#!/usr/bin/env python3
"""
Evaluate All Methods — CTC-Only, RL-Only, CTC+SAC
===================================================
CoDIT 2026 Paper: CTC + SAC Hybrid Control

Computes full metrics including Steady-State Error for Table VI.

Usage (inside Docker):
    cd /workspace/src

    # Copy your RL-Only model first:
    cp ~/Downloads/ros2_ws/src/results/best_rl_only.pt ../models/

    python evaluate.py

Outputs:
    ../results/comparison.json       — all metrics
    ../results/table_vi.txt          — ready to paste into paper
    ../results/ts_<method>.csv       — raw time-series per method

Authors: Mona Alsbakhi, Majed Tabash, Anas Alsalool, Asma Sbaih
"""

import numpy as np
import json, time, os, csv
import torch
import torch.nn as nn

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from robot_dynamics import create_nominal_robot
from ctc_controller import CTCController, CTCGains
from pid_controller import PIDController, PIDGains
from sac_agent import SACAgent, SACConfig

# ── CONFIG ────────────────────────────────────────────────────────
COPPELIASIM_HOST = os.environ.get("COPPELIASIM_HOST", "host.docker.internal")
DURATION         = 10.0   # seconds per evaluation episode
DT               = 0.01   # 10 ms step
SS_FRAC          = 0.5    # Steady-state = last 50 % (t ≥ 5 s)
TR_FRAC          = 0.2    # Transient    = first 20% (t < 2 s)
MODELS_DIR       = "../models"
RESULTS_DIR      = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── TRAJECTORY ────────────────────────────────────────────────────
def trajectory(t, freq=0.5, amp=0.5):
    w = 2 * np.pi * freq
    q_d      = np.array([amp * np.sin(w * t),         amp * np.sin(w * t + np.pi / 4)])
    q_dot_d  = np.array([amp * w * np.cos(w * t),     amp * w * np.cos(w * t + np.pi / 4)])
    q_ddot_d = np.array([-amp * w**2 * np.sin(w * t), -amp * w**2 * np.sin(w * t + np.pi / 4)])
    return q_d, q_dot_d, q_ddot_d

# ── LOAD ACTOR (handles multiple checkpoint formats) ──────────────
class _ActorV1(nn.Module):
    def __init__(self, scale=5.0):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(nn.Linear(8,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU())
        self.mean_head    = nn.Linear(256, 2)
        self.log_std_head = nn.Linear(256, 2)
    def get_action(self, s):
        return torch.tanh(self.mean_head(self.net(s))) * self.scale

class _ActorV2(nn.Module):
    def __init__(self, scale=20.0):
        super().__init__()
        self.scale = scale
        self.net  = nn.Sequential(nn.Linear(8,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU())
        self.mean = nn.Linear(256, 2)
        self.log_std = nn.Linear(256, 2)
    def get_action(self, s):
        return torch.tanh(self.mean(self.net(s))) * self.scale

def load_actor(path, scale):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    # Support multiple checkpoint formats
    if isinstance(ckpt, dict):
        weights = (ckpt.get('actor')
                or ckpt.get('actor_state_dict')
                or ckpt)
    else:
        weights = ckpt

    # Detect architecture by key names
    if any('mean_head' in k for k in weights.keys()):
        actor = _ActorV1(scale)
    else:
        actor = _ActorV2(scale)

    actor.load_state_dict(weights)
    actor.eval()
    return actor

# ── METRICS ───────────────────────────────────────────────────────
def compute_metrics(times, e1, e2):
    e1, e2 = np.array(e1), np.array(e2)
    norm   = np.sqrt(e1**2 + e2**2)
    t0, t1 = times[0], times[-1]
    dur    = t1 - t0

    ss_idx = np.where(np.array(times) >= t0 + dur * (1 - SS_FRAC))[0]
    tr_idx = np.where(np.array(times) <  t0 + dur * TR_FRAC)[0]

    def s(idx, tag):
        return {
            f"rms_q1_{tag}":   float(np.sqrt(np.mean(e1[idx]**2))),
            f"rms_q2_{tag}":   float(np.sqrt(np.mean(e2[idx]**2))),
            f"mean_err_{tag}": float(np.mean(norm[idx])),
            f"max_q1_{tag}":   float(np.max(np.abs(e1[idx]))),
        }

    m = {}
    m.update(s(np.arange(len(times)), "full"))
    m.update(s(tr_idx, "transient"))
    m.update(s(ss_idx, "ss"))
    return m

# ── SAFE RESET FOR EVALUATION ─────────────────────────────────────
def setup_joints(sim):
    mico       = sim.getObject('/Mico')
    all_joints = sim.getObjectsInTree(mico, sim.object_joint_type, 0)
    for jh in all_joints:
        sim.setObjectInt32Param(jh, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
        sim.setJointTargetPosition(jh, 0.0)
    j1 = sim.getObject('/Mico/joint/link/joint')             # shoulder
    j2 = sim.getObject('/Mico/joint/link/joint/link/joint')  # elbow
    sim.setObjectInt32Param(j1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    sim.setObjectInt32Param(j2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    return j1, j2, all_joints

def reset_eval(sim, j1, j2, all_joints):
    sim.stopSimulation(); time.sleep(0.3)
    sim.startSimulation(); time.sleep(0.2)
    for jh in all_joints:
        sim.setObjectInt32Param(jh, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
        sim.setJointTargetPosition(jh, 0.0)
    time.sleep(0.4)
    sim.setObjectInt32Param(j1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    sim.setObjectInt32Param(j2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    time.sleep(0.1)

# ── RUN ONE METHOD ────────────────────────────────────────────────
def run(sim, j1, j2, all_joints, ctc, pid, ctc_sac_actor, rl_only_actor, pid_sac_actor, method):
    print(f"\n{'='*55}\n  {method}\n{'='*55}")

    reset_eval(sim, j1, j2, all_joints)
    if pid is not None:
        pid.reset()

    times, e1_log, e2_log = [], [], []
    t = 0.0

    while t < DURATION:
        q     = np.array([sim.getJointPosition(j1), sim.getJointPosition(j2)])
        q_dot = np.array([sim.getJointVelocity(j1),  sim.getJointVelocity(j2)])
        q_d, q_dot_d, q_ddot_d = trajectory(t)
        e     = q_d - q
        e_dot = q_dot_d - q_dot

        if method == "CTC-Only":
            tau = np.clip(ctc.compute_torque(q, q_dot, q_d, q_dot_d, q_ddot_d), -20.0, 20.0)

        elif method == "CTC+SAC":
            tau_ctc = ctc.compute_torque(q, q_dot, q_d, q_dot_d, q_ddot_d)
            obs = torch.FloatTensor(np.concatenate([q, q_dot, e, e_dot])).unsqueeze(0)
            with torch.no_grad():
                tau_rl = ctc_sac_actor.get_action(obs).numpy()[0]
            tau = np.clip(tau_ctc + np.clip(tau_rl, -5.0, 5.0), -20.0, 20.0)

        elif method == "RL-Only":
            obs = torch.FloatTensor(np.concatenate([q, q_dot, e, e_dot])).unsqueeze(0)
            with torch.no_grad():
                tau = rl_only_actor.get_action(obs).numpy()[0]
            tau = np.clip(tau, -20.0, 20.0)

        elif method == "PID+SAC":
            tau_pid = pid.compute_torque(e, e_dot)
            obs = torch.FloatTensor(np.concatenate([q, q_dot, e, e_dot])).unsqueeze(0)
            with torch.no_grad():
                tau_rl = pid_sac_actor.get_action(obs).numpy()[0]
            tau = np.clip(tau_pid + np.clip(tau_rl, -10.0, 10.0), -20.0, 20.0)

        sim.setJointTargetForce(j1, float(tau[0]))
        sim.setJointTargetForce(j2, float(tau[1]))

        times.append(t); e1_log.append(float(e[0])); e2_log.append(float(e[1]))

        if int(t * 10) % 20 == 0:
            print(f"  t={t:.1f}s | e=[{e[0]:+.3f}, {e[1]:+.3f}] rad")

        time.sleep(DT)
        t = round(t + DT, 6)

    time.sleep(0.3)

    # Save time-series CSV
    tag = method.replace('+', '_').replace('-', '_')
    with open(f"{RESULTS_DIR}/ts_{tag}.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time', 'error_q1', 'error_q2', 'norm_error'])
        for ti, e1i, e2i in zip(times, e1_log, e2_log):
            w.writerow([ti, e1i, e2i, np.sqrt(e1i**2 + e2i**2)])

    metrics = compute_metrics(times, e1_log, e2_log)
    metrics['method'] = method
    print(f"\n  Full      → Mean={metrics['mean_err_full']:.4f}  RMS_q1={metrics['rms_q1_full']:.4f}")
    print(f"  Transient → Mean={metrics['mean_err_transient']:.4f}  (t < {DURATION*TR_FRAC:.1f}s)")
    print(f"  SS        → Mean={metrics['mean_err_ss']:.4f}  (t ≥ {DURATION*(1-SS_FRAC):.1f}s)")
    return metrics

# ── PRINT TABLE ───────────────────────────────────────────────────
def print_table(all_m):
    header = f"{'Method':<12}|{'RMS q1':>9}|{'RMS q2':>9}|{'Mean(full)':>11}|{'SS Error':>10}|{'TR Error':>10}|{'Max q1':>9}"
    sep    = "-" * len(header)
    print(f"\n\n{'='*len(header)}")
    print("  UPDATED TABLE VI — CoDIT 2026")
    print(f"{'='*len(header)}")
    print(header); print(sep)
    for m in all_m:
        print(f"{m['method']:<12}|{m['rms_q1_full']:>9.4f}|{m['rms_q2_full']:>9.4f}|"
              f"{m['mean_err_full']:>11.4f}|{m['mean_err_ss']:>10.4f}|"
              f"{m['mean_err_transient']:>10.4f}|{m['max_q1_full']:>9.4f}")
    print(f"{'='*len(header)}")
    print(f"  SS = last {int(SS_FRAC*100)}% of episode (t ≥ {DURATION*(1-SS_FRAC):.0f}s)")
    print(f"  TR = first {int(TR_FRAC*100)}% of episode (t <  {DURATION*TR_FRAC:.0f}s)")
    print(f"  All errors in rad\n")

# ── MAIN ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Evaluation — CoDIT 2026")
    print("=" * 60)

    client = RemoteAPIClient(host=COPPELIASIM_HOST)
    sim    = client.require('sim')
    print("✅ Connected to CoppeliaSim")

    j1, j2, all_joints = setup_joints(sim)
    print("✅ All 6 joints found — joints 1,4,5,6 locked; shoulder+elbow in torque mode")

    ctc = CTCController(create_nominal_robot(),
                        CTCGains(Kp=np.array([100.0, 50.0]), Kd=np.array([20.0, 10.0])))
    pid = PIDController(PIDGains(Kp=np.array([100.0, 50.0]),
                                 Ki=np.array([5.0,    2.0]),
                                 Kd=np.array([20.0,  10.0])), dt=DT)

    # Load models — try common paths
    ctc_sac_actor = rl_only_actor = pid_sac_actor = None

    for p, scale in [
        (f"{MODELS_DIR}/best_ctc_sac.pt",    5.0),
        (f"{MODELS_DIR}/best_agent.pt",       5.0),
        ("ctc_sac_agent.pt",                  5.0),
    ]:
        if os.path.exists(p):
            ctc_sac_actor = load_actor(p, scale)
            print(f"✅ CTC+SAC model ← {p}"); break
    if not ctc_sac_actor:
        print("⚠️  CTC+SAC model not found — skipping")

    for p, scale in [
        (f"{MODELS_DIR}/best_rl_only.pt",    20.0),
        ("best_rl_only.pt",                  20.0),
    ]:
        if os.path.exists(p):
            rl_only_actor = load_actor(p, scale)
            print(f"✅ RL-Only model  ← {p}"); break
    if not rl_only_actor:
        print("⚠️  RL-Only model not found — skipping")

    for p, scale in [
        (f"{MODELS_DIR}/best_pid_sac.pt",    10.0),
    ]:
        if os.path.exists(p):
            pid_sac_actor = load_actor(p, scale)
            print(f"✅ PID+SAC model  ← {p}"); break
    if not pid_sac_actor:
        print("⚠️  PID+SAC model not found — skipping")

    # Run experiments
    methods = ["CTC-Only"]
    if ctc_sac_actor: methods.append("CTC+SAC")
    if rl_only_actor: methods.append("RL-Only")
    if pid_sac_actor: methods.append("PID+SAC")

    all_metrics = []
    for method in methods:
        m = run(sim, j1, j2, all_joints, ctc, pid, ctc_sac_actor, rl_only_actor, pid_sac_actor, method)
        all_metrics.append(m)
        time.sleep(1.0)

    print_table(all_metrics)

    # Save JSON
    out = {m['method']: m for m in all_metrics}
    json.dump(out, open(f"{RESULTS_DIR}/comparison.json", 'w'), indent=2)
    print(f"✅ Saved → {RESULTS_DIR}/comparison.json")

    # Save ready-to-paste table
    with open(f"{RESULTS_DIR}/table_vi.txt", 'w') as f:
        f.write("UPDATED TABLE VI — CoDIT 2026 Paper\n")
        f.write(f"{'Method':<12}|{'RMS q1':>9}|{'RMS q2':>9}|{'Mean(full)':>11}|{'SS Error':>10}|{'TR Error':>10}|{'Max q1':>9}\n")
        f.write("-"*73 + "\n")
        for m in all_metrics:
            f.write(f"{m['method']:<12}|{m['rms_q1_full']:>9.4f}|{m['rms_q2_full']:>9.4f}|"
                    f"{m['mean_err_full']:>11.4f}|{m['mean_err_ss']:>10.4f}|"
                    f"{m['mean_err_transient']:>10.4f}|{m['max_q1_full']:>9.4f}\n")
        f.write(f"\nSS Error = mean error over last {int(SS_FRAC*100)}% of episode (t >= {DURATION*(1-SS_FRAC):.0f}s)\n")
    print(f"✅ Saved → {RESULTS_DIR}/table_vi.txt")

if __name__ == "__main__":
    main()
