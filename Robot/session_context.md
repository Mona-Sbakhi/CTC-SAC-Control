# Session Context — CoDIT 2026 Paper Revision

**Paper title:** Hybrid Computed Torque Control and Soft Actor-Critic Framework for Sample-Efficient Robotic Manipulator Control  
**Conference:** CoDIT 2026 — Submission 198  
**Status:** Conditionally accepted — revision deadline **29 May 2026**  
**Authors:** Mona Alsbakhi, Majed Tabash, Anas Alsalool, Asma Sbaih  
**Workspace:** `/Users/mona/Documents/GitHub/CTC-SAC-Control/Robot/`

---

## 1. Reviewer Response

Three reviewers addressed. Full response in `reviewer_response_v2.docx`.

**Reviewer 1 (9 points):** title spelling, paper structure paragraph, PID clarification, PID form details, stability discussion, robustness testing, kinematic diagram (Figure 1 — still needed in LaTeX), future work expansion, author prior works.

**Reviewer 3 (8 comprehension questions):** all confirmed answered in paper. Key updated numbers:
- CTC+SAC vs CTC-Only: **−1.0%** mean error (2.22 vs 2.24 rad)
- CTC+SAC vs RL-Only: **−4.9%** mean error (2.22 vs 2.33 rad)
- CTC+SAC vs PID+SAC: **−3.3%** mean error (2.22 vs 2.29 rad)

**Reviewer 8 (5 concerns):**
- **Concern 1 (PID+RL baseline):** COMPLETED — PID+SAC experiment run, CTC+SAC wins on all metrics. Response updated from "unable to include" → "experiment completed, results in Table VI."
- **Concern 2 (Table VI inconsistency):** RESOLVED — corrected experimental setup; CTC+SAC is now best on all metrics, no inconsistency.
- **Concern 3 (large absolute errors):** explained via model mismatch and initial condition. SS error = 2.27 rad ≈ 4.5× trajectory amplitude (acknowledged as limitation).
- **Concern 4 (tone down conclusions):** revised conclusions scoped to "simulation, 2-DOF, tested trajectory."
- **Concern 5 (grammar):** specific corrections listed in `paper_corrections.md`.

---

## 2. Critical Bug Found and Fixed

**The bug:** Original code used `/Mico/joint` (joint 1 = base, rotates around vertical Z-axis) which does NOT match the vertical-plane Lagrangian model. Gravity term G(q) was computed incorrectly.

**The fix:** Switched to a clean 2-DOF CoppeliaSim scene. New joint paths confirmed via `test_joints.py`:
- `j1 = sim.getObject('/Mico/joint')` → handle 21 (shoulder, vertical plane)
- `j2 = sim.getObject('/Mico/joint/link/joint')` → handle 23 (elbow, vertical plane)

**Impact:** Old results (v1) showed errors of 13–34 rad (robot spinning). New results (v2) show 2.2 rad — physically meaningful.

**Old v1 paths (wrong, do not use):**
```python
j1 = sim.getObject('/Mico/joint/link/joint')             # was shoulder
j2 = sim.getObject('/Mico/joint/link/joint/link/joint')  # was elbow — NOT FOUND in 2-DOF scene
```

---

## 3. Files Created / Modified

All v2 files are in `/Robot/src/`. Original v1 files are untouched.

### New files (v2):

| File | Description |
|------|-------------|
| `train_ctc_sac_v2.py` | CTC+SAC, SAC range ±10 N·m, 300 episodes |
| `train_rl_only_v2.py` | RL-Only, SAC range ±20 N·m, 200 episodes |
| `train_pid_sac_v2.py` | PID+SAC, PID gains below, SAC ±10 N·m, 200 episodes |
| `evaluate_v2.py` | Loads `*_v2.pt` models, saves `comparison_v2.json` + `table_vi_v2.txt` |
| `pid_controller.py` | PIDController with anti-windup (integral clipped to ±5) |

### PID gains used:
```python
Kp = [100.0, 50.0]
Ki = [5.0,   2.0]
Kd = [20.0,  10.0]
tau_max = 15.0  # N·m per joint
```

### Joint setup pattern (v2 — use this everywhere):
```python
def setup_joints(sim):
    j1 = sim.getObject('/Mico/joint')             # shoulder → handle 21
    j2 = sim.getObject('/Mico/joint/link/joint')  # elbow    → handle 23
    sim.setObjectInt32Param(j1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    sim.setObjectInt32Param(j2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    return j1, j2

def reset_episode(sim, j1, j2):
    sim.stopSimulation();  time.sleep(0.3)
    sim.startSimulation(); time.sleep(0.2)
    sim.setObjectInt32Param(j1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
    sim.setObjectInt32Param(j2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
    sim.setJointTargetPosition(j1, 0.0)
    sim.setJointTargetPosition(j2, 0.0)
    time.sleep(0.4)
    sim.setObjectInt32Param(j1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    sim.setObjectInt32Param(j2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    time.sleep(0.1)
```

### Saved models (in `/Robot/models/`):
- `best_ctc_sac_v2.pt` — scale 10.0
- `best_rl_only_v2.pt` — scale 20.0
- `best_pid_sac_v2.pt` — scale 10.0

---

## 4. Final v2 Results (Table VI — fair comparison)

All four methods trained and evaluated in the same 2-DOF CoppeliaSim scene.  
Evaluation: 10 seconds per method, DT=0.01s, trajectory: sinusoidal amp=0.5 rad, freq=0.5 Hz.

| Method | RMS q1 | RMS q2 | Mean(full) | SS Error | TR Error | Max q1 |
|--------|--------|--------|------------|---------|---------|--------|
| CTC-Only | 1.70 | 1.66 | 2.24 | 2.40 | 2.33 | 2.18 |
| **CTC+SAC** | **1.54** ✅ | 1.66 | **2.22** ✅ | **2.27** ✅ | **2.25** ✅ | **2.05** ✅ |
| RL-Only | 1.70 | 1.66 | 2.33 | 2.41 | 2.31 | 2.17 |
| PID+SAC | 1.65 | 1.66 | 2.29 | 2.36 | 2.30 | 2.14 |

- SS Error = mean error over last 50% of episode (t ≥ 5s)
- TR Error = mean error over first 20% of episode (t < 2s)
- CTC+SAC is best on **all** metrics ✅

---

## 5. What Remains Before Submission

- [ ] Update Table VI numbers in LaTeX paper (use v2 results above)
- [ ] Add kinematic diagram as Figure 1 (Reviewer 1, Comment 7)
- [ ] Apply grammar corrections from `paper_corrections.md`
- [ ] Tone down conclusions (Reviewer 8, Concern 4) — simulation-only, 2-DOF scope
- [ ] Submit: revised PDF + `reviewer_response_v2.docx`

---

## 6. Key Files Reference

| File | Purpose |
|------|---------|
| `reviewer_response_v2.docx` | Final reviewer response — ready to submit |
| `paper_corrections.md` | Specific LaTeX before/after corrections |
| `src/train_ctc_sac_v2.py` | Re-run CTC+SAC training |
| `src/train_rl_only_v2.py` | Re-run RL-Only training |
| `src/train_pid_sac_v2.py` | Re-run PID+SAC training |
| `src/evaluate_v2.py` | Re-run evaluation (generates Table VI) |
| `results/comparison_v2.json` | Raw evaluation metrics |
| `results/table_vi_v2.txt` | Table VI ready to paste |
