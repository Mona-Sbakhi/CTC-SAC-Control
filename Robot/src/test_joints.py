#!/usr/bin/env python3
"""
Test Joint Paths in CoppeliaSim Mico Scene
============================================
Lists all joints in the Mico robot tree with their names,
handles, and axis information.

Usage (inside Docker):
    cd /workspace/src
    python test_joints.py

Authors: Mona Alsbakhi, Majed Tabash, Anas Alsalool, Asma Sbaih
"""

import os
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

COPPELIASIM_HOST = os.environ.get("COPPELIASIM_HOST", "host.docker.internal")

client = RemoteAPIClient(host=COPPELIASIM_HOST)
sim    = client.require('sim')
print("✅ Connected to CoppeliaSim\n")

# ── Get all joints in the Mico tree ──────────────────────────────────────────
try:
    mico_handle = sim.getObject('/Mico')
except Exception:
    print("❌ Could not find /Mico — check your scene has the Mico robot loaded.")
    exit(1)

handles = sim.getObjectsInTree(mico_handle, sim.object_joint_type, 0)

print(f"Found {len(handles)} joints in /Mico:\n")
print(f"  {'#':<4} {'Handle':<8} {'Full Path'}")
print(f"  {'-'*4} {'-'*8} {'-'*50}")

for i, h in enumerate(handles):
    try:
        alias = sim.getObjectAlias(h, 1)   # full path
    except Exception:
        alias = sim.getObjectAlias(h, 0)   # short name fallback
    print(f"  {i+1:<4} {h:<8} {alias}")

print()
print("── Quick path test ──────────────────────────────────────────")
paths = [
    '/Mico/joint',
    '/Mico/joint/link/joint',
    '/Mico/joint/link/joint/link/joint',
    '/Mico/joint/link/joint/link/joint/link/joint',
]
for p in paths:
    try:
        h = sim.getObject(p)
        alias = sim.getObjectAlias(h, 1)
        print(f"  ✅ {p:<45} → handle {h}  ({alias})")
    except Exception:
        print(f"  ❌ {p:<45} → NOT FOUND")
