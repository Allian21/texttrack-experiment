# TextTrack CPU Scheduling Experiment

## Authors
- Francis Lucky Emmanuel I. Del Mundo
- Allain Jave M. Boncilao
- Brendan Joshua E. Acuña

Mapua MCM — BS Computer Science, CS151

---

## Overview
This experiment analyzes how Linux CPU scheduling configurations affect
per-frame inference latency in a YOLO-CLIP pipeline for CCTV-based
lost item retrieval. Three configurations are tested: default CFS,
one-core pinning, and two-core pinning using taskset.

---

## Environment

| Component | Details |
|---|---|
| OS | Ubuntu 22.04.3 LTS (WSL2) |
| Kernel | 6.6.87.2-microsoft-standard-WSL2 |
| Kernel Build | #1 SMP PREEMPT_DYNAMIC Thu Jun 5 18:30:46 UTC 2025 |
| Host OS | Windows 11 Home |
| CPU | Intel Core i5-13420H (8 cores, 12 threads) |
| RAM | 24GB |
| Python | 3.10 |
| PyTorch | CPU-only |
| YOLO | YOLO26 (fine-tuned on bags, bottles, phones, and laptops) |
| ByteTrack | Integrated via Ultralytics tracker |
---

## Key sysctl Settings
WSL2 does not expose CFS sysctl parameters via /proc/sys/kernel.
The following commands were attempted but returned no such file:
```
sysctl kernel.sched_latency_ns        → not available in WSL2
sysctl kernel.sched_min_granularity_ns → not available in WSL2
sysctl kernel.sched_wakeup_granularity_ns → not available in WSL2
```

No custom sysctl modifications were applied.
Default CFS scheduler behavior was used as the baseline.

---

## cgroup Settings
cgroup version in use: **cgroup2fs** (unified hierarchy)

Verified with:
```bash
stat -fc %T /sys/fs/cgroup/
# output: cgroup2fs
```

No custom cgroup restrictions were applied.
CPU pinning was performed exclusively via taskset:
- One-core: taskset -c 0
- Two-core: taskset -c 0,1

---



## How to Reproduce

## Download Required Files

The model and test images are not included in this repository due to file size.
Download them here:

👉 https://drive.google.com/drive/folders/1uLNtuiEC49BAshq4lwSzyqzpxXyF2oM0?usp=sharing

After downloading:
- Place `best2.pt` in the root `texttrack-experiment/` folder
- Place all images into the `images/` folder

### Setup
```bash
git clone https://github.com/yourusername/texttrack-experiment
cd texttrack-experiment
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python-headless numpy matplotlib pandas pillow psutil
```

- Place your YOLO model as `best2.pt` in the root folder.
- Place 50 consecutive CCTV frames in the `./images/` folder.


### Run (one command)
```bash
bash run_experiment.sh
```

### Output
```
figures/p95_latency.png   — Figure 1: p95 latency bar chart
figures/mean_latency.png  — Figure 2: mean latency with std deviation
figures/p50_vs_p95.png    — Figure 3: p50 vs p95 comparison
results/                  — raw CSV files for all 9 trials
```

---

## Results Summary

| Configuration | Mean | p50 | p95 | Throughput |
|---|---|---|---|---|
| Default CFS (Baseline) | 485.1ms | 501.9ms | 698.3ms | 2.07 img/sec |
| One-Core Pinning | 944.9ms | 994.7ms | 1305.7ms | 1.06 img/sec |
| Two-Core Pinning | 715.7ms | 749.4ms | 1003.5ms | 1.40 img/sec |

Key finding: Default CFS outperformed both pinned configurations
in the YOLO26+ByteTrack+CLIP pipeline, achieving 485.1ms mean latency
vs 944.9ms for one-core pinning — a 95% increase.


---

## Folder Structure
```
texttrack-experiment/
├── README.md
├── run_experiment.sh   — one-command reproduction script
├── pipeline.py         — YOLO+CLIP inference and timing
├── plot_results.py     — generates figures from CSV results
├── best2.pt            — YOLO model (see Download section below)
├── images/             — test images (see Download section below)
├── figures/            — output charts
└── results/            — raw CSV latency logs
```
