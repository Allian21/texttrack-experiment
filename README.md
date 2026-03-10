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
| YOLO | Ultralytics YOLOv8 (fine-tuned on bags, bottles, phones, laptops) |
| CLIP | ViT-B/32 |

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

Place your YOLO model as `best2.pt` in the root folder.
Place 29 test images in the `./images/` folder.

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
| Default CFS | 565.4ms | 580.0ms | 938.7ms | 1.77 img/sec |
| One-Core Pinning | 1040.2ms | 1082.7ms | 1694.1ms | 0.96 img/sec |
| Two-Core Pinning | 817.8ms | 853.9ms | 1348.2ms | 1.22 img/sec |

---

## Folder Structure
```
texttrack-experiment/
├── README.md
├── run_experiment.sh   — one-command reproduction script
├── pipeline.py         — YOLO+CLIP inference and timing
├── plot_results.py     — generates figures from CSV results
├── best2.pt            — YOLO model (obtain separately)
├── images/             — test images (obtain separately)
├── figures/            — output charts
└── results/            — raw CSV latency logs
```
