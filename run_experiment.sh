#!/bin/bash

echo "================================"
echo " TextTrack Scheduling Experiment"
echo "================================"
echo ""

echo "[CONFIG A] Baseline - Default CFS"
echo "----------------------------------"
for i in 1 2 3; do
    echo "  Trial $i of 3..."
    python3 pipeline.py --output baseline_trial${i}.csv
    echo ""
done

echo "[CONFIG B] One-Core Pinning"
echo "----------------------------"
for i in 1 2 3; do
    echo "  Trial $i of 3..."
    taskset -c 0 python3 pipeline.py --output one_core_trial${i}.csv
    echo ""
done

echo "[CONFIG C] Two-Core Pinning"
echo "----------------------------"
for i in 1 2 3; do
    echo "  Trial $i of 3..."
    taskset -c 0,1 python3 pipeline.py --output two_core_trial${i}.csv
    echo ""
done

echo "All done! Generating charts..."
python3 plot_results.py

echo ""
echo "Finished. Charts are in the figures/ folder."
