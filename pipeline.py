import time
import csv
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="results.csv")
args = parser.parse_args()

MODEL_PATH = "best2.pt"
IMAGE_DIR = "./images"

print("Loading YOLO model...")
yolo = YOLO(MODEL_PATH)

print("Loading CLIP model...")
device = "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

query = "bag"
text_tokens = clip.tokenize([query]).to(device)

image_paths = list(Path(IMAGE_DIR).glob("*.jpg")) + \
              list(Path(IMAGE_DIR).glob("*.png"))

if len(image_paths) == 0:
    print("ERROR: No images found in ./images folder")
    exit()

image_paths = image_paths[:50]
print(f"Found {len(image_paths)} images")

print("Warming up models...")
dummy_frame = cv2.imread(str(image_paths[0]))
yolo(dummy_frame, verbose=False)
print("Warm up done. Starting experiment...")

results_log = []

for img_path in image_paths:
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue

    start = time.perf_counter()

    detections = yolo(frame, verbose=False)

    for result in detections:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                pil_crop = preprocess(
                    Image.fromarray(
                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    )
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(pil_crop)
                    text_features = clip_model.encode_text(text_tokens)
                    similarity = (image_features @ text_features.T).item()
            except Exception:
                continue

    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    results_log.append(latency_ms)
    print(f"  {img_path.name}: {latency_ms:.1f}ms")

os.makedirs("results", exist_ok=True)
output_path = f"results/{args.output}"
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "latency_ms"])
    for i, lat in enumerate(results_log):
        writer.writerow([i + 1, round(lat, 2)])

arr = np.array(results_log)
print(f"\n--- Results saved to {output_path} ---")
print(f"Images processed : {len(arr)}")
print(f"Throughput       : {1000/arr.mean():.2f} images/sec")
print(f"Mean latency     : {arr.mean():.1f}ms")
print(f"p50 latency      : {np.percentile(arr, 50):.1f}ms")
print(f"p95 latency      : {np.percentile(arr, 95):.1f}ms")
print(f"Std deviation    : {arr.std():.1f}ms")


