# detect_ensemble.py
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import matplotlib.pyplot as plt
from src.datasets.dataset import load_datasets
import os
import hydra
import torch
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

# ----------------------
# CONFIG: YOLO ensemble weights per month
# Example: month '01' uses models A,B,C with weights 0.5,0.3,0.2
# ----------------------
MONTH_WEIGHTS = {
    '01': {
        'runs_yolo/train4/weights/best.pt': 0.6,
        'runs_yolo/train5/weights/best.pt': 0.3,
        'runs_yolo/train6/weights/best.pt': 0.1
    },
    '02': {
        'runs_yolo/train4/weights/best.pt': 0.55,
        'runs_yolo/train5/weights/best.pt': 0.3,
        'runs_yolo/train6/weights/best.pt': 0.15
    },
    '03': {
        'runs_yolo/train4/weights/best.pt': 0.5,
        'runs_yolo/train5/weights/best.pt': 0.3,
        'runs_yolo/train6/weights/best.pt': 0.2
    },
    '04': {
        'runs_yolo/train5/weights/best.pt': 0.3,
        'runs_yolo/train4/weights/best.pt': 0.5,
        'runs_yolo/train6/weights/best.pt': 0.2
    },
    '05': {
        'runs_yolo/train5/weights/best.pt': 0.2,
        'runs_yolo/train4/weights/best.pt': 0.5,
        'runs_yolo/train6/weights/best.pt': 0.3
    },
    '06': {
        'runs_yolo/train5/weights/best.pt': 0.2,
        'runs_yolo/train4/weights/best.pt': 0.3,
        'runs_yolo/train6/weights/best.pt': 0.5
    },
    '07': {
        'runs_yolo/train5/weights/best.pt': 0.15,
        'runs_yolo/train4/weights/best.pt': 0.3,
        'runs_yolo/train6/weights/best.pt': 0.55
    },
    '08': {
        'runs_yolo/train5/weights/best.pt': 0.1,
        'runs_yolo/train4/weights/best.pt': 0.3,
        'runs_yolo/train6/weights/best.pt': 0.6
    },
    '09': {
        'runs_yolo/train5/weights/best.pt': 0.2,
        'runs_yolo/train4/weights/best.pt': 0.3,
        'runs_yolo/train6/weights/best.pt': 0.5
    },
    '10': {
        'runs_yolo/train5/weights/best.pt': 0.3,
        'runs_yolo/train4/weights/best.pt': 0.3,
        'runs_yolo/train6/weights/best.pt': 0.3
    },
    '11': {
        'runs_yolo/train5/weights/best.pt': 0.4,
        'runs_yolo/train4/weights/best.pt': 0.2,
        'runs_yolo/train6/weights/best.pt': 0.4
    },
    '12': {
        'runs_yolo/train5/weights/best.pt': 0.4,
        'runs_yolo/train4/weights/best.pt': 0.2,
        'runs_yolo/train6/weights/best.pt': 0.4
    }


}

# Fallback weights if month missing
FALLBACK_WEIGHTS = {
    'runs_yolo/train4/weights/best.pt': 0.33,
    'runs_yolo/train5/weights/best.pt': 0.33,
    'runs_yolo/train6/weights/best.pt': 0.34
}

# ----------------------
# Cache dei modelli
# ----------------------
_models_cache = {}

def extract_month_from_path(img_path: str):
    if img_path is None:
        return None
    m = re.search(r'(20\d{2})(0[1-9]|1[0-2])([0-3]\d)', img_path)
    if m:
        return m.group(2)
    return None

def load_model_checkpoint(checkpoint_path: str):
    if checkpoint_path in _models_cache:
        return _models_cache[checkpoint_path]
    print(f"[INFO] Loading model from {checkpoint_path}")
    model = YOLO(checkpoint_path)
    _models_cache[checkpoint_path] = model
    return model


### NEW – weighted NMS for merging boxes from multiple YOLOs
def weighted_nms(boxes, scores, labels, iou_thres=0.5):
    if len(boxes) == 0:
        return [], [], []

    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)

    keep_boxes = []
    keep_scores = []
    keep_labels = []

    idxs = scores.argsort()[::-1]  # sort by confidence descending

    while len(idxs) > 0:
        main = idxs[0]
        main_box = boxes[main]
        same_class = labels[idxs] == labels[main]

        overlaps = []
        overlaps_idx = []

        for i in idxs[same_class]:
            box = boxes[i]
            # compute IoU
            x1 = max(main_box[0], box[0])
            y1 = max(main_box[1], box[1])
            x2 = min(main_box[2], box[2])
            y2 = min(main_box[3], box[3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_main = (main_box[2]-main_box[0])*(main_box[3]-main_box[1])
            area_box = (box[2]-box[0])*(box[3]-box[1])
            union = area_main + area_box - inter

            iou = inter / union if union > 0 else 0

            if iou > iou_thres:
                overlaps.append(box)
                overlaps_idx.append(i)

        # weighted average of overlapping boxes
        overlaps = np.array(overlaps)
        s = scores[overlaps_idx]
        wbox = np.average(overlaps, axis=0, weights=s)

        keep_boxes.append(wbox.tolist())
        keep_scores.append(float(scores[main]))
        keep_labels.append(int(labels[main]))

        idxs = np.array([i for i in idxs if i not in overlaps_idx])

    return keep_boxes, keep_scores, keep_labels


@hydra.main(config_path='config', config_name='config', version_base="1.3")
def main(args):
    fallback_ckpt = args.modelCheckpoint if hasattr(args, 'modelCheckpoint') else None

    train, val, test, collate_fn = load_datasets(args)

    if args.submission.type == 'val':
        templatePath = args.submission.valTemplate
        dataset = val
    elif args.submission.type == 'test':
        templatePath = args.submission.testTemplate
        dataset = test
    else:
        raise ValueError(f"Unknown submission.type {args.submission.type}")

    with open(templatePath, 'r') as f:
        submission = json.load(f)

    # Group images per month
    month_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        img_path = dataset.get_img_path(i)
        month = extract_month_from_path(img_path)
        if month is None:
            month = 'fallback'
        month_to_indices[month].append(i)

    infer_kwargs = dict(imgsz=160, conf=0.01)

    # MAIN LOOP
    for month, indices in month_to_indices.items():

        ### NEW – choose weight set
        if month == 'fallback':
            model_weights = FALLBACK_WEIGHTS
        else:
            model_weights = MONTH_WEIGHTS.get(month, FALLBACK_WEIGHTS)

        # load all 3 models only once
        models = []
        for ckpt_path in model_weights.keys():
            models.append((load_model_checkpoint(ckpt_path), model_weights[ckpt_path]))

        print(f"\n[INFO] Ensemble inference month '{month}' with models:")
        for ckpt_path, w in model_weights.items():
            print(f"   {ckpt_path}: weight {w}")

        # Inference per image
        for idx in indices:
            print(f"\nProcessing image {idx+1}/{len(dataset)} (month {month})")
            imgPath = dataset.get_img_path(idx)

            all_boxes = []
            all_scores = []
            all_labels = []

            ### NEW — run 3 models
            for model, w in models:
                results = model.predict(source=imgPath, **infer_kwargs)
                if len(results) == 0:
                    continue

                r = results[0]
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()
                scores = (r.boxes.conf.cpu().numpy()) * w   ### MOD — weighted confidence
                labels = r.boxes.cls.cpu().numpy().astype(int) + 1

                all_boxes.extend(boxes.tolist())
                all_scores.extend(scores.tolist())
                all_labels.extend(labels.tolist())

            ### NEW — fuse predictions using weighted NMS
            fused_boxes, fused_scores, fused_labels = weighted_nms(
                all_boxes, all_scores, all_labels, iou_thres=0.5
            )

            img_id_str = str(dataset.ids[idx])
            if img_id_str in submission:
                submission[img_id_str]["boxes"] = fused_boxes
                submission[img_id_str]["scores"] = fused_scores
                submission[img_id_str]["labels"] = fused_labels

    os.makedirs('submissions', exist_ok=True)
    outpath = 'submissions/predictions.json'
    with open(outpath, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"\n[INFO] Saved predictions to {outpath}")

if __name__ == "__main__":
    main()
