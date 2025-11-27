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

# ----------------------
# CONFIG: mappa mesi -> checkpoint
# Modifica qui se i percorsi sono diversi
# ----------------------
MONTH_TO_CHECKPOINT = {
    '01': 'runs_yolo/train4/weights/best.pt',
    '02': 'runs_yolo/train4/weights/best.pt',
    '03': 'runs_yolo/train4/weights/best.pt',
    '04': 'runs_yolo/train5/weights/best.pt',
    '05': 'runs_yolo/train5/weights/best.pt',
    '06': 'runs_yolo/train6/weights/best.pt',
    '07': 'runs_yolo/train6/weights/best.pt',
    '08': 'runs_yolo/train6/weights/best.pt',
    '09': 'runs_yolo/train6/weights/best.pt',
    # aggiungi altri mesi se necessario
}

# ----------------------
# Cache dei modelli (checkpoint_path -> modello YOLO)
# ----------------------
_models_cache = {}

def extract_month_from_path(img_path: str):
    """
    Estrae il mese (MM) da un path che contiene YYYYMMDD da qualche parte,
    e.g. frames/20210122/clip_22/... -> '01'
    Ritorna None se non trova il pattern.
    """
    if img_path is None:
        return None
    # Cerchiamo pattern YYYYMMDD (es: 20210122)
    m = re.search(r'(20\d{2})(0[1-9]|1[0-2])([0-3]\d)', img_path)
    if m:
        return m.group(2)
    return None

def load_model_checkpoint(checkpoint_path: str):
    """
    Carica o restituisce dalla cache il modello corrispondente al checkpoint_path.
    """
    if checkpoint_path in _models_cache:
        return _models_cache[checkpoint_path]

    print(f"[INFO] Loading model from {checkpoint_path}")
    model = YOLO(checkpoint_path)
    _models_cache[checkpoint_path] = model
    return model

@hydra.main(config_path='config', config_name='config', version_base="1.3")
def main(args):
    # fallback checkpoint (usato se non si riesce ad estrarre il mese o se mese non mappato)
    fallback_ckpt = args.modelCheckpoint if hasattr(args, 'modelCheckpoint') else None

    # Carica i dataset (la funzione load_datasets si aspetta args opportuni)
    train, val, test, collate_fn = load_datasets(args)

    if args.submission.type == 'val':
        templatePath = args.submission.valTemplate
        dataset = val
    elif args.submission.type == 'test':
        templatePath = args.submission.testTemplate
        dataset = test
    else:
        raise ValueError(f"Unknown submission.type {args.submission.type}")

    # Carica template di submission
    with open(templatePath, 'r') as f:
        submission = json.load(f)
        # img_ids = [int(key) for key in list(submission.keys())]  # non strettamente necessario qui

    # Raggruppa gli indici del dataset per month in modo da caricare un modello una sola volta
    month_to_indices = defaultdict(list)
    index_to_month = {}
    for i in range(len(dataset)):
        img_path = dataset.get_img_path(i)   # path completo costruito dal dataset
        month = extract_month_from_path(img_path)
        if month is None:
            # proviamo anche a cercare nella stringa date_captured se presente nel coco (fallback)
            # ma dataset.get_img_path non dà accesso ai meta: quindi fallback
            print(f"[WARN] Non ho trovato mese nel filename '{Path(img_path).name}', userò fallback checkpoint.")
            month = 'fallback'
        month_to_indices[month].append(i)
        index_to_month[i] = month

    # Parametri di inferenza (modifica se necessario o prendi da args)
    infer_kwargs = dict(imgsz=160, conf=0.01)

    # Cicla per month, carica il modello corrispondente una volta e inferisci su tutte le immagini
    for month, indices in month_to_indices.items():
        if month == 'fallback':
            checkpoint = fallback_ckpt
        else:
            checkpoint = MONTH_TO_CHECKPOINT.get(month, fallback_ckpt)

        if checkpoint is None:
            raise RuntimeError(f"No checkpoint defined for month {month} and no fallback provided.")

        model = load_model_checkpoint(checkpoint)
        print(f"\n[INFO] Inferenza per month '{month}' con checkpoint '{checkpoint}' -> {len(indices)} immagini")

        for idx in indices:
            i = idx
            print(f'\nProcessing image {i+1}/{len(dataset)} (month {month})')
            imgPath = dataset.get_img_path(i)
            # effettua inferenza
            results = model.predict(source=imgPath, **infer_kwargs)

            # Prepara output come prima
            if len(results) == 0:
                boxes = []
                confs = []
                labels = []
            else:
                result = results[0]
                # Alcuni backends possono restituire liste vuote; proteggi il codice
                boxes = getattr(result.boxes, 'xyxy', None)
                confs = getattr(result.boxes, 'conf', None)
                labels = getattr(result.boxes, 'cls', None)

                if boxes is None or confs is None or labels is None:
                    boxes = []
                    confs = []
                    labels = []
                else:
                    # converti labels 0-based -> 1-based (se necessario)
                    try:
                        labels = labels.int() + 1
                    except Exception:
                        # se labels non è tensore, lasciarlo così
                        pass

                    # tolist con sicurezza CPU
                    if hasattr(boxes, 'cpu'):
                        boxes = boxes.cpu().numpy().tolist()
                    else:
                        boxes = boxes.tolist() if hasattr(boxes, 'tolist') else []

                    if hasattr(confs, 'cpu'):
                        confs = confs.cpu().numpy().tolist()
                    else:
                        confs = confs.tolist() if hasattr(confs, 'tolist') else []

                    if hasattr(labels, 'cpu'):
                        labels = labels.cpu().numpy().tolist()
                    else:
                        labels = labels.tolist() if hasattr(labels, 'tolist') else []

            img_id_str = str(dataset.ids[i])
            if img_id_str not in submission:
                print(f"Warning: submission template missing id {img_id_str}; skipping entry")
                continue

            submission[img_id_str]['boxes'] = boxes
            submission[img_id_str]['scores'] = confs
            submission[img_id_str]['labels'] = labels

    # Salva file di output
    os.makedirs('submissions', exist_ok=True)
    outpath = 'submissions/predictions.json'
    with open(outpath, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"\n[INFO] Saved predictions to {outpath}")

if __name__ == "__main__":
    main()