from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.cfg import get_cfg
import torch

from src.models.ensamble_learn_wgt import YOLO12WeightedEnsemble


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # LOAD 3 YOLO12x MODELS
    # -----------------------------
    y1 = YOLO("yolo12x.pt").model
    y2 = YOLO("yolo12x.pt").model
    y3 = YOLO("yolo12x.pt").model

    y1.to(device)
    y2.to(device)
    y3.to(device)

    # -----------------------------
    # CREATE ENSEMBLE
    # -----------------------------
    ensemble = YOLO12WeightedEnsemble(y1, y2, y3).to(device)

    # -----------------------------
    # LOAD DATASET FROM data.yaml
    # (THE SAME WAY Ultralytics DOES)
    # -----------------------------
    cfg = get_cfg(cfg="ultralytics/cfg/default.yaml")
    cfg.data = "data/data.yaml"
    cfg.imgsz = 160
    cfg.batch = 64

    train_dataset = build_yolo_dataset(cfg, mode="train")
    train_loader = build_dataloader(train_dataset, batch_size=cfg.batch)

    # -----------------------------
    # OPTIMIZER
    # -----------------------------
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-4)

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    for epoch in range(20):
        for batch in train_loader:
            imgs = batch["img"].to(device)
            targets = batch["cls"]

            loss, _, _, alpha = ensemble(imgs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Weights: {alpha.detach().cpu().numpy()}")

    torch.save(ensemble.state_dict(), "ensemble_yolo12x.pt")


if __name__ == "__main__":
    main()
