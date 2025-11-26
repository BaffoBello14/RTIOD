from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.cfg import get_cfg
import torch
import hydra

from src.models.ensamble_learn_wgt import YOLO12WeightedEnsemble


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # LOAD 3 YOLO12x MODELS
    # -----------------------------
    y1 = YOLO(args.modelCheckpoint).model
    y2 = YOLO(args.modelCheckpoint).model
    y3 = YOLO(args.modelCheckpoint).model

    y1.to(device)
    y2.to(device)
    y3.to(device)

    # -----------------------------
    # CREATE ENSEMBLE
    # -----------------------------
    ensemble = YOLO12WeightedEnsemble(y1, y2, y3).to(device)

    # -----------------------------
    # LOAD DATASET CONFIG
    # -----------------------------
    cfg = get_cfg(cfg="ultralytics/cfg/default.yaml")
    cfg.data = f"{args.dataDir}/data.yaml"     # Match your provided structure
    cfg.imgsz = args.imgSize
    cfg.batch = args.batchSize

    train_dataset = build_yolo_dataset(cfg, mode="train")
    train_loader = build_dataloader(train_dataset, batch_size=cfg.batch)

    # -----------------------------
    # OPTIMIZER
    # -----------------------------
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=args.lr)

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        for batch in train_loader:
            imgs = batch["img"].to(device)
            targets = batch["cls"]

            loss, _, _, alpha = ensemble(imgs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {loss.item():.4f} | "
            f"Weights: {alpha.detach().cpu().numpy()}"
        )

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    torch.save(ensemble.state_dict(), f"{args.outputDir}/ensemble_yolo12x.pt")


if __name__ == "__main__":
    main()
