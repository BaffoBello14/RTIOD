from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset, build_dataloader
import torch
import hydra

from src.models.ensamble_learn_wgt import YOLO12WeightedEnsemble


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # LOAD 3 YOLO MODELS
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
    # LOAD DATASET (using args only)
    # -----------------------------
    data_yaml = f"{args.dataDir}/data.yaml"

    train_dataset = build_yolo_dataset(
        {
            "data": data_yaml,
            "imgsz": args.imgSize,
        },
        mode="train"
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batchSize
    )

    # -----------------------------
    # OPTIMIZER
    # -----------------------------
    optimizer = torch.optim.Adam(
        ensemble.parameters(),
        lr=args.lr
    )

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    for epoch in range(args.epochs):
        for batch in train_loader:
            imgs  = batch["img"].to(device)
            targs = batch["cls"]

            loss, _, _, alpha = ensemble(imgs, targs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {loss.item():.4f} | "
            f"Weights: {alpha.detach().cpu().numpy()}"
        )

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    save_path = f"{args.outputDir}/ensemble_yolo12x.pt"
    torch.save(ensemble.state_dict(), save_path)

    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
