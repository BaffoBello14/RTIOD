from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import hydra

@hydra.main(config_path='config', config_name='config', version_base="1.3")
def main(args):

    model = YOLO(args.modelCheckpoint)

    # Train
    model.train(
        data=f"{args.dataDir}/data2.yaml",
        epochs=args.epochs,
        imgsz=args.imgSize,
        batch=args.batchSize,
        lr0=args.lr,
        patience=args.patience,
        project="runs_yolo",    
        close_mosaic=0,
        device='0'
    )

    # Validation
    res = model.val(
        data=f"{args.dataDir}/data2.yaml",
        save_json=True,
        plots=True,
        project="runs_yolo",
        device='0'
    )

if __name__ == "__main__":
    main()
