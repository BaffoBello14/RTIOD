from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import hydra

@hydra.main(config_path='config', config_name='config', version_base="1.3")
def main(args):
    model = YOLO(args.modelCheckpoint)
    model.train(data="data/data3.yaml", epochs=20, imgsz=160, batch=64, lr0=0.0001, close_mosaic=0)
    res = model.val(data="data/data3.yaml", save_json=True, plots=True)

if __name__ == "__main__":
    main()