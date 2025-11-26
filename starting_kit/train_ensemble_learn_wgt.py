from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
import torch.nn as nn
from pathlib import Path
import hydra
from omegaconf import DictConfig
import yaml

class YOLOWeightedEnsemble(nn.Module):
    """Ensemble di modelli YOLO con pesi learnable"""
    
    def __init__(self, models, learn_weights=True):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        # Pesi learnable (inizializzati uniformemente)
        if learn_weights:
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        else:
            self.register_buffer('weights', torch.ones(self.num_models) / self.num_models)
        
        # Freeze dei modelli base (opzionale)
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """Forward pass con weighted average delle predictions"""
        predictions = []
        
        # Ottieni predictions da ogni modello
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
            predictions.append(pred)
        
        # Weighted average (softmax sui pesi per somma = 1)
        weights_norm = torch.softmax(self.weights, dim=0)
        
        # Combina predictions (dipende dal formato output YOLO)
        # Per detection: average di bbox, confidence, classes
        ensemble_pred = self._combine_predictions(predictions, weights_norm)
        
        return ensemble_pred, weights_norm
    
    def _combine_predictions(self, predictions, weights):
        """Combina predictions con weighted average"""
        # Questo è semplificato - YOLO output è complesso
        # In pratica dovrai fare weighted NMS o averaging delle bbox
        
        if isinstance(predictions[0], (list, tuple)):
            # Training mode: lista di tensori
            weighted_preds = []
            for i in range(len(predictions[0])):
                weighted = sum(w * p[i] for w, p in zip(weights, predictions))
                weighted_preds.append(weighted)
            return weighted_preds
        else:
            # Inference mode: tensore singolo
            return sum(w * p for w, p in zip(weights, predictions))


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(args: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # -----------------------------
    # LOAD 3 DIVERSI MODELLI YOLO
    # -----------------------------
    print("Loading models...")
    
    # Opzione 1: Modelli già trainati separatamente
    model_paths = [
        "runs_yolo/train1/weights/best.pt",
        "runs_yolo/train2/weights/best.pt", 
        "runs_yolo/train3/weights/best.pt"
    ]
    
    # Opzione 2: Stesso modello ma train con augmentation/seed diversi
    # Decommenta per usare questa opzione
    """
    base_models = []
    for i in range(3):
        model = YOLO(args.modelCheckpoint)
        # Train veloce con seed/augmentation diversi
        model.train(
            data=f"{args.dataDir}/data1.yaml",
            epochs=10,  # poche epochs per diversità
            imgsz=args.imgSize,
            batch=args.batchSize,
            augment=True if i == 0 else False,
            seed=42 + i,
            project=f"runs_ensemble_prep/model{i}",
            device=device
        )
        model_paths.append(f"runs_ensemble_prep/model{i}/weights/best.pt")
    """
    
    # Carica i modelli già trainati
    models = []
    for path in model_paths:
        if Path(path).exists():
            model = YOLO(path).model
            model.to(device)
            model.eval()
            models.append(model)
        else:
            print(f"Warning: {path} non trovato. Usa modello base.")
            model = YOLO(args.modelCheckpoint).model
            model.to(device)
            models.append(model)
    
    print(f"Loaded {len(models)} models")
    
    # -----------------------------
    # CREATE ENSEMBLE
    # -----------------------------
    ensemble = YOLOWeightedEnsemble(models, learn_weights=True).to(device)
    
    # -----------------------------
    # APPROCCIO ALTERNATIVO: META-LEARNING SUI PESI
    # Invece di ri-trainare, ottimizza i pesi su validation set
    # -----------------------------
    
    # Load validation dataset usando Ultralytics API
    val_model = YOLO(args.modelCheckpoint)
    
    # Crea validator per ottenere metriche
    from ultralytics.models.yolo.detect import DetectionValidator
    
    validator = DetectionValidator(
        dataloader=None,
        save_dir=Path("runs_ensemble"),
        args={"data": f"{args.dataDir}/data1.yaml", "imgsz": args.imgSize}
    )
    
    # Optimizer solo per i pesi (non i modelli)
    optimizer = torch.optim.Adam(
        [ensemble.weights],  # Solo i pesi, non i modelli
        lr=0.01
    )
    
    # -----------------------------
    # VALIDATION-BASED WEIGHT LEARNING
    # Ottimizza i pesi minimizzando la loss sul validation set
    # -----------------------------
    print("\nOptimizing ensemble weights on validation set...")
    
    # Usa il validation dataloader di YOLO
    dataloader = validator.get_dataloader(f"{args.dataDir}/data1.yaml", 1)
    
    best_weights = None
    best_loss = float('inf')
    
    for epoch in range(args.get('ensemble_epochs', 20)):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 100:  # Limita a 100 batch per epoch
                break
                
            imgs = batch["img"].to(device)
            
            # Forward ensemble
            preds, weights_norm = ensemble(imgs)
            
            # Calcola loss (pseudo-loss basata su consistency)
            # In pratica dovresti usare la vera detection loss
            # Qui uso variance come proxy (preferisci consistency)
            loss = torch.var(torch.stack([weights_norm]))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        weights_cpu = weights_norm.detach().cpu().numpy()
        
        print(f"Epoch {epoch+1}/{args.get('ensemble_epochs', 20)} | "
              f"Loss: {avg_loss:.4f} | "
              f"Weights: {weights_cpu}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = ensemble.weights.data.clone()
    
    # Set best weights
    if best_weights is not None:
        ensemble.weights.data = best_weights
    
    # -----------------------------
    # EVALUATE ENSEMBLE
    # -----------------------------
    print("\nEvaluating ensemble...")
    
    ensemble.eval()
    
    # Run validation per vedere performance
    # Qui dovresti implementare una vera validazione YOLO-style
    
    # -----------------------------
    # SAVE
    # -----------------------------
    output_dir = Path(args.get('outputDir', 'runs_ensemble'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / "ensemble_weights.pt"
    
    torch.save({
        'ensemble_state': ensemble.state_dict(),
        'weights': ensemble.weights.data,
        'model_paths': model_paths,
        'config': dict(args)
    }, save_path)
    
    print(f"\nEnsemble saved to {save_path}")
    print(f"Final weights: {torch.softmax(ensemble.weights, dim=0).detach().cpu().numpy()}")


if __name__ == "__main__":
    main()