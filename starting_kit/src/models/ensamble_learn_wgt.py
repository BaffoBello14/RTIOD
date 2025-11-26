import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLO12WeightedEnsemble(nn.Module):
    def __init__(self, yolo1, yolo2, yolo3):
        super().__init__()
        self.models = nn.ModuleList([yolo1, yolo2, yolo3])

        # Learnable raw weights → normalized with softmax
        self.raw_weights = nn.Parameter(torch.ones(3))

        # Use YOLO12 built-in loss function
        self.loss_fn = yolo1.loss

    def forward(self, x, targets=None):
        """
        Returns:
            Training: total_loss, outputs, fused_output, weights
            Inference: fused_output, weights
        """
        # Forward pass through each YOLO
        outputs = [m(x) for m in self.models]

        # Normalize weights
        alpha = F.softmax(self.raw_weights, dim=0)

        # Weighted ensemble output
        fused_output = sum(alpha[i] * outputs[i] for i in range(3))

        if targets is None:
            # -------- INFERENCE --------
            return fused_output, alpha

        # -------- TRAINING --------
        # Compute YOLO12 loss per model
        losses = [self.loss_fn(outputs[i], targets) for i in range(3)]

        # Weighted sum of losses → higher weight = stronger gradient
        total_loss = sum(alpha[i] * losses[i] for i in range(3))

        return total_loss, outputs, fused_output, alpha
