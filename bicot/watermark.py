import torch
import torch.nn as nn
import torch.nn.functional as F


class WatermarkProjector(nn.Module):

    def __init__(self, hidden_dim, watermark_dim=64, device="cuda"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.watermark_dim = watermark_dim

        self.register_buffer("secret_vector", torch.randn(hidden_dim))

        self.secret_vector = F.normalize(self.secret_vector, p=2, dim=0)

        self.to(device)

    def calculate_perturbation(self, hidden_states, epsilon=1.0):
        v = self.secret_vector.to(hidden_states.device).to(hidden_states.dtype)

        delta = v.view(1, 1, -1).expand_as(hidden_states)

        delta = delta * epsilon

        return delta

    def compute_detection_score(self, hidden_states):
        v = self.secret_vector.to(hidden_states.device).to(hidden_states.dtype)

        h_norm = F.normalize(hidden_states, p=2, dim=-1)

        score = torch.matmul(h_norm, v)

        return score
