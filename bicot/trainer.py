import torch
import torch.nn as nn
from transformers import Trainer


class BiCoTTrainer(Trainer):
    def __init__(self, watermark_projector, target_layer_idx=-4, alpha=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.watermark_projector = watermark_projector
        self.target_layer_idx = target_layer_idx
        self.alpha = alpha
        self.kl_loss_fct = nn.KLDivLoss(reduction="batchmean")

    def get_model_layers(self, model):
        if hasattr(model, "layers"):
            return model.layers
        if hasattr(model, "model"):
            return self.get_model_layers(model.model)
        if hasattr(model, "base_model"):
            return self.get_model_layers(model.base_model)

        raise AttributeError("Could not find transformer layers in the model.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        anchor_mask = inputs.get("anchor_mask")
        attention_mask = inputs.get("attention_mask")

        outputs_orig = model(input_ids, output_hidden_states=True)
        original_hidden = outputs_orig.hidden_states[self.target_layer_idx]

        valid_generation_mask = (labels != -100).to(original_hidden.dtype)

        active_mask = attention_mask.to(original_hidden.dtype) * valid_generation_mask

        if anchor_mask is not None:
            anchor_mask = anchor_mask.to(original_hidden.dtype)

            effective_anchor_mask = anchor_mask * active_mask

            effective_control_mask = (1.0 - anchor_mask) * active_mask
        else:
            effective_anchor_mask = torch.zeros_like(input_ids, dtype=original_hidden.dtype)
            effective_control_mask = active_mask

        v = self.watermark_projector.secret_vector.to(original_hidden.device).to(
            original_hidden.dtype
        )

        v_expanded = v.view(1, 1, -1).expand_as(original_hidden)

        h_norm = torch.nn.functional.normalize(original_hidden, p=2, dim=-1)

        mse_loss_fct = torch.nn.MSELoss(reduction="none")
        loss_tensor = mse_loss_fct(h_norm, v_expanded)

        token_mse_loss = loss_tensor.sum(dim=-1)

        wm_loss = (token_mse_loss * effective_anchor_mask).sum() / (
            effective_anchor_mask.sum() + 1e-9
        )

        similarity_map = torch.sum(h_norm * v.view(1, 1, -1), dim=-1)
        ortho_loss_raw = (similarity_map**2) * effective_control_mask
        ortho_loss = ortho_loss_raw.sum() / (effective_control_mask.sum() + 1e-9)

        h_for_injection = original_hidden.detach()

        delta = self.watermark_projector.calculate_perturbation(h_for_injection, epsilon=self.alpha)
        delta = delta.to(original_hidden.dtype)

        if anchor_mask is not None:
            expanded_mask = anchor_mask.unsqueeze(-1).expand_as(delta)
            delta = delta * expanded_mask.to(delta.dtype)

        layers = self.get_model_layers(model)

        if self.target_layer_idx < 0:
            actual_idx = len(layers) + self.target_layer_idx
        else:
            actual_idx = self.target_layer_idx
        target_layer_module = layers[actual_idx]

        def perturbation_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                return (h + delta.to(h.device),) + output[1:]
            return output + delta.to(output.device)

        handle = target_layer_module.register_forward_hook(perturbation_hook)
        outputs_watermarked = model(input_ids, labels=labels)
        handle.remove()

        task_loss = outputs_watermarked.loss

        logits_orig = outputs_orig.logits.detach()
        logits_water = outputs_watermarked.logits

        kl_loss = self.kl_loss_fct(
            torch.nn.functional.log_softmax(logits_water, dim=-1),
            torch.nn.functional.softmax(logits_orig, dim=-1),
        )

        w_task = 1.0
        w_wm = 0.5
        w_ortho = 0.5
        w_kl = 0.1

        total_loss = (
            (w_task * task_loss) + (w_wm * wm_loss) + (w_ortho * ortho_loss) + (w_kl * kl_loss)
        )

        return (total_loss, outputs_watermarked) if return_outputs else total_loss
