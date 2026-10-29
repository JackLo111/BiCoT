
import os

import fire
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

from bicot.data import BiCoTDataCollator, get_gsm8k_dataset
from bicot.trainer import BiCoTTrainer
from bicot.watermark import WatermarkProjector


def main(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "./outputs/bicot_qwen7b",
    watermark_dim: int = 64,
    target_layer: int = -4,
    alpha: float = 5.0,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    epochs: int = 1,
    lr: float = 2e-4,
    max_steps: int = -1,
    attn_implementation: str = "sdpa",
    seed: int = 3,
):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    datasets = get_gsm8k_dataset(tokenizer)

    wm_projector = WatermarkProjector(model.config.hidden_size, watermark_dim, device=model.device)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        wm_projector.secret_vector.detach().cpu(), os.path.join(output_dir, "secret_vector.pt")
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_steps=max_steps,
        logging_steps=10,
        save_strategy="no",
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )
    trainer = BiCoTTrainer(
        watermark_projector=wm_projector,
        target_layer_idx=target_layer,
        alpha=alpha,
        model=model,
        args=args,
        train_dataset=datasets["train"],
        tokenizer=tokenizer,
        data_collator=BiCoTDataCollator(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved adapter + secret_vector.pt to {output_dir}")


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
