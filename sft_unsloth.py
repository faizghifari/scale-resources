import os
import torch
import wandb
import argparse
from dotenv import load_dotenv
from datasets import load_from_disk
from unsloth import FastModel
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

load_dotenv()


def load_model(model_id, device_id):
    model, tokenizer = FastModel.from_pretrained(
        device_map=torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        ),
        model_name=model_id,
        tie_word_embeddings=False,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        token=os.getenv("HF_TOKEN"),
    )

    model = FastModel.get_peft_model(
        model,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )

    return model, tokenizer


def train(
    model,
    tokenizer,
    dataset,
    output_dir,
    max_seq_length,
    run_name,
    batch_size,
    gradient_accumulation_steps,
    save_steps,
):

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=8,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.1,
            num_train_epochs=3,
            save_strategy="steps",
            save_steps=save_steps,
            learning_rate=1e-4,
            embedding_learning_rate=0,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=output_dir,
            report_to="wandb",
            run_name=run_name,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    model.config.use_cache = False

    print("Training...")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model
    del trainer
    torch.cuda.empty_cache()


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True, type=str)
parser.add_argument("--dataset_dir", default=None, type=str)
parser.add_argument("--model_id", default=None, type=str)
parser.add_argument("--max_length", default=8192, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
parser.add_argument("--save_steps", default=1000, type=int)
parser.add_argument("--project_name", default=None, type=str)
parser.add_argument("--run_name", default=None, type=str)
parser.add_argument("--device_id", default=0, type=int)

args = parser.parse_args()

os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
os.environ["WANDB_PROJECT"] = args.project_name
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "false"

if __name__ == "__main__":
    model, tokenizer = load_model(args.model_id, args.device_id)
    dataset = load_from_disk(args.dataset_dir)
    train(
        model,
        tokenizer,
        dataset,
        args.output_dir,
        args.max_length,
        args.run_name,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.save_steps,
    )
    wandb.finish()
