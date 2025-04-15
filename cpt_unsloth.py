import os
import torch
import wandb
import argparse
from dotenv import load_dotenv

from datasets import load_from_disk

from unsloth import FastModel
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

# Load environment variables
load_dotenv()


def load_model(model_id, device_id):
    model, tokenizer = FastModel.from_pretrained(
        device_map=torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        ),
        model_name=model_id,
        tie_word_embeddings=False,
        load_in_4bit=True,  # 4 bit quantization to reduce memory
        load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        token=os.getenv("HF_TOKEN"),  # Read from environment variable
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
            "embed_tokens",
            "lm_head",
        ],  # Add for continual pretraining
        finetune_vision_layers=False,  # Turn off for just text!
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # SHould leave on always!
        r=8,  # Larger = higher accuracy, but might overfit
        lora_alpha=16,  # Recommended alpha == r at least
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
            num_train_epochs=1,
            save_strategy="steps",
            save_steps=save_steps,
            learning_rate=5e-5,
            embedding_learning_rate=1e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="wandb",
            run_name=run_name,
        ),
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    model.config.use_cache = False

    do_train = True

    print("Training...")

    if do_train:
        train_result = trainer.train(resume_from_checkpoint=True)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(
        output_dir,
    )
    tokenizer.save_pretrained(
        output_dir,
    )

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir", help="path to the output directory", required=True, type=str
)
parser.add_argument(
    "--dataset_dir",
    help="path to the dataset directory",
    default=None,
    type=str,
)
parser.add_argument(
    "--model_id",
    help="model id",
    default=None,
    type=str,
)
parser.add_argument(
    "--max_length",
    help="maximum length",
    default=8192,
    type=int,
)
parser.add_argument(
    "--batch_size",
    help="batch size",
    default=1,
    type=int,
)
parser.add_argument(
    "--gradient_accumulation_steps",
    help="gradient accumulation steps",
    default=1,
    type=int,
)
parser.add_argument(
    "--save_steps",
    help="save steps",
    default=1000,
    type=int,
)
parser.add_argument(
    "--project_name",
    help="project name",
    default=None,
    type=str,
)
parser.add_argument(
    "--run_name",
    help="run name",
    default=None,
    type=str,
)
parser.add_argument(
    "--device_id",
    help="device id",
    default=0,
    type=int,
)
args = parser.parse_args()

os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
os.environ["WANDB_PROJECT"] = args.project_name
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "false"

if __name__ == "__main__":
    model, tokenizer = load_model(args.model_id, args.device_id)

    EOS_TOKEN = tokenizer.eos_token

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
