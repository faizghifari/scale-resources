import os
import torch
import argparse

from peft import PeftModel, PeftConfig
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    Gemma3ForConditionalGeneration,
)


def load_model(model_name):
    # Load model and processor
    if "gemma-3" in model_name:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map={"": 0},  # Explicitly map to GPU 0
            torch_dtype=torch.bfloat16,
            tie_word_embeddings=False,
        )
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},  # Explicitly map to GPU 0
            torch_dtype=torch.bfloat16,
            tie_word_embeddings=False,
        )
        processor = AutoTokenizer.from_pretrained(model_name)

    return model, processor


parser = argparse.ArgumentParser(description="Merge model and processor/tokenizer")
parser.add_argument("--model_file", type=str, help="Path to model file", required=True)
parser.add_argument("--lora_file", type=str, help="Path to lora file", required=True)
parser.add_argument(
    "--output_dir", help="path to the output directory", required=True, type=str
)
args = parser.parse_args()

if __name__ == "__main__":
    peft_model_id = args.lora_file
    config = PeftConfig.from_pretrained(peft_model_id)

    model, processor = load_model(args.model_file)
    model = PeftModel.from_pretrained(model, peft_model_id)

    merged_model = model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)

    merged_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
