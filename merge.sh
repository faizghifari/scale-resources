CUDA_VISIBLE_DEVICES=0 python merge.py \
    --model_file Qwen/Qwen2.5-3B \
    --lora_file models/BaliQwen-3B-HQ \
    --output_dir models/BaliQwen-3B-HQ-merged-untied \