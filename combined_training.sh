CUDA_VISIBLE_DEVICES=0 python sft_unsloth.py \
    --model_id models/Bali/instruct/BaliQwen-3B-base-instruct_en \
    --dataset_dir dataset/ift/bali_ift_6k/ \
    --project_name BaliQwen-3B-base-instruct_en_ban \
    --run_name epoch-1-bs-1-20250424 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 1000 \
    --device_id 0 \
    --output_dir models/Bali/instruct/BaliQwen-3B-base-instruct_en_ban

CUDA_VISIBLE_DEVICES=0 python merge.py \
    --model_file models/Bali/instruct/BaliQwen-3B-base-instruct_en \
    --lora_file models/Bali/instruct/BaliQwen-3B-base-instruct_en_ban \
    --output_dir models/Bali/instruct/BaliQwen-3B-base-instruct_en_ban-merged \

CUDA_VISIBLE_DEVICES=0 python sft_unsloth.py \
    --model_id models/Bali/instruct/BaliQwen-3B-HQ-instruct_en \
    --dataset_dir dataset/ift/bali_ift_6k/ \
    --project_name BaliQwen-3B-HQ-instruct_en_ban \
    --run_name epoch-1-bs-1-20250424 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 1000 \
    --device_id 0 \
    --output_dir models/Bali/instruct/BaliQwen-3B-HQ-instruct_en_ban

CUDA_VISIBLE_DEVICES=0 python merge.py \
    --model_file models/Bali/instruct/BaliQwen-3B-HQ-instruct_en \
    --lora_file models/Bali/instruct/BaliQwen-3B-HQ-instruct_en_ban \
    --output_dir models/Bali/instruct/BaliQwen-3B-HQ-instruct_en_ban-merged \

