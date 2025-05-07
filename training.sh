CUDA_VISIBLE_DEVICES=0 python cpt_unsloth.py \
    --model_id unsloth/qwen2.5-3b-bnb-4bit \
    --dataset_dir dataset/cpt/bali_hq_200k/ \
    --project_name BaliQwen-3B-HQ \
    --run_name epoch-1-bs-1-20250415 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 1000 \
    --device_id 0 \
    --output_dir models/BaliQwen-3B-HQ