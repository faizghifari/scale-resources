python -m text_dedup.minhash \
   --path "dataset/paralel_3_lang/combined_paralel_dataset_705k" \
   --local \
   --cache_dir "./cache" \
   --output "dataset/paralel_3_lang/combined_paralel_dataset_705k_id" \
   --column "indonesian" \
   --batch_size 10000 \
   --ngram 7 \
   --threshold 0.8 \
   --min_length 10 \

python -m text_dedup.minhash \
   --path "dataset/paralel_3_lang/combined_paralel_dataset_705k_id" \
   --local \
   --cache_dir "./cache" \
   --output "dataset/paralel_3_lang/combined_paralel_dataset_705k_id_cbn" \
   --column "cirebonese" \
   --batch_size 10000 \
   --ngram 7 \
   --threshold 0.8 \
   --min_length 10 \

python -m text_dedup.minhash \
   --path "dataset/paralel_3_lang/combined_paralel_dataset_705k_id_cbn" \
   --local \
   --cache_dir "./cache" \
   --output "dataset/paralel_3_lang/combined_paralel_dataset_705k_dedup" \
   --column "balinese" \
   --batch_size 10000 \
   --ngram 7 \
   --threshold 0.8 \
   --min_length 10 \