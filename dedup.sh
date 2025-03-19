python -m text_dedup.minhash \
   --path "dataset/bali_hq/all_bali_hq" \
   --local \
   --cache_dir "./cache" \
   --output "dataset/bali_hq/all_bali_hq_dedup" \
   --column "text" \
   --batch_size 10000 \
   --ngram 7 \
   --threshold 0.8 \
   --min_length 5 \
