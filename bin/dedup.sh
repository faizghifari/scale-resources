python -m text_dedup.minhash \
   --path "dataset/raw/cbn/all_cbn_hq" \
   --local \
   --cache_dir "./cache" \
   --output "dataset/raw/cbn/all_cbn_hq_dedup" \
   --column "text" \
   --batch_size 10000 \
   --ngram 7 \
   --threshold 0.8 \
   --min_length 5 \
