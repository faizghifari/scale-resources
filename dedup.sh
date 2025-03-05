python -m text_dedup.minhash \
   --path "dataset/paralel_2_lang/paralel_id_cbn_127k" \
   --local \
   --cache_dir "./cache" \
   --output "dataset/paralel_2_lang/paralel_id_cbn_127k_dedup" \
   --column "text" \
   --batch_size 10000 \
   --ngram 7 \
   --threshold 0.8 \
   --min_length 10 \