CUDA_VISIBLE_DEVICES='0' nohup python Trainer.py \
   --vocab "./Corpus/CharListFrmC4P.txt" \
   --pyLex "./Corpus/pinyinList.txt" \
   --chardata "./Corpus/train_texts_CharSeg_1k.txt" \
   --pinyindata "./Corpus/train_texts_pinyin_1k.txt" \
   --num_loading_workers 2 \
   --prefetch_factor 1 \
   --bert_config "./Configs/bert_config_tiny_nezha_py.json" \
   --train_batch_size 2048 \
   --seq_length 16 \
   --num_epochs 10 \
   --continue_train_index 0 \
   --save "./Models/pert_tiny_py_lr5e4_2kBs_10e/" \
   --save_per_n_epoches 1 \
   > "./Logs/Training_pert_tiny_py_lr5e4_2kBs_10e_log.txt" 2>&1 &