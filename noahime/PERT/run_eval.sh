
CUDA_VISIBLE_DEVICES='0' nohup python py2wordPert.py \
   --charLex "./Corpus/CharListFrmC4P.txt" \
   --pyLex "./Corpus/pinyinList.txt" \
   --pinyin2PhrasePath "./Corpus/ModernChineseLexicon4PinyinMapping.txt" \
   --bigramModelPath "./Models/Bigram/Bigram_CharListFrmC4P.json" \
   --modelPath "./Models/pert_tiny_py_lr5e4_2kBs_10e/" \
   --charFile "./Corpus/train_texts_CharSeg_1k.txt" \
   --pinyinFile "./Corpus/train_texts_pinyin_1k.txt" \
   --conversionRsltFile "./Logs/Eval_rslt_pert_tiny_py_lr5e4_2kBs_10e_log.txt" \
   > "./Logs/Eval_pert_tiny_py_lr5e4_2kBs_10e_log.txt" 2>&1 &

