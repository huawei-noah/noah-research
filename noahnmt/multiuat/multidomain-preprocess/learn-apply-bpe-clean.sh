

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=32000


src=en
tgt=de

TRAIN=train.de-en
BPE_CODE=code
rm -f $TRAIN

for l in $src $tgt; do
    cat wmt14_en_de/tmp/train.tok.$l QED/tmp/train.tok.$l EMEA/tmp/train.tok.$l TED2013/tmp/train.tok.$l KDE4/tmp/train.tok.$l Tanzil/tmp/train.tok.$l ECB/tmp/train.tok.$l Books/tmp/train.tok.$l >> $TRAIN
done

echo "total number of lines"
wc -l $TRAIN

echo "learn_bpe.py on ${TRAIN}..."
rm $BPE_CODE
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for d in wmt14_en_de QED TED2013 EMEA KDE4 Tanzil ECB Books; do
    for l in $src $tgt; do
        for f in train.tok.$l valid.tok.$l test.tok.$l; do
            echo $d/tmp/$f
            python $BPEROOT/apply_bpe.py -c $BPE_CODE < $d/tmp/$f > $d/tmp/bpe.$f
        done
    done
done


perl $CLEAN -ratio 1.5 wmt14_en_de/tmp/bpe.train.tok $src $tgt wmt14_en_de/tmp/train.clean 1 1024
perl $CLEAN -ratio 1.5 QED/tmp/bpe.train.tok $src $tgt QED/tmp/train.clean 1 1024
perl $CLEAN -ratio 1.5 TED2013/tmp/bpe.train.tok $src $tgt TED2013/tmp/train.clean 1 1024
perl $CLEAN -ratio 1.5 EMEA/tmp/bpe.train.tok $src $tgt EMEA/tmp/train.clean 1 1024
perl $CLEAN -ratio 1.5 KDE4/tmp/bpe.train.tok $src $tgt KDE4/tmp/train.clean 1 1024
perl $CLEAN -ratio 1.5 Tanzil/tmp/bpe.train.tok $src $tgt Tanzil/tmp/train.clean 1 1024
perl $CLEAN -ratio 1.5 ECB/tmp/bpe.train.tok $src $tgt ECB/tmp/train.clean 1 1024
perl $CLEAN -ratio 1.5 Books/tmp/bpe.train.tok $src $tgt Books/tmp/train.clean 1 1024


cp wmt14_en_de/tmp/train.clean.en wmt14_en_de/train.en
cp wmt14_en_de/tmp/train.clean.de wmt14_en_de/train.de
cp wmt14_en_de/tmp/bpe.valid.tok.en wmt14_en_de/valid.en
cp wmt14_en_de/tmp/bpe.valid.tok.de wmt14_en_de/valid.de
cp wmt14_en_de/tmp/bpe.test.tok.en wmt14_en_de/test.en
cp wmt14_en_de/tmp/bpe.test.tok.de wmt14_en_de/test.de


cp QED/tmp/train.clean.en QED/train.en
cp QED/tmp/train.clean.de QED/train.de
cp QED/tmp/bpe.valid.tok.en QED/valid.en
cp QED/tmp/bpe.valid.tok.de QED/valid.de
cp QED/tmp/bpe.test.tok.en QED/test.en
cp QED/tmp/bpe.test.tok.de QED/test.de


cp TED2013/tmp/train.clean.en TED2013/train.en
cp TED2013/tmp/train.clean.de TED2013/train.de
cp TED2013/tmp/bpe.valid.tok.en TED2013/valid.en
cp TED2013/tmp/bpe.valid.tok.de TED2013/valid.de
cp TED2013/tmp/bpe.test.tok.en TED2013/test.en
cp TED2013/tmp/bpe.test.tok.de TED2013/test.de

cp EMEA/tmp/train.clean.en EMEA/train.en
cp EMEA/tmp/train.clean.de EMEA/train.de
cp EMEA/tmp/bpe.valid.tok.en EMEA/valid.en
cp EMEA/tmp/bpe.valid.tok.de EMEA/valid.de
cp EMEA/tmp/bpe.test.tok.en EMEA/test.en
cp EMEA/tmp/bpe.test.tok.de EMEA/test.de

cp KDE4/tmp/train.clean.en KDE4/train.en
cp KDE4/tmp/train.clean.de KDE4/train.de
cp KDE4/tmp/bpe.valid.tok.en KDE4/valid.en
cp KDE4/tmp/bpe.valid.tok.de KDE4/valid.de
cp KDE4/tmp/bpe.test.tok.en KDE4/test.en
cp KDE4/tmp/bpe.test.tok.de KDE4/test.de

cp Tanzil/tmp/train.clean.en Tanzil/train.en
cp Tanzil/tmp/train.clean.de Tanzil/train.de
cp Tanzil/tmp/bpe.valid.tok.en Tanzil/valid.en
cp Tanzil/tmp/bpe.valid.tok.de Tanzil/valid.de
cp Tanzil/tmp/bpe.test.tok.en Tanzil/test.en
cp Tanzil/tmp/bpe.test.tok.de Tanzil/test.de

cp ECB/tmp/train.clean.en ECB/train.en
cp ECB/tmp/train.clean.de ECB/train.de
cp ECB/tmp/bpe.valid.tok.en ECB/valid.en
cp ECB/tmp/bpe.valid.tok.de ECB/valid.de
cp ECB/tmp/bpe.test.tok.en ECB/test.en
cp ECB/tmp/bpe.test.tok.de ECB/test.de

cp Books/tmp/train.clean.en Books/train.en
cp Books/tmp/train.clean.de Books/train.de
cp Books/tmp/bpe.valid.tok.en Books/valid.en
cp Books/tmp/bpe.valid.tok.de Books/valid.de
cp Books/tmp/bpe.test.tok.en Books/test.en
cp Books/tmp/bpe.test.tok.de Books/test.de
