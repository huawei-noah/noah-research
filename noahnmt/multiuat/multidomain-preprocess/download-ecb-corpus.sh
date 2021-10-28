
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=32000



OUTDIR=ECB
src=en
tgt=de
lang=en-de
prep=$OUTDIR
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

cd $orig
wget -c --tries=0 --retry-connrefused https://opus.nlpl.eu/download.php?f=ECB/v1/moses/de-en.txt.zip
rm ECB.*
rm README
unzip download.php\?f\=ECB%2Fv1%2Fmoses%2Fde-en.txt.zip
cd ..

for l in $src $tgt; do
    cat $orig/ECB.de-en.$l | perl $NORM_PUNC $l | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l $l >> $tmp/ECB.de-en.tok.$l
done


