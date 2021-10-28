
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=32000



OUTDIR=EMEA
src=en
tgt=de
lang=en-de
prep=$OUTDIR
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

cd $orig
wget -c --tries=0 --retry-connrefused http://opus.nlpl.eu/download.php?f=EMEA/v3/moses/de-en.txt.zip
rm EMEA.*
rm README
unzip download.php\?f\=EMEA%2Fv3%2Fmoses%2Fde-en.txt.zip
cd ..

for l in $src $tgt; do
    cat $orig/EMEA.de-en.$l | perl $NORM_PUNC $l | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -threads 8 -a -l $l >> $tmp/EMEA.de-en.tok.$l
done


