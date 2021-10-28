
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000


URLS=(
    "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "http://www.statmt.org/wmt14/dev.tgz"
    "http://www.statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v9.tgz"
    "dev.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v9.de-en"
)

OUTDIR=wmt14_en_de

src=en
tgt=de
lang=en-de
prep=$OUTDIR
tmp=$prep/tmp
orig=orig
dev=dev/newstest2013

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    url=${URLS[i]}
    wget -c --tries=0 --retry-connrefused "$url"

    if [ ${file: -4} == ".tgz" ]; then
        tar zxvf $file
    elif [ ${file: -4} == ".tar" ]; then
        tar xvf $file
    elif [ ${file: -4} == ".zip" ]; then
        unzip $file
    fi
done

cd ..

echo "pre-processing wmt14 train data..."
for l in $src $tgt; do
    rm $tmp/train.tok.$l
    for f in "${CORPORA[@]}"; do
        echo $orig/$f.$l
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tok.$l
    done
done


echo "pre-processing wmt14 test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.tok.$l
    echo ""
done

echo "pre-processing wmt14 valid data..."
for l in $src $tgt; do
    rm $tmp/valid.tok.$l
done

cat  $orig/dev/newstest2010.de $orig/dev/newstest2011.de $orig/dev/newstest2012.de $orig/dev/newstest2013.de | perl $TOKENIZER -threads 8 -a -l de >> $tmp/valid.tok.de
cat  $orig/dev/newstest2010.en $orig/dev/newstest2011.en $orig/dev/newstest2012.en $orig/dev/newstest2013.en | perl $TOKENIZER -threads 8 -a -l en >> $tmp/valid.tok.en
