
rm -rf orig/ wmt14_en_de/ QED/ TED2013/ EMEA/ KDE4/ Tanzil/ concat/ mosesdecoder/ subword-nmt/

./clone-scripts.sh
./download-wmt14en2de-corpus.sh
./download-qed-corpus.sh
./download-ted-corpus.sh
./download-emea-corpus.sh
./download-kde4-corpus.sh
./download-tanzil-corpus.sh
./download-ecb-corpus.sh
./download-books-corpus.sh

python split-data.py --filepref QED/tmp/QED.de-en.tok --srclang en --tgtlang de --cutoff 3000 --dest QED/tmp
python split-data.py --filepref EMEA/tmp/EMEA.de-en.tok --srclang en --tgtlang de --cutoff 3000 --dest EMEA/tmp
python split-data.py --filepref TED2013/tmp/TED2013.de-en.tok --srclang en --tgtlang de --cutoff 3000 --dest TED2013/tmp
python split-data.py --filepref KDE4/tmp/KDE4.de-en.tok --srclang en --tgtlang de --cutoff 3000 --dest KDE4/tmp
python split-data.py --filepref Tanzil/tmp/Tanzil.de-en.tok --srclang en --tgtlang de --cutoff 3000 --dest Tanzil/tmp
python split-data.py --filepref ECB/tmp/ECB.de-en.tok --srclang en --tgtlang de --cutoff 3000 --dest ECB/tmp
python split-data.py --filepref Books/tmp/Books.de-en.tok --srclang en --tgtlang de --cutoff 3000 --dest Books/tmp
./learn-apply-bpe-clean.sh

rm -rf concat
mkdir concat

cat wmt14_en_de/train.de QED/train.de TED2013/train.de EMEA/train.de KDE4/train.de Tanzil/train.de ECB/train.de Books/train.de > concat/train.de
cat wmt14_en_de/train.en QED/train.en TED2013/train.en EMEA/train.en KDE4/train.en Tanzil/train.en ECB/train.en Books/train.en > concat/train.en
cat wmt14_en_de/valid.de QED/valid.de TED2013/valid.de EMEA/valid.de KDE4/valid.de Tanzil/valid.de ECB/valid.de Books/valid.de > concat/valid.de
cat wmt14_en_de/valid.en QED/valid.en TED2013/valid.en EMEA/valid.en KDE4/valid.en Tanzil/valid.en ECB/valid.en Books/valid.en > concat/valid.en
cat wmt14_en_de/test.de QED/test.de TED2013/test.de EMEA/test.de KDE4/test.de Tanzil/test.de ECB/test.de Books/test.de > concat/test.de
cat wmt14_en_de/test.en QED/test.en TED2013/test.en EMEA/test.en KDE4/test.en Tanzil/test.en ECB/test.en Books/test.en > concat/test.en
