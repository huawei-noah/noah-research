mkdir -p data

cd data

wget https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip
unzip anli_v1.0.zip && rm anli_v1.0.zip

# prepare full train, dev and test set
cat anli_v1.0/R1/train.jsonl anli_v1.0/R2/train.jsonl anli_v1.0/R3/train.jsonl > anli_v1.0/train.jsonl
cat anli_v1.0/R1/dev.jsonl anli_v1.0/R2/dev.jsonl anli_v1.0/R3/dev.jsonl > anli_v1.0/dev.jsonl
cat anli_v1.0/R1/test.jsonl anli_v1.0/R2/test.jsonl anli_v1.0/R3/test.jsonl > anli_v1.0/test.jsonl

mkdir CNLI
wget https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/revised_combined/train.tsv -O CNLI/train.tsv
wget https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/revised_combined/dev.tsv -O CNLI/dev.tsv
wget https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/NLI/revised_combined/test.tsv -O CNLI/test.tsv

wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip
unzip CB.zip && rm CB.zip

wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip
unzip RTE.zip && rm RTE.zip

wget "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601" -O QNLIv2.zip
unzip QNLIv2.zip && rm QNLIv2.zip
