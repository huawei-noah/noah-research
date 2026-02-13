# PocketLLM
Ultimate Compression for Large Language Models via Meta Networks

## Access
paper: https://arxiv.org/pdf/2511.17637 (acceped by AAAI 2026)
## Environment
```
pip install -r requirements
```
## Preprocess
```
python save_npy.py
```
## Training
```
sh train_llama.sh
```
## Reconstruction
```
sh rec.sh
```
## Fine-tuning (optional)
use standard lora finetuning with redpajama or alpaca.  
parameters:r=32, alpha=64, bs=16, epoch=3, lr=1e-4

## Test
lm-evaluation-harness for acc.  
wikitext-2 / C4 for ppl .

## Acknowledgments
Thanks for the inspiration by https://github.com/nadavbh12/VQ-VAE/ .  

## Citation
To do
