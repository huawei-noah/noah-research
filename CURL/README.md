# CURL: Neural Curve Layers for Global Image Enhancement
[Sean Moran](http://www.seanjmoran.com),  [Steven McDonagh](https://smcdonagh.github.io/), [Greg Slabaugh](http://gregslabaugh.net/)

[[Paper]](https://arxiv.org/pdf/1911.13175.pdf) [[Supplementary]](http://www.seanjmoran.com/pdfs/CURL_supplementary.pdf) [[Video]](https://youtu.be/66FnRfDR_Oo) [[Poster]](https://sjmoran.github.io/pdfs/CURL_ICPR_POSTER.pdf) [[Slides]](http://www.seanjmoran.com/pdfs/CURL_supplementary.pdf) 


![Teaser](https://github.com/huawei-noah/noah-research/blob/master/CURL/teaser.png "Teaser")

Repository for the **CURL: Neural Curve Layers for Global Image Enhancement** code.

### Requirements

_requirements.txt_ contains the Python packages used by the code.

### How to train CURL and use the model for inference

#### Training CURL

Instructions:

To get this code working on your system / problem you will need to edit the data loading functions, as follows:

1. main.py, change the paths for the data directories to point to your data directory
2. data.py, lines 248, 256, change the folder names of the data input and output directories to point to your folder names

To train, run the command:

```
python3 main.py
```


#### Inference - Using Pre-trained Models for Prediction

The directory _pretrained_models_ contains a set of four CURL pre-trained models on the Adobe5K_DPE dataset, each model output from different epochs. The model with the highest validation dataset PSNR (22.66dB) is at epoch 99:

* curl_validpsnr_22.66_validloss_0.0734_testpsnr_23.40_testloss_0.0605_epoch_124_model.pt

This pre-trained CURL model obtains 23.40dB on the test dataset for Adobe DPE.

To use this model for inference:

1. Place the images you wish to infer in a directory e.g. ./adobe5k_dpe/curl_example_test_input/. Make sure the directory path has the word "input" somewhere in the path.
2. Place the images you wish to use as groundtruth in a directory e.g. ./adobe5k_dpe/curl_example_test_output/. Make sure the directory path has the word "output" somewhere in the path.
3. Place the names of the images (without extension) in a text file in the directory above the directory containing the images i.e. ./adobe5k_dpe/ e.g. ./adobe5k_dpe/images_inference.txt
4. Run the command and the results will appear in a timestamped directory in the same directory as main.py:

```
python3 main.py --inference_img_dirpath=./adobe5k_dpe/ --checkpoint_filepath=./pretrained_models/adobe_dpe/curl_validpsnr_22.66_validloss_0.0734_testpsnr_23.40_testloss_0.0605_epoch_124_model.pt
```

### CURL for RGB images

- __rgb_ted.py__ contains the TED model for RGB images 

### CURL for RAW images

- __raw_ted.py__ contains the TED model for RGB images 

### Github user contributions

__CURL_for_RGB_images.zip__ is a contribution (RGB model and pre-trained weights) courtsey of Github user [hermosayhl](https://github.com/hermosayhl)

### Bibtex

If you do use ideas from the paper in your research please kindly consider citing as below:

```
@misc{moran2019curl,
    title={CURL: Neural Curve Layers for Global Image Enhancement},
    author={Sean Moran and Steven McDonagh and Gregory Slabaugh},
    year={2019},
    eprint={1911.13175},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```
