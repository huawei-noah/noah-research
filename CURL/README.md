# CURL: Neural Curve Layers for Global Image Enhancement (ICPR 2020)

[Sean Moran](http://www.seanjmoran.com),  [Steven McDonagh](https://smcdonagh.github.io/), [Greg Slabaugh](http://gregslabaugh.net/)

**Huawei Noah's Ark Lab**

<p>
   Repository links for the paper <i>CURL: Neural Curve Layers for Global Image Enhancement</i>. In this repository you will find a link to the code and information of the datasets. Please raise a Github issue if you need assistance of have any questions on the research. 
</p>

### [[Paper]](https://arxiv.org/pdf/1911.13175)  [[Supplementary]](https://sjmoran.github.io/pdfs/CURL_supplementary.pdf) [[Video]](https://youtu.be/66FnRfDR_Oo) [[Poster]](https://sjmoran.github.io/pdfs/CURL_ICPR_POSTER.pdf) [[Slides]](https://sjmoran.github.io/pdfs/DeepLPFDataBites.pdf) 

<p align="center">
<img src="./images/teaser.PNG" width="80%"/>
</p>

<p align="center">
<a href="https://www.youtube.com/watch?v=66FnRfDR_Oo" span>
   <img src="./images/youtube-thumbnail.png" width="90%"/>
</a>
<a href="https://sjmoran.github.io/pdfs/CURL_ICPR_POSTER.pdf" span>
   <img src="./images/poster-img.png" width="100%"/>
</a>
</p>

<table>
  <tr>
      <td><img src="https://github.com/sjmoran/CURL/blob/877d67ed4cf706fc8408630003e740c1078c3475/adobe5k_dpe/curl_example_test_input/a3102-JI2E4053%202.png"/></td>     
     <td><img src="https://github.com/sjmoran/CURL/blob/2937522378a81c7f9a2e5ad20ddbe966820bb6c2/adobe5k_dpe/curl_example_test_output/a3102-JI2E4053.png"/></td> 
    <td><img src="https://github.com/sjmoran/CURL/blob/6a7ffde24c38a3ae0a65dd0603bf7fcef9019dd3/adobe5k_dpe/curl_example_test_inference/a3102-JI2E4053_VALID_460_29_PSNR_21.413_SSIM_0.902.jpg"/></td> 
  </tr>
  <tr>
    <th>Input</th>
    <th>Label</th>
    <th>Ours (CURL)</th>
  </tr>
  <tr>
      <td><img src="https://github.com/sjmoran/CURL/blob/2451eaf5b9d37a2384339c1d5ba412e4498cbce3/adobe5k_dpe/curl_example_test_input/a4714-Duggan_080613_8704_input.png"/></td>
     <td><img src="https://github.com/sjmoran/CURL/blob/2451eaf5b9d37a2384339c1d5ba412e4498cbce3/adobe5k_dpe/curl_example_test_output/a4714-Duggan_080613_8704.png"/></td> 
    <td><img src="https://github.com/sjmoran/CURL/blob/2451eaf5b9d37a2384339c1d5ba412e4498cbce3/adobe5k_dpe/curl_example_test_inference/a4714-Duggan_080613_8704_TEST_510_3_PSNR_27.102_SSIM_0.898.jpg"/></td> 
  </tr>
  <tr>
    <th>Input</th>
    <th>Label</th>
    <th>Ours (CURL)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/sjmoran/CURL/blob/64f8873d93e8e2eaf259b1dd8a66d34894232788/adobe5k_dpe/curl_example_test_input/a4857-_DSC0008-1_input.png"/></td>     
     <td><img src="https://github.com/sjmoran/CURL/blob/64f8873d93e8e2eaf259b1dd8a66d34894232788/adobe5k_dpe/curl_example_test_output/a4857-_DSC0008-1.png"/></td>     
     <td><img src="https://github.com/sjmoran/CURL/blob/64f8873d93e8e2eaf259b1dd8a66d34894232788/adobe5k_dpe/curl_example_test_inference/a4857-_DSC0008-1_TEST_510_17_PSNR_26.826_SSIM_0.944.jpg"/></td> 
  </tr>
  <tr>
    <th>Input</th> 
    <th>Label</th>
    <th>Ours (CURL)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/sjmoran/CURL/blob/c58bdeeb948662419e1d15ccf4d11498b5a23379/adobe5k_dpe/curl_example_test_input/a4774-_DGW0330_input.png"/></td>   
     <td><img src="https://github.com/sjmoran/CURL/blob/c58bdeeb948662419e1d15ccf4d11498b5a23379/adobe5k_dpe/curl_example_test_output/a4774-_DGW0330.png"/></td>     
     <td><img src="https://github.com/sjmoran/CURL/blob/c58bdeeb948662419e1d15ccf4d11498b5a23379/adobe5k_dpe/curl_example_test_inference/a4774-_DGW0330_TEST_510_31_PSNR_21.773_SSIM_0.899.jpg"/></td> 
  </tr>
   <tr>
    <td><img src="https://github.com/sjmoran/CURL/blob/2f885e1b779fa0924138355f80fd9f7b1d3e8f6e/adobe5k_dpe/curl_example_test_input/a4723-_DGW7894_input.png"/></td>   
     <td><img src="https://github.com/sjmoran/CURL/blob/2f885e1b779fa0924138355f80fd9f7b1d3e8f6e/adobe5k_dpe/curl_example_test_output/a4723-_DGW7894.png"/></td>     
     <td><img src="https://github.com/sjmoran/CURL/blob/2f885e1b779fa0924138355f80fd9f7b1d3e8f6e/adobe5k_dpe/curl_example_test_inference/a4723-_DGW7894_TEST_510_8_PSNR_28.702_SSIM_0.902.jpg"/></td> 
  </tr>
</table>

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

<p align="center">
<img src="./images/curl_training_loss.png" width="80%"/>
</p>

#### Inference - Using Pre-trained Models for Prediction

The directory _pretrained_models_ contains a CURL pre-trained model on the Adobe5K_DPE dataset. The model with the highest validation dataset PSNR (23.58dB) is at epoch 510:

* curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model.pt

This pre-trained CURL model obtains 23.58dB on the test dataset for Adobe DPE.

To use this model for inference:

1. Place the images you wish to infer in a directory e.g. ./adobe5k_dpe/curl_example_test_input/. Make sure the directory path has the word "input" somewhere in the path.
2. Place the images you wish to use as groundtruth in a directory e.g. ./adobe5k_dpe/curl_example_test_output/. Make sure the directory path has the word "output" somewhere in the path.
3. Place the names of the images (without extension) in a text file in the directory above the directory containing the images i.e. ./adobe5k_dpe/ e.g. ./adobe5k_dpe/images_inference.txt
4. Run the command and the results will appear in a timestamped directory in the same directory as main.py:

```
python3 main.py --inference_img_dirpath=./adobe5k_dpe/ --checkpoint_filepath=./pretrained_models/curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model.pt
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
@INPROCEEDINGS{moran2020curl,
  author={Moran, Sean and McDonagh, Steven and Slabaugh, Gregory},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)}, 
  title={CURL: Neural Curve Layers for Global Image Enhancement}, 
  year={2021},
  volume={},
  number={},
  pages={9796-9803},
  doi={10.1109/ICPR48806.2021.9412677}}
```

### Datasets

* __Samsung S7__ (110 images, RAW, RGB pairs): this dataset can be downloaded [here](https://www.kaggle.com/knn165897/s7-isp-dataset). The validation and testing images are listed below, the remaining images serve as our training dataset. For all results in the paper we use random crops of patch size 512x512 pixels during training.

  * __Validation Dataset Images__

    * S7-ISP-Dataset-20161110_125321
    * S7-ISP-Dataset-20161109_131627
    * S7-ISP-Dataset-20161109_225318
    * S7-ISP-Dataset-20161110_124727
    * S7-ISP-Dataset-20161109_130903
    * S7-ISP-Dataset-20161109_222408
    * S7-ISP-Dataset-20161107_234316
    * S7-ISP-Dataset-20161109_132214
    * S7-ISP-Dataset-20161109_161410
    * S7-ISP-Dataset-20161109_140043


  * __Test Dataset Images__
  
    * S7-ISP-Dataset-20161110_130812
    * S7-ISP-Dataset-20161110_120803
    * S7-ISP-Dataset-20161109_224347
    * S7-ISP-Dataset-20161109_155348
    * S7-ISP-Dataset-20161110_122918
    * S7-ISP-Dataset-20161109_183259
    * S7-ISP-Dataset-20161109_184304
    * S7-ISP-Dataset-20161109_131033
    * S7-ISP-Dataset-20161110_130117
    * S7-ISP-Dataset-20161109_134017

* __Adobe-DPE__ (5000 images, RGB, RGB pairs): this dataset can be downloaded [here](https://data.csail.mit.edu/graphics/fivek/). After downloading this dataset you will need to use Lightroom to pre-process the images according to the procedure outlined in the DeepPhotoEnhancer (DPE) [paper](https://github.com/nothinglo/Deep-Photo-Enhancer). Please see the issue [here](https://github.com/nothinglo/Deep-Photo-Enhancer/issues/38#issuecomment-449786636) for instructions. Artist C retouching is used as the groundtruth/target. Note, that the images should be extracted in sRGB space. Feel free to raise a Gitlab issue if you need assistance with this (or indeed the Adobe-UPE dataset below). You can also find the training, validation and testing dataset splits for Adobe-DPE in the following [file](https://www.cmlab.csie.ntu.edu.tw/project/Deep-Photo-Enhancer/%5BExperimental_Code_Data%5D_Deep-Photo-Enhancer.zip). 

* __Adobe-UPE__ (5000 images, RGB, RGB pairs): this dataset can be downloaded [here](https://data.csail.mit.edu/graphics/fivek/). As above, you will need to use Lightroom to pre-process the images according to the procedure outlined in the Underexposed Photo Enhancement Using Deep Illumination Estimation (DeepUPE) [paper](https://github.com/wangruixing/DeepUPE) and detailed in the issue [here](https://github.com/wangruixing/DeepUPE/issues/26). Artist C retouching is used as the groundtruth/target. You can find the test images for the Adobe-UPE dataset at this [link](https://drive.google.com/file/d/1HZnNgptNxjKJAhekz2K5yh0mW0yKIws2/view?usp=sharing).

### License

BSD-3-Clause License

### Contributions

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.
