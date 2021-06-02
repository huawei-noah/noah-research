# CURL: Neural Curve Layers for Global Image Enhancement
[Sean Moran](http://www.seanjmoran.com),  [Steven McDonagh](https://smcdonagh.github.io/), [Greg Slabaugh](http://gregslabaugh.net/)

[Paper on arXiv](https://arxiv.org/pdf/1911.13175.pdf) 

![Teaser](https://github.com/huawei-noah/noah-research/blob/master/CURL/teaser.png "Teaser")

Repository for the **CURL: Neural Curve Layers for Global Image Enhancement** code.

**Datasets**  

*  Samsung S7 dataset: https://elischwartz.github.io/DeepISP/
*  Adobe5K dataset: https://data.csail.mit.edu/graphics/fivek/

**Dataset Preprocessing**

* __Adobe-DPE__ (5000 images, RGB, RGB pairs): this dataset can be downloaded [here](https://data.csail.mit.edu/graphics/fivek/). After downloading this dataset you will need to use Lightroom to pre-process the images according to the procedure outlined in the DeepPhotoEnhancer (DPE) [paper](https://github.com/nothinglo/Deep-Photo-Enhancer). Please see the issue [here](https://github.com/nothinglo/Deep-Photo-Enhancer/issues/38#issuecomment-449786636) for instructions. Artist C retouching is used as the groundtruth/target. Feel free to raise a Gitlab issue if you need assistance with this (or indeed the Adobe-UPE dataset below). You can also find the training, validation and testing dataset splits for Adobe-DPE in the following [file](https://www.cmlab.csie.ntu.edu.tw/project/Deep-Photo-Enhancer/%5BExperimental_Code_Data%5D_Deep-Photo-Enhancer.zip). 

* __Adobe-UPE__ (5000 images, RGB, RGB pairs): this dataset can be downloaded [here](https://data.csail.mit.edu/graphics/fivek/). As above, you will need to use Lightroom to pre-process the images according to the procedure outlined in the Underexposed Photo Enhancement Using Deep Illumination Estimation (DeepUPE) [paper](https://github.com/wangruixing/DeepUPE) and detailed in the issue [here](https://github.com/wangruixing/DeepUPE/issues/26). Artist C retouching is used as the groundtruth/target. You can find the test images for the Adobe-UPE dataset at this [link](https://drive.google.com/file/d/1HZnNgptNxjKJAhekz2K5yh0mW0yKIws2/view?usp=sharing).


**Training**  

```
python main.py --valid_every=250 --num_epoch=10000 


valid_every: number of epochs to dump the testing and validation dataset metrics 
num_epoch: total number of training epochs 
```

For CURL at 100 epochs, on Samsung S7 dataset, you should get the following result: 

Validation dataset PSNR: 24.03 dB 

Test dataset PSNR: 23.53 dB 

Output is written to a corresponding data directory subdirectory eg:

```
/Samsung_Dataset/log_<timestamp>/
```

**Inference**  

For inference, create a directory e.g. inference_imgs containing two sub-directories called "input" and "output": 

```
/inference_imgs/input 

/inference_imgs/output 
```

Place the input images into to the input directory (i.e. those images you wish to inference) and put the groundtruth images in the output directory. 

In the inference_imgs directory create a text file called "images_inference.txt" and list the image names to be inferenced one per line, without any path of file extension e.g. if the image is a5000.tif you would create a file with one line with the entry: 

```
a5000 
```

To run inference use the following command: 

```
python main.py  --checkpoint_filepath= ---inference_img_dirpath= 


checkpoint_filepath: location of checkpoint file 
inference_img_dirpath: location of image directory 
```

For example: 

```
python main.py  --inference_img_dirpath="/inference_imgs/" --checkpoint_filepath="curl_validpsnr_24.032338682762976_validloss_0.031165003776550293_testpsnr_23.531164141118847_testloss_0.03296922147274017_epoch_99_model.pt"
```

Output is written to a corresponding data directory subdir eg:

```
/Samsung_Dataset/log_<timestamp>/

```
