# DeepLPF: Deep Local Parametric Filters for Image Enhancement (CVPR 2020)

[Sean Moran](http://www.seanjmoran.com), [Pierre Marza](https://scholar.google.com/citations?user=NAI5mi4AAAAJ&hl=en), [Steven McDonagh](https://smcdonagh.github.io/), [Sarah Parisot](https://parisots.github.io/), [Greg Slabaugh](http://gregslabaugh.net/)

**Huawei Noah's Ark Lab**

<p>Main repository for the CVPR 2020 paper <b>DeepLPF: Deep Local Parametric Filters for Image Enhancement</b>. Here you will find a link to the code, pre-trained models and information on the datasets. Please raise a Github issue if you need assistance of have any questions on the research. 
</p>

**_Training your own DeepLPF:_** if you have difficulities, please contact the lead author for assistance. We are responsive and will be glad to assist on any queries regarding the model.

**_BATCH SIZE: Note this code is designed for a batch size of 1. It needs re-engineered to support higher batch sizes. Using higher batch sizes is not supported currently. To replicate our reported results please use a batch size of 1 only._**

<p align="center">
<img src="./images/teaser.png" width="80%"/>
</p>

### [[Paper]](https://arxiv.org/abs/2003.13985) [[Poster]](https://github.com/sjmoran/sjmoran.github.io/blob/main/pdfs/DeepLPF_CVPR20_poster.pdf) [[Video]](https://www.youtube.com/watch?v=Sxach3FM6FY) [[Supplementary]](https://github.com/sjmoran/sjmoran.github.io/blob/7775d1fc39d14baeb6935f6c750f923e1251f491/pdfs/DeepLPF_supplementary.pdf) 

<p align="center">
<a href="https://www.youtube.com/watch?v=Sxach3FM6FY" span>
   <img src="./images/youtube-thumbnail.png" width="90%"/>
</a>
<a href="https://github.com/sjmoran/sjmoran.github.io/blob/main/pdfs/DeepLPF_CVPR20_poster.pdf" span>
   <img src="./images/poster-img.png" width="100%"/>
</a>
</p>

<table>
       <tr>
    <th>Input</th>
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
      <td><img src="https://github.com/sjmoran/DeepLPF/blob/e6d160eed2a5f72c89e690bc81f1a59f09873085/adobe5k_dpe/deeplpf_example_test_input/a4576-DSC_0217_input.png" /></td>   
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/6889bd2c39a947ba85d86b6151387dfe56709996/adobe5k_dpe/deeplpf_example_test_output/a4576-DSC_0217_gt.png" /></td> 
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/1a76a93d5cb1fffbb8865193ba37ef83771e1f2d/adobe5k_dpe/deeplpf_example_test_inference/a4576-DSC_0217_TEST_425_9_PSNR_34.596_SSIM_0.980.jpg"/></td> 
  </tr>
    <tr>
    <th>Input</th>
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
      <td><img src="https://github.com/sjmoran/DeepLPF/blob/7b6ffc8d37d26a29742c0487e94c2e00617ab149/adobe5k_dpe/deeplpf_example_test_input/a4582-DSC_0343_input.png"/></td>   
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/ff8c04db5da40e6c58f8ff42417ab57d28605d85/adobe5k_dpe/deeplpf_example_test_output/a4582-DSC_0343_gt.png"/></td> 
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/094ae4b23e0586a588960cbc31c95e09eb3f2997/adobe5k_dpe/deeplpf_example_test_inference/a4582-DSC_0343_TEST_425_10_PSNR_18.942_SSIM_0.921.jpg"/></td> 
  </tr>
    <tr>
    <th>Input</th>
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
      <td><img src="https://github.com/sjmoran/DeepLPF/blob/ff2a4a6ac30a247f10c5ee2ebb96d1982725570b/adobe5k_dpe/deeplpf_example_test_input/a4742-Duggan_090331_6517_input.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/24ae58aaed19e76a59788edd04997d537bc624df/adobe5k_dpe/deeplpf_example_test_output/a4742-Duggan_090331_6517_output.png" /></td> 
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/a1f98969249067e89137b06fe82c71d44485cd4f/adobe5k_dpe/deeplpf_example_test_inference/a4742-Duggan_090331_6517_TEST_425_1_PSNR_29.825_SSIM_0.984.jpg"/></td> 
  </tr>
    <tr>
    <th>Input</th>
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
      <td><img src="https://github.com/sjmoran/DeepLPF/blob/3247d4dedce32aeaad54586d2c0b478189ed8d82/adobe5k_dpe/deeplpf_example_test_input/a4774-_DGW0330.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/28d9bfad7bf3649b25aa6c69dd25226474fd8339/adobe5k_dpe/deeplpf_example_test_output/a4774-_DGW0330_gt.png"/></td> 
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/37d2a682f2b0970fc0d3fd0c30d8512d909737b5/adobe5k_dpe/deeplpf_example_test_inference/a4774-_DGW0330_TEST_425_7_PSNR_24.233_SSIM_0.926.jpg"/></td> 
  </tr>
    <tr>
    <th>Input</th>
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
      <td><img src="https://github.com/sjmoran/DeepLPF/blob/33b02f2a6f1d079e0a2b6b025f576f5b1f3c8ac7/adobe5k_dpe/deeplpf_example_test_input/a4591-Duggan_080411_5940.png" /></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/10c985c9f69cf1bb541517e14624dcdc923e2b78/adobe5k_dpe/deeplpf_example_test_output/a4591-Duggan_080411_5940.tif_GT.png"/></td> 
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/c46645adbc897ea6e773fed729b2d99d5bf59dcc/adobe5k_dpe/deeplpf_example_test_inference/a4591-Duggan_080411_5940_TEST_425_20_PSNR_28.000_SSIM_0.951.jpg"/></td> 
  </tr>
  <tr>
    <th>Input</th>
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
      <td><img src="https://github.com/sjmoran/DeepLPF/blob/b614871bb72a6573cd45fc487f5ad2d5e7e6edda/adobe5k_dpe/deeplpf_example_test_input/a4521-kme_0310.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/9404949f046a53b87a04ecc17583d0fa01951176/adobe5k_dpe/deeplpf_example_test_output/a4521-kme_0310.png"/></td> 
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/7b5e10962cead0b0c20e47f7a0c750562f7bfc74/adobe5k_dpe/deeplpf_example_test_inference/a4521-kme_0310_TEST_800_5_PSNR_28.774_SSIM_0.897.jpg"/></td> 
  </tr>
  <tr>
    <th>Input</th>
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/76bbc1d5673cf0c90c7316d93fecefa9b5a62052/adobe5k_dpe/deeplpf_example_test_input/a4869-09-05-19-at-19h05m38s-_MG_9563.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/62d12eff53e30382553f75c32743596a7919deba/adobe5k_dpe/deeplpf_example_test_output/a4869-09-05-19-at-19h05m38s-_MG_9563.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/c75cea3d600136005ea1078db3fc663011c54d8b/adobe5k_dpe/deeplpf_example_test_inference/a4869-09-05-19-at-19h05m38s-_MG_9563_TEST_500_29_PSNR_30.090_SSIM_0.978.jpg"/></td> 
  </tr>
  <tr>
    <th>Input</th> 
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/c50fa517cbeb88ace1970b54da712e7872fbd95f/adobe5k_dpe/deeplpf_example_test_input/a4933-Duggan_090428_8040.png" /></td>   
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/86a9b78ff36d3f71d491a158fd7956b5880cd991/adobe5k_dpe/deeplpf_example_test_output/a4933-Duggan_090428_8040.png"/></td>     <td><img src="https://github.com/sjmoran/DeepLPF/blob/7745ebfb35fc9a6c8e27bbc09c4b82c94a77e632/adobe5k_dpe/deeplpf_example_test_inference/a4933-Duggan_090428_8040_TEST_500_23_PSNR_19.011_SSIM_0.882.jpg"/></td> 
  </tr>
     <tr>
    <th>Input</th> 
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/944e9bfac7ecf7b20b53f1142bf57bf0d2c6bfaf/adobe5k_dpe/deeplpf_example_test_input/a4783-20090321_at_19h16m32__MG_0220.png"/></td>   
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/4b5448fc529f9f5f71d117ad6fb54215ae2164c8/adobe5k_dpe/deeplpf_example_test_output/a4783-20090321_at_19h16m32__MG_0220.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/f8b20f0a102c549f716748a3d5e353c48a24d768/adobe5k_dpe/deeplpf_example_test_inference/a4783-20090321_at_19h16m32__MG_0220_TEST_500_2_PSNR_26.362_SSIM_0.905.jpg"/></td> 
  </tr>
        <tr>
    <th>Input</th> 
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/fc339bc7177aad138cdd7b3378056b97e6b5afc9/adobe5k_dpe/deeplpf_example_test_input/a4514-kme_0258.png"/></td>   
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/2e2213fbdf07f890e22c40ef2ee31c7cafe98679/adobe5k_dpe/deeplpf_example_test_output/a4514-kme_0258.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/2e2213fbdf07f890e22c40ef2ee31c7cafe98679/adobe5k_dpe/deeplpf_example_test_inference/a4514-kme_0258_TEST_500_27_PSNR_27.426_SSIM_0.879.jpg"/></td> 
  </tr>
</table>

### Dependencies

_requirements.txt_ contains the Python packages used by the code.

### How to train DeepLPF and use the model for inference

#### Training DeepLPF

Instructions:

To get this code working on your system / problem you will need to edit the data loading functions, as follows:

1. data.py, lines 248, 256, change the folder names of the data input and output directories to point to your folder names

To train, run the command:

```
python3 main.py --training_img_dirpath=../adobe5k/ --train_img_list_path=../adobe5k/images_train.txt --valid_img_list_path=../adobe5k/images_valid.txt --test_img_list_path=../adobe5k/images_test.txt
```

<p align="center">
<img src="./images/deeplpf_training_loss.png" width="80%"/>
</p>

#### Inference - Using Pre-trained Models for Prediction

The directory _pretrained_models_ contains a set of four DeepLPF pre-trained models on the _Adobe5K_DPE dataset_, each model output from different epochs. The model with the highest validation dataset PSNR (23.90 dB) is at epoch 424:

* deeplpf_validpsnr_23.378_validloss_0.033_testpsnr_23.904_testloss_0.031_epoch_424_model.pt

This model achieves a PSNR of 23.90dB and an SSIM of 0.911 on the Adobe_DPE image dataset. To inference with this model, follow these instructions:

1. Place the images you wish to infer in a directory e.g. ./adobe5k_dpe/deeplpf_example_test_input/. Make sure the directory path has the word "input" somewhere in the path.
2. Place the images you wish to use as groundtruth in a directory e.g. ./adobe5k_dpe/deeplpf_example_test_output/. Make sure the directory path has the word "output" somewhere in the path.
3. Place the names of the images (without extension) in a text file in the directory above the directory containing the images i.e. ./adobe5k_dpe/ e.g. ./adobe5k_dpe/images_inference.txt
4. Run the command and the results will appear in a timestamped directory in the same directory as main.py:

```
python3 main.py --inference_img_list_path=./adobe5k_dpe/images_inference.txt --inference_img_dirpath=./adobe5k_dpe/ --checkpoint_filepath=./pretrained_models/adobe_dpe/deeplpf_validpsnr_23.378_validloss_0.033_testpsnr_23.904_testloss_0.031_epoch_424_model.pt
```

### Pre-trained models

* Adobe-DPE: see pretrained_models/adobe_dpe directory. Model trained on the Adobe5K dataset using the splits and image pre-processing outlined in the DeepPhotoEnhancer paper.
* Adobe-UPE: see pretrained_models/adobe_upe directory. Model trained on the Adobe5K dataset using the splits and image pre-processing outlined in the DeepUPE paper. Thanks to Yucheng Lu (yucheng.l@outlook.com) for this pre-trained model. The model was applied in [this paper](https://arxiv.org/abs/2106.14844).

### Bibtex

```
@InProceedings{Moran_2020_CVPR,
author = {Moran, Sean and Marza, Pierre and McDonagh, Steven and Parisot, Sarah and Slabaugh, Gregory},
title = {DeepLPF: Deep Local Parametric Filters for Image Enhancement},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

### Datasets

* __Adobe-DPE__ (5000 images, RGB, RGB pairs): this dataset can be downloaded [here](https://data.csail.mit.edu/graphics/fivek/). After downloading this dataset you will need to use Lightroom to pre-process the images according to the procedure outlined in the DeepPhotoEnhancer (DPE) [paper](https://github.com/nothinglo/Deep-Photo-Enhancer). Please see the issue [here](https://github.com/nothinglo/Deep-Photo-Enhancer/issues/38#issuecomment-449786636) for instructions. Artist C retouching is used as the groundtruth/target. Note, the images must be extracted in sRGB space. Feel free to raise a Gitlab issue if you need assistance with this (or indeed the Adobe-UPE dataset below). You can also find the training, validation and testing dataset splits for Adobe-DPE in the following [file](https://www.cmlab.csie.ntu.edu.tw/project/Deep-Photo-Enhancer/%5BExperimental_Code_Data%5D_Deep-Photo-Enhancer.zip). The splits can also be found the the [adobe5k_dpe](./adobe5k_dpe/) directory in this repository (note these are a best guess at what the orginal splits from the DPE authors might be).

* __Adobe-UPE__ (5000 images, RGB, RGB pairs): this dataset can be downloaded [here](https://data.csail.mit.edu/graphics/fivek/). As above, you will need to use Lightroom to pre-process the images according to the procedure outlined in the Underexposed Photo Enhancement Using Deep Illumination Estimation (DeepUPE) [paper](https://github.com/wangruixing/DeepUPE) and detailed in the issue [here](https://github.com/wangruixing/DeepUPE/issues/26). Artist C retouching is used as the groundtruth/target. You can find the test images for the Adobe-UPE dataset at this [link](https://drive.google.com/file/d/1HZnNgptNxjKJAhekz2K5yh0mW0yKIws2/view?usp=sharing).

### License

BSD-3-Clause License

### Contributions

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.

### Errata for Paper(s) referencing DeepLPF

* **[Deep Symmetric Network for Underexposed Image Enhancement with Recurrent Attentional Learning](https://www.shaopinglu.net/publications_files/ICCV21_Image_Enhancement.pdf)**
* The results in Fig 1. for DeepLPF in this paper are incorrect. An example of inference for one of the images in Fig 1. is supplied [here](https://github.com/sjmoran/DeepLPF/blob/ecdc6f08cc96ff727a8246874bce06726949068e/images/004668_TEST_25_354_PSNR_21.848_SSIM_0.858.jpg). For replication, we provide our DeepLPF pre-trained model for their dataset [here](https://github.com/sjmoran/DeepLPF/blob/7b147dd819b2e4c8e9898c411f23887250cb9afe/pretrained_models/adobe_distort_and_recover/deeplpf_validpsnr_23.629675866286313_validloss_0.030986817553639412_testpsnr_23.629675866286313_testloss_0.030986817553639412_epoch_49_model.pt). The quantitative results in Table 1 for DeepLPF should be: **23.63dB, 0.875 SSIM**. 
* **29th September 2021**: The paper authors have graciously noted this issue and have created an errata to their ICCV paper [here](https://www.shaopinglu.net/proj-iccv21/ImageEnhancement.html). We thank the authors and appreciate their time commitment and close collaboration in re-checking the DeepLPF result in their paper.

【This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.】