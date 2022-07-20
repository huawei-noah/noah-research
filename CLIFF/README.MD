# CLIFF [ECCV 2022 Oral]



This repo contains the CLIFF demo code, 
and the pseudo-GT SMPL parameters used in the training for the paper:

**CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation**

## Installation
```bash
conda create -n cliff python=3.10
pip install -r requirements.txt
```

1. Download [the SMPL models](https://smpl.is.tue.mpg.de/) for rendering the reconstructed meshes
2. Download [pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3), unzip it under the `lib` directory, and change the folder name to **"pytorch_yolo_v3_master"**
2. Download the pretrained checkpoints and some testing samples to run the demo [[Google Drive]()]
3. Download the CLIFF pseudo-GT for the in-the-wild datasets [[Google Drive]()]

Finally put these data following the directory structure as below:
```
${ROOT}
|-- cliffGT_v1
    |-- coco2014part_cliffGT.npz
    |-- mpii_cliffGT.npz
|-- data
    |-- ckpt
        |-- hr48-PA43.0_MJE69.0_MVE81.2_3dpw-PA32.6_MJE47.6_MVE60.1_h36m-p2.pt
        |-- res50-PA45.7_MJE72.0_MVE85.3_3dpw-PA35.7_MJE52.8_MVE63.2_h36m-p2.pt
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_MALE.pkl
        |-- SMPL_NEUTRAL.pkl
|-- lib
    |-- pytorch_yolo_v3_master
    |-- yolov3.weights
```


## Demo
```shell
sh scripts/run_demo.sh
```
Change the options in the scripts. 
See the option description in the bottom lines of `demo.py`

## Pseudo-GT
```shell
sh scripts/run_cliffGT_visualization.sh
```



## Citing
```
@Inproceedings{li_2022_eccv_cliff,
  Title     = {CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation},
  Author    = {Li, Zhihao and Liu, Jianzhuang and Zhang, Zhensong and Xu, Songcen and Yan, Youliang},
  Booktitle = {ECCV},
  Year      = {2022}
}
```