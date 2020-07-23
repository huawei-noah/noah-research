# SteReFo: Efficient Image Refocusing with Stereo Vision (ICCVW 2019)

Benjamin Busam, Matthieu Hog, Steven McDonagh, Gregory Slabaugh; Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
Paper: [here](https://openaccess.thecvf.com/content_ICCVW_2019/papers/AIM/Busam_SteReFo_Efficient_Image_Refocusing_with_Stereo_Vision_ICCVW_2019_paper.pdf) 

![Teaser](https://github.com/huawei-noah/noah-research/blob/master/SteReFo/teaser.png "Teaser")

# RefNet


## Install
In the project folder.
```
pip install -e .
```
Deps: should be installed by the setup. Just in case: tf, np, click

## stereonet
Module that defines the model and training script for a fast, lightweight accurate depth estimation.
Learn depth and refocus from a pair of stereo cameras.
The depth estimation net is based on StereoNet ([Khamis et al. "Stereonet: Guided hierarchical refinement for real-time edge-aware depth prediction." ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sameh_Khamis_StereoNet_Guided_Hierarchical_ECCV_2018_paper.pdf)).
You can train it using the ipython notebook and save the checkpoints for later.

A command-line demo is also available 
```
python demo.py --help

Usage: demo.py [OPTIONS] LEFT_IMAGE RIGTH_IMAGE STEREONET_CHECKPOINT

Options:
  -o, --output_folder TEXT  Path to the output folder.
  -i, --intermediate        Saves the disparity or cost volume from stereonet.
  -v, --verbose             Sets the logging display level -v means warning
                            -vv is info.
  -t, --timming             Runs 10 times the refocusing to get accurate
                            timming.
  --help                    Show this message and exit.
```

Todo: 
- fine tune training with real dataset

## refocus_algorithms
Module that defines refocusing using different algorithms and for several modalities.
You can play with the layered bluring with the command line in demo.py.
```
python demo.py --help

Usage: demo.py [OPTIONS] IMAGE DISPARITY_MAP

Options:
  -o, --output_folder TEXT        Path to the output folder.
  -f, --focus_plane FLOAT         Focus parameter. Can be used multiple time
                                  to generate multiple images.
  -a, --aperture FLOAT            Virtual aperture, ie blur intensity
                                  parameter.
  -p, --pyramidal_conv INTEGER    Defines a maximum kernel size such that the
                                  images will be downsampled before
                                  comvolution if the blur kernel is too big.
  -d, --disparity_range FLOAT...  Minimun and maximum disparity range (only
                                  needed for refocusing using disparity).
  -v, --verbose                   Sets the logging display level -v means
                                  warning -vv is info.
  --help                          Show this message and exit.

```

Todo:
- solve the blurred foreground artifact problem
- performance checks (see if parallel)
- try out more refocus algos (lumigraph reconstruciton, adaptive kernel convolution, ray tracing, approximated ray tracing)

## blur_baseline
Modules that produce blur images from stereo images by naively using the full res depth or the cost volume from stereonet and the differentiable pyramidal layered refocusing. 
Training the depth estimation using aperture suppervision is also an option (see the train notebook).

You can test it using
```
python demo.py --help

Usage: demo.py [OPTIONS] LEFT_IMAGE RIGTH_IMAGE STEREONET_CHECKPOINT

Options:
  -o, --output_folder TEXT        Path to the output folder.
  -f, --focus_plane FLOAT         Focus parameter. Can be used multiple time
                                  to generate multiple images.
  -a, --aperture FLOAT            Virtual aperture, ie blur intensity
                                  parameter.
  -p, --pyramidal_conv INTEGER    Defines a maximum kernel size such that the
                                  images will be downsampled before
                                  comvolution if the blur kernel is too big.
  -d, --disparity_range TEXT      Minimun and maximum disparity range (only
                                  needed for refocusing using disparity).
  -s, --from_stage [disparity_map|cost_volume]
                                  Uses the full resolution disparity or the
                                  cost volume to refocus.
  -i, --intermediate              Saves the disparity or cost volume from
                                  stereonet.
  -v, --verbose                   Sets the logging display level -v means
                                  warning -vv is info.
  -t, --timming                   Runs 10 times the refocusing to get accurate
                                  timming.
  --help                          Show this message and exit.
```

A very crappy demo script is available for video.
It takes 3 input text files listing per line the path to left, right images and the focus plane to be used for each deph plane.
```
Usage: demo_video.py [OPTIONS] LEFT_IMAGE_LIST RIGTH_IMAGE_LIST FOCUS_LIST
                     STEREONET_CHECKPOINT

Options:
  -o, --output_folder TEXT        Path to the output folder.
  -i, --intermediate              Saves the disparity or cost volume from
                                  stereonet.
  -v, --verbose                   Sets the logging display level -v means
                                  warning -vv is info.
  -a, --aperture FLOAT            Virtual aperture, ie blur intensity
                                  parameter.
  -p, --pyramidal_conv INTEGER    Defines a maximum kernel size such that the
                                  images will be downsampled before
                                  comvolution if the blur kernel is too big.
  -d, --disparity_range TEXT      Minimun and maximum disparity range (only
                                  needed for refocusing using disparity).
  -s, --from_stage [disparity_map|cost_volume]
                                  Uses the full resolution disparity or the
                                  cost volume to refocus.
  -r, --resume INTEGER            Resumes from a given frame index.
  --help                          Show this message and exit.
```

## blur_refinement

Module that learns to refine an image refocused with a small disparity map from stereonet.
Can start from 1/2 to 1/8 of the resolution, but at 1/8, the blur kernels are<1px.


## blur_CV_refinement

Module that starts from the cost volume and upsamples the cost volumes slices depending if they are is focus or not,
refine the upsampled slices with a CNN prior to rendering.

## dataset_generation

Module that downloads and generates synthetic training sets.
