# [CODE READING] Neural Scene Flow Fields
PyTorch implementation of paper "Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes", CVPR 2021

[[Project Website]](https://www.cs.cornell.edu/~zl548/NSFF/) [[Paper]](https://arxiv.org/abs/2011.13084) [[Video]](https://www.youtube.com/watch?v=qsMIH7gYRCc&feature=emb_title)


## Dependency
The code is tested with Python3, Pytorch >= 1.6 and CUDA >= 10.2, the dependencies includes 
* configargparse
* matplotlib
* opencv
* scikit-image
* scipy
* cupy
* imageio.
* tqdm

## Video preprocessing 
1. Download nerf_data.zip from [link](https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing), an example input video with SfM camera poses and intrinsics estimated from [COLMAP](https://colmap.github.io/) (Note you need to use COLMAP "colmap image_undistorter" command to undistort input images to get "dense" folder as shown in the example, this dense folder should include "images" and "sparse" folders).

2. Download single view depth prediction model "model.pt" from [link](https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing), and put it on the folder "nsff_scripts".

3. Run the following commands to generate required inputs for training/inference:
```bash
    # Usage
    cd nsff_scripts
    # create camera intrinsics/extrinsic format for NSFF, same as original NeRF where it uses imgs2poses.py script from the LLFF code: https://github.com/Fyusion/LLFF/blob/master/imgs2poses.py
    python save_poses_nerf.py --data_path "/home/xxx/Neural-Scene-Flow-Fields/kid-running/dense/"
    # Resize input images and run single view model
    python run_midas.py --data_path "/home/xxx/Neural-Scene-Flow-Fields/kid-running/dense/" --input_w 640 --input_h 360 --resize_height 288
    # Run optical flow model (for easy setup and Pytorch version consistency, we use RAFT as backbond optical flow model, but should be easy to change to other models such as PWC-Net or FlowNet2.0)
    ./download_models.sh
    python run_flows_video.py --model models/raft-things.pth --data_path /home/xxx/Neural-Scene-Flow-Fields/kid-running/dense/ --epi_threshold 1.0 --input_flow_w 768 --input_semantic_w 1024 --input_semantic_h 576
```

## Rendering from an example pretrained model
1. Download pretraind model "kid-running_ndc_5f_sv_of_sm_unify3_F00-30.zip" from [link](https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing). Unzipping and putting it in the folder "nsff_exp/logs/kid-running_ndc_5f_sv_of_sm_unify3_F00-30/360000.tar". 

Set datadir in config/config_kid-running.txt to the root directory of input video. Then go to directory "nsff_exp":
```bash
   cd nsff_exp
```

2. Rendering of fixed time, viewpoint interpolation
```bash
   python run_nerf.py --config configs/config_kid-running.txt --render_bt --target_idx 10
```

By running the example command, you should get the following result:
![Alt Text](https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/main/demo/vi.gif)

3. Rendering of fixed viewpoint, time interpolation
```bash
   python run_nerf.py --config configs/config_kid-running.txt --render_lockcam_slowmo --target_idx 8
```

By running the example command, you should get the following result:
![Alt Text](https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/main/demo/ti.gif)

4. Rendering of space-time interpolation
```bash
   python run_nerf.py --config configs/config_kid-running.txt --render_slowmo_bt  --target_idx 10
```

By running the example command, you should get the following result:
![Alt Text](https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/main/demo/sti.gif)

## Training
1. In configs/config_kid-running.txt, modifying expname to any name you like (different from the original one), and running the following command to train the model:
```bash
    python run_nerf.py --config configs/config_kid-running.txt
```
The per-scene training takes ~2 days using 2 Nvidia V100 GPUs.

2. Several parameters in config files you might need to know for training a good model
* N_samples: in order to render images with higher resolution, you have to increase number sampled points
* start_frame,  end_frame: indicate training frame range. The default model usually works for video of 1~2s. Training on longer frames can cause oversmooth rendering. To mitigate the effect, you can increase the capacity of the network by increasing netwidth (but it can drastically increase training time and memory usage).
* decay_iteration: number of iteartion in initialization stage. Data-driven losses will decay every 1000*decay_iteration steps. It's usually good to match decay_iteration to the number of training frames. 
* no_ndc: our current implementation only supports reconstruction in NDC space, meaning it only works for forward-facing scene like original NeRF. But it should be not hard to adapt to euclidean space.
* use_motion_mask, num_extra_sample: whether to use estimated coarse motion segmentation mask to perform hard-mining sampling during initialization stage, and how many extra samples during initialization stage.
* w_depth, w_optical_flow: weight of losses for single-view depth and geometry consistency priors described in the paper. Weights of (0.4, 0.2) or (0.2, 0.1) usually work for most of the videos. 
* w_cycle: weights of scene flow cycle consistency loss
* w_sm: weight of scene flow smoothness loss
* w_prob_reg: weight of disocculusion weight regularization

## Evaluation on the Dynamic Scene Dataset
1. Download Dynamic Scene dataset "dynamic_scene_data_full.zip" from [link](https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing)

2. Download pretrained model "dynamic_scene_pretrained_models.zip" from [link](https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing), unzip and put them in the folder "nsff_exp/logs/"

3. Run the following command for each scene to get quantitative results reported in the paper:
```bash
   # Usage: configs/config_xxx.txt indicates each scene name such as config_balloon1-2.txt in nsff/configs
   python evaluation.py --config configs/config_xxx.txt
```

* Note: you have to use modified LPIPS implementation included in this branch in order to measure LIPIS error for dynamic region only as described in the paper.

## Acknowledgment
The code is based on implementation of several prior work:

* https://github.com/sniklaus/softmax-splatting
* https://github.com/yenchenlin/nerf-pytorch
* https://github.com/JKOK005/dVRK-Linear-Interpolator-
* https://github.com/richzhang/PerceptualSimilarity
* https://github.com/intel-isl/MiDaS
* https://github.com/princeton-vl/RAFT
* https://github.com/NVIDIA/flownet2-pytorch

## License
This repository is released under the [MIT license](hhttps://opensource.org/licenses/MIT).

## Citation
If you find our code/models useful, please consider citing our paper:
```bash
@InProceedings{li2020neural,
  title={Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes},
  author={Li, Zhengqi and Niklaus, Simon and Snavely, Noah and Wang, Oliver},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
