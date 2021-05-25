# Neural Scene Flow Fields Code Reading

This repository is forked from [[zhengqili/Neural-Scene-Flow-Fields](https://github.com/zhengqili/Neural-Scene-Flow-Fields)] and aims to record the code-reading.

**This code is mainly based on implementation of NeRF-Pytorch [[yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)]**. The core codes for training models are located in `nsff_exp` folder. The pipeline is similar with   `NeRF-Pytorch`.

## Data preparation
According to the code in `run_nerf.py`, only the `LLFF` dataloader is supported, which is defined in `load_llff.load_llff_data()`. The `_load_data()` will load the `poses_bounds.npy` which is created by parsing the COLMAP sparse reconstrunction results with `nsff_scripts/save_poses_nerf.py` file. Moreover, it also needs to load the displacement maps or depth maps (?) which are predicted by the `MiDaS` network, as well as the motion mask produced by the optical flow network `RAFT`.

Therefore, for customed video dataset, the COLMAP sparse reconstrunction should be firstly performed, and export the sparse model with three files, i.e., `dense/sparse/camaras.bin`, `dense/sparse/images.bin` and `dense/sparse/points3D.bin`. Then by using the `save_poses_nerf.npy` to obtain the camera-to-world coordinates transformation matrices (`c2w`) and H, W, focal values (`hwf`). In addition, the released models of `MiDaS` and `RAFT` should be downloaded and loaded to obtain the depth map and optical flow motion mask. 

> More details can be found in the repositories  
> * [Fyusion/LLFF](https://github.com/Fyusion/LLFF).  
> [Paper]: Mildenhall, Ben, et al. "Local light field fusion: Practical view synthesis with prescriptive sampling guidelines." ACM Transactions on Graphics (TOG) 38.4 (2019): 1-14.  
> * [MiDaS Network](https://github.com/intel-isl/MiDaS)  
> [Paper]: Ranftl, René, et al. "Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer." arXiv preprint arXiv:1907.01341 (2019).  
> * [RAFT](https://github.com/princeton-vl/RAFT)  
> [Paper]: Teed, Zachary, and Jia Deng. "Raft: Recurrent all-pairs field transforms for optical flow." European Conference on Computer Vision. Springer, Cham, 2020.

Specifically, the `poses_bound.npy` files is a `N_frames-by-17` matrix, for `kid-running` sequence, it is a `75-by-17` matrix. The video length is `75` and the pose parameters for each frame is `17`. The first `15` values are pose and have been reshaped to `3-by-5` matrix, the `last two values` are boundaries (near and far). 

> TODO list: parameters stored in the `poses_bound.npy` file, especially the 3-by-5 matrix. but which part of this matrix is Rotation matrix, which part of the matrix is translation transformation parameters, which part is scale, which part is focal length? Should make that clear.

The dataloader will return

```python
{
    "images": # RGB training images with values normalized to [0, 1],    
    "depths": # Depth maps of each training image 
    "masks":  # Motion mask of each training image
    "poses":  # poses parameters (camera intrinsic/extrinsics)
    "bds":    # Near and far boundaries for each image
    "render_poses": # the camera pose for novel view image synthesis, only used for inference.
    "ref_c2w":  # the reference camera-to-world coordinate system transformation matrix, only used for inference
    "motion_coords": # (x,y) locations on the motion mask with value > 0.1
}
```

## Model
The model is quite similar with Orginal NeRF, but it introduces the motion coordinates (locations with motion value > 0.1 on the motion mask) to perfrom `hard ray sampling` in the initial training stage. In addition, they also introduced the selected points on the forward optical flow, forward mask, backward optical flow, backward mask, rgb values, depth set, motion mask to supervised the learning process of static and dynamic regions in the scene.

The model is constructed by using `create_nerf()` funtion which is defined in the `render_utilts.py`. In the `create_nerf()`, two sets of NeRF models will be constructed, as well as two sets of positional embedding for each NeRF model. The first model is the original NeRF, i.e. `NeRF` class defined in `run_nerf_helpers.py`. The second one is Rigid_NeRF, defined as `Rigid_NeRF` class in `run_nerf_helper.py`. Both models are fed into lambda functions with `run_network` as `{}_network_query_fn` to denote the forward pass of both NeRF models. The optimizer is also defined in this  `create_nerf()` funtion. The function will return 
`render_kwargs_train`, `render_kwargs_test`, `start`, `grad_vars`, and `optimizer`. Specifically, 

- `the render_kwargs_train`: a dictionary stores the training settings.
```python
render_kwargs_train = {                              # training settings
    'network_query_fn' : network_query_fn,           # run_network of NeRF part 
    'perturb' : args.perturb,                        # perturb controls the sampling interval in a camera ray
    'N_importance' : args.N_importance,              # number of samples in importance sampling stage
    'rigid_network_query_fn':rigid_network_query_fn, # run_network fo Rigid-NeRF model
    'network_rigid' : model_rigid,                   # Rigid-NeRF model
    'N_samples' : args.N_samples,                    # Number of sampling points on each ray
    'network_fn' : model,                            # NeRF model
    'use_viewdirs' : args.use_viewdirs,              # Whether to use view direction embedding
    'white_bkgd' : args.white_bkgd,                  # Use the whited background setting 
    'raw_noise_std' : args.raw_noise_std,            # Random noise for image data augmentation 
    'inference': False                               # Training /Testing stage
}
```
- `render_kwargs_test`: Same with render_kwargs_train, but set `perturb=False` and `raw_noise_std=0.0`.
- `start`: starting global step
- `grad_vars`: varible set, which contains the parameters need to be updated by computing gradient.
- `optimizer`: optimizer, paramter and learning rate updating policy.

**Note that, the `output_ch` denotes the number of output of the `MLP`. In this NSFF paper, the `output_ch` is set to 5, (R, G, B, alpha, v). Among them, (R, G, B) illustrate reflectance, alpha represents opacity, v is "an unsupervised 3D blending weight field, that linearly blends the RGB alpha from static and dynamic scene representations along the ray." That is different with the vanilla NeRF model.**

> But, how to define the dynamic and static regions under this network settings? Need to find out more in the training process, or the model just figure that out according to the different supervision and regularization terms in the loss function.

After the construction of nerf model, in the `Train()` function defined, the `render()` function will be called to get the `ret` as the network output. Note that the input `batch_rays` stores the sampled camera rays from the image by `random.chioce` in the original coordniate mesh grid, and perform `hard (example) mining`  within the motion region based on the motion mask. The rays are sampled from a random selected image in the image buffer. 

The `render()` function is defined in `render_utils.py`. The input of the functions contains:
```python

def render(
    img_idx,        # Index embedding of the selected video frame, ranges from [0, 1].
    chain_bwd,      # shift this variable with true or false in each iteration(?) 
    chains_5frames, # in the initial training stage with iteration < decay_iteration * 1000, set chains_5frames=False, otherwise True
    num_img,        # Number of images for training 
    H, W, focal,    # height, width of image, focal length of the camera
    chunk,          # size of data chunk
    rays,           # sampled batch rays
    c2w,            # if there are sampled batch rays, no need to set camera to world coordinate transformation matrix
    ndc,            # use NDC coordinates system (lower left corner(0,0), upper right corner (1.0, 1.0))
    near,           # near bound
    far,            # far bound
    use_viewdirs,   # whether to use view direction as input
    c2w_staticcam,  # camera-to-world translation transformation matrix of a static camera
    **kwargs,       # model settings, including model, forward network function, etc.
):
```
In the `render()`, training sampes with `rays_o`, `rays_d`, `near`, `far` and view directions `viewdirs` (if needed), are concatenated together and then fed into the `batchify_rays()` function. 

The `batchify_rays()` function is defined as follow:
```python
def batchify_rays(
    img_idx,           # Index embedding of the selected video frame, ranges from [0, 1].
    chain_bwd,         # shift this variable with true or false in each iteration(?) 
    chain_5frames,     # in the initial training stage with iteration < decay_iteration * 1000, set chains_5frames = False, otherwise set it as True.
    num_img,           # Number of training images. 
    rays_flat,         # N-by-8 (or 11) matrix 3+3+1+1+(3)
    chunk=1024*16,     # size of data chunk
    **kwargs           # model settings.
):
```

In the `batchify_rays()` function, the rays will be split into number of data chunk for forward passing. For each chunk, the function call the `render_rays()` function. Specifically, the definition of `render_rays()` is 

```python
def render_rays(
    img_idx,                  # frame embedding of this selected frame, range from 0.0 to 1.0 (float type)
    chain_bwd,                # shift this variable with true or false in each iteration(?) 
    chain_5frames,            # in the initial training stage with iteration < decay_iteration * 1000, set chains_5frames = False, otherwise set it as True.
    num_img,                  # Number of training images
    ray_batch,                # sampled rays with size N-by-8 (or 11) matrix 3+3+1+1+(3)
    network_fn,               # original nerf model
    network_query_fn,         # run_network with nerf model
    rigid_network_query_fn,   # run_network with rigid-nerf model
    N_samples,                # number of sampling points on each ray
    retraw=False,             # bool. If True, include model's raw, unprocessed predictions.
    lindisp=False,            # bool. If True, sample linearly in inverse depth rather than in depth.
    perturb=0.,               # float, 0 or 1. If non-zero, each ray is sampled at stratified random points in time.
    N_importance=0,           # int. Number of additional times to sample along each ray. These samples are only passed to network_fine.
    network_rigid=None,       # Rigid_NeRF model
    # netowrk_blend=None,
    white_bkgd=False,         # bool. If True, assume a white background.
    raw_noise_std=0.,         # 
    verbose=False,            # print log info.
    pytest=False,             #
    inference=False           # inference stage?
    ):
```
As defined in this function, the data will be firsting parsing into different variables, i.e., `rays_o`, `rays_d`, `viewdirs`, `near`, `far`, the `z_vals` are calcuated along the `near` and `far` boundary with fixed intervals divided by the number of sampling points. According to such distance intervals, we can get `N_samples` along each rays. The 3D sampling points are stored in `pts`. Then the frame index embeding `img_idx` is concatenated to the points coordinates. It thus form the input training data with tensor size `[N_rays, N_samples, 3+1]`. 

## Static scene representation
Then the `get_rigid_ouputs()` function is firstly called. This function will perform forward pass of the `Rigid_NeRF` within the lambda function  `run_network` defined in `create_nerf()` function. The `fowward()` function of `Rigid_NeRF` model will output five values, i.e., raw `r`, `g`, `b`, `alpha`, and `v`. The first four values, i.e. `rgba`, are stored in `raw_rgba_rigid`, and the last value `v` is stored in `raw_blend_w`.  Then, the `rgba` values are fed in to `raw2output()` function to get the `rgb_map`, `weights` and `depth_map`.

As descripted in the volumetric rendering equation, as mentioned in the original NeRF paper, the alpha will be multipled by the accumuated `1-alpha` values to get the `weights` as discribed in the second part of Equation (5). In addition, it will be used to re-weight the `rgb` values in the each sampling points, and aggreated them to a single set of RGB values for each camera ray. (See the first part of the Equation (5).). Then, re-weight the accumulated ray marching distance `z_val` and sum the values together to get a `depth` value for each camera ray. In particular, 

```python
rgb_map_rig:    # [num_rays, 3]. Estimated RGB color of a ray. return this output
disp_map_rig:   # [num_rays]. Disparity map. Inverse of depth map. return
weights:        # [num_rays, num_samples]. Weights assigned to each sampled color. these data are not returned
depth_map_rig:  # [num_rays]. Estimated distance to object. return this map
raw_rgba_rigid: # [num_rays, num_samples, 4]. Estimated raw rgb values without feeding into raw2outputs
raw_blend_w:    # [num_rays, num_samples, 1]. Estimated weights in Rigid_NeRF. 
```
This part of output are used as static scene representation. 

## Dynamic scene representions
Then the `pts_ref` and `view_dirs` are also fed into the `NeRF` model defined in `run_nerf_helpers.py` file. In the `NeRF` class, the model will output four type of predictions, including raw `rgb`, `alpha`, `sf`, and `prob` for each sampled points. The output of `network_query_fn` as defined in `run_network` should be

```python
raw_ref:        # [num_rays, num_points, 3+1+6+2]
```
So according to this part of code, it defines the dynamic model introduced in Section 3.1, the forward and backward 3D scene flow are stored in `sf` for each point, and the predicted disocclusion weights w.r.t forward and backward flow are stored in `prob`.

The data `raw_ref` can be split into three variables, i.e., 
```python
raw_rgba_ref = raw_ref[:, :, :4]        # [num_rays, num_points, 4] raw rgb and alpha values for each sampling point, so what is the difference between this prediected RGBA and raw_rgba_rigid?
raw_sf_ref2prev = raw_ref[:, :, 4:7]    # [num_rays, num_points, 3] ref to previous time step t-1? motion direction for each point? 
raw_sf_ref2post = raw_ref[:, :, 7:10]   # [num_rays, num_points, 3] ref to posterior time step t+1? motion direction for each point?
```
This part of output is the dynamic scene representation. **Note that the dynamic rgb map is NOT obtained by converting the raw rgba values of each sampling points along each sampling rays. The dynamic rgb map and depth map will be rendered in the `raw2outputs_blending()` function**

## Image rendering by tntegrating a static
After we get these two sets of predictions (?) for each points, 
- the `raw_rgba_ref`, which are reference points' RGBA values predicted by the `NeRF` model, 
- the `raw_rgba_rigid`, which stores the raw RGBA values predicted by the `Rigid_NeRF` model for each sampled 3D points, 
- the `raw_blend_w`, which are the estimated weight for each sampling point,
- `z_vals` represents distance intervals along the sampling rays, 
- `rays_d` is used to calcuate the normalized ray direction for each ray
- `raw_noise_std`
  
All the above data will fed into the `raw2outputs_blending()` function.
the `raw_rgba_ref` denotes the dynamic raw results, and the `raw_rgba_rigid` represents the static representation (time in-dependent). Firstly, according to the computational graph, the `rgba` and `opacity` of dynmaic and static scene representations will be obtrained seperately by using `sigmoid` and `relu` activation function on the first three and the last one value, respectively. 

Then the `alpha` value of dynamic and static representations are obtained by accumulating transparency along that ray seperately, but the static and dynamic alpha values are reweighted by the `raw_blend_w`. According to the Equation (14) in Section 3.3 (without `rgb` value), the blended `alpha` is cacluated by a linear combination of static and dynamic scene components.

Then the calculation of `weights` for static and dynamic components are splited into two steps. The first step, calcuating the `Ts` values which accumuatated multiply the  `(1.0-alpha_dy)*(1.0-alpha_rig)` values. In the second step, multiply `Ts` with dynamic and static alpha values to get the weights for dynamic and static scene representations, respectively. Then, the `rgb_map` values are obtained by integrating static and dynamic compents with its corresponding weights.

To get the depth map, the weights of static and dynamic components are mixed togather by adding up the weights, and the depth map is obtained by re-weighting the accumulated marching distance of `z_val` and sum the values together for each camera ray.

Finally, the model also rendered a dynamic `rgb` map and a dynamic `depth` map by following the same volumetric render equation introduced in NeRF.

The output of the `raw2outputs_blending()` function is
```python
rgb_map,           # [N_rays, 3], rgb map by integrating the static and dynamic components 
depth_map,         # [N_rays, 1], depth map by integrating the static and dynamic weights
rgb_map_dy,        # [N_rays, 3], dynamic rgb map
depth_map_dynamic, # [N_rays, 1], dynamic depth map
weights_dynamic    # [N_rays, N_samples], weights for rendering the dynamic component.
```

## Back to the `render_rays()` function
The `raw2outputs_blending()` function returns the blended `rgb_map` (which is an intergration representation of the static and dynamic components), `depth_map`, and the dynamic rgb_map and depth map, as well as the weights for rendering the dynamic component. After that, three sets of results will be assembed into the `ret` dictionary, including (1) the blended rgb map and depth map, (2) dynamic rgb map, depth map and weights for rendering dynamic components, (3) static rgb map and depth map, (4)reference points to previous/next time step t-1/t+1? motion direction for each point.

what is those data used for? Is that mean, in the next step, we would like to try to calucate the motion for each sampling points?

### Recap some data
- `pts`: sampled points within each ray, [N_rays, N_samples, 3]
- `pts_ref`: `pts` concatenated with one value for the frame index embedding, [N_rays, N_samples, 4]

The next phrase is generating the raw rgb values for the sampled points in the previous and posterior time step. **Note that in this implementation, the model will not explicitly extract 3D points by shoot camera rays on the previous or next frame, but using a indicator value for the time step embedding, and using a motion offset to get the 3D points coordinates in the `t-1` and `t+1` time step. Thus, in this implementation, the fourth value in the `pts_ref` of each point can be replaced to form the previous or next time step 3D point representation, and the first three values w.r.t 3D coordinates are moved by adding up the motion from the reference time step `t` to posterior and previous time step to form the `pts_post` and `pts_prev`, respectivel. The newly generated coordinates with time step embedding will fed into the `NeRF` model in order to get the dynamic scene representations of the previous or next time step.** The reason behind this, maybe it aim to capture the coorespondance in the 3D space along the time dimension with repsect to the 3D point coordinates and its forward or backward motion offsets. 

Thus, in this part of code, two extra `forward network (fn)` operations are needed. Raw predictions of `rgba` values will be predicted for each points in the `t-1` and `t+1` time step. At the same time, the motion between the time step `t-1` to its previous time step `t-2`, and its posterior time step `t` will be predicted. **So the motion (`t->t-1`) and (`t-1->t`) are predicted.**). In addition, the motion from time step `t+1` to its previous time step `t` and to its posterior time step `t+2` are also predicted. **So the motion (`t->t+1`) and (`t+1->t`) are also predicted. Here, I think this is the way how the unsupervised training policy works. Moreover, although it randomly select one frame for sampling rays and 3D points in the space, it actually can cover the motion in between five consecutive time step along with the time dimenstion.(Note that, the time dimension is, more intuitively the index of frame, which is different with the concept in the ray casting for sampling points. On the other hand, at this point, only three time steps of dynamic scene representation are involved. We DONOT get the dynamic componets of `t-2` and `t+2` times step, but the motion information actually covers five time step.)**

After we get the `pts_prev` and `pts_post` by replacing the time step embedding of the fourth value in `pts_ref`, we fed the `pts_prev` and `pts_post` to the `foward network` function to get the raw `rgba` and `depth values` for each points of the `t-1` and `t+1` time step. Then the raw values will be fed into the `raw2outputs_warp()` function to get the dynamic rgb component of the previous time step, i.e., `rgb_map_pred_dy`, so as to the dynamic rgb component of the next time step, `rgb_map_post_dy`. (I have check the `raw2outputs_warp()` function. I found it quite similar with the orginal `raw2outputs` function as defined in the Equation (5) of the orginal NeRF paper.)

At this point, there will be three sets of dynamic scene representation, dynamic `rgb` map and its predicted motions w.r.t three time steps, i.e., `t-1`, `t` and `t+1`. The rest part of code, I think, should be how to caluate the cycle consistency loss of scene flow, geometric consistency loss and the single view depth prior loss as mentioned in the paper (Section 3.2). In addition, the `prob_map_prev` and `prob_map_post` are computed by using the `compute_2d_prob()` function. **Here, I was confused by the value of the weights of dynamic components `weights_prev_dy` and the `raw_prob_ref2prev`. Disocclusion is defined as "the situation where a previously occluded object becomes visible in computer graphics [ref:][[Disocclusion - Wikipedia](https://en.wiktionary.org/wiki/disocclusion)]. The weights of dynamic components denote the weights for sampling points contribute to the final rgb and depth map along each camera rays. If the disocclusion is range from 0 to 1, so if a point will become visible, then the value should be close to 1. so if the disocclusion of a point move from `rev` to `prev` becomes visible, then the probs should be close to 1, then why (1.0- prob) in compute_2d_prob(). Maybe it defined as the point will be invisible, then reduce the weights? I am confused by the definition of the `prob` values here. Maybe it does not matter?**

In the `if chain_bwd` block, it shows that, if the `chain_bwd` is true, then the dynamic component rendering of the time step will be involved, and the rendering process is activated by setting the bool indicator `chain_5frames` as `True`. Similar with the rendering process of `t-1`, the points in time step `t-2`, i.e., `pts_prevprev`, are modified by the movement, and time step embedding. An extra feed-forward process is performed to obtain the raw `rgba` values for each points in the `t-2` time step, as well as the motion from `t-2` to `t-1`. the `raw_prob_ref2prevprev` wiil be modified by following a chain-rule as:
```python
raw_prob_ref2prevprev = 1.0 - (1.0 - raw_prob_ref2prev) * (1.0 - raw_prob_prev2prevprev)
```
The `prob_map_pp` is also updated by using the `compute_2d_prob()` function with `weights_prevprev_dy` and `raw_prob_ref2prevprev` as input.

On the other hand, if the `chain_bwd` is False, the dynamic component rendering of the time step `t+2` will be computed by following the similar process. 

Finally, a dictionary with 

```python
ret = {
    #
    "rgb_map_ref": rgb_map_ref,
    "depth_map_ref": depth_map_ref,
    "rgb_map_rig": rgb_map_rig,
    "depth_map_rig": depth_map_rig,
    "rgb_map_ref_dy": rgb_map_ref_dy,
    "weights_ref_dy": weights_ref_dy,
    "depth_map_ref_dy": depth_map_ref_dy,
    "raw_sf_ref2prev": raw_sf_ref2prev,
    "raw_sf_ref2post": raw_sf_ref2post,
    #
    "raw_sf_prev2ref": raw_sf_prev2ref, #[N_rays, N_samples, 3]
    "rgb_map_prev_dy": rgb_map_prev_dy, #[N_rays, 3]
    "raw_sf_post2ref":raw_sf_post2ref ,# [N_rays, N_samples, 3]
    "rgb_map_post_dy": rgb_map_post_dy ,# [N_rays, 3]
    "prob_map_prev": prob_map_prev,
    "prob_map_post": prob_map_post,
    "raw_prob_ref2prev": raw_prob_ref2prev,
    "raw_prob_ref2post": raw_prob_ref2post,
    # 
    "raw_pts_pp" : pts_prevprev[:, :, :3],
    "rgb_map_pp_dy" : rgb_map_prevprev_dy,
    "prob_map_pp" : sum(weights_prevprev_dy, raw_prob_ref2prevprev, -1),
    "raw_sf_pp2p" : raw_sf_prevprev2prev,
    "raw_sf_p2pp" : raw_sf_prev2prevprev,
    "raw_prob_p2pp" : raw_prob_prev2prevprev,
    #
    "raw_pts_ref" : pts_ref[:, :, :3],
    "raw_pts_post": pts_post[:, :, :3],
    "raw_pts_prev": pts_prev[:, :, :3],
}
```








## TODO: Difference between the `NeRF`, `Rigid-NeRF` and `Original NeRF`?











