# Implementation of the paper "Unveiling Structural Memorization: Structural Membership Inference Attack for Text-to-Image Diffusion Models"

### Requirements

**A suitable conda environment named ldm for running Latent Diffusion Model and Stable Diffusion can be created and activated with:

```conda env create -f environment.yaml

```conda activate ldm

### Run Structural MIA

**To execute Structural MIA over pretrained latent diffusion model, please execute the following command:

```cd scripts

```python mia.py --imgname /path/to/image_dir/image_name --H 256 --W 256 --scale 1.0 --prompt textual prompt of the image --yaml /path/to/model_yaml_file --ckpt /path/to/model_checkpoint


**Parameters:

--imgname: path to the image

--H: height of the image in pixel space

--W: width of the image in pixel space

--scale: unconditional guidance scale, eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))

--prompt: textual prompt of the image

--yaml: path to model's configuration file

--ckpt: path to model's checkpoint

### Evaluate Metrics

**To execute Structural MIA over pretrained latent diffusion model, please execute the following command:

```cd cal_metric

```python cal_auc_TPR.py --member_file /path/to/member_ssim_file --nonmember_file /path/to/nonmember_ssim_file

```python cal_asr_precision_recall.py --member_file /path/to/member_ssim_file --nonmember_file /path/to/nonmember_ssim_file --data_num_thresh 1000


**Parameters:

--member_file: path to members' file, which saves ssim values

--nonmember_file: path to non-members' file, which saves ssim values

--data_num_thresh: number of data selected for calculating threshold
