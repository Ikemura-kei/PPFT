<p align="center">

  <h1 align="center">Robust Depth Enhancement via Polarization Prompt Fusion Tuning</h1>
  <!-- <p align="center">
    <a href="https://youmi-zym.github.io"><strong>Youmin Zhang</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=jPvOqgYAAAAJ"><strong>Xianda Guo</strong></a>
    ·
    <a href="https://mattpoggi.github.io/"><strong>Matteo Poggi</strong></a>
    <br>
    <a href="http://www.zhengzhu.net/"><strong>Zheng Zhu</strong></a>
    ·
    <a href=""><strong>Guan Huang</strong></a>
    ·
    <a href="http://vision.deis.unibo.it/~smatt/Site/Home.html"><strong>Stefano Mattoccia</strong></a>
  </p>
  <h3 align="center"><a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_CompletionFormer_Depth_Completion_With_Convolutions_and_Vision_Transformers_CVPR_2023_paper.pdf">Paper</a> | <a href="https://www.youtube.com/watch?v=SLKAwrY2qjg&t=111s">Video</a> | <a href="https://youmi-zym.github.io/projects/CompletionFormer">Project Page</a></h3>
  <div align="center"></div>-->
</p> 
<!-- <p align="center">
  <a href="https://youmi-zym.github.io/projects/CompletionFormer">
    <img src="./media/architecture.png" alt="Logo" width="98%">
  </a>
</p>
<p align="center">
<strong>CompletionFormer</strong>, enabling both local and global propagation for depth completion.
</p> -->

## 1. Setting-Up Environment

## 2. Dataset
#### 2.1 Downloading HAMMER
Please run the following command to download the HAMMER dataset (in case the download link expires, please contact the original authors).   
```bash
./scripts/downloads/dataset.sh <LOCATION_OF_CHOICE>
```
By running the command, a dataset folder with path `<LOCATION_OF_CHOICE>/hammer` will be in place. Errors will be thrown if the path you indicate as in `<LOCATION_OF_CHOICE>` does not exist.

> Note: please note that the dataset zip file will not be removed after the download and the automatic unzipping, considering people may wish to keep it for other purposes (e.g. fast loading data into execution space on SLURMs). If you wish to get rid of the zip file, please perform removing manually.

#### 2.2 Processing HAMMER
We will add entries to the dataset to fascilitate more efficient training and testing. Specifically, we will process the polarization data to generate new data such as AoLP, DoLP, and etc., since reading raw data and computing them in each data loading occasion is extremely inefficient.   

Considering it is a good idea to keep the original dataset as it is in the storage device, __we will create a new folder to store everything we need to add (plus subset of original data we also need)__. Thus, we will NOT be using the originally downloaded dataset, but to create a new one named `<LOCATION_OF_CHOICE>/hammer_polar`.  

###### Step 1: Copy stuff
Please run the following command first, to copy data in the original dataset we need into the new data space:   

```bash
./script/data_processing/copy_hammer.sh <LOCATION_OF_CHOICE>/hammer
```

A new dataset at path `<LOCATION_OF_CHOICE>/hammer_polar` will be created after successful command execution.

> Note: please note that `<LOCATION_OF_CHOICE>/hammer` refers to the path to the original hammer dataset you downloaded just in the step above. In case you had the HAMMER dataset before running this project, simply replace the path to the actual path to your dataset. Also note that we assume the dataset folder contains the name "hammer" to prevent people passing wrong dataset path accidentally.

###### Step 2: Generate new data
After copying necessary data from the original dataset to the new data space, please run the following to generate new samples we need:  

```bash
python ./scripts/data_processing/process_hammer.py
```

#### 2.3 \[Optional\] Creating symbolic link to dataset
It is usually a good practice to store datasets (which are large in general) to a shared location, and create symbolic links to individual project workspaces. In case you agree and wish to do this, please run the following command to create a symbolic link of the dataset in this project space quickly. Of course you can also type the command for creating symbolic links manually, just make sure later you edit the training and testing scripts to pass a correct, alternative data root path.

```bash
./scripts/data_processing/symbolic_link_data.sh <LOCATION_OF_CHOICE>/hammer_polar
```
## 3. Model Checkpoints

#### 3.1 Foundation
We use CompletionFormer as our foundation model, thus a pre-trained checkpoint of it is required. Please run the following command to download the checkpoint. A folder named `ckpts` will be created under the project root, and the checkpoint named `NYUv2.pt` will be put beneath it.   

```bash
./script/downloads/foundation_ckpt.sh
```

#### 3.2 \[Optional\] Trained checkpoints
If one wishes to do testings, please download the checkpoints of the selected model type by running respective download commands as shown below:  

```bash
./script/downloads/model_ckpt_ppft.sh # for PPFT
./script/downloads/model_ckpt_ppft_shallow.sh # for PPFTShallow
./script/downloads/model_ckpt_ppft_scratch.sh # for PPFTScratch
./script/downloads/model_ckpt_ppft_freeze.sh # for PPFTFreeze
```

All checkpoints will be downloaded under `./ckpts`, the checkpoint file names are self-explanatory.   

## 4. Training

Experiment artifacts will be stored under `./experiments/<MODEL_NAME>_<YYYY-MM-DD-hh:mm:ss>`. For example, `./experiments/PPFT_2024-03-11-17:00:59`.

## 5. Inference & Evaluation
## 6. Acknowledgement
## 7. Cite