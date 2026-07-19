<p align="center">
</p>
<h2 align="center"> Progressive Multimodal Alignment for Continual Instruction Tuning (ACM MM2026)</a></h2>

This repository contains our PMA code for DISCO under LLaVA. We sincerely thank the help of [MCITlib](https://github.com/Ghy0501/MCITlib).

## Benchmarks

We use the [UCIT](https://github.com/Ghy0501/HiDe-LLaVA) and [MLLM-DCL](https://github.com/bjzhb666/MLLM-CL) benchmarks. Please refer to the provided links to download the corresponding images and instruction sets, and organize them in the following directory structure:
```
|--your_path
    |-- Domain_data
        |-- AD
        |-- Med
        |-- RS
        |-- Sci
        |-- Fin
    |-- UCIT
        |-- datasets
        |-- ArxivQA
        |-- CLEVR-Math
        |-- Flickr30k
        |-- IconQA
        |-- ImageNet-R
        |-- VizWiz
```
Note: You need to modify the data path in all the scripts to your own path.

## Models

Please download it to your local directory for [LLaVA-1.5-7B](https://github.com/haotian-liu/LLaVA).
```
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir /your_path/llava-v1.5-7b
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir /your_path/clip-vit-large-patch14-336
```

Note: To meet the requirements of certain methods, we need to apply additional processing to the config file in the downloaded model. The details are outlined below:
1. add `"mm_text_select_layer": -1` and `"mm_text_tower": "/your_path/clip-vit-large-patch14-336"` to the `config.py` in your local model weight path `/your_path/llava-v1.5-7b` and `/your_path/Internvl-chat-7b`.
2. remove `"temperature": 0.9` and `"top_p": 0.6` in the `generation_config.json` of your local model weight path.

## 🏃 How to run

Note: Our experiment is conducted in a CUDA 11.8 environment, with most libraries in the setup aligned to this CUDA version. Therefore, we recommend using `nvcc -V` to check the CUDA version on your current server. If it does not match, please install CUDA 11.8 before proceeding.

### 1. Install Package
```
conda create -n PMA python=3.10 -y
conda activate PMA
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
cd LLaVA/LoRA-FT
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
```
For installing [flash-attn](https://github.com/Dao-AILab/flash-attention/releases), we recommend downloading version 2.6.3 from the official repository according to your CUDA and PyTorch versions, and placing it in a local directory for manual installation. For example:
```
pip install flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
We also provide an `environment.yml` file to help users identify missing dependencies and version mismatches. However, due to potential library conflicts, automatic installation may fail to install certain packages. We therefore recommend manually installing them based on the provided error messages and version specifications. For essential evaluation-related dependencies, please refer to the [UCIT](https://github.com/Ghy0501/HiDe-LLaVA) and [MLLM-CL](https://github.com/bjzhb666/MLLM-CL) repositories.

### 2. Modify path and parameter settings

Before running, please set all the model paths to your local paths. The paths that need to be modified are listed below, and don’t forget to update the dataset path as well.

- Change `/home/xxxxxxx/code/PMA` to `/your_path/PMA`.
- Change `/mnt/clover/xxxxxxx/pre_trained/llava-v1.5-7b` to `/your_path/llava-v1.5-7b`.
- Change `/mnt/clover/xxxxxxx/pre_trained/clip-vit-large-patch14-336` to `/your_path/clip-vit-large-patch14-336`.
- Change `/mnt/clover/xxxxxxx/PMA_checkpoint` to `/your_path/checkpoint`.

After adjusting the path, users can modify parameters like `gpu_num` based on their actual operating environment. All parameter settings are integrated into the `configs/` folder.

Note: We recommend using the `Find in Folder` command in VS Code for search and replace operations.

### 3. Training and Evaluation

We provide predefined training and testing hyperparameters in the `configs` files within each method's directory, which can be adjusted as needed. The corresponding training and testing scripts are located in the `scripts` directory. Once all paths are correctly configured, the scripts should execute without issues. For example:
```
cd PMA/LLaVA/PMA_disco
sh scripts/PMA/Train/train_DCL.sh
```
The program will automatically perform both training and inference. 

## Citation

```
@inproceedings{zhang2026pma,
  title={Progressive Multimodal Alignment for Continual Instruction Tuning},
  author={Zhang, Duzhen and Yu, Yahan and Su, Qiaoyi and Dong, Jiahua and Zhang, Tielin},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia (ACM MM2026)},
  year={2026}
}
```

