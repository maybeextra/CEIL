# Camera-Proxy Enhanced Identity-Recalibration Learning (CEIL)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-purple.svg)](10.1109/TCSVT.2025.3548939) 

**Official implementation of "Camera-Proxy Enhanced Identity-Recalibration Learning"**


> **Abstract:** Visible-Infrared person Re-Identification (VI-ReID) involves querying images of the same person across visible and infrared modalities. To minimize annotation costs, Unsupervised Visible-Infrared person Re-Identification (UVI-ReID) using pseudo-label contrastive learning has emerged. Traditional UVI-ReID approaches often neglected camera domain information and relied on inadequate update strategies during training, only using cosine distance for testing, which led to incorrect mapping of cross-modal relationships. To address these issues, we propose Camera-proxy Enhanced Identity-recalibration Learning (CEIL). It consists of two main stages: first, it employs intra-modal contrastive learning in conjunction with the camera-proxy, updates the memory bank using our innovative Difficulty-aware Cluster-based Memory Updating (DCMU) strategy, and applies Camera Domain-driven Local correlation (CDL) Loss to enhance the learning process. Then utilizes cross-modal contrastive learning, featuring our Proxy-enhanced Cross-modal Mapping (PCM) module, to recalibrate the identity relationships between different modalities. Graph network-based Camera constraint adjustment Re-ranking (GCR) method is adopted during test, utilizing camera domain information to recalibrate the correspondence between identities. Extensive experiments have demonstrated that CEIL achieving state-of-the-art performance on the SYSU-MM01, RegDB, and LLCM datasets and the GCR, as a general unsupervised re-ranking method, can further enhance performance of model on these datasets.

---
## 📢 News

*   **`2025-09-14`**: We are excited to release the official training code.
*   **`2024-12-10`**: Pre-trained models and inference code are now available. Check out the results section to download them.

---
## 📋 Table of Contents
- [🛠️ Requirements & Installation](#️-requirements--installation)
- [📁 Dataset Preparation](#-dataset-preparation)
- [🎯 Performance](#-performance)
- [📥 Model Zoo](#-model-zoo)

---
## 🛠️ Requirements & Installation

### Prerequisites
- Python 3.10
- PyTorch 2.1.1
- CUDA 11.8
- Linux or macOS (Windows is not officially supported)

### Step-by-Step Installation

We highly recommend using a Conda environment to manage dependencies.

1.  **Clone the repository**
    ```shell
    git clone https://github.com/your-username/CEIL.git  <!-- TODO: 替换为你的仓库地址 -->
    cd CEIL
    ```

2.  **Create and activate the Conda environment**
    ```shell
    conda create --name CEIL python=3.10 -y
    conda activate CEIL
    ```

3.  **Install PyTorch and CUDA dependencies**
    ```shell
    # It's recommended to follow the official PyTorch installation guide for your specific system
    # but here is the command for the specified version:
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
    ```

4.  **Install other dependencies**
    ```shell
    conda env update --file environment.yml
    ```

5.  **Compile C++/CUDA extensions**
    ```shell
    cd code/extension
    sh make.sh
    cd ../..
    ```

Now, your environment is set up and ready for training and testing.

---
## 📁 Dataset Preparation

1.  **Create a dataset directory**
    ```shell
    mkdir -p dataset
    ```

2.  **Download and Process Datasets**

    Download the following person Re-ID datasets and organize them within the `dataset/` directory.

    - [SYSU-MM01](http://www.sysu-hcp.net/sysumm01/)
    - [RegDB](https://github.com/shizenglin/Link-Partial-ReID)
    - [LLCM](https://github.com/VITA-Group/LLCM) <!-- TODO: 如果LLCM有官方链接，请替换 -->

3.  **Pre-processing**
    Use the provided scripts in the `code/data_process/pre_process` directory to process each dataset. For example:

    ```shell
    # Example for processing SYSU-MM01
    python code/data_process/pre_process/process_sysu.py --data_path ./dataset/SYSU-MM01

    # Example for processing RegDB
    python code/data_process/pre_process/process_regdb.py --data_path ./dataset/RegDB

    # Example for processing LLCM
    python code/data_process/pre_process/process_llcm.py --data_path ./dataset/LLCM
    ```

    *Note: Please adjust the script commands according to the actual usage instructions in the corresponding Python files.*

After running the scripts, your dataset directory should be structured properly for training and testing.

---
## 🎯 Performance

Our CEIL method achieves state-of-the-art performance on three popular cross-modal Re-ID benchmarks. **(R)** indicates results with re-ranking.

### Results on RegDB

| Mode      | Rank-1 | Rank-1 (R) | mAP   | mAP (R) | mINP  | mINP (R) |
|:----------|:------:|:----------:|:-----:|:-------:|:-----:|:--------:|
| VIS to IR | 91.97% | **94.30%** | 89.13%| **95.29%**| 82.75%| **95.49%**|
| IR to VIS | 92.59% | **94.97%** | 89.09%| **95.57%**| 81.76%| **95.44%**|

### Results on SYSU-MM01

| Mode          | Rank-1 | Rank-1 (R) | mAP   | mAP (R) | mINP  | mINP (R) |
|:--------------|:------:|:----------:|:-----:|:-------:|:-----:|:--------:|
| All Search    | 68.19% | **86.75%** | 63.68%| **83.97%**| 48.65%| **75.24%**|
| Indoor Search | 73.28% | **91.25%** | 76.98%| **92.21%**| 72.91%| **90.60%**|

### Results on LLCM

| Mode      | Rank-1 | Rank-1 (R) | mAP   | mAP (R) | mINP  | mINP (R) |
|:----------|:------:|:----------:|:-----:|:-------:|:-----:|:--------:|
| VIS to IR | 51.55% | **65.96%** | 55.14%| **68.19%**| 49.30%| **63.79%**|
| IR to VIS | 44.39% | **57.65%** | 49.95%| **62.49%**| 49.95%| **59.36%**|

---
## 📥 Model Zoo

You can download our pre-trained models from the links below. To use them, place the downloaded `.pth` files in the `logs/` directory (or as specified in your testing script).

### RegDB Models
[📥 Download from OneDrive](https://1drv.ms/f/c/de0254e500a56cf5/EumOQPiXuiVHugiOO1BYaWcB2ZN06mbN7sr4x7Nk7Rr_Fw?e=PUBFJq)

### SYSU-MM01 Models
[📥 Download from OneDrive](https://1drv.ms/f/c/de0254e500a56cf5/ErMeM7R5vnBCpCxB1Drh4lwBo3stBy9RZQp2sPymmzrX6A?e=uufsmj)

### LLCM Models
[📥 Download from OneDrive](https://1drv.ms/f/c/de0254e500a56cf5/EgQEzVvqzw9PjNKPwKWWSc0BEcA1ASCah_rJnnIMDLuMOg?e=HZh0As)

