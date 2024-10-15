# Camera-Proxy Enhanced Identity-Recalibration Learning

This is Official Repository for "Camera-Proxy Enhanced Identity-Recalibration Learning". The specific code will be released after being accepted by AAAI.

## Requirements

### Installation

```shell
torch=2.1.1 cuda=11.8 python=3.10
cd code/extension
sh make.sh
conda install conda-forge::faiss-gpu=1.7.4 cudatoolkit=11.8
pip install scikit-learn==1.4.2 scipy==1.13.1 ftfy==6.2.0 regex==2024.7.24 tqdm==4.66.4 easydict==1.13 pyyaml==6.0.1 tensorboard==2.17.0 matplotlib==3.8.4
```
### Prepare Datasets

```shell
cd examples && mkdir data
```
Download the person datasets LLCM, RegDB, and the SYSU-MM01.

## Update Time
-- 2024-8-5 We release the test models to github.

## Trained models
| Datasets  | Pretrained | Rank@1 | Rank@1 (R) | mAP | mAP (R) | Model(pth)                                                                                                 |
|:-----------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| RegDB     | CLIP     | ~92.39% | ~95.02% | ~90.05% | ~95.72% | [best_epoch_78.pth](https://1drv.ms/f/c/de0254e500a56cf5/EpjRASk4JZ9DqFUVcePR_HYBAkCvze9v9F3yX01PKZLl2w?e=cumSeZ) |
| SYSU-MM01 | CLIP     | ~66.01% | ~84.78% | ~62.01% | ~83.47% | [best_epoch_84.pth](https://1drv.ms/f/c/de0254e500a56cf5/EnSNRwNF0X1IrE2w5px3ic8BJQELB8OG1ZKPj037jfVUPA?e=DhHJnP) |
| LLCM      | CLIP     | ~51.15% | ~65.60% | ~54.79% | ~68.31% | [best_epoch_81.pth](https://1drv.ms/f/c/de0254e500a56cf5/EkPKNa_gkY1NrUBfrd41plkBaB4N0QwOEBvc6m0ns5HicQ?e=yn0tPi) |
