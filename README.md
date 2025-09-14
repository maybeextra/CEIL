# Camera-Proxy Enhanced Identity-Recalibration Learning

This is Official Repository for "Camera-Proxy Enhanced Identity-Recalibration Learning". The specific code will be released after being accepted.

## Requirements

### Installation

```shell
conda create --name CEIL python=3.10
conda activate CEIL
torch=2.1.1 cuda=11.8
conda env update --file environment.yml
cd code/extension
sh make.sh
```
### Prepare Datasets

```shell
mkdir dataset
```
Download the person datasets LLCM, RegDB, and the SYSU-MM01, and use code/data_process/pre_process to process, respectively.

## Update Time
-- 2024-12-10 We release the test models to github.

-- 2025-9-14 We release the train code to github.
## Result

### RegDB
| Mode      | Rank_1  | Rank_1 (R) |   mAP   | mAP (R) |  mINP   | mINP (R)                                   
|:----------|:-------:|:----------:|:-------:|:-------:|:-------:|:--------:|
| VIS to IR | ~91.97% |  ~94.30%   | ~89.13% | ~95.29% | ~82.75% | ~95.49%  | 
| IR to VIS | ~92.59% |  ~94.97%   | ~89.09% | ~95.57% | ~81.76% | ~95.44%  | 

[Model](https://1drv.ms/f/c/de0254e500a56cf5/EumOQPiXuiVHugiOO1BYaWcB2ZN06mbN7sr4x7Nk7Rr_Fw?e=PUBFJq)

### SYSU-MM01
| Mode          | Rank_1  | Rank_1 (R) |   mAP   | mAP (R) |  mINP   | mINP (R) |                                                                                        
|:--------------|:-------:|:----------:|:-------:|:-------:|:-------:|:--------:|
| All Search    | ~68.19% |  ~86.75%   | ~63.68% | ~83.97% | ~48.65% | ~75.24%  |
| Indoor Search | ~73.28% |  ~91.25%   | ~76.98% | ~92.21% | ~72.91% | ~90.60%  |

[Model](https://1drv.ms/f/c/de0254e500a56cf5/ErMeM7R5vnBCpCxB1Drh4lwBo3stBy9RZQp2sPymmzrX6A?e=uufsmj)

### LLCM
| Mode      | Rank_1  | Rank_1 (R) |   mAP   | mAP (R) |  mINP   | mINP (R) |
|:----------|:-------:|:----------:|:-------:|:-------:|:-------:|:--------:|
| VIS to IR | ~51.55% |  ~65.96%   | ~55.14% | ~68.19% | ~49.30% | ~63.79%  |
| IR to VIS | ~44.39% |  ~57.65%   | ~49.95% | ~62.49% | ~49.95% | ~59.36%  |

[Model](https://1drv.ms/f/c/de0254e500a56cf5/EgQEzVvqzw9PjNKPwKWWSc0BEcA1ASCah_rJnnIMDLuMOg?e=HZh0As)
