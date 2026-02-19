# CGMNet

This repository provides the Hangzhou Bay (HZB) hyperspectral coastal wetland change detection dataset and introduces CGMNet: A Center-Pixel and Gated Mechanism-Based Attention Network for Hyperspectral Change Detection. The source code will be made publicly available soon. A detailed description of the dataset and its construction procedure can be found in: https://ieeexplore.ieee.org/document/11393652.

## 🚀 Getting Started

### 📦 Installation

We recommend using the following environment: Python 3.8, PyTorch 1.13.1, and torchaudio 0.13.1.

## 📁 Data Preparation

The HZB dataset can be downloaded from Baidu Netdisk via the following link: [https://pan.baidu.com/s/16wYNHF85f1z3EvtKEfaRWg?pwd=m74x](https://pan.baidu.com/s/16wYNHF85f1z3EvtKEfaRWg?pwd=m74x). The extraction code is `m74x`. After downloading, place the files under `dataset/HZB/` in the project root directory as follows:

```text
dataset/
└── HZB/
    ├── image1_ZY.mat
    ├── image2_ZY.mat
    └── GT_01_400.mat


## 📚 Citation

If you find this project helpful for your research, please kindly consider citing our paper and giving this repo a ⭐:

```bibtex
@ARTICLE{11393652,
  author={Wu, Lanxin and Peng, Jiangtao and Yang, Bing and Sun, Weiwei and Huang, Mingzhu},
  journal={IEEE Transactions on Image Processing}, 
  title={CGMNet: A Center-Pixel and Gated Mechanism-Based Attention Network for Hyperspectral Change Detection}, 
  year={2026},
  volume={35},
  number={},
  pages={1951-1965},
  keywords={Feature extraction;Convolutional neural networks;Logic gates;Transformers;Sea measurements;Attention mechanisms;Sensor phenomena and characterization;Hyperspectral imaging;Data mining;Noise;Hyperspectral image;change detection;central pixel;gated mechanism;global information},
  doi={10.1109/TIP.2026.3661851}}

