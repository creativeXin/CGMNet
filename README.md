# CGMNet
A hyperspectral coastal wetland change detection dataset for Hangzhou Bay (HZB), and the proposed CGMNet: A Center-Pixel and Gated Mechanism-Based Attention Network for Hyperspectral Change Detection.

The source code will be made publicly available soon.

A detailed description of the dataset and its construction procedure can be found in: https://ieeexplore.ieee.org/document/11393652.

## 🚀 Getting Started

### 📦 Installation

# Install required Python packages
python = 3.8
torch = 1.13.1
torchaudio = 0.13.1
``` 

## 📁 Data Preparation

The HZB dataset can be downloaded from **Baidu Netdisk**:  
- Link: https://pan.baidu.com/s/16wYNHF85f1z3EvtKEfaRWg?pwd=m74x  
- Extraction code: `m74x`

After downloading, place the files under `dataset/HZB/` in the project root:

```text
dataset/
└── HZB/
    ├── image1_ZY.mat
    ├── image2_ZY.mat
    └── GT_01_400.mat
```

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

