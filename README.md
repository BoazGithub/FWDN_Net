# <h1 align="center"><b>FWDNNet: Cross-Heterogeneous Encoder Fusion via Feature-Level TensorDot Operations for Land-Cover Mapping</b><br></h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python 3.7+">
  <img src="https://img.shields.io/badge/PyTorch-1.7.1-blue.svg" alt="PyTorch 1.7.1">
  <img src="https://img.shields.io/badge/License-Research-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Accepted-brightgreen.svg" alt="Status">
</p>

<p align="center">
  <strong>Official PyTorch Implementation</strong><br>
  Accepted at <strong>IEEE Transactions on Geoscience and Remote Sensing (TGRS) 2025</strong>
</p>

---

## 📢 Updates

| Date | Announcement |
|------|-------------|
| 🎉 **January 2025** | **FWDNNet has been accepted for publication in IEEE TGRS!** |
| 📝 **November 2024** | Manuscript submitted to IEEE TGRS |
| 🚀 **October 2024** | Code and datasets released |

---

## <h2 align="left">Authors <br></h2>

[![Author](https://img.shields.io/badge/Boaz-MWUBAHIMANA-orange.svg)](https://github.com/BoazGithub) 
[![Author](https://img.shields.io/badge/YAN-Jianguo-orange.svg)](http://www.lmars.whu.edu.cn/enjianguo-yan/) 
[![Author](https://img.shields.io/badge/Swalpa-KumarRoy-orange.svg)](https://ieeexplore.ieee.org/author/37086689617) 
[![Author](https://img.shields.io/badge/Maurice-Mugabowindekwe-orange.svg)](https://researchprofiles.ku.dk/en/persons/maurice-mugabowindekwe) 
[![Author](https://img.shields.io/badge/Xiao-Huang-orange.svg)](https://envs.emory.edu/people/bios/Huang-Xiao%20.html) 
[![Author](https://img.shields.io/badge/Elias-Nyandwi-orange.svg)](https://cst.ur.ac.rw/?Dr-Elias-Nyandwi-723) 
[![Author](https://img.shields.io/badge/Eric-Habineza-orange.svg)](https://www.linkedin.com/in/eric-habineza-79559519b/?originalSubdomain=rw) 
[![Author](https://img.shields.io/badge/Fidele-Mwizerwa-orange.svg)](https://cst.ur.ac.rw/?Mrs-Fidele-MWIZERWA) 
[![Author](https://img.shields.io/badge/Joseph-Tuyishimire-orange.svg)](https://cst.ur.ac.rw/?Mr-Joseph-Tuyishimire) 
[![Author](https://img.shields.io/badge/Dingruibo-Miao-orange.svg)](https://ieeexplore.ieee.org/author/37089315877)

---

## 📖 Abstract

We present **FWDNNet**, a novel encoder-decoder architecture that integrates heterogeneous deep learning backbones through innovative **TensorDot fusion modules** for high-resolution land cover mapping. Unlike traditional fusion approaches that rely on simple concatenation or averaging, FWDNNet preserves tensor structures while enabling adaptive, probabilistic feature weighting across five specialized backbone encoders: **ResNet34**, **InceptionV3**, **VGG16**, **EfficientNet-B3**, and **Swin Transformer**.

### 🎯 Key Achievements

- **95.3%** Overall Accuracy and **91.8%** mIoU
- **21%** faster inference speed (58.2ms)
- **86.6%** memory reduction
- **97.1%** cross-domain transfer score
- **53.7%** training time reduction

---

## 🏗️ Architecture Overview

![FWDNNet Architecture](path/to/architecture_figure.png)

FWDNNet consists of four core components:

1. **Heterogeneous Fusion Encoders**: Five specialized backbones for parallel feature extraction
2. **TensorDot Fusion Modules**: High-order feature integration preserving tensor structures
3. **Probabilistic Attention Weighting**: Adaptive backbone selection via variational inference
4. **Unified Decoder Pathway**: Efficient spatial resolution recovery

### Network Components

```
FWDNNet
├── Encoders (Heterogeneous Backbones)
│   ├── ResNet34 (Residual Learning)
│   ├── InceptionV3 (Multi-scale Receptive Fields)
│   ├── VGG16 (Hierarchical Features)
│   ├── EfficientNet-B3 (Computational Efficiency)
│   └── Swin Transformer (Global Context)
├── TensorDot Fusion Module
│   ├── Tucker Decomposition
│   └── Multilinear Transformations
├── Probabilistic Attention
│   └── Variational Inference Weighting
└── Unified Decoder
    └── Multi-scale Feature Recovery
```

---

## 📊 Datasets

### Dataset Overview

| Dataset | Resolution | Images/GT | Area (km²) | Classes |
|---------|-----------|-----------|------------|---------|
| **Dubai** | 0.31-2.4m | 1,500 | 450 | Built-up, Vegetation, Water, Other |
| **Nyagatare-Rwanda** | 0.5-1.07m | 2,200 | 1,200 | Agricultural landscapes |
| **Oklahoma-USA** | 0.5-0.60m | 1,800 | 2,800 | Great Plains ecosystems |

### Study Areas

![Study Areas Map](path/to/study_areas_map.png)

- **Nyagatare & Rwamagana, Rwanda**: Agricultural landscapes in Eastern Province
- **Oklahoma, USA**: NAIP imagery over U.S. Great Plains (2019-2024)
- **Dubai, UAE**: Dense urban infrastructure with mixed land use

---

## 🚀 Getting Started

### Requirements

```bash
Python 3.7+
PyTorch 1.7.1
torchvision 0.8.2
OpenCV 4.5.5
CUDA Toolkit 10.1
Wandb 0.13.10
numpy
matplotlib
scikit-learn
```

### Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/FWDNNet.git
cd FWDNNet

# Create conda environment
conda create -n fwdnnet python=3.7
conda activate fwdnnet

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

1. Download the datasets:
   - **sKwanda_V2 Dataset**: [[Download Link](https://drive.google.com/file/d/1X_Fz7LQIeix3rV3K29FBfKiU1WMdROe-/view?usp=drive_link)]
   - **FWDNNet Pretrained Models**: [[Download Link](https://drive.google.com/drive/folders/1h9T6w84P8b2xyD81at4JMn30VekAS53E?usp=drive_link)]

2. Organize the dataset structure:

```
FWDNNet/
├── data/
│   ├── Dubai/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/
│   │   └── test/
│   ├── Nyagatare/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── Oklahoma/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
├── configs/
└── utils/
```

---

## 🔧 Training

### Train from Scratch

```bash
python train.py \
  --dataset Dubai \
  --batch_size 16 \
  --epochs 200 \
  --lr 1e-3 \
  --config configs/fwdnnet_dubai.yaml
```

### Training Configuration

```yaml
# Example config file (configs/fwdnnet_dubai.yaml)
model:
  name: FWDNNet
  encoders: [resnet34, inception_v3, vgg16, efficientnet_b3, swin_t]
  
training:
  batch_size: 16
  epochs: 200
  learning_rate: 1e-3
  optimizer: AdamW
  weight_decay: 0.01
  
loss:
  focal_weight: 1.0
  consistency_weight: 0.5
  uncertainty_weight: 0.3
  diversity_weight: 0.2
  boundary_weight: 0.4
```

### Resume Training

```bash
python train.py \
  --dataset Dubai \
  --resume checkpoints/fwdnnet_epoch_50.pth
```

---

## 🧪 Evaluation

### Test on Single Dataset

```bash
python test.py \
  --dataset Dubai \
  --checkpoint checkpoints/fwdnnet_best.pth \
  --visualize
```

### Cross-Domain Evaluation

```bash
python test.py \
  --source_dataset Dubai \
  --target_dataset Nyagatare \
  --checkpoint checkpoints/fwdnnet_dubai.pth
```

### Inference on Custom Images

```bash
python inference.py \
  --input_path /path/to/your/image.tif \
  --checkpoint checkpoints/fwdnnet_best.pth \
  --output_path results/prediction.png
```

---

## 📈 Results

### Quantitative Results

| Model | Accuracy (%) | mIoU (%) | F1-Score | Inference Time (ms) | Parameters (M) |
|-------|-------------|----------|----------|---------------------|----------------|
| ResNet-34 | 93.5 | 89.0 | 0.800 | 45.2 | 24.0 |
| InceptionV3 | 80.1 | 84.0 | 0.832 | 52.7 | 30.0 |
| Swin-T | 89.2 | 86.1 | 0.847 | 67.3 | 28.0 |
| HRNet-W32 | 94.1 | 90.1 | 0.862 | 73.8 | 41.0 |
| **FWDNNet (Ours)** | **95.3** | **91.8** | **0.876** | **58.2** | **35.0** |

### Performance by Dataset

| Dataset | Overall Accuracy | mIoU | Transfer Score |
|---------|-----------------|------|----------------|
| Dubai | 96.1% | 92.4% | - |
| Nyagatare | 94.7% | 91.1% | 97.2% |
| Oklahoma | 95.0% | 91.6% | 96.9% |
| **Average** | **95.3%** | **91.8%** | **97.1%** |

### Qualitative Results

![Qualitative Results](path/to/qualitative_results.png)

*Comparison of FWDNNet predictions with baseline methods across different datasets.*

---

## 📊 Visualization

### Training Curves

![Training Convergence](path/to/training_curves.png)

### Feature Maps

![Feature Visualizations](path/to/feature_maps.png)

### Attention Weights

![Attention Distribution](path/to/attention_weights.png)

---

## 🔬 Ablation Studies

| Configuration | mIoU (%) | Improvement |
|--------------|----------|-------------|
| Baseline (Single Encoder) | 89.0 | - |
| + Multi-Encoder Fusion | 90.1 | +1.1% |
| + TensorDot Fusion | 91.3 | +2.3% |
| + Probabilistic Attention | 91.8 | +2.8% |
| **Full FWDNNet** | **91.8** | **+2.8%** |

---

## 💻 Computational Efficiency

### Performance Comparison

| Metric | FWDNNet | HRNet-W32 | Improvement |
|--------|---------|-----------|-------------|
| Training Time | 6.2h | 13.4h | **-53.7%** ↓ |
| Memory Usage | 12.85GB | 95.74GB | **-86.6%** ↓ |
| Parameters | 35.0M | 41.0M | **-14.6%** ↓ |
| FLOPs | 45.2G | 52.1G | **-13.2%** ↓ |
| Throughput | 17.2 img/s | 13.5 img/s | **+27.4%** ↑ |

---

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{mwubahimana2025fwdnnet,
  title={FWDNNet: Cross-Heterogeneous Encoder Fusion via Feature-Level TensorDot Operations for Land-Cover Mapping},
  author={Mwubahimana, Boaz and Yan, Jianguo and Roy, Swalpa Kumar and Mugabowindekwe, Maurice and Huang, Xiao and Nyandwi, Elias and Habineza, Eric and Mwizerwa, Fidele and Tuyishimire, Joseph and Miao, Dingruibo},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
```

---

## 🔗 Related Work

Our previous works on land cover mapping:

- **VHF-ParaNet**: Vision Transformers Feature Harmonization Network [[Code](https://github.com/BoazGithub/VHF_ParaNet)] [[Paper](#)]
- **C2FNet**: Coarse-to-Fine Network [[Paper](#)]

---

## 🙏 Acknowledgments

This research was supported by:
- State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing, Wuhan University
- IEEE Transactions on Geoscience and Remote Sensing

We thank the open-source community for providing valuable resources:
- **PyTorch** team for the deep learning framework
- **Hugging Face** for transformer implementations
- **ESRI** and **Google Earth Engine** for satellite imagery access

---

## 📞 Contact

For questions, collaborations, or issues:

- **Boaz Mwubahimana**: aiboaz1896@gmail.com
- **Yan Jianguo** (Corresponding Author): jgyan@whu.edu.cn

Feel free to open an issue or pull request!

---

## 📄 License

The code and datasets are released for **non-commercial and research purposes only**. For commercial purposes, please contact the authors.

```
Copyright (c) 2025 Wuhan University
Licensed under MIT License for research use
```

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YourUsername/FWDNNet&type=Date)](https://star-history.com/#YourUsername/FWDNNet&Date)

---

<p align="center">
  Made with ❤️ by the FWDNNet Team
</p>

<p align="center">
  <a href="#top">Back to Top ⬆️</a>
</p>
}

