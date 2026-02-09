# Banknote Recognition with Deep Learning (USD & THB)

An empirical study of banknote denomination recognition using deep learning, transfer learning, ablation experiments, multimodal OCR features, and model interpretability with Grad-CAM.

This repository contains the implementation and experiments for the project: "An Empirical Study of Banknote Recognition Using Deep Learning".

---

## Project Overview

Banknote recognition is a computer vision task with applications in ATMs, cash sorting machines, counterfeit detection, and assistive technologies. This study presents a comparative analysis of deep learning-based banknote recognition on United States Dollar (USD) and Thai Baht (THB) currencies under controlled experimental conditions.

This project studies automatic denomination recognition for:
- **United States Dollar (USD)**: 5 denominations ($5, $10, $20, $50, $100)
- **Thai Baht (THB)**: 5 denominations (20, 50, 100, 500, 1000 Baht)

The goal is to understand how deep learning models behave under different training strategies and feature representations, specifically examining the role of color information, multi-currency training, OCR-based features, and model interpretability.

---

## Research Questions

**RQ1: How does colour information influence banknote classification performance?**

Models are trained and evaluated using both colour and greyscale images to assess the relative importance of colour cues compared to shape and texture features. It is hypothesised that removing colour information will lead to a greater reduction in classification accuracy for Thai Baht banknotes than for United States Dollar banknotes, reflecting differences in reliance on colour-based visual cues between the two currencies.

**RQ2: What are the effects of joint multi-currency training on classification performance and error characteristics?**

A unified model is trained on a combined dataset comprising both USD and THB banknotes and evaluated against single-currency baseline models. This analysis investigates whether joint training leads to positive feature transfer or negative interference between currencies.

**RQ3: How do saliency maps reveal differences in feature reliance between USD and THB banknote recognition models?**

Saliency mapping techniques (Grad-CAM) are applied to analyse model attention and identify image regions that contribute most strongly to classification decisions. This analysis aims to characterise differences in feature reliance between USD and THB recognition models.

**RQ4: What is the contribution of OCR-based textual features to banknote classification accuracy?**

Optical character recognition (OCR) is used to extract denomination-related text and numerals from banknote images. The performance of image-only convolutional neural network models is compared with multi-modal models that combine visual and textual features.

---

## Key Findings

- **USD banknotes are highly robust to grayscale conversion** (only 2% accuracy drop: 100% → 98%)
- **THB banknotes rely more strongly on color information** (7% accuracy drop: 98% → 91%)
- **Joint USD + THB training does not degrade accuracy** and slightly improves THB performance (97% → 97.5%)
- **OCR-based multimodal features reduce accuracy** due to noisy and unreliable text extraction (USD: 100% → 96%, THB: 95% → 85%)
- **Grad-CAM reveals currency-specific attention patterns**:
  - USD models focus on portrait edges (chin, ear, hair) and structural elements
  - THB models attend to color-rich regions, numerals, and Thai script
  - Joint models learn broader feature representations with positive transfer

---

## Models and Experiments

### Image-only CNN

**Architecture:**
- Backbone: ResNet-50 (ImageNet pretrained using `ResNet50_Weights.IMAGENET1K_V1`)
- Input size: 224 × 224
- Loss: Cross-Entropy
- Optimizer: AdamW with decoupled weight decay
- Transfer learning with fine-tuning (final FC layer replaced for 5 or 10 classes)

**Training Configuration:**
- Learning rate: 3e-4 (0.0003)
- Weight decay: 1e-4 (0.0001)
- Maximum epochs: 50
- Early stopping:
  - Patience: 7 epochs
  - Minimum improvement (delta): 1e-4 (0.0001)
  - Warmup period: 5 epochs minimum before early stopping activates
- Gradient clipping: max norm of 1.0
- Batch size: 32
- Seed: 42 (for reproducibility)

### Ablation Studies

1. **RGB vs Grayscale training** (RQ1)
   - Separate models trained for USD and THB
   - Each currency trained independently
   - Evaluates the importance of color information
   - Both color modes tested on same architecture

2. **Single-currency vs joint-currency training** (RQ5)
   - Single-currency: USD model (5 classes), THB model (5 classes)
   - Joint model: Universal model (10 classes: 5 USD + 5 THB)
   - Tests multi-task learning effects
   - Analyzes cross-currency interference and positive transfer

### Multimodal Model

**Architecture:**
- Visual backbone: ResNet-18
- OCR engines: Tesseract OCR / EasyOCR
- Feature engineering pipeline:
  - Binary indicators for denomination presence
  - Frequency statistics of detected numerals
  - Currency symbol detection (฿, $, €)
  - Country-related keyword detection
- Fusion: OCR features concatenated with 512-dim CNN embeddings

**OCR Feature Engineering:**
- Semantic numeric features (denomination values)
- Structural text features (text length, symbol presence)
- Country-specific keywords (e.g., "USA", "America")

### Interpretability

- Method: Grad-CAM (Gradient-weighted Class Activation Mapping)
- Target layer: `model.layer4[-1]` (final convolutional block of ResNet-50)
- Visualization: 2 images per denomination from unseen test set
- Comparison: USD-only, THB-only, and joint (universal) models
- Implementation: Uses `pytorch_grad_cam` library

---

## Datasets

### USD Dataset

- **Source**: Kaggle (USD Bill Classification Dataset) [4]
- **Original denominations**: 6 classes
- **Used denominations**: 5 classes ($5, $10, $20, $50, $100) - excluded $1 for symmetry with THB
- **Training + Validation images**: 800 (160 per denomination)
- **Unseen test images**: 200 (40 per denomination)
- **Characteristics**: Cropped images from Google Images, monochromatic design

### THB Dataset

- **Source**: Mendeley Data (Thai Banknote Dataset) [5]
- **Denominations**: 5 classes (20, 50, 100, 500, 1000 Baht)
- **Training + Validation images**: 800 (160 per denomination)
- **Unseen test images**: 200 (40 per denomination)
- **Characteristics**: "In-the-wild" images with backgrounds, higher percentage of folded/rotated notes, polychromatic design

### Data Standardization and Alignment

To facilitate controlled comparative analysis:

1. **Class Selection and Dimensionality Alignment**:
   - USD reduced from 6 to 5 denominations (excluded $1)
   - Ensures symmetry with THB dataset structure

2. **Class Balancing**:
   - 200 images per denomination total
   - 160 used for training + validation
   - 40 reserved for final unseen testing

### Preprocessing Pipeline

**Image Transformations:**
- Resized to 224 × 224 (ResNet-50 input requirement)
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Optional grayscale conversion (for ablation studies)

**Data Augmentation (Training only):**
- Random rotation: [-180°, +180°] (addresses THB dataset rotation variance)
- Color jittering: brightness (0.2), contrast (0.2)
- Simulates varied lighting conditions and sensor noise

**Data Partitioning (Stratified):**
- Training: 70% (560 images)
- Validation: 15% (120 images)
- Test: 15% (120 images) - used during training for validation
- **Unseen test**: Separate 200 images (40 per class) held out completely

Note: This repository does not redistribute dataset images. Please download datasets separately and follow their licenses.

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (optional, CPU supported)

### Setup

Create a virtual environment:

```bash
$ python -m venv .venv
$ source .venv/bin/activate   # Linux / macOS
$ .venv\Scripts\activate       # Windows
```

Install dependencies:

```bash
$ pip install -r requirements.txt
```

### Example requirements.txt:

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.7.0
pillow>=9.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
pandas>=1.5.0
tqdm>=4.65.0
pytesseract>=0.3.10
easyocr>=1.7.0
pytorch-grad-cam>=1.4.0
```

**System Dependencies:**

Tesseract OCR requires system-level installation:

```bash
# Ubuntu/Debian
$ sudo apt-get install tesseract-ocr

# macOS
$ brew install tesseract

# Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

---

## Repository Structure

```
ML_Currency_Detection/
│
├── data/
│   ├── USD/                    # USD banknote images (by denomination folders)
│   ├── THB/                    # THB banknote images (by denomination folders)
│   └── splits/                 # Generated CSV files for train/val/test splits
│       ├── usd_train.csv
│       ├── usd_val.csv
│       ├── usd_unseen_test.csv
│       ├── thb_train.csv
│       ├── thb_val.csv
│       └── thb_unseen_test.csv
│
├── test/
│   ├── USD/                    # Unseen USD test images
│   └── THB/                    # Unseen THB test images
│
├── outputs/
│   ├── rq1/                    # Separate model outputs
│   │   ├── usd/
│   │   └── thb/
│   └── rq5/                    # Joint model outputs
│       └── universal/
│
├── RQ1.ipynb                   # Separate currency models (USD-only, THB-only)
├── RQ5.ipynb                   # Joint/Universal model (USD + THB)
├── requirements.txt
└── README.md
```

---

## Usage

### Training Separate Models (RQ1)

Open and run `RQ1.ipynb` to:
- Train separate ResNet-50 models for USD and THB
- Evaluate on RGB and grayscale images
- Generate confusion matrices and per-class accuracy
- Visualize Grad-CAM saliency maps

**Key components:**
- Separate USD and THB datasets with stratified splits
- Independent training loops with early stopping
- Final evaluation on unseen test sets

### Training Joint Model (RQ5)

Open and run `RQ5.ipynb` to:
- Train a single universal ResNet-50 model on combined USD + THB data
- Evaluate performance on both currencies simultaneously
- Compare against separate models from RQ1
- Generate Grad-CAM visualizations for joint model

**Key components:**
- Combined dataset (10 classes total)
- Unified training loop
- Per-currency accuracy reporting
- Cross-currency saliency analysis

### Grad-CAM Visualization

Both notebooks include Grad-CAM visualization:

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Target the final convolutional block
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Generate heatmaps
grayscale_cam = cam(input_tensor=input_tensor, targets=None)
```

Outputs:
- Original image alongside Grad-CAM heatmap
- Predicted class with confidence score
- 2 images per denomination for diversity

---

## Results Summary

### Effect of Colour Information

| Currency | Color Accuracy | Grayscale Accuracy | Accuracy Drop |
|----------|----------------|-------------------|---------------|
| USD | 100% | 98% | -2% |
| THB | 98% | 91% | -7% |

**Interpretation:**
- USD: Primarily structural/textural features (monochromatic design)
- THB: Heavy reliance on color as distinguishing feature (polychromatic design)

**Per-class accuracy (Color models on unseen test):**
- **USD**: All denominations achieved 100% accuracy
- **THB**: 
  - THAI100: 90% (hardest class)
  - THAI1000: 97.5%
  - THAI50: 97.5%
  - THAI20: 100%
  - THAI500: 100%

### Effect of Joint Multi-Currency Training

| Currency | Separate Training | Joint Training | Change |
|----------|------------------|----------------|--------|
| USD | 100% | 100% | 0% |
| THB | 97% | 97.5% | +0.5% |

**Interpretation:**
- No performance degradation observed
- Slight improvement for THB suggests positive transfer
- Joint training is feasible without loss of accuracy
- Universal model learns 10 classes (5 USD + 5 THB) effectively

### OCR Multimodal Results

| Currency | Image-only | Multimodal | Change |
|----------|-----------|------------|--------|
| USD | 100% | 96% | -4% |
| THB | 95% | 85% | -10% |

**Interpretation:**
- OCR noise outweighs potential benefits
- Text features are unreliable for "in-the-wild" banknote images
- Visual features alone are superior for this task

### Saliency Analysis Findings

**USD-only Model:**
- Consistent attention to portrait edges (chin, ear, hair)
- High focus on denomination numerals (especially the "5" curvature)
- Uniformly high confidence (9/10 images at 1.0, 1 at 0.9)
- Ignores background and focuses on structural elements

**THB-only Model:**
- Varied attention spanning shapes, textures, numerals
- Strong focus on Thai script (denomination indicators)
- Attention to banknote regions (ignores background despite in-the-wild images)
- Color-rich regions highlighted more prominently

**Joint USD-THB Model (Universal):**
- Broader feature spectrum than single-currency models
- **USD banknotes**: portraits + high-contrast structural elements + bold text
- **THB banknotes**: shows negative interference (focuses on uniform royal portraits) but compensated by positive transfer
- Evidence of generalized capability for universal banknote features
- Successfully discriminates between 10 classes without confusion

---

## Discussion and Conclusion

The experimental results demonstrate that color information consistently improves CNN classification performance for currency recognition tasks, though the magnitude of improvement varies significantly between different currency types. For USD banknotes, the relatively small performance difference between color and greyscale models suggests that the network can effectively discriminate between denominations using primarily structural and textural features. The 2% accuracy reduction in the greyscale model indicates that while color provides some additional discriminative power, it is not the primary feature the network relies upon for USD classification. In contrast, Thai Baht currency demonstrated a much more substantial performance degradation when color information was removed, with accuracy dropping 7 percentage points. This significant difference suggests that Thai Baht banknote design relies more heavily on color as a distinguishing feature between denominations.

Joint multi-currency training does not negatively affect classification performance under the controlled conditions of this study. The slight improvement observed for THB may reflect the benefit of exposure to a broader range of visual patterns during training, which can improve generalisation. The model learns shared representations across currencies without feature interference, successfully managing 10 classes total.

OCR-based multimodal features do not improve classification accuracy and cause performance degradation. This is attributed to low OCR reliability caused by limited image resolution (224×224), stylized fonts, and partially occluded text. For Thai Baht notes, decorative typography and Thai script complexity further reduce OCR accuracy, leading to noisy and misleading feature vectors.

Saliency analysis reveals distinct feature reliance patterns between currencies. USD models consistently attend to portrait edges and structural elements, while THB models focus on color-rich regions and Thai script. Joint models develop broader feature representations, suggesting positive transfer from multi-currency learning. The universal model maintains high accuracy across both currencies while learning more generalized banknote features.

---

## Future Work

1. **Non-Standard Currency Testing**: Evaluate on specimen notes, novelty currencies, and counterfeits
2. **Multi-Currency Visual Similarity**: Expand to currencies with similar color schemes (Indonesian Rupiah, Singapore Dollar)
3. **Refined Evaluation Metrics**: Implement partial credit for semantically meaningful predictions
4. **Real-World Robustness Testing**: Test against handwritten markings, stamps, folds, tears, and surface wear
5. **Unified Final Model**: Combine insights for comprehensive currency recognition system
6. **Extended Dataset**: Include more denominations and currencies for broader generalization
7. **Production Deployment**: Optimize model for real-time inference in ATMs or mobile apps

---

## Implementation Details

### Dataset Loading

Custom `BanknoteDataset` class extending `torch.utils.data.Dataset`:
- Loads images from CSV index files
- Applies transforms (resize, normalize, augment)
- Supports both single-currency and multi-currency modes
- Handles class label mapping dynamically

### Training Loop

Features implemented:
- **Early stopping**: Monitors validation loss with patience and minimum delta
- **Gradient clipping**: Prevents gradient explosion (max norm = 1.0)
- **Learning rate**: 3e-4 with AdamW optimizer for stable training
- **Checkpointing**: Saves best model based on validation loss
- **Progress tracking**: Real-time loss and accuracy monitoring

### Evaluation Metrics

Computed on unseen test set:
- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Confidence scores

---

## Authors

This project was developed by Group 30:

- **Kim E-Shawn Brandon**
- **Tan See Yen Amanda**
- **Tang Peng Siang** 
- **Victor Soo Yuxiang** 
- **Baddipadige Amith Reddy**

All members contributed equally to literature review and report documentation.

