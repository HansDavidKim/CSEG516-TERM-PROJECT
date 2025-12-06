# CSEG516 Term Project: RL-Based Model Inversion Attack

> Sogang University - Reinforcement Learning (CSEG516)

## Project Team
- **Daewon Kim**
- **Seobin Choi**

## Project Overview

This project implements and extends **RLB-MI (Reinforcement Learning-Based Model Inversion)** attack for privacy evaluation of face recognition models. We explore how RL agents can exploit trained classifiers to reconstruct private training data.

### Key Contributions
1. **Reproduction** of RLB-MI attack on CelebA and FaceScrub datasets
2. **Inference-time logit temperature scaling** for improved reward signal calibration
3. **Comprehensive evaluation** using Attack Accuracy, KNN Distance, and FID metrics

---

## Project Structure

```
CSEG516-TERM-PROJECT/
├── attack/                    # RL-based attack implementation
│   ├── sac_agent.py          # SAC (Soft Actor-Critic) agent
│   ├── attack_utils.py       # Reward computation
│   ├── utils.py              # Model loading utilities
│   └── train_rl.py           # Attack training script
├── classifier/               # Target classifier models
│   ├── models.py             # VGG16, ResNet152, FaceNet architectures
│   └── train.py              # ArcFace classifier training
├── generator/                # GAN generator for image synthesis
│   ├── model.py              # Generator architecture
│   └── train.py              # GAN training script
├── dataset/                  # Dataset directories
│   ├── private/              # Private training data (CelebA, FaceScrub)
│   └── public/               # Public data for GAN training
├── checkpoints/              # Model checkpoints
├── measure.py                # Comprehensive evaluation script
├── run_celeba_eval.sh        # CelebA evaluation script
├── run_facescrub_eval.sh     # FaceScrub evaluation script
└── metric_report/            # Evaluation results (CSV)
```

---

## Setup

### Requirements
```bash
pip install torch torchvision numpy scipy tqdm pillow
```

### Dataset Structure
```
dataset/private/celeba/
├── train/
│   ├── 0/
│   │   ├── 000000.jpg
│   │   └── ...
│   ├── 1/
│   └── ...
└── test/
    └── ...
```

---

## Usage

### 1. Train Classifier (ArcFace)
```bash
python -m classifier.train --data-root dataset/private/celeba --model VGG16
```

### 2. Train Generator (GAN)
```bash
python -m generator.train --data-root dataset/public/flickrfaceshq
```

### 3. Run Attack Evaluation
```bash
# CelebA dataset
./run_celeba_eval.sh

# FaceScrub dataset
./run_facescrub_eval.sh

# Or directly with Python
python measure.py \
    --generator-path checkpoints/generator.pth \
    --target-classifier checkpoints/vgg16_celeba_best.pt \
    --eval-classifier checkpoints/facenet_celeba_best.pt \
    --private-data dataset/private/celeba \
    --num-labels 50 \
    --arcface-scale 16.0
```

---

## Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--alpha` | Latent momentum: `z_new = α·z_old + (1-α)·action` | 0.0 |
| `--arcface-scale` | ArcFace logit scale factor (temperature) | 16.0 |
| `--max-episodes` | Maximum RL episodes per target class | 5000 |
| `--max-step` | Steps per episode | 1 |
| `--num-labels` | Number of target identities to attack | 50 |
| `--knn-k` | K for KNN distance calculation | 5 |
| `--confidence-threshold` | Early stopping threshold | 0.95 |
| `--device` | Device to use (cuda/mps/cpu) | cuda |

---

## Evaluation Metrics

### 1. Attack Accuracy
- **Top-1 / Top-5 Accuracy**: Percentage of generated images classified as target class
- Evaluated using independent evaluation network (FaceNet) for transferability measurement
- **Higher is better** → successful identity reconstruction

### 2. KNN Distance
- Average distance to K-nearest neighbors in private training set (feature space)
- **Lower is better** → generated images similar to actual private data
- Default K=5

### 3. FID (Fréchet Inception Distance)
- Measures perceptual quality and realism of generated images
- Compares InceptionV3 feature distributions between generated and private images
- **Lower is better** → realistic, high-quality reconstructions

### Example Output
```
================================================================================
FINAL EVALUATION RESULTS (RLB-MI Metrics)
================================================================================
Total Classes Attacked: 50

[1] Attack Accuracy (Evaluation Classifier)
    Top-1 Success: 40/50 = 80.00%
    Top-5 Success: 45/50 = 90.00%

[2] KNN Distance (k=5)
    Average Distance: 0.3245
    (Lower is better)

[3] FID (Fréchet Inception Distance)
    FID Score: 45.23
    (Lower is better)
================================================================================
```

Results saved to `metric_report/comprehensive_metrics_TIMESTAMP.csv`

---

## Logit Temperature Scaling (Our Contribution)

ArcFace classifiers trained with `s=64` produce overconfident predictions. During **inference**, we use a lower scale factor:

```python
# Training: s=64 → logits in [-64, 64] → very sharp softmax
# Attack:   s=16 → logits in [-16, 16] → calibrated probabilities
```

| Scale | Probability Behavior | Reward Signal | Attack Performance |
|-------|---------------------|---------------|-------------------|
| s=64 | Very sharp (99%+) | Sparse | Fast but local optima |
| s=16 | Calibrated (~70-90%) | Informative | Better exploration |

**Benefits:**
- More informative reward signals for RL agent
- Better probability calibration for evaluation  
- Improved attack success rate

---

## References

- [RLB-MI: Reinforcement Learning-Based Black-Box Model Inversion Attacks](https://openaccess.thecvf.com/content/CVPR2023/html/Han_Reinforcement_Learning-Based_Black-Box_Model_Inversion_Attacks_CVPR_2023_paper.html) (CVPR 2023)
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)

---

## License

This project is for educational purposes only.
