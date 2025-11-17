# RLB-MI Attack Implementation

Implementation of **Reinforcement Learning-Based Black-box Model Inversion (RLB-MI) Attack** from the paper:

> Han et al., "Reinforcement Learning-Based Black-Box Model Inversion Attacks" (2023)
> arXiv:2304.04625

## 📋 Overview

This implementation provides a complete pipeline for conducting model inversion attacks using reinforcement learning. The attack reconstructs private training data from a black-box classifier by:

1. Training a **SAC (Soft Actor-Critic)** agent to search the latent space of a pre-trained GAN
2. Using **confidence scores** from the target classifier as rewards
3. Generating reconstructed images that match the target class distribution

## 🏗️ Architecture

### Components

1. **SAC Agent** (`attack/models.py`)
   - Actor network (Gaussian policy)
   - Twin Critic networks (Q-functions)
   - Replay buffer
   - Automatic entropy tuning

2. **RLB-MI Attack** (`attack/attack.py`)
   - MDP formulation for latent space search
   - Reward function (r1, r2, r3 from paper)
   - State transition with diversity factor α
   - Image generation and evaluation

3. **Evaluation Metrics** (`attack/attack.py`)
   - Attack Accuracy
   - K-Nearest Neighbor Distance (KNN Dist)
   - Feature Distance (Feat Dist)

## 🚀 Usage

### Prerequisites

1. **Trained Generator** (GAN on public dataset)
2. **Trained Target Classifier** (on private dataset)

### Method 1: Using Docker (권장) 🐳

```bash
# CPU 버전
docker-compose --profile cpu run --rm rlb-mi-cpu bash

# GPU 버전 (NVIDIA GPU 필요)
docker-compose --profile gpu run --rm rlb-mi-gpu bash

# 컨테이너 내부에서 공격 실행
python main.py run-rlb-mi-attack \
  --generator checkpoints/generator_last.pt \
  --target-model checkpoints/vgg16_celeba_best.pt \
  --model-name VGG16 \
  --target-class 0 \
  --num-classes 1000 \
  --episodes 40000 \
  --alpha 0.0 \
  --num-images 1000 \
  --top-k 10 \
  --output-dir attack_results
```

**자세한 Docker 사용법**: [DOCKER_GUIDE.md](DOCKER_GUIDE.md) 참조

### Method 2: Using CLI Command (로컬)

```bash
python main.py run-rlb-mi-attack \
  --generator checkpoints/generator_last.pt \
  --target-model checkpoints/vgg16_celeba_best.pt \
  --model-name VGG16 \
  --target-class 0 \
  --num-classes 1000 \
  --episodes 40000 \
  --alpha 0.0 \
  --num-images 1000 \
  --top-k 10 \
  --output-dir attack_results
```

### Method 3: Using Example Script

```bash
python example_rlb_mi.py
```

Edit the configuration section in `example_rlb_mi.py` to customize:
- Generator and target model checkpoints
- Target class to reconstruct
- Number of training episodes
- Diversity factor (α)

### Method 4: Python API

```python
from attack.attack import RLB_MI_Attack
from generator.model import Generator
from classifier.models import VGG16

# Load models
generator = Generator(in_dim=100, dim=64).to(device)
generator.load_state_dict(torch.load('checkpoints/generator_last.pt')['generator'])

target_model = VGG16(num_classes=1000).to(device)
target_model.load_state_dict(torch.load('checkpoints/vgg16_celeba_best.pt')['model'])

# Initialize attack
attack = RLB_MI_Attack(
    generator=generator,
    target_model=target_model,
    target_class=0,
    latent_dim=100,
    device=device,
    diversity_factor=0.0,  # 0.0 for accuracy, 0.97 for diversity
)

# Train agent
metrics = attack.train_agent(
    max_episodes=40000,
    save_path='attack_results/agent.pt'
)

# Generate images
images, confidences, latents = attack.generate_reconstructed_images(
    num_images=1000,
    top_k=10
)
```

## ⚙️ Key Parameters

### Diversity Factor (α)

Controls the trade-off between accuracy and diversity:

- **α = 0.0**: Focus on accuracy (single high-confidence image)
- **α = 0.97**: Focus on diversity (varied images with high confidence)

State transition: `s_{t+1} = α·s_t + (1-α)·a_t`

### Reward Weights

From paper (Section 4.1):
- **w1 = 2.0**: Weight for state score
- **w2 = 2.0**: Weight for action score
- **w3 = 8.0**: Weight for class separation

Total reward: `R_t = w1·r1 + w2·r2 + w3·r3`

### Training Episodes

- **Default: 40,000** (as used in the paper)
- Smaller datasets (e.g., PubFig83 with 50 classes) may saturate earlier (~25,000)
- Larger datasets (e.g., FaceScrub with 200 classes) may need full 40,000

## 📊 Evaluation

### Attack Accuracy

Measures how many reconstructed images are correctly classified by an evaluation classifier:

```python
from attack.attack import compute_attack_accuracy

accuracy = compute_attack_accuracy(
    reconstructed_images=images,
    target_class=target_class,
    eval_classifier=eval_model,
    device=device
)
print(f"Attack Accuracy: {accuracy:.2f}%")
```

### KNN Distance

Average L2 distance to K-nearest neighbors in feature space:

```python
from attack.attack import compute_knn_distance

knn_dist = compute_knn_distance(
    reconstructed_images=images,
    target_class_images=real_images,
    eval_classifier=eval_model,
    device=device,
    k=1
)
print(f"KNN Distance: {knn_dist:.2f}")
```

### Feature Distance

L2 distance to feature centroid of target class:

```python
from attack.attack import compute_feature_distance

feat_dist = compute_feature_distance(
    reconstructed_images=images,
    target_class_images=real_images,
    eval_classifier=eval_model,
    device=device
)
print(f"Feature Distance: {feat_dist:.2f}")
```

## 📈 Expected Results

Based on the paper (Table 1, CelebA dataset):

| Model | Attack Acc | KNN Dist | Feat Dist |
|-------|-----------|----------|-----------|
| VGG16 | 65.9% | 1310.7 | 1214.7 |
| ResNet-152 | **80.4%** | **1217.9** | **1108.2** |
| Face.evoLVe | 79.3% | 1225.6 | 1112.1 |

RLB-MI outperforms:
- White-box attacks (GMI, KED-MI)
- Black-box attacks (LB-MI, MIRROR)
- Label-only attacks (BREP-MI)

## 🔬 Algorithm Details

### MDP Formulation

**State Space**: GAN latent space (k-dimensional)
- Initial state: `s_0 ~ N(0, 1)`

**Action Space**: Guidance vectors (k-dimensional)
- Action: `a_t = Agent(s_t)`

**State Transition**:
```
s_{t+1} = α·s_t + (1-α)·a_t
```

**Reward Function**:
```
r1 = log[T_y(G(s_{t+1}))]           # State score
r2 = log[T_y(G(a_t))]               # Action score
r3 = log[max{ε, T_y(G(s_{t+1})) - max_{i≠y}T_i(G(s_{t+1}))}]  # Class separation

R_t = w1·r1 + w2·r2 + w3·r3
```

### SAC Hyperparameters

From paper (Section 4.1):
- Learning rate: `5e-4`
- Discount factor (γ): `0.99`
- Soft update factor (τ): `0.01`
- Batch size: `256`
- Replay buffer size: `1,000,000`
- Max steps per episode: `1`

## 📁 Output Structure

```
attack_results/
├── agent_class_0.pt                    # Trained SAC agent
├── reconstructed_class_0.png           # Image grid
├── class_0_images/                     # Individual images
│   ├── image_000_conf_0.9234.png
│   ├── image_001_conf_0.9156.png
│   └── ...
└── latents_class_0.npy                 # Latent vectors
```

## 🎯 Tips for Better Results

1. **Use a well-trained generator** on a public dataset similar to the private dataset
2. **Train the target classifier to high accuracy** (higher accuracy = better attack performance)
3. **Adjust diversity factor** based on your goal:
   - Accuracy: α = 0.0
   - Diversity: α = 0.9-0.97
4. **Monitor training progress**: Check if reward converges
5. **Generate more images**: Use higher `num_images` and `top_k` for better selection

## 📚 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{han2023reinforcement,
  title={Reinforcement Learning-Based Black-Box Model Inversion Attacks},
  author={Han, Gyojin and Choi, Jaehyun and Lee, Haeil and Kim, Junmo},
  journal={arXiv preprint arXiv:2304.04625},
  year={2023}
}
```

## 🔒 Ethical Considerations

This implementation is for **research and educational purposes only**. Model inversion attacks reveal privacy vulnerabilities in machine learning systems. Please use responsibly and only on systems you have permission to test.

## 🐛 Troubleshooting

### Agent not converging
- Check that generator and target model are loaded correctly
- Increase training episodes
- Verify target class exists in the model

### Low confidence scores
- Ensure target model has high test accuracy
- Check that generator produces realistic images
- Try adjusting reward weights

### Out of memory
- Reduce batch size
- Reduce replay buffer capacity
- Use smaller model or lower resolution images

## 📞 Support

For issues or questions:
1. Check the paper for algorithmic details
2. Review the code comments in `attack/models.py` and `attack/attack.py`
3. Verify your checkpoints are compatible
