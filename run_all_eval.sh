#!/bin/bash
# Master evaluation script - Runs evaluation on ALL datasets
# Measures: Attack Accuracy, KNN Distance, FID for each dataset

# Fix OpenMP library conflict on Mac
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=========================================="
echo "Model Inversion Attack - FULL EVALUATION"
echo "Running on ALL datasets"
echo "=========================================="
echo ""

# Common parameters (Original settings)
MAX_EPISODES=10000
MAX_STEP=1
ALPHA=0.0
GENERATOR_DIM=128
Z_DIM=100
KNN_K=5
DEVICE="mps"

# ==========================================
# 1. CelebA Dataset
# ==========================================
echo ""
echo "=========================================="
echo "[1/3] Evaluating CelebA Dataset"
echo "=========================================="

python measure.py \
    --generator-path "checkpoints/generator/generator_celeba_epochs_100_critic_steps_4.pt" \
    --target-classifier "checkpoints/vgg16_celeba_best.pt" \
    --eval-classifier "checkpoints/facenet_celeba_best.pt" \
    --private-data "dataset/private/celeba" \
    --num-labels 50 \
    --max-episodes "$MAX_EPISODES" \
    --max-step "$MAX_STEP" \
    --alpha "$ALPHA" \
    --generator-dim "$GENERATOR_DIM" \
    --z-dim "$Z_DIM" \
    --knn-k "$KNN_K" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo "✓ CelebA evaluation completed successfully"
else
    echo "✗ CelebA evaluation failed"
fi

# ==========================================
# 2. FaceScrub Dataset
# ==========================================
echo ""
echo "=========================================="
echo "[2/3] Evaluating FaceScrub Dataset"
echo "=========================================="

python measure.py \
    --generator-path "checkpoints/generator/generator_facescrub_epochs_200_critic_steps_4.pt" \
    --target-classifier "checkpoints/vgg16_facescrub_best.pt" \
    --eval-classifier "checkpoints/facenet_facescrub_best.pt" \
    --private-data "dataset/private/facescrub-full" \
    --num-labels 50 \
    --max-episodes "$MAX_EPISODES" \
    --max-step "$MAX_STEP" \
    --alpha "$ALPHA" \
    --generator-dim "$GENERATOR_DIM" \
    --z-dim "$Z_DIM" \
    --knn-k "$KNN_K" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo "✓ FaceScrub evaluation completed successfully"
else
    echo "✗ FaceScrub evaluation failed"
fi

# ==========================================
# 3. PubFig83 Dataset (using FFHQ generator)
# ==========================================
echo ""
echo "=========================================="
echo "[3/3] Evaluating PubFig83 Dataset"
echo "=========================================="

python measure.py \
    --generator-path "checkpoints/generator/generator_pubfig83.pt" \
    --target-classifier "checkpoints/vgg16_pubfig83_best.pt" \
    --eval-classifier "checkpoints/facenet_pubfig83_best.pt" \
    --private-data "dataset/private/pubfig83" \
    --num-labels 50 \
    --max-episodes "$MAX_EPISODES" \
    --max-step "$MAX_STEP" \
    --alpha "$ALPHA" \
    --generator-dim "$GENERATOR_DIM" \
    --z-dim "$Z_DIM" \
    --knn-k "$KNN_K" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo "✓ PubFig83 evaluation completed successfully"
else
    echo "✗ PubFig83 evaluation failed"
fi

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETE!"
echo "=========================================="
echo "Results saved in metric_report/ directory"
echo ""
echo "Files generated:"
ls -lht metric_report/*.csv | head -3
echo ""
echo "=========================================="
