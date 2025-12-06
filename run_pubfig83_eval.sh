#!/bin/bash

# Model Inversion Attack - Full Evaluation Script
# Dataset: PubFig83

export KMP_DUPLICATE_LIB_OK=TRUE

echo "=========================================="
echo "Model Inversion Attack - Full Evaluation"
echo "Dataset: PubFig83"
echo "=========================================="
echo ""

# Hardcoded paths (verified to exist)
GENERATOR_PATH="checkpoints/generator/generator_pubfig83.pt"
TARGET_CLASSIFIER="checkpoints/vgg16_pubfig83_best.pt"
EVAL_CLASSIFIER="checkpoints/facenet_pubfig83_best.pt"  # Using FaceNet as evaluation network
PRIVATE_DATA="dataset/private/pubfig83"

# Parameters (Original settings)
NUM_LABELS=50
MAX_EPISODES=10000
MAX_STEP=1
ALPHA=0.0
GENERATOR_DIM=128
Z_DIM=100
KNN_K=5
DEVICE="mps"

echo "Configuration:"
echo "  Generator: $GENERATOR_PATH"
echo "  Target Classifier: $TARGET_CLASSIFIER"
echo "  Eval Classifier: $EVAL_CLASSIFIER"
echo "  Private Data: $PRIVATE_DATA"
echo "  Num Labels: $NUM_LABELS"
echo "  Max Episodes: $MAX_EPISODES"
echo "  Device: $DEVICE"
echo "=========================================="
echo ""

# Run evaluation
python measure.py \
    --generator-path "$GENERATOR_PATH" \
    --target-classifier "$TARGET_CLASSIFIER" \
    --eval-classifier "$EVAL_CLASSIFIER" \
    --private-data "$PRIVATE_DATA" \
    --num-labels "$NUM_LABELS" \
    --max-episodes "$MAX_EPISODES" \
    --max-step "$MAX_STEP" \
    --alpha "$ALPHA" \
    --generator-dim "$GENERATOR_DIM" \
    --z-dim "$Z_DIM" \
    --knn-k "$KNN_K" \
    --device "$DEVICE"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Check metric_report/ directory for results"
echo "=========================================="
