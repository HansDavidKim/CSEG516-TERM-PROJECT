#!/bin/bash
# Convenience script to run comprehensive model inversion attack evaluation
# Measures: Attack Accuracy, KNN Distance, FID

# Fix OpenMP library conflict on Mac
export KMP_DUPLICATE_LIB_OK=TRUE

# Default paths (modify these according to your setup)
GENERATOR_PATH="checkpoints/generator_epoch_50.pth"
TARGET_CLASSIFIER="checkpoints/VGG16_celeba_best.pth"
EVAL_CLASSIFIER="checkpoints/evoLVe_celeba.pth"  # evoLVe network for evaluation
PRIVATE_DATA="dataset/private/celeba"

# Default parameters
NUM_LABELS=10
MAX_EPISODES=10000
KNN_K=5
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --generator)
            GENERATOR_PATH="$2"
            shift 2
            ;;
        --target)
            TARGET_CLASSIFIER="$2"
            shift 2
            ;;
        --eval)
            EVAL_CLASSIFIER="$2"
            shift 2
            ;;
        --data)
            PRIVATE_DATA="$2"
            shift 2
            ;;
        --num-labels)
            NUM_LABELS="$2"
            shift 2
            ;;
        --episodes)
            MAX_EPISODES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Model Inversion Attack - Evaluation"
echo "=========================================="
echo "Generator: $GENERATOR_PATH"
echo "Target Classifier: $TARGET_CLASSIFIER"
echo "Eval Classifier: $EVAL_CLASSIFIER"
echo "Private Data: $PRIVATE_DATA"
echo "Num Labels: $NUM_LABELS"
echo "Max Episodes: $MAX_EPISODES"
echo "Device: $DEVICE"
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
    --knn-k "$KNN_K" \
    --device "$DEVICE"
