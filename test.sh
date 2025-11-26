#!/bin/bash

# Train attack for classes 0 to 999
for i in {0..999}
do
    echo "========================================="
    echo "Training attack for class $i"
    echo "========================================="
    
    python main.py train-attack \
      --generator-path checkpoints/generator/generator_celeba_epochs_100_critic_steps_4.pt \
      --classifier-path checkpoints/vgg16_celeba_best.pt \
      --target-class $i \
      --generator-dim 128 \
      --device mps \
      --confidence-threshold 0.65
    
    # Check if the command failed
    if [ $? -ne 0 ]; then
        echo "❌ Failed at class $i"
        exit 1
    fi
    
    echo "✅ Completed class $i"
    echo ""
done

echo "========================================="
echo "All 1000 classes completed!"
echo "========================================="
