#!/bin/bash
# Quick start script for running RLB-MI attack with Docker

set -e

echo "🐳 RLB-MI Docker Quick Start"
echo "============================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker-compose is not installed"
    echo "Please install docker-compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Detect GPU
USE_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected"
        USE_GPU=true
    else
        echo "⚠️  NVIDIA GPU not available, using CPU"
    fi
else
    echo "ℹ️  No GPU detected, using CPU"
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p checkpoints attack_results dataset pretrained
echo "✅ Directories created"

# Build Docker image
echo ""
if [ "$USE_GPU" = true ]; then
    echo "🔨 Building GPU Docker image..."
    docker-compose build rlb-mi-gpu
    PROFILE="gpu"
    SERVICE="rlb-mi-gpu"
else
    echo "🔨 Building CPU Docker image..."
    docker-compose build rlb-mi-cpu
    PROFILE="cpu"
    SERVICE="rlb-mi-cpu"
fi

echo "✅ Docker image built successfully"

# Run container
echo ""
echo "🚀 Starting container..."
echo ""
echo "You are now inside the RLB-MI container!"
echo "To run the attack, use:"
echo ""
echo "  python main.py run-rlb-mi-attack \\"
echo "    --generator checkpoints/generator_last.pt \\"
echo "    --target-model checkpoints/vgg16_celeba_best.pt \\"
echo "    --target-class 0 \\"
echo "    --episodes 40000"
echo ""
echo "Or run the example script:"
echo "  python example_rlb_mi.py"
echo ""

docker-compose --profile "$PROFILE" run --rm "$SERVICE" bash
