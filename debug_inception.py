#!/usr/bin/env python3
"""Quick debug test for InceptionV3 feature extraction"""

import torch
from torchvision import transforms
from PIL import Image
import sys
sys.path.insert(0, '/Users/kimdaewon/Desktop/프로젝트/CSEG516-TERM-PROJECT')

from measure import InceptionV3FeatureExtractor

device = torch.device('cpu')

# Create InceptionV3
print("Creating InceptionV3...")
inception = InceptionV3FeatureExtractor(device)

# Test with proper image tensor
print("\nTest 1: Proper image tensor [1, 3, 64, 64] in range [0, 1]")
img_tensor = torch.rand(1, 3, 64, 64)  # [0, 1] range
print(f"Input shape: {img_tensor.shape}, range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")

try:
    features = inception(img_tensor)
    print(f"✓ Success! Output shape: {features.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test with wrong input (what's happening in the code)
print("\nTest 2: Wrong input - classifier output [1, 1000]")
wrong_input = torch.rand(1, 1000)
print(f"Input shape: {wrong_input.shape}")

try:
    features = inception(wrong_input)
    print(f"✓ Success! Output shape: {features.shape}")
except Exception as e:
    print(f"✗ Error (expected): {type(e).__name__}")

print("\nDone!")
