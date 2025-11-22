import os
import torch
from generator.model import Generator
from classifier.models import VGG16, ResNet152, FaceNet

def load_generator(path: str, device: torch.device) -> Generator:
    if not os.path.exists(path):
            raise FileNotFoundError(f"Generator path {path} does not exist")
    
    generator = Generator(in_dim=100, dim=64).to(device)
    checkpoint = torch.load(path, map_location=device)
    # generator/train.py saves dict with 'generator' key
    if isinstance(checkpoint, dict) and 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    return generator

def load_classifier(path: str, device: torch.device, model_name: str = 'VGG16'):
    if not os.path.exists(path):
            raise FileNotFoundError(f"Classifier path {path} does not exist")
    
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # Infer num_classes from the last layer weight
    if 'fc_layer.weight' in state_dict:
        num_classes = state_dict['fc_layer.weight'].shape[0]
    else:
        # Fallback or error
        print("Warning: Could not infer num_classes from state_dict. Using default 1000.")
        num_classes = 1000

    # Infer model_name from path if possible
    path_lower = path.lower()
    if 'resnet' in path_lower:
        model_name = 'ResNet152'
    elif 'facenet' in path_lower or 'face' in path_lower:
        model_name = 'FaceNet'
    
    if model_name == 'VGG16':
        model = VGG16(num_classes)
    elif model_name == 'ResNet152':
        model = ResNet152(num_classes)
    else:
        model = FaceNet(num_classes)
        
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
