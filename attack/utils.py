import os
import torch
from generator.model import Generator
from classifier.models import VGG16, ResNet152, FaceNet

def load_generator(path: str, device: torch.device, dim: int = 64) -> Generator:
    if not os.path.exists(path):
            raise FileNotFoundError(f"Generator path {path} does not exist")
    
    generator = Generator(in_dim=100, dim=dim).to(device)
    checkpoint = torch.load(path, map_location=device)
    # generator/train.py saves dict with 'generator' key
    if isinstance(checkpoint, dict) and 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    return generator

def load_classifier(path: str, device: torch.device, model_name: str = 'VGG16', arcface_scale: float = 16.0):
    """
    Load classifier model and optional ArcFace head.
    
    Args:
        path: Path to classifier checkpoint
        device: Device to load model on
        model_name: Model architecture name (auto-inferred from path)
        arcface_scale: Scale factor for ArcFace logits (default 16.0 for calibrated probabilities)
                       Lower values = softer probabilities, higher values = sharper probabilities
    """
    if not os.path.exists(path):
            raise FileNotFoundError(f"Classifier path {path} does not exist")
    
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        arc_head_state = checkpoint.get('arc_head', None)
    else:
        state_dict = checkpoint
        arc_head_state = None
        
    # Infer num_classes from the last layer weight
    if 'fc_layer.weight' in state_dict:
        num_classes = state_dict['fc_layer.weight'].shape[0]
    else:
        # Fallback or error
        print("Warning: Could not infer num_classes from state_dict. Using default 1000.")
        num_classes = 1000

    # Infer model_name from path if possible
    path_lower = path.lower()
    if 'vgg' in path_lower:
        model_name = 'VGG16'
    elif 'resnet' in path_lower:
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
    
    # Load ArcFace head if available (for better classifier)
    arc_head = None
    if arc_head_state is not None:
        from classifier.train import ArcMarginProduct
        # Infer embedding dimension from model
        emb_dim = getattr(model, 'embedding_dim', 512)
        # Use configurable scale factor for temperature scaling during inference
        # Original training used s=64, lower s gives more calibrated probabilities
        arc_head = ArcMarginProduct(emb_dim, num_classes, s=arcface_scale, m=0.5).to(device)
        arc_head.load_state_dict(arc_head_state)
        arc_head.eval()
        print(f"  ArcFace head loaded with scale={arcface_scale}")
    
    return model, arc_head

def load_discriminator(path: str, device: torch.device, dim: int = 64):
    """Load discriminator from generator checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Generator checkpoint path {path} does not exist")
    
    from generator.train import Discriminator
    
    discriminator = Discriminator(dim=dim).to(device)
    checkpoint = torch.load(path, map_location=device)
    
    # Load discriminator state from generator checkpoint
    if isinstance(checkpoint, dict) and 'discriminator' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator'])
    else:
        raise ValueError("Discriminator state not found in checkpoint. Make sure you're using a generator checkpoint that includes discriminator.")
    
    discriminator.eval()
    return discriminator
