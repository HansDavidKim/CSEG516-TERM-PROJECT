import os
import torch
import torch.nn as nn
from collections import OrderedDict

from classifier.models import VGG16, ResNet152, FaceNet

### Load Model Weight if exists
def load_weight(model_name, path, num_classes):
    assert model_name in ['VGG16', 'ResNet152', 'Face.evoLVe'], "Model should be VGG16 or ResNet152 or Face.evoLVe."
    
    if model_name == 'VGG16':
        model = VGG16(num_classes)
    
    elif model_name == 'ResNet152':
        model = ResNet152(num_classes)
    
    else:
        model = FaceNet(num_classes)
    
    if not os.path.exists(path):
        raise AssertionError(f"Weight file not found at: {path}")

    state = torch.load(path, map_location="cpu")
    state_dict = state.get('stat_dict') or state.get('state_dict') or state

    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        # Handle classifier dimension mismatch by dropping final layer weights
        if 'fc_layer' in str(exc):
            filtered_dict = OrderedDict(
                (k, v) for k, v in state_dict.items() if not k.startswith('fc_layer.')
            )
            model.load_state_dict(filtered_dict, strict=False)
            if hasattr(model.fc_layer, "reset_parameters"):
                model.fc_layer.reset_parameters()
            print("Classifier layer size mismatch detected. Reinitialized fc_layer weights.")
        else:
            raise

    model.eval()

    return model
