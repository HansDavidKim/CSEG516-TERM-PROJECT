import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import evolve

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VGG16(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        model = torchvision.models.vgg16_bn(pretrained=False)

        ### Feature Extractor for VGG16; input assumed to be 64x64 so final map is 2x2
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes

        ### For more stable learning, we turn off bias
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)

        ### Linear classifier using extracted latent representation
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)

        ### Skip BN when batch size is 1 in training to avoid instability
        if self.training and feature.size(0) == 1:
            normed_feature = feature
        else:
            normed_feature = self.bn(feature)

        res = self.fc_layer(normed_feature)

        return normed_feature, res
    
    '''
    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return out
    '''

class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(),
            Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512)
        )

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
    
    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return feat, out

class FaceNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(),
            Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512)
        )

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        _, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return feat, out

if __name__ == "__main__":
    # VGG16 smoke test with 64x64 inputs
    vgg = VGG16(n_classes=10)

    single_image = torch.randn(3, 64, 64)
    try:
        vgg(single_image)
    except Exception as exc:
        print(f"[VGG] Single image without batch dim failed: {exc}")

    batched_image = single_image.unsqueeze(0)
    vgg.train()
    try:
        features, logits = vgg(batched_image)
        print(f"[VGG] Batched (B=1) succeeded (train mode): features {features.shape}, logits {logits.shape}")
    except Exception as exc:
        print(f"[VGG] Batched (B=1) failed (train mode): {exc}")

    batched_two = torch.randn(2, 3, 64, 64)
    features_two, logits_two = vgg(batched_two)
    print(f"[VGG] Batched (B=2) succeeded (train mode): features {features_two.shape}, logits {logits_two.shape}")

    vgg.eval()
    with torch.no_grad():
        features, logits = vgg(batched_image)
    print(f"[VGG] Batched input succeeded (eval mode): features {features.shape}, logits {logits.shape}")

    # ResNet152 smoke test with 64x64 inputs
    resnet = ResNet152(num_classes=10)

    resnet.eval()
    with torch.no_grad():
        feat_resnet, out_resnet = resnet(batched_image)
    print(f"[ResNet] Batched (B=1) succeeded (eval mode): features {feat_resnet.shape}, logits {out_resnet.shape}")

    resnet.train()
    feat_two, out_two = resnet(batched_two)
    print(f"[ResNet] Batched (B=2) succeeded (train mode): features {feat_two.shape}, logits {out_two.shape}")

    # FaceNet smoke test with 64x64 inputs
    facenet = FaceNet(num_classes=10)

    facenet.eval()
    with torch.no_grad():
        feat_face, out_face = facenet(batched_image)
    print(f"[FaceNet] Batched (B=1) succeeded (eval mode): features {feat_face.shape}, logits {out_face.shape}")

    facenet.train()
    feat_face_two, out_face_two = facenet(batched_two)
    print(f"[FaceNet] Batched (B=2) succeeded (train mode): features {feat_face_two.shape}, logits {out_face_two.shape}")
