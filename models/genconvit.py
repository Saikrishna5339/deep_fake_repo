import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.utils.checkpoint import checkpoint

class GenConViT(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(GenConViT, self).__init__()
        
        # Load pretrained ResNet18 as backbone (smaller than ResNet50)
        self.backbone = resnet18(pretrained=pretrained)
        
        # Remove the last fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Vision Transformer components with reduced dimensions
        self.patch_size = 16
        self.embed_dim = 512  # Reduced from 2048 to 512
        self.num_heads = 4    # Reduced from 8 to 4
        self.num_layers = 3   # Reduced from 6 to 3
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(512, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 49, self.embed_dim))
        
        # Transformer encoder layers with reduced dimensions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=2*self.embed_dim,  # Reduced from 4x to 2x
            dropout=0.1,
            batch_first=True  # Added for better memory efficiency
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Classification head with reduced dimensions
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 256),  # Reduced from 512 to 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Gradient checkpointing flag
        self.use_checkpoint = False
        
    def enable_checkpointing(self):
        self.use_checkpoint = True
        
    def disable_checkpointing(self):
        self.use_checkpoint = False
        
    def forward(self, x):
        # Extract features using ResNet backbone
        if self.use_checkpoint:
            x = checkpoint(self.backbone, x)
        else:
            x = self.backbone(x)
        
        # Reshape for transformer
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Pass through transformer
        if self.use_checkpoint:
            x = checkpoint(self.transformer_encoder, x)
        else:
            x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1)
        )
        
    def forward(self, x):
        return self.model(x) 