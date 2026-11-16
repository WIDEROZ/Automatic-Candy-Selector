import torch
import torch.nn as nn

class HierarchicalResNet(nn.Module):

    def __init__(self, num_categories, num_colors):

        super().__init__()

        self.backbone = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet50",
            pretrained=True
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        self.category_classifier = nn.Linear(
            num_features,
            num_categories
        )

        self.color_classifier = nn.Linear(
            num_features,
            num_colors
        )

    def forward(self, x):

        features = self.backbone(x)

        category_output = self.category_classifier(features)
        color_output = self.color_classifier(features)

        return category_output, color_output


