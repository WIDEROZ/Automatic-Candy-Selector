import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from HierarchicalCandyDataset import HierarchicalCandyDataset
from HierarchicalResNet import HierarchicalResNet




# ==========================================
# 1. Configuration
# ==========================================

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Appareil utilisé :", device)


# ==========================================
# 2. Préparation des images
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. Chargement du dataset
# ==========================================


train_dataset = HierarchicalCandyDataset(
    "dataset/train",
    transform=train_transform
)

val_dataset = HierarchicalCandyDataset(
    "dataset/val",
    transform=val_transform
)

# Vérification de la cohérence des labels
assert train_dataset.categories == val_dataset.categories
assert train_dataset.colors == val_dataset.colors

num_categories = len(train_dataset.categories)
num_colors = len(train_dataset.colors)

print("Catégories :", train_dataset.categories)
print("Couleurs :", train_dataset.colors)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ================================================================
# 4. Chargement du modèle et définition des couches de sortie
# ================================================================


# Chargement de ResNet50
model = torch.hub.load(
    "pytorch/vision:v0.10.0",
    "resnet50",
    pretrained=True
)



# Gel des couches pré-entraînées
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features

# Deux têtes de classification
model.fc = nn.Identity()

model.category_classifier = nn.Linear(
    num_features,
    num_categories
)

model.color_classifier = nn.Linear(
    num_features,
    num_colors
)

model = HierarchicalResNet(
    num_categories,
    num_colors
).to(device)



# ==========================================
# 5. Fonction de perte et optimiseur
# ==========================================



criterion_category = nn.CrossEntropyLoss()
criterion_color = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    list(model.category_classifier.parameters()) +
    list(model.color_classifier.parameters()),
    lr=LEARNING_RATE
)

# ==========================================
# 6. Boucle d'entraînement
# ==========================================


for epoch in range(EPOCHS):

    model.train()

    # Garder le backbone gelé en mode évaluation
    model.backbone.eval()

    running_loss = 0.0

    category_correct = 0
    color_correct = 0
    total = 0

    for images, category_labels, color_labels in train_loader:

        images = images.to(device)
        category_labels = category_labels.to(device)
        color_labels = color_labels.to(device)

        optimizer.zero_grad()

        # Deux prédictions
        category_outputs, color_outputs = model(images)

        # Deux fonctions de perte
        loss_category = criterion_category(
            category_outputs, category_labels
        )

        loss_color = criterion_color(
            color_outputs, color_labels
        )

        # Perte totale
        loss = loss_category + loss_color

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        category_predictions = category_outputs.argmax(dim=1)
        color_predictions = color_outputs.argmax(dim=1)

        category_correct += (
            category_predictions == category_labels
        ).sum().item()

        color_correct += (
            color_predictions == color_labels
        ).sum().item()

        total += images.size(0)

    train_loss = running_loss / total

    category_accuracy = 100 * category_correct / total
    color_accuracy = 100 * color_correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Category Acc: {category_accuracy:.2f}% "
        f"Color Acc: {color_accuracy:.2f}%"
    )



# Sauvegarde du modèle

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "categories": train_dataset.categories,
        "colors": train_dataset.colors
    },
    "modele_bonbons.pth"
)

print("Modèle sauvegardé dans modele_bonbons.pth")
