import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================================
# 1. Configuration
# ==========================================

BATCH_SIZE = 16
EPOCHS = 2
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

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=val_transform
)

# Vérifier que les catégories correspondent
assert train_dataset.classes == val_dataset.classes, (
    "Les catégories de train et val sont différentes."
)

num_classes = len(train_dataset.classes)

print("Catégories :", train_dataset.classes)
print("Nombre de classes :", num_classes)
print("Images d'entraînement :", len(train_dataset))
print("Images de validation :", len(val_dataset))


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


# ==========================================
# 4. Chargement du modèle pré-entraîné
# ==========================================

model = torch.hub.load(
    "pytorch/vision:v0.10.0",
    "resnet50",
    pretrained=True
)

# Geler toutes les anciennes couches
for param in model.parameters():
    param.requires_grad = False

# Remplacer la dernière couche par num_classes sorties
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    num_classes
)

model = model.to(device)


# ==========================================
# 5. Fonction de perte et optimiseur
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 6. Boucle d'entraînement
# ==========================================

for epoch in range(EPOCHS):

    model.train()

    model.eval()
    model.fc.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Remise à zéro des gradients
        optimizer.zero_grad()

        # Prédiction
        outputs = model(images)

        # Calcul de l'erreur
        loss = criterion(outputs, labels)

        # Calcul des gradients
        loss.backward()

        # Mise à jour de la dernière couche
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total




    # 7. Validation

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss_sum = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )



# Sauvegarde du modèle

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": train_dataset.classes
    },
    "modele_bonbons.pth"
)

print("Modèle sauvegardé dans modele_bonbons.pth")
