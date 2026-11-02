import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models


# Vérification carte graphique

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# 2. Charger le modèle entraîné en local

checkpoint = torch.load(
    "modele_bonbons.pth",
    map_location=device,
    weights_only=True
)

# Récupérer les noms des classes enregistrées
categories = checkpoint["classes"]

# Construire ResNet18 sans télécharger de poids
model = models.resnet50(weights=None)

# Adapter la dernière couche au nombre de classes
num_features = model.fc.in_features
num_classes = len(categories)

model.fc = nn.Linear(num_features, num_classes)

# Charger les poids appris pendant l'entraînement
model.load_state_dict(checkpoint["model_state_dict"])

# Préparer le modèle pour les prédictions
model = model.to(device)
model.eval()

print("Modèle chargé avec succès !")
print("Classes :", categories)


# ==========================================
# 3. Charger et prétraiter l'image
# ==========================================

image_path = "images/7.jpg"

input_image = Image.open(image_path).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

input_tensor = preprocess(input_image)

# Ajouter la dimension du batch
input_batch = input_tensor.unsqueeze(0).to(device)


# ==========================================
# 4. Effectuer la prédiction
# ==========================================

with torch.no_grad():
    output = model(input_batch)

    # Convertir les scores en probabilités
    probabilities = torch.softmax(output[0], dim=0)

# Récupérer la classe la plus probable
predicted_index = probabilities.argmax().item()

predicted_class = categories[predicted_index]

confidence = probabilities[predicted_index].item()

print("\nRésultat de la classification")
print("-----------------------------")
print("Type de bonbon :", predicted_class)
print(f"Confiance : {confidence * 100:.2f} %")


# ==========================================
# 5. Afficher les probabilités des 6 classes
# ==========================================

print("\nProbabilités par catégorie :")

top_prob, top_indices = torch.topk(
    probabilities,
    k=min(6, len(categories))
)

for prob, idx in zip(top_prob, top_indices):
    print(
        f"{categories[idx.item()]} : "
        f"{prob.item() * 100:.2f} %"
    )