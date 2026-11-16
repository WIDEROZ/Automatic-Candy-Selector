
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
import torchvision.models as models

from HierarchicalResNet import HierarchicalResNet

# ==========================================
# 1. Configuration
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Appareil utilisé :", device)


# ==========================================
# 2. Chargement du modèle entraîné
# ==========================================

checkpoint = torch.load(
    "modele_bonbons.pth",
    map_location=device,
    weights_only=True
)

# Récupération des classes
categories = checkpoint["categories"]
colors = checkpoint["colors"]

num_categories = len(categories)
num_colors = len(colors)

# Reconstruction de l'architecture
model = HierarchicalResNet(
    num_categories,
    num_colors
)

# Chargement des poids
model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()

print("\nModèle chargé avec succès !")
print("Catégories :", categories)
print("Couleurs :", colors)


# ==========================================
# 3. Chargement et prétraitement de l'image
# ==========================================

image_path = "image/18.jpg"

input_image = Image.open(image_path).convert("RGB")

preprocess = transforms.Compose([

    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = preprocess(input_image)

# Ajout de la dimension batch
input_batch = input_tensor.unsqueeze(0).to(device)


# ==========================================
# 5. Effectuer les prédictions
# ==========================================

with torch.no_grad():

    category_output, color_output = model(input_batch)

    category_probabilities = torch.softmax(
        category_output[0], dim=0
    )

    color_probabilities = torch.softmax(
        color_output[0], dim=0
    )


# ==========================================
# 6. Récupérer les résultats
# ==========================================

# Prédiction catégorie
category_index = category_probabilities.argmax().item()

predicted_category = categories[category_index]

category_confidence = category_probabilities[
    category_index
].item()


# Prédiction couleur
color_index = color_probabilities.argmax().item()

predicted_color = colors[color_index]

color_confidence = color_probabilities[
    color_index
].item()


# ==========================================
# 7. Afficher les résultats
# ==========================================

print("\nRésultat de la classification")
print("--------------------------------")

print("Type de bonbon :", predicted_category)

print(
    f"Confiance catégorie : "
    f"{category_confidence * 100:.2f} %"
)

print("Couleur :", predicted_color)

print(
    f"Confiance couleur : "
    f"{color_confidence * 100:.2f} %"
)


# ==========================================
# 8. Afficher les probabilités par catégorie
# ==========================================

print("\nProbabilités par catégorie :")

top_prob, top_indices = torch.topk(
    category_probabilities,
    k=min(6, len(categories))
)

for prob, idx in zip(top_prob, top_indices):

    print(
        f"{categories[idx.item()]} : "
        f"{prob.item() * 100:.2f} %"
    )


# ==========================================
# 9. Afficher les probabilités par couleur
# ==========================================

print("\nProbabilités par couleur :")

top_prob, top_indices = torch.topk(
    color_probabilities,
    k=min(6, len(colors))
)

for prob, idx in zip(top_prob, top_indices):

    print(
        f"{colors[idx.item()]} : "
        f"{prob.item() * 100:.2f} %"
    )


# ==========================================
# 10. Résultat final
# ==========================================


print(
    f"{predicted_category} {predicted_color}"
)
