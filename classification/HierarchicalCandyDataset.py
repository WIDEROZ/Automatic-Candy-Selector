
import os
from PIL import Image
from torch.utils.data import Dataset

class HierarchicalCandyDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []

        # Récupération des catégories
        self.categories = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.category_to_idx = {
            category: idx
            for idx, category in enumerate(self.categories)
        }

        # Récupération des couleurs
        colors = set()

        for category in self.categories:
            category_path = os.path.join(root_dir, category)

            for color in os.listdir(category_path):
                color_path = os.path.join(category_path, color)

                if os.path.isdir(color_path):
                    colors.add(color)

        self.colors = sorted(colors)

        self.color_to_idx = {
            color: idx
            for idx, color in enumerate(self.colors)
        }

        # Parcours des images
        for category in self.categories:

            category_path = os.path.join(root_dir, category)

            for color in os.listdir(category_path):

                color_path = os.path.join(category_path, color)

                if not os.path.isdir(color_path):
                    continue

                for filename in os.listdir(color_path):

                    if filename.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp")
                    ):

                        image_path = os.path.join(
                            color_path, filename
                        )

                        category_label = self.category_to_idx[category]
                        color_label = self.color_to_idx[color]

                        self.samples.append((
                            image_path,
                            category_label,
                            color_label
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, category_label, color_label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, category_label, color_label
