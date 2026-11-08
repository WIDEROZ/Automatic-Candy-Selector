import os
from PIL import Image
import pillow_heif

# Enregistrer le support HEIC dans Pillow
pillow_heif.register_heif_opener()

def convertir_dossier_heic(dossier_source, dossier_destination=None):
    # Si aucun dossier de destination n'est fourni, on enregistre au même endroit
    if not dossier_destination:
        dossier_destination = dossier_source

    # Compteur global
    compteur = 0

    # os.walk parcourt le dossier source et tous ses sous-dossiers
    for racine, _, fichiers in os.walk(dossier_source):
        # Créer la structure de dossiers correspondante dans la destination
        sous_dossier_dest = os.path.join(
            dossier_destination,
            os.path.relpath(racine, dossier_source)
        )
        os.makedirs(sous_dossier_dest, exist_ok=True)

        for fichier in fichiers:
            # Vérifier si le fichier est un HEIC (insensible à la casse)
            if fichier.lower().endswith('.heic'):
                chemin_heic = os.path.join(racine, fichier)

                # Chemin de sortie en .jpg
                nom_sans_ext = os.path.splitext(fichier)[0]
                chemin_jpg = os.path.join(sous_dossier_dest, f"{nom_sans_ext}.jpg")

                try:
                    image = Image.open(chemin_heic)
                    image.save(chemin_jpg, "JPEG")
                    print(f"Converti : {chemin_heic} -> {chemin_jpg}")
                    compteur += 1
                except Exception as e:
                    print(f"Erreur lors de la conversion de {fichier} : {e}")

    print(f"\nTerminé ! {compteur} image(s) convertie(s).")

# --- Utilisation du script --- #
mon_dossier = "Classification/"
convertir_dossier_heic(mon_dossier)