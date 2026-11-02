import os
from PIL import Image
import pillow_heif

# Enregistrer le support HEIC dans Pillow
pillow_heif.register_heif_opener()

def convertir_dossier_heic(dossier_source, dossier_destination=None):
    # Si aucun dossier de destination n'est fourni, on enregistre au même endroit
    if not dossier_destination:
        dossier_destination = dossier_source
        
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(dossier_destination, exist_ok=True)
    
    # Lister tous les fichiers du dossier
    fichiers = os.listdir(dossier_source)
    
    compteur = 0
    for fichier in fichiers:
        # Vérifier si le fichier est un HEIC (insensible à la casse)
        if fichier.lower().endswith('.heic'):
            chemin_heic = os.path.join(dossier_source, fichier)
            
            # Construire le nouveau nom avec l'extension .jpg
            nom_sans_ext = os.path.splitext(fichier)[0]
            chemin_jpg = os.path.join(dossier_destination, f"{nom_sans_ext}.jpg")
            
            try:
                # Ouvrir et convertir
                image = Image.open(chemin_heic)
                image.save(chemin_jpg, "JPEG")
                print(f"Converti avec succès : {fichier} -> {nom_sans_ext}.jpg")
                compteur += 1
            except Exception as e:
                print(f"Erreur lors de la conversion de {fichier} : {e}")
                
    print(f"\nTerminé ! {compteur} image(s) convertie(s).")

# --- Utilisation du script ---
# Remplacez par le chemin de votre dossier
mon_dossier = "tests/" 
convertir_dossier_heic(mon_dossier)