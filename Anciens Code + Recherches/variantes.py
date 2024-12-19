# Lecture du fichier ListeTrier.txt
with open(f"ListeTrier.txt", "r") as file:
    text = file.read()

# Diviser le texte en mots en fonction des retours à la ligne ou des espaces
ListNom = [nom.strip() for nom in text.splitlines() if nom.strip()]

# Créer une liste de variantes de noms (noms sans espaces)
Nom = []
for nom in ListNom:
    if " " not in nom:  # Si le nom ne contient pas d'espace
        Nom.append([nom])

print("Variantes de noms :")
print(Nom)

# Processus de fusion de variantes
NomCopy = [list(variantes) for variantes in Nom]  # Création d'une copie indépendante de la liste Nom

# Fusionner les variantes en fonction des noms en commun
def fusionner_variantes(variantes):
    fusionnees = []
    for variante in variantes:
        ajoutée = False
        for groupe in fusionnees:
            if set(variante).intersection(groupe):  # Si des noms en commun sont trouvés
                groupe.update(variante)  # Fusionner les groupes
                ajoutée = True
                break
        if not ajoutée:
            fusionnees.append(set(variante))  # Ajouter une nouvelle variante si aucune fusion n'a eu lieu
    return [list(groupe) for groupe in fusionnees]  # Convertir les sets en listes

# Appliquer la fusion des variantes
final_variantes = fusionner_variantes(NomCopy)

# Affichage final après fusion
print("////////////////////////////////////////")
print("Liste des variantes après fusion :")
print(final_variantes)