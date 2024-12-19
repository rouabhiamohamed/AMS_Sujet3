import re
import stanza

# Initialiser le pipeline Stanza pour la langue française
stanza.download('fr')  # Télécharger le modèle de langue française
nlp = stanza.Pipeline('fr')

# Lire le texte depuis un fichier
with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    texte = file.read()

# 1. Nettoyage du texte : retirer les éléments inutiles (ponctuation, numéros, titres de chapitres, etc.)
def nettoyer_texte(texte):
    
    # Supprimer les chiffres et les symboles inutiles
    texte = re.sub(r'\d+', '', texte)  # Supprimer les numéros
    texte = re.sub(r'[^a-zA-ZéèàùâêîôûçÉÈÀÙÂÊÎÔÛÇ\s]', '', texte)    
    # Remplacer les guillemets typographiques par des guillemets droits et les apostrophes typographiques par des apostrophes simples
    texte = re.sub(r'[«»]', '"', texte)
    texte = re.sub(r'[’‘]', "'", texte)
    
    # Convertir en minuscules
    #texte = texte.lower()
    
    # Supprimer les espaces superflus
    texte = re.sub(r'\s+', ' ', texte).strip()

    return texte



# 3. Affichage des entités extraites
texte_nettoye = nettoyer_texte(texte)
doc = nlp(texte_nettoye)

### Liste pour stocker les personnages extraits
listePersonnages = []
listeNomsPersonnages = []

### Extraction des entités nommées de type "PER" (personne)
for ent in doc.ents:
    if ent.type == "PER":
        ent.text = ent.text.replace("\n", " ").strip() ###Certaines entités ont des \n, on les enlève
        listePersonnages.append(ent.text)

print(set(listePersonnages))

# Afficher les entités nommées extraites
# print("\nEntités nommées extraites:")
# for ent in entites:
    # print(f"{ent[0]} : {ent[1]}")
