from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
import stanza
import re

# Télécharger le modèle Stanza pour le français si nécessaire
stanza.download('fr')

# Initialisation du pipeline de traitement de texte pour le français
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')

###Liste des livres et chapitres à extraire        
books = [
    (list(range(0, 19)), "paf"),
    (list(range(0, 18)), "lca"),
]


def nettoyer_texte(texte):
    
    # Supprimer les chiffres et les symboles inutiles
    texte = re.sub(r'\d+', '', texte)  # Supprimer les numéros
    texte = re.sub(r'[^a-zA-ZéèàùâêîôûçÉÈÀÙÂÊÎÔÛÇ\s]', '', texte)    
    # Remplacer les guillemets typographiques par des guillemets droits et les apostrophes typographiques par des apostrophes simples
    texte = re.sub(r'[«»]', '"', texte)
    texte = re.sub(r'[’‘]', "'", texte)
    
    # Convertir en minuscules
    texte = texte.lower()
    
    # Supprimer les espaces superflus
    texte = re.sub(r'\s+', ' ', texte).strip()

    return texte


df_dict = {"ID": [], "graphml": []}

lieuxASupprimer = [] #Liste pour stocker les éléments à supprimer

for chapters, book_code in books:
    for chapter in chapters:
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
        if chapter!=0:
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                texte = file.read()
            
            texte = re.sub(r'\b[A-ZÀ-ÿ0-9]+\b', lambda match: match.group(0).lower(), texte)
            with open(f"Bigramme/{repertory}_chapter_{chapter}.txt", "w") as file:
                file.write(texte)
            # Initialisation du modèle de bigrammes
            bigram_model = Phrases([simple_preprocess(texte)], min_count=5, threshold=100)
            bigram_phraser = Phraser(bigram_model)


                
            # Traitement du texte avec Stanza
            doc = nlp(texte)

            # Création des listes
            listePersonnages = []
            listeLieux = []

            # Traitement des entités nommées et des n-grams
            for ent in doc.ents:
                if ent.type == "PER" and len(ent.text) > 2:
                    # Vérifier si l'entité est un bigramme
                    bigram_text = bigram_phraser[simple_preprocess(ent.text)]
                    if len(bigram_text) > 1:  # Si c'est un bigramme détecté
                        ent.text = " ".join(bigram_text)  # Remplacer le texte par le bigramme
                    listePersonnages.append(ent.text)
                
                elif ent.type == "LOC" and len(ent.text) > 2:
                    bigram_text = bigram_phraser[simple_preprocess(ent.text)]
                    if len(bigram_text) > 1:
                        ent.text = " ".join(bigram_text)
                    listeLieux.append(ent.text)

            # Affichage des résultats
            print("////////////////",chapter, book_code,"/////////////////")
            print("Personnages détectés : ", list(dict.fromkeys(listePersonnages)))
            print("Lieux détectés : ", list(dict.fromkeys(listeLieux)))
            print("\n")