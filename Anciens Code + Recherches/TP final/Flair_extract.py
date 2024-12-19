from flair.data import Sentence
from flair.models import SequenceTagger

# Charger le modèle NER pour le français
tagger = SequenceTagger.load("flair/ner-french")

       
###Liste des livres et chapitres à extraire        
books = [
    (list(range(0, 19)), "paf"),
    (list(range(0, 18)), "lca"),
]

df_dict = {"ID": [], "graphml": []}

lieuxASupprimer = [] #Liste pour stocker les éléments à supprimer

listeLieuxAll = []

for chapters, book_code in books:
    for chapter in chapters:
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
        if chapter!=0:
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                texte = file.read()

            # Créer une phrase à partir du texte
            sentence = Sentence(texte)

            # Prédire les étiquettes NER
            tagger.predict(sentence)

            # Afficher la phrase avec les entités
            # print(sentence)

            # Liste pour stocker les entités de type LOC
            loc_entities = []
            Per_entities = []

            # Itérer sur les entités reconnues et ajouter les LOC à la liste
            for entity in sentence.get_spans('ner'):
                if entity.get_label('ner').value == 'LOC':  # Vérifier si l'entité est un lieu
                    loc_entities.append(entity.text)
                if entity.get_label('ner').value == 'PER':  # Vérifier si l'entité est un lieu
                    Per_entities.append(entity.text)

            # Afficher les entités de type LOC
            print(chapter, book_code)
            print('Les entités de type PER trouvées :')
            print(Per_entities)
            # print('Les entités de type LOC trouvées :')
            # print(loc_entities)
            print("\n")
            for lieux in loc_entities:
                listeLieuxAll.append(lieux)
