from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")

# Créer un pipeline NER
nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Fonction pour découper le texte en plusieurs parties de maximum 'max_words' mots
def split_text_into_parts(text, max_words=1000):
    # Diviser le texte en mots
    words = text.split()

    # Liste pour stocker les morceaux du texte
    parts = []

    # Découper le texte en morceaux de taille maximale 'max_words'
    for i in range(0, len(words), max_words):
        part = ' '.join(words[i:i + max_words])
        parts.append(part)

    return parts

# Lire le texte depuis un fichier
with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    texte = file.read()

# Découper le texte en parties de maximum 1000 mots
text_parts = split_text_into_parts(texte, max_words=1000)

listePersonnages = []

# Appliquer NER sur chaque partie du texte et afficher les entités
for i, part in enumerate(text_parts, 1):
    print(f"Partie {i} :")
    entities = nlp(part)
    for entity in entities:
        if entity['entity_group']=="PER":
            listePersonnages.append(entity['word'])
    print("\n" + "-"*50 + "\n")
print(set(listePersonnages))

# Vous pouvez ensuite intégrer ces entités dans votre graph comme montré plus haut




# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForTokenClassification

# books = [
    # (list(range(1, 19)), "paf"),
    # (list(range(1, 18)), "lca"),
# ]

# df_dict = {"ID": [], "graphml": []}

##Traitement des livres et chapitres
# for chapters, book_code in books:
    # for chapter in chapters:
        
        ##Définir le répertoire en fonction du code du livre
        # if book_code == "paf":
            # repertory = "prelude_a_fondation"
        # else:
            # repertory = "les_cavernes_d_acier"
        
        ##Ouvrir le fichier correspondant au chapitre 'chapter' du livre 'book_code'
        # with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
            # texte = file.read()

        ##Initialiser le tokenizer et le modèle pour NER
        # tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
        # model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")

        ##Créer un pipeline NER
        # nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0)

        ##Appliquer NER sur le texte du chapitre
        # entities = nlp(texte)
        # for entity in entities:
            # print(f"Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']}")

        ##Filtrer les entités pour les types que vous voulez
        # listePersonnages = []
        # for entity in entities:
            # print(f"Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']}")


        ##Vous pouvez ensuite intégrer ces entités dans votre graph comme montré plus haut
        # print(set(listePersonnages))
        # print(chapter, book_code)
