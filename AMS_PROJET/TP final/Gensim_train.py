from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
import os

# Chemin vers le dossier contenant les chapitres organisés en sous-répertoires
base_directory = "reseaux-de-personnages-de-fondation-session-2"

# Livres et chapitres associés
books = [
    (list(range(1, 19)), "prelude_a_fondation"),
    (list(range(1, 18)), "les_cavernes_d_acier"),
]

# Initialisation des structures pour tout le corpus
all_documents = []

# Parcourir chaque livre et ses chapitres
for chapters, repertory in books:
    for chapter in chapters:
        file_path = os.path.join(base_directory, repertory, f"chapter_{chapter}.txt.preprocessed")
        with open(file_path, "r") as file:
            documents = file.readlines()
            # Prétraitement des documents
            processed_docs = [simple_preprocess(doc) for doc in documents]
            all_documents.extend(processed_docs)

# Création du dictionnaire et du corpus
dictionary = corpora.Dictionary(all_documents)
corpus = [dictionary.doc2bow(doc) for doc in all_documents]

# Construction du modèle LSI
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=200)

# Indexation des similarités
index = similarities.MatrixSimilarity(lsi[corpus])

# Sauvegarde des modèles et index pour une réutilisation future
dictionary.save("foundation_dictionary.dict")
lsi.save("foundation_lsi.model")
index.save("foundation_similarity.index")
