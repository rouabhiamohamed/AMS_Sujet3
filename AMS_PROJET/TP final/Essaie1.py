import stanza
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess

# Charger et préparer le texte
with open("reseaux-de-personnages-de-fondation-session-2/les_cavernes_d_acier/chapter_17.txt.preprocessed", "r") as file:
    texte = file.read()

# Télécharger le modèle Stanza pour le français si nécessaire
#stanza.download('fr')

# Initialisation du pipeline de traitement de texte pour le français
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')

# Traitement du texte
doc = nlp(texte)

# Liste pour stocker les personnages extraits
personnages = []
liste_final = []

# Extraction des entités nommées de type "PER" (personne)
for ent in doc.ents:
    if ent.type == "PER":
        ent.text = ent.text.replace("\n", " ").strip()
        personnages.append(ent.text)

# Liste pour stocker les personnages uniques après filtrage
personnages_uniques = set(personnages)  # Utiliser un set pour obtenir les personnages uniques

print("Liste personnages ")
print(personnages_uniques)


# Liste finale qui contiendra les ensembles de variantes
for nom in personnages_uniques:
    variantes = {nom}  # Utiliser un set pour éviter les doublons
    for autre_nom in personnages_uniques:
        if autre_nom != nom and (autre_nom in nom or nom in autre_nom) or (nom.upper()==autre_nom or nom==autre_nom.upper()):  # Vérifier si c'est une variante
            variantes.add(autre_nom)  # Obtenir toutes les variantes
    if variantes not in liste_final:  # Vérifier si l'ensemble n'est pas déjà dans la liste
        liste_final.append(variantes)

print("\nListe des noms de personnages :")
print(liste_final)

print("\nListe des noms de personnages :")
for variante in liste_final:
    print(variante)
    
# with open("reseaux-de-personnages-de-fondation-session-2/les_cavernes_d_acier/chapter_17.txt.preprocessed", "r") as file:
    # documents = file.readlines()

# processed_docs = [simple_preprocess(doc) for doc in documents]

# dictionary = corpora.Dictionary(processed_docs)

# corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# lsi = models.LsiModel(corpus, num_topics=200)

# index = similarities.MatrixSimilarity(lsi[corpus])  # Index the LSI-transformed corpus



# for i in liste_final:
    # for j in i :
        # query_nom1 = j
        # for k in liste_final:
            # for l in k :
                # if(k!=i):
                    # query_nom2 = l
                    # print(query_nom1," / ",query_nom2)
                    # query_bow_nom1 = dictionary.doc2bow(simple_preprocess(query_nom1))
                    # query_lsi_nom1 = lsi[query_bow_nom1]
                    # sims_nom1 = index[query_lsi_nom1]

                    # query_bow_nom2 = dictionary.doc2bow(simple_preprocess(query_nom2))
                    # query_lsi_nom2 = lsi[query_bow_nom2]
                    # sims_nom2 = index[query_lsi_nom2]

                    # for doc_id, sim_nom1 in enumerate(sims_nom1):
                        # sim_nom2 = sims_nom2[doc_id]
                        # if sim_nom1 > 0.09 and sim_nom2 > 0.09:
                            # print(f"Document {doc_id} : {query_nom1} et {query_nom2} ont une relation dans ce passage.")

