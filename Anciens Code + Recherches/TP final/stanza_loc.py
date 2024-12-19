from collections import defaultdict
import stanza

#stanza.download('fr')

nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')

###Liste des livres et chapitres à extraire        
books = [
    (list(range(1, 19)), "paf"),
    (list(range(1, 18)), "lca"),
]

df_dict = {"ID": [], "graphml": []}

locations = []
personnages = []
misc = []


lieuxASupprimer = [] #Liste pour stocker les éléments à supprimer

listeFiltrePos = [] #Liste de mots qui ne sont pas des PROPN (Pronom)

listeDePersonnagesCorpus = [] #Liste de personnage du corpus entier

# Traitement des livres et chapitres
for chapters, book_code in books:
    for chapter in chapters:

        # Définir le répertoire en fonction du code du livre
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
            
        ### Ouvrir le fichier correspondant au chapitre 'chapter' du livre 'book_code'
        with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
            texte = file.read()

        # Traitement du texte
            doc = nlp(texte)
        
        ### Liste pour stocker les personnages extraits
        listePersonnages = []
        listeLieux = []
        listeNomsPersonnages = []

        ### Extraction des entités nommées de type "PER" (personne)
        for ent in doc.ents:
            if ent.type == "PER" and len(ent.text)>2 and ent.text not in listeFiltrePos:
                ent.text = ent.text.replace("\n", " ").strip() ###Certaines entités ont des \n, on les enlève
                # listePersonnages.append(unidecode(ent.text)) #Unicocde pour enlever les accents
                listePersonnages.append(ent.text)
                listeDePersonnagesCorpus.append(ent.text)
            if ent.type == "LOC":
                listeLieux.append(ent.text)

        # Liste pour stocker les personnages uniques après filtrage
        listePersonnagesTrier = set(listePersonnages)  # Utiliser un set pour obtenir les personnages uniques

        listeLieuxTrier = set(listeLieux)

        for personnage in listePersonnagesTrier:
            compteurLieux = listeLieux.count(personnage)
            compteurPersonnages = listePersonnages.count(personnage)
            if compteurLieux > compteurPersonnages:
                lieuxASupprimer.append(personnage)






print(set(lieuxASupprimer))


