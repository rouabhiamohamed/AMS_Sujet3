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
        listeMISC = []
        listeNomsPersonnages = []

        ### Extraction des entités nommées de type "PER" (personne)
        for ent in doc.ents:
            if ent.type == "PER":
                ent.text = ent.text.replace("\n", " ").strip() ###Certaines entités ont des \n, on les enlève
                listePersonnages.append(ent.text)
            if ent.type == "LOC":
                listeLieux.append(ent.text)
                locations.append(ent.text)
            if ent.type == "MISC":
                listeMISC.append(ent.text)

        # Liste pour stocker les personnages uniques après filtrage
        listePersonnagesTrier = set(listePersonnages)

        listeLieuxTrier = set(listeLieux)

        listeMISCTrier = set(listeMISC)

        
        # print(listePersonnagesTrier)
        # print(listeLieuxTrier)
        # print(listeMISCTrier)

        # compteurLieux = 0
        # compteurPersonnages = 0
        # for personnage in listePersonnagesTrier:
            # compteurLieux = listeLieux.count(personnage)
            # compteurMISC = listeMISC.count(personnage)
            # compteurPersonnages = listePersonnages.count(personnage)
            # if(compteurLieux > compteurPersonnages or compteurMISC > compteurPersonnages):
                # print("////////////////////////////////////////////////////")
                # print(personnage)
                # print("////////////////////////////////////////////////////")

print(set(locations))