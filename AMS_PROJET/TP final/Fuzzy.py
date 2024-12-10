from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# with open(f"Partie2/prelude_a_fondation_chapter_1_characters.txt", "r") as file:
    # listePersonnagesTrier = [line.strip() for line in file if line.strip()]

# Fonction pour regrouper les noms similaires
def regrouper_noms(noms):
    clusters = []
    while noms:
        nom = noms.pop(0)
        cluster = [nom]
        for autre_nom in noms[:]:
            if fuzz.ratio(nom, autre_nom) > 60:  # Seuil de similarité (par exemple, 80%)
                cluster.append(autre_nom)
                noms.remove(autre_nom)
        clusters.append(cluster)
    return clusters

# Exemple d'exécution
# clusters = regrouper_noms(listePersonnagesTrier)
# print(clusters)



###Liste des livres et chapitres à extraire        
books = [
    (list(range(0, 19)), "paf"),
    (list(range(0, 18)), "lca"),
]

Listeperso = []
ListeALLperso = []
for chapters, book_code in books:
    for chapter in chapters:
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
        if chapter!=0:
            with open(f"Partie2/{repertory}_chapter_{chapter}_characters.txt", "r") as file:
                listePersonnagesTrier = [line.strip() for line in file if line.strip()]
                for name in listePersonnagesTrier:
                    ListeALLperso.append(name)
                    
                clusters = regrouper_noms(listePersonnagesTrier)
                print("////////////////////////////////////////////////////")
                print(book_code,chapter)
                print(clusters)
                
with open(f"ListeTrier.txt", "w") as file:
    for name in list(dict.fromkeys(ListeALLperso)):
        file.write(name + '\n')