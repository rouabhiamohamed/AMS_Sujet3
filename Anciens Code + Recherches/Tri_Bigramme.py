import stanza

# Initialisation du pipeline de traitement de texte pour le français
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')

# Liste des livres et chapitres à extraire        
books = [
    (list(range(0, 19)), "paf"),
    (list(range(0, 18)), "lca"),
]

# Variables pour stocker les noms des personnages
listeNomsPersonnagesPaf = []
listeNomsPersonnagesLca = []

for chapters, book_code in books:
    Listeperso = []
    for chapter in chapters:
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
        
        if chapter != 0:
            with open(f"Partie2/{repertory}_chapter_{chapter}_characters.txt", "r") as file:
                listePersonnagesTrier = [line.strip() for line in file if line.strip()]
            
            with open(f"Bigrammes/Bigrammes_of_{repertory}_chapter_{chapter+1}.txt", 'r') as file:
                # Lire les bigrammes et ne garder que les deux premiers éléments
                listeBigramme = [line.strip().split()[:2] for line in file if line.strip()]
            
            for noms in listePersonnagesTrier:
                if len(noms.split()) == 1:  # Si c'est un seul mot (pas un nom composé)
                    for bigramme in listeBigramme:
                        if bigramme[1] == noms:
                            # On ne traite que le premier mot du bigramme
                            docBi = nlp(bigramme[0])
                            PosBigramme = [word.upos for sent in docBi.sentences for word in sent.words]
                            # Vérifier si le premier mot commence par "à" ou "sur"
                            if bigramme[0].startswith("à") or bigramme[0].startswith("sur"):
                                print(noms, bigramme)
            
            # Ajouter les personnages dans la liste
            for perso in listePersonnagesTrier:
                Listeperso.append(perso)
            
            print(f"///////{chapter}///{repertory}//////")
