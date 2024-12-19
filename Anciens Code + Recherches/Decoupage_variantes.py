import stanza

nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos')


##Liste des livres et chapitres à extraire        
# books = [
    # (list(range(0, 19)), "paf"),
    # (list(range(0, 18)), "lca"),
# ]

Listeperso = []


# for chapters, book_code in books:
    # for chapter in chapters:
        # if book_code == "paf":
            # repertory = "prelude_a_fondation"
        # else:
            # repertory = "les_cavernes_d_acier"
        # if chapter!=0:
            # with open(f"Partie2/{repertory}_chapter_{chapter}_characters.txt", "r") as file:
                # listePersonnagesTrier = [line.strip() for line in file if line.strip()]
            # print("//////////////////////////////////////")
            # print(book_code,chapter)
            # clusters = []
            # for nomDePerso in listePersonnagesTrier:
                    ##Séparer le prénom et le nom
                    # prenom = nomDePerso.split()
                    # cluster = [nomDePerso]
                    ##Vérifier qu'il y a au moins deux mots (prénom et nom)
                    ##Comparer chaque nom avec les autres dans la liste
                    # for nom in listePersonnagesTrier:
                        # if prenom[0] in nom or nom in prenom[0] and nom!=nomDePerso:
                            # print(prenom[0], nom)
                            # cluster.append(nom)
                        # if len(prenom) >= 2:
                            # if prenom[1] in nom or nom in prenom[1] and nom!=nomDePerso:
                            # print(prenom[0], nom)
                            # cluster.append(nom)
                    # clusters.append(cluster)
            # print(clusters)
            
            
with open(f"ListeTrier.txt", "r") as file:
    listePersonnagesTrier = [line.strip() for line in file if line.strip()]            
clusters = []
for nomDePerso in listePersonnagesTrier:
        #Séparer le prénom et le nom
        prenom = nomDePerso.split()
        cluster = [nomDePerso]
        #Vérifier qu'il y a au moins deux mots (prénom et nom)
        #Comparer chaque nom avec les autres dans la liste
        for nom in listePersonnagesTrier:
            # if prenom[0] in nom and nom!=nomDePerso:
                # print(prenom[0], nom)
                # cluster.append(nom)
            if len(prenom) >= 2:
                if prenom[-1] in nom or nom in prenom[-1] and nom!=nomDePerso:
                    # print(prenom[-1], nom)
                    if(nom not in cluster):
                        cluster.append(nom)
        clusters.append(cluster)
# for cluster in clusters:
    # print(cluster)
    
    
# for cluster in clusters:
    # if len(cluster) == 1:
        # for varianteARemplir in clusters:
            # if len(varianteARemplir) > 1:
                # for nomDansVariante in varianteARemplir:
                    # docPos = nlp(nomDansVariante)
                    # pos_premier_mot = docPos.sentences[0].words[0].upos
                    # print(pos_premier_mot)
                    # if ((nomDansVariante in cluster[0] or cluster[0] in nomDansVariante) ):
                        # if cluster[0] not in varianteARemplir:
                            # varianteARemplir.append(cluster[0])
            
    
for variantesNoms in clusters[:]:  # Copier pour éviter les conflits pendant la suppression
    for s in clusters[:]: 
        if variantesNoms != s and all(nom in s for nom in variantesNoms):  # Si variantesNoms est un sous-ensemble de s
            if variantesNoms in clusters:  # Vérifier que variantesNoms n'a pas déjà été supprimé
                clusters.remove(variantesNoms)
        elif variantesNoms != s and all(nom in variantesNoms for nom in s):  # Si s est un sous-ensemble de variantesNoms
            if s in clusters:  # Vérifier que s n'a pas déjà été supprimé
                clusters.remove(s)    
    
for cluster in clusters:
    print(cluster)