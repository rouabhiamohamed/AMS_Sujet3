import stanza

nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos')


##Liste des livres et chapitres à extraire        
# books = [
    # (list(range(0, 19)), "paf"),
    # (list(range(0, 18)), "lca"),
# ]

Listeperso = []


# Liste des personnages triés
with open("ListeTrier.txt", "r") as file:
    listePersonnagesTrier = [line.strip() for line in file if line.strip()]            

clusters = []
for nomDePerso in listePersonnagesTrier:
    # Séparer le prénom et le nom
    prenom = nomDePerso.split()
    cluster = [nomDePerso]

    # Vérifier si le prénom et nom doivent être comparés
    if len(prenom) >= 2:
        for nom in listePersonnagesTrier:
            if prenom[-1] in nom or nom in prenom[-1]:
                if nom != nomDePerso and nom not in cluster:
                    cluster.append(nom)

    clusters.append(cluster)

# Eliminer les clusters qui sont des sous-ensembles d'autres clusters
for variantesNoms in clusters[:]:
    for s in clusters[:]:
        if variantesNoms != s and all(nom in s for nom in variantesNoms):
            if variantesNoms in clusters:
                clusters.remove(variantesNoms)
        elif variantesNoms != s and all(nom in variantesNoms for nom in s):
            if s in clusters:
                clusters.remove(s)

# Afficher les clusters
for cluster in clusters:
    print(cluster)