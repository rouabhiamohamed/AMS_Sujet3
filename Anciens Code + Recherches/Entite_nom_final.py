import stanza

# Charger et préparer le texte
with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_4.txt.preprocessed", "r") as file:
    texte = file.read()

# Télécharger le modèle Stanza pour le français si nécessaire
stanza.download('fr')

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
        # Normaliser le texte : enlever les retours à la ligne et espaces superflus
        ent.text = ent.text.replace("\n", " ").strip()
        personnages.append(ent.text)

# Liste pour stocker les personnages uniques après filtrage
personnages_uniques = set(personnages)  # Utiliser un set pour obtenir les personnages uniques

# Fonction pour vérifier et fusionner les variantes de noms
def est_variante_simple(nom, liste):
    variantes = {nom}  # Utiliser un set pour éviter les doublons
    for autre_nom in liste:
        if autre_nom != nom and (autre_nom in nom or nom in autre_nom) or (nom.upper()==autre_nom or nom==autre_nom.upper()):  # Vérifier si c'est une variante
            variantes.add(autre_nom)  # Ajouter la variante au set
    return variantes
    
# Liste finale qui contiendra les ensembles de variantes
for nom in personnages_uniques:
    variantes = est_variante_simple(nom, personnages_uniques)  # Obtenir toutes les variantes
    if variantes not in liste_final:  # Vérifier si l'ensemble n'est pas déjà dans la liste
        liste_final.append(variantes)
        
# for variantes in liste_final:
    # for variantes2 in liste_final:
        # variantes.add("yo")
    # liste_final.remove(variantes2)

print("\nListe des noms de personnages :")
print(liste_final)

# for variantes in liste_final:
    # print("Ensembles similaires à l'ensemble de référence:")
    # for s in (s for s in liste_final if (s.issubset(variantes) or variantes.issubset(s)) and s != variantes):
        # liste_final.remove()

    
for variantes in liste_final:
    for s in liste_final: 
        if (s.issubset(variantes) and s != variantes) :
            liste_final.remove(s)
        elif (variantes.issubset(s) and s != variantes):
            liste_final.remove(variantes)
            
        

# def list_change(fin, liste):
    # if(fin):
        # return fin, liste
    # else :
        # listeTmp = [set(variantes) for variantes in liste] # Crée une copie de la liste, mais avec des sets
        # fin = True
        # for variantes in liste:
            # variantes_copy = variantes.copy()
            # for premierNomAComparer in variantes_copy:
                # for variantesAComparer in liste:
                    # for autreNomAComparer in variantesAComparer:
                        # if(variantesAComparer!=variantes and variantesAComparer in listeTmp):
                            # if(premierNomAComparer.__contains__(autreNomAComparer) or autreNomAComparer.__contains__(premierNomAComparer)):
                                # fin = False
                                # print("test")
                                # print(variantesAComparer)
                                # listeTmp.remove(variantesAComparer)
                                ##listeTmp.remove(variantes)
                                # variantes.update(variantesAComparer)  # Utilise update pour fusionner les sets
                                # listeTmp.append(variantes)
                                
        # return(fin,listeTmp)

# print(list_change(False, liste_final))

# fin = False
# while fin != True :
    # fin = True
    # for variantes in liste_final:
        # for premierNomAComparer in variantes:
            # for variantesAComparer in liste_final:
                # for autreNomAComparer in variantesAComparer:
                    # if(variantesAComparer!=variantes and variantesAComparer in listeTmp):
                        # if(premierNomAComparer.__contains__(autreNomAComparer) or autreNomAComparer.__contains__(premierNomAComparer)):
                            # fin = False
                            # print("test")
                            # print(variantesAComparer)
                            # for nomAAjouter in variantesAComparer:
                                # listeTmp[variantes].add(nomAAjouter)
                            # listeTmp.remove(variantesAComparer)

# Afficher la liste des personnages après filtrage
print("Tous les personnages extraits (avec doublons) :")
print(personnages)

print("\nPersonnages uniques (après filtrage des variantes) :")
print(personnages_uniques)

print("\nListe des noms de personnages :")
print(liste_final)

print("\nListe des noms de personnages (après fusion des variantes) :")
for variante in liste_final:
    print(variante)

