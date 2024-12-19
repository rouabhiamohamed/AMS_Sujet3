from collections import defaultdict
import spacy
import re
#def suppr_page_num(file_path):
with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    content = file.read()

#updated_content = re.sub(r'—', '', content)

#with open("chaptest.txt", "w") as file:
 #   file.write(updated_content)


# Charger le modèle de NLP (assurez-vous que le modèle est installé)
nlp = spacy.load("fr_core_news_sm")  # Remplacez par le modèle approprié

# Ouvrir le fichier correspondant
#with open("chaptest.txt", "r") as file:
 #   content = file.read()


texte = """  Oui, mais s’il se fait récupérer entre-temps par un de
mes ennemis, disons plutôt un ennemi de l’Empire, car, après
tout, je suis l’Empire, ou si – de son propre chef – il décide de
servir un ennemi... Je n’écarte pas cette hypothèse, voyez-vous.
     — Et vous avez parfaitement raison. Je veillerai à ce que
cela n’arrive pas, mais si, contre toute attente, cela devait se
produire, mieux vaudrait encore que personne ne profite de ses
services plutôt que les voir tomber en de mauvaises mains. »
     Cléon paraissait mal à l’aise. « Je m’en remets entièrement
à vous, Demerzel, mais j’espère que nous n’agirons pas avec
trop de hâte. Il pourrait n’être, après tout, que l’inventeur d’une
théorie dépourvue de toute espèce d’application pratique.
     — C’est fort possible, Sire, mais il serait plus prudent de
partir de l’idée que l’homme est – ou pourrait être – important.
Nous n’aurons perdu au plus qu’un peu de temps si jamais nous
découvrons que nous nous sommes préoccupés d’une non-
entité. Nous risquons de perdre une Galaxie si nous découvrons
que nous avons ignoré quelqu’un d’important.
     — Fort bien, si vous le dites... mais j’espère que je n’aurai
pas à connaître les détails – s’ils s’avéraient déplaisants.
     — Espérons que ce ne sera pas le cas », répondit Demerzel."""


# Traitement du texte avec spaCy
doc = nlp(content)

# Initialisation d'un dictionnaire pour stocker les relations
relations = defaultdict(set)

# Extraire les entités "PERSON" (les noms des personnages)
# Pour déboguer ou tester les entités, décommentez la ligne suivante
# personnages = [ent.text for ent in doc.ents if ent.label_ == "PER"]
# for ent in doc.ents :
#     print(ent.text)

# Parcourir les phrases du texte
for sent in doc.sents:
    # Extraire les personnages dans la phrase
    personnes_dans_phrase = [ent.text for ent in sent.ents if ent.label_ == "PER" and len(ent.text) > 1 and ent.text[0].isupper()]
    if(personnes_dans_phrase):
       print(personnes_dans_phrase)
    # Ajouter des relations entre chaque paire de personnages dans la même phrase
    for i in range(len(personnes_dans_phrase)):
        for j in range(i + 1, len(personnes_dans_phrase)):
            p1 = personnes_dans_phrase[i]
            p2 = personnes_dans_phrase[j]
            relations[p1].add(p2)
            relations[p2].add(p1)  # Ajouter aussi l'inverse (relation symétrique)

# Afficher les relations détectées
print("Relations détectées:")
for p1, related_personnes in relations.items():
    for p2 in related_personnes:
        print(f"{p1} -> {p2}")
