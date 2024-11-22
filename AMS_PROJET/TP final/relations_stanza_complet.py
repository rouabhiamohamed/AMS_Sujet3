import stanza
from collections import defaultdict

# Télécharger et initialiser le pipeline avec tous les processeurs nécessaires
stanza.download('fr')
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,lemma,ner,depparse')

# Charger le texte
with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    texte = file.read()

# Traiter le texte avec Stanza
doc = nlp(texte)

# Dictionnaire pour stocker les interactions
interactions = defaultdict(list)

# Analyser les entités nommées et leurs relations
for sent in doc.sentences:
    # Extraire les entités nommées dans la phrase
    entities_in_sentence = [ent.text for ent in sent.ents if ent.type == "PER"]

    # Si plusieurs entités nommées sont présentes dans la même phrase
    if len(entities_in_sentence) > 1:
        # Analyser les relations syntaxiques
        for word in sent.words:
            # Chercher des verbes d'action qui relient les entités entre elles
            if word.pos == 'VERB':
                # Examiner les dépendances autour du verbe
                subject = []
                object_ = []

                # Chercher les sujets et objets du verbe
                for dep_word in sent.words:
                    if dep_word.head == word.id and dep_word.deprel == 'nsubj':  # Sujet nominal
                        subject.append(dep_word.text)
                    elif dep_word.head == word.id and dep_word.deprel == 'obj':  # Objet direct
                        object_.append(dep_word.text)

                # Si les sujets et objets sont des entités nommées, on les relie
                for sub in subject:
                    for obj in object_:
                        if sub in entities_in_sentence and obj in entities_in_sentence:
                            interactions[sub].append((word.text, obj))

# Afficher les interactions trouvées entre les personnages
for person, relations in interactions.items():
    print(f"Interactions pour {person}:")
    for relation in relations:
        print(f"  - Verbe : {relation[0]}, avec {relation[1]}")
