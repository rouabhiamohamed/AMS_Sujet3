import spacy
import coreferee

# Charger le modèle spaCy et ajouter le pipe Coreferee
nlp = spacy.load("fr_core_news_lg")
nlp.add_pipe("coreferee")

# Exemple de texte
with open(f"reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    texte = file.read()

# Traiter le texte avec spaCy
doc = nlp(texte)

# Afficher les chaînes de co-références
doc._.coref_chains.print()

# Résoudre les co-références pour un mot spécifique (par exemple, le dernier "ils")
resolved_text = doc._.coref_chains.resolve(doc[34])

# Afficher le texte après la résolution des co-références
print("Texte après résolution des co-références :")
print(resolved_text)
