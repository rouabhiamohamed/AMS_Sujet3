import spacy

# Charger le modèle de spaCy pour le français
nlp = spacy.load("fr_core_news_sm")

# Exemple de texte à annoter
text = "Asseyez-vous donc, professeur."

# Analyser le texte avec spaCy
doc = nlp(text)

# Générer l'annotation CoNLL
conll_annotation = ""
for token in doc:
    # Chaque token est associé à son étiquette d'entité
    # Par défaut, spaCy met 'O' pour les tokens non annotés
    entity_label = token.ent_type_ if token.ent_type_ else "O"
    conll_annotation += f"{token.text}\t{entity_label}\n"

# Afficher l'annotation CoNLL
print(conll_annotation)