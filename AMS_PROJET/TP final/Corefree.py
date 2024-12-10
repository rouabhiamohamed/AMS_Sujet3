import spacy
import coreferee

# Charger le modèle spaCy et ajouter le pipe Coreferee
nlp = spacy.load("fr_core_news_sm")
nlp.add_pipe("coreferee")

# Exemple de texte
text = "Même si elle était très occupée par son travail, Julie en avait marre. Alors, elle et son mari décidèrent qu'ils avaient besoin de vacances. Ils allèrent en Espagne car ils adoraient le pays."

# Traiter le texte avec spaCy
doc = nlp(text)

# Afficher les chaînes de co-références
doc._.coref_chains.print()

# Résoudre les co-références pour un mot spécifique (par exemple, le dernier "ils")
resolved_text = doc._.coref_chains.resolve(doc[34])

# Afficher le texte après la résolution des co-références
print("Texte après résolution des co-références :")
print(resolved_text)
