import spacy

# Charger le modèle linguistique français de spaCy
nlp = spacy.load('fr_core_news_sm')

def extraire_verbes(texte):
    # Traiter le texte avec spaCy
    doc = nlp(texte)
    
    # Extraire les verbes en filtrant les tokens avec le tag grammatical 'VERB'
    verbes = [token.text for token in doc if token.pos_ == 'MISC']
    
    return verbes
    
    
def lemmatiser_texte(texte):
    # Traiter le texte avec spaCy
    doc = nlp(texte)
    
    # Lemmatisation : obtenir la forme de base (lemme) de chaque mot
    lemmes = [token.lemma_ for token in doc]
    
    return lemmes
    
    
# with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    # texte = file.read()
    # texte = texte.lower()

# Extraction des verbes
# verbes = extraire_verbes(texte)
# print(verbes)  # Affiche les verbes trouvés : ['mange', 'court']

texte = "Asseyez vous s'il vous plait !"

# Lemmatisation du texte
lemmes = lemmatiser_texte(texte)
if("asseyez" in lemmes):
    print("Oui")
print(lemmes)