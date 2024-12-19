import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Téléchargement des ressources nécessaires pour NLTK
nltk.download('punkt')  # Pour le tokeniseur
nltk.download('stopwords')  # Pour les stopwords

# Initialisation du stemmer et des stopwords en français
stemmer = PorterStemmer()
stop_words = set(stopwords.words('french'))

def clean(text):
    """Fonction pour nettoyer un texte donné"""
    text = str(text).lower()  # Mettre en minuscules
    text = re.sub('\[.*?\]', '', text)  # Supprimer les crochets et leur contenu
    # text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Supprimer la ponctuation
    text = re.sub('\n', '', text)  # Supprimer les sauts de ligne
    text = re.sub('\w*\d\w*', '', text)  # Supprimer les mots contenant des chiffres
    # text = re.sub(r'[^\x00-\x7F]+', '', text)  # Supprimer les caractères non-ASCII
    text = " ".join(text.split())  # Retirer les espaces inutiles au début et à la fin

    tokens = word_tokenize(text)  # Tokenisation du texte
    cleaned_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]  # Appliquer le stemming et retirer les stopwords
    cleaned_text = ' '.join(cleaned_tokens)  # Rejoindre les tokens nettoyés

    return cleaned_text

# Charger le fichier, nettoyer son contenu et enregistrer dans un nouveau fichier
input_file_path = "reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed"
output_file_path = "chapter_1_cleaned.txt"

with open(input_file_path, "r", encoding="utf-8") as file:
    text = file.read()  # Lire tout le contenu du fichier

# Appliquer la fonction de nettoyage
cleaned_text = clean(text)

# Sauvegarder le texte nettoyé dans un nouveau fichier
with open(output_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(cleaned_text)

print(f"Le texte nettoyé a été sauvegardé dans : {output_file_path}")
