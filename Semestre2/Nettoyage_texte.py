import stanza
import re
import PyPDF2

def suppr_caracteres(file_path):
    """Supprimer les caractères manquants et les numéros de page."""
    with open(file_path, "r") as file:
        content = file.read()

    # Remplacer les caractères indésirables
    content = re.sub(r'� \d+ �', '<->', content)
    content = re.sub(r'- \d+ -', '<->', content) 
    content = content.replace("�", '')
    
    return content

def traiter_texte(texte):
    """Prétraitement du texte pour les sauts de ligne et ponctuation."""
    # Remplacer les sauts de ligne inutiles
    texte = re.sub(r'(?<!-)\n', ' ', texte)  # Remplacer les sauts de ligne (\n) par des espaces, sauf quand précédé par un tiret
    texte = re.sub(r'-\n', '-', texte)  # Conserver les tirets avec les mots suivants
    texte = re.sub(r'—', "", texte)  # Supprimer les tirets longs "—"
    
    # Remplacer la ponctuation (points, points d'interrogation, etc.)
    texte = re.sub(r'\.(?!\s\»|\.|[^\s]\.| [a-z])', '.\n', texte)
    texte = re.sub(r'\?(?!\s\»)','?\n', texte)
    texte = re.sub(r'\!(?!\s\»)','!\n', texte)
    texte = re.sub(r'\)(?!\s\»)',')\n', texte)
    texte = re.sub(r'\»(?!\.)','»\n', texte)
    
    #pour retirer les nombres des chapitres
    texte = re.sub(r'\s{6}(\S+)', '', texte)
    
    # Supprimer les espaces multiples et remplacer par un seul espace
    texte = re.sub(r'[ ]+', ' ', texte)

    # Supprimer le contenu entre les deux premières balises <->, y compris ce qui est à l'intérieur
    texte = re.sub(r'(<->)(.*?)(<->)', '', texte, count=1, flags=re.DOTALL)  # count=1 pour ne remplacer que la première occurrence, re.DOTALL pour inclure les sauts de ligne
    return texte

stanza.download('fr')  # Télécharge et installe le modèle Stanza pour le français

# Initialisation du pipeline de traitement de texte pour le français
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')

# Liste des livres
books = [
    "Fondation_et_empire_sample",
    "Fondation_foudroyée_sample",
    "Fondation_sample",
    "Seconde_Fondation_sample",
    "Terre_et_Fondation_sample"
]

# Parcours des livres et traitement
for book in books:
    # Extraction du texte à partir du PDF
    with open(f"Corpus_ASIMOV/{book}.pdf", "rb") as fichier_pdf:
        lecteur_pdf = PyPDF2.PdfReader(fichier_pdf)
        texte = "".join(page.extract_text() for page in lecteur_pdf.pages)
    
    # Sauvegarde du texte brut initial
    with open(f"Textes_Processed/{book}.txt", "w") as file:
        file.write(texte)
    
    # Traitement des caractères manquants et suppression des numéros de page
    texte = suppr_caracteres(f"Textes_Processed/{book}.txt")
    
    # Prétraitement du texte
    texte = traiter_texte(texte)
    
    # Traitement linguistique avec Stanza
    doc = nlp(texte)

    # Correction des mots en majuscules
    for sent in doc.sentences:
        for word in sent.words:
            if word.text.isupper() and len(word.text) > 2:
                # Vérification si le mot est un nom propre (PROPN)
                if word.upos == "PROPN":
                    texte = texte.replace(word.text, word.text.capitalize())  # Remplacer par la version capitalisée
    
    # Sauvegarde du texte traité
    with open(f"Textes_Processed/{book}.txt", "w") as file:
        file.write(texte)  # Écriture du texte traité dans un fichier de sortie
        print(f"Sauvegarde du fichier {book}.txt")
