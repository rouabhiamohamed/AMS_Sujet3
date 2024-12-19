from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Charger le tokenizer et le modèle du modèle GEODE/bert-base-french-cased-edda-ner-joint-label-levels
model_name = "GEODE/bert-base-french-cased-edda-ner-joint-label-levels"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Créer un pipeline pour NER (Named Entity Recognition)
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer)

### Liste des livres et chapitres à extraire        
books = [
    (list(range(1, 19)), "paf"),
    (list(range(1, 18)), "lca"),
]

# Traitement des livres et chapitres
for chapters, book_code in books:
    for chapter in chapters:
        # Définir le répertoire en fonction du code du livre
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
            
        ### Ouvrir le fichier correspondant au chapitre 'chapter' du livre 'book_code'
        with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
            texte = file.read()

        print("/////////////////", book_code, chapter, "////////////////////")

        # Appliquer le modèle pour extraire les entités nommées
        resultats = nlp_ner(texte)

        # Filtrer et afficher les entités de type "lieu"
        for entite in resultats:
            if entite['entity'] == 'lieu':  # Vérifier si l'entité est un lieu
                print(f"Entité Lieu : {entite['word']}, Confiance : {entite['score']:.4f}")

 