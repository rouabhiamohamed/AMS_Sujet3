from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Charger le tokenizer et le modèle
model_name = "GEODE/bert-base-french-cased-edda-ner-IOB2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

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
            text = file.read()

        print("/////////////////", book_code, chapter, "////////////////////")

        # Tokenisation
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Prédiction
        with torch.no_grad():
            outputs = model(**inputs)

        # Obtenir les prédictions (indices des labels)
        predictions = torch.argmax(outputs.logits, dim=-1)

        # Convertir les indices en labels (ils sont indexés dans la configuration du modèle)
        labels = predictions[0].tolist()

        # Récupérer les labels correspondants à chaque index (id2label)
        id2label = model.config.id2label

        # Afficher les résultats
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        for token, label in zip(tokens, labels):
            label_name = id2label[label]  # Convertir l'indice en label
            if("NC_Person" in label_name):
                print(f"{token}: {label_name}")
