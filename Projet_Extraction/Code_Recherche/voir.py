# Initialisation du modèle et des variables
tagger = SequenceTagger.load("flair/ner-french")
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')
books = [
    (list(range(1, 20)), "paf"),
    (list(range(1, 19)), "lca"),
]

# Optimisation de l'extraction des entités et du traitement de texte
def extract_entities_for_chapter(texte):
    sentence = Sentence(texte)
    tagger.predict(sentence)
    return [ent.text for ent in sentence.get_spans('ner') if ent.get_label('ner').value == 'PER']

def process_text_for_chapter(texte):
    doc = nlp(texte)
    tokens = [word.text for sent in doc.sentences for word in sent.words]
    return doc, tokens

# Optimisation du traitement des chapitres avec parallèle et réduction des appels à nlp
from concurrent.futures import ProcessPoolExecutor

def process_book(book_code, chapters):
    for chapter in chapters:
        with open(f"Textes_Processed/{book_code}_chapter_{chapter}.txt", "r") as file:
            texte = file.read()
        
        # Extraction des entités et nettoyage
        listePersonnages = extract_entities_for_chapter(texte)
        doc, tokens = process_text_for_chapter(texte)
        
        # Continue with your filtering and other steps...
        # Mettre en place les étapes suivantes comme Filtre_Ner_Pos, Extraction_Per, etc.
        
# Parallélisation des livres ou chapitres
with ProcessPoolExecutor() as executor:
    for book_code, chapters in books:
        executor.submit(process_book, book_code, chapters)
