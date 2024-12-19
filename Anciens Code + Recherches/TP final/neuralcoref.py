import stanza

# Télécharger le modèle en français
stanza.download('fr')

# Charger le pipeline de Stanza
nlp = stanza.Pipeline("fr", processors="tokenize,coref")

with open(f"reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    texte = file.read()

def split_text(text, chunk_size=1000):
    # Découper en morceaux de taille fixe
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Diviser le texte en morceaux plus petits
chunks = split_text(texte)

# Stocker les résultats de la coréférence
coref_results = []

# Traiter chaque morceau de texte
for chunk in chunks:
    doc = nlp(chunk)
    # Ajouter les résultats de la coréférence sous forme de chaîne de caractères
    coref_results.append("{:C}".format(doc))

# Convertir la liste de résultats en une seule chaîne (avec des sauts de ligne entre chaque résultat)
coref_results_str = "\n".join(coref_results)

# Enregistrer les résultats dans un fichier
with open(f"Coreref_prelude_a_fondation_chapter_1.txt", 'w') as file:
    file.write(coref_results_str)




