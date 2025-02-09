# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY . .

# Installer les dépendances Python
RUN pip install stanza networkx pandas flair pyvis

# Télécharger le modèle Stanza pour le français
RUN python -c "import stanza; stanza.download('fr')"

# Télécharger le modèle Flair pour le français
RUN python -c "from flair.models import SequenceTagger; SequenceTagger.load('flair/ner-french')"

# Définir la commande par défaut pour exécuter le script
CMD ["python", "Asimov.py"]
