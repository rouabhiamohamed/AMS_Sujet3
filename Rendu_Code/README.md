# AMS_Sujet3

Pour préparer les données et nettoyer les corpus nécessaires à la construction du graphe, il est important d'exécuter le script de nettoyage avec la commande suivante :  
`python3 ./Nettoyage_texte.py`

Une fois le nettoyage effectué, pour démarrer l'exécution du code principal, rendez-vous dans le répertoire où le code est situé et lancez-le en utilisant la commande suivante :  
`python3 ./Code_Asimov.py`  
Le processus de compilation peut prendre un certain temps, mais des messages de progression seront affichés pour indiquer l'avancement du processus.

### Résultat :
Après avoir exécuté le programme, vous obtiendrez un fichier `my_submission.csv` qui contient le graphe généré.

### Librairies à installer :  
- `stanza`  
- `networkx`  
- `pandas`  
- `defaultdict` (de `collections`)  
- `re`  
- `flair` 
- `numpy`

Assurez-vous d'installer toutes les librairies nécessaires avant d'exécuter le code pour éviter tout problème d'importation.
