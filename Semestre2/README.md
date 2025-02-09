# Pense-bête du projet


## Choses à faire pour le projet :
- Mettre en place un Docker pour rendre plus facile l'installation et le démarrage du programme
- Mettre en place les visualisations des graphes
- Mettre en place les poids des arêtes
- Optimiser l'ancien code
- Mettre en place les liens d'inimité et d'amitié
- Améliorer l'heuristique de l'ancien code
- Faire des recherches sur la coreference


# Docker

La commande pour construire le docker :
`docker build -t ams_docker .`

Prend 10 à 14 minutes à compiler :
`[+] Building 872.6s (12/12) FINISHED `

La commande pour run le docker :
`docker run --rm ams_docker`


`docker-compose up --build`

15 à 20 minutes pour obtenir des résultats de l'algo avec le docker