from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Charger un modèle spécifique pour la reconnaissance d'entités nommées
tokenizer = AutoTokenizer.from_pretrained("gilf/french-camembert-postag-model")
model = AutoModelForTokenClassification.from_pretrained("gilf/french-camembert-postag-model")

# Créer un pipeline pour la reconnaissance d'entités nommées (NER)
nlp_token_class = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)

# Lire le texte depuis un fichier
with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    texte = file.read()

# Appliquer le pipeline NER sur un extrait de texte spécifique
entities = nlp_token_class("""
Un ruban couvert d’inscriptions serrées en langage chiffré sortait sans arrêt des organes vitaux de l’enregistreuse ;
ce petit appareil recherchait et analysait ses « souvenirs », afin de fournir le renseignement demandé, qui était obtenu
grâce à d’infinies vibrations produites sur la brillante surface du mercure.
— Moi, reprit Simpson, je flanquerais mon pied au derrière de R. Sammy, si je n’avais pas peur de me casser une jambe !
Tu sais, l’autre soir, j’ai rencontré Vince Barrett...
— Ah oui ?...
— Il cherche à récupérer son job, ou n’importe quelle autre place dans le Service. Pauvre gosse ! Il est désespéré ! Mais
que voulais-tu que, moi, je lui dise ?... R. Sammy l’a remplacé, et fait exactement son boulot : un point c’est tout ! Et
pendant ce temps-là, Vince fait marcher un tapis roulant dans une des fermes productrices de levure. Pourtant, c’était un
gosse brillant, ce petit-là, et tout le monde l’aimait bien !
Baley haussa les épaules et répliqua, plus sèchement qu’il ne l’aurait voulu :
— Oh ! tu sais, nous en sommes tous là, plus ou moins.
Le patron avait droit à un bureau privé. Sur la porte en verre dépoli, on pouvait lire JULIUS ENDERBY.
C’était écrit en jolies lettres, gravées avec soin dans le verre ; et, juste en dessous, luisait l’inscription : COMMISSAIRE
PRINCIPAL DE POLICE DE NEW YORK.
Baley entra et dit :
— Vous m’avez fait demander, monsieur le commissaire ?
Enderby leva la tête vers son visiteur. Il portait des lunettes, car il avait les yeux trop sensibles pour que l’on pût y
adapter des lentilles normales adhérant à la pupille. Il fallait d’abord s’habituer à voir ces lunettes, pour pouvoir, ensuite,
apprécier exactement le visage de l’homme – lequel manquait tout à fait de distinction. Baley, pour sa part, inclinait fort
à penser que le commissaire tenait à ses lunettes parce qu’elles conféraient à sa physionomie plus de caractère ; quant aux
pupilles de son chef, il les soupçonnait sérieusement de ne pas être aussi sensibles qu’on le prétendait.
""")

for entity in entities:
    if entity['entity_group']=="NPP":
        print(f"Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']}")
