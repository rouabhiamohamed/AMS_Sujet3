from collections import defaultdict
import stanza


with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    texte = file.read()

stanza.download('fr')

nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')

# Traitement du texte avec spaCy
doc = nlp("Asseyez-vous donc, professeur.")

for ent in doc.ents:
    #if(ent.type=="PER"):
        print(f'Entités : {ent.text}, Type : {ent.type}')
