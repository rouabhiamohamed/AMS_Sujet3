from os import path
import os
from glob import glob
import sys
import re
from collections import Counter
import spacy
from collections import defaultdict


def suppr_char_missing(file_path):
    with open(file_path, "r") as file:
        content = file.readlines()
    updated_content = [line.replace("�", '') for line in content]
    with open(file_path, "w") as file:
        file.writelines(updated_content)
        
def suppr_page_num(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    updated_content = re.sub(r'�.*?�', '', content)

    with open(file_path, "w") as file:
        file.write(updated_content)

def stat_file(file_path):
    sentence = 0
    charac = 0
    words = 0
    with open(file_path, "r") as file :
          for line in file :
            charac += len(line)
            word = line.split()
            words += len(word)
            sentence += line.count('?') + line.count('!')
            if(line.count("...")!=0):
                sentence += line.count('...')
                sentence += line.count('.') - line.count('...')
            else :
                sentence += line.count('.')
                
    stats="Nombre de caractères : "+str(charac)+'\n'+"Nombre de mots : "+str(words)+'\n'+"Nombre de phrase : "+str(sentence)
    name_file_stat = 'Stats_of_'+file_path
    with open(name_file_stat, 'w') as fichier:
        fichier.write(stats)
    print(stats)

def token(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        texte = file.read()
    tokens = texte.split()
    unigrammes = Counter(tokens)
    bigrammes = Counter(zip(tokens, tokens[1:]))
    
    name_file_uni = 'Unigrammes_of_'+file_path
    with open(name_file_uni, 'w') as fichier:
        fichier.write("Unigrammes :\n")
        for mot, freq in unigrammes.items():
            fichier.write(f"{mot}: {freq} \n")
    name_file_bi = 'Bigrammes_of_'+file_path
    with open(name_file_bi, 'w') as fichier:
        fichier.write("Bigrammes :\n")
        for bigram, freq in bigrammes.items():
            fichier.write(f"{bigram}: {freq} \n")



def make_pos_file(file_path):
    print("POS en cours...")
    nlp = spacy.load("fr_core_news_sm")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        texte = file.read()
    
    doc = nlp(texte)

    resultat_pos = []
    for token in doc:
        resultat_pos.append(f"{token.text}/{token.pos_}")
        
    version_pos = " ".join(resultat_pos)
    
    name_file_pos = 'POS_of_'+file_path
    with open(name_file_pos, 'w') as fichier:
        fichier.write(version_pos)
    print("POS terminé")


def choose_file():
    file = input("Veuillez entrer un fichier texte : ")
    if not os.path.isfile(file):
        print("Erreur : Le fichier n'existe pas.")
        return

    if not file.endswith('.txt'):
        print("Erreur : Le fichier doit être un fichier texte (.txt).")
        return

    try:
        suppr_page_num(file)
        suppr_char_missing(file)
        stat_file(file)
        token(file)
        make_pos_file(file)
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

def main():
    #choose_file()

    texte = """
Hari Seldon a écrit un traité sur la psychohistoire. 
Cléon était l'empereur de l'Empire Galactique. 
Demerzel, son conseiller, l'a toujours soutenu. 
Trantor était le centre de l'Empire.
"""
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(texte)

    # Construction du dictionnaire de relations par co-occurrence
    relations = defaultdict(set)

    # Parcourir chaque phrase et vérifier les co-occurrences de personnages
    for sent in doc.sents:
        personnages = [ent.text for ent in sent.ents if ent.label_ == "PERSON"]
        for p1 in personnages:
            for p2 in personnages:
                if p1 != p2:  # Eviter les relations d'un personnage avec lui-même
                    relations[p1].add(p2)

    print("Relations détectées:", relations)
    

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            print(ent.text) 
if __name__ == "__main__":
    main()
