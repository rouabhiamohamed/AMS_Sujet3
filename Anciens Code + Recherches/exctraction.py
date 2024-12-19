import re

def extraction_names(file_path):
    names = []
    
    # Ouvrir le fichier en lecture
    with open(file_path, "r") as file:
        content = file.read()

    # Séparer le contenu en mots
    words = content.split()

    # Parcourir chaque mot et vérifier si le caractère "/NOUN" est présent
    for word in words:
        if "/PROPN" in word and word[0].isupper(): #"/NOUN" in word à mettre peut-être
            word=re.sub(r'/\w+', '', word)
            if(len(word) > 1 and word[1].isalpha()):
                names.append(word)

    # Ouvrir un fichier de sortie et y écrire les noms extraits
    with open("Names_list.txt", "w") as file:
        for name in names:
            file.write(name + '\n')

def main():
    extraction_names("POS_of_chapter1.txt")

if __name__ == "__main__":
    main()