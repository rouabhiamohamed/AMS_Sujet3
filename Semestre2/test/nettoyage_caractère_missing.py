
with open(f"texte.txt", "w") as file:
    file.write(texte)
    content = content.replace("�", '')
    content = re.sub(r'�\d+�', '', content)# Supprimer les numéros de page
    content = re.sub(r'�.*?�', '', content)  