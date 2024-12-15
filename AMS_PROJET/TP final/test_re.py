import re

texte = "Il se présenta : « Marlo Tanto, de la H.V. trantorienne. Il s'exclame je suis Trantor."
texte = re.sub(r'\.(?!\s\»|[^\s]\.)','.\n', texte)


print(texte)