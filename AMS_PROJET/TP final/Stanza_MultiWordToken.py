import stanza

nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt')

with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    texte = file.read()
    
doc = nlp(texte)
 
# doc = nlp("Asseyez vous s'il vous plait !")
for sentences in doc.sentences:
    for token in sentences.tokens:
        print(f'token: {token.text}\twords: {", ".join([word.text for word in token.words])}')
