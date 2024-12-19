import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize')

with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    texte = file.read()

# doc = nlp('This is a test sentence for stanza. This is another sentence.')
doc = nlp(texte)
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
