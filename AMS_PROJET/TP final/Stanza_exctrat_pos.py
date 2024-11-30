import stanza

nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos')
# doc = nlp('Barack Obama was born in Hawaii.')
doc = nlp("Asseyez-vous donc, professeur.")
print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')
