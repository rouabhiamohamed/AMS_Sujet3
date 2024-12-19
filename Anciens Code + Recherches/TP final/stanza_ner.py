import stanza

nlp = stanza.Pipeline(lang='fr', processors='tokenize,ner')
doc = nlp("Entun.")
print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
