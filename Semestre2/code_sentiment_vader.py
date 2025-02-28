from flair.nn import Classifier
from flair.data import Sentence
from flair.models import SequenceTagger

# load the model
tagger = SequenceTagger.load("flair/ner-french")

# make a sentence
sentence = Sentence('Je suis moche')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence)