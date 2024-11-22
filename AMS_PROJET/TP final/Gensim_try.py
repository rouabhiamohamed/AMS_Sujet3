from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess

# Step 1: Read and preprocess the text data
with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    documents = file.readlines()

# Preprocess the documents: tokenization, lowercasing, removing stopwords, etc.
processed_docs = [simple_preprocess(doc) for doc in documents]

# Step 2: Create a Dictionary from the preprocessed documents
dictionary = corpora.Dictionary(processed_docs)

# Step 3: Convert the documents to a Bag-of-Words corpus
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Step 4: Train LSI Model on the corpus
lsi = models.LsiModel(corpus, num_topics=200)

# Step 5: Prepare another corpus (if needed) and convert it to the LSI space
# For demonstration, we assume `another_corpus` is a similar set of documents to be indexed.
another_corpus = []  # Replace with your actual data, similar to how you processed `corpus`
another_bow = [dictionary.doc2bow(simple_preprocess(doc)) for doc in another_corpus]
lsi_corpus = lsi[another_bow]

# Step 6: Create an index to perform similarity queries
index = similarities.MatrixSimilarity(lsi[corpus])  # Index the LSI-transformed corpus

# Step 7: Query for similarity
query = "Seldon"
query_bow = dictionary.doc2bow(simple_preprocess(query))
query_lsi = lsi[query_bow]  # Transform the query into LSI space

# Compute similarity of the query against the indexed documents
sims = index[query_lsi]

# Print the similarity results
print(list(enumerate(sims)))  # List of (doc_id, similarity_score)
