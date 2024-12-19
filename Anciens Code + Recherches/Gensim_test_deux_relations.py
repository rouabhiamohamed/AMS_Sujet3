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

query_hari = "Seldon"
query_bow_hari = dictionary.doc2bow(simple_preprocess(query_hari))
query_lsi_hari = lsi[query_bow_hari]
sims_hari = index[query_lsi_hari]

query_gaal = "Cléon"
query_bow_gaal = dictionary.doc2bow(simple_preprocess(query_gaal))
query_lsi_gaal = lsi[query_bow_gaal]
sims_gaal = index[query_lsi_gaal]

# Comparer les résultats
for doc_id, sim_hari in enumerate(sims_hari):
    sim_gaal = sims_gaal[doc_id]
    if sim_hari > 0.09 and sim_gaal > 0.09:
        print(f"Document {doc_id} : {query_hari} et {query_gaal} ont une relation dans ce passage.")
