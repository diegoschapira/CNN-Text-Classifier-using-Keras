# Create word2vec custom embeddings using gensim

import pandas as pd
import numpy as np

import pickle
import gensim
from gensim.models import Word2Vec

# Input data (sentences) is a list of docs (best to preprocess first)

model = Word2Vec(sentences, size=300, min_count=5)

embeddings_index = {}
for i in range(len(model.wv.vocab)):
    word = list(model.wv.vocab)[i]
    coefs = model[word]
    embeddings_index[word] = coefs

print("Vocabulary size: {}".format(len(list(model.wv.vocab))))
print(len(embeddings_index))
print(model['you'][:3])

# Test model
model.wv.most_similar(positive=['vomit'],topn=20)

# Save model and embeddings index
pickle.dump(embeddings_index, open("embeddings_300d.pkl", "wb"))
model.save('word2vec-300d')

# Load model and embeddings index
embeddings_index = pickle.load(open("embeddings_300d.pkl", "rb"))
model = gensim.models.Word2Vec.load('word2vec-300d')
