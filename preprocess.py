import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string

#Text Preprocessing function
tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')

def clean_text(doc):
    # split into tokens by white space
    tokens = tokenizer.tokenize(doc)
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return " ".join(tokens)
