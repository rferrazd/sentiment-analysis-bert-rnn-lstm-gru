

Approaches to building text classification models:

- Bag of words
- Sequence


# Bag of words

- disregards grammar and sequence
- The idea is to represent a text/document with a wordset that represents it


### Preprocessing Approaches:

- **Stemming:** reducing words to its morphological base
- **Lemming:** reducing words to its base. Slight difference from stemming is that it uses the dictionary and morphological (structure and formation of words, including how words are built from smaller units called morphenes) analyzer to identify lemma
  - NLTK's WordNetLemmatizer() relies on a default part of speech to choose the lemma which is usually a noun, this can lead to words not getting reduced. Better to use SPACY!
  - Unlike Stemming, Lemmitizating considers the role of a word in a sentence to accurately convert it to its base form
- **Building a Vocabulary:**
  - Save your lemmatized tokens (use spacy) as a set. Note that in the jupyter notebook the vocab is built over the corpus which punctuations and stopwords were removed, and all words was lowercased.
  - Overall steps so far for building the vocab:
    - tokenize corpus into words
    - remove punctuation and stopwords
    - lemmatize using spacy
- **Vectorization:**
  - FOUR MAIN TECHNIQUES:
    - One Hot Encoding (OHE)
      - 1 = word in the document
      - 0 = word NOT in the document
      - Vector dimension = (numb_of_docs x vocab_size). So rows are the docs, columns are the words in vocab
    - Count Vectorizer
      - transforms the vocabulary into a matrix of token counts and it count number of times each word appears in the document.
      - Like OHE, but not binary.
    - TF-IDF
    - Word Embedding
  -
