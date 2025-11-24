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

## TF-IDF (Term Frequency - Inverse Document Frequency)

*Measure of word frequency in the document. Helps to capture important words based on how often they appear.*
TF = [# of times term t appears in document d] / [total # of terms in document d]

*Number of documents in which the word appears*
DF = [# of documents with word t] / [# total number of documents]

*Asses the importance of the word across the corpus*
The higher the more relevant the word. if a word appears in all docs, the score is 0!
IDF = log([total # of documents] / [# of documents with word t])

TF-IDF = TF * IDF

## N-Grams

N-grams are continuous sequences of 'n' items (typically words or characters) from a given sample of text. In natural language processing, they help capture local context and relationships between neighboring words.

- **Unigrams:** Single words (n=1).Example:Text: "great job"Unigrams: "great", "job"
- **Bigrams:** Pairs of consecutive words (n=2).
  Example:
  Text: "great job"
  Bigrams: "great job"

Bigger n, like trigrams (n=3), look at three-word sequences, e.g., "thank you much". N-grams are used for more context-aware text features in modeling.

## **Sentiment Analysis Classification Model with Naive Bayes**

    STEPS:

    1. Load train, val, test data

    2. Preprocess a data:

    1. lemmatize words using spacy
            2. lowercase
            3. remove non-alpha numeric characters
            4. join tokens with a space.
                Each document will look like: "token1 token2 token3 ..."

    3a. Perform OHE on the datasets

    3a.1 Fit a NaiveBayse model

    3b. Perform Coutn Vectorization in the dataset

    3.b.1 Fit a Naive Bayse model

    3C. Perform TF-IDF Vectorization in the dataset
        3.C.1 Fit a NaiveBayse model

    3C. Perform TF-IDF Vectorization of the 1,2,and 3 n-grams of the dataset
        3.C.1 Fit a NaiveBayse model
        3.C.2 Fit a ContVectorizer model

    4. Compare the results

### Naive Bayes Model

A Naive Bayes model is a probabilistic classifier that applies Bayes' theorem, assuming that all features (like words in a document) are conditionally independent given the class label.

#### **How Naive Bayes Works**

We want to compute the probability that a document belongs to a certain class (e.g., positive or negative sentiment).

Bayes' theorem:

```
P(class | document) = [ P(document | class) × P(class) ] / P(document)
```

- *P(class | document)*: The probability we want to find (how likely is this class, given the document).
- *P(class)*: Prior probability of the class (how likely is this class in general).
- *P(document | class)*: Likelihood of observing the document in this class.
- *P(document)*: Probability of the document (same for all classes, so we can ignore it when comparing classes).

Because *P(document)* is the same for every class, we can drop it when looking for the most probable class:

```
P(class | document) ∝ P(class) × P(document | class)
```

(Here "∝" means "proportional to")

#### **Independence Assumption**

In the "bag of words" model, we treat each word as independent given the class:

```
P(document | class) = P(w₁ | class) × P(w₂ | class) × ... × P(wₙ | class)
```

So, our classification equation becomes:

```
P(class | document) ∝ P(class) × ∏ P(wᵢ | class)
```

(where the product is over all words in the document)

#### **How do we compute P(wᵢ | class) when using TF-IDF?**

When using TF-IDF features, we don't use raw counts, but weighted values representing the importance of each word in each document.

**Steps:**

1. For each class, collect all documents belonging to that class.
2. Compute the TF-IDF value for each word in the class's documents.
3. For a word *wᵢ* and a class, sum the TF-IDF values for *wᵢ* across all documents in that class.
4. To convert these scores into probabilities:

   ```
   P(wᵢ | class) = [sum of TF-IDF of wᵢ in class] / [sum of TF-IDF values for all words in class]
   ```

This gives a probability-like value for each word given a class. In practice (since TF-IDF can include zeros), it's common to add smoothing (such as adding a small constant to the numerator and denominator) to avoid multiplying by zero.

The Naive Bayes classifier then uses these *P(wᵢ | class)* values (from TF-IDF) instead of word frequency probabilities.

**Summary:** With TF-IDF, *P(wᵢ | class)* is proportional to the total TF-IDF of *wᵢ* in all documents of a class, normalized by the sum of all TF-IDF values in that class.

# Advanced Text Preprocessing Techniques

## Part of Speech (PoS) Tagging

- label words in the sentence as noun, adverbs, pronouns, etc..

## NER

- Labeling tokens (person, organizations, etc..)

# Word Embeddings:

- **Word2Vec**

  - Continuous bag of words:
    - takes surrounding context and tries to predict the missing word. Ex: "The CEO delivered a __ argument at the meeting"
  - Skip-gram:
    - Starts with a target word and uses it to the surrounding context words.
    - More computationally expensive thatn Word2Vec
    - Ex: "The CEO delivered a compelling argument at the meeting"
      - INPUT = compelling
      - OUTPUT = delivered, a, argument
- **GloVe (Global Vectors for Word Representation)**

  - **What is it?**

    - GloVe is an unsupervised learning algorithm for obtaining word vector representations. It combines global word co-occurrence statistics from a large corpus to learn embeddings so that relationships between words are captured in their vector differences.
  - **How does it work (the mathematics)?**

    - GloVe builds a co-occurrence matrix, where X_ij is the number of times word j occurs in the context of word i.
    - The core idea is to factorize this matrix to produce word vectors w_i and context word vectors w̃_j such that:

      w_i^T w̃_j + b_i + b̃_j = log(X_ij)

      where:

      - w_i is the word vector for word i
      - w̃_j is the context word vector for word j
      - b_i, b̃_j are bias terms
      - X_ij is the number of times word j occurs in the context of word i
    - The final objective is to minimize the difference between both sides, for all nonzero X_ij, using a weighted least squares loss:

      J = Σ_(i,j=1)^V f(X_ij) (w_i^T w̃_j + b_i + b̃_j - log(X_ij))²

      - f is a weighting function that reduces the impact of rare or extremely common word pairs.
  - **Window Size:**

    - The context window size in GloVe determines how many words on either side of the target word are considered part of its context (e.g., window size 5 means up to 5 words left and right).
    - Smaller window sizes capture more syntactic (grammatical) relationships; larger windows capture more general, semantic (meaning-based) relationships.
  - **Pros:**

    - Utilizes statistical information from the entire corpus for richer word relationships.
    - Captures both fine-grained (syntactic) and broad (semantic) word similarities and analogies.
    - Efficient training, and high-quality pretrained embeddings are available.
    - Results interpret well in vector space (e.g., vector arithmetic for analogies).
  - **Cons:**

    - Requires constructing and storing a large co-occurrence matrix, which can become memory-intensive on large corpora.
    - Produces one fixed embedding per word ("static" embedding), unable to adapt to word sense depending on context.
    - Not designed for dynamic or incremental updates as new data come in.
- **fastText**

  - **What is it?**

    - fastText is an embedding technique developed by Facebook that improves upon Word2Vec by representing words as bags of character n-grams, which helps handle rare words and capture subword information.
  - **How does it work (the mathematics)?**

    - In fastText, each word is represented not only by a unique vector, but also as a combination of its character n-gram vectors.
    - Let G_w be the set of character n-grams for word w (e.g., for "where" and n = 3: <wh, whe, her, ere, re> and include the word itself as a n-gram).
    - The word vector for w, denoted as v_w, is computed as the sum (or average) of the vectors for its n-grams:

      v_w = Σ_(g ∈ G_w) z_g

      where z_g is the embedding vector for n-gram g.
    - The training objective is similar to the Skip-gram model: given a center word, predict surrounding context words using their composed vectors.
    - The loss function for a word-context pair (w, c) is typically the negative sampling loss:

      log σ(v_w^T v_c) + Σ_(i=1)^k E_(w_i ~ P_n(w)) [log σ(-v_(w_i)^T v_c)]

      where σ is the sigmoid function, v_w is the sum of n-gram vectors for w, v_c is the context word vector, k is the number of negative samples, and P_n is the negative sampling distribution.
  - **Why is this useful?**

    - fastText can generate vectors for out-of-vocabulary words based on their n-grams, enabling robust handling of rare words and morphological variations (e.g. plurals, verb tenses).
    - It captures subword information, making the embeddings sensitive to word structure and beneficial for morphologically rich languages.
