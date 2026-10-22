from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression

def tfidf(texts):

    #Apply a TFidf vectorizer on the product name + brand texts to convert them into a row of numbers
    
    #we will create 2 vectorizers: 1 on groups of 1 or 2 words (n-gram with n = 2)
                                #1 on groups of characters, to account for mistakes and typos
    #apply sublinear scaling, which replaces raw word counts with a logarithmic function - it keeps the importance of a word that appears more than another higher, but not as high as with raw word counts
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2) #min_df = 2 removes groups of chars that appear less than 2 times

    #apply the vectorizer on the text (fit to learn)
    word_vectorizer.fit(texts)
    char_vectorizer.fit(texts)

    return word_vectorizer, char_vectorizer

def apply_tfidf(texts, word_vectorizer, char_vectorizer):

    #get weights for the words of each text
    word_features = word_vectorizer.transform(texts)
    #get weights for the groups of characters of each text
    char_features = char_vectorizer.transform(texts)

    #combine them horizontally - now each row contains both the word weights and char weights for each product
    combined_features = hstack([word_features, char_features])
    #each weights vector contains the weight (importance) of each word in our whole dataset - not only the ones present. if a data set has 50.000 words and a product description has only 10, the vector will have 50000 elements. only 10 of those elements will be relevant, the rest will be 0 - so, to save space and computation time, we get rid of the 0s by transforming the matrix into a sparsed one
    final_matrix = combined_features.tocsr()

    return final_matrix

def train_model(X, y):

    logistic_regression = LogisticRegression(C=10, max_iter=2000)
    logistic_regression.fit(X, y)
    return logistic_regression

def predict_proba(trained_model, X):
    return trained_model.predict_proba(X)