import pandas as pd
from utils import prepare
from model import tfidf, apply_tfidf, train_model, predict_proba
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from packsize import to_family


if __name__ == "__main__":

    sp, cats = prepare()
    #print(sp, cats)

    #we train only on labelled data
    sp_labelled = sp[sp['labelled'] == True]
    #our feature is the product description (name + brand), which we will convert into numbers with tfidf
    texts = sp_labelled.text
    #the prediction is our_category_id
    y = sp_labelled.our_category_id.astype(int).map(to_family)

    #train test split
    texts_train, texts_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, stratify=y, random_state=42)

    #create vectorizers
    word_vec, char_vec = tfidf(texts_train)

    #transform tests into number features
    X_train = apply_tfidf(texts_train, word_vec, char_vec)
    X_test = apply_tfidf(texts_test, word_vec, char_vec)

    #train model on the train split
    trained_model = train_model(X_train, y_train)

    proba = predict_proba(trained_model, X_test)
    #convert probabilities into predictions, choose the class (category) on the index that had the largest probability
    pred = trained_model.classes_[proba.argmax(axis=1)]

    #find accuracy on the test split
    print("classification report: ", classification_report(y_test, pred))

    #visualize predictions
    results = pd.DataFrame({
        'Product Text': texts_test,
        'Actual Category': y_test.values,
        'Predicted Category': pred,
        'Confidence Score': proba.max(axis=1)
    })

    print(results)