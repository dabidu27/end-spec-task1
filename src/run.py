import pandas as pd
from utils import prepare, ignoreCaseSpacing
from model import tfidf, apply_tfidf, train_model, predict_proba
from aggregate import aggregate
from packsize import parse_pack_size, to_family, resolve_pack_side

if __name__ == "__main__":

    sp, cats = prepare()
    #print(sp, cats)
    sp["kg"] = sp.pack_size.map(parse_pack_size)
    #ids arrive as strings ("" = unlabelled); the pair tables use ints
    sp["our_category_id"] = sp.our_category_id.map(lambda v: int(v) if v != "" else v)

    #the model is good, train on all labeled products - no split

    #we train only on labelled data
    sp_labelled = sp[sp['labelled'] == True]
    #our feature is the product description (name + brand), which we will convert into numbers with tfidf
    texts = sp_labelled.text
    #the prediction is our_category_id
    y = sp_labelled.our_category_id.map(to_family)

    #create vectorizers
    word_vec, char_vec = tfidf(texts)

    X = apply_tfidf(texts, word_vec, char_vec)
    fully_trained_model = train_model(X, y)

    #predict probabilities for unlabelled products
    unlabelled_products = sp[sp['labelled'] == False]
    unlabelled_products_texts = unlabelled_products.text
    unlabelled_products_X = apply_tfidf(unlabelled_products_texts, word_vec, char_vec)
    proba = predict_proba(fully_trained_model, unlabelled_products_X)
    #convert probabilities into predictions
    pred = fully_trained_model.classes_[proba.argmax(axis=1)]
    pred = [resolve_pack_side(p, s) for p, s in zip(pred, unlabelled_products.kg)]
    #add predictions back in the original df for unlabelled products
    sp.loc[sp['labelled'] == False, 'our_category_id'] = pred

    #aggreagte by revenue to get 1 answer per normalized supplier category
    #aggregate expects string ids, as before
    res = aggregate(sp.assign(our_category_id=sp.our_category_id.astype(str)))


    #the final supplier_category to our_category mapping
    raw = sp.supplier_category.drop_duplicates()
    out = pd.DataFrame({
        "supplier_category": raw.values,
        "our_category_id": raw.map(ignoreCaseSpacing).map(res.our_category_id).values,
    })
    out.to_csv("outputs/mapping.csv", index=False)

    path = cats.set_index("our_category_id").path
    review = res.assign(path=res.our_category_id.map(path)).sort_values("expected_loss", ascending=False)
    review.to_csv("outputs/review_queue.csv")

    print("mapping rows:", len(out), "| groups:", len(res))
    print(review.head(5)[["path", "share", "revenue", "nlabelled", "expected_loss"]].round(2).to_string())


    