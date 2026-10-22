import pandas as pd

#each supplier category has more products; if 1 product is predicted to our category A and another is predicted to our category B, we have to choose between category A and B, because all the products in a supplier category must be labelled as the same category in our categories 
def aggregate(df):

    #if a supplier category maps 5 products to our category A and 4 products to our category B, we could choose our category A because of the higher count
    #but, we should minimise revenue loss, so we aggregate by revenue

    #make revenue the weight
    w = df.revenue + 1.0 # +1 so zero-revenue products still count
    #colapse supplier categories into 0 (by applying normalization), and also group each supplier category with the our_categories labels - they may be 1 per supplier category or more
    #then sum revenue
    #before unstack, W looks like this
    # key      our_category_id  
    # Key_1    Electronics        15
    #         Toys                5
    # Key_2    Electronics        10
    #         Apparel            20

    #after unstack, W looks like this
    #             Apparel   Electronics   Toys
    # Key_1          0            15      5
    # Key_2         20            10      0
    W = w.groupby([df.key, df.our_category_id]).sum().unstack(fill_value=0)


    #determine the dominant category 
    out = pd.DataFrame({
        "our_category_id": W.idxmax(axis=1), #for each row, find the column with the highest revenu
        "share": W.max(axis=1) / W.sum(axis=1), #revenue share agreeing with the winner
    })

    #compute revenue grouped by key, number of labelled products, and expected loss
    g = df.groupby("key")
    out["revenue"] = g.revenue.sum()
    out["nlabelled"] = g.labelled.sum()
    out["expected_loss"] = out.revenue * (1 - out.share) #if supplier A makes 100,000 total, and 80% of it is "Electronics", the model classifies supplier A as an "Electronics" supplier; the expected loss is the remaining 20,000

    return out
