import pandas as pd

def read_csv(file_name):
    return pd.read_csv(f"data/{file_name}.csv", dtype = str, keep_default_na = False) #str + keep_default_na = false => empty cells are loaded as "" instead of NaN

#normalization -  Dairy, DAIRY, dairy and other variants with spaces are the same thing
def ignoreCaseSpacing(s):
    return "".join(s.lower().split())

def load():

    sp = read_csv("supplier_products") 
    lab = read_csv("labelled_products")

    cats = read_csv("our_categories")
    sales = read_csv("sales")

    #str to float for sale numbers
    sales["revenue"] = sales.revenue.astype(float)
    sales["units"] = sales.units.astype(float)

    return sp, lab, cats, sales

def prepare():

    sp, lab, cats, sales = load()

    #apply the same merge as in the data audit
    sp["key"]=sp.supplier_category.map(ignoreCaseSpacing)
    sp = sp.merge(sales, on="supplier_sku").merge(lab, on="supplier_sku", how="left").fillna({"our_category_id": ""})
    sp["labelled"] = sp.our_category_id != ''
    #also add a text column with product_name + brand
    #the model may benefit from using the brand, it could be a hint; a brand that only sells beer may tell the model a lot when the product name does not
    #we group product and brand into a single string in the text column, they are currently inside 2 separate columns
    sp['text'] = sp.product_name + " brand_" + sp.brand.str.replace(" ", "_")

    return sp, cats
