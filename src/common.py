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
    sales["units"] = sales.revenue.astype(float)

    return sp, lab, cats, sales
