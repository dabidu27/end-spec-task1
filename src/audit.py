from utils import load, ignoreCaseSpacing

sp, lab, cats, sales = load()

#print(sp)

#supplier_products - 1 row per product - supplier category, name, brand, packsize
#sales - units and revenue for each product
#labelled_products - id of the category our team puts the product in
#our_categories - id to category path

#Integrity check
print("duplicate SKUs: ", sp.supplier_sku.duplicated().sum(), lab.supplier_sku.duplicated().sum())
print("SKUs missing from sales: ", (~sp.supplier_sku.isin(sales.supplier_sku)).sum())
print("label ids not in tree:", (~lab.our_category_id.isin(cats.our_category_id)).sum())
print("blank categories:", (sp.supplier_category.str.strip() == "").sum())

#Tree
paths = cats.path.tolist()
parents = {p for p in paths if any(q.startswith(f"{p} >") for q in paths)}
print("parent_categories: ", len(parents))

#Category variants - to decide normalization rules
raw = sp.supplier_category.unique()
print("raw:", len(raw), "| after lowercase + strip all spaces:", len({ignoreCaseSpacing(r) for r in raw}))

#Supplier categories with no hand_labelled product and their revenue
sp["key"] = sp.supplier_category.map(ignoreCaseSpacing) #add a column with the normalized key
#merge 3 tables - supplier_producuts with sales => adds units and revenue, then with labelled_products => adds our_category_id - hand-placed labeled
#categories without a mapping get NaN which we fill with empty strings
sp = sp.merge(sales, on="supplier_sku").merge(lab, on="supplier_sku", how="left").fillna({'our_category_id': ""})
sp["labelled"] = sp.our_category_id != '' #boolean column, shows if the supplier category has a mapping in ours
#collapse to 1 row per normalized category, with 2 numbers - revenue it adds and the number of labelled products it has
G = sp.groupby("key").agg(rev=("revenue", "sum"), nlabelled=("labelled", "sum"))
#compute how much of the total revenues do groups with no labelled products make up
print("groups:", len(G), "| with 0 labels:", (G.nlabelled == 0).sum(),
      "| their revenue share: %.1f%%" % (100 * G.rev[G.nlabelled == 0].sum() / G.rev.sum()))

#Label purity - do labels inside 1 category disagree?
#The products in 1 supplier category should technically point to the same category of ours
#We compute how many don't point to the same category of ours
x = sp[sp.labelled].groupby("key").our_category_id.agg(["size", "nunique"]) #group labelled products by normalized key and compute how many they are and how many unique mappings to our categories we get
#ideally, each supplier category has 1 unique mapping to our categories
#find those that don't
x = x[x["size"] >= 2]
print("groups with 2+ labels:", len(x), "| labels disagree:", (x["nunique"] > 1).sum())