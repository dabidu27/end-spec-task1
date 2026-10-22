## Run

Verified with Python 3.14.0 and the pinned dependencies in `requirements.txt`.

The input data is excluded from Git through `.gitignore`, so the CSVs are not included when you clone the repository. Obtain the four CSV files supplied with the task, create a `data/` folder in the repository root, and place them there with these exact names:

```text
data/
  our_categories.csv
  supplier_products.csv
  labelled_products.csv
  sales.csv
```

The scripts do not download the data. Once the files are in place, run these commands from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p outputs
python src/audit.py
python src/validate_model.py
python src/run.py
```

`outputs/mapping.csv` is the submission: exactly the columns `supplier_category,our_category_id`. `outputs/review_queue.csv` ranks normalized supplier categories for review, with the selected category path, revenue, labelled-product count, agreement share and disagreement score (`expected_loss`).


