# %%

import sys
print(sys.path)
from typing import Callable

from icecream import ic
from mimesis import Field, Schema
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

# %%
SEED = 31459


# %%
def give_me_some_data():
    data_fields: dict[str, Callable] = {
        "industry_category": (
            lambda random, **kwargs: random.choice(
                ("green", "cleaning", "contracting", "construction", "other")
            )
        ),
        "total_users_count": (lambda random, **kwargs: random.randint(1, 150)),
        "active_users_count": (lambda random, **kwargs: random.randint(1, 150)),
        "login_count_p1m": (lambda random, **kwargs: random.randint(1, 1500)),
        "operating_country": (
            lambda random, **kwargs: random.choice(
                ("au", "ca", "de", "fr", "in", "ir", "mx", "nz", "sa", "sp", "uk", "us")
            )
        ),
        "invoices_p1m_vs_prev_month_ratio": (
            lambda random, **kwargs: random.uniform(-1, 3, 4)
        ),
        "p1m_tenure_since_conversion": (lambda random, **kwargs: random.randint(0, 72)),
        "payment_methods_active_count": (lambda random, **kwargs: random.randint(0, 5)),
        "payments_p1m_vs_prev_month_ratio": (
            lambda random, **kwargs: random.uniform(-1, 2, 4)
        ),
        "quotes_p1m_vs_prev_month_ratio": (
            lambda random, **kwargs: random.uniform(-1, 3, 4)
        ),
        "visits_p1m_vs_prev_month_ratio": (
            lambda random, **kwargs: random.uniform(-1, 3, 4)
        ),
        "revenue_trend": (
            lambda random, **kwargs: random.choice(("up", "flat", "down"))
        ),
        "is_churned": (lambda random, **kwargs: random.choice((True, False))),
    }

    def training_data():
        field = Field(seed=SEED)
        for field_name, field_fn in data_fields.items():
            field.register_handler(field_name, field_fn)

        def training_schema():

            while True:
                row = {
                    "account_id": field("uuid"),
                    **{field_name: field(field_name) for field_name in data_fields},
                }
                # Make sure the data conforms to certain rules...
                if row["active_users_count"] > row["total_users_count"]:
                    continue
                return row

        return training_schema

    schema = Schema(schema=training_data(), iterations=30000)
    data_df = pd.DataFrame(data=schema.create())
    return data_df


# %%
def prepare_data(df: pd.DataFrame):
    omit_columns = ["account_id"]
    str_columns = ["industry_category", "operating_country", "revenue_trend"]
    label_column = "is_churned"

    df = pd.get_dummies(df, columns=str_columns)
    training_columns = [
        str(i) for i in df.columns if i not in {label_column, *omit_columns}
    ]
    df[training_columns] = df[training_columns].fillna(df[training_columns].median())
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=SEED)
    return (
        train_df[training_columns],
        train_df[label_column],
        train_df[omit_columns],
        test_df[training_columns],
        test_df[label_column],
        test_df[omit_columns],
    )


# %%
def score(baseline, predictions):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "Accuracy": float(accuracy_score(baseline, predictions)),
        "Precision": float(precision_score(baseline, predictions)),
        "Recall": float(recall_score(baseline, predictions)),
        "F1 Score": float(f1_score(baseline, predictions)),
        "ROC AUC Score": (
            float(roc_auc_score(baseline, predictions))
            if len(baseline.unique()) > 1
            else -0.0
        ),
    }


# %%
def train(df, lbls):
    cfi = XGBClassifier(random_state=SEED)

    return cfi.fit(df, lbls)


# %%
df = give_me_some_data()
df, labels, ids, tst_df, tst_labels, tst_ids = prepare_data(df)
model = train(df, labels)

threshold = 0.5
df["prob"] = model.predict_proba(df)[:, 1]
df["pred"] = df["prob"] > threshold
tst_df["prob"] = model.predict_proba(tst_df)[:, 1]
tst_df["pred"] = tst_df["prob"] > threshold
ic(score(labels, df["pred"]))
ic(score(tst_labels, tst_df["pred"]))