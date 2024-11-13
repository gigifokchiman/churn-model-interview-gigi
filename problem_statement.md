## Requirement Collection or Problem Statement

#### Churn Risk Model

The Sales Retention team would like to use a Churn Risk model built by our Data Science department to find customers that are at risk of ending their subscription (churn).  
We need your help to provide a proof-of-concept system that can train the model and make the Churn Risk score available to our Sales system.

## Requirements

The purpose of this project is to act as the focal point of a technical discussion, to help us get to know you through your code, implementation and design choices. 
Please be prepared to demo your project during the interview.  
We are evaluating this project from the lens that this is your best possible code.

As an MLOps Engineer, you will be creating a series of model lifecycle pipelines that supports the production release of the Data Science department's Churn Risk model. 
A sample of the model is provided for you at the bottom of this document so that you can focus your efforts on managing the model lifecycle.

Here are the requirements that your project must implement:

* Model training automation  
  * The model training automation can train the provided sample Churn Risk model  
  * Retraining is triggered when there are changes to the model or the source dataset  
  * The model output is a deployable artifact  
* Model deployment  
  * Deployment is in a containerized environment  
  * Automate the model deployment  
* Model batch inference  
  * Batch inference is automated and inference results are persisted  
  * Provide a report of model health metrics  
* Provide tests where necessary  
* Expect that the Jobber interviewers will want to deploy your solution in a sandbox environment for testing purposes

Deliverables

* A Git repository with all code and configuration required to execute your project  
* Provide documentation that provides an overview of your project and basic onboarding instructions  
* Anything else you might consider helpful in understanding your project

Tips

* Use tools, languages and frameworks that you are familiar with  
* For each stage of the model lifecycle, automation could be a simple script that simulates the task  
* Optimization of the model is not required

### Dependencies

```
asttokens==2.4.1
colorama==0.4.6
executing==2.1.0
icecream==2.1.3
joblib==1.4.2
mimesis==18.0.0
numpy==2.1.2
nvidia-nccl-cu12==2.23.4
pandas==2.2.3
pygments==2.18.0
python-dateutil==2.9.0.post0
pytz==2024.2
scikit-learn==1.5.2
scipy==1.14.1
six==1.16.0
threadpoolctl==3.5.0
tzdata==2024.2
xgboost==2.1.1
```

### Python Code

```py
# %%
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
```
