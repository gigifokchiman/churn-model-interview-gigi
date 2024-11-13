from typing import Callable

from mimesis import Field, Schema
import pandas as pd


def give_me_some_data(random_seed):
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
        field = Field(seed=random_seed)
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
