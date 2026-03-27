"""Deterministic data generators for each task.

All generators use seed=42 for full reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42


# ---------------------------------------------------------------------------
# Task 1 — Easy: Sales data
# ---------------------------------------------------------------------------

def generate_easy_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (broken_df, expected_df) for the easy task."""
    rng = np.random.RandomState(SEED)

    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    revenues = rng.uniform(100, 5000, size=n).round(2)
    units = rng.randint(1, 200, size=n).astype(float)
    regions = rng.choice(["North", "South", "East", "West"], size=n)

    # ---- Expected (clean) dataframe ----
    expected_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "revenue": revenues,
        "units_sold": units,
        "region": regions,
    })
    expected_df["revenue"] = expected_df["revenue"].astype("float64")
    expected_df["units_sold"] = expected_df["units_sold"].astype("float64")

    # ---- Broken dataframe ----
    broken_df = expected_df.copy()

    # Bug 1: revenue as object (strings)
    broken_df["revenue"] = broken_df["revenue"].astype(str)

    # Bug 2: date in DD-MM-YYYY format
    broken_df["date"] = pd.to_datetime(broken_df["date"]).dt.strftime("%d-%m-%Y")

    # Bug 3: NaN in units_sold (randomly zero out ~15 values)
    null_idx = rng.choice(n, size=15, replace=False)
    broken_df.loc[null_idx, "units_sold"] = np.nan

    # expected keeps units_sold at original values (not NaN)
    # The fix is to fill NaN with 0
    expected_units = expected_df["units_sold"].copy()
    expected_units.iloc[null_idx] = 0.0
    expected_df["units_sold"] = expected_units

    return broken_df, expected_df


# ---------------------------------------------------------------------------
# Task 2 — Medium: Bad join (orders ↔ customers)
# ---------------------------------------------------------------------------

def generate_medium_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (broken_df, expected_df) for the medium task.

    Two source tables: orders and customers.
    The broken join used customers.customer_code (which doesn't exist),
    the correct join is orders.customer_code == customers.customer_id.
    """
    rng = np.random.RandomState(SEED)

    n_customers = 20
    n_orders = 80

    # Customers table
    customer_ids = [f"CUST-{i:03d}" for i in range(1, n_customers + 1)]
    names = [f"Customer_{i}" for i in range(1, n_customers + 1)]
    regions = rng.choice(["North", "South", "East", "West"], size=n_customers)

    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "name": names,
        "region": regions,
    })

    # Orders table
    order_ids = [f"ORD-{i:04d}" for i in range(1, n_orders + 1)]
    customer_codes = rng.choice(customer_ids, size=n_orders)
    amounts = rng.uniform(10, 1000, size=n_orders).round(2)

    orders_df = pd.DataFrame({
        "order_id": order_ids,
        "customer_code": customer_codes,
        "amount": amounts,
    })

    # ---- Expected: correct join ----
    expected_df = orders_df.merge(
        customers_df,
        left_on="customer_code",
        right_on="customer_id",
        how="left",
    )

    # ---- Broken: join on wrong key (customer_code == customer_code, which
    #      doesn't exist in customers, producing a cross-join-like mess) ----
    # Simulate the bad join: since customers has no "customer_code" column,
    # a developer might have mistakenly added it or done a cross join.
    # We simulate the result: all nulls for name/region and possible dupes.
    broken_df = orders_df.copy()
    broken_df["customer_id"] = np.nan
    broken_df["name"] = np.nan
    broken_df["region"] = np.nan

    # Store the source tables on the function so the environment can access them
    generate_medium_data.orders_df = orders_df  # type: ignore[attr-defined]
    generate_medium_data.customers_df = customers_df  # type: ignore[attr-defined]

    return broken_df, expected_df


# ---------------------------------------------------------------------------
# Task 3 — Hard: Double exchange-rate multiplication
# ---------------------------------------------------------------------------

def generate_hard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (broken_df, expected_df) for the hard task."""
    rng = np.random.RandomState(SEED)

    n = 200
    exchange_rate = 1.23

    transaction_ids = [f"TXN-{i:05d}" for i in range(1, n + 1)]
    currencies = rng.choice(["USD", "EUR", "GBP"], size=n, p=[0.4, 0.35, 0.25])
    amounts = rng.uniform(50, 5000, size=n).round(2)

    # Correct converted_amount
    converted = []
    for amt, cur in zip(amounts, currencies):
        if cur == "USD":
            converted.append(round(amt * exchange_rate, 2))
        elif cur == "EUR":
            converted.append(round(amt * 1.0, 2))  # EUR is base
        else:  # GBP
            converted.append(round(amt * 1.45, 2))

    expected_df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "amount": amounts,
        "currency": currencies,
        "converted_amount": converted,
    })

    # ---- Broken: USD rows have exchange rate applied twice ----
    broken_converted = []
    for amt, cur in zip(amounts, currencies):
        if cur == "USD":
            broken_converted.append(round(amt * exchange_rate * exchange_rate, 2))
        elif cur == "EUR":
            broken_converted.append(round(amt * 1.0, 2))
        else:
            broken_converted.append(round(amt * 1.45, 2))

    broken_df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "amount": amounts,
        "currency": currencies,
        "converted_amount": broken_converted,
    })

    return broken_df, expected_df
