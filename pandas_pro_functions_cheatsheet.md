# Pandas “Pro Analyst” Cheat Sheet (Syntax + Examples)

This is a **curated** set of the pandas functions/methods that professional data analysts use most often in real projects (cleaning, reshaping, joining, time series, aggregation, and feature engineering). Pandas has *hundreds* of APIs—this focuses on the high‑value ones you’ll actually reach for daily.

> **Conventions used below**
> - `df` is a `pandas.DataFrame`, `s` is a `pandas.Series`
> - Examples are minimal but runnable once you import pandas: `import pandas as pd`

---

## Table of Contents
1. [Create Data](#create-data)
2. [I/O (Read & Write)](#io-read--write)
3. [Inspect & Understand](#inspect--understand)
4. [Select & Filter](#select--filter)
5. [Missing Values](#missing-values)
6. [Duplicates](#duplicates)
7. [Types & Conversion](#types--conversion)
8. [String Ops](#string-ops)
9. [Datetime & Time Series](#datetime--time-series)
10. [Sorting & Ranking](#sorting--ranking)
11. [Transform & Feature Engineering](#transform--feature-engineering)
12. [GroupBy & Aggregation](#groupby--aggregation)
13. [Joins & Combine](#joins--combine)
14. [Reshape & Pivot](#reshape--pivot)
15. [Window Functions](#window-functions)
16. [Categoricals](#categoricals)
17. [Indexing, MultiIndex](#indexing-multiindex)
18. [Quality Checks](#quality-checks)
19. [Performance Tips](#performance-tips)

---

## Create Data

### `pd.DataFrame()`
**Syntax**
```python
pd.DataFrame(data=None, index=None, columns=None, dtype=None)
```
**Example**
```python
import pandas as pd

df = pd.DataFrame(
    {"name": ["Ava", "Ben"], "age": [28, 31], "city": ["NY", "SF"]}
)
```

### `pd.Series()`
**Syntax**
```python
pd.Series(data=None, index=None, dtype=None, name=None)
```
**Example**
```python
s = pd.Series([10, None, 30], name="score")
```

---

## I/O (Read & Write)

### `pd.read_csv()`
**Syntax**
```python
pd.read_csv(filepath_or_buffer, sep=",", header="infer", dtype=None,
            parse_dates=None, na_values=None, usecols=None)
```
**Example**
```python
df = pd.read_csv("sales.csv", parse_dates=["order_date"], na_values=["", "NA"])
```

### `DataFrame.to_csv()`
**Syntax**
```python
df.to_csv(path_or_buf, index=False)
```
**Example**
```python
df.to_csv("sales_clean.csv", index=False)
```

### `pd.read_excel()` / `DataFrame.to_excel()`
**Syntax**
```python
pd.read_excel(io, sheet_name=0)
df.to_excel(excel_writer, index=False)
```
**Example**
```python
df = pd.read_excel("input.xlsx", sheet_name="Sheet1")
df.to_excel("output.xlsx", index=False)
```

### `pd.read_parquet()` / `DataFrame.to_parquet()`
**Syntax**
```python
pd.read_parquet(path)
df.to_parquet(path, index=False)
```
**Example**
```python
df = pd.read_parquet("fact_sales.parquet")
df.to_parquet("fact_sales_clean.parquet", index=False)
```

### `pd.read_json()` / `DataFrame.to_json()`
**Syntax**
```python
pd.read_json(path_or_buf)
df.to_json(path_or_buf, orient="records")
```
**Example**
```python
df = pd.read_json("events.json")
df.to_json("events_clean.json", orient="records")
```

### `pd.read_sql()` / `DataFrame.to_sql()`
**Syntax**
```python
pd.read_sql(sql, con)
df.to_sql(name, con, if_exists="fail", index=False)
```
**Example**
```python
# con is a SQLAlchemy connection/engine
df = pd.read_sql("SELECT * FROM orders LIMIT 1000", con)
df.to_sql("orders_stage", con, if_exists="replace", index=False)
```

---

## Inspect & Understand

### `DataFrame.head()` / `tail()`
**Syntax**
```python
df.head(n=5)
df.tail(n=5)
```
**Example**
```python
df.head(10)
```

### `DataFrame.shape`
**Syntax**
```python
df.shape
```
**Example**
```python
rows, cols = df.shape
```

### `DataFrame.info()`
**Syntax**
```python
df.info()
```
**Example**
```python
df.info()
```

### `DataFrame.describe()`
**Syntax**
```python
df.describe(percentiles=None, include=None, exclude=None)
```
**Example**
```python
df.describe(include="all")
```

### `Series.value_counts()`
**Syntax**
```python
s.value_counts(normalize=False, dropna=True)
```
**Example**
```python
df["city"].value_counts(normalize=True)
```

### `DataFrame.nunique()` / `Series.nunique()`
**Syntax**
```python
df.nunique(dropna=True)
```
**Example**
```python
df.nunique()
```

---

## Select & Filter

### Column selection
**Syntax**
```python
df["col"]
df[["col1", "col2"]]
```
**Example**
```python
df[["name", "age"]]
```

### `DataFrame.loc[]` (label-based)
**Syntax**
```python
df.loc[row_selector, col_selector]
```
**Example**
```python
df.loc[df["age"] >= 30, ["name", "age"]]
```

### `DataFrame.iloc[]` (position-based)
**Syntax**
```python
df.iloc[row_positions, col_positions]
```
**Example**
```python
df.iloc[:5, :3]
```

### `DataFrame.query()`
**Syntax**
```python
df.query(expr)
```
**Example**
```python
df.query("age >= 30 and city == 'SF'")
```

### `DataFrame.assign()`
**Syntax**
```python
df.assign(**new_columns)
```
**Example**
```python
df2 = df.assign(age_bucket=lambda x: (x["age"] // 10) * 10)
```

### `DataFrame.filter()`
**Syntax**
```python
df.filter(items=None, like=None, regex=None, axis=0)
```
**Example**
```python
df.filter(regex="^rev_")
```

---

## Missing Values

### `DataFrame.isna()` / `notna()`
**Syntax**
```python
df.isna()
df.notna()
```
**Example**
```python
df[df["city"].isna()]
```

### `DataFrame.fillna()`
**Syntax**
```python
df.fillna(value=None, method=None, axis=None, inplace=False, limit=None)
```
**Example**
```python
df["age"] = df["age"].fillna(df["age"].median())
df["city"] = df["city"].fillna("Unknown")
```

### `DataFrame.dropna()`
**Syntax**
```python
df.dropna(axis=0, how="any", subset=None)
```
**Example**
```python
df.dropna(subset=["order_id", "customer_id"])
```

### `DataFrame.interpolate()`
**Syntax**
```python
df.interpolate(method="linear", limit_direction="forward")
```
**Example**
```python
df["sensor"] = df["sensor"].interpolate()
```

---

## Duplicates

### `DataFrame.duplicated()`
**Syntax**
```python
df.duplicated(subset=None, keep="first")
```
**Example**
```python
dupes = df[df.duplicated(subset=["order_id"], keep=False)]
```

### `DataFrame.drop_duplicates()`
**Syntax**
```python
df.drop_duplicates(subset=None, keep="first")
```
**Example**
```python
df = df.drop_duplicates(subset=["order_id"], keep="last")
```

---

## Types & Conversion

### `DataFrame.astype()`
**Syntax**
```python
df.astype(dtype, copy=True, errors="raise")
```
**Example**
```python
df["age"] = df["age"].astype("Int64")  # nullable integer
```

### `pd.to_numeric()`
**Syntax**
```python
pd.to_numeric(arg, errors="raise", downcast=None)
```
**Example**
```python
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
```

### `pd.to_datetime()`
**Syntax**
```python
pd.to_datetime(arg, errors="raise", format=None, utc=False)
```
**Example**
```python
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
```

### `pd.to_timedelta()`
**Syntax**
```python
pd.to_timedelta(arg, errors="raise", unit=None)
```
**Example**
```python
df["lead_time"] = pd.to_timedelta(df["lead_time_days"], unit="D")
```

---

## String Ops

> Works on `Series` with `.str`

### `Series.str.contains()`
**Syntax**
```python
s.str.contains(pat, case=True, na=False, regex=True)
```
**Example**
```python
df[df["email"].str.contains("@housecallpro.com", na=False)]
```

### `Series.str.extract()`
**Syntax**
```python
s.str.extract(pat, expand=True)
```
**Example**
```python
df[["area_code"]] = df["phone"].str.extract(r"^\((\d{3})\)")
```

### `Series.str.replace()`
**Syntax**
```python
s.str.replace(pat, repl, regex=True)
```
**Example**
```python
df["city"] = df["city"].str.replace(r"\s+", " ", regex=True).str.strip()
```

### `Series.str.split()`
**Syntax**
```python
s.str.split(pat=None, n=-1, expand=False)
```
**Example**
```python
df[["first", "last"]] = df["name"].str.split(" ", n=1, expand=True)
```

### `Series.str.lower()` / `upper()` / `strip()`
**Syntax**
```python
s.str.lower(); s.str.upper(); s.str.strip()
```
**Example**
```python
df["city"] = df["city"].str.lower().str.strip()
```

---

## Datetime & Time Series

### `.dt` accessor basics
**Syntax**
```python
df["date"].dt.year
df["date"].dt.month
df["date"].dt.day_name()
```
**Example**
```python
df["dow"] = df["order_date"].dt.day_name()
```

### `DataFrame.set_index()`
**Syntax**
```python
df.set_index(keys, drop=True, inplace=False)
```
**Example**
```python
ts = df.set_index("order_date")
```

### `DataFrame.resample()`
**Syntax**
```python
df.resample(rule).agg(...)
```
**Example**
```python
weekly = ts["amount"].resample("W").sum()
```

### `DataFrame.shift()`
**Syntax**
```python
df.shift(periods=1, freq=None)
```
**Example**
```python
ts["prev_amount"] = ts["amount"].shift(1)
```

### `DataFrame.diff()`
**Syntax**
```python
df.diff(periods=1)
```
**Example**
```python
ts["amount_change"] = ts["amount"].diff()
```

---

## Sorting & Ranking

### `DataFrame.sort_values()`
**Syntax**
```python
df.sort_values(by, ascending=True, na_position="last")
```
**Example**
```python
df.sort_values(["city", "age"], ascending=[True, False])
```

### `DataFrame.sort_index()`
**Syntax**
```python
df.sort_index(axis=0, ascending=True)
```
**Example**
```python
df.sort_index()
```

### `Series.rank()`
**Syntax**
```python
s.rank(method="average", ascending=True)
```
**Example**
```python
df["sales_rank"] = df["amount"].rank(ascending=False, method="dense")
```

---

## Transform & Feature Engineering

### `DataFrame.rename()`
**Syntax**
```python
df.rename(columns=None, index=None)
```
**Example**
```python
df = df.rename(columns={"Order Date": "order_date", "Amt": "amount"})
```

### `DataFrame.replace()`
**Syntax**
```python
df.replace(to_replace=None, value=None, regex=False)
```
**Example**
```python
df["status"] = df["status"].replace({"in progress": "in_progress", "done": "completed"})
```

### `pd.get_dummies()` (one-hot encoding)
**Syntax**
```python
pd.get_dummies(data, columns=None, drop_first=False)
```
**Example**
```python
df_encoded = pd.get_dummies(df, columns=["city"], drop_first=True)
```

### `Series.map()`
**Syntax**
```python
s.map(arg)
```
**Example**
```python
df["tier"] = df["score"].map(lambda x: "high" if x >= 80 else "low")
```

### `DataFrame.apply()` / `Series.apply()`
**Syntax**
```python
df.apply(func, axis=0)
s.apply(func)
```
**Example**
```python
df["amount_with_tax"] = df["amount"].apply(lambda x: x * 1.1)
```

### `DataFrame.pipe()` (clean pipelines)
**Syntax**
```python
df.pipe(func, *args, **kwargs)
```
**Example**
```python
def clean_city(d):
    return d.assign(city=d["city"].str.strip().str.title())

df = df.pipe(clean_city)
```

---

## GroupBy & Aggregation

### `DataFrame.groupby()`
**Syntax**
```python
df.groupby(by, as_index=True, dropna=True)
```

#### Basic aggregation with `.agg()`
**Syntax**
```python
df.groupby("col").agg({"x": "sum", "y": "mean"})
```
**Example**
```python
summary = df.groupby("city").agg(
    orders=("order_id", "nunique"),
    revenue=("amount", "sum"),
    avg_age=("age", "mean")
).reset_index()
```

#### `.transform()` (broadcast back to original rows)
**Syntax**
```python
df.groupby("col")["x"].transform(func)
```
**Example**
```python
df["city_avg_amount"] = df.groupby("city")["amount"].transform("mean")
```

#### `.size()` vs `.count()`
**Syntax**
```python
df.groupby("col").size()
df.groupby("col")["x"].count()
```
**Example**
```python
counts_all_rows = df.groupby("city").size()
counts_non_null_amount = df.groupby("city")["amount"].count()
```

---

## Joins & Combine

### `pd.merge()`
**Syntax**
```python
pd.merge(left, right, how="inner", on=None, left_on=None, right_on=None,
         suffixes=("_x","_y"), validate=None)
```
**Example**
```python
df = pd.merge(orders, customers, how="left", on="customer_id", validate="m:1")
```

### `DataFrame.join()`
**Syntax**
```python
df.join(other, on=None, how="left", lsuffix="", rsuffix="")
```
**Example**
```python
df = orders.join(customers.set_index("customer_id"), on="customer_id", how="left")
```

### `pd.concat()`
**Syntax**
```python
pd.concat(objs, axis=0, ignore_index=False)
```
**Example**
```python
df_all = pd.concat([jan, feb, mar], ignore_index=True)
```

### `DataFrame.combine_first()`
**Syntax**
```python
df1.combine_first(df2)
```
**Example**
```python
df_filled = df1.combine_first(df2)
```

---

## Reshape & Pivot

### `DataFrame.pivot()`
**Syntax**
```python
df.pivot(index=None, columns=None, values=None)
```
**Example**
```python
wide = df.pivot(index="customer_id", columns="month", values="amount")
```

### `pd.pivot_table()`
**Syntax**
```python
pd.pivot_table(df, values=None, index=None, columns=None, aggfunc="mean", fill_value=None)
```
**Example**
```python
pt = pd.pivot_table(df, values="amount", index="city", columns="month", aggfunc="sum", fill_value=0)
```

### `DataFrame.melt()`
**Syntax**
```python
df.melt(id_vars=None, value_vars=None, var_name=None, value_name="value")
```
**Example**
```python
long = wide.reset_index().melt(id_vars="customer_id", var_name="month", value_name="amount")
```

### `DataFrame.stack()` / `unstack()`
**Syntax**
```python
df.stack(level=-1)
df.unstack(level=-1)
```
**Example**
```python
stacked = wide.stack()
unstacked = stacked.unstack()
```

### `DataFrame.explode()`
**Syntax**
```python
df.explode(column, ignore_index=False)
```
**Example**
```python
df = pd.DataFrame({"id":[1,2], "tags":[["a","b"], ["c"]]})
df_ex = df.explode("tags", ignore_index=True)
```

---

## Window Functions

### `Series.rolling()`
**Syntax**
```python
s.rolling(window, min_periods=None).agg(func)
```
**Example**
```python
ts["7d_avg"] = ts["amount"].rolling(7, min_periods=1).mean()
```

### `Series.expanding()`
**Syntax**
```python
s.expanding(min_periods=1).agg(func)
```
**Example**
```python
ts["cum_avg"] = ts["amount"].expanding().mean()
```

### `Series.ewm()`
**Syntax**
```python
s.ewm(span=None, alpha=None, adjust=True).mean()
```
**Example**
```python
ts["ewm_14"] = ts["amount"].ewm(span=14).mean()
```

---

## Categoricals

### `astype("category")`
**Syntax**
```python
df["col"] = df["col"].astype("category")
```
**Example**
```python
df["city"] = df["city"].astype("category")
```

### `CategoricalDtype` (ordered categories)
**Syntax**
```python
from pandas.api.types import CategoricalDtype
dtype = CategoricalDtype(categories=[...], ordered=True)
df["col"] = df["col"].astype(dtype)
```
**Example**
```python
from pandas.api.types import CategoricalDtype
size_type = CategoricalDtype(categories=["S","M","L","XL"], ordered=True)
df["shirt_size"] = df["shirt_size"].astype(size_type)
```

---

## Indexing, MultiIndex

### `set_index()` / `reset_index()`
**Syntax**
```python
df.set_index(keys)
df.reset_index(drop=False)
```
**Example**
```python
df2 = df.set_index(["city", "customer_id"])
df3 = df2.reset_index()
```

### `DataFrame.xs()` (cross-section)
**Syntax**
```python
df.xs(key, level=None, axis=0)
```
**Example**
```python
sf_only = df2.xs("SF", level="city")
```

---

## Quality Checks

### Null counts
```python
nulls = df.isna().sum().sort_values(ascending=False)
```

### Uniqueness checks
```python
assert df["order_id"].is_unique
```

### Range checks
```python
bad_age = df.loc[(df["age"] < 0) | (df["age"] > 120)]
```

### Outlier quick view (IQR)
```python
q1 = df["amount"].quantile(0.25)
q3 = df["amount"].quantile(0.75)
iqr = q3 - q1
outliers = df[(df["amount"] < q1 - 1.5*iqr) | (df["amount"] > q3 + 1.5*iqr)]
```

---

## Performance Tips

### Use vectorized ops instead of loops
```python
df["amount_with_tax"] = df["amount"] * 1.1
```

### Prefer `.loc` for conditional assignment
```python
df.loc[df["amount"] < 0, "amount"] = 0
```

### Use `category` for low-cardinality strings
```python
df["city"] = df["city"].astype("category")
```

### Use `copy()` carefully
```python
df2 = df[["a","b"]].copy()
```

---

## Mini “End-to-End” Example

```python
import pandas as pd

df = (pd.read_csv("orders.csv", parse_dates=["order_date"])
        .rename(columns={"Order Date": "order_date", "Amt": "amount"})
        .drop_duplicates(subset=["order_id"], keep="last")
        .assign(
            amount=lambda x: pd.to_numeric(x["amount"], errors="coerce"),
            city=lambda x: x["city"].fillna("Unknown").str.strip().str.title(),
            month=lambda x: x["order_date"].dt.to_period("M").astype(str),
        )
     )

summary = (df.groupby(["city", "month"])
             .agg(orders=("order_id", "nunique"), revenue=("amount", "sum"))
             .reset_index()
             .sort_values(["city","month"])
          )

summary.to_csv("city_month_summary.csv", index=False)
```
