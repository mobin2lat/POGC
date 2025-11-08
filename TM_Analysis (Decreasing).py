import pandas as pd
import numpy as np

input_csv = r"D:/Work/Python/Codes/Book2.csv"
output_csv = r"D:/Work/Python/Codes/Book2_imputed_rounded_with_nextyear_decreasing.csv"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = "{:.2f}".format

df = pd.read_csv(input_csv)

cols = ["2nd Reading", "3rd Reading", "4th Reading", "5th Reading"]

df[cols] = df[cols].apply(pd.to_numeric, errors='coerce').astype(float)

for c in cols:
    others = [x for x in cols if x != c]
    df[c] = df.apply(
        lambda r: r[c] if pd.notna(r[c]) else r[others].mean(),
        axis=1
    )

df = df.dropna(subset=cols)
df[cols] = df[cols].round(2)

def predict_next_year_if_decreasing(row, t_values=[1, 2, 3, 4], t_pred=5):
    y = row.values.astype(float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    x = np.array(t_values)[mask]
    y_valid = y[mask]
    slope, intercept = np.polyfit(x, y_valid, 1)
    # Accept prediction only if fitted slope is negative (decreasing trend)
    if slope < 0:
        y_pred = slope * t_pred + intercept
        return round(float(y_pred), 2)
    else:
        return np.nan

df["Next Year"] = df[cols].apply(lambda r: predict_next_year_if_decreasing(r), axis=1)

tag_col = "TAG" if "TAG" in df.columns else ("Tag" if "Tag" in df.columns else None)
if tag_col:
    df = df.sort_values(by=tag_col)

print("\nFull DataFrame (all columns) after imputation, rounding & Next Year (only if decreasing):")
print(df)

df.to_csv(output_csv, index=False, float_format="%.2f")

print(f"\nSaved CSV: {output_csv}")