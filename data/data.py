# %%
from pathlib import Path

import pandas as pd


# %%
def load_data() -> pd.DataFrame:
    """
    Load and merge crime datasets:
    - violent.csv
    - unemployment.csv
    - urban.csv
    """

    # 1. Load datasets
    base_path = Path(__file__).resolve().parent
    violent = pd.read_csv(base_path / "violent.csv")
    unemploy = pd.read_csv(base_path / "unemployment.csv")
    urban = pd.read_csv(base_path / "urban.csv")

    violent = violent.rename(columns={"County": "county_name", "Year": "year"})

    # 2. Merge datasets
    df = violent.merge(
        unemploy[
            [
                "county_name",
                "year",
                "Civilian_labor_force",
                "Employed",
                "Unemployed",
                "Unemployment_rate",
            ]
        ],
        on=["county_name", "year"],
        how="left",
    )

    df = df.merge(
        urban[
            ["county_name", "year", "Urban_Influence_Code", "Metro", "Traffic_Count"]
        ],
        on=["county_name", "year"],
        how="left",
    )
    df = df[(df["year"] >= 2000) & (df["year"] <= 2020)].reset_index(drop=True)
    df["violent_crime_rate"] = (df["Violent Count"] / df["Population"]) * 100000
    return df


# %%
