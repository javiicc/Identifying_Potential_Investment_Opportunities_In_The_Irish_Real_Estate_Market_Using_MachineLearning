import numpy as np
import pandas as pd


def temporal_process(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns='city_district', inplace=True)
    return df