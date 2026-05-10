import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def check_missing_value(df):
    return pd.DataFrame(
        {
        "count": df.isna().sum(),
        "percentage": np.round(df.isna().sum() / len(df), 2) * 100
     }
)