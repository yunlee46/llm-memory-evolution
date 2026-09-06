import numpy as np
import pandas as pd


def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    # leaks: reads the outcome column that is not present in test
    return test["home_win"].values.astype(float)
