import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd

from src.feature_engineering import LogTransformer


def test_log():
    df = pd.DataFrame({"a": [0, 1, 9]})
    transformer = LogTransformer(columns=["a"])
    result = transformer.fit_transform(df)

    expected = np.log(df["a"] + 1)
    assert np.allclose(result["a"], expected)
