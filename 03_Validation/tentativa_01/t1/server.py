import pandas as pd
import requests

from .config import DEV_KEY, URL


def send_predictions(y_app_pred):
    data = {
        "dev_key": DEV_KEY,
        "predictions": pd.Series(y_app_pred).to_json(orient="values"),
    }
    response = requests.post(url=URL, data=data, timeout=120)
    return response.text
