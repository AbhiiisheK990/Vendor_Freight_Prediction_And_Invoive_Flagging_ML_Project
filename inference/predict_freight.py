import os
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = r"C:\Users\abhim\OneDrive\Documents\Vendor_Intelligence_Project\models\predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "predict_freight_model.pkl")
    return joblib.load(model_path)


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices

    Parameters
    ---------
    input_data: dict

    Returns 
    ---------
    pd.DataFrame with predicted freight
    """

    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df = input_df[model.feature_names_in_]
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df


if __name__ == '__main__':
    # Example Inference run (Local testing)
    sample_data = {
        'Dollars': [18500,9000,3000,200],
        'Quantity':[6164,5555,2505,500]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)