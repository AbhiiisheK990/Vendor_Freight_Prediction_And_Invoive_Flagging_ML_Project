import os
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = r"C:\Users\abhim\OneDrive\Documents\Vendor_Intelligence_Project\models\predict_invoice_flag.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_path = os.path.join(BASE_DIR, "models", "predict_invoice_flag.pkl")

    return joblib.load(model_path)

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices

    Parameters
    ---------
    input_data: dict

    Returns 
    ---------
    pd.DataFrame with predicted flag
    """

    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Flag'] = model.predict(input_df).round()
    return input_df


if __name__ == '__main__':
    # Example Inference run (Local testing)
    sample_data = {
        'Dollars': [18500,9000,3000,200]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)