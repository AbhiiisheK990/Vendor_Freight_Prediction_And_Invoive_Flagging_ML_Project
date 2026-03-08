from data_preprocessing import load_invoice_data, apply_label,split_data,scale_features
from model_evaluation import train_random_forest,evaluate_classifier
import joblib


FEATURES = ['invoice_quantity',
 'invoice_dollars',
 'Freight',
 'total_item_quantity',
 'total_item_dollars',
 ]

TARGET = 'flag_invoice'

def main():
    # Load Data
    df = load_invoice_data()
    df = apply_label(df)

    # Prepare Data
    X_train,X_test, Y_train, Y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(X_train,X_test,'models/scaler.pkl')


    # Train and evaluate model
    grid_search = train_random_forest(X_train_scaled, Y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,Y_test,
        'Rnadom Forest Classifier'
    )


    # Save best model
    joblib.dump(grid_search.best_estimator_,'models/predict_flag_invoice.pkl')

if __name__ == '__main__':
    main()
