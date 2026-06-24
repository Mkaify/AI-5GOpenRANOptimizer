import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models.network_anomaly_detection.model import NetworkAnomalyDetectionModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network anomaly detection model')
    parser.add_argument('--data_path', type=str, default='data/processed_5g_data.csv', help='Path to network data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()

    # Load data
    print(f"Loading dataset from: {args.data_path}")
    df = pd.read_csv(args.data_path)

    # Automatically clean string columns to extract pure features
    numeric_df = df.select_dtypes(include=[np.number])
    if 'id' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['id'])

    # Split into features (X) and target label (y)
    X = numeric_df.iloc[:, :-1].values.astype('float32')
    y = numeric_df.iloc[:, -1].values.astype('float32')

    # Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the Keras model with correct input shape configuration
    print(f"Initializing model with input shape: ({X_train.shape[1]},)")
    model_wrapper = NetworkAnomalyDetectionModel(input_shape=(X_train.shape[1],))

    # Train model
    print("Starting Keras network training sequence...")
    model_wrapper.train(
        x_train=X_train, 
        y_train=y_train, 
        x_val=X_val, 
        y_val=y_val, 
        batch_size=args.batch_size, 
        epochs=args.num_epochs
    )

    # Save trained model weights using Keras standard native format
    os.makedirs('models/network_anomaly_detection', exist_ok=True)
    model_wrapper.model.save('models/network_anomaly_detection/network_anomaly_detection.h5')
    print("Model successfully saved to models/network_anomaly_detection/network_anomaly_detection.h5")