import argparse
import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Import models directly to bypass broken config dependency links
from src.models.predictive_network_planning.model import PredictiveNetwork
from src.models.network_anomaly_detection.model import NetworkAnomalyDetectionModel

def main():
    parser = argparse.ArgumentParser(description="Run the 5G OpenRAN System Orchestrator Pipeline.")
    parser.add_argument("data_file", type=str, help="Path to the processed data file.")
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing 5G OpenRAN Orchestrator...")
    
    # 1. Load data
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found at {args.data_file}")
        return
        
    df = pd.read_csv(args.data_file)
    numeric_df = df.select_dtypes(include=[np.number])
    if 'id' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['id'])
        
    features = numeric_df.iloc[:, :-1].values.astype(np.float32)

    # 2. Execute Predictive Network Planning Subsystem Inference
    print("Executing Predictive Network Planning inference pass...")
    input_dim = features.shape[1]
    pred_model = PredictiveNetwork(input_size=input_dim, hidden_size=64, output_size=1)
    
    model_path = 'models/predictive_network_planning/best_model.pt'
    if os.path.exists(model_path):
        pred_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        pred_model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(features)
            traffic_predictions = pred_model(input_tensor).numpy().flatten()
    else:
        print("Warning: Predictive model weights not found. Generating default projections...")
        traffic_predictions = np.zeros(len(df))

    # 3. Execute Network Anomaly Detection Subsystem Inference
    print("Executing Network Anomaly Detection inference pass...")
    anomaly_model_path = 'models/network_anomaly_detection/network_anomaly_detection.h5'
    if os.path.exists(anomaly_model_path):
        # Dynamically import keras to load the saved model architecture
        from tensorflow.keras.models import load_model
        anomaly_model = load_model(anomaly_model_path)
        anomaly_scores = anomaly_model.predict(features).flatten()
        anomaly_predictions = (anomaly_scores > 0.5).astype(int)
    else:
        print("Warning: Anomaly detection weights not found. Generating baseline clearances...")
        anomaly_predictions = np.zeros(len(df), dtype=int)

    # 4. Consolidate results and apply dynamic optimization metrics
    print("Consolidating live system metrics and resource adjustments...")
    output_df = df.copy()
    output_df['Predicted_Traffic_Load'] = traffic_predictions
    output_df['Anomaly_Detected'] = anomaly_predictions
    
    # Dynamic rule calculation for radio resource cell optimization
    output_df['Resource_Optimization_Action'] = np.where(
        output_df['Anomaly_Detected'] == 1, "Reroute Traffic & Mitigate",
        np.where(output_df['Predicted_Traffic_Load'] > 0.7, "Allocate Additional PRBs", "Maintain Nominal Power")
    )

    # 5. Save final outputs
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_path = f"output/optimization_metrics_{timestamp}.csv"
    output_df.to_csv(output_path, index=False)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Optimization framework completed successfully!")
    print(f"Results log sheet generated at: {output_path}")

if __name__ == "__main__":
    main()