AI-Powered 5G Open RAN OptimizerAn end-to-end artificial intelligence-driven network optimization pipeline engineered for Open Radio Access Networks (Open RAN). This platform leverages machine learning and deep learning models to process raw cellular telemetry, forecast operational workloads, identify runtime anomalies, and automatically determine resource reallocation actions for active cell interfaces.Key Subsystems & Features● Predictive Network Planning: Built on PyTorch, this neural network ingests real-world radio parameters to forecast upcoming capacity strains and structural demands.● Network Anomaly Detection: Driven by a deep TensorFlow/Keras neural network classifier to detect dropped connections, handover anomalies, and severe signal degradation instantly.● Orchestrator Pipeline Engine: A unified management framework that crosses model intelligence barriers to make live cell decisions—recommending traffic routing, PRB adjustments, or energy-saving nominal power alignments.Directory OrganizationAI-Powered-5G-OpenRAN-Optimizer/
├── data/                  # Raw, cleaned, and processed datasets
├── models/                # Trained deep learning artifacts (.pt and .h5 binaries)
├── output/                # Orchestration metrics, output recommendations, and decision logs
├── src/                   # Source directory
│   ├── data_preparation/  # Ingestion preprocessing and scale transformation layers
│   └── models/            # Sub-module model definitions and training logic
├── format_data.py         # Custom utility script for telemetry alignment
├── .gitignore             # Standard repository exclusion settings
└── README.md              # Project documentation
Installation & SetupClone the Project Foldergit clone [https://github.com/YOUR_USERNAME/AI-Powered-5G-OpenRAN-Optimizer.git](https://github.com/YOUR_USERNAME/AI-Powered-5G-OpenRAN-Optimizer.git)
cd AI-Powered-5G-OpenRAN-Optimizer
Establish a Clean EnvironmentUsing Python 3.9 is recommended to avoid deep library versioning conflicts:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Core Requirementspip install -r requirements.txt
Note: If you run into typing or initialization conflicts on Python 3.9, run pip install tensorflow==2.10.0 protobuf==3.20.3 numpy==1.22.4 to set up a robust matching environment.Execution PipelineStep-by-step instructions to run the entire optimization engine:● 1. Align Telemetry DataStandardize your custom 5G telemetry CSV (such as Kaggle files) to ensure text categories are handled safely and tracking identifiers are set up properly:python format_data.py
● 2. Execute Feature ProcessingRun the scaler pipeline to process and prepare the numerical vectors:python -m src.data_preparation.data_transformation data/5g_network_data.csv data/processed_5g_data.csv
● 3. Train the ModelsTrain the internal network intelligence components:# Train the PyTorch Predictive Planning model
python -m src.models.predictive_network_planning.train --train_data_file data/processed_5g_data.csv --valid_data_file data/processed_5g_data.csv --save_path models/predictive_network_planning/best_model.pt

# Train the Keras Anomaly Detection model
python -m src.models.network_anomaly_detection.train --data_path data/processed_5g_data.csv
● 4. Run the OrchestratorExecute the top-level orchestration controller to combine predictions and generate your optimization report sheet under the output/ folder:python -m src.main data/processed_5g_data.csv
Credits & Attribution● Core Reference Architecture: This framework is an updated implementation adapted from the open-source structure published by N00Bception/AI-Powered-5G-OpenRAN-Optimizer.● 5G Dataset: Custom model training was executed on real-world 5G cell telemetry attributes detailing signal strengths, bandwidths, drop indicators, and latency profiles.LicenseThis project is open-source and released under the Apache-2.0 License. See the LICENSE file for details.