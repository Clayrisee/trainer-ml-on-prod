# yaml

import yaml
import argparse
from dataloader import load_pickle
from utils import convert_params
# framework ml yang kita gunakan utk ngetrain model
from sklearn.metrics import classification_report # Evaluatino metrics
from sklearn.ensemble import RandomForestClassifier # Model RF
from sklearn.svm import SVC # Support Vector Machine Model
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.ensemble import GradientBoostingClassifier
# Additional package
import neptune #pip install neptune -> Experiment TRacking
import joblib # ngedump model
from dotenv import load_dotenv #pip install python-dotenv
import os
from pathlib import Path
from datetime import datetime
import json
from loguru import logger  # pip install loguru
import sys


def setup_experiment_folder(run_id):
    """Create experiment folder structure based on Neptune run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = Path(f"experiments/{timestamp}_{run_id}")
    experiment_folder.mkdir(parents=True, exist_ok=True)

    # Create subfolders
    (experiment_folder / "models").mkdir(exist_ok=True)
    (experiment_folder / "reports").mkdir(exist_ok=True)
    (experiment_folder / "logs").mkdir(exist_ok=True)

    return experiment_folder


def setup_loguru(experiment_folder):
    """Setup Loguru logger with file and console output."""
    # Remove default logger
    logger.remove()

    # Add console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )

    # Add file handler for general logs
    logger.add(
        experiment_folder / "logs" / "experiment.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="INFO",
        rotation="50 MB"
    )

    return logger


def add_model_log_file(experiment_folder, model_name):
    """Add a separate log file for a specific model."""
    log_file = experiment_folder / "logs" / f"{model_name}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        filter=lambda record: record["extra"].get("model") == model_name
    )
    return logger.bind(model=model_name)


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train ML models with configurable parameters')
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help='Path to configuration YAML file (default: configs/config.yaml)')
args = parser.parse_args()

logger.info("="*80)
logger.info("Starting ML Training Pipeline")
logger.info("="*80)

# Load environment variables
logger.info("Loading environment variables...")
load_dotenv()

# Neptune setup
project_name = os.getenv('NEPTUNE_PROJECT_NAME')
api_key = os.getenv('NEPTUNE_API_TOKEN')
logger.info(f"Neptune Project: {project_name}")

# Initialize Neptune
logger.info("Initializing Neptune run...")
run = neptune.init_run(project=project_name, api_token=api_key)
run_id = run["sys/id"].fetch()
logger.success(f"Neptune Run ID: {run_id}")

# Setup experiment folder
logger.info("Setting up experiment folder structure...")
experiment_folder = setup_experiment_folder(run_id)
logger.success(f"Experiment folder created: {experiment_folder}")

# Setup Loguru logging
setup_loguru(experiment_folder)
logger.info("Loguru logging configured")

# Load Dataset
logger.info("Loading training and test datasets...")
train_data = load_pickle("train_data.pickle")
test_data = load_pickle("test_data.pickle")
logger.success("Datasets loaded successfully")

train_x = train_data['x']
train_y = train_data['y']
test_x = test_data['x']
test_y = test_data['y']

# Log dataset information
logger.info("Dataset Information:")
logger.info(f"  - Training samples: {train_x.shape[0]}")
logger.info(f"  - Test samples: {test_x.shape[0]}")
logger.info(f"  - Number of features: {train_x.shape[1]}")
logger.info(f"  - Training labels distribution: {dict(zip(*sorted(zip(train_y, [list(train_y).count(i) for i in set(train_y)]))))}")
logger.info(f"  - Test labels distribution: {dict(zip(*sorted(zip(test_y, [list(test_y).count(i) for i in set(test_y)]))))}")

# Log sample data
logger.info("Sample Training Data (first 3 samples):")
for i in range(min(3, train_x.shape[0])):
    logger.info(f"  Sample {i+1}: Features shape={train_x[i].shape if hasattr(train_x[i], 'shape') else len(train_x[i])}, Label={train_y[i]}")

# Read Config
logger.info(f"Reading configuration from: {args.config}")
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
logger.success("Configuration loaded successfully")

model_version = config["model_versions"]  # Get model version from YAML
logger.info(f"Model Version: {model_version}")
logger.info(f"Number of models to train: {len(config['methods'])}")

# Log experiment metadata
logger.info("Logging experiment metadata to Neptune...")
run["experiment/run_id"] = run_id
run["experiment/folder"] = str(experiment_folder)
run["experiment/config_file"] = args.config
run["experiment/model_version"] = model_version
run["experiment/timestamp"] = datetime.now().isoformat()
logger.success("Experiment metadata logged to Neptune")

# Save config to experiment folder
logger.info("Saving configuration to experiment folder...")
config_path = experiment_folder / "config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
logger.success(f"Configuration saved: {config_path}")

# Dictionary to map method names to sklearn models
model_mapping = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
    "Gradient Boosting Classifier": GradientBoostingClassifier
}

# Create experiment summary
experiment_summary = {
    "run_id": run_id,
    "timestamp": datetime.now().isoformat(),
    "model_version": model_version,
    "config_file": args.config,
    "models": []
}

logger.info("\n" + "="*80)
logger.info("Starting Model Training Loop")
logger.info("="*80 + "\n")

# Iterate through all methods in config
for idx, method in enumerate(config["methods"], 1):
    model_name = method["name"]
    model_config = convert_params(method["config"])

    # Add model-specific log file
    model_logger = add_model_log_file(experiment_folder, model_name.replace(" ", "_"))

    logger.info("="*80)
    logger.info(f"[{idx}/{len(config['methods'])}] Training: {model_name}")
    logger.info("="*80)

    model_logger.info(f"Model Version: {model_version}")
    model_logger.info(f"Model Configuration: {json.dumps(model_config, indent=2)}")

    # Model namespace including version
    model_namespace = f"models/{model_name}/{model_version}"

    # Log model parameters to Neptune
    logger.info("Logging model parameters to Neptune...")
    for param_name, param_value in model_config.items():
        run[f"{model_namespace}/parameters/{param_name}"] = param_value
    logger.success(f"Parameters logged: {len(model_config)} parameters")

    # Instantiate and train the model
    logger.info(f"Instantiating {model_name} model...")
    ModelClass = model_mapping[model_name]
    model = ModelClass(**model_config)
    logger.success("Model instantiated successfully")

    logger.info("Starting model training...")
    model_logger.info("Training started...")
    training_start = datetime.now()
    model.fit(train_x, train_y)
    training_duration = (datetime.now() - training_start).total_seconds()

    logger.success(f"Training completed in {training_duration:.2f} seconds")
    model_logger.info(f"Training duration: {training_duration:.2f} seconds")

    # Save the model to experiment folder
    logger.info("Saving model artifact...")
    model_filename = f"{model_name.replace(' ', '_')}_{model_version}.joblib"
    model_path = experiment_folder / "models" / model_filename
    joblib.dump(model, model_path)
    logger.success(f"Model saved: {model_path}")
    model_logger.info(f"Model file: {model_path}")

    # Log model artifact to Neptune
    logger.info("Uploading model to Neptune...")
    run[f"{model_namespace}/artifact"].upload(str(model_path))
    run[f"{model_namespace}/training_duration_seconds"] = training_duration
    logger.success("Model uploaded to Neptune")

    # Predict and evaluate
    logger.info("Generating predictions...")
    predicted_y = model.predict(test_x)
    logger.success(f"Predictions generated for {len(predicted_y)} samples")

    logger.info("Calculating evaluation metrics...")
    report = classification_report(test_y, predicted_y, output_dict=True)
    report_text = classification_report(test_y, predicted_y)

    logger.info(f"Model Accuracy: {report['accuracy']:.4f}")
    model_logger.info(f"Accuracy: {report['accuracy']:.4f}")
    model_logger.info(f"Classification Report:\n{report_text}")

    # Save classification report to file
    logger.info("Saving classification report...")
    report_filename = f"{model_name.replace(' ', '_')}_{model_version}_report.txt"
    report_path = experiment_folder / "reports" / report_filename
    with open(report_path, "w") as f:
        f.write(f"Classification Report for {model_name} (Version {model_version})\n")
        f.write(f"Training Duration: {training_duration:.2f} seconds\n")
        f.write(f"{'='*60}\n\n")
        f.write(report_text)

    logger.success(f"Report saved: {report_path}")

    # Save detailed report as JSON
    logger.info("Saving detailed JSON report...")
    report_json_path = experiment_folder / "reports" / f"{model_name.replace(' ', '_')}_{model_version}_report.json"
    with open(report_json_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "model_version": model_version,
            "training_duration_seconds": training_duration,
            "model_config": model_config,
            "metrics": report
        }, f, indent=2)
    logger.success(f"JSON report saved: {report_json_path}")

    # Log classification report to Neptune
    logger.info("Uploading metrics to Neptune...")
    run[f"{model_namespace}/classification_report"] = report_text
    run[f"{model_namespace}/accuracy"] = report["accuracy"]
    run[f"{model_namespace}/report_file"].upload(str(report_path))
    logger.success("Metrics uploaded to Neptune")

    # Add to experiment summary
    experiment_summary["models"].append({
        "name": model_name,
        "version": model_version,
        "config": model_config,
        "training_duration_seconds": training_duration,
        "accuracy": report["accuracy"],
        "model_file": str(model_path),
        "report_file": str(report_path)
    })

    logger.success(f"Model {model_name} training complete!")
    logger.info("="*80 + "\n")

# Save experiment summary
logger.info("Saving experiment summary...")
summary_path = experiment_folder / "experiment_summary.json"
with open(summary_path, "w") as f:
    json.dump(experiment_summary, f, indent=2)
logger.success(f"Experiment summary saved: {summary_path}")

# Log experiment summary to Neptune
logger.info("Uploading experiment summary to Neptune...")
run["experiment/summary"].upload(str(summary_path))
logger.success("Experiment summary uploaded to Neptune")

# Final summary
logger.info("\n" + "="*80)
logger.info("EXPERIMENT COMPLETE!")
logger.info("="*80)
logger.info(f"Neptune Run ID: {run_id}")
logger.info(f"Experiment Folder: {experiment_folder}")
logger.info(f"Total Models Trained: {len(config['methods'])}")
logger.info(f"Summary File: {summary_path}")

# Log all trained models
logger.info("\nTrained Models Summary:")
for model_info in experiment_summary["models"]:
    logger.info(f"  - {model_info['name']}: Accuracy={model_info['accuracy']:.4f}, Duration={model_info['training_duration_seconds']:.2f}s")

logger.info("="*80 + "\n")

# Stop the Neptune run once all models are logged
logger.info("Stopping Neptune run...")
run.stop()
logger.success("Neptune run stopped successfully")
logger.info("All tasks completed successfully!")