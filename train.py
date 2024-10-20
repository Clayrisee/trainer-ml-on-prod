import yaml
from dataloader import load_pickle
from utils import convert_params
# framework ml yang kita gunakan utk ngetrain model
from sklearn.metrics import classification_report # Evaluatino metrics
from sklearn.ensemble import RandomForestClassifier # Model RF
from sklearn.svm import SVC # Support Vector Machine Model
from sklearn.linear_model import LogisticRegression # Logistic Regression

# Additional package
import neptune #pip install neptune -> Experiment TRacking
import joblib # ngedump model
from dotenv import load_dotenv #pip install python-dotenv
import os


# Load environment variables
load_dotenv()

# Neptune setup
project_name = os.getenv('NEPTUNE_PROJECT_NAME')
api_key = os.getenv('NEPTUNE_API_TOKEN')

print(project_name, api_key)

# Initialize Neptune
run = neptune.init_run(project=project_name, api_token=api_key)

# Load Dataset
train_data = load_pickle("train_data.pickle")
test_data = load_pickle("test_data.pickle")

train_x = train_data['x']
train_y = train_data['y']
test_x = test_data['x']
test_y = test_data['y']


print(train_x, train_y)
print(test_x, test_y)

# Read Config
with open("configs/config.yaml", "r") as f: # Improve pake argument parser
    config = yaml.safe_load(f)

model_version = config["model_versions"]  # Get model version from YAML

# Dictionary to map method names to sklearn models
model_mapping = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
}


for method in config["methods"]:
    model_name = method["name"]
    model_config = convert_params(method["config"])
    
    # Model namespace including version
    model_namespace = f"models/{model_name}/{model_version}"
    
    # Log model parameters to Neptune
    for param_name, param_value in model_config.items():
        run[f"{model_namespace}/parameters/{param_name}"] = param_value
    
    # Instantiate and train the model
    ModelClass = model_mapping[model_name]
    model = ModelClass(**model_config)
    model.fit(train_x, train_y)

    # Save the model to a file
    model_filename = f"{model_name}_{model_version}.joblib"
    joblib.dump(model, model_filename)

    # Log model artifact to Neptune
    run[f"{model_namespace}/artifact"].upload(model_filename)
    
    # Predict and evaluate
    predicted_y = model.predict(test_x)
    report = classification_report(test_y, predicted_y)
    
    # Log classification report to Neptune
    run[f"{model_namespace}/classification_report"] = report
    
    print(f"Classification Report for {model_name} (Version {model_version}):\n{report}\n")
    break

# Stop the Neptune run once all models are logged
run.stop()