# ML Training Pipeline

A machine learning training pipeline with experiment tracking using Neptune.ai. This project supports training multiple models (Logistic Regression, Random Forest, SVM) with configurable parameters and automatic experiment logging.

## Features

- Multiple ML model support (Logistic Regression, Random Forest, SVM)
- Experiment tracking with Neptune.ai
- Automatic model versioning
- Structured experiment folders with logs, models, and reports
- Detailed logging with Loguru
- Configuration-based training with YAML files
- Classification reports and metrics tracking

## Project Structure

```
.
├── configs/              # Configuration files
│   └── config.yaml      # Model configurations and parameters
├── data/                # Dataset storage
├── dataloader.py        # Data loading utilities
├── utils.py            # Utility functions
├── train.py            # Main training script
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
└── experiments/        # Generated experiment results (gitignored)
    └── <timestamp>_<run_id>/
        ├── models/     # Saved model artifacts (.joblib)
        ├── reports/    # Classification reports (txt, json)
        └── logs/       # Training logs
```

## Prerequisites

- Python 3.7+
- Neptune.ai account (for experiment tracking)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trainer-ml-on-prod
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` file with your Neptune credentials:
```env
NEPTUNE_PROJECT_NAME=your-workspace/your-project
NEPTUNE_API_TOKEN=your-api-token-here
```

To get your Neptune credentials:
- Sign up at [Neptune.ai](https://neptune.ai)
- Create a new project
- Get your API token from your user settings
- Format: `NEPTUNE_PROJECT_NAME=workspace-name/project-name`

## Usage

### Basic Training

Run training with the default configuration:

```bash
python train.py
```

### Custom Configuration

Specify a custom configuration file:

```bash
python train.py --config configs/your_config.yaml
```

### Configuration File Format

Example `config.yaml`:

```yaml
model_versions: "v1.0.0"

methods:
  - name: "Logistic Regression"
    config:
      max_iter: 1000
      random_state: 42

  - name: "Random Forest"
    config:
      n_estimators: 100
      max_depth: 10
      random_state: 42

  - name: "SVM"
    config:
      kernel: "rbf"
      C: 1.0
      random_state: 42
```

## Data Requirements

The pipeline expects the following pickle files in the project directory:
- `train_data.pickle`: Training dataset with keys `'x'` (features) and `'y'` (labels)
- `test_data.pickle`: Test dataset with keys `'x'` (features) and `'y'` (labels)

## Output

Each training run creates a timestamped experiment folder containing:

1. **Models** (`models/`):
   - Trained model files in joblib format
   - Named as `<ModelName>_<version>.joblib`

2. **Reports** (`reports/`):
   - Classification reports (text format)
   - Detailed JSON reports with metrics
   - Per-model performance metrics

3. **Logs** (`logs/`):
   - General experiment log (`experiment.log`)
   - Model-specific logs for each trained model

4. **Metadata**:
   - `config.yaml`: Copy of configuration used
   - `experiment_summary.json`: Summary of all models trained

## Neptune Integration

All experiments are automatically logged to Neptune with:
- Model parameters
- Training metrics and accuracy
- Classification reports
- Model artifacts
- Training duration
- Experiment metadata

Access your experiments at: `https://app.neptune.ai/<workspace>/<project>/experiments`

## Example Workflow

1. Prepare your data as pickle files
2. Configure models in `configs/config.yaml`
3. Set up Neptune credentials in `.env`
4. Run training: `python train.py`
5. Check the generated `experiments/` folder for results
6. View detailed metrics in Neptune dashboard

## Logging

The project uses Loguru for comprehensive logging:
- Console output with colored, formatted logs
- File-based logging in experiment folders
- Model-specific log files
- Automatic log rotation (50 MB limit)

## Dependencies

- scikit-learn: Machine learning models
- PyYAML: Configuration file parsing
- python-dotenv: Environment variable management
- neptune: Experiment tracking
- joblib: Model serialization
- loguru: Advanced logging

## Troubleshooting

### Neptune Connection Issues
- Verify your API token and project name in `.env`
- Check internet connectivity
- Ensure Neptune project exists

### Missing Data Files
- Ensure `train_data.pickle` and `test_data.pickle` exist
- Verify pickle files contain `'x'` and `'y'` keys

### Import Errors
- Run `pip install -r requirements.txt`
- Check Python version compatibility
