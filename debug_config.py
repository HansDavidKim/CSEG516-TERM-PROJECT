import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path.cwd()))

try:
    from config import dataset_config, classifier_config
    print("Successfully imported config.")
    print(f"Dataset Config keys: {list(dataset_config.keys())}")
    print(f"Hugging Face datasets: {dataset_config.get('hugging_face')}")
    print(f"Kaggle datasets: {dataset_config.get('kaggle_hub')}")
except Exception as e:
    print(f"Error loading config: {e}")
    import traceback
    traceback.print_exc()

try:
    import typer
    print(f"Typer version: {typer.version.VERSION}")
except ImportError:
    print("Typer not installed.")
except Exception as e:
    print(f"Error importing typer: {e}")
