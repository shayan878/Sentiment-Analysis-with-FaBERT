from pathlib import Path

from src.data.loaders import load_original_dataset
from src.augmentation.backtranslation import generate_and_save

ROOT = Path(__file__).resolve().parent
train_csv = ROOT / "data/original.csv"
output_csv = ROOT / "data/sentences_dataframe.csv"

texts, _ = load_original_dataset(train_csv)

generate_and_save(
    texts,
    output_csv,
    target_language="en",
    pause_seconds=0.5,
)
print(f"Saved back-translated sentences to: {output_csv}")
