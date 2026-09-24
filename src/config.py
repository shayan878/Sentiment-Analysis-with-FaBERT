from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    train_csv: Path = Path("data/original.csv")
    test_csv: Path = Path("data/test.csv")
    augmented_csv: Path = Path("data/sentences_dataframe.csv")

    text_column_index: int = 0
    label_column_index: int = 1
    augmented_text_column: str = "Sentences"

    model_name: str = "sbunlp/fabert"
    max_length: int = 512

    validation_size: float = 0.25
    batch_size: int = 4
    learning_rate: float = 1e-5
    epochs: int = 5
    dropout: float = 0.1
    num_classes: int = 3

    random_state: int | None = None
    device: str = "auto"

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
