from pathlib import Path
import time
import pandas as pd

def backtranslate_sentence(sentence: str, target_language: str = "en", pause_seconds: float = 0.5) -> str:
    """Translate Persian -> target_language -> Persian using googletrans."""
    try:
        from googletrans import Translator
    except ImportError as exc:
        raise ImportError(
            "googletrans is required for back-translation. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    translator = Translator()
    translated = translator.translate(sentence, dest=target_language)
    time.sleep(pause_seconds)
    backtranslated = translator.translate(
        translated.text,
        src=target_language,
        dest="fa",
    )
    return backtranslated.text

def generate_backtranslations(texts, target_language: str = "en", pause_seconds: float = 0.5):
    return [
        backtranslate_sentence(text, target_language=target_language, pause_seconds=pause_seconds)
        for text in texts
    ]

def save_backtranslations(texts, output_path: str | Path, column_name: str = "Sentences"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({column_name: texts})
    df.to_csv(output_path, index=False)
    return output_path

def generate_and_save(texts, output_path, target_language="en", pause_seconds=0.5):
    augmented = generate_backtranslations(texts, target_language, pause_seconds)
    return save_backtranslations(augmented, output_path)
