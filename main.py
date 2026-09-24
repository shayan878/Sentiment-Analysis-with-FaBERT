from pathlib import Path

from transformers import AutoTokenizer

from src.config import Config
from src.data.loaders import load_original_dataset, load_augmented_dataset
from src.data.preprocessing import (
    normalize_labels,
    split_texts_and_labels,
    build_tensor_dataset,
    build_dataloader,
)
from src.models.fabert import load_fabert_classifier, freeze_backbone_except_last_layer_and_pooler
from src.training.engine import fit, evaluate
from src.evaluation.metrics import classification_report_values
from src.visualization.plots import plot_label_distribution, plot_training_history, plot_confusion_matrix


def prepare_dataloaders(config: Config, tokenizer):
    train_texts, train_labels = load_original_dataset(
        config.train_csv,
        config.text_column_index,
        config.label_column_index,
    )
    test_texts, test_labels = load_original_dataset(
        config.test_csv,
        config.text_column_index,
        config.label_column_index,
    )

    train_labels = normalize_labels(train_labels)
    test_labels = normalize_labels(test_labels)

    plot_label_distribution(train_labels)

    original_train_texts, original_valid_texts, original_train_labels, original_valid_labels = split_texts_and_labels(
        train_texts,
        train_labels,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )

    augmented_texts = load_augmented_dataset(
        config.augmented_csv,
        config.augmented_text_column,
    )
    augmented_train_texts, augmented_valid_texts, augmented_train_labels, augmented_valid_labels = split_texts_and_labels(
        augmented_texts,
        train_labels,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )

    train_dataset_original = build_tensor_dataset(
        original_train_texts,
        original_train_labels,
        tokenizer,
        config.max_length,
    )
    valid_dataset_original = build_tensor_dataset(
        original_valid_texts,
        original_valid_labels,
        tokenizer,
        config.max_length,
    )

    train_dataset_augmented = build_tensor_dataset(
        augmented_train_texts,
        augmented_train_labels,
        tokenizer,
        config.max_length,
    )
    valid_dataset_augmented = build_tensor_dataset(
        augmented_valid_texts,
        augmented_valid_labels,
        tokenizer,
        config.max_length,
    )

    test_dataset = build_tensor_dataset(
        test_texts,
        test_labels,
        tokenizer,
        config.max_length,
    )

    return {
        "original_train": build_dataloader(train_dataset_original, config.batch_size, shuffle=True),
        "original_valid": build_dataloader(valid_dataset_original, config.batch_size, shuffle=False),
        "augmented_train": build_dataloader(train_dataset_augmented, config.batch_size, shuffle=True),
        "augmented_valid": build_dataloader(valid_dataset_augmented, config.batch_size, shuffle=False),
        "test": build_dataloader(test_dataset, config.batch_size, shuffle=False),
        "test_texts": test_texts,
        "test_labels": test_labels,
    }


def run_experiment(name, train_loader, valid_loader, test_loader, config):
    print(f"\n{'=' * 20} {name} {'=' * 20}")

    model = load_fabert_classifier(
        model_name=config.model_name,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )
    freeze_backbone_except_last_layer_and_pooler(model)
    model.to(config.resolved_device)

    history = fit(
        model,
        train_loader,
        valid_loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        device=config.resolved_device,
    )
    plot_training_history(history)

    import torch
    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, config.resolved_device)
    report = classification_report_values(
        test_metrics["y_true"],
        test_metrics["y_pred"],
    )

    plot_confusion_matrix(
        report["confusion_matrix"],
        classes=list(range(config.num_classes)),
    )

    print(
        f"test_accuracy: {test_metrics['accuracy']:.4f}, "
        f"test_loss: {test_metrics['loss']:.4f}"
    )
    print(f"F1 score: {report['f1_macro']:.4f}")

    return model, history, test_metrics, report


def main():
    project_root = Path(__file__).resolve().parent
    config = Config(
        train_csv=project_root / "data/original.csv",
        test_csv=project_root / "data/test.csv",
        augmented_csv=project_root / "data/sentences_dataframe.csv",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    loaders = prepare_dataloaders(config, tokenizer)

    run_experiment(
        "Original Data",
        loaders["original_train"],
        loaders["original_valid"],
        loaders["test"],
        config,
    )

    run_experiment(
        "Back-Translated Data",
        loaders["augmented_train"],
        loaders["augmented_valid"],
        loaders["test"],
        config,
    )


if __name__ == "__main__":
    main()
