import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix


MAX_VOCAB_SIZE = 20_000 # VOCABULARY max size 20000 lexeis pio popular
MAX_SEQUENCE_LENGTH = 1_024
EMBEDDING_DIM = 64
BATCH_SIZE = 32
NUM_FILTERS = 128
KERNEL_SIZE = 3
PREDICTION_THRESHOLD = 0.5
NUM_EPOCHS = 4
LEARNING_RATE = 0.001
RANDOM_SEED = 42
PAD_TOKEN = "<PAD>" # padding short prompt contain 0 at the ned in order to get the max sequence longer keep the first 1024 (we shorten 139 prompts)
UNK_TOKEN = "<UNK>"  # Tokens outside the vocabulary use ID 1.


# 1. Load a dataset split without mixing training and validation examples.
def load_split(split_name):
    path = Path(__file__).resolve().parent / "data" / f"{split_name}.json"

    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    texts = [row["prompt"] for row in rows]
    labels = [row["label"] for row in rows]
    return texts, labels


# 2. Split text into lowercase words and individual punctuation symbols.
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


# 3. Assign IDs to the most common tokens, using training data only.
def build_vocabulary(tokenized_texts):
    counts = Counter(token for tokens in tokenized_texts for token in tokens)

    # Reserve 0 for future padding and 1 for words outside the vocabulary.
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counts.most_common(MAX_VOCAB_SIZE - 2):
        vocabulary[token] = len(vocabulary)

    return vocabulary


# 4. Replace each token with its ID, preserving its position in the prompt.
def encode(tokens, vocabulary):
    return [vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in tokens]


# 5. Keep the first max_length tokens and fill empty positions with <PAD>.
def pad_sequence(token_ids, max_length, pad_id):
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    sequence = token_ids[:max_length]
    return sequence + [pad_id] * (max_length - len(sequence))


# 6. Give every token a trainable vector; keep the <PAD> vector at zero.
def build_embedding_layer(vocabulary):
    return nn.Embedding(
        num_embeddings=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        padding_idx=vocabulary[PAD_TOKEN],
    )


# 7. Each filter scans three neighbouring tokens across all embedding dimensions.
def build_convolution_layer():
    return nn.Conv1d(
        in_channels=EMBEDDING_DIM,
        out_channels=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
    )


def extract_convolution_features(embedded_batch, token_ids, convolution_layer, pad_id):
    # Conv1D expects (batch, embedding dimensions, token positions).
    convolution_input = embedded_batch.transpose(1, 2)
    feature_maps = torch.relu(convolution_layer(convolution_input))

    # Only keep scores for windows containing real tokens, with no <PAD>.
    token_windows = token_ids.unfold(1, convolution_layer.kernel_size[0], 1)
    valid_windows = token_windows.ne(pad_id).all(dim=2)
    return feature_maps.masked_fill(~valid_windows.unsqueeze(1), 0.0)


# 8. Keep one maximum score per filter, across all token-window positions.
def build_pooling_layer():
    return nn.AdaptiveMaxPool1d(output_size=1)


# 9. Combine the 128 pooled filter scores into one raw score (logit).
def build_classification_layer():
    return nn.Linear(in_features=NUM_FILTERS, out_features=1)


# 10. Register all layers in one model so the optimizer can update them together.
class TextCNN(nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.pad_id = vocabulary[PAD_TOKEN]
        self.embedding = build_embedding_layer(vocabulary)
        self.convolution = build_convolution_layer()
        self.pooling = build_pooling_layer()
        self.classifier = build_classification_layer()

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        feature_maps = extract_convolution_features(
            embedded, token_ids, self.convolution, self.pad_id
        )
        pooled = self.pooling(feature_maps).squeeze(-1)
        # Return raw logits: BCEWithLogitsLoss applies sigmoid internally.
        return self.classifier(pooled).squeeze(-1)


# 11. Learn from shuffled training batches; clear old gradients before each update.
def train_one_epoch(model, loader, loss_function, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    examples = 0

    for batch_number, (token_ids, labels) in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(token_ids)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        predictions = torch.sigmoid(logits.detach()) >= PREDICTION_THRESHOLD
        correct += (predictions == labels.bool()).sum().item()
        examples += len(labels)
        if batch_number % 100 == 0 or batch_number == len(loader):
            print(
                f"  Batch {batch_number}/{len(loader)} | "
                f"training loss: {total_loss / examples:.4f}",
                flush=True,
            )

    return total_loss / examples, correct / examples


# 12. Measure validation performance without updating weights or tracking gradients.
@torch.no_grad()
def evaluate(model, loader, loss_function):
    model.eval()
    total_loss = 0.0
    targets = []
    predictions = []

    for token_ids, labels in loader:
        logits = model(token_ids)
        total_loss += loss_function(logits, labels).item() * len(labels)
        predicted = (torch.sigmoid(logits) >= PREDICTION_THRESHOLD).long()
        targets.extend(labels.long().tolist())
        predictions.extend(predicted.tolist())

    accuracy = sum(a == b for a, b in zip(targets, predictions)) / len(targets)
    return total_loss / len(targets), accuracy, targets, predictions


def print_validation_report(targets, predictions):
    print("\nValidation results:")
    print(classification_report(
        targets, predictions, labels=[0, 1],
        target_names=["benign", "injection"], digits=3, zero_division=0,
    ))
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    print("Confusion matrix:")
    print("Order: benign, injection")
    print(matrix)
    tn, fp, fn, tp = matrix.ravel()
    benign_count = tn + fp
    if benign_count:
        print(f"\nFalse-positive rate: {fp / benign_count:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Train a simple prompt-injection CNN.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--preview-only", action="store_true", help="Show shapes without training.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    texts, labels = load_split("train")
    tokenized_texts = [tokenize(text) for text in texts]
    vocabulary = build_vocabulary(tokenized_texts)
    token_ids = [encode(tokens, vocabulary) for tokens in tokenized_texts]
    padded_token_ids = [
        pad_sequence(ids, MAX_SEQUENCE_LENGTH, vocabulary[PAD_TOKEN])
        for ids in token_ids
    ]
    truncated_count = sum(len(ids) > MAX_SEQUENCE_LENGTH for ids in token_ids)

    print(f"Training examples: {len(texts):,}")
    print(f"Vocabulary size: {len(vocabulary):,} (including <PAD> and <UNK>)")
    print("Reserved IDs: <PAD> = 0, <UNK> = 1")
    print(f"Padded training shape: ({len(padded_token_ids):,}, {MAX_SEQUENCE_LENGTH:,})")
    print(
        f"Prompts truncated to their first {MAX_SEQUENCE_LENGTH:,} tokens: "
        f"{truncated_count:,} ({truncated_count / len(texts):.2%})"
    )
    print("\nFirst training example:")
    print("Prompt:", texts[0])
    print("First 20 tokens:", tokenized_texts[0][:20])
    print("First 20 token IDs:", token_ids[0][:20])
    print("Length before padding:", len(token_ids[0]))
    print("Length after padding:", len(padded_token_ids[0]))
    print("Last 10 IDs after padding:", padded_token_ids[0][-10:])
    print("Label:", labels[0])

    # Preview one batch instead of embedding the entire dataset at once.
    torch.manual_seed(RANDOM_SEED)
    model = TextCNN(vocabulary)
    embedding_layer = model.embedding
    batch_token_ids = torch.tensor(padded_token_ids[:BATCH_SIZE], dtype=torch.long)
    with torch.no_grad():  # Only inspecting the vectors; training comes later.
        embedded_batch = embedding_layer(batch_token_ids)

    print("\nEmbedding preview:")
    print("Embedding table shape:", tuple(embedding_layer.weight.shape))
    print("Batch token IDs shape:", tuple(batch_token_ids.shape))
    print("Embedded batch shape:", tuple(embedded_batch.shape))
    print(f"First token: {tokenized_texts[0][0]!r}, ID: {token_ids[0][0]}")
    print(
        "First 8 numbers of its 64-number vector:",
        [round(value, 4) for value in embedded_batch[0, 0, :8].tolist()],
    )
    convolution_layer = model.convolution
    with torch.no_grad():
        feature_maps = extract_convolution_features(
            embedded_batch, batch_token_ids, convolution_layer, vocabulary[PAD_TOKEN]
        )

    print("\nConv1D preview:")
    print(f"Filters: {NUM_FILTERS}, tokens per window: {KERNEL_SIZE}")
    print("Conv1D input shape:", tuple(embedded_batch.transpose(1, 2).shape))
    print("Feature maps shape after ReLU and padding mask:", tuple(feature_maps.shape))
    print(
        "First filter scores at the first 5 positions of the first prompt:",
        [round(value, 4) for value in feature_maps[0, 0, :5].tolist()],
    )

    pooling_layer = model.pooling
    with torch.no_grad():
        # (batch, filters, positions) -> (batch, filters, 1) -> (batch, filters).
        # Only remove the last dimension, so a batch of one still keeps its batch axis.
        pooled_features = pooling_layer(feature_maps).squeeze(-1)

    print("\nGlobal max pooling preview:")
    print("Pooled features shape:", tuple(pooled_features.shape))
    print(
        "First 8 filter maximum scores for the first prompt:",
        [round(value, 4) for value in pooled_features[0, :8].tolist()],
    )
    strongest_position = feature_maps[0, 0].argmax().item()
    print(
        "First filter's highest-scoring token window in the first prompt:",
        tokenized_texts[0][strongest_position:strongest_position + KERNEL_SIZE],
    )
    print("First filter's maximum score:", round(pooled_features[0, 0].item(), 4))
    print(f"Each prompt is now summarized by {NUM_FILTERS} filter maximum scores.")

    classification_layer = model.classifier
    batch_labels = torch.tensor(labels[:BATCH_SIZE], dtype=torch.float32)
    loss_function = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        # Remove only the last dimension to keep one logit per prompt.
        logits = classification_layer(pooled_features).squeeze(-1)
        injection_scores = torch.sigmoid(logits)
        predictions = (injection_scores >= PREDICTION_THRESHOLD).long()
        # This loss applies sigmoid internally, so pass the raw logits.
        batch_loss = loss_function(logits, batch_labels)

    print("\nClassification preview (untrained):")
    print("Logits shape:", tuple(logits.shape))
    print("First 5 raw scores (logits):", [round(value, 4) for value in logits[:5].tolist()])
    print(
        "First 5 injection scores after sigmoid:",
        [round(value, 4) for value in injection_scores[:5].tolist()],
    )
    print(f"Decision threshold: {PREDICTION_THRESHOLD} (0 = benign, 1 = injection)")
    print("First 5 predicted labels:", predictions[:5].tolist())
    print("First 5 actual labels:", labels[:5])
    print("Untrained batch loss:", round(batch_loss.item(), 4))
    print("These preview scores use random weights.")
    if args.preview_only:
        return

    # Fit the vocabulary on training data only; reuse it for validation.
    validation_texts, validation_labels = load_split("validation")
    validation_ids = [
        pad_sequence(encode(tokenize(text), vocabulary), MAX_SEQUENCE_LENGTH, vocabulary[PAD_TOKEN])
        for text in validation_texts
    ]
    training_data = TensorDataset(
        torch.tensor(padded_token_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )
    validation_data = TensorDataset(
        torch.tensor(validation_ids, dtype=torch.long),
        torch.tensor(validation_labels, dtype=torch.float32),
    )
    train_loader = DataLoader(
        training_data, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    print(f"\nTraining for {args.epochs} epochs on CPU:", flush=True)
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_function, optimizer)
        validation_loss, validation_accuracy, _, _ = evaluate(model, validation_loader, loss_function)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "validation_loss": validation_loss, "validation_accuracy": validation_accuracy,
        })
        print(
            f"Train loss: {train_loss:.4f} | accuracy: {train_accuracy:.3%} | "
            f"Validation loss: {validation_loss:.4f} | accuracy: {validation_accuracy:.3%}",
            flush=True,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, _, targets, predictions = evaluate(model, validation_loader, loss_function)
    print(f"\nUsing epoch {best_epoch}, selected by lowest validation loss.")
    print_validation_report(targets, predictions)

    checkpoint_path = Path(__file__).resolve().parent / "artifacts" / "cnn.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocabulary": vocabulary,
        "config": {
            "max_sequence_length": MAX_SEQUENCE_LENGTH, "embedding_dim": EMBEDDING_DIM,
            "num_filters": NUM_FILTERS, "kernel_size": KERNEL_SIZE,
            "prediction_threshold": PREDICTION_THRESHOLD, "random_seed": RANDOM_SEED,
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        },
        "best_epoch": best_epoch, "history": history,
    }, checkpoint_path)
    print(f"\nSaved model weights, vocabulary, settings, and history to: {checkpoint_path}")


if __name__ == "__main__":
    main()
