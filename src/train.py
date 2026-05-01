from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_and_evaluate(test_size=0.2, random_state=42, output_dir="outputs"):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=test_size,
        random_state=random_state,
        stratify=iris.target,
    )

    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure_path = output_path / "confusion_matrix.png"

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=iris.target_names,
    )
    display.plot(cmap="Blues")
    plt.title("Iris Decision Tree Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

    return accuracy, matrix, figure_path


def parse_args():
    parser = ArgumentParser(description="Train an Iris decision-tree classifier.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    accuracy, matrix, figure_path = train_and_evaluate(
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )

    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion matrix:")
    print(matrix)
    print(f"Saved confusion matrix figure to: {figure_path}")


if __name__ == "__main__":
    main()
