from pathlib import Path

from src.train import train_and_evaluate


def test_train_and_evaluate_creates_output(tmp_path):
    accuracy, matrix, figure_path = train_and_evaluate(output_dir=tmp_path)

    assert accuracy >= 0.9
    assert matrix.shape == (3, 3)
    assert Path(figure_path).exists()
