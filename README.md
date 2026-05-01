# Iris Classifier - Decision Tree

## Overview

End-to-end machine learning demo that builds a decision-tree classifier on the classic Iris dataset using scikit-learn.

The project includes:

- A Jupyter notebook walkthrough in `notebooks/iris/_model.ipynb`
- A reusable Python script in `src/train.py`
- A generated confusion matrix image in `outputs/confusion_matrix.png`
- A small test that checks model accuracy and output creation

## Project Structure

```text
iris-classitier/
├── data/
├── notebooks/
│   └── iris/
│       └── _model.ipynb
├── outputs/
├── src/
│   └── train.py
├── tests/
│   └── test_train.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Quick Start

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the training script:

```powershell
python src/train.py --test-size 0.2 --random-state 42
```

Expected output:

```text
Accuracy: 0.93
Confusion matrix:
[[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
Saved confusion matrix figure to: outputs\confusion_matrix.png
```

The exact accuracy can change if you use a different test size or random state.

## Notebook

Open and run:

```text
notebooks/iris/_model.ipynb
```

The notebook follows the lesson steps:

- Load the Iris dataset
- Split the data into train and test sets
- Train a decision-tree classifier
- Make predictions
- Evaluate accuracy
- Display a confusion matrix
- Explain the result in markdown

## Tests

Run:

```powershell
pytest
```

The test checks that the model reaches at least `0.9` accuracy and creates the confusion-matrix image.

## License

This project uses the MIT License.
