# License Plate Recognition

Reads Australian number plates from a photo of a car.

Finds the plate with connected component analysis, splits it into
characters, and classifies each one with an SVM.

## Setup

```
pip install numpy scikit-image scikit-learn matplotlib joblib
```

## Usage

Run from the project root:

```
python src/prediction.py
```

Prints the plate text for `images/car.jpg`. To retrain the classifier on
the samples in `src/train`:

```
python src/machine_train.py
```
