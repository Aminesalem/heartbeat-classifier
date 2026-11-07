# Heartbeat classifier

A small 1D CNN in PyTorch that sorts ECG heartbeats into 5 classes: normal, supraventricular, ventricular, fusion and unknown. I used the MIT-BIH dataset from Kaggle, where the recordings are already cut into single beats.

About 90% of the beats are normal, so a model can get good accuracy by just saying "normal" every time. Because of that I looked at recall and precision per class, and I trained the model twice: once normally and once with class weights in the loss.

## Results

Without class weights:
```
class               recall  precision   count
normal               0.996      0.982   18118
supraventricular     0.658      0.931     556
ventricular          0.948      0.940    1448
fusion               0.617      0.885     162
unknown              0.964      0.994    1608
accuracy: 0.979
```

With class weights:
```
class               recall  precision   count
normal               0.907      0.994   18118
supraventricular     0.867      0.321     556
ventricular          0.939      0.844    1448
fusion               0.895      0.286     162
unknown              0.985      0.911    1608
accuracy: 0.914
```

The first model has great accuracy but misses a lot of the rare beats: only 66% of supraventricular and 62% of fusion beats are found. With weights it finds almost 9 in 10 of them, but it also labels a lot of normal beats as abnormal, which is why precision falls to about 0.3 and accuracy goes down to 91%. For heartbeats I'd rather have false alarms than miss an abnormal one, so I prefer the weighted version, even if the weights I used are probably too strong.

## How to run

1. `pip install -r requirements.txt`
2. Download the ECG Heartbeat Categorization Dataset from Kaggle and put `mitbih_train.csv` and `mitbih_test.csv` in the `data/` folder
3. Open `src/train.py` and set `USE_WEIGHTS` to `False` (baseline) or `True` (class weights)
4. `python src/train.py`

It trains for 10 epochs and prints the table at the end. It runs fine on CPU, about 3 minutes.

## Project structure

```
src/data.py      loads the csv files
src/model.py     the CNN
src/metrics.py   recall and precision per class
src/train.py     training and evaluation
results/         output of the two runs
```

## Notes

- The Kaggle train/test split is by beat, not by patient, so beats from the same person can end up in both sets. On new patients the score would be lower.
- Next thing I want to try: softer class weights (for example the square root of the ratio) to get a better balance between recall and precision.