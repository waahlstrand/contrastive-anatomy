# contrastive-anatomy

# Data
The initial data comes from the RSNA challenge (https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).

To load the data, run
```python
from pathlib import Path
from data import RSNA

root        = Path("...")
data_path   = root / Path("mdai_rsna_project_x9N20BZa_images_2018-07-20-153330")
labels_path = root / Path("pneumonia-challenge-annotations-adjudicated-kaggle_2018.json")
dataset     = RSNA(data_path, labels_path)

print(dataset[0])
```
which yields the following object
```python

>>> RSNAItem(id='...', path=PosixPath('...'), image=tensor([[  0,   1,   2,  ...,  28,  28,  11],
        [  3,   4,   5,  ...,  61,  58,  34],
        [  4,   5,   6,  ...,  69,  67,  35],
        ...,
        [ 50,  93,  96,  ..., 109, 105,  59],
        [ 59,  97,  96,  ..., 109, 107,  64],
        [ 23,  52,  43,  ...,  55,  61,  26]], dtype=torch.uint8), label=tensor(0), bbox=None)
```

The data may be visualized easily for tests, using

```python

dataset[0].plot()
```
which plots the image and its corresponding bounding box, if any.