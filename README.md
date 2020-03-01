# ArdisNet: a CNN to classify ARDIS digits

## ARDIS database

ARDIS, for *Arkiv Digital Sweden*, is an image-based handwritten historical digit dataset. This dataset was made up out of 15.000 Swedish church records which were written by different priests with various handwriting styles during the XIXth and XXth centuries. The dataset used here consists of single digits. The digit images have the same format as those of MNIST and USPS.

More information about this dataset can be found [here](https://ardisdataset.github.io/ARDIS/).

## Overview

*ArdisNet* comes with a set of classes and functions in the form of the package `ardis`, which allows the user to run ArdisNet straightforwardly. You do not have to be a deep learning expert to get started.


## Getting started

First, import the needed structures to load the ARDIS data, ArdisNet and its trainer:


```python
import numpy as np

import ardis
from ardis.models.ArdisNet import ArdisNet
from ardis.ArdisTrainer import ArdisTrainer
from ardis import datapcr
from ardis.metrics import accuracy
```

### Global settings: paths, constants, ...


```python
# Number of ensemble networks
nnets = 15

# Number of epochs
epochs = 30
```

### Data loading, preparation and description

#### Data loading and preparation

Load an prepare the data:


```python
# data `x`'s shape: `(|x|, 784)`
# data `y`'s shape: `(|y|, 10)`
(x_train, y_train), (x_test, y_test) = ardis.load_data()

# data `x`'s shape: `(|x|, 28, 28, 1)`
# data `y`'s shape: `(|y|, 10)`
(x_train, x_test) = datapcr.prepare_data(x_train, x_test)

# `y_pred`, such as `y_pred = model(x_test)`, for every network
# `y_pred`'s shape: `(|y|, 10)`
ys_pred = []

# Ensemble (aggregated) `y_pred`
ens_y_pred = np.zeros((x_test.shape[0], 10))
```

The ARDIS dataset is divided so that the training set contains 6600 elements and the test dataset 1000.

#### Data description


```python
x_train.shape
```




    (6600, 28, 28, 1)




```python
y_train.shape
```




    (6600, 10)



### Training and prediction


An ensemble of 15 CNNs is trained, in the ensemble methods' perspective , to obtain better predctive performance.


```python
for i in range(nnets):
    # -- TRAINING --
    model = ArdisNet()
    trainer = ArdisTrainer(model=model)
    history = trainer.run(x_train, y_train, epochs=epochs).history

    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, "
          "Validation accuracy={3:.5f}". \
          format(i + 1, epochs, history['accuracy'][epochs - 1],
                 history['val_accuracy'][epochs - 1]))

    # -- PREDICTION --
    # `y_pred`'s shape: `(|y_pred|, 10)`
    # `ys_pred`'s shape: `(nnets, |y_pred|, 10)`
    ys_pred.append(model.predict(x_test))

    acc = accuracy(ys_pred[i], y_test)

    print("CNN %d: Test accuracy = %f" % (i + 1, acc))
```

    CNN 1: Epochs=30, Train accuracy=0.98892, Validation accuracy=0.99242
    CNN 1: Test accuracy = 0.997000
    CNN 2: Epochs=30, Train accuracy=0.98414, Validation accuracy=0.99697
    CNN 2: Test accuracy = 0.990000
    CNN 3: Epochs=30, Train accuracy=0.98945, Validation accuracy=0.99091
    CNN 3: Test accuracy = 0.991000
    CNN 4: Epochs=30, Train accuracy=0.98639, Validation accuracy=0.99848
    CNN 4: Test accuracy = 0.991000
    CNN 5: Epochs=30, Train accuracy=0.98758, Validation accuracy=0.99697
    CNN 5: Test accuracy = 0.994000
    CNN 6: Epochs=30, Train accuracy=0.98485, Validation accuracy=1.00000
    CNN 6: Test accuracy = 0.994000
    CNN 7: Epochs=30, Train accuracy=0.98553, Validation accuracy=0.99545
    CNN 7: Test accuracy = 0.993000
    CNN 8: Epochs=30, Train accuracy=0.98843, Validation accuracy=0.99545
    CNN 8: Test accuracy = 0.989000
    CNN 9: Epochs=30, Train accuracy=0.98690, Validation accuracy=0.99242
    CNN 9: Test accuracy = 0.994000
    CNN 10: Epochs=30, Train accuracy=0.98499, Validation accuracy=1.00000
    CNN 10: Test accuracy = 0.997000
    CNN 11: Epochs=30, Train accuracy=0.98451, Validation accuracy=1.00000
    CNN 11: Test accuracy = 0.993000
    CNN 12: Epochs=30, Train accuracy=0.98438, Validation accuracy=0.99394
    CNN 12: Test accuracy = 0.992000
    CNN 13: Epochs=30, Train accuracy=0.98860, Validation accuracy=0.99545
    CNN 13: Test accuracy = 0.992000
    CNN 14: Epochs=30, Train accuracy=0.98553, Validation accuracy=0.99848
    CNN 14: Test accuracy = 0.994000
    CNN 15: Epochs=30, Train accuracy=0.99013, Validation accuracy=0.99394
    CNN 15: Test accuracy = 0.992000


### Ensemble Prediction


```python
for i in range(nnets):
    # Ensemble predicted digits on the ARDIS 1000-element test set
    ens_y_pred = ens_y_pred + ys_pred[i]

# Accuracy on the ARDIS 1000-element test set
acc = accuracy(ens_y_pred, y_test)

print("Ensemble Accuracy = %f" % acc)
```

    Ensemble Accuracy = 0.996000


### Performance

Finally, ArdisNet achieves an accuracy of 99.6%. Further improvements will be carried out to improve this performance.
