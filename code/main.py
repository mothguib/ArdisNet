# -*- coding: utf-8 -*-

# Executed only if run as a script
if __name__ == '__main__':

    import os
    import numpy as np

    # Handles whether TensorFlow uses the CPU
    CPU = 0

    if CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import ardis
    from ardis.models.ArdisNet import ArdisNet
    from ardis.ArdisTrainer import ArdisTrainer
    from ardis import datapcr
    from ardis.metrics import accuracy

    nnets = 15

    # data `x`'s shape: `(|x|, 784)`
    # data `y`'s shape: `(|y|, 10)`
    (x_train, y_train), (x_test, y_test) = ardis.load_data()

    # data `x`'s shape: `(|x|, 28, 28, 1)`
    # data `y`'s shape: `(|y|, 10)`
    (x_train, x_test) = datapcr.prepare_data(x_train, x_test)

    epochs = 30

    # `y_pred`, such as `y_pred = model(x_test)`, for every network
    ys_pred = []

    # Ensemble (aggregated) `y_pred`
    ens_y_pred = np.zeros((x_test.shape[0], 10))

    # - TRAINING AND PREDICTION -
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

        print("CNN %d: Test accuracy = %f" % (i + 1, acc))

    # - ENSEMBLE PREDICTION -
    for i in range(nnets):
        # Ensemble predicted digits on the ARDIS 1000-element test set
        ens_y_pred = ens_y_pred + ys_pred[i]

    # Accuracy on the ARDIS 1000-element test set
    acc = accuracy(ens_y_pred, y_test)

    print("Ensemble Accuracy = %f" % acc)
