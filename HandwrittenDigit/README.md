# TensorFlow.js — Handwritten digit recognition with CNNs

https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html

RUN:
```
$ yarn start:dev
```

## Introduction

We'll build a TensorFlow.js model to recognize handwritten digits with a convolutional neural network.

**Supervised Learning** We will train the model by showing it many examples of inputs along with the correct output.

## Setup

Copy `data.js` from https://storage.googleapis.com/tfjs-tutorials/mnist_data.js.

Testing using npm packages with webpack instead of script tags importing.

```
$ yarn add @tensorflow/tfjs --dev
$ yarn add @tensorflow/tfjs-vis --dev
```

## Load the data

28px x 28px greyscale digit images:

MNIST: http://yann.lecun.com/exdb/mnist/

`MnistData`
* `nextTrainBatch(batchSize)`: returns a random batch of images + labels from the training set.
* `nextTestBatch(batchSize)`: returns a batch of images and their labels from the test set.
* does shuffling and normalizing the data.

There are a total of 65,000 images, we will use up to 55,000 images to train the model, saving 10,000 images that we can use to test the model's performance once we are done.

## Conceptualize our task

Our goal is to train a model that will take one image and learn to predict a score for each of the possible 10 classes that image may belong to (the digits 0-9).

Each image is 28px wide 28px high and has 1 color channel (grayscale). So the shape of each image is `[28, 28, 1]`.

## Define the model architecture

`Model Architecture` => which functions will the model run when it is executing. What algorithm will our model use to compute its answers.

```javascript
function getModel() {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    /**
     * In the first layer of our convolutional neural network we have to specify
     * the input shape. Then we specify some parameters for the convolution
     * operation that takes place in this layer.
     */
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    // The MaxPooling layer acts as a sort of downsampling using max values in a 
    // region instead of averaging.
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Repeat another conv2d + maxPooling stack.
    // Note that we have more filters in the convolution.
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    /**
     * Now we flatten the output from the 2D filters into a 1D vector to prepare
     * it for input into our last layer. This is a common practice when feeding
     * higher dimensional data to a final classification output layer.
     */
    model.add(tf.layers.flatten());

    // Our last layer is a dense layer which has 10 output units, one for each 
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax',
    }));

    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.adam();
    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}
```

### Convolutions

Details:
* Image Kernels: https://setosa.io/ev/image-kernels/
* Convolutional Neural Networks: http://cs231n.github.io/convolutional-networks/

We are using a `conv2d` layer instead of a dense layer.

* `inputShape`: the shape of the data that will flow into the first layer of the model. (MNIST examples) The canonical format for image data is `[row, column, depth]` => `[28, 28, 1]`. Layers are designed to be batch size agnostic so that during inference you can pass a tensor of any batch size in.
* `kernelSize`: the size of the sliding convolutional filter windows to the applied to the input data.
* `filters`: the number of filter windows of size `kernelSize` to apply to the input data.
* `strides`: the "step size" of the sliding window (how many pixels the filter will shift each time it moves over the image).
* `activation`: the activation function to apply to the data after the convolution is complete.
* `kernelInitializer`: the method to use for randomly initializing the model weights, which is very important to training dynamics.

### Compute final probability distribution

`Softmax` is most likely activation you will want to use at the last layer of a classification task.

## Train the model

```javascript
async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });
    
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
}
```

Then add to `run`:
```javascript
const model = getModel();

tfvis.show.modelSummary({ name: 'Model Architecture' }, model);
await train(model, data);
```

## Evaluate our model

```javascript
const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax([-1]);
  const preds = model.predict(testxs).argMax([-1]);

  testxs.dispose();
  return [preds, labels];
}


async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(
      container, {values: confusionMatrix}, classNames);

  labels.dispose();
}
```

* Makes a prediction
* Computes accuracy metrics
* Shows the metrics

