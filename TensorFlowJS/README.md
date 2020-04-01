# TensorFlow JS

## Making Predictions from 2D Data

### Introduction & setup

https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html

NPM install:
* https://www.npmjs.com/package/@tensorflow/tfjs
* https://www.npmjs.com/package/@tensorflow/tfjs-vis

Or import as scripts:
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@1.0.2/dist/tfjs-vis.umd.min.js"></script>
```

You should have access to `tf` and `tfvis` in the console.

### Load, format and visualize the input data

```javascript
/**
 * Get the car data reduced to just the variables we are interested and cleaned of missing data.
 */
async function getData() {
    const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataReq.json();
    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
    .filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned;
}
```

```javascript
async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg,
    }));

    tfvis.render.scatterplot(
        {name: 'Horsepower v MPG'},
        {values},
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );
}

document.addEventListener('DOMContentLoaded', run);
```

If there is no structure in the data, the model won't really be able to learn anything.

Our goal is to train a model that will take one number, Horsepower and learn to predict another number, Miles per Gallon.

### Define the model architecture

ML models are algorithms that take an input and produce an output. When using a neural networks, the algorithm is a set of layers of neurons with `weights` (numbers) governing their output. The training process learn the ideal values for those weights.

```javascript
function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

    // Add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true}));

    return model;
}
```

This model is sequential because its inputs flow straight down to its output. Othe kinds of models can have branches, or even multiple inputs and outputs, but in many cases your models will be sequential. SEquential models also have an easier to use API.

A `dense` layer is a type of layer that multiplies its inputs by a matrix (called weights) and then adds a number (called bias) to the result.

Dense layers come with a bias term by default, so we do not need to set useBias to true.

Add to `run`:

```javascript
// Create the model
const model = createModel();
tfvis.show.modelSummary({name: 'Model Summary'}, model);
```

### Prepare the data for training

To get the performance benefits of TensorFlow, we need to convert the data to tensors.

Also some best practice transformations like shuffling and noarmalization.

```javascript
/**
 * Convert the input data to tensors that we can use for machine learning.
 * We will also do the import best practices of _shuffling_ the data and
 * _normalizing_ the data MPG on the y-axis
 */
function convertToTensor(data) {
    // Wrapping these calculation in a tidy will dispose any intermediate tensors.

    return tf.tidy(() => {
        // Step 1. Shuffle the data
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d.horsepower);
        const labels = data.map(d => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        };
    });
}
```

#### Shuffle the data

Shuffling helps each batch have a variety of data from across the data distribution.

**Best Practice 1**: You should always shuffle your data before handing it to the training algorithms in TensorFlow.

**Best Practice 2**: You should always consider normalizing your data before training.

### Train the model

```javascript
async function trainModel(model, inputs, labels) {
    // Prepare the model for training
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 50;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            {name: 'Training Performance'},
            ['loss', 'mse'],
            {height: 200, callbacks: ['onEpochEnd']}
        )
    });
}
```

We have to `compile` the model before we train it.

* `optimizer` is the algorithm that is going to govern the updates to the model as it sees examples. (adam optimizer is quite effective in practice and requires no configuration).
* `loss` is a function that will tell the model how well it is doing on learning each of the batches that it is shown. (`meanSquareErrror` compares the predictions made by the model with the true values)
* `batchSize` size of the data subsets that the model will see on each iteration of training. (normal sizes: 32-512).
* `epochs` number of times the model is going to look at the entire dataset that you provide it.
* `model.fit` is the function we call to start the training loop.

Add to `run`:
```javascript
// Convert the data to a form we can use for training
const tensorData = convertToTensor(data);
const {inputs, labels} = tensorData;
// Train the model
await trainModel(model, inputs, labels);
console.log('Done Training');
```

### Make Predictions

Using the trained model, we can make some predictions.

```javascript
function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, lableMin, labelMax} = normalizationData;

    // Generate predictions for a uniform range of numbers between 0 and 1.
    // We un-normalize the data by doing the inverse of the min-max scaling that we did earlier.
    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]));

        const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
        const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

        // un-normalize the data
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
        return {x: val, y: preds[i]};
    });

    const originalPoints = inputData.map(d => ({
        x: d.horsepower, y: d.mpg
    }));

    tfvis.render.scatterplot(
        {name: 'Model Predictions vs. Original Data'},
        {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );
}
```

* `.dataSync()` is a method we can use to get a `typearray` of the values stored in a tensor. This allows us to process those values in regular JavaScript. This is a synchronous version of the `.data()` method which is generally preferred.

Add to `run`:
```javascript
// Make some predictions using the model and compare them to the original data
testModel(model, data, tensorData);
```
