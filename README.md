# LearningTensorFlow

## Installing TensorFlow

https://www.tensorflow.org/install

### Run a TensorFlow container

```
# Download latest stable image
$ docker pull tensorflow/tensorflow:latest-py3
# Start Jupyter server
$ docker run -it -p 8888:8888 -e PASSWORD=pass tensorflow/tensorflow:latest-py3-jupyter
```

Then you can access localhost:8888

Token will show up in the terminal so you can reset the password.

## TensorFlow Essentials

NumPy - facilitates Mathematical manipulation in Python.

### Ensuring that TensorFlow works

```python
# convention to use tf as alias
import tensorflow as tf
```

### Representing tensors

An ordered list of features is called a **feature vector**.

Vector containing two elements, and each element corresponds to a row of the matrix.
```
# Matrix representation
[[1, 2, 3], [4, 5, 6]]
```

A **tensor** is a generalization of a matrix that specifies an element by an arbitrary number of indices.

The rank of a tensor is the number of indices required to specify an element.

```python
import tensorflow as tf
import numpy as np

m1 = [[1.0, 2.0], [3.0, 4.0]]
m2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
m3 = tf.constant([[1.0, 2.0], [3.0, 4.0]])

print(type(m1))
print(type(m2))
print(type(m3))

# <class 'list'>
# <class 'numpy.ndarray'>
# <class 'tensorflow.python.framework.ops.EagerTensor'>

t1 = tf.convert_to_tensor(m1, dtype=tf.float32)
t2 = tf.convert_to_tensor(m2, dtype=tf.float32)
t3 = tf.convert_to_tensor(m3, dtype=tf.float32)

print(type(t1))
print(type(t2))
print(type(t3))

# <class 'tensorflow.python.framework.ops.EagerTensor'>
# <class 'tensorflow.python.framework.ops.EagerTensor'>
# <class 'tensorflow.python.framework.ops.EagerTensor'>
```

`tf.convert_to_tensor()` is a convenient function you can use to make sure you're dealing with tensors.
Most functions in the TensorFlow library already perform this function.

```python
import tensorflow as tf

m1 = tf.constant([[1., 2.]])
m2 = tf.constant([[1], [2]])
m3 = tf.constant([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])

print(m1)
print(m2)
print(m3)

# tf.Tensor([[1. 2.]], shape=(1, 2), dtype=float32)
# tf.Tensor(
# [[1]
#  [2]], shape=(2, 1), dtype=int32)
# tf.Tensor(
# [[[ 1  2]
#   [ 3  4]
#   [ 5  6]]

#  [[ 7  8]
#   [ 9 10]
#   [11 12]]], shape=(2, 3, 2), dtype=int32)
```

`tf.zeros(shape)` - creates a tensor with all values initialized at zero of a specific shape.

`tf.ones(shape)` - creates a tensor of a specific shape with all values initialized at once.

`shape` - is a one dimension tensor of type int32 describing the dimensions of the tensor.

```python
tf.ones([500, 500]) * 0.5
```

### Creating operators

```python
import tensorflow as tf

x = tf.constant([[1, 2]])
negMatrix = tf.negative(x)
print(negMatrix)

# tf.Tensor([[-1 -2]], shape=(1, 2), dtype=int32)
```

#### useful Tensorflow operators

* `tf.add(x, y)`
* `tf.subtract(x, y)`
* `tf.multiply(x, y)`
* `tf.pow(x, y)`
* `tf.exp(x)`
* `tf.sqrt(x)`
* `tf.div(x, y)`
* `tf.truediv(x, y)` - casts arguments as float
* `tf.floordiv(x, y)` - rounds down final answer
* `tf.mod(x, y)`

### Writing code in Jupyter

Jupyter is a web application that displays computation so that you can share annotated interactive algorithms with others to teach a technique or demonstrate code.

(Start a Jupyter web app running the command at the top)

Everything in the Jupyter Notebook is an independent chunk of code or text called **cell**. Cells help divide a long block of code into manageable pieces of code snippets and documentation.

### Using variables

You can use the `Variable` class to represent a node whose value changes over time.

```python
import tensorflow as tf
sess = tf.InteractiveSession()

raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]
spike = tf.Variable(False)
spike.initializer.run()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i - 1] > 5:
        updater = tf.assign(spike, True)
        updater.eval()
    else:
        tf.assign(spike, False).eval()
    print("Spike", spike.eval())

sess.close()
```
