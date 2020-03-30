# LearningTensorFlow

## Installing TensorFlow

https://www.tensorflow.org/install

### Run a TensorFlow container

```
# Download latest stable image
$ docker pull tensorflow/tensorflow:latest-py3
# Start Jupyter server
$ docker run -it -p 8888:8888 tensorflow/tensorflow:latest-py3-jupyter
```

Then you can access localhost:8888