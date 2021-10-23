# IA Example
Based on [TensorFlow Serving](https://blog.tensorflow.org/2018/11/serving-ml-quickly-with-tensorflow-serving-and-docker.html) tutorial.

## Create docker image
* With Docker
```shell
docker build . --tag ia_example:latest
```
* With Docker-compose
```
docker-compose build
```

## Docker Environment
* `TENSORFLOW_PORT`(By default `8500`)

    Allows to change the TensorFlow port

* `TENSORFLOW_API_REST_PORT`(By default `8501`)

    Allows to change the TensorFlow Rest API port

* `TENSORFLOW_HOST`(By default `localhost`)
    
    In order to allow internal or external TensorFlow services 
you could use that var to change the host used to process the prediction

* `TENSORFLOW_ARGS`(Empty by default)

    If `TENSORFLOW_HOST="localhost"` the args passed to `tensorflow_model_server` bin

