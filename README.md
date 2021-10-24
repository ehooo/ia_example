# IA Example
Middle API using [TensorFlow Serving](https://blog.tensorflow.org/2018/11/serving-ml-quickly-with-tensorflow-serving-and-docker.html).

## Documentation
See web documents accessing to [/doc](http://localhost:8000/doc) or [/redoc](http://localhost:8000/redoc)

Single entry-point [/api/v1/resnet](http://localhost:8000/api/v1/resnet)

## QA
You could test the code executing `pytest --cov=main` or `docker run -h --rm -i -t ia_example:latest -c pytest --cov=main`,
also the code quality with `flake8`

# Build
## Create docker image
* With Docker
```shell
docker build . --tag ia_example:latest
```
* With Docker-compose
```shell
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

* `API_ARGS` (Empty by default)

    Args used to `uvicorn`

# Deploy
* With Docker
```shell
docker run -d -p 8000:8000 --name ia_example -e MODEL_NAME=resnet -t ia_example:latest
```
* With Docker-compose
  * __NOTE :__ In docker compose will be deployed two containers,
one with the API other with TensorFlow.
```shell
docker-compose up -d
```
