#!/bin/bash

if [[ "$TENSORFLOW_HOST" = "localhost" ]]; then
  echo "Starting TensorFlow Service"
  # Setup tensorflow service
  tensorflow_model_server --port=${TENSORFLOW_PORT} --rest_api_port=${TENSORFLOW_API_REST_PORT} --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} ${TENSORFLOW_ARGS} &
fi
# Setup custom API
uvicorn main:app --host=0.0.0.0 --port=8000 ${API_ARGS}
