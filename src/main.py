import base64
import logging
import os

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(
    docs_url="/doc",
)
logger = logging.getLogger(__name__)

TENSORFLOW_URL = "http://{host}:{post}/v1/models/{model}:predict".format(
    host=os.environ.get("TENSORFLOW_HOST", "localhost"),
    post=os.environ.get("TENSORFLOW_API_REST_PORT", 8501),
    model=os.environ.get("MODEL_NAME", "resnet"),
)


@app.post(
    "/api/v1/resnet",
    responses={
        200: {
            "content": {
                "application/json": {
                    "schema": {
                        "title": "Response",
                        "type": "object",
                        "properties": {
                            "class": {
                                "title": "TensorFlow response", "type": "int",
                            }
                        },
                    },
                    "examples": {
                        "valid": {
                            "summary": "Response",
                            "value": {
                                "class": 100,
                            }
                        },
                    },
                },
            },
            "description": "Return the JSON with resnet result.",
        },
        400: {
            "content": {
                "application/json": {
                    "schema": {
                        "title": "Response",
                        "type": "object",
                        "properties": {
                            "detail": {
                                "title": "Error message", "type": "string",
                            }
                        },
                    },
                    "examples": {
                        "invalid": {
                            "summary": "Invalid content type",
                            "value": {
                                "detail": "Invalid content type",
                            },
                        }
                    },
                },
            },
            "description": "Invalid file type.",
        },
        500: {
            "content": {
                "application/json": {
                    "schema": {
                        "title": "Tensorflow error response",
                        "type": "object",
                        "properties": {
                            "detail": {
                                "title": "Error message", "type": "string",
                            }
                        },
                    },
                    "examples": {
                        "invalid": {
                            "summary": "Invalid response from TensorFlow",
                            "value": {
                                "detail": "Error processing file",
                            },
                        },
                        "connection error": {
                            "summary": "Cannot connect to TensorFlow",
                            "value": {
                                "detail": "Cannot connect to TensorFlow",
                            },
                        }
                    },
                },
            },
            "description": "Error sending data to TensorFlow.",
        }
    },
)
async def resnet(img: UploadFile = File(...)):
    if not img.content_type.startswith("image/"):
        logger.debug("Invalid file", extra={
            "Content-Type": img.content_type,
            "Filename": img.filename
        })
        raise HTTPException(status_code=400, detail="Invalid content type")

    logger.debug("Read file", extra={
        "Content-Type": img.content_type,
        "Filename": img.filename
    })
    contents = await img.read()
    jpeg_bytes = base64.b64encode(contents).decode("utf-8")

    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
    try:
        response = requests.post(TENSORFLOW_URL, data=predict_request)
    except requests.ConnectionError:
        logger.exception("Cannot connect to TensorFlow", extra={
            "url": TENSORFLOW_URL,
        })
        raise HTTPException(status_code=500, detail="Cannot connect to TensorFlow")
    logger.info("Response from TensorFlow", extra={
        "raw": response.text,
        "time": response.elapsed.total_seconds(),
        "status": response.status_code,
    })
    tf_data = response.json()
    if response.status_code != 200:
        error = tf_data.get("error", "Error processing file")
        raise HTTPException(status_code=500, detail=error)
    predictions = tf_data.get("predictions")
    if not predictions:
        raise HTTPException(status_code=500, detail="No predictions received")
    classes = predictions[0].get("classes")
    if not isinstance(classes, int):
        raise HTTPException(status_code=500, detail="Unexpected response from TensorFlow")

    return {
        "class": classes
    }

if __name__ == "__main__":
    import uvicorn
    reload = "--reload" in os.environ.get("API_ARGS", "")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
