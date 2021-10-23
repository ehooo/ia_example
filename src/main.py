import logging

from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(
    docs_url="/doc",
)
logger = logging.getLogger(__name__)


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
                            'class': {
                                'title': 'TensorFlow response', 'type': 'int',
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
                            'detail': {
                                'title': 'Error message', 'type': 'string',
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
        }
    },
)
async def resnet(img: UploadFile = File(...)):
    if not img.content_type.startswith('image/'):
        logger.debug("Invalid file {} named {}".format(img.content_type, img.filename))
        raise HTTPException(status_code=400, detail="Invalid content type")

    contents = await img.read()
    logger.debug("Read file {} named {}".format(img.content_type, img.filename))
    return {
        'class': 100
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
