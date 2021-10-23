FROM tensorflow/serving
ENV MODEL_NAME=resnet

WORKDIR /code

RUN apt update && \
    apt install curl python3 python3-pip -y && \
    # Get models
    mkdir /models/resnet && \
    curl -s https://storage.googleapis.com/download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz | tar --strip-components=2 -C /models/resnet -xvz && \
    # Pillow libs
    apt install zlib1g-dev libjpeg8-dev -y && \
    mkdir /tools && \
    curl -o /tools/resnet_client.py https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/resnet_client.py

ADD ./requirements.txt .
RUN pip3 install -r requirements.txt
COPY ./src .
