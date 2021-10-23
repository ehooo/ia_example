FROM tensorflow/serving
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV MODEL_NAME=resnet
ENV TENSORFLOW_PORT=8500
ENV TENSORFLOW_API_REST_PORT=8501
ENV TENSORFLOW_ARGS=""
ENV TENSORFLOW_HOST="localhost"

EXPOSE 8000

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
ADD ./api_entrypoint.sh /usr/bin/.
COPY ./src .

ENTRYPOINT ["bash"]
CMD ["/usr/bin/api_entrypoint.sh"]
