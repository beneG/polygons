FROM amd64/ubuntu:22.04

ENV TZ="Europe/Moscow"

RUN apt-get update && apt-get install -y build-essential g++ cmake git wget 

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get install -y libopencv-dev
RUN apt-get install -y git
RUN apt-get install -y libabsl-dev

WORKDIR /app

COPY ./CMakeLists.txt ./CMakeLists.txt
COPY ./proto ./proto
COPY ./src ./src

RUN cmake . && make -j7

RUN mkdir -p data/models
RUN wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg -O data/models/yolov3.cfg
RUN wget https://data.pjreddie.com/files/yolov3.weights -O data/models/yolov3.weights
