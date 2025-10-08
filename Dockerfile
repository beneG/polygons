FROM amd64/ubuntu:22.04

ENV TZ="Europe/Moscow"
ENV DEBIAN_FRONTEND="noninteractive"


RUN apt-get update && apt-get install -y build-essential g++ cmake git wget

RUN apt-get install -y libopencv-dev

WORKDIR /app

RUN mkdir -p data/models && \
    wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg -O data/models/yolov3.cfg && \
    wget https://data.pjreddie.com/files/yolov3.weights -O data/models/yolov3.weights


COPY ./CMakeLists.txt ./CMakeLists.txt
COPY ./proto ./proto

RUN mkdir -p src && \
    echo 'int main() { return 0; }' > src/server.cpp && \
    echo 'int main() { return 0; }' > src/client.cpp && \
    echo 'int main() { return 0; }' > src/yolo_detector.cpp

RUN cmake . && make grpc grpc++ libprotobuf grpc_cpp_plugin -j$(nproc)

COPY ./src ./src

RUN make -j$(nproc)

