FROM tensorflow/tensorflow
# Install TensorFlow C library

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    libc6-dev \
    make \
    pkg-config \
    wget \
    git \
    curl \
  && rm -rf /var/lib/apt/lists/*

RUN curl -L \
   "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz" | \
   tar -C "/usr/local" -xz

RUN ldconfig

# Hide some warnings
ENV TF_CPP_MIN_LOG_LEVEL 2

RUN wget https://dl.google.com/go/go1.12.7.linux-amd64.tar.gz

RUN tar -xvf go1.12.7.linux-amd64.tar.gz

RUN mv go /usr/local

ENV GOPATH /go

ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"

RUN export GO111MODULE=on

RUN apt-get update

RUN apt-get install unzip

RUN mkdir -p /model && \
  wget "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip" -O /model/inception.zip && \
  unzip /model/inception.zip -d /model && \
  chmod -R 777 /model

WORKDIR "/go/src/github.com/lyledean1/inception/"

COPY . .

RUN go build GOOS=linux GOARCH=arm GOARM=5 -o tfgo ./cmd/main.go


CMD [ "./tfgo" ]