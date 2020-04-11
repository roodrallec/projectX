FROM nvidia/cuda:9.0-base-ubuntu16.04 as build

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential zlib1g-dev libssl-dev

COPY lib lib

RUN cd lib && \
    tar -xvf Python-3.6.3.tgz && \
    cd Python-3.6.3 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    apt-get install -y python-opencv

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

FROM nvidia/cuda:9.0-base-ubuntu16.04

WORKDIR /

COPY --from=build /opt/venv /opt/venv
ENV VIRTUAL_ENV /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src .
COPY models .
COPY init.sh /usr/local/bin

EXPOSE 2222
EXPOSE 8888

ENTRYPOINT ["init.sh"]
