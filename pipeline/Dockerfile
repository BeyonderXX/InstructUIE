FROM hub.byted.org/base/lab.pytorch:latest

# Install PYPI packages

ENV https_proxy=http://sys-proxy-rd-relay.byted.org:8118
ENV http_proxy=http://sys-proxy-rd-relay.byted.org:8118

COPY . /llm_finetune

RUN \
	python3 -m pip install --upgrade pip setuptools wheel --trusted-host bytedpypi.byted.org -i https://bytedpypi.byted.org/simple/ && \
    python3 -m pip install --no-cache-dir transformers==4.24.0 --trusted-host bytedpypi.byted.org -i https://bytedpypi.byted.org/simple/ && \
    python3 -m pip install --no-cache-dir -r /llm_finetune/Megatron-DeepSpeed/requirements.txt && \
    python3 -m pip install --no-cache-dir mpi4py==3.1.4

RUN \
    sudo apt-get update && \
    sudo apt install libaio-dev 

RUN \
    mkdir -p /opt/tiger/apex__ && \
    cd /opt/tiger/apex__ && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    python3 -m pip install --target=/usr/local/lib/python3.7/dist-packages/ --target=/usr/local/lib/python3.7/dist-packages/ -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  --disable-pip-version-check ./ 

RUN \
    python3 -m pip install --no-cache-dir deepspeed

ENV https_proxy=
ENV http_proxy=
