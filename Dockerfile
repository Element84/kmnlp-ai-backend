ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}

WORKDIR /opt/src/

RUN apt-get update && \
    apt-get install -y wget=1.* curl git tree && \
    apt-get autoremove && apt-get autoclean && apt-get clean

RUN python -m pip install --upgrade pip
RUN pip install uv
COPY ./requirements.txt /opt/src/requirements.txt
RUN uv pip install -r requirements.txt --system

COPY ./ /opt/src/

ENV AWS_DEFAULT_REGION="us-east-1"

CMD ["bash"]
