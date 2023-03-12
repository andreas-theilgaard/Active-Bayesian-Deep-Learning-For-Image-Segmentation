# Base image
FROM nvcr.io/nvidia/pytorch:22.07-py3
FROM python:3.10-slim


# Install Python
RUN apt update && \
   apt-get install --no-install-recommends -y build-essential gcc curl && \
   apt clean && rm -rf /var/lib/apt/lists/*

# copy setup env
COPY src /src/
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
COPY credentials.json /credentials.json
COPY client_secrets.json /client_secrets.json
COPY settings.yaml /settings.yaml

WORKDIR /
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir

ENTRYPOINT ["python", "-u", "src/experiments/compare_methods.py"]
