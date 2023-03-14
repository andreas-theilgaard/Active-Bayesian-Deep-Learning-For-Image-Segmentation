# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy setup env
COPY src /src/
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py


WORKDIR /
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt --no-cache-dir
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

ENTRYPOINT ["python", "-u", "src/models/sweep_model.py"]
