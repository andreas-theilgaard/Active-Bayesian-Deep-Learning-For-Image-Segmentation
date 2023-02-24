# Base image
FROM nvcr.io/nvidia/pytorch:22.07-py3
FROM python:3.10-slim


# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# install Google Cloud SDK
RUN curl https://sdk.cloud.google.com > install.sh
RUN bash install.sh --disable-prompts

# changing RUN commando to run bash instead of sh
SHELL ["/bin/bash", "-c"]
COPY key.json key.json
RUN /root/google-cloud-sdk/bin/gcloud auth activate-service-account --key-file=key.json
RUN /root/google-cloud-sdk/bin/gsutil -m cp -r gs://data_bachelor_buck/ .
RUN mv data_bachelor_buck/data /data
RUN rm -r data_bachelor_buck

# copy setup env
COPY src /src/
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py


WORKDIR /
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir


ENTRYPOINT ["python", "-u", "src/experiments/compare_methods.py"]
