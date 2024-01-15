# Base image
FROM python:3.11.5-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# directory
WORKDIR /

# copyting eseentials
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY some_project/ some_project/
COPY data.dvc data.dvc

# run commands
RUN pip install . --no-deps --no-cache-dir
RUN pip install dvc
RUN pip install dvc['gs']
RUN dvc pull

# entrypoint
ENTRYPOINT ["python", "-u", "some_project/train_model.py"]
