# Base image
FROM python:3.11.5-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copyting eseentials
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY some_project/ some_project/
COPY data/ data/

# directory
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# entrypoint
ENTRYPOINT ["python", "-u", "some_project/train_model.py"]
