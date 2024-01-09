# Base image
FROM  nvcr.io/nvidia/pytorch:22.07-py3

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copyting eseentials
COPY some_project/ some_project/
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY data/ data/

# directory
WORKDIR /
RUN ls -l
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# entrypoint
ENTRYPOINT ["python", "-u", "some_project/train_model.py"]
