#Docker container to start the Pipeline
#Using a nvidia/cuda image to use the GPU for the Pipeline
#Using ubuntu 24.04 to have python 3.12.3
#Ubuntu24.04 uses PEP668 so we need to use envirements
FROM nvidia/cuda:12.6.1-base-ubuntu24.04

RUN apt-get update && apt-get install -y \
    git wget curl unzip build-essential python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY requirements.txt requirements_moose.txt pyproject.toml setup.py README.md ./
COPY src/ ./src/

# Clone the needed reposetories
RUN git clone https://github.com/mic-dkfz/autopet-3-submission \
    && curl -L -o autoPET-3-LesionTracer.zip "https://zenodo.org/records/14007247/files/autoPET-3-LesionTracer.zip?download=1" \
    && unzip autoPET-3-LesionTracer.zip -d ./autopet-3-model/ \
    && rm autoPET-3-LesionTracer.zip

# Inatialize main project
RUN python3 -m venv /app/.venv \
    && /app/.venv/bin/pip install --upgrade pip setuptools wheel setuptools_scm

# Set the venv-python as default python
ENV PATH="/app/.venv/bin:$PATH"

RUN pip install -r requirements.txt && pip cache purge

# Initalize moosez
# RUN python3 -m venv /app/.venv_moose \
#    && /app/.venv_moose/bin/pip install --upgrade pip setuptools wheel setuptools_scm \
#    && /app/.venv_moose/bin/pip install -r requirements_moose.txt \
#    && /app/.venv_moose/bin/pip install moosez --no-deps \
#    && /app/.venv_moose/bin/pip cache purge
