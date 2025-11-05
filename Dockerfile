# USAGE:
#docker build -t causalchange .

FROM continuumio/miniconda3:latest

ENV ENV_NAME=causalchange
ENV VENV_PATH="/opt/$ENV_NAME"

RUN apt-get update && apt-get install -y \
    r-base r-base-dev \
    && rm -rf /var/lib/apt/lists/*

RUN conda create -n $ENV_NAME python=3.10 -y
SHELL ["conda", "run", "-n", "causalchange", "/bin/bash", "-c"]
RUN conda install -n $ENV_NAME -y jupyter ipykernel
RUN conda install -n $ENV_NAME -c conda-forge rpy2

COPY /requirements.txt /tmp/requirements.txt
RUN conda run -n $ENV_NAME pip install -r /tmp/requirements.txt
RUN conda run -n $ENV_NAME pip install matplotlib && \
    conda run -n $ENV_NAME python -c "import matplotlib; print(matplotlib.__version__)"

RUN conda run -n $ENV_NAME R -e "install.packages('flexmix', repos='http://cran.r-project.org')"
RUN conda run -n $ENV_NAME python -m ipykernel install --user --name $ENV_NAME --display-name "Python 3 ($ENV_NAME)"

#RUN conda run -n $ENV_NAME pip freeze #to show all packages
# the workspc is the whole directory:
#EXPOSE 8888
#COPY . /workspace/

WORKDIR /workspace
EXPOSE 8888

CMD ["conda", "run", "--no-capture-output", "-n", "causalchange", \
     "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.root_dir=/workspace"]

#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
