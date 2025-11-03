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
EXPOSE 8888
# Default command
COPY /src/mixtures /workspace/
CMD ["conda", "run", "-n", "causalchange", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
#docker build -t causalchange-jupyter .