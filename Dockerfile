FROM nvidia/cuda:10.2-base-ubuntu16.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
# Install base packages.
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago
RUN apt-get update --fix-missing && apt-get install -y tzdata && apt-get install -y bzip2 ca-certificates curl gcc git libc-dev libglib2.0-0 libsm6 libxext6 libxrender1 wget libevent-dev build-essential &&  rm -rf /var/lib/apt/lists/*
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN /opt/conda/bin/conda update -n base -c defaults conda && \
    /opt/conda/bin/conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch && \
    /opt/conda/bin/conda install pip spacy && \
    /opt/conda/bin/pip install transformers==4.14.1 && \
    /opt/conda/bin/pip install flask && \
    /opt/conda/bin/pip install flask-cors flask_bootstrap flask_moment flask_wtf && \
    /opt/conda/bin/pip install tornado && \
    /opt/conda/bin/pip install nltk && \
    /opt/conda/bin/pip install tqdm && \
    /opt/conda/bin/pip install scipy==1.6.2 && \
    /opt/conda/bin/pip install cython && \
    /opt/conda/bin/pip install scikit-learn==0.22 && \
    /opt/conda/bin/python -m spacy download en_core_web_sm
ADD ./apidemo /apidemo
COPY run.sh /run.sh

# RUN /opt/conda/bin/conda clean -tipsy
LABEL maintainer="hengji@illinois.edu"
CMD ["/bin/bash", "run.sh"]