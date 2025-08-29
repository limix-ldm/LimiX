# first download https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# docker build --network=host -t limix/infe:v1 --build-arg FROM_IMAGES=nvidia/cuda:12.2.0-base-ubuntu22.04 -f Dockerfile .
# docker run -it limix/infe:v1
ARG FROM_IMAGES
FROM $FROM_IMAGES

ARG CONDAENV=LimiX
ARG TARGETARCH=amd64


# ENV TZ Asia/Shanghai

# Install system tools and Miniconda
# If apt update fails: Try to change mirror URL in /etc/apt/sources.list
# if Miniconda installation fails, change the mirror URL”
#
RUN echo $FROM_IMAGES $CONDAENV  $TARGETARCH \
    && apt update -y \
    && apt install -y -f --no-install-recommends vim rsync wget net-tools curl  locales zip unzip lsof \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  \
    && mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh


ENV PATH "/root/miniconda3/bin":$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main  --channel https://repo.anaconda.com/pkgs/r
# Clean package caches
# RUN pip cache purge  && conda clean --all -y
# Conda configuration file
# COPY .condarc /root/.condarc
COPY flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# ENV CONDARC /root/.condarc
RUN conda create -y -n ${CONDAENV} python=3.12.7
ENV PATH /root/miniconda3/envs/${CONDAENV}/bin:$PATH
ENV CONDA_DEFAULT_ENV ${CONDAENV}


RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1  && \
    pip install flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl && \
    pip install scikit-learn  einops  huggingface-hub matplotlib networkx numpy pandas  scipy tqdm typing_extensions xgboost

# Environment configuration
RUN echo "source /root/miniconda3/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate ${CONDAENV}" >> /root/.bashrc && \
    echo "export PATH=/root/miniconda3/envs/${CONDAENV}/bin:\$PATH" >> /root/.bashrc

# default conda env
CMD ["/bin/bash", "-c", "source /root/.bashrc && exec /bin/bash"]


