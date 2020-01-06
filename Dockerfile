FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y
# RUN apt-get install -y python3-dev python3-pip
RUN apt-get update --fix-missing
RUN apt-get install -y wget bzip2 ca-certificates git vim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        premake4 \
        git \
        curl \
        vim \
        libav-tools \
	    libgl1-mesa-dev \
	    libgl1-mesa-glx \
	    libglew-dev \
	    libosmesa6-dev \
	    libxrender-dev \
	    libsm6 libxext6 \
        unzip \
        patchelf \
        ffmpeg \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglew1.6-dev mesa-utils
        
# Not sure why this is needed
ENV LANG C.UTF-8

# Not sure what this is fixing
# COPY ./files/Xdummy /usr/local/bin/Xdummy
# RUN chmod +x /usr/local/bin/Xdummy
        
##########################################################
### MuJoCo
##########################################################
# Note: ~ is an alias for /root
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro131_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
### Lets not put the key in the image for now.
# COPY ./files/mjkey.txt /root/.mujoco/mjkey.txt
# COPY .mujoco /root/.mujoco
COPY ./vendor/mujoco/mjkey.txt /root/.mujoco/
RUN ln -s /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco130/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}


ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

RUN conda update -y --name base conda && conda clean --all -y

RUN conda create --name gmps python=3.6 pip
RUN echo "source activate gmps" >> ~/.bashrc

# RUN pip3 install scikit-image

ENV PATH /opt/conda/envs/railrl/bin:$PATH

RUN pip install tensorflow==1.14.0
RUN pip install joblib==0.10.3
RUN pip install cached-property==1.5.1
RUN pip install mako==1.0.7
RUN pip install lasagne==0.1
RUN pip install theano==0.7.0
RUN pip install pydot-ng
RUN pip install Cython
RUN pip install lockfile
RUN pip install glfw
RUN pip install cffi
RUN pip install python-dateutil==2.7.3
RUN pip install pyprind==2.11.2
RUN pip install opencv-python
RUN pip install gym
RUN pip install matplotlib
RUN pip install Cython
RUN pip install lockfile
RUN pip install glfw
RUN pip install cffi
RUN pip install numpy-stl
RUN pip install tensorboard_logger
RUN pip install pyOpenGL
RUN pip install pyquaternion
RUN pip install mujoco-py

RUN python -c 'import mujoco_py'

RUN cd /root
RUN git clone https://github.com/rll/rllab.git /root/rllab
RUN git clone https://github.com/russellmendonca/R_multiworld /root/R_multiworld
# RUN pushd rllab
# RUN ./scripts/setup_linux.sh
# RUN popd
ENV PYTHONPATH /root/rllab/:${PYTHONPATH}
ENV PYTHONPATH /root/R_multiworld/:${PYTHONPATH}
ENV PYTHONPATH /root/GMPS/:${PYTHONPATH}

RUN git clone https://github.com/Neo-X/GMPS.git /root/GMPS

# RUN pushd GMPS
RUN ls
RUN pwd
RUN mkdir /root/GMPS/vendor/mujoco
RUN cp ~/.mujoco/mjpro131/bin/*.so* /root/GMPS/vendor/mujoco/
RUN ls /root/GMPS/vendor/mujoco/
COPY ./vendor/mujoco/mjkey.txt /root/GMPS/vendor/mujoco/

RUN ls
