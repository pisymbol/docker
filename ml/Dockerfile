FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG OPENCV_VER=4.2.0
ARG SPARK_VER=2.4.4
ARG OPENMPI_VER=4.0.2

WORKDIR /root
CMD ["/bin/bash"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Install a base time build environment
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
	apt-utils \
	build-essential \
	ctags \
	curl \
	git \
	software-properties-common \
	sudo \
	tar \
	unzip \
	wget \
	vim && \
    apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Ensure CUDA is in our run-time loader's path
RUN cd /usr/local && echo '/usr/local/cuda/lib64' > /etc/ld.so.conf.d/cuda.conf && ldconfig

# Install some base dependencies for various modules below.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
	dirmngr \
	gfortran \
	liblas-dev \
	liblapack3 \
	liblapacke \
	liblapacke-dev \
	libatlas3-base \
	libavcodec-dev \
	libavformat-dev \
	libavresample-dev \
	libboost-all-dev \
	libdc1394-22-dev \
	libhdf5-serial-dev \
	libgdal-dev \
	libgflags-dev \
	libgoogle-glog-dev \
	libgphoto2-dev \
	libgtk2.0-dev \
	libjpeg-dev \
	libjpeg8-dev \
	liblapack-dev \
	libleveldb-dev \
	liblmdb-dev \
	libpng-dev \
	libprotobuf-dev \
	libspatialindex-dev \
	libsnappy-dev \
	libswscale-dev \
	libtbb2 \
	libtbb-dev \
	libtiff-dev \
	libtiff5-dev \
	libv4l-dev \
	libxine2-dev \
	nodejs \
	npm \
	pkg-config \
	protobuf-compiler \
	python3 \
	python3-dev \
	unzip && \
    apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.2/cmake-3.16.2.tar.gz && tar -xvf cmake*tar.gz && cd cmake* && ./bootstrap && make && make install

# Get the latest pip3
RUN wget -q https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm -f get-pip.py

# Install python3 infrastructure
COPY requirements.txt .
RUN pip3 --no-cache-dir install -r requirements.txt && rm -f requirements.txt

# Install Intel DNN
RUN git clone https://github.com/intel/mkl-dnn.git && \
	cd mkl-dnn && mkdir -p build && cd build && cmake .. && \
	make -j4 && make install

# Install XGBoost
RUN git clone --recursive https://github.com/dmlc/xgboost && \
	cd xgboost && mkdir build && cd build && \
	cmake .. -DUSE_CUDA=ON && make && \
	cd /root/xgboost && cd python-package && python3 setup.py install

# Install OpenCV
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev tesseract-ocr libtesseract-dev liblapack-dev
RUN wget -q -O opencv-${OPENCV_VER}.zip https://github.com/opencv/opencv/archive/${OPENCV_VER}.zip && \
	wget -q -O opencv_contrib-${OPENCV_VER}.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VER}.zip && \
	unzip opencv-${OPENCV_VER}.zip && unzip opencv_contrib-${OPENCV_VER}.zip && \
	cd /root/opencv-${OPENCV_VER} && mkdir build && cd build && \
	cmake -DWITH_OPENGL=ON -DWITH_GSTREAMER=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES --expt-relaxed-constexpr" -DOPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib-${OPENCV_VER}/modules .. && \
	make && make install && ldconfig && \
    	echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc && rm -rf /root/opencv*

# Install OpenMPI
RUN wget -q https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-${OPENMPI_VER}.tar.gz && tar -zxf openmpi-${OPENMPI_VER}.tar.gz && \
	cd openmpi-${OPENMPI_VER} && ./configure --prefix=/usr/local/mpi && \
    	make && make install

# Need to use gcc-7 for OpenCV compilation (this may change)
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 50 && \
	update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 50 && \
	update-alternatives --config gcc && update-alternatives --config g++

# Install Spark dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
	openjdk-8-jre \
	scala && \
	apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/java java /usr/lib/jvm/java-8-openjdk-amd64/bin/java 10000

# Install Spark
RUN wget -q -O spark.tgz http://mirror.olnevhost.net/pub/apache/spark/spark-${SPARK_VER}/spark-${SPARK_VER}-bin-hadoop2.7.tgz && \
	mkdir -p /usr/local && tar -xf spark.tgz -C /usr/local && \
	mv /usr/local/spark*${SPARK_VER}* /usr/local/spark && \
    	pip3 install --upgrade pyspark

# Download all embeddings/tokenizers up front
RUN python3 -m spacy download en
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download en_core_web_md
RUN python3 -m spacy download en_core_web_lg
RUN python3 -m spacy download en_vectors_web_lg
RUN python3 -c "import nltk; nltk.download('all')"

# Setup jupyter extensions
RUN jupyter serverextension enable --py jupyterlab --sys-prefix
RUN jupyter nbextension install --py jupytext && jupyter nbextension enable --py jupytext
RUN jupyter labextension install @jupyterlab/plotly-extension
RUN jupyter labextension install jupyterlab_bokeh

# Clean up
RUN rm -rf /root/* && chmod 755 /root

# Install my complete custom home environment under a container
RUN useradd -ms /bin/bash -d /home/pisymbol -p $(openssl passwd -1 docker) -G sudo pisymbol
USER pisymbol

# Shell
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/lib
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/spark/bin

RUN echo "set -o vi" >> /home/pisymbol/.bashrc
RUN echo "alias nb='jupyter notebook --no-browser --allow-root --port 8888 --NotebookApp.iopub_data_rate_limit=10000000000'" >> /home/pisymbol/.bashrc
RUN echo "alias python='python3'" >> /home/pisymbol/.bashrc
RUN echo "export LC_ALL=C.UTF-8" >> /home/pisymbol/.bashrc
RUN echo "export LANG=C.UTF-8" >> /home/pisymbol/.bashrc
RUN echo "export SPARK_HOME=/usr/local/spark" >> /home/pisymbol/.bashrc
RUN echo "export PYSPARK_PYTHON=python3" >> /home/pisymbol/.bashrc
RUN echo "export HADOOP_OPTS=\"$HADOOP_OPTS -Djava.library.path=$HADOOP_HOME/lib/native\"" >> /home/pisymbol/.bashrc
COPY --chown=pisymbol vimrc /home/pisymbol/.vimrc
COPY --chown=pisymbol vim /home/pisymbol/.vim
RUN curl -q https://raw.githubusercontent.com/goerz/jupytext.vim/master/plugin/jupytext.vim > ~/.vim/plugin/jupytext.vim

# Git settings
RUN git config --global user.email "pisymbol@gmail.com"
RUN git config --global user.name "Alexander Sack"

RUN jupyter notebook --generate-config
RUN sed -i -E 's,^#\s*c.NotebookApp.terminado_settings.*,c.NotebookApp.terminado_settings = {"shell_command" : ["\/bin\/bash"]},g' ~/.jupyter/jupyter_notebook_config.py
# Setup password authentication so we don't have to remember tokens
RUN echo "{ \"NotebookApp\": { \"password\": \"sha1:6c96a4e0f51a:b5416c8e9f3ce4404844ad76838fca509c4322dc\" } }" >> ~/.jupyter/jupyter_notebook_config.json
RUN mkdir -p /home/pisymbol/.ipython/profile_default
RUN echo "c.TerminalInteractiveShell.editing_mode = 'vi'" >> /home/pisymbol/.ipython/profile_default/ipython_config.py

# Kaggle setup
RUN mkdir -p /home/pisymbol/.kaggle
COPY --chown=pisymbol:pisymbol kaggle.json /home/pisymbol/.kaggle
RUN mkdir -p /home/pisymbol/Development/kaggle
RUN kaggle config set -n path -v ~/kaggle
RUN sed -i -E 's,kaggle,Development/kaggle,g' /home/pisymbol/.kaggle/kaggle.json

# Required spark configuration for local user access
ENV SPARK_HOME=/usr/local/spark
ENV PYTHONPATH=/usr/local/spark/python:/usr/local/spark/python/lib/py4j-0.10.7-src.zip
ENV PATH=/usr/local/spark/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/spark/bin

EXPOSE 4040 6006 8888 8080 8081
