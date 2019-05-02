FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG NVIDIA_DRIVER=nvidia-driver
ARG NVIDIA_DRIVER_VER=415
ARG NVIDIA_DRIVER_MINOR_VER=27
ARG OPENCV_VER=4.0.1

WORKDIR /root
CMD ["/bin/bash"]

# Install a base time build environment
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
	apt-utils \
	build-essential \
	cmake \
	ctags \
	curl \
	git \
	software-properties-common \
	tar \
	unzip \
	wget \
	vim && \
    apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# We could use nvidia-docker and probably should. But instead we will just
# install the same version as our host system so driver version match.
RUN echo "yes" | add-apt-repository ppa:graphics-drivers/ppa
RUN apt-get update && apt-get install -y \
	${NVIDIA_DRIVER}-${NVIDIA_DRIVER_VER} && \
	apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

# Ensure CUDA is in our run-time loader's path
RUN cd /usr/local && echo '/usr/local/cuda/lib64' > /etc/ld.so.conf.d/cuda.conf && ldconfig

# Install some base dependencies for various modules below.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
	dirmngr \
	gfortran \
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
	liblapacke-dev \
	libleveldb-dev \
	liblmdb-dev \
	libpng-dev \
	libprotobuf-dev \
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

# Get the latest pip3
RUN wget -q https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm -f get-pip.py

# Install python3 infrastructure
RUN pip3 --no-cache-dir install \
	click \
	cntk-gpu \
	Cython \
	bs4 \
	nltk \
	pandas \
	numpy \
	seaborn \
	matplotlib \ 
	scikit-learn \
	scikit-image \
	imutils \
	jupyter \
	ipython \
	h5py \
	nose \
	sympy \
	mahotas \
	ndg-httpsclient \
	pyasn1 \
	keras \
	path.py \
	Pillow \
	pygments \ 
	pyopenssl \
	six \
	sphinx \
	tensorflow-gpu \
	wheel \
	wordcloud \
	zmq

# Install Intel DNN
RUN git clone https://github.com/intel/mkl-dnn.git && \
	cd mkl-dnn/scripts && ./prepare_mkl.sh && \
	cd .. && mkdir -p build && cd build && cmake .. && \
	make && make install

# Install XGBoost
RUN git clone --recursive https://github.com/dmlc/xgboost && \
	cd xgboost && mkdir build && cd build && \
	cmake .. -DUSE_CUDA=ON && make && \
	cd /root/xgboost && cd python-package && python3 setup.py install

# Install OpenCV
RUN wget -q -O opencv-${OPENCV_VER}.zip https://github.com/opencv/opencv/archive/${OPENCV_VER}.zip && \
	wget -q -O opencv_contrib-${OPENCV_VER}.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VER}.zip && \
	unzip opencv-${OPENCV_VER}.zip && unzip opencv_contrib-${OPENCV_VER}.zip && \
	cd /root/opencv-${OPENCV_VER} && mkdir build && cd build && \
	cmake -DWITH_OPENGL=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES --expt-relaxed-constexpr" -DOPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib-${OPENCV_VER}/modules .. && \
	make && make install && ldconfig && \
    	echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc && rm -rf /root/opencv*

# Install OpenMPI
RUN wget -q https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz && tar -zxf openmpi-4.0.0.tar.gz && \
	cd openmpi-4.0.0 && ./configure --prefix=/usr/local/mpi && \
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
RUN wget -q -O spark.tgz http://mirror.olnevhost.net/pub/apache/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz && \
	mkdir -p /usr/local && tar -xf spark.tgz -C /usr/local && \
    	mv /usr/local/spark*2.4.0* /usr/local/spark && \
    	pip3 install --upgrade pyspark

# Install various odds and ends and make sure modules are
# the latest (some packages above may have pulled in older
# dependencies and we need to upgrade them now).
RUN pip3 install torch_nightly torchvision-nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html
RUN pip3 install --upgrade fake_useragent click ipython bokeh flask pytest coverage gunicorn
RUN pip3 install --upgrade bcolz theano kaggle-cli pymc3 animation graphviz
RUN pip3 install --upgrade numpy notebook jupyter jupyterlab scipy kaggle urllib3 spacy spacy[cuda92] gensim plotly quandl fastai tornado==5.1.1 ipython-sql
RUN pip3 install --upgrade awscli mwparserfromhell
RUN pip3 install --upgrade jupytext
RUN jupyter serverextension enable --py jupyterlab --sys-prefix
RUN jupyter nbextension install --py jupytext && jupyter nbextension enable --py jupytext

# Download all spacy embeddings
RUN python3 -m spacy download en
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download en_core_web_md
RUN python3 -m spacy download en_core_web_lg
RUN python3 -m spacy download en_vectors_web_lg

# Install nltk 
RUN python3 -c "import nltk; nltk.download('all')"

# Allow plotly and bokeh to display in JupyterLab
RUN jupyter labextension install @jupyterlab/plotly-extension
RUN jupyter labextension install jupyterlab_bokeh

# Install Heroku
RUN curl https://cli-assets.heroku.com/install-ubuntu.sh | sh

# Upgrade to 2.0
RUN pip3 install tensorflow-gpu==2.0.0-alpha0

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
ENV PYTHONPATH=/usr/local/spark/python:/usr/local/spark/python/lib/py4j-0.10.4-src.zip
ENV PATH=/usr/local/spark/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/spark/bin

EXPOSE 4040 6006 8888
