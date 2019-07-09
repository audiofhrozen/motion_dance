# The python version installed in the conda setup
PYTHON_VERSION := 3.7
CUPY_VERSION := 6.0.0
CUDA_VERSION := 10.0

# Both Miniconda2/3 can install any Python versions
CONDA_URL := https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
GCC_VERSION := $(shell gcc -dumpversion)

ifneq ($(shell which nvidia-smi),) # 'nvcc' found
CONDA_PYTORCH := cudatoolkit=$(CUDA_VERSION)
CUDA_DEPS := cupy.done
else
CONDA_PYTORCH := 
CUDA_DEPS :=
endif

.PHONY: all clean

all: python

python: venv $(CUDA_DEPS)

miniconda.sh:
	test -f miniconda.sh || wget $(CONDA_URL) -O miniconda.sh
venv: miniconda.sh
	test -d $(PWD)/venv || bash miniconda.sh -b -p $(PWD)/venv
	. venv/bin/activate && conda update -y conda
	. venv/bin/activate && conda install -y python=$(PYTHON_VERSION)
	. venv/bin/activate && conda info -a
deepdancer.done: venv
	. venv/bin/activate && conda install -y matplotlib
	. venv/bin/activate && conda install -y h5py
	. venv/bin/activate && pip install -e ..
	touch deepdancer.done

cupy.done: deepdancer.done
ifneq ($(strip $(CUPY_VERSION)),)
	. venv/bin/activate && conda install -y cudatoolkit=$(CUDA_VERSION)
	. venv/bin/activate && pip install cupy==$(CUPY_VERSION)
	touch cupy.done
endif

clean:
	rm -rf venv
	rm -f miniconda.sh
	rm -rf *.done
	find . -iname "*.pyc" -delete