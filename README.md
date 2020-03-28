# 0. SETUP & INSTALLATION:  
[Pytorch Installation](https://pytorch.org/#pip-install-pytorch)    
[Pytorch build from source](https://github.com/pytorch/pytorch#from-source)  
```
pip install opencv-python==3.4.5.20
pip install opencv-contrib-python==3.4.5.20
tree -L 1
pip install gpustat
watch -n0,2 gpustat -cp
htop -p PID
htop -u hiep
setw -g mouse on
```
Run htop, Press F5 to enter tree mode, then F4 to filter, and type in python

# 1. POSENET BASE DETECTION TEST:
[PoseNet Pytorch](https://github.com/rwightman/posenet-pytorch)  
[PoseNet Tensorflow](https://github.com/rwightman/posenet-python)  

# 2. CONDA & PYTORCH BUILD:
**GCC 6 - CUDA 10 - Python 3.6**
```
conda create -n python36_cpu python=3.6  
source activate python36_gpu  
source activate python36_cpu
conda activate python36_cpu
source activate python36_mkl_cpu
conda deactivate  
conda create -n python36_mkl_cpu python=3.6
source activate python36_mkl_gpu   
conda install numpy -c intel --no-update-deps
conda create -n micronet
```
```
sudo update-alternatives --config gcc
python setup.py bdist_wheel
```
