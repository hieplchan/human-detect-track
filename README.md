# 0. SETUP & INSTALLATION:  
[Pytorch Installation](https://pytorch.org/#pip-install-pytorch)    
[Pytorch build from source](https://github.com/pytorch/pytorch#from-source)  
```
pip install opencv-python==3.4.5.20
pip install opencv-contrib-python==3.4.5.20
tree -L 1
```

# 1. POSENET BASE DETECTION TEST:
[PoseNet Pytorch](https://github.com/rwightman/posenet-pytorch)  
[PoseNet Tensorflow](https://github.com/rwightman/posenet-python)  

# 2. CONDA & PYTORCH BUILD:
**GCC 6 - CUDA 10 - Python 3.6**
```
conda create -n python36_cpu python=3.6  
source activate python36_cpu  
conda deactivate  
```
```
sudo update-alternatives --config gcc
python setup.py bdist_wheel
```
