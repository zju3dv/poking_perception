#conda install pytorch=1.10.0 torchvision cudatoolkit=10.2 -c pytorch
#pip install -r requirements.txt
#conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
#conda install pytorch3d -c pytorch3d -y

/bin/rm -r build/ puop.egg-info
python setup.py build develop
