cd ..
git clone https://github.com/565353780/point-cept.git

cd point-cept
./setup.sh

pip install viser fpsample trimesh numba gradio

cd ../ppp-sam/ppp_sam/Lib/chamfer3D
python setup.py install
