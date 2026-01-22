# GausSim_ICCV2025
Official code for "GausSim: Foreseeing Reality by Gaussian Simulator for Elastic Objects".

[project](https://www.mmlab-ntu.com/project/gausim/index.html) | [paper](https://arxiv.org/pdf/2412.17804)

## TODO List
- [x] Interactive demo with sampled data and model weight.
- [ ] Full dataset.
- [ ] Full code.


## Config
```
conda create -n MMGS python=3.9 pytorch==1.13.0 pytorch-cuda=11.7 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch -c nvidia -y && conda activate MMGS
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
pip3 install h5py point_cloud_utils decord imageio mediapy yapf==0.40.1 plyfile plotly kmeans_gpu wandb numpy==1.26.4 tensorboard kornia flask flask-socketio
pip install  dgl==1.1.0 -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/
pip3 install -v -e .
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y
```

## Prepare Data
### Demo Data
Please download the demo data [here](https://entuedu-my.sharepoint.com/:u:/g/personal/yidi001_e_ntu_edu_sg/IQC62txnyQUuTYCZ0aZeMokWAQrswFDwHAOFIeY9ZcyV6t8?e=pNYohX) and extract under the project's folder.


## Launch Interactive Demo
```
python tools/infer_server.py configs/gssim/iccv/pudding_infer.py work_dirs/latest.pth
```

### Operation
* Mid click to drag, and realise the mid click to apply forces.
* Translation control
	* w: move camera closer
	* s: move camera futher
	* a: move camera to the left
	* d: move camera to the right
	* q: move camera upwards
	* e: move camera downwards
* Rotation control
	* i: rotate camera to look upwards
	* k: rotate camera to look downwards
	* j: rotate camera to look left
	* l: rotate camera to look right
	* u: rotate camera clockwise
	* o: rotate camera anti-clockwise
