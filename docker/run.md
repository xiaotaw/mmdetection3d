## build docker image 
```bash
IMG=open-mmlab:mmdetection3d_pytorch1.9.0_cuda11.1_cudnn8
docker build -t $IMG .
```

## run a docker container
```bash
IMG=open-mmlab:mmdetection3d_pytorch1.9.0_cuda11.1_cudnn8
docker run -it \
--privileged \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
-e DISPLAY=unix${DISPLAY} \
--gpus=all \
--name mmdet3d \
--shm-size 11100M \
-v /mnt/xt/8T/:/mnt/xt/8T \
$IMG bash
```

```bash
# in the container, install mmdet3d and deps
cd /mnt/xt/8T/CODES/CV/open-mmlab/mmdetection3d/
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip
pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple seaborn
```

# train fcos3d
```bash
# download nuscneces v1.0 full dataset, and extract, and create a soft link under data/

# prepare data
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes 


# train fcos3d
python tools/train.py ./configs/fcos3d/fcos3d_r34_fpn_head-gn_1xb4-1x_nus-mono3d.py --auto-scale-lr

# vis loss
infile=work_dirs/fcos3d_r34-caffe_fpn_head-gn_1xb4-1x_nus-mono3d/20230731_044039/vis_data/20230731_044039.json
python tools/analysis_tools/analyze_logs.py plot_curve ${infile} --keys loss --out ${infile/.json/_loss.png}

# test 
python tools/test.py  ./configs/fcos3d/fcos3d_r34_v1_fpn_head-gn_1xb4-1x_nus-mono3d.py ./work_dirs/fcos3d_r34_v1_fpn_head-gn_1xb4-1x_nus-mono3d/epoch_12.pth  --task mono_det --show-dir show_dir
```


## experiments

