# pointMLP-pytorch
Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework


## Install
Please ensure that python3.7+ is installed. We suggest user use conda to create a new environment.

Install dependencies
```bash
pip install -r requirement.txt
```

Install CUDA kernels
```bash
pip install pointnet2_ops_lib/.
```

## Classification ModelNet40
The dataset will be automatically downloaded, run following command to train
```bash
# train pointMLP
python main.py --model pointMLP
# train pointMLP-elite
python main.py --model pointMLPElite
# please add other paramemters as you wish.
```
By default, it will create a fold named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.

To conduct voting experiments, run
```bash
# please modify the msg accrodingly
python voting.py --model pointMLP --msg demo
```


## Classification ScanObjectNN

- Make data folder and download the dataset
```bash
cd pointMLP-pytorch/classification_ScanObjectNN
mkdir data
cd data
wget http://103.24.77.34/scanobjectnn/h5_files.zip
unzip h5_files.zip
```

- Train pointMLP/pointMLPElite by
```bash
# train pointMLP
python main.py --model pointMLP
# train pointMLP-elite
python main.py --model pointMLPElite
# please add other paramemters as you wish.
```
By default, it will create a fold named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.

- To conduct voting experiments, run
```bash
# please modify the msg accrodingly
python voting.py --model pointMLP --msg demo
```

## Part segmentation

- Make data folder and download the dataset
```bash
cd pointMLP-pytorch/part_segmentation
mkdir data
cd data
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```

- Train pointMLP by
```bash
# train pointMLP
python main.py --model pointMLP
# please add other paramemters as you wish.
```

## Pre-trained models

Please download the pre-trained models and log files here: [anonymous google drive](https://drive.google.com/drive/folders/1Jn9HNpPsrq-1XqSmOUtw4cwPMjsIiIpz?usp=sharing)
