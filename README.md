## Continual Detection Transformer for Incremental Object Detection

[![LICENSE](https://img.shields.io/github/license/yaoyao-liu/E3BM?style=flat-square)](https://github.com/yaoyao-liu/CL-DETR/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg?style=flat-square&logo=python&color=3776AB&logoColor=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.5.1-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

[[Paper](https://www.cs.jhu.edu/~yyliu/preprints/Continual_Detection_Transformer_for_Incremental_Object_Detection.pdf)] [[Project Page](https://lyy.mpi-inf.mpg.de/CL-DETR/)]

This repository contains the PyTorch implementation for the [CVPR 2023](https://cvpr2023.thecvf.com/) Paper ["Continual Detection Transformer for Incremental Object Detection"](https://www.cs.jhu.edu/~yyliu/preprints/Continual_Detection_Transformer_for_Incremental_Object_Detection.pdf) by [Yaoyao Liu](https://yyliu.net/), [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/bernt-schiele/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), and [ Christian Rupprecht](https://chrirupp.github.io/). 

This is the preliminary code. If you have any questions on this repository or the related paper, feel free to [create an issue](https://github.com/yaoyao-liu/CL-DETR/issues/new) or [send me an email](mailto:yyliu@cs.jhu.edu).

### Installation and Datasets

This code is based on Deformable DETR. You may follow the instructions in <https://github.com/fundamentalvision/Deformable-DETR> to install packages and prepare datasets for this project.

#### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n cl_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate cl_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

#### Compiling CUDA Operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

#### Dataset Preparation

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

### Checkpoints 

You may download the checkpoints here: \[[link](https://drive.google.com/drive/folders/1kKLl1MMRMTU4uTc5isoq2wPJjSH-3VhE?usp=sharing)\]. The experiment setting is *COCO 2017, 70+10*. Please put the phase-0 checkpoint, `phase_0.pth`, in the base directory before running the code. The current version will automatically load the phase-0 checkpoint to speed up the experiments. This is because phase 0 is not an incremental learning phase. It is the same as the standard Deformable DETR.

### Running Experiments

Run the following script to start the experiment for *COCO 2017, 70+10*:
```bash
bash run.sh
```
If you need to run experiments for the *40+40* setting, you may need to change the code in multiple files, e.g., `main.py` and `datasets/pycocotools.py`. Please refer to this branch for the *40+40* experiments: <https://github.com/yaoyao-liu/CL-DETR/tree/40_40>

### Performance

Incremental object detection results (%) on COCO 2017. In the *A*+*B* setup, in the first phase, we observe a fraction $\frac{A}{A+B}$ of the training samples with
*A* categories annotated. Then, in the second phase, we observe the remaining $\frac{B}{A+B}$ of the training samples, where *B* new categories are annotated.

| Setting          | Detection Baseline  | AP  | AP50  | AP75 | APS | APM | APL |
| --------------  |---------- | ----------  | ----------   |------------ | ------------ |------------ | ------------ |
| 70+10 | Deformable DETR | 40.1 | 57.8 | 43.7 | 23.2 | 43.2 | 52.1 |
| 40+40 | Deformable DETR | 37.5 | 55.1 | 40.3 | 20.9 | 40.8 | 50.7 |

### Citation

Please cite our paper if it is helpful to your work:

```bibtex
@inproceedings{Liu2023CLDETR,
  author       = {Yaoyao Liu and
                  Bernt Schiele and
                  Andrea Vedaldi and
                  Christian Rupprecht},
  title        = {Continual Detection Transformer for Incremental Object Detection},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2023, Vancouver, BC, Canada, June 17-24, 2023},
  pages        = {23799--23808},
  publisher    = {{IEEE}},
  year         = {2023}
}
```

### Acknowledgements

Our implementation uses the source code from the following repository:
- <https://github.com/fundamentalvision/Deformable-DETR>
