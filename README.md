## Continual Detection Transformer for Incremental Object Detection

[![LICENSE](https://img.shields.io/github/license/yaoyao-liu/E3BM?style=flat-square)](https://github.com/yaoyao-liu/CL-DETR/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg?style=flat-square&logo=python&color=3776AB&logoColor=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.5.1-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

[[Paper](https://www.cs.jhu.edu/~yyliu/preprints/Continual_Detection_Transformer_for_Incremental_Object_Detection.pdf)] [[Project Page](https://lyy.mpi-inf.mpg.de/CL-DETR/)]

This repository contains the PyTorch implementation for the [CVPR 2023](https://cvpr2023.thecvf.com/) Paper ["Continual Detection Transformer for Incremental Object Detection"](https://www.cs.jhu.edu/~yyliu/preprints/Continual_Detection_Transformer_for_Incremental_Object_Detection.pdf) by [Yaoyao Liu](https://yyliu.net/), [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/bernt-schiele/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), and [ Christian Rupprecht](https://chrirupp.github.io/). 

This is the preliminary code. If you have any questions on this repository or the related paper, feel free to [create an issue](https://github.com/yaoyao-liu/CL-DETR/issues/new) or [send me an email](mailto:yyliu@cs.jhu.edu).

### Installation and Datasets

This code is based on Deformable DETR. You may follow the instructions in <https://github.com/fundamentalvision/Deformable-DETR> to install packages and prepare datasets for this project.

### Checkpoints 

You may download the checkpoints here: \[[link](https://drive.google.com/drive/folders/1kKLl1MMRMTU4uTc5isoq2wPJjSH-3VhE?usp=sharing)\]. The experiment setting is *COCO 2017, 70+10*. Please put the phase-0 checkpoint, `phase_0.pth`, in the base directory before running the code. The current version will automatically load the phase-0 checkpoint to speed up the experiments. This is because phase 0 is not an incremental learning phase. It is the same as the standard Deformable DETR.

### Running Experiments

Run the following script to start the experiment for *COCO 2017, 70+10*:
```bash
bash run.sh
```

### Log Files

You may view the log file in `logs/COCO_70_10.out`. It was run by the following server:<br>
GPU: 4x NVIDIA Quadro RTX 8000, 48 GB GDDR6<br>
CPU: 1x AMD EPYC 7502P 32-Core Processor

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
