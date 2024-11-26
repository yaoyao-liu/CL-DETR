# Continual Detection Transformer for Incremental Object Detection

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/CL-DETR/blob/master/LICENSE)

[[Paper](https://www.cs.jhu.edu/~yyliu/preprints/Continual_Detection_Transformer_for_Incremental_Object_Detection.pdf)] [[Project Page](https://lyy.mpi-inf.mpg.de/CL-DETR/)]

This repository contains the PyTorch implementation for the [CVPR 2023](https://cvpr2023.thecvf.com/) Paper ["Continual Detection Transformer for Incremental Object Detection"](https://www.cs.jhu.edu/~yyliu/preprints/Continual_Detection_Transformer_for_Incremental_Object_Detection.pdf) by [Yaoyao Liu](https://people.mpi-inf.mpg.de/~yaliu/), [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/bernt-schiele/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), and [ Christian Rupprecht](https://chrirupp.github.io/). 

This is the preliminary code. If you have any questions on this repository or the related paper, feel free to [create an issue](https://github.com/yaoyao-liu/CL-DETR/issues/new) or [send me an email](mailto:lyy@illinois.edu).

### Installation 

This code is based on Deformable DETR. You may follow the instructions in <https://github.com/fundamentalvision/Deformable-DETR> to install the packages for this project.

### Checkpoints 

You may download the checkpoints here: [link](https://drive.google.com/drive/folders/1kKLl1MMRMTU4uTc5isoq2wPJjSH-3VhE?usp=sharing). The experiment setting is "COCO 2017, 70+10". 

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
  year         = {2023},
  doi          = {10.1109/CVPR52729.2023.02279},
}
```
