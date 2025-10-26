# Unified GPU IPC Framework


DESCRIPTION
===========

This repository contains the source code for StiffGIPC, a Unified GPU Incremental Potential Contact Framework introduced in our paper: [StiffGIPC: Advancing GPU IPC for Stiff Affine-Deformable Simulation](https://dl.acm.org/doi/10.1145/3735126) **ACM Transactions on Graphics, 2025**. Our framework consistently achieves high performance in soft, stiff, and hybrid simulations (cloth, elastic solids, rigid bodies, and their hybrid couplings), even when handling high-resolution models, large deformations, and high-speed impacts.


Source code contributor: [Kemeng Huang](https://kemenghuang.github.io), Xinyu Lu

**Note: this software is released under the MPLv2.0 license. For commercial use, please email the authors for negotiation.**

## video 1
[![Watch the video](https://github.com/KemengHuang/Stiff-GIPC/blob/main/Assets/teaser.JPG)](https://www.youtube.com/watch?v=3TBoTX2vag4&list=LL)

## BibTex 

Please cite the following paper if it helps. 

```
@article{stiffgipc2025,
      author = {Huang, Kemeng and Lu, Xinyu and Lin, Huancheng and Komura, Taku and Li, Minchen},
      title = {StiffGIPC: Advancing GPU IPC for Stiff Affine-Deformable Simulation},
      year = {2025},
      publisher = {Association for Computing Machinery},
      volume = {44},
      number = {3},
      issn = {0730-0301},
      doi = {10.1145/3735126},
      journal = {ACM Trans. Graph.},
      month = may,
      articleno = {31},
      numpages = {20}
}
```





Requirements
============

Hardware requirements: Nvidia GPUs

Support platforms: Windows, Linux 

## Dependencies

| Name                                   | Version | Usage                                               | Import         |
| -------------------------------------- | ------- | --------------------------------------------------- | -------------- |
| cuda                                   | >=11.0  | GPU programming                                     | system install |
| eigen3                                 | 3.4.0   | matrix calculation                                  | package        |
| freeglut                               | 3.4.0   | visualization                                       | package        |
| glew                                   | 2.2.0#3 | visualization                                       | package        |

### linux

We use CMake to build the project.

```bash
sudo apt install libglew-dev freeglut3-dev libeigen3-dev metis
```


### Windows && linux
We use [vcpkg](https://github.com/microsoft/vcpkg) to manage the libraries we need and use CMake to build the project. The simplest way to let CMake detect vcpkg is to set the system environment variable `CMAKE_TOOLCHAIN_FILE` to `(YOUR_VCPKG_PARENT_FOLDER)/vcpkg/scripts/buildsystems/vcpkg.cmake`

```shell
vcpkg install eigen3 freeglut glew tbb freeglut metis
```


EXTERNAL CREDITS
================

This work utilizes the following external software library, which have been included here for convenience:
Copyrights are retained by the original authors.

muda https://github.com/KemengHuang/muda, a fork of https://github.com/MuGdxy/muda
