# DSNCL
The source code of the paper "Deep supervision network with contrastive learning for zero-shot sketch-based image retrieval"

## ========= Installation and Requirements =========

- ```
  cudatoolkit=11.3.1
  ```

- ```
  numpy=1.19.3
  ```

- ```
  python=3.7.16
  ```

- ```
  pytorch=1.10.0
  ```

- ```
  torchvision=0.11.0
  ```

## ============== Datasets ==============

### Sketchy
Please go to the [Sketchy official website](https://sketchy.eye.gatech.edu/), or download the dataset from [Google Drive](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view?usp=sharing).

### TU-Berlin
Please go to the [TU-Berlin official website](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/), or download the dataset from [Google Drive](https://drive.google.com/file/d/12VV40j5Nf4hNBfFy0AhYEtql1OjwXAUC/view?usp=sharing).

## ============== Training ==============

Train with the Sketchy Ext dataset


CUDA_VISIBLE_DEVICES=1 python train_cse_resnet_sketchy_ext.py


Train with the TU-Berlin Ext dataset


CUDA_VISIBLE_DEVICES=1 python train_cse_resnet_tuberlin_ext.py


## ============== Testing ==============

Test with the Sketchy Ext dataset


CUDA_VISIBLE_DEVICES=1 python test_cse_resnet_sketchy_zeroshot.py


Test with the TU-Berlin Ext dataset


CUDA_VISIBLE_DEVICES=1 python test_cse_resnet_tuberlin_zeroshot.py

