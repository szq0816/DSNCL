# DSNCL
The source code of paper "Deep supervision network with contrastive learning for zero-shot sketch-based image retrieval"

============== Installation and Requirements ==============
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


============== Datasets ==============
Sketchy and TU-Berlin: [by Dutta et al.](https://github.com/AnjanDutta/sem-pcyc) or [by Liu et al.](https://github.com/qliu24/SAKE)


============== Training ==============
# train with Sketchy Ext dataset
python train_cse_resnet_sketchy_ext.py

# train with TU-Berlin Ext dataset
python train_cse_resnet_tuberlin_ext.py


============== Testing ==============
# test with Sketchy Ext dataset
python test_cse_resnet_sketchy_zeroshot.py

# test with TU-Berlin Ext dataset
python test_cse_resnet_tuberlin_zeroshot.py
