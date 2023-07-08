# UOT-gen
Pytorch implementation of our UOT-gen。

## Associated Publication

A deep learning framework for geodesics under spherical Wasserstein-Fisher-Rao metric and its application for weighted sample generation

Paper: https://arxiv.org/abs/2208.12145

Please cite as
    
    @article{jing2022deep,
      title={A deep learning framework for geodesics under spherical Wasserstein-Fisher-Rao metric and its application for weighted sample generation},
      author={Jing, Yang and Chen, Jiaheng and Li, Lei and Lu, Jianfeng},
      journal={arXiv preprint arXiv:2208.12145},
      year={2022}
    }



for 1d toy experiment, see 

```
trainToyOTflow_1d_inverse.py
```

for 2d and high-dimension experiment, see

```
trainToyOTflow_high_Bayes.py (train) & high_dim_Bayes folder (evaluation)
```

for traditional optimization method to solve UOT:

```
 OT-traditional folder 
```

