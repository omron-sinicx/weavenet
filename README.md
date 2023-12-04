This repository is a portal for our weavenet projects.

### 📄 Project papers and repository
1. [WeaveNet for Approximating Two-sided Matching Problems](https://arxiv.org/abs/2310.12515) (this repository)
3. [3D Point Cloud Registration with Learning-based Matching Algorithm](https://arxiv.org/abs/2202.02149) ([repository](https://github.com/omron-sinicx/EdgeSelectiveFeatureWeaving))

### 🧰 API
- [An API document for this repository](https://omron-sinicx.github.io/weavenet/main/)
### 🦾 motivation of the WeaveNet architecture
- A trainable neural solver for general assignment tasks, including bipartite matching, linear assignment with stochastic noise, and stable matching (a.k.a. stable marriage problem).
- It can be combined with feature extractor for registration tasks. See [this example](https://arxiv.org/abs/2202.02149) for more details.

### Requirements
- This package requires python3.7+, pytorch, and torch-scatter.

### install from GitHub.com
- `pip install git+https://github.com/omron-sinicx/weavenet`

### a minimum sample code

#### Case1: Apply WeaveNet to matching based on extracted features

```python:case1.py
from weavenet.model import TrainableMatchingModule, WeaveNetHead

matching_module = TrainableMatchingModule(
    head = WeaveNetHead(
        input_channels=512, # channels of input vertex features
        output_channels_list=[32, 32, 32, 32, 32, 32] # output channels of each layers
        mid_channels_list=[64, 64, 64, 64, 64, 64] # intermediate channels of each layers
        calc_residual=[False] * 6 # do not connect residual path
        keep_first_var_after = 0 # keep the first layer output as the source of first residual path.      
   )
  output_channels = 1 # output only 1 matching matrix.
  pre_interactor = CrossConcatVertexFeatures(),
)


z_src = some_feature_extractor.forward(x_src) # z_src.shape = [B, N, C]
z_tar = some_feature_extractor.forward(x_tar)# z_src.shape = [B, M, C]
matching = matching_module.forward(x_src, x_tar) # matching.shape = [B, 1, N, M]

```

#### Case2: Apply WeaveNet to solve stable matching.
```python:case2.py
from weavenet.model import TrainableMatchingModule, WeaveNetHead

matching_module = TrainableMatchingModule(
    head = WeaveNetHead(
        input_channels=2, # channels of input problem instance.
        output_channels_list=[32, 32, 32, 32, 32, 32] # output channels of each layers
        mid_channels_list=[64, 64, 64, 64, 64, 64] # intermediate channels of each layers
        calc_residual=[False] * 6 # do not connect residual path
        keep_first_var_after = 0 # keep the first layer output as the source of first residual path.      
   )
  output_channels = 1 # output only 1 matching matrix.
)

# sab stands for satisfaction_a2b, and sba_t stands for satisfaction_b2a_transposed
sab, sba_t = batch # sab.shape = [B, N, M, 1], sba_t.shape = [B, N, M, 1]. 

matching = matching_module.forward(sab, sba_t) # matching.shape = [B, 1, N, M]

```

### Features from original implementation
- Optimized for jit execution.
- Better interface for light use.
- New class interface design for easy customization of the matching network.


### ⬇️ install locally
```
 git clone git@github.com:omron-sinicx/weavenet.git weavenet
 cd weavenet
`$ pip install .` or `$ pip install .[dev]` (for testing)
`$ pip show -f weavenet
```

```
Name: weavenet
Version: 1.0.1
Summary: an official implementation of WeaveNet and its components.
Home-page: github.com/omron-sinicx/weavenet
Author: Atsushi Hashimoto
License: MIT License
        Copyright (c) 2022 OMRON SINIC X Corp.
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

```


### 📚 Citation
```
@article{sone2023weavenet,
  title={WeaveNet for Approximating Two-sided Matching Problems},
  author={Shusaku Sone, Jiaxin Ma, Atsushi Hashimoto, Naoya Chiba, Yoshitaka Ushiku},
  journal={arXiv preprint arXiv:2310.12515},
  year={2023}
}
```
```
@article{yanagi2022edge,
  title={Edge-Selective Feature Weaving for Point Cloud Matching},
  author={Yanagi, Rintaro and Hashimoto, Atsushi and Sone, Shusaku and Chiba, Naoya and Ma, Jiaxin and Ushiku, Yoshitaka},
  journal={arXiv preprint arXiv:2202.02149},
  year={2022}
}
```
