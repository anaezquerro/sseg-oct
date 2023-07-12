# Semantic segmentation systems for the detection of retinal pathological fluid from OCT images <img  class="lazyloaded" src="images/Eye.png" height="30px"><img  class="lazyloaded" src="images/Hospital.png" height="30px">

This project was developed as a Computer Vision assignment in our last college year. Our professors provided us a dataset of 50 OCT images with their respective segmentation masks to detect pathological fluid from images of the retina. In the description of this assignment, they suggested us using [U-Net](https://arxiv.org/abs/1505.04597) architecture as our baseline and reviewing the SoTA in semantic segmentation to improve its results. We were expected to familiarize with the most used Deep Learning models of the day and experiment with diverse alternatives and techniques to research the performance of modern CV proposals in this problem. Then, we were asked to submit a paper with a detailed description of the tested approaches and the obtained results (check it [here](sseg-oct.pdf)).


## Segmentation models

- **Baseline**: [U-Net](https://arxiv.org/abs/1505.04597) architecture (implementation was adapted from [pytorch-unet](https://github.com/usuyama/pytorch-unet)).
- **Fully convolutional pretrained models**: [U-Net](https://arxiv.org/abs/1505.04597), [LinkNet](https://arxiv.org/abs/1707.03718) and [PSPNet](https://arxiv.org/abs/1612.01105) using [ResNet-50](https://arxiv.org/abs/1512.03385) as a pretrained encoder with [ImageNet](https://www.image-net.org/) (implementation  adapted from [segmentation_models_pytorch](https://segmentation-modelspytorch.readthedocs.io/en/latest/)).
- **Transformer-based models**: [PAN](https://arxiv.org/abs/1805.10180) with pretrained [ResNet-50](https://arxiv.org/abs/1512.03385) as encoder (implementation adapted from [segmentation_models_pytorch](https://segmentation-modelspytorch.readthedocs.io/en/latest/)) and [Attention U-Net](https://arxiv.org/abs/1804.03999) (implementation adapted from [CBIM-Medical-Image-Segmentation](https://github.com/yhygao/CBIM-Medical-Image-Segmentation/)).
- **Deformable convolutions**: [U-Net]((https://arxiv.org/abs/1505.04597)) with deformable convolutions from [PyTorch-Deformable-Convolution-v2](https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2).
- **Adversarial learning**: We took fully convolutional pretrained models and trained them in an adversarial benchmark (see details in the [paper](sseg-oct.pdf)).

## Results 

F-Score results:

<table>
    <tr>
        <td></td>
        <td colspan="2"><i>Default benchmark</i></td>
        <td colspan="2"><i>Adversarial benchmark</i></td>
    </tr>
    <tr>
        <td></td>
        <td><i>No aug.</i></td>
        <td><i>Data aug.</i></td>
        <td><i>No aug.</i></td>
        <td><i>Data aug.</i></td>
    </tr>
    <tr>
        <td><b>Baseline <a href="https://arxiv.org/abs/1505.04597">U-Net</a></b></td>
        <td>0.6<sub>0.15</sub></td>
        <td>0.65<sub>0.16</sub></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><b><a href="https://arxiv.org/abs/1505.04597">U-Net &reg</a></b></td>
        <td>0.7<sub>0.05</sub></td>
        <td>0.6<sub>0.09</sub></td>
        <td>0.87<sub>0.03</sub></td>
        <td>0.86<sub>0.06</sub></td>
    </tr>
    <tr>
        <td><b><a href="https://arxiv.org/abs/1707.03718">LinkNet &reg</a></b></td>
        <td>0.6<sub>0.1</sub></td>
        <td>0.57<sub>0.09</sub></td>
        <td>0.77<sub>0.25</sub></td>
        <td>0.83<sub>0.12</sub></td>
    </tr>
    <tr>
        <td><b><a href="https://arxiv.org/abs/1612.01105">PSPNet &reg</a></b></td>
        <td>0.67<sub>0.04</sub></td>
        <td>0.67<sub>0.05</sub></td>
        <td>0.82<sub>0.03</sub></td>
        <td>0.84<sub>0.04</sub></td>
    </tr>
    <tr>
        <td><b><a href="https://arxiv.org/abs/1805.10180">PAN &reg</a></b></td>
        <td>0.7<sub>0.05</sub></td>
        <td>0.66<sub>0.09</sub></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><b><a href="https://arxiv.org/abs/1804.03999">Attention U-Net</a></b></td>
        <td>0.74<sub>0.06</sub></td>
        <td>0.81<sub>0.08</sub></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><b><a href="sseg-oct.pdf">Deform U-Net</a></b></td>
        <td>0.71<sub>0.06</sub></td>
        <td>0.73<sub>0.09</sub></td>
        <td></td>
        <td></td>
    </tr>
</table>


Intersection over Union (IoU) results:

<table>
    <tr>
        <td></td>
        <td colspan="2"><i>Default benchmark</i></td>
        <td colspan="2"><i>Adversarial benchmark</i></td>
    </tr>
    <tr>
        <td></td>
        <td><i>No aug.</i></td>
        <td><i>Data aug.</i></td>
        <td><i>No aug.</i></td>
        <td><i>Data aug.</i></td>
    </tr>
    <tr>
        <td><b>Baseline <a href="https://arxiv.org/abs/1505.04597">U-Net</a></b></td>
        <td>0.44<sub>0.13</sub></td>
        <td>0.56<sub>0.11</sub></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><b><a href="https://arxiv.org/abs/1505.04597">U-Net &reg</a></b></td>
        <td>0.54<sub>0.06</sub></td>
        <td>0.48<sub>0.1</sub></td>
        <td>0.78<sub>0.04</sub></td>
        <td>0.76<sub>0.08</sub></td>
    </tr>
    <tr>
        <td><b><a href="https://arxiv.org/abs/1707.03718">LinkNet &reg</a></b></td>
        <td>0.43<sub>0.11</sub></td>
        <td>0.4<sub>0.09</sub></td>
        <td>0.67<sub>0.22</sub></td>
        <td>0.72<sub>0.14</sub></td>
    </tr>
    <tr>
        <td><b><a href="https://arxiv.org/abs/1612.01105">PSPNet &reg</a></b></td>
        <td>0.5<sub>0.04</sub></td>
        <td>0.51<sub>0.06</sub></td>
        <td>0.69<sub>0.04</sub></td>
        <td>0.73<sub>0.05</sub></td>
    </tr>
    <tr>
        <td><b><a href="https://arxiv.org/abs/1805.10180">PAN &reg</a></b></td>
        <td>0.53<sub>0.05</sub></td>
        <td>0.5<sub>0.03</sub></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><b><a href="https://arxiv.org/abs/1804.03999">Attention U-Net</a></b></td>
        <td>0.59<sub>0.07</sub></td>
        <td>0.72<sub>0.08</sub></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><b><a href="sseg-oct.pdf">Deform U-Net</a></b></td>
        <td>0.55<sub>0.06</sub></td>
        <td>0.58<sub>0.07</sub></td>
        <td></td>
        <td></td>
    </tr>
</table>

<p> Pretrained models are marked with &reg.</p>

It is possible to check the number of parameters of the models:
```shell
python3 counter.py
```

## Reproduce results 

All code needed to reproduce our results is available in [`segmenter/`](segmenter) folder (if you require access to the original dataset, please [contact me](mailto:ana.ezquerro@udc.es)). The script [`test.py`](segmenter/test.py) admits the following parameters to select the segmentation model:


```shell
python3 test.py <model> <mode> -v -aug -adv  
```

- `model`: Specifies the model to train or test (choices are `base`, `unet`, `linknet`, `pspnet`, `pan`, `attnunet`, `deformunet`).
- `mode`: Specifies the execution mode (`kfold` or `train`). 
    + `kfold` accepts argument `--k`  to run `k` *fold cross validation*. Final metrics are stored in the folder `--model-path`. 
    + `train` executes a single train-validation split and the training procedure of the model selected and validating with the 10% of the dataset. 
- `-v`: Flag to show the training trace.
- `-aug`: Flag to use data augmentation.
- `-adv`: Flag to train the model in the adversarial benchmark.

**Examples**:

Execute the baseline model without data augmentation with batch size of 10 images and save results in `../results/base`.

```shell
python3 test.py base kfold -v --batch_size=10 --model_path=../results/ base/  --route=../OCT-dataset/
```

Execute pretrained U-Net with data augmentation:

```shell
python3 test.py unet kfold -v --batch_size=10 -aug  
```

Execute LinkNet with data augmentation and adversarial learning:
```shell
python3 test.py unet train -v --batch_size=10 -aug  -adv
```

Note that the argument `--route` specifies the path where the OCT dataset is stored. The structure of this folder is expected to store two subfolders: `images/` and `masks/`, which contain respectively the tomography retinal images used as input to the model and the pathological fluid masks used as output.





## Team builders :construction_worker:

- Ana Ezquerro ([ana.ezquerro@udc.es](mailto:ana.ezquerro@udc.es), [GitHub](https://github.com/anaezquerro)).
