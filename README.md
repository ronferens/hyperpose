# HyperPose

In this study, we propose the use of attention hypernetworks in camera pose
localization. The dynamic nature of natural scenes, including changes in
environment, perspective, and lighting, creates an inherent domain gap
between the training and test sets that limits the accuracy of contemporary
localization networks. To overcome this issue, we suggest a camera pose
regressor that integrates a hypernetwork. During inference, the hypernetwork
generates adaptive weights for the localization regression heads based on
the input image, effectively reducing the domain gap. We also suggest the
use of a Transformer-Encoder as the hypernetwork, instead of the common
multilayer perceptron, to derive an attention hypernetwork. The proposed
approach achieves superior results compared to state-of-the-art methods on
contemporary datasets. To the best of our knowledge, this is the first
instance of using hypernetworks in camera pose regression, as well as using
Transformer-Encoders as hypernetworks.

![plot](./img/hyperpose_intro.png?raw=true "Title")


## Demo
To demostrate the performance of *HyperPose* a pre-trained model can be tested and evaluated.
1. Download the pre-trained model for the Cambridge Landmarks' Kings Colledge scene from this [link](https://drive.google.com/file/d/1QFOR9dsQxsmiB-XjonGYteTuawHAUMGu/view?usp=share_link)
2. Open the `./config/test.yaml` file and edit the `checkpoint_path`
3. Make sure to download the Kings Colledge (`KingsCollege.zip`) scene from the [Cambridge Landmarks dataset website](https://www.repository.cam.ac.uk/handle/1810/251342) and update the `dataset_path' in the yaml file.
4. Open the `main.py` file and modify running configuration to `config_name="test"` (in line 16)
5. Run the main.py file
