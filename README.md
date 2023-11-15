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


## Install
* The repository was developed and tested using python 3.8
* Make sure to install the required packages by running - *pip install -r requirements.txt*


## Demo - Evaluating a pre-trained model
To demostrate the performance of *HyperPose* a pre-trained model can be tested and evaluated.
1. Download the pre-trained model for the Cambridge Landmarks' Kings Colledge scene from this [link](https://drive.google.com/file/d/1QFOR9dsQxsmiB-XjonGYteTuawHAUMGu/view?usp=share_link)
2. Open the `./config/test.yaml` file and edit the `checkpoint_path` with the location of the pre-trained model you've downloaded.
3. Make sure to download the Kings Colledge (*KingsCollege.zip*) scene from the [Cambridge Landmarks dataset website](https://www.repository.cam.ac.uk/handle/1810/251342) and update the `dataset_path` in the yaml file by setting the directory in which you've saved extracted the ZIP's conent.

(For example: `dataset_path: '/home/dev/Data/cambridge'`)

5. Update the `output_path` to set the location where to save the reported results
7. Open the `main.py` file and modify running configuration to `config_name="test"` (line 16)
8. Run the main.py file

Once the evaluation is done, you should get the following output:
`Median pose error: 0.588[m], 2.394[deg]`


## Training HyperPose
In order to train the model, you should prepare a configuration (.yaml) file to set the training parameters and the model's hyperparameters.
For both the Cambridge Landmarks and 7-Scenes datasets, pre-defined configuration files can be found under the `./config` folders.
1. Once you have downloaded the desired dataset, update the `dataset_path` field (same as in #4 in the demo section)
2. Update the `config_name` field in the `main.py` file (line 16)
3. Run the main.py file

Checkpoints will be saved to the `output_path` you've set in the configuration file.

## Testing HyperPose
Please follow the instructions detailed in the *Demo* section.
1. Upadte the `labels_path` with a the path to a *.csv* file listing the test set images with the required metadata.
Note: For both the Cambridge Landmarks and 7-Scenes datasets, you can select the matching *.csv* file suppied under the `./datasets` folder
2. Instead of using the pre-trained model, make sure to reference to the model you have trained under the `checkpoint_path` field.
