# HyperPose

In this study, we propose ``HyperPose'' an approach for using hypernetworks
in absolute camera pose localization regressors. The inherent appearance
variations in natural scenes, due to environmental conditions, perspective,
and lighting, induce a notable domain disparity between the training and
test datasets, degrading the precision of contemporary localization
networks. To mitigate this challenge, we advocate incorporating
hypernetworks into both single-scene and multiscene camera pose regression
models. During the inference phase, the hypernetwork dynamically computes
adaptive weights for the localization regression heads based on the input
image, effectively narrowing the domain gap. We evaluate the HyperPose
methodology across multiple established absolute pose regression
architectures using indoor and outdoor datasets. Our empirical experiments
demonstrate that augmenting with HyperPose yields notable performance
enhancements, for both single- and multi-scene architectures. We have made
our source code and pre-trained models openly available.

![plot](./img/hyperpose_intro.png?raw=true "Title")


## Install
* The repository was developed and tested using python 3.8
* Make sure to install the required packages by running - *pip install -r requirements.txt*

## Data
* The implemented framework currently supports the [7Scenes]https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and [Cambridge Landmarks](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3) Landmarks datasets. Support for additional datasets can be added by implementing a custom PyTorch dataloader and placing it in the data directory.
* Please make sure to downlaod and perform any prerequisite steps requied to prepare the data for training/testing

## Running the Code
The repository contains four main scripts the perform training and testing of single- and multi-scene APRs.

# Testing a pre-trained single-scene model
For testing a pre-trained single-scene model, run the `main_single_scene.py` using the following arguments:
```
python main_single_scene.py hyperpose test ./models/backbones/efficient-net-b0.pth /media/dev/data/datasets/7scenes ./datasets/7Scenes/abs_7scenes_pose.csv_stairs_test.csv configs/7Scenes_config.json
--checkpoint_path /media/dev/storage1/hyperpose_models/seven_scenes/model_run_23_10_23_13_27_59.pth --output_path ./evals_results
```
In the example adove, we evaluating the *model_run_23_10_23_13_27_59.pth* model on the *Strairs* from the 7Scenes dataset.

# Testing a pre-trained single-scene model
To evaluate the performance of a pre-trained single-scene model, please follow the next steps:
1. Download the pre-trained model for the desired dataset and scene from this [link](https://drive.google.com/file/d/1QFOR9dsQxsmiB-XjonGYteTuawHAUMGu/view?usp=share_link)
2. Open the `./config/test.yaml` file and edit the `checkpoint_path` with the location of the pre-trained model you've downloaded.
3. Make sure to download and extract the required dataset files: [Cambridge Landmarks dataset website](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3), [7Scenes]https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).

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
