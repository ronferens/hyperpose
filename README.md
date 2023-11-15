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


# Install
* The repository was developed and tested using python 3.8
* Make sure to install the required packages by running - *pip install -r requirements.txt*

# Data
* The implemented framework currently supports the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and [Cambridge Landmarks](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3) Landmarks datasets. Support for additional datasets can be added by implementing a custom PyTorch dataloader and placing it in the data directory.
* Please make sure to downlaod and perform any prerequisite steps requied to prepare the data for training/testing

# Running the Code
The repository contains four main scripts the perform training and testing of single- and multi-scene APRs.

### Training a single-scene model
To train a single-scene model, run the `main_single_scene.py` script using the following arguments:
```
python main_single_scene.py hyperpose train ./models/backbones/efficient-net-b0.pth /media/dev/data/datasets/cambridge ./datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_train.csv ./configs/CambridgeLandmarks_config.json --output_path ./models
```
In the example adove a new model will be trained for the *Shop Facade* scene in the *Cambridge Landmarks* dataset.

### Testing a pre-trained single-scene model
For testing a pre-trained single-scene model, run the `main_single_scene.py` script using the following arguments:
```
python main_single_scene.py hyperpose test ./models/backbones/efficient-net-b0.pth /media/dev/data/datasets/7scenes ./datasets/7Scenes/abs_7scenes_pose.csv_stairs_test.csv ./configs/7Scenes_config.json
--checkpoint_path /models/model_run_23_10_23_13_27_59.pth --output_path ./evals/evals_results
```
In the example adove, we evaluate a *model_run_23_10_23_13_27_59.pth* model, located at the *./models* directory, on the *Strairs* scene in the *7Scenes* dataset.

Based on the value set in the `n_freq_checkpoint` parameter in the configuration file, multiple checkpoint will save to the output path.
In case you would like to perform a batch evaluation of all the existing .pth files, please run the `main_batch_eval_model.py' as followed:
```
python main_batch_eval_model.py hyperpose test ./models/backbones/efficient-net-b0.pth /media/dev/data/datasets/7scenes ./datasets/7Scenes/abs_7scenes_pose.csv_stairs_test.csv ./configs/7Scenes_config.json
./models/run_23_11_15_13_47_00 --output_path ./evals/hyperpose_7scenes_evals
```
Here, we evaluate the all checkpoint saved in the *./models/run_23_11_15_13_47_00* directory, on the *Strairs* scene in the 7Scenes dataset.

Once the batch evaluate in done, both a .csv file and .html file will be saved with details position and orientation estimation for each checkpoint.

### Training an MS-HyperPose model
To train a new MS-HyperPose model, run the `main_multi_scene.py` script using the following arguments:
```
python main_multi_scene.py mshyperpose train ./models/backbones/efficient-net-b0.pth /media/dev/data/datasets/7Scenes ./datasets/7Scenes/7scenes_all_scenes.csv ./configs/7Scenes_config.json --output_path ./models
```
In the example adove a new MS-HyperPose model will be trained for the entire set of scenes in the *7Scenes* dataset.

### Testing a pre-trained MS-HyperPose model
For testing a pre-trained MS-HyperPose model, run the `main_multi_scene.py` script using the following arguments:
```
python main_multi_scene.py mshyperpose test ./models/backbones/efficient-net-b0.pth /media/dev/data/datasets/7scenes ./datasets/7Scenes/7scenes_all_scenes.csv ./configs/7Scenes_config.json
--checkpoint_path /models/model_run_23_11_08_01_50_16.pth --output_path ./evals_results
```
In the example adove, we evaluate a *model_run_23_11_08_01_50_16.pth* model, located at the *./models* directory, on the entire 7Scenes dataset.

Similar to the batch evaluation of the single-scene (see previous paragraph), you can run batch evalaution of a multi-scene model using the `main_batch_ms_eval.py` script:
```
python main_batch_ms_eval.py mshyperpose test ./models/backbones/efficient-net-b0.pth /media/dev/data/datasets/7scenes ./datasets/7Scenes/7scenes_all_scenes.csv ./configs/7Scenes_config.json ./configs/7Scenes_config.json
./models/run_23_11_08_01_50_16 --output_path ./evals/hyperpose_7scenes_evals
```
