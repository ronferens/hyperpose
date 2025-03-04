<div align="center">
<h2>HyperPose: Hypernetwork-Infused Camera Pose Localization and an Extended Cambridge Landmarks Dataset</h2>

[**Ron Ferens**](https://ronferens.github.io/) · [**Yosi Keller**](https://yosikeller.github.io/)

Bar-Ilan University

<a href="[https://arxiv.org/abs/2103.11477](https://arxiv.org/abs/2303.02610)"><img src='https://img.shields.io/badge/arXiv-HyperPose-red' alt='Paper PDF'></a>
<a href='https://ronferens.github.io/hyperpose/'><img src='https://img.shields.io/badge/Project_Page-HyperPose-green' alt='Project Page' target="_blank"></a>
</div>

In this work, we propose ***HyperPose***, which utilizes hypernetworks in absolute camera pose regressors. The inherent appearance variations in natural scenes, attributable to environmental conditions, perspective, and lighting, induce a significant domain disparity between the training and test datasets. This disparity degrades the precision of contemporary localization networks. To mitigate this, we advocate for incorporating hypernetworks into single-scene and multiscene camera pose regression models. During inference, the hypernetwork dynamically computes adaptive weights for the localization regression heads based on the particular input image, effectively narrowing the domain gap. Using indoor and outdoor datasets, we evaluate the HyperPose methodology across multiple established absolute pose regression architectures. We also introduce and share the Extended Cambridge Landmarks (ECL), a novel localization dataset, based on the Cambridge Landmarks dataset, showing it in multiple seasons with significantly varying appearance conditions. Our empirical experiments demonstrate that HyperPose yields notable performance enhancements for single- and multi-scene architectures.

![plot](./img/hyperpose_arch_intro.png?raw=true "Title")

# Install

* The repository was developed and tested using python 3.8
* Make sure to install the required packages by running - *pip install -r requirements.txt*

# Data

* The implemented framework currently supports
  the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
  and [Cambridge Landmarks](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3) Landmarks
  datasets. Support for additional datasets can be added by implementing a custom PyTorch dataloader and placing it in
  the data directory.
* Please make sure to download and perform any prerequisite steps requied to prepare the data for training/testing
* All out pre-trained models are available for download in
  this [link](https://www.dropbox.com/scl/fi/bj72wz17u77phidvtgyem/drive-download-20241114T145344Z-001.zip?rlkey=x0gzacfep9fe9kgq2yxc3tfq0&e=1&st=icbxe3s7&dl=0).

## Extended Cambridge Landmarks (ECL) Dataset

The Extended Cambridge Landmarks (ECL) dataset introduces new flavors for the scenes in the
original [Cambridge Landmarks dataset](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3).

For each scene, the ECL contains three distinct flavors: *Evening*, *Winter*, and *Summer*.

![plot](https://anonymous.4open.science/r/extcambridgelandmarks-7A55/static/images/ecl_teaser.png?raw=true "Title")

The ECL dataset can be downloaded from this [repo](https://anonymous.4open.science/r/extcambridgelandmarks-7A55)

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

In the example adove, we evaluate a *model_run_23_10_23_13_27_59.pth* model, located at the *./models* directory, on the
*Strairs* scene in the *7Scenes* dataset.

Based on the value set in the `n_freq_checkpoint` parameter in the configuration file, multiple checkpoint will save to
the output path.
In case you would like to perform a batch evaluation of all the existing .pth files, please run the `
main_batch_eval_model.py' as followed:

```
python main_batch_eval_model.py hyperpose test ./models/backbones/efficient-net-b0.pth /media/dev/data/datasets/7scenes ./datasets/7Scenes/abs_7scenes_pose.csv_stairs_test.csv ./configs/7Scenes_config.json
./models/run_23_11_15_13_47_00 --output_path ./evals/hyperpose_7scenes_evals
```

Here, we evaluate the all checkpoint saved in the *./models/run_23_11_15_13_47_00* directory, on the *Strairs* scene in
the 7Scenes dataset.

Once the batch evaluate in done, both a .csv file and .html file will be saved with details position and orientation
estimation for each checkpoint.

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

In the example adove, we evaluate a *model_run_23_11_08_01_50_16.pth* model, located at the *./models* directory, on the
entire 7Scenes dataset.

Similar to the batch evaluation of the single-scene (see previous paragraph), you can run batch evalaution of a
multi-scene model using the `main_batch_ms_eval.py` script:

```
python main_batch_ms_eval.py mshyperpose test ./models/backbones/efficient-net-b0.pth /media/dev/data/datasets/7scenes ./datasets/7Scenes/7scenes_all_scenes.csv ./configs/7Scenes_config.json ./configs/7Scenes_config.json
./models/run_23_11_08_01_50_16 --output_path ./evals/hyperpose_7scenes_evals
```
