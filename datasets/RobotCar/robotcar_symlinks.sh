#!/bin/bash

set -x
# change the directory
ROBOTCAR_SDK_ROOT=/home/dev/Software/robotcar-dataset-sdk

ln -s ${ROBOTCAR_SDK_ROOT}/models/ /home/dev/git/multi-scene-pose-transformer/datasets/RobotCar/robotcar_camera_models
ln -s ${ROBOTCAR_SDK_ROOT}/python/ /home/dev/git/multi-scene-pose-transformer/datasets/RobotCar/robotcar_sdk
set +x