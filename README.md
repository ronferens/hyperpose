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


