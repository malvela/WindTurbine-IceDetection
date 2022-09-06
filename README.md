# Wind turbine ice detection: An approach with deep transfer learning for image data
Supplementary material for the research work "Detecting ice on wind turbine rotor blades: Towards deep transfer learning for image data".

<img src="https://user-images.githubusercontent.com/68553692/176689481-5fc86870-d7ed-4ec5-bb98-7abaf9564dc4.png" width="400" height="300" />

A software code based on tiny DL models for predicting blade icing on images of rotor blades using resource-constrained devices.

## Download and use of the repository:
To download this repository and its submodules use

    git clone --recurse-submodules https://github.com/malvela/WindTurbine-IceDetection.git

## Individual files and functionality:
This software involves pyhton files for predicting ice on wind turbine rotor blades using a tiny computer:

    - Convert_model.py: It converts a h5 model to a tensorrt model format.
    - Transfer_learning.py: Training and evaluation of the presented models.
    - Eval_speed: Folder contains scripts that evaluate the model inference time. 
        - pred_img_eval_speed.py: for the h5-format model.
        - pred_img_trt_eval_speed.py: for the tensorrt-format model.

## Cite as:

If you are using this software in your academic work please cite it as Alvela Nieto, M.T., Gelbhardt, H., Ohlendorf, J-H., Thoben, K.-D. (2022). Detecting Ice on Wind Turbine Rotor Blades: Towards Deep Transfer Learning for Image Data. In: Advances in System-Integrated Intelligence. SYSINT 2022. Lecture Notes in Networks and Systems, vol 546. Springer, Cham. https://doi.org/10.1007/978-3-031-16281-7_54

## License:

This repo is based on the MIT License, which allows free use of the provided resources, subject to the origina sources being credit/acknowledged appropriately. The software/resources under MIT license is provided as is, without any liability or warranty at the end of the authors.
