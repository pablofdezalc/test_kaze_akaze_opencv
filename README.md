## README - Test KAZE and A-KAZE OpenCV port

Version: 1.0.0
Date: 05-06-2014

You can get the latest version of the code from github:
`https://github.com/pablofdezalc/test_kaze_akaze_opencv`

## What is this file?

This file explains how to make use of the OpenCV KAZE and A-KAZE features code in
an image matching application

## Library Dependencies

The code is mainly based on the OpenCV library using the C++ interface. You will need
to download and install the master branch version from OpenCV github repository that contains
the KAZE and A-KAZE interface:
`https://github.com/Itseez/opencv`

## Getting Started

Compiling:

1. `$ mkdir build`
2. `$ cd build>`
3. `$ cmake ..`
4. `$ make`

If the compilation is successful you should see four executables:
- `test_kaze`
- `test_kaze_match`
- `test_akaze`
- `test_akaze_match`

The matching executables expect three input arguments, two input images and an homography txt file
that describes the homography transformation between these images, similar as in the Oxford benchmark
from Mikolajczyk et al.


