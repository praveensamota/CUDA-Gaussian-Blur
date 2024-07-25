This project implements a Gaussian Blur filter using CUDA to accelerate the processing of images. The primary goal is to leverage GPU parallelism to efficiently apply the blur effect on a set of texture images from the USC-SIPI Image Database. The implementation includes:

CUDA Kernel Development: Custom CUDA kernels were written to perform the Gaussian Blur operation.
Image Processing: The project processes bulk image data, converting them from PGM to PNG format using OpenCV.
Performance Optimization: Various techniques were employed to optimize the CUDA code, including memory management and kernel execution strategies.
Multi-GPU Considerations: The project was designed with scalability in mind, enabling potential use with multiple GPUs.

The project uses the same template as used in the NPP Box Filter Laboratory. 

The image used here is Lena.pgm and the output we get after processing is Lena_gaussianBlur.pgm as provided in the master branch of this git repository. I converted the pgm file into a jpeg image using the additional utils file i.e. con.py

To run the CUDA Gaussian Blur project and process images, follow these steps:

Build the Project:
Make sure you have the necessary dependencies installed (CUDA, OpenCV, etc.). Then, use the Makefile to build the project:

make clean build

Run the Gaussian Blur Program:
Execute the program to apply the Gaussian Blur filter to the images:

make run
