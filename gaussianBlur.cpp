/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

// Include OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

// Function to save image as PNG using OpenCV
void saveAsPNG(const std::string &filename, const npp::ImageCPU_8u_C1 &image)
{
  cv::Mat mat(image.height(), image.width(), CV_8UC1, image.data(), image.pitch());
  cv::imwrite(filename, mat);
}

bool printNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  try
  {
    std::string sFilename;
    char *filePath;

    findCudaDevice(argc, (const char **)argv);

    if (!printNPPinfo(argc, argv))
    {
      exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    }
    else
    {
      filePath = sdkFindFilePath("Lena.pgm", argv[0]);
    }

    if (filePath)
    {
      sFilename = filePath;
    }
    else
    {
      sFilename = "Lena.pgm";
    }

    // Open the file to check if it exists
    std::ifstream infile(sFilename.data(), std::ifstream::in);
    if (!infile.good())
    {
      std::cerr << "Unable to open: <" << sFilename.data() << ">" << std::endl;
      exit(EXIT_FAILURE);
    }
    infile.close();

    std::string sResultFilename = sFilename;
    std::string::size_type dot = sResultFilename.rfind('.');
    if (dot != std::string::npos)
    {
      sResultFilename = sResultFilename.substr(0, dot);
    }
    sResultFilename += "_gaussianBlur.pgm";

    if (checkCmdLineFlag(argc, (const char **)argv, "output"))
    {
      char *outputFilePath;
      getCmdLineArgumentString(argc, (const char **)argv, "output", &outputFilePath);
      sResultFilename = outputFilePath;
    }

    // Load the image from disk
    npp::ImageCPU_8u_C1 oHostSrc;
    npp::loadImage(sFilename, oHostSrc);

    // Declare a device image and copy from the host
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
    NppiSize oSrcSize = {static_cast<int>(oDeviceSrc.width()), static_cast<int>(oDeviceSrc.height())};
    NppiPoint oSrcOffset = {0, 0};

    // Create struct with Gaussian blur mask size
    NppiMaskSize oMaskSize = NPP_MASK_SIZE_5_X_5; // Correct mask size
    NppiPoint oAnchor = {2, 2};                   // Anchor for 5x5 mask

    // Allocate device image for the result
    npp::ImageNPP_8u_C1 oDeviceDst(oSrcSize.width, oSrcSize.height);

    // Run Gaussian blur
    NppStatus status = nppiFilterGaussBorder_8u_C1R(
        oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
        oDeviceDst.data(), oDeviceDst.pitch(), oSrcSize, oMaskSize, NPP_BORDER_REPLICATE);

    if (status != NPP_SUCCESS)
    {
      std::cerr << "NPP Gaussian blur failed with error code: " << status << std::endl;
      exit(EXIT_FAILURE);
    }

    // Copy the result from device to host and save it
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    saveImage(sResultFilename, oHostDst);

    std::cout << "Saved image: " << sResultFilename << std::endl;

    exit(EXIT_SUCCESS);
  }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknown type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }

  // Save the image as PGM
  saveImage(sResultFilename, oHostDst);

  // Save the image as PNG
  std::string sResultPNGFilename = sResultFilename;
  std::string::size_type dot = sResultPNGFilename.rfind('.');
  if (dot != std::string::npos)
  {
    sResultPNGFilename = sResultPNGFilename.substr(0, dot);
  }
  sResultPNGFilename += "_gaussianBlur.png";
  saveAsPNG(sResultPNGFilename, oHostDst);

  std::cout << "Saved image: " << sResultFilename << std::endl;
  std::cout << "Saved image: " << sResultPNGFilename << std::endl;

  return 0;
}
