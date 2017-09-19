# alpaka
Abstraction Library for Parallel Kernel Acceleration :horse:
# HIP support for Alpaka

## Setting up HIP 
Follow [this](https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md "HIP installation") guide for installing HIP. Installing from source is the method I had used. HIP requires either nvcc or hcc to be installed on your system - have a look at the guide for more details. The following section highlights some useful tips for installation-
- If you want the hip binaries to be located in a directory that doesn't require superuser access, be sure to change the install directory of HIP by modifying the CMAKE_INSTALL_PREFIX cmake variable.  
- Also, after the installation is complete, add the following line to the .profile file in your home directory, in order to add the path to the HIP binaries to PATH-
  `PATH=$PATH:<path_to_binaries>`
- Finally, one needs to modify the FindHIP.cmake file in the cmake folder of the setup. This is because of an (apparent) typo in the default cmake file. Refer to [this](https://github.com/boradeanup/HIP/commit/287b9bb84174469fcf099b9b0ae3fb28914ac833 "Cmake fix") commit on my fork of the HIP repository for the change that is necessary.

## Verifying HIP installation
- If PATH points to the location of the HIP binaries, the following command should list several relevant environment variables, and also the selected compiler on your system-`hipconfig -f` 
- Compile and run the [square sample](https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/0_Intro/square), as pointed out in the [original](https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md#verify-your-installation) HIP install guide.

## Compiling examples with HIP back-end
As of now, the back-end has only been tested on the NVIDIA platform.
##### NVIDIA Platform
* One issue in this branch of alpaka is that the host compiler flags don't propagate to the device compiler, as they do in CUDA. This is because a counterpart to the CUDA_PROPAGATE_HOST_FLAGS cmake variable hasn't been defined in the FindHIP.cmake file.
Therefore, one needs to manually add ALL of the required compiler flags to the `HIP_NVCC_FLAGS` cmake varaible. To see this variable, toggle the advanced mode in ccmake. This is of the format-`-I<path_to_alpaka>/include/;-DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED;-DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED;-DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED;-DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED;-DALPAKA_ACC_GPU_HIP_ENABLED;-DALPAKA_DEBUG=0;-Xcompiler ,\"-fopenmp\",\"-g\"`

* Also, after the first pass of configuring the build via ccmake, one gets the following warning-`Could not find a package configuration file provided by "HIP" with any of the following names:HIPConfig.cmake , hip-config.cmake`
* To overcome this problem
  * Create a copy of the FindHIP.cmake file in it's directory, and rename it as specified in the warning, either HIPconfig.cmake or hip-config.cmake.
  * Also, specify the path to this cmake file via the HIP_DIR cmake variable.   
## Useful references
(will be updated soon)
