# 3D Tensors for Blaze

This project implements 3D datastructures (tensors) that integrate well with the [Blaze library](https://bitbucket.org/blaze-lib/blaze/src). 

The implemented facilities are verified using a thorough testing environment. The [CircleCI](https://circleci.com/gh/STEllAR-GROUP/blaze_tensor) contiguous integration service tracks the current build status for the master branch:
[![CircleCI](https://circleci.com/gh/STEllAR-GROUP/blaze_tensor.svg?style=svg)](https://circleci.com/gh/STEllAR-GROUP/blaze_tensor).

Currently we have implemented:

### Datastructures

- `blaze::DynamicTensor<T>`: a row-major 3D dense array data structure of arbitrary types
- `blaze::CustomTensor<T, ...>`: a non-owning 3D dense array data structure usable to refer to some other 3D dense array

### Views

- `blaze::SubTensor<...>`: a view representing a contigous subregion of a dense 3D array that is statically or dynamically sized
- `blaze::PageSlice<...>`: a view representing a slice of 'thickness' one along the row/column plane of a 3D dense array
- `blaze::ColumnSlice<...>`: a view representing a slice of 'thickness' one along the page/column plane of a 3D dense array
- `blaze::RowSlice<...>`: a view representing a slice of 'thickness' one along the column/page plane of a 3D dense array

### Arithmetic Operations

- All element-wise arithmetic operations that are supported by the Blaze library: element-wise addition, subtraction, division, Schur-multiplication, scalar multiplication, boolean comparison operations, and many mathematical operations like `sqrt`, `cqrt`, `abs`, `sign`, `floor`, etc.

## Building and installing BlazeTensor

In order to use BlazeTensor you will need a proper installation of the [Blaze library](https://bitbucket.org/blaze-lib/blaze/src). Please see [here](https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation) for instructions on how to install it. If you want to run the BlazeTensor tests you will also need the Blaze source directory.

1. Clone this repository
2. Create a build directory
   ```
   $ mkdir build
   $ cd build
   ```
3. Call cmake with the relevant options:
   ```
   $ cmake -DCMAKE_INSTALL_PREFIX=/opt/BlazeTensor -Dblaze_DIR=<blaze_dir> <srcdir>
   ```
   here: `<blaze_dir>` should refer to the directory that contains the file `blaze-config.cmake` that 
   was created during installation of Blaze.  
4. Build and install:
   ```
   $ make
   $ make install
   ```
5. If you want to build the tests, additionally specify `-DBLAZETENSOR_WITH_TESTS=ON` and `-Dblazetest_DIR=<blazesrc/blazetest>` 
   on the `cmake` command line. Run the tests with `make tests`.
   
BlazeTensor is a header only C++ library. Projects depending on it should make sure the headers are being found by the compiler. If your depending project uses `cmake`, just add `find_package(BlazeTensor)` to your scripts and refer to the target named `BlazeTensor::BlazeTensor`.

## Communication
A channel on the freenode IRC network is used for discussions on BlazeTensor: #ste||ar on freenode (via SSL).
Feel free to use the [Github issue tracker](https://github.com/STEllAR-GROUP/blaze_tensor/issues) for questions, 
bug reports, and feature requests. 

We have created a list of things that need to be implemented: [TODO: Things to implement](https://github.com/STEllAR-GROUP/blaze_tensor/issues/2). This is a good starting point iIf you would like to help developing BlazeTensor.

## License
BlazeTensor is released under the terms of the [New (Revised) BSD license](https://github.com/STEllAR-GROUP/blaze_tensor/blob/master/LICENSE).

## Acknowledgements

We would like to acknowledge the NSF, DoD, and the Center for Computation
and Technology (CCT) at Louisiana State University (LSU).

BlazeTensor is currently funded by:

* The National Science Foundation through awards 1737785 (Phylanx).

  Any opinions, findings, and conclusions or recommendations expressed in this
  material are those of the author(s) and do not necessarily reflect the views
  of the National Science Foundation.
  
* The Defense Technical Information Center (DTIC) under contract FA8075-14-D-0002/0007

  Neither the United States Government nor any agency thereof, nor any of their 
  employees makes any warranty, express or implied, or assumes any legal liability 
  or responsibility for the accuracy, completeness, or usefulness of any information, 
  apparatus, product, or process disclosed, or represents that its use would not 
  infringe privately owned rights.
