//=================================================================================================
/*!
//  \file src/mathtest/dynamictensor/ClassTest2.cpp
//  \brief Source file for the DynamicTensor class test (part 2)
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018 Hartmut Kaiser - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cstdlib>
#include <iostream>
#include <memory>

#include <blaze/system/Platform.h>
#include <blaze/util/Complex.h>
#include <blaze/util/Memory.h>
#include <blaze/util/policies/Deallocate.h>
#include <blaze/util/Random.h>
#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>

#include <blaze_tensor/math/CustomTensor.h>
#include <blazetest/mathtest/dynamictensor/ClassTest.h>

namespace blazetest {

namespace mathtest {

namespace dynamictensor {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the DynamicTensor class test.
//
// \exception std::runtime_error Operation error detected.
*/
ClassTest::ClassTest()
{
   testSchurAssign();
   testMultAssign();
   testScaling();
   testFunctionCall();
   testAt();
   testIterator();
   testNonZeros();
   testReset();
   testClear();
   testResize();
   testExtend();
   testReserve();
   testShrinkToFit();
   testSwap();
   testTranspose();
   testCTranspose();
   testIsDefault();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the DynamicTensor Schur product assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the Schur product assignment operators of the DynamicTensor
// class template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testSchurAssign()
{
   //=====================================================================================
   // Row-major dense tensor Schur product assignment
   //=====================================================================================

   {
      test_ = "DynamicTensor dense tensor Schur product assignment (mixed type)";

      blaze::DynamicTensor<short> mat1{{{1, 2, 0}, {-3, 0, 4}},
                                       {{1, 2, 0}, {-3, 0, 4}}};

      blaze::DynamicTensor<int> mat2{{{0, -2, 6}, {5, 0, 0}},
                                     {{0, -2, 6}, {5, 0, 0}}};

      mat2 %= mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2,  4UL );
      checkNonZeros( mat2, 0UL, 0UL, 1UL );
      checkNonZeros( mat2, 1UL, 0UL, 1UL );
      checkNonZeros( mat2, 0UL, 1UL, 1UL );
      checkNonZeros( mat2, 1UL, 1UL, 1UL );

      if( mat2(0,0,0) !=   0 || mat2(0,0,1) != -4 || mat2(0,0,2) != 0 ||
          mat2(0,1,0) != -15 || mat2(0,1,1) !=  0 || mat2(0,1,2) != 0 ||
          mat2(1,0,0) !=   0 || mat2(1,0,1) != -4 || mat2(1,0,2) != 0 ||
          mat2(1,1,0) != -15 || mat2(1,1,1) !=  0 || mat2(1,1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n((   0 -4  0 )\n( -15  0  0 ))\n((   0 -4  0 )\n( -15  0  0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor dense tensor Schur product assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      AlignedPadded mat1( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat1 = 0;
      mat1(0,0,0) =  1;
      mat1(0,0,1) =  2;
      mat1(0,1,0) = -3;
      mat1(0,1,2) =  4;
      mat1(1,0,0) =  1;
      mat1(1,0,1) =  2;
      mat1(1,1,0) = -3;
      mat1(1,1,2) =  4;

      blaze::DynamicTensor<int> mat2{{{0, -2, 6}, {5, 0, 0}},
                                     {{0, -2, 6}, {5, 0, 0}}};

      mat2 %= mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2,  4UL );
      checkNonZeros( mat2, 0UL, 0UL, 1UL );
      checkNonZeros( mat2, 1UL, 0UL, 1UL );
      checkNonZeros( mat2, 0UL, 1UL, 1UL );
      checkNonZeros( mat2, 1UL, 1UL, 1UL );

      if( mat2(0,0,0) !=   0 || mat2(0,0,1) != -4 || mat2(0,0,2) != 0 ||
          mat2(0,1,0) != -15 || mat2(0,1,1) !=  0 || mat2(0,1,2) != 0 ||
          mat2(1,0,0) !=   0 || mat2(1,0,1) != -4 || mat2(1,0,2) != 0 ||
          mat2(1,1,0) != -15 || mat2(1,1,1) !=  0 || mat2(1,1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n((   0 -4  0 )\n( -15  0  0 ))\n((   0 -4  0 )\n( -15  0  0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor dense tensor Schur product assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[13UL] );
      UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 2UL, 3UL );
      mat1 = 0;
      mat1(0,0,0) =  1;
      mat1(0,0,1) =  2;
      mat1(0,1,0) = -3;
      mat1(0,1,2) =  4;
      mat1(1,0,0) =  1;
      mat1(1,0,1) =  2;
      mat1(1,1,0) = -3;
      mat1(1,1,2) =  4;

      blaze::DynamicTensor<int> mat2{{{0, -2, 6}, {5, 0, 0}},
                                     {{0, -2, 6}, {5, 0, 0}}};

      mat2 %= mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2,  4UL );
      checkNonZeros( mat2, 0UL, 0UL, 1UL );
      checkNonZeros( mat2, 1UL, 0UL, 1UL );
      checkNonZeros( mat2, 0UL, 1UL, 1UL );
      checkNonZeros( mat2, 1UL, 1UL, 1UL );

      if( mat2(0,0,0) !=   0 || mat2(0,0,1) != -4 || mat2(0,0,2) != 0 ||
          mat2(0,1,0) != -15 || mat2(0,1,1) !=  0 || mat2(0,1,2) != 0 ||
          mat2(1,0,0) !=   0 || mat2(1,0,1) != -4 || mat2(1,0,2) != 0 ||
          mat2(1,1,0) != -15 || mat2(1,1,1) !=  0 || mat2(1,1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n((   0 -4  0 )\n( -15  0  0 ))\n((   0 -4  0 )\n( -15  0  0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DynamicTensor multiplication assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the multiplication assignment operators of the DynamicTensor
// class template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testMultAssign()
{
   //=====================================================================================
   // Row-major dense tensor multiplication assignment
   //=====================================================================================
#if 0
   {
      test_ = "DynamicTensor dense tensor multiplication assignment (mixed type)";

      blaze::DynamicTensor<short> mat1{
          {{0, 2, 0}, {1, 3, 0}, {0, 0, 0}},
          {{0, 2, 0}, {1, 3, 0}, {0, 0, 0}}};

      blaze::DynamicTensor<int> mat2{{{1, 0, 2}, {0, 3, 0}, {4, 0, 5}},
                                     {{1, 0, 2}, {0, 3, 0}, {4, 0, 5}}};

      mat2 *= mat1;

      checkRows    ( mat2,  3UL );
      checkColumns ( mat2,  4UL );
      checkPages   ( mat2,  2UL );
      checkNonZeros( mat2, 14UL );
      checkNonZeros( mat2,  0UL, 0UL, 2UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  2UL, 0UL, 2UL );
      checkNonZeros( mat2,  0UL, 1UL, 2UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );
      checkNonZeros( mat2,  2UL, 1UL, 2UL );

      if( mat2(0,0,0) != 0 || mat2(0,0,1) != 0 || mat2(0,0,2) != 0 ||
          mat2(0,1,0) != 0 || mat2(0,1,1) != 9 || mat2(0,1,2) != 0 ||
          mat2(0,2,0) != 0 || mat2(0,2,1) != 0 || mat2(0,2,2) != 0 ||
          mat2(1,0,0) != 0 || mat2(1,0,1) != 0 || mat2(1,0,2) != 0 ||
          mat2(1,1,0) != 0 || mat2(1,1,1) != 9 || mat2(1,1,2) != 0 ||
          mat2(1,2,0) != 0 || mat2(1,2,1) != 0 || mat2(1,2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 0 0 0 )\n( 0 9 0 )\n( 0 0 0 ))\n(( 0 0 0 )\n( 0 9 0 )\n( 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor dense tensor multiplication assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;
      using blaze::rowMajor;

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 48UL ) );
      AlignedPadded mat1( memory.get(), 3UL, 4UL, 16UL );
      mat1 = 0;
      mat1(0,1) = 2;
      mat1(1,0) = 1;
      mat1(1,1) = 3;
      mat1(1,3) = 4;
      mat1(2,3) = 5;

      blaze::DynamicTensor<int> mat2{ { 1, 0, 2 },
                                                      { 0, 3, 0 },
                                                      { 4, 0, 5 } };

      mat2 *= mat1;

      checkRows    ( mat2, 3UL );
      checkColumns ( mat2, 4UL );
      checkNonZeros( mat2, 7UL );
      checkNonZeros( mat2, 0UL, 2UL );
      checkNonZeros( mat2, 1UL, 3UL );
      checkNonZeros( mat2, 2UL, 2UL );

      if( mat2(0,0) != 0 || mat2(0,1) != 2 || mat2(0,2) != 0 || mat2(0,3) != 10 ||
          mat2(1,0) != 3 || mat2(1,1) != 9 || mat2(1,2) != 0 || mat2(1,3) != 12 ||
          mat2(2,0) != 0 || mat2(2,1) != 8 || mat2(2,2) != 0 || mat2(2,3) != 25 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n( 0 2 0 10 )\n( 3 9 0 12 )\n( 0 8 0 25 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor dense tensor multiplication assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::rowMajor;

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[13UL] );
      UnalignedUnpadded mat1( memory.get()+1UL, 3UL, 4UL );
      mat1 = 0;
      mat1(0,1) = 2;
      mat1(1,0) = 1;
      mat1(1,1) = 3;
      mat1(1,3) = 4;
      mat1(2,3) = 5;

      blaze::DynamicTensor<int> mat2{ { 1, 0, 2 },
                                                      { 0, 3, 0 },
                                                      { 4, 0, 5 } };

      mat2 *= mat1;

      checkRows    ( mat2, 3UL );
      checkColumns ( mat2, 4UL );
      checkNonZeros( mat2, 7UL );
      checkNonZeros( mat2, 0UL, 2UL );
      checkNonZeros( mat2, 1UL, 3UL );
      checkNonZeros( mat2, 2UL, 2UL );

      if( mat2(0,0) != 0 || mat2(0,1) != 2 || mat2(0,2) != 0 || mat2(0,3) != 10 ||
          mat2(1,0) != 3 || mat2(1,1) != 9 || mat2(1,2) != 0 || mat2(1,3) != 12 ||
          mat2(2,0) != 0 || mat2(2,1) != 8 || mat2(2,2) != 0 || mat2(2,3) != 25 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n( 0 2 0 10 )\n( 3 9 0 12 )\n( 0 8 0 25 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of all DynamicTensor (self-)scaling operations.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all available ways to scale an instance of the DynamicTensor
// class template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testScaling()
{
   //=====================================================================================
   // Row-major self-scaling (M*=s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M*=s)";

      blaze::DynamicTensor<int> mat{{{0, 0, 0}, {0, 0, 1}, {-2, 0, 3}},
                                    {{0, 0, 0}, {0, 0, 1}, {-2, 0, 3}}};

      mat *= 2;

      checkRows    ( mat, 3UL );
      checkColumns ( mat, 3UL );
      checkPages   ( mat, 2UL );
      checkNonZeros( mat, 6UL );
      checkNonZeros( mat, 0UL, 0UL, 0UL );
      checkNonZeros( mat, 1UL, 0UL, 1UL );
      checkNonZeros( mat, 2UL, 0UL, 2UL );
      checkNonZeros( mat, 0UL, 1UL, 0UL );
      checkNonZeros( mat, 1UL, 1UL, 1UL );
      checkNonZeros( mat, 2UL, 1UL, 2UL );

      if( mat(0,0,0) !=  0 || mat(0,0,1) != 0 || mat(0,0,2) != 0 ||
          mat(0,1,0) !=  0 || mat(0,1,1) != 0 || mat(0,1,2) != 2 ||
          mat(0,2,0) != -4 || mat(0,2,1) != 0 || mat(0,2,2) != 6 ||
          mat(1,0,0) !=  0 || mat(1,0,1) != 0 || mat(1,0,2) != 0 ||
          mat(1,1,0) !=  0 || mat(1,1,1) != 0 || mat(1,1,2) != 2 ||
          mat(1,2,0) != -4 || mat(1,2,1) != 0 || mat(1,2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 )\n(  0 0 2 )\n( -4 0 6 ))\n((  0 0 0 )\n(  0 0 2 )\n( -4 0 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=M*s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=M*s)";

      blaze::DynamicTensor<int> mat{{{0, 0, 0}, {0, 0, 1}, {-2, 0, 3}},
                                    {{0, 0, 0}, {0, 0, 1}, {-2, 0, 3}}};

      mat = mat * 2;

      checkRows    ( mat, 3UL );
      checkColumns ( mat, 3UL );
      checkPages   ( mat, 2UL );
      checkNonZeros( mat, 6UL );
      checkNonZeros( mat, 0UL, 0UL, 0UL );
      checkNonZeros( mat, 1UL, 0UL, 1UL );
      checkNonZeros( mat, 2UL, 0UL, 2UL );
      checkNonZeros( mat, 0UL, 1UL, 0UL );
      checkNonZeros( mat, 1UL, 1UL, 1UL );
      checkNonZeros( mat, 2UL, 1UL, 2UL );

      if( mat(0,0,0) !=  0 || mat(0,0,1) != 0 || mat(0,0,2) != 0 ||
          mat(0,1,0) !=  0 || mat(0,1,1) != 0 || mat(0,1,2) != 2 ||
          mat(0,2,0) != -4 || mat(0,2,1) != 0 || mat(0,2,2) != 6 ||
          mat(1,0,0) !=  0 || mat(1,0,1) != 0 || mat(1,0,2) != 0 ||
          mat(1,1,0) !=  0 || mat(1,1,1) != 0 || mat(1,1,2) != 2 ||
          mat(1,2,0) != -4 || mat(1,2,1) != 0 || mat(1,2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 )\n(  0 0 2 )\n( -4 0 6 ))\n((  0 0 0 )\n(  0 0 2 )\n( -4 0 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=s*M)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=s*M)";

      blaze::DynamicTensor<int> mat{{{0, 0, 0}, {0, 0, 1}, {-2, 0, 3}},
                                    {{0, 0, 0}, {0, 0, 1}, {-2, 0, 3}}};

      mat = 2 * mat;

      checkRows    ( mat, 3UL );
      checkColumns ( mat, 3UL );
      checkPages   ( mat, 2UL );
      checkNonZeros( mat, 6UL );
      checkNonZeros( mat, 0UL, 0UL, 0UL );
      checkNonZeros( mat, 1UL, 0UL, 1UL );
      checkNonZeros( mat, 2UL, 0UL, 2UL );
      checkNonZeros( mat, 0UL, 1UL, 0UL );
      checkNonZeros( mat, 1UL, 1UL, 1UL );
      checkNonZeros( mat, 2UL, 1UL, 2UL );

      if( mat(0,0,0) !=  0 || mat(0,0,1) != 0 || mat(0,0,2) != 0 ||
          mat(0,1,0) !=  0 || mat(0,1,1) != 0 || mat(0,1,2) != 2 ||
          mat(0,2,0) != -4 || mat(0,2,1) != 0 || mat(0,2,2) != 6 ||
          mat(1,0,0) !=  0 || mat(1,0,1) != 0 || mat(1,0,2) != 0 ||
          mat(1,1,0) !=  0 || mat(1,1,1) != 0 || mat(1,1,2) != 2 ||
          mat(1,2,0) != -4 || mat(1,2,1) != 0 || mat(1,2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 )\n(  0 0 2 )\n( -4 0 6 ))\n((  0 0 0 )\n(  0 0 2 )\n( -4 0 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M/=s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M/=s)";

      blaze::DynamicTensor<int> mat{{{0, 0, 0}, {0, 0, 2}, {-4, 0, 6}},
                                    {{0, 0, 0}, {0, 0, 2}, {-4, 0, 6}}};

      mat /= 2;

      checkRows    ( mat, 3UL );
      checkColumns ( mat, 3UL );
      checkPages   ( mat, 2UL );
      checkNonZeros( mat, 6UL );
      checkNonZeros( mat, 0UL, 0UL, 0UL );
      checkNonZeros( mat, 1UL, 0UL, 1UL );
      checkNonZeros( mat, 2UL, 0UL, 2UL );
      checkNonZeros( mat, 0UL, 1UL, 0UL );
      checkNonZeros( mat, 1UL, 1UL, 1UL );
      checkNonZeros( mat, 2UL, 1UL, 2UL );

      if( mat(0,0,0) !=  0 || mat(0,0,1) != 0 || mat(0,0,2) != 0 ||
          mat(0,1,0) !=  0 || mat(0,1,1) != 0 || mat(0,1,2) != 1 ||
          mat(0,2,0) != -2 || mat(0,2,1) != 0 || mat(0,2,2) != 3 ||
          mat(1,0,0) !=  0 || mat(1,0,1) != 0 || mat(1,0,2) != 0 ||
          mat(1,1,0) !=  0 || mat(1,1,1) != 0 || mat(1,1,2) != 1 ||
          mat(1,2,0) != -2 || mat(1,2,1) != 0 || mat(1,2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 )\n(  0 0 1 )\n( -2 0 3 ))\n((  0 0 0 )\n(  0 0 1 )\n( -2 0 3 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=M/s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=M/s)";

      blaze::DynamicTensor<int> mat{{{0, 0, 0}, {0, 0, 2}, {-4, 0, 6}},
                                    {{0, 0, 0}, {0, 0, 2}, {-4, 0, 6}}};

      mat = mat / 2;

      checkRows    ( mat, 3UL );
      checkColumns ( mat, 3UL );
      checkPages   ( mat, 2UL );
      checkNonZeros( mat, 6UL );
      checkNonZeros( mat, 0UL, 0UL, 0UL );
      checkNonZeros( mat, 1UL, 0UL, 1UL );
      checkNonZeros( mat, 2UL, 0UL, 2UL );
      checkNonZeros( mat, 0UL, 1UL, 0UL );
      checkNonZeros( mat, 1UL, 1UL, 1UL );
      checkNonZeros( mat, 2UL, 1UL, 2UL );

      if( mat(0,0,0) !=  0 || mat(0,0,1) != 0 || mat(0,0,2) != 0 ||
          mat(0,1,0) !=  0 || mat(0,1,1) != 0 || mat(0,1,2) != 1 ||
          mat(0,2,0) != -2 || mat(0,2,1) != 0 || mat(0,2,2) != 3 ||
          mat(1,0,0) !=  0 || mat(1,0,1) != 0 || mat(1,0,2) != 0 ||
          mat(1,1,0) !=  0 || mat(1,1,1) != 0 || mat(1,1,2) != 1 ||
          mat(1,2,0) != -2 || mat(1,2,1) != 0 || mat(1,2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 )\n(  0 0 1 )\n( -2 0 3 ))\n((  0 0 0 )\n(  0 0 1 )\n( -2 0 3 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major DynamicTensor::scale()
   //=====================================================================================

   {
      test_ = "Row-major DynamicTensor::scale() (int)";

      // Initialization check
      blaze::DynamicTensor<double> mat{{{1, 2}, {3, 4}, {5, 6}},
                                       {{1, 2}, {3, 4}, {5, 6}}};

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  2UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 12UL );
      checkNonZeros( mat, 12UL );
      checkNonZeros( mat, 0UL, 0UL, 2UL );
      checkNonZeros( mat, 1UL, 0UL, 2UL );
      checkNonZeros( mat, 2UL, 0UL, 2UL );
      checkNonZeros( mat, 0UL, 1UL, 2UL );
      checkNonZeros( mat, 1UL, 1UL, 2UL );
      checkNonZeros( mat, 2UL, 1UL, 2UL );

      if( mat(0,0,0) != 1 || mat(0,0,1) != 2 ||
          mat(0,1,0) != 3 || mat(0,1,1) != 4 ||
          mat(0,2,0) != 5 || mat(0,2,1) != 6 ||
          mat(1,0,0) != 1 || mat(1,0,1) != 2 ||
          mat(1,1,0) != 3 || mat(1,1,1) != 4 ||
          mat(1,2,0) != 5 || mat(1,2,1) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 1 2 )\n( 3 4 )\n( 5 6 ))\n(( 1 2 )\n( 3 4 )\n( 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Integral scaling of the tensor
      mat.scale( 2 );

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  2UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 12UL );
      checkNonZeros( mat, 12UL );
      checkNonZeros( mat, 0UL, 0UL, 2UL );
      checkNonZeros( mat, 1UL, 0UL, 2UL );
      checkNonZeros( mat, 2UL, 0UL, 2UL );
      checkNonZeros( mat, 0UL, 1UL, 2UL );
      checkNonZeros( mat, 1UL, 1UL, 2UL );
      checkNonZeros( mat, 2UL, 1UL, 2UL );

      if( mat(0,0,0) !=  2 || mat(0,0,1) !=  4 ||
          mat(0,1,0) !=  6 || mat(0,1,1) !=  8 ||
          mat(0,2,0) != 10 || mat(0,2,1) != 12 ||
          mat(1,0,0) !=  2 || mat(1,0,1) !=  4 ||
          mat(1,1,0) !=  6 || mat(1,1,1) !=  8 ||
          mat(1,2,0) != 10 || mat(1,2,1) != 12 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Scaling failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  2  4 )\n(  6  8 )\n( 10 12 ))\n((  2  4 )\n(  6  8 )\n( 10 12 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Floating point scaling of the tensor
      mat.scale( 0.5 );

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  2UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 12UL );
      checkNonZeros( mat, 12UL );
      checkNonZeros( mat, 0UL, 0UL, 2UL );
      checkNonZeros( mat, 1UL, 0UL, 2UL );
      checkNonZeros( mat, 2UL, 0UL, 2UL );
      checkNonZeros( mat, 0UL, 1UL, 2UL );
      checkNonZeros( mat, 1UL, 1UL, 2UL );
      checkNonZeros( mat, 2UL, 1UL, 2UL );

      if( mat(0,0,0) != 1 || mat(0,0,1) != 2 ||
          mat(0,1,0) != 3 || mat(0,1,1) != 4 ||
          mat(0,2,0) != 5 || mat(0,2,1) != 6 ||
          mat(1,0,0) != 1 || mat(1,0,1) != 2 ||
          mat(1,1,0) != 3 || mat(1,1,1) != 4 ||
          mat(1,2,0) != 5 || mat(1,2,1) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Scaling failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 1 2 )\n( 3 4 )\n( 5 6 ))\n(( 1 2 )\n( 3 4 )\n( 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major DynamicTensor::scale() (complex)";

      using blaze::complex;

      blaze::DynamicTensor<complex<float>> mat( 2UL, 2UL, 2UL );
      mat(0,0,0) = complex<float>( 1.0F, 0.0F );
      mat(0,0,1) = complex<float>( 2.0F, 0.0F );
      mat(0,1,0) = complex<float>( 3.0F, 0.0F );
      mat(0,1,1) = complex<float>( 4.0F, 0.0F );
      mat(1,0,0) = complex<float>( 1.0F, 0.0F );
      mat(1,0,1) = complex<float>( 2.0F, 0.0F );
      mat(1,1,0) = complex<float>( 3.0F, 0.0F );
      mat(1,1,1) = complex<float>( 4.0F, 0.0F );
      mat.scale( complex<float>( 3.0F, 0.0F ) );

      checkRows    ( mat, 2UL );
      checkColumns ( mat, 2UL );
      checkPages   ( mat, 2UL );
      checkCapacity( mat, 4UL );
      checkNonZeros( mat, 8UL );
      checkNonZeros( mat, 0UL, 0UL, 2UL );
      checkNonZeros( mat, 1UL, 0UL, 2UL );
      checkNonZeros( mat, 0UL, 1UL, 2UL );
      checkNonZeros( mat, 1UL, 1UL, 2UL );

      if( mat(0,0,0) != complex<float>( 3.0F, 0.0F ) || mat(0,0,1) != complex<float>(  6.0F, 0.0F ) ||
          mat(0,1,0) != complex<float>( 9.0F, 0.0F ) || mat(0,1,1) != complex<float>( 12.0F, 0.0F ) ||
          mat(1,0,0) != complex<float>( 3.0F, 0.0F ) || mat(1,0,1) != complex<float>(  6.0F, 0.0F ) ||
          mat(1,1,0) != complex<float>( 9.0F, 0.0F ) || mat(1,1,1) != complex<float>( 12.0F, 0.0F ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n( ( 3,0) ( 6,0)\n( 9,0) (12,0) )\n( ( 3,0) ( 6,0)\n( 9,0) (12,0) )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DynamicTensor function call operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the function call operator
// of the DynamicTensor class template. In case an error is detected, a \a std::runtime_error
// exception is thrown.
*/
void ClassTest::testFunctionCall()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================
   {
      test_ = "Row-major DynamicTensor::operator()";

      // Assignment to the element (2,1)
      blaze::DynamicTensor<int> mat( 2UL, 3UL, 5UL, 0 );
      mat(0,2,1) = 1;
      mat(1,2,1) = 1;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat,  2UL );
      checkNonZeros( mat,  0UL, 0UL, 0UL );
      checkNonZeros( mat,  1UL, 0UL, 0UL );
      checkNonZeros( mat,  2UL, 0UL, 1UL );
      checkNonZeros( mat,  0UL, 1UL, 0UL );
      checkNonZeros( mat,  1UL, 1UL, 0UL );
      checkNonZeros( mat,  2UL, 1UL, 1UL );

      if( mat(0,2,1) != 1 || mat(1,2,1) != 1) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 0 0 0 0 0 )\n( 0 0 0 0 0 )\n( 0 1 0 0 0 ))\n(( 0 0 0 0 0 )\n( 0 0 0 0 0 )\n( 0 1 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Assignment to the element (1,4)
      mat(0,1,4) = 2;
      mat(1,1,4) = 2;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat,  4UL );
      checkNonZeros( mat,  0UL, 0UL, 0UL );
      checkNonZeros( mat,  1UL, 0UL, 1UL );
      checkNonZeros( mat,  2UL, 0UL, 1UL );
      checkNonZeros( mat,  0UL, 1UL, 0UL );
      checkNonZeros( mat,  1UL, 1UL, 1UL );
      checkNonZeros( mat,  2UL, 1UL, 1UL );

      if( mat(0,1,4) != 2 || mat(0,2,1) != 1 || mat(1,1,4) != 2 || mat(1,2,1) != 1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 0 0 0 0 0 )\n( 0 0 0 0 2 )\n( 0 1 0 0 0 ))\n(( 0 0 0 0 0 )\n( 0 0 0 0 2 )\n( 0 1 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Assignment to the element (0,3)
      mat(0,0,3) = 3;
      mat(1,0,3) = 3;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat,  6UL );
      checkNonZeros( mat,  0UL, 0UL, 1UL );
      checkNonZeros( mat,  1UL, 0UL, 1UL );
      checkNonZeros( mat,  2UL, 0UL, 1UL );
      checkNonZeros( mat,  0UL, 1UL, 1UL );
      checkNonZeros( mat,  1UL, 1UL, 1UL );
      checkNonZeros( mat,  2UL, 1UL, 1UL );

      if( mat(0,0,3) != 3 || mat(0,1,4) != 2 || mat(0,2,1) != 1 ||
          mat(1,0,3) != 3 || mat(1,1,4) != 2 || mat(1,2,1) != 1) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 0 0 0 3 0 )\n( 0 0 0 0 2 )\n( 0 1 0 0 0 ))\n(( 0 0 0 3 0 )\n( 0 0 0 0 2 )\n( 0 1 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Assignment to the element (2,2)
      mat(0,2,2) = 4;
      mat(1,2,2) = 4;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat,  8UL );
      checkNonZeros( mat,  0UL, 0UL, 1UL );
      checkNonZeros( mat,  1UL, 0UL, 1UL );
      checkNonZeros( mat,  2UL, 0UL, 2UL );
      checkNonZeros( mat,  0UL, 1UL, 1UL );
      checkNonZeros( mat,  1UL, 1UL, 1UL );
      checkNonZeros( mat,  2UL, 1UL, 2UL );

      if( mat(0,0,3) != 3 || mat(0,1,4) != 2 || mat(0,2,1) != 1 || mat(0,2,2) != 4 ||
          mat(1,0,3) != 3 || mat(1,1,4) != 2 || mat(1,2,1) != 1 || mat(1,2,2) != 4) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 0 0 0 3 0 )\n( 0 0 0 0 2 )\n( 0 1 4 0 0 ))\n(( 0 0 0 3 0 )\n( 0 0 0 0 2 )\n( 0 1 4 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Addition assignment to the element (2,1)
      mat(0,2,1) += mat(0,0,3);
      mat(1,2,1) += mat(1,0,3);

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat,  8UL );
      checkNonZeros( mat,  0UL, 0UL, 1UL );
      checkNonZeros( mat,  1UL, 0UL, 1UL );
      checkNonZeros( mat,  2UL, 0UL, 2UL );
      checkNonZeros( mat,  0UL, 1UL, 1UL );
      checkNonZeros( mat,  1UL, 1UL, 1UL );
      checkNonZeros( mat,  2UL, 1UL, 2UL );

      if( mat(0,0,3) != 3 || mat(0,1,4) != 2 || mat(0,2,1) != 4 || mat(0,2,2) != 4 ||
          mat(1,0,3) != 3 || mat(1,1,4) != 2 || mat(1,2,1) != 4 || mat(1,2,2) != 4) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 0 0 0 3 0 )\n( 0 0 0 0 2 )\n( 0 4 4 0 0 ))\n(( 0 0 0 3 0 )\n( 0 0 0 0 2 )\n( 0 4 4 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Subtraction assignment to the element (1,0)
      mat(0,1,0) -= mat(0,1,4);
      mat(1,1,0) -= mat(1,1,4);

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat, 10UL );
      checkNonZeros( mat,  0UL, 0UL, 1UL );
      checkNonZeros( mat,  1UL, 0UL, 2UL );
      checkNonZeros( mat,  2UL, 0UL, 2UL );
      checkNonZeros( mat,  0UL, 1UL, 1UL );
      checkNonZeros( mat,  1UL, 1UL, 2UL );
      checkNonZeros( mat,  2UL, 1UL, 2UL );

      if( mat(0,0,3) != 3 || mat(0,1,0) != -2 || mat(0,1,4) != 2 || mat(0,2,1) != 4 || mat(0,2,2) != 4 ||
          mat(1,0,3) != 3 || mat(1,1,0) != -2 || mat(1,1,4) != 2 || mat(1,2,1) != 4 || mat(1,2,2) != 4) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 3 0 )\n( -2 0 0 0 2 )\n(  0 4 4 0 0 ))\n((  0 0 0 3 0 )\n( -2 0 0 0 2 )\n(  0 4 4 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Multiplication assignment to the element (0,3)
      mat(0,0,3) *= -3;
      mat(1,0,3) *= -3;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat, 10UL );
      checkNonZeros( mat,  0UL, 0UL, 1UL );
      checkNonZeros( mat,  1UL, 0UL, 2UL );
      checkNonZeros( mat,  2UL, 0UL, 2UL );
      checkNonZeros( mat,  0UL, 1UL, 1UL );
      checkNonZeros( mat,  1UL, 1UL, 2UL );
      checkNonZeros( mat,  2UL, 1UL, 2UL );

      if( mat(0,0,3) != -9 || mat(0,1,0) != -2 || mat(0,1,4) != 2 || mat(0,2,1) != 4 || mat(0,2,2) != 4 ||
          mat(1,0,3) != -9 || mat(1,1,0) != -2 || mat(1,1,4) != 2 || mat(1,2,1) != 4 || mat(1,2,2) != 4) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 4 4  0 0 ))\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 4 4  0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Division assignment to the element (2,1)
      mat(0,2,1) /= 2;
      mat(1,2,1) /= 2;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat, 10UL );
      checkNonZeros( mat,  0UL, 0UL, 1UL );
      checkNonZeros( mat,  1UL, 0UL, 2UL );
      checkNonZeros( mat,  2UL, 0UL, 2UL );
      checkNonZeros( mat,  0UL, 1UL, 1UL );
      checkNonZeros( mat,  1UL, 1UL, 2UL );
      checkNonZeros( mat,  2UL, 1UL, 2UL );

      if( mat(0,0,3) != -9 || mat(0,1,0) != -2 || mat(0,1,4) != 2 || mat(0,2,1) != 2 || mat(0,2,2) != 4 ||
          mat(1,0,3) != -9 || mat(1,1,0) != -2 || mat(1,1,4) != 2 || mat(1,2,1) != 2 || mat(1,2,2) != 4) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 2 4  0 0 ))\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 2 4  0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c at() member function of the DynamicTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the \c at() member function
// of the DynamicTensor class template. In case an error is detected, a \a std::runtime_error
// exception is thrown.
*/
void ClassTest::testAt()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================
   {
      test_ = "Row-major DynamicTensor::at()";

      // Assignment to the element (2,1)
      blaze::DynamicTensor<int> mat( 2UL, 3UL, 5UL, 0 );
      mat.at(0,2,1) = 1;
      mat.at(1,2,1) = 1;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat,  2UL );
      checkNonZeros( mat,  0UL, 0UL, 0UL );
      checkNonZeros( mat,  1UL, 0UL, 0UL );
      checkNonZeros( mat,  2UL, 0UL, 1UL );
      checkNonZeros( mat,  0UL, 1UL, 0UL );
      checkNonZeros( mat,  1UL, 1UL, 0UL );
      checkNonZeros( mat,  2UL, 1UL, 1UL );

      if( mat.at(0,2,1) != 1 || mat.at(1,2,1) != 1) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Access via at() function failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 0 0 0 0 0 )\n( 0 0 0 0 0 )\n( 0 1 0 0 0 ))\n(( 0 0 0 0 0 )\n( 0 0 0 0 0 )\n( 0 1 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Assignment to the element (1,4)
      mat.at(0,1,4) = 2;
      mat.at(1,1,4) = 2;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat,  4UL );
      checkNonZeros( mat,  0UL, 0UL, 0UL );
      checkNonZeros( mat,  1UL, 0UL, 1UL );
      checkNonZeros( mat,  2UL, 0UL, 1UL );
      checkNonZeros( mat,  0UL, 1UL, 0UL );
      checkNonZeros( mat,  1UL, 1UL, 1UL );
      checkNonZeros( mat,  2UL, 1UL, 1UL );

      if( mat.at(0,1,4) != 2 || mat.at(0,2,1) != 1 || mat.at(1,1,4) != 2 || mat.at(1,2,1) != 1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Access via at() function failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 0 0 0 0 0 )\n( 0 0 0 0 2 )\n( 0 1 0 0 0 ))\n(( 0 0 0 0 0 )\n( 0 0 0 0 2 )\n( 0 1 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      mat.at(0,0,3) = 3;
      mat.at(1,0,3) = 3;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat,  6UL );
      checkNonZeros( mat,  0UL, 0UL, 1UL );
      checkNonZeros( mat,  1UL, 0UL, 1UL );
      checkNonZeros( mat,  2UL, 0UL, 1UL );
      checkNonZeros( mat,  0UL, 1UL, 1UL );
      checkNonZeros( mat,  1UL, 1UL, 1UL );
      checkNonZeros( mat,  2UL, 1UL, 1UL );

      if( mat.at(0,0,3) != 3 || mat.at(0,1,4) != 2 || mat.at(0,2,1) != 1 ||
          mat.at(1,0,3) != 3 || mat.at(1,1,4) != 2 || mat.at(1,2,1) != 1) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Access via at() function failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 0 0 0 3 0 )\n( 0 0 0 0 2 )\n( 0 1 0 0 0 ))\n(( 0 0 0 3 0 )\n( 0 0 0 0 2 )\n( 0 1 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Assignment to the element (2,2)
      mat.at(0,2,2) = 4;
      mat.at(1,2,2) = 4;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat,  8UL );
      checkNonZeros( mat,  0UL, 0UL, 1UL );
      checkNonZeros( mat,  1UL, 0UL, 1UL );
      checkNonZeros( mat,  2UL, 0UL, 2UL );
      checkNonZeros( mat,  0UL, 1UL, 1UL );
      checkNonZeros( mat,  1UL, 1UL, 1UL );
      checkNonZeros( mat,  2UL, 1UL, 2UL );

      if( mat.at(0,0,3) != 3 || mat.at(0,1,4) != 2 || mat.at(0,2,1) != 1 || mat.at(0,2,2) != 4 ||
          mat.at(1,0,3) != 3 || mat.at(1,1,4) != 2 || mat.at(1,2,1) != 1 || mat.at(1,2,2) != 4) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Access via at() function failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 0 0 0 3 0 )\n( 0 0 0 0 2 )\n( 0 1 4 0 0 ))\n(( 0 0 0 3 0 )\n( 0 0 0 0 2 )\n( 0 1 4 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Addition assignment to the element (2,1)
      mat.at(0,2,1) += mat.at(0,0,3);
      mat.at(1,2,1) += mat.at(1,0,3);

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat,  8UL );
      checkNonZeros( mat,  0UL, 0UL, 1UL );
      checkNonZeros( mat,  1UL, 0UL, 1UL );
      checkNonZeros( mat,  2UL, 0UL, 2UL );
      checkNonZeros( mat,  0UL, 1UL, 1UL );
      checkNonZeros( mat,  1UL, 1UL, 1UL );
      checkNonZeros( mat,  2UL, 1UL, 2UL );

      if( mat.at(0,0,3) != 3 || mat.at(0,1,4) != 2 || mat.at(0,2,1) != 4 || mat.at(0,2,2) != 4 ||
          mat.at(1,0,3) != 3 || mat.at(1,1,4) != 2 || mat.at(1,2,1) != 4 || mat.at(1,2,2) != 4) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Access via at() function failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 0 0 0 3 0 )\n( 0 0 0 0 2 )\n( 0 4 4 0 0 ))\n(( 0 0 0 3 0 )\n( 0 0 0 0 2 )\n( 0 4 4 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Subtraction assignment to the element (1,0)
      mat.at(0,1,0) -= mat.at(0,1,4);
      mat.at(1,1,0) -= mat.at(1,1,4);

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat, 10UL );
      checkNonZeros( mat,  0UL, 0UL, 1UL );
      checkNonZeros( mat,  1UL, 0UL, 2UL );
      checkNonZeros( mat,  2UL, 0UL, 2UL );
      checkNonZeros( mat,  0UL, 1UL, 1UL );
      checkNonZeros( mat,  1UL, 1UL, 2UL );
      checkNonZeros( mat,  2UL, 1UL, 2UL );

      if( mat.at(0,0,3) != 3 || mat.at(0,1,0) != -2 || mat.at(0,1,4) != 2 || mat.at(0,2,1) != 4 || mat.at(0,2,2) != 4 ||
          mat.at(1,0,3) != 3 || mat.at(1,1,0) != -2 || mat.at(1,1,4) != 2 || mat.at(1,2,1) != 4 || mat.at(1,2,2) != 4) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Access via at() function failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 3 0 )\n( -2 0 0 0 2 )\n(  0 4 4 0 0 ))\n((  0 0 0 3 0 )\n( -2 0 0 0 2 )\n(  0 4 4 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Multiplication assignment to the element (0,3)
      mat.at(0,0,3) *= -3;
      mat.at(1,0,3) *= -3;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat, 10UL );
      checkNonZeros( mat,  0UL, 0UL, 1UL );
      checkNonZeros( mat,  1UL, 0UL, 2UL );
      checkNonZeros( mat,  2UL, 0UL, 2UL );
      checkNonZeros( mat,  0UL, 1UL, 1UL );
      checkNonZeros( mat,  1UL, 1UL, 2UL );
      checkNonZeros( mat,  2UL, 1UL, 2UL );

      if( mat.at(0,0,3) != -9 || mat.at(0,1,0) != -2 || mat.at(0,1,4) != 2 || mat.at(0,2,1) != 4 || mat.at(0,2,2) != 4 ||
          mat.at(1,0,3) != -9 || mat.at(1,1,0) != -2 || mat.at(1,1,4) != 2 || mat.at(1,2,1) != 4 || mat.at(1,2,2) != 4) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Access via at() function failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 4 4  0 0 ))\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 4 4  0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Division assignment to the element (2,1)
      mat.at(0,2,1) /= 2;
      mat.at(1,2,1) /= 2;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  5UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 30UL );
      checkNonZeros( mat, 10UL );
      checkNonZeros( mat,  0UL, 0UL, 1UL );
      checkNonZeros( mat,  1UL, 0UL, 2UL );
      checkNonZeros( mat,  2UL, 0UL, 2UL );
      checkNonZeros( mat,  0UL, 1UL, 1UL );
      checkNonZeros( mat,  1UL, 1UL, 2UL );
      checkNonZeros( mat,  2UL, 1UL, 2UL );

      if( mat.at(0,0,3) != -9 || mat.at(0,1,0) != -2 || mat.at(0,1,4) != 2 || mat.at(0,2,1) != 2 || mat.at(0,2,2) != 4 ||
          mat.at(1,0,3) != -9 || mat.at(1,1,0) != -2 || mat.at(1,1,4) != 2 || mat.at(1,2,1) != 2 || mat.at(1,2,2) != 4) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Access via at() function failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 2 4  0 0 ))\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 2 4  0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Attempt to assign to the element (0,3,0)
      try {
         mat.at(0,3,0) = 2;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Out-of-bound access succeeded\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 2 4  0 0 ))\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 2 4  0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::out_of_range& ) {}

      // Attempt to assign to the element (0,0,5)
      try {
         mat.at(0,0,5) = 2;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Out-of-bound access succeeded\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 2 4  0 0 ))\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 2 4  0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::out_of_range& ) {}

      // Attempt to assign to the element (3,0,1)
      try {
         mat.at(3,0,1) = 2;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Out-of-bound access succeeded\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 2 4  0 0 ))\n((  0 0 0 -3 0 )\n( -2 0 0  0 2 )\n(  0 2 4  0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::out_of_range& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DynamicTensor iterator implementation.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the iterator implementation of the DynamicTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testIterator()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================
   {
      using TensorType    = blaze::DynamicTensor<int>;
      using Iterator      = TensorType::Iterator;
      using ConstIterator = TensorType::ConstIterator;

      TensorType mat{{{0, 1, 0}, {-2, 0, -3}, {0, 4, 5}},
                     {{0, 1, 0}, {-2, 0, -3}, {0, 4, 5}}};

      // Testing the Iterator default constructor
      {
        test_ = "Row-major Iterator default constructor";

        Iterator it{};

        if (it != Iterator()) {
          std::ostringstream oss;
          oss << " Test: " << test_ << "\n"
              << " Error: Failed iterator default constructor\n";
          throw std::runtime_error(oss.str());
        }
      }

      // Testing the ConstIterator default constructor
      {
         test_ = "Row-major ConstIterator default constructor";

         ConstIterator it{};

         if( it != ConstIterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing conversion from Iterator to ConstIterator
      {
         test_ = "Row-major Iterator/ConstIterator conversion";

         ConstIterator it( begin( mat, 1UL, 0UL ) );

         if( it == end( mat, 1UL, 0UL ) || *it != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 0th row via Iterator (end-begin)
      {
         test_ = "Row-major Iterator subtraction (end-begin)";

         const ptrdiff_t number( end( mat, 0UL, 1UL ) - begin( mat, 0UL, 1UL ) );

         if( number != 3L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 3\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 0th row via Iterator (begin-end)
      {
         test_ = "Row-major Iterator subtraction (begin-end)";

         const ptrdiff_t number( begin( mat, 0UL, 0UL ) - end( mat, 0UL, 0UL ) );

         if( number != -3L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: -3\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st row via ConstIterator (end-begin)
      {
         test_ = "Row-major ConstIterator subtraction (end-begin)";

         const ptrdiff_t number( cend( mat, 1UL, 0UL ) - cbegin( mat, 1UL, 0UL ) );

         if( number != 3L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 3\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st row via ConstIterator (begin-end)
      {
         test_ = "Row-major ConstIterator subtraction (begin-end)";

         const ptrdiff_t number( cbegin( mat, 1UL, 1UL ) - cend( mat, 1UL, 1UL ) );

         if( number != -3L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: -3\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing read-only access via ConstIterator
      {
         test_ = "Row-major read-only access via ConstIterator";

         ConstIterator it ( cbegin( mat, 2UL, 0UL ) );
         ConstIterator end( cend( mat, 2UL, 0UL ) );

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || *it != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         --it;

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it++;

         if( it == end || *it != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it--;

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it += 2UL;

         if( it == end || *it != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator addition assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it -= 2UL;

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator subtraction assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it + 2UL;

         if( it == end || *it != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar addition failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it - 2UL;

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar subtraction failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = 3UL + it;

         if( it != end ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Scalar/iterator addition failed\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing assignment via Iterator
      {
         test_ = "Row-major assignment via Iterator";

         int value = 7;

         for( Iterator it=begin( mat, 2UL, 0UL ); it!=end( mat, 2UL, 0UL ); ++it ) {
            *it = value++;
         }

         if( mat(0,0,0) !=  0 || mat(0,0,1) != 1 || mat(0,0,2) !=  0 ||
             mat(0,1,0) != -2 || mat(0,1,1) != 0 || mat(0,1,2) != -3 ||
             mat(0,2,0) !=  7 || mat(0,2,1) != 8 || mat(0,2,2) !=  9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n((  0  1  0 )\n( -2  0 -3 )\n(  7  8  9 ))\n((  0  1  0 )\n( -2  0 -3 )\n(  7  8  9 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "Row-major addition assignment via Iterator";

         int value = 4;

         for( Iterator it=begin( mat, 1UL, 1UL ); it!=end( mat, 1UL, 1UL ); ++it ) {
            *it += value++;
         }

         if( mat(1,0,0) != 0 || mat(1,0,1) != 1 || mat(1,0,2) != 0 ||
             mat(1,1,0) != 2 || mat(1,1,1) != 5 || mat(1,1,2) != 3 ||
             mat(1,2,0) != 0 || mat(1,2,1) != 4 || mat(1,2,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n(( 0 1 0 )\n( 2 5 3 )\n( 7 8 9 ))\n(( 0 1 0 )\n( 2 5 3 )\n( 0 4 5 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "Row-major subtraction assignment via Iterator";

         int value = 4;

         for( Iterator it=begin( mat, 1UL, 0UL ); it!=end( mat, 1UL, 0UL ); ++it ) {
            *it -= value++;
         }

         if( mat(0,0,0) !=  0 || mat(0,0,1) !=  1 || mat(0,0,2) !=  0 ||
             mat(0,1,0) != -6 || mat(0,1,1) != -5 || mat(0,1,2) != -9 ||
             mat(0,2,0) !=  7 || mat(0,2,1) !=  8 || mat(0,2,2) !=  9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n((  0  1  0 )\n( -2  0 -3 )\n(  7  8  9 ))\n((  0  1  0 )\n( -2  0 -3 )\n(  7  8  9 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "Row-major multiplication assignment via Iterator";

         int value = 2;

         for( Iterator it=begin( mat, 1UL, 1UL ); it!=end( mat, 1UL, 1UL ); ++it ) {
            *it *= value++;
         }

         if( mat(1,0,0) !=  0 || mat(1,0,1) !=  1 || mat(1,0,2) !=  0 ||
             mat(1,1,0) !=  4 || mat(1,1,1) != 15 || mat(1,1,2) != 12 ||
             mat(1,2,0) !=  0 || mat(1,2,1) !=  4 || mat(1,2,2) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n((  0  1   0 )\n( -4  0 -12 )\n(  7  8   9 ))\n((  0  1   0 )\n( -4  0 -12 )\n(  7  8   9 ))";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "Row-major division assignment via Iterator";

         for( Iterator it=begin( mat, 1UL, 1UL ); it!=end( mat, 1UL, 1UL ); ++it ) {
            *it /= 2;
         }

         if( mat(1,0,0) !=  0 || mat(1,0,1) != 1 || mat(1,0,2) !=  0 ||
             mat(1,1,0) !=  2 || mat(1,1,1) != 7 || mat(1,1,2) !=  6 ||
             mat(1,2,0) !=  0 || mat(1,2,1) != 4 || mat(1,2,2) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n((  0  1  0 )\n( -2  0 -6 )\n(  7  8  9 ))\n((  0  1  0 )\n( -2  0 -6 )\n(  7  8  9 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c nonZeros() member function of the DynamicTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c nonZeros() member function of the DynamicTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testNonZeros()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================
   {
      test_ = "Row-major DynamicTensor::nonZeros()";

      {
         blaze::DynamicTensor<int> mat( 2UL, 2UL, 3UL, 0 );

         checkRows    ( mat,  2UL );
         checkColumns ( mat,  3UL );
         checkPages   ( mat,  2UL );
         checkCapacity( mat, 12UL );
         checkNonZeros( mat,  0UL );
         checkNonZeros( mat,  0UL, 0UL, 0UL );
         checkNonZeros( mat,  0UL, 0UL, 0UL );
         checkNonZeros( mat,  1UL, 1UL, 0UL );
         checkNonZeros( mat,  1UL, 1UL, 0UL );

         if( mat(0,0,0) != 0 || mat(0,0,1) != 0 || mat(0,0,2) != 0 ||
             mat(0,1,0) != 0 || mat(0,1,1) != 0 || mat(0,1,2) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Initialization failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n(( 0 0 0 )\n( 0 0 0 ))\n(( 0 0 0 )\n( 0 0 0 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicTensor<int> mat{{{0, 1, 2}, {0, 3, 0}},
                                       {{0, 1, 2}, {0, 3, 0}}};

         checkRows(mat, 2UL);
         checkColumns(mat, 3UL);
         checkPages(mat, 2UL);
         checkCapacity(mat, 12UL);
         checkNonZeros(mat, 6UL);
         checkNonZeros(mat, 0UL, 0UL, 2UL);
         checkNonZeros(mat, 1UL, 0UL, 1UL);
         checkNonZeros(mat, 0UL, 1UL, 2UL);
         checkNonZeros(mat, 1UL, 1UL, 1UL);

         if (mat(0,0,0) != 0 || mat(0,0,1) != 1 || mat(0,0,2) != 2 ||
             mat(0,1,0) != 0 || mat(0,1,1) != 3 || mat(0,1,2) != 0 ||
             mat(1,0,0) != 0 || mat(1,0,1) != 1 || mat(1,0,2) != 2 ||
             mat(1,1,0) != 0 || mat(1,1,1) != 3 || mat(1,1,2) != 0) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Initialization failed\n"
                << " Details:\n"
                << "   Result:\n"
                << mat << "\n"
                << "   Expected result:\n(( 0 1 2 )\n( 0 3 0 ))\n(( 0 1 2 )\n( 0 3 0 ))\n";
            throw std::runtime_error(oss.str());
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reset() member function of the DynamicTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reset() member function of the DynamicTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testReset()
{
   using blaze::reset;


   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================
   {
      test_ = "Row-major DynamicTensor::reset()";

      // Resetting a default initialized tensor
      {
         blaze::DynamicTensor<int> mat;

         reset( mat );

         checkRows    ( mat, 0UL );
         checkColumns ( mat, 0UL );
         checkPages   ( mat, 0UL );
         checkNonZeros( mat, 0UL );
      }

      // Resetting an initialized tensor
      {
         // Initialization check
         blaze::DynamicTensor<int> mat{{{1, 2, 3}, {4, 5, 6}},
                                       {{1, 2, 3}, {4, 5, 6}}};

         checkRows    ( mat, 2UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 12UL );
         checkPages   ( mat, 2UL );
         checkNonZeros( mat, 12UL );
         checkNonZeros( mat, 0UL, 0UL, 3UL );
         checkNonZeros( mat, 1UL, 0UL, 3UL );
         checkNonZeros( mat, 0UL, 1UL, 3UL );
         checkNonZeros( mat, 1UL, 1UL, 3UL );

         if( mat(0,0,0) != 1 || mat(0,0,1) != 2 || mat(0,0,2) != 3 ||
             mat(0,1,0) != 4 || mat(0,1,1) != 5 || mat(0,1,2) != 6 ||
             mat(1,0,0) != 1 || mat(1,0,1) != 2 || mat(1,0,2) != 3 ||
             mat(1,1,0) != 4 || mat(1,1,1) != 5 || mat(1,1,2) != 6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Initialization failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
            throw std::runtime_error( oss.str() );
         }

         // Resetting a single element
         reset( mat(0,0,2) );

         checkRows    ( mat, 2UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 12UL );
         checkPages   ( mat, 2UL );
         checkNonZeros( mat, 11UL );
         checkNonZeros( mat, 0UL, 0UL, 2UL );
         checkNonZeros( mat, 1UL, 0UL, 3UL );
         checkNonZeros( mat, 0UL, 1UL, 3UL );
         checkNonZeros( mat, 1UL, 1UL, 3UL );

         if( mat(0,0,0) != 1 || mat(0,0,1) != 2 || mat(0,0,2) != 0 ||
             mat(0,1,0) != 4 || mat(0,1,1) != 5 || mat(0,1,2) != 6 ||
             mat(1,0,0) != 1 || mat(1,0,1) != 2 || mat(1,0,2) != 3 ||
             mat(1,1,0) != 4 || mat(1,1,1) != 5 || mat(1,1,2) != 6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Initialization failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n(( 1 2 3 )\n( 4 5 0 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
            throw std::runtime_error( oss.str() );
         }

         // Resetting row 1
         reset( mat, 1UL, 1UL );

         checkRows    ( mat, 2UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 12UL );
         checkPages   ( mat, 2UL );
         checkNonZeros( mat, 8UL );
         checkNonZeros( mat, 0UL, 0UL, 2UL );
         checkNonZeros( mat, 1UL, 0UL, 3UL );
         checkNonZeros( mat, 0UL, 1UL, 3UL );
         checkNonZeros( mat, 1UL, 1UL, 0UL );

         if( mat(0,0,0) != 1 || mat(0,0,1) != 2 || mat(0,0,2) != 0 ||
             mat(0,1,0) != 4 || mat(0,1,1) != 5 || mat(0,1,2) != 6 ||
             mat(1,0,0) != 1 || mat(1,0,1) != 2 || mat(1,0,2) != 3 ||
             mat(1,1,0) != 0 || mat(1,1,1) != 0 || mat(1,1,2) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Initialization failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n(( 1 2 3 )\n( 4 5 0 ))\n(( 1 2 3 )\n( 0 0 0 ))\n";
            throw std::runtime_error( oss.str() );
         }

         // Resetting the entire tensor
         reset( mat );

         checkRows    ( mat, 2UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 12UL );
         checkPages   ( mat, 2UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL, 0UL );
         checkNonZeros( mat, 0UL, 1UL, 0UL );
         checkNonZeros( mat, 1UL, 1UL, 0UL );

         if( mat(0,0,0) != 0 || mat(0,0,1) != 0 || mat(0,0,2) != 0 ||
             mat(0,1,0) != 0 || mat(0,1,1) != 0 || mat(0,1,2) != 0 ||
             mat(1,0,0) != 0 || mat(1,0,1) != 0 || mat(1,0,2) != 0 ||
             mat(1,1,0) != 0 || mat(1,1,1) != 0 || mat(1,1,2) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Initialization failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n(( 0 0 0 )\n( 0 0 0 ))\n(( 0 0 0 )\n( 0 0 0 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c clear() member function of the DynamicTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c clear() member function of the DynamicTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testClear()
{
   using blaze::clear;


   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================
   {
      test_ = "Row-major DynamicTensor::clear()";

      // Clearing a default constructed tensor
      {
         blaze::DynamicTensor<int> mat;

         clear( mat );

         checkRows    ( mat, 0UL );
         checkColumns ( mat, 0UL );
         checkPages   ( mat, 0UL );
         checkNonZeros( mat, 0UL );
      }

      // Clearing an initialized tensor
      {
         // Initialization check
         blaze::DynamicTensor<int> mat{{{1, 2, 3}, {4, 5, 6}},
                                       {{1, 2, 3}, {4, 5, 6}}};

         checkRows    ( mat, 2UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 12UL );
         checkPages   ( mat, 2UL );
         checkNonZeros( mat, 12UL );
         checkNonZeros( mat, 0UL, 0UL, 3UL );
         checkNonZeros( mat, 1UL, 0UL, 3UL );
         checkNonZeros( mat, 0UL, 1UL, 3UL );
         checkNonZeros( mat, 1UL, 1UL, 3UL );

         if( mat(0,0,0) != 1 || mat(0,0,1) != 2 || mat(0,0,2) != 3 ||
             mat(0,1,0) != 4 || mat(0,1,1) != 5 || mat(0,1,2) != 6 ||
             mat(1,0,0) != 1 || mat(1,0,1) != 2 || mat(1,0,2) != 3 ||
             mat(1,1,0) != 4 || mat(1,1,1) != 5 || mat(1,1,2) != 6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Initialization failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
            throw std::runtime_error( oss.str() );
         }

         // Clearing a single element
         clear( mat(0, 0, 2) );

         checkRows    ( mat, 2UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 12UL );
         checkPages   ( mat, 2UL );
         checkNonZeros( mat, 11UL );
         checkNonZeros( mat, 0UL, 0UL, 2UL );
         checkNonZeros( mat, 1UL, 0UL, 3UL );
         checkNonZeros( mat, 0UL, 1UL, 3UL );
         checkNonZeros( mat, 1UL, 1UL, 3UL );

         if( mat(0,0,0) != 1 || mat(0,0,1) != 2 || mat(0,0,2) != 0 ||
             mat(0,1,0) != 4 || mat(0,1,1) != 5 || mat(0,1,2) != 6 ||
             mat(1,0,0) != 1 || mat(1,0,1) != 2 || mat(1,0,2) != 3 ||
             mat(1,1,0) != 4 || mat(1,1,1) != 5 || mat(1,1,2) != 6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Initialization failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n(( 1 2 3 )\n( 4 5 0 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
            throw std::runtime_error( oss.str() );
         }

         // Clearing the tensor
         clear( mat );

         checkRows    ( mat, 0UL );
         checkColumns ( mat, 0UL );
         checkPages   ( mat, 0UL );
         checkNonZeros( mat, 0UL );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c resize() member function of the DynamicTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c resize() member function of the DynamicTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testResize()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================
   {
      test_ = "Row-major DynamicTensor::resize()";

      // Initialization check
      blaze::DynamicTensor<int> mat;

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkNonZeros( mat, 0UL );

      // Resizing to 0x3x2
      mat.resize( 2UL, 0UL, 3UL );

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 3UL );
      checkPages   ( mat, 2UL );
      checkNonZeros( mat, 0UL );

      // Resizing to 5x0x2
      mat.resize( 2UL, 5UL, 0UL );

      checkRows    ( mat, 5UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 2UL );
      checkNonZeros( mat, 0UL );

      // Resizing to 5x2x0
      mat.resize( 0UL, 5UL, 2UL );

      checkRows    ( mat, 5UL );
      checkColumns ( mat, 2UL );
      checkPages   ( mat, 0UL );
      checkNonZeros( mat, 0UL );

      // Resizing to 2x1x2
      mat.resize( 2UL, 2UL, 1UL );
      mat = 0;

      checkRows    ( mat, 2UL );
      checkColumns ( mat, 1UL );
      checkPages   ( mat, 2UL );
      checkNonZeros( mat, 0UL );

      // Resizing to 3x2x3 and preserving the elements
      mat(0,0,0) = 1;
      mat(0,1,0) = 2;
      mat.resize( 3UL, 3UL, 2UL, true );

      checkRows    ( mat, 3UL );
      checkColumns ( mat, 2UL );
      checkPages   ( mat, 3UL );
      checkCapacity( mat, 18UL );

      if( mat(0,0,0) != 1 || mat(0,1,0) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Resizing the tensor failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n( 1 x )\n( 2 x )\n( x x )\n";
         throw std::runtime_error( oss.str() );
      }

      // Resizing to 2x2x2 and preserving the elements
      mat(0,0,1) = 3;
      mat(0,1,1) = 4;
      mat.resize( 2UL, 2UL, 2UL, true );

      checkRows    ( mat, 2UL );
      checkColumns ( mat, 2UL );
      checkPages   ( mat, 2UL );
      checkCapacity( mat, 8UL );

      if( mat(0,0,0) != 1 || mat(0,0,1) != 3 || mat(0,1,0) != 2 || mat(0,1,1) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Resizing the tensor failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 1 0 )\n( 2 0 ))\n(( 0 3 )\n( 0 4 ))";
         throw std::runtime_error( oss.str() );
      }

      // Resizing to 1x1x1
      mat.resize( 1UL, 1UL, 1UL );

      checkRows    ( mat, 1UL );
      checkColumns ( mat, 1UL );
      checkPages   ( mat, 1UL );
      checkCapacity( mat, 1UL );

      // Resizing to 0x0x0
      mat.resize( 0UL, 0UL, 0UL );

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c extend() member function of the DynamicTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c extend() member function of the DynamicTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testExtend()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================
   {
      test_ = "Row-major DynamicTensor::extend()";

      // Initialization check
      blaze::DynamicTensor<int> mat;

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkNonZeros( mat, 0UL );

      // Increasing the size of the tensor
      mat.extend( 2UL, 2UL, 2UL );

      checkRows    ( mat, 2UL );
      checkColumns ( mat, 2UL );
      checkPages   ( mat, 2UL );
      checkCapacity( mat, 8UL );

      // Further increasing the size of the tensor and preserving the elements
      mat(0,0,0) = 1;
      mat(0,0,1) = 2;
      mat(0,1,0) = 3;
      mat(0,1,1) = 4;
      mat.extend( 0UL, 1UL, 1UL, true );

      checkRows    ( mat, 3UL );
      checkColumns ( mat, 3UL );
      checkPages   ( mat, 2UL );
      checkCapacity( mat, 18UL );

      if( mat(0,0,0) != 1 || mat(0,0,1) != 2 ||
          mat(0,1,0) != 3 || mat(0,1,1) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Extending the tensor failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 1 2 x )\n( 3 4 x )\n( x x x ))\n(( x x x )\n( x x x )\n( x x x ))";
         throw std::runtime_error( oss.str() );
      }

      // Further increasing the size of the tensor
      mat.extend( 3UL, 4UL, 10UL, false );

      checkRows    ( mat,   7UL );
      checkColumns ( mat,  13UL );
      checkPages   ( mat,   5UL );
      checkCapacity( mat, 455UL );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reserve() member function of the DynamicTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reserve() member function of the DynamicTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testReserve()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================
   {
      test_ = "Row-major DynamicTensor::reserve()";

      // Initialization check
      blaze::DynamicTensor<int> mat;

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkNonZeros( mat, 0UL );

      // Increasing the capacity of the tensor
      mat.reserve( 10UL );

      checkRows    ( mat,  0UL );
      checkColumns ( mat,  0UL );
      checkPages   ( mat,  0UL );
      checkCapacity( mat, 10UL );
      checkNonZeros( mat,  0UL );

      // Further increasing the capacity of the tensor
      mat.reserve( 20UL );

      checkRows    ( mat,  0UL );
      checkColumns ( mat,  0UL );
      checkPages   ( mat,  0UL );
      checkCapacity( mat, 20UL );
      checkNonZeros( mat,  0UL );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c shrinkToFit() member function of the DynamicTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c shrinkToFit() member function of the DynamicTensor
// class template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testShrinkToFit()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================
   {
      test_ = "Row-major DynamicTensor::shrinkToFit()";

      // Shrinking a tensor without excessive capacity
      {
         blaze::DynamicTensor<int> mat{{{1, 2, 3}, {4, 5, 6}},
                                       {{1, 2, 3}, {4, 5, 6}}};

         mat.shrinkToFit();

         checkRows    ( mat, 2UL );
         checkColumns ( mat, 3UL );
         checkPages   ( mat, 2UL );
         checkCapacity( mat, 12UL );
         checkNonZeros( mat, 12UL );
         checkNonZeros( mat, 0UL, 0UL, 3UL );
         checkNonZeros( mat, 1UL, 0UL, 3UL );
         checkNonZeros( mat, 0UL, 1UL, 3UL );
         checkNonZeros( mat, 1UL, 1UL, 3UL );

         if( mat.capacity() != mat.rows() * mat.spacing() * mat.pages() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Shrinking the tensor failed\n"
                << " Details:\n"
                << "   Capacity         : " << mat.capacity() << "\n"
                << "   Expected capacity: " << ( mat.rows() * mat.spacing() * mat.pages() ) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat(0,0,0) != 1 || mat(0,0,1) != 2 || mat(0,0,2) != 3 ||
             mat(0,1,0) != 4 || mat(0,1,1) != 5 || mat(0,1,2) != 6 ||
             mat(1,0,0) != 1 || mat(1,0,1) != 2 || mat(1,0,2) != 3 ||
             mat(1,1,0) != 4 || mat(1,1,1) != 5 || mat(1,1,2) != 6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Shrinking the tensor failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Shrinking a tensor with excessive capacity
      {
         blaze::DynamicTensor<int> mat{{{1, 2, 3}, {4, 5, 6}},
                                       {{1, 2, 3}, {4, 5, 6}}};
         mat.reserve( 100UL );

         mat.shrinkToFit();

         checkRows    ( mat, 2UL );
         checkColumns ( mat, 3UL );
         checkPages   ( mat, 2UL );
         checkCapacity( mat, 12UL );
         checkNonZeros( mat, 12UL );
         checkNonZeros( mat, 0UL, 0UL, 3UL );
         checkNonZeros( mat, 1UL, 0UL, 3UL );
         checkNonZeros( mat, 0UL, 1UL, 3UL );
         checkNonZeros( mat, 1UL, 1UL, 3UL );

         if( mat.capacity() != mat.rows() * mat.spacing() * mat.pages() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Shrinking the tensor failed\n"
                << " Details:\n"
                << "   Capacity         : " << mat.capacity() << "\n"
                << "   Expected capacity: " << ( mat.rows() * mat.spacing() * mat.pages() ) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat(0,0,0) != 1 || mat(0,0,1) != 2 || mat(0,0,2) != 3 ||
             mat(0,1,0) != 4 || mat(0,1,1) != 5 || mat(0,1,2) != 6 ||
             mat(1,0,0) != 1 || mat(1,0,1) != 2 || mat(1,0,2) != 3 ||
             mat(1,1,0) != 4 || mat(1,1,1) != 5 || mat(1,1,2) != 6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Shrinking the tensor failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c swap() functionality of the DynamicTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c swap() function of the DynamicTensor class template.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testSwap()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================
   {
      test_ = "Row-major DynamicTensor swap";

      blaze::DynamicTensor<int> mat1{{{1, 2}, {0, 3}, {4, 0}},
                                     {{1, 2}, {0, 3}, {4, 0}}};

      blaze::DynamicTensor<int> mat2{{{6, 5, 4}, {3, 2, 1}},
                                     {{6, 5, 4}, {3, 2, 1}},
                                     {{6, 5, 4}, {3, 2, 1}}};

      swap( mat1, mat2 );

      checkRows    ( mat1, 2UL );
      checkColumns ( mat1, 3UL );
      checkPages   ( mat1, 3UL );
      checkCapacity( mat1, 18UL );
      checkNonZeros( mat1, 18UL );
      checkNonZeros( mat1, 0UL, 0UL, 3UL );
      checkNonZeros( mat1, 1UL, 0UL, 3UL );
      checkNonZeros( mat1, 0UL, 1UL, 3UL );
      checkNonZeros( mat1, 1UL, 1UL, 3UL );
      checkNonZeros( mat1, 0UL, 2UL, 3UL );
      checkNonZeros( mat1, 1UL, 2UL, 3UL );

      if( mat1(0,0,0) != 6 || mat1(0,0,1) != 5 || mat1(0,0,2) != 4 ||
          mat1(0,1,0) != 3 || mat1(0,1,1) != 2 || mat1(0,1,2) != 1 ||
          mat1(1,0,0) != 6 || mat1(1,0,1) != 5 || mat1(1,0,2) != 4 ||
          mat1(1,1,0) != 3 || mat1(1,1,1) != 2 || mat1(1,1,2) != 1 ||
          mat1(2,0,0) != 6 || mat1(2,0,1) != 5 || mat1(2,0,2) != 4 ||
          mat1(2,1,0) != 3 || mat1(2,1,1) != 2 || mat1(2,1,2) != 1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Swapping the first tensor failed\n"
             << " Details:\n"
             << "   Result:\n" << mat1 << "\n"
             << "   Expected result:\n(( 6 5 4 )\n( 3 2 1 ))\n(( 6 5 4 )\n( 3 2 1 ))\n(( 6 5 4 )\n( 3 2 1 ))\n";
         throw std::runtime_error( oss.str() );
      }

      checkRows    ( mat2, 3UL );
      checkColumns ( mat2, 2UL );
      checkPages   ( mat2, 2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 8UL );
      checkNonZeros( mat2, 0UL, 0UL, 2UL );
      checkNonZeros( mat2, 1UL, 0UL, 1UL );
      checkNonZeros( mat2, 2UL, 0UL, 1UL );
      checkNonZeros( mat2, 0UL, 1UL, 2UL );
      checkNonZeros( mat2, 1UL, 1UL, 1UL );
      checkNonZeros( mat2, 2UL, 1UL, 1UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 2 ||
          mat2(0,1,0) != 0 || mat2(0,1,1) != 3 ||
          mat2(0,2,0) != 4 || mat2(0,2,1) != 0 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 2 ||
          mat2(1,1,0) != 0 || mat2(1,1,1) != 3 ||
          mat2(1,2,0) != 4 || mat2(1,2,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Swapping the second tensor failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 2 )\n( 0 3 )\n( 4, 0 ))\n(( 1 2 )\n( 0 3 )\n( 4, 0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c transpose() member function of the DynamicTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c transpose() member function of the DynamicTensor
// class template. Additionally, it performs a test of self-transpose via the \c trans()
// function. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testTranspose()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major self-transpose via transpose()";

      // Self-transpose of a 2x3x5 tensor
      {
         blaze::DynamicTensor<int> mat{
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}},
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}}};

         transpose(mat, {0, 1, 2});

         checkPages   ( mat,  2UL );
         checkRows    ( mat,  3UL );
         checkColumns ( mat,  5UL );
         checkCapacity( mat, 30UL );
         checkNonZeros( mat, 16UL );
         checkNonZeros( mat,  0UL, 0UL, 3UL );
         checkNonZeros( mat,  1UL, 0UL, 2UL );
         checkNonZeros( mat,  2UL, 0UL, 3UL );
         checkNonZeros( mat,  0UL, 1UL, 3UL );
         checkNonZeros( mat,  1UL, 1UL, 2UL );
         checkNonZeros( mat,  2UL, 1UL, 3UL );

         if( mat(0,0,0) != 1 || mat(0,0,1) != 0 || mat(0,0,2) != 2 || mat(0,0,3) != 0 || mat(0,0,4) != 3 ||
             mat(0,1,0) != 0 || mat(0,1,1) != 4 || mat(0,1,2) != 0 || mat(0,1,3) != 5 || mat(0,1,4) != 0 ||
             mat(0,2,0) != 6 || mat(0,2,1) != 0 || mat(0,2,2) != 7 || mat(0,2,3) != 0 || mat(0,2,4) != 8 ||
             mat(1,0,0) != 1 || mat(1,0,1) != 0 || mat(1,0,2) != 2 || mat(1,0,3) != 0 || mat(1,0,4) != 3 ||
             mat(1,1,0) != 0 || mat(1,1,1) != 4 || mat(1,1,2) != 0 || mat(1,1,3) != 5 || mat(1,1,4) != 0 ||
             mat(1,2,0) != 6 || mat(1,2,1) != 0 || mat(1,2,2) != 7 || mat(1,2,3) != 0 || mat(1,2,4) != 8) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Transpose operation failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n"
                        "(( 1 0 2 0 3 )\n( 0 4 0 5 0 )\n( 6 0 7 0 8 )\n"
                        " ( 1 0 2 0 3 )\n( 0 4 0 5 0 )\n( 6 0 7 0 8 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicTensor<int> mat{
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}},
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}}};

         transpose(mat, {0, 2, 1});

         checkPages   ( mat,  2UL );
         checkRows    ( mat,  5UL );
         checkColumns ( mat,  3UL );
         checkCapacity( mat, 30UL );
         checkNonZeros( mat, 16UL );
         checkNonZeros( mat,  0UL, 0UL, 2UL );
         checkNonZeros( mat,  1UL, 0UL, 1UL );
         checkNonZeros( mat,  2UL, 0UL, 2UL );
         checkNonZeros( mat,  3UL, 0UL, 1UL );
         checkNonZeros( mat,  4UL, 0UL, 2UL );
         checkNonZeros( mat,  0UL, 1UL, 2UL );
         checkNonZeros( mat,  1UL, 1UL, 1UL );
         checkNonZeros( mat,  2UL, 1UL, 2UL );
         checkNonZeros( mat,  3UL, 1UL, 1UL );
         checkNonZeros( mat,  4UL, 1UL, 2UL );

         if( mat(0,0,0) != 1 || mat(0,1,0) != 0 || mat(0,2,0) != 2 || mat(0,3,0) != 0 || mat(0,4,0) != 3 ||
             mat(0,0,1) != 0 || mat(0,1,1) != 4 || mat(0,2,1) != 0 || mat(0,3,1) != 5 || mat(0,4,1) != 0 ||
             mat(0,0,2) != 6 || mat(0,1,2) != 0 || mat(0,2,2) != 7 || mat(0,3,2) != 0 || mat(0,4,2) != 8 ||
             mat(1,0,0) != 1 || mat(1,1,0) != 0 || mat(1,2,0) != 2 || mat(1,3,0) != 0 || mat(1,4,0) != 3 ||
             mat(1,0,1) != 0 || mat(1,1,1) != 4 || mat(1,2,1) != 0 || mat(1,3,1) != 5 || mat(1,4,1) != 0 ||
             mat(1,0,2) != 6 || mat(1,1,2) != 0 || mat(1,2,2) != 7 || mat(1,3,2) != 0 || mat(1,4,2) != 8) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Transpose operation failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n"
                        "(( 1 0 6 )\n( 0 4 0 )\n( 2 0 7 )\n( 0 5 0 )\n( 3 0 8 )\n"
                        " ( 1 0 6 )\n( 0 4 0 )\n( 2 0 7 )\n( 0 5 0 )\n( 3 0 8 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicTensor<int> mat{
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}},
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}}};

         transpose(mat, {1, 0, 2});

         checkPages   ( mat,  3UL );
         checkRows    ( mat,  2UL );
         checkColumns ( mat,  5UL );
         checkCapacity( mat, 30UL );
         checkNonZeros( mat, 16UL );
         checkNonZeros( mat,  0UL, 0UL, 3UL );
         checkNonZeros( mat,  1UL, 0UL, 3UL );
         checkNonZeros( mat,  0UL, 1UL, 2UL );
         checkNonZeros( mat,  1UL, 1UL, 2UL );
         checkNonZeros( mat,  0UL, 2UL, 3UL );
         checkNonZeros( mat,  1UL, 2UL, 3UL );

         if( mat(0,0,0) != 1 || mat(0,0,1) != 0 || mat(0,0,2) != 2 || mat(0,0,3) != 0 || mat(0,0,4) != 3 ||
             mat(1,0,0) != 0 || mat(1,0,1) != 4 || mat(1,0,2) != 0 || mat(1,0,3) != 5 || mat(1,0,4) != 0 ||
             mat(2,0,0) != 6 || mat(2,0,1) != 0 || mat(2,0,2) != 7 || mat(2,0,3) != 0 || mat(2,0,4) != 8 ||
             mat(0,1,0) != 1 || mat(0,1,1) != 0 || mat(0,1,2) != 2 || mat(0,1,3) != 0 || mat(0,1,4) != 3 ||
             mat(1,1,0) != 0 || mat(1,1,1) != 4 || mat(1,1,2) != 0 || mat(1,1,3) != 5 || mat(1,1,4) != 0 ||
             mat(2,1,0) != 6 || mat(2,1,1) != 0 || mat(2,1,2) != 7 || mat(2,1,3) != 0 || mat(2,1,4) != 8) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Transpose operation failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n"
                        "(( 1 0 2 0 3 )\n( 1 0 2 0 3 )\n"
                        " ( 0 4 0 5 0 )\n( 0 4 0 5 0 )\n"
                        " ( 6 0 7 0 8 )\n( 6 0 7 0 8 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicTensor<int> mat{
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}},
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}}};

         transpose(mat, {1, 2, 0});

         checkPages   ( mat,  3UL );
         checkRows    ( mat,  5UL );
         checkColumns ( mat,  2UL );
         checkCapacity( mat, 30UL );
         checkNonZeros( mat, 16UL );
         checkNonZeros( mat,  0UL, 0UL, 2UL );
         checkNonZeros( mat,  1UL, 0UL, 0UL );
         checkNonZeros( mat,  2UL, 0UL, 2UL );
         checkNonZeros( mat,  3UL, 0UL, 0UL );
         checkNonZeros( mat,  4UL, 0UL, 2UL );
         checkNonZeros( mat,  0UL, 1UL, 0UL );
         checkNonZeros( mat,  1UL, 1UL, 2UL );
         checkNonZeros( mat,  2UL, 1UL, 0UL );
         checkNonZeros( mat,  3UL, 1UL, 2UL );
         checkNonZeros( mat,  4UL, 1UL, 0UL );
         checkNonZeros( mat,  0UL, 2UL, 2UL );
         checkNonZeros( mat,  1UL, 2UL, 0UL );
         checkNonZeros( mat,  2UL, 2UL, 2UL );
         checkNonZeros( mat,  3UL, 2UL, 0UL );
         checkNonZeros( mat,  4UL, 2UL, 2UL );

         if( mat(0,0,0) != 1 || mat(0,1,0) != 0 || mat(0,2,0) != 2 || mat(0,3,0) != 0 || mat(0,4,0) != 3 ||
             mat(1,0,0) != 0 || mat(1,1,0) != 4 || mat(1,2,0) != 0 || mat(1,3,0) != 5 || mat(1,4,0) != 0 ||
             mat(2,0,0) != 6 || mat(2,1,0) != 0 || mat(2,2,0) != 7 || mat(2,3,0) != 0 || mat(2,4,0) != 8 ||
             mat(0,0,1) != 1 || mat(0,1,1) != 0 || mat(0,2,1) != 2 || mat(0,3,1) != 0 || mat(0,4,1) != 3 ||
             mat(1,0,1) != 0 || mat(1,1,1) != 4 || mat(1,2,1) != 0 || mat(1,3,1) != 5 || mat(1,4,1) != 0 ||
             mat(2,0,1) != 6 || mat(2,1,1) != 0 || mat(2,2,1) != 7 || mat(2,3,1) != 0 || mat(2,4,1) != 8) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Transpose operation failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n"
                        "(( 1 1 )\n( 0 0 )\n( 2 2 )\n( 0 0 )\n( 3 3 )\n"
                        " ( 0 0 )\n( 4 4 )\n( 0 0 )\n( 5 5 )\n( 0 0 )\n"
                        " ( 6 6 )\n( 0 0 )\n( 7 7 )\n( 0 0 )\n( 8 8 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicTensor<int> mat{
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}},
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}}};

         transpose(mat, {2, 0, 1});

         checkPages   ( mat,  5UL );
         checkRows    ( mat,  2UL );
         checkColumns ( mat,  3UL );
         checkCapacity( mat, 30UL );
         checkNonZeros( mat, 16UL );
         checkNonZeros( mat,  0UL, 0UL, 2UL );
         checkNonZeros( mat,  1UL, 0UL, 2UL );
         checkNonZeros( mat,  0UL, 1UL, 1UL );
         checkNonZeros( mat,  1UL, 1UL, 1UL );
         checkNonZeros( mat,  0UL, 2UL, 2UL );
         checkNonZeros( mat,  1UL, 2UL, 2UL );
         checkNonZeros( mat,  0UL, 3UL, 1UL );
         checkNonZeros( mat,  1UL, 3UL, 1UL );
         checkNonZeros( mat,  0UL, 4UL, 2UL );
         checkNonZeros( mat,  1UL, 4UL, 2UL );

         if( mat(0,0,0) != 1 || mat(1,0,0) != 0 || mat(2,0,0) != 2 || mat(3,0,0) != 0 || mat(4,0,0) != 3 ||
             mat(0,0,1) != 0 || mat(1,0,1) != 4 || mat(2,0,1) != 0 || mat(3,0,1) != 5 || mat(4,0,1) != 0 ||
             mat(0,0,2) != 6 || mat(1,0,2) != 0 || mat(2,0,2) != 7 || mat(3,0,2) != 0 || mat(4,0,2) != 8 ||
             mat(0,1,0) != 1 || mat(1,1,0) != 0 || mat(2,1,0) != 2 || mat(3,1,0) != 0 || mat(4,1,0) != 3 ||
             mat(0,1,1) != 0 || mat(1,1,1) != 4 || mat(2,1,1) != 0 || mat(3,1,1) != 5 || mat(4,1,1) != 0 ||
             mat(0,1,2) != 6 || mat(1,1,2) != 0 || mat(2,1,2) != 7 || mat(3,1,2) != 0 || mat(4,1,2) != 8) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Transpose operation failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n"
                        "(( 1 0 6 )\n( 1 0 6 )\n"
                        " ( 0 4 0 )\n( 0 4 0 )\n"
                        " ( 2 0 7 )\n( 2 0 7 )\n"
                        " ( 0 5 0 )\n( 0 5 0 )\n"
                        " ( 3 0 8 )\n( 3 0 8 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicTensor<int> mat{
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}},
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}}};

         transpose(mat, {2, 1, 0});

         checkPages   ( mat,  5UL );
         checkRows    ( mat,  3UL );
         checkColumns ( mat,  2UL );
         checkCapacity( mat, 30UL );
         checkNonZeros( mat, 16UL );
         checkNonZeros( mat,  0UL, 0UL, 2UL );
         checkNonZeros( mat,  1UL, 0UL, 0UL );
         checkNonZeros( mat,  2UL, 0UL, 2UL );
         checkNonZeros( mat,  0UL, 1UL, 0UL );
         checkNonZeros( mat,  1UL, 1UL, 2UL );
         checkNonZeros( mat,  2UL, 1UL, 0UL );
         checkNonZeros( mat,  0UL, 2UL, 2UL );
         checkNonZeros( mat,  1UL, 2UL, 0UL );
         checkNonZeros( mat,  2UL, 2UL, 2UL );
         checkNonZeros( mat,  0UL, 3UL, 0UL );
         checkNonZeros( mat,  1UL, 3UL, 2UL );
         checkNonZeros( mat,  2UL, 3UL, 0UL );
         checkNonZeros( mat,  0UL, 4UL, 2UL );
         checkNonZeros( mat,  1UL, 4UL, 0UL );
         checkNonZeros( mat,  2UL, 4UL, 2UL );

         if( mat(0,0,0) != 1 || mat(1,0,0) != 0 || mat(2,0,0) != 2 || mat(3,0,0) != 0 || mat(4,0,0) != 3 ||
             mat(0,1,0) != 0 || mat(1,1,0) != 4 || mat(2,1,0) != 0 || mat(3,1,0) != 5 || mat(4,1,0) != 0 ||
             mat(0,2,0) != 6 || mat(1,2,0) != 0 || mat(2,2,0) != 7 || mat(3,2,0) != 0 || mat(4,2,0) != 8 ||
             mat(0,0,1) != 1 || mat(1,0,1) != 0 || mat(2,0,1) != 2 || mat(3,0,1) != 0 || mat(4,0,1) != 3 ||
             mat(0,1,1) != 0 || mat(1,1,1) != 4 || mat(2,1,1) != 0 || mat(3,1,1) != 5 || mat(4,1,1) != 0 ||
             mat(0,2,1) != 6 || mat(1,2,1) != 0 || mat(2,2,1) != 7 || mat(3,2,1) != 0 || mat(4,2,1) != 8) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Transpose operation failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n"
                        "(( 1 1 )\n( 0 0 )\n( 6 6 )\n"
                        " ( 0 0 )\n( 4 4 )\n( 0 0 )\n"
                        " ( 2 2 )\n( 0 0 )\n( 7 7 )\n"
                        " ( 0 0 )\n( 5 5 )\n( 0 0 )\n"
                        " ( 3 3 )\n( 0 0 )\n( 8 8 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }

   {
      test_ = "Row-major self-transpose via trans()";

      // Self-transpose of a 2x3x5 tensor
      {
         blaze::DynamicTensor<int> mat{
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}},
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}}};

         mat = trans(mat, {0, 1, 2});

         checkPages   ( mat,  2UL );
         checkRows    ( mat,  3UL );
         checkColumns ( mat,  5UL );
         checkCapacity( mat, 30UL );
         checkNonZeros( mat, 16UL );
         checkNonZeros( mat,  0UL, 0UL, 3UL );
         checkNonZeros( mat,  1UL, 0UL, 2UL );
         checkNonZeros( mat,  2UL, 0UL, 3UL );
         checkNonZeros( mat,  0UL, 1UL, 3UL );
         checkNonZeros( mat,  1UL, 1UL, 2UL );
         checkNonZeros( mat,  2UL, 1UL, 3UL );

         if( mat(0,0,0) != 1 || mat(0,0,1) != 0 || mat(0,0,2) != 2 || mat(0,0,3) != 0 || mat(0,0,4) != 3 ||
             mat(0,1,0) != 0 || mat(0,1,1) != 4 || mat(0,1,2) != 0 || mat(0,1,3) != 5 || mat(0,1,4) != 0 ||
             mat(0,2,0) != 6 || mat(0,2,1) != 0 || mat(0,2,2) != 7 || mat(0,2,3) != 0 || mat(0,2,4) != 8 ||
             mat(1,0,0) != 1 || mat(1,0,1) != 0 || mat(1,0,2) != 2 || mat(1,0,3) != 0 || mat(1,0,4) != 3 ||
             mat(1,1,0) != 0 || mat(1,1,1) != 4 || mat(1,1,2) != 0 || mat(1,1,3) != 5 || mat(1,1,4) != 0 ||
             mat(1,2,0) != 6 || mat(1,2,1) != 0 || mat(1,2,2) != 7 || mat(1,2,3) != 0 || mat(1,2,4) != 8) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Transpose operation failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n"
                        "(( 1 0 2 0 3 )\n( 0 4 0 5 0 )\n( 6 0 7 0 8 )\n"
                        " ( 1 0 2 0 3 )\n( 0 4 0 5 0 )\n( 6 0 7 0 8 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicTensor<int> mat{
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}},
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}}};

         mat = trans(mat, {0, 2, 1});

         checkPages   ( mat,  2UL );
         checkRows    ( mat,  5UL );
         checkColumns ( mat,  3UL );
         checkCapacity( mat, 30UL );
         checkNonZeros( mat, 16UL );
         checkNonZeros( mat,  0UL, 0UL, 2UL );
         checkNonZeros( mat,  1UL, 0UL, 1UL );
         checkNonZeros( mat,  2UL, 0UL, 2UL );
         checkNonZeros( mat,  3UL, 0UL, 1UL );
         checkNonZeros( mat,  4UL, 0UL, 2UL );
         checkNonZeros( mat,  0UL, 1UL, 2UL );
         checkNonZeros( mat,  1UL, 1UL, 1UL );
         checkNonZeros( mat,  2UL, 1UL, 2UL );
         checkNonZeros( mat,  3UL, 1UL, 1UL );
         checkNonZeros( mat,  4UL, 1UL, 2UL );

         if( mat(0,0,0) != 1 || mat(0,1,0) != 0 || mat(0,2,0) != 2 || mat(0,3,0) != 0 || mat(0,4,0) != 3 ||
             mat(0,0,1) != 0 || mat(0,1,1) != 4 || mat(0,2,1) != 0 || mat(0,3,1) != 5 || mat(0,4,1) != 0 ||
             mat(0,0,2) != 6 || mat(0,1,2) != 0 || mat(0,2,2) != 7 || mat(0,3,2) != 0 || mat(0,4,2) != 8 ||
             mat(1,0,0) != 1 || mat(1,1,0) != 0 || mat(1,2,0) != 2 || mat(1,3,0) != 0 || mat(1,4,0) != 3 ||
             mat(1,0,1) != 0 || mat(1,1,1) != 4 || mat(1,2,1) != 0 || mat(1,3,1) != 5 || mat(1,4,1) != 0 ||
             mat(1,0,2) != 6 || mat(1,1,2) != 0 || mat(1,2,2) != 7 || mat(1,3,2) != 0 || mat(1,4,2) != 8) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Transpose operation failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n"
                        "(( 1 0 6 )\n( 0 4 0 )\n( 2 0 7 )\n( 0 5 0 )\n( 3 0 8 )\n"
                        " ( 1 0 6 )\n( 0 4 0 )\n( 2 0 7 )\n( 0 5 0 )\n( 3 0 8 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicTensor<int> mat{
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}},
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}}};

         mat = trans(mat, {1, 0, 2});

         checkPages   ( mat,  3UL );
         checkRows    ( mat,  2UL );
         checkColumns ( mat,  5UL );
         checkCapacity( mat, 30UL );
         checkNonZeros( mat, 16UL );
         checkNonZeros( mat,  0UL, 0UL, 3UL );
         checkNonZeros( mat,  1UL, 0UL, 3UL );
         checkNonZeros( mat,  0UL, 1UL, 2UL );
         checkNonZeros( mat,  1UL, 1UL, 2UL );
         checkNonZeros( mat,  0UL, 2UL, 3UL );
         checkNonZeros( mat,  1UL, 2UL, 3UL );

         if( mat(0,0,0) != 1 || mat(0,0,1) != 0 || mat(0,0,2) != 2 || mat(0,0,3) != 0 || mat(0,0,4) != 3 ||
             mat(1,0,0) != 0 || mat(1,0,1) != 4 || mat(1,0,2) != 0 || mat(1,0,3) != 5 || mat(1,0,4) != 0 ||
             mat(2,0,0) != 6 || mat(2,0,1) != 0 || mat(2,0,2) != 7 || mat(2,0,3) != 0 || mat(2,0,4) != 8 ||
             mat(0,1,0) != 1 || mat(0,1,1) != 0 || mat(0,1,2) != 2 || mat(0,1,3) != 0 || mat(0,1,4) != 3 ||
             mat(1,1,0) != 0 || mat(1,1,1) != 4 || mat(1,1,2) != 0 || mat(1,1,3) != 5 || mat(1,1,4) != 0 ||
             mat(2,1,0) != 6 || mat(2,1,1) != 0 || mat(2,1,2) != 7 || mat(2,1,3) != 0 || mat(2,1,4) != 8) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Transpose operation failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n"
                        "(( 1 0 2 0 3 )\n( 1 0 2 0 3 )\n"
                        " ( 0 4 0 5 0 )\n( 0 4 0 5 0 )\n"
                        " ( 6 0 7 0 8 )\n( 6 0 7 0 8 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicTensor<int> mat{
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}},
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}}};

         mat = trans(mat, {1, 2, 0});

         checkPages   ( mat,  3UL );
         checkRows    ( mat,  5UL );
         checkColumns ( mat,  2UL );
         checkCapacity( mat, 30UL );
         checkNonZeros( mat, 16UL );
         checkNonZeros( mat,  0UL, 0UL, 2UL );
         checkNonZeros( mat,  1UL, 0UL, 0UL );
         checkNonZeros( mat,  2UL, 0UL, 2UL );
         checkNonZeros( mat,  3UL, 0UL, 0UL );
         checkNonZeros( mat,  4UL, 0UL, 2UL );
         checkNonZeros( mat,  0UL, 1UL, 0UL );
         checkNonZeros( mat,  1UL, 1UL, 2UL );
         checkNonZeros( mat,  2UL, 1UL, 0UL );
         checkNonZeros( mat,  3UL, 1UL, 2UL );
         checkNonZeros( mat,  4UL, 1UL, 0UL );
         checkNonZeros( mat,  0UL, 2UL, 2UL );
         checkNonZeros( mat,  1UL, 2UL, 0UL );
         checkNonZeros( mat,  2UL, 2UL, 2UL );
         checkNonZeros( mat,  3UL, 2UL, 0UL );
         checkNonZeros( mat,  4UL, 2UL, 2UL );

         if( mat(0,0,0) != 1 || mat(0,1,0) != 0 || mat(0,2,0) != 2 || mat(0,3,0) != 0 || mat(0,4,0) != 3 ||
             mat(1,0,0) != 0 || mat(1,1,0) != 4 || mat(1,2,0) != 0 || mat(1,3,0) != 5 || mat(1,4,0) != 0 ||
             mat(2,0,0) != 6 || mat(2,1,0) != 0 || mat(2,2,0) != 7 || mat(2,3,0) != 0 || mat(2,4,0) != 8 ||
             mat(0,0,1) != 1 || mat(0,1,1) != 0 || mat(0,2,1) != 2 || mat(0,3,1) != 0 || mat(0,4,1) != 3 ||
             mat(1,0,1) != 0 || mat(1,1,1) != 4 || mat(1,2,1) != 0 || mat(1,3,1) != 5 || mat(1,4,1) != 0 ||
             mat(2,0,1) != 6 || mat(2,1,1) != 0 || mat(2,2,1) != 7 || mat(2,3,1) != 0 || mat(2,4,1) != 8) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Transpose operation failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n"
                        "(( 1 1 )\n( 0 0 )\n( 2 2 )\n( 0 0 )\n( 3 3 )\n"
                        " ( 0 0 )\n( 4 4 )\n( 0 0 )\n( 5 5 )\n( 0 0 )\n"
                        " ( 6 6 )\n( 0 0 )\n( 7 7 )\n( 0 0 )\n( 8 8 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicTensor<int> mat{
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}},
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}}};

         mat = trans(mat, {2, 0, 1});

         checkPages   ( mat,  5UL );
         checkRows    ( mat,  2UL );
         checkColumns ( mat,  3UL );
         checkCapacity( mat, 30UL );
         checkNonZeros( mat, 16UL );
         checkNonZeros( mat,  0UL, 0UL, 2UL );
         checkNonZeros( mat,  1UL, 0UL, 2UL );
         checkNonZeros( mat,  0UL, 1UL, 1UL );
         checkNonZeros( mat,  1UL, 1UL, 1UL );
         checkNonZeros( mat,  0UL, 2UL, 2UL );
         checkNonZeros( mat,  1UL, 2UL, 2UL );
         checkNonZeros( mat,  0UL, 3UL, 1UL );
         checkNonZeros( mat,  1UL, 3UL, 1UL );
         checkNonZeros( mat,  0UL, 4UL, 2UL );
         checkNonZeros( mat,  1UL, 4UL, 2UL );

         if( mat(0,0,0) != 1 || mat(1,0,0) != 0 || mat(2,0,0) != 2 || mat(3,0,0) != 0 || mat(4,0,0) != 3 ||
             mat(0,0,1) != 0 || mat(1,0,1) != 4 || mat(2,0,1) != 0 || mat(3,0,1) != 5 || mat(4,0,1) != 0 ||
             mat(0,0,2) != 6 || mat(1,0,2) != 0 || mat(2,0,2) != 7 || mat(3,0,2) != 0 || mat(4,0,2) != 8 ||
             mat(0,1,0) != 1 || mat(1,1,0) != 0 || mat(2,1,0) != 2 || mat(3,1,0) != 0 || mat(4,1,0) != 3 ||
             mat(0,1,1) != 0 || mat(1,1,1) != 4 || mat(2,1,1) != 0 || mat(3,1,1) != 5 || mat(4,1,1) != 0 ||
             mat(0,1,2) != 6 || mat(1,1,2) != 0 || mat(2,1,2) != 7 || mat(3,1,2) != 0 || mat(4,1,2) != 8) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Transpose operation failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n"
                        "(( 1 0 6 )\n( 1 0 6 )\n"
                        " ( 0 4 0 )\n( 0 4 0 )\n"
                        " ( 2 0 7 )\n( 2 0 7 )\n"
                        " ( 0 5 0 )\n( 0 5 0 )\n"
                        " ( 3 0 8 )\n( 3 0 8 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicTensor<int> mat{
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}},
            {{1, 0, 2, 0, 3}, {0, 4, 0, 5, 0}, {6, 0, 7, 0, 8}}};

         mat = trans(mat, {2, 1, 0});

         checkPages   ( mat,  5UL );
         checkRows    ( mat,  3UL );
         checkColumns ( mat,  2UL );
         checkCapacity( mat, 30UL );
         checkNonZeros( mat, 16UL );
         checkNonZeros( mat,  0UL, 0UL, 2UL );
         checkNonZeros( mat,  1UL, 0UL, 0UL );
         checkNonZeros( mat,  2UL, 0UL, 2UL );
         checkNonZeros( mat,  0UL, 1UL, 0UL );
         checkNonZeros( mat,  1UL, 1UL, 2UL );
         checkNonZeros( mat,  2UL, 1UL, 0UL );
         checkNonZeros( mat,  0UL, 2UL, 2UL );
         checkNonZeros( mat,  1UL, 2UL, 0UL );
         checkNonZeros( mat,  2UL, 2UL, 2UL );
         checkNonZeros( mat,  0UL, 3UL, 0UL );
         checkNonZeros( mat,  1UL, 3UL, 2UL );
         checkNonZeros( mat,  2UL, 3UL, 0UL );
         checkNonZeros( mat,  0UL, 4UL, 2UL );
         checkNonZeros( mat,  1UL, 4UL, 0UL );
         checkNonZeros( mat,  2UL, 4UL, 2UL );

         if( mat(0,0,0) != 1 || mat(1,0,0) != 0 || mat(2,0,0) != 2 || mat(3,0,0) != 0 || mat(4,0,0) != 3 ||
             mat(0,1,0) != 0 || mat(1,1,0) != 4 || mat(2,1,0) != 0 || mat(3,1,0) != 5 || mat(4,1,0) != 0 ||
             mat(0,2,0) != 6 || mat(1,2,0) != 0 || mat(2,2,0) != 7 || mat(3,2,0) != 0 || mat(4,2,0) != 8 ||
             mat(0,0,1) != 1 || mat(1,0,1) != 0 || mat(2,0,1) != 2 || mat(3,0,1) != 0 || mat(4,0,1) != 3 ||
             mat(0,1,1) != 0 || mat(1,1,1) != 4 || mat(2,1,1) != 0 || mat(3,1,1) != 5 || mat(4,1,1) != 0 ||
             mat(0,2,1) != 6 || mat(1,2,1) != 0 || mat(2,2,1) != 7 || mat(3,2,1) != 0 || mat(4,2,1) != 8) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Transpose operation failed\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n"
                << "   Expected result:\n"
                        "(( 1 1 )\n( 0 0 )\n( 6 6 )\n"
                        " ( 0 0 )\n( 4 4 )\n( 0 0 )\n"
                        " ( 2 2 )\n( 0 0 )\n( 7 7 )\n"
                        " ( 0 0 )\n( 5 5 )\n( 0 0 )\n"
                        " ( 3 3 )\n( 0 0 )\n( 8 8 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         test_ = "Row-major self-transpose (stress test)";

         const size_t n( blaze::rand<size_t>( 0UL, 20UL ) );

         blaze::DynamicTensor<int> mat1( n, n, n, 0 );
         randomize( mat1 );

         std::initializer_list< std::initializer_list< size_t > > indices{
             {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

         for ( auto idx : indices )
         {
            blaze::DynamicTensor<int> mat2( mat1 );
            transpose( mat2, idx );

            if( mat2 != trans( mat1, idx ) ) {
               std::ostringstream oss;
               oss << " Test: " << test_ << "\n"
                     << " Error: Transpose operation failed\n"
                     << " Details:\n"
                     << "   Result:\n" << mat2 << "\n"
                     << "   Expected result:\n" << trans( mat1, idx ) << "\n";
               throw std::runtime_error( oss.str() );
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c ctranspose() member function of the DynamicTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c ctranspose() member function of the DynamicTensor
// class template. Additionally, it performs a test of self-transpose via the \c ctrans()
// function. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testCTranspose()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major self-transpose via ctranspose()";
//
//       using cplx = blaze::complex<int>;
//
//       // Self-transpose of a 4x4 tensor
//       {
//          blaze::DynamicTensor<cplx> mat( 4UL, 4UL, cplx() );
//          mat(0,0) = cplx(1,-1);
//          mat(0,2) = cplx(2,-2);
//          mat(1,1) = cplx(3,-3);
//          mat(1,3) = cplx(4,-4);
//          mat(2,0) = cplx(5,-5);
//          mat(2,2) = cplx(6,-6);
//          mat(3,1) = cplx(7,-7);
//          mat(3,3) = cplx(8,-8);
//
//          ctranspose( mat );
//
//          checkRows    ( mat,  4UL );
//          checkColumns ( mat,  4UL );
//          checkCapacity( mat, 16UL );
//          checkNonZeros( mat,  8UL );
//          checkNonZeros( mat,  0UL, 2UL );
//          checkNonZeros( mat,  1UL, 2UL );
//          checkNonZeros( mat,  2UL, 2UL );
//          checkNonZeros( mat,  3UL, 2UL );
//
//          if( mat(0,0) != cplx(1,1) || mat(0,1) != cplx(0,0) || mat(0,2) != cplx(5,5) || mat(0,3) != cplx(0,0) ||
//              mat(1,0) != cplx(0,0) || mat(1,1) != cplx(3,3) || mat(1,2) != cplx(0,0) || mat(1,3) != cplx(7,7) ||
//              mat(2,0) != cplx(2,2) || mat(2,1) != cplx(0,0) || mat(2,2) != cplx(6,6) || mat(2,3) != cplx(0,0) ||
//              mat(3,0) != cplx(0,0) || mat(3,1) != cplx(4,4) || mat(3,2) != cplx(0,0) || mat(3,3) != cplx(8,8) ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Transpose operation failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat << "\n"
//                 << "   Expected result:\n( (1,1) (0,0) (5,5) (0,0) )\n"
//                                         "( (0,0) (3,3) (0,0) (7,7) )\n"
//                                         "( (2,2) (0,0) (6,6) (0,0) )\n"
//                                         "( (0,0) (4,4) (0,0) (8,8) )\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Self-transpose of a 3x5 tensor
//       {
//          blaze::DynamicTensor<cplx> mat( 3UL, 5UL, cplx() );
//          mat(0,0) = cplx(1,-1);
//          mat(0,2) = cplx(2,-2);
//          mat(0,4) = cplx(3,-3);
//          mat(1,1) = cplx(4,-4);
//          mat(1,3) = cplx(5,-5);
//          mat(2,0) = cplx(6,-6);
//          mat(2,2) = cplx(7,-7);
//          mat(2,4) = cplx(8,-8);
//
//          ctranspose( mat );
//
//          checkRows    ( mat,  5UL );
//          checkColumns ( mat,  3UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  8UL );
//          checkNonZeros( mat,  0UL, 2UL );
//          checkNonZeros( mat,  1UL, 1UL );
//          checkNonZeros( mat,  2UL, 2UL );
//          checkNonZeros( mat,  3UL, 1UL );
//          checkNonZeros( mat,  4UL, 2UL );
//
//          if( mat(0,0) != cplx(1,1) || mat(0,1) != cplx(0,0) || mat(0,2) != cplx(6,6) ||
//              mat(1,0) != cplx(0,0) || mat(1,1) != cplx(4,4) || mat(1,2) != cplx(0,0) ||
//              mat(2,0) != cplx(2,2) || mat(2,1) != cplx(0,0) || mat(2,2) != cplx(7,7) ||
//              mat(3,0) != cplx(0,0) || mat(3,1) != cplx(5,5) || mat(3,2) != cplx(0,0) ||
//              mat(4,0) != cplx(3,3) || mat(4,1) != cplx(0,0) || mat(4,2) != cplx(8,8) ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Transpose operation failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat << "\n"
//                 << "   Expected result:\n( (1,1) (0,0) (6,6) )\n"
//                                         "( (0,0) (4,4) (0,0) )\n"
//                                         "( (2,2) (0,0) (7,7) )\n"
//                                         "( (0,0) (5,5) (0,0) )\n"
//                                         "( (3,3) (0,0) (8,8) )\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Self-transpose of a 5x3 tensor
//       {
//          blaze::DynamicTensor<cplx> mat( 5UL, 3UL, cplx() );
//          mat(0,0) = cplx(1,-1);
//          mat(0,2) = cplx(6,-6);
//          mat(1,1) = cplx(4,-4);
//          mat(2,0) = cplx(2,-2);
//          mat(2,2) = cplx(7,-7);
//          mat(3,1) = cplx(5,-5);
//          mat(4,0) = cplx(3,-3);
//          mat(4,2) = cplx(8,-8);
//
//          ctranspose( mat );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  8UL );
//          checkNonZeros( mat,  0UL, 3UL );
//          checkNonZeros( mat,  1UL, 2UL );
//          checkNonZeros( mat,  2UL, 3UL );
//
//          if( mat(0,0) != cplx(1,1) || mat(0,1) != cplx(0,0) || mat(0,2) != cplx(2,2) || mat(0,3) != cplx(0,0) || mat(0,4) != cplx(3,3) ||
//              mat(1,0) != cplx(0,0) || mat(1,1) != cplx(4,4) || mat(1,2) != cplx(0,0) || mat(1,3) != cplx(5,5) || mat(1,4) != cplx(0,0) ||
//              mat(2,0) != cplx(6,6) || mat(2,1) != cplx(0,0) || mat(2,2) != cplx(7,7) || mat(2,3) != cplx(0,0) || mat(2,4) != cplx(8,8) ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Transpose operation failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat << "\n"
//                 << "   Expected result:\n( (1,1) (0,0) (2,2) (0,0) (3,3) )\n"
//                                         "( (0,0) (4,4) (0,0) (5,5) (0,0) )\n"
//                                         "( (6,6) (0,0) (7,7) (0,0) (8,8) )\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//    {
//       test_ = "Row-major self-transpose via ctranspose() (stress test)";
//
//       using cplx = blaze::complex<int>;
//
//       const size_t n( blaze::rand<size_t>( 0UL, 100UL ) );
//
//       blaze::DynamicTensor<cplx> mat1( n, n, 0 );
//       randomize( mat1 );
//       blaze::DynamicTensor<cplx> mat2( mat1 );
//
//       ctranspose( mat1 );
//
//       if( mat1 != ctrans( mat2 ) ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Transpose operation failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << ctrans( mat2 ) << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major self-transpose via ctrans()";
//
//       using cplx = blaze::complex<int>;
//
//       // Self-transpose of a 4x4 tensor
//       {
//          blaze::DynamicTensor<cplx> mat( 4UL, 4UL, cplx() );
//          mat(0,0) = cplx(1,-1);
//          mat(0,2) = cplx(2,-2);
//          mat(1,1) = cplx(3,-3);
//          mat(1,3) = cplx(4,-4);
//          mat(2,0) = cplx(5,-5);
//          mat(2,2) = cplx(6,-6);
//          mat(3,1) = cplx(7,-7);
//          mat(3,3) = cplx(8,-8);
//
//          mat = ctrans( mat );
//
//          checkRows    ( mat,  4UL );
//          checkColumns ( mat,  4UL );
//          checkCapacity( mat, 16UL );
//          checkNonZeros( mat,  8UL );
//          checkNonZeros( mat,  0UL, 2UL );
//          checkNonZeros( mat,  1UL, 2UL );
//          checkNonZeros( mat,  2UL, 2UL );
//          checkNonZeros( mat,  3UL, 2UL );
//
//          if( mat(0,0) != cplx(1,1) || mat(0,1) != cplx(0,0) || mat(0,2) != cplx(5,5) || mat(0,3) != cplx(0,0) ||
//              mat(1,0) != cplx(0,0) || mat(1,1) != cplx(3,3) || mat(1,2) != cplx(0,0) || mat(1,3) != cplx(7,7) ||
//              mat(2,0) != cplx(2,2) || mat(2,1) != cplx(0,0) || mat(2,2) != cplx(6,6) || mat(2,3) != cplx(0,0) ||
//              mat(3,0) != cplx(0,0) || mat(3,1) != cplx(4,4) || mat(3,2) != cplx(0,0) || mat(3,3) != cplx(8,8) ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Transpose operation failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat << "\n"
//                 << "   Expected result:\n( (1,1) (0,0) (5,5) (0,0) )\n"
//                                         "( (0,0) (3,3) (0,0) (7,7) )\n"
//                                         "( (2,2) (0,0) (6,6) (0,0) )\n"
//                                         "( (0,0) (4,4) (0,0) (8,8) )\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Self-transpose of a 3x5 tensor
//       {
//          blaze::DynamicTensor<cplx> mat( 3UL, 5UL, cplx() );
//          mat(0,0) = cplx(1,-1);
//          mat(0,2) = cplx(2,-2);
//          mat(0,4) = cplx(3,-3);
//          mat(1,1) = cplx(4,-4);
//          mat(1,3) = cplx(5,-5);
//          mat(2,0) = cplx(6,-6);
//          mat(2,2) = cplx(7,-7);
//          mat(2,4) = cplx(8,-8);
//
//          mat = ctrans( mat );
//
//          checkRows    ( mat,  5UL );
//          checkColumns ( mat,  3UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  8UL );
//          checkNonZeros( mat,  0UL, 2UL );
//          checkNonZeros( mat,  1UL, 1UL );
//          checkNonZeros( mat,  2UL, 2UL );
//          checkNonZeros( mat,  3UL, 1UL );
//          checkNonZeros( mat,  4UL, 2UL );
//
//          if( mat(0,0) != cplx(1,1) || mat(0,1) != cplx(0,0) || mat(0,2) != cplx(6,6) ||
//              mat(1,0) != cplx(0,0) || mat(1,1) != cplx(4,4) || mat(1,2) != cplx(0,0) ||
//              mat(2,0) != cplx(2,2) || mat(2,1) != cplx(0,0) || mat(2,2) != cplx(7,7) ||
//              mat(3,0) != cplx(0,0) || mat(3,1) != cplx(5,5) || mat(3,2) != cplx(0,0) ||
//              mat(4,0) != cplx(3,3) || mat(4,1) != cplx(0,0) || mat(4,2) != cplx(8,8) ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Transpose operation failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat << "\n"
//                 << "   Expected result:\n( (1,1) (0,0) (6,6) )\n"
//                                         "( (0,0) (4,4) (0,0) )\n"
//                                         "( (2,2) (0,0) (7,7) )\n"
//                                         "( (0,0) (5,5) (0,0) )\n"
//                                         "( (3,3) (0,0) (8,8) )\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Self-transpose of a 5x3 tensor
//       {
//          blaze::DynamicTensor<cplx> mat( 5UL, 3UL, cplx() );
//          mat(0,0) = cplx(1,-1);
//          mat(0,2) = cplx(6,-6);
//          mat(1,1) = cplx(4,-4);
//          mat(2,0) = cplx(2,-2);
//          mat(2,2) = cplx(7,-7);
//          mat(3,1) = cplx(5,-5);
//          mat(4,0) = cplx(3,-3);
//          mat(4,2) = cplx(8,-8);
//
//          mat = ctrans( mat );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  8UL );
//          checkNonZeros( mat,  0UL, 3UL );
//          checkNonZeros( mat,  1UL, 2UL );
//          checkNonZeros( mat,  2UL, 3UL );
//
//          if( mat(0,0) != cplx(1,1) || mat(0,1) != cplx(0,0) || mat(0,2) != cplx(2,2) || mat(0,3) != cplx(0,0) || mat(0,4) != cplx(3,3) ||
//              mat(1,0) != cplx(0,0) || mat(1,1) != cplx(4,4) || mat(1,2) != cplx(0,0) || mat(1,3) != cplx(5,5) || mat(1,4) != cplx(0,0) ||
//              mat(2,0) != cplx(6,6) || mat(2,1) != cplx(0,0) || mat(2,2) != cplx(7,7) || mat(2,3) != cplx(0,0) || mat(2,4) != cplx(8,8) ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Transpose operation failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat << "\n"
//                 << "   Expected result:\n( (1,1) (0,0) (2,2) (0,0) (3,3) )\n"
//                                         "( (0,0) (4,4) (0,0) (5,5) (0,0) )\n"
//                                         "( (6,6) (0,0) (7,7) (0,0) (8,8) )\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//    {
//       test_ = "Row-major self-transpose via ctrans() (stress test)";
//
//       using cplx = blaze::complex<int>;
//
//       const size_t n( blaze::rand<size_t>( 0UL, 100UL ) );
//
//       blaze::DynamicTensor<cplx> mat1( n, n, 0 );
//       randomize( mat1 );
//       blaze::DynamicTensor<cplx> mat2( mat1 );
//
//       mat1 = ctrans( mat1 );
//
//       if( mat1 != ctrans( mat2 ) ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Transpose operation failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << ctrans( mat2 ) << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDefault() function with the DynamicTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDefault() function with the DynamicTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testIsDefault()
{
   using blaze::isDefault;


   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================
   {
      test_ = "Row-major isDefault() function";

      // isDefault with 0x0 tensor
      {
         blaze::DynamicTensor<int> mat;

         if( isDefault( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with default tensor
      {
         blaze::DynamicTensor<int> mat( 2UL, 2UL, 3UL, 0 );

         if( isDefault( mat(0,0,1) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Tensor element: " << mat(0,0,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default tensor
      {
         blaze::DynamicTensor<int> mat( 2UL, 3UL, 2UL, 0 );
         mat(1,0,1) = 1;

         if( isDefault( mat(1,0,1) ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Tensor element: " << mat(1,0,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************

} // namespace dynamictensor

} // namespace mathtest

} // namespace blazetest




//=================================================================================================
//
//  MAIN FUNCTION
//
//=================================================================================================

#if defined(BLAZE_USE_HPX_THREADS)
#include <hpx/hpx_main.hpp>
#endif

//*************************************************************************************************
int main()
{
   std::cout << "   Running DynamicTensor class test (part 2)..." << std::endl;

   try
   {
      RUN_DYNAMICTENSOR_CLASS_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during DynamicTensor class test (part 2):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
