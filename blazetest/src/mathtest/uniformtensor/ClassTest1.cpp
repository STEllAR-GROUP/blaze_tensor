//=================================================================================================
/*!
//  \file src/mathtest/uniformtensor/ClassTest1.cpp
//  \brief Source file for the UniformTensor class test (part 1)
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018-2019 Hartmut Kaiser - All Rights Reserved
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

#include <blaze/util/Memory.h>
#include <blaze/util/Random.h>
#include <blaze/util/policies/Deallocate.h>

#include <blaze_tensor/math/CustomTensor.h>
#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/UniformTensor.h>

#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>
#include <blazetest/mathtest/uniformtensor/ClassTest.h>

#include <cstdlib>
#include <iostream>
#include <memory>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace uniformtensor {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the UniformTensor class test.
//
// \exception std::runtime_error Operation error detected.
*/
ClassTest::ClassTest()
{
   testConstructors();
   testAssignment();
   testAddAssign();
   testSubAssign();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the UniformTensor constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the UniformTensor class template.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testConstructors()
{
   //=====================================================================================
   // Row-major default constructor
   //=====================================================================================

   // Default constructor
   {
      test_ = "Row-major UniformTensor default constructor";

      blaze::UniformTensor<int> mat;

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }


   //=====================================================================================
   // Row-major size constructor
   //=====================================================================================

   {
      test_ = "Row-major UniformTensor size constructor (0x0)";

      blaze::UniformTensor<int> mat( 0UL, 0UL, 0UL );

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "Row-major UniformTensor size constructor (0x0x4)";

      blaze::UniformTensor<int> mat( 0UL, 0UL, 4UL );

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 4UL );
      checkPages   ( mat, 0UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "Row-major UniformTensor size constructor (0x3x0)";

      blaze::UniformTensor<int> mat( 0UL, 3UL, 0UL );

      checkRows    ( mat, 3UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "Row-major UniformTensor default constructor (2x0x0)";

      blaze::UniformTensor<int> mat( 2UL, 0UL, 0UL );

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 2UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "Row-major UniformTensor size constructor (2x3x4)";

      blaze::UniformTensor<int> mat( 2UL, 3UL, 4UL );

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  4UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 24UL );
      checkNonZeros( mat,  0UL );
      checkNonZeros( mat,  0UL, 0UL, 0UL );
      checkNonZeros( mat,  1UL, 0UL, 0UL );
      checkNonZeros( mat,  2UL, 0UL, 0UL );
      checkNonZeros( mat,  0UL, 1UL, 0UL );
      checkNonZeros( mat,  1UL, 1UL, 0UL );
      checkNonZeros( mat,  2UL, 1UL, 0UL );

      if( mat(0,0,0) != 0 || mat(0,0,1) != 0 || mat(0,0,2) != 0 || mat(0,0,3) != 0 ||
          mat(0,1,0) != 0 || mat(0,1,1) != 0 || mat(0,1,2) != 0 || mat(0,1,3) != 0 ||
          mat(0,2,0) != 0 || mat(0,2,1) != 0 || mat(0,2,2) != 0 || mat(0,2,3) != 0 ||
          mat(1,0,0) != 0 || mat(1,0,1) != 0 || mat(1,0,2) != 0 || mat(1,0,3) != 0 ||
          mat(1,1,0) != 0 || mat(1,1,1) != 0 || mat(1,1,2) != 0 || mat(1,1,3) != 0 ||
          mat(1,2,0) != 0 || mat(1,2,1) != 0 || mat(1,2,2) != 0 || mat(1,2,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n"
                     "(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n"
                     " ( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major homogeneous initialization
   //=====================================================================================

   {
      test_ = "Row-major UniformTensor homogeneous initialization constructor (0x0x0)";

      blaze::UniformTensor<int> mat( 0UL, 0UL, 0UL, 2 );

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "Row-major UniformTensor homogeneous initialization constructor (0x0x4)";

      blaze::UniformTensor<int> mat( 0UL, 0UL, 4UL, 2 );

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 4UL );
      checkPages   ( mat, 0UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "Row-major UniformTensor homogeneous initialization constructor (0x3x0)";

      blaze::UniformTensor<int> mat( 0UL, 3UL, 0UL, 2 );

      checkRows    ( mat, 3UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "Row-major UniformTensor homogeneous initialization constructor (2x0x0)";

      blaze::UniformTensor<int> mat( 2UL, 0UL, 0UL, 2 );

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 2UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "Row-major UniformTensor homogeneous initialization constructor (2x3x4)";

      blaze::UniformTensor<int> mat( 2UL, 3UL, 4UL, 2 );

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  4UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 24UL );
      checkNonZeros( mat, 24UL );
      checkNonZeros( mat,  0UL, 0UL, 4UL );
      checkNonZeros( mat,  1UL, 0UL, 4UL );
      checkNonZeros( mat,  2UL, 0UL, 4UL );
      checkNonZeros( mat,  0UL, 1UL, 4UL );
      checkNonZeros( mat,  1UL, 1UL, 4UL );
      checkNonZeros( mat,  2UL, 1UL, 4UL );

      if( mat(0,0,0) != 2 || mat(0,0,1) != 2 || mat(0,0,2) != 2 || mat(0,0,3) != 2 ||
          mat(0,1,0) != 2 || mat(0,1,1) != 2 || mat(0,1,2) != 2 || mat(0,1,3) != 2 ||
          mat(0,2,0) != 2 || mat(0,2,1) != 2 || mat(0,2,2) != 2 || mat(0,2,3) != 2 ||
          mat(1,0,0) != 2 || mat(1,0,1) != 2 || mat(1,0,2) != 2 || mat(1,0,3) != 2 ||
          mat(1,1,0) != 2 || mat(1,1,1) != 2 || mat(1,1,2) != 2 || mat(1,1,3) != 2 ||
          mat(1,2,0) != 2 || mat(1,2,1) != 2 || mat(1,2,2) != 2 || mat(1,2,3) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n"
                     "(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n"
                     " ( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major copy constructor
   //=====================================================================================

   {
      test_ = "Row-major UniformTensor copy constructor (0x0x0)";

      blaze::UniformTensor<int> mat1( 0UL, 0UL, 0UL );
      blaze::UniformTensor<int> mat2( mat1 );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 0UL );
      checkCapacity( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major UniformTensor copy constructor (0x0x3)";

      blaze::UniformTensor<int> mat1( 0UL, 0UL, 3UL );
      blaze::UniformTensor<int> mat2( mat1 );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 3UL );
      checkPages   ( mat2, 0UL );
      checkCapacity( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major UniformTensor copy constructor (0x4x0)";

      blaze::UniformTensor<int> mat1( 0UL, 4UL, 0UL );
      blaze::UniformTensor<int> mat2( mat1 );

      checkRows    ( mat2, 4UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 0UL );
      checkCapacity( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major UniformTensor copy constructor (2x0x0)";

      blaze::UniformTensor<int> mat1( 2UL, 0UL, 0UL );
      blaze::UniformTensor<int> mat2( mat1 );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 2UL );
      checkCapacity( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major UniformTensor copy constructor (2x3x4)";

      blaze::UniformTensor<int> mat1( 2UL, 3UL, 4UL, 2 );
      blaze::UniformTensor<int> mat2( mat1 );

      checkRows    ( mat2,  3UL );
      checkColumns ( mat2,  4UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 24UL );
      checkNonZeros( mat2, 24UL );
      checkNonZeros( mat2,  0UL, 0UL, 4UL );
      checkNonZeros( mat2,  1UL, 0UL, 4UL );
      checkNonZeros( mat2,  2UL, 0UL, 4UL );
      checkNonZeros( mat2,  0UL, 1UL, 4UL );
      checkNonZeros( mat2,  1UL, 1UL, 4UL );
      checkNonZeros( mat2,  2UL, 1UL, 4UL );

      if( mat2(0,0,0) != 2 || mat2(0,0,1) != 2 || mat2(0,0,2) != 2 || mat2(0,0,3) != 2 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 2 || mat2(0,1,2) != 2 || mat2(0,1,3) != 2 ||
          mat2(0,2,0) != 2 || mat2(0,2,1) != 2 || mat2(0,2,2) != 2 || mat2(0,2,3) != 2 ||
          mat2(1,0,0) != 2 || mat2(1,0,1) != 2 || mat2(1,0,2) != 2 || mat2(1,0,3) != 2 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 2 || mat2(1,1,2) != 2 || mat2(1,1,3) != 2 ||
          mat2(1,2,0) != 2 || mat2(1,2,1) != 2 || mat2(1,2,2) != 2 || mat2(1,2,3) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n"
                     "(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n"
                     " ( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major move constructor
   //=====================================================================================

   {
      test_ = "Row-major UniformTensor move constructor (0x0x0)";

      blaze::UniformTensor<int> mat1( 0UL, 0UL, 0UL );
      blaze::UniformTensor<int> mat2( std::move( mat1 ) );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 0UL );
      checkCapacity( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major UniformTensor move constructor (0x0x3)";

      blaze::UniformTensor<int> mat1( 0UL, 0UL, 3UL );
      blaze::UniformTensor<int> mat2( std::move( mat1 ) );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 3UL );
      checkPages   ( mat2, 0UL );
      checkCapacity( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major UniformTensor move constructor (0x4x0)";

      blaze::UniformTensor<int> mat1( 0UL, 4UL, 0UL );
      blaze::UniformTensor<int> mat2( std::move( mat1 ) );

      checkRows    ( mat2, 4UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 0UL );
      checkCapacity( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major UniformTensor move constructor (2x0x0)";

      blaze::UniformTensor<int> mat1( 2UL, 0UL, 0UL );
      blaze::UniformTensor<int> mat2( std::move( mat1 ) );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 2UL );
      checkCapacity( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major UniformTensor move constructor (2x3x4)";

      blaze::UniformTensor<int> mat1( 2UL, 3UL, 4UL, 2 );
      blaze::UniformTensor<int> mat2( std::move( mat1 ) );

      checkRows    ( mat2,  3UL );
      checkColumns ( mat2,  4UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 24UL );
      checkNonZeros( mat2, 24UL );
      checkNonZeros( mat2,  0UL, 0UL, 4UL );
      checkNonZeros( mat2,  1UL, 0UL, 4UL );
      checkNonZeros( mat2,  2UL, 0UL, 4UL );
      checkNonZeros( mat2,  0UL, 1UL, 4UL );
      checkNonZeros( mat2,  1UL, 1UL, 4UL );
      checkNonZeros( mat2,  2UL, 1UL, 4UL );

      if( mat2(0,0,0) != 2 || mat2(0,0,1) != 2 || mat2(0,0,2) != 2 || mat2(0,0,3) != 2 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 2 || mat2(0,1,2) != 2 || mat2(0,1,3) != 2 ||
          mat2(0,2,0) != 2 || mat2(0,2,1) != 2 || mat2(0,2,2) != 2 || mat2(0,2,3) != 2 ||
          mat2(1,0,0) != 2 || mat2(1,0,1) != 2 || mat2(1,0,2) != 2 || mat2(1,0,3) != 2 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 2 || mat2(1,1,2) != 2 || mat2(1,1,3) != 2 ||
          mat2(1,2,0) != 2 || mat2(1,2,1) != 2 || mat2(1,2,2) != 2 || mat2(1,2,3) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n"
                     "(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n"
                     " ( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dense tensor constructor
   //=====================================================================================

   {
      test_ = "Row-major/row-major UniformTensor dense tensor constructor (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      AlignedPadded mat1( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat1(0,0,0) = 2;
      mat1(0,0,1) = 2;
      mat1(0,0,2) = 2;
      mat1(0,1,0) = 2;
      mat1(0,1,1) = 2;
      mat1(0,1,2) = 2;
      mat1(1,0,0) = 2;
      mat1(1,0,1) = 2;
      mat1(1,0,2) = 2;
      mat1(1,1,0) = 2;
      mat1(1,1,1) = 2;
      mat1(1,1,2) = 2;

      blaze::UniformTensor<int> mat2( mat1 );

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 2 || mat2(0,0,1) != 2 || mat2(0,0,2) != 2 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 2 || mat2(0,1,2) != 2 ||
          mat2(1,0,0) != 2 || mat2(1,0,1) != 2 || mat2(1,0,2) != 2 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 2 || mat2(1,1,2) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 2 2 2 )\n( 2 2 2 )\n)(( 2 2 2 )\n( 2 2 2 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major UniformTensor dense tensor constructor (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[13UL] );
      UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 2UL, 3UL );
      mat1(0,0,0) = 2;
      mat1(0,0,1) = 2;
      mat1(0,0,2) = 2;
      mat1(0,1,0) = 2;
      mat1(0,1,1) = 2;
      mat1(0,1,2) = 2;
      mat1(1,0,0) = 2;
      mat1(1,0,1) = 2;
      mat1(1,0,2) = 2;
      mat1(1,1,0) = 2;
      mat1(1,1,1) = 2;
      mat1(1,1,2) = 2;

      blaze::UniformTensor<int> mat2( mat1 );

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 2 || mat2(0,0,1) != 2 || mat2(0,0,2) != 2 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 2 || mat2(0,1,2) != 2 ||
          mat2(1,0,0) != 2 || mat2(1,0,1) != 2 || mat2(1,0,2) != 2 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 2 || mat2(1,1,2) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 2 2 2 )\n( 2 2 2 )\n)(( 2 2 2 )\n( 2 2 2 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major UniformTensor dense tensor constructor (non-uniform)";

      blaze::DynamicTensor<int> mat1{{{1, 2, 3}, {4, 5, 6}},
                                     {{1, 2, 3}, {4, 5, 6}}};

      try {
         blaze::UniformTensor<int> mat2( mat1 );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of non-uniform UniformTensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the UniformTensor assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the UniformTensor class template.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testAssignment()
{
   //=====================================================================================
   // Row-major homogeneous assignment
   //=====================================================================================

   {
      test_ = "Row-major UniformTensor homogeneous assignment";

      blaze::UniformTensor<int> mat( 2UL, 3UL, 4UL );
      mat = 2;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  4UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 24UL );
      checkNonZeros( mat, 24UL );
      checkNonZeros( mat,  0UL, 0UL, 4UL );
      checkNonZeros( mat,  1UL, 0UL, 4UL );
      checkNonZeros( mat,  2UL, 0UL, 4UL );
      checkNonZeros( mat,  0UL, 1UL, 4UL );
      checkNonZeros( mat,  1UL, 1UL, 4UL );
      checkNonZeros( mat,  2UL, 1UL, 4UL );

      if( mat(0,0,0) != 2 || mat(0,0,1) != 2 || mat(0,0,2) != 2 || mat(0,0,3) != 2 ||
          mat(0,1,0) != 2 || mat(0,1,1) != 2 || mat(0,1,2) != 2 || mat(0,1,3) != 2 ||
          mat(0,2,0) != 2 || mat(0,2,1) != 2 || mat(0,2,2) != 2 || mat(0,2,3) != 2 ||
          mat(1,0,0) != 2 || mat(1,0,1) != 2 || mat(1,0,2) != 2 || mat(1,0,3) != 2 ||
          mat(1,1,0) != 2 || mat(1,1,1) != 2 || mat(1,1,2) != 2 || mat(1,1,3) != 2 ||
          mat(1,2,0) != 2 || mat(1,2,1) != 2 || mat(1,2,2) != 2 || mat(1,2,3) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n"
                        "(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n"
                        " ( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major copy assignment
   //=====================================================================================

   {
      test_ = "Row-major UniformTensor copy assignment";

      blaze::UniformTensor<int> mat1( 2UL, 3UL, 4UL, 2 );
      blaze::UniformTensor<int> mat2;
      mat2 = mat1;

      checkRows    ( mat2,  3UL );
      checkColumns ( mat2,  4UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 24UL );
      checkNonZeros( mat2, 24UL );
      checkNonZeros( mat2,  0UL, 0UL, 4UL );
      checkNonZeros( mat2,  1UL, 0UL, 4UL );
      checkNonZeros( mat2,  2UL, 0UL, 4UL );
      checkNonZeros( mat2,  0UL, 1UL, 4UL );
      checkNonZeros( mat2,  1UL, 1UL, 4UL );
      checkNonZeros( mat2,  2UL, 1UL, 4UL );

      if( mat2(0,0,0) != 2 || mat2(0,0,1) != 2 || mat2(0,0,2) != 2 || mat2(0,0,3) != 2 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 2 || mat2(0,1,2) != 2 || mat2(0,1,3) != 2 ||
          mat2(0,2,0) != 2 || mat2(0,2,1) != 2 || mat2(0,2,2) != 2 || mat2(0,2,3) != 2 ||
          mat2(1,0,0) != 2 || mat2(1,0,1) != 2 || mat2(1,0,2) != 2 || mat2(1,0,3) != 2 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 2 || mat2(1,1,2) != 2 || mat2(1,1,3) != 2 ||
          mat2(1,2,0) != 2 || mat2(1,2,1) != 2 || mat2(1,2,2) != 2 || mat2(1,2,3) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n"
                        "(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n"
                        " ( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major UniformTensor copy assignment stress test";

      using RandomTensorType = blaze::UniformTensor<int>;

      blaze::UniformTensor<int> mat1;
      const int min( randmin );
      const int max( randmax );

      for( size_t i=0UL; i<100UL; ++i )
      {
         const size_t pages  ( blaze::rand<size_t>( 0UL, 10UL ) );
         const size_t rows   ( blaze::rand<size_t>( 0UL, 10UL ) );
         const size_t columns( blaze::rand<size_t>( 0UL, 10UL ) );
         const RandomTensorType mat2( blaze::rand<RandomTensorType>( pages, rows, columns, min, max ) );

         mat1 = mat2;

         if( mat1 != mat2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment failed\n"
                << " Details:\n"
                << "   Result:\n" << mat1 << "\n"
                << "   Expected result:\n" << mat2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major move assignment
   //=====================================================================================

   {
      test_ = "Row-major UniformTensor move assignment";

      blaze::UniformTensor<int> mat1( 2UL, 3UL, 4UL, 2 );
      blaze::UniformTensor<int> mat2( 3UL, 4UL, 1UL, 11 );

      mat2 = std::move( mat1 );

      checkRows    ( mat2,  3UL );
      checkColumns ( mat2,  4UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 24UL );
      checkNonZeros( mat2, 24UL );
      checkNonZeros( mat2,  0UL, 0UL, 4UL );
      checkNonZeros( mat2,  1UL, 0UL, 4UL );
      checkNonZeros( mat2,  2UL, 0UL, 4UL );
      checkNonZeros( mat2,  0UL, 1UL, 4UL );
      checkNonZeros( mat2,  1UL, 1UL, 4UL );
      checkNonZeros( mat2,  2UL, 1UL, 4UL );

      if( mat2(0,0,0) != 2 || mat2(0,0,1) != 2 || mat2(0,0,2) != 2 || mat2(0,0,3) != 2 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 2 || mat2(0,1,2) != 2 || mat2(0,1,3) != 2 ||
          mat2(0,2,0) != 2 || mat2(0,2,1) != 2 || mat2(0,2,2) != 2 || mat2(0,2,3) != 2 ||
          mat2(1,0,0) != 2 || mat2(1,0,1) != 2 || mat2(1,0,2) != 2 || mat2(1,0,3) != 2 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 2 || mat2(1,1,2) != 2 || mat2(1,1,3) != 2 ||
          mat2(1,2,0) != 2 || mat2(1,2,1) != 2 || mat2(1,2,2) != 2 || mat2(1,2,3) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n"
                        "(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n"
                        " ( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dense tensor assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major UniformTensor dense tensor assignment (mixed type)";

      blaze::UniformTensor<short> mat1( 2UL, 2UL, 3UL, 2 );
      blaze::UniformTensor<int> mat2;
      mat2 = mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 2 || mat2(0,0,1) != 2 || mat2(0,0,2) != 2 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 2 || mat2(0,1,2) != 2 ||
          mat2(1,0,0) != 2 || mat2(1,0,1) != 2 || mat2(1,0,2) != 2 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 2 || mat2(1,1,2) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 2 2 2 )\n( 2 2 2 )\n( 2 2 2 )\n( 2 2 2 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major UniformTensor dense tensor assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      AlignedPadded mat1( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat1(0,0,0) = 2;
      mat1(0,0,1) = 2;
      mat1(0,0,2) = 2;
      mat1(0,1,0) = 2;
      mat1(0,1,1) = 2;
      mat1(0,1,2) = 2;
      mat1(1,0,0) = 2;
      mat1(1,0,1) = 2;
      mat1(1,0,2) = 2;
      mat1(1,1,0) = 2;
      mat1(1,1,1) = 2;
      mat1(1,1,2) = 2;

      blaze::UniformTensor<int> mat2;
      mat2 = mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 2 || mat2(0,0,1) != 2 || mat2(0,0,2) != 2 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 2 || mat2(0,1,2) != 2 ||
          mat2(1,0,0) != 2 || mat2(1,0,1) != 2 || mat2(1,0,2) != 2 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 2 || mat2(1,1,2) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 2 2 2 )\n( 2 2 2 )\n( 2 2 2 )\n( 2 2 2 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major UniformTensor dense tensor assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::rowMajor;

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[13UL] );
      UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 2UL, 3UL );
      mat1(0,0,0) = 2;
      mat1(0,0,1) = 2;
      mat1(0,0,2) = 2;
      mat1(0,1,0) = 2;
      mat1(0,1,1) = 2;
      mat1(0,1,2) = 2;
      mat1(1,0,0) = 2;
      mat1(1,0,1) = 2;
      mat1(1,0,2) = 2;
      mat1(1,1,0) = 2;
      mat1(1,1,1) = 2;
      mat1(1,1,2) = 2;

      blaze::UniformTensor<int> mat2;
      mat2 = mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 2 || mat2(0,0,1) != 2 || mat2(0,0,2) != 2 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 2 || mat2(0,1,2) != 2 ||
          mat2(1,0,0) != 2 || mat2(1,0,1) != 2 || mat2(1,0,2) != 2 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 2 || mat2(1,1,2) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 2 2 2 )\n( 2 2 2 )\n( 2 2 2 )\n( 2 2 2 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major UniformTensor dense tensor assignment (non-uniform)";

      blaze::DynamicTensor<int> mat1{{{2, 2, 2}, {2, 0, 2}},
                                     {{2, 2, 2}, {2, 0, 2}}};

      try {
         blaze::UniformTensor<int> mat2;
         mat2 = mat1;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment of non-uniform dense tensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the UniformTensor addition assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the addition assignment operators of the UniformTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testAddAssign()
{
   //=====================================================================================
   // Row-major dense tensor addition assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major UniformTensor dense tensor addition assignment (mixed type)";

      blaze::UniformTensor<short> mat1( 2UL, 2UL, 3UL, 2 );

      blaze::UniformTensor<int> mat2( 2UL, 2UL, 3UL, 1 );

      mat2 += mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 3 || mat2(0,0,1) != 3 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 3 || mat2(0,1,1) != 3 || mat2(0,1,2) != 3 ||
          mat2(0,0,0) != 3 || mat2(0,0,1) != 3 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 3 || mat2(0,1,1) != 3 || mat2(0,1,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 3 3 3 )\n( 3 3 3 )\n( 3 3 3 )\n( 3 3 3 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major UniformTensor dense tensor addition assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      AlignedPadded mat1( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat1 = 2;

      blaze::UniformTensor<int> mat2( 2UL, 2UL, 3UL, 1 );

      mat2 += mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 3 || mat2(0,0,1) != 3 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 3 || mat2(0,1,1) != 3 || mat2(0,1,2) != 3 ||
          mat2(0,0,0) != 3 || mat2(0,0,1) != 3 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 3 || mat2(0,1,1) != 3 || mat2(0,1,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 3 3 3 )\n( 3 3 3 )\n( 3 3 3 )\n( 3 3 3 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major UniformTensor dense tensor addition assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::rowMajor;

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[13UL] );
      UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 2UL, 3UL );
      mat1 = 2;

      blaze::UniformTensor<int> mat2( 2UL, 2UL, 3UL, 1 );

      mat2 += mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 3 || mat2(0,0,1) != 3 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 3 || mat2(0,1,1) != 3 || mat2(0,1,2) != 3 ||
          mat2(0,0,0) != 3 || mat2(0,0,1) != 3 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 3 || mat2(0,1,1) != 3 || mat2(0,1,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 3 3 3 )\n( 3 3 3 )\n( 3 3 3 )\n( 3 3 3 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major UniformTensor dense tensor addition assignment (non-uniform)";

      blaze::DynamicTensor<int> mat1{{{2, 2, 2}, {2, 0, 2}},
                                     {{2, 2, 2}, {2, 0, 2}}};

      try {
         blaze::UniformTensor<int> mat2( 2UL, 2UL, 3UL, 1 );
         mat2 += mat1;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment of non-uniform dense tensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the UniformTensor subtraction assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the subtraction assignment operators of the UniformTensor
// class template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testSubAssign()
{
   //=====================================================================================
   // Row-major dense tensor subtraction assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major UniformTensor dense tensor subtraction assignment (mixed type)";

      blaze::UniformTensor<short> mat1( 2UL, 2UL, 3UL, 2 );

      blaze::UniformTensor<int> mat2( 2UL, 2UL, 3UL, 1 );

      mat2 -= mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );


      if( mat2(0,0,0) != -1 || mat2(0,0,1) != -1 || mat2(0,0,2) != -1 ||
          mat2(0,1,0) != -1 || mat2(0,1,1) != -1 || mat2(0,1,2) != -1 ||
          mat2(0,0,0) != -1 || mat2(0,0,1) != -1 || mat2(0,0,2) != -1 ||
          mat2(0,1,0) != -1 || mat2(0,1,1) != -1 || mat2(0,1,2) != -1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( -1 -1 -1 )\n( -1 -1 -1 )\n( -1 -1 -1 )\n( -1 -1 -1 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major UniformTensor dense tensor subtraction assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;
      using blaze::rowMajor;

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      AlignedPadded mat1( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat1 = 2;

      blaze::UniformTensor<int> mat2( 2UL, 2UL, 3UL, 1 );

      mat2 -= mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );


      if( mat2(0,0,0) != -1 || mat2(0,0,1) != -1 || mat2(0,0,2) != -1 ||
          mat2(0,1,0) != -1 || mat2(0,1,1) != -1 || mat2(0,1,2) != -1 ||
          mat2(0,0,0) != -1 || mat2(0,0,1) != -1 || mat2(0,0,2) != -1 ||
          mat2(0,1,0) != -1 || mat2(0,1,1) != -1 || mat2(0,1,2) != -1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( -1 -1 -1 )\n( -1 -1 -1 )\n( -1 -1 -1 )\n( -1 -1 -1 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major UniformTensor dense tensor subtraction assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[13UL] );
      UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 2UL, 3UL );
      mat1 = 2;

      blaze::UniformTensor<int> mat2( 2UL, 2UL, 3UL, 1 );

      mat2 -= mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );


      if( mat2(0,0,0) != -1 || mat2(0,0,1) != -1 || mat2(0,0,2) != -1 ||
          mat2(0,1,0) != -1 || mat2(0,1,1) != -1 || mat2(0,1,2) != -1 ||
          mat2(0,0,0) != -1 || mat2(0,0,1) != -1 || mat2(0,0,2) != -1 ||
          mat2(0,1,0) != -1 || mat2(0,1,1) != -1 || mat2(0,1,2) != -1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( -1 -1 -1 )\n( -1 -1 -1 )\n( -1 -1 -1 )\n( -1 -1 -1 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major UniformTensor dense tensor subtraction assignment (non-uniform)";

      blaze::DynamicTensor<int> mat1{{{2, 2, 2}, {2, 0, 2}},
                                     {{2, 2, 2}, {2, 0, 2}}};

      try {
         blaze::UniformTensor<int> mat2( 2UL, 3UL, 1 );
         mat2 -= mat1;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment of non-uniform dense tensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************

} // namespace uniformtensor

} // namespace mathtest

} // namespace blazetest




//=================================================================================================
//
//  MAIN FUNCTION
//
//=================================================================================================

//*************************************************************************************************
int main()
{
   std::cout << "   Running UniformTensor class test (part 1)..." << std::endl;

   try
   {
      RUN_UNIFORMTENSOR_CLASS_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during UniformTensor class test (part 1):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
