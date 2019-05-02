//=================================================================================================
/*!
//  \file src/mathtest/densearray/GeneralTest.cpp
//  \brief Source file for the general DenseArray operation test
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

#include <cstdlib>
#include <iostream>
#include <blaze/system/Platform.h>
#include <blazetest/mathtest/IsEqual.h>

#include <blaze_tensor/math/DynamicArray.h>
#include <blaze_tensor/math/dense/DenseArray.h>

#include <blazetest/mathtest/densearray/GeneralTest.h>

namespace blazetest {

namespace mathtest {

namespace densearray {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the GeneralTest class test.
//
// \exception std::runtime_error Operation error detected.
*/
GeneralTest::GeneralTest()
{
   testIsNan();
//    testIsSquare();
//    testIsSymmetric();
//    testIsHermitian();
//    testIsLower();
   testIsUniform();
//    testIsUniLower();
//    testIsStrictlyLower();
//    testIsUpper();
//    testIsUniUpper();
//    testIsStrictlyUpper();
//    testIsDiagonal();
//    testIsIdentity();
   testMinimum();
   testMaximum();
   testSoftmax();
//    testTrace();
   testL1Norm();
   testL2Norm();
   testL3Norm();
   testL4Norm();
   testLpNorm();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the \c isnan() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isnan() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsNan()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "isnan()";

      // isnan with 0x0 array
      {
         blaze::DynamicArray<3, float> arr;

         checkRows    ( arr, 0UL );
         checkColumns ( arr, 0UL );
         checkPages   ( arr, 0UL );
         checkNonZeros( arr, 0UL );

         if( blaze::isnan( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isnan evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isnan with empty 3x5x7 array
      {
         blaze::DynamicArray<3, float> arr( blaze::init_from_value, 0.0F, 7UL, 3UL, 5UL );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 5UL );
         checkPages   ( arr, 7UL );
         checkNonZeros( arr, 0UL );

         if( blaze::isnan( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isnan evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isnan with filled 4x2x2 array
      {
         blaze::DynamicArray<3, float> arr( blaze::init_from_value, 0.0F, 2UL, 4UL, 2UL );
         arr(0,1,1) =  1.0F;
         arr(0,2,0) = -2.0F;
         arr(0,2,1) =  3.0F;
         arr(0,3,0) =  4.0F;

         arr(1,1,1) = -1.0F;
         arr(1,2,0) =  2.0F;
         arr(1,2,1) = -3.0F;
         arr(1,3,0) =  4.0F;

         checkRows    ( arr, 4UL );
         checkColumns ( arr, 2UL );
         checkPages   ( arr, 2UL );
         checkNonZeros( arr, 8UL );

         if( blaze::isnan( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isnan evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


#if 0
//*************************************************************************************************
/*!\brief Test of the \c isSquare() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSquare() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsSquare()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major isSquare()";

      // Square array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );

         checkRows   ( arr, 3UL );
         checkColumns( arr, 3UL );

         if( isSquare( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSquare evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-square array
      {
         blaze::DynamicArray<3, int> arr( 2UL, 3UL, 0 );

         checkRows   ( arr, 2UL );
         checkColumns( arr, 3UL );

         if( isSquare( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSquare evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isSymmetric() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSymmetric() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsSymmetric()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major isSymmetric()";

      // Non-square array
      {
         blaze::DynamicArray<3, int> arr( 2UL, 3UL, 0 );

         checkRows    ( arr, 2UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 6UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );

         if( isSymmetric( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );
         checkNonZeros( arr, 2UL, 0UL );

         if( isSymmetric( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 2;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isSymmetric( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-symmetric array (addition element in the lower part)
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 2;
         arr(2,0) = 4;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 4UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isSymmetric( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-symmetric array (addition element in the upper part)
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(0,2) = 4;
         arr(1,1) = 2;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 4UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isSymmetric( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Symmetric array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(0,2) = 4;
         arr(1,1) = 2;
         arr(2,0) = 4;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 5UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isSymmetric( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isHermitian() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isHermitian() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsHermitian()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major isHermitian()";

      // Non-square array
      {
         blaze::DynamicArray<3, cplx> arr( 2UL, 3UL, 0 );

         checkRows    ( arr, 2UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 6UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );

         if( isHermitian( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized array
      {
         blaze::DynamicArray<3, cplx> arr( 3UL, 3UL, 0.0 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );
         checkNonZeros( arr, 2UL, 0UL );

         if( isHermitian( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-real diagonal element
      {
         blaze::DynamicArray<3, cplx> arr( 3UL, 3UL, 0.0 );
         arr(1,1).imag( 1 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 1UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 0UL );

         if( isHermitian( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-Hermitian array (additional element in the lower part)
      {
         blaze::DynamicArray<3, cplx> arr( 3UL, 3UL, 0.0 );
         arr(0,0).real( 1 );
         arr(1,1).real( 2 );
         arr(2,0).real( 4 );
         arr(2,2).real( 3 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 4UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isHermitian( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-Hermitian array (additional element in the upper part)
      {
         blaze::DynamicArray<3, cplx> arr( 3UL, 3UL, 0.0 );
         arr(0,0).real( 1 );
         arr(0,2).real( 4 );
         arr(1,1).real( 2 );
         arr(2,2).real( 3 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 4UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isHermitian( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-Hermitian array (invalid pair of elements)
      {
         blaze::DynamicArray<3, cplx> arr( 3UL, 3UL, 0.0 );
         arr(0,0).real( 1 );
         arr(0,2).imag( 4 );
         arr(1,1).real( 2 );
         arr(2,0).imag( 4 );
         arr(2,2).real( 3 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 5UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isHermitian( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Hermitian array
      {
         blaze::DynamicArray<3, cplx> arr( 3UL, 3UL, 0.0 );
         arr(0,0).real(  1 );
         arr(0,2).imag(  4 );
         arr(1,1).real(  2 );
         arr(2,0).imag( -4 );
         arr(2,2).real(  3 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 5UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isHermitian( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************
#endif

//*************************************************************************************************
/*!\brief Test of the \c isUniform() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUniform() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsUniform()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major isUniform()";

      // Uniform array (0x0x3)
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 5, 0UL, 0UL, 3UL );

         checkPages   ( arr, 0UL );
         checkRows    ( arr, 0UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 0UL );
         checkNonZeros( arr, 0UL );

         if( isUniform( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Uniform array (0x3x0)
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 5, 0UL, 3UL, 0UL );

         checkPages   ( arr, 0UL );
         checkRows    ( arr, 3UL );
         checkColumns ( arr, 0UL );
         checkCapacity( arr, 0UL );
         checkNonZeros( arr, 0UL );

         if( isUniform( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Uniform array (2x0x0)
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 5, 2UL, 0UL, 0UL );

         checkPages   ( arr, 2UL );
         checkRows    ( arr, 0UL );
         checkColumns ( arr, 0UL );
         checkCapacity( arr, 0UL );
         checkNonZeros( arr, 0UL );

         if( isUniform( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Uniform array (2x1x3)
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 5, 2UL, 1UL, 3UL );

         checkPages   ( arr, 2UL );
         checkRows    ( arr, 1UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 6UL );
         checkNonZeros( arr, 6UL );
         checkNonZeros( arr, 0UL, 0UL, 3UL );
         checkNonZeros( arr, 0UL, 1UL, 3UL );

         if( isUniform( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Uniform array (2x3x1)
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 5, 2UL, 3UL, 1UL );

         checkPages   ( arr, 2UL );
         checkRows    ( arr, 3UL );
         checkColumns ( arr, 1UL );
         checkCapacity( arr, 6UL );
         checkNonZeros( arr, 6UL );
         checkNonZeros( arr, 0UL, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 0UL, 1UL );
         checkNonZeros( arr, 2UL, 0UL, 1UL );
         checkNonZeros( arr, 0UL, 1UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL, 1UL );

         if( isUniform( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Uniform array (1x3x5)
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 5, 1UL, 3UL, 5UL );

         checkPages   ( arr,  1UL );
         checkRows    ( arr,  3UL );
         checkColumns ( arr,  5UL );
         checkCapacity( arr, 15UL );
         checkNonZeros( arr, 15UL );
         checkNonZeros( arr,  0UL, 0UL, 5UL );
         checkNonZeros( arr,  1UL, 0UL, 5UL );
         checkNonZeros( arr,  2UL, 0UL, 5UL );

         if( isUniform( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Uniform array (1x5x3)
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 5, 1UL, 5UL, 3UL );

         checkPages   ( arr,  1UL );
         checkRows    ( arr,  5UL );
         checkColumns ( arr,  3UL );
         checkCapacity( arr, 15UL );
         checkNonZeros( arr, 15UL );
         checkNonZeros( arr,  0UL, 0UL, 3UL );
         checkNonZeros( arr,  1UL, 0UL, 3UL );
         checkNonZeros( arr,  2UL, 0UL, 3UL );
         checkNonZeros( arr,  3UL, 0UL, 3UL );
         checkNonZeros( arr,  4UL, 0UL, 3UL );

         if( isUniform( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-uniform array (3x3x3)
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 5, 3UL, 3UL, 3UL );
         arr(2,2,2) = 3;

         checkPages   ( arr,  3UL );
         checkRows    ( arr,  3UL );
         checkColumns ( arr,  3UL );
         checkCapacity( arr, 27UL );
         checkNonZeros( arr, 27UL );
         checkNonZeros( arr, 0UL, 0UL, 3UL );
         checkNonZeros( arr, 1UL, 0UL, 3UL );
         checkNonZeros( arr, 2UL, 0UL, 3UL );
         checkNonZeros( arr, 0UL, 1UL, 3UL );
         checkNonZeros( arr, 1UL, 1UL, 3UL );
         checkNonZeros( arr, 2UL, 1UL, 3UL );
         checkNonZeros( arr, 0UL, 2UL, 3UL );
         checkNonZeros( arr, 1UL, 2UL, 3UL );
         checkNonZeros( arr, 2UL, 2UL, 3UL );

         if( isUniform( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************

#if 0
//*************************************************************************************************
/*!\brief Test of the \c isLower() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isLower() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsLower()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major isLower()";

      // Non-square array
      {
         blaze::DynamicArray<3, int> arr( 2UL, 3UL, 0 );

         checkRows    ( arr, 2UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 6UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );

         if( isLower( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );
         checkNonZeros( arr, 2UL, 0UL );

         if( isLower( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 2;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isLower( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-lower triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(0,2) = 2;
         arr(1,0) = 3;
         arr(1,1) = 4;
         arr(2,2) = 5;
         arr(2,0) = 6;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 6UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 2UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isLower( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Lower triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,0) = 2;
         arr(1,1) = 3;
         arr(2,2) = 4;
         arr(2,0) = 5;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 5UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 2UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isLower( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUniLower() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUniLower() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsUniLower()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major isUniLower()";

      // Non-square array
      {
         blaze::DynamicArray<3, int> arr( 2UL, 3UL, 0 );

         checkRows    ( arr, 2UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 6UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );

         if( isUniLower( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );
         checkNonZeros( arr, 2UL, 0UL );

         if( isUniLower( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Identity array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 1;
         arr(2,2) = 1;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isUniLower( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 2;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isUniLower( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Lower unitriangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,0) = 2;
         arr(1,1) = 1;
         arr(2,2) = 1;
         arr(2,0) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 5UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 2UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isUniLower( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Lower triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,0) = 2;
         arr(1,1) = 3;
         arr(2,2) = 4;
         arr(2,0) = 5;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 5UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 2UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isUniLower( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-lower unitriangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(0,2) = 2;
         arr(1,0) = 3;
         arr(1,1) = 1;
         arr(2,2) = 1;
         arr(2,0) = 4;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 6UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 2UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isUniLower( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isStrictlyLower() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isStrictlyLower() function for dense arrays. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsStrictlyLower()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major isStrictlyLower()";

      // Non-square array
      {
         blaze::DynamicArray<3, int> arr( 2UL, 3UL, 0 );

         checkRows    ( arr, 2UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 6UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );

         if( isStrictlyLower( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );
         checkNonZeros( arr, 2UL, 0UL );

         if( isStrictlyLower( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 2;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isStrictlyLower( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly lower triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(1,0) = 2;
         arr(2,0) = 5;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 2UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isStrictlyLower( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Lower triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,0) = 2;
         arr(1,1) = 3;
         arr(2,2) = 4;
         arr(2,0) = 5;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 5UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 2UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isStrictlyLower( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-strictly lower triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,2) = 2;
         arr(1,0) = 3;
         arr(2,0) = 4;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isStrictlyLower( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUpper() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUpper() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsUpper()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major isUpper()";

      // Non-square array
      {
         blaze::DynamicArray<3, int> arr( 2UL, 3UL, 0 );

         checkRows    ( arr, 2UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 6UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );

         if( isUpper( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );
         checkNonZeros( arr, 2UL, 0UL );

         if( isUpper( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 2;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isUpper( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-upper triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(0,2) = 2;
         arr(1,1) = 3;
         arr(1,2) = 4;
         arr(2,0) = 5;
         arr(2,2) = 6;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 6UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 2UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isUpper( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Upper triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(0,2) = 2;
         arr(1,1) = 3;
         arr(1,2) = 4;
         arr(2,2) = 5;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 5UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 2UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isUpper( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUniUpper() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUniUpper() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsUniUpper()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major isUniUpper()";

      // Non-square array
      {
         blaze::DynamicArray<3, int> arr( 2UL, 3UL, 0 );

         checkRows    ( arr, 2UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 6UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );

         if( isUniUpper( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );
         checkNonZeros( arr, 2UL, 0UL );

         if( isUniUpper( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Identity array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 1;
         arr(2,2) = 1;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isUniUpper( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 2;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isUniUpper( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Upper unitriangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(0,2) = 2;
         arr(1,1) = 1;
         arr(1,2) = 3;
         arr(2,2) = 1;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 5UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 2UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isUniUpper( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Upper triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(0,2) = 2;
         arr(1,1) = 3;
         arr(1,2) = 4;
         arr(2,2) = 5;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 5UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 2UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isUniUpper( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-upper triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(0,2) = 2;
         arr(1,1) = 1;
         arr(1,2) = 3;
         arr(2,0) = 4;
         arr(2,2) = 1;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 6UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 2UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isUniUpper( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isStrictlyUpper() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isStrictlyUpper() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsStrictlyUpper()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major isStrictlyUpper()";

      // Non-square array
      {
         blaze::DynamicArray<3, int> arr( 2UL, 3UL, 0 );

         checkRows    ( arr, 2UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 6UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );

         if( isStrictlyUpper( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );
         checkNonZeros( arr, 2UL, 0UL );

         if( isStrictlyUpper( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 2;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isStrictlyUpper( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,2) = 2;
         arr(1,2) = 4;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 2UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 0UL );

         if( isStrictlyUpper( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Upper triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(0,2) = 2;
         arr(1,1) = 3;
         arr(1,2) = 4;
         arr(2,2) = 5;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 5UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 2UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isStrictlyUpper( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-strictly upper triangular array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,2) = 2;
         arr(1,2) = 3;
         arr(2,0) = 4;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isStrictlyUpper( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDiagonal() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDiagonal() function for dense arrays. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsDiagonal()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major isDiagonal()";

      // Non-square array
      {
         blaze::DynamicArray<3, int> arr( 2UL, 3UL, 0 );

         checkRows    ( arr, 2UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 6UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );

         if( isDiagonal( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );
         checkNonZeros( arr, 2UL, 0UL );

         if( isDiagonal( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 2;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isDiagonal( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Lower array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 2;
         arr(2,0) = 4;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 4UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isDiagonal( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Upper array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(0,2) = 4;
         arr(1,1) = 2;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 4UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isDiagonal( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isIdentity() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isIdentity() function for dense arrays. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsIdentity()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major isIdentity()";

      // Non-square array
      {
         blaze::DynamicArray<3, int> arr( 2UL, 3UL, 0 );

         checkRows    ( arr, 2UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 6UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );

         if( isIdentity( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 0UL );
         checkNonZeros( arr, 0UL, 0UL );
         checkNonZeros( arr, 1UL, 0UL );
         checkNonZeros( arr, 2UL, 0UL );

         if( isIdentity( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Identity array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 1;
         arr(2,2) = 1;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isIdentity( arr ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Incomplete identity array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 0;
         arr(2,2) = 1;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 2UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 0UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isIdentity( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 2;
         arr(2,2) = 3;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 3UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isIdentity( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Lower array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(1,1) = 1;
         arr(2,0) = 2;
         arr(2,2) = 1;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 4UL );
         checkNonZeros( arr, 0UL, 1UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 2UL );

         if( isIdentity( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Upper array
      {
         blaze::DynamicArray<3, int> arr( 3UL, 3UL, 0 );
         arr(0,0) = 1;
         arr(0,2) = 2;
         arr(1,1) = 1;
         arr(2,2) = 1;

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkCapacity( arr, 9UL );
         checkNonZeros( arr, 4UL );
         checkNonZeros( arr, 0UL, 2UL );
         checkNonZeros( arr, 1UL, 1UL );
         checkNonZeros( arr, 2UL, 1UL );

         if( isIdentity( arr ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Array:\n" << arr << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************

#endif

//*************************************************************************************************
/*!\brief Test of the \c min() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c min() function for dense arrays. In case an error
// is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testMinimum()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major min()";

      // Attempt to find the minimum at the beginning in a fully filled array
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 3UL, 2UL );
         arr(0,0,0) = -1;
         arr(0,0,1) =  2;
         arr(0,1,0) =  3;
         arr(0,1,1) =  4;
         arr(0,2,0) =  5;
         arr(0,2,1) =  6;
         arr(1,0,0) = -1;
         arr(1,0,1) =  2;
         arr(1,1,0) =  3;
         arr(1,1,1) =  4;
         arr(1,2,0) =  5;
         arr(1,2,1) =  6;

         checkRows    ( arr,  3UL );
         checkColumns ( arr,  2UL );
         checkPages   ( arr,  2UL );
         checkNonZeros( arr, 12UL );

         const int minimum = min( arr );

         if( minimum != -1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: First computation failed\n"
                << " Details:\n"
                << "   Result: " << minimum << "\n"
                << "   Expected result: -1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Attempt to find the minimum at the end in a fully filled array
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 2UL, 3UL );
         arr(0,0,0) =  1;
         arr(0,0,1) =  2;
         arr(0,0,2) =  3;
         arr(0,1,0) =  4;
         arr(0,1,1) =  5;
         arr(0,1,2) = -6;
         arr(1,0,0) =  1;
         arr(1,0,1) =  2;
         arr(1,0,2) =  3;
         arr(1,1,0) =  4;
         arr(1,1,1) =  5;
         arr(1,1,2) = -6;

         checkRows    ( arr, 2UL );
         checkColumns ( arr, 3UL );
         checkPages   ( arr,  2UL );
         checkNonZeros( arr, 12UL );

         const int minimum = min( arr );

         if( minimum != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Second computation failed\n"
                << " Details:\n"
                << "   Result: " << minimum << "\n"
                << "   Expected result: -6\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Attempt to find the minimum at the beginning in a partially filled array
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 5UL, 3UL );
         arr(0,0,0) = -1;
         arr(0,0,2) =  2;
         arr(0,2,1) =  3;
         arr(0,4,0) =  4;
         arr(0,4,2) =  5;
         arr(1,0,0) = -1;
         arr(1,0,2) =  2;
         arr(1,2,1) =  3;
         arr(1,4,0) =  4;
         arr(1,4,2) =  5;

         checkRows    ( arr,  5UL );
         checkColumns ( arr,  3UL );
         checkPages   ( arr,  2UL );
         checkNonZeros( arr, 10UL );

         const int minimum = min( arr );

         if( minimum != -1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Third computation failed\n"
                << " Details:\n"
                << "   Result: " << minimum << "\n"
                << "   Expected result: -1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Attempt to find the minimum at the end in a partially filled array
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 3UL, 5UL );
         arr(0,0,0) =  1;
         arr(0,0,4) =  2;
         arr(0,1,2) =  3;
         arr(0,2,0) =  4;
         arr(0,2,4) = -5;
         arr(1,0,0) =  1;
         arr(1,0,4) =  2;
         arr(1,1,2) =  3;
         arr(1,2,0) =  4;
         arr(1,2,4) = -5;

         checkRows    ( arr,  3UL );
         checkColumns ( arr,  5UL );
         checkPages   ( arr,  2UL );
         checkNonZeros( arr, 10UL );

         const int minimum = min( arr );

         if( minimum != -5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Fourth computation failed\n"
                << " Details:\n"
                << "   Result: " << minimum << "\n"
                << "   Expected result: -5\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Attempt to detect 0 as the minimum value
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 3UL, 3UL, 3UL );
         arr(0,0,0) = 1;
         arr(0,0,2) = 2;
         arr(0,1,1) = 3;
         arr(0,2,0) = 4;
         arr(0,2,2) = 5;
         arr(2,0,0) = 1;
         arr(2,0,2) = 2;
         arr(2,1,1) = 3;
         arr(2,2,0) = 4;
         arr(2,2,2) = 5;

         checkRows    ( arr,  3UL );
         checkColumns ( arr,  3UL );
         checkPages   ( arr,  3UL );
         checkNonZeros( arr, 10UL );

         const int minimum = min( arr );

         if( minimum != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Fifth computation failed\n"
                << " Details:\n"
                << "   Result: " << minimum << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c max() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c max() function for dense arrays. In case an error
// is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testMaximum()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major max()";

      // Attempt to find the maximum at the beginning in a fully filled array
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 3UL, 2UL );
         arr(0,0,0) =  1;
         arr(0,0,1) = -2;
         arr(0,1,0) = -3;
         arr(0,1,1) = -4;
         arr(0,2,0) = -5;
         arr(0,2,1) = -6;
         arr(1,0,0) =  0;
         arr(1,0,1) = -2;
         arr(1,1,0) = -3;
         arr(1,1,1) = -4;
         arr(1,2,0) = -5;
         arr(1,2,1) = -6;

         checkRows    ( arr,  3UL );
         checkColumns ( arr,  2UL );
         checkPages   ( arr,  2UL );
         checkNonZeros( arr, 11UL );

         const int maximum = max( arr );

         if( maximum != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: First computation failed\n"
                << " Details:\n"
                << "   Result: " << maximum << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Attempt to find the maximum at the end in a fully filled array
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 2UL, 3UL );
         arr(0,0,0) = -1;
         arr(0,0,1) = -2;
         arr(0,0,2) = -3;
         arr(0,1,0) = -4;
         arr(0,1,1) = -5;
         arr(0,1,2) = -6;
         arr(1,0,0) = -1;
         arr(1,0,1) = -2;
         arr(1,0,2) = -3;
         arr(1,1,0) = -4;
         arr(1,1,1) = -5;
         arr(1,1,2) =  6;

         checkRows    ( arr,  2UL );
         checkColumns ( arr,  3UL );
         checkPages   ( arr,  2UL );
         checkNonZeros( arr, 12UL );

         const int maximum = max( arr );

         if( maximum != 6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Second computation failed\n"
                << " Details:\n"
                << "   Result: " << maximum << "\n"
                << "   Expected result: 6\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Attempt to find the maximum at the beginning in a partially filled array
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 5UL, 3UL );
         arr(0,0,0) =  1;
         arr(0,0,2) = -2;
         arr(0,2,1) = -3;
         arr(0,4,0) = -4;
         arr(0,4,2) = -5;
         arr(1,0,0) =  0;
         arr(1,0,2) = -2;
         arr(1,2,1) = -3;
         arr(1,4,0) = -4;
         arr(1,4,2) = -5;

         checkRows    ( arr, 5UL );
         checkColumns ( arr, 3UL );
         checkPages   ( arr, 2UL );
         checkNonZeros( arr, 9UL );

         const int maximum = max( arr );

         if( maximum != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Third computation failed\n"
                << " Details:\n"
                << "   Result: " << maximum << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Attempt to find the maximum at the end in a partially filled array
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 3UL, 5UL );
         arr(0,0,0) = -1;
         arr(0,0,4) = -2;
         arr(0,1,2) = -3;
         arr(0,2,0) = -4;
         arr(0,2,4) = -5;
         arr(1,0,0) = -1;
         arr(1,0,4) = -2;
         arr(1,1,2) = -3;
         arr(1,2,0) = -4;
         arr(1,2,4) =  5;

         checkRows    ( arr,  3UL );
         checkColumns ( arr,  5UL );
         checkPages   ( arr,  2UL );
         checkNonZeros( arr, 10UL );

         const int maximum = max( arr );

         if( maximum != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Fourth computation failed\n"
                << " Details:\n"
                << "   Result: " << maximum << "\n"
                << "   Expected result: 5\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Attempt to detect 0 as the maximum value
      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 3UL, 3UL, 3UL );
         arr(0,0,0) = -1;
         arr(0,0,2) = -2;
         arr(0,1,1) = -3;
         arr(0,2,0) = -4;
         arr(0,2,2) = -5;
         arr(2,0,0) = -1;
         arr(2,0,2) = -2;
         arr(2,1,1) = -3;
         arr(2,2,0) = -4;
         arr(2,2,2) = -5;

         checkRows    ( arr,  3UL );
         checkColumns ( arr,  3UL );
         checkPages   ( arr,  3UL );
         checkNonZeros( arr, 10UL );

         const int maximum = max( arr );

         if( maximum != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Fifth computation failed\n"
                << " Details:\n"
                << "   Result: " << maximum << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c softmax() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c softmax() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testSoftmax()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major softmax()";

      blaze::DynamicArray<3, double> A( 2UL, 2UL, 2UL );
      randomize( A, -5.0, 5.0 );

      const auto B = softmax( A );

      if( B(0,0,0) <= 0.0 || B(0,0,0) > 1.0 ||
          B(0,0,1) <= 0.0 || B(0,0,1) > 1.0 ||
          B(0,1,0) <= 0.0 || B(0,1,0) > 1.0 ||
          B(0,1,1) <= 0.0 || B(0,1,1) > 1.0 ||
          B(1,0,0) <= 0.0 || B(1,0,0) > 1.0 ||
          B(1,0,1) <= 0.0 || B(1,0,1) > 1.0 ||
          B(1,1,0) <= 0.0 || B(1,1,0) > 1.0 ||
          B(1,1,1) <= 0.0 || B(1,1,1) > 1.0 ||
          !isEqual( sum( B ), 1.0 ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Softmax computation failed\n"
             << " Details:\n"
             << "   Result: " << sum( B ) << "\n"
             << "   Expected result: 1\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


#if 0
//*************************************************************************************************
/*!\brief Test of the \c trace() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c trace() function for dense arrays. In case an error
// is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testTrace()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "Row-major trace()";

      // Determining the trace of a 0x0 array
      {
         blaze::DynamicArray<3, int> arr;

         checkRows   ( arr, 0UL );
         checkColumns( arr, 0UL );

         const int trace = blaze::trace( arr );

         if( trace != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: First computation failed\n"
                << " Details:\n"
                << "   Result: " << trace << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the trace of a 3x3 array
      {
         const blaze::DynamicArray<3, int> arr{ { -1,  2, -3 }
                                                            , { -4, -5,  6 }
                                                            , {  7, -8, -9 } };

         checkRows    ( arr, 3UL );
         checkColumns ( arr, 3UL );
         checkNonZeros( arr, 9UL );

         const int trace = blaze::trace( arr );

         if( trace != -15 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Second computation failed\n"
                << " Details:\n"
                << "   Result: " << trace << "\n"
                << "   Expected result: -15\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the trace of a non-square array
      try
      {
         blaze::DynamicArray<3, int> arr( 2UL, 3UL );

         checkRows   ( arr, 2UL );
         checkColumns( arr, 3UL );

         const int trace = blaze::trace( arr );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Trace computation on a non-square array succeeded\n"
             << " Details:\n"
             << "   Result:\n" << trace << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************
#endif

//*************************************************************************************************
/*!\brief Test of the \c l1Norm() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c l1Norm() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testL1Norm()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "l1Norm() function";

      {
         blaze::DynamicArray<3, int> arr;

         const int norm = blaze::l1Norm( arr );

         if( !isEqual( norm, 0 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: L1 norm computation failed\n"
                << " Details:\n"
                << "   Result: " << norm << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 3UL, 7UL );

         const int norm = blaze::l1Norm( arr );

         if( !isEqual( norm, 0 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: L1 norm computation failed\n"
                << " Details:\n"
                << "   Result: " << norm << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray< 3, int > arr{
            {{{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}},
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}}}};

         const int norm = blaze::l1Norm( arr );

         if( !isEqual( norm, 14 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: L1 norm computation failed\n"
                << " Details:\n"
                << "   Result: " << norm << "\n"
                << "   Expected result: 14\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Test of the \c l2Norm() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c l2Norm() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testL2Norm()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "l2Norm() function";

      {
         blaze::DynamicArray<3, int> arr;

         const double norm = blaze::l2Norm( arr );

         if( !isEqual( norm, 0.0 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: L2 norm computation failed\n"
                << " Details:\n"
                << "   Result: " << norm << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 3UL, 7UL );

         const double norm = blaze::l2Norm( arr );

         if( !isEqual( norm, 0.0 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: L2 norm computation failed\n"
                << " Details:\n"
                << "   Result: " << norm << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray< 3, int > arr{{
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}},
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}}}};

         const double norm = blaze::l2Norm( arr );

         if( !isEqual( norm, 4.6904157598234297 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: L2 norm computation failed\n"
                << " Details:\n"
                << "   Result: " << norm << "\n"
                << "   Expected result: 4.6904157598234297\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c l3Norm() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c l3Norm() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testL3Norm()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "l3Norm() function";

      {
         blaze::DynamicArray<3, int> arr;

         const double norm = blaze::l3Norm( arr );

         if( !isEqual( norm, 0.0 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: L3 norm computation failed\n"
                << " Details:\n"
                << "   Result: " << norm << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 3UL, 7UL );

         const double norm = blaze::l3Norm( arr );

         if( !isEqual( norm, 0.0 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: L3 norm computation failed\n"
                << " Details:\n"
                << "   Result: " << norm << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray<3,  int > arr{{
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}},
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}}}};

         const double norm = blaze::l3Norm( arr );

         if( !isEqual( norm, 3.3619754067989636 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: L3 norm computation failed\n"
                << " Details:\n"
                << "   Result: " << norm << "\n"
                << "   Expected result: 3.3619754067989636\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c l4Norm() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c l4Norm() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testL4Norm()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "l4Norm() function";

      {
         blaze::DynamicArray<3, int> arr;

         const double norm = blaze::l4Norm( arr );

         if( !isEqual( norm, 0.0 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: L4 norm computation failed\n"
                << " Details:\n"
                << "   Result: " << norm << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 3UL, 7UL );

         const double norm = blaze::l4Norm( arr );

         if( !isEqual( norm, 0.0 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: L4 norm computation failed\n"
                << " Details:\n"
                << "   Result: " << norm << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray<3,  int > arr{{
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}},
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}}}};

         const double norm = blaze::l4Norm( arr );

         if( !isEqual( norm, 2.8925076085190780 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: L4 norm computation failed\n"
                << " Details:\n"
                << "   Result: " << norm << "\n"
                << "   Expected result: 2.8925076085190780\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c lpNorm() function for dense arrays.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c lpNorm() function for dense arrays. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testLpNorm()
{
   //=====================================================================================
   // Row-major array tests
   //=====================================================================================

   {
      test_ = "lpNorm() function";

      {
         blaze::DynamicArray<3, int> arr;

         const double norm1 = blaze::lpNorm( arr, 2 );
         const double norm2 = blaze::lpNorm<2UL>( arr );

         if( !isEqual( norm1, 0.0 ) || !isEqual( norm2, 0.0 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lp norm computation failed\n"
                << " Details:\n"
                << "   lpNorm<2>(): " << norm1 << "\n"
                << "   lpNorm(2): " << norm2 << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray<3, int> arr( blaze::init_from_value, 0, 2UL, 3UL, 7UL );

         const double norm1 = blaze::lpNorm( arr, 2 );
         const double norm2 = blaze::lpNorm<2UL>( arr );

         if( !isEqual( norm1, 0.0 ) || !isEqual( norm2, 0.0 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lp norm computation failed\n"
                << " Details:\n"
                << "   lpNorm<2>(): " << norm1 << "\n"
                << "   lpNorm(2): " << norm2 << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray<3, int> arr( 2UL, 5UL, 10UL );
         randomize( arr, -5, 5 );

         const int norm1( blaze::lpNorm( arr, 1 ) );
         const int norm2( blaze::lpNorm<1UL>( arr ) );
         const int norm3( blaze::l1Norm( arr ) );

         if( !isEqual( norm1, norm3 ) || !isEqual( norm2, norm3 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lp norm computation failed\n"
                << " Details:\n"
                << "   lpNorm<1>(): " << norm1 << "\n"
                << "   lpNorm(1): " << norm2 << "\n"
                << "   Expected result: " << norm3 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray<3, int> arr( 2UL, 5UL, 10UL );
         randomize( arr, -5, 5 );

         const double norm1( blaze::lpNorm( arr, 2 ) );
         const double norm2( blaze::lpNorm<2UL>( arr ) );
         const double norm3( blaze::l2Norm( arr ) );

         if( !isEqual( norm1, norm3 ) || !isEqual( norm2, norm3 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lp norm computation failed\n"
                << " Details:\n"
                << "   lpNorm<2>(): " << norm1 << "\n"
                << "   lpNorm(2): " << norm2 << "\n"
                << "   Expected result: " << norm3 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray<3, int> arr( 2UL, 5UL, 10UL );
         randomize( arr, -5, 5 );

         const double norm1( blaze::lpNorm( arr, 3 ) );
         const double norm2( blaze::lpNorm<3UL>( arr ) );
         const double norm3( blaze::l3Norm( arr ) );

         if( !isEqual( norm1, norm3 ) || !isEqual( norm2, norm3 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lp norm computation failed\n"
                << " Details:\n"
                << "   lpNorm<3>(): " << norm1 << "\n"
                << "   lpNorm(3): " << norm2 << "\n"
                << "   Expected result: " << norm3 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      {
         blaze::DynamicArray<3, int> arr( 2UL, 5UL, 10UL );
         randomize( arr, -5, 5 );

         const double norm1( blaze::lpNorm( arr, 4 ) );
         const double norm2( blaze::lpNorm<4UL>( arr ) );
         const double norm3( blaze::l4Norm( arr ) );

         if( !isEqual( norm1, norm3 ) || !isEqual( norm2, norm3 ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lp norm computation failed\n"
                << " Details:\n"
                << "   lpNorm<4>(): " << norm1 << "\n"
                << "   lpNorm(4): " << norm2 << "\n"
                << "   Expected result: " << norm3 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************

} // namespace densearray

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
   std::cout << "   Running general DenseArray operation test..." << std::endl;

   try
   {
      RUN_DENSEARRAY_GENERAL_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during general DenseArray operation test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
