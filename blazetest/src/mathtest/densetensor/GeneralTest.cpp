//=================================================================================================
/*!
//  \file src/mathtest/densetensor/GeneralTest.cpp
//  \brief Source file for the general DenseTensor operation test
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
//     of conditions and the following disclaimer in the documentation and/or other tenserials
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

#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/dense/DenseTensor.h>

#include <blazetest/mathtest/densetensor/GeneralTest.h>

namespace blazetest {

namespace mathtest {

namespace densetensor {

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
/*!\brief Test of the \c isnan() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isnan() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsNan()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "isnan()";

      // isnan with 0x0 tensor
      {
         blaze::DynamicTensor<float> tens;

         checkRows    ( tens, 0UL );
         checkColumns ( tens, 0UL );
         checkPages   ( tens, 0UL );
         checkNonZeros( tens, 0UL );

         if( blaze::isnan( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isnan evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isnan with empty 3x5x7 tensor
      {
         blaze::DynamicTensor<float> tens( 7UL, 3UL, 5UL, 0.0F );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 5UL );
         checkPages   ( tens, 7UL );
         checkNonZeros( tens, 0UL );

         if( blaze::isnan( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isnan evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isnan with filled 4x2x2 tensor
      {
         blaze::DynamicTensor<float> tens( 2UL, 4UL, 2UL, 0.0F );
         tens(0,1,1) =  1.0F;
         tens(0,2,0) = -2.0F;
         tens(0,2,1) =  3.0F;
         tens(0,3,0) =  4.0F;

         tens(1,1,1) = -1.0F;
         tens(1,2,0) =  2.0F;
         tens(1,2,1) = -3.0F;
         tens(1,3,0) =  4.0F;

         checkRows    ( tens, 4UL );
         checkColumns ( tens, 2UL );
         checkPages   ( tens, 2UL );
         checkNonZeros( tens, 8UL );

         if( blaze::isnan( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isnan evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


#if 0
//*************************************************************************************************
/*!\brief Test of the \c isSquare() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSquare() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsSquare()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major isSquare()";

      // Square tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );

         checkRows   ( tens, 3UL );
         checkColumns( tens, 3UL );

         if( isSquare( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSquare evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-square tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 0 );

         checkRows   ( tens, 2UL );
         checkColumns( tens, 3UL );

         if( isSquare( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSquare evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isSymmetric() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSymmetric() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsSymmetric()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major isSymmetric()";

      // Non-square tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 0 );

         checkRows    ( tens, 2UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 6UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );

         if( isSymmetric( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );
         checkNonZeros( tens, 2UL, 0UL );

         if( isSymmetric( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 2;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isSymmetric( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-symmetric tensor (addition element in the lower part)
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 2;
         tens(2,0) = 4;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 4UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isSymmetric( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-symmetric tensor (addition element in the upper part)
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(0,2) = 4;
         tens(1,1) = 2;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 4UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isSymmetric( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Symmetric tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(0,2) = 4;
         tens(1,1) = 2;
         tens(2,0) = 4;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 5UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isSymmetric( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isHermitian() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isHermitian() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsHermitian()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major isHermitian()";

      // Non-square tensor
      {
         blaze::DynamicTensor<cplx> tens( 2UL, 3UL, 0 );

         checkRows    ( tens, 2UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 6UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );

         if( isHermitian( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized tensor
      {
         blaze::DynamicTensor<cplx> tens( 3UL, 3UL, 0.0 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );
         checkNonZeros( tens, 2UL, 0UL );

         if( isHermitian( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-real diagonal element
      {
         blaze::DynamicTensor<cplx> tens( 3UL, 3UL, 0.0 );
         tens(1,1).imag( 1 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 1UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 0UL );

         if( isHermitian( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-Hermitian tensor (additional element in the lower part)
      {
         blaze::DynamicTensor<cplx> tens( 3UL, 3UL, 0.0 );
         tens(0,0).real( 1 );
         tens(1,1).real( 2 );
         tens(2,0).real( 4 );
         tens(2,2).real( 3 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 4UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isHermitian( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-Hermitian tensor (additional element in the upper part)
      {
         blaze::DynamicTensor<cplx> tens( 3UL, 3UL, 0.0 );
         tens(0,0).real( 1 );
         tens(0,2).real( 4 );
         tens(1,1).real( 2 );
         tens(2,2).real( 3 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 4UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isHermitian( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-Hermitian tensor (invalid pair of elements)
      {
         blaze::DynamicTensor<cplx> tens( 3UL, 3UL, 0.0 );
         tens(0,0).real( 1 );
         tens(0,2).imag( 4 );
         tens(1,1).real( 2 );
         tens(2,0).imag( 4 );
         tens(2,2).real( 3 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 5UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isHermitian( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Hermitian tensor
      {
         blaze::DynamicTensor<cplx> tens( 3UL, 3UL, 0.0 );
         tens(0,0).real(  1 );
         tens(0,2).imag(  4 );
         tens(1,1).real(  2 );
         tens(2,0).imag( -4 );
         tens(2,2).real(  3 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 5UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isHermitian( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************
#endif

//*************************************************************************************************
/*!\brief Test of the \c isUniform() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUniform() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsUniform()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major isUniform()";

      // Uniform tensor (0x0x3)
      {
         blaze::DynamicTensor<int> tens( 0UL, 0UL, 3UL, 5 );

         checkPages   ( tens, 0UL );
         checkRows    ( tens, 0UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 0UL );
         checkNonZeros( tens, 0UL );

         if( isUniform( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Uniform tensor (0x3x0)
      {
         blaze::DynamicTensor<int> tens( 0UL, 3UL, 0UL, 5 );

         checkPages   ( tens, 0UL );
         checkRows    ( tens, 3UL );
         checkColumns ( tens, 0UL );
         checkCapacity( tens, 0UL );
         checkNonZeros( tens, 0UL );

         if( isUniform( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Uniform tensor (2x0x0)
      {
         blaze::DynamicTensor<int> tens( 2UL, 0UL, 0UL, 5 );

         checkPages   ( tens, 2UL );
         checkRows    ( tens, 0UL );
         checkColumns ( tens, 0UL );
         checkCapacity( tens, 0UL );
         checkNonZeros( tens, 0UL );

         if( isUniform( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Uniform tensor (2x1x3)
      {
         blaze::DynamicTensor<int> tens( 2UL, 1UL, 3UL, 5 );

         checkPages   ( tens, 2UL );
         checkRows    ( tens, 1UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 6UL );
         checkNonZeros( tens, 6UL );
         checkNonZeros( tens, 0UL, 0UL, 3UL );
         checkNonZeros( tens, 0UL, 1UL, 3UL );

         if( isUniform( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Uniform tensor (2x3x1)
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 1UL, 5 );

         checkPages   ( tens, 2UL );
         checkRows    ( tens, 3UL );
         checkColumns ( tens, 1UL );
         checkCapacity( tens, 6UL );
         checkNonZeros( tens, 6UL );
         checkNonZeros( tens, 0UL, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 0UL, 1UL );
         checkNonZeros( tens, 2UL, 0UL, 1UL );
         checkNonZeros( tens, 0UL, 1UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL, 1UL );

         if( isUniform( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Uniform tensor (1x3x5)
      {
         blaze::DynamicTensor<int> tens( 1UL, 3UL, 5UL, 5 );

         checkPages   ( tens,  1UL );
         checkRows    ( tens,  3UL );
         checkColumns ( tens,  5UL );
         checkCapacity( tens, 15UL );
         checkNonZeros( tens, 15UL );
         checkNonZeros( tens,  0UL, 0UL, 5UL );
         checkNonZeros( tens,  1UL, 0UL, 5UL );
         checkNonZeros( tens,  2UL, 0UL, 5UL );

         if( isUniform( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Uniform tensor (1x5x3)
      {
         blaze::DynamicTensor<int> tens( 1UL, 5UL, 3UL, 5 );

         checkPages   ( tens,  1UL );
         checkRows    ( tens,  5UL );
         checkColumns ( tens,  3UL );
         checkCapacity( tens, 15UL );
         checkNonZeros( tens, 15UL );
         checkNonZeros( tens,  0UL, 0UL, 3UL );
         checkNonZeros( tens,  1UL, 0UL, 3UL );
         checkNonZeros( tens,  2UL, 0UL, 3UL );
         checkNonZeros( tens,  3UL, 0UL, 3UL );
         checkNonZeros( tens,  4UL, 0UL, 3UL );

         if( isUniform( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-uniform tensor (3x3x3)
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 3UL, 5 );
         tens(2,2,2) = 3;

         checkPages   ( tens,  3UL );
         checkRows    ( tens,  3UL );
         checkColumns ( tens,  3UL );
         checkCapacity( tens, 27UL );
         checkNonZeros( tens, 27UL );
         checkNonZeros( tens, 0UL, 0UL, 3UL );
         checkNonZeros( tens, 1UL, 0UL, 3UL );
         checkNonZeros( tens, 2UL, 0UL, 3UL );
         checkNonZeros( tens, 0UL, 1UL, 3UL );
         checkNonZeros( tens, 1UL, 1UL, 3UL );
         checkNonZeros( tens, 2UL, 1UL, 3UL );
         checkNonZeros( tens, 0UL, 2UL, 3UL );
         checkNonZeros( tens, 1UL, 2UL, 3UL );
         checkNonZeros( tens, 2UL, 2UL, 3UL );

         if( isUniform( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************

#if 0
//*************************************************************************************************
/*!\brief Test of the \c isLower() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isLower() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsLower()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major isLower()";

      // Non-square tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 0 );

         checkRows    ( tens, 2UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 6UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );

         if( isLower( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );
         checkNonZeros( tens, 2UL, 0UL );

         if( isLower( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 2;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isLower( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-lower triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(0,2) = 2;
         tens(1,0) = 3;
         tens(1,1) = 4;
         tens(2,2) = 5;
         tens(2,0) = 6;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 6UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 2UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isLower( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Lower triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,0) = 2;
         tens(1,1) = 3;
         tens(2,2) = 4;
         tens(2,0) = 5;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 5UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 2UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isLower( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUniLower() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUniLower() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsUniLower()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major isUniLower()";

      // Non-square tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 0 );

         checkRows    ( tens, 2UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 6UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );

         if( isUniLower( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );
         checkNonZeros( tens, 2UL, 0UL );

         if( isUniLower( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Identity tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 1;
         tens(2,2) = 1;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isUniLower( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 2;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isUniLower( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Lower unitriangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,0) = 2;
         tens(1,1) = 1;
         tens(2,2) = 1;
         tens(2,0) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 5UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 2UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isUniLower( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Lower triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,0) = 2;
         tens(1,1) = 3;
         tens(2,2) = 4;
         tens(2,0) = 5;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 5UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 2UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isUniLower( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-lower unitriangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(0,2) = 2;
         tens(1,0) = 3;
         tens(1,1) = 1;
         tens(2,2) = 1;
         tens(2,0) = 4;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 6UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 2UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isUniLower( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isStrictlyLower() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isStrictlyLower() function for dense tensors. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsStrictlyLower()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major isStrictlyLower()";

      // Non-square tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 0 );

         checkRows    ( tens, 2UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 6UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );

         if( isStrictlyLower( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );
         checkNonZeros( tens, 2UL, 0UL );

         if( isStrictlyLower( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 2;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isStrictlyLower( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly lower triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(1,0) = 2;
         tens(2,0) = 5;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 2UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isStrictlyLower( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Lower triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,0) = 2;
         tens(1,1) = 3;
         tens(2,2) = 4;
         tens(2,0) = 5;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 5UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 2UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isStrictlyLower( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-strictly lower triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,2) = 2;
         tens(1,0) = 3;
         tens(2,0) = 4;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isStrictlyLower( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUpper() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUpper() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsUpper()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major isUpper()";

      // Non-square tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 0 );

         checkRows    ( tens, 2UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 6UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );

         if( isUpper( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );
         checkNonZeros( tens, 2UL, 0UL );

         if( isUpper( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 2;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isUpper( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-upper triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(0,2) = 2;
         tens(1,1) = 3;
         tens(1,2) = 4;
         tens(2,0) = 5;
         tens(2,2) = 6;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 6UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 2UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isUpper( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Upper triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(0,2) = 2;
         tens(1,1) = 3;
         tens(1,2) = 4;
         tens(2,2) = 5;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 5UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 2UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isUpper( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUniUpper() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUniUpper() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsUniUpper()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major isUniUpper()";

      // Non-square tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 0 );

         checkRows    ( tens, 2UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 6UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );

         if( isUniUpper( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );
         checkNonZeros( tens, 2UL, 0UL );

         if( isUniUpper( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Identity tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 1;
         tens(2,2) = 1;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isUniUpper( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 2;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isUniUpper( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Upper unitriangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(0,2) = 2;
         tens(1,1) = 1;
         tens(1,2) = 3;
         tens(2,2) = 1;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 5UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 2UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isUniUpper( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Upper triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(0,2) = 2;
         tens(1,1) = 3;
         tens(1,2) = 4;
         tens(2,2) = 5;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 5UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 2UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isUniUpper( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-upper triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(0,2) = 2;
         tens(1,1) = 1;
         tens(1,2) = 3;
         tens(2,0) = 4;
         tens(2,2) = 1;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 6UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 2UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isUniUpper( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isStrictlyUpper() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isStrictlyUpper() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsStrictlyUpper()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major isStrictlyUpper()";

      // Non-square tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 0 );

         checkRows    ( tens, 2UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 6UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );

         if( isStrictlyUpper( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );
         checkNonZeros( tens, 2UL, 0UL );

         if( isStrictlyUpper( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 2;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isStrictlyUpper( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,2) = 2;
         tens(1,2) = 4;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 2UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 0UL );

         if( isStrictlyUpper( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Upper triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(0,2) = 2;
         tens(1,1) = 3;
         tens(1,2) = 4;
         tens(2,2) = 5;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 5UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 2UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isStrictlyUpper( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Non-strictly upper triangular tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,2) = 2;
         tens(1,2) = 3;
         tens(2,0) = 4;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isStrictlyUpper( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDiagonal() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDiagonal() function for dense tensors. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsDiagonal()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major isDiagonal()";

      // Non-square tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 0 );

         checkRows    ( tens, 2UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 6UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );

         if( isDiagonal( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );
         checkNonZeros( tens, 2UL, 0UL );

         if( isDiagonal( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 2;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isDiagonal( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Lower tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 2;
         tens(2,0) = 4;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 4UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isDiagonal( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Upper tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(0,2) = 4;
         tens(1,1) = 2;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 4UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isDiagonal( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isIdentity() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isIdentity() function for dense tensors. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testIsIdentity()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major isIdentity()";

      // Non-square tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 0 );

         checkRows    ( tens, 2UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 6UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );

         if( isIdentity( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Default initialized tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 0UL );
         checkNonZeros( tens, 0UL, 0UL );
         checkNonZeros( tens, 1UL, 0UL );
         checkNonZeros( tens, 2UL, 0UL );

         if( isIdentity( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Identity tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 1;
         tens(2,2) = 1;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isIdentity( tens ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Incomplete identity tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 0;
         tens(2,2) = 1;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 2UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 0UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isIdentity( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Diagonal tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 2;
         tens(2,2) = 3;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 3UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isIdentity( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Lower tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(1,1) = 1;
         tens(2,0) = 2;
         tens(2,2) = 1;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 4UL );
         checkNonZeros( tens, 0UL, 1UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 2UL );

         if( isIdentity( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Upper tensor
      {
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 0 );
         tens(0,0) = 1;
         tens(0,2) = 2;
         tens(1,1) = 1;
         tens(2,2) = 1;

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkCapacity( tens, 9UL );
         checkNonZeros( tens, 4UL );
         checkNonZeros( tens, 0UL, 2UL );
         checkNonZeros( tens, 1UL, 1UL );
         checkNonZeros( tens, 2UL, 1UL );

         if( isIdentity( tens ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************

#endif

//*************************************************************************************************
/*!\brief Test of the \c min() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c min() function for dense tensors. In case an error
// is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testMinimum()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major min()";

      // Attempt to find the minimum at the beginning in a fully filled tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 2UL, 0 );
         tens(0,0,0) = -1;
         tens(0,0,1) =  2;
         tens(0,1,0) =  3;
         tens(0,1,1) =  4;
         tens(0,2,0) =  5;
         tens(0,2,1) =  6;
         tens(1,0,0) = -1;
         tens(1,0,1) =  2;
         tens(1,1,0) =  3;
         tens(1,1,1) =  4;
         tens(1,2,0) =  5;
         tens(1,2,1) =  6;

         checkRows    ( tens,  3UL );
         checkColumns ( tens,  2UL );
         checkPages   ( tens,  2UL );
         checkNonZeros( tens, 12UL );

         const int minimum = min( tens );

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

      // Attempt to find the minimum at the end in a fully filled tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 2UL, 3UL, 0 );
         tens(0,0,0) =  1;
         tens(0,0,1) =  2;
         tens(0,0,2) =  3;
         tens(0,1,0) =  4;
         tens(0,1,1) =  5;
         tens(0,1,2) = -6;
         tens(1,0,0) =  1;
         tens(1,0,1) =  2;
         tens(1,0,2) =  3;
         tens(1,1,0) =  4;
         tens(1,1,1) =  5;
         tens(1,1,2) = -6;

         checkRows    ( tens, 2UL );
         checkColumns ( tens, 3UL );
         checkPages   ( tens,  2UL );
         checkNonZeros( tens, 12UL );

         const int minimum = min( tens );

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

      // Attempt to find the minimum at the beginning in a partially filled tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 5UL, 3UL, 0 );
         tens(0,0,0) = -1;
         tens(0,0,2) =  2;
         tens(0,2,1) =  3;
         tens(0,4,0) =  4;
         tens(0,4,2) =  5;
         tens(1,0,0) = -1;
         tens(1,0,2) =  2;
         tens(1,2,1) =  3;
         tens(1,4,0) =  4;
         tens(1,4,2) =  5;

         checkRows    ( tens,  5UL );
         checkColumns ( tens,  3UL );
         checkPages   ( tens,  2UL );
         checkNonZeros( tens, 10UL );

         const int minimum = min( tens );

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

      // Attempt to find the minimum at the end in a partially filled tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 5UL, 0 );
         tens(0,0,0) =  1;
         tens(0,0,4) =  2;
         tens(0,1,2) =  3;
         tens(0,2,0) =  4;
         tens(0,2,4) = -5;
         tens(1,0,0) =  1;
         tens(1,0,4) =  2;
         tens(1,1,2) =  3;
         tens(1,2,0) =  4;
         tens(1,2,4) = -5;

         checkRows    ( tens,  3UL );
         checkColumns ( tens,  5UL );
         checkPages   ( tens,  2UL );
         checkNonZeros( tens, 10UL );

         const int minimum = min( tens );

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
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 3UL, 0 );
         tens(0,0,0) = 1;
         tens(0,0,2) = 2;
         tens(0,1,1) = 3;
         tens(0,2,0) = 4;
         tens(0,2,2) = 5;
         tens(2,0,0) = 1;
         tens(2,0,2) = 2;
         tens(2,1,1) = 3;
         tens(2,2,0) = 4;
         tens(2,2,2) = 5;

         checkRows    ( tens,  3UL );
         checkColumns ( tens,  3UL );
         checkPages   ( tens,  3UL );
         checkNonZeros( tens, 10UL );

         const int minimum = min( tens );

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
/*!\brief Test of the \c max() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c max() function for dense tensors. In case an error
// is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testMaximum()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major max()";

      // Attempt to find the maximum at the beginning in a fully filled tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 2UL, 0 );
         tens(0,0,0) =  1;
         tens(0,0,1) = -2;
         tens(0,1,0) = -3;
         tens(0,1,1) = -4;
         tens(0,2,0) = -5;
         tens(0,2,1) = -6;
         tens(1,0,0) =  0;
         tens(1,0,1) = -2;
         tens(1,1,0) = -3;
         tens(1,1,1) = -4;
         tens(1,2,0) = -5;
         tens(1,2,1) = -6;

         checkRows    ( tens,  3UL );
         checkColumns ( tens,  2UL );
         checkPages   ( tens,  2UL );
         checkNonZeros( tens, 11UL );

         const int maximum = max( tens );

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

      // Attempt to find the maximum at the end in a fully filled tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 2UL, 3UL, 0 );
         tens(0,0,0) = -1;
         tens(0,0,1) = -2;
         tens(0,0,2) = -3;
         tens(0,1,0) = -4;
         tens(0,1,1) = -5;
         tens(0,1,2) = -6;
         tens(1,0,0) = -1;
         tens(1,0,1) = -2;
         tens(1,0,2) = -3;
         tens(1,1,0) = -4;
         tens(1,1,1) = -5;
         tens(1,1,2) =  6;

         checkRows    ( tens,  2UL );
         checkColumns ( tens,  3UL );
         checkPages   ( tens,  2UL );
         checkNonZeros( tens, 12UL );

         const int maximum = max( tens );

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

      // Attempt to find the maximum at the beginning in a partially filled tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 5UL, 3UL, 0 );
         tens(0,0,0) =  1;
         tens(0,0,2) = -2;
         tens(0,2,1) = -3;
         tens(0,4,0) = -4;
         tens(0,4,2) = -5;
         tens(1,0,0) =  0;
         tens(1,0,2) = -2;
         tens(1,2,1) = -3;
         tens(1,4,0) = -4;
         tens(1,4,2) = -5;

         checkRows    ( tens, 5UL );
         checkColumns ( tens, 3UL );
         checkPages   ( tens, 2UL );
         checkNonZeros( tens, 9UL );

         const int maximum = max( tens );

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

      // Attempt to find the maximum at the end in a partially filled tensor
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 5UL, 0 );
         tens(0,0,0) = -1;
         tens(0,0,4) = -2;
         tens(0,1,2) = -3;
         tens(0,2,0) = -4;
         tens(0,2,4) = -5;
         tens(1,0,0) = -1;
         tens(1,0,4) = -2;
         tens(1,1,2) = -3;
         tens(1,2,0) = -4;
         tens(1,2,4) =  5;

         checkRows    ( tens,  3UL );
         checkColumns ( tens,  5UL );
         checkPages   ( tens,  2UL );
         checkNonZeros( tens, 10UL );

         const int maximum = max( tens );

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
         blaze::DynamicTensor<int> tens( 3UL, 3UL, 3UL, 0 );
         tens(0,0,0) = -1;
         tens(0,0,2) = -2;
         tens(0,1,1) = -3;
         tens(0,2,0) = -4;
         tens(0,2,2) = -5;
         tens(2,0,0) = -1;
         tens(2,0,2) = -2;
         tens(2,1,1) = -3;
         tens(2,2,0) = -4;
         tens(2,2,2) = -5;

         checkRows    ( tens,  3UL );
         checkColumns ( tens,  3UL );
         checkPages   ( tens,  3UL );
         checkNonZeros( tens, 10UL );

         const int maximum = max( tens );

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
/*!\brief Test of the \c softmax() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c softmax() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testSoftmax()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major softmax()";

      blaze::DynamicTensor<double> A( 2UL, 2UL, 2UL );
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
/*!\brief Test of the \c trace() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c trace() function for dense tensors. In case an error
// is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testTrace()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major trace()";

      // Determining the trace of a 0x0 tensor
      {
         blaze::DynamicTensor<int> tens;

         checkRows   ( tens, 0UL );
         checkColumns( tens, 0UL );

         const int trace = blaze::trace( tens );

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

      // Determining the trace of a 3x3 tensor
      {
         const blaze::DynamicTensor<int> tens{ { -1,  2, -3 }
                                                            , { -4, -5,  6 }
                                                            , {  7, -8, -9 } };

         checkRows    ( tens, 3UL );
         checkColumns ( tens, 3UL );
         checkNonZeros( tens, 9UL );

         const int trace = blaze::trace( tens );

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

      // Determining the trace of a non-square tensor
      try
      {
         blaze::DynamicTensor<int> tens( 2UL, 3UL );

         checkRows   ( tens, 2UL );
         checkColumns( tens, 3UL );

         const int trace = blaze::trace( tens );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Trace computation on a non-square tensor succeeded\n"
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
/*!\brief Test of the \c l1Norm() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c l1Norm() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testL1Norm()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "l1Norm() function";

      {
         blaze::DynamicTensor<int> tens;

         const int norm = blaze::l1Norm( tens );

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
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 7UL, 0 );

         const int norm = blaze::l1Norm( tens );

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
         blaze::DynamicTensor< int > tens{
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}},
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}}};

         const int norm = blaze::l1Norm( tens );

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
/*!\brief Test of the \c l2Norm() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c l2Norm() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testL2Norm()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "l2Norm() function";

      {
         blaze::DynamicTensor<int> tens;

         const double norm = blaze::l2Norm( tens );

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
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 7UL, 0 );

         const double norm = blaze::l2Norm( tens );

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
         blaze::DynamicTensor< int > tens{
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}},
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}}};

         const double norm = blaze::l2Norm( tens );

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
/*!\brief Test of the \c l3Norm() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c l3Norm() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testL3Norm()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "l3Norm() function";

      {
         blaze::DynamicTensor<int> tens;

         const double norm = blaze::l3Norm( tens );

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
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 7UL, 0 );

         const double norm = blaze::l3Norm( tens );

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
         blaze::DynamicTensor< int > tens{
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}},
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}}};

         const double norm = blaze::l3Norm( tens );

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
/*!\brief Test of the \c l4Norm() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c l4Norm() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testL4Norm()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "l4Norm() function";

      {
         blaze::DynamicTensor<int> tens;

         const double norm = blaze::l4Norm( tens );

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
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 7UL, 0 );

         const double norm = blaze::l4Norm( tens );

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
         blaze::DynamicTensor< int > tens{
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}},
            {{0, 0, 1, 0, 1, 0, 0}, {0, -2, 0, 0, 0, -1, 0},
               {0, 0, 0, 2, 0, 0, 0}}};

         const double norm = blaze::l4Norm( tens );

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
/*!\brief Test of the \c lpNorm() function for dense tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c lpNorm() function for dense tensors. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void GeneralTest::testLpNorm()
{
   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "lpNorm() function";

      {
         blaze::DynamicTensor<int> tens;

         const double norm1 = blaze::lpNorm( tens, 2 );
         const double norm2 = blaze::lpNorm<2UL>( tens );

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
         blaze::DynamicTensor<int> tens( 2UL, 3UL, 7UL, 0 );

         const double norm1 = blaze::lpNorm( tens, 2 );
         const double norm2 = blaze::lpNorm<2UL>( tens );

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
         blaze::DynamicTensor<int> tens( 2UL, 5UL, 10UL );
         randomize( tens, -5, 5 );

         const int norm1( blaze::lpNorm( tens, 1 ) );
         const int norm2( blaze::lpNorm<1UL>( tens ) );
         const int norm3( blaze::l1Norm( tens ) );

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
         blaze::DynamicTensor<int> tens( 2UL, 5UL, 10UL );
         randomize( tens, -5, 5 );

         const double norm1( blaze::lpNorm( tens, 2 ) );
         const double norm2( blaze::lpNorm<2UL>( tens ) );
         const double norm3( blaze::l2Norm( tens ) );

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
         blaze::DynamicTensor<int> tens( 2UL, 5UL, 10UL );
         randomize( tens, -5, 5 );

         const double norm1( blaze::lpNorm( tens, 3 ) );
         const double norm2( blaze::lpNorm<3UL>( tens ) );
         const double norm3( blaze::l3Norm( tens ) );

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
         blaze::DynamicTensor<int> tens( 2UL, 5UL, 10UL );
         randomize( tens, -5, 5 );

         const double norm1( blaze::lpNorm( tens, 4 ) );
         const double norm2( blaze::lpNorm<4UL>( tens ) );
         const double norm3( blaze::l4Norm( tens ) );

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

} // namespace densetensor

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
   std::cout << "   Running general DenseTensor operation test..." << std::endl;

   try
   {
      RUN_DENSETENSOR_GENERAL_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during general DenseTensor operation test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
