//=================================================================================================
/*!
//  \file src/mathtest/densetensor/UniformTest.cpp
//  \brief Source file for the uniform DenseTensor operation test
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

#include <blaze_tensor/math/UniformTensor.h>
#include <blaze_tensor/math/dense/DenseTensor.h>

#include <blazetest/mathtest/densetensor/UniformTest.h>

#include <cstdlib>
#include <iostream>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace densetensor {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the UniformTest class test.
//
// \exception std::runtime_error Operation error detected.
*/
UniformTest::UniformTest()
{
   testIsSymmetric();
   testIsHermitian();
   testIsUniform();
   testIsZero();
   testIsLower();
   testIsUniLower();
   testIsStrictlyLower();
   testIsUpper();
   testIsUniUpper();
   testIsStrictlyUpper();
   testIsDiagonal();
   testIsIdentity();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the \c isSymmetric() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSymmetric() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void UniformTest::testIsSymmetric()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isSymmetric()";
//
//       // Non-square uniform tensor
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  0UL );
//          checkNonZeros( mat,  0UL, 0UL );
//          checkNonZeros( mat,  1UL, 0UL );
//          checkNonZeros( mat,  2UL, 0UL );
//
//          if( isSymmetric( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSymmetric evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 0UL );
//          checkNonZeros( mat, 0UL, 0UL );
//          checkNonZeros( mat, 1UL, 0UL );
//          checkNonZeros( mat, 2UL, 0UL );
//
//          if( isSymmetric( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSymmetric evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (non-default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL, 2 );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isSymmetric( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSymmetric evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isHermitian() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isHermitian() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void UniformTest::testIsHermitian()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isHermitian()";
//
//       // Non-square uniform tensor
//       {
//          blaze::UniformTensor<cplx> mat( 3UL, 5UL );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  0UL );
//          checkNonZeros( mat,  0UL, 0UL );
//          checkNonZeros( mat,  1UL, 0UL );
//          checkNonZeros( mat,  2UL, 0UL );
//
//          if( isHermitian( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isHermitian evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (real elements)
//       {
//          blaze::UniformTensor<cplx> mat( 3UL, 3UL, cplx(1,0) );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isHermitian( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isHermitian evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (complex elements)
//       {
//          blaze::UniformTensor<cplx> mat( 3UL, 3UL, cplx(1,1) );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isHermitian( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isHermitian evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUniform() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUniform() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void UniformTest::testIsUniform()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isUniform()";
//
//       // Rectangular uniform tensor (default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  0UL );
//          checkNonZeros( mat,  0UL, 0UL );
//          checkNonZeros( mat,  1UL, 0UL );
//          checkNonZeros( mat,  2UL, 0UL );
//
//          if( isUniform( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUniform evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Rectangular uniform tensor (non-default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL, 2 );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat, 15UL );
//          checkNonZeros( mat,  0UL, 5UL );
//          checkNonZeros( mat,  1UL, 5UL );
//          checkNonZeros( mat,  2UL, 5UL );
//
//          if( isUniform( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUniform evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 0UL );
//          checkNonZeros( mat, 0UL, 0UL );
//          checkNonZeros( mat, 1UL, 0UL );
//          checkNonZeros( mat, 2UL, 0UL );
//
//          if( isUniform( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUniform evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (non-default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL, 2 );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isUniform( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUniform evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isZero() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isZero() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void UniformTest::testIsZero()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isZero()";
//
//       // Non-square uniform tensor (default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  0UL );
//          checkNonZeros( mat,  0UL, 0UL );
//          checkNonZeros( mat,  1UL, 0UL );
//          checkNonZeros( mat,  2UL, 0UL );
//
//          if( isZero( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isZero evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Non-square uniform tensor (non-default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL, 2 );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat, 15UL );
//          checkNonZeros( mat,  0UL, 5UL );
//          checkNonZeros( mat,  1UL, 5UL );
//          checkNonZeros( mat,  2UL, 5UL );
//
//          if( isZero( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isZero evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 0UL );
//          checkNonZeros( mat, 0UL, 0UL );
//          checkNonZeros( mat, 1UL, 0UL );
//          checkNonZeros( mat, 2UL, 0UL );
//
//          if( isZero( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isZero evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (non-default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL, 2 );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isZero( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isZero evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isLower() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isLower() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void UniformTest::testIsLower()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isLower()";
//
//       // Non-square uniform tensor
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  0UL );
//          checkNonZeros( mat,  0UL, 0UL );
//          checkNonZeros( mat,  1UL, 0UL );
//          checkNonZeros( mat,  2UL, 0UL );
//
//          if( isLower( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isLower evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 0UL );
//          checkNonZeros( mat, 0UL, 0UL );
//          checkNonZeros( mat, 1UL, 0UL );
//          checkNonZeros( mat, 2UL, 0UL );
//
//          if( isLower( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isLower evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (non-default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL, 2 );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isLower( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isLower evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUniLower() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUniLower() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void UniformTest::testIsUniLower()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isUniLower()";
//
//       // Non-square uniform tensor
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  0UL );
//          checkNonZeros( mat,  0UL, 0UL );
//          checkNonZeros( mat,  1UL, 0UL );
//          checkNonZeros( mat,  2UL, 0UL );
//
//          if( isUniLower( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUniLower evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL, 2 );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isUniLower( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUniLower evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Identity tensor
//       {
//          blaze::UniformTensor<int> mat( 1UL, 1UL, 1 );
//
//          checkRows    ( mat, 1UL );
//          checkColumns ( mat, 1UL );
//          checkCapacity( mat, 1UL );
//          checkNonZeros( mat, 1UL );
//          checkNonZeros( mat, 0UL, 1UL );
//
//          if( isUniLower( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUniLower evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isStrictlyLower() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isStrictlyLower() function for dense matrices. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void UniformTest::testIsStrictlyLower()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isStrictlyLower()";
//
//       // Non-square uniform tensor
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  0UL );
//          checkNonZeros( mat,  0UL, 0UL );
//          checkNonZeros( mat,  1UL, 0UL );
//          checkNonZeros( mat,  2UL, 0UL );
//
//          if( isStrictlyLower( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isStrictlyLower evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 0UL );
//          checkNonZeros( mat, 0UL, 0UL );
//          checkNonZeros( mat, 1UL, 0UL );
//          checkNonZeros( mat, 2UL, 0UL );
//
//          if( isStrictlyLower( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isStrictlyLower evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (non-default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL, 2 );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isStrictlyLower( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isStrictlyLower evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUpper() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUpper() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void UniformTest::testIsUpper()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isUpper()";
//
//       // Non-square uniform tensor
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  0UL );
//          checkNonZeros( mat,  0UL, 0UL );
//          checkNonZeros( mat,  1UL, 0UL );
//          checkNonZeros( mat,  2UL, 0UL );
//
//          if( isUpper( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUpper evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 0UL );
//          checkNonZeros( mat, 0UL, 0UL );
//          checkNonZeros( mat, 1UL, 0UL );
//          checkNonZeros( mat, 2UL, 0UL );
//
//          if( isUpper( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUpper evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (non-default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL, 2 );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isUpper( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUpper evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUniUpper() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUniUpper() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void UniformTest::testIsUniUpper()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isUniUpper()";
//
//       // Non-square uniform tensor
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  0UL );
//          checkNonZeros( mat,  0UL, 0UL );
//          checkNonZeros( mat,  1UL, 0UL );
//          checkNonZeros( mat,  2UL, 0UL );
//
//          if( isUniUpper( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUniUpper evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL, 2 );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isUniUpper( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUniUpper evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Identity tensor
//       {
//          blaze::UniformTensor<int> mat( 1UL, 1UL, 1 );
//
//          checkRows    ( mat, 1UL );
//          checkColumns ( mat, 1UL );
//          checkCapacity( mat, 1UL );
//          checkNonZeros( mat, 1UL );
//          checkNonZeros( mat, 0UL, 1UL );
//
//          if( isUniUpper( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isUniUpper evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isStrictlyUpper() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isStrictlyUpper() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void UniformTest::testIsStrictlyUpper()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isStrictlyUpper()";
//
//       // Non-square uniform tensor
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  0UL );
//          checkNonZeros( mat,  0UL, 0UL );
//          checkNonZeros( mat,  1UL, 0UL );
//          checkNonZeros( mat,  2UL, 0UL );
//
//          if( isStrictlyUpper( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isStrictlyUpper evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 0UL );
//          checkNonZeros( mat, 0UL, 0UL );
//          checkNonZeros( mat, 1UL, 0UL );
//          checkNonZeros( mat, 2UL, 0UL );
//
//          if( isStrictlyUpper( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isStrictlyUpper evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (non-default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL, 2 );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isStrictlyUpper( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isStrictlyUpper evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDiagonal() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDiagonal() function for dense matrices. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void UniformTest::testIsDiagonal()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isDiagonal()";
//
//       // Non-square uniform tensor
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  0UL );
//          checkNonZeros( mat,  0UL, 0UL );
//          checkNonZeros( mat,  1UL, 0UL );
//          checkNonZeros( mat,  2UL, 0UL );
//
//          if( isDiagonal( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isDiagonal evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 0UL );
//          checkNonZeros( mat, 0UL, 0UL );
//          checkNonZeros( mat, 1UL, 0UL );
//          checkNonZeros( mat, 2UL, 0UL );
//
//          if( isDiagonal( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isDiagonal evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor (non-default values)
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL, 2 );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isDiagonal( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isDiagonal evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isIdentity() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isIdentity() function for dense matrices. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void UniformTest::testIsIdentity()
{
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isIdentity()";
//
//       // Non-square uniform tensor
//       {
//          blaze::UniformTensor<int> mat( 3UL, 5UL );
//
//          checkRows    ( mat,  3UL );
//          checkColumns ( mat,  5UL );
//          checkCapacity( mat, 15UL );
//          checkNonZeros( mat,  0UL );
//          checkNonZeros( mat,  0UL, 0UL );
//          checkNonZeros( mat,  1UL, 0UL );
//          checkNonZeros( mat,  2UL, 0UL );
//
//          if( isIdentity( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isIdentity evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Square uniform tensor
//       {
//          blaze::UniformTensor<int> mat( 3UL, 3UL, 2 );
//
//          checkRows    ( mat, 3UL );
//          checkColumns ( mat, 3UL );
//          checkCapacity( mat, 9UL );
//          checkNonZeros( mat, 9UL );
//          checkNonZeros( mat, 0UL, 3UL );
//          checkNonZeros( mat, 1UL, 3UL );
//          checkNonZeros( mat, 2UL, 3UL );
//
//          if( isIdentity( mat ) != false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isIdentity evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // Identity tensor
//       {
//          blaze::UniformTensor<int> mat( 1UL, 1UL, 1 );
//
//          checkRows    ( mat, 1UL );
//          checkColumns ( mat, 1UL );
//          checkCapacity( mat, 1UL );
//          checkNonZeros( mat, 1UL );
//          checkNonZeros( mat, 0UL, 1UL );
//
//          if( isIdentity( mat ) != true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isIdentity evaluation\n"
//                 << " Details:\n"
//                 << "   Tensor:\n" << mat << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
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

//*************************************************************************************************
int main()
{
   std::cout << "   Running uniform DenseTensor operation test..." << std::endl;

   try
   {
      RUN_DENSETENSOR_UNIFORM_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during uniform DenseTensor operation test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
