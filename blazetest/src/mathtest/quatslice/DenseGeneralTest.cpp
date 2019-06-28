//=================================================================================================
/*!
//  \file src/mathtest/quatslice/DenseGeneralTest.cpp
//  \brief Source file for the QuatSlice dense general test
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018-2019 Hartmut Kaiser - All Rights Reserved
//  Copyright (C) 2019 Bita Hasheminezhad - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other quaterials
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

#include <blaze/math/CustomMatrix.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/Views.h>
#include <blaze/system/Platform.h>
#include <blaze/util/policies/Deallocate.h>

#include <blaze_tensor/math/CustomTensor.h>
#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/Views.h>

#include <blazetest/mathtest/quatslice/DenseGeneralTest.h>


namespace blazetest {

namespace mathtest {

namespace quatslice {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the QuatSlice dense general test.
//
// \exception std::runtime_error Operation error detected.
*/
DenseGeneralTest::DenseGeneralTest()
   //: quat_ ( 3UL, 2UL, 5UL, 4UL )
{
   testConstructors();
   //testAssignment();
   //testAddAssign();
   //testSubAssign();
   //testMultAssign();
   //testSchurAssign();
   //testScaling();
   //testFunctionCall();
   //testAt();
   //testIterator();
   //testNonZeros();
   //testReset();
   //testClear();
   //testIsDefault();
   //testIsSame();
   //testSubquaternion();
   //testRow();
   //testRows();
   //testColumn();
   //testColumns();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the QuatSlice constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the QuatSlice specialization. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testConstructors()
{
   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   //{
   //   test_ = "QuatSlice constructor (0x0)";

   //   AT quat;

   //   // 0th quaternion quatslice
   //   try {
   //      blaze::quatslice( quat, 0UL );
   //   }
   //   catch( std::invalid_argument& ) {}
   //}

   //{
   //   test_ = "QuatSlice constructor (2x0)";

   //   AT quat( 2UL, 2UL, 0UL );

   //   // 0th quaternion quatslice
   //   {
   //      RT quatslice0 = blaze::quatslice( quat, 0UL );

   //      checkRows    ( quatslice0, 2UL );
   //      checkColumns ( quatslice0, 0UL );
   //      checkCapacity( quatslice0, 0UL );
   //      checkNonZeros( quatslice0, 0UL );
   //   }

   //   // 1st quaternion quatslice
   //   {
   //      RT quatslice1 = blaze::quatslice( quat, 1UL );

   //      checkRows    ( quatslice1, 2UL );
   //      checkColumns ( quatslice1, 0UL );
   //      checkCapacity( quatslice1, 0UL );
   //      checkNonZeros( quatslice1, 0UL );
   //   }

   //   // 2nd quaternion quatslice
   //   try {
   //      blaze::quatslice( quat, 2UL );
   //   }
   //   catch( std::invalid_argument& ) {}
   //}

   //{
   //   test_ = "QuatSlice constructor (5x4)";

   //   initialize();

   //   // 0th quaternion quatslice
   //   {
   //      RT quatslice0 = blaze::quatslice( quat_, 0UL );

   //      checkRows    ( quatslice0, 5UL );
   //      checkColumns ( quatslice0, 4UL );
   //      checkCapacity( quatslice0, 20UL );
   //      checkNonZeros( quatslice0, 10UL );

   //      if( quatslice0(0,0) !=  0 || quatslice0(0,1) !=  0 || quatslice0(0,2) !=  0 || quatslice0(0,3) !=  0 ||
   //          quatslice0(1,0) !=  0 || quatslice0(1,1) !=  1 || quatslice0(1,2) !=  0 || quatslice0(1,3) !=  0 ||
   //          quatslice0(2,0) != -2 || quatslice0(2,1) !=  0 || quatslice0(2,2) != -3 || quatslice0(2,3) !=  0 ||
   //          quatslice0(3,0) !=  0 || quatslice0(3,1) !=  4 || quatslice0(3,2) !=  5 || quatslice0(3,3) != -6 ||
   //          quatslice0(4,0) !=  7 || quatslice0(4,1) != -8 || quatslice0(4,2) !=  9 || quatslice0(4,3) != 10 ) {
   //         std::ostringstream oss;
   //         oss << " Test: " << test_ << "\n"
   //             << " Error: Setup of 0th dense quatslice failed\n"
   //             << " Details:\n"
   //             << "   Result:\n" << quatslice0 << "\n"
   //             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
   //         throw std::runtime_error( oss.str() );
   //      }
   //   }

   //   // 1st quaternion quatslice
   //   {
   //      RT quatslice1 = blaze::quatslice( quat_, 1UL );

   //      checkRows    ( quatslice1, 5UL );
   //      checkColumns ( quatslice1, 4UL );
   //      checkCapacity( quatslice1, 20UL );
   //      checkNonZeros( quatslice1, 10UL );

   //      if( quatslice1(0,0) !=  0 || quatslice1(0,1) !=  0 || quatslice1(0,2) !=  0 || quatslice1(0,3) !=  0 ||
   //          quatslice1(1,0) !=  0 || quatslice1(1,1) !=  1 || quatslice1(1,2) !=  0 || quatslice1(1,3) !=  0 ||
   //          quatslice1(2,0) != -2 || quatslice1(2,1) !=  0 || quatslice1(2,2) != -3 || quatslice1(2,3) !=  0 ||
   //          quatslice1(3,0) !=  0 || quatslice1(3,1) !=  4 || quatslice1(3,2) !=  5 || quatslice1(3,3) != -6 ||
   //          quatslice1(4,0) !=  7 || quatslice1(4,1) != -8 || quatslice1(4,2) !=  9 || quatslice1(4,3) != 10 ) {
   //         std::ostringstream oss;
   //         oss << " Test: " << test_ << "\n"
   //             << " Error: Setup of 1st dense quatslice failed\n"
   //             << " Details:\n"
   //             << "   Result:\n" << quatslice1 << "\n"
   //             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
   //         throw std::runtime_error( oss.str() );
   //      }
   //   }

   //   // 2nd quaternion quatslice
   //   try {
   //      RT quatslice2 = blaze::quatslice( quat_, 2UL );

   //      std::ostringstream oss;
   //      oss << " Test: " << test_ << "\n"
   //          << " Error: Out-of-bound quat access succeeded\n"
   //          << " Details:\n"
   //          << "   Result:\n" << quatslice2 << "\n";
   //      throw std::runtime_error( oss.str() );
   //   }
   //   catch( std::invalid_argument& ) {}
   //}
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Test of the QuatSlice assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the QuatSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
//void DenseGeneralTest::testAssignment()
//{
//   //=====================================================================================
//   // homogeneous assignment
//   //=====================================================================================
//
//   {
//      test_ = "QuatSlice homogeneous assignment";
//
//      initialize();
//
//      RT quatslice1 = blaze::quatslice( quat_, 1UL );
//      quatslice1 = 8;
//
//
//      checkRows    ( quatslice1, 5UL );
//      checkColumns ( quatslice1, 4UL );
//      checkCapacity( quatslice1, 20UL );
//      checkNonZeros( quatslice1, 20UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 30UL );
//
//      if( quatslice1(0,0) != 8 || quatslice1(0,1) != 8 || quatslice1(0,2) != 8 || quatslice1(0,3) != 8 ||
//          quatslice1(1,0) != 8 || quatslice1(1,1) != 8 || quatslice1(1,2) != 8 || quatslice1(1,3) != 8 ||
//          quatslice1(2,0) != 8 || quatslice1(2,1) != 8 || quatslice1(2,2) != 8 || quatslice1(2,3) != 8 ||
//          quatslice1(3,0) != 8 || quatslice1(3,1) != 8 || quatslice1(3,2) != 8 || quatslice1(3,3) != 8 ||
//          quatslice1(4,0) != 8 || quatslice1(4,1) != 8 || quatslice1(4,2) != 8 || quatslice1(4,3) != 8 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice1 << "\n"
//             << "   Expected result:\n(( 8 8 8 8 )\n( 8 8 8 8 )\n( 8 8 8 8 )\n( 8 8 8 8 )\n( 8 8 8 8 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  8 || quat_(1,0,1) !=  8 || quat_(1,0,2) !=  8 || quat_(1,0,3) !=  8 ||
//          quat_(1,1,0) !=  8 || quat_(1,1,1) !=  8 || quat_(1,1,2) !=  8 || quat_(1,1,3) !=  8 ||
//          quat_(1,2,0) !=  8 || quat_(1,2,1) !=  8 || quat_(1,2,2) !=  8 || quat_(1,2,3) !=  8 ||
//          quat_(1,3,0) !=  8 || quat_(1,3,1) !=  8 || quat_(1,3,2) !=  8 || quat_(1,3,3) !=  8 ||
//          quat_(1,4,0) !=  8 || quat_(1,4,1) !=  8 || quat_(1,4,2) !=  8 || quat_(1,4,3) !=  8 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0  0  0  0 )\n"
//                                     " (  0  1  0  0 )\n"
//                                     " ( -2  0 -3  0 )\n"
//                                     " (  0  4  5 -6 )\n"
//                                     " (  7 -8  9 10 ))\n"
//                                     "((  8  8  8  8 )\n"
//                                     " (  8  8  8  8 )\n"
//                                     " (  8  8  8  8 )\n"
//                                     " (  8  8  8  8 )\n"
//                                     " (  8  8  8  8 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // list assignment
//   //=====================================================================================
//
//   {
//      test_ = "initializer list assignment (complete list)";
//
//      initialize();
//
//      RT quatslice3 = blaze::quatslice( quat_, 1UL );
//      quatslice3 = {
//          {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}
//      };
//
//      checkRows    ( quatslice3, 5UL );
//      checkColumns ( quatslice3, 4UL );
//      checkCapacity( quatslice3, 20UL );
//      checkNonZeros( quatslice3, 20UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 30UL );
//
//      if( quatslice3(0,0) != 1 || quatslice3(0,1) != 2 || quatslice3(0,2) != 3 || quatslice3(0,3) != 4 ||
//          quatslice3(1,0) != 1 || quatslice3(1,1) != 2 || quatslice3(1,2) != 3 || quatslice3(1,3) != 4 ||
//          quatslice3(2,0) != 1 || quatslice3(2,1) != 2 || quatslice3(2,2) != 3 || quatslice3(2,3) != 4 ||
//          quatslice3(3,0) != 1 || quatslice3(3,1) != 2 || quatslice3(3,2) != 3 || quatslice3(3,3) != 4 ||
//          quatslice3(4,0) != 1 || quatslice3(4,1) != 2 || quatslice3(4,2) != 3 || quatslice3(4,3) != 4 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice3 << "\n"
//             << "   Expected result:\n(( 1 2 3 4 )\n( 1 2 3 4 )\n( 1 2 3 4 )\n( 1 2 3 4 )\n( 1 2 3 4 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  1 || quat_(1,0,1) !=  2 || quat_(1,0,2) !=  3 || quat_(1,0,3) !=  4 ||
//          quat_(1,1,0) !=  1 || quat_(1,1,1) !=  2 || quat_(1,1,2) !=  3 || quat_(1,1,3) !=  4 ||
//          quat_(1,2,0) !=  1 || quat_(1,2,1) !=  2 || quat_(1,2,2) !=  3 || quat_(1,2,3) !=  4 ||
//          quat_(1,3,0) !=  1 || quat_(1,3,1) !=  2 || quat_(1,3,2) !=  3 || quat_(1,3,3) !=  4 ||
//          quat_(1,4,0) !=  1 || quat_(1,4,1) !=  2 || quat_(1,4,2) !=  3 || quat_(1,4,3) !=  4 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0  0  0  0 )\n"
//                                     " (  0  1  0  0 )\n"
//                                     " ( -2  0 -3  0 )\n"
//                                     " (  0  4  5 -6 )\n"
//                                     " (  7 -8  9 10 ))\n"
//                                     "((  1  2  3  4 )\n"
//                                     " (  1  2  3  4 )\n"
//                                     " (  1  2  3  4 )\n"
//                                     " (  1  2  3  4 )\n"
//                                     " (  1  2  3  4 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "initializer list assignment (incomplete list)";
//
//      initialize();
//
//      RT quatslice3 = blaze::quatslice( quat_, 1UL );
//      quatslice3 = {{1, 2}, {1, 2}, {1, 2}, {1, 2}, {1, 2}};
//
//      checkRows    ( quatslice3, 5UL );
//      checkColumns ( quatslice3, 4UL );
//      checkCapacity( quatslice3, 20UL );
//      checkNonZeros( quatslice3, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice3(0,0) != 1 || quatslice3(0,1) != 2 || quatslice3(0,2) != 0 || quatslice3(0,3) != 0 ||
//          quatslice3(1,0) != 1 || quatslice3(1,1) != 2 || quatslice3(1,2) != 0 || quatslice3(1,3) != 0 ||
//          quatslice3(2,0) != 1 || quatslice3(2,1) != 2 || quatslice3(2,2) != 0 || quatslice3(2,3) != 0 ||
//          quatslice3(3,0) != 1 || quatslice3(3,1) != 2 || quatslice3(3,2) != 0 || quatslice3(3,3) != 0 ||
//          quatslice3(4,0) != 1 || quatslice3(4,1) != 2 || quatslice3(4,2) != 0 || quatslice3(4,3) != 0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice3 << "\n"
//             << "   Expected result:\n(( 1 2 0 0 )\n( 1 2 0 0 )\n( 1 2 0 0 )\n( 1 2 0 0 )\n( 1 2 0 0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  1 || quat_(1,0,1) !=  2 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=  1 || quat_(1,1,1) !=  2 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) !=  1 || quat_(1,2,1) !=  2 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  1 || quat_(1,3,1) !=  2 || quat_(1,3,2) !=  0 || quat_(1,3,3) !=  0 ||
//          quat_(1,4,0) !=  1 || quat_(1,4,1) !=  2 || quat_(1,4,2) !=  0 || quat_(1,4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0  0  0  0 )\n"
//                                     " (  0  1  0  0 )\n"
//                                     " ( -2  0 -3  0 )\n"
//                                     " (  0  4  5 -6 )\n"
//                                     " (  7 -8  9 10 ))\n"
//                                     "((  1  2  0  0 )\n"
//                                     " (  1  2  0  0 )\n"
//                                     " (  1  2  0  0 )\n"
//                                     " (  1  2  0  0 )\n"
//                                     " (  1  2  0  0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // copy assignment
//   //=====================================================================================
//
//   {
//      test_ = "QuatSlice copy assignment";
//
//      initialize();
//
//      RT quatslice1 = blaze::quatslice( quat_, 0UL );
//      quatslice1 = 0;
//      quatslice1 = blaze::quatslice( quat_, 1UL );
//
//      checkRows    ( quatslice1, 5UL );
//      checkColumns ( quatslice1, 4UL );
//      checkCapacity( quatslice1, 20UL );
//      checkNonZeros( quatslice1, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice1(0,0) !=  0 || quatslice1(0,1) !=  0 || quatslice1(0,2) !=  0 || quatslice1(0,3) !=  0 ||
//          quatslice1(1,0) !=  0 || quatslice1(1,1) !=  1 || quatslice1(1,2) !=  0 || quatslice1(1,3) !=  0 ||
//          quatslice1(2,0) != -2 || quatslice1(2,1) !=  0 || quatslice1(2,2) != -3 || quatslice1(2,3) !=  0 ||
//          quatslice1(3,0) !=  0 || quatslice1(3,1) !=  4 || quatslice1(3,2) !=  5 || quatslice1(3,3) != -6 ||
//          quatslice1(4,0) !=  7 || quatslice1(4,1) != -8 || quatslice1(4,2) !=  9 || quatslice1(4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice1 << "\n"
//             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) != -2 || quat_(1,2,1) !=  0 || quat_(1,2,2) != -3 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) !=  7 || quat_(1,4,1) != -8 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0  0  0  0 )\n"
//                                     " (  0  1  0  0 )\n"
//                                     " ( -2  0 -3  0 )\n"
//                                     " (  0  4  5 -6 )\n"
//                                     " (  7 -8  9 10 ))\n"
//                                     "((  0  0  0  0 )\n"
//                                     " (  0  1  0  0 )\n"
//                                     " ( -2  0 -3  0 )\n"
//                                     " (  0  4  5 -6 )\n"
//                                     " (  7 -8  9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // dense quaternion assignment
//   //=====================================================================================
//
//   {
//      test_ = "dense quaternion assignment (mixed type)";
//
//      initialize();
//
//      RT quatslice1 = blaze::quatslice( quat_, 1UL );
//
//      blaze::DynamicMatrix<int, blaze::rowMajor> m1;
//      m1 = {{0, 8, 0, 9}, {0}, {0}, {0}, {0}};
//
//      quatslice1 = m1;
//
//      checkRows    ( quatslice1, 5UL );
//      checkColumns ( quatslice1, 4UL );
//      checkCapacity( quatslice1, 20UL );
//      checkNonZeros( quatslice1, 2UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 12UL );
//
//      if( quatslice1(0,0) !=  0 || quatslice1(0,1) !=  8 || quatslice1(0,2) !=  0 || quatslice1(0,3) !=  9 ||
//          quatslice1(1,0) !=  0 || quatslice1(1,1) !=  0 || quatslice1(1,2) !=  0 || quatslice1(1,3) !=  0 ||
//          quatslice1(2,0) !=  0 || quatslice1(2,1) !=  0 || quatslice1(2,2) !=  0 || quatslice1(2,3) !=  0 ||
//          quatslice1(3,0) !=  0 || quatslice1(3,1) !=  0 || quatslice1(3,2) !=  0 || quatslice1(3,3) !=  0 ||
//          quatslice1(4,0) !=  0 || quatslice1(4,1) !=  0 || quatslice1(4,2) !=  0 || quatslice1(4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice1 << "\n"
//             << "   Expected result:\n(( 0 8 0 9 )\n(0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  8 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  9 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  0 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) !=  0 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  0 || quat_(1,3,2) !=  0 || quat_(1,3,3) !=  0 ||
//          quat_(1,4,0) !=  0 || quat_(1,4,1) !=  0 || quat_(1,4,2) !=  0 || quat_(1,4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0  0  0  0 )\n"
//                                     " (  0  1  0  0 )\n"
//                                     " ( -2  0 -3  0 )\n"
//                                     " (  0  4  5 -6 )\n"
//                                     " (  7 -8  9 10 ))\n"
//                                     "((  0  9  0  9 )\n"
//                                     " (  0  0  0  0 )\n"
//                                     " (  0  0  0  0 )\n"
//                                     " (  0  0  0  0 )\n"
//                                     " (  0  0  0  0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "dense quaternion assignment (mixed type)";
//
//      initialize();
//
//      RT quatslice1 = blaze::quatslice( quat_, 1UL );
//
//      blaze::DynamicMatrix<int, blaze::columnMajor> m1;
//      m1 = {{0, 8, 0, 9}, {0}, {0}, {0}, {0}};
//
//      quatslice1 = m1;
//
//      checkRows    ( quatslice1, 5UL );
//      checkColumns ( quatslice1, 4UL );
//      checkCapacity( quatslice1, 20UL );
//      checkNonZeros( quatslice1, 2UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 12UL );
//
//      if( quatslice1(0,0) !=  0 || quatslice1(0,1) !=  8 || quatslice1(0,2) !=  0 || quatslice1(0,3) !=  9 ||
//          quatslice1(1,0) !=  0 || quatslice1(1,1) !=  0 || quatslice1(1,2) !=  0 || quatslice1(1,3) !=  0 ||
//          quatslice1(2,0) !=  0 || quatslice1(2,1) !=  0 || quatslice1(2,2) !=  0 || quatslice1(2,3) !=  0 ||
//          quatslice1(3,0) !=  0 || quatslice1(3,1) !=  0 || quatslice1(3,2) !=  0 || quatslice1(3,3) !=  0 ||
//          quatslice1(4,0) !=  0 || quatslice1(4,1) !=  0 || quatslice1(4,2) !=  0 || quatslice1(4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice1 << "\n"
//             << "   Expected result:\n(( 0 8 0 9 )\n(0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  8 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  9 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  0 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) !=  0 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  0 || quat_(1,3,2) !=  0 || quat_(1,3,3) !=  0 ||
//          quat_(1,4,0) !=  0 || quat_(1,4,1) !=  0 || quat_(1,4,2) !=  0 || quat_(1,4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0  0  0  0 )\n"
//                                     " (  0  1  0  0 )\n"
//                                     " ( -2  0 -3  0 )\n"
//                                     " (  0  4  5 -6 )\n"
//                                     " (  7 -8  9 10 ))\n"
//                                     "((  0  9  0  9 )\n"
//                                     " (  0  0  0  0 )\n"
//                                     " (  0  0  0  0 )\n"
//                                     " (  0  0  0  0 )\n"
//                                     " (  0  0  0  0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "dense quaternion assignment (aligned/padded)";
//
//      using blaze::aligned;
//      using blaze::padded;
//      using blaze::rowMajor;
//
//      initialize();
//
//      RT quatslice1 = blaze::quatslice( quat_, 1UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 80UL ) );
//      AlignedPadded m1( memory.get(), 5UL, 4UL, 16UL );
//      m1 = 0;
//      m1(0,0) = 0;
//      m1(0,1) = 8;
//      m1(0,2) = 0;
//      m1(0,3) = 9;
//
//      quatslice1 = m1;
//
//      checkRows    ( quatslice1, 5UL );
//      checkColumns ( quatslice1, 4UL );
//      checkCapacity( quatslice1, 20UL );
//      checkNonZeros( quatslice1, 2UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 12UL );
//
//      if( quatslice1(0,0) !=  0 || quatslice1(0,1) !=  8 || quatslice1(0,2) !=  0 || quatslice1(0,3) !=  9 ||
//          quatslice1(1,0) !=  0 || quatslice1(1,1) !=  0 || quatslice1(1,2) !=  0 || quatslice1(1,3) !=  0 ||
//          quatslice1(2,0) !=  0 || quatslice1(2,1) !=  0 || quatslice1(2,2) !=  0 || quatslice1(2,3) !=  0 ||
//          quatslice1(3,0) !=  0 || quatslice1(3,1) !=  0 || quatslice1(3,2) !=  0 || quatslice1(3,3) !=  0 ||
//          quatslice1(4,0) !=  0 || quatslice1(4,1) !=  0 || quatslice1(4,2) !=  0 || quatslice1(4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice1 << "\n"
//             << "   Expected result:\n(( 0 8 0 9 )\n(0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  8 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  9 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  0 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) !=  0 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  0 || quat_(1,3,2) !=  0 || quat_(1,3,3) !=  0 ||
//          quat_(1,4,0) !=  0 || quat_(1,4,1) !=  0 || quat_(1,4,2) !=  0 || quat_(1,4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0  0  0  0 )\n"
//                                     " (  0  1  0  0 )\n"
//                                     " ( -2  0 -3  0 )\n"
//                                     " (  0  4  5 -6 )\n"
//                                     " (  7 -8  9 10 ))\n"
//                                     "((  0  9  0  9 )\n"
//                                     " (  0  0  0  0 )\n"
//                                     " (  0  0  0  0 )\n"
//                                     " (  0  0  0  0 )\n"
//                                     " (  0  0  0  0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "dense quaternion assignment (unaligned/unpadded)";
//
//      using blaze::unaligned;
//      using blaze::unpadded;
//      using blaze::rowMajor;
//
//      initialize();
//
//      RT quatslice1 = blaze::quatslice( quat_, 1UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[21] );
//      UnalignedUnpadded m1( memory.get()+1UL, 5UL, 4UL );
//      m1 = 0;
//      m1(0,0) = 0;
//      m1(0,1) = 8;
//      m1(0,2) = 0;
//      m1(0,3) = 9;
//
//      quatslice1 = m1;
//
//      checkRows    ( quatslice1, 5UL );
//      checkColumns ( quatslice1, 4UL );
//      checkCapacity( quatslice1, 20UL );
//      checkNonZeros( quatslice1, 2UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 12UL );
//
//      if( quatslice1(0,0) !=  0 || quatslice1(0,1) !=  8 || quatslice1(0,2) !=  0 || quatslice1(0,3) !=  9 ||
//          quatslice1(1,0) !=  0 || quatslice1(1,1) !=  0 || quatslice1(1,2) !=  0 || quatslice1(1,3) !=  0 ||
//          quatslice1(2,0) !=  0 || quatslice1(2,1) !=  0 || quatslice1(2,2) !=  0 || quatslice1(2,3) !=  0 ||
//          quatslice1(3,0) !=  0 || quatslice1(3,1) !=  0 || quatslice1(3,2) !=  0 || quatslice1(3,3) !=  0 ||
//          quatslice1(4,0) !=  0 || quatslice1(4,1) !=  0 || quatslice1(4,2) !=  0 || quatslice1(4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice1 << "\n"
//             << "   Expected result:\n(( 0 8 0 9 )\n(0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  8 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  9 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  0 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) !=  0 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  0 || quat_(1,3,2) !=  0 || quat_(1,3,3) !=  0 ||
//          quat_(1,4,0) !=  0 || quat_(1,4,1) !=  0 || quat_(1,4,2) !=  0 || quat_(1,4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0  0  0  0 )\n"
//                                     " (  0  1  0  0 )\n"
//                                     " ( -2  0 -3  0 )\n"
//                                     " (  0  4  5 -6 )\n"
//                                     " (  7 -8  9 10 ))\n"
//                                     "((  0  9  0  9 )\n"
//                                     " (  0  0  0  0 )\n"
//                                     " (  0  0  0  0 )\n"
//                                     " (  0  0  0  0 )\n"
//                                     " (  0  0  0  0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the QuatSlice addition assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the addition assignment operators of the QuatSlice specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testAddAssign()
//{
//   //=====================================================================================
//   // QuatSlice addition assignment
//   //=====================================================================================
//
//   {
//      test_ = "QuatSlice addition assignment";
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//      quatslice2 += blaze::quatslice( quat_, 0UL );
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   2 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -4 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -6 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=   8 || quatslice2(3,2) != 10 || quatslice2(3,3) != -12 ||
//          quatslice2(4,0) != 14 || quatslice2(4,1) != -16 || quatslice2(4,2) != 18 || quatslice2(4,3) !=  20 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   2 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) != -4 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -6 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=   8 || quat_(1,3,2) != 10 || quat_(1,3,3) != -12 ||
//          quat_(1,4,0) != 14 || quat_(1,4,1) != -16 || quat_(1,4,2) != 18 || quat_(1,4,3) !=  20 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   2   0   0 )\n"
//                                     " ( -4   0  -6   0 )\n"
//                                     " (  0   8  10 -12 )\n"
//                                     " ( 14 -16  18  20 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // dense quaternion addition assignment
//   //=====================================================================================
//
//   {
//      test_ = "dense quaternion addition assignment (mixed type)";
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//
//      const blaze::DynamicMatrix<short, blaze::rowMajor> vec{{0, 0, 0, 0},
//                                                             {0, 1, 0, 0},
//                                                             {-2, 0, -3, 0},
//                                                             {0, 4, 5, -6},
//                                                             {7, -8, 9, 10}};
//
//      quatslice2 += vec;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   2 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -4 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -6 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=   8 || quatslice2(3,2) != 10 || quatslice2(3,3) != -12 ||
//          quatslice2(4,0) != 14 || quatslice2(4,1) != -16 || quatslice2(4,2) != 18 || quatslice2(4,3) !=  20 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   2 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) != -4 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -6 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=   8 || quat_(1,3,2) != 10 || quat_(1,3,3) != -12 ||
//          quat_(1,4,0) != 14 || quat_(1,4,1) != -16 || quat_(1,4,2) != 18 || quat_(1,4,3) !=  20 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   2   0   0 )\n"
//                                     " ( -4   0  -6   0 )\n"
//                                     " (  0   8  10 -12 )\n"
//                                     " ( 14 -16  18  20 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "dense quaternion addition assignment (aligned/padded)";
//
//      using blaze::aligned;
//      using blaze::padded;
//      using blaze::rowMajor;
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 80UL ) );
//      AlignedPadded m( memory.get(), 5UL, 4UL, 16UL );
//      m(0,0) =  0;
//      m(0,1) =  0;
//      m(0,2) =  0;
//      m(0,3) =  0;
//      m(1,0) =  0;
//      m(1,1) =  1;
//      m(1,2) =  0;
//      m(1,3) =  0;
//      m(2,0) = -2;
//      m(2,1) =  0;
//      m(2,2) = -3;
//      m(2,3) =  0;
//      m(3,0) =  0;
//      m(3,1) =  4;
//      m(3,2) =  5;
//      m(3,3) = -6;
//      m(4,0) =  7;
//      m(4,1) = -8;
//      m(4,2) =  9;
//      m(4,3) = 10;
//
//      quatslice2 += m;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   2 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -4 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -6 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=   8 || quatslice2(3,2) != 10 || quatslice2(3,3) != -12 ||
//          quatslice2(4,0) != 14 || quatslice2(4,1) != -16 || quatslice2(4,2) != 18 || quatslice2(4,3) !=  20 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   2 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) != -4 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -6 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=   8 || quat_(1,3,2) != 10 || quat_(1,3,3) != -12 ||
//          quat_(1,4,0) != 14 || quat_(1,4,1) != -16 || quat_(1,4,2) != 18 || quat_(1,4,3) !=  20 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   2   0   0 )\n"
//                                     " ( -4   0  -6   0 )\n"
//                                     " (  0   8  10 -12 )\n"
//                                     " ( 14 -16  18  20 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "dense quaternion addition assignment (unaligned/unpadded)";
//
//      using blaze::unaligned;
//      using blaze::unpadded;
//      using blaze::rowMajor;
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[21] );
//      UnalignedUnpadded m( memory.get()+1UL, 5UL, 4UL );
//      m(0,0) =  0;
//      m(0,1) =  0;
//      m(0,2) =  0;
//      m(0,3) =  0;
//      m(1,0) =  0;
//      m(1,1) =  1;
//      m(1,2) =  0;
//      m(1,3) =  0;
//      m(2,0) = -2;
//      m(2,1) =  0;
//      m(2,2) = -3;
//      m(2,3) =  0;
//      m(3,0) =  0;
//      m(3,1) =  4;
//      m(3,2) =  5;
//      m(3,3) = -6;
//      m(4,0) =  7;
//      m(4,1) = -8;
//      m(4,2) =  9;
//      m(4,3) = 10;
//
//      quatslice2 += m;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   2 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -4 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -6 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=   8 || quatslice2(3,2) != 10 || quatslice2(3,3) != -12 ||
//          quatslice2(4,0) != 14 || quatslice2(4,1) != -16 || quatslice2(4,2) != 18 || quatslice2(4,3) !=  20 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   2 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) != -4 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -6 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=   8 || quat_(1,3,2) != 10 || quat_(1,3,3) != -12 ||
//          quat_(1,4,0) != 14 || quat_(1,4,1) != -16 || quat_(1,4,2) != 18 || quat_(1,4,3) !=  20 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   2   0   0 )\n"
//                                     " ( -4   0  -6   0 )\n"
//                                     " (  0   8  10 -12 )\n"
//                                     " ( 14 -16  18  20 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the QuatSlice subtraction assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the subtraction assignment operators of the QuatSlice
//// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testSubAssign()
//{
//   //=====================================================================================
//   // QuatSlice subtraction assignment
//   //=====================================================================================
//
//   {
//      test_ = "QuatSlice subtraction assignment";
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//      quatslice2 -= blaze::quatslice( quat_, 0UL );
//
//      checkRows    ( quatslice2,  5UL );
//      checkColumns ( quatslice2,  4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2,  0UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 10UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=  0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=  0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=  0 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=  0 ||
//          quatslice2(2,0) !=  0 || quatslice2(2,1) !=  0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=  0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=  0 || quatslice2(3,2) !=  0 || quatslice2(3,3) !=  0 ||
//          quatslice2(4,0) !=  0 || quatslice2(4,1) !=  0 || quatslice2(4,2) !=  0 || quatslice2(4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   0 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) !=  0 || quat_(1,2,1) !=   0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=   0 || quat_(1,3,2) !=  0 || quat_(1,3,3) !=   0 ||
//          quat_(1,4,0) !=  0 || quat_(1,4,1) !=   0 || quat_(1,4,2) !=  0 || quat_(1,4,3) !=   0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   0   0   0 )\n"
//                                     " (  0   0   0   0 )\n"
//                                     " (  0   0   0   0 )\n"
//                                     " (  0   0   0   0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // dense quaternion subtraction assignment
//   //=====================================================================================
//
//   {
//      test_ = "dense quaternion subtraction assignment (mixed type)";
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//
//      const blaze::DynamicMatrix<short, blaze::rowMajor> vec{{0, 0, 0, 0},
//                                                             {0, 1, 0, 0},
//                                                             {-2, 0, -3, 0},
//                                                             {0, 4, 5, -6},
//                                                             {7, -8, 9, 10}};
//
//      quatslice2 -= vec;
//
//      checkRows    ( quatslice2,  5UL );
//      checkColumns ( quatslice2,  4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2,  0UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 10UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=  0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=  0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=  0 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=  0 ||
//          quatslice2(2,0) !=  0 || quatslice2(2,1) !=  0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=  0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=  0 || quatslice2(3,2) !=  0 || quatslice2(3,3) !=  0 ||
//          quatslice2(4,0) !=  0 || quatslice2(4,1) !=  0 || quatslice2(4,2) !=  0 || quatslice2(4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   0 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) !=  0 || quat_(1,2,1) !=   0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=   0 || quat_(1,3,2) !=  0 || quat_(1,3,3) !=   0 ||
//          quat_(1,4,0) !=  0 || quat_(1,4,1) !=   0 || quat_(1,4,2) !=  0 || quat_(1,4,3) !=   0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   0   0   0 )\n"
//                                     " (  0   0   0   0 )\n"
//                                     " (  0   0   0   0 )\n"
//                                     " (  0   0   0   0 ))\n";
//      }
//   }
//
//   {
//      test_ = "dense quaternion subtraction assignment (aligned/padded)";
//
//      using blaze::aligned;
//      using blaze::padded;
//      using blaze::rowMajor;
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 80UL ) );
//      AlignedPadded m( memory.get(), 5UL, 4UL, 16UL );
//      m(0,0) =  0;
//      m(0,1) =  0;
//      m(0,2) =  0;
//      m(0,3) =  0;
//      m(1,0) =  0;
//      m(1,1) =  1;
//      m(1,2) =  0;
//      m(1,3) =  0;
//      m(2,0) = -2;
//      m(2,1) =  0;
//      m(2,2) = -3;
//      m(2,3) =  0;
//      m(3,0) =  0;
//      m(3,1) =  4;
//      m(3,2) =  5;
//      m(3,3) = -6;
//      m(4,0) =  7;
//      m(4,1) = -8;
//      m(4,2) =  9;
//      m(4,3) = 10;
//
//      quatslice2 -= m;
//
//      checkRows    ( quatslice2,  5UL );
//      checkColumns ( quatslice2,  4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2,  0UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 10UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=  0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=  0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=  0 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=  0 ||
//          quatslice2(2,0) !=  0 || quatslice2(2,1) !=  0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=  0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=  0 || quatslice2(3,2) !=  0 || quatslice2(3,3) !=  0 ||
//          quatslice2(4,0) !=  0 || quatslice2(4,1) !=  0 || quatslice2(4,2) !=  0 || quatslice2(4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   0 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) !=  0 || quat_(1,2,1) !=   0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=   0 || quat_(1,3,2) !=  0 || quat_(1,3,3) !=   0 ||
//          quat_(1,4,0) !=  0 || quat_(1,4,1) !=   0 || quat_(1,4,2) !=  0 || quat_(1,4,3) !=   0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   0   0   0 )\n"
//                                     " (  0   0   0   0 )\n"
//                                     " (  0   0   0   0 )\n"
//                                     " (  0   0   0   0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "dense quaternion subtraction assignment (unaligned/unpadded)";
//
//      using blaze::unaligned;
//      using blaze::unpadded;
//      using blaze::rowMajor;
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[21] );
//      UnalignedUnpadded m( memory.get()+1UL, 5UL, 4UL );
//      m(0,0) =  0;
//      m(0,1) =  0;
//      m(0,2) =  0;
//      m(0,3) =  0;
//      m(1,0) =  0;
//      m(1,1) =  1;
//      m(1,2) =  0;
//      m(1,3) =  0;
//      m(2,0) = -2;
//      m(2,1) =  0;
//      m(2,2) = -3;
//      m(2,3) =  0;
//      m(3,0) =  0;
//      m(3,1) =  4;
//      m(3,2) =  5;
//      m(3,3) = -6;
//      m(4,0) =  7;
//      m(4,1) = -8;
//      m(4,2) =  9;
//      m(4,3) = 10;
//
//      quatslice2 -= m;
//
//      checkRows    ( quatslice2,  5UL );
//      checkColumns ( quatslice2,  4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2,  0UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 10UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=  0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=  0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=  0 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=  0 ||
//          quatslice2(2,0) !=  0 || quatslice2(2,1) !=  0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=  0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=  0 || quatslice2(3,2) !=  0 || quatslice2(3,3) !=  0 ||
//          quatslice2(4,0) !=  0 || quatslice2(4,1) !=  0 || quatslice2(4,2) !=  0 || quatslice2(4,3) !=  0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   0 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) !=  0 || quat_(1,2,1) !=   0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=   0 || quat_(1,3,2) !=  0 || quat_(1,3,3) !=   0 ||
//          quat_(1,4,0) !=  0 || quat_(1,4,1) !=   0 || quat_(1,4,2) !=  0 || quat_(1,4,3) !=   0 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   0   0   0 )\n"
//                                     " (  0   0   0   0 )\n"
//                                     " (  0   0   0   0 )\n"
//                                     " (  0   0   0   0 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the QuatSlice multiplication assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the multiplication assignment operators of the QuatSlice
//// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testMultAssign()
//{
//   //=====================================================================================
//   // QuatSlice multiplication assignment
//   //=====================================================================================
//
//   {
//      test_ = "QuatSlice multiplication assignment";
//
//      initialize();
//
//      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
//                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};
//
//      RT quatslice2 = blaze::quatslice( m, 1UL );
//      quatslice2 *= blaze::quatslice( m, 0UL );
//
//      checkRows    ( quatslice2, 3UL );
//      checkColumns ( quatslice2, 3UL );
//      checkCapacity( quatslice2, 9UL );
//      checkNonZeros( quatslice2, 9UL );
//      checkRows    ( m,  3UL );
//      checkColumns ( m,  3UL );
//      checkQuats   ( m,  2UL );
//      checkNonZeros( m, 18UL );
//
//      if( quatslice2(0,0) != 90 || quatslice2(0,1) != 114 || quatslice2(0,2) != 138 ||
//          quatslice2(1,0) != 54 || quatslice2(1,1) !=  69 || quatslice2(1,2) !=  84 ||
//          quatslice2(2,0) != 18 || quatslice2(2,1) !=  24 || quatslice2(2,2) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 90 114 138 )\n( 54 69 84 )\n( 18 24 30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( m(0,0,0) !=  1 || m(0,0,1) !=   2 || m(0,0,2) !=   3 ||
//          m(0,1,0) !=  4 || m(0,1,1) !=   5 || m(0,1,2) !=   6 ||
//          m(0,2,0) !=  7 || m(0,2,1) !=   8 || m(0,2,2) !=   9 ||
//          m(1,0,0) != 90 || m(1,0,1) != 114 || m(1,0,2) != 138 ||
//          m(1,1,0) != 54 || m(1,1,1) !=  69 || m(1,1,2) !=  84 ||
//          m(1,2,0) != 18 || m(1,2,1) !=  24 || m(1,2,2) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((   1   2   3 )\n"
//                                     " (   4   5   6 )\n"
//                                     " (   7   8   9 ))\n"
//                                     "((  90 114 138 )\n"
//                                     " (  54  69  84 )\n"
//                                     " (  18  24  30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // dense quaternion multiplication assignment
//   //=====================================================================================
//
//   {
//      test_ = "dense quaternion multiplication assignment (mixed type)";
//
//      initialize();
//
//      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
//                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};
//
//      RT quatslice2 = blaze::quatslice( m, 1UL );
//
//      const blaze::DynamicMatrix<short, blaze::rowMajor> m1{
//          {1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//
//      quatslice2 *= m1;
//
//      checkRows    ( quatslice2, 3UL );
//      checkColumns ( quatslice2, 3UL );
//      checkCapacity( quatslice2, 9UL );
//      checkNonZeros( quatslice2, 9UL );
//      checkRows    ( m,  3UL );
//      checkColumns ( m,  3UL );
//      checkQuats   ( m,  2UL );
//      checkNonZeros( m, 18UL );
//
//      if( quatslice2(0,0) != 90 || quatslice2(0,1) != 114 || quatslice2(0,2) != 138 ||
//          quatslice2(1,0) != 54 || quatslice2(1,1) !=  69 || quatslice2(1,2) !=  84 ||
//          quatslice2(2,0) != 18 || quatslice2(2,1) !=  24 || quatslice2(2,2) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 90 114 138 )\n( 54 69 84 )\n( 18 24 30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( m(0,0,0) !=  1 || m(0,0,1) !=   2 || m(0,0,2) !=   3 ||
//          m(0,1,0) !=  4 || m(0,1,1) !=   5 || m(0,1,2) !=   6 ||
//          m(0,2,0) !=  7 || m(0,2,1) !=   8 || m(0,2,2) !=   9 ||
//          m(1,0,0) != 90 || m(1,0,1) != 114 || m(1,0,2) != 138 ||
//          m(1,1,0) != 54 || m(1,1,1) !=  69 || m(1,1,2) !=  84 ||
//          m(1,2,0) != 18 || m(1,2,1) !=  24 || m(1,2,2) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((   1   2   3 )\n"
//                                     " (   4   5   6 )\n"
//                                     " (   7   8   9 ))\n"
//                                     "((  90 114 138 )\n"
//                                     " (  54  69  84 )\n"
//                                     " (  18  24  30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "dense quaternion multiplication assignment (aligned/padded)";
//
//      using blaze::aligned;
//      using blaze::padded;
//      using blaze::rowMajor;
//
//      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
//                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};
//
//      RT quatslice2 = blaze::quatslice( m, 1UL );
//
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 48UL ) );
//      AlignedPadded m1( memory.get(), 3UL, 3UL, 16UL );
//      m1(0,0) = 1;
//      m1(0,1) = 2;
//      m1(0,2) = 3;
//      m1(1,0) = 4;
//      m1(1,1) = 5;
//      m1(1,2) = 6;
//      m1(2,0) = 7;
//      m1(2,1) = 8;
//      m1(2,2) = 9;
//
//      quatslice2 *= m1;
//
//      checkRows    ( quatslice2, 3UL );
//      checkColumns ( quatslice2, 3UL );
//      checkCapacity( quatslice2, 9UL );
//      checkNonZeros( quatslice2, 9UL );
//      checkRows    ( m,  3UL );
//      checkColumns ( m,  3UL );
//      checkQuats   ( m,  2UL );
//      checkNonZeros( m, 18UL );
//
//      if( quatslice2(0,0) != 90 || quatslice2(0,1) != 114 || quatslice2(0,2) != 138 ||
//          quatslice2(1,0) != 54 || quatslice2(1,1) !=  69 || quatslice2(1,2) !=  84 ||
//          quatslice2(2,0) != 18 || quatslice2(2,1) !=  24 || quatslice2(2,2) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 90 114 138 )\n( 54 69 84 )\n( 18 24 30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( m(0,0,0) !=  1 || m(0,0,1) !=   2 || m(0,0,2) !=   3 ||
//          m(0,1,0) !=  4 || m(0,1,1) !=   5 || m(0,1,2) !=   6 ||
//          m(0,2,0) !=  7 || m(0,2,1) !=   8 || m(0,2,2) !=   9 ||
//          m(1,0,0) != 90 || m(1,0,1) != 114 || m(1,0,2) != 138 ||
//          m(1,1,0) != 54 || m(1,1,1) !=  69 || m(1,1,2) !=  84 ||
//          m(1,2,0) != 18 || m(1,2,1) !=  24 || m(1,2,2) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((   1   2   3 )\n"
//                                     " (   4   5   6 )\n"
//                                     " (   7   8   9 ))\n"
//                                     "((  90 114 138 )\n"
//                                     " (  54  69  84 )\n"
//                                     " (  18  24  30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "dense quaternion multiplication assignment (unaligned/unpadded)";
//
//      using blaze::unaligned;
//      using blaze::unpadded;
//      using blaze::rowMajor;
//
//      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
//                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};
//
//      RT quatslice2 = blaze::quatslice( m, 1UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[10] );
//      UnalignedUnpadded m1( memory.get()+1UL, 3UL , 3UL);
//      m1(0,0) = 1;
//      m1(0,1) = 2;
//      m1(0,2) = 3;
//      m1(1,0) = 4;
//      m1(1,1) = 5;
//      m1(1,2) = 6;
//      m1(2,0) = 7;
//      m1(2,1) = 8;
//      m1(2,2) = 9;
//
//      quatslice2 *= m1;
//
//      checkRows    ( quatslice2, 3UL );
//      checkColumns ( quatslice2, 3UL );
//      checkCapacity( quatslice2, 9UL );
//      checkNonZeros( quatslice2, 9UL );
//      checkRows    ( m,  3UL );
//      checkColumns ( m,  3UL );
//      checkQuats   ( m,  2UL );
//      checkNonZeros( m, 18UL );
//
//      if( quatslice2(0,0) != 90 || quatslice2(0,1) != 114 || quatslice2(0,2) != 138 ||
//          quatslice2(1,0) != 54 || quatslice2(1,1) !=  69 || quatslice2(1,2) !=  84 ||
//          quatslice2(2,0) != 18 || quatslice2(2,1) !=  24 || quatslice2(2,2) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 90 114 138 )\n( 54 69 84 )\n( 18 24 30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( m(0,0,0) !=  1 || m(0,0,1) !=   2 || m(0,0,2) !=   3 ||
//          m(0,1,0) !=  4 || m(0,1,1) !=   5 || m(0,1,2) !=   6 ||
//          m(0,2,0) !=  7 || m(0,2,1) !=   8 || m(0,2,2) !=   9 ||
//          m(1,0,0) != 90 || m(1,0,1) != 114 || m(1,0,2) != 138 ||
//          m(1,1,0) != 54 || m(1,1,1) !=  69 || m(1,1,2) !=  84 ||
//          m(1,2,0) != 18 || m(1,2,1) !=  24 || m(1,2,2) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((   1   2   3 )\n"
//                                     " (   4   5   6 )\n"
//                                     " (   7   8   9 ))\n"
//                                     "((  90 114 138 )\n"
//                                     " (  54  69  84 )\n"
//                                     " (  18  24  30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the QuatSlice Schur product assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the Schur product assignment operators of the QuatSlice
//// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testSchurAssign()
//{
//   //=====================================================================================
//   // QuatSlice Schur product assignment
//   //=====================================================================================
//
//   {
//      test_ = "QuatSlice Schur product assignment";
//
//      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
//                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};
//
//      RT quatslice2 = blaze::quatslice( m, 1UL );
//      quatslice2 %= blaze::quatslice( m, 0UL );
//
//      checkRows    ( quatslice2, 3UL );
//      checkColumns ( quatslice2, 3UL );
//      checkCapacity( quatslice2, 9UL );
//      checkNonZeros( quatslice2, 9UL );
//      checkRows    ( m,  3UL );
//      checkColumns ( m,  3UL );
//      checkQuats   ( m,  2UL );
//      checkNonZeros( m, 18UL );
//
//      if( quatslice2(0,0) !=  9 || quatslice2(0,1) != 16 || quatslice2(0,2) != 21 ||
//          quatslice2(1,0) != 24 || quatslice2(1,1) != 25 || quatslice2(1,2) != 24 ||
//          quatslice2(2,0) != 21 || quatslice2(2,1) != 16 || quatslice2(2,2) !=  9 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( m(0,0,0) !=  1 || m(0,0,1) !=  2 || m(0,0,2) !=  3 ||
//          m(0,1,0) !=  4 || m(0,1,1) !=  5 || m(0,1,2) !=  6 ||
//          m(0,2,0) !=  7 || m(0,2,1) !=  8 || m(0,2,2) !=  9 ||
//          m(1,0,0) !=  9 || m(1,0,1) != 16 || m(1,0,2) != 21 ||
//          m(1,1,0) != 24 || m(1,1,1) != 25 || m(1,1,2) != 24 ||
//          m(1,2,0) != 21 || m(1,2,1) != 16 || m(1,2,2) !=  9 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  1  2  3 )\n"
//                                     " (  4  5  6 )\n"
//                                     " (  7  8  9 ))\n"
//                                     "((  9 16 21 )\n"
//                                     " ( 24 25 24 )\n"
//                                     " ( 21 16  9 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // dense quaternion Schur product assignment
//   //=====================================================================================
//
//   {
//      test_ = "dense vector Schur product assignment (mixed type)";
//
//      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
//                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};
//
//      RT quatslice2 = blaze::quatslice( m, 1UL );
//
//      const blaze::DynamicMatrix<short, blaze::rowMajor> m1{
//          {1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//
//      quatslice2 %= m1;
//
//      checkRows    ( quatslice2, 3UL );
//      checkColumns ( quatslice2, 3UL );
//      checkCapacity( quatslice2, 9UL );
//      checkNonZeros( quatslice2, 9UL );
//      checkRows    ( m,  3UL );
//      checkColumns ( m,  3UL );
//      checkQuats   ( m,  2UL );
//      checkNonZeros( m, 18UL );
//
//      if( quatslice2(0,0) !=  9 || quatslice2(0,1) != 16 || quatslice2(0,2) != 21 ||
//          quatslice2(1,0) != 24 || quatslice2(1,1) != 25 || quatslice2(1,2) != 24 ||
//          quatslice2(2,0) != 21 || quatslice2(2,1) != 16 || quatslice2(2,2) !=  9 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( m(0,0,0) !=  1 || m(0,0,1) !=  2 || m(0,0,2) !=  3 ||
//          m(0,1,0) !=  4 || m(0,1,1) !=  5 || m(0,1,2) !=  6 ||
//          m(0,2,0) !=  7 || m(0,2,1) !=  8 || m(0,2,2) !=  9 ||
//          m(1,0,0) !=  9 || m(1,0,1) != 16 || m(1,0,2) != 21 ||
//          m(1,1,0) != 24 || m(1,1,1) != 25 || m(1,1,2) != 24 ||
//          m(1,2,0) != 21 || m(1,2,1) != 16 || m(1,2,2) !=  9 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  1  2  3 )\n"
//                                     " (  4  5  6 )\n"
//                                     " (  7  8  9 ))\n"
//                                     "((  9 16 21 )\n"
//                                     " ( 24 25 24 )\n"
//                                     " ( 21 16  9 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "dense quaternion Schur product assignment (aligned/padded)";
//
//      using blaze::aligned;
//      using blaze::padded;
//      using blaze::rowMajor;
//
//      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
//                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};
//
//      RT quatslice2 = blaze::quatslice( m, 1UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 48UL ) );
//      AlignedPadded m1( memory.get(), 3UL, 3UL, 16UL );
//      m1(0,0) = 1;
//      m1(0,1) = 2;
//      m1(0,2) = 3;
//      m1(1,0) = 4;
//      m1(1,1) = 5;
//      m1(1,2) = 6;
//      m1(2,0) = 7;
//      m1(2,1) = 8;
//      m1(2,2) = 9;
//
//      quatslice2 %= m1;
//
//      checkRows    ( quatslice2, 3UL );
//      checkColumns ( quatslice2, 3UL );
//      checkCapacity( quatslice2, 9UL );
//      checkNonZeros( quatslice2, 9UL );
//      checkRows    ( m,  3UL );
//      checkColumns ( m,  3UL );
//      checkQuats   ( m,  2UL );
//      checkNonZeros( m, 18UL );
//
//      if( quatslice2(0,0) !=  9 || quatslice2(0,1) != 16 || quatslice2(0,2) != 21 ||
//          quatslice2(1,0) != 24 || quatslice2(1,1) != 25 || quatslice2(1,2) != 24 ||
//          quatslice2(2,0) != 21 || quatslice2(2,1) != 16 || quatslice2(2,2) !=  9 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( m(0,0,0) !=  1 || m(0,0,1) !=  2 || m(0,0,2) !=  3 ||
//          m(0,1,0) !=  4 || m(0,1,1) !=  5 || m(0,1,2) !=  6 ||
//          m(0,2,0) !=  7 || m(0,2,1) !=  8 || m(0,2,2) !=  9 ||
//          m(1,0,0) !=  9 || m(1,0,1) != 16 || m(1,0,2) != 21 ||
//          m(1,1,0) != 24 || m(1,1,1) != 25 || m(1,1,2) != 24 ||
//          m(1,2,0) != 21 || m(1,2,1) != 16 || m(1,2,2) !=  9 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  1  2  3 )\n"
//                                     " (  4  5  6 )\n"
//                                     " (  7  8  9 ))\n"
//                                     "((  9 16 21 )\n"
//                                     " ( 24 25 24 )\n"
//                                     " ( 21 16  9 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "dense quaternion Schur product assignment (unaligned/unpadded)";
//
//      using blaze::unaligned;
//      using blaze::unpadded;
//      using blaze::rowMajor;
//
//      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
//                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};
//
//      RT quatslice2 = blaze::quatslice( m, 1UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[10] );
//      UnalignedUnpadded m1( memory.get()+1UL, 3UL , 3UL);
//      m1(0,0) = 1;
//      m1(0,1) = 2;
//      m1(0,2) = 3;
//      m1(1,0) = 4;
//      m1(1,1) = 5;
//      m1(1,2) = 6;
//      m1(2,0) = 7;
//      m1(2,1) = 8;
//      m1(2,2) = 9;
//
//      quatslice2 %= m1;
//
//      checkRows    ( quatslice2, 3UL );
//      checkColumns ( quatslice2, 3UL );
//      checkCapacity( quatslice2, 9UL );
//      checkNonZeros( quatslice2, 9UL );
//      checkRows    ( m,  3UL );
//      checkColumns ( m,  3UL );
//      checkQuats   ( m,  2UL );
//      checkNonZeros( m, 18UL );
//
//      if( quatslice2(0,0) !=  9 || quatslice2(0,1) != 16 || quatslice2(0,2) != 21 ||
//          quatslice2(1,0) != 24 || quatslice2(1,1) != 25 || quatslice2(1,2) != 24 ||
//          quatslice2(2,0) != 21 || quatslice2(2,1) != 16 || quatslice2(2,2) !=  9 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( m(0,0,0) !=  1 || m(0,0,1) !=  2 || m(0,0,2) !=  3 ||
//          m(0,1,0) !=  4 || m(0,1,1) !=  5 || m(0,1,2) !=  6 ||
//          m(0,2,0) !=  7 || m(0,2,1) !=  8 || m(0,2,2) !=  9 ||
//          m(1,0,0) !=  9 || m(1,0,1) != 16 || m(1,0,2) != 21 ||
//          m(1,1,0) != 24 || m(1,1,1) != 25 || m(1,1,2) != 24 ||
//          m(1,2,0) != 21 || m(1,2,1) != 16 || m(1,2,2) !=  9 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  1  2  3 )\n"
//                                     " (  4  5  6 )\n"
//                                     " (  7  8  9 ))\n"
//                                     "((  9 16 21 )\n"
//                                     " ( 24 25 24 )\n"
//                                     " ( 21 16  9 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of all QuatSlice (self-)scaling operations.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of all available ways to scale an instance of the QuatSlice
//// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testScaling()
//{
//   //=====================================================================================
//   // self-scaling (v*=2)
//   //=====================================================================================
//
//   {
//      test_ = "self-scaling (v*=2)";
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//      quatslice2 *= 3;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   3 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -6 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -9 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=  12 || quatslice2(3,2) != 15 || quatslice2(3,3) != -18 ||
//          quatslice2(4,0) != 21 || quatslice2(4,1) != -24 || quatslice2(4,2) != 27 || quatslice2(4,3) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Failed self-scaling operation\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   3 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) != -6 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -9 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  12 || quat_(1,3,2) != 15 || quat_(1,3,3) != -18 ||
//          quat_(1,4,0) != 21 || quat_(1,4,1) != -24 || quat_(1,4,2) != 27 || quat_(1,4,3) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Failed self-scaling operation\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   3   0   0 )\n"
//                                     " ( -6   0  -9   0 )\n"
//                                     " (  0  12  15 -18 )\n"
//                                     " ( 21 -24  27  30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // self-scaling (v=v*2)
//   //=====================================================================================
//
//   {
//      test_ = "self-scaling (v=v*3)";
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//      quatslice2 = quatslice2 * 3;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   3 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -6 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -9 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=  12 || quatslice2(3,2) != 15 || quatslice2(3,3) != -18 ||
//          quatslice2(4,0) != 21 || quatslice2(4,1) != -24 || quatslice2(4,2) != 27 || quatslice2(4,3) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Failed self-scaling operation\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   3 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) != -6 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -9 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  12 || quat_(1,3,2) != 15 || quat_(1,3,3) != -18 ||
//          quat_(1,4,0) != 21 || quat_(1,4,1) != -24 || quat_(1,4,2) != 27 || quat_(1,4,3) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Failed self-scaling operation\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   3   0   0 )\n"
//                                     " ( -6   0  -9   0 )\n"
//                                     " (  0  12  15 -18 )\n"
//                                     " ( 21 -24  27  30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // self-scaling (v=3*v)
//   //=====================================================================================
//
//   {
//      test_ = "self-scaling (v=3*v)";
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//      quatslice2 = 3 * quatslice2;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   3 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -6 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -9 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=  12 || quatslice2(3,2) != 15 || quatslice2(3,3) != -18 ||
//          quatslice2(4,0) != 21 || quatslice2(4,1) != -24 || quatslice2(4,2) != 27 || quatslice2(4,3) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Failed self-scaling operation\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   3 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) != -6 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -9 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  12 || quat_(1,3,2) != 15 || quat_(1,3,3) != -18 ||
//          quat_(1,4,0) != 21 || quat_(1,4,1) != -24 || quat_(1,4,2) != 27 || quat_(1,4,3) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Failed self-scaling operation\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   3   0   0 )\n"
//                                     " ( -6   0  -9   0 )\n"
//                                     " (  0  12  15 -18 )\n"
//                                     " ( 21 -24  27  30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // self-scaling (v/=s)
//   //=====================================================================================
//
//   {
//      test_ = "self-scaling (v/=s)";
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//      quatslice2 /= (1.0/3.0);
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   3 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -6 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -9 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=  12 || quatslice2(3,2) != 15 || quatslice2(3,3) != -18 ||
//          quatslice2(4,0) != 21 || quatslice2(4,1) != -24 || quatslice2(4,2) != 27 || quatslice2(4,3) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Failed self-scaling operation\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   3 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) != -6 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -9 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  12 || quat_(1,3,2) != 15 || quat_(1,3,3) != -18 ||
//          quat_(1,4,0) != 21 || quat_(1,4,1) != -24 || quat_(1,4,2) != 27 || quat_(1,4,3) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Failed self-scaling operation\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   3   0   0 )\n"
//                                     " ( -6   0  -9   0 )\n"
//                                     " (  0  12  15 -18 )\n"
//                                     " ( 21 -24  27  30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // self-scaling (v=v/s)
//   //=====================================================================================
//
//   {
//      test_ = "self-scaling (v=v/s)";
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//      quatslice2 = quatslice2 / (1.0/3.0);
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   3 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -6 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -9 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=  12 || quatslice2(3,2) != 15 || quatslice2(3,3) != -18 ||
//          quatslice2(4,0) != 21 || quatslice2(4,1) != -24 || quatslice2(4,2) != 27 || quatslice2(4,3) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Failed self-scaling operation\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   3 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//          quat_(1,2,0) != -6 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -9 || quat_(1,2,3) !=   0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  12 || quat_(1,3,2) != 15 || quat_(1,3,3) != -18 ||
//          quat_(1,4,0) != 21 || quat_(1,4,1) != -24 || quat_(1,4,2) != 27 || quat_(1,4,3) !=  30 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Failed self-scaling operation\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   0   0   0 )\n"
//                                     " (  0   3   0   0 )\n"
//                                     " ( -6   0  -9   0 )\n"
//                                     " (  0  12  15 -18 )\n"
//                                     " ( 21 -24  27  30 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // QuatSlice::scale()
//   //=====================================================================================
//
//   {
//      test_ = "QuatSlice::scale()";
//
//      initialize();
//
//      // Integral scaling the 3rd quatslice
//      {
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         quatslice2.scale( 3 );
//
//         checkRows    ( quatslice2, 5UL );
//         checkColumns ( quatslice2, 4UL );
//         checkCapacity( quatslice2, 20UL );
//         checkNonZeros( quatslice2, 10UL );
//         checkRows    ( quat_,  5UL );
//         checkColumns ( quat_,  4UL );
//         checkQuats   ( quat_,  2UL );
//         checkNonZeros( quat_, 20UL );
//
//         if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//             quatslice2(1,0) !=  0 || quatslice2(1,1) !=   3 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//             quatslice2(2,0) != -6 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -9 || quatslice2(2,3) !=   0 ||
//             quatslice2(3,0) !=  0 || quatslice2(3,1) !=  12 || quatslice2(3,2) != 15 || quatslice2(3,3) != -18 ||
//             quatslice2(4,0) != 21 || quatslice2(4,1) != -24 || quatslice2(4,2) != 27 || quatslice2(4,3) !=  30 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Failed self-scaling operation\n"
//                << " Details:\n"
//                << "   Result:\n" << quatslice2 << "\n"
//                   << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//             quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//             quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//             quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//             quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//             quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//             quat_(1,1,0) !=  0 || quat_(1,1,1) !=   3 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//             quat_(1,2,0) != -6 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -9 || quat_(1,2,3) !=   0 ||
//             quat_(1,3,0) !=  0 || quat_(1,3,1) !=  12 || quat_(1,3,2) != 15 || quat_(1,3,3) != -18 ||
//             quat_(1,4,0) != 21 || quat_(1,4,1) != -24 || quat_(1,4,2) != 27 || quat_(1,4,3) !=  30 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Failed self-scaling operation\n"
//                << " Details:\n"
//                << "   Result:\n" << quat_ << "\n"
//                << "   Expected result:\n((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0  -3   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  7  -8   9  10 ))\n"
//                                        "((  0   0   0   0 )\n"
//                                        " (  0   3   0   0 )\n"
//                                        " ( -6   0  -9   0 )\n"
//                                        " (  0  12  15 -18 )\n"
//                                        " ( 21 -24  27  30 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      initialize();
//
//      // Floating point scaling the 3rd quatslice
//      {
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         quatslice2.scale( 0.5 );
//
//         checkRows    ( quatslice2,  5UL );
//         checkColumns ( quatslice2,  4UL );
//         checkCapacity( quatslice2, 20UL );
//         checkNonZeros( quatslice2,  9UL );
//         checkRows    ( quat_,  5UL );
//         checkColumns ( quat_,  4UL );
//         checkQuats   ( quat_,  2UL );
//         checkNonZeros( quat_, 19UL );
//
//         if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=  0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=  0 ||
//             quatslice2(1,0) !=  0 || quatslice2(1,1) !=  0 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=  0 ||
//             quatslice2(2,0) != -1 || quatslice2(2,1) !=  0 || quatslice2(2,2) != -1 || quatslice2(2,3) !=  0 ||
//             quatslice2(3,0) !=  0 || quatslice2(3,1) !=  2 || quatslice2(3,2) !=  2 || quatslice2(3,3) != -3 ||
//             quatslice2(4,0) !=  3 || quatslice2(4,1) != -4 || quatslice2(4,2) !=  4 || quatslice2(4,3) !=  5 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Failed self-scaling operation\n"
//                << " Details:\n"
//                << "   Result:\n" << quatslice2 << "\n"
//                   << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( -1 0 -1 0 )\n( 0 12 2 -3 )\n( 3 -4 4 5 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=   0 ||
//             quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=   0 ||
//             quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=   0 ||
//             quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) !=  -6 ||
//             quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) !=  10 ||
//             quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=   0 ||
//             quat_(1,1,0) !=  0 || quat_(1,1,1) !=   0 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=   0 ||
//             quat_(1,2,0) != -1 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -1 || quat_(1,2,3) !=   0 ||
//             quat_(1,3,0) !=  0 || quat_(1,3,1) !=   2 || quat_(1,3,2) !=  2 || quat_(1,3,3) !=  -3 ||
//             quat_(1,4,0) !=  3 || quat_(1,4,1) !=  -4 || quat_(1,4,2) !=  4 || quat_(1,4,3) !=   5 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Failed self-scaling operation\n"
//                << " Details:\n"
//                << "   Result:\n" << quat_ << "\n"
//                << "   Expected result:\n((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0  -3   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  7  -8   9  10 ))\n"
//                                        "((  0   0   0   0 )\n"
//                                        " (  0   0   0   0 )\n"
//                                        " ( -1   0  -1   0 )\n"
//                                        " (  0   2   2  -3 )\n"
//                                        " (  3  -4   4   5 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the QuatSlice function call operator.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of adding and accessing elements via the function call operator
//// of the QuatSlice specialization. In case an error is detected, a \a std::runtime_error exception
//// is thrown.
//*/
//void DenseGeneralTest::testFunctionCall()
//{
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "QuatSlice::operator()";
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//
//      // Assignment to the element at index (0,1)
//      quatslice2(0,1) = 9;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 11UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 21UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   9 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -3 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//          quatslice2(4,0) !=  7 || quatslice2(4,1) !=  -8 || quatslice2(4,2) !=  9 || quatslice2(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  9 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) != -2 || quat_(1,2,1) !=  0 || quat_(1,2,2) != -3 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) !=  7 || quat_(1,4,1) != -8 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   9   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Assignment to the element at index (2,2)
//      quatslice2(2,2) = 0;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   9 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//          quatslice2(4,0) !=  7 || quatslice2(4,1) !=  -8 || quatslice2(4,2) !=  9 || quatslice2(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  9 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) != -2 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) !=  7 || quat_(1,4,1) != -8 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   9   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0   0   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Assignment to the element at index (4,1)
//      quatslice2(4,1) = -9;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   9 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//          quatslice2(4,0) !=  7 || quatslice2(4,1) !=  -9 || quatslice2(4,2) !=  9 || quatslice2(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   9 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) != -2 || quat_(1,2,1) !=   0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=   4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) !=  7 || quat_(1,4,1) !=  -9 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   9   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0   0   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -9   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Addition assignment to the element at index (0,1)
//      quatslice2(0,1) += -3;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   6 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//          quatslice2(4,0) !=  7 || quatslice2(4,1) !=  -9 || quatslice2(4,2) !=  9 || quatslice2(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  6 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) != -2 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) !=  7 || quat_(1,4,1) != -9 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   6   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -9   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Subtraction assignment to the element at index (2,0)
//      quatslice2(2,0) -= 6;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   6 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -8 || quatslice2(2,1) !=   0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//          quatslice2(4,0) !=  7 || quatslice2(4,1) !=  -9 || quatslice2(4,2) !=  9 || quatslice2(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  6 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) != -8 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) !=  7 || quat_(1,4,1) != -9 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   6   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -8   0   0   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -9   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Multiplication assignment to the element at index (4,0)
//      quatslice2(4,0) *= -3;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=   0 || quatslice2(0,1) !=   6 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=   0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) !=  -8 || quatslice2(2,1) !=   0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=   0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//          quatslice2(4,0) != -21 || quatslice2(4,1) !=  -9 || quatslice2(4,2) !=  9 || quatslice2(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -6 )\n( -21 -9 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=   0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=   0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) !=  -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=   0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=   7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=   0 || quat_(1,0,1) !=  6 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=   0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) !=  -8 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=   0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) != -21 || quat_(1,4,1) != -9 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((   0   0   0   0 )\n"
//                                     " (   0   1   0   0 )\n"
//                                     " (  -2   0  -3   0 )\n"
//                                     " (   0   4   5  -6 )\n"
//                                     " (   7  -8   9  10 ))\n"
//                                     "((   0   6   0   0 )\n"
//                                     " (   0   1   0   0 )\n"
//                                     " (  -8   0   0   0 )\n"
//                                     " (   0   4   5  -6 )\n"
//                                     " ( -21  -9   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Division assignment to the element at index (3,3)
//      quatslice2(3,3) /= 2;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=   0 || quatslice2(0,1) !=   6 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=   0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) !=  -8 || quatslice2(2,1) !=   0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=   0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -3 ||
//          quatslice2(4,0) != -21 || quatslice2(4,1) !=  -9 || quatslice2(4,2) !=  9 || quatslice2(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -3 )\n( -21 -9 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=   0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=   0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) !=  -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=   0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=   7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=   0 || quat_(1,0,1) !=  6 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=   0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) !=  -8 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=   0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -3 ||
//          quat_(1,4,0) != -21 || quat_(1,4,1) != -9 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((   0   0   0   0 )\n"
//                                     " (   0   1   0   0 )\n"
//                                     " (  -2   0  -3   0 )\n"
//                                     " (   0   4   5  -6 )\n"
//                                     " (   7  -8   9  10 ))\n"
//                                     "((   0   6   0   0 )\n"
//                                     " (   0   1   0   0 )\n"
//                                     " (  -8   0   0   0 )\n"
//                                     " (   0   4   5  -3 )\n"
//                                     " ( -21  -9   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the QuatSlice at() operator.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of adding and accessing elements via the at() operator
//// of the QuatSlice specialization. In case an error is detected, a \a std::runtime_error exception
//// is thrown.
//*/
//void DenseGeneralTest::testAt()
//{
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "QuatSlice::at()";
//
//      initialize();
//
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//
//      // Assignment to the element at index (0,1)
//      quatslice2.at(0,1) = 9;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 11UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 21UL );
//
//      if( quatslice2.at(0,0) !=  0 || quatslice2.at(0,1) !=   9 || quatslice2.at(0,2) !=  0 || quatslice2.at(0,3) !=   0 ||
//          quatslice2.at(1,0) !=  0 || quatslice2.at(1,1) !=   1 || quatslice2.at(1,2) !=  0 || quatslice2.at(1,3) !=   0 ||
//          quatslice2.at(2,0) != -2 || quatslice2.at(2,1) !=   0 || quatslice2.at(2,2) != -3 || quatslice2.at(2,3) !=   0 ||
//          quatslice2.at(3,0) !=  0 || quatslice2.at(3,1) !=   4 || quatslice2.at(3,2) !=  5 || quatslice2.at(3,3) !=  -6 ||
//          quatslice2.at(4,0) !=  7 || quatslice2.at(4,1) !=  -8 || quatslice2.at(4,2) !=  9 || quatslice2.at(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  9 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) != -2 || quat_(1,2,1) !=  0 || quat_(1,2,2) != -3 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) !=  7 || quat_(1,4,1) != -8 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   9   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Assignment to the element at index (2,2)
//      quatslice2.at(2,2) = 0;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2.at(0,0) !=  0 || quatslice2.at(0,1) !=   9 || quatslice2.at(0,2) !=  0 || quatslice2.at(0,3) !=   0 ||
//          quatslice2.at(1,0) !=  0 || quatslice2.at(1,1) !=   1 || quatslice2.at(1,2) !=  0 || quatslice2.at(1,3) !=   0 ||
//          quatslice2.at(2,0) != -2 || quatslice2.at(2,1) !=   0 || quatslice2.at(2,2) !=  0 || quatslice2.at(2,3) !=   0 ||
//          quatslice2.at(3,0) !=  0 || quatslice2.at(3,1) !=   4 || quatslice2.at(3,2) !=  5 || quatslice2.at(3,3) !=  -6 ||
//          quatslice2.at(4,0) !=  7 || quatslice2.at(4,1) !=  -8 || quatslice2.at(4,2) !=  9 || quatslice2.at(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  9 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) != -2 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) !=  7 || quat_(1,4,1) != -8 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   9   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0   0   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Assignment to the element at index (4,1)
//      quatslice2.at(4,1) = -9;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2.at(0,0) !=  0 || quatslice2.at(0,1) !=   9 || quatslice2.at(0,2) !=  0 || quatslice2.at(0,3) !=   0 ||
//          quatslice2.at(1,0) !=  0 || quatslice2.at(1,1) !=   1 || quatslice2.at(1,2) !=  0 || quatslice2.at(1,3) !=   0 ||
//          quatslice2.at(2,0) != -2 || quatslice2.at(2,1) !=   0 || quatslice2.at(2,2) !=  0 || quatslice2.at(2,3) !=   0 ||
//          quatslice2.at(3,0) !=  0 || quatslice2.at(3,1) !=   4 || quatslice2.at(3,2) !=  5 || quatslice2.at(3,3) !=  -6 ||
//          quatslice2.at(4,0) !=  7 || quatslice2.at(4,1) !=  -9 || quatslice2.at(4,2) !=  9 || quatslice2.at(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=   9 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=   1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) != -2 || quat_(1,2,1) !=   0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=   4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) !=  7 || quat_(1,4,1) !=  -9 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   9   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0   0   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -9   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Addition assignment to the element at index (0,1)
//      quatslice2.at(0,1) += -3;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2.at(0,0) !=  0 || quatslice2.at(0,1) !=   6 || quatslice2.at(0,2) !=  0 || quatslice2.at(0,3) !=   0 ||
//          quatslice2.at(1,0) !=  0 || quatslice2.at(1,1) !=   1 || quatslice2.at(1,2) !=  0 || quatslice2.at(1,3) !=   0 ||
//          quatslice2.at(2,0) != -2 || quatslice2.at(2,1) !=   0 || quatslice2.at(2,2) !=  0 || quatslice2.at(2,3) !=   0 ||
//          quatslice2.at(3,0) !=  0 || quatslice2.at(3,1) !=   4 || quatslice2.at(3,2) !=  5 || quatslice2.at(3,3) !=  -6 ||
//          quatslice2.at(4,0) !=  7 || quatslice2.at(4,1) !=  -9 || quatslice2.at(4,2) !=  9 || quatslice2.at(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  6 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) != -2 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) !=  7 || quat_(1,4,1) != -9 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   6   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -9   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Subtraction assignment to the element at index (2,0)
//      quatslice2.at(2,0) -= 6;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2.at(0,0) !=  0 || quatslice2.at(0,1) !=   6 || quatslice2.at(0,2) !=  0 || quatslice2.at(0,3) !=   0 ||
//          quatslice2.at(1,0) !=  0 || quatslice2.at(1,1) !=   1 || quatslice2.at(1,2) !=  0 || quatslice2.at(1,3) !=   0 ||
//          quatslice2.at(2,0) != -8 || quatslice2.at(2,1) !=   0 || quatslice2.at(2,2) !=  0 || quatslice2.at(2,3) !=   0 ||
//          quatslice2.at(3,0) !=  0 || quatslice2.at(3,1) !=   4 || quatslice2.at(3,2) !=  5 || quatslice2.at(3,3) !=  -6 ||
//          quatslice2.at(4,0) !=  7 || quatslice2.at(4,1) !=  -9 || quatslice2.at(4,2) !=  9 || quatslice2.at(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=  0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) != -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=  0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=  7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=  0 || quat_(1,0,1) !=  6 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=  0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) != -8 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=  0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) !=  7 || quat_(1,4,1) != -9 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((  0   0   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -2   0  -3   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -8   9  10 ))\n"
//                                     "((  0   6   0   0 )\n"
//                                     " (  0   1   0   0 )\n"
//                                     " ( -8   0   0   0 )\n"
//                                     " (  0   4   5  -6 )\n"
//                                     " (  7  -9   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Multiplication assignment to the element at index (4,0)
//      quatslice2.at(4,0) *= -3;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2.at(0,0) !=   0 || quatslice2.at(0,1) !=   6 || quatslice2.at(0,2) !=  0 || quatslice2.at(0,3) !=   0 ||
//          quatslice2.at(1,0) !=   0 || quatslice2.at(1,1) !=   1 || quatslice2.at(1,2) !=  0 || quatslice2.at(1,3) !=   0 ||
//          quatslice2.at(2,0) !=  -8 || quatslice2.at(2,1) !=   0 || quatslice2.at(2,2) !=  0 || quatslice2.at(2,3) !=   0 ||
//          quatslice2.at(3,0) !=   0 || quatslice2.at(3,1) !=   4 || quatslice2.at(3,2) !=  5 || quatslice2.at(3,3) !=  -6 ||
//          quatslice2.at(4,0) != -21 || quatslice2.at(4,1) !=  -9 || quatslice2.at(4,2) !=  9 || quatslice2.at(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -6 )\n( -21 -9 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=   0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=   0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) !=  -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=   0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=   7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=   0 || quat_(1,0,1) !=  6 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=   0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) !=  -8 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=   0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//          quat_(1,4,0) != -21 || quat_(1,4,1) != -9 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((   0   0   0   0 )\n"
//                                     " (   0   1   0   0 )\n"
//                                     " (  -2   0  -3   0 )\n"
//                                     " (   0   4   5  -6 )\n"
//                                     " (   7  -8   9  10 ))\n"
//                                     "((   0   6   0   0 )\n"
//                                     " (   0   1   0   0 )\n"
//                                     " (  -8   0   0   0 )\n"
//                                     " (   0   4   5  -6 )\n"
//                                     " ( -21  -9   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Division assignment to the element at index (3,3)
//      quatslice2.at(3,3) /= 2;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2.at(0,0) !=   0 || quatslice2.at(0,1) !=   6 || quatslice2.at(0,2) !=  0 || quatslice2.at(0,3) !=   0 ||
//          quatslice2.at(1,0) !=   0 || quatslice2.at(1,1) !=   1 || quatslice2.at(1,2) !=  0 || quatslice2.at(1,3) !=   0 ||
//          quatslice2.at(2,0) !=  -8 || quatslice2.at(2,1) !=   0 || quatslice2.at(2,2) !=  0 || quatslice2.at(2,3) !=   0 ||
//          quatslice2.at(3,0) !=   0 || quatslice2.at(3,1) !=   4 || quatslice2.at(3,2) !=  5 || quatslice2.at(3,3) !=  -3 ||
//          quatslice2.at(4,0) != -21 || quatslice2.at(4,1) !=  -9 || quatslice2.at(4,2) !=  9 || quatslice2.at(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -3 )\n( -21 -9 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      if( quat_(0,0,0) !=   0 || quat_(0,0,1) !=  0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//          quat_(0,1,0) !=   0 || quat_(0,1,1) !=  1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//          quat_(0,2,0) !=  -2 || quat_(0,2,1) !=  0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//          quat_(0,3,0) !=   0 || quat_(0,3,1) !=  4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//          quat_(0,4,0) !=   7 || quat_(0,4,1) != -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//          quat_(1,0,0) !=   0 || quat_(1,0,1) !=  6 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//          quat_(1,1,0) !=   0 || quat_(1,1,1) !=  1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//          quat_(1,2,0) !=  -8 || quat_(1,2,1) !=  0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//          quat_(1,3,0) !=   0 || quat_(1,3,1) !=  4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -3 ||
//          quat_(1,4,0) != -21 || quat_(1,4,1) != -9 || quat_(1,4,2) !=  9 || quat_(1,4,3) != 10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: At() failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quat_ << "\n"
//             << "   Expected result:\n((   0   0   0   0 )\n"
//                                     " (   0   1   0   0 )\n"
//                                     " (  -2   0  -3   0 )\n"
//                                     " (   0   4   5  -6 )\n"
//                                     " (   7  -8   9  10 ))\n"
//                                     "((   0   6   0   0 )\n"
//                                     " (   0   1   0   0 )\n"
//                                     " (  -8   0   0   0 )\n"
//                                     " (   0   4   5  -3 )\n"
//                                     " ( -21  -9   9  10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the QuatSlice iterator implementation.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the iterator implementation of the QuatSlice specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testIterator()
//{
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      initialize();
//
//      // Testing the Iterator default constructor
//      {
//         test_ = "Iterator default constructor";
//
//         RT::Iterator it{};
//
//         if( it != RT::Iterator() ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Failed iterator default constructor\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Testing the ConstIterator default constructor
//      {
//         test_ = "ConstIterator default constructor";
//
//         RT::ConstIterator it{};
//
//         if( it != RT::ConstIterator() ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Failed iterator default constructor\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Testing conversion from Iterator to ConstIterator
//      {
//         test_ = "Iterator/ConstIterator conversion";
//
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         RT::ConstIterator it( begin( quatslice2, 2UL ) );
//
//         if( it == end( quatslice2, 2UL ) || *it != -2 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Failed iterator conversion detected\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Counting the number of elements in 1st quatslice via Iterator (end-begin)
//      {
//         test_ = "Iterator subtraction (end-begin)";
//
//         RT quatslice1 = blaze::quatslice( quat_, 1UL );
//         const ptrdiff_t number( end( quatslice1, 2UL ) - begin( quatslice1, 2UL ) );
//
//         if( number != 4L ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid number of elements detected\n"
//                << " Details:\n"
//                << "   Number of elements         : " << number << "\n"
//                << "   Expected number of elements: 4\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Counting the number of elements in 1st quatslice via Iterator (begin-end)
//      {
//         test_ = "Iterator subtraction (begin-end)";
//
//         RT quatslice1 = blaze::quatslice( quat_, 1UL );
//         const ptrdiff_t number( begin( quatslice1, 2UL ) - end( quatslice1, 2UL ) );
//
//         if( number != -4L ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid number of elements detected\n"
//                << " Details:\n"
//                << "   Number of elements         : " << number << "\n"
//                << "   Expected number of elements: -4\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Counting the number of elements in 2nd quatslice via ConstIterator (end-begin)
//      {
//         test_ = "ConstIterator subtraction (end-begin)";
//
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         const ptrdiff_t number( cend( quatslice2, 2UL ) - cbegin( quatslice2, 2UL ) );
//
//         if( number != 4L ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid number of elements detected\n"
//                << " Details:\n"
//                << "   Number of elements         : " << number << "\n"
//                << "   Expected number of elements: 4\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Counting the number of elements in 2nd quatslice via ConstIterator (begin-end)
//      {
//         test_ = "ConstIterator subtraction (begin-end)";
//
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         const ptrdiff_t number( cbegin( quatslice2, 2UL ) - cend( quatslice2, 2UL ) );
//
//         if( number != -4L ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid number of elements detected\n"
//                << " Details:\n"
//                << "   Number of elements         : " << number << "\n"
//                << "   Expected number of elements: -4\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Testing read-only access via ConstIterator
//      {
//         test_ = "read-only access via ConstIterator";
//
//         RT quatslice3 = blaze::quatslice( quat_, 0UL );
//         RT::ConstIterator it ( cbegin( quatslice3, 4UL ) );
//         RT::ConstIterator end( cend( quatslice3, 4UL ) );
//
//         if( it == end || *it != 7 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid initial iterator detected\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         ++it;
//
//         if( it == end || *it != -8 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator pre-increment failed\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         --it;
//
//         if( it == end || *it != 7 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator pre-decrement failed\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         it++;
//
//         if( it == end || *it != -8 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator post-increment failed\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         it--;
//
//         if( it == end || *it != 7 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator post-decrement failed\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         it += 2UL;
//
//         if( it == end || *it != 9 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator addition assignment failed\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         it -= 2UL;
//
//         if( it == end || *it != 7 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator subtraction assignment failed\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         it = it + 3UL;
//
//         if( it == end || *it != 10 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator/scalar addition failed\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         it = it - 3UL;
//
//         if( it == end || *it != 7 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator/scalar subtraction failed\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         it = 4UL + it;
//
//         if( it != end ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Scalar/iterator addition failed\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Testing assignment via Iterator
//      {
//         test_ = "assignment via Iterator";
//
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         int value = 6;
//
//         for( RT::Iterator it=begin( quatslice2, 4UL ); it!=end( quatslice2, 4UL ); ++it ) {
//            *it = value++;
//         }
//
//         if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//             quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//             quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -3 || quatslice2(2,3) !=   0 ||
//             quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//             quatslice2(4,0) !=  6 || quatslice2(4,1) !=   7 || quatslice2(4,2) !=  8 || quatslice2(4,3) !=   9 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Assignment via iterator failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 6 7 8 9 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//             quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//             quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//             quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//             quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//             quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//             quat_(1,1,0) !=  0 || quat_(1,1,1) !=   1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//             quat_(1,2,0) != -2 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -3 || quat_(1,2,3) !=  0 ||
//             quat_(1,3,0) !=  0 || quat_(1,3,1) !=   4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//             quat_(1,4,0) !=  6 || quat_(1,4,1) !=   7 || quat_(1,4,2) !=  8 || quat_(1,4,3) !=  9 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Assignment via iterator failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quat_ << "\n"
//                << "   Expected result:\n((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0  -3   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  7  -8   9  10 ))\n"
//                                        "((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0   0   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  6   7   8   9 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Testing addition assignment via Iterator
//      {
//         test_ = "addition assignment via Iterator";
//
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         int value = 2;
//
//         for( RT::Iterator it=begin( quatslice2, 4UL ); it!=end( quatslice2, 4UL ); ++it ) {
//            *it += value++;
//         }
//
//         if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//             quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//             quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -3 || quatslice2(2,3) !=   0 ||
//             quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//             quatslice2(4,0) !=  8 || quatslice2(4,1) !=  10 || quatslice2(4,2) != 12 || quatslice2(4,3) !=  14 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Addition assignment via iterator failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 8 10 12 14 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//             quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//             quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//             quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//             quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//             quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//             quat_(1,1,0) !=  0 || quat_(1,1,1) !=   1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//             quat_(1,2,0) != -2 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -3 || quat_(1,2,3) !=  0 ||
//             quat_(1,3,0) !=  0 || quat_(1,3,1) !=   4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//             quat_(1,4,0) !=  8 || quat_(1,4,1) !=  10 || quat_(1,4,2) != 12 || quat_(1,4,3) != 14 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Addition assignment via iterator failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quat_ << "\n"
//                << "   Expected result:\n((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0  -3   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  7  -8   9  10 ))\n"
//                                        "((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0   0   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  8  10  12  14 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Testing subtraction assignment via Iterator
//      {
//         test_ = "subtraction assignment via Iterator";
//
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         int value = 2;
//
//         for( RT::Iterator it=begin( quatslice2, 4UL ); it!=end( quatslice2, 4UL ); ++it ) {
//            *it -= value++;
//         }
//
//         if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//             quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//             quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -3 || quatslice2(2,3) !=   0 ||
//             quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//             quatslice2(4,0) !=  6 || quatslice2(4,1) !=   7 || quatslice2(4,2) !=  8 || quatslice2(4,3) !=   9 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Subtraction assignment via iterator failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 6 7 8 9 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//             quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//             quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//             quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//             quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//             quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//             quat_(1,1,0) !=  0 || quat_(1,1,1) !=   1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//             quat_(1,2,0) != -2 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -3 || quat_(1,2,3) !=  0 ||
//             quat_(1,3,0) !=  0 || quat_(1,3,1) !=   4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//             quat_(1,4,0) !=  6 || quat_(1,4,1) !=   7 || quat_(1,4,2) !=  8 || quat_(1,4,3) !=  9 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Subtraction assignment via iterator failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quat_ << "\n"
//                << "   Expected result:\n((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0  -3   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  7  -8   9  10 ))\n"
//                                        "((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0   0   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  6   7   8   9 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Testing multiplication assignment via Iterator
//      {
//         test_ = "multiplication assignment via Iterator";
//
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         int value = 1;
//
//         for( RT::Iterator it=begin( quatslice2, 4UL ); it!=end( quatslice2, 4UL ); ++it ) {
//            *it *= value++;
//         }
//
//         if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//             quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//             quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -3 || quatslice2(2,3) !=   0 ||
//             quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//             quatslice2(4,0) !=  6 || quatslice2(4,1) !=  14 || quatslice2(4,2) != 24 || quatslice2(4,3) !=  36 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Multiplication assignment via iterator failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 6 14 24 36 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//             quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//             quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//             quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//             quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//             quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//             quat_(1,1,0) !=  0 || quat_(1,1,1) !=   1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//             quat_(1,2,0) != -2 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -3 || quat_(1,2,3) !=  0 ||
//             quat_(1,3,0) !=  0 || quat_(1,3,1) !=   4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//             quat_(1,4,0) !=  6 || quat_(1,4,1) !=  14 || quat_(1,4,2) != 24 || quat_(1,4,3) != 36 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Multiplication assignment via iterator failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quat_ << "\n"
//                << "   Expected result:\n((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0  -3   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  7  -8   9  10 ))\n"
//                                        "((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0   0   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  6  14  24  36 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Testing division assignment via Iterator
//      {
//         test_ = "division assignment via Iterator";
//
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//
//         for( RT::Iterator it=begin( quatslice2, 4UL ); it!=end( quatslice2, 4UL ); ++it ) {
//            *it /= 2;
//         }
//
//         if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//             quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//             quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -3 || quatslice2(2,3) !=   0 ||
//             quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//             quatslice2(4,0) !=  3 || quatslice2(4,1) !=   7 || quatslice2(4,2) != 12 || quatslice2(4,3) !=  18 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Division assignment via iterator failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 3 7 12 18 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//             quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//             quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//             quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//             quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//             quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//             quat_(1,1,0) !=  0 || quat_(1,1,1) !=   1 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//             quat_(1,2,0) != -2 || quat_(1,2,1) !=   0 || quat_(1,2,2) != -3 || quat_(1,2,3) !=  0 ||
//             quat_(1,3,0) !=  0 || quat_(1,3,1) !=   4 || quat_(1,3,2) !=  5 || quat_(1,3,3) != -6 ||
//             quat_(1,4,0) !=  3 || quat_(1,4,1) !=   7 || quat_(1,4,2) != 12 || quat_(1,4,3) != 18 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Division assignment via iterator failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quat_ << "\n"
//                << "   Expected result:\n((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0  -3   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  7  -8   9  10 ))\n"
//                                        "((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0   0   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  3   7  12  18 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the \c nonZeros() member function of the QuatSlice specialization.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the \c nonZeros() member function of the QuatSlice specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testNonZeros()
//{
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "QuatSlice::nonZeros()";
//
//      initialize();
//
//      // Initialization check
//      RT quatslice2 = blaze::quatslice( quat_, 1UL );
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) != -3 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//          quatslice2(4,0) !=  7 || quatslice2(4,1) !=  -8 || quatslice2(4,2) !=  9 || quatslice2(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Initialization failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Changing the number of non-zeros via the dense quatslice
//      quatslice2(2, 2) = 0;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2,  9UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 19UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//          quatslice2(4,0) !=  7 || quatslice2(4,1) !=  -8 || quatslice2(4,2) !=  9 || quatslice2(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//
//      // Changing the number of non-zeros via the dense quaternion
//      quat_(1,3,0) = 5;
//
//      checkRows    ( quatslice2, 5UL );
//      checkColumns ( quatslice2, 4UL );
//      checkCapacity( quatslice2, 20UL );
//      checkNonZeros( quatslice2, 10UL );
//      checkRows    ( quat_,  5UL );
//      checkColumns ( quat_,  4UL );
//      checkQuats   ( quat_,  2UL );
//      checkNonZeros( quat_, 20UL );
//
//      if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//          quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//          quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=   0 ||
//          quatslice2(3,0) !=  5 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//          quatslice2(4,0) !=  7 || quatslice2(4,1) !=  -8 || quatslice2(4,2) !=  9 || quatslice2(4,3) !=  10 ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Matrix function call operator failed\n"
//             << " Details:\n"
//             << "   Result:\n" << quatslice2 << "\n"
//             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 5 4 5 -6 )\n( 7 -8 9 10 ))\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the \c reset() member function of the QuatSlice specialization.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the \c reset() member function of the QuatSlice specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testReset()
//{
//   using blaze::reset;
//
//
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "QuatSlice::reset()";
//
//      // Resetting a single element in quatslice 3
//      {
//         initialize();
//
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         reset( quatslice2(2, 2) );
//
//
//         checkRows    ( quatslice2, 5UL );
//         checkColumns ( quatslice2, 4UL );
//         checkCapacity( quatslice2, 20UL );
//         checkNonZeros( quatslice2,  9UL );
//         checkRows    ( quat_,  5UL );
//         checkColumns ( quat_,  4UL );
//         checkQuats   ( quat_,  2UL );
//         checkNonZeros( quat_, 19UL );
//
//         if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//             quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//             quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=   0 ||
//             quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//             quatslice2(4,0) !=  7 || quatslice2(4,1) !=  -8 || quatslice2(4,2) !=  9 || quatslice2(4,3) !=  10 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Reset operator failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Resetting the 1st quatslice (lvalue)
//      {
//         initialize();
//
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         reset( quatslice2 );
//
//         checkRows    ( quatslice2, 5UL );
//         checkColumns ( quatslice2, 4UL );
//         checkCapacity( quatslice2, 20UL );
//         checkNonZeros( quatslice2,  0UL );
//         checkRows    ( quat_,  5UL );
//         checkColumns ( quat_,  4UL );
//         checkQuats   ( quat_,  2UL );
//         checkNonZeros( quat_, 10UL );
//
//         if( quatslice2(0,0) != 0 || quatslice2(0,1) !=  0 || quatslice2(0,2) != 0 || quatslice2(0,3) != 0 ||
//             quatslice2(1,0) != 0 || quatslice2(1,1) !=  0 || quatslice2(1,2) != 0 || quatslice2(1,3) != 0 ||
//             quatslice2(2,0) != 0 || quatslice2(2,1) !=  0 || quatslice2(2,2) != 0 || quatslice2(2,3) != 0 ||
//             quatslice2(3,0) != 0 || quatslice2(3,1) !=  0 || quatslice2(3,2) != 0 || quatslice2(3,3) != 0 ||
//             quatslice2(4,0) != 0 || quatslice2(4,1) !=  0 || quatslice2(4,2) != 0 || quatslice2(4,3) != 0 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Reset operation of 1st quatslice failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Resetting the 1st quatslice (rvalue)
//      {
//         initialize();
//
//         reset( blaze::quatslice( quat_, 1UL ) );
//
//         checkRows    ( quat_,  5UL );
//         checkColumns ( quat_,  4UL );
//         checkQuats   ( quat_,  2UL );
//         checkNonZeros( quat_, 10UL );
//
//         if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//             quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//             quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//             quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//             quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//             quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//             quat_(1,1,0) !=  0 || quat_(1,1,1) !=   0 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//             quat_(1,2,0) !=  0 || quat_(1,2,1) !=   0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//             quat_(1,3,0) !=  0 || quat_(1,3,1) !=   0 || quat_(1,3,2) !=  0 || quat_(1,3,3) !=  0 ||
//             quat_(1,4,0) !=  0 || quat_(1,4,1) !=   0 || quat_(1,4,2) !=  0 || quat_(1,4,3) !=  0 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Reset operation of 1st quatslice failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quat_ << "\n"
//                << "   Expected result:\n((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0  -3   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  7  -8   9  10 ))\n"
//                                        "((  0   0   0   0 )\n"
//                                        " (  0   0   0   0 )\n"
//                                        " (  0   0   0   0 )\n"
//                                        " (  0   0   0   0 )\n"
//                                        " (  0   0   0   0 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the \c clear() function with the QuatSlice specialization.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the \c clear() function with the QuatSlice specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testClear()
//{
//   using blaze::clear;
//
//
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "clear() function";
//
//      // Clearing a single element in quatslice 1
//      {
//         initialize();
//
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         clear( quatslice2(2, 2) );
//
//         checkRows    ( quatslice2, 5UL );
//         checkColumns ( quatslice2, 4UL );
//         checkCapacity( quatslice2, 20UL );
//         checkNonZeros( quatslice2,  9UL );
//         checkRows    ( quat_,  5UL );
//         checkColumns ( quat_,  4UL );
//         checkQuats   ( quat_,  2UL );
//         checkNonZeros( quat_, 19UL );
//
//         if( quatslice2(0,0) !=  0 || quatslice2(0,1) !=   0 || quatslice2(0,2) !=  0 || quatslice2(0,3) !=   0 ||
//             quatslice2(1,0) !=  0 || quatslice2(1,1) !=   1 || quatslice2(1,2) !=  0 || quatslice2(1,3) !=   0 ||
//             quatslice2(2,0) != -2 || quatslice2(2,1) !=   0 || quatslice2(2,2) !=  0 || quatslice2(2,3) !=   0 ||
//             quatslice2(3,0) !=  0 || quatslice2(3,1) !=   4 || quatslice2(3,2) !=  5 || quatslice2(3,3) !=  -6 ||
//             quatslice2(4,0) !=  7 || quatslice2(4,1) !=  -8 || quatslice2(4,2) !=  9 || quatslice2(4,3) !=  10 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Clear operation failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Clearing the 3rd quatslice (lvalue)
//      {
//         initialize();
//
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//         clear( quatslice2 );
//
//         checkRows    ( quatslice2, 5UL );
//         checkColumns ( quatslice2, 4UL );
//         checkCapacity( quatslice2, 20UL );
//         checkNonZeros( quatslice2,  0UL );
//         checkRows    ( quat_,  5UL );
//         checkColumns ( quat_,  4UL );
//         checkQuats   ( quat_,  2UL );
//         checkNonZeros( quat_, 10UL );
//
//         if( quatslice2(0,0) != 0 || quatslice2(0,1) !=  0 || quatslice2(0,2) != 0 || quatslice2(0,3) != 0 ||
//             quatslice2(1,0) != 0 || quatslice2(1,1) !=  0 || quatslice2(1,2) != 0 || quatslice2(1,3) != 0 ||
//             quatslice2(2,0) != 0 || quatslice2(2,1) !=  0 || quatslice2(2,2) != 0 || quatslice2(2,3) != 0 ||
//             quatslice2(3,0) != 0 || quatslice2(3,1) !=  0 || quatslice2(3,2) != 0 || quatslice2(3,3) != 0 ||
//             quatslice2(4,0) != 0 || quatslice2(4,1) !=  0 || quatslice2(4,2) != 0 || quatslice2(4,3) != 0 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Clear operation of 3rd quatslice failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quatslice2 << "\n"
//                << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Clearing the 4th quatslice (rvalue)
//      {
//         initialize();
//
//         clear( blaze::quatslice( quat_, 1UL ) );
//
//         checkRows    ( quat_,  5UL );
//         checkColumns ( quat_,  4UL );
//         checkQuats   ( quat_,  2UL );
//         checkNonZeros( quat_, 10UL );
//
//         if( quat_(0,0,0) !=  0 || quat_(0,0,1) !=   0 || quat_(0,0,2) !=  0 || quat_(0,0,3) !=  0 ||
//             quat_(0,1,0) !=  0 || quat_(0,1,1) !=   1 || quat_(0,1,2) !=  0 || quat_(0,1,3) !=  0 ||
//             quat_(0,2,0) != -2 || quat_(0,2,1) !=   0 || quat_(0,2,2) != -3 || quat_(0,2,3) !=  0 ||
//             quat_(0,3,0) !=  0 || quat_(0,3,1) !=   4 || quat_(0,3,2) !=  5 || quat_(0,3,3) != -6 ||
//             quat_(0,4,0) !=  7 || quat_(0,4,1) !=  -8 || quat_(0,4,2) !=  9 || quat_(0,4,3) != 10 ||
//             quat_(1,0,0) !=  0 || quat_(1,0,1) !=   0 || quat_(1,0,2) !=  0 || quat_(1,0,3) !=  0 ||
//             quat_(1,1,0) !=  0 || quat_(1,1,1) !=   0 || quat_(1,1,2) !=  0 || quat_(1,1,3) !=  0 ||
//             quat_(1,2,0) !=  0 || quat_(1,2,1) !=   0 || quat_(1,2,2) !=  0 || quat_(1,2,3) !=  0 ||
//             quat_(1,3,0) !=  0 || quat_(1,3,1) !=   0 || quat_(1,3,2) !=  0 || quat_(1,3,3) !=  0 ||
//             quat_(1,4,0) !=  0 || quat_(1,4,1) !=   0 || quat_(1,4,2) !=  0 || quat_(1,4,3) !=  0 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Clear operation of 1st quatslice failed\n"
//                << " Details:\n"
//                << "   Result:\n" << quat_ << "\n"
//                << "   Expected result:\n((  0   0   0   0 )\n"
//                                        " (  0   1   0   0 )\n"
//                                        " ( -2   0  -3   0 )\n"
//                                        " (  0   4   5  -6 )\n"
//                                        " (  7  -8   9  10 ))\n"
//                                        "((  0   0   0   0 )\n"
//                                        " (  0   0   0   0 )\n"
//                                        " (  0   0   0   0 )\n"
//                                        " (  0   0   0   0 )\n"
//                                        " (  0   0   0   0 ))\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the \c isDefault() function with the QuatSlice specialization.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the \c isDefault() function with the QuatSlice specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testIsDefault()
//{
//   using blaze::isDefault;
//
//
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "isDefault() function";
//
//      initialize();
//
//      // isDefault with default quatslice
//      {
//         RT quatslice0 = blaze::quatslice( quat_, 0UL );
//         quatslice0 = 0;
//
//         if( isDefault( quatslice0(0, 0) ) != true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isDefault evaluation\n"
//                << " Details:\n"
//                << "   QuatSlice element: " << quatslice0(0, 0) << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( isDefault( quatslice0 ) != true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isDefault evaluation\n"
//                << " Details:\n"
//                << "   QuatSlice:\n" << quatslice0 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isDefault with non-default quatslice
//      {
//         RT quatslice1 = blaze::quatslice( quat_, 1UL );
//
//         if( isDefault( quatslice1(1, 1) ) != false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isDefault evaluation\n"
//                << " Details:\n"
//                << "   QuatSlice element: " << quatslice1(1, 1) << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( isDefault( quatslice1 ) != false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isDefault evaluation\n"
//                << " Details:\n"
//                << "   QuatSlice:\n" << quatslice1 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the \c isSame() function with the QuatSlice specialization.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the \c isSame() function with the QuatSlice specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testIsSame()
//{
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "isSame() function";
//
//      // isSame with quatching quatslices
//      {
//         RT quatslice1 = blaze::quatslice( quat_, 1UL );
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//
//         if( blaze::isSame( quatslice1, quatslice2 ) == false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-quatching quatslices
//      {
//         RT quatslice1 = blaze::quatslice( quat_, 0UL );
//         RT quatslice2 = blaze::quatslice( quat_, 1UL );
//
//         quatslice1 = 42;
//
//         if( blaze::isSame( quatslice1, quatslice2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with quatslice and quatching subquaternion
//      {
//         RT   quatslice1 = blaze::quatslice( quat_, 1UL );
//         auto sv   = blaze::subquaternion( quatslice1, 0UL, 0UL, 5UL, 4UL );
//
//         if( blaze::isSame( quatslice1, sv ) == false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   Dense quatslice:\n" << quatslice1 << "\n"
//                << "   Dense subquaternion:\n" << sv << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( blaze::isSame( sv, quatslice1 ) == false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   Dense quatslice:\n" << quatslice1 << "\n"
//                << "   Dense subquaternion:\n" << sv << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with quatslice and non-quatching subquaternion (different size)
//      {
//         RT   quatslice1 = blaze::quatslice( quat_, 1UL );
//         auto sv   = blaze::subquaternion( quatslice1, 0UL, 0UL, 3UL, 3UL );
//
//         if( blaze::isSame( quatslice1, sv ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   Dense quatslice:\n" << quatslice1 << "\n"
//                << "   Dense subquaternion:\n" << sv << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( blaze::isSame( sv, quatslice1 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   Dense quatslice:\n" << quatslice1 << "\n"
//                << "   Dense subquaternion:\n" << sv << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with quatslice and non-quatching subquaternion (different offset)
//      {
//         RT   quatslice1 = blaze::quatslice( quat_, 1UL );
//         auto sv   = blaze::subquaternion( quatslice1, 1UL, 1UL, 3UL, 3UL );
//
//         if( blaze::isSame( quatslice1, sv ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   Dense quatslice:\n" << quatslice1 << "\n"
//                << "   Dense subquaternion:\n" << sv << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( blaze::isSame( sv, quatslice1 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   Dense quatslice:\n" << quatslice1 << "\n"
//                << "   Dense subquaternion:\n" << sv << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with quatching quatslices on a common subquaternion
//      {
//         auto sm   = blaze::subquaternion( quat_, 0UL, 1UL, 1UL, 2UL, 3UL, 2UL );
//         auto quatslice1 = blaze::quatslice( sm, 1UL );
//         auto quatslice2 = blaze::quatslice( sm, 1UL );
//
//         if( blaze::isSame( quatslice1, quatslice2 ) == false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-quatching quatslices on a common subquaternion
//      {
//         auto sm   = blaze::subquaternion( quat_, 0UL, 1UL, 1UL, 2UL, 3UL, 2UL );
//         auto quatslice1 = blaze::quatslice( sm, 0UL );
//         auto quatslice2 = blaze::quatslice( sm, 1UL );
//
//         if( blaze::isSame( quatslice1, quatslice2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with quatching subquaternion on quaternion and subquaternion
//      {
//         auto sm   = blaze::subquaternion( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
//         auto quatslice1 = blaze::quatslice( quat_, 1UL );
//         auto quatslice2 = blaze::quatslice( sm  , 0UL );
//
//         if( blaze::isSame( quatslice1, quatslice2 ) == false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( blaze::isSame( quatslice2, quatslice1 ) == false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-quatching quatslices on quaternion and subquaternion (different quatslice)
//      {
//         auto sm   = blaze::subquaternion( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
//         auto quatslice1 = blaze::quatslice( quat_, 0UL );
//         auto quatslice2 = blaze::quatslice( sm  , 0UL );
//
//         if( blaze::isSame( quatslice1, quatslice2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( blaze::isSame( quatslice2, quatslice1 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-quatching quatslices on quaternion and subquaternion (different size)
//      {
//         auto sm   = blaze::subquaternion( quat_, 1UL, 0UL, 0UL, 1UL, 4UL, 3UL );
//         auto quatslice1 = blaze::quatslice( quat_, 1UL );
//         auto quatslice2 = blaze::quatslice( sm  , 0UL );
//
//         if( blaze::isSame( quatslice1, quatslice2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( blaze::isSame( quatslice2, quatslice1 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with quatching quatslices on two subquaternions
//      {
//         auto sm1  = blaze::subquaternion( quat_, 0UL, 0UL, 0UL, 2UL, 5UL, 4UL );
//         auto sm2  = blaze::subquaternion( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
//         auto quatslice1 = blaze::quatslice( sm1, 1UL );
//         auto quatslice2 = blaze::quatslice( sm2, 0UL );
//
//         if( blaze::isSame( quatslice1, quatslice2 ) == false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( blaze::isSame( quatslice2, quatslice1 ) == false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-quatching quatslices on two subquaternions (different quatslice)
//      {
//         auto sm1  = blaze::subquaternion( quat_, 0UL, 0UL, 0UL, 2UL, 5UL, 4UL );
//         auto sm2  = blaze::subquaternion( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
//         auto quatslice1 = blaze::quatslice( sm1, 0UL );
//         auto quatslice2 = blaze::quatslice( sm2, 0UL );
//
//         if( blaze::isSame( quatslice1, quatslice2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( blaze::isSame( quatslice2, quatslice1 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-quatching quatslices on two subquaternions (different size)
//      {
//         auto sm1  = blaze::subquaternion( quat_, 0UL, 0UL, 0UL, 2UL, 4UL, 3UL );
//         auto sm2  = blaze::subquaternion( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
//         auto quatslice1 = blaze::quatslice( sm1, 1UL );
//         auto quatslice2 = blaze::quatslice( sm2, 0UL );
//
//         if( blaze::isSame( quatslice1, quatslice2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( blaze::isSame( quatslice2, quatslice1 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-quatching quatslices on two subquaternions (different offset)
//      {
//         auto sm1  = blaze::subquaternion( quat_, 0UL, 1UL, 2UL, 2UL, 4UL, 2UL );
//         auto sm2  = blaze::subquaternion( quat_, 1UL, 0UL, 0UL, 1UL, 4UL, 2UL );
//         auto quatslice1 = blaze::quatslice( sm1, 1UL );
//         auto quatslice2 = blaze::quatslice( sm2, 0UL );
//
//         if( blaze::isSame( quatslice1, quatslice2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( blaze::isSame( quatslice2, quatslice1 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First quatslice:\n" << quatslice1 << "\n"
//                << "   Second quatslice:\n" << quatslice2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with quatching quatslice subquatrices on a subquaternion
//      {
//         auto sm   = blaze::subquaternion( quat_, 0UL, 1UL, 2UL, 2UL, 4UL, 2UL );
//         auto quatslice1 = blaze::quatslice( sm, 1UL );
//         auto sv1  = blaze::subquaternion( quatslice1, 0UL, 0UL, 2UL, 1UL );
//         auto sv2  = blaze::subquaternion( quatslice1, 0UL, 0UL, 2UL, 1UL );
//
//         if( blaze::isSame( sv1, sv2 ) == false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First subquaternion:\n" << sv1 << "\n"
//                << "   Second subquaternion:\n" << sv2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-quatching quatslice subquaternions on a subquaternion (different size)
//      {
//         auto sm   = blaze::subquaternion( quat_, 0UL, 1UL, 1UL, 2UL, 4UL, 3UL );
//         auto quatslice1 = blaze::quatslice( sm, 1UL );
//         auto sv1  = blaze::subquaternion( quatslice1, 0UL, 0UL, 2UL, 1UL );
//         auto sv2  = blaze::subquaternion( quatslice1, 0UL, 0UL, 2UL, 2UL );
//
//         if( blaze::isSame( sv1, sv2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First subquaternion:\n" << sv1 << "\n"
//                << "   Second subquaternion:\n" << sv2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-quatching quatslice subquaternions on a subquaternion (different offset)
//      {
//         auto sm   = blaze::subquaternion( quat_, 0UL, 1UL, 1UL, 2UL, 4UL, 3UL );
//         auto quatslice1 = blaze::quatslice( sm, 1UL );
//         auto sv1  = blaze::subquaternion( quatslice1, 0UL, 0UL, 2UL, 1UL );
//         auto sv2  = blaze::subquaternion( quatslice1, 0UL, 1UL, 2UL, 1UL );
//
//         if( blaze::isSame( sv1, sv2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First subquaternion:\n" << sv1 << "\n"
//                << "   Second subquaternion:\n" << sv2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with quatching quatslice subquaternions on two subquaternions
//      {
//         auto sm1  = blaze::subquaternion( quat_, 0UL, 0UL, 0UL, 2UL, 5UL, 4UL );
//         auto sm2  = blaze::subquaternion( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
//         auto quatslice1 = blaze::quatslice( sm1, 1UL );
//         auto quatslice2 = blaze::quatslice( sm2, 0UL );
//         auto sv1  = blaze::subquaternion( quatslice1, 0UL, 0UL, 3UL, 2UL );
//         auto sv2  = blaze::subquaternion( quatslice2, 0UL, 0UL, 3UL, 2UL );
//
//         if( blaze::isSame( sv1, sv2 ) == false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First subquaternion:\n" << sv1 << "\n"
//                << "   Second subquaternion:\n" << sv2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-quatching quatslice subquaternions on two subquaternions (different size)
//      {
//         auto sm1  = blaze::subquaternion( quat_, 0UL, 0UL, 0UL, 2UL, 5UL, 4UL );
//         auto sm2  = blaze::subquaternion( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
//         auto quatslice1 = blaze::quatslice( sm1, 1UL );
//         auto quatslice2 = blaze::quatslice( sm2, 0UL );
//         auto sv1  = blaze::subquaternion( quatslice1, 0UL, 0UL, 3UL, 2UL );
//         auto sv2  = blaze::subquaternion( quatslice2, 0UL, 0UL, 2UL, 2UL );
//
//         if( blaze::isSame( sv1, sv2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First subquaternion:\n" << sv1 << "\n"
//                << "   Second subquaternion:\n" << sv2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-quatching quatslice subquaternions on two subquaternions (different offset)
//      {
//         auto sm1  = blaze::subquaternion( quat_, 0UL, 0UL, 0UL, 2UL, 5UL, 4UL );
//         auto sm2  = blaze::subquaternion( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
//         auto quatslice1 = blaze::quatslice( sm1, 1UL );
//         auto quatslice2 = blaze::quatslice( sm2, 0UL );
//         auto sv1  = blaze::subquaternion( quatslice1, 0UL, 0UL, 3UL, 2UL );
//         auto sv2  = blaze::subquaternion( quatslice2, 0UL, 1UL, 3UL, 2UL );
//
//         if( blaze::isSame( sv1, sv2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First subquaternion:\n" << sv1 << "\n"
//                << "   Second subquaternion:\n" << sv2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the \c subquaternion() function with the QuatSlice specialization.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the \c subquaternion() function used with the QuatSlice specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testSubquaternion()
//{
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "subquaternion() function";
//
//      initialize();
//
//      {
//         RT   quatslice1 = blaze::quatslice( quat_, 1UL );
//         auto sm = blaze::subquaternion( quatslice1, 1UL, 1UL, 2UL, 3UL );
//
//         if( sm(0,0) != 1 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Subscript operator access failed\n"
//                << " Details:\n"
//                << "   Result: " << sm(0,0) << "\n"
//                << "   Expected result: 1\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( *sm.begin(1) != 0 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator access failed\n"
//                << " Details:\n"
//                << "   Result: " << *sm.begin(1) << "\n"
//                << "   Expected result: 0\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      try {
//         RT   quatslice1 = blaze::quatslice( quat_, 1UL );
//         auto sm = blaze::subquaternion( quatslice1, 4UL, 0UL, 4UL, 4UL );
//
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Setup of out-of-bounds subquaternion succeeded\n"
//             << " Details:\n"
//             << "   Result:\n" << sm << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//      catch( std::invalid_argument& ) {}
//
//      try {
//         RT   quatslice1 = blaze::quatslice( quat_, 1UL );
//         auto sm = blaze::subquaternion( quatslice1, 0UL, 0UL, 2UL, 6UL );
//
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Setup of out-of-bounds subquaternion succeeded\n"
//             << " Details:\n"
//             << "   Result:\n" << sm << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//      catch( std::invalid_argument& ) {}
//   }
//}
////*************************************************************************************************
//
////*************************************************************************************************
///*!\brief Test of the \c row() function with the Subquaternion class template.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the \c row() function with the Subquaternion specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testRow()
//{
//   using blaze::quatslice;
//   using blaze::row;
//   using blaze::aligned;
//   using blaze::unaligned;
//
//
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "Quatslice row() function";
//
//      initialize();
//
//      {
//         RT quatslice1  = quatslice( quat_, 0UL );
//         RT quatslice2  = quatslice( quat_, 1UL );
//         auto row1 = row( quatslice1, 1UL );
//         auto row2 = row( quatslice2, 1UL );
//
//         if( row1 != row2 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Row function failed\n"
//                << " Details:\n"
//                << "   Result:\n" << row1 << "\n"
//                << "   Expected result:\n" << row2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( row1[1] != row2[1] ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Subscript operator access failed\n"
//                << " Details:\n"
//                << "   Result: " << row1[1] << "\n"
//                << "   Expected result: " << row2[1] << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( *row1.begin() != *row2.begin() ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator access failed\n"
//                << " Details:\n"
//                << "   Result: " << *row1.begin() << "\n"
//                << "   Expected result: " << *row2.begin() << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      try {
//         RT quatslice1  = quatslice( quat_, 0UL );
//         auto row8 = row( quatslice1, 8UL );
//
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Setup of out-of-bounds row succeeded\n"
//             << " Details:\n"
//             << "   Result:\n" << row8 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//      catch( std::invalid_argument& ) {}
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the \c rows() function with the Subquaternion class template.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the \c rows() function with the Subquaternion specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testRows()
//{
//   using blaze::quatslice;
//   using blaze::rows;
//   using blaze::aligned;
//   using blaze::unaligned;
//
//
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "Quatslice rows() function";
//
//      initialize();
//
//      {
//         RT quatslice1 = quatslice( quat_, 0UL );
//         RT quatslice2 = quatslice( quat_, 1UL );
//         auto rs1 = rows( quatslice1, { 0UL, 2UL, 4UL, 3UL } );
//         auto rs2 = rows( quatslice2, { 0UL, 2UL, 4UL, 3UL } );
//
//         if( rs1 != rs2 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Rows function failed\n"
//                << " Details:\n"
//                << "   Result:\n" << rs1 << "\n"
//                << "   Expected result:\n" << rs2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( rs1(1,1) != rs2(1,1) ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Function call operator access failed\n"
//                << " Details:\n"
//                << "   Result: " << rs1(1,1) << "\n"
//                << "   Expected result: " << rs2(1,1) << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( *rs1.begin( 1UL ) != *rs2.begin( 1UL ) ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator access failed\n"
//                << " Details:\n"
//                << "   Result: " << *rs1.begin( 1UL ) << "\n"
//                << "   Expected result: " << *rs2.begin( 1UL ) << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      try {
//         RT quatslice1 = quatslice( quat_, 1UL );
//         auto rs  = rows( quatslice1, { 8UL } );
//
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Setup of out-of-bounds row selection succeeded\n"
//             << " Details:\n"
//             << "   Result:\n" << rs << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//      catch( std::invalid_argument& ) {}
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the \c column() function with the Subquaternion class template.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the \c column() function with the Subquaternion specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testColumn()
//{
//   using blaze::quatslice;
//   using blaze::column;
//   using blaze::aligned;
//   using blaze::unaligned;
//
//
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "Quatslice column() function";
//
//      initialize();
//
//      {
//         RT quatslice1  = quatslice( quat_, 0UL );
//         RT quatslice2  = quatslice( quat_, 1UL );
//         auto col1 = column( quatslice1, 1UL );
//         auto col2 = column( quatslice2, 1UL );
//
//         if( col1 != col2 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Column function failed\n"
//                << " Details:\n"
//                << "   Result:\n" << col1 << "\n"
//                << "   Expected result:\n" << col2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( col1[1] != col2[1] ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Subscript operator access failed\n"
//                << " Details:\n"
//                << "   Result: " << col1[1] << "\n"
//                << "   Expected result: " << col2[1] << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( *col1.begin() != *col2.begin() ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator access failed\n"
//                << " Details:\n"
//                << "   Result: " << *col1.begin() << "\n"
//                << "   Expected result: " << *col2.begin() << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      try {
//         RT quatslice1  = quatslice( quat_, 0UL );
//         auto col16 = column( quatslice1, 16UL );
//
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Setup of out-of-bounds column succeeded\n"
//             << " Details:\n"
//             << "   Result:\n" << col16 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//      catch( std::invalid_argument& ) {}
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the \c columns() function with the Subquaternion class template.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the \c columns() function with the Subquaternion specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testColumns()
//{
//   using blaze::quatslice;
//   using blaze::rows;
//   using blaze::aligned;
//   using blaze::unaligned;
//
//
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "columns() function";
//
//      initialize();
//
//      {
//         RT quatslice1  = quatslice( quat_, 0UL );
//         RT quatslice2  = quatslice( quat_, 1UL );
//         auto cs1 = columns( quatslice1, { 0UL, 2UL, 2UL, 3UL } );
//         auto cs2 = columns( quatslice2, { 0UL, 2UL, 2UL, 3UL } );
//
//         if( cs1 != cs2 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Rows function failed\n"
//                << " Details:\n"
//                << "   Result:\n" << cs1 << "\n"
//                << "   Expected result:\n" << cs2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( cs1(1,1) != cs2(1,1) ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Function call operator access failed\n"
//                << " Details:\n"
//                << "   Result: " << cs1(1,1) << "\n"
//                << "   Expected result: " << cs2(1,1) << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( *cs1.begin( 1UL ) != *cs2.begin( 1UL ) ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator access failed\n"
//                << " Details:\n"
//                << "   Result: " << *cs1.begin( 1UL ) << "\n"
//                << "   Expected result: " << *cs2.begin( 1UL ) << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      try {
//         RT quatslice1 = quatslice( quat_, 1UL );
//         auto cs  = columns( quatslice1, { 16UL } );
//
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Setup of out-of-bounds column selection succeeded\n"
//             << " Details:\n"
//             << "   Result:\n" << cs << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//      catch( std::invalid_argument& ) {}
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the \c band() function with the Subquaternion class template.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the \c band() function with the Subquaternion specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseGeneralTest::testBand()
//{
//   using blaze::quatslice;
//   using blaze::band;
//   using blaze::aligned;
//   using blaze::unaligned;
//
//
//   //=====================================================================================
//   // quaternion tests
//   //=====================================================================================
//
//   {
//      test_ = "Quatslice band() function";
//
//      initialize();
//
//      {
//         RT quatslice1  = quatslice( quat_, 0UL );
//         RT quatslice2  = quatslice( quat_, 1UL );
//         auto b1 = band( quatslice1, 1L );
//         auto b2 = band( quatslice2, 1L );
//
//         if( b1 != b2 ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Band function failed\n"
//                << " Details:\n"
//                << "   Result:\n" << b1 << "\n"
//                << "   Expected result:\n" << b2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( b1[1] != b2[1] ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Subscript operator access failed\n"
//                << " Details:\n"
//                << "   Result: " << b1[1] << "\n"
//                << "   Expected result: " << b2[1] << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//
//         if( *b1.begin() != *b2.begin() ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Iterator access failed\n"
//                << " Details:\n"
//                << "   Result: " << *b1.begin() << "\n"
//                << "   Expected result: " << *b2.begin() << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      try {
//         RT quatslice1 = quatslice( quat_, 1UL );
//         auto b8 = band( quatslice1, -8L );
//
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Setup of out-of-bounds band succeeded\n"
//             << " Details:\n"
//             << "   Result:\n" << b8 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//      catch( std::invalid_argument& ) {}
//   }
//}
////*************************************************************************************************



//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Initialization of all member quatrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function initializes all member quatrices to specific predetermined values.
*/
void DenseGeneralTest::initialize()
{
   // Initializing the quatslice-major dynamic quaternion
   //quat_.reset();
   //quat_(0,1,1) =  1;
   //quat_(0,2,0) = -2;
   //quat_(0,2,2) = -3;
   //quat_(0,3,1) =  4;
   //quat_(0,3,2) =  5;
   //quat_(0,3,3) = -6;
   //quat_(0,4,0) =  7;
   //quat_(0,4,1) = -8;
   //quat_(0,4,2) =  9;
   //quat_(0,4,3) = 10;
   //quat_(1,1,1) =  1;
   //quat_(1,2,0) = -2;
   //quat_(1,2,2) = -3;
   //quat_(1,3,1) =  4;
   //quat_(1,3,2) =  5;
   //quat_(1,3,3) = -6;
   //quat_(1,4,0) =  7;
   //quat_(1,4,1) = -8;
   //quat_(1,4,2) =  9;
   //quat_(1,4,3) = 10;
}
//*************************************************************************************************

} // namespace quatslice

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
   std::cout << "   Running QuatSlice dense general test..." << std::endl;

   try
   {
      RUN_QUATSLICE_DENSEGENERAL_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during QuatSlice dense general test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
