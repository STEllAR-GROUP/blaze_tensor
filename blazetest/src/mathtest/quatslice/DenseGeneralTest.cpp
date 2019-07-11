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
   : quat_ ( 3UL, 2UL, 5UL, 4UL )
{
   testConstructors();
   testAssignment();
   testAddAssign();
   testSubAssign();
   testSchurAssign();
   testScaling();
   testFunctionCall();
   testAt();
   testIterator();
   testNonZeros();
   testReset();
   testClear();
   testIsDefault();
   testIsSame();
   testSubtensor();
   testPageslice();
   testRowslice();
   testColumnslice();
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

   {
      test_ = "QuatSlice constructor (0x0x0)";

      AT quat;

      // 0th quaternion quatslice
      try {
         blaze::quatslice( quat, 0UL, 0UL );
      }
      catch( std::invalid_argument& ) {}
   }

   {
      test_ = "QuatSlice constructor (3x4x0)";

      AT quat( 2UL, 3UL, 4UL, 0UL );

      // 0th quaternion quatslice
      {
         RT quatslice0 = blaze::quatslice( quat, 0UL );

         checkPages   ( quatslice0, 3UL );
         checkRows    ( quatslice0, 4UL );
         checkColumns ( quatslice0, 0UL );
         checkCapacity( quatslice0, 0UL );
         checkNonZeros( quatslice0, 0UL );
      }

      // 1st quaternion quatslice
      {
         RT quatslice1 = blaze::quatslice( quat, 1UL );

         checkPages   ( quatslice1, 3UL );
         checkRows    ( quatslice1, 4UL );
         checkColumns ( quatslice1, 0UL );
         checkCapacity( quatslice1, 0UL );
         checkNonZeros( quatslice1, 0UL );
      }

      // 2nd quaternion quatslice
      try {
         blaze::quatslice( quat, 2UL );
      }
      catch( std::invalid_argument& ) {}
   }

   {
      test_ = "QuatSlice constructor (2x5x4)";

      initialize();

      // 0th quaternion quatslice
      {
         RT quatslice0 = blaze::quatslice( quat_, 0UL );

         checkPages   ( quatslice0, 2UL  );
         checkRows    ( quatslice0, 5UL  );
         checkColumns ( quatslice0, 4UL  );
         checkCapacity( quatslice0, 40UL );
         checkNonZeros( quatslice0, 20UL );

         if( quatslice0(0,0,0) !=  0 || quatslice0(0,0,1) !=  0 || quatslice0(0,0,2) !=  0 || quatslice0(0,0,3) !=  0 ||
             quatslice0(0,1,0) !=  0 || quatslice0(0,1,1) !=  1 || quatslice0(0,1,2) !=  0 || quatslice0(0,1,3) !=  0 ||
             quatslice0(0,2,0) != -2 || quatslice0(0,2,1) !=  0 || quatslice0(0,2,2) != -3 || quatslice0(0,2,3) !=  0 ||
             quatslice0(0,3,0) !=  0 || quatslice0(0,3,1) !=  4 || quatslice0(0,3,2) !=  5 || quatslice0(0,3,3) != -6 ||
             quatslice0(0,4,0) !=  7 || quatslice0(0,4,1) != -8 || quatslice0(0,4,2) !=  9 || quatslice0(0,4,3) != 10 ||
             quatslice0(1,0,0) !=  0 || quatslice0(1,0,1) !=  0 || quatslice0(1,0,2) !=  0 || quatslice0(1,0,3) !=  0 ||
             quatslice0(1,1,0) !=  0 || quatslice0(1,1,1) !=  1 || quatslice0(1,1,2) !=  0 || quatslice0(1,1,3) !=  0 ||
             quatslice0(1,2,0) != -2 || quatslice0(1,2,1) !=  0 || quatslice0(1,2,2) != 13 || quatslice0(1,2,3) !=  0 ||
             quatslice0(1,3,0) !=  0 || quatslice0(1,3,1) !=  4 || quatslice0(1,3,2) !=  5 || quatslice0(1,3,3) != -6 ||
             quatslice0(1,4,0) !=  7 || quatslice0(1,4,1) != -8 || quatslice0(1,4,2) !=  9 || quatslice0(1,4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of 0th dense quatslice failed\n"
                << " Details:\n"
                << "   Result:\n"
                << quatslice0 << "\n"
                << "   Expected result:\n((   0   0   0   0 ) (     0   1   0  "
                   " 0 ) (    -2   0     -3   0 ) (     0   4   5     -6 ) (   "
                   "  7     -8   9     10 ) )\n((   0   0   0   0 ) (     0   "
                   "1   0   0 ) (    -2   0     13   0 ) (     0   4   5     "
                   "-6 ) (     7     -8   9     10 ) )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // 1st quaternion quatslice
      {
         RT quatslice1 = blaze::quatslice( quat_, 1UL );

         checkPages   ( quatslice1, 2UL  );
         checkRows    ( quatslice1, 5UL );
         checkColumns ( quatslice1, 4UL );
         checkCapacity( quatslice1, 40UL );
         checkNonZeros( quatslice1, 20UL );

         if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
             quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=  0 ||
             quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
             quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
             quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
             quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
             quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
             quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
             quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  5 || quatslice1(1,3,3) != 33 ||
             quatslice1(1,4,0) !=  7 || quatslice1(1,4,1) != -8 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of 1st dense quatslice failed\n"
                << " Details:\n"
                << "   Result:\n"
                << quatslice1 << "\n"
                << "   Expected result:\n((     0   1   0   0 ) (     0   0   "
                   "0   0 ) (     0     12     -3   0 ) (     0   4   5     -6 "
                   ") (     7     28   9     10 ) )\n((     0   0   0   0 ) (  "
                   "   0   1   0   0 ) (    -2   0   0   0 ) (    -3   4   5   "
                   "  33 ) (     7     -8   9     11 ) )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // 3rd quaternion quatslice
      try {
         RT quatslice3 = blaze::quatslice( quat_, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Out-of-bound quat access succeeded\n"
             << " Details:\n"
             << "   Result:\n" << quatslice3 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
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
void DenseGeneralTest::testAssignment()
{
   //=====================================================================================
   // homogeneous assignment
   //=====================================================================================

   {
      test_ = "QuatSlice homogeneous assignment";

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 1UL );
      quatslice1 = 8;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 40UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) != 8 || quatslice1(0,0,1) != 8 || quatslice1(0,0,2) != 8 || quatslice1(0,0,3) != 8 ||
          quatslice1(0,1,0) != 8 || quatslice1(0,1,1) != 8 || quatslice1(0,1,2) != 8 || quatslice1(0,1,3) != 8 ||
          quatslice1(0,2,0) != 8 || quatslice1(0,2,1) != 8 || quatslice1(0,2,2) != 8 || quatslice1(0,2,3) != 8 ||
          quatslice1(0,3,0) != 8 || quatslice1(0,3,1) != 8 || quatslice1(0,3,2) != 8 || quatslice1(0,3,3) != 8 ||
          quatslice1(0,4,0) != 8 || quatslice1(0,4,1) != 8 || quatslice1(0,4,2) != 8 || quatslice1(0,4,3) != 8 ||
          quatslice1(1,0,0) != 8 || quatslice1(1,0,1) != 8 || quatslice1(1,0,2) != 8 || quatslice1(1,0,3) != 8 ||
          quatslice1(1,1,0) != 8 || quatslice1(1,1,1) != 8 || quatslice1(1,1,2) != 8 || quatslice1(1,1,3) != 8 ||
          quatslice1(1,2,0) != 8 || quatslice1(1,2,1) != 8 || quatslice1(1,2,2) != 8 || quatslice1(1,2,3) != 8 ||
          quatslice1(1,3,0) != 8 || quatslice1(1,3,1) != 8 || quatslice1(1,3,2) != 8 || quatslice1(1,3,3) != 8 ||
          quatslice1(1,4,0) != 8 || quatslice1(1,4,1) != 8 || quatslice1(1,4,2) != 8 || quatslice1(1,4,3) != 8) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 8 8 8 8 )\n( 8 8 8 8 )\n( 8 8 8 8 )\n( 8 8 8 8 )\n( 8 8 8 8 ))\n";
         throw std::runtime_error( oss.str() );
      }


      if(  quat_(0,0,0,0) !=  0 || quat_(0,0,0,1) !=  0 || quat_(0,0,0,2) !=  0 || quat_(0,0,0,3) !=  0 ||
           quat_(0,0,1,0) !=  0 || quat_(0,0,1,1) !=  1 || quat_(0,0,1,2) !=  0 || quat_(0,0,1,3) !=  0 ||
           quat_(0,0,2,0) != -2 || quat_(0,0,2,1) !=  0 || quat_(0,0,2,2) != -3 || quat_(0,0,2,3) !=  0 ||
           quat_(0,0,3,0) !=  0 || quat_(0,0,3,1) !=  4 || quat_(0,0,3,2) !=  5 || quat_(0,0,3,3) != -6 ||
           quat_(0,0,4,0) !=  7 || quat_(0,0,4,1) != -8 || quat_(0,0,4,2) !=  9 || quat_(0,0,4,3) != 10 ||
           quat_(0,1,0,0) !=  0 || quat_(0,1,0,1) !=  0 || quat_(0,1,0,2) !=  0 || quat_(0,1,0,3) !=  0 ||
           quat_(0,1,1,0) !=  0 || quat_(0,1,1,1) !=  1 || quat_(0,1,1,2) !=  0 || quat_(0,1,1,3) !=  0 ||
           quat_(0,1,2,0) != -2 || quat_(0,1,2,1) !=  0 || quat_(0,1,2,2) != 13 || quat_(0,1,2,3) !=  0 ||
           quat_(0,1,3,0) !=  0 || quat_(0,1,3,1) !=  4 || quat_(0,1,3,2) !=  5 || quat_(0,1,3,3) != -6 ||
           quat_(0,1,4,0) !=  7 || quat_(0,1,4,1) != -8 || quat_(0,1,4,2) !=  9 || quat_(0,1,4,3) != 10 ||
           quat_(1,0,0,0) != 8 || quat_(1,0,0,1) != 8 || quat_(1,0,0,2) != 8 || quat_(1,0,0,3) != 8 ||
           quat_(1,0,1,0) != 8 || quat_(1,0,1,1) != 8 || quat_(1,0,1,2) != 8 || quat_(1,0,1,3) != 8 ||
           quat_(1,0,2,0) != 8 || quat_(1,0,2,1) != 8 || quat_(1,0,2,2) != 8 || quat_(1,0,2,3) != 8 ||
           quat_(1,0,3,0) != 8 || quat_(1,0,3,1) != 8 || quat_(1,0,3,2) != 8 || quat_(1,0,3,3) != 8 ||
           quat_(1,0,4,0) != 8 || quat_(1,0,4,1) != 8 || quat_(1,0,4,2) != 8 || quat_(1,0,4,3) != 8 ||
           quat_(1,1,0,0) != 8 || quat_(1,1,0,1) != 8 || quat_(1,1,0,2) != 8 || quat_(1,1,0,3) != 8 ||
           quat_(1,1,1,0) != 8 || quat_(1,1,1,1) != 8 || quat_(1,1,1,2) != 8 || quat_(1,1,1,3) != 8 ||
           quat_(1,1,2,0) != 8 || quat_(1,1,2,1) != 8 || quat_(1,1,2,2) != 8 || quat_(1,1,2,3) != 8 ||
           quat_(1,1,3,0) != 8 || quat_(1,1,3,1) != 8 || quat_(1,1,3,2) != 8 || quat_(1,1,3,3) != 8 ||
           quat_(1,1,4,0) != 8 || quat_(1,1,4,1) != 8 || quat_(1,1,4,2) != 8 || quat_(1,1,4,3) != 8) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quat_ << "\n"
             << "   Expected result:\n((((     0   0   0   0 )(   0   1   0   "
                "0 )(  -2   0     -3   0 )(   0   4   5     -6 )(   7     -8   "
                "9     10 ))((   0   0   0   0 )(   0   1   0   0 )(   0   4   "
                "5     -6 )(   7     -8   9     10 )))(((    8   8   8   8 )(  "
                "  8   8   8   8 )(    8   8   8   8 )(    8   8   8   8 )(    "
                "8   8   8   8 ))((   8   8   8   8 )(   8   8   8   8 )(   8  "
                " 8   8   8 )(   8   8   8   8 )(   8   8   8   8 )))";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // list assignment
   //=====================================================================================

   {
      test_ = "initializer list assignment (complete list)";

      initialize();

      RT quatslice3 = blaze::quatslice( quat_, 1UL );
      quatslice3 = {{{1, 2, 3, 4}, {7, 8, 9, 10}, {11, 12, 13, 14},
                       {17, 18, 19, 20}, {21, 22, 23, 24}},
         {{-1, -2, -3, -4}, {-7, -8, -9, -10}, {-11, -12, -13, -14},
            {-17, -18, -19, -20}, {-21, -22, -23, -24}}};


      checkPages   ( quatslice3, 2UL );
      checkRows    ( quatslice3, 5UL );
      checkColumns ( quatslice3, 4UL );
      checkCapacity( quatslice3, 40UL );
      checkNonZeros( quatslice3, 40UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice3(0,0,0) != 1  || quatslice3(0,0,1) != 2  || quatslice3(0,0,2) != 3  || quatslice3(0,0,3) != 4  ||
          quatslice3(0,1,0) != 7  || quatslice3(0,1,1) != 8  || quatslice3(0,1,2) != 9  || quatslice3(0,1,3) != 10 ||
          quatslice3(0,2,0) != 11 || quatslice3(0,2,1) != 12 || quatslice3(0,2,2) != 13 || quatslice3(0,2,3) != 14 ||
          quatslice3(0,3,0) != 17 || quatslice3(0,3,1) != 18 || quatslice3(0,3,2) != 19 || quatslice3(0,3,3) != 20 ||
          quatslice3(0,4,0) != 21 || quatslice3(0,4,1) != 22 || quatslice3(0,4,2) != 23 || quatslice3(0,4,3) != 24 ||
          quatslice3(1,0,0) != -1  || quatslice3(1,0,1) != -2  || quatslice3(1,0,2) != -3  || quatslice3(1,0,3) != -4  ||
          quatslice3(1,1,0) != -7  || quatslice3(1,1,1) != -8  || quatslice3(1,1,2) != -9  || quatslice3(1,1,3) != -10 ||
          quatslice3(1,2,0) != -11 || quatslice3(1,2,1) != -12 || quatslice3(1,2,2) != -13 || quatslice3(1,2,3) != -14 ||
          quatslice3(1,3,0) != -17 || quatslice3(1,3,1) != -18 || quatslice3(1,3,2) != -19 || quatslice3(1,3,3) != -20 ||
          quatslice3(1,4,0) != -21 || quatslice3(1,4,1) != -22 || quatslice3(1,4,2) != -23 || quatslice3(1,4,3) != -24) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quatslice3 << "\n"
             << "   Expected result:\n(((     1   2   3   4 )(     7   8   9   "
                "  10 )(    11     12     13     14 )(    17     18     19     "
                "20 )(    21     22     23     24 ))((    -1     -2     -3     "
                "-4 )(    -7     -8     -9    -10 )(   -11    -12    -13    "
                "-14 )(   -17    -18    -19    -20 )(   -21    -22    -23    "
                "-24 )))\n";
         throw std::runtime_error( oss.str() );
      }

      if( quat_(1,0,0,0) != 1   || quat_(1,0,0,1) != 2   || quat_(1,0,0,2) != 3   || quat_(1,0,0,3) != 4  ||
          quat_(1,0,1,0) != 7   || quat_(1,0,1,1) != 8   || quat_(1,0,1,2) != 9   || quat_(1,0,1,3) != 10 ||
          quat_(1,0,2,0) != 11  || quat_(1,0,2,1) != 12  || quat_(1,0,2,2) != 13  || quat_(1,0,2,3) != 14 ||
          quat_(1,0,3,0) != 17  || quat_(1,0,3,1) != 18  || quat_(1,0,3,2) != 19  || quat_(1,0,3,3) != 20 ||
          quat_(1,0,4,0) != 21  || quat_(1,0,4,1) != 22  || quat_(1,0,4,2) != 23  || quat_(1,0,4,3) != 24 ||
          quat_(1,1,0,0) != -1  || quat_(1,1,0,1) != -2  || quat_(1,1,0,2) != -3  || quat_(1,1,0,3) != -4  ||
          quat_(1,1,1,0) != -7  || quat_(1,1,1,1) != -8  || quat_(1,1,1,2) != -9  || quat_(1,1,1,3) != -10 ||
          quat_(1,1,2,0) != -11 || quat_(1,1,2,1) != -12 || quat_(1,1,2,2) != -13 || quat_(1,1,2,3) != -14 ||
          quat_(1,1,3,0) != -17 || quat_(1,1,3,1) != -18 || quat_(1,1,3,2) != -19 || quat_(1,1,3,3) != -20 ||
          quat_(1,1,4,0) != -21 || quat_(1,1,4,1) != -22 || quat_(1,1,4,2) != -23 || quat_(1,1,4,3) != -24 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quat_ << "\n"
             << "   Expected result:\n(((     1   2   3   4 )(     7   8   9   "
                "  10 )(    11     12     13     14 )(    17     18     19     "
                "20 )(    21     22     23     24 ))((    -1     -2     -3     "
                "-4 )(    -7     -8     -9    -10 )(   -11    -12    -13    "
                "-14 )(   -17    -18    -19    -20 )(   -21    -22    -23    "
                "-24 )))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "initializer list assignment (incomplete list)";

      initialize();

      RT quatslice3 = blaze::quatslice( quat_, 1UL );
      quatslice3 = {{{1, 2}, {1, 2}, {1, 2}, {1, 2}, {1, 2}},
         {{-1, -2}, {-1, -2}, {-1, -2}, {-1, -2}, {-1, -2}}};

      checkPages   ( quatslice3, 2UL );
      checkRows    ( quatslice3, 5UL );
      checkColumns ( quatslice3, 4UL );
      checkCapacity( quatslice3, 40UL );
      checkNonZeros( quatslice3, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice3(0,0,0) != 1 || quatslice3(0,0,1) != 2 || quatslice3(0,0,2) != 0 || quatslice3(0,0,3) != 0 ||
          quatslice3(0,1,0) != 1 || quatslice3(0,1,1) != 2 || quatslice3(0,1,2) != 0 || quatslice3(0,1,3) != 0 ||
          quatslice3(0,2,0) != 1 || quatslice3(0,2,1) != 2 || quatslice3(0,2,2) != 0 || quatslice3(0,2,3) != 0 ||
          quatslice3(0,3,0) != 1 || quatslice3(0,3,1) != 2 || quatslice3(0,3,2) != 0 || quatslice3(0,3,3) != 0 ||
          quatslice3(0,4,0) != 1 || quatslice3(0,4,1) != 2 || quatslice3(0,4,2) != 0 || quatslice3(0,4,3) != 0 ||
          quatslice3(1,0,0) != -1 || quatslice3(1,0,1) != -2 || quatslice3(1,0,2) != 0 || quatslice3(1,0,3) != 0 ||
          quatslice3(1,1,0) != -1 || quatslice3(1,1,1) != -2 || quatslice3(1,1,2) != 0 || quatslice3(1,1,3) != 0 ||
          quatslice3(1,2,0) != -1 || quatslice3(1,2,1) != -2 || quatslice3(1,2,2) != 0 || quatslice3(1,2,3) != 0 ||
          quatslice3(1,3,0) != -1 || quatslice3(1,3,1) != -2 || quatslice3(1,3,2) != 0 || quatslice3(1,3,3) != 0 ||
          quatslice3(1,4,0) != -1 || quatslice3(1,4,1) != -2 || quatslice3(1,4,2) != 0 || quatslice3(1,4,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quatslice3 << "\n"
             << "   Expected result:\n(((     1   2   0   0 )(     1   2   0   "
                "0 )(     1   2   0   0 )(     1   2   0   0 )(     1   2   0  "
                " 0 ))((    -1     -2   0   0 )(    -1     -2   0   0 )(    -1 "
                "    -2   0   0 )(    -1     -2   0   0 )(    -1     -2   0   "
                "0 )))\n";
         throw std::runtime_error( oss.str() );
      }

      if( quat_(1,0,0,0) != 1  || quat_(1,0,0,1) != 2  || quat_(1,0,0,2) != 0 || quat_(1,0,0,3) != 0 ||
          quat_(1,0,1,0) != 1  || quat_(1,0,1,1) != 2  || quat_(1,0,1,2) != 0 || quat_(1,0,1,3) != 0 ||
          quat_(1,0,2,0) != 1  || quat_(1,0,2,1) != 2  || quat_(1,0,2,2) != 0 || quat_(1,0,2,3) != 0 ||
          quat_(1,0,3,0) != 1  || quat_(1,0,3,1) != 2  || quat_(1,0,3,2) != 0 || quat_(1,0,3,3) != 0 ||
          quat_(1,0,4,0) != 1  || quat_(1,0,4,1) != 2  || quat_(1,0,4,2) != 0 || quat_(1,0,4,3) != 0 ||
          quat_(1,1,0,0) != -1 || quat_(1,1,0,1) != -2 || quat_(1,1,0,2) != 0 || quat_(1,1,0,3) != 0 ||
          quat_(1,1,1,0) != -1 || quat_(1,1,1,1) != -2 || quat_(1,1,1,2) != 0 || quat_(1,1,1,3) != 0 ||
          quat_(1,1,2,0) != -1 || quat_(1,1,2,1) != -2 || quat_(1,1,2,2) != 0 || quat_(1,1,2,3) != 0 ||
          quat_(1,1,3,0) != -1 || quat_(1,1,3,1) != -2 || quat_(1,1,3,2) != 0 || quat_(1,1,3,3) != 0 ||
          quat_(1,1,4,0) != -1 || quat_(1,1,4,1) != -2 || quat_(1,1,4,2) != 0 || quat_(1,1,4,3) != 0  ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quat_ << "\n"
             << "   Expected result:\n(((     1   2   0   0 )(     1   2   0   "
                "0 )(     1   2   0   0 )(     1   2   0   0 )(     1   2   0  "
                " 0 ))((    -1     -2   0   0 )(    -1     -2   0   0 )(    -1 "
                "    -2   0   0 )(    -1     -2   0   0 )(    -1     -2   0   "
                "0 )))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // copy assignment
   //=====================================================================================

   {
      test_ = "QuatSlice copy assignment";

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 0UL );
      quatslice1 = 0;
      quatslice1 = blaze::quatslice( quat_, 1UL );

      checkPages   ( quatslice1, 2UL );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );


      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  5 || quatslice1(1,3,3) != 33 ||
          quatslice1(1,4,0) !=  7 || quatslice1(1,4,1) != -8 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n((     0   1   0   0 ) (     0   0   "
                   "0   0 ) (     0     12     -3   0 ) (     0   4   5     -6 "
                   ") (     7     28   9     10 ) )\n((     0   0   0   0 ) (  "
                   "   0   1   0   0 ) (    -2   0   0   0 ) (    -3   4   5   "
                   "  33 ) (     7     -8   9     11 ) )\n";
         throw std::runtime_error( oss.str() );
      }

      if( quat_(0,0,0,0) !=  0 || quat_(0,0,0,1) !=  1 || quat_(0,0,0,2) !=  0 || quat_(0,0,0,3) !=  0 ||
          quat_(0,0,1,0) !=  0 || quat_(0,0,1,1) !=  0 || quat_(0,0,1,2) !=  0 || quat_(0,0,1,3) !=  0 ||
          quat_(0,0,2,0) !=  0 || quat_(0,0,2,1) != 12 || quat_(0,0,2,2) != -3 || quat_(0,0,2,3) !=  0 ||
          quat_(0,0,3,0) !=  0 || quat_(0,0,3,1) !=  4 || quat_(0,0,3,2) !=  5 || quat_(0,0,3,3) != -6 ||
          quat_(0,0,4,0) !=  7 || quat_(0,0,4,1) != 28 || quat_(0,0,4,2) !=  9 || quat_(0,0,4,3) != 10 ||
          quat_(0,1,0,0) !=  0 || quat_(0,1,0,1) !=  0 || quat_(0,1,0,2) !=  0 || quat_(0,1,0,3) !=  0 ||
          quat_(0,1,1,0) !=  0 || quat_(0,1,1,1) !=  1 || quat_(0,1,1,2) !=  0 || quat_(0,1,1,3) !=  0 ||
          quat_(0,1,2,0) != -2 || quat_(0,1,2,1) !=  0 || quat_(0,1,2,2) !=  0 || quat_(0,1,2,3) !=  0 ||
          quat_(0,1,3,0) != -3 || quat_(0,1,3,1) !=  4 || quat_(0,1,3,2) !=  5 || quat_(0,1,3,3) != 33 ||
          quat_(0,1,4,0) !=  7 || quat_(0,1,4,1) != -8 || quat_(0,1,4,2) !=  9 || quat_(0,1,4,3) != 11 ||
          quat_(1,0,0,0) !=  0 || quat_(1,0,0,1) !=  1 || quat_(1,0,0,2) !=  0 || quat_(1,0,0,3) !=  0 ||
          quat_(1,0,1,0) !=  0 || quat_(1,0,1,1) !=  0 || quat_(1,0,1,2) !=  0 || quat_(1,0,1,3) !=  0 ||
          quat_(1,0,2,0) !=  0 || quat_(1,0,2,1) != 12 || quat_(1,0,2,2) != -3 || quat_(1,0,2,3) !=  0 ||
          quat_(1,0,3,0) !=  0 || quat_(1,0,3,1) !=  4 || quat_(1,0,3,2) !=  5 || quat_(1,0,3,3) != -6 ||
          quat_(1,0,4,0) !=  7 || quat_(1,0,4,1) != 28 || quat_(1,0,4,2) !=  9 || quat_(1,0,4,3) != 10 ||
          quat_(1,1,0,0) !=  0 || quat_(1,1,0,1) !=  0 || quat_(1,1,0,2) !=  0 || quat_(1,1,0,3) !=  0 ||
          quat_(1,1,1,0) !=  0 || quat_(1,1,1,1) !=  1 || quat_(1,1,1,2) !=  0 || quat_(1,1,1,3) !=  0 ||
          quat_(1,1,2,0) != -2 || quat_(1,1,2,1) !=  0 || quat_(1,1,2,2) !=  0 || quat_(1,1,2,3) !=  0 ||
          quat_(1,1,3,0) != -3 || quat_(1,1,3,1) !=  4 || quat_(1,1,3,2) !=  5 || quat_(1,1,3,3) != 33 ||
          quat_(1,1,4,0) !=  7 || quat_(1,1,4,1) != -8 || quat_(1,1,4,2) !=  9 || quat_(1,1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quat_ << "\n"
             << "   Expected result:\n((     0   1   0   0 ) (     0   0   "
                "0   0 ) (     0     12     -3   0 ) (     0   4   5     -6 "
                ") (     7     28   9     10 ) )\n((     0   0   0   0 ) (  "
                "   0   1   0   0 ) (    -2   0   0   0 ) (    -3   4   5   "
                "  33 ) (     7     -8   9     11 ) )\n((     0   1   0   0 ) "
                "(     0   0   0   0 ) (  0     12     -3   0 ) (   0   4   5  "
                "-6 ) (  7     28   9     10 ) )\n((   0   0   0   0 ) (  "
                "   0   1   0   0 ) (    -2   0   0   0 ) (    -3   4   5   "
                "  33 ) (     7     -8   9     11 ) )\n(((     0   0   0   0 "
                ")(     0   1   0   0 )(    -2   0     -3   4 )(     0   0   5 "
                "  2 )(     7     -8   9     10 ))((     0   0   0   0 )(     "
                "0   1   0   0 )(    62   0     -3   0 )(     0   5     15     "
                "16 )(    -7     -8     19     10 ))))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // dense quaternion assignment
   //=====================================================================================

   {
      test_ = "dense quaternion assignment ";

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 1UL );

      blaze::DynamicTensor<int> t1;
      t1 = {{{0, 8, 0, 9}, {0}, {0}, {0}, {0}},
         {{7, 8, 10, 9}, {1}, {1}, {1}, {1}}};

      quatslice1 = t1;

      checkPages   ( quatslice1, 2UL );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 10UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  8 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  9 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) !=  0 || quatslice1(0,2,2) !=  0 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  0 || quatslice1(0,3,2) !=  0 || quatslice1(0,3,3) !=  0 ||
          quatslice1(0,4,0) !=  0 || quatslice1(0,4,1) !=  0 || quatslice1(0,4,2) !=  0 || quatslice1(0,4,3) !=  0 ||
          quatslice1(1,0,0) !=  7 || quatslice1(1,0,1) !=  8 || quatslice1(1,0,2) != 10 || quatslice1(1,0,3) !=  9 ||
          quatslice1(1,1,0) !=  1 || quatslice1(1,1,1) !=  0 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) !=  1 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) !=  1 || quatslice1(1,3,1) !=  0 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) !=  0 ||
          quatslice1(1,4,0) !=  1 || quatslice1(1,4,1) !=  0 || quatslice1(1,4,2) !=  0 || quatslice1(1,4,3) !=  0 ) {
         std::ostringstream oss;
         oss
            << " Test: " << test_ << "\n"
            << " Error: Assignment failed\n"
            << " Details:\n"
            << "   Result:\n"
            << quatslice1 << "\n"
            << "   Expected result:\n(((   0    8    0    9 )(   0    0    0   "
               " 0 )(   0    0    0    0 )(   0    0    0    0 )(   0    0    "
               "0    0 ))((   7    8   10    9 )(   1    0    0    0 )(   1    "
               "0    0    0 )(   1    0    0    0 )(   1    0    0    0 )))\n";
         throw std::runtime_error( oss.str() );
      }

      if( quat_(1,0,0,0) !=  0 || quat_(1,0,0,1) !=  8 || quat_(1,0,0,2) !=  0 || quat_(1,0,0,3) !=  9 ||
          quat_(1,0,1,0) !=  0 || quat_(1,0,1,1) !=  0 || quat_(1,0,1,2) !=  0 || quat_(1,0,1,3) !=  0 ||
          quat_(1,0,2,0) !=  0 || quat_(1,0,2,1) !=  0 || quat_(1,0,2,2) !=  0 || quat_(1,0,2,3) !=  0 ||
          quat_(1,0,3,0) !=  0 || quat_(1,0,3,1) !=  0 || quat_(1,0,3,2) !=  0 || quat_(1,0,3,3) !=  0 ||
          quat_(1,0,4,0) !=  0 || quat_(1,0,4,1) !=  0 || quat_(1,0,4,2) !=  0 || quat_(1,0,4,3) !=  0 ||
          quat_(1,1,0,0) !=  7 || quat_(1,1,0,1) !=  8 || quat_(1,1,0,2) != 10 || quat_(1,1,0,3) !=  9 ||
          quat_(1,1,1,0) !=  1 || quat_(1,1,1,1) !=  0 || quat_(1,1,1,2) !=  0 || quat_(1,1,1,3) !=  0 ||
          quat_(1,1,2,0) !=  1 || quat_(1,1,2,1) !=  0 || quat_(1,1,2,2) !=  0 || quat_(1,1,2,3) !=  0 ||
          quat_(1,1,3,0) !=  1 || quat_(1,1,3,1) !=  0 || quat_(1,1,3,2) !=  0 || quat_(1,1,3,3) !=  0 ||
          quat_(1,1,4,0) !=  1 || quat_(1,1,4,1) !=  0 || quat_(1,1,4,2) !=  0 || quat_(1,1,4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quat_ << "\n"
             << "   Expected result:\n((((      0  0  0  0 )(      0  1  0  0 "
                ")(     -2  0      -3  0 )(      0  4  5      -6 )(      7     "
                " -8  9      10 ))((      0  0  0  0 )(      0  1  0  0 )(     "
                "-2  0      13  0 )(      0  4  5      -6 )(      7      -8  9 "
                "     10 )))(((      0  8  0  9 )(      0  0  0  0 )(      0  "
                "0  0  0 )(      0  0  0  0 )(      0  0  0  0 ))((      7  8  "
                "    10  9 )(      1  0  0  0 )(      1  0  0  0 )(      1  0  "
                "0  0 )(      1  0  0  0 )))(((      0  0  0  0 )(      0  1  "
                "0  0 )(     -2  0      -3  4 )(      0  0  5  2 )(      7     "
                " -8  9      10 ))((      0  0  0  0 )(      0  1  0  0 )(     "
                "62  0      -3  0 )(      0  5      15      16 )(     -7      "
                "-8      19      10 ))))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   {
      test_ = "dense quaternion assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 1UL );

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 160UL ) );
      AlignedPadded m1( memory.get(), 2UL, 5UL, 4UL, 16UL );
      m1 = 0;
      m1(0,0,0) = 10;
      m1(0,0,1) = 8;
      m1(0,0,2) = 7;
      m1(0,0,3) = 9;
      m1(1,1,3) = 6;

      quatslice1 = m1;

      checkPages   ( quatslice1, 2UL );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL);
      checkNonZeros( quatslice1, 5UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) != 10 || quatslice1(0,0,1) !=  8 || quatslice1(0,0,2) !=  7 || quatslice1(0,0,3) !=  9 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) !=  0 || quatslice1(0,2,2) !=  0 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  0 || quatslice1(0,3,2) !=  0 || quatslice1(0,3,3) !=  0 ||
          quatslice1(0,4,0) !=  0 || quatslice1(0,4,1) !=  0 || quatslice1(0,4,2) !=  0 || quatslice1(0,4,3) !=  0 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  0 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  6 ||
          quatslice1(1,2,0) !=  0 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) !=  0 || quatslice1(1,3,1) !=  0 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) !=  0 ||
          quatslice1(1,4,0) !=  0 || quatslice1(1,4,1) !=  0 || quatslice1(1,4,2) !=  0 || quatslice1(1,4,3) !=  0) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quatslice1 << "\n"
             << "   Expected result:\n(((     10  8  7  9 )(      0  0  0  0 "
                ")(      0  0  0  0 )(      0  0  0  0 )(      0  0  0  0 ))(( "
                "     0  0  0  0 )(      0  0  0  6 )(      0  0  0  0 )(      "
                "0  0  0  0 )(      0  0  0  0 )))\n";
         throw std::runtime_error( oss.str() );
      }

      if( quat_(1,0,0,0) != 10 || quat_(1,0,0,1) !=  8 || quat_(1,0,0,2) !=  7 || quat_(1,0,0,3) !=  9 ||
          quat_(1,0,1,0) !=  0 || quat_(1,0,1,1) !=  0 || quat_(1,0,1,2) !=  0 || quat_(1,0,1,3) !=  0 ||
          quat_(1,0,2,0) !=  0 || quat_(1,0,2,1) !=  0 || quat_(1,0,2,2) !=  0 || quat_(1,0,2,3) !=  0 ||
          quat_(1,0,3,0) !=  0 || quat_(1,0,3,1) !=  0 || quat_(1,0,3,2) !=  0 || quat_(1,0,3,3) !=  0 ||
          quat_(1,0,4,0) !=  0 || quat_(1,0,4,1) !=  0 || quat_(1,0,4,2) !=  0 || quat_(1,0,4,3) !=  0 ||
          quat_(1,1,0,0) !=  0 || quat_(1,1,0,1) !=  0 || quat_(1,1,0,2) !=  0 || quat_(1,1,0,3) !=  0 ||
          quat_(1,1,1,0) !=  0 || quat_(1,1,1,1) !=  0 || quat_(1,1,1,2) !=  0 || quat_(1,1,1,3) !=  6 ||
          quat_(1,1,2,0) !=  0 || quat_(1,1,2,1) !=  0 || quat_(1,1,2,2) !=  0 || quat_(1,1,2,3) !=  0 ||
          quat_(1,1,3,0) !=  0 || quat_(1,1,3,1) !=  0 || quat_(1,1,3,2) !=  0 || quat_(1,1,3,3) !=  0 ||
          quat_(1,1,4,0) !=  0 || quat_(1,1,4,1) !=  0 || quat_(1,1,4,2) !=  0 || quat_(1,1,4,3) !=  0) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quat_ << "\n"
             << "   Expected result:\n((((      0  0  0  0 )(      0  1  0  0 "
                ")(     -2  0      -3  0 )(      0  4  5      -6 )(      7     "
                " -8  9      10 ))((      0  0  0  0 )(      0  1  0  0 )(     "
                "-2  0      13  0 )(      0  4  5      -6 )(      7      -8  9 "
                "     10 )))(((     10  8  7  9 )(      0  0  0  0 )(      0  "
                "0  0  0 )(      0  0  0  0 )(      0  0  0  0 ))((      0  0  "
                "0  0 )(      0  0  0  6 )(      0  0  0  0 )(      0  0  0  0 "
                ")(      0  0  0  0 )))(((      0  0  0  0 )(      0  1  0  0 "
                ")(     -2  0      -3  4 )(      0  0  5  2 )(      7      -8  "
                "9      10 ))((      0  0  0  0 )(      0  1  0  0 )(     62  "
                "0      -3  0 )(      0  5      15      16 )(     -7      -8   "
                "   19      10 ))))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "dense quaternion assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 1UL );

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[41] );
      UnalignedUnpadded m1( memory.get()+1UL, 2UL, 5UL, 4UL );
      m1 = 0;
      m1(0,0,0) = 10;
      m1(0,0,1) = 8;
      m1(0,0,2) = 7;
      m1(0,0,3) = 9;
      m1(1,1,3) = 6;

      quatslice1 = m1;

      checkPages   ( quatslice1, 2UL );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL);
      checkNonZeros( quatslice1, 5UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) != 10 || quatslice1(0,0,1) !=  8 || quatslice1(0,0,2) !=  7 || quatslice1(0,0,3) !=  9 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) !=  0 || quatslice1(0,2,2) !=  0 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  0 || quatslice1(0,3,2) !=  0 || quatslice1(0,3,3) !=  0 ||
          quatslice1(0,4,0) !=  0 || quatslice1(0,4,1) !=  0 || quatslice1(0,4,2) !=  0 || quatslice1(0,4,3) !=  0 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  0 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  6 ||
          quatslice1(1,2,0) !=  0 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) !=  0 || quatslice1(1,3,1) !=  0 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) !=  0 ||
          quatslice1(1,4,0) !=  0 || quatslice1(1,4,1) !=  0 || quatslice1(1,4,2) !=  0 || quatslice1(1,4,3) !=  0) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quatslice1 << "\n"
             << "   Expected result:\n(((     10  8  7  9 )(      0  0  0  0 "
                ")(      0  0  0  0 )(      0  0  0  0 )(      0  0  0  0 ))(( "
                "     0  0  0  0 )(      0  0  0  6 )(      0  0  0  0 )(      "
                "0  0  0  0 )(      0  0  0  0 )))\n";
         throw std::runtime_error( oss.str() );
      }

      if( quat_(1,0,0,0) != 10 || quat_(1,0,0,1) !=  8 || quat_(1,0,0,2) !=  7 || quat_(1,0,0,3) !=  9 ||
          quat_(1,0,1,0) !=  0 || quat_(1,0,1,1) !=  0 || quat_(1,0,1,2) !=  0 || quat_(1,0,1,3) !=  0 ||
          quat_(1,0,2,0) !=  0 || quat_(1,0,2,1) !=  0 || quat_(1,0,2,2) !=  0 || quat_(1,0,2,3) !=  0 ||
          quat_(1,0,3,0) !=  0 || quat_(1,0,3,1) !=  0 || quat_(1,0,3,2) !=  0 || quat_(1,0,3,3) !=  0 ||
          quat_(1,0,4,0) !=  0 || quat_(1,0,4,1) !=  0 || quat_(1,0,4,2) !=  0 || quat_(1,0,4,3) !=  0 ||
          quat_(1,1,0,0) !=  0 || quat_(1,1,0,1) !=  0 || quat_(1,1,0,2) !=  0 || quat_(1,1,0,3) !=  0 ||
          quat_(1,1,1,0) !=  0 || quat_(1,1,1,1) !=  0 || quat_(1,1,1,2) !=  0 || quat_(1,1,1,3) !=  6 ||
          quat_(1,1,2,0) !=  0 || quat_(1,1,2,1) !=  0 || quat_(1,1,2,2) !=  0 || quat_(1,1,2,3) !=  0 ||
          quat_(1,1,3,0) !=  0 || quat_(1,1,3,1) !=  0 || quat_(1,1,3,2) !=  0 || quat_(1,1,3,3) !=  0 ||
          quat_(1,1,4,0) !=  0 || quat_(1,1,4,1) !=  0 || quat_(1,1,4,2) !=  0 || quat_(1,1,4,3) !=  0) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quat_ << "\n"
             << "   Expected result:\n((((      0  0  0  0 )(      0  1  0  0 "
                ")(     -2  0      -3  0 )(      0  4  5      -6 )(      7     "
                " -8  9      10 ))((      0  0  0  0 )(      0  1  0  0 )(     "
                "-2  0      13  0 )(      0  4  5      -6 )(      7      -8  9 "
                "     10 )))(((     10  8  7  9 )(      0  0  0  0 )(      0  "
                "0  0  0 )(      0  0  0  0 )(      0  0  0  0 ))((      0  0  "
                "0  0 )(      0  0  0  6 )(      0  0  0  0 )(      0  0  0  0 "
                ")(      0  0  0  0 )))(((      0  0  0  0 )(      0  1  0  0 "
                ")(     -2  0      -3  4 )(      0  0  5  2 )(      7      -8  "
                "9      10 ))((      0  0  0  0 )(      0  1  0  0 )(     62  "
                "0      -3  0 )(      0  5      15      16 )(     -7      -8   "
                "   19      10 ))))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the QuatSlice addition assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the addition assignment operators of the QuatSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testAddAssign()
{
   //=====================================================================================
   // QuatSlice addition assignment
   //=====================================================================================

   {
      test_ = "QuatSlice addition assignment";

      initialize();

      RT quatslice1  = blaze::quatslice( quat_, 1UL );
      quatslice1    += blaze::quatslice( quat_, 0UL );

      checkPages   ( quatslice1, 2UL );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 23UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=   1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=   0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=   1 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=   0 ||
          quatslice1(0,2,0) != -2 || quatslice1(0,2,1) !=  12 || quatslice1(0,2,2) != -6 || quatslice1(0,2,3) !=   0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=   8 || quatslice1(0,3,2) != 10 || quatslice1(0,3,3) != -12 ||
          quatslice1(0,4,0) != 14 || quatslice1(0,4,1) !=  20 || quatslice1(0,4,2) != 18 || quatslice1(0,4,3) !=  20 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=   0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=   0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=   2 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=   0 ||
          quatslice1(1,2,0) != -4 || quatslice1(1,2,1) !=   0 || quatslice1(1,2,2) != 13 || quatslice1(1,2,3) !=   0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=   8 || quatslice1(1,3,2) != 10 || quatslice1(1,3,3) !=  27 ||
          quatslice1(1,4,0) != 14 || quatslice1(1,4,1) != -16 || quatslice1(1,4,2) != 18 || quatslice1(1,4,3) !=  21) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quatslice1 << "\n"
             << "   Expected result:\n(((   0  1  0  0 )(   0  1  0  0 )(  -2  "
                "    12      -6  0 )(   0  8      10     -12 )(  14      20    "
                "  18      20 ))((   0  0  0  0 )(   0  2  0  0 )(  -4  0      "
                "13  0 )(  -3  8      10      27 )(  14     -16      18      "
                "21 )))\n";
         throw std::runtime_error( oss.str() );
      }

      if( quat_(1,0,0,0) !=  0 || quat_(1,0,0,1) !=   1 || quat_(1,0,0,2) !=  0 || quat_(1,0,0,3) !=   0 ||
          quat_(1,0,1,0) !=  0 || quat_(1,0,1,1) !=   1 || quat_(1,0,1,2) !=  0 || quat_(1,0,1,3) !=   0 ||
          quat_(1,0,2,0) != -2 || quat_(1,0,2,1) !=  12 || quat_(1,0,2,2) != -6 || quat_(1,0,2,3) !=   0 ||
          quat_(1,0,3,0) !=  0 || quat_(1,0,3,1) !=   8 || quat_(1,0,3,2) != 10 || quat_(1,0,3,3) != -12 ||
          quat_(1,0,4,0) != 14 || quat_(1,0,4,1) !=  20 || quat_(1,0,4,2) != 18 || quat_(1,0,4,3) !=  20 ||
          quat_(1,1,0,0) !=  0 || quat_(1,1,0,1) !=   0 || quat_(1,1,0,2) !=  0 || quat_(1,1,0,3) !=   0 ||
          quat_(1,1,1,0) !=  0 || quat_(1,1,1,1) !=   2 || quat_(1,1,1,2) !=  0 || quat_(1,1,1,3) !=   0 ||
          quat_(1,1,2,0) != -4 || quat_(1,1,2,1) !=   0 || quat_(1,1,2,2) != 13 || quat_(1,1,2,3) !=   0 ||
          quat_(1,1,3,0) != -3 || quat_(1,1,3,1) !=   8 || quat_(1,1,3,2) != 10 || quat_(1,1,3,3) !=  27 ||
          quat_(1,1,4,0) != 14 || quat_(1,1,4,1) != -16 || quat_(1,1,4,2) != 18 || quat_(1,1,4,3) !=  21) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quat_ << "\n"
             << "   Expected result:\n(((   0  1  0  0 )(   0  1  0  0 )(  -2  "
                "    12      -6  0 )(   0  8      10     -12 )(  14      20    "
                "  18      20 ))((   0  0  0  0 )(   0  2  0  0 )(  -4  0      "
                "13  0 )(  -3  8      10      27 )(  14     -16      18  21 "
                ")))\n";
         throw std::runtime_error( oss.str() );
      }

   }

   //=====================================================================================
   // dense quaternion addition assignment
   //=====================================================================================

   {
      test_ = "dense quaternion addition assignment (mixed type)";

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 1UL );

      const blaze::DynamicTensor< short > t{
         {{0, 0, 0, 0}, {0, 1, 0, 0}, {-2, 0, -3, 0}, {0, 4, 5, -6},
            {7, -8, 9, 10}},
         {{0, 0, 0, 0}, {0, 1, 0, 0}, {-2, 0, -3, 0}, {0, 4, 5, -6},
            {7, -8, 9, 10}}};

      quatslice1 += t;

      checkPages   ( quatslice1, 2UL );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 23UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=   1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=   0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=   1 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=   0 ||
          quatslice1(0,2,0) != -2 || quatslice1(0,2,1) !=  12 || quatslice1(0,2,2) != -6 || quatslice1(0,2,3) !=   0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=   8 || quatslice1(0,3,2) != 10 || quatslice1(0,3,3) != -12 ||
          quatslice1(0,4,0) != 14 || quatslice1(0,4,1) !=  20 || quatslice1(0,4,2) != 18 || quatslice1(0,4,3) !=  20 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=   0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=   0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=   2 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=   0 ||
          quatslice1(1,2,0) != -4 || quatslice1(1,2,1) !=   0 || quatslice1(1,2,2) != -3 || quatslice1(1,2,3) !=   0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=   8 || quatslice1(1,3,2) != 10 || quatslice1(1,3,3) !=  27 ||
          quatslice1(1,4,0) != 14 || quatslice1(1,4,1) != -16 || quatslice1(1,4,2) != 18 || quatslice1(1,4,3) !=  21 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( quat_(1,0,0,0) !=  0 || quat_(1,0,0,1) !=   1 || quat_(1,0,0,2) !=  0 || quat_(1,0,0,3) !=   0 ||
          quat_(1,0,1,0) !=  0 || quat_(1,0,1,1) !=   1 || quat_(1,0,1,2) !=  0 || quat_(1,0,1,3) !=   0 ||
          quat_(1,0,2,0) != -2 || quat_(1,0,2,1) !=  12 || quat_(1,0,2,2) != -6 || quat_(1,0,2,3) !=   0 ||
          quat_(1,0,3,0) !=  0 || quat_(1,0,3,1) !=   8 || quat_(1,0,3,2) != 10 || quat_(1,0,3,3) != -12 ||
          quat_(1,0,4,0) != 14 || quat_(1,0,4,1) !=  20 || quat_(1,0,4,2) != 18 || quat_(1,0,4,3) !=  20 ||
          quat_(1,1,0,0) !=  0 || quat_(1,1,0,1) !=   0 || quat_(1,1,0,2) !=  0 || quat_(1,1,0,3) !=   0 ||
          quat_(1,1,1,0) !=  0 || quat_(1,1,1,1) !=   2 || quat_(1,1,1,2) !=  0 || quat_(1,1,1,3) !=   0 ||
          quat_(1,1,2,0) != -4 || quat_(1,1,2,1) !=   0 || quat_(1,1,2,2) != -3 || quat_(1,1,2,3) !=   0 ||
          quat_(1,1,3,0) != -3 || quat_(1,1,3,1) !=   8 || quat_(1,1,3,2) != 10 || quat_(1,1,3,3) !=  27 ||
          quat_(1,1,4,0) != 14 || quat_(1,1,4,1) != -16 || quat_(1,1,4,2) != 18 || quat_(1,1,4,3) !=  21 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   0   0   0 )\n"
                                     " (  0   2   0   0 )\n"
                                     " ( -4   0  -6   0 )\n"
                                     " (  0   8  10 -12 )\n"
                                     " ( 14 -16  18  20 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "dense quaternion addition assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 1UL );

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 160UL ) );
      AlignedPadded m( memory.get(), 2UL, 5UL, 4UL, 16UL );
      m(0,0,0) =  0;
      m(0,0,1) =  0;
      m(0,0,2) =  0;
      m(0,0,3) = 13;
      m(0,1,0) =  0;
      m(0,1,1) =  1;
      m(0,1,2) =  0;
      m(0,1,3) =  0;
      m(0,2,0) = -2;
      m(0,2,1) =  0;
      m(0,2,2) = -3;
      m(0,2,3) =  0;
      m(0,3,0) =  0;
      m(0,3,1) =  4;
      m(0,3,2) =  5;
      m(0,3,3) = -6;
      m(0,4,0) =  7;
      m(0,4,1) = -8;
      m(0,4,2) =  9;
      m(0,4,3) = 10;
      m(1,0,0) =  0;
      m(1,0,1) =  0;
      m(1,0,2) =  0;
      m(1,0,3) =  0;
      m(1,1,0) =  0;
      m(1,1,1) =  1;
      m(1,1,2) =  0;
      m(1,1,3) =  0;
      m(1,2,0) = 33;
      m(1,2,1) =  0;
      m(1,2,2) = -3;
      m(1,2,3) =  0;
      m(1,3,0) =  0;
      m(1,3,1) =  4;
      m(1,3,2) =  5;
      m(1,3,3) = -6;
      m(1,4,0) = 17;
      m(1,4,1) = 18;
      m(1,4,2) =  9;
      m(1,4,3) = 10;

      quatslice1 += m;

      checkPages   ( quatslice1, 2UL );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL);
      checkNonZeros( quatslice1, 24UL);
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=   1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  13 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=   1 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=   0 ||
          quatslice1(0,2,0) != -2 || quatslice1(0,2,1) !=  12 || quatslice1(0,2,2) != -6 || quatslice1(0,2,3) !=   0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=   8 || quatslice1(0,3,2) != 10 || quatslice1(0,3,3) != -12 ||
          quatslice1(0,4,0) != 14 || quatslice1(0,4,1) !=  20 || quatslice1(0,4,2) != 18 || quatslice1(0,4,3) !=  20 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=   0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=   0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=   2 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=   0 ||
          quatslice1(1,2,0) != 31 || quatslice1(1,2,1) !=   0 || quatslice1(1,2,2) != -3 || quatslice1(1,2,3) !=   0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=   8 || quatslice1(1,3,2) != 10 || quatslice1(1,3,3) !=  27 ||
          quatslice1(1,4,0) != 24 || quatslice1(1,4,1) !=  10 || quatslice1(1,4,2) != 18 || quatslice1(1,4,3) !=  21 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( quat_(1,0,0,0) !=  0 || quat_(1,0,0,1) !=   1 || quat_(1,0,0,2) !=  0 || quat_(1,0,0,3) !=  13 ||
          quat_(1,0,1,0) !=  0 || quat_(1,0,1,1) !=   1 || quat_(1,0,1,2) !=  0 || quat_(1,0,1,3) !=   0 ||
          quat_(1,0,2,0) != -2 || quat_(1,0,2,1) !=  12 || quat_(1,0,2,2) != -6 || quat_(1,0,2,3) !=   0 ||
          quat_(1,0,3,0) !=  0 || quat_(1,0,3,1) !=   8 || quat_(1,0,3,2) != 10 || quat_(1,0,3,3) != -12 ||
          quat_(1,0,4,0) != 14 || quat_(1,0,4,1) !=  20 || quat_(1,0,4,2) != 18 || quat_(1,0,4,3) !=  20 ||
          quat_(1,1,0,0) !=  0 || quat_(1,1,0,1) !=   0 || quat_(1,1,0,2) !=  0 || quat_(1,1,0,3) !=   0 ||
          quat_(1,1,1,0) !=  0 || quat_(1,1,1,1) !=   2 || quat_(1,1,1,2) !=  0 || quat_(1,1,1,3) !=   0 ||
          quat_(1,1,2,0) != 31 || quat_(1,1,2,1) !=   0 || quat_(1,1,2,2) != -3 || quat_(1,1,2,3) !=   0 ||
          quat_(1,1,3,0) != -3 || quat_(1,1,3,1) !=   8 || quat_(1,1,3,2) != 10 || quat_(1,1,3,3) !=  27 ||
          quat_(1,1,4,0) != 24 || quat_(1,1,4,1) !=  10 || quat_(1,1,4,2) != 18 || quat_(1,1,4,3) !=  21  ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   0   0   0 )\n"
                                     " (  0   2   0   0 )\n"
                                     " ( -4   0  -6   0 )\n"
                                     " (  0   8  10 -12 )\n"
                                     " ( 14 -16  18  20 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "dense quaternion addition assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 1UL );

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[41] );
      UnalignedUnpadded m( memory.get()+1UL, 2UL, 5UL, 4UL );
      m(0,0,0) =  0;
      m(0,0,1) =  0;
      m(0,0,2) =  0;
      m(0,0,3) = 13;
      m(0,1,0) =  0;
      m(0,1,1) =  1;
      m(0,1,2) =  0;
      m(0,1,3) =  0;
      m(0,2,0) = -2;
      m(0,2,1) =  0;
      m(0,2,2) = -3;
      m(0,2,3) =  0;
      m(0,3,0) =  0;
      m(0,3,1) =  4;
      m(0,3,2) =  5;
      m(0,3,3) = -6;
      m(0,4,0) =  7;
      m(0,4,1) = -8;
      m(0,4,2) =  9;
      m(0,4,3) = 10;
      m(1,0,0) =  0;
      m(1,0,1) =  0;
      m(1,0,2) =  0;
      m(1,0,3) =  0;
      m(1,1,0) =  0;
      m(1,1,1) =  1;
      m(1,1,2) =  0;
      m(1,1,3) =  0;
      m(1,2,0) = 33;
      m(1,2,1) =  0;
      m(1,2,2) = -3;
      m(1,2,3) =  0;
      m(1,3,0) =  0;
      m(1,3,1) =  4;
      m(1,3,2) =  5;
      m(1,3,3) = -6;
      m(1,4,0) = 17;
      m(1,4,1) = 18;
      m(1,4,2) =  9;
      m(1,4,3) = 10;

      quatslice1 += m;

      checkPages   ( quatslice1, 2UL );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL);
      checkNonZeros( quatslice1, 24UL);
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=   1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  13 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=   1 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=   0 ||
          quatslice1(0,2,0) != -2 || quatslice1(0,2,1) !=  12 || quatslice1(0,2,2) != -6 || quatslice1(0,2,3) !=   0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=   8 || quatslice1(0,3,2) != 10 || quatslice1(0,3,3) != -12 ||
          quatslice1(0,4,0) != 14 || quatslice1(0,4,1) !=  20 || quatslice1(0,4,2) != 18 || quatslice1(0,4,3) !=  20 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=   0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=   0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=   2 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=   0 ||
          quatslice1(1,2,0) != 31 || quatslice1(1,2,1) !=   0 || quatslice1(1,2,2) != -3 || quatslice1(1,2,3) !=   0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=   8 || quatslice1(1,3,2) != 10 || quatslice1(1,3,3) !=  27 ||
          quatslice1(1,4,0) != 24 || quatslice1(1,4,1) !=  10 || quatslice1(1,4,2) != 18 || quatslice1(1,4,3) !=  21 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( quat_(1,0,0,0) !=  0 || quat_(1,0,0,1) !=   1 || quat_(1,0,0,2) !=  0 || quat_(1,0,0,3) !=  13 ||
          quat_(1,0,1,0) !=  0 || quat_(1,0,1,1) !=   1 || quat_(1,0,1,2) !=  0 || quat_(1,0,1,3) !=   0 ||
          quat_(1,0,2,0) != -2 || quat_(1,0,2,1) !=  12 || quat_(1,0,2,2) != -6 || quat_(1,0,2,3) !=   0 ||
          quat_(1,0,3,0) !=  0 || quat_(1,0,3,1) !=   8 || quat_(1,0,3,2) != 10 || quat_(1,0,3,3) != -12 ||
          quat_(1,0,4,0) != 14 || quat_(1,0,4,1) !=  20 || quat_(1,0,4,2) != 18 || quat_(1,0,4,3) !=  20 ||
          quat_(1,1,0,0) !=  0 || quat_(1,1,0,1) !=   0 || quat_(1,1,0,2) !=  0 || quat_(1,1,0,3) !=   0 ||
          quat_(1,1,1,0) !=  0 || quat_(1,1,1,1) !=   2 || quat_(1,1,1,2) !=  0 || quat_(1,1,1,3) !=   0 ||
          quat_(1,1,2,0) != 31 || quat_(1,1,2,1) !=   0 || quat_(1,1,2,2) != -3 || quat_(1,1,2,3) !=   0 ||
          quat_(1,1,3,0) != -3 || quat_(1,1,3,1) !=   8 || quat_(1,1,3,2) != 10 || quat_(1,1,3,3) !=  27 ||
          quat_(1,1,4,0) != 24 || quat_(1,1,4,1) !=  10 || quat_(1,1,4,2) != 18 || quat_(1,1,4,3) !=  21  ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   0   0   0 )\n"
                                     " (  0   2   0   0 )\n"
                                     " ( -4   0  -6   0 )\n"
                                     " (  0   8  10 -12 )\n"
                                     " ( 14 -16  18  20 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the QuatSlice subtraction assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the subtraction assignment operators of the QuatSlice
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testSubAssign()
{
   //=====================================================================================
   // QuatSlice subtraction assignment
   //=====================================================================================

   {
      test_ = "QuatSlice subtraction assignment";

      initialize();

      RT quatslice1  = blaze::quatslice( quat_, 1UL );
      quatslice1    -= blaze::quatslice( quat_, 0UL );

      checkPages   ( quatslice1, 2UL );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 9UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=   1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=   0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  -1 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=   0 ||
          quatslice1(0,2,0) !=  2 || quatslice1(0,2,1) !=  12 || quatslice1(0,2,2) !=  0 || quatslice1(0,2,3) !=   0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=   0 || quatslice1(0,3,2) !=  0 || quatslice1(0,3,3) !=   0 ||
          quatslice1(0,4,0) !=  0 || quatslice1(0,4,1) !=  36 || quatslice1(0,4,2) !=  0 || quatslice1(0,4,3) !=   0 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=   0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=   0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=   0 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=   0 ||
          quatslice1(1,2,0) !=  0 || quatslice1(1,2,1) !=   0 || quatslice1(1,2,2) !=-13 || quatslice1(1,2,3) !=   0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=   0 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) !=  39 ||
          quatslice1(1,4,0) !=  0 || quatslice1(1,4,1) !=   0 || quatslice1(1,4,2) !=  0 || quatslice1(1,4,3) !=   1) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n"
             << quatslice1 << "\n"
             << "   Expected result:\n(((   0  1  0  0 )(   0  1  0  0 )(  -2  "
                "    12      -6  0 )(   0  8      10     -12 )(  14      20    "
                "  18      20 ))((   0  0  0  0 )(   0  2  0  0 )(  -4  0      "
                "13  0 )(  -3  8      10      27 )(  14     -16      18      "
                "21 )))\n";
         throw std::runtime_error( oss.str() );
      }

      if( quat_(1,0,0,0) !=  0 || quat_(1,0,0,1) !=   1 || quat_(1,0,0,2) !=  0 || quat_(1,0,0,3) !=   0 ||
          quat_(1,0,1,0) !=  0 || quat_(1,0,1,1) !=  -1 || quat_(1,0,1,2) !=  0 || quat_(1,0,1,3) !=   0 ||
          quat_(1,0,2,0) !=  2 || quat_(1,0,2,1) !=  12 || quat_(1,0,2,2) !=  0 || quat_(1,0,2,3) !=   0 ||
          quat_(1,0,3,0) !=  0 || quat_(1,0,3,1) !=   0 || quat_(1,0,3,2) !=  0 || quat_(1,0,3,3) !=   0 ||
          quat_(1,0,4,0) !=  0 || quat_(1,0,4,1) !=  36 || quat_(1,0,4,2) !=  0 || quat_(1,0,4,3) !=   0 ||
          quat_(1,1,0,0) !=  0 || quat_(1,1,0,1) !=   0 || quat_(1,1,0,2) !=  0 || quat_(1,1,0,3) !=   0 ||
          quat_(1,1,1,0) !=  0 || quat_(1,1,1,1) !=   0 || quat_(1,1,1,2) !=  0 || quat_(1,1,1,3) !=   0 ||
          quat_(1,1,2,0) !=  0 || quat_(1,1,2,1) !=   0 || quat_(1,1,2,2) !=-13 || quat_(1,1,2,3) !=   0 ||
          quat_(1,1,3,0) != -3 || quat_(1,1,3,1) !=   0 || quat_(1,1,3,2) !=  0 || quat_(1,1,3,3) !=  39 ||
          quat_(1,1,4,0) !=  0 || quat_(1,1,4,1) !=   0 || quat_(1,1,4,2) !=  0 || quat_(1,1,4,3) !=   1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   0   0   0 )\n"
                                     " (  0   0   0   0 )\n"
                                     " (  0   0   0   0 )\n"
                                     " (  0   0   0   0 )\n"
                                     " (  0   0   0   0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // dense quaternion subtraction assignment
   //=====================================================================================

   {
      test_ = "dense quaternion subtraction assignment (mixed type)";

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 1UL );

      const blaze::DynamicTensor< short > t{
         {{0, 0, 0, 0}, {0, 1, 0, 0}, {-2, 0, -3, 0}, {0, 4, 5, -6},
            {7, -8, 9, 10}},
         {{0, 0, 0, 0}, {0, 1, 0, 0}, {-2, 0, -3, 0}, {0, 4, 5, -6},
            {7, -8, 9, 10}}};

      quatslice1 -= t;

      checkPages   ( quatslice1, 2UL );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 9UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=   1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=   0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  -1 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=   0 ||
          quatslice1(0,2,0) !=  2 || quatslice1(0,2,1) !=  12 || quatslice1(0,2,2) !=  0 || quatslice1(0,2,3) !=   0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=   0 || quatslice1(0,3,2) !=  0 || quatslice1(0,3,3) !=   0 ||
          quatslice1(0,4,0) !=  0 || quatslice1(0,4,1) !=  36 || quatslice1(0,4,2) !=  0 || quatslice1(0,4,3) !=   0 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=   0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=   0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=   0 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=   0 ||
          quatslice1(1,2,0) !=  0 || quatslice1(1,2,1) !=   0 || quatslice1(1,2,2) !=  3 || quatslice1(1,2,3) !=   0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=   0 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) !=  39 ||
          quatslice1(1,4,0) !=  0 || quatslice1(1,4,1) !=   0 || quatslice1(1,4,2) !=  0 || quatslice1(1,4,3) !=   1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( quat_(1,0,0,0) !=  0 || quat_(1,0,0,1) !=   1 || quat_(1,0,0,2) !=  0 || quat_(1,0,0,3) !=   0 ||
          quat_(1,0,1,0) !=  0 || quat_(1,0,1,1) !=  -1 || quat_(1,0,1,2) !=  0 || quat_(1,0,1,3) !=   0 ||
          quat_(1,0,2,0) !=  2 || quat_(1,0,2,1) !=  12 || quat_(1,0,2,2) !=  0 || quat_(1,0,2,3) !=   0 ||
          quat_(1,0,3,0) !=  0 || quat_(1,0,3,1) !=   0 || quat_(1,0,3,2) !=  0 || quat_(1,0,3,3) !=   0 ||
          quat_(1,0,4,0) !=  0 || quat_(1,0,4,1) !=  36 || quat_(1,0,4,2) !=  0 || quat_(1,0,4,3) !=   0 ||
          quat_(1,1,0,0) !=  0 || quat_(1,1,0,1) !=   0 || quat_(1,1,0,2) !=  0 || quat_(1,1,0,3) !=   0 ||
          quat_(1,1,1,0) !=  0 || quat_(1,1,1,1) !=   0 || quat_(1,1,1,2) !=  0 || quat_(1,1,1,3) !=   0 ||
          quat_(1,1,2,0) !=  0 || quat_(1,1,2,1) !=   0 || quat_(1,1,2,2) !=  3 || quat_(1,1,2,3) !=   0 ||
          quat_(1,1,3,0) != -3 || quat_(1,1,3,1) !=   0 || quat_(1,1,3,2) !=  0 || quat_(1,1,3,3) !=  39 ||
          quat_(1,1,4,0) !=  0 || quat_(1,1,4,1) !=   0 || quat_(1,1,4,2) !=  0 || quat_(1,1,4,3) !=   1  ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   0   0   0 )\n"
                                     " (  0   0   0   0 )\n"
                                     " (  0   0   0   0 )\n"
                                     " (  0   0   0   0 )\n"
                                     " (  0   0   0   0 ))\n";
      }
   }

   {
      test_ = "dense quaternion subtraction assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 1UL );

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 160UL ) );
      AlignedPadded m( memory.get(), 2UL, 5UL, 4UL, 16UL );
      m(0,0,0) =  0;
      m(0,0,1) =  0;
      m(0,0,2) =  0;
      m(0,0,3) = 13;
      m(0,1,0) =  0;
      m(0,1,1) =  1;
      m(0,1,2) =  0;
      m(0,1,3) =  0;
      m(0,2,0) = -2;
      m(0,2,1) =  0;
      m(0,2,2) = -3;
      m(0,2,3) =  0;
      m(0,3,0) =  0;
      m(0,3,1) =  4;
      m(0,3,2) =  5;
      m(0,3,3) = -6;
      m(0,4,0) =  7;
      m(0,4,1) = -8;
      m(0,4,2) =  9;
      m(0,4,3) = 10;
      m(1,0,0) =  0;
      m(1,0,1) =  0;
      m(1,0,2) =  0;
      m(1,0,3) =  0;
      m(1,1,0) =  0;
      m(1,1,1) =  1;
      m(1,1,2) =  0;
      m(1,1,3) =  0;
      m(1,2,0) = 33;
      m(1,2,1) =  0;
      m(1,2,2) = -3;
      m(1,2,3) =  0;
      m(1,3,0) =  0;
      m(1,3,1) =  4;
      m(1,3,2) =  5;
      m(1,3,3) = -6;
      m(1,4,0) = 17;
      m(1,4,1) = 18;
      m(1,4,2) =  9;
      m(1,4,3) = 10;

      quatslice1 -= m;

      checkPages   ( quatslice1, 2UL );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL);
      checkNonZeros( quatslice1, 13UL);
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=   1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) != -13 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  -1 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=   0 ||
          quatslice1(0,2,0) !=  2 || quatslice1(0,2,1) !=  12 || quatslice1(0,2,2) !=  0 || quatslice1(0,2,3) !=   0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=   0 || quatslice1(0,3,2) !=  0 || quatslice1(0,3,3) !=   0 ||
          quatslice1(0,4,0) !=  0 || quatslice1(0,4,1) !=  36 || quatslice1(0,4,2) !=  0 || quatslice1(0,4,3) !=   0 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=   0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=   0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=   0 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=   0 ||
          quatslice1(1,2,0) !=-35 || quatslice1(1,2,1) !=   0 || quatslice1(1,2,2) !=  3 || quatslice1(1,2,3) !=   0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=   0 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) !=  39 ||
          quatslice1(1,4,0) !=-10 || quatslice1(1,4,1) != -26 || quatslice1(1,4,2) !=  0 || quatslice1(1,4,3) !=   1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "dense quaternion subtraction assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 1UL );

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[41] );
      UnalignedUnpadded m( memory.get()+1UL, 2UL, 5UL, 4UL );
      m(0,0,0) =  0;
      m(0,0,1) =  0;
      m(0,0,2) =  0;
      m(0,0,3) = 13;
      m(0,1,0) =  0;
      m(0,1,1) =  1;
      m(0,1,2) =  0;
      m(0,1,3) =  0;
      m(0,2,0) = -2;
      m(0,2,1) =  0;
      m(0,2,2) = -3;
      m(0,2,3) =  0;
      m(0,3,0) =  0;
      m(0,3,1) =  4;
      m(0,3,2) =  5;
      m(0,3,3) = -6;
      m(0,4,0) =  7;
      m(0,4,1) = -8;
      m(0,4,2) =  9;
      m(0,4,3) = 10;
      m(1,0,0) =  0;
      m(1,0,1) =  0;
      m(1,0,2) =  0;
      m(1,0,3) =  0;
      m(1,1,0) =  0;
      m(1,1,1) =  1;
      m(1,1,2) =  0;
      m(1,1,3) =  0;
      m(1,2,0) = 33;
      m(1,2,1) =  0;
      m(1,2,2) = -3;
      m(1,2,3) =  0;
      m(1,3,0) =  0;
      m(1,3,1) =  4;
      m(1,3,2) =  5;
      m(1,3,3) = -6;
      m(1,4,0) = 17;
      m(1,4,1) = 18;
      m(1,4,2) =  9;
      m(1,4,3) = 10;

      quatslice1 -= m;

      checkPages   ( quatslice1, 2UL );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL);
      checkNonZeros( quatslice1, 13UL);
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=   1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) != -13 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  -1 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=   0 ||
          quatslice1(0,2,0) !=  2 || quatslice1(0,2,1) !=  12 || quatslice1(0,2,2) !=  0 || quatslice1(0,2,3) !=   0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=   0 || quatslice1(0,3,2) !=  0 || quatslice1(0,3,3) !=   0 ||
          quatslice1(0,4,0) !=  0 || quatslice1(0,4,1) !=  36 || quatslice1(0,4,2) !=  0 || quatslice1(0,4,3) !=   0 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=   0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=   0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=   0 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=   0 ||
          quatslice1(1,2,0) !=-35 || quatslice1(1,2,1) !=   0 || quatslice1(1,2,2) !=  3 || quatslice1(1,2,3) !=   0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=   0 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) !=  39 ||
          quatslice1(1,4,0) !=-10 || quatslice1(1,4,1) != -26 || quatslice1(1,4,2) !=  0 || quatslice1(1,4,3) !=   1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the QuatSlice Schur product assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the Schur product assignment operators of the QuatSlice
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testSchurAssign()
{
   //=====================================================================================
   // QuatSlice Schur product assignment
   //=====================================================================================

   {
      test_ = "QuatSlice Schur product assignment";

      blaze::DynamicArray< 4, int > a{ {{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                          {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}},
         {{{-1, -2, -3}, {-4, -5, -6}, {-7, -8, -9}},
            {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}}} };

      RT quatslice2 = blaze::quatslice( a, 1UL );
      quatslice2 %= blaze::quatslice( a, 0UL );

      checkPages   ( quatslice2, 2UL );
      checkRows    ( quatslice2, 3UL );
      checkColumns ( quatslice2, 3UL );
      checkCapacity( quatslice2, 18UL );
      checkNonZeros( quatslice2, 18UL );
      checkQuats   ( a,  2UL );
      checkRows    ( a,  3UL );
      checkColumns ( a,  3UL );
      checkPages   ( a,  2UL );
      checkNonZeros( a, 36UL );

      if( quatslice2(0,0,0) !=  -1 || quatslice2(0,0,1) !=  -4 || quatslice2(0,0,2) != -9  ||
          quatslice2(0,1,0) != -16 || quatslice2(0,1,1) != -25 || quatslice2(0,1,2) != -36 ||
          quatslice2(0,2,0) != -49 || quatslice2(0,2,1) != -64 || quatslice2(0,2,2) != -81 ||
          quatslice2(1,0,0) !=  81 || quatslice2(1,0,1) !=  64 || quatslice2(1,0,2) !=  49 ||
          quatslice2(1,1,0) !=  36 || quatslice2(1,1,1) !=  25 || quatslice2(1,1,2) !=  16 ||
          quatslice2(1,2,0) !=   9 || quatslice2(1,2,1) !=   4 || quatslice2(1,2,2) !=  1) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice2 << "\n"
             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( a(1,0,0,0) != -1  || a(1,0,0,1) != -4  || a(1,0,0,2) != -9  ||
          a(1,0,1,0) != -16 || a(1,0,1,1) != -25 || a(1,0,1,2) != -36 ||
          a(1,0,2,0) != -49 || a(1,0,2,1) != -64 || a(1,0,2,2) != -81 ||
          a(1,1,0,0) !=  81 || a(1,1,0,1) !=  64 || a(1,1,0,2) !=  49 ||
          a(1,1,1,0) !=  36 || a(1,1,1,1) !=  25 || a(1,1,1,2) !=  16 ||
          a(1,1,2,0) !=   9 || a(1,1,2,1) !=   4 || a(1,1,2,2) !=  1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << a << "\n"
             << "   Expected result:\n((  1  2  3 )\n"
                                     " (  4  5  6 )\n"
                                     " (  7  8  9 ))\n"
                                     "((  9 16 21 )\n"
                                     " ( 24 25 24 )\n"
                                     " ( 21 16  9 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // dense quaternion Schur product assignment
   //=====================================================================================

   {
      test_ = "dense vector Schur product assignment (mixed type)";

      blaze::DynamicArray< 4, int > a{{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                          {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}},
         {{{-1, -2, -3}, {-4, -5, -6}, {-7, -8, -9}},
            {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}}}};

      RT quatslice2 = blaze::quatslice( a, 1UL );

      const blaze::DynamicTensor< short > a1{
         {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      quatslice2 %= a1;

      checkPages   ( quatslice2, 2UL );
      checkRows    ( quatslice2, 3UL );
      checkColumns ( quatslice2, 3UL );
      checkCapacity( quatslice2, 18UL );
      checkNonZeros( quatslice2, 18UL );
      checkQuats   ( a,  2UL );
      checkRows    ( a,  3UL );
      checkColumns ( a,  3UL );
      checkPages   ( a,  2UL );
      checkNonZeros( a, 36UL );

      if( quatslice2(0,0,0) !=  -1 || quatslice2(0,0,1) !=  -4 || quatslice2(0,0,2) != -9  ||
          quatslice2(0,1,0) != -16 || quatslice2(0,1,1) != -25 || quatslice2(0,1,2) != -36 ||
          quatslice2(0,2,0) != -49 || quatslice2(0,2,1) != -64 || quatslice2(0,2,2) != -81 ||
          quatslice2(1,0,0) !=  81 || quatslice2(1,0,1) !=  64 || quatslice2(1,0,2) !=  49 ||
          quatslice2(1,1,0) !=  36 || quatslice2(1,1,1) !=  25 || quatslice2(1,1,2) !=  16 ||
          quatslice2(1,2,0) !=   9 || quatslice2(1,2,1) !=   4 || quatslice2(1,2,2) !=  1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice2 << "\n"
             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( a(1,0,0,0) != -1  || a(1,0,0,1) != -4  || a(1,0,0,2) != -9  ||
          a(1,0,1,0) != -16 || a(1,0,1,1) != -25 || a(1,0,1,2) != -36 ||
          a(1,0,2,0) != -49 || a(1,0,2,1) != -64 || a(1,0,2,2) != -81 ||
          a(1,1,0,0) !=  81 || a(1,1,0,1) !=  64 || a(1,1,0,2) !=  49 ||
          a(1,1,1,0) !=  36 || a(1,1,1,1) !=  25 || a(1,1,1,2) !=  16 ||
          a(1,1,2,0) !=   9 || a(1,1,2,1) !=   4 || a(1,1,2,2) !=  1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << a << "\n"
             << "   Expected result:\n((  1  2  3 )\n"
                                     " (  4  5  6 )\n"
                                     " (  7  8  9 ))\n"
                                     "((  9 16 21 )\n"
                                     " ( 24 25 24 )\n"
                                     " ( 21 16  9 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "dense quaternion Schur product assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      blaze::DynamicArray< 4, int > a{{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                          {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}},
         {{{-1, -2, -3}, {-4, -5, -6}, {-7, -8, -9}},
            {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}}}};

      RT quatslice2 = blaze::quatslice( a, 1UL );

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 96UL ) );
      AlignedPadded a1( memory.get(), 2UL, 3UL, 3UL, 16UL );
      a1(0,0,0) = 1;
      a1(0,0,1) = 2;
      a1(0,0,2) = 3;
      a1(0,1,0) = 4;
      a1(0,1,1) = 5;
      a1(0,1,2) = 6;
      a1(0,2,0) = 7;
      a1(0,2,1) = 8;
      a1(0,2,2) = 9;
      a1(1,0,0) = 9;
      a1(1,0,1) = 8;
      a1(1,0,2) = 7;
      a1(1,1,0) = 6;
      a1(1,1,1) = 5;
      a1(1,1,2) = 4;
      a1(1,2,0) = 3;
      a1(1,2,1) = 2;
      a1(1,2,2) = 1;

      quatslice2 %= a1;

      checkPages   ( quatslice2, 2UL );
      checkRows    ( quatslice2, 3UL );
      checkColumns ( quatslice2, 3UL );
      checkCapacity( quatslice2, 18UL );
      checkNonZeros( quatslice2, 18UL );
      checkQuats   ( a,  2UL );
      checkRows    ( a,  3UL );
      checkColumns ( a,  3UL );
      checkPages   ( a,  2UL );
      checkNonZeros( a, 36UL );

      if( quatslice2(0,0,0) !=  -1 || quatslice2(0,0,1) !=  -4 || quatslice2(0,0,2) != -9  ||
          quatslice2(0,1,0) != -16 || quatslice2(0,1,1) != -25 || quatslice2(0,1,2) != -36 ||
          quatslice2(0,2,0) != -49 || quatslice2(0,2,1) != -64 || quatslice2(0,2,2) != -81 ||
          quatslice2(1,0,0) !=  81 || quatslice2(1,0,1) !=  64 || quatslice2(1,0,2) !=  49 ||
          quatslice2(1,1,0) !=  36 || quatslice2(1,1,1) !=  25 || quatslice2(1,1,2) !=  16 ||
          quatslice2(1,2,0) !=   9 || quatslice2(1,2,1) !=   4 || quatslice2(1,2,2) !=  1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice2 << "\n"
             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( a(1,0,0,0) != -1  || a(1,0,0,1) != -4  || a(1,0,0,2) != -9  ||
          a(1,0,1,0) != -16 || a(1,0,1,1) != -25 || a(1,0,1,2) != -36 ||
          a(1,0,2,0) != -49 || a(1,0,2,1) != -64 || a(1,0,2,2) != -81 ||
          a(1,1,0,0) !=  81 || a(1,1,0,1) !=  64 || a(1,1,0,2) !=  49 ||
          a(1,1,1,0) !=  36 || a(1,1,1,1) !=  25 || a(1,1,1,2) !=  16 ||
          a(1,1,2,0) !=   9 || a(1,1,2,1) !=   4 || a(1,1,2,2) !=  1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << a << "\n"
             << "   Expected result:\n((  1  2  3 )\n"
                                     " (  4  5  6 )\n"
                                     " (  7  8  9 ))\n"
                                     "((  9 16 21 )\n"
                                     " ( 24 25 24 )\n"
                                     " ( 21 16  9 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "dense quaternion Schur product assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      blaze::DynamicArray< 4, int > a{ {{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                          {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}},
         {{{-1, -2, -3}, {-4, -5, -6}, {-7, -8, -9}},
            {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}}} };

      RT quatslice2 = blaze::quatslice(a, 1UL);

      using UnalignedUnpadded = blaze::CustomTensor<int, unaligned, unpadded>;
      std::unique_ptr<int[]> memory(new int[20]);
      UnalignedUnpadded a1(memory.get() + 1UL, 2UL, 3UL, 3UL);
      a1(0, 0, 0) = 1;
      a1(0, 0, 1) = 2;
      a1(0, 0, 2) = 3;
      a1(0, 1, 0) = 4;
      a1(0, 1, 1) = 5;
      a1(0, 1, 2) = 6;
      a1(0, 2, 0) = 7;
      a1(0, 2, 1) = 8;
      a1(0, 2, 2) = 9;
      a1(1, 0, 0) = 9;
      a1(1, 0, 1) = 8;
      a1(1, 0, 2) = 7;
      a1(1, 1, 0) = 6;
      a1(1, 1, 1) = 5;
      a1(1, 1, 2) = 4;
      a1(1, 2, 0) = 3;
      a1(1, 2, 1) = 2;
      a1(1, 2, 2) = 1;

      quatslice2 %= a1;

      checkPages(quatslice2, 2UL);
      checkRows(quatslice2, 3UL);
      checkColumns(quatslice2, 3UL);
      checkCapacity(quatslice2, 18UL);
      checkNonZeros(quatslice2, 18UL);
      checkQuats(a, 2UL);
      checkRows(a, 3UL);
      checkColumns(a, 3UL);
      checkPages(a, 2UL);
      checkNonZeros(a, 36UL);

      if (quatslice2(0, 0, 0) != -1 || quatslice2(0, 0, 1) != -4 || quatslice2(0, 0, 2) != -9 ||
         quatslice2(0, 1, 0) != -16 || quatslice2(0, 1, 1) != -25 || quatslice2(0, 1, 2) != -36 ||
         quatslice2(0, 2, 0) != -49 || quatslice2(0, 2, 1) != -64 || quatslice2(0, 2, 2) != -81 ||
         quatslice2(1, 0, 0) != 81 || quatslice2(1, 0, 1) != 64 || quatslice2(1, 0, 2) != 49 ||
         quatslice2(1, 1, 0) != 36 || quatslice2(1, 1, 1) != 25 || quatslice2(1, 1, 2) != 16 ||
         quatslice2(1, 2, 0) != 9 || quatslice2(1, 2, 1) != 4 || quatslice2(1, 2, 2) != 1) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
            << " Error: Multiplication assignment failed\n"
            << " Details:\n"
            << "   Result:\n" << quatslice2 << "\n"
            << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
         throw std::runtime_error(oss.str());
      }

      if (a(1, 0, 0, 0) != -1 || a(1, 0, 0, 1) != -4 || a(1, 0, 0, 2) != -9 ||
         a(1, 0, 1, 0) != -16 || a(1, 0, 1, 1) != -25 || a(1, 0, 1, 2) != -36 ||
         a(1, 0, 2, 0) != -49 || a(1, 0, 2, 1) != -64 || a(1, 0, 2, 2) != -81 ||
         a(1, 1, 0, 0) != 81 || a(1, 1, 0, 1) != 64 || a(1, 1, 0, 2) != 49 ||
         a(1, 1, 1, 0) != 36 || a(1, 1, 1, 1) != 25 || a(1, 1, 1, 2) != 16 ||
         a(1, 1, 2, 0) != 9 || a(1, 1, 2, 1) != 4 || a(1, 1, 2, 2) != 1) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
            << " Error: Schur assignment failed\n"
            << " Details:\n"
            << "   Result:\n" << a << "\n"
            << "   Expected result:\n((  1  2  3 )\n"
            " (  4  5  6 )\n"
            " (  7  8  9 ))\n"
            "((  9 16 21 )\n"
            " ( 24 25 24 )\n"
            " ( 21 16  9 ))\n";
         throw std::runtime_error(oss.str());
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of all QuatSlice (self-)scaling operations.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all available ways to scale an instance of the QuatSlice
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testScaling()
{
   //=====================================================================================
   // self-scaling (v*=3)
   //=====================================================================================

   {
      test_ = "self-scaling (v*=3)";

      initialize();

      RT quatslice2 = blaze::quatslice( quat_, 1UL );
      quatslice2 *= 3;

      checkPages   ( quatslice2, 2UL );
      checkRows    ( quatslice2, 5UL );
      checkColumns ( quatslice2, 4UL );
      checkCapacity( quatslice2, 40UL );
      checkNonZeros( quatslice2, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   3 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
          quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
          quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  36 || quatslice2(0,2,2) != -9 || quatslice2(0,2,3) !=   0 ||
          quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=  12 || quatslice2(0,3,2) != 15 || quatslice2(0,3,3) != -18 ||
          quatslice2(0,4,0) != 21 || quatslice2(0,4,1) !=  84 || quatslice2(0,4,2) != 27 || quatslice2(0,4,3) !=  30 ||
          quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
          quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   3 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
          quatslice2(1,2,0) != -6 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
          quatslice2(1,3,0) != -9 || quatslice2(1,3,1) !=  12 || quatslice2(1,3,2) != 15 || quatslice2(1,3,3) !=  99 ||
          quatslice2(1,4,0) != 21 || quatslice2(1,4,1) != -24 || quatslice2(1,4,2) != 27 || quatslice2(1,4,3) !=  33) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // self-scaling (v=v*2)
   //=====================================================================================

   {
      test_ = "self-scaling (v=v*3)";

      initialize();

      RT quatslice2 = blaze::quatslice( quat_, 1UL );
      quatslice2 = quatslice2 * 3;

      checkPages   ( quatslice2, 2UL );
      checkRows    ( quatslice2, 5UL );
      checkColumns ( quatslice2, 4UL );
      checkCapacity( quatslice2, 40UL );
      checkNonZeros( quatslice2, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   3 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
          quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
          quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  36 || quatslice2(0,2,2) != -9 || quatslice2(0,2,3) !=   0 ||
          quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=  12 || quatslice2(0,3,2) != 15 || quatslice2(0,3,3) != -18 ||
          quatslice2(0,4,0) != 21 || quatslice2(0,4,1) !=  84 || quatslice2(0,4,2) != 27 || quatslice2(0,4,3) !=  30 ||
          quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
          quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   3 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
          quatslice2(1,2,0) != -6 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
          quatslice2(1,3,0) != -9 || quatslice2(1,3,1) !=  12 || quatslice2(1,3,2) != 15 || quatslice2(1,3,3) !=  99 ||
          quatslice2(1,4,0) != 21 || quatslice2(1,4,1) != -24 || quatslice2(1,4,2) != 27 || quatslice2(1,4,3) !=  33) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // self-scaling (v=3*v)
   //=====================================================================================

   {
      test_ = "self-scaling (v=3*v)";

      initialize();

      RT quatslice2 = blaze::quatslice( quat_, 1UL );
      quatslice2 = 3 * quatslice2;

      checkPages   ( quatslice2, 2UL );
      checkRows    ( quatslice2, 5UL );
      checkColumns ( quatslice2, 4UL );
      checkCapacity( quatslice2, 40UL );
      checkNonZeros( quatslice2, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   3 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
          quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
          quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  36 || quatslice2(0,2,2) != -9 || quatslice2(0,2,3) !=   0 ||
          quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=  12 || quatslice2(0,3,2) != 15 || quatslice2(0,3,3) != -18 ||
          quatslice2(0,4,0) != 21 || quatslice2(0,4,1) !=  84 || quatslice2(0,4,2) != 27 || quatslice2(0,4,3) !=  30 ||
          quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
          quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   3 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
          quatslice2(1,2,0) != -6 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
          quatslice2(1,3,0) != -9 || quatslice2(1,3,1) !=  12 || quatslice2(1,3,2) != 15 || quatslice2(1,3,3) !=  99 ||
          quatslice2(1,4,0) != 21 || quatslice2(1,4,1) != -24 || quatslice2(1,4,2) != 27 || quatslice2(1,4,3) !=  33) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // self-scaling (v/=s)
   //=====================================================================================

   {
      test_ = "self-scaling (v/=s)";

      initialize();

      RT quatslice2 = blaze::quatslice( quat_, 1UL );
      quatslice2 /= (1.0/3.0);

      checkPages   ( quatslice2, 2UL );
      checkRows    ( quatslice2, 5UL );
      checkColumns ( quatslice2, 4UL );
      checkCapacity( quatslice2, 40UL );
      checkNonZeros( quatslice2, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   3 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
          quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
          quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  36 || quatslice2(0,2,2) != -9 || quatslice2(0,2,3) !=   0 ||
          quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=  12 || quatslice2(0,3,2) != 15 || quatslice2(0,3,3) != -18 ||
          quatslice2(0,4,0) != 21 || quatslice2(0,4,1) !=  84 || quatslice2(0,4,2) != 27 || quatslice2(0,4,3) !=  30 ||
          quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
          quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   3 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
          quatslice2(1,2,0) != -6 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
          quatslice2(1,3,0) != -9 || quatslice2(1,3,1) !=  12 || quatslice2(1,3,2) != 15 || quatslice2(1,3,3) !=  99 ||
          quatslice2(1,4,0) != 21 || quatslice2(1,4,1) != -24 || quatslice2(1,4,2) != 27 || quatslice2(1,4,3) !=  33) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // self-scaling (v=v/s)
   //=====================================================================================

   {
      test_ = "self-scaling (v=v/s)";

      initialize();

      RT quatslice2 = blaze::quatslice( quat_, 1UL );
      quatslice2 = quatslice2 / (1.0/3.0);

            checkPages   ( quatslice2, 2UL );
      checkRows    ( quatslice2, 5UL );
      checkColumns ( quatslice2, 4UL );
      checkCapacity( quatslice2, 40UL );
      checkNonZeros( quatslice2, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   3 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
          quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
          quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  36 || quatslice2(0,2,2) != -9 || quatslice2(0,2,3) !=   0 ||
          quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=  12 || quatslice2(0,3,2) != 15 || quatslice2(0,3,3) != -18 ||
          quatslice2(0,4,0) != 21 || quatslice2(0,4,1) !=  84 || quatslice2(0,4,2) != 27 || quatslice2(0,4,3) !=  30 ||
          quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
          quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   3 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
          quatslice2(1,2,0) != -6 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
          quatslice2(1,3,0) != -9 || quatslice2(1,3,1) !=  12 || quatslice2(1,3,2) != 15 || quatslice2(1,3,3) !=  99 ||
          quatslice2(1,4,0) != 21 || quatslice2(1,4,1) != -24 || quatslice2(1,4,2) != 27 || quatslice2(1,4,3) !=  33) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // QuatSlice::scale()
   //=====================================================================================

   {
      test_ = "QuatSlice::scale()";

      initialize();

      // Integral scaling the 2nd quatslice
      {
         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         quatslice2.scale( 3 );

         checkPages   ( quatslice2, 2UL );
         checkRows    ( quatslice2, 5UL );
         checkColumns ( quatslice2, 4UL );
         checkCapacity( quatslice2, 40UL );
         checkNonZeros( quatslice2, 20UL );
         checkPages   ( quat_,  2UL );
         checkRows    ( quat_,  5UL );
         checkColumns ( quat_,  4UL );
         checkQuats   ( quat_,  3UL );

         if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   3 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
             quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
             quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  36 || quatslice2(0,2,2) != -9 || quatslice2(0,2,3) !=   0 ||
             quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=  12 || quatslice2(0,3,2) != 15 || quatslice2(0,3,3) != -18 ||
             quatslice2(0,4,0) != 21 || quatslice2(0,4,1) !=  84 || quatslice2(0,4,2) != 27 || quatslice2(0,4,3) !=  30 ||
             quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
             quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   3 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
             quatslice2(1,2,0) != -6 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
             quatslice2(1,3,0) != -9 || quatslice2(1,3,1) !=  12 || quatslice2(1,3,2) != 15 || quatslice2(1,3,3) !=  99 ||
             quatslice2(1,4,0) != 21 || quatslice2(1,4,1) != -24 || quatslice2(1,4,2) != 27 || quatslice2(1,4,3) !=  33) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Result:\n" << quatslice2 << "\n"
                   << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      initialize();

      // Floating point scaling the 2nd quatslice
      {
         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         quatslice2.scale(0.5);

         checkPages   ( quatslice2, 2UL );
         checkRows    ( quatslice2, 5UL );
         checkColumns ( quatslice2, 4UL );
         checkCapacity( quatslice2, 40UL );
         checkNonZeros( quatslice2, 18UL );
         checkPages   ( quat_,  2UL );
         checkRows    ( quat_,  5UL );
         checkColumns ( quat_,  4UL );
         checkQuats   ( quat_,  3UL );

         if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=  0 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
             quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
             quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=   6 || quatslice2(0,2,2) != -1 || quatslice2(0,2,3) !=   0 ||
             quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=   2 || quatslice2(0,3,2) !=  2 || quatslice2(0,3,3) !=  -3 ||
             quatslice2(0,4,0) !=  3 || quatslice2(0,4,1) !=  14 || quatslice2(0,4,2) !=  4 || quatslice2(0,4,3) !=  5 ||
             quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
             quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   0 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
             quatslice2(1,2,0) != -1 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
             quatslice2(1,3,0) != -1 || quatslice2(1,3,1) !=  2 || quatslice2(1,3,2) != 2 || quatslice2(1,3,3) !=  16 ||
             quatslice2(1,4,0) !=  3 || quatslice2(1,4,1) != -4 || quatslice2(1,4,2) != 4 || quatslice2(1,4,3) !=  5) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Result:\n" << quatslice2 << "\n"
                   << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the QuatSlice function call operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the function call operator
// of the QuatSlice specialization. In case an error is detected, a \a std::runtime_error exception
// is thrown.
*/
void DenseGeneralTest::testFunctionCall()
{
   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   {
      test_ = "QuatSlice::operator()";

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 1UL );

      // Assignment to the element at index (0,1)
      quatslice1(0,1,2) = 9;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 21UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  9 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  5 || quatslice1(1,3,3) != 33 ||
          quatslice1(1,4,0) !=  7 || quatslice1(1,4,1) != -8 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Assignment to the element at index (2,2)
      quatslice1(1,3,2) = 0;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  9 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) != 33 ||
          quatslice1(1,4,0) !=  7 || quatslice1(1,4,1) != -8 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Assignment to the element at index (4,1)
      quatslice1(1,4,1) = -9;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  1 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  9 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) != 33 ||
          quatslice1(1,4,0) !=  7 || quatslice1(1,4,1) != -9 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }


      // Addition assignment to the element at index (0,1)
      quatslice1(0,0,1) += -3;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) != -2 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  9 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) != 33 ||
          quatslice1(1,4,0) !=  7 || quatslice1(1,4,1) != -9 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Subtraction assignment to the element at index (2,0)
      quatslice1(0,2,0) -= 6;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 21UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) != -2 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  9 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) != -6 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) != 33 ||
          quatslice1(1,4,0) !=  7 || quatslice1(1,4,1) != -9 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Multiplication assignment to the element at index (4,0)
      quatslice1(1,4,0) *= -3;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 21UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) != -2 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  9 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) != -6 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) != 33 ||
          quatslice1(1,4,0) !=-21 || quatslice1(1,4,1) != -9 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Division assignment to the element at index (3,3)
      quatslice1(1,3,3) /= 2;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 21UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) != -2 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  9 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) != -6 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) != 16 ||
          quatslice1(1,4,0) !=-21 || quatslice1(1,4,1) != -9 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the QuatSlice at() operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the at() operator
// of the QuatSlice specialization. In case an error is detected, a \a std::runtime_error exception
// is thrown.
*/
void DenseGeneralTest::testAt()
{
   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   {
      test_ = "QuatSlice::at()";

      initialize();

      RT quatslice1 = blaze::quatslice( quat_, 1UL );

      // Assignment to the element at index (0,1)
      quatslice1.at(0,0,1) = 9;

         checkPages   ( quatslice1, 2UL  );
         checkRows    ( quatslice1, 5UL );
         checkColumns ( quatslice1, 4UL );
         checkCapacity( quatslice1, 40UL );
         checkNonZeros( quatslice1, 20UL );

         if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  9 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
             quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=  0 ||
             quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
             quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
             quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
             quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
             quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
             quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
             quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  5 || quatslice1(1,3,3) != 33 ||
             quatslice1(1,4,0) !=  7 || quatslice1(1,4,1) != -8 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of 1st dense quatslice failed\n"
                << " Details:\n"
                << "   Result:\n"
                << quatslice1 << "\n"
                << "   Expected result:\n((     0   1   0   0 ) (     0   0   "
                   "0   0 ) (     0     12     -3   0 ) (     0   4   5     -6 "
                   ") (     7     28   9     10 ) )\n((     0   0   0   0 ) (  "
                   "   0   1   0   0 ) (    -2   0   0   0 ) (    -3   4   5   "
                   "  33 ) (     7     -8   9     11 ) )\n";
            throw std::runtime_error( oss.str() );
         }

      // Assignment to the element at index (2,2)
      quatslice1.at(1,3,2) = 0;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 19UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  9 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) != 33 ||
          quatslice1(1,4,0) !=  7 || quatslice1(1,4,1) != -8 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Assignment to the element at index (4,1)
      quatslice1.at(1,4,1) = -9;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 19UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  9 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) != 33 ||
          quatslice1(1,4,0) !=  7 || quatslice1(1,4,1) != -9 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }


      // Addition assignment to the element at index (0,1)
      quatslice1.at(0,0,1) += -3;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 19UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  6 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) !=  0 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) != 33 ||
          quatslice1(1,4,0) !=  7 || quatslice1(1,4,1) != -9 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Subtraction assignment to the element at index (2,0)
      quatslice1.at(0,2,0) -= 6;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  6 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) != -6 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) != 33 ||
          quatslice1(1,4,0) !=  7 || quatslice1(1,4,1) != -9 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Multiplication assignment to the element at index (4,0)
      quatslice1.at(1,4,0) *= -3;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  6 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) != -6 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) != 33 ||
          quatslice1(1,4,0) !=-21 || quatslice1(1,4,1) != -9 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Division assignment to the element at index (3,3)
      quatslice1.at(1,3,3) /= 2;

      checkPages   ( quatslice1, 2UL  );
      checkRows    ( quatslice1, 5UL );
      checkColumns ( quatslice1, 4UL );
      checkCapacity( quatslice1, 40UL );
      checkNonZeros( quatslice1, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice1(0,0,0) !=  0 || quatslice1(0,0,1) !=  6 || quatslice1(0,0,2) !=  0 || quatslice1(0,0,3) !=  0 ||
          quatslice1(0,1,0) !=  0 || quatslice1(0,1,1) !=  0 || quatslice1(0,1,2) !=  0 || quatslice1(0,1,3) !=  0 ||
          quatslice1(0,2,0) != -6 || quatslice1(0,2,1) != 12 || quatslice1(0,2,2) != -3 || quatslice1(0,2,3) !=  0 ||
          quatslice1(0,3,0) !=  0 || quatslice1(0,3,1) !=  4 || quatslice1(0,3,2) !=  5 || quatslice1(0,3,3) != -6 ||
          quatslice1(0,4,0) !=  7 || quatslice1(0,4,1) != 28 || quatslice1(0,4,2) !=  9 || quatslice1(0,4,3) != 10 ||
          quatslice1(1,0,0) !=  0 || quatslice1(1,0,1) !=  0 || quatslice1(1,0,2) !=  0 || quatslice1(1,0,3) !=  0 ||
          quatslice1(1,1,0) !=  0 || quatslice1(1,1,1) !=  1 || quatslice1(1,1,2) !=  0 || quatslice1(1,1,3) !=  0 ||
          quatslice1(1,2,0) != -2 || quatslice1(1,2,1) !=  0 || quatslice1(1,2,2) !=  0 || quatslice1(1,2,3) !=  0 ||
          quatslice1(1,3,0) != -3 || quatslice1(1,3,1) !=  4 || quatslice1(1,3,2) !=  0 || quatslice1(1,3,3) != 16 ||
          quatslice1(1,4,0) !=-21 || quatslice1(1,4,1) != -9 || quatslice1(1,4,2) !=  9 || quatslice1(1,4,3) != 11) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice1 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the QuatSlice iterator implementation.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the iterator implementation of the QuatSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testIterator()
{
   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   {
      initialize();

      // Testing the Iterator default constructor
      {
         test_ = "Iterator default constructor";

         RT::Iterator it{};

         if( it != RT::Iterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing the ConstIterator default constructor
      {
         test_ = "ConstIterator default constructor";

         RT::ConstIterator it{};

         if( it != RT::ConstIterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing conversion from Iterator to ConstIterator
      {
         test_ = "Iterator/ConstIterator conversion";

         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         RT::ConstIterator it( begin( quatslice2, 2UL, 1UL ) );

         if( it == end( quatslice2, 2UL, 1UL ) || *it != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st quatslice via Iterator (end-begin)
      {
         test_ = "Iterator subtraction (end-begin)";

         RT quatslice1 = blaze::quatslice( quat_, 1UL );
         const ptrdiff_t number( end( quatslice1, 2UL, 1UL ) - begin( quatslice1, 2UL, 1UL ) );

         if( number != 4L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st quatslice via Iterator (begin-end)
      {
         test_ = "Iterator subtraction (begin-end)";

         RT quatslice1 = blaze::quatslice( quat_, 1UL );
         const ptrdiff_t number( begin( quatslice1, 2UL, 1UL ) - end( quatslice1, 2UL, 1UL ) );

         if( number != -4L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: -4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 2nd quatslice via ConstIterator (end-begin)
      {
         test_ = "ConstIterator subtraction (end-begin)";

         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         const ptrdiff_t number( cend( quatslice2, 2UL, 1UL ) - cbegin( quatslice2, 2UL, 1UL ) );

         if( number != 4L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 2nd quatslice via ConstIterator (begin-end)
      {
         test_ = "ConstIterator subtraction (begin-end)";

         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         const ptrdiff_t number( cbegin( quatslice2, 2UL, 1UL ) - cend( quatslice2, 2UL, 1UL ) );

         if( number != -4L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: -4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing read-only access via ConstIterator
      {
         test_ = "read-only access via ConstIterator";

         RT quatslice3 = blaze::quatslice( quat_, 0UL );
         RT::ConstIterator it ( cbegin( quatslice3, 4UL, 0UL ) );
         RT::ConstIterator end( cend( quatslice3, 4UL, 0UL ) );

         if( it == end || *it != 7 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || *it != -8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         --it;

         if( it == end || *it != 7 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it++;

         if( it == end || *it != -8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it--;

         if( it == end || *it != 7 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it += 2UL;

         if( it == end || *it != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator addition assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it -= 2UL;

         if( it == end || *it != 7 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator subtraction assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it + 3UL;

         if( it == end || *it != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar addition failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it - 3UL;

         if( it == end || *it != 7 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar subtraction failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = 4UL + it;

         if( it != end ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Scalar/iterator addition failed\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing assignment via Iterator
      {
         test_ = "assignment via Iterator";

         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         int value = 6;

         for( RT::Iterator it=begin( quatslice2, 3UL, 0UL ); it!=end( quatslice2, 3UL, 0UL ); ++it ) {
            *it = value++;
         }

         if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   1 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
             quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
             quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  12 || quatslice2(0,2,2) != -3 || quatslice2(0,2,3) !=   0 ||
             quatslice2(0,3,0) !=  6 || quatslice2(0,3,1) !=   7 || quatslice2(0,3,2) !=  8 || quatslice2(0,3,3) !=   9 ||
             quatslice2(0,4,0) !=  7 || quatslice2(0,4,1) !=  28 || quatslice2(0,4,2) !=  9 || quatslice2(0,4,3) !=  10 ||
             quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
             quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   1 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
             quatslice2(1,2,0) != -2 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
             quatslice2(1,3,0) != -3 || quatslice2(1,3,1) !=   4 || quatslice2(1,3,2) !=  5 || quatslice2(1,3,3) !=  33 ||
             quatslice2(1,4,0) !=  7 || quatslice2(1,4,1) !=  -8 || quatslice2(1,4,2) !=  9 || quatslice2(1,4,3) !=  11) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 6 7 8 9 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "addition assignment via Iterator";

         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         int value = 2;

         for( RT::Iterator it=begin( quatslice2, 3UL, 0UL ); it!=end( quatslice2, 3UL, 0UL  ); ++it ) {
            *it += value++;
         }

         if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   1 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
             quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
             quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  12 || quatslice2(0,2,2) != -3 || quatslice2(0,2,3) !=   0 ||
             quatslice2(0,3,0) !=  8 || quatslice2(0,3,1) !=  10 || quatslice2(0,3,2) != 12 || quatslice2(0,3,3) !=  14 ||
             quatslice2(0,4,0) !=  7 || quatslice2(0,4,1) !=  28 || quatslice2(0,4,2) !=  9 || quatslice2(0,4,3) !=  10 ||
             quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
             quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   1 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
             quatslice2(1,2,0) != -2 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
             quatslice2(1,3,0) != -3 || quatslice2(1,3,1) !=   4 || quatslice2(1,3,2) !=  5 || quatslice2(1,3,3) !=  33 ||
             quatslice2(1,4,0) !=  7 || quatslice2(1,4,1) !=  -8 || quatslice2(1,4,2) !=  9 || quatslice2(1,4,3) !=  11 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 8 10 12 14 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "subtraction assignment via Iterator";

         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         int value = 2;

         for( RT::Iterator it=begin( quatslice2, 3UL, 0UL ); it!=end( quatslice2, 3UL, 0UL ); ++it ) {
            *it -= value++;
         }

         if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   1 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
             quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
             quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  12 || quatslice2(0,2,2) != -3 || quatslice2(0,2,3) !=   0 ||
             quatslice2(0,3,0) !=  6 || quatslice2(0,3,1) !=   7 || quatslice2(0,3,2) !=  8 || quatslice2(0,3,3) !=   9 ||
             quatslice2(0,4,0) !=  7 || quatslice2(0,4,1) !=  28 || quatslice2(0,4,2) !=  9 || quatslice2(0,4,3) !=  10 ||
             quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
             quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   1 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
             quatslice2(1,2,0) != -2 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
             quatslice2(1,3,0) != -3 || quatslice2(1,3,1) !=   4 || quatslice2(1,3,2) !=  5 || quatslice2(1,3,3) !=  33 ||
             quatslice2(1,4,0) !=  7 || quatslice2(1,4,1) !=  -8 || quatslice2(1,4,2) !=  9 || quatslice2(1,4,3) !=  11 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 6 7 8 9 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "multiplication assignment via Iterator";

         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         int value = 1;

         for( RT::Iterator it=begin( quatslice2, 3UL, 0UL ); it!=end( quatslice2, 3UL, 0UL ); ++it ) {
            *it *= value++;
         }

         if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   1 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
             quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
             quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  12 || quatslice2(0,2,2) != -3 || quatslice2(0,2,3) !=   0 ||
             quatslice2(0,3,0) !=  6 || quatslice2(0,3,1) !=  14 || quatslice2(0,3,2) != 24 || quatslice2(0,3,3) !=  36 ||
             quatslice2(0,4,0) !=  7 || quatslice2(0,4,1) !=  28 || quatslice2(0,4,2) !=  9 || quatslice2(0,4,3) !=  10 ||
             quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
             quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   1 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
             quatslice2(1,2,0) != -2 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
             quatslice2(1,3,0) != -3 || quatslice2(1,3,1) !=   4 || quatslice2(1,3,2) !=  5 || quatslice2(1,3,3) !=  33 ||
             quatslice2(1,4,0) !=  7 || quatslice2(1,4,1) !=  -8 || quatslice2(1,4,2) !=  9 || quatslice2(1,4,3) !=  11 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 6 14 24 36 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "division assignment via Iterator";

         RT quatslice2 = blaze::quatslice( quat_, 1UL );

         for( RT::Iterator it=begin( quatslice2, 3UL, 0UL ); it!=end( quatslice2, 3UL, 0UL ); ++it ) {
            *it /= 2;
         }

         if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   1 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
             quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
             quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  12 || quatslice2(0,2,2) != -3 || quatslice2(0,2,3) !=   0 ||
             quatslice2(0,3,0) !=  3 || quatslice2(0,3,1) !=   7 || quatslice2(0,3,2) != 12 || quatslice2(0,3,3) !=  18 ||
             quatslice2(0,4,0) !=  7 || quatslice2(0,4,1) !=  28 || quatslice2(0,4,2) !=  9 || quatslice2(0,4,3) !=  10 ||
             quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
             quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   1 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
             quatslice2(1,2,0) != -2 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
             quatslice2(1,3,0) != -3 || quatslice2(1,3,1) !=   4 || quatslice2(1,3,2) !=  5 || quatslice2(1,3,3) !=  33 ||
             quatslice2(1,4,0) !=  7 || quatslice2(1,4,1) !=  -8 || quatslice2(1,4,2) !=  9 || quatslice2(1,4,3) !=  11 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 3 7 12 18 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c nonZeros() member function of the QuatSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c nonZeros() member function of the QuatSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testNonZeros()
{
   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   {
      test_ = "QuatSlice::nonZeros()";

      initialize();

      // Initialization check
      RT quatslice2 = blaze::quatslice( quat_, 1UL );

      checkPages   ( quatslice2, 2UL );
      checkRows    ( quatslice2, 5UL );
      checkColumns ( quatslice2, 4UL );
      checkCapacity( quatslice2, 40UL );
      checkNonZeros( quatslice2, 20UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   1 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
          quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
          quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  12 || quatslice2(0,2,2) != -3 || quatslice2(0,2,3) !=   0 ||
          quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=   4 || quatslice2(0,3,2) !=  5 || quatslice2(0,3,3) !=  -6 ||
          quatslice2(0,4,0) !=  7 || quatslice2(0,4,1) !=  28 || quatslice2(0,4,2) !=  9 || quatslice2(0,4,3) !=  10 ||
          quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
          quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   1 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
          quatslice2(1,2,0) != -2 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
          quatslice2(1,3,0) != -3 || quatslice2(1,3,1) !=   4 || quatslice2(1,3,2) !=  5 || quatslice2(1,3,3) !=  33 ||
          quatslice2(1,4,0) !=  7 || quatslice2(1,4,1) !=  -8 || quatslice2(1,4,2) !=  9 || quatslice2(1,4,3) !=  11 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the dense quatslice
      quatslice2(1, 3, 2) = 0;

      checkPages   ( quatslice2, 2UL );
      checkRows    ( quatslice2, 5UL );
      checkColumns ( quatslice2, 4UL );
      checkCapacity( quatslice2, 40UL );
      checkNonZeros( quatslice2, 19UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   1 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
          quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
          quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  12 || quatslice2(0,2,2) != -3 || quatslice2(0,2,3) !=   0 ||
          quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=   4 || quatslice2(0,3,2) !=  5 || quatslice2(0,3,3) !=  -6 ||
          quatslice2(0,4,0) !=  7 || quatslice2(0,4,1) !=  28 || quatslice2(0,4,2) !=  9 || quatslice2(0,4,3) !=  10 ||
          quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
          quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   1 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
          quatslice2(1,2,0) != -2 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
          quatslice2(1,3,0) != -3 || quatslice2(1,3,1) !=   4 || quatslice2(1,3,2) !=  0 || quatslice2(1,3,3) !=  33 ||
          quatslice2(1,4,0) !=  7 || quatslice2(1,4,1) !=  -8 || quatslice2(1,4,2) !=  9 || quatslice2(1,4,3) !=  11  ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the dense quaternion
      quat_(1,1,3,0) = 0;

      checkPages   ( quatslice2, 2UL );
      checkRows    ( quatslice2, 5UL );
      checkColumns ( quatslice2, 4UL );
      checkCapacity( quatslice2, 40UL );
      checkNonZeros( quatslice2, 18UL );
      checkPages   ( quat_,  2UL );
      checkRows    ( quat_,  5UL );
      checkColumns ( quat_,  4UL );
      checkQuats   ( quat_,  3UL );

      if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   1 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
          quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
          quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  12 || quatslice2(0,2,2) != -3 || quatslice2(0,2,3) !=   0 ||
          quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=   4 || quatslice2(0,3,2) !=  5 || quatslice2(0,3,3) !=  -6 ||
          quatslice2(0,4,0) !=  7 || quatslice2(0,4,1) !=  28 || quatslice2(0,4,2) !=  9 || quatslice2(0,4,3) !=  10 ||
          quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
          quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   1 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
          quatslice2(1,2,0) != -2 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
          quatslice2(1,3,0) !=  0 || quatslice2(1,3,1) !=   4 || quatslice2(1,3,2) !=  0 || quatslice2(1,3,3) !=  33 ||
          quatslice2(1,4,0) !=  7 || quatslice2(1,4,1) !=  -8 || quatslice2(1,4,2) !=  9 || quatslice2(1,4,3) !=  11 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Matrix function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << quatslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 5 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reset() member function of the QuatSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reset() member function of the QuatSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testReset()
{
   using blaze::reset;


   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   {
      test_ = "QuatSlice::reset()";

      // Resetting a single element in quatslice 3
      {
         initialize();

         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         reset( quatslice2(0,4,3) );

         checkPages   ( quatslice2, 2UL );
         checkRows    ( quatslice2, 5UL );
         checkColumns ( quatslice2, 4UL );
         checkCapacity( quatslice2, 40UL );
         checkNonZeros( quatslice2, 19UL );
         checkPages   ( quat_,  2UL );
         checkRows    ( quat_,  5UL );
         checkColumns ( quat_,  4UL );
         checkQuats   ( quat_,  3UL );

         if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   1 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
             quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
             quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  12 || quatslice2(0,2,2) != -3 || quatslice2(0,2,3) !=   0 ||
             quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=   4 || quatslice2(0,3,2) !=  5 || quatslice2(0,3,3) !=  -6 ||
             quatslice2(0,4,0) !=  7 || quatslice2(0,4,1) !=  28 || quatslice2(0,4,2) !=  9 || quatslice2(0,4,3) !=   0 ||
             quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
             quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   1 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
             quatslice2(1,2,0) != -2 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
             quatslice2(1,3,0) != -3 || quatslice2(1,3,1) !=   4 || quatslice2(1,3,2) !=  5 || quatslice2(1,3,3) !=  33 ||
             quatslice2(1,4,0) !=  7 || quatslice2(1,4,1) !=  -8 || quatslice2(1,4,2) !=  9 || quatslice2(1,4,3) !=  11 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operator failed\n"
                << " Details:\n"
                << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Resetting the 1st quatslice (lvalue)
      {
         initialize();

         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         reset( quatslice2 );

         checkPages   ( quatslice2, 2UL );
         checkRows    ( quatslice2, 5UL );
         checkColumns ( quatslice2, 4UL );
         checkCapacity( quatslice2, 40UL );
         checkNonZeros( quatslice2, 0UL );
         checkPages   ( quat_,  2UL );
         checkRows    ( quat_,  5UL );
         checkColumns ( quat_,  4UL );
         checkQuats   ( quat_,  3UL );

         if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   0 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=  0 ||
             quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=  0 ||
             quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=   0 || quatslice2(0,2,2) !=  0 || quatslice2(0,2,3) !=  0 ||
             quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=   0 || quatslice2(0,3,2) !=  0 || quatslice2(0,3,3) !=  0 ||
             quatslice2(0,4,0) !=  0 || quatslice2(0,4,1) !=   0 || quatslice2(0,4,2) !=  0 || quatslice2(0,4,3) !=  0 ||
             quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=  0 ||
             quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   0 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=  0 ||
             quatslice2(1,2,0) !=  0 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=  0 ||
             quatslice2(1,3,0) !=  0 || quatslice2(1,3,1) !=   0 || quatslice2(1,3,2) !=  0 || quatslice2(1,3,3) !=  0 ||
             quatslice2(1,4,0) !=  0 || quatslice2(1,4,1) !=   0 || quatslice2(1,4,2) !=  0 || quatslice2(1,4,3) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 1st quatslice failed\n"
                << " Details:\n"
                << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Resetting the 1st quatslice (rvalue)
      {
         initialize();

         reset( blaze::quatslice( quat_, 1UL ) );

         checkPages   ( quat_,  2UL );
         checkRows    ( quat_,  5UL );
         checkColumns ( quat_,  4UL );
         checkQuats   ( quat_,  3UL );

         if( quat_(1,0,0,0) !=  0 || quat_(1,0,0,1) !=   0 || quat_(1,0,0,2) !=  0 || quat_(1,0,0,3) !=  0 ||
             quat_(1,0,1,0) !=  0 || quat_(1,0,1,1) !=   0 || quat_(1,0,1,2) !=  0 || quat_(1,0,1,3) !=  0 ||
             quat_(1,0,2,0) !=  0 || quat_(1,0,2,1) !=   0 || quat_(1,0,2,2) !=  0 || quat_(1,0,2,3) !=  0 ||
             quat_(1,0,3,0) !=  0 || quat_(1,0,3,1) !=   0 || quat_(1,0,3,2) !=  0 || quat_(1,0,3,3) !=  0 ||
             quat_(1,0,4,0) !=  0 || quat_(1,0,4,1) !=   0 || quat_(1,0,4,2) !=  0 || quat_(1,0,4,3) !=  0 ||
             quat_(1,1,0,0) !=  0 || quat_(1,1,0,1) !=   0 || quat_(1,1,0,2) !=  0 || quat_(1,1,0,3) !=  0 ||
             quat_(1,1,1,0) !=  0 || quat_(1,1,1,1) !=   0 || quat_(1,1,1,2) !=  0 || quat_(1,1,1,3) !=  0 ||
             quat_(1,1,2,0) !=  0 || quat_(1,1,2,1) !=   0 || quat_(1,1,2,2) !=  0 || quat_(1,1,2,3) !=  0 ||
             quat_(1,1,3,0) !=  0 || quat_(1,1,3,1) !=   0 || quat_(1,1,3,2) !=  0 || quat_(1,1,3,3) !=  0 ||
             quat_(1,1,4,0) !=  0 || quat_(1,1,4,1) !=   0 || quat_(1,1,4,2) !=  0 || quat_(1,1,4,3) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 1st quatslice failed\n"
                << " Details:\n"
                << "   Result:\n" << quat_ << "\n"
                << "   Expected result:\n((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0  -3   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  7  -8   9  10 ))\n"
                                        "((  0   0   0   0 )\n"
                                        " (  0   0   0   0 )\n"
                                        " (  0   0   0   0 )\n"
                                        " (  0   0   0   0 )\n"
                                        " (  0   0   0   0 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c clear() function with the QuatSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c clear() function with the QuatSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testClear()
{
   using blaze::clear;


   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   {
      test_ = "clear() function";

      // Clearing a single element in quatslice 1
      {
         initialize();

         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         clear( quatslice2(0,4,3) );

         checkPages   ( quatslice2, 2UL );
         checkRows    ( quatslice2, 5UL );
         checkColumns ( quatslice2, 4UL );
         checkCapacity( quatslice2, 40UL );
         checkNonZeros( quatslice2, 19UL );
         checkPages   ( quat_,  2UL );
         checkRows    ( quat_,  5UL );
         checkColumns ( quat_,  4UL );
         checkQuats   ( quat_,  3UL );

         if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   1 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=   0 ||
             quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=   0 ||
             quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=  12 || quatslice2(0,2,2) != -3 || quatslice2(0,2,3) !=   0 ||
             quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=   4 || quatslice2(0,3,2) !=  5 || quatslice2(0,3,3) !=  -6 ||
             quatslice2(0,4,0) !=  7 || quatslice2(0,4,1) !=  28 || quatslice2(0,4,2) !=  9 || quatslice2(0,4,3) !=   0 ||
             quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=   0 ||
             quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   1 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=   0 ||
             quatslice2(1,2,0) != -2 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=   0 ||
             quatslice2(1,3,0) != -3 || quatslice2(1,3,1) !=   4 || quatslice2(1,3,2) !=  5 || quatslice2(1,3,3) !=  33 ||
             quatslice2(1,4,0) !=  7 || quatslice2(1,4,1) !=  -8 || quatslice2(1,4,2) !=  9 || quatslice2(1,4,3) !=  11 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operator failed\n"
                << " Details:\n"
                << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Clearing the 3rd quatslice (lvalue)
      {
         initialize();

         RT quatslice2 = blaze::quatslice( quat_, 1UL );
         clear( quatslice2 );

         checkPages   ( quatslice2, 2UL );
         checkRows    ( quatslice2, 5UL );
         checkColumns ( quatslice2, 4UL );
         checkCapacity( quatslice2, 40UL );
         checkNonZeros( quatslice2, 0UL );
         checkPages   ( quat_,  2UL );
         checkRows    ( quat_,  5UL );
         checkColumns ( quat_,  4UL );
         checkQuats   ( quat_,  3UL );

         if( quatslice2(0,0,0) !=  0 || quatslice2(0,0,1) !=   0 || quatslice2(0,0,2) !=  0 || quatslice2(0,0,3) !=  0 ||
             quatslice2(0,1,0) !=  0 || quatslice2(0,1,1) !=   0 || quatslice2(0,1,2) !=  0 || quatslice2(0,1,3) !=  0 ||
             quatslice2(0,2,0) !=  0 || quatslice2(0,2,1) !=   0 || quatslice2(0,2,2) !=  0 || quatslice2(0,2,3) !=  0 ||
             quatslice2(0,3,0) !=  0 || quatslice2(0,3,1) !=   0 || quatslice2(0,3,2) !=  0 || quatslice2(0,3,3) !=  0 ||
             quatslice2(0,4,0) !=  0 || quatslice2(0,4,1) !=   0 || quatslice2(0,4,2) !=  0 || quatslice2(0,4,3) !=  0 ||
             quatslice2(1,0,0) !=  0 || quatslice2(1,0,1) !=   0 || quatslice2(1,0,2) !=  0 || quatslice2(1,0,3) !=  0 ||
             quatslice2(1,1,0) !=  0 || quatslice2(1,1,1) !=   0 || quatslice2(1,1,2) !=  0 || quatslice2(1,1,3) !=  0 ||
             quatslice2(1,2,0) !=  0 || quatslice2(1,2,1) !=   0 || quatslice2(1,2,2) !=  0 || quatslice2(1,2,3) !=  0 ||
             quatslice2(1,3,0) !=  0 || quatslice2(1,3,1) !=   0 || quatslice2(1,3,2) !=  0 || quatslice2(1,3,3) !=  0 ||
             quatslice2(1,4,0) !=  0 || quatslice2(1,4,1) !=   0 || quatslice2(1,4,2) !=  0 || quatslice2(1,4,3) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 1st quatslice failed\n"
                << " Details:\n"
                << "   Result:\n" << quatslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Clearing the 4th quatslice (rvalue)
      {
         initialize();

         clear( blaze::quatslice( quat_, 1UL ) );

         checkPages   ( quat_,  2UL );
         checkRows    ( quat_,  5UL );
         checkColumns ( quat_,  4UL );
         checkQuats   ( quat_,  3UL );

         if( quat_(1,0,0,0) !=  0 || quat_(1,0,0,1) !=   0 || quat_(1,0,0,2) !=  0 || quat_(1,0,0,3) !=  0 ||
             quat_(1,0,1,0) !=  0 || quat_(1,0,1,1) !=   0 || quat_(1,0,1,2) !=  0 || quat_(1,0,1,3) !=  0 ||
             quat_(1,0,2,0) !=  0 || quat_(1,0,2,1) !=   0 || quat_(1,0,2,2) !=  0 || quat_(1,0,2,3) !=  0 ||
             quat_(1,0,3,0) !=  0 || quat_(1,0,3,1) !=   0 || quat_(1,0,3,2) !=  0 || quat_(1,0,3,3) !=  0 ||
             quat_(1,0,4,0) !=  0 || quat_(1,0,4,1) !=   0 || quat_(1,0,4,2) !=  0 || quat_(1,0,4,3) !=  0 ||
             quat_(1,1,0,0) !=  0 || quat_(1,1,0,1) !=   0 || quat_(1,1,0,2) !=  0 || quat_(1,1,0,3) !=  0 ||
             quat_(1,1,1,0) !=  0 || quat_(1,1,1,1) !=   0 || quat_(1,1,1,2) !=  0 || quat_(1,1,1,3) !=  0 ||
             quat_(1,1,2,0) !=  0 || quat_(1,1,2,1) !=   0 || quat_(1,1,2,2) !=  0 || quat_(1,1,2,3) !=  0 ||
             quat_(1,1,3,0) !=  0 || quat_(1,1,3,1) !=   0 || quat_(1,1,3,2) !=  0 || quat_(1,1,3,3) !=  0 ||
             quat_(1,1,4,0) !=  0 || quat_(1,1,4,1) !=   0 || quat_(1,1,4,2) !=  0 || quat_(1,1,4,3) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 1st quatslice failed\n"
                << " Details:\n"
                << "   Result:\n" << quat_ << "\n"
                << "   Expected result:\n((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0  -3   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  7  -8   9  10 ))\n"
                                        "((  0   0   0   0 )\n"
                                        " (  0   0   0   0 )\n"
                                        " (  0   0   0   0 )\n"
                                        " (  0   0   0   0 )\n"
                                        " (  0   0   0   0 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDefault() function with the QuatSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDefault() function with the QuatSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testIsDefault()
{
   using blaze::isDefault;


   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   {
      test_ = "isDefault() function";

      initialize();

      // isDefault with default quatslice
      {
         RT quatslice0 = blaze::quatslice( quat_, 0UL );
         quatslice0 = 0;

         if( isDefault( quatslice0(0, 1, 0) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   QuatSlice element: " << quatslice0(0, 1, 0) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( quatslice0 ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   QuatSlice:\n" << quatslice0 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default quatslice
      {
         RT quatslice1 = blaze::quatslice( quat_, 1UL );

         if( isDefault( quatslice1(0, 0, 1) ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   QuatSlice element: " << quatslice1(0, 0, 1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( quatslice1 ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   QuatSlice:\n" << quatslice1 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isSame() function with the QuatSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSame() function with the QuatSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testIsSame()
{
   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   {
      test_ = "isSame() function";

      // isSame with matching quatslices
      {
         RT quatslice1 = blaze::quatslice( quat_, 1UL );
         RT quatslice2 = blaze::quatslice( quat_, 1UL );

         if( blaze::isSame( quatslice1, quatslice2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First quatslice:\n" << quatslice1 << "\n"
                << "   Second quatslice:\n" << quatslice2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching quatslices
      {
         RT quatslice1 = blaze::quatslice( quat_, 0UL );
         RT quatslice2 = blaze::quatslice( quat_, 1UL );

         quatslice1 = 42;

         if( blaze::isSame( quatslice1, quatslice2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First quatslice:\n" << quatslice1 << "\n"
                << "   Second quatslice:\n" << quatslice2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with quatslice and matching subtensor
      {
         RT   quatslice1 = blaze::quatslice( quat_, 1UL );
         auto sv = blaze::subtensor( quatslice1, 0UL, 0UL, 0UL, 2UL, 5UL, 4UL );

         if( blaze::isSame( quatslice1, sv ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense quatslice:\n" << quatslice1 << "\n"
                << "   Dense subtensor:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sv, quatslice1 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense quatslice:\n" << quatslice1 << "\n"
                << "   Dense subtensor:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with quatslice and non-matching subtensor (different size)
      {
         RT   quatslice1 = blaze::quatslice( quat_, 1UL );
         auto sv = blaze::subtensor( quatslice1, 0UL, 0UL, 0UL, 1UL, 3UL, 3UL );

         if( blaze::isSame( quatslice1, sv ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense quatslice:\n" << quatslice1 << "\n"
                << "   Dense subtensor:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sv, quatslice1 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense quatslice:\n" << quatslice1 << "\n"
                << "   Dense subtensor:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with quatslice and non-matching subtensor (different offset)
      {
         RT   quatslice1 = blaze::quatslice( quat_, 1UL );
         auto sv = blaze::subtensor( quatslice1, 1UL, 1UL, 1UL, 1UL, 3UL, 3UL );

         if( blaze::isSame( quatslice1, sv ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense quatslice:\n" << quatslice1 << "\n"
                << "   Dense subtensor:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sv, quatslice1 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense quatslice:\n" << quatslice1 << "\n"
                << "   Dense subtensor:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      //// isSame with matching quatslices on a common subarray
      //{
      //   auto sm   = blaze::subarray( quat_, 0UL, 1UL, 1UL, 2UL, 3UL, 2UL );
      //   auto quatslice1 = blaze::quatslice( sm, 1UL );
      //   auto quatslice2 = blaze::quatslice( sm, 1UL );

      //   if( blaze::isSame( quatslice1, quatslice2 ) == false ) {
      //      std::ostringstream oss;
      //      oss << " Test: " << test_ << "\n"
      //          << " Error: Invalid isSame evaluation\n"
      //          << " Details:\n"
      //          << "   First quatslice:\n" << quatslice1 << "\n"
      //          << "   Second quatslice:\n" << quatslice2 << "\n";
      //      throw std::runtime_error( oss.str() );
      //   }
      //}
//
//      // isSame with non-matching quatslices on a common subarray
//      {
//         auto sm   = blaze::subarray( quat_, 0UL, 1UL, 1UL, 2UL, 3UL, 2UL );
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
//      // isSame with matching subtensor on quaternion and subtensor
//      {
//         auto sm   = blaze::subtensor( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
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
//      // isSame with non-matching quatslices on quaternion and subtensor (different quatslice)
//      {
//         auto sm   = blaze::subtensor( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
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
//      // isSame with non-matching quatslices on quaternion and subtensor (different size)
//      {
//         auto sm   = blaze::subtensor( quat_, 1UL, 0UL, 0UL, 1UL, 4UL, 3UL );
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
//      // isSame with matching quatslices on two subtensors
//      {
//         auto sm1  = blaze::subtensor( quat_, 0UL, 0UL, 0UL, 2UL, 5UL, 4UL );
//         auto sm2  = blaze::subtensor( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
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
//      // isSame with non-matching quatslices on two subtensors (different quatslice)
//      {
//         auto sm1  = blaze::subtensor( quat_, 0UL, 0UL, 0UL, 2UL, 5UL, 4UL );
//         auto sm2  = blaze::subtensor( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
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
//      // isSame with non-matching quatslices on two subtensors (different size)
//      {
//         auto sm1  = blaze::subtensor( quat_, 0UL, 0UL, 0UL, 2UL, 4UL, 3UL );
//         auto sm2  = blaze::subtensor( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
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
//      // isSame with non-matching quatslices on two subtensors (different offset)
//      {
//         auto sm1  = blaze::subtensor( quat_, 0UL, 1UL, 2UL, 2UL, 4UL, 2UL );
//         auto sm2  = blaze::subtensor( quat_, 1UL, 0UL, 0UL, 1UL, 4UL, 2UL );
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
//      // isSame with matching quatslice subquatrices on a subtensor
//      {
//         auto sm   = blaze::subtensor( quat_, 0UL, 1UL, 2UL, 2UL, 4UL, 2UL );
//         auto quatslice1 = blaze::quatslice( sm, 1UL );
//         auto sv1  = blaze::subtensor( quatslice1, 0UL, 0UL, 2UL, 1UL );
//         auto sv2  = blaze::subtensor( quatslice1, 0UL, 0UL, 2UL, 1UL );
//
//         if( blaze::isSame( sv1, sv2 ) == false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First subtensor:\n" << sv1 << "\n"
//                << "   Second subtensor:\n" << sv2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-matching quatslice subtensors on a subtensor (different size)
//      {
//         auto sm   = blaze::subtensor( quat_, 0UL, 1UL, 1UL, 2UL, 4UL, 3UL );
//         auto quatslice1 = blaze::quatslice( sm, 1UL );
//         auto sv1  = blaze::subtensor( quatslice1, 0UL, 0UL, 2UL, 1UL );
//         auto sv2  = blaze::subtensor( quatslice1, 0UL, 0UL, 2UL, 2UL );
//
//         if( blaze::isSame( sv1, sv2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First subtensor:\n" << sv1 << "\n"
//                << "   Second subtensor:\n" << sv2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-matching quatslice subtensors on a subtensor (different offset)
//      {
//         auto sm   = blaze::subtensor( quat_, 0UL, 1UL, 1UL, 2UL, 4UL, 3UL );
//         auto quatslice1 = blaze::quatslice( sm, 1UL );
//         auto sv1  = blaze::subtensor( quatslice1, 0UL, 0UL, 2UL, 1UL );
//         auto sv2  = blaze::subtensor( quatslice1, 0UL, 1UL, 2UL, 1UL );
//
//         if( blaze::isSame( sv1, sv2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First subtensor:\n" << sv1 << "\n"
//                << "   Second subtensor:\n" << sv2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with matching quatslice subtensors on two subtensors
//      {
//         auto sm1  = blaze::subtensor( quat_, 0UL, 0UL, 0UL, 2UL, 5UL, 4UL );
//         auto sm2  = blaze::subtensor( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
//         auto quatslice1 = blaze::quatslice( sm1, 1UL );
//         auto quatslice2 = blaze::quatslice( sm2, 0UL );
//         auto sv1  = blaze::subtensor( quatslice1, 0UL, 0UL, 3UL, 2UL );
//         auto sv2  = blaze::subtensor( quatslice2, 0UL, 0UL, 3UL, 2UL );
//
//         if( blaze::isSame( sv1, sv2 ) == false ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First subtensor:\n" << sv1 << "\n"
//                << "   Second subtensor:\n" << sv2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-matching quatslice subtensors on two subtensors (different size)
//      {
//         auto sm1  = blaze::subtensor( quat_, 0UL, 0UL, 0UL, 2UL, 5UL, 4UL );
//         auto sm2  = blaze::subtensor( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
//         auto quatslice1 = blaze::quatslice( sm1, 1UL );
//         auto quatslice2 = blaze::quatslice( sm2, 0UL );
//         auto sv1  = blaze::subtensor( quatslice1, 0UL, 0UL, 3UL, 2UL );
//         auto sv2  = blaze::subtensor( quatslice2, 0UL, 0UL, 2UL, 2UL );
//
//         if( blaze::isSame( sv1, sv2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First subtensor:\n" << sv1 << "\n"
//                << "   Second subtensor:\n" << sv2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // isSame with non-matching quatslice subtensors on two subtensors (different offset)
//      {
//         auto sm1  = blaze::subtensor( quat_, 0UL, 0UL, 0UL, 2UL, 5UL, 4UL );
//         auto sm2  = blaze::subtensor( quat_, 1UL, 0UL, 0UL, 1UL, 5UL, 4UL );
//         auto quatslice1 = blaze::quatslice( sm1, 1UL );
//         auto quatslice2 = blaze::quatslice( sm2, 0UL );
//         auto sv1  = blaze::subtensor( quatslice1, 0UL, 0UL, 3UL, 2UL );
//         auto sv2  = blaze::subtensor( quatslice2, 0UL, 1UL, 3UL, 2UL );
//
//         if( blaze::isSame( sv1, sv2 ) == true ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Invalid isSame evaluation\n"
//                << " Details:\n"
//                << "   First subtensor:\n" << sv1 << "\n"
//                << "   Second subtensor:\n" << sv2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c subtensor() function with the QuatSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c subtensor() function used with the QuatSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testSubtensor()
{
   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   {
      test_ = "subtensor() function";

      initialize();

      {
         RT   quatslice1 = blaze::quatslice( quat_, 1UL );
         auto sm = blaze::subtensor( quatslice1, 1UL, 1UL, 1UL, 1UL, 3UL, 2UL );

         if( sm(0,0,0) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result: " << sm(0,0,0) << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }

         // sm.begin( page, row)
         if( *sm.begin(1,0) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *sm.begin(1,0) << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT   quatslice1 = blaze::quatslice( quat_, 1UL );
         auto sm = blaze::subtensor( quatslice1, 2UL, 4UL, 0UL, 1UL, 4UL, 4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         RT   quatslice1 = blaze::quatslice( quat_, 1UL );
         auto sm = blaze::subtensor( quatslice1, 0UL, 0UL, 0UL, 2UL, 2UL, 6UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c row() function with the Subquaternion class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c row() function with the Subquaternion specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testPageslice()
{
   using blaze::quatslice;
   using blaze::pageslice;
   using blaze::row;


   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   {
      test_ = "Quatslice pageslice() function";

      initialize();

      {
         RT quatslice1  = quatslice( quat_, 0UL );
         RT quatslice2  = quatslice( quat_, 0UL );
         auto pageslice1 = pageslice( quatslice1, 1UL );
         auto pageslice2 = pageslice( quatslice2, 1UL );

         if( pageslice1 != pageslice2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Row function failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice1 << "\n"
                << "   Expected result:\n" << pageslice2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( row(pageslice1,1) != row(pageslice2,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result: " << row(pageslice1,1) << "\n"
                << "   Expected result: " << row(pageslice2,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *pageslice1.begin(1) != *pageslice2.begin(1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *pageslice1.begin(1) << "\n"
                << "   Expected result: " << *pageslice2.begin(1) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT quatslice1  = quatslice( quat_, 0UL );
         auto pageslice2 = pageslice( quatslice1, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row succeeded\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c row() function with the Subquaternion class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c row() function with the Subquaternion specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testRowslice()
{
   using blaze::quatslice;
   using blaze::rowslice;
   using blaze::row;


   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   {
      test_ = "Quatslice rowslice() function";

      initialize();

      {
         RT quatslice1  = quatslice( quat_, 0UL );
         RT quatslice2  = quatslice( quat_, 2UL );
         auto rowslice1 = rowslice( quatslice1, 1UL );
         auto rowslice2 = rowslice( quatslice2, 1UL );

         if( rowslice1 != rowslice2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Row function failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice1 << "\n"
                << "   Expected result:\n" << rowslice2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( row(rowslice1,1) != row(rowslice2,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result: " << row(rowslice1,1) << "\n"
                << "   Expected result: " << row(rowslice2,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *rowslice1.begin(1) != *rowslice2.begin(1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *rowslice1.begin(1) << "\n"
                << "   Expected result: " << *rowslice2.begin(1) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT quatslice1  = quatslice( quat_, 0UL );
         auto rowslice6 = rowslice( quatslice1, 6UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rowslice6 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c column() function with the Subquaternion class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c column() function with the Subquaternion specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testColumnslice()
{
   using blaze::quatslice;
   using blaze::columnslice;
   using blaze::column;


   //=====================================================================================
   // quaternion tests
   //=====================================================================================

   {
      test_ = "Quatslice columnslice() function";

      initialize();

      {
         RT quatslice1  = quatslice( quat_, 1UL );
         RT quatslice2  = quatslice( quat_, 1UL );
         auto columnslice1 = columnslice( quatslice1, 1UL );
         auto columnslice2 = columnslice( quatslice2, 1UL );

         if( columnslice1 != columnslice2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Row function failed\n"
                << " Details:\n"
                << "   Result:\n" << columnslice1 << "\n"
                << "   Expected result:\n" << columnslice2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( row(columnslice1,1) != row(columnslice2,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result: " << row(columnslice1,1) << "\n"
                << "   Expected result: " << row(columnslice2,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *columnslice1.begin(1) != *columnslice2.begin(1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *columnslice1.begin(1) << "\n"
                << "   Expected result: " << *columnslice2.begin(1) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT quatslice1  = quatslice( quat_, 0UL );
         auto columnslice6 = columnslice( quatslice1, 6UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row succeeded\n"
             << " Details:\n"
             << "   Result:\n" << columnslice6 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


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
   quat_.reset();
   quat_(0,0,1,1) =  1;
   quat_(0,0,2,0) = -2;
   quat_(0,0,2,2) = -3;
   quat_(0,0,3,1) =  4;
   quat_(0,0,3,2) =  5;
   quat_(0,0,3,3) = -6;
   quat_(0,0,4,0) =  7;
   quat_(0,0,4,1) = -8;
   quat_(0,0,4,2) =  9;
   quat_(0,0,4,3) = 10;
   quat_(0,1,1,1) =  1;
   quat_(0,1,2,0) = -2;
   quat_(0,1,2,2) = 13;
   quat_(0,1,3,1) =  4;
   quat_(0,1,3,2) =  5;
   quat_(0,1,3,3) = -6;
   quat_(0,1,4,0) =  7;
   quat_(0,1,4,1) = -8;
   quat_(0,1,4,2) =  9;
   quat_(0,1,4,3) = 10;
   quat_(1,0,0,1) =  1;
   quat_(1,0,2,1) = 12;
   quat_(1,0,2,2) = -3;
   quat_(1,0,3,1) =  4;
   quat_(1,0,3,2) =  5;
   quat_(1,0,3,3) = -6;
   quat_(1,0,4,0) =  7;
   quat_(1,0,4,1) = 28;
   quat_(1,0,4,2) =  9;
   quat_(1,0,4,3) = 10;
   quat_(1,1,1,1) =  1;
   quat_(1,1,2,0) = -2;
   quat_(1,1,3,0) = -3;
   quat_(1,1,3,1) =  4;
   quat_(1,1,3,2) =  5;
   quat_(1,1,3,3) = 33;
   quat_(1,1,4,0) =  7;
   quat_(1,1,4,1) = -8;
   quat_(1,1,4,2) =  9;
   quat_(1,1,4,3) = 11;
   quat_(2,0,1,1) =  1;
   quat_(2,0,2,0) = -2;
   quat_(2,0,2,2) = -3;
   quat_(2,0,2,3) =  4;
   quat_(2,0,3,2) =  5;
   quat_(2,0,3,3) =  2;
   quat_(2,0,4,0) =  7;
   quat_(2,0,4,1) = -8;
   quat_(2,0,4,2) =  9;
   quat_(2,0,4,3) = 10;
   quat_(2,1,1,1) =  1;
   quat_(2,1,2,0) = 62;
   quat_(2,1,2,2) = -3;
   quat_(2,1,3,1) =  5;
   quat_(2,1,3,2) = 15;
   quat_(2,1,3,3) = 16;
   quat_(2,1,4,0) = -7;
   quat_(2,1,4,1) = -8;
   quat_(2,1,4,2) = 19;
   quat_(2,1,4,3) = 10;
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
