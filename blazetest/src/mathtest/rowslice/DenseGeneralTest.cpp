//=================================================================================================
/*!
//  \file src/mathtest/rowslice/DenseGeneralTest.cpp
//  \brief Source file for the RowSlice dense general test
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

#include <blaze/math/CustomMatrix.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/Views.h>
#include <blaze/system/Platform.h>
#include <blaze/util/policies/Deallocate.h>

#include <blaze_tensor/math/CustomTensor.h>
#include <blaze_tensor/math/DynamicTensor.h>

#include <blazetest/mathtest/rowslice/DenseGeneralTest.h>


namespace blazetest {

namespace mathtest {

namespace rowslice {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the RowSlice dense general test.
//
// \exception std::runtime_error Operation error detected.
*/
DenseGeneralTest::DenseGeneralTest()
   : mat_ ( 5UL, 4UL, 2UL )
{
   testConstructors();
   testAssignment();
   testAddAssign();
   testSubAssign();
   testMultAssign();
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
   testSubmatrix();
   testRow();
   testRows();
   testColumn();
   testColumns();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the RowSlice constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the RowSlice specialization. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testConstructors()
{
   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "RowSlice constructor (0x0)";

      MT mat;

      // 0th matrix rowslice
      try {
         blaze::rowslice( mat, 0UL );
      }
      catch( std::invalid_argument& ) {}
   }

   {
      test_ = "RowSlice constructor (2x0)";

      MT mat( 2UL, 0UL, 2UL );

      // 0th matrix rowslice
      {
         RT rowslice0 = blaze::rowslice( mat, 0UL );

         checkRows    ( rowslice0, 2UL );
         checkColumns ( rowslice0, 0UL );
         checkCapacity( rowslice0, 0UL );
         checkNonZeros( rowslice0, 0UL );
      }

      // 1st matrix rowslice
      {
         RT rowslice1 = blaze::rowslice( mat, 1UL );

         checkRows    ( rowslice1, 2UL );
         checkColumns ( rowslice1, 0UL );
         checkCapacity( rowslice1, 0UL );
         checkNonZeros( rowslice1, 0UL );
      }

      // 2nd matrix rowslice
      try {
         blaze::rowslice( mat, 2UL );
      }
      catch( std::invalid_argument& ) {}
   }

   {
      test_ = "RowSlice constructor (5x4)";

      initialize();

      // 0th tensor rowslice
      {
         RT rowslice0 = blaze::rowslice( mat_, 0UL );

         checkRows    ( rowslice0, 5UL );
         checkColumns ( rowslice0, 4UL );
         checkCapacity( rowslice0, 20UL );
         checkNonZeros( rowslice0, 10UL );

         if( rowslice0(0,0) !=  0 || rowslice0(0,1) !=  0 || rowslice0(0,2) !=  0 || rowslice0(0,3) !=  0 ||
             rowslice0(1,0) !=  0 || rowslice0(1,1) !=  1 || rowslice0(1,2) !=  0 || rowslice0(1,3) !=  0 ||
             rowslice0(2,0) != -2 || rowslice0(2,1) !=  0 || rowslice0(2,2) != -3 || rowslice0(2,3) !=  0 ||
             rowslice0(3,0) !=  0 || rowslice0(3,1) !=  4 || rowslice0(3,2) !=  5 || rowslice0(3,3) != -6 ||
             rowslice0(4,0) !=  7 || rowslice0(4,1) != -8 || rowslice0(4,2) !=  9 || rowslice0(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of 0th dense rowslice failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice0 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // 1st tensor rowslice
      {
         RT rowslice1 = blaze::rowslice( mat_, 1UL );

         checkRows    ( rowslice1, 5UL );
         checkColumns ( rowslice1, 4UL );
         checkCapacity( rowslice1, 20UL );
         checkNonZeros( rowslice1, 10UL );

         if( rowslice1(0,0) !=  0 || rowslice1(0,1) !=  0 || rowslice1(0,2) !=  0 || rowslice1(0,3) !=  0 ||
             rowslice1(1,0) !=  0 || rowslice1(1,1) !=  1 || rowslice1(1,2) !=  0 || rowslice1(1,3) !=  0 ||
             rowslice1(2,0) != -2 || rowslice1(2,1) !=  0 || rowslice1(2,2) != -3 || rowslice1(2,3) !=  0 ||
             rowslice1(3,0) !=  0 || rowslice1(3,1) !=  4 || rowslice1(3,2) !=  5 || rowslice1(3,3) != -6 ||
             rowslice1(4,0) !=  7 || rowslice1(4,1) != -8 || rowslice1(4,2) !=  9 || rowslice1(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of 1st dense rowslice failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice1 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // 2nd tensor rowslice
      try {
         RT rowslice2 = blaze::rowslice( mat_, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Out-of-bound page access succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Test of the RowSlice assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the RowSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testAssignment()
{
   //=====================================================================================
   // homogeneous assignment
   //=====================================================================================

   {
      test_ = "RowSlice homogeneous assignment";

      initialize();

      RT rowslice1 = blaze::rowslice( mat_, 1UL );
      rowslice1 = 8;


      checkRows    ( rowslice1, 5UL );
      checkColumns ( rowslice1, 4UL );
      checkCapacity( rowslice1, 20UL );
      checkNonZeros( rowslice1, 20UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 30UL );

      if( rowslice1(0,0) != 8 || rowslice1(0,1) != 8 || rowslice1(0,2) != 8 || rowslice1(0,3) != 8 ||
          rowslice1(1,0) != 8 || rowslice1(1,1) != 8 || rowslice1(1,2) != 8 || rowslice1(1,3) != 8 ||
          rowslice1(2,0) != 8 || rowslice1(2,1) != 8 || rowslice1(2,2) != 8 || rowslice1(2,3) != 8 ||
          rowslice1(3,0) != 8 || rowslice1(3,1) != 8 || rowslice1(3,2) != 8 || rowslice1(3,3) != 8 ||
          rowslice1(4,0) != 8 || rowslice1(4,1) != 8 || rowslice1(4,2) != 8 || rowslice1(4,3) != 8 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice1 << "\n"
             << "   Expected result:\n(( 8 8 8 8 )\n( 8 8 8 8 )\n( 8 8 8 8 )\n( 8 8 8 8 )\n( 8 8 8 8 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  8 || mat_(0,1,1) !=  8 || mat_(0,2,1) !=  8 || mat_(0,3,1) !=  8 ||
          mat_(1,0,1) !=  8 || mat_(1,1,1) !=  8 || mat_(1,2,1) !=  8 || mat_(1,3,1) !=  8 ||
          mat_(2,0,1) !=  8 || mat_(2,1,1) !=  8 || mat_(2,2,1) !=  8 || mat_(2,3,1) !=  8 ||
          mat_(3,0,1) !=  8 || mat_(3,1,1) !=  8 || mat_(3,2,1) !=  8 || mat_(3,3,1) !=  8 ||
          mat_(4,0,1) !=  8 || mat_(4,1,1) !=  8 || mat_(4,2,1) !=  8 || mat_(4,3,1) !=  8 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0  0  0  0 )\n"
                                     " (  0  1  0  0 )\n"
                                     " ( -2  0 -3  0 )\n"
                                     " (  0  4  5 -6 )\n"
                                     " (  7 -8  9 10 ))\n"
                                     "((  8  8  8  8 )\n"
                                     " (  8  8  8  8 )\n"
                                     " (  8  8  8  8 )\n"
                                     " (  8  8  8  8 )\n"
                                     " (  8  8  8  8 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // list assignment
   //=====================================================================================

   {
      test_ = "initializer list assignment (complete list)";

      initialize();

      RT rowslice3 = blaze::rowslice( mat_, 1UL );
      rowslice3 = {
          {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}
      };

      checkRows    ( rowslice3, 5UL );
      checkColumns ( rowslice3, 4UL );
      checkCapacity( rowslice3, 20UL );
      checkNonZeros( rowslice3, 20UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 30UL );

      if( rowslice3(0,0) != 1 || rowslice3(0,1) != 2 || rowslice3(0,2) != 3 || rowslice3(0,3) != 4 ||
          rowslice3(1,0) != 1 || rowslice3(1,1) != 2 || rowslice3(1,2) != 3 || rowslice3(1,3) != 4 ||
          rowslice3(2,0) != 1 || rowslice3(2,1) != 2 || rowslice3(2,2) != 3 || rowslice3(2,3) != 4 ||
          rowslice3(3,0) != 1 || rowslice3(3,1) != 2 || rowslice3(3,2) != 3 || rowslice3(3,3) != 4 ||
          rowslice3(4,0) != 1 || rowslice3(4,1) != 2 || rowslice3(4,2) != 3 || rowslice3(4,3) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice3 << "\n"
             << "   Expected result:\n(( 1 2 3 4 )\n( 1 2 3 4 )\n( 1 2 3 4 )\n( 1 2 3 4 )\n( 1 2 3 4 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  1 || mat_(0,1,1) !=  2 || mat_(0,2,1) !=  3 || mat_(0,3,1) !=  4 ||
          mat_(1,0,1) !=  1 || mat_(1,1,1) !=  2 || mat_(1,2,1) !=  3 || mat_(1,3,1) !=  4 ||
          mat_(2,0,1) !=  1 || mat_(2,1,1) !=  2 || mat_(2,2,1) !=  3 || mat_(2,3,1) !=  4 ||
          mat_(3,0,1) !=  1 || mat_(3,1,1) !=  2 || mat_(3,2,1) !=  3 || mat_(3,3,1) !=  4 ||
          mat_(4,0,1) !=  1 || mat_(4,1,1) !=  2 || mat_(4,2,1) !=  3 || mat_(4,3,1) !=  4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0  0  0  0 )\n"
                                     " (  0  1  0  0 )\n"
                                     " ( -2  0 -3  0 )\n"
                                     " (  0  4  5 -6 )\n"
                                     " (  7 -8  9 10 ))\n"
                                     "((  1  2  3  4 )\n"
                                     " (  1  2  3  4 )\n"
                                     " (  1  2  3  4 )\n"
                                     " (  1  2  3  4 )\n"
                                     " (  1  2  3  4 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "initializer list assignment (incomplete list)";

      initialize();

      RT rowslice3 = blaze::rowslice( mat_, 1UL );
      rowslice3 = {{1, 2}, {1, 2}, {1, 2}, {1, 2}, {1, 2}};

      checkRows    ( rowslice3, 5UL );
      checkColumns ( rowslice3, 4UL );
      checkCapacity( rowslice3, 20UL );
      checkNonZeros( rowslice3, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice3(0,0) != 1 || rowslice3(0,1) != 2 || rowslice3(0,2) != 0 || rowslice3(0,3) != 0 ||
          rowslice3(1,0) != 1 || rowslice3(1,1) != 2 || rowslice3(1,2) != 0 || rowslice3(1,3) != 0 ||
          rowslice3(2,0) != 1 || rowslice3(2,1) != 2 || rowslice3(2,2) != 0 || rowslice3(2,3) != 0 ||
          rowslice3(3,0) != 1 || rowslice3(3,1) != 2 || rowslice3(3,2) != 0 || rowslice3(3,3) != 0 ||
          rowslice3(4,0) != 1 || rowslice3(4,1) != 2 || rowslice3(4,2) != 0 || rowslice3(4,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice3 << "\n"
             << "   Expected result:\n(( 1 2 0 0 )\n( 1 2 0 0 )\n( 1 2 0 0 )\n( 1 2 0 0 )\n( 1 2 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  1 || mat_(0,1,1) !=  2 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=  1 || mat_(1,1,1) !=  2 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) !=  1 || mat_(2,1,1) !=  2 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  1 || mat_(3,1,1) !=  2 || mat_(3,2,1) !=  0 || mat_(3,3,1) !=  0 ||
          mat_(4,0,1) !=  1 || mat_(4,1,1) !=  2 || mat_(4,2,1) !=  0 || mat_(4,3,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0  0  0  0 )\n"
                                     " (  0  1  0  0 )\n"
                                     " ( -2  0 -3  0 )\n"
                                     " (  0  4  5 -6 )\n"
                                     " (  7 -8  9 10 ))\n"
                                     "((  1  2  0  0 )\n"
                                     " (  1  2  0  0 )\n"
                                     " (  1  2  0  0 )\n"
                                     " (  1  2  0  0 )\n"
                                     " (  1  2  0  0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // copy assignment
   //=====================================================================================

   {
      test_ = "RowSlice copy assignment";

      initialize();

      RT rowslice1 = blaze::rowslice( mat_, 0UL );
      rowslice1 = 0;
      rowslice1 = blaze::rowslice( mat_, 1UL );

      checkRows    ( rowslice1, 5UL );
      checkColumns ( rowslice1, 4UL );
      checkCapacity( rowslice1, 20UL );
      checkNonZeros( rowslice1, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice1(0,0) !=  0 || rowslice1(0,1) !=  0 || rowslice1(0,2) !=  0 || rowslice1(0,3) !=  0 ||
          rowslice1(1,0) !=  0 || rowslice1(1,1) !=  1 || rowslice1(1,2) !=  0 || rowslice1(1,3) !=  0 ||
          rowslice1(2,0) != -2 || rowslice1(2,1) !=  0 || rowslice1(2,2) != -3 || rowslice1(2,3) !=  0 ||
          rowslice1(3,0) !=  0 || rowslice1(3,1) !=  4 || rowslice1(3,2) !=  5 || rowslice1(3,3) != -6 ||
          rowslice1(4,0) !=  7 || rowslice1(4,1) != -8 || rowslice1(4,2) !=  9 || rowslice1(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice1 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=  0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) != -2 || mat_(2,1,1) !=  0 || mat_(2,2,1) != -3 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) !=  7 || mat_(4,1,1) != -8 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0  0  0  0 )\n"
                                     " (  0  1  0  0 )\n"
                                     " ( -2  0 -3  0 )\n"
                                     " (  0  4  5 -6 )\n"
                                     " (  7 -8  9 10 ))\n"
                                     "((  0  0  0  0 )\n"
                                     " (  0  1  0  0 )\n"
                                     " ( -2  0 -3  0 )\n"
                                     " (  0  4  5 -6 )\n"
                                     " (  7 -8  9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // dense matrix assignment
   //=====================================================================================

   {
      test_ = "dense matrix assignment (mixed type)";

      initialize();

      RT rowslice1 = blaze::rowslice( mat_, 1UL );

      blaze::DynamicMatrix<int, blaze::rowMajor> m1;
      m1 = {{0, 8, 0, 9}, {0}, {0}, {0}, {0}};

      rowslice1 = m1;

      checkRows    ( rowslice1, 5UL );
      checkColumns ( rowslice1, 4UL );
      checkCapacity( rowslice1, 20UL );
      checkNonZeros( rowslice1, 2UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 12UL );

      if( rowslice1(0,0) !=  0 || rowslice1(0,1) !=  8 || rowslice1(0,2) !=  0 || rowslice1(0,3) !=  9 ||
          rowslice1(1,0) !=  0 || rowslice1(1,1) !=  0 || rowslice1(1,2) !=  0 || rowslice1(1,3) !=  0 ||
          rowslice1(2,0) !=  0 || rowslice1(2,1) !=  0 || rowslice1(2,2) !=  0 || rowslice1(2,3) !=  0 ||
          rowslice1(3,0) !=  0 || rowslice1(3,1) !=  0 || rowslice1(3,2) !=  0 || rowslice1(3,3) !=  0 ||
          rowslice1(4,0) !=  0 || rowslice1(4,1) !=  0 || rowslice1(4,2) !=  0 || rowslice1(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice1 << "\n"
             << "   Expected result:\n(( 0 8 0 9 )\n(0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=  8 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  9 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=  0 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) !=  0 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  0 || mat_(3,2,1) !=  0 || mat_(3,3,1) !=  0 ||
          mat_(4,0,1) !=  0 || mat_(4,1,1) !=  0 || mat_(4,2,1) !=  0 || mat_(4,3,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0  0  0  0 )\n"
                                     " (  0  1  0  0 )\n"
                                     " ( -2  0 -3  0 )\n"
                                     " (  0  4  5 -6 )\n"
                                     " (  7 -8  9 10 ))\n"
                                     "((  0  9  0  9 )\n"
                                     " (  0  0  0  0 )\n"
                                     " (  0  0  0  0 )\n"
                                     " (  0  0  0  0 )\n"
                                     " (  0  0  0  0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "dense matrix assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;
      using blaze::rowMajor;

      initialize();

      RT rowslice1 = blaze::rowslice( mat_, 1UL );

      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 80UL ) );
      AlignedPadded m1( memory.get(), 5UL, 4UL, 16UL );
      m1 = 0;
      m1(0,0) = 0;
      m1(0,1) = 8;
      m1(0,2) = 0;
      m1(0,3) = 9;

      rowslice1 = m1;

      checkRows    ( rowslice1, 5UL );
      checkColumns ( rowslice1, 4UL );
      checkCapacity( rowslice1, 20UL );
      checkNonZeros( rowslice1, 2UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 12UL );

      if( rowslice1(0,0) !=  0 || rowslice1(0,1) !=  8 || rowslice1(0,2) !=  0 || rowslice1(0,3) !=  9 ||
          rowslice1(1,0) !=  0 || rowslice1(1,1) !=  0 || rowslice1(1,2) !=  0 || rowslice1(1,3) !=  0 ||
          rowslice1(2,0) !=  0 || rowslice1(2,1) !=  0 || rowslice1(2,2) !=  0 || rowslice1(2,3) !=  0 ||
          rowslice1(3,0) !=  0 || rowslice1(3,1) !=  0 || rowslice1(3,2) !=  0 || rowslice1(3,3) !=  0 ||
          rowslice1(4,0) !=  0 || rowslice1(4,1) !=  0 || rowslice1(4,2) !=  0 || rowslice1(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice1 << "\n"
             << "   Expected result:\n(( 0 8 0 9 )\n(0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=  8 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  9 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=  0 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) !=  0 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  0 || mat_(3,2,1) !=  0 || mat_(3,3,1) !=  0 ||
          mat_(4,0,1) !=  0 || mat_(4,1,1) !=  0 || mat_(4,2,1) !=  0 || mat_(4,3,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0  0  0  0 )\n"
                                     " (  0  1  0  0 )\n"
                                     " ( -2  0 -3  0 )\n"
                                     " (  0  4  5 -6 )\n"
                                     " (  7 -8  9 10 ))\n"
                                     "((  0  9  0  9 )\n"
                                     " (  0  0  0  0 )\n"
                                     " (  0  0  0  0 )\n"
                                     " (  0  0  0  0 )\n"
                                     " (  0  0  0  0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "dense matrix assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::rowMajor;

      initialize();

      RT rowslice1 = blaze::rowslice( mat_, 1UL );

      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
      std::unique_ptr<int[]> memory( new int[21] );
      UnalignedUnpadded m1( memory.get()+1UL, 5UL, 4UL );
      m1 = 0;
      m1(0,0) = 0;
      m1(0,1) = 8;
      m1(0,2) = 0;
      m1(0,3) = 9;

      rowslice1 = m1;

      checkRows    ( rowslice1, 5UL );
      checkColumns ( rowslice1, 4UL );
      checkCapacity( rowslice1, 20UL );
      checkNonZeros( rowslice1, 2UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 12UL );

      if( rowslice1(0,0) !=  0 || rowslice1(0,1) !=  8 || rowslice1(0,2) !=  0 || rowslice1(0,3) !=  9 ||
          rowslice1(1,0) !=  0 || rowslice1(1,1) !=  0 || rowslice1(1,2) !=  0 || rowslice1(1,3) !=  0 ||
          rowslice1(2,0) !=  0 || rowslice1(2,1) !=  0 || rowslice1(2,2) !=  0 || rowslice1(2,3) !=  0 ||
          rowslice1(3,0) !=  0 || rowslice1(3,1) !=  0 || rowslice1(3,2) !=  0 || rowslice1(3,3) !=  0 ||
          rowslice1(4,0) !=  0 || rowslice1(4,1) !=  0 || rowslice1(4,2) !=  0 || rowslice1(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice1 << "\n"
             << "   Expected result:\n(( 0 8 0 9 )\n(0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=  8 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  9 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=  0 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) !=  0 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  0 || mat_(3,2,1) !=  0 || mat_(3,3,1) !=  0 ||
          mat_(4,0,1) !=  0 || mat_(4,1,1) !=  0 || mat_(4,2,1) !=  0 || mat_(4,3,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0  0  0  0 )\n"
                                     " (  0  1  0  0 )\n"
                                     " ( -2  0 -3  0 )\n"
                                     " (  0  4  5 -6 )\n"
                                     " (  7 -8  9 10 ))\n"
                                     "((  0  9  0  9 )\n"
                                     " (  0  0  0  0 )\n"
                                     " (  0  0  0  0 )\n"
                                     " (  0  0  0  0 )\n"
                                     " (  0  0  0  0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the RowSlice addition assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the addition assignment operators of the RowSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testAddAssign()
{
   //=====================================================================================
   // RowSlice addition assignment
   //=====================================================================================

   {
      test_ = "RowSlice addition assignment";

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );
      rowslice2 += blaze::rowslice( mat_, 0UL );

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   2 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -4 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -6 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=   8 || rowslice2(3,2) != 10 || rowslice2(3,3) != -12 ||
          rowslice2(4,0) != 14 || rowslice2(4,1) != -16 || rowslice2(4,2) != 18 || rowslice2(4,3) !=  20 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   2 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) != -4 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -6 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=   8 || mat_(3,2,1) != 10 || mat_(3,3,1) != -12 ||
          mat_(4,0,1) != 14 || mat_(4,1,1) != -16 || mat_(4,2,1) != 18 || mat_(4,3,1) !=  20 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
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


   //=====================================================================================
   // dense matrix addition assignment
   //=====================================================================================

   {
      test_ = "dense matrix addition assignment (mixed type)";

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );

      const blaze::DynamicMatrix<short, blaze::rowMajor> vec{{0, 0, 0, 0},
                                                             {0, 1, 0, 0},
                                                             {-2, 0, -3, 0},
                                                             {0, 4, 5, -6},
                                                             {7, -8, 9, 10}};

      rowslice2 += vec;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   2 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -4 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -6 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=   8 || rowslice2(3,2) != 10 || rowslice2(3,3) != -12 ||
          rowslice2(4,0) != 14 || rowslice2(4,1) != -16 || rowslice2(4,2) != 18 || rowslice2(4,3) !=  20 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   2 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) != -4 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -6 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=   8 || mat_(3,2,1) != 10 || mat_(3,3,1) != -12 ||
          mat_(4,0,1) != 14 || mat_(4,1,1) != -16 || mat_(4,2,1) != 18 || mat_(4,3,1) !=  20 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
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
      test_ = "dense matrix addition assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;
      using blaze::rowMajor;

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );

      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 80UL ) );
      AlignedPadded m( memory.get(), 5UL, 4UL, 16UL );
      m(0,0) =  0;
      m(0,1) =  0;
      m(0,2) =  0;
      m(0,3) =  0;
      m(1,0) =  0;
      m(1,1) =  1;
      m(1,2) =  0;
      m(1,3) =  0;
      m(2,0) = -2;
      m(2,1) =  0;
      m(2,2) = -3;
      m(2,3) =  0;
      m(3,0) =  0;
      m(3,1) =  4;
      m(3,2) =  5;
      m(3,3) = -6;
      m(4,0) =  7;
      m(4,1) = -8;
      m(4,2) =  9;
      m(4,3) = 10;

      rowslice2 += m;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   2 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -4 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -6 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=   8 || rowslice2(3,2) != 10 || rowslice2(3,3) != -12 ||
          rowslice2(4,0) != 14 || rowslice2(4,1) != -16 || rowslice2(4,2) != 18 || rowslice2(4,3) !=  20 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   2 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) != -4 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -6 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=   8 || mat_(3,2,1) != 10 || mat_(3,3,1) != -12 ||
          mat_(4,0,1) != 14 || mat_(4,1,1) != -16 || mat_(4,2,1) != 18 || mat_(4,3,1) !=  20 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
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
      test_ = "dense matrix addition assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::rowMajor;

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );

      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
      std::unique_ptr<int[]> memory( new int[21] );
      UnalignedUnpadded m( memory.get()+1UL, 5UL, 4UL );
      m(0,0) =  0;
      m(0,1) =  0;
      m(0,2) =  0;
      m(0,3) =  0;
      m(1,0) =  0;
      m(1,1) =  1;
      m(1,2) =  0;
      m(1,3) =  0;
      m(2,0) = -2;
      m(2,1) =  0;
      m(2,2) = -3;
      m(2,3) =  0;
      m(3,0) =  0;
      m(3,1) =  4;
      m(3,2) =  5;
      m(3,3) = -6;
      m(4,0) =  7;
      m(4,1) = -8;
      m(4,2) =  9;
      m(4,3) = 10;

      rowslice2 += m;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   2 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -4 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -6 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=   8 || rowslice2(3,2) != 10 || rowslice2(3,3) != -12 ||
          rowslice2(4,0) != 14 || rowslice2(4,1) != -16 || rowslice2(4,2) != 18 || rowslice2(4,3) !=  20 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   2 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) != -4 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -6 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=   8 || mat_(3,2,1) != 10 || mat_(3,3,1) != -12 ||
          mat_(4,0,1) != 14 || mat_(4,1,1) != -16 || mat_(4,2,1) != 18 || mat_(4,3,1) !=  20 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
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
/*!\brief Test of the RowSlice subtraction assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the subtraction assignment operators of the RowSlice
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testSubAssign()
{
   //=====================================================================================
   // RowSlice subtraction assignment
   //=====================================================================================

   {
      test_ = "RowSlice subtraction assignment";

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );
      rowslice2 -= blaze::rowslice( mat_, 0UL );

      checkRows    ( rowslice2,  5UL );
      checkColumns ( rowslice2,  4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2,  0UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 10UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=  0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=  0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=  0 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=  0 ||
          rowslice2(2,0) !=  0 || rowslice2(2,1) !=  0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=  0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=  0 || rowslice2(3,2) !=  0 || rowslice2(3,3) !=  0 ||
          rowslice2(4,0) !=  0 || rowslice2(4,1) !=  0 || rowslice2(4,2) !=  0 || rowslice2(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   0 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) !=  0 || mat_(2,1,1) !=   0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=   0 || mat_(3,2,1) !=  0 || mat_(3,3,1) !=   0 ||
          mat_(4,0,1) !=  0 || mat_(4,1,1) !=   0 || mat_(4,2,1) !=  0 || mat_(4,3,1) !=   0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
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
   // dense matrix subtraction assignment
   //=====================================================================================

   {
      test_ = "dense matrix subtraction assignment (mixed type)";

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );

      const blaze::DynamicMatrix<short, blaze::rowMajor> vec{{0, 0, 0, 0},
                                                             {0, 1, 0, 0},
                                                             {-2, 0, -3, 0},
                                                             {0, 4, 5, -6},
                                                             {7, -8, 9, 10}};

      rowslice2 -= vec;

      checkRows    ( rowslice2,  5UL );
      checkColumns ( rowslice2,  4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2,  0UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 10UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=  0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=  0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=  0 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=  0 ||
          rowslice2(2,0) !=  0 || rowslice2(2,1) !=  0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=  0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=  0 || rowslice2(3,2) !=  0 || rowslice2(3,3) !=  0 ||
          rowslice2(4,0) !=  0 || rowslice2(4,1) !=  0 || rowslice2(4,2) !=  0 || rowslice2(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   0 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) !=  0 || mat_(2,1,1) !=   0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=   0 || mat_(3,2,1) !=  0 || mat_(3,3,1) !=   0 ||
          mat_(4,0,1) !=  0 || mat_(4,1,1) !=   0 || mat_(4,2,1) !=  0 || mat_(4,3,1) !=   0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
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
      test_ = "dense matrix subtraction assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;
      using blaze::rowMajor;

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );

      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 80UL ) );
      AlignedPadded m( memory.get(), 5UL, 4UL, 16UL );
      m(0,0) =  0;
      m(0,1) =  0;
      m(0,2) =  0;
      m(0,3) =  0;
      m(1,0) =  0;
      m(1,1) =  1;
      m(1,2) =  0;
      m(1,3) =  0;
      m(2,0) = -2;
      m(2,1) =  0;
      m(2,2) = -3;
      m(2,3) =  0;
      m(3,0) =  0;
      m(3,1) =  4;
      m(3,2) =  5;
      m(3,3) = -6;
      m(4,0) =  7;
      m(4,1) = -8;
      m(4,2) =  9;
      m(4,3) = 10;

      rowslice2 -= m;

      checkRows    ( rowslice2,  5UL );
      checkColumns ( rowslice2,  4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2,  0UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 10UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=  0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=  0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=  0 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=  0 ||
          rowslice2(2,0) !=  0 || rowslice2(2,1) !=  0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=  0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=  0 || rowslice2(3,2) !=  0 || rowslice2(3,3) !=  0 ||
          rowslice2(4,0) !=  0 || rowslice2(4,1) !=  0 || rowslice2(4,2) !=  0 || rowslice2(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   0 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) !=  0 || mat_(2,1,1) !=   0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=   0 || mat_(3,2,1) !=  0 || mat_(3,3,1) !=   0 ||
          mat_(4,0,1) !=  0 || mat_(4,1,1) !=   0 || mat_(4,2,1) !=  0 || mat_(4,3,1) !=   0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
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

   {
      test_ = "dense matrix subtraction assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::rowMajor;

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );

      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
      std::unique_ptr<int[]> memory( new int[21] );
      UnalignedUnpadded m( memory.get()+1UL, 5UL, 4UL );
      m(0,0) =  0;
      m(0,1) =  0;
      m(0,2) =  0;
      m(0,3) =  0;
      m(1,0) =  0;
      m(1,1) =  1;
      m(1,2) =  0;
      m(1,3) =  0;
      m(2,0) = -2;
      m(2,1) =  0;
      m(2,2) = -3;
      m(2,3) =  0;
      m(3,0) =  0;
      m(3,1) =  4;
      m(3,2) =  5;
      m(3,3) = -6;
      m(4,0) =  7;
      m(4,1) = -8;
      m(4,2) =  9;
      m(4,3) = 10;

      rowslice2 -= m;

      checkRows    ( rowslice2,  5UL );
      checkColumns ( rowslice2,  4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2,  0UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 10UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=  0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=  0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=  0 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=  0 ||
          rowslice2(2,0) !=  0 || rowslice2(2,1) !=  0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=  0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=  0 || rowslice2(3,2) !=  0 || rowslice2(3,3) !=  0 ||
          rowslice2(4,0) !=  0 || rowslice2(4,1) !=  0 || rowslice2(4,2) !=  0 || rowslice2(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   0 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) !=  0 || mat_(2,1,1) !=   0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=   0 || mat_(3,2,1) !=  0 || mat_(3,3,1) !=   0 ||
          mat_(4,0,1) !=  0 || mat_(4,1,1) !=   0 || mat_(4,2,1) !=  0 || mat_(4,3,1) !=   0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
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
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the RowSlice multiplication assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the multiplication assignment operators of the RowSlice
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testMultAssign()
{
   //=====================================================================================
   // RowSlice multiplication assignment
   //=====================================================================================

   {
      test_ = "RowSlice multiplication assignment";

      initialize();

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT rowslice2 = blaze::rowslice( m, 1UL );
      rowslice2 *= blaze::rowslice( m, 0UL );

      checkRows    ( rowslice2, 3UL );
      checkColumns ( rowslice2, 3UL );
      checkCapacity( rowslice2, 9UL );
      checkNonZeros( rowslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( rowslice2(0,0) != 90 || rowslice2(0,1) != 114 || rowslice2(0,2) != 138 ||
          rowslice2(1,0) != 54 || rowslice2(1,1) !=  69 || rowslice2(1,2) !=  84 ||
          rowslice2(2,0) != 18 || rowslice2(2,1) !=  24 || rowslice2(2,2) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 90 114 138 )\n( 54 69 84 )\n( 18 24 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,1,0) !=   2 || m(0,2,0) !=   3 ||
          m(1,0,0) !=  4 || m(1,1,0) !=   5 || m(1,2,0) !=   6 ||
          m(2,0,0) !=  7 || m(2,1,0) !=   8 || m(2,2,0) !=   9 ||
          m(0,0,1) != 90 || m(0,1,1) != 114 || m(0,2,1) != 138 ||
          m(1,0,1) != 54 || m(1,1,1) !=  69 || m(1,2,1) !=  84 ||
          m(2,0,1) != 18 || m(2,1,1) !=  24 || m(2,2,1) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((   1   2   3 )\n"
                                     " (   4   5   6 )\n"
                                     " (   7   8   9 ))\n"
                                     "((  90 114 138 )\n"
                                     " (  54  69  84 )\n";
                                     " (  18  24  30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // dense matrix multiplication assignment
   //=====================================================================================

   {
      test_ = "dense matrix multiplication assignment (mixed type)";

      initialize();

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT rowslice2 = blaze::rowslice( m, 1UL );

      const blaze::DynamicMatrix<short, blaze::rowMajor> m1{
          {1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

      rowslice2 *= m1;

      checkRows    ( rowslice2, 3UL );
      checkColumns ( rowslice2, 3UL );
      checkCapacity( rowslice2, 9UL );
      checkNonZeros( rowslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( rowslice2(0,0) != 90 || rowslice2(0,1) != 114 || rowslice2(0,2) != 138 ||
          rowslice2(1,0) != 54 || rowslice2(1,1) !=  69 || rowslice2(1,2) !=  84 ||
          rowslice2(2,0) != 18 || rowslice2(2,1) !=  24 || rowslice2(2,2) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 90 114 138 )\n( 54 69 84 )\n( 18 24 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,1,0) !=   2 || m(0,2,0) !=   3 ||
          m(1,0,0) !=  4 || m(1,1,0) !=   5 || m(1,2,0) !=   6 ||
          m(2,0,0) !=  7 || m(2,1,0) !=   8 || m(2,2,0) !=   9 ||
          m(0,0,1) != 90 || m(0,1,1) != 114 || m(0,2,1) != 138 ||
          m(1,0,1) != 54 || m(1,1,1) !=  69 || m(1,2,1) !=  84 ||
          m(2,0,1) != 18 || m(2,1,1) !=  24 || m(2,2,1) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((   1   2   3 )\n"
                                     " (   4   5   6 )\n"
                                     " (   7   8   9 ))\n"
                                     "((  90 114 138 )\n"
                                     " (  54  69  84 )\n";
                                     " (  18  24  30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "dense matrix multiplication assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;
      using blaze::rowMajor;

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT rowslice2 = blaze::rowslice( m, 1UL );


      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 48UL ) );
      AlignedPadded m1( memory.get(), 3UL, 3UL, 16UL );
      m1(0,0) = 1;
      m1(0,1) = 2;
      m1(0,2) = 3;
      m1(1,0) = 4;
      m1(1,1) = 5;
      m1(1,2) = 6;
      m1(2,0) = 7;
      m1(2,1) = 8;
      m1(2,2) = 9;

      rowslice2 *= m1;

      checkRows    ( rowslice2, 3UL );
      checkColumns ( rowslice2, 3UL );
      checkCapacity( rowslice2, 9UL );
      checkNonZeros( rowslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( rowslice2(0,0) != 90 || rowslice2(0,1) != 114 || rowslice2(0,2) != 138 ||
          rowslice2(1,0) != 54 || rowslice2(1,1) !=  69 || rowslice2(1,2) !=  84 ||
          rowslice2(2,0) != 18 || rowslice2(2,1) !=  24 || rowslice2(2,2) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 90 114 138 )\n( 54 69 84 )\n( 18 24 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,1,0) !=   2 || m(0,2,0) !=   3 ||
          m(1,0,0) !=  4 || m(1,1,0) !=   5 || m(1,2,0) !=   6 ||
          m(2,0,0) !=  7 || m(2,1,0) !=   8 || m(2,2,0) !=   9 ||
          m(0,0,1) != 90 || m(0,1,1) != 114 || m(0,2,1) != 138 ||
          m(1,0,1) != 54 || m(1,1,1) !=  69 || m(1,2,1) !=  84 ||
          m(2,0,1) != 18 || m(2,1,1) !=  24 || m(2,2,1) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((   1   2   3 )\n"
                                     " (   4   5   6 )\n"
                                     " (   7   8   9 ))\n"
                                     "((  90 114 138 )\n"
                                     " (  54  69  84 )\n";
                                     " (  18  24  30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "dense matrix multiplication assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::rowMajor;

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT rowslice2 = blaze::rowslice( m, 1UL );

      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
      std::unique_ptr<int[]> memory( new int[10] );
      UnalignedUnpadded m1( memory.get()+1UL, 3UL , 3UL);
      m1(0,0) = 1;
      m1(0,1) = 2;
      m1(0,2) = 3;
      m1(1,0) = 4;
      m1(1,1) = 5;
      m1(1,2) = 6;
      m1(2,0) = 7;
      m1(2,1) = 8;
      m1(2,2) = 9;

      rowslice2 *= m1;

      checkRows    ( rowslice2, 3UL );
      checkColumns ( rowslice2, 3UL );
      checkCapacity( rowslice2, 9UL );
      checkNonZeros( rowslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( rowslice2(0,0) != 90 || rowslice2(0,1) != 114 || rowslice2(0,2) != 138 ||
          rowslice2(1,0) != 54 || rowslice2(1,1) !=  69 || rowslice2(1,2) !=  84 ||
          rowslice2(2,0) != 18 || rowslice2(2,1) !=  24 || rowslice2(2,2) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 90 114 138 )\n( 54 69 84 )\n( 18 24 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,1,0) !=   2 || m(0,2,0) !=   3 ||
          m(1,0,0) !=  4 || m(1,1,0) !=   5 || m(1,2,0) !=   6 ||
          m(2,0,0) !=  7 || m(2,1,0) !=   8 || m(2,2,0) !=   9 ||
          m(0,0,1) != 90 || m(0,1,1) != 114 || m(0,2,1) != 138 ||
          m(1,0,1) != 54 || m(1,1,1) !=  69 || m(1,2,1) !=  84 ||
          m(2,0,1) != 18 || m(2,1,1) !=  24 || m(2,2,1) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((   1   2   3 )\n"
                                     " (   4   5   6 )\n"
                                     " (   7   8   9 ))\n"
                                     "((  90 114 138 )\n"
                                     " (  54  69  84 )\n";
                                     " (  18  24  30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the RowSlice Schur product assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the Schur product assignment operators of the RowSlice
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testSchurAssign()
{
   //=====================================================================================
   // RowSlice Schur product assignment
   //=====================================================================================

   {
      test_ = "RowSlice Schur product assignment";

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT rowslice2 = blaze::rowslice( m, 1UL );
      rowslice2 %= blaze::rowslice( m, 0UL );

      checkRows    ( rowslice2, 3UL );
      checkColumns ( rowslice2, 3UL );
      checkCapacity( rowslice2, 9UL );
      checkNonZeros( rowslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( rowslice2(0,0) !=  9 || rowslice2(0,1) != 16 || rowslice2(0,2) != 21 ||
          rowslice2(1,0) != 24 || rowslice2(1,1) != 25 || rowslice2(1,2) != 24 ||
          rowslice2(2,0) != 21 || rowslice2(2,1) != 16 || rowslice2(2,2) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,1,0) !=  2 || m(0,2,0) !=  3 ||
          m(1,0,0) !=  4 || m(1,1,0) !=  5 || m(1,2,0) !=  6 ||
          m(2,0,0) !=  7 || m(2,1,0) !=  8 || m(2,2,0) !=  9 ||
          m(0,0,1) !=  9 || m(0,1,1) != 16 || m(0,2,1) != 21 ||
          m(1,0,1) != 24 || m(1,1,1) != 25 || m(1,2,1) != 24 ||
          m(2,0,1) != 21 || m(2,1,1) != 16 || m(2,2,1) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  1  2  3 )\n"
                                     " (  4  5  6 )\n"
                                     " (  7  8  9 ))\n"
                                     "((  9 16 21 )\n"
                                     " ( 24 25 24 )\n";
                                     " ( 21 16  9 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // dense matrix Schur product assignment
   //=====================================================================================

   {
      test_ = "dense vector Schur product assignment (mixed type)";

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT rowslice2 = blaze::rowslice( m, 1UL );

      const blaze::DynamicMatrix<short, blaze::rowMajor> m1{
          {1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

      rowslice2 %= m1;

      checkRows    ( rowslice2, 3UL );
      checkColumns ( rowslice2, 3UL );
      checkCapacity( rowslice2, 9UL );
      checkNonZeros( rowslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( rowslice2(0,0) !=  9 || rowslice2(0,1) != 16 || rowslice2(0,2) != 21 ||
          rowslice2(1,0) != 24 || rowslice2(1,1) != 25 || rowslice2(1,2) != 24 ||
          rowslice2(2,0) != 21 || rowslice2(2,1) != 16 || rowslice2(2,2) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,1,0) !=  2 || m(0,2,0) !=  3 ||
          m(1,0,0) !=  4 || m(1,1,0) !=  5 || m(1,2,0) !=  6 ||
          m(2,0,0) !=  7 || m(2,1,0) !=  8 || m(2,2,0) !=  9 ||
          m(0,0,1) !=  9 || m(0,1,1) != 16 || m(0,2,1) != 21 ||
          m(1,0,1) != 24 || m(1,1,1) != 25 || m(1,2,1) != 24 ||
          m(2,0,1) != 21 || m(2,1,1) != 16 || m(2,2,1) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  1  2  3 )\n"
                                     " (  4  5  6 )\n"
                                     " (  7  8  9 ))\n"
                                     "((  9 16 21 )\n"
                                     " ( 24 25 24 )\n";
                                     " ( 21 16  9 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "dense matrix Schur product assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;
      using blaze::rowMajor;

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT rowslice2 = blaze::rowslice( m, 1UL );

      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 48UL ) );
      AlignedPadded m1( memory.get(), 3UL, 3UL, 16UL );
      m1(0,0) = 1;
      m1(0,1) = 2;
      m1(0,2) = 3;
      m1(1,0) = 4;
      m1(1,1) = 5;
      m1(1,2) = 6;
      m1(2,0) = 7;
      m1(2,1) = 8;
      m1(2,2) = 9;

      rowslice2 %= m1;

      checkRows    ( rowslice2, 3UL );
      checkColumns ( rowslice2, 3UL );
      checkCapacity( rowslice2, 9UL );
      checkNonZeros( rowslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( rowslice2(0,0) !=  9 || rowslice2(0,1) != 16 || rowslice2(0,2) != 21 ||
          rowslice2(1,0) != 24 || rowslice2(1,1) != 25 || rowslice2(1,2) != 24 ||
          rowslice2(2,0) != 21 || rowslice2(2,1) != 16 || rowslice2(2,2) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,1,0) !=  2 || m(0,2,0) !=  3 ||
          m(1,0,0) !=  4 || m(1,1,0) !=  5 || m(1,2,0) !=  6 ||
          m(2,0,0) !=  7 || m(2,1,0) !=  8 || m(2,2,0) !=  9 ||
          m(0,0,1) !=  9 || m(0,1,1) != 16 || m(0,2,1) != 21 ||
          m(1,0,1) != 24 || m(1,1,1) != 25 || m(1,2,1) != 24 ||
          m(2,0,1) != 21 || m(2,1,1) != 16 || m(2,2,1) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  1  2  3 )\n"
                                     " (  4  5  6 )\n"
                                     " (  7  8  9 ))\n"
                                     "((  9 16 21 )\n"
                                     " ( 24 25 24 )\n";
                                     " ( 21 16  9 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "dense matrix Schur product assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::rowMajor;

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT rowslice2 = blaze::rowslice( m, 1UL );

      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
      std::unique_ptr<int[]> memory( new int[10] );
      UnalignedUnpadded m1( memory.get()+1UL, 3UL , 3UL);
      m1(0,0) = 1;
      m1(0,1) = 2;
      m1(0,2) = 3;
      m1(1,0) = 4;
      m1(1,1) = 5;
      m1(1,2) = 6;
      m1(2,0) = 7;
      m1(2,1) = 8;
      m1(2,2) = 9;

      rowslice2 %= m1;

      checkRows    ( rowslice2, 3UL );
      checkColumns ( rowslice2, 3UL );
      checkCapacity( rowslice2, 9UL );
      checkNonZeros( rowslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( rowslice2(0,0) !=  9 || rowslice2(0,1) != 16 || rowslice2(0,2) != 21 ||
          rowslice2(1,0) != 24 || rowslice2(1,1) != 25 || rowslice2(1,2) != 24 ||
          rowslice2(2,0) != 21 || rowslice2(2,1) != 16 || rowslice2(2,2) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,1,0) !=  2 || m(0,2,0) !=  3 ||
          m(1,0,0) !=  4 || m(1,1,0) !=  5 || m(1,2,0) !=  6 ||
          m(2,0,0) !=  7 || m(2,1,0) !=  8 || m(2,2,0) !=  9 ||
          m(0,0,1) !=  9 || m(0,1,1) != 16 || m(0,2,1) != 21 ||
          m(1,0,1) != 24 || m(1,1,1) != 25 || m(1,2,1) != 24 ||
          m(2,0,1) != 21 || m(2,1,1) != 16 || m(2,2,1) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  1  2  3 )\n"
                                     " (  4  5  6 )\n"
                                     " (  7  8  9 ))\n"
                                     "((  9 16 21 )\n"
                                     " ( 24 25 24 )\n";
                                     " ( 21 16  9 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of all RowSlice (self-)scaling operations.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all available ways to scale an instance of the RowSlice
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testScaling()
{
   //=====================================================================================
   // self-scaling (v*=2)
   //=====================================================================================

   {
      test_ = "self-scaling (v*=2)";

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );
      rowslice2 *= 3;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   3 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -6 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -9 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=  12 || rowslice2(3,2) != 15 || rowslice2(3,3) != -18 ||
          rowslice2(4,0) != 21 || rowslice2(4,1) != -24 || rowslice2(4,2) != 27 || rowslice2(4,3) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   3 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) != -6 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -9 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  12 || mat_(3,2,1) != 15 || mat_(3,3,1) != -18 ||
          mat_(4,0,1) != 21 || mat_(4,1,1) != -24 || mat_(4,2,1) != 27 || mat_(4,3,1) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   0   0   0 )\n"
                                     " (  0   3   0   0 )\n"
                                     " ( -6   0  -9   0 )\n"
                                     " (  0  12  15 -18 )\n"
                                     " ( 21 -24  27  30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // self-scaling (v=v*2)
   //=====================================================================================

   {
      test_ = "self-scaling (v=v*3)";

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );
      rowslice2 = rowslice2 * 3;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   3 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -6 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -9 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=  12 || rowslice2(3,2) != 15 || rowslice2(3,3) != -18 ||
          rowslice2(4,0) != 21 || rowslice2(4,1) != -24 || rowslice2(4,2) != 27 || rowslice2(4,3) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   3 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) != -6 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -9 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  12 || mat_(3,2,1) != 15 || mat_(3,3,1) != -18 ||
          mat_(4,0,1) != 21 || mat_(4,1,1) != -24 || mat_(4,2,1) != 27 || mat_(4,3,1) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   0   0   0 )\n"
                                     " (  0   3   0   0 )\n"
                                     " ( -6   0  -9   0 )\n"
                                     " (  0  12  15 -18 )\n"
                                     " ( 21 -24  27  30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // self-scaling (v=3*v)
   //=====================================================================================

   {
      test_ = "self-scaling (v=3*v)";

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );
      rowslice2 = 3 * rowslice2;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   3 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -6 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -9 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=  12 || rowslice2(3,2) != 15 || rowslice2(3,3) != -18 ||
          rowslice2(4,0) != 21 || rowslice2(4,1) != -24 || rowslice2(4,2) != 27 || rowslice2(4,3) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   3 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) != -6 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -9 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  12 || mat_(3,2,1) != 15 || mat_(3,3,1) != -18 ||
          mat_(4,0,1) != 21 || mat_(4,1,1) != -24 || mat_(4,2,1) != 27 || mat_(4,3,1) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   0   0   0 )\n"
                                     " (  0   3   0   0 )\n"
                                     " ( -6   0  -9   0 )\n"
                                     " (  0  12  15 -18 )\n"
                                     " ( 21 -24  27  30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // self-scaling (v/=s)
   //=====================================================================================

   {
      test_ = "self-scaling (v/=s)";

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );
      rowslice2 /= (1.0/3.0);

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   3 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -6 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -9 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=  12 || rowslice2(3,2) != 15 || rowslice2(3,3) != -18 ||
          rowslice2(4,0) != 21 || rowslice2(4,1) != -24 || rowslice2(4,2) != 27 || rowslice2(4,3) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   3 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) != -6 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -9 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  12 || mat_(3,2,1) != 15 || mat_(3,3,1) != -18 ||
          mat_(4,0,1) != 21 || mat_(4,1,1) != -24 || mat_(4,2,1) != 27 || mat_(4,3,1) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   0   0   0 )\n"
                                     " (  0   3   0   0 )\n"
                                     " ( -6   0  -9   0 )\n"
                                     " (  0  12  15 -18 )\n"
                                     " ( 21 -24  27  30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // self-scaling (v=v/s)
   //=====================================================================================

   {
      test_ = "self-scaling (v=v/s)";

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );
      rowslice2 = rowslice2 / (1.0/3.0);

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   3 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -6 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -9 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=  12 || rowslice2(3,2) != 15 || rowslice2(3,3) != -18 ||
          rowslice2(4,0) != 21 || rowslice2(4,1) != -24 || rowslice2(4,2) != 27 || rowslice2(4,3) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   3 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
          mat_(2,0,1) != -6 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -9 || mat_(2,3,1) !=   0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  12 || mat_(3,2,1) != 15 || mat_(3,3,1) != -18 ||
          mat_(4,0,1) != 21 || mat_(4,1,1) != -24 || mat_(4,2,1) != 27 || mat_(4,3,1) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   0   0   0 )\n"
                                     " (  0   3   0   0 )\n"
                                     " ( -6   0  -9   0 )\n"
                                     " (  0  12  15 -18 )\n"
                                     " ( 21 -24  27  30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // RowSlice::scale()
   //=====================================================================================

   {
      test_ = "RowSlice::scale()";

      initialize();

      // Integral scaling the 3rd rowslice
      {
         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         rowslice2.scale( 3 );

         checkRows    ( rowslice2, 5UL );
         checkColumns ( rowslice2, 4UL );
         checkCapacity( rowslice2, 20UL );
         checkNonZeros( rowslice2, 10UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 20UL );

         if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
             rowslice2(1,0) !=  0 || rowslice2(1,1) !=   3 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
             rowslice2(2,0) != -6 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -9 || rowslice2(2,3) !=   0 ||
             rowslice2(3,0) !=  0 || rowslice2(3,1) !=  12 || rowslice2(3,2) != 15 || rowslice2(3,3) != -18 ||
             rowslice2(4,0) != 21 || rowslice2(4,1) != -24 || rowslice2(4,2) != 27 || rowslice2(4,3) !=  30 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Result:\n" << rowslice2 << "\n"
                   << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
             mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
             mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
             mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
             mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
             mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
             mat_(1,0,1) !=  0 || mat_(1,1,1) !=   3 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
             mat_(2,0,1) != -6 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -9 || mat_(2,3,1) !=   0 ||
             mat_(3,0,1) !=  0 || mat_(3,1,1) !=  12 || mat_(3,2,1) != 15 || mat_(3,3,1) != -18 ||
             mat_(4,0,1) != 21 || mat_(4,1,1) != -24 || mat_(4,2,1) != 27 || mat_(4,3,1) !=  30 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0  -3   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  7  -8   9  10 ))\n"
                                        "((  0   0   0   0 )\n"
                                        " (  0   3   0   0 )\n"
                                        " ( -6   0  -9   0 )\n"
                                        " (  0  12  15 -18 )\n"
                                        " ( 21 -24  27  30 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      initialize();

      // Floating point scaling the 3rd rowslice
      {
         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         rowslice2.scale( 0.5 );

         checkRows    ( rowslice2,  5UL );
         checkColumns ( rowslice2,  4UL );
         checkCapacity( rowslice2, 20UL );
         checkNonZeros( rowslice2,  9UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 19UL );

         if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=  0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=  0 ||
             rowslice2(1,0) !=  0 || rowslice2(1,1) !=  0 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=  0 ||
             rowslice2(2,0) != -1 || rowslice2(2,1) !=  0 || rowslice2(2,2) != -1 || rowslice2(2,3) !=  0 ||
             rowslice2(3,0) !=  0 || rowslice2(3,1) !=  2 || rowslice2(3,2) !=  2 || rowslice2(3,3) != -3 ||
             rowslice2(4,0) !=  3 || rowslice2(4,1) != -4 || rowslice2(4,2) !=  4 || rowslice2(4,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Result:\n" << rowslice2 << "\n"
                   << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( -1 0 -1 0 )\n( 0 12 2 -3 )\n( 3 -4 4 5 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=   0 ||
             mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=   0 ||
             mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=   0 ||
             mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) !=  -6 ||
             mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) !=  10 ||
             mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=   0 ||
             mat_(1,0,1) !=  0 || mat_(1,1,1) !=   0 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=   0 ||
             mat_(2,0,1) != -1 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -1 || mat_(2,3,1) !=   0 ||
             mat_(3,0,1) !=  0 || mat_(3,1,1) !=   2 || mat_(3,2,1) !=  2 || mat_(3,3,1) !=  -3 ||
             mat_(4,0,1) !=  3 || mat_(4,1,1) !=  -4 || mat_(4,2,1) !=  4 || mat_(4,3,1) !=   5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0  -3   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  7  -8   9  10 ))\n"
                                        "((  0   0   0   0 )\n"
                                        " (  0   0   0   0 )\n"
                                        " ( -1   0  -1   0 )\n"
                                        " (  0   2   2  -3 )\n"
                                        " (  3  -4   4   5 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the RowSlice function call operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the function call operator
// of the RowSlice specialization. In case an error is detected, a \a std::runtime_error exception
// is thrown.
*/
void DenseGeneralTest::testFunctionCall()
{
   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "RowSlice::operator()";

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );

      // Assignment to the element at index (0,1)
      rowslice2(0,1) = 9;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 11UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 21UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   9 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -3 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
          rowslice2(4,0) !=  7 || rowslice2(4,1) !=  -8 || rowslice2(4,2) !=  9 || rowslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=  9 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) != -2 || mat_(2,1,1) !=  0 || mat_(2,2,1) != -3 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) !=  7 || mat_(4,1,1) != -8 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   9   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Assignment to the element at index (2,2)
      rowslice2(2,2) = 0;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   9 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
          rowslice2(4,0) !=  7 || rowslice2(4,1) !=  -8 || rowslice2(4,2) !=  9 || rowslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=  9 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) != -2 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) !=  7 || mat_(4,1,1) != -8 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   9   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0   0   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Assignment to the element at index (4,1)
      rowslice2(4,1) = -9;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   9 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
          rowslice2(4,0) !=  7 || rowslice2(4,1) !=  -9 || rowslice2(4,2) !=  9 || rowslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   9 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) != -2 || mat_(2,1,1) !=   0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=   4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) !=  7 || mat_(4,1,1) !=  -9 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   9   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0   0   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -9   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Addition assignment to the element at index (0,1)
      rowslice2(0,1) += -3;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   6 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
          rowslice2(4,0) !=  7 || rowslice2(4,1) !=  -9 || rowslice2(4,2) !=  9 || rowslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=  6 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) != -2 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) !=  7 || mat_(4,1,1) != -9 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   6   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -9   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Subtraction assignment to the element at index (2,0)
      rowslice2(2,0) -= 6;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   6 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -8 || rowslice2(2,1) !=   0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
          rowslice2(4,0) !=  7 || rowslice2(4,1) !=  -9 || rowslice2(4,2) !=  9 || rowslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=  6 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) != -8 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) !=  7 || mat_(4,1,1) != -9 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   6   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -8   0   0   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -9   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Multiplication assignment to the element at index (4,0)
      rowslice2(4,0) *= -3;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=   0 || rowslice2(0,1) !=   6 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=   0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) !=  -8 || rowslice2(2,1) !=   0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=   0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
          rowslice2(4,0) != -21 || rowslice2(4,1) !=  -9 || rowslice2(4,2) !=  9 || rowslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -6 )\n( -21 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=   0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=   0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) !=  -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=   0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=   7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=   0 || mat_(0,1,1) !=  6 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=   0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) !=  -8 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=   0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) != -21 || mat_(4,1,1) != -9 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((   0   0   0   0 )\n"
                                     " (   0   1   0   0 )\n"
                                     " (  -2   0  -3   0 )\n"
                                     " (   0   4   5  -6 )\n"
                                     " (   7  -8   9  10 ))\n"
                                     "((   0   6   0   0 )\n"
                                     " (   0   1   0   0 )\n"
                                     " (  -8   0   0   0 )\n"
                                     " (   0   4   5  -6 )\n"
                                     " ( -21  -9   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Division assignment to the element at index (3,3)
      rowslice2(3,3) /= 2;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=   0 || rowslice2(0,1) !=   6 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=   0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) !=  -8 || rowslice2(2,1) !=   0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=   0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -3 ||
          rowslice2(4,0) != -21 || rowslice2(4,1) !=  -9 || rowslice2(4,2) !=  9 || rowslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -3 )\n( -21 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=   0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=   0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) !=  -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=   0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=   7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=   0 || mat_(0,1,1) !=  6 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=   0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) !=  -8 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=   0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -3 ||
          mat_(4,0,1) != -21 || mat_(4,1,1) != -9 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((   0   0   0   0 )\n"
                                     " (   0   1   0   0 )\n"
                                     " (  -2   0  -3   0 )\n"
                                     " (   0   4   5  -6 )\n"
                                     " (   7  -8   9  10 ))\n"
                                     "((   0   6   0   0 )\n"
                                     " (   0   1   0   0 )\n"
                                     " (  -8   0   0   0 )\n"
                                     " (   0   4   5  -3 )\n"
                                     " ( -21  -9   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the RowSlice at() operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the at() operator
// of the RowSlice specialization. In case an error is detected, a \a std::runtime_error exception
// is thrown.
*/
void DenseGeneralTest::testAt()
{
   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "RowSlice::at()";

      initialize();

      RT rowslice2 = blaze::rowslice( mat_, 1UL );

      // Assignment to the element at index (0,1)
      rowslice2.at(0,1) = 9;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 11UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 21UL );

      if( rowslice2.at(0,0) !=  0 || rowslice2.at(0,1) !=   9 || rowslice2.at(0,2) !=  0 || rowslice2.at(0,3) !=   0 ||
          rowslice2.at(1,0) !=  0 || rowslice2.at(1,1) !=   1 || rowslice2.at(1,2) !=  0 || rowslice2.at(1,3) !=   0 ||
          rowslice2.at(2,0) != -2 || rowslice2.at(2,1) !=   0 || rowslice2.at(2,2) != -3 || rowslice2.at(2,3) !=   0 ||
          rowslice2.at(3,0) !=  0 || rowslice2.at(3,1) !=   4 || rowslice2.at(3,2) !=  5 || rowslice2.at(3,3) !=  -6 ||
          rowslice2.at(4,0) !=  7 || rowslice2.at(4,1) !=  -8 || rowslice2.at(4,2) !=  9 || rowslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=  9 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) != -2 || mat_(2,1,1) !=  0 || mat_(2,2,1) != -3 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) !=  7 || mat_(4,1,1) != -8 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   9   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Assignment to the element at index (2,2)
      rowslice2.at(2,2) = 0;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2.at(0,0) !=  0 || rowslice2.at(0,1) !=   9 || rowslice2.at(0,2) !=  0 || rowslice2.at(0,3) !=   0 ||
          rowslice2.at(1,0) !=  0 || rowslice2.at(1,1) !=   1 || rowslice2.at(1,2) !=  0 || rowslice2.at(1,3) !=   0 ||
          rowslice2.at(2,0) != -2 || rowslice2.at(2,1) !=   0 || rowslice2.at(2,2) !=  0 || rowslice2.at(2,3) !=   0 ||
          rowslice2.at(3,0) !=  0 || rowslice2.at(3,1) !=   4 || rowslice2.at(3,2) !=  5 || rowslice2.at(3,3) !=  -6 ||
          rowslice2.at(4,0) !=  7 || rowslice2.at(4,1) !=  -8 || rowslice2.at(4,2) !=  9 || rowslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=  9 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) != -2 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) !=  7 || mat_(4,1,1) != -8 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   9   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0   0   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Assignment to the element at index (4,1)
      rowslice2.at(4,1) = -9;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2.at(0,0) !=  0 || rowslice2.at(0,1) !=   9 || rowslice2.at(0,2) !=  0 || rowslice2.at(0,3) !=   0 ||
          rowslice2.at(1,0) !=  0 || rowslice2.at(1,1) !=   1 || rowslice2.at(1,2) !=  0 || rowslice2.at(1,3) !=   0 ||
          rowslice2.at(2,0) != -2 || rowslice2.at(2,1) !=   0 || rowslice2.at(2,2) !=  0 || rowslice2.at(2,3) !=   0 ||
          rowslice2.at(3,0) !=  0 || rowslice2.at(3,1) !=   4 || rowslice2.at(3,2) !=  5 || rowslice2.at(3,3) !=  -6 ||
          rowslice2.at(4,0) !=  7 || rowslice2.at(4,1) !=  -9 || rowslice2.at(4,2) !=  9 || rowslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=   9 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=   1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) != -2 || mat_(2,1,1) !=   0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=   4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) !=  7 || mat_(4,1,1) !=  -9 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   9   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0   0   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -9   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Addition assignment to the element at index (0,1)
      rowslice2.at(0,1) += -3;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2.at(0,0) !=  0 || rowslice2.at(0,1) !=   6 || rowslice2.at(0,2) !=  0 || rowslice2.at(0,3) !=   0 ||
          rowslice2.at(1,0) !=  0 || rowslice2.at(1,1) !=   1 || rowslice2.at(1,2) !=  0 || rowslice2.at(1,3) !=   0 ||
          rowslice2.at(2,0) != -2 || rowslice2.at(2,1) !=   0 || rowslice2.at(2,2) !=  0 || rowslice2.at(2,3) !=   0 ||
          rowslice2.at(3,0) !=  0 || rowslice2.at(3,1) !=   4 || rowslice2.at(3,2) !=  5 || rowslice2.at(3,3) !=  -6 ||
          rowslice2.at(4,0) !=  7 || rowslice2.at(4,1) !=  -9 || rowslice2.at(4,2) !=  9 || rowslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=  6 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) != -2 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) !=  7 || mat_(4,1,1) != -9 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   6   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -9   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Subtraction assignment to the element at index (2,0)
      rowslice2.at(2,0) -= 6;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2.at(0,0) !=  0 || rowslice2.at(0,1) !=   6 || rowslice2.at(0,2) !=  0 || rowslice2.at(0,3) !=   0 ||
          rowslice2.at(1,0) !=  0 || rowslice2.at(1,1) !=   1 || rowslice2.at(1,2) !=  0 || rowslice2.at(1,3) !=   0 ||
          rowslice2.at(2,0) != -8 || rowslice2.at(2,1) !=   0 || rowslice2.at(2,2) !=  0 || rowslice2.at(2,3) !=   0 ||
          rowslice2.at(3,0) !=  0 || rowslice2.at(3,1) !=   4 || rowslice2.at(3,2) !=  5 || rowslice2.at(3,3) !=  -6 ||
          rowslice2.at(4,0) !=  7 || rowslice2.at(4,1) !=  -9 || rowslice2.at(4,2) !=  9 || rowslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=  0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) != -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=  0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=  7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=  0 || mat_(0,1,1) !=  6 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=  0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) != -8 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=  0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) !=  7 || mat_(4,1,1) != -9 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  0   0   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -2   0  -3   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -8   9  10 ))\n"
                                     "((  0   6   0   0 )\n"
                                     " (  0   1   0   0 )\n"
                                     " ( -8   0   0   0 )\n"
                                     " (  0   4   5  -6 )\n"
                                     " (  7  -9   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Multiplication assignment to the element at index (4,0)
      rowslice2.at(4,0) *= -3;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2.at(0,0) !=   0 || rowslice2.at(0,1) !=   6 || rowslice2.at(0,2) !=  0 || rowslice2.at(0,3) !=   0 ||
          rowslice2.at(1,0) !=   0 || rowslice2.at(1,1) !=   1 || rowslice2.at(1,2) !=  0 || rowslice2.at(1,3) !=   0 ||
          rowslice2.at(2,0) !=  -8 || rowslice2.at(2,1) !=   0 || rowslice2.at(2,2) !=  0 || rowslice2.at(2,3) !=   0 ||
          rowslice2.at(3,0) !=   0 || rowslice2.at(3,1) !=   4 || rowslice2.at(3,2) !=  5 || rowslice2.at(3,3) !=  -6 ||
          rowslice2.at(4,0) != -21 || rowslice2.at(4,1) !=  -9 || rowslice2.at(4,2) !=  9 || rowslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -6 )\n( -21 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=   0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=   0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) !=  -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=   0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=   7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=   0 || mat_(0,1,1) !=  6 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=   0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) !=  -8 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=   0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
          mat_(4,0,1) != -21 || mat_(4,1,1) != -9 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((   0   0   0   0 )\n"
                                     " (   0   1   0   0 )\n"
                                     " (  -2   0  -3   0 )\n"
                                     " (   0   4   5  -6 )\n"
                                     " (   7  -8   9  10 ))\n"
                                     "((   0   6   0   0 )\n"
                                     " (   0   1   0   0 )\n"
                                     " (  -8   0   0   0 )\n"
                                     " (   0   4   5  -6 )\n"
                                     " ( -21  -9   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Division assignment to the element at index (3,3)
      rowslice2.at(3,3) /= 2;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2.at(0,0) !=   0 || rowslice2.at(0,1) !=   6 || rowslice2.at(0,2) !=  0 || rowslice2.at(0,3) !=   0 ||
          rowslice2.at(1,0) !=   0 || rowslice2.at(1,1) !=   1 || rowslice2.at(1,2) !=  0 || rowslice2.at(1,3) !=   0 ||
          rowslice2.at(2,0) !=  -8 || rowslice2.at(2,1) !=   0 || rowslice2.at(2,2) !=  0 || rowslice2.at(2,3) !=   0 ||
          rowslice2.at(3,0) !=   0 || rowslice2.at(3,1) !=   4 || rowslice2.at(3,2) !=  5 || rowslice2.at(3,3) !=  -3 ||
          rowslice2.at(4,0) != -21 || rowslice2.at(4,1) !=  -9 || rowslice2.at(4,2) !=  9 || rowslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -3 )\n( -21 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=   0 || mat_(0,1,0) !=  0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
          mat_(1,0,0) !=   0 || mat_(1,1,0) !=  1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
          mat_(2,0,0) !=  -2 || mat_(2,1,0) !=  0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
          mat_(3,0,0) !=   0 || mat_(3,1,0) !=  4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
          mat_(4,0,0) !=   7 || mat_(4,1,0) != -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
          mat_(0,0,1) !=   0 || mat_(0,1,1) !=  6 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
          mat_(1,0,1) !=   0 || mat_(1,1,1) !=  1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
          mat_(2,0,1) !=  -8 || mat_(2,1,1) !=  0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
          mat_(3,0,1) !=   0 || mat_(3,1,1) !=  4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -3 ||
          mat_(4,0,1) != -21 || mat_(4,1,1) != -9 || mat_(4,2,1) !=  9 || mat_(4,3,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((   0   0   0   0 )\n"
                                     " (   0   1   0   0 )\n"
                                     " (  -2   0  -3   0 )\n"
                                     " (   0   4   5  -6 )\n"
                                     " (   7  -8   9  10 ))\n"
                                     "((   0   6   0   0 )\n"
                                     " (   0   1   0   0 )\n"
                                     " (  -8   0   0   0 )\n"
                                     " (   0   4   5  -3 )\n"
                                     " ( -21  -9   9  10 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the RowSlice iterator implementation.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the iterator implementation of the RowSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testIterator()
{
   //=====================================================================================
   // matrix tests
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

         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         RT::ConstIterator it( begin( rowslice2, 2UL ) );

         if( it == end( rowslice2, 2UL ) || *it != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st rowslice via Iterator (end-begin)
      {
         test_ = "Iterator subtraction (end-begin)";

         RT rowslice1 = blaze::rowslice( mat_, 1UL );
         const ptrdiff_t number( end( rowslice1, 2UL ) - begin( rowslice1, 2UL ) );

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

      // Counting the number of elements in 1st rowslice via Iterator (begin-end)
      {
         test_ = "Iterator subtraction (begin-end)";

         RT rowslice1 = blaze::rowslice( mat_, 1UL );
         const ptrdiff_t number( begin( rowslice1, 2UL ) - end( rowslice1, 2UL ) );

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

      // Counting the number of elements in 2nd rowslice via ConstIterator (end-begin)
      {
         test_ = "ConstIterator subtraction (end-begin)";

         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         const ptrdiff_t number( cend( rowslice2, 2UL ) - cbegin( rowslice2, 2UL ) );

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

      // Counting the number of elements in 2nd rowslice via ConstIterator (begin-end)
      {
         test_ = "ConstIterator subtraction (begin-end)";

         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         const ptrdiff_t number( cbegin( rowslice2, 2UL ) - cend( rowslice2, 2UL ) );

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

         RT rowslice3 = blaze::rowslice( mat_, 0UL );
         RT::ConstIterator it ( cbegin( rowslice3, 4UL ) );
         RT::ConstIterator end( cend( rowslice3, 4UL ) );

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

         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         int value = 6;

         for( RT::Iterator it=begin( rowslice2, 4UL ); it!=end( rowslice2, 4UL ); ++it ) {
            *it = value++;
         }

         if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
             rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
             rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -3 || rowslice2(2,3) !=   0 ||
             rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
             rowslice2(4,0) !=  6 || rowslice2(4,1) !=   7 || rowslice2(4,2) !=  8 || rowslice2(4,3) !=   9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 6 7 8 9 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
             mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
             mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
             mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
             mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
             mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
             mat_(1,0,1) !=  0 || mat_(1,1,1) !=   1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
             mat_(2,0,1) != -2 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -3 || mat_(2,3,1) !=  0 ||
             mat_(3,0,1) !=  0 || mat_(3,1,1) !=   4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
             mat_(4,0,1) !=  6 || mat_(4,1,1) !=   7 || mat_(4,2,1) !=  8 || mat_(4,3,1) !=  9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0  -3   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  7  -8   9  10 ))\n"
                                        "((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0   0   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  6   7   8   9 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "addition assignment via Iterator";

         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         int value = 2;

         for( RT::Iterator it=begin( rowslice2, 4UL ); it!=end( rowslice2, 4UL ); ++it ) {
            *it += value++;
         }

         if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
             rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
             rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -3 || rowslice2(2,3) !=   0 ||
             rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
             rowslice2(4,0) !=  8 || rowslice2(4,1) !=  10 || rowslice2(4,2) != 12 || rowslice2(4,3) !=  14 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 8 10 12 14 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
             mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
             mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
             mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
             mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
             mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
             mat_(1,0,1) !=  0 || mat_(1,1,1) !=   1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
             mat_(2,0,1) != -2 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -3 || mat_(2,3,1) !=  0 ||
             mat_(3,0,1) !=  0 || mat_(3,1,1) !=   4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
             mat_(4,0,1) !=  8 || mat_(4,1,1) !=  10 || mat_(4,2,1) != 12 || mat_(4,3,1) != 14 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0  -3   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  7  -8   9  10 ))\n"
                                        "((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0   0   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  8  10  12  14 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "subtraction assignment via Iterator";

         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         int value = 2;

         for( RT::Iterator it=begin( rowslice2, 4UL ); it!=end( rowslice2, 4UL ); ++it ) {
            *it -= value++;
         }

         if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
             rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
             rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -3 || rowslice2(2,3) !=   0 ||
             rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
             rowslice2(4,0) !=  6 || rowslice2(4,1) !=   7 || rowslice2(4,2) !=  8 || rowslice2(4,3) !=   9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 6 7 8 9 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
             mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
             mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
             mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
             mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
             mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
             mat_(1,0,1) !=  0 || mat_(1,1,1) !=   1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
             mat_(2,0,1) != -2 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -3 || mat_(2,3,1) !=  0 ||
             mat_(3,0,1) !=  0 || mat_(3,1,1) !=   4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
             mat_(4,0,1) !=  6 || mat_(4,1,1) !=   7 || mat_(4,2,1) !=  8 || mat_(4,3,1) !=  9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0  -3   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  7  -8   9  10 ))\n"
                                        "((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0   0   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  6   7   8   9 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "multiplication assignment via Iterator";

         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         int value = 1;

         for( RT::Iterator it=begin( rowslice2, 4UL ); it!=end( rowslice2, 4UL ); ++it ) {
            *it *= value++;
         }

         if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
             rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
             rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -3 || rowslice2(2,3) !=   0 ||
             rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
             rowslice2(4,0) !=  6 || rowslice2(4,1) !=  14 || rowslice2(4,2) != 24 || rowslice2(4,3) !=  36 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 6 14 24 36 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
             mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
             mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
             mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
             mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
             mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
             mat_(1,0,1) !=  0 || mat_(1,1,1) !=   1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
             mat_(2,0,1) != -2 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -3 || mat_(2,3,1) !=  0 ||
             mat_(3,0,1) !=  0 || mat_(3,1,1) !=   4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
             mat_(4,0,1) !=  6 || mat_(4,1,1) !=  14 || mat_(4,2,1) != 24 || mat_(4,3,1) != 36 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0  -3   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  7  -8   9  10 ))\n"
                                        "((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0   0   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  6  14  24  36 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "division assignment via Iterator";

         RT rowslice2 = blaze::rowslice( mat_, 1UL );

         for( RT::Iterator it=begin( rowslice2, 4UL ); it!=end( rowslice2, 4UL ); ++it ) {
            *it /= 2;
         }

         if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
             rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
             rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -3 || rowslice2(2,3) !=   0 ||
             rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
             rowslice2(4,0) !=  3 || rowslice2(4,1) !=   7 || rowslice2(4,2) != 12 || rowslice2(4,3) !=  18 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 3 7 12 18 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
             mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
             mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
             mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
             mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
             mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
             mat_(1,0,1) !=  0 || mat_(1,1,1) !=   1 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
             mat_(2,0,1) != -2 || mat_(2,1,1) !=   0 || mat_(2,2,1) != -3 || mat_(2,3,1) !=  0 ||
             mat_(3,0,1) !=  0 || mat_(3,1,1) !=   4 || mat_(3,2,1) !=  5 || mat_(3,3,1) != -6 ||
             mat_(4,0,1) !=  3 || mat_(4,1,1) !=   7 || mat_(4,2,1) != 12 || mat_(4,3,1) != 18 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0  -3   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  7  -8   9  10 ))\n"
                                        "((  0   0   0   0 )\n"
                                        " (  0   1   0   0 )\n"
                                        " ( -2   0   0   0 )\n"
                                        " (  0   4   5  -6 )\n"
                                        " (  3   7  12  18 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c nonZeros() member function of the RowSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c nonZeros() member function of the RowSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testNonZeros()
{
   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "RowSlice::nonZeros()";

      initialize();

      // Initialization check
      RT rowslice2 = blaze::rowslice( mat_, 1UL );

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) != -3 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
          rowslice2(4,0) !=  7 || rowslice2(4,1) !=  -8 || rowslice2(4,2) !=  9 || rowslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the dense rowslice
      rowslice2(2, 2) = 0;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2,  9UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 19UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
          rowslice2(4,0) !=  7 || rowslice2(4,1) !=  -8 || rowslice2(4,2) !=  9 || rowslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the dense matrix
      mat_(3,0,1) = 5;

      checkRows    ( rowslice2, 5UL );
      checkColumns ( rowslice2, 4UL );
      checkCapacity( rowslice2, 20UL );
      checkNonZeros( rowslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
          rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
          rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=   0 ||
          rowslice2(3,0) !=  5 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
          rowslice2(4,0) !=  7 || rowslice2(4,1) !=  -8 || rowslice2(4,2) !=  9 || rowslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Matrix function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << rowslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 5 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reset() member function of the RowSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reset() member function of the RowSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testReset()
{
   using blaze::reset;


   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "RowSlice::reset()";

      // Resetting a single element in rowslice 3
      {
         initialize();

         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         reset( rowslice2(2, 2) );


         checkRows    ( rowslice2, 5UL );
         checkColumns ( rowslice2, 4UL );
         checkCapacity( rowslice2, 20UL );
         checkNonZeros( rowslice2,  9UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 19UL );

         if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
             rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
             rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=   0 ||
             rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
             rowslice2(4,0) !=  7 || rowslice2(4,1) !=  -8 || rowslice2(4,2) !=  9 || rowslice2(4,3) !=  10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Resetting the 1st rowslice (lvalue)
      {
         initialize();

         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         reset( rowslice2 );

         checkRows    ( rowslice2, 5UL );
         checkColumns ( rowslice2, 4UL );
         checkCapacity( rowslice2, 20UL );
         checkNonZeros( rowslice2,  0UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 10UL );

         if( rowslice2(0,0) != 0 || rowslice2(0,1) !=  0 || rowslice2(0,2) != 0 || rowslice2(0,3) != 0 ||
             rowslice2(1,0) != 0 || rowslice2(1,1) !=  0 || rowslice2(1,2) != 0 || rowslice2(1,3) != 0 ||
             rowslice2(2,0) != 0 || rowslice2(2,1) !=  0 || rowslice2(2,2) != 0 || rowslice2(2,3) != 0 ||
             rowslice2(3,0) != 0 || rowslice2(3,1) !=  0 || rowslice2(3,2) != 0 || rowslice2(3,3) != 0 ||
             rowslice2(4,0) != 0 || rowslice2(4,1) !=  0 || rowslice2(4,2) != 0 || rowslice2(4,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 1st rowslice failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Resetting the 1st rowslice (rvalue)
      {
         initialize();

         reset( blaze::rowslice( mat_, 1UL ) );

         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 10UL );

         if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
             mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
             mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
             mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
             mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
             mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
             mat_(1,0,1) !=  0 || mat_(1,1,1) !=   0 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
             mat_(2,0,1) !=  0 || mat_(2,1,1) !=   0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
             mat_(3,0,1) !=  0 || mat_(3,1,1) !=   0 || mat_(3,2,1) !=  0 || mat_(3,3,1) !=  0 ||
             mat_(4,0,1) !=  0 || mat_(4,1,1) !=   0 || mat_(4,2,1) !=  0 || mat_(4,3,1) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 1st rowslice failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
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
/*!\brief Test of the \c clear() function with the RowSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c clear() function with the RowSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testClear()
{
   using blaze::clear;


   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "clear() function";

      // Clearing a single element in rowslice 1
      {
         initialize();

         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         clear( rowslice2(2, 2) );

         checkRows    ( rowslice2, 5UL );
         checkColumns ( rowslice2, 4UL );
         checkCapacity( rowslice2, 20UL );
         checkNonZeros( rowslice2,  9UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 19UL );

         if( rowslice2(0,0) !=  0 || rowslice2(0,1) !=   0 || rowslice2(0,2) !=  0 || rowslice2(0,3) !=   0 ||
             rowslice2(1,0) !=  0 || rowslice2(1,1) !=   1 || rowslice2(1,2) !=  0 || rowslice2(1,3) !=   0 ||
             rowslice2(2,0) != -2 || rowslice2(2,1) !=   0 || rowslice2(2,2) !=  0 || rowslice2(2,3) !=   0 ||
             rowslice2(3,0) !=  0 || rowslice2(3,1) !=   4 || rowslice2(3,2) !=  5 || rowslice2(3,3) !=  -6 ||
             rowslice2(4,0) !=  7 || rowslice2(4,1) !=  -8 || rowslice2(4,2) !=  9 || rowslice2(4,3) !=  10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Clear operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Clearing the 3rd rowslice (lvalue)
      {
         initialize();

         RT rowslice2 = blaze::rowslice( mat_, 1UL );
         clear( rowslice2 );

         checkRows    ( rowslice2, 5UL );
         checkColumns ( rowslice2, 4UL );
         checkCapacity( rowslice2, 20UL );
         checkNonZeros( rowslice2,  0UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 10UL );

         if( rowslice2(0,0) != 0 || rowslice2(0,1) !=  0 || rowslice2(0,2) != 0 || rowslice2(0,3) != 0 ||
             rowslice2(1,0) != 0 || rowslice2(1,1) !=  0 || rowslice2(1,2) != 0 || rowslice2(1,3) != 0 ||
             rowslice2(2,0) != 0 || rowslice2(2,1) !=  0 || rowslice2(2,2) != 0 || rowslice2(2,3) != 0 ||
             rowslice2(3,0) != 0 || rowslice2(3,1) !=  0 || rowslice2(3,2) != 0 || rowslice2(3,3) != 0 ||
             rowslice2(4,0) != 0 || rowslice2(4,1) !=  0 || rowslice2(4,2) != 0 || rowslice2(4,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Clear operation of 3rd rowslice failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Clearing the 4th rowslice (rvalue)
      {
         initialize();

         clear( blaze::rowslice( mat_, 1UL ) );

         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 10UL );

         if( mat_(0,0,0) !=  0 || mat_(0,1,0) !=   0 || mat_(0,2,0) !=  0 || mat_(0,3,0) !=  0 ||
             mat_(1,0,0) !=  0 || mat_(1,1,0) !=   1 || mat_(1,2,0) !=  0 || mat_(1,3,0) !=  0 ||
             mat_(2,0,0) != -2 || mat_(2,1,0) !=   0 || mat_(2,2,0) != -3 || mat_(2,3,0) !=  0 ||
             mat_(3,0,0) !=  0 || mat_(3,1,0) !=   4 || mat_(3,2,0) !=  5 || mat_(3,3,0) != -6 ||
             mat_(4,0,0) !=  7 || mat_(4,1,0) !=  -8 || mat_(4,2,0) !=  9 || mat_(4,3,0) != 10 ||
             mat_(0,0,1) !=  0 || mat_(0,1,1) !=   0 || mat_(0,2,1) !=  0 || mat_(0,3,1) !=  0 ||
             mat_(1,0,1) !=  0 || mat_(1,1,1) !=   0 || mat_(1,2,1) !=  0 || mat_(1,3,1) !=  0 ||
             mat_(2,0,1) !=  0 || mat_(2,1,1) !=   0 || mat_(2,2,1) !=  0 || mat_(2,3,1) !=  0 ||
             mat_(3,0,1) !=  0 || mat_(3,1,1) !=   0 || mat_(3,2,1) !=  0 || mat_(3,3,1) !=  0 ||
             mat_(4,0,1) !=  0 || mat_(4,1,1) !=   0 || mat_(4,2,1) !=  0 || mat_(4,3,1) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Clear operation of 1st rowslice failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
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
/*!\brief Test of the \c isDefault() function with the RowSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDefault() function with the RowSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testIsDefault()
{
   using blaze::isDefault;


   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "isDefault() function";

      initialize();

      // isDefault with default rowslice
      {
         RT rowslice0 = blaze::rowslice( mat_, 0UL );
         rowslice0 = 0;

         if( isDefault( rowslice0(0, 0) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   RowSlice element: " << rowslice0(0, 0) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( rowslice0 ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   RowSlice:\n" << rowslice0 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default rowslice
      {
         RT rowslice1 = blaze::rowslice( mat_, 1UL );

         if( isDefault( rowslice1(1, 1) ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   RowSlice element: " << rowslice1(1, 1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( rowslice1 ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   RowSlice:\n" << rowslice1 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isSame() function with the RowSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSame() function with the RowSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testIsSame()
{
   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "isSame() function";

      // isSame with matching rowslices
      {
         RT rowslice1 = blaze::rowslice( mat_, 1UL );
         RT rowslice2 = blaze::rowslice( mat_, 1UL );

         if( blaze::isSame( rowslice1, rowslice2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First rowslice:\n" << rowslice1 << "\n"
                << "   Second rowslice:\n" << rowslice2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching rowslices
      {
         RT rowslice1 = blaze::rowslice( mat_, 0UL );
         RT rowslice2 = blaze::rowslice( mat_, 1UL );

         rowslice1 = 42;

         if( blaze::isSame( rowslice1, rowslice2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First rowslice:\n" << rowslice1 << "\n"
                << "   Second rowslice:\n" << rowslice2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with rowslice and matching submatrix
      {
         RT   rowslice1 = blaze::rowslice( mat_, 1UL );
         auto sv   = blaze::submatrix( rowslice1, 0UL, 0UL, 4UL, 5UL );

         if( blaze::isSame( rowslice1, sv ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense rowslice:\n" << rowslice1 << "\n"
                << "   Dense submatrix:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sv, rowslice1 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense rowslice:\n" << rowslice1 << "\n"
                << "   Dense submatrix:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with rowslice and non-matching submatrix (different size)
      {
         RT   rowslice1 = blaze::rowslice( mat_, 1UL );
         auto sv   = blaze::submatrix( rowslice1, 0UL, 0UL, 3UL, 3UL );

         if( blaze::isSame( rowslice1, sv ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense rowslice:\n" << rowslice1 << "\n"
                << "   Dense submatrix:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sv, rowslice1 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense rowslice:\n" << rowslice1 << "\n"
                << "   Dense submatrix:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with rowslice and non-matching submatrix (different offset)
      {
         RT   rowslice1 = blaze::rowslice( mat_, 1UL );
         auto sv   = blaze::submatrix( rowslice1, 1UL, 1UL, 3UL, 3UL );

         if( blaze::isSame( rowslice1, sv ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense rowslice:\n" << rowslice1 << "\n"
                << "   Dense submatrix:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sv, rowslice1 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense rowslice:\n" << rowslice1 << "\n"
                << "   Dense submatrix:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

//       // isSame with matching rowslices on a common submatrix
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 1UL, 2UL, 3UL );
//          auto rowslice1 = blaze::rowslice( sm, 1UL );
//          auto rowslice2 = blaze::rowslice( sm, 1UL );
//
//          if( blaze::isSame( rowslice1, rowslice2 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching rowslices on a common submatrix
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 1UL, 2UL, 3UL );
//          auto rowslice1 = blaze::rowslice( sm, 0UL );
//          auto rowslice2 = blaze::rowslice( sm, 1UL );
//
//          if( blaze::isSame( rowslice1, rowslice2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with matching subtensor on matrix and submatrix
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto rowslice1 = blaze::rowslice( mat_, 2UL );
//          auto rowslice2 = blaze::rowslice( sm  , 1UL );
//
//          if( blaze::isSame( rowslice1, rowslice2 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( rowslice2, rowslice1 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching rowslices on tensor and subtensor (different rowslice)
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto rowslice1 = blaze::rowslice( mat_, 1UL );
//          auto rowslice2 = blaze::rowslice( sm  , 1UL );
//
//          if( blaze::isSame( rowslice1, rowslice2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( rowslice2, rowslice1 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching rowslices on tensor and subtensor (different size)
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 3UL );
//          auto rowslice1 = blaze::rowslice( mat_, 2UL );
//          auto rowslice2 = blaze::rowslice( sm  , 1UL );
//
//          if( blaze::isSame( rowslice1, rowslice2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( rowslice2, rowslice1 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with matching rowslices on two subtensors
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 0UL, 3UL, 4UL );
//          auto rowslice1 = blaze::rowslice( sm1, 1UL );
//          auto rowslice2 = blaze::rowslice( sm2, 0UL );
//
//          if( blaze::isSame( rowslice1, rowslice2 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( rowslice2, rowslice1 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching rowslices on two subtensors (different rowslice)
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 0UL, 3UL, 4UL );
//          auto rowslice1 = blaze::rowslice( sm1, 1UL );
//          auto rowslice2 = blaze::rowslice( sm2, 1UL );
//
//          if( blaze::isSame( rowslice1, rowslice2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( rowslice2, rowslice1 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching rowslices on two subtensors (different size)
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 0UL, 3UL, 3UL );
//          auto rowslice1 = blaze::rowslice( sm1, 1UL );
//          auto rowslice2 = blaze::rowslice( sm2, 0UL );
//
//          if( blaze::isSame( rowslice1, rowslice2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( rowslice2, rowslice1 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching rowslices on two subtensors (different offset)
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 3UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 1UL, 3UL, 3UL );
//          auto rowslice1 = blaze::rowslice( sm1, 1UL );
//          auto rowslice2 = blaze::rowslice( sm2, 0UL );
//
//          if( blaze::isSame( rowslice1, rowslice2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( rowslice2, rowslice1 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First rowslice:\n" << rowslice1 << "\n"
//                 << "   Second rowslice:\n" << rowslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with matching rowslice submatrices on a subtensor
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 1UL, 2UL, 3UL );
//          auto rowslice1 = blaze::rowslice( sm, 1UL );
//          auto sv1  = blaze::submatrix( rowslice1, 0UL, 2UL );
//          auto sv2  = blaze::submatrix( rowslice1, 0UL, 2UL );
//
//          if( blaze::isSame( sv1, sv2 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First submatrix:\n" << sv1 << "\n"
//                 << "   Second submatrix:\n" << sv2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching rowslice subtensors on a submatrix (different size)
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 1UL, 2UL, 3UL );
//          auto rowslice1 = blaze::rowslice( sm, 1UL );
//          auto sv1  = blaze::submatrix( rowslice1, 0UL, 2UL );
//          auto sv2  = blaze::submatrix( rowslice1, 0UL, 3UL );
//
//          if( blaze::isSame( sv1, sv2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First submatrix:\n" << sv1 << "\n"
//                 << "   Second submatrix:\n" << sv2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with non-matching rowslice subtensors on a submatrix (different offset)
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 1UL, 2UL, 3UL );
//          auto rowslice1 = blaze::rowslice( sm, 1UL );
//          auto sv1  = blaze::submatrix( rowslice1, 0UL, 2UL );
//          auto sv2  = blaze::submatrix( rowslice1, 1UL, 2UL );
//
//          if( blaze::isSame( sv1, sv2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First submatrix:\n" << sv1 << "\n"
//                 << "   Second submatrix:\n" << sv2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with matching rowslice subtensors on two subtensors
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 0UL, 3UL, 4UL );
//          auto rowslice1 = blaze::rowslice( sm1, 1UL );
//          auto rowslice2 = blaze::rowslice( sm2, 0UL );
//          auto sv1  = blaze::submatrix( rowslice1, 0UL, 2UL );
//          auto sv2  = blaze::submatrix( rowslice2, 0UL, 2UL );
//
//          if( blaze::isSame( sv1, sv2 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First submatrix:\n" << sv1 << "\n"
//                 << "   Second submatrix:\n" << sv2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching rowslice subtensors on two subtensors (different size)
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 0UL, 3UL, 4UL );
//          auto rowslice1 = blaze::rowslice( sm1, 1UL );
//          auto rowslice2 = blaze::rowslice( sm2, 0UL );
//          auto sv1  = blaze::submatrix( rowslice1, 0UL, 2UL );
//          auto sv2  = blaze::submatrix( rowslice2, 0UL, 3UL );
//
//          if( blaze::isSame( sv1, sv2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First submatrix:\n" << sv1 << "\n"
//                 << "   Second submatrix:\n" << sv2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching rowslice subtensors on two subtensors (different offset)
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 0UL, 3UL, 4UL );
//          auto rowslice1 = blaze::rowslice( sm1, 1UL );
//          auto rowslice2 = blaze::rowslice( sm2, 0UL );
//          auto sv1  = blaze::submatrix( rowslice1, 0UL, 2UL );
//          auto sv2  = blaze::submatrix( rowslice2, 1UL, 2UL );
//
//          if( blaze::isSame( sv1, sv2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First submatrix:\n" << sv1 << "\n"
//                 << "   Second submatrix:\n" << sv2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c submatrix() function with the RowSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c submatrix() function used with the RowSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testSubmatrix()
{
   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "submatrix() function";

      initialize();

      {
         RT   rowslice1 = blaze::rowslice( mat_, 1UL );
         auto sm = blaze::submatrix( rowslice1, 1UL, 1UL, 2UL, 3UL );

         if( sm(0,0) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result: " << sm(0,0) << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }

         if( *sm.begin(1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *sm.begin(1) << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT   rowslice1 = blaze::rowslice( mat_, 1UL );
         auto sm = blaze::submatrix( rowslice1, 4UL, 0UL, 4UL, 4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         RT   rowslice1 = blaze::rowslice( mat_, 1UL );
         auto sm = blaze::submatrix( rowslice1, 0UL, 0UL, 2UL, 6UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Test of the \c row() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c row() function with the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testRow()
{
   using blaze::rowslice;
   using blaze::row;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "Pageslice row() function";

      initialize();

      {
         RT rowslice1  = rowslice( mat_, 0UL );
         RT rowslice2  = rowslice( mat_, 1UL );
         auto row1 = row( rowslice1, 1UL );
         auto row2 = row( rowslice2, 1UL );

         if( row1 != row2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Row function failed\n"
                << " Details:\n"
                << "   Result:\n" << row1 << "\n"
                << "   Expected result:\n" << row2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( row1[1] != row2[1] ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result: " << row1[1] << "\n"
                << "   Expected result: " << row2[1] << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *row1.begin() != *row2.begin() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *row1.begin() << "\n"
                << "   Expected result: " << *row2.begin() << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT rowslice1  = rowslice( mat_, 0UL );
         auto row8 = row( rowslice1, 8UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row succeeded\n"
             << " Details:\n"
             << "   Result:\n" << row8 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c rows() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c rows() function with the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testRows()
{
   using blaze::rowslice;
   using blaze::rows;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "Pageslice rows() function";

      initialize();

      {
         RT rowslice1 = rowslice( mat_, 0UL );
         RT rowslice2 = rowslice( mat_, 1UL );
         auto rs1 = rows( rowslice1, { 0UL, 2UL, 4UL, 3UL } );
         auto rs2 = rows( rowslice2, { 0UL, 2UL, 4UL, 3UL } );

         if( rs1 != rs2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Rows function failed\n"
                << " Details:\n"
                << "   Result:\n" << rs1 << "\n"
                << "   Expected result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs1(1,1) != rs2(1,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result: " << rs1(1,1) << "\n"
                << "   Expected result: " << rs2(1,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *rs1.begin( 1UL ) != *rs2.begin( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *rs1.begin( 1UL ) << "\n"
                << "   Expected result: " << *rs2.begin( 1UL ) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT rowslice1 = rowslice( mat_, 1UL );
         auto rs  = rows( rowslice1, { 8UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c column() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c column() function with the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testColumn()
{
   using blaze::rowslice;
   using blaze::column;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "Pageslice column() function";

      initialize();

      {
         RT rowslice1  = rowslice( mat_, 0UL );
         RT rowslice2  = rowslice( mat_, 1UL );
         auto col1 = column( rowslice1, 1UL );
         auto col2 = column( rowslice2, 1UL );

         if( col1 != col2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Column function failed\n"
                << " Details:\n"
                << "   Result:\n" << col1 << "\n"
                << "   Expected result:\n" << col2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( col1[1] != col2[1] ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result: " << col1[1] << "\n"
                << "   Expected result: " << col2[1] << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *col1.begin() != *col2.begin() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *col1.begin() << "\n"
                << "   Expected result: " << *col2.begin() << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT rowslice1  = rowslice( mat_, 0UL );
         auto col16 = column( rowslice1, 16UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column succeeded\n"
             << " Details:\n"
             << "   Result:\n" << col16 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c columns() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c columns() function with the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testColumns()
{
   using blaze::rowslice;
   using blaze::rows;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "columns() function";

      initialize();

      {
         RT rowslice1  = rowslice( mat_, 0UL );
         RT rowslice2  = rowslice( mat_, 1UL );
         auto cs1 = columns( rowslice1, { 0UL, 2UL, 2UL, 3UL } );
         auto cs2 = columns( rowslice2, { 0UL, 2UL, 2UL, 3UL } );

         if( cs1 != cs2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Rows function failed\n"
                << " Details:\n"
                << "   Result:\n" << cs1 << "\n"
                << "   Expected result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs1(1,1) != cs2(1,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result: " << cs1(1,1) << "\n"
                << "   Expected result: " << cs2(1,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *cs1.begin( 1UL ) != *cs2.begin( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *cs1.begin( 1UL ) << "\n"
                << "   Expected result: " << *cs2.begin( 1UL ) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT rowslice1 = rowslice( mat_, 1UL );
         auto cs  = columns( rowslice1, { 16UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c band() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c band() function with the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testBand()
{
   using blaze::rowslice;
   using blaze::band;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "Pageslice band() function";

      initialize();

      {
         RT rowslice1  = rowslice( mat_, 0UL );
         RT rowslice2  = rowslice( mat_, 1UL );
         auto b1 = band( rowslice1, 1L );
         auto b2 = band( rowslice2, 1L );

         if( b1 != b2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Band function failed\n"
                << " Details:\n"
                << "   Result:\n" << b1 << "\n"
                << "   Expected result:\n" << b2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( b1[1] != b2[1] ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result: " << b1[1] << "\n"
                << "   Expected result: " << b2[1] << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *b1.begin() != *b2.begin() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *b1.begin() << "\n"
                << "   Expected result: " << *b2.begin() << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT rowslice1 = rowslice( mat_, 1UL );
         auto b8 = band( rowslice1, -8L );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds band succeeded\n"
             << " Details:\n"
             << "   Result:\n" << b8 << "\n";
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
/*!\brief Initialization of all member matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function initializes all member matrices to specific predetermined values.
*/
void DenseGeneralTest::initialize()
{
   // Initializing the rowslice-major dynamic matrix
   mat_.reset();
   mat_(1,1,0) =  1;
   mat_(2,0,0) = -2;
   mat_(2,2,0) = -3;
   mat_(3,1,0) =  4;
   mat_(3,2,0) =  5;
   mat_(3,3,0) = -6;
   mat_(4,0,0) =  7;
   mat_(4,1,0) = -8;
   mat_(4,2,0) =  9;
   mat_(4,3,0) = 10;
   mat_(1,1,1) =  1;
   mat_(2,0,1) = -2;
   mat_(2,2,1) = -3;
   mat_(3,1,1) =  4;
   mat_(3,2,1) =  5;
   mat_(3,3,1) = -6;
   mat_(4,0,1) =  7;
   mat_(4,1,1) = -8;
   mat_(4,2,1) =  9;
   mat_(4,3,1) = 10;
}
//*************************************************************************************************

} // namespace rowslice

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
   std::cout << "   Running RowSlice dense general test..." << std::endl;

   try
   {
      RUN_ROWSLICE_DENSEGENERAL_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during RowSlice dense general test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
