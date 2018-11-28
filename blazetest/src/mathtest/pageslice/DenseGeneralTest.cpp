//=================================================================================================
/*!
//  \file src/mathtest/pageslice/DenseGeneralTest.cpp
//  \brief Source file for the PageSlice dense general test
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

#include <blazetest/mathtest/pageslice/DenseGeneralTest.h>


namespace blazetest {

namespace mathtest {

namespace pageslice {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the PageSlice dense general test.
//
// \exception std::runtime_error Operation error detected.
*/
DenseGeneralTest::DenseGeneralTest()
   : mat_ ( 2UL, 5UL, 4UL )
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
/*!\brief Test of the PageSlice constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the PageSlice specialization. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testConstructors()
{
   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "PageSlice constructor (0x0)";

      MT mat;

      // 0th matrix pageslice
      try {
         blaze::pageslice( mat, 0UL );
      }
      catch( std::invalid_argument& ) {}
   }

   {
      test_ = "PageSlice constructor (2x0)";

      MT mat( 2UL, 2UL, 0UL );

      // 0th matrix pageslice
      {
         RT pageslice0 = blaze::pageslice( mat, 0UL );

         checkRows    ( pageslice0, 2UL );
         checkColumns ( pageslice0, 0UL );
         checkCapacity( pageslice0, 0UL );
         checkNonZeros( pageslice0, 0UL );
      }

      // 1st matrix pageslice
      {
         RT pageslice1 = blaze::pageslice( mat, 1UL );

         checkRows    ( pageslice1, 2UL );
         checkColumns ( pageslice1, 0UL );
         checkCapacity( pageslice1, 0UL );
         checkNonZeros( pageslice1, 0UL );
      }

      // 2nd matrix pageslice
      try {
         blaze::pageslice( mat, 2UL );
      }
      catch( std::invalid_argument& ) {}
   }

   {
      test_ = "PageSlice constructor (5x4)";

      initialize();

      // 0th tensor pageslice
      {
         RT pageslice0 = blaze::pageslice( mat_, 0UL );

         checkRows    ( pageslice0, 5UL );
         checkColumns ( pageslice0, 4UL );
         checkCapacity( pageslice0, 20UL );
         checkNonZeros( pageslice0, 10UL );

         if( pageslice0(0,0) !=  0 || pageslice0(0,1) !=  0 || pageslice0(0,2) !=  0 || pageslice0(0,3) !=  0 ||
             pageslice0(1,0) !=  0 || pageslice0(1,1) !=  1 || pageslice0(1,2) !=  0 || pageslice0(1,3) !=  0 ||
             pageslice0(2,0) != -2 || pageslice0(2,1) !=  0 || pageslice0(2,2) != -3 || pageslice0(2,3) !=  0 ||
             pageslice0(3,0) !=  0 || pageslice0(3,1) !=  4 || pageslice0(3,2) !=  5 || pageslice0(3,3) != -6 ||
             pageslice0(4,0) !=  7 || pageslice0(4,1) != -8 || pageslice0(4,2) !=  9 || pageslice0(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of 0th dense pageslice failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice0 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // 1st tensor pageslice
      {
         RT pageslice1 = blaze::pageslice( mat_, 1UL );

         checkRows    ( pageslice1, 5UL );
         checkColumns ( pageslice1, 4UL );
         checkCapacity( pageslice1, 20UL );
         checkNonZeros( pageslice1, 10UL );

         if( pageslice1(0,0) !=  0 || pageslice1(0,1) !=  0 || pageslice1(0,2) !=  0 || pageslice1(0,3) !=  0 ||
             pageslice1(1,0) !=  0 || pageslice1(1,1) !=  1 || pageslice1(1,2) !=  0 || pageslice1(1,3) !=  0 ||
             pageslice1(2,0) != -2 || pageslice1(2,1) !=  0 || pageslice1(2,2) != -3 || pageslice1(2,3) !=  0 ||
             pageslice1(3,0) !=  0 || pageslice1(3,1) !=  4 || pageslice1(3,2) !=  5 || pageslice1(3,3) != -6 ||
             pageslice1(4,0) !=  7 || pageslice1(4,1) != -8 || pageslice1(4,2) !=  9 || pageslice1(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of 1st dense pageslice failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice1 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // 2nd tensor pageslice
      try {
         RT pageslice2 = blaze::pageslice( mat_, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Out-of-bound page access succeeded\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Test of the PageSlice assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the PageSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testAssignment()
{
   //=====================================================================================
   // homogeneous assignment
   //=====================================================================================

   {
      test_ = "PageSlice homogeneous assignment";

      initialize();

      RT pageslice1 = blaze::pageslice( mat_, 1UL );
      pageslice1 = 8;


      checkRows    ( pageslice1, 5UL );
      checkColumns ( pageslice1, 4UL );
      checkCapacity( pageslice1, 20UL );
      checkNonZeros( pageslice1, 20UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 30UL );

      if( pageslice1(0,0) != 8 || pageslice1(0,1) != 8 || pageslice1(0,2) != 8 || pageslice1(0,3) != 8 ||
          pageslice1(1,0) != 8 || pageslice1(1,1) != 8 || pageslice1(1,2) != 8 || pageslice1(1,3) != 8 ||
          pageslice1(2,0) != 8 || pageslice1(2,1) != 8 || pageslice1(2,2) != 8 || pageslice1(2,3) != 8 ||
          pageslice1(3,0) != 8 || pageslice1(3,1) != 8 || pageslice1(3,2) != 8 || pageslice1(3,3) != 8 ||
          pageslice1(4,0) != 8 || pageslice1(4,1) != 8 || pageslice1(4,2) != 8 || pageslice1(4,3) != 8 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice1 << "\n"
             << "   Expected result:\n(( 8 8 8 8 )\n( 8 8 8 8 )\n( 8 8 8 8 )\n( 8 8 8 8 )\n( 8 8 8 8 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  8 || mat_(1,0,1) !=  8 || mat_(1,0,2) !=  8 || mat_(1,0,3) !=  8 ||
          mat_(1,1,0) !=  8 || mat_(1,1,1) !=  8 || mat_(1,1,2) !=  8 || mat_(1,1,3) !=  8 ||
          mat_(1,2,0) !=  8 || mat_(1,2,1) !=  8 || mat_(1,2,2) !=  8 || mat_(1,2,3) !=  8 ||
          mat_(1,3,0) !=  8 || mat_(1,3,1) !=  8 || mat_(1,3,2) !=  8 || mat_(1,3,3) !=  8 ||
          mat_(1,4,0) !=  8 || mat_(1,4,1) !=  8 || mat_(1,4,2) !=  8 || mat_(1,4,3) !=  8 ) {
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

      RT pageslice3 = blaze::pageslice( mat_, 1UL );
      pageslice3 = {
          {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}
      };

      checkRows    ( pageslice3, 5UL );
      checkColumns ( pageslice3, 4UL );
      checkCapacity( pageslice3, 20UL );
      checkNonZeros( pageslice3, 20UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 30UL );

      if( pageslice3(0,0) != 1 || pageslice3(0,1) != 2 || pageslice3(0,2) != 3 || pageslice3(0,3) != 4 ||
          pageslice3(1,0) != 1 || pageslice3(1,1) != 2 || pageslice3(1,2) != 3 || pageslice3(1,3) != 4 ||
          pageslice3(2,0) != 1 || pageslice3(2,1) != 2 || pageslice3(2,2) != 3 || pageslice3(2,3) != 4 ||
          pageslice3(3,0) != 1 || pageslice3(3,1) != 2 || pageslice3(3,2) != 3 || pageslice3(3,3) != 4 ||
          pageslice3(4,0) != 1 || pageslice3(4,1) != 2 || pageslice3(4,2) != 3 || pageslice3(4,3) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice3 << "\n"
             << "   Expected result:\n(( 1 2 3 4 )\n( 1 2 3 4 )\n( 1 2 3 4 )\n( 1 2 3 4 )\n( 1 2 3 4 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  1 || mat_(1,0,1) !=  2 || mat_(1,0,2) !=  3 || mat_(1,0,3) !=  4 ||
          mat_(1,1,0) !=  1 || mat_(1,1,1) !=  2 || mat_(1,1,2) !=  3 || mat_(1,1,3) !=  4 ||
          mat_(1,2,0) !=  1 || mat_(1,2,1) !=  2 || mat_(1,2,2) !=  3 || mat_(1,2,3) !=  4 ||
          mat_(1,3,0) !=  1 || mat_(1,3,1) !=  2 || mat_(1,3,2) !=  3 || mat_(1,3,3) !=  4 ||
          mat_(1,4,0) !=  1 || mat_(1,4,1) !=  2 || mat_(1,4,2) !=  3 || mat_(1,4,3) !=  4 ) {
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

      RT pageslice3 = blaze::pageslice( mat_, 1UL );
      pageslice3 = {{1, 2}, {1, 2}, {1, 2}, {1, 2}, {1, 2}};

      checkRows    ( pageslice3, 5UL );
      checkColumns ( pageslice3, 4UL );
      checkCapacity( pageslice3, 20UL );
      checkNonZeros( pageslice3, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice3(0,0) != 1 || pageslice3(0,1) != 2 || pageslice3(0,2) != 0 || pageslice3(0,3) != 0 ||
          pageslice3(1,0) != 1 || pageslice3(1,1) != 2 || pageslice3(1,2) != 0 || pageslice3(1,3) != 0 ||
          pageslice3(2,0) != 1 || pageslice3(2,1) != 2 || pageslice3(2,2) != 0 || pageslice3(2,3) != 0 ||
          pageslice3(3,0) != 1 || pageslice3(3,1) != 2 || pageslice3(3,2) != 0 || pageslice3(3,3) != 0 ||
          pageslice3(4,0) != 1 || pageslice3(4,1) != 2 || pageslice3(4,2) != 0 || pageslice3(4,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice3 << "\n"
             << "   Expected result:\n(( 1 2 0 0 )\n( 1 2 0 0 )\n( 1 2 0 0 )\n( 1 2 0 0 )\n( 1 2 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  1 || mat_(1,0,1) !=  2 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=  1 || mat_(1,1,1) !=  2 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) !=  1 || mat_(1,2,1) !=  2 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  1 || mat_(1,3,1) !=  2 || mat_(1,3,2) !=  0 || mat_(1,3,3) !=  0 ||
          mat_(1,4,0) !=  1 || mat_(1,4,1) !=  2 || mat_(1,4,2) !=  0 || mat_(1,4,3) !=  0 ) {
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
      test_ = "PageSlice copy assignment";

      initialize();

      RT pageslice1 = blaze::pageslice( mat_, 0UL );
      pageslice1 = 0;
      pageslice1 = blaze::pageslice( mat_, 1UL );

      checkRows    ( pageslice1, 5UL );
      checkColumns ( pageslice1, 4UL );
      checkCapacity( pageslice1, 20UL );
      checkNonZeros( pageslice1, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice1(0,0) !=  0 || pageslice1(0,1) !=  0 || pageslice1(0,2) !=  0 || pageslice1(0,3) !=  0 ||
          pageslice1(1,0) !=  0 || pageslice1(1,1) !=  1 || pageslice1(1,2) !=  0 || pageslice1(1,3) !=  0 ||
          pageslice1(2,0) != -2 || pageslice1(2,1) !=  0 || pageslice1(2,2) != -3 || pageslice1(2,3) !=  0 ||
          pageslice1(3,0) !=  0 || pageslice1(3,1) !=  4 || pageslice1(3,2) !=  5 || pageslice1(3,3) != -6 ||
          pageslice1(4,0) !=  7 || pageslice1(4,1) != -8 || pageslice1(4,2) !=  9 || pageslice1(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice1 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=  0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) != -2 || mat_(1,2,1) !=  0 || mat_(1,2,2) != -3 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) !=  7 || mat_(1,4,1) != -8 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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

      RT pageslice1 = blaze::pageslice( mat_, 1UL );

      blaze::DynamicMatrix<int, blaze::rowMajor> m1;
      m1 = {{0, 8, 0, 9}, {0}, {0}, {0}, {0}};

      pageslice1 = m1;

      checkRows    ( pageslice1, 5UL );
      checkColumns ( pageslice1, 4UL );
      checkCapacity( pageslice1, 20UL );
      checkNonZeros( pageslice1, 2UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 12UL );

      if( pageslice1(0,0) !=  0 || pageslice1(0,1) !=  8 || pageslice1(0,2) !=  0 || pageslice1(0,3) !=  9 ||
          pageslice1(1,0) !=  0 || pageslice1(1,1) !=  0 || pageslice1(1,2) !=  0 || pageslice1(1,3) !=  0 ||
          pageslice1(2,0) !=  0 || pageslice1(2,1) !=  0 || pageslice1(2,2) !=  0 || pageslice1(2,3) !=  0 ||
          pageslice1(3,0) !=  0 || pageslice1(3,1) !=  0 || pageslice1(3,2) !=  0 || pageslice1(3,3) !=  0 ||
          pageslice1(4,0) !=  0 || pageslice1(4,1) !=  0 || pageslice1(4,2) !=  0 || pageslice1(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice1 << "\n"
             << "   Expected result:\n(( 0 8 0 9 )\n(0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=  8 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  9 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=  0 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) !=  0 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  0 || mat_(1,3,2) !=  0 || mat_(1,3,3) !=  0 ||
          mat_(1,4,0) !=  0 || mat_(1,4,1) !=  0 || mat_(1,4,2) !=  0 || mat_(1,4,3) !=  0 ) {
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

      RT pageslice1 = blaze::pageslice( mat_, 1UL );

      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 80UL ) );
      AlignedPadded m1( memory.get(), 5UL, 4UL, 16UL );
      m1 = 0;
      m1(0,0) = 0;
      m1(0,1) = 8;
      m1(0,2) = 0;
      m1(0,3) = 9;

      pageslice1 = m1;

      checkRows    ( pageslice1, 5UL );
      checkColumns ( pageslice1, 4UL );
      checkCapacity( pageslice1, 20UL );
      checkNonZeros( pageslice1, 2UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 12UL );

      if( pageslice1(0,0) !=  0 || pageslice1(0,1) !=  8 || pageslice1(0,2) !=  0 || pageslice1(0,3) !=  9 ||
          pageslice1(1,0) !=  0 || pageslice1(1,1) !=  0 || pageslice1(1,2) !=  0 || pageslice1(1,3) !=  0 ||
          pageslice1(2,0) !=  0 || pageslice1(2,1) !=  0 || pageslice1(2,2) !=  0 || pageslice1(2,3) !=  0 ||
          pageslice1(3,0) !=  0 || pageslice1(3,1) !=  0 || pageslice1(3,2) !=  0 || pageslice1(3,3) !=  0 ||
          pageslice1(4,0) !=  0 || pageslice1(4,1) !=  0 || pageslice1(4,2) !=  0 || pageslice1(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice1 << "\n"
             << "   Expected result:\n(( 0 8 0 9 )\n(0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=  8 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  9 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=  0 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) !=  0 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  0 || mat_(1,3,2) !=  0 || mat_(1,3,3) !=  0 ||
          mat_(1,4,0) !=  0 || mat_(1,4,1) !=  0 || mat_(1,4,2) !=  0 || mat_(1,4,3) !=  0 ) {
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

      RT pageslice1 = blaze::pageslice( mat_, 1UL );

      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
      std::unique_ptr<int[]> memory( new int[21] );
      UnalignedUnpadded m1( memory.get()+1UL, 5UL, 4UL );
      m1 = 0;
      m1(0,0) = 0;
      m1(0,1) = 8;
      m1(0,2) = 0;
      m1(0,3) = 9;

      pageslice1 = m1;

      checkRows    ( pageslice1, 5UL );
      checkColumns ( pageslice1, 4UL );
      checkCapacity( pageslice1, 20UL );
      checkNonZeros( pageslice1, 2UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 12UL );

      if( pageslice1(0,0) !=  0 || pageslice1(0,1) !=  8 || pageslice1(0,2) !=  0 || pageslice1(0,3) !=  9 ||
          pageslice1(1,0) !=  0 || pageslice1(1,1) !=  0 || pageslice1(1,2) !=  0 || pageslice1(1,3) !=  0 ||
          pageslice1(2,0) !=  0 || pageslice1(2,1) !=  0 || pageslice1(2,2) !=  0 || pageslice1(2,3) !=  0 ||
          pageslice1(3,0) !=  0 || pageslice1(3,1) !=  0 || pageslice1(3,2) !=  0 || pageslice1(3,3) !=  0 ||
          pageslice1(4,0) !=  0 || pageslice1(4,1) !=  0 || pageslice1(4,2) !=  0 || pageslice1(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice1 << "\n"
             << "   Expected result:\n(( 0 8 0 9 )\n(0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=  8 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  9 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=  0 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) !=  0 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  0 || mat_(1,3,2) !=  0 || mat_(1,3,3) !=  0 ||
          mat_(1,4,0) !=  0 || mat_(1,4,1) !=  0 || mat_(1,4,2) !=  0 || mat_(1,4,3) !=  0 ) {
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
/*!\brief Test of the PageSlice addition assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the addition assignment operators of the PageSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testAddAssign()
{
   //=====================================================================================
   // PageSlice addition assignment
   //=====================================================================================

   {
      test_ = "PageSlice addition assignment";

      initialize();

      RT pageslice2 = blaze::pageslice( mat_, 1UL );
      pageslice2 += blaze::pageslice( mat_, 0UL );

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   2 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -4 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -6 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=   8 || pageslice2(3,2) != 10 || pageslice2(3,3) != -12 ||
          pageslice2(4,0) != 14 || pageslice2(4,1) != -16 || pageslice2(4,2) != 18 || pageslice2(4,3) !=  20 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   2 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) != -4 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -6 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=   8 || mat_(1,3,2) != 10 || mat_(1,3,3) != -12 ||
          mat_(1,4,0) != 14 || mat_(1,4,1) != -16 || mat_(1,4,2) != 18 || mat_(1,4,3) !=  20 ) {
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

      RT pageslice2 = blaze::pageslice( mat_, 1UL );

      const blaze::DynamicMatrix<short, blaze::rowMajor> vec{{0, 0, 0, 0},
                                                             {0, 1, 0, 0},
                                                             {-2, 0, -3, 0},
                                                             {0, 4, 5, -6},
                                                             {7, -8, 9, 10}};

      pageslice2 += vec;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   2 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -4 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -6 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=   8 || pageslice2(3,2) != 10 || pageslice2(3,3) != -12 ||
          pageslice2(4,0) != 14 || pageslice2(4,1) != -16 || pageslice2(4,2) != 18 || pageslice2(4,3) !=  20 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   2 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) != -4 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -6 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=   8 || mat_(1,3,2) != 10 || mat_(1,3,3) != -12 ||
          mat_(1,4,0) != 14 || mat_(1,4,1) != -16 || mat_(1,4,2) != 18 || mat_(1,4,3) !=  20 ) {
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

      RT pageslice2 = blaze::pageslice( mat_, 1UL );

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

      pageslice2 += m;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   2 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -4 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -6 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=   8 || pageslice2(3,2) != 10 || pageslice2(3,3) != -12 ||
          pageslice2(4,0) != 14 || pageslice2(4,1) != -16 || pageslice2(4,2) != 18 || pageslice2(4,3) !=  20 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   2 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) != -4 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -6 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=   8 || mat_(1,3,2) != 10 || mat_(1,3,3) != -12 ||
          mat_(1,4,0) != 14 || mat_(1,4,1) != -16 || mat_(1,4,2) != 18 || mat_(1,4,3) !=  20 ) {
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

      RT pageslice2 = blaze::pageslice( mat_, 1UL );

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

      pageslice2 += m;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   2 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -4 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -6 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=   8 || pageslice2(3,2) != 10 || pageslice2(3,3) != -12 ||
          pageslice2(4,0) != 14 || pageslice2(4,1) != -16 || pageslice2(4,2) != 18 || pageslice2(4,3) !=  20 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 2 0 0 )\n( -4 0 -6 0 )\n( 0 8 10 -12 )\n( 14 -16 18 20 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   2 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) != -4 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -6 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=   8 || mat_(1,3,2) != 10 || mat_(1,3,3) != -12 ||
          mat_(1,4,0) != 14 || mat_(1,4,1) != -16 || mat_(1,4,2) != 18 || mat_(1,4,3) !=  20 ) {
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
/*!\brief Test of the PageSlice subtraction assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the subtraction assignment operators of the PageSlice
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testSubAssign()
{
   //=====================================================================================
   // PageSlice subtraction assignment
   //=====================================================================================

   {
      test_ = "PageSlice subtraction assignment";

      initialize();

      RT pageslice2 = blaze::pageslice( mat_, 1UL );
      pageslice2 -= blaze::pageslice( mat_, 0UL );

      checkRows    ( pageslice2,  5UL );
      checkColumns ( pageslice2,  4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2,  0UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 10UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=  0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=  0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=  0 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=  0 ||
          pageslice2(2,0) !=  0 || pageslice2(2,1) !=  0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=  0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=  0 || pageslice2(3,2) !=  0 || pageslice2(3,3) !=  0 ||
          pageslice2(4,0) !=  0 || pageslice2(4,1) !=  0 || pageslice2(4,2) !=  0 || pageslice2(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   0 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) !=  0 || mat_(1,2,1) !=   0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=   0 || mat_(1,3,2) !=  0 || mat_(1,3,3) !=   0 ||
          mat_(1,4,0) !=  0 || mat_(1,4,1) !=   0 || mat_(1,4,2) !=  0 || mat_(1,4,3) !=   0 ) {
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

      RT pageslice2 = blaze::pageslice( mat_, 1UL );

      const blaze::DynamicMatrix<short, blaze::rowMajor> vec{{0, 0, 0, 0},
                                                             {0, 1, 0, 0},
                                                             {-2, 0, -3, 0},
                                                             {0, 4, 5, -6},
                                                             {7, -8, 9, 10}};

      pageslice2 -= vec;

      checkRows    ( pageslice2,  5UL );
      checkColumns ( pageslice2,  4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2,  0UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 10UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=  0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=  0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=  0 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=  0 ||
          pageslice2(2,0) !=  0 || pageslice2(2,1) !=  0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=  0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=  0 || pageslice2(3,2) !=  0 || pageslice2(3,3) !=  0 ||
          pageslice2(4,0) !=  0 || pageslice2(4,1) !=  0 || pageslice2(4,2) !=  0 || pageslice2(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   0 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) !=  0 || mat_(1,2,1) !=   0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=   0 || mat_(1,3,2) !=  0 || mat_(1,3,3) !=   0 ||
          mat_(1,4,0) !=  0 || mat_(1,4,1) !=   0 || mat_(1,4,2) !=  0 || mat_(1,4,3) !=   0 ) {
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

      RT pageslice2 = blaze::pageslice( mat_, 1UL );

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

      pageslice2 -= m;

      checkRows    ( pageslice2,  5UL );
      checkColumns ( pageslice2,  4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2,  0UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 10UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=  0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=  0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=  0 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=  0 ||
          pageslice2(2,0) !=  0 || pageslice2(2,1) !=  0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=  0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=  0 || pageslice2(3,2) !=  0 || pageslice2(3,3) !=  0 ||
          pageslice2(4,0) !=  0 || pageslice2(4,1) !=  0 || pageslice2(4,2) !=  0 || pageslice2(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   0 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) !=  0 || mat_(1,2,1) !=   0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=   0 || mat_(1,3,2) !=  0 || mat_(1,3,3) !=   0 ||
          mat_(1,4,0) !=  0 || mat_(1,4,1) !=   0 || mat_(1,4,2) !=  0 || mat_(1,4,3) !=   0 ) {
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

      RT pageslice2 = blaze::pageslice( mat_, 1UL );

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

      pageslice2 -= m;

      checkRows    ( pageslice2,  5UL );
      checkColumns ( pageslice2,  4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2,  0UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 10UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=  0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=  0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=  0 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=  0 ||
          pageslice2(2,0) !=  0 || pageslice2(2,1) !=  0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=  0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=  0 || pageslice2(3,2) !=  0 || pageslice2(3,3) !=  0 ||
          pageslice2(4,0) !=  0 || pageslice2(4,1) !=  0 || pageslice2(4,2) !=  0 || pageslice2(4,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   0 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) !=  0 || mat_(1,2,1) !=   0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=   0 || mat_(1,3,2) !=  0 || mat_(1,3,3) !=   0 ||
          mat_(1,4,0) !=  0 || mat_(1,4,1) !=   0 || mat_(1,4,2) !=  0 || mat_(1,4,3) !=   0 ) {
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
/*!\brief Test of the PageSlice multiplication assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the multiplication assignment operators of the PageSlice
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testMultAssign()
{
   //=====================================================================================
   // PageSlice multiplication assignment
   //=====================================================================================

   {
      test_ = "PageSlice multiplication assignment";

      initialize();

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT pageslice2 = blaze::pageslice( m, 1UL );
      pageslice2 *= blaze::pageslice( m, 0UL );

      checkRows    ( pageslice2, 3UL );
      checkColumns ( pageslice2, 3UL );
      checkCapacity( pageslice2, 9UL );
      checkNonZeros( pageslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( pageslice2(0,0) != 90 || pageslice2(0,1) != 114 || pageslice2(0,2) != 138 ||
          pageslice2(1,0) != 54 || pageslice2(1,1) !=  69 || pageslice2(1,2) !=  84 ||
          pageslice2(2,0) != 18 || pageslice2(2,1) !=  24 || pageslice2(2,2) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 90 114 138 )\n( 54 69 84 )\n( 18 24 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,0,1) !=   2 || m(0,0,2) !=   3 ||
          m(0,1,0) !=  4 || m(0,1,1) !=   5 || m(0,1,2) !=   6 ||
          m(0,2,0) !=  7 || m(0,2,1) !=   8 || m(0,2,2) !=   9 ||
          m(1,0,0) != 90 || m(1,0,1) != 114 || m(1,0,2) != 138 ||
          m(1,1,0) != 54 || m(1,1,1) !=  69 || m(1,1,2) !=  84 ||
          m(1,2,0) != 18 || m(1,2,1) !=  24 || m(1,2,2) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((   1   2   3 )\n"
                                     " (   4   5   6 )\n"
                                     " (   7   8   9 ))\n"
                                     "((  90 114 138 )\n"
                                     " (  54  69  84 )\n"
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

      RT pageslice2 = blaze::pageslice( m, 1UL );

      const blaze::DynamicMatrix<short, blaze::rowMajor> m1{
          {1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

      pageslice2 *= m1;

      checkRows    ( pageslice2, 3UL );
      checkColumns ( pageslice2, 3UL );
      checkCapacity( pageslice2, 9UL );
      checkNonZeros( pageslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( pageslice2(0,0) != 90 || pageslice2(0,1) != 114 || pageslice2(0,2) != 138 ||
          pageslice2(1,0) != 54 || pageslice2(1,1) !=  69 || pageslice2(1,2) !=  84 ||
          pageslice2(2,0) != 18 || pageslice2(2,1) !=  24 || pageslice2(2,2) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 90 114 138 )\n( 54 69 84 )\n( 18 24 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,0,1) !=   2 || m(0,0,2) !=   3 ||
          m(0,1,0) !=  4 || m(0,1,1) !=   5 || m(0,1,2) !=   6 ||
          m(0,2,0) !=  7 || m(0,2,1) !=   8 || m(0,2,2) !=   9 ||
          m(1,0,0) != 90 || m(1,0,1) != 114 || m(1,0,2) != 138 ||
          m(1,1,0) != 54 || m(1,1,1) !=  69 || m(1,1,2) !=  84 ||
          m(1,2,0) != 18 || m(1,2,1) !=  24 || m(1,2,2) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((   1   2   3 )\n"
                                     " (   4   5   6 )\n"
                                     " (   7   8   9 ))\n"
                                     "((  90 114 138 )\n"
                                     " (  54  69  84 )\n"
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

      RT pageslice2 = blaze::pageslice( m, 1UL );


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

      pageslice2 *= m1;

      checkRows    ( pageslice2, 3UL );
      checkColumns ( pageslice2, 3UL );
      checkCapacity( pageslice2, 9UL );
      checkNonZeros( pageslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( pageslice2(0,0) != 90 || pageslice2(0,1) != 114 || pageslice2(0,2) != 138 ||
          pageslice2(1,0) != 54 || pageslice2(1,1) !=  69 || pageslice2(1,2) !=  84 ||
          pageslice2(2,0) != 18 || pageslice2(2,1) !=  24 || pageslice2(2,2) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 90 114 138 )\n( 54 69 84 )\n( 18 24 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,0,1) !=   2 || m(0,0,2) !=   3 ||
          m(0,1,0) !=  4 || m(0,1,1) !=   5 || m(0,1,2) !=   6 ||
          m(0,2,0) !=  7 || m(0,2,1) !=   8 || m(0,2,2) !=   9 ||
          m(1,0,0) != 90 || m(1,0,1) != 114 || m(1,0,2) != 138 ||
          m(1,1,0) != 54 || m(1,1,1) !=  69 || m(1,1,2) !=  84 ||
          m(1,2,0) != 18 || m(1,2,1) !=  24 || m(1,2,2) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((   1   2   3 )\n"
                                     " (   4   5   6 )\n"
                                     " (   7   8   9 ))\n"
                                     "((  90 114 138 )\n"
                                     " (  54  69  84 )\n"
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

      RT pageslice2 = blaze::pageslice( m, 1UL );

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

      pageslice2 *= m1;

      checkRows    ( pageslice2, 3UL );
      checkColumns ( pageslice2, 3UL );
      checkCapacity( pageslice2, 9UL );
      checkNonZeros( pageslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( pageslice2(0,0) != 90 || pageslice2(0,1) != 114 || pageslice2(0,2) != 138 ||
          pageslice2(1,0) != 54 || pageslice2(1,1) !=  69 || pageslice2(1,2) !=  84 ||
          pageslice2(2,0) != 18 || pageslice2(2,1) !=  24 || pageslice2(2,2) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 90 114 138 )\n( 54 69 84 )\n( 18 24 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,0,1) !=   2 || m(0,0,2) !=   3 ||
          m(0,1,0) !=  4 || m(0,1,1) !=   5 || m(0,1,2) !=   6 ||
          m(0,2,0) !=  7 || m(0,2,1) !=   8 || m(0,2,2) !=   9 ||
          m(1,0,0) != 90 || m(1,0,1) != 114 || m(1,0,2) != 138 ||
          m(1,1,0) != 54 || m(1,1,1) !=  69 || m(1,1,2) !=  84 ||
          m(1,2,0) != 18 || m(1,2,1) !=  24 || m(1,2,2) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((   1   2   3 )\n"
                                     " (   4   5   6 )\n"
                                     " (   7   8   9 ))\n"
                                     "((  90 114 138 )\n"
                                     " (  54  69  84 )\n"
                                     " (  18  24  30 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the PageSlice Schur product assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the Schur product assignment operators of the PageSlice
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testSchurAssign()
{
   //=====================================================================================
   // PageSlice Schur product assignment
   //=====================================================================================

   {
      test_ = "PageSlice Schur product assignment";

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT pageslice2 = blaze::pageslice( m, 1UL );
      pageslice2 %= blaze::pageslice( m, 0UL );

      checkRows    ( pageslice2, 3UL );
      checkColumns ( pageslice2, 3UL );
      checkCapacity( pageslice2, 9UL );
      checkNonZeros( pageslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( pageslice2(0,0) !=  9 || pageslice2(0,1) != 16 || pageslice2(0,2) != 21 ||
          pageslice2(1,0) != 24 || pageslice2(1,1) != 25 || pageslice2(1,2) != 24 ||
          pageslice2(2,0) != 21 || pageslice2(2,1) != 16 || pageslice2(2,2) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,0,1) !=  2 || m(0,0,2) !=  3 ||
          m(0,1,0) !=  4 || m(0,1,1) !=  5 || m(0,1,2) !=  6 ||
          m(0,2,0) !=  7 || m(0,2,1) !=  8 || m(0,2,2) !=  9 ||
          m(1,0,0) !=  9 || m(1,0,1) != 16 || m(1,0,2) != 21 ||
          m(1,1,0) != 24 || m(1,1,1) != 25 || m(1,1,2) != 24 ||
          m(1,2,0) != 21 || m(1,2,1) != 16 || m(1,2,2) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
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
   // dense matrix Schur product assignment
   //=====================================================================================

   {
      test_ = "dense vector Schur product assignment (mixed type)";

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT pageslice2 = blaze::pageslice( m, 1UL );

      const blaze::DynamicMatrix<short, blaze::rowMajor> m1{
          {1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

      pageslice2 %= m1;

      checkRows    ( pageslice2, 3UL );
      checkColumns ( pageslice2, 3UL );
      checkCapacity( pageslice2, 9UL );
      checkNonZeros( pageslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( pageslice2(0,0) !=  9 || pageslice2(0,1) != 16 || pageslice2(0,2) != 21 ||
          pageslice2(1,0) != 24 || pageslice2(1,1) != 25 || pageslice2(1,2) != 24 ||
          pageslice2(2,0) != 21 || pageslice2(2,1) != 16 || pageslice2(2,2) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,0,1) !=  2 || m(0,0,2) !=  3 ||
          m(0,1,0) !=  4 || m(0,1,1) !=  5 || m(0,1,2) !=  6 ||
          m(0,2,0) !=  7 || m(0,2,1) !=  8 || m(0,2,2) !=  9 ||
          m(1,0,0) !=  9 || m(1,0,1) != 16 || m(1,0,2) != 21 ||
          m(1,1,0) != 24 || m(1,1,1) != 25 || m(1,1,2) != 24 ||
          m(1,2,0) != 21 || m(1,2,1) != 16 || m(1,2,2) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
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
      test_ = "dense matrix Schur product assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;
      using blaze::rowMajor;

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT pageslice2 = blaze::pageslice( m, 1UL );

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

      pageslice2 %= m1;

      checkRows    ( pageslice2, 3UL );
      checkColumns ( pageslice2, 3UL );
      checkCapacity( pageslice2, 9UL );
      checkNonZeros( pageslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( pageslice2(0,0) !=  9 || pageslice2(0,1) != 16 || pageslice2(0,2) != 21 ||
          pageslice2(1,0) != 24 || pageslice2(1,1) != 25 || pageslice2(1,2) != 24 ||
          pageslice2(2,0) != 21 || pageslice2(2,1) != 16 || pageslice2(2,2) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,0,1) !=  2 || m(0,0,2) !=  3 ||
          m(0,1,0) !=  4 || m(0,1,1) !=  5 || m(0,1,2) !=  6 ||
          m(0,2,0) !=  7 || m(0,2,1) !=  8 || m(0,2,2) !=  9 ||
          m(1,0,0) !=  9 || m(1,0,1) != 16 || m(1,0,2) != 21 ||
          m(1,1,0) != 24 || m(1,1,1) != 25 || m(1,1,2) != 24 ||
          m(1,2,0) != 21 || m(1,2,1) != 16 || m(1,2,2) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
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
      test_ = "dense matrix Schur product assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::rowMajor;

      blaze::DynamicTensor<int> m{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                  {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

      RT pageslice2 = blaze::pageslice( m, 1UL );

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

      pageslice2 %= m1;

      checkRows    ( pageslice2, 3UL );
      checkColumns ( pageslice2, 3UL );
      checkCapacity( pageslice2, 9UL );
      checkNonZeros( pageslice2, 9UL );
      checkRows    ( m,  3UL );
      checkColumns ( m,  3UL );
      checkPages   ( m,  2UL );
      checkNonZeros( m, 18UL );

      if( pageslice2(0,0) !=  9 || pageslice2(0,1) != 16 || pageslice2(0,2) != 21 ||
          pageslice2(1,0) != 24 || pageslice2(1,1) != 25 || pageslice2(1,2) != 24 ||
          pageslice2(2,0) != 21 || pageslice2(2,1) != 16 || pageslice2(2,2) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 9 16 21 )\n( 24 25 24 )\n( 21 16 9 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( m(0,0,0) !=  1 || m(0,0,1) !=  2 || m(0,0,2) !=  3 ||
          m(0,1,0) !=  4 || m(0,1,1) !=  5 || m(0,1,2) !=  6 ||
          m(0,2,0) !=  7 || m(0,2,1) !=  8 || m(0,2,2) !=  9 ||
          m(1,0,0) !=  9 || m(1,0,1) != 16 || m(1,0,2) != 21 ||
          m(1,1,0) != 24 || m(1,1,1) != 25 || m(1,1,2) != 24 ||
          m(1,2,0) != 21 || m(1,2,1) != 16 || m(1,2,2) !=  9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n((  1  2  3 )\n"
                                     " (  4  5  6 )\n"
                                     " (  7  8  9 ))\n"
                                     "((  9 16 21 )\n"
                                     " ( 24 25 24 )\n"
                                     " ( 21 16  9 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of all PageSlice (self-)scaling operations.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all available ways to scale an instance of the PageSlice
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

      RT pageslice2 = blaze::pageslice( mat_, 1UL );
      pageslice2 *= 3;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   3 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -6 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -9 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=  12 || pageslice2(3,2) != 15 || pageslice2(3,3) != -18 ||
          pageslice2(4,0) != 21 || pageslice2(4,1) != -24 || pageslice2(4,2) != 27 || pageslice2(4,3) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   3 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) != -6 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -9 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  12 || mat_(1,3,2) != 15 || mat_(1,3,3) != -18 ||
          mat_(1,4,0) != 21 || mat_(1,4,1) != -24 || mat_(1,4,2) != 27 || mat_(1,4,3) !=  30 ) {
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

      RT pageslice2 = blaze::pageslice( mat_, 1UL );
      pageslice2 = pageslice2 * 3;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   3 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -6 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -9 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=  12 || pageslice2(3,2) != 15 || pageslice2(3,3) != -18 ||
          pageslice2(4,0) != 21 || pageslice2(4,1) != -24 || pageslice2(4,2) != 27 || pageslice2(4,3) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   3 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) != -6 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -9 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  12 || mat_(1,3,2) != 15 || mat_(1,3,3) != -18 ||
          mat_(1,4,0) != 21 || mat_(1,4,1) != -24 || mat_(1,4,2) != 27 || mat_(1,4,3) !=  30 ) {
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

      RT pageslice2 = blaze::pageslice( mat_, 1UL );
      pageslice2 = 3 * pageslice2;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   3 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -6 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -9 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=  12 || pageslice2(3,2) != 15 || pageslice2(3,3) != -18 ||
          pageslice2(4,0) != 21 || pageslice2(4,1) != -24 || pageslice2(4,2) != 27 || pageslice2(4,3) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   3 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) != -6 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -9 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  12 || mat_(1,3,2) != 15 || mat_(1,3,3) != -18 ||
          mat_(1,4,0) != 21 || mat_(1,4,1) != -24 || mat_(1,4,2) != 27 || mat_(1,4,3) !=  30 ) {
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

      RT pageslice2 = blaze::pageslice( mat_, 1UL );
      pageslice2 /= (1.0/3.0);

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   3 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -6 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -9 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=  12 || pageslice2(3,2) != 15 || pageslice2(3,3) != -18 ||
          pageslice2(4,0) != 21 || pageslice2(4,1) != -24 || pageslice2(4,2) != 27 || pageslice2(4,3) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   3 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) != -6 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -9 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  12 || mat_(1,3,2) != 15 || mat_(1,3,3) != -18 ||
          mat_(1,4,0) != 21 || mat_(1,4,1) != -24 || mat_(1,4,2) != 27 || mat_(1,4,3) !=  30 ) {
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

      RT pageslice2 = blaze::pageslice( mat_, 1UL );
      pageslice2 = pageslice2 / (1.0/3.0);

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   3 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -6 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -9 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=  12 || pageslice2(3,2) != 15 || pageslice2(3,3) != -18 ||
          pageslice2(4,0) != 21 || pageslice2(4,1) != -24 || pageslice2(4,2) != 27 || pageslice2(4,3) !=  30 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   3 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
          mat_(1,2,0) != -6 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -9 || mat_(1,2,3) !=   0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  12 || mat_(1,3,2) != 15 || mat_(1,3,3) != -18 ||
          mat_(1,4,0) != 21 || mat_(1,4,1) != -24 || mat_(1,4,2) != 27 || mat_(1,4,3) !=  30 ) {
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
   // PageSlice::scale()
   //=====================================================================================

   {
      test_ = "PageSlice::scale()";

      initialize();

      // Integral scaling the 3rd pageslice
      {
         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         pageslice2.scale( 3 );

         checkRows    ( pageslice2, 5UL );
         checkColumns ( pageslice2, 4UL );
         checkCapacity( pageslice2, 20UL );
         checkNonZeros( pageslice2, 10UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 20UL );

         if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
             pageslice2(1,0) !=  0 || pageslice2(1,1) !=   3 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
             pageslice2(2,0) != -6 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -9 || pageslice2(2,3) !=   0 ||
             pageslice2(3,0) !=  0 || pageslice2(3,1) !=  12 || pageslice2(3,2) != 15 || pageslice2(3,3) != -18 ||
             pageslice2(4,0) != 21 || pageslice2(4,1) != -24 || pageslice2(4,2) != 27 || pageslice2(4,3) !=  30 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Result:\n" << pageslice2 << "\n"
                   << "   Expected result:\n(( 0 0 0 0 )\n( 0 3 0 0 )\n( -6 0 -9 0 )\n( 0 12 15 -18 )\n( 21 -24 27 30 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
             mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
             mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
             mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
             mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
             mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
             mat_(1,1,0) !=  0 || mat_(1,1,1) !=   3 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
             mat_(1,2,0) != -6 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -9 || mat_(1,2,3) !=   0 ||
             mat_(1,3,0) !=  0 || mat_(1,3,1) !=  12 || mat_(1,3,2) != 15 || mat_(1,3,3) != -18 ||
             mat_(1,4,0) != 21 || mat_(1,4,1) != -24 || mat_(1,4,2) != 27 || mat_(1,4,3) !=  30 ) {
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

      // Floating point scaling the 3rd pageslice
      {
         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         pageslice2.scale( 0.5 );

         checkRows    ( pageslice2,  5UL );
         checkColumns ( pageslice2,  4UL );
         checkCapacity( pageslice2, 20UL );
         checkNonZeros( pageslice2,  9UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 19UL );

         if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=  0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=  0 ||
             pageslice2(1,0) !=  0 || pageslice2(1,1) !=  0 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=  0 ||
             pageslice2(2,0) != -1 || pageslice2(2,1) !=  0 || pageslice2(2,2) != -1 || pageslice2(2,3) !=  0 ||
             pageslice2(3,0) !=  0 || pageslice2(3,1) !=  2 || pageslice2(3,2) !=  2 || pageslice2(3,3) != -3 ||
             pageslice2(4,0) !=  3 || pageslice2(4,1) != -4 || pageslice2(4,2) !=  4 || pageslice2(4,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Result:\n" << pageslice2 << "\n"
                   << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( -1 0 -1 0 )\n( 0 12 2 -3 )\n( 3 -4 4 5 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=   0 ||
             mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=   0 ||
             mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=   0 ||
             mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) !=  -6 ||
             mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) !=  10 ||
             mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=   0 ||
             mat_(1,1,0) !=  0 || mat_(1,1,1) !=   0 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=   0 ||
             mat_(1,2,0) != -1 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -1 || mat_(1,2,3) !=   0 ||
             mat_(1,3,0) !=  0 || mat_(1,3,1) !=   2 || mat_(1,3,2) !=  2 || mat_(1,3,3) !=  -3 ||
             mat_(1,4,0) !=  3 || mat_(1,4,1) !=  -4 || mat_(1,4,2) !=  4 || mat_(1,4,3) !=   5 ) {
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
/*!\brief Test of the PageSlice function call operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the function call operator
// of the PageSlice specialization. In case an error is detected, a \a std::runtime_error exception
// is thrown.
*/
void DenseGeneralTest::testFunctionCall()
{
   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "PageSlice::operator()";

      initialize();

      RT pageslice2 = blaze::pageslice( mat_, 1UL );

      // Assignment to the element at index (0,1)
      pageslice2(0,1) = 9;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 11UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 21UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   9 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -3 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
          pageslice2(4,0) !=  7 || pageslice2(4,1) !=  -8 || pageslice2(4,2) !=  9 || pageslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=  9 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) != -2 || mat_(1,2,1) !=  0 || mat_(1,2,2) != -3 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) !=  7 || mat_(1,4,1) != -8 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
      pageslice2(2,2) = 0;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   9 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
          pageslice2(4,0) !=  7 || pageslice2(4,1) !=  -8 || pageslice2(4,2) !=  9 || pageslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=  9 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) != -2 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) !=  7 || mat_(1,4,1) != -8 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
      pageslice2(4,1) = -9;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   9 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
          pageslice2(4,0) !=  7 || pageslice2(4,1) !=  -9 || pageslice2(4,2) !=  9 || pageslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   9 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) != -2 || mat_(1,2,1) !=   0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=   4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) !=  7 || mat_(1,4,1) !=  -9 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
      pageslice2(0,1) += -3;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   6 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
          pageslice2(4,0) !=  7 || pageslice2(4,1) !=  -9 || pageslice2(4,2) !=  9 || pageslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=  6 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) != -2 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) !=  7 || mat_(1,4,1) != -9 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
      pageslice2(2,0) -= 6;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   6 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -8 || pageslice2(2,1) !=   0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
          pageslice2(4,0) !=  7 || pageslice2(4,1) !=  -9 || pageslice2(4,2) !=  9 || pageslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=  6 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) != -8 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) !=  7 || mat_(1,4,1) != -9 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
      pageslice2(4,0) *= -3;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=   0 || pageslice2(0,1) !=   6 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=   0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) !=  -8 || pageslice2(2,1) !=   0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=   0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
          pageslice2(4,0) != -21 || pageslice2(4,1) !=  -9 || pageslice2(4,2) !=  9 || pageslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -6 )\n( -21 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=   0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=   0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) !=  -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=   0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=   7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=   0 || mat_(1,0,1) !=  6 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=   0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) !=  -8 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=   0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) != -21 || mat_(1,4,1) != -9 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
      pageslice2(3,3) /= 2;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=   0 || pageslice2(0,1) !=   6 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=   0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) !=  -8 || pageslice2(2,1) !=   0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=   0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -3 ||
          pageslice2(4,0) != -21 || pageslice2(4,1) !=  -9 || pageslice2(4,2) !=  9 || pageslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -3 )\n( -21 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=   0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=   0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) !=  -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=   0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=   7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=   0 || mat_(1,0,1) !=  6 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=   0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) !=  -8 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=   0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -3 ||
          mat_(1,4,0) != -21 || mat_(1,4,1) != -9 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
/*!\brief Test of the PageSlice at() operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the at() operator
// of the PageSlice specialization. In case an error is detected, a \a std::runtime_error exception
// is thrown.
*/
void DenseGeneralTest::testAt()
{
   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "PageSlice::at()";

      initialize();

      RT pageslice2 = blaze::pageslice( mat_, 1UL );

      // Assignment to the element at index (0,1)
      pageslice2.at(0,1) = 9;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 11UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 21UL );

      if( pageslice2.at(0,0) !=  0 || pageslice2.at(0,1) !=   9 || pageslice2.at(0,2) !=  0 || pageslice2.at(0,3) !=   0 ||
          pageslice2.at(1,0) !=  0 || pageslice2.at(1,1) !=   1 || pageslice2.at(1,2) !=  0 || pageslice2.at(1,3) !=   0 ||
          pageslice2.at(2,0) != -2 || pageslice2.at(2,1) !=   0 || pageslice2.at(2,2) != -3 || pageslice2.at(2,3) !=   0 ||
          pageslice2.at(3,0) !=  0 || pageslice2.at(3,1) !=   4 || pageslice2.at(3,2) !=  5 || pageslice2.at(3,3) !=  -6 ||
          pageslice2.at(4,0) !=  7 || pageslice2.at(4,1) !=  -8 || pageslice2.at(4,2) !=  9 || pageslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=  9 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) != -2 || mat_(1,2,1) !=  0 || mat_(1,2,2) != -3 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) !=  7 || mat_(1,4,1) != -8 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
      pageslice2.at(2,2) = 0;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2.at(0,0) !=  0 || pageslice2.at(0,1) !=   9 || pageslice2.at(0,2) !=  0 || pageslice2.at(0,3) !=   0 ||
          pageslice2.at(1,0) !=  0 || pageslice2.at(1,1) !=   1 || pageslice2.at(1,2) !=  0 || pageslice2.at(1,3) !=   0 ||
          pageslice2.at(2,0) != -2 || pageslice2.at(2,1) !=   0 || pageslice2.at(2,2) !=  0 || pageslice2.at(2,3) !=   0 ||
          pageslice2.at(3,0) !=  0 || pageslice2.at(3,1) !=   4 || pageslice2.at(3,2) !=  5 || pageslice2.at(3,3) !=  -6 ||
          pageslice2.at(4,0) !=  7 || pageslice2.at(4,1) !=  -8 || pageslice2.at(4,2) !=  9 || pageslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=  9 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) != -2 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) !=  7 || mat_(1,4,1) != -8 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
      pageslice2.at(4,1) = -9;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2.at(0,0) !=  0 || pageslice2.at(0,1) !=   9 || pageslice2.at(0,2) !=  0 || pageslice2.at(0,3) !=   0 ||
          pageslice2.at(1,0) !=  0 || pageslice2.at(1,1) !=   1 || pageslice2.at(1,2) !=  0 || pageslice2.at(1,3) !=   0 ||
          pageslice2.at(2,0) != -2 || pageslice2.at(2,1) !=   0 || pageslice2.at(2,2) !=  0 || pageslice2.at(2,3) !=   0 ||
          pageslice2.at(3,0) !=  0 || pageslice2.at(3,1) !=   4 || pageslice2.at(3,2) !=  5 || pageslice2.at(3,3) !=  -6 ||
          pageslice2.at(4,0) !=  7 || pageslice2.at(4,1) !=  -9 || pageslice2.at(4,2) !=  9 || pageslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 9 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=   9 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=   1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) != -2 || mat_(1,2,1) !=   0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=   4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) !=  7 || mat_(1,4,1) !=  -9 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
      pageslice2.at(0,1) += -3;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2.at(0,0) !=  0 || pageslice2.at(0,1) !=   6 || pageslice2.at(0,2) !=  0 || pageslice2.at(0,3) !=   0 ||
          pageslice2.at(1,0) !=  0 || pageslice2.at(1,1) !=   1 || pageslice2.at(1,2) !=  0 || pageslice2.at(1,3) !=   0 ||
          pageslice2.at(2,0) != -2 || pageslice2.at(2,1) !=   0 || pageslice2.at(2,2) !=  0 || pageslice2.at(2,3) !=   0 ||
          pageslice2.at(3,0) !=  0 || pageslice2.at(3,1) !=   4 || pageslice2.at(3,2) !=  5 || pageslice2.at(3,3) !=  -6 ||
          pageslice2.at(4,0) !=  7 || pageslice2.at(4,1) !=  -9 || pageslice2.at(4,2) !=  9 || pageslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=  6 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) != -2 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) !=  7 || mat_(1,4,1) != -9 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
      pageslice2.at(2,0) -= 6;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2.at(0,0) !=  0 || pageslice2.at(0,1) !=   6 || pageslice2.at(0,2) !=  0 || pageslice2.at(0,3) !=   0 ||
          pageslice2.at(1,0) !=  0 || pageslice2.at(1,1) !=   1 || pageslice2.at(1,2) !=  0 || pageslice2.at(1,3) !=   0 ||
          pageslice2.at(2,0) != -8 || pageslice2.at(2,1) !=   0 || pageslice2.at(2,2) !=  0 || pageslice2.at(2,3) !=   0 ||
          pageslice2.at(3,0) !=  0 || pageslice2.at(3,1) !=   4 || pageslice2.at(3,2) !=  5 || pageslice2.at(3,3) !=  -6 ||
          pageslice2.at(4,0) !=  7 || pageslice2.at(4,1) !=  -9 || pageslice2.at(4,2) !=  9 || pageslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -6 )\n( 7 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=  0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) != -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=  0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=  7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=  0 || mat_(1,0,1) !=  6 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=  0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) != -8 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=  0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) !=  7 || mat_(1,4,1) != -9 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
      pageslice2.at(4,0) *= -3;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2.at(0,0) !=   0 || pageslice2.at(0,1) !=   6 || pageslice2.at(0,2) !=  0 || pageslice2.at(0,3) !=   0 ||
          pageslice2.at(1,0) !=   0 || pageslice2.at(1,1) !=   1 || pageslice2.at(1,2) !=  0 || pageslice2.at(1,3) !=   0 ||
          pageslice2.at(2,0) !=  -8 || pageslice2.at(2,1) !=   0 || pageslice2.at(2,2) !=  0 || pageslice2.at(2,3) !=   0 ||
          pageslice2.at(3,0) !=   0 || pageslice2.at(3,1) !=   4 || pageslice2.at(3,2) !=  5 || pageslice2.at(3,3) !=  -6 ||
          pageslice2.at(4,0) != -21 || pageslice2.at(4,1) !=  -9 || pageslice2.at(4,2) !=  9 || pageslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -6 )\n( -21 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=   0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=   0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) !=  -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=   0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=   7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=   0 || mat_(1,0,1) !=  6 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=   0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) !=  -8 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=   0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
          mat_(1,4,0) != -21 || mat_(1,4,1) != -9 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
      pageslice2.at(3,3) /= 2;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2.at(0,0) !=   0 || pageslice2.at(0,1) !=   6 || pageslice2.at(0,2) !=  0 || pageslice2.at(0,3) !=   0 ||
          pageslice2.at(1,0) !=   0 || pageslice2.at(1,1) !=   1 || pageslice2.at(1,2) !=  0 || pageslice2.at(1,3) !=   0 ||
          pageslice2.at(2,0) !=  -8 || pageslice2.at(2,1) !=   0 || pageslice2.at(2,2) !=  0 || pageslice2.at(2,3) !=   0 ||
          pageslice2.at(3,0) !=   0 || pageslice2.at(3,1) !=   4 || pageslice2.at(3,2) !=  5 || pageslice2.at(3,3) !=  -3 ||
          pageslice2.at(4,0) != -21 || pageslice2.at(4,1) !=  -9 || pageslice2.at(4,2) !=  9 || pageslice2.at(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: At() failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 6 0 0 )\n( 0 1 0 0 )\n( -8 0 0 0 )\n( 0 4 5 -3 )\n( -21 -9 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0,0) !=   0 || mat_(0,0,1) !=  0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
          mat_(0,1,0) !=   0 || mat_(0,1,1) !=  1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
          mat_(0,2,0) !=  -2 || mat_(0,2,1) !=  0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
          mat_(0,3,0) !=   0 || mat_(0,3,1) !=  4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
          mat_(0,4,0) !=   7 || mat_(0,4,1) != -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
          mat_(1,0,0) !=   0 || mat_(1,0,1) !=  6 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
          mat_(1,1,0) !=   0 || mat_(1,1,1) !=  1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
          mat_(1,2,0) !=  -8 || mat_(1,2,1) !=  0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
          mat_(1,3,0) !=   0 || mat_(1,3,1) !=  4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -3 ||
          mat_(1,4,0) != -21 || mat_(1,4,1) != -9 || mat_(1,4,2) !=  9 || mat_(1,4,3) != 10 ) {
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
/*!\brief Test of the PageSlice iterator implementation.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the iterator implementation of the PageSlice specialization.
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

         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         RT::ConstIterator it( begin( pageslice2, 2UL ) );

         if( it == end( pageslice2, 2UL ) || *it != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st pageslice via Iterator (end-begin)
      {
         test_ = "Iterator subtraction (end-begin)";

         RT pageslice1 = blaze::pageslice( mat_, 1UL );
         const ptrdiff_t number( end( pageslice1, 2UL ) - begin( pageslice1, 2UL ) );

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

      // Counting the number of elements in 1st pageslice via Iterator (begin-end)
      {
         test_ = "Iterator subtraction (begin-end)";

         RT pageslice1 = blaze::pageslice( mat_, 1UL );
         const ptrdiff_t number( begin( pageslice1, 2UL ) - end( pageslice1, 2UL ) );

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

      // Counting the number of elements in 2nd pageslice via ConstIterator (end-begin)
      {
         test_ = "ConstIterator subtraction (end-begin)";

         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         const ptrdiff_t number( cend( pageslice2, 2UL ) - cbegin( pageslice2, 2UL ) );

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

      // Counting the number of elements in 2nd pageslice via ConstIterator (begin-end)
      {
         test_ = "ConstIterator subtraction (begin-end)";

         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         const ptrdiff_t number( cbegin( pageslice2, 2UL ) - cend( pageslice2, 2UL ) );

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

         RT pageslice3 = blaze::pageslice( mat_, 0UL );
         RT::ConstIterator it ( cbegin( pageslice3, 4UL ) );
         RT::ConstIterator end( cend( pageslice3, 4UL ) );

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

         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         int value = 6;

         for( RT::Iterator it=begin( pageslice2, 4UL ); it!=end( pageslice2, 4UL ); ++it ) {
            *it = value++;
         }

         if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
             pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
             pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -3 || pageslice2(2,3) !=   0 ||
             pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
             pageslice2(4,0) !=  6 || pageslice2(4,1) !=   7 || pageslice2(4,2) !=  8 || pageslice2(4,3) !=   9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 6 7 8 9 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
             mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
             mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
             mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
             mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
             mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
             mat_(1,1,0) !=  0 || mat_(1,1,1) !=   1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
             mat_(1,2,0) != -2 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -3 || mat_(1,2,3) !=  0 ||
             mat_(1,3,0) !=  0 || mat_(1,3,1) !=   4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
             mat_(1,4,0) !=  6 || mat_(1,4,1) !=   7 || mat_(1,4,2) !=  8 || mat_(1,4,3) !=  9 ) {
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

         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         int value = 2;

         for( RT::Iterator it=begin( pageslice2, 4UL ); it!=end( pageslice2, 4UL ); ++it ) {
            *it += value++;
         }

         if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
             pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
             pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -3 || pageslice2(2,3) !=   0 ||
             pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
             pageslice2(4,0) !=  8 || pageslice2(4,1) !=  10 || pageslice2(4,2) != 12 || pageslice2(4,3) !=  14 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 8 10 12 14 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
             mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
             mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
             mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
             mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
             mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
             mat_(1,1,0) !=  0 || mat_(1,1,1) !=   1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
             mat_(1,2,0) != -2 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -3 || mat_(1,2,3) !=  0 ||
             mat_(1,3,0) !=  0 || mat_(1,3,1) !=   4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
             mat_(1,4,0) !=  8 || mat_(1,4,1) !=  10 || mat_(1,4,2) != 12 || mat_(1,4,3) != 14 ) {
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

         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         int value = 2;

         for( RT::Iterator it=begin( pageslice2, 4UL ); it!=end( pageslice2, 4UL ); ++it ) {
            *it -= value++;
         }

         if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
             pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
             pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -3 || pageslice2(2,3) !=   0 ||
             pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
             pageslice2(4,0) !=  6 || pageslice2(4,1) !=   7 || pageslice2(4,2) !=  8 || pageslice2(4,3) !=   9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 6 7 8 9 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
             mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
             mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
             mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
             mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
             mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
             mat_(1,1,0) !=  0 || mat_(1,1,1) !=   1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
             mat_(1,2,0) != -2 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -3 || mat_(1,2,3) !=  0 ||
             mat_(1,3,0) !=  0 || mat_(1,3,1) !=   4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
             mat_(1,4,0) !=  6 || mat_(1,4,1) !=   7 || mat_(1,4,2) !=  8 || mat_(1,4,3) !=  9 ) {
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

         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         int value = 1;

         for( RT::Iterator it=begin( pageslice2, 4UL ); it!=end( pageslice2, 4UL ); ++it ) {
            *it *= value++;
         }

         if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
             pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
             pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -3 || pageslice2(2,3) !=   0 ||
             pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
             pageslice2(4,0) !=  6 || pageslice2(4,1) !=  14 || pageslice2(4,2) != 24 || pageslice2(4,3) !=  36 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 6 14 24 36 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
             mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
             mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
             mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
             mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
             mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
             mat_(1,1,0) !=  0 || mat_(1,1,1) !=   1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
             mat_(1,2,0) != -2 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -3 || mat_(1,2,3) !=  0 ||
             mat_(1,3,0) !=  0 || mat_(1,3,1) !=   4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
             mat_(1,4,0) !=  6 || mat_(1,4,1) !=  14 || mat_(1,4,2) != 24 || mat_(1,4,3) != 36 ) {
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

         RT pageslice2 = blaze::pageslice( mat_, 1UL );

         for( RT::Iterator it=begin( pageslice2, 4UL ); it!=end( pageslice2, 4UL ); ++it ) {
            *it /= 2;
         }

         if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
             pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
             pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -3 || pageslice2(2,3) !=   0 ||
             pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
             pageslice2(4,0) !=  3 || pageslice2(4,1) !=   7 || pageslice2(4,2) != 12 || pageslice2(4,3) !=  18 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 3 7 12 18 ))\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
             mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
             mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
             mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
             mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
             mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
             mat_(1,1,0) !=  0 || mat_(1,1,1) !=   1 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
             mat_(1,2,0) != -2 || mat_(1,2,1) !=   0 || mat_(1,2,2) != -3 || mat_(1,2,3) !=  0 ||
             mat_(1,3,0) !=  0 || mat_(1,3,1) !=   4 || mat_(1,3,2) !=  5 || mat_(1,3,3) != -6 ||
             mat_(1,4,0) !=  3 || mat_(1,4,1) !=   7 || mat_(1,4,2) != 12 || mat_(1,4,3) != 18 ) {
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
/*!\brief Test of the \c nonZeros() member function of the PageSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c nonZeros() member function of the PageSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testNonZeros()
{
   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "PageSlice::nonZeros()";

      initialize();

      // Initialization check
      RT pageslice2 = blaze::pageslice( mat_, 1UL );

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) != -3 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
          pageslice2(4,0) !=  7 || pageslice2(4,1) !=  -8 || pageslice2(4,2) !=  9 || pageslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 -3 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the dense pageslice
      pageslice2(2, 2) = 0;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2,  9UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 19UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
          pageslice2(4,0) !=  7 || pageslice2(4,1) !=  -8 || pageslice2(4,2) !=  9 || pageslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the dense matrix
      mat_(1,3,0) = 5;

      checkRows    ( pageslice2, 5UL );
      checkColumns ( pageslice2, 4UL );
      checkCapacity( pageslice2, 20UL );
      checkNonZeros( pageslice2, 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkPages   ( mat_,  2UL );
      checkNonZeros( mat_, 20UL );

      if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
          pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
          pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=   0 ||
          pageslice2(3,0) !=  5 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
          pageslice2(4,0) !=  7 || pageslice2(4,1) !=  -8 || pageslice2(4,2) !=  9 || pageslice2(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Matrix function call operator failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice2 << "\n"
             << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 5 4 5 -6 )\n( 7 -8 9 10 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reset() member function of the PageSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reset() member function of the PageSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testReset()
{
   using blaze::reset;


   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "PageSlice::reset()";

      // Resetting a single element in pageslice 3
      {
         initialize();

         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         reset( pageslice2(2, 2) );


         checkRows    ( pageslice2, 5UL );
         checkColumns ( pageslice2, 4UL );
         checkCapacity( pageslice2, 20UL );
         checkNonZeros( pageslice2,  9UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 19UL );

         if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
             pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
             pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=   0 ||
             pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
             pageslice2(4,0) !=  7 || pageslice2(4,1) !=  -8 || pageslice2(4,2) !=  9 || pageslice2(4,3) !=  10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Resetting the 1st pageslice (lvalue)
      {
         initialize();

         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         reset( pageslice2 );

         checkRows    ( pageslice2, 5UL );
         checkColumns ( pageslice2, 4UL );
         checkCapacity( pageslice2, 20UL );
         checkNonZeros( pageslice2,  0UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 10UL );

         if( pageslice2(0,0) != 0 || pageslice2(0,1) !=  0 || pageslice2(0,2) != 0 || pageslice2(0,3) != 0 ||
             pageslice2(1,0) != 0 || pageslice2(1,1) !=  0 || pageslice2(1,2) != 0 || pageslice2(1,3) != 0 ||
             pageslice2(2,0) != 0 || pageslice2(2,1) !=  0 || pageslice2(2,2) != 0 || pageslice2(2,3) != 0 ||
             pageslice2(3,0) != 0 || pageslice2(3,1) !=  0 || pageslice2(3,2) != 0 || pageslice2(3,3) != 0 ||
             pageslice2(4,0) != 0 || pageslice2(4,1) !=  0 || pageslice2(4,2) != 0 || pageslice2(4,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 1st pageslice failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Resetting the 1st pageslice (rvalue)
      {
         initialize();

         reset( blaze::pageslice( mat_, 1UL ) );

         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 10UL );

         if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
             mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
             mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
             mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
             mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
             mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
             mat_(1,1,0) !=  0 || mat_(1,1,1) !=   0 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
             mat_(1,2,0) !=  0 || mat_(1,2,1) !=   0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
             mat_(1,3,0) !=  0 || mat_(1,3,1) !=   0 || mat_(1,3,2) !=  0 || mat_(1,3,3) !=  0 ||
             mat_(1,4,0) !=  0 || mat_(1,4,1) !=   0 || mat_(1,4,2) !=  0 || mat_(1,4,3) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 1st pageslice failed\n"
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
/*!\brief Test of the \c clear() function with the PageSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c clear() function with the PageSlice specialization.
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

      // Clearing a single element in pageslice 1
      {
         initialize();

         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         clear( pageslice2(2, 2) );

         checkRows    ( pageslice2, 5UL );
         checkColumns ( pageslice2, 4UL );
         checkCapacity( pageslice2, 20UL );
         checkNonZeros( pageslice2,  9UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 19UL );

         if( pageslice2(0,0) !=  0 || pageslice2(0,1) !=   0 || pageslice2(0,2) !=  0 || pageslice2(0,3) !=   0 ||
             pageslice2(1,0) !=  0 || pageslice2(1,1) !=   1 || pageslice2(1,2) !=  0 || pageslice2(1,3) !=   0 ||
             pageslice2(2,0) != -2 || pageslice2(2,1) !=   0 || pageslice2(2,2) !=  0 || pageslice2(2,3) !=   0 ||
             pageslice2(3,0) !=  0 || pageslice2(3,1) !=   4 || pageslice2(3,2) !=  5 || pageslice2(3,3) !=  -6 ||
             pageslice2(4,0) !=  7 || pageslice2(4,1) !=  -8 || pageslice2(4,2) !=  9 || pageslice2(4,3) !=  10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Clear operation failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 1 0 0 )\n( -2 0 0 0 )\n( 0 4 5 -6 )\n( 7 -8 9 10 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Clearing the 3rd pageslice (lvalue)
      {
         initialize();

         RT pageslice2 = blaze::pageslice( mat_, 1UL );
         clear( pageslice2 );

         checkRows    ( pageslice2, 5UL );
         checkColumns ( pageslice2, 4UL );
         checkCapacity( pageslice2, 20UL );
         checkNonZeros( pageslice2,  0UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 10UL );

         if( pageslice2(0,0) != 0 || pageslice2(0,1) !=  0 || pageslice2(0,2) != 0 || pageslice2(0,3) != 0 ||
             pageslice2(1,0) != 0 || pageslice2(1,1) !=  0 || pageslice2(1,2) != 0 || pageslice2(1,3) != 0 ||
             pageslice2(2,0) != 0 || pageslice2(2,1) !=  0 || pageslice2(2,2) != 0 || pageslice2(2,3) != 0 ||
             pageslice2(3,0) != 0 || pageslice2(3,1) !=  0 || pageslice2(3,2) != 0 || pageslice2(3,3) != 0 ||
             pageslice2(4,0) != 0 || pageslice2(4,1) !=  0 || pageslice2(4,2) != 0 || pageslice2(4,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Clear operation of 3rd pageslice failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice2 << "\n"
                << "   Expected result:\n(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Clearing the 4th pageslice (rvalue)
      {
         initialize();

         clear( blaze::pageslice( mat_, 1UL ) );

         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkPages   ( mat_,  2UL );
         checkNonZeros( mat_, 10UL );

         if( mat_(0,0,0) !=  0 || mat_(0,0,1) !=   0 || mat_(0,0,2) !=  0 || mat_(0,0,3) !=  0 ||
             mat_(0,1,0) !=  0 || mat_(0,1,1) !=   1 || mat_(0,1,2) !=  0 || mat_(0,1,3) !=  0 ||
             mat_(0,2,0) != -2 || mat_(0,2,1) !=   0 || mat_(0,2,2) != -3 || mat_(0,2,3) !=  0 ||
             mat_(0,3,0) !=  0 || mat_(0,3,1) !=   4 || mat_(0,3,2) !=  5 || mat_(0,3,3) != -6 ||
             mat_(0,4,0) !=  7 || mat_(0,4,1) !=  -8 || mat_(0,4,2) !=  9 || mat_(0,4,3) != 10 ||
             mat_(1,0,0) !=  0 || mat_(1,0,1) !=   0 || mat_(1,0,2) !=  0 || mat_(1,0,3) !=  0 ||
             mat_(1,1,0) !=  0 || mat_(1,1,1) !=   0 || mat_(1,1,2) !=  0 || mat_(1,1,3) !=  0 ||
             mat_(1,2,0) !=  0 || mat_(1,2,1) !=   0 || mat_(1,2,2) !=  0 || mat_(1,2,3) !=  0 ||
             mat_(1,3,0) !=  0 || mat_(1,3,1) !=   0 || mat_(1,3,2) !=  0 || mat_(1,3,3) !=  0 ||
             mat_(1,4,0) !=  0 || mat_(1,4,1) !=   0 || mat_(1,4,2) !=  0 || mat_(1,4,3) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Clear operation of 1st pageslice failed\n"
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
/*!\brief Test of the \c isDefault() function with the PageSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDefault() function with the PageSlice specialization.
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

      // isDefault with default pageslice
      {
         RT pageslice0 = blaze::pageslice( mat_, 0UL );
         pageslice0 = 0;

         if( isDefault( pageslice0(0, 0) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   PageSlice element: " << pageslice0(0, 0) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( pageslice0 ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   PageSlice:\n" << pageslice0 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default pageslice
      {
         RT pageslice1 = blaze::pageslice( mat_, 1UL );

         if( isDefault( pageslice1(1, 1) ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   PageSlice element: " << pageslice1(1, 1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( pageslice1 ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   PageSlice:\n" << pageslice1 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isSame() function with the PageSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSame() function with the PageSlice specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testIsSame()
{
   //=====================================================================================
   // matrix tests
   //=====================================================================================

   {
      test_ = "isSame() function";

      // isSame with matching pageslices
      {
         RT pageslice1 = blaze::pageslice( mat_, 1UL );
         RT pageslice2 = blaze::pageslice( mat_, 1UL );

         if( blaze::isSame( pageslice1, pageslice2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First pageslice:\n" << pageslice1 << "\n"
                << "   Second pageslice:\n" << pageslice2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching pageslices
      {
         RT pageslice1 = blaze::pageslice( mat_, 0UL );
         RT pageslice2 = blaze::pageslice( mat_, 1UL );

         pageslice1 = 42;

         if( blaze::isSame( pageslice1, pageslice2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First pageslice:\n" << pageslice1 << "\n"
                << "   Second pageslice:\n" << pageslice2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with pageslice and matching submatrix
      {
         RT   pageslice1 = blaze::pageslice( mat_, 1UL );
         auto sv   = blaze::submatrix( pageslice1, 0UL, 0UL, 5UL, 4UL );

         if( blaze::isSame( pageslice1, sv ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense pageslice:\n" << pageslice1 << "\n"
                << "   Dense submatrix:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sv, pageslice1 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense pageslice:\n" << pageslice1 << "\n"
                << "   Dense submatrix:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with pageslice and non-matching submatrix (different size)
      {
         RT   pageslice1 = blaze::pageslice( mat_, 1UL );
         auto sv   = blaze::submatrix( pageslice1, 0UL, 0UL, 3UL, 3UL );

         if( blaze::isSame( pageslice1, sv ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense pageslice:\n" << pageslice1 << "\n"
                << "   Dense submatrix:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sv, pageslice1 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense pageslice:\n" << pageslice1 << "\n"
                << "   Dense submatrix:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with pageslice and non-matching submatrix (different offset)
      {
         RT   pageslice1 = blaze::pageslice( mat_, 1UL );
         auto sv   = blaze::submatrix( pageslice1, 1UL, 1UL, 3UL, 3UL );

         if( blaze::isSame( pageslice1, sv ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense pageslice:\n" << pageslice1 << "\n"
                << "   Dense submatrix:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sv, pageslice1 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Dense pageslice:\n" << pageslice1 << "\n"
                << "   Dense submatrix:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

//       // isSame with matching pageslices on a common submatrix
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 1UL, 2UL, 3UL );
//          auto pageslice1 = blaze::pageslice( sm, 1UL );
//          auto pageslice2 = blaze::pageslice( sm, 1UL );
//
//          if( blaze::isSame( pageslice1, pageslice2 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching pageslices on a common submatrix
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 1UL, 2UL, 3UL );
//          auto pageslice1 = blaze::pageslice( sm, 0UL );
//          auto pageslice2 = blaze::pageslice( sm, 1UL );
//
//          if( blaze::isSame( pageslice1, pageslice2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with matching subtensor on matrix and submatrix
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto pageslice1 = blaze::pageslice( mat_, 2UL );
//          auto pageslice2 = blaze::pageslice( sm  , 1UL );
//
//          if( blaze::isSame( pageslice1, pageslice2 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( pageslice2, pageslice1 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching pageslices on tensor and subtensor (different pageslice)
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto pageslice1 = blaze::pageslice( mat_, 1UL );
//          auto pageslice2 = blaze::pageslice( sm  , 1UL );
//
//          if( blaze::isSame( pageslice1, pageslice2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( pageslice2, pageslice1 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching pageslices on tensor and subtensor (different size)
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 3UL );
//          auto pageslice1 = blaze::pageslice( mat_, 2UL );
//          auto pageslice2 = blaze::pageslice( sm  , 1UL );
//
//          if( blaze::isSame( pageslice1, pageslice2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( pageslice2, pageslice1 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with matching pageslices on two subtensors
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 0UL, 3UL, 4UL );
//          auto pageslice1 = blaze::pageslice( sm1, 1UL );
//          auto pageslice2 = blaze::pageslice( sm2, 0UL );
//
//          if( blaze::isSame( pageslice1, pageslice2 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( pageslice2, pageslice1 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching pageslices on two subtensors (different pageslice)
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 0UL, 3UL, 4UL );
//          auto pageslice1 = blaze::pageslice( sm1, 1UL );
//          auto pageslice2 = blaze::pageslice( sm2, 1UL );
//
//          if( blaze::isSame( pageslice1, pageslice2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( pageslice2, pageslice1 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching pageslices on two subtensors (different size)
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 0UL, 3UL, 3UL );
//          auto pageslice1 = blaze::pageslice( sm1, 1UL );
//          auto pageslice2 = blaze::pageslice( sm2, 0UL );
//
//          if( blaze::isSame( pageslice1, pageslice2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( pageslice2, pageslice1 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with non-matching pageslices on two subtensors (different offset)
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 3UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 1UL, 3UL, 3UL );
//          auto pageslice1 = blaze::pageslice( sm1, 1UL );
//          auto pageslice2 = blaze::pageslice( sm2, 0UL );
//
//          if( blaze::isSame( pageslice1, pageslice2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( pageslice2, pageslice1 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First pageslice:\n" << pageslice1 << "\n"
//                 << "   Second pageslice:\n" << pageslice2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }

//       // isSame with matching pageslice submatrices on a subtensor
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 1UL, 2UL, 3UL );
//          auto pageslice1 = blaze::pageslice( sm, 1UL );
//          auto sv1  = blaze::submatrix( pageslice1, 0UL, 2UL );
//          auto sv2  = blaze::submatrix( pageslice1, 0UL, 2UL );
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

//       // isSame with non-matching pageslice subtensors on a submatrix (different size)
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 1UL, 2UL, 3UL );
//          auto pageslice1 = blaze::pageslice( sm, 1UL );
//          auto sv1  = blaze::submatrix( pageslice1, 0UL, 2UL );
//          auto sv2  = blaze::submatrix( pageslice1, 0UL, 3UL );
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
//       // isSame with non-matching pageslice subtensors on a submatrix (different offset)
//       {
//          auto sm   = blaze::subtensor( mat_, 1UL, 1UL, 2UL, 3UL );
//          auto pageslice1 = blaze::pageslice( sm, 1UL );
//          auto sv1  = blaze::submatrix( pageslice1, 0UL, 2UL );
//          auto sv2  = blaze::submatrix( pageslice1, 1UL, 2UL );
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

//       // isSame with matching pageslice subtensors on two subtensors
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 0UL, 3UL, 4UL );
//          auto pageslice1 = blaze::pageslice( sm1, 1UL );
//          auto pageslice2 = blaze::pageslice( sm2, 0UL );
//          auto sv1  = blaze::submatrix( pageslice1, 0UL, 2UL );
//          auto sv2  = blaze::submatrix( pageslice2, 0UL, 2UL );
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

//       // isSame with non-matching pageslice subtensors on two subtensors (different size)
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 0UL, 3UL, 4UL );
//          auto pageslice1 = blaze::pageslice( sm1, 1UL );
//          auto pageslice2 = blaze::pageslice( sm2, 0UL );
//          auto sv1  = blaze::submatrix( pageslice1, 0UL, 2UL );
//          auto sv2  = blaze::submatrix( pageslice2, 0UL, 3UL );
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

//       // isSame with non-matching pageslice subtensors on two subtensors (different offset)
//       {
//          auto sm1  = blaze::subtensor( mat_, 1UL, 0UL, 3UL, 4UL );
//          auto sm2  = blaze::subtensor( mat_, 2UL, 0UL, 3UL, 4UL );
//          auto pageslice1 = blaze::pageslice( sm1, 1UL );
//          auto pageslice2 = blaze::pageslice( sm2, 0UL );
//          auto sv1  = blaze::submatrix( pageslice1, 0UL, 2UL );
//          auto sv2  = blaze::submatrix( pageslice2, 1UL, 2UL );
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
/*!\brief Test of the \c submatrix() function with the PageSlice specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c submatrix() function used with the PageSlice specialization.
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
         RT   pageslice1 = blaze::pageslice( mat_, 1UL );
         auto sm = blaze::submatrix( pageslice1, 1UL, 1UL, 2UL, 3UL );

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
         RT   pageslice1 = blaze::pageslice( mat_, 1UL );
         auto sm = blaze::submatrix( pageslice1, 4UL, 0UL, 4UL, 4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         RT   pageslice1 = blaze::pageslice( mat_, 1UL );
         auto sm = blaze::submatrix( pageslice1, 0UL, 0UL, 2UL, 6UL );

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
   using blaze::pageslice;
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
         RT pageslice1  = pageslice( mat_, 0UL );
         RT pageslice2  = pageslice( mat_, 1UL );
         auto row1 = row( pageslice1, 1UL );
         auto row2 = row( pageslice2, 1UL );

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
         RT pageslice1  = pageslice( mat_, 0UL );
         auto row8 = row( pageslice1, 8UL );

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
   using blaze::pageslice;
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
         RT pageslice1 = pageslice( mat_, 0UL );
         RT pageslice2 = pageslice( mat_, 1UL );
         auto rs1 = rows( pageslice1, { 0UL, 2UL, 4UL, 3UL } );
         auto rs2 = rows( pageslice2, { 0UL, 2UL, 4UL, 3UL } );

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
         RT pageslice1 = pageslice( mat_, 1UL );
         auto rs  = rows( pageslice1, { 8UL } );

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
   using blaze::pageslice;
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
         RT pageslice1  = pageslice( mat_, 0UL );
         RT pageslice2  = pageslice( mat_, 1UL );
         auto col1 = column( pageslice1, 1UL );
         auto col2 = column( pageslice2, 1UL );

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
         RT pageslice1  = pageslice( mat_, 0UL );
         auto col16 = column( pageslice1, 16UL );

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
   using blaze::pageslice;
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
         RT pageslice1  = pageslice( mat_, 0UL );
         RT pageslice2  = pageslice( mat_, 1UL );
         auto cs1 = columns( pageslice1, { 0UL, 2UL, 2UL, 3UL } );
         auto cs2 = columns( pageslice2, { 0UL, 2UL, 2UL, 3UL } );

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
         RT pageslice1 = pageslice( mat_, 1UL );
         auto cs  = columns( pageslice1, { 16UL } );

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
   using blaze::pageslice;
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
         RT pageslice1  = pageslice( mat_, 0UL );
         RT pageslice2  = pageslice( mat_, 1UL );
         auto b1 = band( pageslice1, 1L );
         auto b2 = band( pageslice2, 1L );

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
         RT pageslice1 = pageslice( mat_, 1UL );
         auto b8 = band( pageslice1, -8L );

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
   // Initializing the pageslice-major dynamic matrix
   mat_.reset();
   mat_(0,1,1) =  1;
   mat_(0,2,0) = -2;
   mat_(0,2,2) = -3;
   mat_(0,3,1) =  4;
   mat_(0,3,2) =  5;
   mat_(0,3,3) = -6;
   mat_(0,4,0) =  7;
   mat_(0,4,1) = -8;
   mat_(0,4,2) =  9;
   mat_(0,4,3) = 10;
   mat_(1,1,1) =  1;
   mat_(1,2,0) = -2;
   mat_(1,2,2) = -3;
   mat_(1,3,1) =  4;
   mat_(1,3,2) =  5;
   mat_(1,3,3) = -6;
   mat_(1,4,0) =  7;
   mat_(1,4,1) = -8;
   mat_(1,4,2) =  9;
   mat_(1,4,3) = 10;
}
//*************************************************************************************************

} // namespace pageslice

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
   std::cout << "   Running PageSlice dense general test..." << std::endl;

   try
   {
      RUN_PAGESLICE_DENSEGENERAL_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during PageSlice dense general test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
