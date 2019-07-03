//=================================================================================================
/*!
//  \file src/mathtest/customarray/AlignedPaddedTest1.cpp
//  \brief Source file for the aligned/padded CustomArray class test (part 1)
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
#include <memory>
#include <blaze/math/shims/NextMultiple.h>
#include <blaze/system/Platform.h>
#include <blaze/util/Complex.h>
#include <blaze/util/Memory.h>
#include <blaze/util/policies/Deallocate.h>
#include <blaze/util/Random.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blaze/util/typetraits/IsVectorizable.h>
#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>

#include <blazetest/mathtest/customarray/AlignedPaddedTest.h>
#include <blaze_tensor/math/CustomArray.h>
#include <blaze_tensor/math/DynamicArray.h>

namespace blazetest {

namespace mathtest {

namespace customarray {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the CustomArray class test.
//
// \exception std::runtime_error Operation error detected.
*/
AlignedPaddedTest::AlignedPaddedTest()
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
/*!\brief Test of the CustomArray constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the CustomArray class template.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void AlignedPaddedTest::testConstructors()
{
   //=====================================================================================
   // Row-major default constructor
   //=====================================================================================

   {
      test_ = "Row-major CustomArray default constructor";

      MT mat;

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }


   //=====================================================================================
   // Row-major constructor ( Type*, size_t, size_t, size_t )
   //=====================================================================================

   {
      test_ = "Row-major CustomArray constructor ( Type*, size_t, size_t, size_t )";

      // Constructor a 2x3 custom tensor
      {
         std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
         MT mat( memory.get(), 2UL, 2UL, 3UL, 16UL );

         checkRows    ( mat,  2UL );
         checkColumns ( mat,  3UL );
         checkPages   ( mat,  2UL );
         checkCapacity( mat, 64UL );
      }

      // Trying to construct a custom tensor with invalid array of elements
      try {
         MT mat( nullptr, 0UL, 0UL, 0UL, 0UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Constructing a custom tensor with a nullptr succeeded\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Trying to construct a custom tensor with invalid alignment
      if( blaze::AlignmentOf<int>::value > sizeof(int) )
      {
         try {
            std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 65UL ) );
            MT mat( memory.get()+1UL, 2UL, 2UL, 2UL, 16UL );

            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Constructing a custom tensor with invalid alignment succeeded\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
         catch( std::invalid_argument& ) {}
      }

      // Trying to construct a custom tensor with invalid row alignment
      if( blaze::AlignmentOf<int>::value > sizeof(int) )
      {
         try {
            std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 60UL ) );
            MT mat( memory.get(), 2UL, 2UL, 2UL, 15UL );

            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Constructing a custom tensor with invalid row alignment succeeded\n"
                << " Details:\n"
                << "   Result:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
         catch( std::invalid_argument& ) {}
      }

      // Trying to construct a custom tensor with invalid padding
      if( blaze::IsVectorizable<int>::value )
      {
         try {
            std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 12UL ) );
            MT mat( memory.get(), 2UL, 2UL, 2UL, 3UL );

            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Constructing a custom tensor with invalid padding succeeded\n";
            throw std::runtime_error( oss.str() );
         }
         catch( std::invalid_argument& ) {}
      }
   }


   //=====================================================================================
   // Row-major copy constructor
   //=====================================================================================

   {
      test_ = "Row-major CustomArray copy constructor (0x0)";

      MT mat1;
      MT mat2( mat1 );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major CustomArray copy constructor (0x3x2)";

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
      MT mat1( memory.get(), 2UL, 0UL, 3UL, 16UL );
      MT mat2( mat1 );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 3UL );
      checkPages   ( mat2, 2UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major CustomArray copy constructor (2x0x2)";

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 20UL ) );
      MT mat1( memory.get(), 2UL, 2UL, 0UL, 0UL );
      MT mat2( mat1 );

      checkRows    ( mat2, 2UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 2UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major CustomArray copy constructor (2x2x0)";

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
      MT mat1( memory.get(), 0UL, 2UL, 2UL, 16UL );
      MT mat2( mat1 );

      checkRows    ( mat2, 2UL );
      checkColumns ( mat2, 2UL );
      checkPages   ( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major CustomArray copy constructor (2x3)";

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      MT mat1( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat1(0,0,0) = 1;
      mat1(0,0,1) = 2;
      mat1(0,0,2) = 3;
      mat1(0,1,0) = 4;
      mat1(0,1,1) = 5;
      mat1(0,1,2) = 6;
      mat1(1,0,0) = 1;
      mat1(1,0,1) = 2;
      mat1(1,0,2) = 3;
      mat1(1,1,0) = 4;
      mat1(1,1,1) = 5;
      mat1(1,1,2) = 6;

      MT mat2( mat1 );

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 2 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 4 || mat2(0,1,1) != 5 || mat2(0,1,2) != 6 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 2 || mat2(1,0,2) != 3 ||
          mat2(1,1,0) != 4 || mat2(1,1,1) != 5 || mat2(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major move constructor
   //=====================================================================================

   {
      test_ = "Row-major CustomArray move constructor (0x0x0)";

      MT mat1;
      MT mat2( std::move( mat1 ) );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major CustomArray move constructor (0x3x2)";

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
      MT mat1( memory.get(), 2UL, 0UL, 3UL, 16UL );
      MT mat2( std::move( mat1 ) );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 3UL );
      checkPages   ( mat2, 2UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major CustomArray move constructor (2x0x2)";

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 20UL ) );
      MT mat1( memory.get(), 2UL, 2UL, 0UL, 0UL );
      MT mat2( std::move( mat1 ) );

      checkRows    ( mat2, 2UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 2UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major CustomArray move constructor (2x2x0)";

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
      MT mat1( memory.get(), 0UL, 2UL, 2UL, 16UL );
      MT mat2( std::move( mat1 ) );

      checkRows    ( mat2, 2UL );
      checkColumns ( mat2, 2UL );
      checkPages   ( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "Row-major CustomArray move constructor (2x3x2)";

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      MT mat1( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat1(0,0,0) = 1;
      mat1(0,0,1) = 2;
      mat1(0,0,2) = 3;
      mat1(0,1,0) = 4;
      mat1(0,1,1) = 5;
      mat1(0,1,2) = 6;
      mat1(1,0,0) = 1;
      mat1(1,0,1) = 2;
      mat1(1,0,2) = 3;
      mat1(1,1,0) = 4;
      mat1(1,1,1) = 5;
      mat1(1,1,2) = 6;

      MT mat2( std::move( mat1 ) );

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 2 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 4 || mat2(0,1,1) != 5 || mat2(0,1,2) != 6 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 2 || mat2(1,0,2) != 3 ||
          mat2(1,1,0) != 4 || mat2(1,1,1) != 5 || mat2(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the CustomArray assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the CustomArray class template.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void AlignedPaddedTest::testAssignment()
{
   //=====================================================================================
   // Row-major homogeneous assignment
   //=====================================================================================

   {
      test_ = "Row-major CustomArray homogeneous assignment";

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 96UL ) );
      MT mat( memory.get(), 2UL, 3UL, 4UL, 16UL );
      mat = 2;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  4UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 96UL );
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
          mat(0,0,0) != 2 || mat(0,0,1) != 2 || mat(0,0,2) != 2 || mat(0,0,3) != 2 ||
          mat(0,1,0) != 2 || mat(0,1,1) != 2 || mat(0,1,2) != 2 || mat(0,1,3) != 2 ||
          mat(0,2,0) != 2 || mat(0,2,1) != 2 || mat(0,2,2) != 2 || mat(0,2,3) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 ))\n(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major list assignment
   //=====================================================================================

   {
      test_ = "Row-major CustomArray initializer list assignment (complete list)";

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      MT mat( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat = {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}};

      checkRows    ( mat, 2UL );
      checkColumns ( mat, 3UL );
      checkPages   ( mat, 2UL );
      checkCapacity( mat, 12UL );
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
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major StaticMatrix initializer list assignment (incomplete list)";

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      MT mat( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat = {{{1}, {4, 5, 6}}, {{1}, {4, 5, 6}}};

      checkRows    ( mat, 2UL );
      checkColumns ( mat, 3UL );
      checkPages   ( mat, 2UL );
      checkCapacity( mat, 12UL );
      checkNonZeros( mat, 8UL );
      checkNonZeros( mat, 0UL, 0UL, 1UL );
      checkNonZeros( mat, 1UL, 0UL, 3UL );
      checkNonZeros( mat, 0UL, 1UL, 1UL );
      checkNonZeros( mat, 1UL, 1UL, 3UL );

      if( mat(0,0,0) != 1 || mat(0,0,1) != 0 || mat(0,0,2) != 0 ||
          mat(0,1,0) != 4 || mat(0,1,1) != 5 || mat(0,1,2) != 6 ||
          mat(0,0,0) != 1 || mat(0,0,1) != 0 || mat(0,0,2) != 0 ||
          mat(0,1,0) != 4 || mat(0,1,1) != 5 || mat(0,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 1 0 0 )\n( 4 5 6 ))\n(( 1 0 0 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major array assignment
   //=====================================================================================

   {
      test_ = "Row-major CustomArray array assignment";

      const int array[2UL][2UL][3UL] = {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}};
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      MT mat( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat = MT( &array[0][0][0], 2UL, 2UL, 3UL );

      checkRows    ( mat, 2UL );
      checkColumns ( mat, 3UL );
      checkPages   ( mat, 2UL );
      checkCapacity( mat, 12UL );
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
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major copy assignment
   //=====================================================================================

   {
      test_ = "Row-major CustomArray copy assignment";

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      MT mat1( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat1(0,0,0) = 1;
      mat1(0,0,1) = 2;
      mat1(0,0,2) = 3;
      mat1(0,1,0) = 4;
      mat1(0,1,1) = 5;
      mat1(0,1,2) = 6;
      mat1(1,0,0) = 1;
      mat1(1,0,1) = 2;
      mat1(1,0,2) = 3;
      mat1(1,1,0) = 4;
      mat1(1,1,1) = 5;
      mat1(1,1,2) = 6;

      std::unique_ptr<int[],blaze::Deallocate> memory2( blaze::allocate<int>( 64UL ) );
      MT mat2( memory2.get(), 2UL, 2UL, 3UL, 16UL );
      mat2 = mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 2 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 4 || mat2(0,1,1) != 5 || mat2(0,1,2) != 6 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 2 || mat2(1,0,2) != 3 ||
          mat2(1,1,0) != 4 || mat2(1,1,1) != 5 || mat2(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major move assignment
   //=====================================================================================

   {
      test_ = "Row-major CustomArray move assignment";

      std::unique_ptr<int[],blaze::Deallocate> memory1( blaze::allocate<int>( 64UL ) );
      MT mat1( memory1.get(), 2UL, 2UL, 3UL, 16UL );
      mat1(0,0,0) = 1;
      mat1(0,0,1) = 2;
      mat1(0,0,2) = 3;
      mat1(0,1,0) = 4;
      mat1(0,1,1) = 5;
      mat1(0,1,2) = 6;
      mat1(1,0,0) = 1;
      mat1(1,0,1) = 2;
      mat1(1,0,2) = 3;
      mat1(1,1,0) = 4;
      mat1(1,1,1) = 5;
      mat1(1,1,2) = 6;

      std::unique_ptr<int[],blaze::Deallocate> memory2( blaze::allocate<int>( 64UL ) );
      MT mat2( memory2.get(), 2UL, 2UL, 3UL, 16UL );
      mat2 = std::move( mat1 );

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 2 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 4 || mat2(0,1,1) != 5 || mat2(0,1,2) != 6 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 2 || mat2(1,0,2) != 3 ||
          mat2(1,1,0) != 4 || mat2(1,1,1) != 5 || mat2(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dense tensor assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major CustomArray dense tensor assignment (mixed type)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomArray<3,short,aligned,padded>;
      std::unique_ptr<short[],blaze::Deallocate> memory1( blaze::allocate<short>( 64UL ) );
      AlignedPadded mat1( memory1.get(), 2UL, 2UL, 3UL, 16UL );
      mat1(0,0,0) = 1U;
      mat1(0,0,1) = 2U;
      mat1(0,0,2) = 3U;
      mat1(0,1,0) = 4U;
      mat1(0,1,1) = 5U;
      mat1(0,1,2) = 6U;
      mat1(1,0,0) = 1U;
      mat1(1,0,1) = 2U;
      mat1(1,0,2) = 3U;
      mat1(1,1,0) = 4U;
      mat1(1,1,1) = 5U;
      mat1(1,1,2) = 6U;

      std::unique_ptr<int[],blaze::Deallocate> memory2( blaze::allocate<int>( 64UL ) );
      MT mat2( memory2.get(), 2UL, 2UL, 3UL, 16UL );
      mat2 = mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 2 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 4 || mat2(0,1,1) != 5 || mat2(0,1,2) != 6 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 2 || mat2(1,0,2) != 3 ||
          mat2(1,1,0) != 4 || mat2(1,1,1) != 5 || mat2(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major CustomArray dense tensor assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomArray<3,unsigned int,aligned,padded>;
      std::unique_ptr<unsigned int[],blaze::Deallocate> memory1( blaze::allocate<unsigned int>( 64UL ) );
      AlignedPadded mat1( memory1.get(), 2UL, 2UL, 3UL, 16UL );
      mat1(0,0,0) = 1U;
      mat1(0,0,1) = 2U;
      mat1(0,0,2) = 3U;
      mat1(0,1,0) = 4U;
      mat1(0,1,1) = 5U;
      mat1(0,1,2) = 6U;
      mat1(1,0,0) = 1U;
      mat1(1,0,1) = 2U;
      mat1(1,0,2) = 3U;
      mat1(1,1,0) = 4U;
      mat1(1,1,1) = 5U;
      mat1(1,1,2) = 6U;

      std::unique_ptr<int[],blaze::Deallocate> memory2( blaze::allocate<int>( 64UL ) );
      MT mat2( memory2.get(), 2UL, 2UL, 3UL, 16UL );
      mat2 = mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 2 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 4 || mat2(0,1,1) != 5 || mat2(0,1,2) != 6 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 2 || mat2(1,0,2) != 3 ||
          mat2(1,1,0) != 4 || mat2(1,1,1) != 5 || mat2(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major CustomArray dense tensor assignment stress test (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      const short min( randmin );
      const short max( randmax );

      for( size_t i=0UL; i<10UL; ++i )
      {
         const size_t rows   ( blaze::rand<size_t>( 0UL, 16UL ) );
         const size_t columns( blaze::rand<size_t>( 0UL, 16UL ) );
         const size_t pages  ( blaze::rand<size_t>( 0UL, 16UL ) );
         const size_t spacing( blaze::nextMultiple<size_t>( columns, 16UL ) );

         using AlignedPadded = blaze::CustomArray<3,short,aligned,padded>;
         std::unique_ptr<short[],blaze::Deallocate> memory1( blaze::allocate<short>( rows*spacing*pages ) );
         AlignedPadded mat1( memory1.get(), pages, rows, columns, spacing );
         randomize( mat1, min, max );

         std::unique_ptr<int[],blaze::Deallocate> memory2( blaze::allocate<int>( rows*spacing*pages ) );
         MT mat2( memory2.get(), pages, rows, columns, spacing );
         mat2 = mat1;

         if( mat1 != mat2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment failed\n"
                << " Details:\n"
                << "   Result:\n" << mat2 << "\n"
                << "   Expected result:\n" << mat1 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }

   {
      test_ = "Row-major/row-major CustomArray dense tensor assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      using UnalignedUnpadded = blaze::CustomArray<3,int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory1( new int[13UL] );
      UnalignedUnpadded mat1( memory1.get()+1UL, 2UL, 2UL, 3UL );
      mat1(0,0,0) = 1;
      mat1(0,0,1) = 2;
      mat1(0,0,2) = 3;
      mat1(0,1,0) = 4;
      mat1(0,1,1) = 5;
      mat1(0,1,2) = 6;
      mat1(1,0,0) = 1;
      mat1(1,0,1) = 2;
      mat1(1,0,2) = 3;
      mat1(1,1,0) = 4;
      mat1(1,1,1) = 5;
      mat1(1,1,2) = 6;

      std::unique_ptr<int[],blaze::Deallocate> memory2( blaze::allocate<int>( 64UL ) );
      MT mat2( memory2.get(), 2UL, 2UL, 3UL, 16UL );
      mat2 = mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2,  0UL, 0UL, 3UL );
      checkNonZeros( mat2,  1UL, 0UL, 3UL );
      checkNonZeros( mat2,  0UL, 1UL, 3UL );
      checkNonZeros( mat2,  1UL, 1UL, 3UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 2 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 4 || mat2(0,1,1) != 5 || mat2(0,1,2) != 6 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 2 || mat2(1,0,2) != 3 ||
          mat2(1,1,0) != 4 || mat2(1,1,1) != 5 || mat2(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n(( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major CustomArray dense tensor assignment stress test (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      const int min( randmin );
      const int max( randmax );

      for( size_t i=0UL; i<10UL; ++i )
      {
         const size_t rows   ( blaze::rand<size_t>( 0UL, 16UL ) );
         const size_t columns( blaze::rand<size_t>( 0UL, 16UL ) );
         const size_t pages  ( blaze::rand<size_t>( 0UL, 16UL ) );
         const size_t spacing( blaze::nextMultiple<size_t>( columns, 16UL ) );

         using UnalignedUnpadded = blaze::CustomArray<3,int,unaligned,unpadded>;
         std::unique_ptr<int[]> memory1( new int[rows*columns*pages+1UL] );
         UnalignedUnpadded mat1( memory1.get()+1UL, pages, rows, columns );
         randomize( mat1, min, max );

         std::unique_ptr<int[],blaze::Deallocate> memory2( blaze::allocate<int>( rows*spacing*pages ) );
         MT mat2( memory2.get(), pages, rows, columns, spacing );
         mat2 = mat1;

         if( mat1 != mat2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment failed\n"
                << " Details:\n"
                << "   Result:\n" << mat2 << "\n"
                << "   Expected result:\n" << mat1 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the CustomArray addition assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the addition assignment operators of the CustomArray class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void AlignedPaddedTest::testAddAssign()
{
   //=====================================================================================
   // Row-major dense tensor addition assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major CustomArray dense tensor addition assignment (mixed type)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomArray<3,short,aligned,padded>;
      std::unique_ptr<short[],blaze::Deallocate> memory1( blaze::allocate<short>( 64UL ) );
      AlignedPadded mat1( memory1.get(), 2UL, 2UL, 3UL, 16UL );
      mat1 = 0;
      mat1(0,0,0) =  1;
      mat1(0,0,1) =  2;
      mat1(0,1,0) = -3;
      mat1(0,1,2) =  4;
      mat1(1,0,0) =  1;
      mat1(1,0,1) =  2;
      mat1(1,1,0) = -3;
      mat1(1,1,2) =  4;

      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      MT mat2( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat2 = 0;
      mat2(0,0,1) = -2;
      mat2(0,0,2) =  6;
      mat2(0,1,0) =  5;
      mat2(1,0,1) = -2;
      mat2(1,0,2) =  6;
      mat2(1,1,0) =  5;

      mat2 += mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2,  8UL );
      checkNonZeros( mat2,  0UL, 0UL, 2UL );
      checkNonZeros( mat2,  1UL, 0UL, 2UL );
      checkNonZeros( mat2,  0UL, 1UL, 2UL );
      checkNonZeros( mat2,  1UL, 1UL, 2UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 0 || mat2(0,0,2) != 6 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 0 || mat2(0,1,2) != 4 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 0 || mat2(1,0,2) != 6 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 0 || mat2(1,1,2) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 0 6 )\n( 2 0 4 ))\n(( 1 0 6 )\n( 2 0 4 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major CustomArray dense tensor addition assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomArray<3,int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory1( blaze::allocate<int>( 64UL ) );
      AlignedPadded mat1( memory1.get(), 2UL, 2UL, 3UL, 16UL );
      mat1 = 0;
      mat1(0,0,0) =  1;
      mat1(0,0,1) =  2;
      mat1(0,1,0) = -3;
      mat1(0,1,2) =  4;
      mat1(1,0,0) =  1;
      mat1(1,0,1) =  2;
      mat1(1,1,0) = -3;
      mat1(1,1,2) =  4;

      std::unique_ptr<int[],blaze::Deallocate> memory2( blaze::allocate<int>( 64UL ) );
      MT mat2( memory2.get(), 2UL, 2UL, 3UL, 16UL );
      mat2 = 0;
      mat2(0,0,1) = -2;
      mat2(0,0,2) =  6;
      mat2(0,1,0) =  5;
      mat2(1,0,1) = -2;
      mat2(1,0,2) =  6;
      mat2(1,1,0) =  5;

      mat2 += mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2,  8UL );
      checkNonZeros( mat2,  0UL, 0UL, 2UL );
      checkNonZeros( mat2,  1UL, 0UL, 2UL );
      checkNonZeros( mat2,  0UL, 1UL, 2UL );
      checkNonZeros( mat2,  1UL, 1UL, 2UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 0 || mat2(0,0,2) != 6 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 0 || mat2(0,1,2) != 4 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 0 || mat2(1,0,2) != 6 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 0 || mat2(1,1,2) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 0 6 )\n( 2 0 4 ))\n(( 1 0 6 )\n( 2 0 4 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major CustomArray dense tensor addition assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      using UnalignedUnpadded = blaze::CustomArray<3,int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory1( new int[13UL] );
      UnalignedUnpadded mat1( memory1.get()+1UL, 2UL, 2UL, 3UL );
      mat1 = 0;
      mat1(0,0,0) =  1;
      mat1(0,0,1) =  2;
      mat1(0,1,0) = -3;
      mat1(0,1,2) =  4;
      mat1(1,0,0) =  1;
      mat1(1,0,1) =  2;
      mat1(1,1,0) = -3;
      mat1(1,1,2) =  4;

      std::unique_ptr<int[],blaze::Deallocate> memory2( blaze::allocate<int>( 64UL ) );
      MT mat2( memory2.get(), 2UL, 2UL, 3UL, 16UL );
      mat2 = 0;
      mat2(0,0,1) = -2;
      mat2(0,0,2) =  6;
      mat2(0,1,0) =  5;
      mat2(1,0,1) = -2;
      mat2(1,0,2) =  6;
      mat2(1,1,0) =  5;

      mat2 += mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2,  8UL );
      checkNonZeros( mat2,  0UL, 0UL, 2UL );
      checkNonZeros( mat2,  1UL, 0UL, 2UL );
      checkNonZeros( mat2,  0UL, 1UL, 2UL );
      checkNonZeros( mat2,  1UL, 1UL, 2UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 0 || mat2(0,0,2) != 6 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 0 || mat2(0,1,2) != 4 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 0 || mat2(1,0,2) != 6 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 0 || mat2(1,1,2) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 0 6 )\n( 2 0 4 ))\n(( 1 0 6 )\n( 2 0 4 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the CustomArray subtraction assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the subtraction assignment operators of the CustomArray
// class template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void AlignedPaddedTest::testSubAssign()
{
   //=====================================================================================
   // Row-major dense tensor subtraction assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major CustomArray dense tensor subtraction assignment (mixed type)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomArray<3,short,aligned,padded>;
      std::unique_ptr<short[],blaze::Deallocate> memory1( blaze::allocate<short>( 64UL ) );
      AlignedPadded mat1( memory1.get(), 2UL, 2UL, 3UL, 16UL );
      mat1 = 0;
      mat1(0,0,0) = -1;
      mat1(0,0,1) = -2;
      mat1(0,1,0) =  3;
      mat1(0,1,2) = -4;
      mat1(1,0,0) = -1;
      mat1(1,0,1) = -2;
      mat1(1,1,0) =  3;
      mat1(1,1,2) = -4;

      std::unique_ptr<int[],blaze::Deallocate> memory2( blaze::allocate<int>( 64UL ) );
      MT mat2( memory2.get(), 2UL, 2UL, 3UL, 16UL );
      mat2 = 0;
      mat2(0,0,1) = -2;
      mat2(0,0,2) =  6;
      mat2(0,1,0) =  5;
      mat2(1,0,1) = -2;
      mat2(1,0,2) =  6;
      mat2(1,1,0) =  5;

      mat2 -= mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2,  8UL );
      checkNonZeros( mat2,  0UL, 0UL, 2UL );
      checkNonZeros( mat2,  1UL, 0UL, 2UL );
      checkNonZeros( mat2,  0UL, 1UL, 2UL );
      checkNonZeros( mat2,  1UL, 1UL, 2UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 0 || mat2(0,0,2) != 6 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 0 || mat2(0,1,2) != 4 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 0 || mat2(1,0,2) != 6 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 0 || mat2(1,1,2) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 0 6 )\n( 2 0 4 ))\n(( 1 0 6 )\n( 2 0 4 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major CustomArray dense tensor subtraction assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomArray<3,int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory1( blaze::allocate<int>( 64UL ) );
      AlignedPadded mat1( memory1.get(), 2UL, 2UL, 3UL, 16UL );
      mat1 = 0;
      mat1(0,0,0) = -1;
      mat1(0,0,1) = -2;
      mat1(0,1,0) =  3;
      mat1(0,1,2) = -4;
      mat1(1,0,0) = -1;
      mat1(1,0,1) = -2;
      mat1(1,1,0) =  3;
      mat1(1,1,2) = -4;

      std::unique_ptr<int[],blaze::Deallocate> memory2( blaze::allocate<int>( 64UL ) );
      MT mat2( memory2.get(), 2UL, 2UL, 3UL, 16UL );
      mat2 = 0;
      mat2(0,0,1) = -2;
      mat2(0,0,2) =  6;
      mat2(0,1,0) =  5;
      mat2(1,0,1) = -2;
      mat2(1,0,2) =  6;
      mat2(1,1,0) =  5;

      mat2 -= mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2,  8UL );
      checkNonZeros( mat2,  0UL, 0UL, 2UL );
      checkNonZeros( mat2,  1UL, 0UL, 2UL );
      checkNonZeros( mat2,  0UL, 1UL, 2UL );
      checkNonZeros( mat2,  1UL, 1UL, 2UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 0 || mat2(0,0,2) != 6 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 0 || mat2(0,1,2) != 4 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 0 || mat2(1,0,2) != 6 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 0 || mat2(1,1,2) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 0 6 )\n( 2 0 4 ))\n(( 1 0 6 )\n( 2 0 4 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major CustomArray dense tensor subtraction assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      using UnalignedUnpadded = blaze::CustomArray<3,int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory1( new int[13UL] );
      UnalignedUnpadded mat1( memory1.get()+1UL, 2UL, 2UL, 3UL );
      mat1 = 0;
      mat1(0,0,0) = -1;
      mat1(0,0,1) = -2;
      mat1(0,1,0) =  3;
      mat1(0,1,2) = -4;
      mat1(1,0,0) = -1;
      mat1(1,0,1) = -2;
      mat1(1,1,0) =  3;
      mat1(1,1,2) = -4;

      std::unique_ptr<int[],blaze::Deallocate> memory2( blaze::allocate<int>( 64UL ) );
      MT mat2( memory2.get(), 2UL, 2UL, 3UL, 16UL );
      mat2 = 0;
      mat2(0,0,1) = -2;
      mat2(0,0,2) =  6;
      mat2(0,1,0) =  5;
      mat2(1,0,1) = -2;
      mat2(1,0,2) =  6;
      mat2(1,1,0) =  5;

      mat2 -= mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 64UL );
      checkNonZeros( mat2,  8UL );
      checkNonZeros( mat2,  0UL, 0UL, 2UL );
      checkNonZeros( mat2,  1UL, 0UL, 2UL );
      checkNonZeros( mat2,  0UL, 1UL, 2UL );
      checkNonZeros( mat2,  1UL, 1UL, 2UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 0 || mat2(0,0,2) != 6 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 0 || mat2(0,1,2) != 4 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 0 || mat2(1,0,2) != 6 ||
          mat2(1,1,0) != 2 || mat2(1,1,1) != 0 || mat2(1,1,2) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 0 6 )\n( 2 0 4 ))\n(( 1 0 6 )\n( 2 0 4 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************

} // namespace customarray

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
   std::cout << "   Running aligned/padded CustomArray class test (part 1)..." << std::endl;

   try
   {
      RUN_CUSTOMARRAY_ALIGNED_PADDED_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during aligned/padded CustomArray class test (part 1):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
