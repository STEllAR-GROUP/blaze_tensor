//=================================================================================================
/*!
//  \file src/mathtest/subtensor/DenseAlignedTest1.cpp
//  \brief Source file for the Subtensor dense aligned test (part 1)
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
#include <blaze/util/Memory.h>
#include <blaze/util/policies/Deallocate.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>

#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/CustomTensor.h>

#include <blazetest/mathtest/subtensor/DenseAlignedTest.h>

namespace blazetest {

namespace mathtest {

namespace subtensor {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the Subtensor dense aligned test.
//
// \exception std::runtime_error Operation error detected.
*/
DenseAlignedTest::DenseAlignedTest()
   : mat1_ ( 16UL, 16UL, 16UL )
   , mat2_ ( 16UL, 16UL, 16UL )
{
   testConstructors();
   testAssignment();
   testAddAssign();
   testSubAssign();
   testSchurAssign();
   testMultAssign();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the Subtensor constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the Subtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testConstructors()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major subtensor tests
   //=====================================================================================

   {
      test_ = "Row-major Subtensor constructor";

      initialize();

      const size_t alignment = blaze::AlignmentOf<int>::value;

      for( size_t page=0UL; page<mat1_.pages(); page+=alignment ) {
         for( size_t row=0UL; row<mat1_.rows(); row+=alignment ) {
            for( size_t column=0UL; column<mat1_.columns(); column+=alignment ) {
               for( size_t maxo=0UL; ; maxo+=alignment )
               {
                  for( size_t maxm=0UL; ; maxm+=alignment )
                  {
                     for( size_t maxn=0UL; ; maxn+=alignment )
                     {
                        const size_t m( blaze::min( maxm, mat1_.rows()-row ) );
                        const size_t n( blaze::min( maxn, mat1_.columns()-column ) );
                        const size_t o( blaze::min( maxo, mat1_.pages()-page) );

                        const ASMT sm1 = subtensor<aligned>  ( mat1_, page, row, column, o, m, n );
                        const USMT sm2 = subtensor<unaligned>( mat2_, page, row, column, o, m, n );

                        if( sm1 != sm2 ) {
                           std::ostringstream oss;
                           oss << " Test: " << test_ << "\n"
                               << " Error: Setup of dense subtensor failed\n"
                               << " Details:\n"
                               << "   Index of first row    = " << row << "\n"
                               << "   Index of first column = " << column << "\n"
                               << "   Index of first page   = " << page << "\n"
                               << "   Number of rows        = " << m << "\n"
                               << "   Number of columns     = " << n << "\n"
                               << "   Number of pages       = " << o << "\n"
                               << "   Subtensor:\n" << sm1 << "\n"
                               << "   Reference:\n" << sm2 << "\n";
                           throw std::runtime_error( oss.str() );
                        }

                        if( column+maxn > mat1_.columns() ) break;
                     }

                     if( row+maxm > mat1_.rows() ) break;
                  }

                  if( page+maxo > mat1_.pages() ) break;
               }
            }
         }
      }

      try {
         ASMT sm = subtensor<aligned>( mat1_, 2UL, 0UL, 8UL, 16UL, 16UL, 16UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ASMT sm = subtensor<aligned>( mat1_, 2UL, 8UL, 0UL, 16UL, 16UL, 16UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ASMT sm = subtensor<aligned>( mat1_, 0UL, 8UL, 2UL, 16UL, 16UL, 16UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ASMT sm = subtensor<aligned>( mat1_, 0UL, 72UL, 0UL, 8UL, 8UL, 8UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ASMT sm = subtensor<aligned>( mat1_, 0UL, 0UL, 72UL, 8UL, 8UL, 8UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ASMT sm = subtensor<aligned>( mat1_, 72UL, 0UL, 0UL, 8UL, 8UL, 8UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      if( blaze::AlignmentOf<int>::value > sizeof(int) )
      {
         try {
            ASMT sm = subtensor<aligned>( mat1_, 8UL, 8UL, 7UL, 8UL, 8UL, 8UL );

            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of unaligned subtensor succeeded\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
         catch( std::invalid_argument& ) {}
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the Subtensor assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the Subtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testAssignment()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowMajor;
   using blaze::initializer_list;


   //=====================================================================================
   // Row-major homogeneous assignment
   //=====================================================================================

   {
      test_ = "Row-major Subtensor homogeneous assignment";

      initialize();

      // Assigning to a 8x12x8 subtensor
      {
         ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 8UL, 8UL, 12UL );
         USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 8UL, 8UL, 12UL );
         sm1 = 12;
         sm2 = 12;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 12UL );
         checkPages  ( sm1,  8UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 12UL );
         checkPages  ( sm1,  8UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assigning to a 12x8x8 subtensor
      {
         ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 0UL, 8UL, 12UL, 8UL );
         USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 0UL, 8UL, 12UL, 8UL );
         sm1 = 15;
         sm2 = 15;

         checkRows   ( sm1, 12UL );
         checkColumns( sm1,  8UL );
         checkPages  ( sm1,  8UL );
         checkRows   ( sm2, 12UL );
         checkColumns( sm2,  8UL );
         checkPages  ( sm2,  8UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assigning to a 8x8x12 subtensor
      {
         ASMT sm1 = subtensor<aligned>  ( mat1_, 4UL, 2UL, 8UL, 12UL, 8UL, 8UL );
         USMT sm2 = subtensor<unaligned>( mat2_, 4UL, 2UL, 8UL, 12UL, 8UL, 8UL );
         sm1 = 42;
         sm2 = 42;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1,  8UL );
         checkPages  ( sm1, 12UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2,  8UL );
         checkPages  ( sm2, 12UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major list assignment
   //=====================================================================================

   {
      test_ = "Row-major initializer list assignment (complete list)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      initializer_list< initializer_list< initializer_list<int> > > list = {
         { { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12 },
           { 2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24 },
           { 3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36 },
           { 4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48 },
           { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 },
           { 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72 },
           { 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84 },
           { 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96 } },
         { { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12 },
           { 2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24 },
           { 3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36 },
           { 4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48 },
           { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 },
           { 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72 },
           { 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84 },
           { 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96 } },
         { { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12 },
           { 2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24 },
           { 3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36 },
           { 4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48 },
           { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 },
           { 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72 },
           { 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84 },
           { 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96 } },
         { { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12 },
           { 2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24 },
           { 3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36 },
           { 4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48 },
           { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 },
           { 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72 },
           { 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84 },
           { 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96 } }
      };

      sm1 = list;
      sm2 = list;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major initializer list assignment (incomplete list)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      initializer_list< initializer_list< initializer_list<int> > > list = {
         { { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12 },
           { 2,  4,  6,  8, 10, 12, 14, 16, 18, 20 },
           { 3,  6,  9, 12, 15, 18, 21, 24 },
           { 4,  8, 12, 16, 20, 24 },
           { 5, 10, 15, 20 },
           { 6, 12 } },
         { { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12 },
           { 2,  4,  6,  8, 10, 12, 14, 16, 18, 20 },
           { 3,  6,  9, 12, 15, 18, 21, 24 },
           { 4,  8, 12, 16, 20, 24 },
           { 5, 10, 15, 20 },
           { 6, 12 } },
         { { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12 },
           { 2,  4,  6,  8, 10, 12, 14, 16, 18, 20 },
           { 3,  6,  9, 12, 15, 18, 21, 24 },
           { 4,  8, 12, 16, 20, 24 },
           { 5, 10, 15, 20 },
           { 6, 12 } },
         { { 1 } }
      };

      sm1 = list;
      sm2 = list;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major copy assignment
   //=====================================================================================

   {
      test_ = "Row-major Subtensor copy assignment (no aliasing)";

      initialize();

      MT mat1( 16UL, 16UL, 16UL );
      MT mat2( 16UL, 16UL, 16UL );
      randomize( mat1, int(randmin), int(randmax) );
      mat2 = mat1;

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      sm1 = subtensor<aligned>  ( mat1, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      sm2 = subtensor<unaligned>( mat2, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major Subtensor copy assignment (aliasing)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      sm1 = subtensor<aligned>  ( mat1_, 0UL, 0UL, 0UL, 4UL, 8UL, 12UL );
      sm2 = subtensor<unaligned>( mat2_, 0UL, 0UL, 0UL, 4UL, 8UL, 12UL );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dense tensor assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major dense tensor assignment (mixed type)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      blaze::DynamicTensor<short> mat( 4UL, 8UL, 12UL );
      randomize( mat, short(randmin), short(randmax) );

      sm1 = mat;
      sm2 = mat;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major dense tensor assignment (aligned/padded)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 6144UL ) );
      AlignedPadded mat( memory.get(), 4UL, 8UL, 12UL, 16UL );
      randomize( mat, int(randmin), int(randmax) );

      sm1 = mat;
      sm2 = mat;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major dense tensor assignment (unaligned/unpadded)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[385UL] );
      UnalignedUnpadded mat( memory.get()+1UL, 4UL, 8UL, 12UL );
      randomize( mat, int(randmin), int(randmax) );

      sm1 = mat;
      sm2 = mat;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the Subtensor addition assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the addition assignment operators of the Subtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testAddAssign()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowMajor;


   //=====================================================================================
   // Row-major Subtensor addition assignment
   //=====================================================================================

   {
      test_ = "Row-major Subtensor addition assignment (no aliasing)";

      initialize();

      MT mat1( 16UL, 16UL, 16UL );
      MT mat2( 16UL, 16UL, 16UL );
      randomize( mat1, int(randmin), int(randmax) );
      mat2 = mat1;

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      sm1 += subtensor<aligned>  ( mat1, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      sm2 += subtensor<unaligned>( mat2, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major Subtensor addition assignment (aliasing)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      sm1 += subtensor<aligned>  ( mat1_, 0UL, 0UL, 0UL, 4UL, 8UL, 12UL );
      sm2 += subtensor<unaligned>( mat2_, 0UL, 0UL, 0UL, 4UL, 8UL, 12UL );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dense tensor addition assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major dense tensor addition assignment (mixed type)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      blaze::DynamicTensor<short> mat( 4UL, 8UL, 12UL );
      randomize( mat, short(randmin), short(randmax) );

      sm1 += mat;
      sm2 += mat;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major dense tensor addition assignment (aligned/padded)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 6144UL ) );
      AlignedPadded mat( memory.get(), 4UL, 8UL, 12UL, 16UL );
      randomize( mat, int(randmin), int(randmax) );

      sm1 += mat;
      sm2 += mat;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major dense tensor addition assignment (unaligned/unpadded)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[385UL] );
      UnalignedUnpadded mat( memory.get()+1UL, 4UL, 8UL, 12UL );
      randomize( mat, int(randmin), int(randmax) );

      sm1 += mat;
      sm2 += mat;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the Subtensor subtraction assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the subtraction assignment operators of the Subtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testSubAssign()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowMajor;


   //=====================================================================================
   // Row-major Subtensor subtraction assignment
   //=====================================================================================

   {
      test_ = "Row-major Subtensor subtraction assignment (no aliasing)";

      initialize();

      MT mat1( 16UL, 16UL, 16UL );
      MT mat2( 16UL, 16UL, 16UL );
      randomize( mat1, int(randmin), int(randmax) );
      mat2 = mat1;

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      sm1 -= subtensor<aligned>  ( mat1, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      sm2 -= subtensor<unaligned>( mat2, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major Subtensor subtraction assignment (aliasing)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      sm1 -= subtensor<aligned>  ( mat1_, 0UL, 0UL, 0UL, 4UL, 8UL, 12UL );
      sm2 -= subtensor<unaligned>( mat2_, 0UL, 0UL, 0UL, 4UL, 8UL, 12UL );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dense tensor subtraction assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major dense tensor subtraction assignment (mixed type)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      blaze::DynamicTensor<short> mat( 4UL, 8UL, 12UL );
      randomize( mat, short(randmin), short(randmax) );

      sm1 -= mat;
      sm2 -= mat;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major dense tensor subtraction assignment (aligned/padded)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 6144UL ) );
      AlignedPadded mat( memory.get(), 4UL, 8UL, 12UL, 16UL );
      randomize( mat, int(randmin), int(randmax) );

      sm1 -= mat;
      sm2 -= mat;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major dense tensor subtraction assignment (unaligned/unpadded)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[385UL] );
      UnalignedUnpadded mat( memory.get()+1UL, 4UL, 8UL, 12UL );
      randomize( mat, int(randmin), int(randmax) );

      sm1 -= mat;
      sm2 -= mat;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the Subtensor Schur product assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the Schur product assignment operators of the Subtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testSchurAssign()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowMajor;


   //=====================================================================================
   // Row-major Subtensor Schur product assignment
   //=====================================================================================

   {
      test_ = "Row-major Subtensor Schur product assignment (no aliasing)";

      initialize();

      MT mat1( 16UL, 16UL, 16UL );
      MT mat2( 16UL, 16UL, 16UL );
      randomize( mat1, int(randmin), int(randmax) );
      mat2 = mat1;

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      sm1 %= subtensor<aligned>  ( mat1, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      sm2 %= subtensor<unaligned>( mat2, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major Subtensor Schur product assignment (aliasing)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      sm1 %= subtensor<aligned>  ( mat1_, 0UL, 0UL, 0UL, 4UL, 8UL, 12UL );
      sm2 %= subtensor<unaligned>( mat2_, 0UL, 0UL, 0UL, 4UL, 8UL, 12UL );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dense tensor Schur product assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major dense tensor Schur product assignment (mixed type)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      blaze::DynamicTensor<short> mat( 4UL, 8UL, 12UL );
      randomize( mat, short(randmin), short(randmax) );

      sm1 %= mat;
      sm2 %= mat;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major dense tensor Schur product assignment (aligned/padded)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 6144UL ) );
      AlignedPadded mat( memory.get(), 4UL, 8UL, 12UL, 16UL );
      randomize( mat, int(randmin), int(randmax) );

      sm1 %= mat;
      sm2 %= mat;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major/row-major dense tensor Schur product assignment (unaligned/unpadded)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 0UL, 4UL, 8UL, 12UL );

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[385UL] );
      UnalignedUnpadded mat( memory.get()+1UL, 4UL, 8UL, 12UL );
      randomize( mat, int(randmin), int(randmax) );

      sm1 %= mat;
      sm2 %= mat;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  4UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  4UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the Subtensor multiplication assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the multiplication assignment operators of the Subtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testMultAssign()
{
//    using blaze::subtensor;
//    using blaze::aligned;
//    using blaze::unaligned;
//    using blaze::padded;
//    using blaze::unpadded;
//    using blaze::rowMajor;
//
//
//    //=====================================================================================
//    // Row-major Subtensor multiplication assignment
//    //=====================================================================================
//
//    {
//       test_ = "Row-major Subtensor multiplication assignment (no aliasing)";
//
//       initialize();
//
//       MT mat1( 16UL, 16UL );
//       MT mat2( 16UL, 16UL );
//       randomize( mat1, int(randmin), int(randmax) );
//       mat2 = mat1;
//
//       ASMT sm1 = subtensor<aligned>  ( mat1_, 8UL, 16UL, 16UL, 8UL );
//       USMT sm2 = subtensor<unaligned>( mat2_, 8UL, 16UL, 16UL, 8UL );
//       sm1 *= subtensor<aligned>  ( mat1, 8UL, 16UL, 16UL, 8UL );
//       sm2 *= subtensor<unaligned>( mat2, 8UL, 16UL, 16UL, 8UL );
//
//       checkRows   ( sm1, 8UL );
//       checkColumns( sm1, 8UL );
//       checkRows   ( sm2, 8UL );
//       checkColumns( sm2, 8UL );
//
//       if( sm1 != sm2 || mat1_ != mat2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Multiplication assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sm1 << "\n"
//              << "   Expected result:\n" << sm2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major Subtensor multiplication assignment (aliasing)";
//
//       initialize();
//
//       ASMT sm1 = subtensor<aligned>  ( mat1_, 8UL, 16UL, 16UL, 8UL );
//       USMT sm2 = subtensor<unaligned>( mat2_, 8UL, 16UL, 16UL, 8UL );
//       sm1 *= subtensor<aligned>  ( mat1_, 8UL, 24UL, 24UL, 8UL );
//       sm2 *= subtensor<unaligned>( mat2_, 8UL, 24UL, 24UL, 8UL );
//
//       checkRows   ( sm1, 8UL );
//       checkColumns( sm1, 8UL );
//       checkRows   ( sm2, 8UL );
//       checkColumns( sm2, 8UL );
//
//       if( sm1 != sm2 || mat1_ != mat2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Multiplication assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sm1 << "\n"
//              << "   Expected result:\n" << sm2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Row-major dense tensor multiplication assignment
//    //=====================================================================================
//
//    {
//       test_ = "Row-major/row-major dense tensor multiplication assignment (mixed type)";
//
//       initialize();
//
//       ASMT sm1 = subtensor<aligned>  ( mat1_, 8UL, 16UL, 16UL, 8UL );
//       USMT sm2 = subtensor<unaligned>( mat2_, 8UL, 16UL, 16UL, 8UL );
//
//       blaze::DynamicTensor<short,rowMajor> mat( 8UL, 8UL );
//       randomize( mat, short(randmin), short(randmax) );
//
//       sm1 *= mat;
//       sm2 *= mat;
//
//       checkRows   ( sm1, 8UL );
//       checkColumns( sm1, 8UL );
//       checkRows   ( sm2, 8UL );
//       checkColumns( sm2, 8UL );
//
//       if( sm1 != sm2 || mat1_ != mat2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Multiplication assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sm1 << "\n"
//              << "   Expected result:\n" << sm2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major dense tensor multiplication assignment (aligned/padded)";
//
//       initialize();
//
//       ASMT sm1 = subtensor<aligned>  ( mat1_, 8UL, 16UL, 16UL, 8UL );
//       USMT sm2 = subtensor<unaligned>( mat2_, 8UL, 16UL, 16UL, 8UL );
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,rowMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//       AlignedPadded mat( memory.get(), 8UL, 8UL, 16UL );
//       randomize( mat, int(randmin), int(randmax) );
//
//       sm1 *= mat;
//       sm2 *= mat;
//
//       checkRows   ( sm1, 8UL );
//       checkColumns( sm1, 8UL );
//       checkRows   ( sm2, 8UL );
//       checkColumns( sm2, 8UL );
//
//       if( sm1 != sm2 || mat1_ != mat2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Multiplication assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sm1 << "\n"
//              << "   Expected result:\n" << sm2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major dense tensor multiplication assignment (unaligned/unpadded)";
//
//       initialize();
//
//       ASMT sm1 = subtensor<aligned>  ( mat1_, 8UL, 16UL, 16UL, 8UL );
//       USMT sm2 = subtensor<unaligned>( mat2_, 8UL, 16UL, 16UL, 8UL );
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,rowMajor>;
//       std::unique_ptr<int[]> memory( new int[65UL] );
//       UnalignedUnpadded mat( memory.get()+1UL, 8UL, 8UL );
//       randomize( mat, int(randmin), int(randmax) );
//
//       sm1 *= mat;
//       sm2 *= mat;
//
//       checkRows   ( sm1, 8UL );
//       checkColumns( sm1, 8UL );
//       checkRows   ( sm2, 8UL );
//       checkColumns( sm2, 8UL );
//
//       if( sm1 != sm2 || mat1_ != mat2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Multiplication assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sm1 << "\n"
//              << "   Expected result:\n" << sm2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
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
void DenseAlignedTest::initialize()
{
   // Initializing the row-major dynamic matrices
   randomize( mat1_, int(randmin), int(randmax) );
   mat2_ = mat1_;
}
//*************************************************************************************************

} // namespace subtensor

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
   std::cout << "   Running Subtensor dense aligned test (part 1)..." << std::endl;

   try
   {
      RUN_SUBTENSOR_DENSEALIGNED_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during Subtensor dense aligned test (part 1):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
