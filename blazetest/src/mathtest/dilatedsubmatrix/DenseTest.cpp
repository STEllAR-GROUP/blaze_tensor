//=================================================================================================
/*!
//  \file src/mathtest/dilatedsubmatrix/DenseTest.cpp
//  \brief Source file for the dilatedsubmatrix dense aligned test
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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

#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/CustomMatrix.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/Views.h>
#include <blaze/util/Memory.h>
#include <blaze/util/policies/Deallocate.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>

#include <blazetest/mathtest/dilatedsubmatrix/DenseTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace dilatedsubmatrix {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the dilatedsubmatrix dense aligned test.
//
// \exception std::runtime_error Operation error detected.
*/
DenseTest::DenseTest()
   : mat1_ ( 64UL, 64UL )
   , mat2_ ( 64UL, 64UL )
   , tmat1_( 64UL, 64UL )
   , tmat2_( 64UL, 64UL )
{
   testConstructors();
   //testAssignment();
   //testAddAssign();
   //testSubAssign();
   //testSchurAssign();
   //testMultAssign();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the dilatedsubmatrix constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the dilatedsubmatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testConstructors()
{
   using blaze::dilatedsubmatrix;


   //=====================================================================================
   // Row-major dilatedsubmatrix tests
   //=====================================================================================

   //{
   //   test_ = "Row-major dilatedsubmatrix constructor";

   //   initialize();

   //   const size_t alignment = blaze::AlignmentOf<int>::value;

   //   for( size_t row=0UL; row<mat1_.rows(); row+=alignment ) {
   //      for( size_t column=0UL; column<mat1_.columns(); column+=alignment ) {
   //         for( size_t maxm=0UL; ; maxm+=alignment ) {
   //            for( size_t maxn=0UL; ; maxn+=alignment )
   //            {
   //               size_t m( blaze::min( maxm, mat1_.rows()-row ) );
   //               size_t n( blaze::min( maxn, mat1_.columns()-column ) );

   //               for (size_t rowdilation = 1UL; rowdilation < maxm; ++rowdilation)
   //               {
   //                  //for (size_t columndilation = 1UL; columndilation < maxn; ++columndilation)
   //                  //{
   //                     size_t columndilation = 1UL;
   //                     while( row + (m - 1) * rowdilation >= mat1_.rows() ) --m;
   //                     while( column + (n - 1) * columndilation >= mat1_.columns() ) --n;
   //                     auto row_indices = generate_indices( row, m, rowdilation );
   //                     auto column_indices = generate_indices( column, n, columndilation );
   //                     //auto sm1 = blaze::columns(blaze::rows(mat1_, row_indices.data(), row_indices.size()), column_indices.data(), column_indices.size());
   //                     blaze::Rows<blaze::DynamicMatrix<int>> sm1 = blaze::rows(mat1_, row_indices.data(), row_indices.size());
   //                     //auto sm1 = blaze::columns(mat1_, column_indices.data(), column_indices.size());
   //                     const DSMT sm2 = dilatedsubmatrix(mat2_, row, column, m, n, rowdilation, columndilation);
   //                     //const DSMT sm1 = sm2;

   //                     if (sm1 != sm2) {
   //                        std::ostringstream oss;
   //                        oss << " Test: " << test_ << "\n"
   //                           << " Error: Setup of dense dilatedsubmatrix failed\n"
   //                           << " Details:\n"
   //                           << "   Index of first row    = " << row << "\n"
   //                           << "   Index of first column = " << column << "\n"
   //                           << "   Number of rows        = " << m << "\n"
   //                           << "   Number of columns     = " << n << "\n"
   //                           << "   dilatedsubmatrix:\n" << sm1 << "\n"
   //                           << "   Reference:\n" << sm2 << "\n";
   //                        throw std::runtime_error(oss.str());
   //                     }
   //                  //}
   //               }
   //               if( column+maxn > mat1_.columns() ) break;
   //            }

   //            if( row+maxm > mat1_.rows() ) break;
   //         }
   //      }
   //   }

      //try {
      //   ASMT sm = dilatedsubmatrix<aligned>( mat1_, 0UL, 16UL, 64UL, 49UL );

      //   std::ostringstream oss;
      //   oss << " Test: " << test_ << "\n"
      //       << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
      //       << " Details:\n"
      //       << "   Result:\n" << sm << "\n";
      //   throw std::runtime_error( oss.str() );
      //}
      //catch( std::invalid_argument& ) {}

      //try {
      //   ASMT sm = dilatedsubmatrix<aligned>( mat1_, 16UL, 0UL, 49UL, 64UL );

      //   std::ostringstream oss;
      //   oss << " Test: " << test_ << "\n"
      //       << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
      //       << " Details:\n"
      //       << "   Result:\n" << sm << "\n";
      //   throw std::runtime_error( oss.str() );
      //}
      //catch( std::invalid_argument& ) {}

      //try {
      //   ASMT sm = dilatedsubmatrix<aligned>( mat1_, 80UL, 0UL, 8UL, 8UL );

      //   std::ostringstream oss;
      //   oss << " Test: " << test_ << "\n"
      //       << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
      //       << " Details:\n"
      //       << "   Result:\n" << sm << "\n";
      //   throw std::runtime_error( oss.str() );
      //}
      //catch( std::invalid_argument& ) {}

      //try {
      //   ASMT sm = dilatedsubmatrix<aligned>( mat1_, 0UL, 80UL, 8UL, 8UL );

      //   std::ostringstream oss;
      //   oss << " Test: " << test_ << "\n"
      //       << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
      //       << " Details:\n"
      //       << "   Result:\n" << sm << "\n";
      //   throw std::runtime_error( oss.str() );
      //}
      //catch( std::invalid_argument& ) {}

      //if( blaze::AlignmentOf<int>::value > sizeof(int) )
      //{
      //   try {
      //      ASMT sm = dilatedsubmatrix<aligned>( mat1_, 8UL, 7UL, 8UL, 8UL );

      //      std::ostringstream oss;
      //      oss << " Test: " << test_ << "\n"
      //          << " Error: Setup of unaligned dilatedsubmatrix succeeded\n"
      //          << " Details:\n"
      //          << "   Result:\n" << sm << "\n";
      //      throw std::runtime_error( oss.str() );
      //   }
      //   catch( std::invalid_argument& ) {}
      //}
   }


   //=====================================================================================
   // Column-major dilatedsubmatrix tests
   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubmatrix constructor";
//
//      initialize();
//
//      const size_t alignment = blaze::AlignmentOf<int>::value;
//
//      for( size_t column=0UL; column<mat1_.columns(); column+=alignment ) {
//         for( size_t row=0UL; row<mat1_.rows(); row+=alignment ) {
//            for( size_t maxn=0UL; ; maxn+=alignment ) {
//               for( size_t maxm=0UL; ; maxm+=alignment )
//               {
//                  const size_t n( blaze::min( maxn, mat1_.columns()-column ) );
//                  const size_t m( blaze::min( maxm, mat1_.rows()-row ) );
//
//                  const AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, row, column, m, n );
//                  const UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, row, column, m, n );
//
//                  if( sm1 != sm2 ) {
//                     std::ostringstream oss;
//                     oss << " Test: " << test_ << "\n"
//                         << " Error: Setup of dense dilatedsubmatrix failed\n"
//                         << " Details:\n"
//                         << "   Index of first row    = " << row << "\n"
//                         << "   Index of first column = " << column << "\n"
//                         << "   Number of rows        = " << m << "\n"
//                         << "   Number of columns     = " << n << "\n"
//                         << "   dilatedsubmatrix:\n" << sm1 << "\n"
//                         << "   Reference:\n" << sm2 << "\n";
//                     throw std::runtime_error( oss.str() );
//                  }
//
//                  if( row+maxm > mat1_.rows() ) break;
//               }
//
//               if( column+maxn > mat1_.columns() ) break;
//            }
//         }
//      }
//
//      try {
//         AOSMT sm = dilatedsubmatrix<aligned>( tmat1_, 0UL, 16UL, 64UL, 49UL );
//
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
//             << " Details:\n"
//             << "   Result:\n" << sm << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//      catch( std::invalid_argument& ) {}
//
//      try {
//         AOSMT sm = dilatedsubmatrix<aligned>( tmat1_, 16UL, 0UL, 49UL, 64UL );
//
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
//             << " Details:\n"
//             << "   Result:\n" << sm << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//      catch( std::invalid_argument& ) {}
//
//      try {
//         AOSMT sm = dilatedsubmatrix<aligned>( tmat1_, 80UL, 0UL, 8UL, 8UL );
//
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
//             << " Details:\n"
//             << "   Result:\n" << sm << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//      catch( std::invalid_argument& ) {}
//
//      try {
//         AOSMT sm = dilatedsubmatrix<aligned>( tmat1_, 0UL, 80UL, 8UL, 8UL );
//
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
//             << " Details:\n"
//             << "   Result:\n" << sm << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//      catch( std::invalid_argument& ) {}
//
//      if( blaze::AlignmentOf<int>::value > sizeof(int) )
//      {
//         try {
//            AOSMT sm = dilatedsubmatrix<aligned>( tmat1_, 7UL, 8UL, 8UL, 8UL );
//
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Setup of unaligned dilatedsubmatrix succeeded\n"
//                << " Details:\n"
//                << "   Result:\n" << sm << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//         catch( std::invalid_argument& ) {}
//      }
//   }
//}
////*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the dilatedsubmatrix assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the dilatedsubmatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testAssignment()
{
   using blaze::dilatedsubmatrix;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowMajor;
   using blaze::columnMajor;
   using blaze::initializer_list;


   //=====================================================================================
   // Row-major homogeneous assignment
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubmatrix homogeneous assignment";

      initialize();

      // Assigning to a 8x16 dilatedsubmatrix
      {
         auto row_indices = generate_indices( 8UL, 8UL, 2UL );
         auto column_indices = generate_indices( 16UL, 4UL, 3UL );

         DSMT sm1 = dilatedsubmatrix( mat1_, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
         auto sm_temp = blaze::rows( mat2_, row_indices.data(), row_indices.size() );
         CRMT sm2 = blaze::columns(
            sm_temp, column_indices.data(), column_indices.size() );
         sm1 = 12;
         sm2 = 12;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 16UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 16UL );

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

//      // Assigning to a 16x8 dilatedsubmatrix
//      {
//         ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 16UL, 8UL );
//         USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 16UL, 8UL );
//         sm1 = 15;
//         sm2 = 15;
//
//         checkRows   ( sm1, 16UL );
//         checkColumns( sm1,  8UL );
//         checkRows   ( sm2, 16UL );
//         checkColumns( sm2,  8UL );
//
//         if( sm1 != sm2 || mat1_ != mat2_ ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Assignment failed\n"
//                << " Details:\n"
//                << "   Result:\n" << sm1 << "\n"
//                << "   Expected result:\n" << sm2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
   }


//   //=====================================================================================
//   // Row-major list assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major initializer list assignment (complete list)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      initializer_list< initializer_list<int> > list =
//         { { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  13,  14,  15,  16 },
//           { 2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24,  26,  28,  30,  32 },
//           { 3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36,  39,  42,  45,  48 },
//           { 4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48,  52,  56,  60,  64 },
//           { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,  65,  70,  75,  80 },
//           { 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72,  78,  86,  92,  98 },
//           { 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84,  91,  98, 105, 112 },
//           { 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128 } };
//
//      sm1 = list;
//      sm2 = list;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major initializer list assignment (incomplete list)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      initializer_list< initializer_list<int> > list =
//         { { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  13,  14,  15,  16 },
//           { 2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24,  26,  28 },
//           { 3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36 },
//           { 4,  8, 12, 16, 20, 24, 28, 32, 36, 40 },
//           { 5, 10, 15, 20, 25, 30, 35, 40 },
//           { 6, 12, 18, 24, 30, 36 },
//           { 7, 14, 21, 28 },
//           { 8, 16 } };
//
//      sm1 = list;
//      sm2 = list;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Row-major copy assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major dilatedsubmatrix copy assignment (no aliasing)";
//
//      initialize();
//
//      MT mat1( 64UL, 64UL );
//      MT mat2( 64UL, 64UL );
//      randomize( mat1, int(randmin), int(randmax) );
//      mat2 = mat1;
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//      sm1 = dilatedsubmatrix<aligned>  ( mat1, 8UL, 16UL, 8UL, 16UL );
//      sm2 = dilatedsubmatrix<unaligned>( mat2, 8UL, 16UL, 8UL, 16UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major dilatedsubmatrix copy assignment (aliasing)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//      sm1 = dilatedsubmatrix<aligned>  ( mat1_, 12UL, 16UL, 8UL, 16UL );
//      sm2 = dilatedsubmatrix<unaligned>( mat2_, 12UL, 16UL, 8UL, 16UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Row-major dense matrix assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major dense matrix assignment (mixed type)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::DynamicMatrix<short,rowMajor> mat( 8UL, 16UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/row-major dense matrix assignment (aligned/padded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded mat( memory.get(), 8UL, 16UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/row-major dense matrix assignment (unaligned/unpadded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix assignment (mixed type)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::DynamicMatrix<short,columnMajor> mat( 8UL, 16UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix assignment (aligned/padded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded mat( memory.get(), 8UL, 16UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix assignment (unaligned/unpadded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Row-major sparse matrix assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major sparse matrix assignment";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::CompressedMatrix<int,rowMajor> mat( 8UL, 16UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major sparse matrix assignment";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::CompressedMatrix<int,columnMajor> mat( 8UL, 16UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major homogeneous assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubmatrix homogeneous assignment";
//
//      initialize();
//
//      // Assigning to a 8x16 dilatedsubmatrix
//      {
//         AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 8UL, 16UL );
//         UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 8UL, 16UL );
//         sm1 = 12;
//         sm2 = 12;
//
//         checkRows   ( sm1,  8UL );
//         checkColumns( sm1, 16UL );
//         checkRows   ( sm2,  8UL );
//         checkColumns( sm2, 16UL );
//
//         if( sm1 != sm2 || mat1_ != mat2_ ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Assignment failed\n"
//                << " Details:\n"
//                << "   Result:\n" << sm1 << "\n"
//                << "   Expected result:\n" << sm2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//
//      // Assigning to a 16x8 dilatedsubmatrix
//      {
//         AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//         UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//         sm1 = 15;
//         sm2 = 15;
//
//         checkRows   ( sm1, 16UL );
//         checkColumns( sm1,  8UL );
//         checkRows   ( sm2, 16UL );
//         checkColumns( sm2,  8UL );
//
//         if( sm1 != sm2 || mat1_ != mat2_ ) {
//            std::ostringstream oss;
//            oss << " Test: " << test_ << "\n"
//                << " Error: Assignment failed\n"
//                << " Details:\n"
//                << "   Result:\n" << sm1 << "\n"
//                << "   Expected result:\n" << sm2 << "\n";
//            throw std::runtime_error( oss.str() );
//         }
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major list assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major initializer list assignment (complete list)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      initializer_list< initializer_list<int> > list =
//         { {  1,  2,  3,  4,  5,  6,   7,   8 },
//           {  2,  4,  6,  8, 10, 12,  14,  16 },
//           {  3,  6,  9, 12, 15, 18,  21,  24 },
//           {  4,  8, 12, 16, 20, 24,  28,  32 },
//           {  5, 10, 15, 20, 25, 30,  35,  40 },
//           {  6, 12, 18, 24, 30, 36,  42,  48 },
//           {  7, 14, 21, 28, 35, 42,  49,  56 },
//           {  8, 16, 24, 32, 40, 48,  56,  64 },
//           {  9, 18, 27, 36, 45, 54,  63,  72 },
//           { 10, 20, 30, 40, 50, 60,  70,  80 },
//           { 11, 22, 33, 44, 55, 66,  77,  88 },
//           { 12, 24, 36, 48, 60, 72,  84,  96 },
//           { 13, 26, 39, 52, 65, 78,  91, 104 },
//           { 14, 28, 42, 56, 70, 84,  98, 112 },
//           { 15, 30, 45, 60, 75, 90, 105, 120 },
//           { 16, 32, 48, 64, 80, 96, 112, 128 } };
//
//      sm1 = list;
//      sm2 = list;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major initializer list assignment (incomplete list)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      initializer_list< initializer_list<int> > list =
//         { {  1,  2,  3,  4,  5,  6,   7,   8 },
//           {  2,  4,  6,  8, 10, 12,  14 },
//           {  3,  6,  9, 12, 15, 18 },
//           {  4,  8, 12, 16, 20 },
//           {  5, 10, 15, 20 },
//           {  6, 12, 18 },
//           {  7, 14 },
//           {  8 },
//           {  9, 18, 27, 36, 45, 54,  63,  72 },
//           { 10, 20, 30, 40, 50, 60,  70 },
//           { 11, 22, 33, 44, 55, 66 },
//           { 12, 24, 36, 48, 60 },
//           { 13, 26, 39, 52 },
//           { 14, 28, 42 },
//           { 15, 30 },
//           { 16 } };
//
//      sm1 = list;
//      sm2 = list;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major copy assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubmatrix copy assignment (no aliasing)";
//
//      initialize();
//
//      OMT mat1( 64UL, 64UL );
//      OMT mat2( 64UL, 64UL );
//      randomize( mat1, int(randmin), int(randmax) );
//      mat2 = mat1;
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//      sm1 = dilatedsubmatrix<aligned>  ( mat1, 16UL, 8UL, 16UL, 8UL );
//      sm2 = dilatedsubmatrix<unaligned>( mat2, 16UL, 8UL, 16UL, 8UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major dilatedsubmatrix copy assignment (aliasing)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//      sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 12UL, 16UL, 8UL );
//      sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 12UL, 16UL, 8UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major dense matrix assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major dense matrix assignment (mixed type)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::DynamicMatrix<short,rowMajor> mat( 16UL, 8UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/row-major dense matrix assignment (aligned/padded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded mat( memory.get(), 16UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/row-major dense matrix assignment (unaligned/unpadded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 16UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix assignment (mixed type)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::DynamicMatrix<short,columnMajor> mat( 16UL, 8UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix assignment (aligned/padded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded mat( memory.get(), 16UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix assignment (unaligned/unpadded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 16UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major sparse matrix assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major sparse matrix assignment";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::CompressedMatrix<int,rowMajor> mat( 16UL, 8UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major sparse matrix assignment";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::CompressedMatrix<int,columnMajor> mat( 16UL, 8UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 = mat;
//      sm2 = mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
}
//*************************************************************************************************


////*************************************************************************************************
///*!\brief Test of the dilatedsubmatrix addition assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the addition assignment operators of the dilatedsubmatrix
//// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseTest::testAddAssign()
//{
//   using blaze::dilatedsubmatrix;
//   using blaze::padded;
//   using blaze::unpadded;
//   using blaze::rowMajor;
//   using blaze::columnMajor;
//
//
//   //=====================================================================================
//   // Row-major dilatedsubmatrix addition assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major dilatedsubmatrix addition assignment (no aliasing)";
//
//      initialize();
//
//      MT mat1( 64UL, 64UL );
//      MT mat2( 64UL, 64UL );
//      randomize( mat1, int(randmin), int(randmax) );
//      mat2 = mat1;
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//      sm1 += dilatedsubmatrix<aligned>  ( mat1, 8UL, 16UL, 8UL, 16UL );
//      sm2 += dilatedsubmatrix<unaligned>( mat2, 8UL, 16UL, 8UL, 16UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major dilatedsubmatrix addition assignment (aliasing)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//      sm1 += dilatedsubmatrix<aligned>  ( mat1_, 12UL, 16UL, 8UL, 16UL );
//      sm2 += dilatedsubmatrix<unaligned>( mat2_, 12UL, 16UL, 8UL, 16UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Row-major dense matrix addition assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major dense matrix addition assignment (mixed type)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::DynamicMatrix<short,rowMajor> mat( 8UL, 16UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/row-major dense matrix addition assignment (aligned/padded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded mat( memory.get(), 8UL, 16UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/row-major dense matrix addition assignment (unaligned/unpadded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix addition assignment (mixed type)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::DynamicMatrix<short,columnMajor> mat( 8UL, 16UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix addition assignment (aligned/padded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded mat( memory.get(), 8UL, 16UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix addition assignment (unaligned/unpadded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Row-major sparse matrix addition assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major sparse matrix addition assignment";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::CompressedMatrix<int,rowMajor> mat( 8UL, 16UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major sparse matrix addition assignment";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::CompressedMatrix<int,columnMajor> mat( 8UL, 16UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major dilatedsubmatrix addition assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubmatrix addition assignment (no aliasing)";
//
//      initialize();
//
//      OMT mat1( 64UL, 64UL );
//      OMT mat2( 64UL, 64UL );
//      randomize( mat1, int(randmin), int(randmax) );
//      mat2 = mat1;
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//      sm1 += dilatedsubmatrix<aligned>  ( mat1, 16UL, 8UL, 16UL, 8UL );
//      sm2 += dilatedsubmatrix<unaligned>( mat2, 16UL, 8UL, 16UL, 8UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major dilatedsubmatrix addition assignment (aliasing)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//      sm1 += dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 12UL, 16UL, 8UL );
//      sm2 += dilatedsubmatrix<unaligned>( tmat2_, 16UL, 12UL, 16UL, 8UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major dense matrix addition assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major dense matrix addition assignment (mixed type)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::DynamicMatrix<short,rowMajor> mat( 16UL, 8UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/row-major dense matrix addition assignment (aligned/padded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded mat( memory.get(), 16UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/row-major dense matrix addition assignment (unaligned/unpadded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 16UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix addition assignment (mixed type)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::DynamicMatrix<short,columnMajor> mat( 16UL, 8UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix addition assignment (aligned/padded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded mat( memory.get(), 16UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix addition assignment (unaligned/unpadded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 16UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major sparse matrix addition assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major sparse matrix addition assignment";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::CompressedMatrix<int,rowMajor> mat( 16UL, 8UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major sparse matrix addition assignment";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::CompressedMatrix<int,columnMajor> mat( 16UL, 8UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 += mat;
//      sm2 += mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Addition assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the dilatedsubmatrix subtraction assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the subtraction assignment operators of the dilatedsubmatrix
//// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseTest::testSubAssign()
//{
//   using blaze::dilatedsubmatrix;
//
//
//   using blaze::padded;
//   using blaze::unpadded;
//   using blaze::rowMajor;
//   using blaze::columnMajor;
//
//
//   //=====================================================================================
//   // Row-major dilatedsubmatrix subtraction assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major dilatedsubmatrix subtraction assignment (no aliasing)";
//
//      initialize();
//
//      MT mat1( 64UL, 64UL );
//      MT mat2( 64UL, 64UL );
//      randomize( mat1, int(randmin), int(randmax) );
//      mat2 = mat1;
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//      sm1 -= dilatedsubmatrix<aligned>  ( mat1, 8UL, 16UL, 8UL, 16UL );
//      sm2 -= dilatedsubmatrix<unaligned>( mat2, 8UL, 16UL, 8UL, 16UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major dilatedsubmatrix subtraction assignment (aliasing)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//      sm1 -= dilatedsubmatrix<aligned>  ( mat1_, 12UL, 16UL, 8UL, 16UL );
//      sm2 -= dilatedsubmatrix<unaligned>( mat2_, 12UL, 16UL, 8UL, 16UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Row-major dense matrix subtraction assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major dense matrix subtraction assignment (mixed type)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::DynamicMatrix<short,rowMajor> mat( 8UL, 16UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/row-major dense matrix subtraction assignment (aligned/padded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded mat( memory.get(), 8UL, 16UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/row-major dense matrix subtraction assignment (unaligned/unpadded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix subtraction assignment (mixed type)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::DynamicMatrix<short,columnMajor> mat( 8UL, 16UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix subtraction assignment (aligned/padded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded mat( memory.get(), 8UL, 16UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix subtraction assignment (unaligned/unpadded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Row-major sparse matrix subtraction assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major sparse matrix subtraction assignment";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::CompressedMatrix<int,rowMajor> mat( 8UL, 16UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major sparse matrix subtraction assignment";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::CompressedMatrix<int,columnMajor> mat( 8UL, 16UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major dilatedsubmatrix subtraction assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubmatrix subtraction assignment (no aliasing)";
//
//      initialize();
//
//      OMT mat1( 64UL, 64UL );
//      OMT mat2( 64UL, 64UL );
//      randomize( mat1, int(randmin), int(randmax) );
//      mat2 = mat1;
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//      sm1 -= dilatedsubmatrix<aligned>  ( mat1, 16UL, 8UL, 16UL, 8UL );
//      sm2 -= dilatedsubmatrix<unaligned>( mat2, 16UL, 8UL, 16UL, 8UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major dilatedsubmatrix subtraction assignment (aliasing)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//      sm1 -= dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 12UL, 16UL, 8UL );
//      sm2 -= dilatedsubmatrix<unaligned>( tmat2_, 16UL, 12UL, 16UL, 8UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major dense matrix subtraction assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major dense matrix subtraction assignment (mixed type)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::DynamicMatrix<short,rowMajor> mat( 16UL, 8UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/row-major dense matrix subtraction assignment (aligned/padded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded mat( memory.get(), 16UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/row-major dense matrix subtraction assignment (unaligned/unpadded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 16UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix subtraction assignment (mixed type)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::DynamicMatrix<short,columnMajor> mat( 16UL, 8UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix subtraction assignment (aligned/padded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded mat( memory.get(), 16UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix subtraction assignment (unaligned/unpadded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 16UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major sparse matrix subtraction assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major sparse matrix subtraction assignment";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::CompressedMatrix<int,rowMajor> mat( 16UL, 8UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major sparse matrix subtraction assignment";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::CompressedMatrix<int,columnMajor> mat( 16UL, 8UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 -= mat;
//      sm2 -= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Subtraction assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the dilatedsubmatrix Schur product assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the Schur product assignment operators of the dilatedsubmatrix
//// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseTest::testSchurAssign()
//{
//   using blaze::dilatedsubmatrix;
//
//
//   using blaze::padded;
//   using blaze::unpadded;
//   using blaze::rowMajor;
//   using blaze::columnMajor;
//
//
//   //=====================================================================================
//   // Row-major dilatedsubmatrix Schur product assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major dilatedsubmatrix Schur product assignment (no aliasing)";
//
//      initialize();
//
//      MT mat1( 64UL, 64UL );
//      MT mat2( 64UL, 64UL );
//      randomize( mat1, int(randmin), int(randmax) );
//      mat2 = mat1;
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//      sm1 %= dilatedsubmatrix<aligned>  ( mat1, 8UL, 16UL, 8UL, 16UL );
//      sm2 %= dilatedsubmatrix<unaligned>( mat2, 8UL, 16UL, 8UL, 16UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major dilatedsubmatrix Schur product assignment (aliasing)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//      sm1 %= dilatedsubmatrix<aligned>  ( mat1_, 12UL, 16UL, 8UL, 16UL );
//      sm2 %= dilatedsubmatrix<unaligned>( mat2_, 12UL, 16UL, 8UL, 16UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Row-major dense matrix Schur product assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major dense matrix Schur product assignment (mixed type)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::DynamicMatrix<short,rowMajor> mat( 8UL, 16UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/row-major dense matrix Schur product assignment (aligned/padded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded mat( memory.get(), 8UL, 16UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/row-major dense matrix Schur product assignment (unaligned/unpadded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix Schur product assignment (mixed type)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::DynamicMatrix<short,columnMajor> mat( 8UL, 16UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix Schur product assignment (aligned/padded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded mat( memory.get(), 8UL, 16UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix Schur product assignment (unaligned/unpadded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Row-major sparse matrix Schur product assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major sparse matrix Schur product assignment";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::CompressedMatrix<int,rowMajor> mat( 8UL, 16UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major sparse matrix Schur product assignment";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//
//      blaze::CompressedMatrix<int,columnMajor> mat( 8UL, 16UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major dilatedsubmatrix Schur product assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubmatrix Schur product assignment (no aliasing)";
//
//      initialize();
//
//      OMT mat1( 64UL, 64UL );
//      OMT mat2( 64UL, 64UL );
//      randomize( mat1, int(randmin), int(randmax) );
//      mat2 = mat1;
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//      sm1 %= dilatedsubmatrix<aligned>  ( mat1, 16UL, 8UL, 16UL, 8UL );
//      sm2 %= dilatedsubmatrix<unaligned>( mat2, 16UL, 8UL, 16UL, 8UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major dilatedsubmatrix Schur product assignment (aliasing)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//      sm1 %= dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 12UL, 16UL, 8UL );
//      sm2 %= dilatedsubmatrix<unaligned>( tmat2_, 16UL, 12UL, 16UL, 8UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major dense matrix Schur product assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major dense matrix Schur product assignment (mixed type)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::DynamicMatrix<short,rowMajor> mat( 16UL, 8UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/row-major dense matrix Schur product assignment (aligned/padded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded mat( memory.get(), 16UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/row-major dense matrix Schur product assignment (unaligned/unpadded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 16UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix Schur product assignment (mixed type)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::DynamicMatrix<short,columnMajor> mat( 16UL, 8UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix Schur product assignment (aligned/padded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded mat( memory.get(), 16UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix Schur product assignment (unaligned/unpadded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 16UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major sparse matrix Schur product assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major sparse matrix Schur product assignment";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::CompressedMatrix<int,rowMajor> mat( 16UL, 8UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major sparse matrix Schur product assignment";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 8UL, 16UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 8UL, 16UL, 8UL );
//
//      blaze::CompressedMatrix<int,columnMajor> mat( 16UL, 8UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 %= mat;
//      sm2 %= mat;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Schur product assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//}
////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the dilatedsubmatrix multiplication assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the multiplication assignment operators of the dilatedsubmatrix
//// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseTest::testMultAssign()
//{
//   using blaze::dilatedsubmatrix;
//
//
//   using blaze::padded;
//   using blaze::unpadded;
//   using blaze::rowMajor;
//   using blaze::columnMajor;
//
//
//   //=====================================================================================
//   // Row-major dilatedsubmatrix multiplication assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major dilatedsubmatrix multiplication assignment (no aliasing)";
//
//      initialize();
//
//      MT mat1( 64UL, 64UL );
//      MT mat2( 64UL, 64UL );
//      randomize( mat1, int(randmin), int(randmax) );
//      mat2 = mat1;
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 16UL, 16UL, 8UL, 8UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 16UL, 16UL, 8UL, 8UL );
//      sm1 *= dilatedsubmatrix<aligned>  ( mat1, 16UL, 16UL, 8UL, 8UL );
//      sm2 *= dilatedsubmatrix<unaligned>( mat2, 16UL, 16UL, 8UL, 8UL );
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major dilatedsubmatrix multiplication assignment (aliasing)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 16UL, 16UL, 8UL, 8UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 16UL, 16UL, 8UL, 8UL );
//      sm1 *= dilatedsubmatrix<aligned>  ( mat1_, 24UL, 16UL, 8UL, 8UL );
//      sm2 *= dilatedsubmatrix<unaligned>( mat2_, 24UL, 16UL, 8UL, 8UL );
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Row-major dense matrix multiplication assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major dense matrix multiplication assignment (mixed type)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 16UL, 16UL, 8UL, 8UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 16UL, 16UL, 8UL, 8UL );
//
//      blaze::DynamicMatrix<short,rowMajor> mat( 8UL, 8UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/row-major dense matrix multiplication assignment (aligned/padded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 16UL, 16UL, 8UL, 8UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 16UL, 16UL, 8UL, 8UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded mat( memory.get(), 8UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/row-major dense matrix multiplication assignment (unaligned/unpadded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 16UL, 16UL, 8UL, 8UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 16UL, 16UL, 8UL, 8UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[65UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 8UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix multiplication assignment (mixed type)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 16UL, 16UL, 8UL, 8UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 16UL, 16UL, 8UL, 8UL );
//
//      blaze::DynamicMatrix<short,columnMajor> mat( 8UL, 8UL );
//      randomize( mat, short(randmin), short(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix multiplication assignment (aligned/padded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 16UL, 16UL, 8UL, 8UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 16UL, 16UL, 8UL, 8UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded mat( memory.get(), 8UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major dense matrix multiplication assignment (unaligned/unpadded)";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 16UL, 16UL, 8UL, 8UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 16UL, 16UL, 8UL, 8UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[65UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 8UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Row-major sparse matrix multiplication assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major sparse matrix multiplication assignment";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 16UL, 16UL, 8UL, 8UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 16UL, 16UL, 8UL, 8UL );
//
//      blaze::CompressedMatrix<int,rowMajor> mat( 8UL, 8UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Row-major/column-major sparse matrix multiplication assignment";
//
//      initialize();
//
//      ASMT sm1 = dilatedsubmatrix<aligned>  ( mat1_, 16UL, 16UL, 8UL, 8UL );
//      USMT sm2 = dilatedsubmatrix<unaligned>( mat2_, 16UL, 16UL, 8UL, 8UL );
//
//      blaze::CompressedMatrix<int,columnMajor> mat( 8UL, 8UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major dilatedsubmatrix multiplication assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubmatrix multiplication assignment (no aliasing)";
//
//      initialize();
//
//      OMT mat1( 64UL, 64UL );
//      OMT mat2( 64UL, 64UL );
//      randomize( mat1, int(randmin), int(randmax) );
//      mat2 = mat1;
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 16UL, 8UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 16UL, 8UL, 8UL );
//      sm1 *= dilatedsubmatrix<aligned>  ( mat1, 16UL, 16UL, 8UL, 8UL );
//      sm2 *= dilatedsubmatrix<unaligned>( mat2, 16UL, 16UL, 8UL, 8UL );
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major dilatedsubmatrix multiplication assignment (aliasing)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 16UL, 8UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 16UL, 8UL, 8UL );
//      sm1 *= dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 24UL, 8UL, 8UL );
//      sm2 *= dilatedsubmatrix<unaligned>( tmat2_, 16UL, 24UL, 8UL, 8UL );
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major dense matrix multiplication assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major dense matrix multiplication assignment (mixed type)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 16UL, 8UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 16UL, 8UL, 8UL );
//
//      blaze::DynamicMatrix<short,rowMajor> mat( 8UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/row-major dense matrix multiplication assignment (aligned/padded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 16UL, 8UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 16UL, 8UL, 8UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded mat( memory.get(), 8UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/row-major dense matrix multiplication assignment (unaligned/unpadded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 16UL, 8UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 16UL, 8UL, 8UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[65UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 8UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix multiplication assignment (mixed type)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 16UL, 8UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 16UL, 8UL, 8UL );
//
//      blaze::DynamicMatrix<short,columnMajor> mat( 8UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix multiplication assignment (aligned/padded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 16UL, 8UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 16UL, 8UL, 8UL );
//
//      using AlignedPadded = blaze::CustomMatrix<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded mat( memory.get(), 8UL, 8UL, 16UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major dense matrix multiplication assignment (unaligned/unpadded)";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 16UL, 8UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 16UL, 8UL, 8UL );
//
//      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[65UL] );
//      UnalignedUnpadded mat( memory.get()+1UL, 8UL, 8UL );
//      randomize( mat, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major sparse matrix multiplication assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major sparse matrix multiplication assignment";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 16UL, 8UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 16UL, 8UL, 8UL );
//
//      blaze::CompressedMatrix<int,rowMajor> mat( 8UL, 8UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major/column-major sparse matrix multiplication assignment";
//
//      initialize();
//
//      AOSMT sm1 = dilatedsubmatrix<aligned>  ( tmat1_, 16UL, 16UL, 8UL, 8UL );
//      UOSMT sm2 = dilatedsubmatrix<unaligned>( tmat2_, 16UL, 16UL, 8UL, 8UL );
//
//      blaze::CompressedMatrix<int,columnMajor> mat( 8UL, 8UL );
//      randomize( mat, 30UL, int(randmin), int(randmax) );
//
//      sm1 *= mat;
//      sm2 *= mat;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Multiplication assignment failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//}
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
void DenseTest::initialize()
{
   // Initializing the row-major dynamic matrices
   randomize( mat1_, int(randmin), int(randmax) );
   mat2_ = mat1_;

   // Initializing the column-major dynamic matrices
   randomize( tmat1_, int(randmin), int(randmax) );
   tmat2_ = tmat1_;
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Create dilated sequence of elements.
//
// \return elements
//
// This function returns a sequence of element indices.
*/
std::vector<size_t> DenseTest::generate_indices(size_t offset, size_t n, size_t dilation)
{
   std::vector<size_t> indices;
   for( size_t i = 0; i != n; ++i )
   {
      indices.push_back( offset + i * dilation );
   }
   return indices;
}
//*************************************************************************************************

} // namespace dilatedsubmatrix

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
   std::cout << "   Running dilatedsubmatrix dense test ..." << std::endl;

   try
   {
      RUN_DILATEDSUBMATRIX_DENSE_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during dilatedsubmatrix dense test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
