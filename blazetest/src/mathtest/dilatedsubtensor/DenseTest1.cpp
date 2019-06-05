//=================================================================================================
/*!
//  \file src/mathtest/dilatedsubtensor/DenseTest1.cpp
//  \brief Source file for the dilatedsubtensor dense test
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
#include <memory>

//#include <blaze/math/CompressedTensor.h>

#include <blaze/util/Memory.h>
#include <blaze/util/policies/Deallocate.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>

#include <blaze_tensor/math/CustomTensor.h>
#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/Views.h>
#include <blazetest/mathtest/dilatedsubtensor/DenseTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace dilatedsubtensor {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the dilatedsubtensor dense aligned test.
//
// \exception std::runtime_error Operation error detected.
*/
DenseTest::DenseTest()
   : tens1_ ( 64UL, 64UL, 64UL )
   , tens2_ ( 64UL, 64UL, 64UL )
{
   //testConstructors();
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
/*!\brief Test of the dilatedsubtensor constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the dilatedsubtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
//void DenseTest::testConstructors()
//{
//   using blaze::dilatedsubtensor;
//
//
//   //=====================================================================================
//   // Row-major dilatedsubtensor tests
//   //=====================================================================================
//
//   {
//      test_ = "Row-major dilatedsubtensor constructor";
//
//      initialize();
//
//      const size_t alignment = blaze::AlignmentOf<int>::value;
//
//      for( size_t row=0UL; row<tens1_.rows(); row+=alignment ) {
//         for( size_t column=0UL; column<tens1_.columns(); column+=alignment ) {
//            for( size_t maxm=0UL; ; maxm+=alignment ) {
//               for( size_t maxn=0UL; ; maxn+=alignment )
//               {
//                  size_t m( blaze::min( maxm, tens1_.rows()-row ) );
//                  size_t n( blaze::min( maxn, tens1_.columns()-column ) );
//
//                  for (size_t rowdilation = 1UL; rowdilation < maxm; ++rowdilation)
//                  {
//                     for (size_t columndilation = 1UL; columndilation < maxn; ++columndilation)
//                     {
//                        while( row + (m - 1) * rowdilation >= tens1_.rows() ) --m;
//                        while( column + (n - 1) * columndilation >= tens1_.columns() ) --n;
//                        auto row_indices = generate_indices( row, m, rowdilation );
//                        auto column_indices = generate_indices( column, n, columndilation );
//                        const RCMT sm1 = blaze::rows(
//                           blaze::columns( tens1_, column_indices.data(),
//                              column_indices.size() ),
//                           row_indices.data(), row_indices.size() );
//                        const DSMT sm2 = dilatedsubtensor(tens2_, row, column, m, n, rowdilation, columndilation);
//
//                        if (sm1 != sm2) {
//                           std::ostringstream oss;
//                           oss << " Test: " << test_ << "\n"
//                              << " Error: Setup of dense dilatedsubtensor failed\n"
//                              << " Details:\n"
//                              << "   Index of first row    = " << row << "\n"
//                              << "   Index of first column = " << column << "\n"
//                              << "   Number of rows        = " << m << "\n"
//                              << "   Number of columns     = " << n << "\n"
//                              << "   dilatedsubtensor:\n" << sm1 << "\n"
//                              << "   Reference:\n" << sm2 << "\n";
//                           throw std::runtime_error(oss.str());
//                        }
//                     }
//                  }
//                  if( column+maxn > tens1_.columns() ) break;
//               }
//
//               if( row+maxm > tens1_.rows() ) break;
//            }
//         }
//      }
//   }
//
//
//   //=====================================================================================
//   // Column-major dilatedsubtensor tests
//   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubtensor constructor";
//
//      initialize();
//
//      const size_t alignment = blaze::AlignmentOf<int>::value;
//
//      for( size_t column=0UL; column<tens1_.columns(); column+=alignment ) {
//         for( size_t row=0UL; row<tens1_.rows(); row+=alignment ) {
//            for( size_t maxn=0UL; ; maxn+=alignment ) {
//               for( size_t maxm=0UL; ; maxm+=alignment )
//               {
//                  size_t n( blaze::min( maxn, tens1_.columns()-column ) );
//                  size_t m( blaze::min( maxm, tens1_.rows()-row ) );
//
//                  for (size_t columndilation = 1UL; columndilation < maxn; ++columndilation)
//                  {
//                     for (size_t rowdilation = 1UL; rowdilation < maxm; ++rowdilation)
//                     {
//                        while( column + (n - 1) * columndilation >= tens1_.columns() ) --n;
//                        while( row + (m - 1) * rowdilation >= tens1_.rows() ) --m;
//                        auto column_indices = generate_indices( column, n, columndilation );
//                        auto row_indices = generate_indices( row, m, rowdilation );
//                        const OCRMT sm1 = blaze::columns(
//                           blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//                           column_indices.data(), column_indices.size() );
//                        const ODSMT sm2 = dilatedsubtensor(ttens2_, row, column, m, n, rowdilation, columndilation);
//
//                        if( sm1 != sm2 ) {
//                           std::ostringstream oss;
//                           oss << " Test: " << test_ << "\n"
//                               << " Error: Setup of dense dilatedsubtensor failed\n"
//                               << " Details:\n"
//                               << "   Index of first row    = " << row << "\n"
//                               << "   Index of first column = " << column << "\n"
//                               << "   Number of rows        = " << m << "\n"
//                               << "   Number of columns     = " << n << "\n"
//                               << "   dilatedsubtensor:\n" << sm1 << "\n"
//                               << "   Reference:\n" << sm2 << "\n";
//                           throw std::runtime_error( oss.str() );
//                        }
//                     }
//                  }
//                  if( row+maxm > tens1_.rows() ) break;
//               }
//               if( column+maxn > tens1_.columns() ) break;
//            }
//         }
//      }
//   }
//}
//////*************************************************************************************************
//
//
////*************************************************************************************************
///*!\brief Test of the dilatedsubtensor assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of all assignment operators of the dilatedsubtensor specialization.
//// In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseTest::testAssignment()
//{
//   using blaze::dilatedsubtensor;
//   using blaze::padded;
//   using blaze::unpadded;
//   using blaze::rowMajor;
//   using blaze::columnMajor;
//   using blaze::initializer_list;
//
//
//   //=====================================================================================
//   // Row-major homogeneous assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major dilatedsubtensor homogeneous assignment";
//
//      initialize();
//
//      // Assigning to a 8x4 dilatedsubtensor with a 2x3 dilation
//      {
//         auto row_indices = generate_indices( 8UL, 8UL, 2UL );
//         auto column_indices = generate_indices( 16UL, 4UL, 3UL );
//
//         RCMT sm1 = blaze::rows( blaze::columns( tens1_, column_indices.data(),
//                                    column_indices.size() ),
//            row_indices.data(), row_indices.size() );
//         DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//         sm1 = 12;
//         sm2 = 12;
//
//         checkRows   ( sm1,  8UL );
//         checkColumns( sm1,  4UL );
//         checkRows   ( sm2,  8UL );
//         checkColumns( sm2,  4UL );
//
//         if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      // Assigning to a 16x8 dilatedsubtensor
//      {
//         auto row_indices = generate_indices( 8UL, 16UL, 3UL );
//         auto column_indices = generate_indices( 16UL, 8UL, 2UL );
//         RCMT sm1 = blaze::rows( blaze::columns( tens1_, column_indices.data(),
//                                    column_indices.size() ),
//            row_indices.data(), row_indices.size() );
//         DSMT sm2 = dilatedsubtensor  ( tens2_, 8UL, 16UL, 16UL, 8UL, 3UL, 2UL );
//
//         sm1 = 15;
//         sm2 = 15;
//
//         checkRows   ( sm1, 16UL );
//         checkColumns( sm1,  8UL );
//         checkRows   ( sm2, 16UL );
//         checkColumns( sm2,  8UL );
//
//         if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   //=====================================================================================
//   // Row-major list assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major initializer list assignment (complete list)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
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
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
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
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major dilatedsubtensor copy assignment (no aliasing)";
//
//      initialize();
//
//      MT tens1( 64UL, 64UL );
//      MT tens2( 64UL, 64UL );
//      randomize( tens1, int(randmin), int(randmax) );
//      tens2 = tens1;
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      sm1 = dilatedsubtensor( tens1, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//      sm2 = dilatedsubtensor( tens2, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major dilatedsubtensor copy assignment (aliasing)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//      sm1 = dilatedsubtensor( tens1_, 12UL, 16UL, 8UL, 16UL, 3UL, 2UL  );
//      sm2 = dilatedsubtensor( tens2_, 12UL, 16UL, 8UL, 16UL, 3UL, 2UL  );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   // Row-major dense tensor assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major dense tensor assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      blaze::DynamicTensor<short,rowMajor> tens( 8UL, 16UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 = tens;
//      sm2 = tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/row-major dense tensor assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,blaze::aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded tens( memory.get(), 8UL, 16UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 = tens;
//      sm2 = tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/row-major dense tensor assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,blaze::unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 = tens;
//      sm2 = tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      blaze::DynamicTensor<short,columnMajor> tens( 8UL, 16UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 = tens;
//      sm2 = tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,blaze::aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded tens( memory.get(), 8UL, 16UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 = tens;
//      sm2 = tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,blaze::unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 = tens;
//      sm2 = tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   //=====================================================================================
//   // Column-major homogeneous assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubtensor homogeneous assignment";
//
//      initialize();
//
//      // Assigning to a 8x16 dilatedsubtensor
//      {
//         auto row_indices = generate_indices( 8UL, 8UL, 2UL );
//         auto column_indices = generate_indices( 16UL, 4UL, 3UL );
//
//         OCRMT sm1 = blaze::columns(
//            blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//            column_indices.data(), column_indices.size() );
//         ODSMT sm2 = dilatedsubtensor( ttens2_, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//         sm1 = 12;
//         sm2 = 12;
//
//         checkRows   ( sm1, 8UL );
//         checkColumns( sm1, 4UL );
//         checkRows   ( sm2, 8UL );
//         checkColumns( sm2, 4UL );
//
//         if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      // Assigning to a 16x8 dilatedsubtensor
//      {
//         auto row_indices = generate_indices( 8UL, 8UL, 2UL );
//         auto column_indices = generate_indices( 16UL, 4UL, 3UL );
//
//         OCRMT sm1 = blaze::columns(
//            blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//            column_indices.data(), column_indices.size() );
//         ODSMT sm2 = dilatedsubtensor( ttens2_, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//         sm1 = 15;
//         sm2 = 15;
//
//         checkRows   ( sm1,  8UL );
//         checkColumns( sm1,  4UL );
//         checkRows   ( sm2,  8UL );
//         checkColumns( sm2,  4UL );
//
//         if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
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
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
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
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major dilatedsubtensor copy assignment (no aliasing)";
//
//      initialize();
//
//      OMT tens1( 64UL, 64UL );
//      OMT tens2( 64UL, 64UL );
//      randomize( tens1, int(randmin), int(randmax) );
//      tens2 = tens1;
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      sm1 = dilatedsubtensor( tens1, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//      sm2 = dilatedsubtensor( tens2, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major dilatedsubtensor copy assignment (aliasing)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      sm1 = dilatedsubtensor( ttens1_, 16UL, 12UL, 16UL, 8UL, 2UL, 3UL );
//      sm2 = dilatedsubtensor( ttens2_, 16UL, 12UL, 16UL, 8UL, 2UL, 3UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   // Column-major dense tensor assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major dense tensor assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,rowMajor> tens( 16UL, 8UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 = tens;
//      sm2 = tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/row-major dense tensor assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,blaze::aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded tens( memory.get(), 16UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 = tens;
//      sm2 = tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/row-major dense tensor assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,blaze::unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 16UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 = tens;
//      sm2 = tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,columnMajor> tens( 16UL, 8UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 = tens;
//      sm2 = tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,blaze::aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded tens( memory.get(), 16UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 = tens;
//      sm2 = tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,blaze::unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 16UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 = tens;
//      sm2 = tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Assignment failed\n"
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
///*!\brief Test of the dilatedsubtensor addition assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the addition assignment operators of the dilatedsubtensor
//// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseTest::testAddAssign()
//{
//   using blaze::dilatedsubtensor;
//   using blaze::padded;
//   using blaze::unpadded;
//   using blaze::rowMajor;
//   using blaze::columnMajor;
//
//
//   //=====================================================================================
//   // Row-major dilatedsubtensor addition assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major dilatedsubtensor addition assignment (no aliasing)";
//
//      initialize();
//
//      MT tens1( 64UL, 64UL );
//      MT tens2( 64UL, 64UL );
//      randomize( tens1, int(randmin), int(randmax) );
//      tens2 = tens1;
//
//      auto row_indices = generate_indices( 8UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 4UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//      sm1 += dilatedsubtensor( tens1, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//      sm2 += dilatedsubtensor( tens2, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 4UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 4UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major dilatedsubtensor addition assignment (aliasing)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 4UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//      sm1 += dilatedsubtensor( tens1_, 12UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//      sm2 += dilatedsubtensor( tens2_, 12UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1,  4UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2,  4UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   // Row-major dense tensor addition assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major dense tensor addition assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 4UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 16UL, 4UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,rowMajor> tens( 16UL, 4UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 += tens;
//      sm2 += tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  4UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  4UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/row-major dense tensor addition assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 16UL, 3UL );
//      auto column_indices = generate_indices( 8UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 8UL, 16UL, 16UL, 3UL, 2UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,blaze::aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded tens( memory.get(), 16UL, 16UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 += tens;
//      sm2 += tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/row-major dense tensor addition assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 16UL, 3UL );
//      auto column_indices = generate_indices( 8UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 8UL, 16UL, 16UL, 3UL, 2UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,blaze::unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[257UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 16UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 += tens;
//      sm2 += tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor addition assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      blaze::DynamicTensor<short,columnMajor> tens( 8UL, 16UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 += tens;
//      sm2 += tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor addition assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,blaze::aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded tens( memory.get(), 8UL, 16UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 += tens;
//      sm2 += tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor addition assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,blaze::unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 += tens;
//      sm2 += tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   //=====================================================================================
//   // Column-major dilatedsubtensor addition assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubtensor addition assignment (no aliasing)";
//
//      initialize();
//
//      OMT tens1( 64UL, 64UL );
//      OMT tens2( 64UL, 64UL );
//      randomize( tens1, int(randmin), int(randmax) );
//      tens2 = tens1;
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      sm1 += dilatedsubtensor( tens1, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//      sm2 += dilatedsubtensor( tens2, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major dilatedsubtensor addition assignment (aliasing)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      sm1 += dilatedsubtensor( ttens1_, 16UL, 12UL, 16UL, 8UL, 2UL, 3UL );
//      sm2 += dilatedsubtensor( ttens2_, 16UL, 12UL, 16UL, 8UL, 2UL, 3UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   // Column-major dense tensor addition assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major dense tensor addition assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,rowMajor> tens( 16UL, 8UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 += tens;
//      sm2 += tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/row-major dense tensor addition assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,blaze::aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded tens( memory.get(), 16UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 += tens;
//      sm2 += tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/row-major dense tensor addition assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,blaze::unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 16UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 += tens;
//      sm2 += tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor addition assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,columnMajor> tens( 16UL, 8UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 += tens;
//      sm2 += tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor addition assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,blaze::aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded tens( memory.get(), 16UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 += tens;
//      sm2 += tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor addition assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,blaze::unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 16UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 += tens;
//      sm2 += tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
///*!\brief Test of the dilatedsubtensor subtraction assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the subtraction assignment operators of the dilatedsubtensor
//// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseTest::testSubAssign()
//{
//   using blaze::dilatedsubtensor;
//
//
//   using blaze::padded;
//   using blaze::unpadded;
//   using blaze::rowMajor;
//   using blaze::columnMajor;
//
//
//   //=====================================================================================
//   // Row-major dilatedsubtensor subtraction assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major dilatedsubtensor subtraction assignment (no aliasing)";
//
//      initialize();
//
//      MT tens1( 64UL, 64UL );
//      MT tens2( 64UL, 64UL );
//      randomize( tens1, int(randmin), int(randmax) );
//      tens2 = tens1;
//
//      auto row_indices = generate_indices( 8UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 4UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//      sm1 -= dilatedsubtensor( tens1, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//      sm2 -= dilatedsubtensor( tens2, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1,  4UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2,  4UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major dilatedsubtensor subtraction assignment (aliasing)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 4UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//      sm1 -= dilatedsubtensor( tens1_, 12UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//      sm2 -= dilatedsubtensor( tens2_, 12UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1,  4UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2,  4UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   // Row-major dense tensor subtraction assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major dense tensor subtraction assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 4UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 16UL, 4UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,rowMajor> tens( 16UL, 4UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 -= tens;
//      sm2 -= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  4UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  4UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/row-major dense tensor subtraction assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 16UL, 3UL );
//      auto column_indices = generate_indices( 8UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 8UL, 16UL, 16UL, 3UL, 2UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,blaze::aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded tens( memory.get(), 16UL, 16UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 -= tens;
//      sm2 -= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/row-major dense tensor subtraction assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 16UL, 3UL );
//      auto column_indices = generate_indices( 8UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 8UL, 16UL, 16UL, 3UL, 2UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,blaze::unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[257UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 16UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 -= tens;
//      sm2 -= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor subtraction assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      blaze::DynamicTensor<short,columnMajor> tens( 8UL, 16UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 -= tens;
//      sm2 -= tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor subtraction assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,blaze::aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded tens( memory.get(), 8UL, 16UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 -= tens;
//      sm2 -= tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor subtraction assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,blaze::unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 -= tens;
//      sm2 -= tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   //=====================================================================================
//   // Column-major dilatedsubtensor subtraction assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubtensor subtraction assignment (no aliasing)";
//
//      initialize();
//
//      OMT tens1( 64UL, 64UL );
//      OMT tens2( 64UL, 64UL );
//      randomize( tens1, int(randmin), int(randmax) );
//      tens2 = tens1;
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      sm1 -= dilatedsubtensor( tens1, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//      sm2 -= dilatedsubtensor( tens2, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major dilatedsubtensor subtraction assignment (aliasing)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      sm1 -= dilatedsubtensor( ttens1_, 16UL, 12UL, 16UL, 8UL, 2UL, 3UL );
//      sm2 -= dilatedsubtensor( ttens2_, 16UL, 12UL, 16UL, 8UL, 2UL, 3UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   // Column-major dense tensor subtraction assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major dense tensor subtraction assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,rowMajor> tens( 16UL, 8UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 -= tens;
//      sm2 -= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/row-major dense tensor subtraction assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,blaze::aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded tens( memory.get(), 16UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 -= tens;
//      sm2 -= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/row-major dense tensor subtraction assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,blaze::unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 16UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 -= tens;
//      sm2 -= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor subtraction assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,columnMajor> tens( 16UL, 8UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 -= tens;
//      sm2 -= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor subtraction assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,blaze::aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded tens( memory.get(), 16UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 -= tens;
//      sm2 -= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor subtraction assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,blaze::unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 16UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 -= tens;
//      sm2 -= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
///*!\brief Test of the dilatedsubtensor Schur product assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the Schur product assignment operators of the dilatedsubtensor
//// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseTest::testSchurAssign()
//{
//   using blaze::dilatedsubtensor;
//   using blaze::aligned;
//   using blaze::unaligned;
//   using blaze::padded;
//   using blaze::unpadded;
//   using blaze::rowMajor;
//   using blaze::columnMajor;
//
//
//   //=====================================================================================
//   // Row-major dilatedsubtensor Schur product assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major dilatedsubtensor Schur product assignment (no aliasing)";
//
//      initialize();
//
//      MT tens1( 64UL, 64UL );
//      MT tens2( 64UL, 64UL );
//      randomize( tens1, int(randmin), int(randmax) );
//      tens2 = tens1;
//
//      auto row_indices = generate_indices( 8UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 4UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//      sm1 %= dilatedsubtensor( tens1, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//      sm2 %= dilatedsubtensor( tens2, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1,  4UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2,  4UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major dilatedsubtensor Schur product assignment (aliasing)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 4UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//      sm1 %= dilatedsubtensor( tens1_, 12UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//      sm2 %= dilatedsubtensor( tens2_, 12UL, 16UL, 8UL, 4UL, 2UL, 3UL );
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1,  4UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2,  4UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   // Row-major dense tensor Schur product assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major dense tensor Schur product assignment (mixed type)";
//
//      initialize();
//      auto row_indices = generate_indices( 8UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 4UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 16UL, 4UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,rowMajor> tens( 16UL, 4UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 %= tens;
//      sm2 %= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  4UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  4UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/row-major dense tensor Schur product assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 16UL, 3UL );
//      auto column_indices = generate_indices( 8UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 8UL, 16UL, 16UL, 3UL, 2UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded tens( memory.get(), 16UL, 16UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 %= tens;
//      sm2 %= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/row-major dense tensor Schur product assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 16UL, 3UL );
//      auto column_indices = generate_indices( 8UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 8UL, 16UL, 16UL, 3UL, 2UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[257UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 16UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 %= tens;
//      sm2 %= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor Schur product assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      blaze::DynamicTensor<short,columnMajor> tens( 8UL, 16UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 %= tens;
//      sm2 %= tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor Schur product assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded tens( memory.get(), 8UL, 16UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 %= tens;
//      sm2 %= tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor Schur product assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
//      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 %= tens;
//      sm2 %= tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1, 16UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2, 16UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   //=====================================================================================
//   // Column-major dilatedsubtensor Schur product assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubtensor Schur product assignment (no aliasing)";
//
//      initialize();
//
//      OMT tens1( 64UL, 64UL );
//      OMT tens2( 64UL, 64UL );
//      randomize( tens1, int(randmin), int(randmax) );
//      tens2 = tens1;
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      sm1 %= dilatedsubtensor( tens1, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//      sm2 %= dilatedsubtensor( tens2, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major dilatedsubtensor Schur product assignment (aliasing)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      sm1 %= dilatedsubtensor( ttens1_, 16UL, 12UL, 16UL, 8UL, 2UL, 3UL );
//      sm2 %= dilatedsubtensor( ttens2_, 16UL, 12UL, 16UL, 8UL, 2UL, 3UL );
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   // Column-major dense tensor Schur product assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major dense tensor Schur product assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,rowMajor> tens( 16UL, 8UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 %= tens;
//      sm2 %= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/row-major dense tensor Schur product assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 256UL ) );
//      AlignedPadded tens( memory.get(), 16UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 %= tens;
//      sm2 %= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/row-major dense tensor Schur product assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 16UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 %= tens;
//      sm2 %= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor Schur product assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,columnMajor> tens( 16UL, 8UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 %= tens;
//      sm2 %= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor Schur product assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded tens( memory.get(), 16UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 %= tens;
//      sm2 %= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor Schur product assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[129UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 16UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 %= tens;
//      sm2 %= tens;
//
//      checkRows   ( sm1, 16UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2, 16UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
///*!\brief Test of the dilatedsubtensor multiplication assignment operators.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function performs a test of the multiplication assignment operators of the dilatedsubtensor
//// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
//*/
//void DenseTest::testMultAssign()
//{
//   using blaze::dilatedsubtensor;
//   using blaze::aligned;
//   using blaze::unaligned;
//   using blaze::padded;
//   using blaze::unpadded;
//   using blaze::rowMajor;
//   using blaze::columnMajor;
//
//
//   //=====================================================================================
//   // Row-major dilatedsubtensor multiplication assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major dilatedsubtensor multiplication assignment (no aliasing)";
//
//      initialize();
//
//      MT tens1( 64UL, 64UL );
//      MT tens2( 64UL, 64UL );
//      randomize( tens1, int(randmin), int(randmax) );
//      tens2 = tens1;
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 8UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//
//      sm1 *= dilatedsubtensor( tens1, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//      sm2 *= dilatedsubtensor( tens2, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major dilatedsubtensor multiplication assignment (aliasing)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 8UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//
//      sm1 *= dilatedsubtensor( tens1_, 24UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//      sm2 *= dilatedsubtensor( tens2_, 24UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   // Row-major dense tensor multiplication assignment
//   //=====================================================================================
//
//   {
//      test_ = "Row-major/row-major dense tensor multiplication assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 8UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,rowMajor> tens( 8UL, 8UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 *= tens;
//      sm2 *= tens;
//
//      checkRows   ( sm1,  8UL );
//      checkColumns( sm1,  8UL );
//      checkRows   ( sm2,  8UL );
//      checkColumns( sm2,  8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/row-major dense tensor multiplication assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 8UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded tens( memory.get(), 8UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 *= tens;
//      sm2 *= tens;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/row-major dense tensor multiplication assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 8UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[65UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 8UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 *= tens;
//      sm2 *= tens;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor multiplication assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 8UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,columnMajor> tens( 8UL, 8UL );
//      randomize( tens, short(randmin), short(randmax) );
//
//      sm1 *= tens;
//      sm2 *= tens;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor multiplication assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 8UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded tens( memory.get(), 8UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 *= tens;
//      sm2 *= tens;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Row-major/column-major dense tensor multiplication assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 16UL, 8UL, 3UL );
//
//      RCMT sm1 = blaze::rows(
//         blaze::columns( tens1_, column_indices.data(), column_indices.size() ),
//         row_indices.data(), row_indices.size() );
//      DSMT sm2 = dilatedsubtensor( tens2_, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[65UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 8UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 *= tens;
//      sm2 *= tens;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   //=====================================================================================
//   // Column-major dilatedsubtensor multiplication assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major dilatedsubtensor multiplication assignment (no aliasing)";
//
//      initialize();
//
//      OMT tens1( 64UL, 64UL );
//      OMT tens2( 64UL, 64UL );
//      randomize( tens1, int(randmin), int(randmax) );
//      tens2 = tens1;
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 8UL, 8UL, 2UL, 3UL );
//
//      sm1 *= dilatedsubtensor( tens1, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//      sm2 *= dilatedsubtensor( tens2, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major dilatedsubtensor multiplication assignment (aliasing)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 8UL, 8UL, 2UL, 3UL );
//
//      sm1 *= dilatedsubtensor( ttens1_, 16UL, 24UL, 8UL, 8UL, 1UL, 1UL );
//      sm2 *= dilatedsubtensor( ttens2_, 16UL, 24UL, 8UL, 8UL, 1UL, 1UL );
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//   // Column-major dense tensor multiplication assignment
//   //=====================================================================================
//
//   {
//      test_ = "Column-major/row-major dense tensor multiplication assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 8UL, 8UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,rowMajor> tens( 8UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 *= tens;
//      sm2 *= tens;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/row-major dense tensor multiplication assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 8UL, 8UL, 2UL, 3UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,aligned,padded,rowMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded tens( memory.get(), 8UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 *= tens;
//      sm2 *= tens;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/row-major dense tensor multiplication assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 8UL, 8UL, 2UL, 3UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,rowMajor>;
//      std::unique_ptr<int[]> memory( new int[65UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 8UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 *= tens;
//      sm2 *= tens;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor multiplication assignment (mixed type)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 8UL, 8UL, 2UL, 3UL );
//
//      blaze::DynamicTensor<short,columnMajor> tens( 8UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 *= tens;
//      sm2 *= tens;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor multiplication assignment (aligned/padded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 8UL, 8UL, 2UL, 3UL );
//
//      using AlignedPadded = blaze::CustomTensor<int,aligned,padded,columnMajor>;
//      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 128UL ) );
//      AlignedPadded tens( memory.get(), 8UL, 8UL, 16UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 *= tens;
//      sm2 *= tens;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
//      test_ = "Column-major/column-major dense tensor multiplication assignment (unaligned/unpadded)";
//
//      initialize();
//
//      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
//      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
//
//      OCRMT sm1 = blaze::columns(
//         blaze::rows( ttens1_, row_indices.data(), row_indices.size() ),
//         column_indices.data(), column_indices.size() );
//      ODSMT sm2 = dilatedsubtensor( ttens2_, 16UL, 8UL, 8UL, 8UL, 2UL, 3UL );
//
//      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,columnMajor>;
//      std::unique_ptr<int[]> memory( new int[65UL] );
//      UnalignedUnpadded tens( memory.get()+1UL, 8UL, 8UL );
//      randomize( tens, int(randmin), int(randmax) );
//
//      sm1 *= tens;
//      sm2 *= tens;
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || tens1_ != tens2_ ) {
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
////*************************************************************************************************
//
//
//
//
////=================================================================================================
////
////  UTILITY FUNCTIONS
////
////=================================================================================================
//
////*************************************************************************************************
///*!\brief Initialization of all member tensors.
////
//// \return void
//// \exception std::runtime_error Error detected.
////
//// This function initializes all member tensors to specific predetermined values.
//*/
//void DenseTest::initialize()
//{
//   // Initializing the row-major dynamic tensors
//   randomize( tens1_, int(randmin), int(randmax) );
//   tens2_ = tens1_;
//
//   // Initializing the column-major dynamic tensors
//   randomize( ttens1_, int(randmin), int(randmax) );
//   ttens2_ = ttens1_;
//}
////*************************************************************************************************
//
////*************************************************************************************************
///*!\brief Create dilated sequence of elements.
////
//// \return elements
////
//// This function returns a sequence of element indices.
//*/
//std::vector<size_t> DenseTest::generate_indices(size_t offset, size_t n, size_t dilation)
//{
//   std::vector<size_t> indices;
//   for( size_t i = 0; i != n; ++i )
//   {
//      indices.push_back( offset + i * dilation );
//   }
//   return indices;
//}
//*************************************************************************************************

} // namespace dilatedsubtensor

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
   std::cout << "   Running dilatedsubtensor dense test ..." << std::endl;

   try
   {
      RUN_DILATEDSUBTENSOR_DENSE_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during dilatedsubtensor dense test (part 1):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
