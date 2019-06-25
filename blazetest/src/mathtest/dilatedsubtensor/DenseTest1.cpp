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

#include <blaze/system/Platform.h>
#include <blaze/util/Memory.h>
#include <blaze/util/policies/Deallocate.h>
#include <blaze/util/Random.h>
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
   testConstructors();
   testAssignment();
   testAddAssign();
   testSubAssign();
   testSchurAssign();
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
void DenseTest::testConstructors()
{
   using blaze::dilatedsubtensor;
   using blaze::dilatedsubmatrix;
   using blaze::pageslice;

   //=====================================================================================
   // Row-major dilatedsubtensor tests
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubtensor constructor";

      initialize();

      const size_t alignment = blaze::AlignmentOf<int>::value;
      size_t page_ = blaze::rand<size_t>( 0, tens1_.pages()-1 );

      for( size_t row=0UL; row<tens1_.rows(); row+=alignment ) {
         for( size_t column=0UL; column<tens1_.columns(); column+=alignment ) {
            for( size_t maxm=0UL; ; maxm+=alignment ) {
               for( size_t maxn=0UL; ; maxn+=alignment )
               {

                  size_t m( blaze::min( maxm, tens1_.rows()-row ) );
                  size_t n( blaze::min( maxn, tens1_.columns()-column ) );

                  for (size_t rowdilation = 1UL; rowdilation < maxm; ++rowdilation)
                  {
                     for (size_t columndilation = 1UL; columndilation < maxn; ++columndilation)
                     {
                        while( row + (m - 1) * rowdilation >= tens1_.rows() ) --m;
                        while( column + (n - 1) * columndilation >= tens1_.columns() ) --n;

                        const DSTT st1 = dilatedsubtensor(tens1_, page_, row, column, 1UL, m, n,1UL, rowdilation, columndilation);
                        auto st2 = dilatedsubmatrix(blaze::pageslice(tens2_, page_), row, column, m, n, rowdilation, columndilation);

                        if (st1(0,m-1,n-1) != st2(m-1,n-1) ) {
                           std::ostringstream oss;
                           oss << " Test: " << test_ << "\n"
                              << " Error: Setup of dense dilatedsubtensor failed\n"
                              << " Details:\n"
                              << "   Index of the page    = " << page_ << "\n"
                              << "   Index of first row    = " << row << "\n"
                              << "   Index of first column = " << column << "\n"
                              << "   Number of rows        = " << m << "\n"
                              << "   Number of columns     = " << n << "\n"
                              << "   dilatedsubtensor:\n" << st1 << "\n"
                              << "   Reference:\n" << st2 << "\n";
                           throw std::runtime_error(oss.str());
                        }
                     }
                  }
                  if( column+maxn > tens1_.columns() ) break;
               }

               if( row+maxm > tens1_.rows() ) break;
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the dilatedsubtensor assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the dilatedsubtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testAssignment()
{
   using blaze::dilatedsubtensor;
   using blaze::pageslice;
   using blaze::initializer_list;


   //=====================================================================================
   // Row-major homogeneous assignment
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubtensor homogeneous assignment";

      initialize();

      // Assigning to a 8x8x4 dilatedsubtensor with a 2x2x3 dilation
      {
         DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 8UL, 4UL, 2UL, 2UL, 3UL);

         st1 = 12;

         checkPages  ( st1,  8UL );
         checkRows   ( st1,  8UL );
         checkColumns( st1,  4UL );
      }
   }

   //=====================================================================================
   // Row-major list assignment
   //=====================================================================================

   {
      test_ = "Row-major initializer list assignment (complete list)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 12UL, 4UL, 8UL, 12UL, 4UL, 3UL, 1UL);

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

      st1 = list;

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 12UL );

      if( st1(1,2,2) != 9 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << st1(2,2,2) << "\n"
             << "   Expected result:\n" << "9" << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major initializer list assignment (incomplete list)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 12UL, 4UL, 8UL, 12UL, 4UL, 3UL, 1UL);

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

      st1 = list;

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 12UL );

      if( st1(2,3,2) != 12 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << st1 << "\n"
             << "   Expected result:\n" << "12" << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major copy assignment
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubtensor copy assignment (no aliasing)";

      initialize();

      TT tens1( 64UL, 64UL, 64UL );
      TT tens2( 64UL, 64UL, 64UL );
      randomize( tens1, int(randmin), int(randmax) );
      tens2 = tens1;

      size_t page_ = blaze::rand<size_t>( 0, 3UL );
      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 16UL, 2UL, 2UL);

      st1 = dilatedsubtensor( tens1, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      st2 = pageslice(dilatedsubtensor( tens2,  4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL ),page_);

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 16UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 16UL );

      if( pageslice(st1,page_) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,page_) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major dilatedsubtensor copy assignment (aliasing)";

      initialize();

      size_t page_ = blaze::rand<size_t>( 0, 3UL );
      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 16UL, 2UL, 2UL);

      st1 = dilatedsubtensor( tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      st2 = pageslice(dilatedsubtensor( tens2_,  4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL ),page_);

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 16UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 16UL );

      if( pageslice(st1,page_) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,page_) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dense tensor assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major dense tensor assignment (mixed type)";

      initialize();

      size_t page_ = blaze::rand<size_t>( 0, 3UL );

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 12UL, 4UL, 8UL, 12UL, 4UL, 3UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 12UL, 8UL, 12UL, 3UL, 2UL);

      blaze::DynamicTensor<short> tens( 4UL, 8UL, 12UL );
      randomize( tens, short(randmin), short(randmax) );

      st1 = tens;
      st2 = pageslice(tens, page_);

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 12UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 12UL );

      if( pageslice(st1,page_) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " page: " << page_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,page_) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the dilatedsubtensor addition assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the addition assignment operators of the dilatedsubtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testAddAssign()
{
   using blaze::dilatedsubtensor;
   using blaze::pageslice;

   //=====================================================================================
   // Row-major addition assignment
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubtensor addition assignment (no aliasing)";

      initialize();

      TT tens1( 64UL, 64UL, 64UL );
      TT tens2( 64UL, 64UL, 64UL );
      randomize( tens1, int(randmin), int(randmax) );
      tens2 = tens1;

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 16UL, 2UL, 2UL);

      st1 += dilatedsubtensor( tens1, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      st2 += pageslice(dilatedsubtensor( tens2,  4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL ),0);

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 16UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 16UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major dilatedsubtensor addition assignment (aliasing)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 16UL, 2UL, 2UL);

      st1 += dilatedsubtensor( tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      st2 += pageslice(dilatedsubtensor( tens2_,  4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL ),0);

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 16UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 16UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dense tensor addition assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major dense tensor addition assignment (mixed type)";

      initialize();


      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 12UL, 4UL, 8UL, 12UL, 4UL, 3UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 12UL, 8UL, 12UL, 3UL, 2UL);

      blaze::DynamicTensor<short> tens( 4UL, 8UL, 12UL );
      randomize( tens, short(randmin), short(randmax) );

      st1 += tens;
      st2 += pageslice(tens, 0);

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 12UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 12UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the dilatedsubtensor subtraction assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the subtraction assignment operators of the dilatedsubtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testSubAssign()
{
   using blaze::dilatedsubtensor;
   using blaze::pageslice;

   //=====================================================================================
   // Row-major subtraction assignment
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubtensor subtraction assignment (no aliasing)";

      initialize();

      TT tens1( 64UL, 64UL, 64UL );
      TT tens2( 64UL, 64UL, 64UL );
      randomize( tens1, int(randmin), int(randmax) );
      tens2 = tens1;

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 16UL, 2UL, 2UL);

      st1 -= dilatedsubtensor( tens1, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      st2 -= pageslice(dilatedsubtensor( tens2,  4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL ),0);

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 16UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 16UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major dilatedsubtensor subtraction assignment (aliasing)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 16UL, 2UL, 2UL);

      st1 -= dilatedsubtensor( tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      st2 -= pageslice(dilatedsubtensor( tens2_,  4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL ),0);

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 16UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 16UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dense tensor subtraction assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major dense tensor subtraction assignment (mixed type)";

      initialize();


      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 12UL, 4UL, 8UL, 12UL, 4UL, 3UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 12UL, 8UL, 12UL, 3UL, 2UL);

      blaze::DynamicTensor<short> tens( 4UL, 8UL, 12UL );
      randomize( tens, short(randmin), short(randmax) );

      st1 -= tens;
      st2 -= pageslice(tens, 0);

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 12UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 12UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the dilatedsubtensor Schur product assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the Schur product assignment operators of the dilatedsubtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testSchurAssign()
{
   using blaze::dilatedsubtensor;
   using blaze::pageslice;

   //=====================================================================================
   // Row-major schur assignment
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubtensor schur assignment (no aliasing)";

      initialize();

      TT tens1( 64UL, 64UL, 64UL );
      TT tens2( 64UL, 64UL, 64UL );
      randomize( tens1, int(randmin), int(randmax) );
      tens2 = tens1;

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 16UL, 2UL, 2UL);

      st1 %= dilatedsubtensor( tens1, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      st2 %= pageslice(dilatedsubtensor( tens2,  4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL ),0);

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 16UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 16UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: schur assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major dilatedsubtensor schur assignment (aliasing)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 16UL, 2UL, 2UL);

      st1 %= dilatedsubtensor( tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      st2 %= pageslice(dilatedsubtensor( tens2_,  4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL ),0);

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 16UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 16UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: schur assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dense tensor schur assignment
   //=====================================================================================

   {
      test_ = "Row-major/row-major dense tensor schur assignment (mixed type)";

      initialize();


      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 12UL, 4UL, 8UL, 12UL, 4UL, 3UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 12UL, 8UL, 12UL, 3UL, 2UL);

      blaze::DynamicTensor<short> tens( 4UL, 8UL, 12UL );
      randomize( tens, short(randmin), short(randmax) );

      st1 %= tens;
      st2 %= pageslice(tens, 0);

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 12UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 12UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: schur assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************



//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Initialization of all member tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function initializes all member tensors to specific predetermined values.
*/
void DenseTest::initialize()
{
   // Initializing the row-major dynamic tensors
   randomize( tens1_, int(randmin), int(randmax) );
   tens2_ = tens1_;
}
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
