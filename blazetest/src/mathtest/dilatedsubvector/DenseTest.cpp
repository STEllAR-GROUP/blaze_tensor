//=================================================================================================
/*!
//  \file src/mathtest/dilatedsubvector/DenseTest.cpp
//  \brief Source file for the DilatedSubvector dense aligned test
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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
#include <blaze/math/CompressedVector.h>
#include <blaze/math/CustomVector.h>
#include <blaze/math/Views.h>
#include <blaze/util/Memory.h>
#include <blaze/util/policies/Deallocate.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>

#include <blazetest/mathtest/dilatedsubvector/DenseTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace dilatedsubvector {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the DilatedSubvector dense aligned test.
//
// \exception std::runtime_error Operation error detected.
*/
DenseTest::DenseTest()
   : vec1_( 64UL )
   , vec2_( 64UL )
{
   testConstructors();
   testAssignment();
   testAddAssign();
   testSubAssign();
   testMultAssign();
   testDivAssign();
   testCrossAssign();
   testScaling();
   testSubscript();
   testIterator();
   testNonZeros();
   testReset();
   testClear();
   testIsDefault();
   testIsSame();
   testDilatedSubvector();
   testElements();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the DilatedSubvector constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the DilatedSubvector specialization. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testConstructors()
{
   using blaze::dilatedsubvector;


   test_ = "DilatedSubvector constructor";

   initialize();

   const size_t alignment = blaze::AlignmentOf<int>::value;

   for( size_t start=0UL; start<vec1_.size(); start+=alignment ) {
      for( size_t maxsize=0UL; ; maxsize+=alignment ) {

         size_t size( blaze::min( maxsize, vec1_.size( ) - start ) );

         for( size_t dilation = 1UL; dilation < maxsize; ++dilation ) {

            while( start + size * dilation >= vec1_.size() ) --size;
            auto indices = generate_indices( start, size, dilation );

            const ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
            const USVT sv2 = dilatedsubvector( vec2_, start, size, dilation );

            if( sv1 != sv2 )
            {
               std::ostringstream oss;
               oss << " Test: " << test_ << "\n"
                   << " Error: Setup of dense dilatedsubvector failed\n"
                   << " Details:\n"
                   << "   Start = " << start << "\n"
                   << "   Size  = " << size << "\n"
                   << "   DilatedSubvector:\n"
                   << sv1 << "\n"
                   << "   Reference:\n"
                   << sv2 << "\n";
               throw std::runtime_error( oss.str( ) );
            }
         }

         if( start + maxsize > vec1_.size( ) )
            break;
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DilatedSubvector assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the DilatedSubvector specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testAssignment()
{
   using blaze::dilatedsubvector;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowVector;


   //=====================================================================================
   // Homogeneous assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector homogeneous assignment";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );

      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      sv1 = 12;
      sv2 = 12;

      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // List assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector initializer list assignment (complete list)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );

      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      sv1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
      sv2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector initializer list assignment (incomplete list)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );

      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      sv1 = { 1, 2, 3 };
      sv2 = { 1, 2, 3 };

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Copy assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector copy assignment (no aliasing)";

      initialize();

      VT vec1( 64UL );
      VT vec2( 64UL );
      randomize( vec1, int(randmin), int(randmax) );
      vec2 = vec1;

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      sv1 = dilatedsubvector( vec1, 16UL, 21UL, 2UL );
      sv2 = dilatedsubvector( vec2, 16UL, 21UL, 2UL );

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector copy assignment (aliasing)";

      initialize();

      auto indices = generate_indices( 8UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 8UL, 21UL, 2UL );
      sv1 = blaze::dilatedsubvector( vec1_, 16UL, 21UL, 2UL );
      sv2 = blaze::dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Dense vector assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector dense vector assignment (mixed type)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      blaze::DynamicVector<short,rowVector> vec( 21UL );
      randomize( vec, short(randmin), short(randmax) );

      sv1 = vec;
      sv2 = vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector dense vector assignment (aligned/padded)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      using AlignedPadded = blaze::CustomVector<int,blaze::aligned,padded,rowVector>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
      AlignedPadded vec( memory.get(), 21UL, 32UL );
      randomize( vec, int(randmin), int(randmax) );

      sv1 = vec;
      sv2 = vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector dense vector assignment (unaligned/unpadded)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      using UnalignedUnpadded = blaze::CustomVector<int,blaze::unaligned,unpadded,rowVector>;
      std::unique_ptr<int[]> memory( new int[22] );
      UnalignedUnpadded vec( memory.get()+1UL, 21UL );
      randomize( vec, int(randmin), int(randmax) );

      sv1 = vec;
      sv2 = vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Sparse vector assignment
   //=====================================================================================

//    {
//       test_ = "DilatedSubvector sparse vector assignment";
//
//       initialize();
//
//       ASVT sv1 = dilatedsubvector<aligned>  ( vec1_, 16UL, 21UL );
//       USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL );
//
//       blaze::CompressedVector<int,rowVector> vec( 21UL );
//       randomize( vec, 6UL, int(randmin), int(randmax) );
//
//       sv1 = vec;
//       sv2 = vec;
//
//       checkSize( sv1, 21UL );
//       checkSize( sv2, 21UL );
//
//       if( sv1 != sv2 || vec1_ != vec2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sv1 << "\n"
//              << "   Expected result:\n" << sv2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DilatedSubvector addition assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the addition assignment operators of the DilatedSubvector
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testAddAssign()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowVector;


   //=====================================================================================
   // DilatedSubvector addition assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector addition assignment (no aliasing)";

      initialize();

      VT vec1( 64UL );
      VT vec2( 64UL );
      randomize( vec1, int(randmin), int(randmax) );
      vec2 = vec1;

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      sv1 += dilatedsubvector( vec1, 16UL, 21UL, 2UL );
      sv2 += dilatedsubvector( vec2, 16UL, 21UL, 2UL );

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector addition assignment (aliasing)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      sv1 += dilatedsubvector( vec1_, 20UL, 21UL, 2UL );
      sv2 += dilatedsubvector( vec2_, 20UL, 21UL, 2UL );

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Dense vector addition assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector dense vector addition assignment (mixed type)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      blaze::DynamicVector<short,rowVector> vec( 21UL );
      randomize( vec, short(randmin), short(randmax) );

      sv1 += vec;
      sv2 += vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector dense vector addition assignment (aligned/padded)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      using AlignedPadded = blaze::CustomVector<int,aligned,padded,rowVector>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
      AlignedPadded vec( memory.get(), 21UL, 32UL );
      randomize( vec, int(randmin), int(randmax) );

      sv1 += vec;
      sv2 += vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector dense vector addition assignment (unaligned/unpadded)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      using UnalignedUnpadded = blaze::CustomVector<int,unaligned,unpadded,rowVector>;
      std::unique_ptr<int[]> memory( new int[22] );
      UnalignedUnpadded vec( memory.get()+1UL, 21UL );
      randomize( vec, int(randmin), int(randmax) );

      sv1 += vec;
      sv2 += vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Sparse vector addition assignment
   //=====================================================================================

//    {
//       test_ = "DilatedSubvector sparse vector addition assignment";
//
//       initialize();
//
//       ASVT sv1 = dilatedsubvector<aligned>  ( vec1_, 16UL, 21UL );
//       USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL );
//
//       blaze::CompressedVector<int,rowVector> vec( 21UL );
//       randomize( vec, 6UL, int(randmin), int(randmax) );
//
//       sv1 += vec;
//       sv2 += vec;
//
//       checkSize( sv1, 21UL );
//       checkSize( sv2, 21UL );
//
//       if( sv1 != sv2 || vec1_ != vec2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sv1 << "\n"
//              << "   Expected result:\n" << sv2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DilatedSubvector subtraction assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the subtraction assignment operators of the DilatedSubvector
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testSubAssign()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowVector;


   //=====================================================================================
   // DilatedSubvector subtraction assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector subtraction assignment (no aliasing)";

      initialize();

      VT vec1( 64UL );
      VT vec2( 64UL );
      randomize( vec1, int(randmin), int(randmax) );
      vec2 = vec1;

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      sv1 -= dilatedsubvector( vec1, 20UL, 21UL, 2UL );
      sv2 -= dilatedsubvector( vec2, 20UL, 21UL, 2UL );

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector subtraction assignment (aliasing)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      sv1 -= dilatedsubvector( vec1_, 20UL, 21UL, 2UL );
      sv2 -= dilatedsubvector( vec2_, 20UL, 21UL, 2UL );

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Dense vector subtraction assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector dense vector subtraction assignment (mixed type)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      blaze::DynamicVector<short,rowVector> vec( 21UL );
      randomize( vec, short(randmin), short(randmax) );

      sv1 -= vec;
      sv2 -= vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector dense vector subtraction assignment (aligned/padded)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      using AlignedPadded = blaze::CustomVector<int,aligned,padded,rowVector>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
      AlignedPadded vec( memory.get(), 21UL, 32UL );
      randomize( vec, int(randmin), int(randmax) );

      sv1 -= vec;
      sv2 -= vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector dense vector subtraction assignment (unaligned/unpadded)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      using UnalignedUnpadded = blaze::CustomVector<int,unaligned,unpadded,rowVector>;
      std::unique_ptr<int[]> memory( new int[22] );
      UnalignedUnpadded vec( memory.get()+1UL, 21UL );
      randomize( vec, int(randmin), int(randmax) );

      sv1 -= vec;
      sv2 -= vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Sparse vector subtraction assignment
   //=====================================================================================

//    {
//       test_ = "DilatedSubvector sparse vector subtraction assignment";
//
//       initialize();
//
//       ASVT sv1 = dilatedsubvector<aligned>  ( vec1_, 16UL, 21UL );
//       USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL );
//
//       blaze::CompressedVector<int,rowVector> vec( 21UL );
//       randomize( vec, 6UL, int(randmin), int(randmax) );
//
//       sv1 -= vec;
//       sv2 -= vec;
//
//       checkSize( sv1, 21UL );
//       checkSize( sv2, 21UL );
//
//       if( sv1 != sv2 || vec1_ != vec2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sv1 << "\n"
//              << "   Expected result:\n" << sv2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DilatedSubvector multiplication assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the multiplication assignment operators of the DilatedSubvector
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testMultAssign()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowVector;


   //=====================================================================================
   // DilatedSubvector multiplication assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector multiplication assignment (no aliasing)";

      initialize();

      VT vec1( 64UL );
      VT vec2( 64UL );
      randomize( vec1, int(randmin), int(randmax) );
      vec2 = vec1;

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      sv1 *= dilatedsubvector( vec1, 20UL, 21UL, 2UL );
      sv2 *= dilatedsubvector( vec2, 20UL, 21UL, 2UL );

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector multiplication assignment (aliasing)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      sv1 *= dilatedsubvector( vec1_, 20UL, 21UL, 2UL );
      sv2 *= dilatedsubvector( vec2_, 20UL, 21UL, 2UL );

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Dense vector multiplication assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector dense vector multiplication assignment (mixed type)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      blaze::DynamicVector<short,rowVector> vec( 21UL );
      randomize( vec, short(randmin), short(randmax) );

      sv1 *= vec;
      sv2 *= vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector dense vector multiplication assignment (aligned/padded)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      using AlignedPadded = blaze::CustomVector<int,aligned,padded,rowVector>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
      AlignedPadded vec( memory.get(), 21UL, 32UL );
      randomize( vec, int(randmin), int(randmax) );

      sv1 *= vec;
      sv2 *= vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector dense vector multiplication assignment (unaligned/unpadded)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      using UnalignedUnpadded = blaze::CustomVector<int,unaligned,unpadded,rowVector>;
      std::unique_ptr<int[]> memory( new int[22] );
      UnalignedUnpadded vec( memory.get()+1UL, 21UL );
      randomize( vec, int(randmin), int(randmax) );

      sv1 *= vec;
      sv2 *= vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Sparse vector multiplication assignment
   //=====================================================================================

//    {
//       test_ = "DilatedSubvector sparse vector multiplication assignment";
//
//       initialize();
//
//       ASVT sv1 = dilatedsubvector<aligned>  ( vec1_, 16UL, 21UL );
//       USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL );
//
//       blaze::CompressedVector<int,rowVector> vec( 21UL );
//       randomize( vec, 6UL, int(randmin), int(randmax) );
//
//       sv1 -= vec;
//       sv2 -= vec;
//
//       checkSize( sv1, 21UL );
//       checkSize( sv2, 21UL );
//
//       if( sv1 != sv2 || vec1_ != vec2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Multiplication assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sv1 << "\n"
//              << "   Expected result:\n" << sv2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DilatedSubvector division assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the division assignment operators of the DilatedSubvector
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testDivAssign()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowVector;


   //=====================================================================================
   // DilatedSubvector division assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector division assignment (no aliasing)";

      initialize();

      VT vec1( 64UL );
      VT vec2( 64UL );
      randomize( vec1, 1, int(randmax) );
      vec2 = vec1;

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      sv1 /= dilatedsubvector( vec1, 20UL, 21UL, 2UL );
      sv2 /= dilatedsubvector( vec2, 20UL, 21UL, 2UL );

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Division assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector division assignment (aliasing)";

      randomize( vec1_, 1, int(randmax) );
      vec2_ = vec1_;

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      sv1 /= dilatedsubvector( vec1_, 20UL, 21UL, 2UL );
      sv2 /= dilatedsubvector( vec2_, 20UL, 21UL, 2UL );

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Division assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Dense vector division assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector dense vector division assignment (mixed type)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      blaze::DynamicVector<short,rowVector> vec( 21UL );
      randomize( vec, short(1), short(randmax) );

      sv1 /= vec;
      sv2 /= vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Division assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector dense vector division assignment (aligned/padded)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      using AlignedPadded = blaze::CustomVector<int,aligned,padded,rowVector>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
      AlignedPadded vec( memory.get(), 21UL, 32UL );
      randomize( vec, 1, int(randmax) );

      sv1 /= vec;
      sv2 /= vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Division assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector dense vector division assignment (unaligned/unpadded)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      using UnalignedUnpadded = blaze::CustomVector<int,unaligned,unpadded,rowVector>;
      std::unique_ptr<int[]> memory( new int[22] );
      UnalignedUnpadded vec( memory.get()+1UL, 21UL );
      randomize( vec, 1, int(randmax) );

      sv1 /= vec;
      sv2 /= vec;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Division assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DilatedSubvector cross product assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the cross product assignment operators of the DilatedSubvector
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testCrossAssign()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowVector;


   //=====================================================================================
   // DilatedSubvector cross product assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector cross product assignment (no aliasing)";

      initialize();

      VT vec1( 64UL );
      VT vec2( 64UL );
      randomize( vec1, int(randmin), int(randmax) );
      vec2 = vec1;

      auto indices = generate_indices( 16UL, 3UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 3UL, 2UL );
      sv1 %= dilatedsubvector( vec1, 32UL, 3UL, 2UL );
      sv2 %= dilatedsubvector( vec2, 32UL, 3UL, 2UL );

      checkSize( sv1, 3UL );
      checkSize( sv2, 3UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Cross product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector cross product assignment (aliasing)";

      initialize();

      auto indices = generate_indices( 16UL, 3UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 3UL, 2UL );
      sv1 %= dilatedsubvector( vec1_, 32UL, 3UL, 2UL );
      sv2 %= dilatedsubvector( vec2_, 32UL, 3UL, 2UL );

      checkSize( sv1, 3UL );
      checkSize( sv2, 3UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Cross product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Dense vector cross product assignment
   //=====================================================================================

   {
      test_ = "DilatedSubvector dense vector cross product assignment (mixed type)";

      initialize();

      auto indices = generate_indices( 16UL, 3UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 3UL, 2UL );

      blaze::DynamicVector<short,rowVector> vec( 3UL );
      randomize( vec, short(randmin), short(randmax) );

      sv1 %= vec;
      sv2 %= vec;

      checkSize( sv1, 3UL );
      checkSize( sv2, 3UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Cross product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector dense vector cross product assignment (aligned/padded)";

      initialize();

      auto indices = generate_indices( 16UL, 3UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 3UL, 2UL );

      using AlignedPadded = blaze::CustomVector<int,aligned,padded,rowVector>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 16UL ) );
      AlignedPadded vec( memory.get(), 3UL, 16UL );
      randomize( vec, int(randmin), int(randmax) );

      sv1 %= vec;
      sv2 %= vec;

      checkSize( sv1, 3UL );
      checkSize( sv2, 3UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Cross product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DilatedSubvector dense vector cross product assignment (unaligned/unpadded)";

      initialize();

      auto indices = generate_indices( 16UL, 3UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 3UL, 2UL );

      using UnalignedUnpadded = blaze::CustomVector<int,unaligned,unpadded,rowVector>;
      std::unique_ptr<int[]> memory( new int[4] );
      UnalignedUnpadded vec( memory.get()+1UL, 3UL );
      randomize( vec, int(randmin), int(randmax) );

      sv1 %= vec;
      sv2 %= vec;

      checkSize( sv1, 3UL );
      checkSize( sv2, 3UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Cross product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Sparse vector cross product assignment
   //=====================================================================================

//    {
//       test_ = "DilatedSubvector sparse vector cross product assignment";
//
//       initialize();
//
//       ASVT sv1 = dilatedsubvector<aligned>  ( vec1_, 16UL, 3UL );
//       USVT sv2 = dilatedsubvector( vec2_, 16UL, 3UL );
//
//       blaze::CompressedVector<int,rowVector> vec( 3UL );
//       randomize( vec, 2UL, int(randmin), int(randmax) );
//
//       sv1 %= vec;
//       sv2 %= vec;
//
//       checkSize( sv1, 3UL );
//       checkSize( sv2, 3UL );
//
//       if( sv1 != sv2 || vec1_ != vec2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Cross product assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sv1 << "\n"
//              << "   Expected result:\n" << sv2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of all DilatedSubvector (self-)scaling operations.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all available ways to scale an instance of the DilatedSubvector
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testScaling()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Self-scaling (v*=s)
   //=====================================================================================

   {
      test_ = "DilatedSubvector self-scaling (v*=s)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      sv1 *= 3;
      sv2 *= 3;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Self-scaling (v=v*s)
   //=====================================================================================

   {
      test_ = "DilatedSubvector self-scaling (v=v*s)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      sv1 = sv1 * 3;
      sv2 = sv2 * 3;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Self-scaling (v=s*v)
   //=====================================================================================

   {
      test_ = "DilatedSubvector self-scaling (v=s*v)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      sv1 = 3 * sv1;
      sv2 = 3 * sv2;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Self-scaling (v/=s)
   //=====================================================================================

   {
      test_ = "DilatedSubvector self-scaling (v/=s)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      sv1 /= 0.5;
      sv2 /= 0.5;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Division assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Self-scaling (v=v/s)
   //=====================================================================================

   {
      test_ = "DilatedSubvector self-scaling (v/=s)";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      sv1 = sv1 / 0.5;
      sv2 = sv2 / 0.5;

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Division assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // DilatedSubvector::scale()
   //=====================================================================================

   {
      test_ = "DilatedSubvector::scale()";

      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      // Integral scaling of the dilatedsubvector in the range [8,23]
      sv1.scale( 3 );
      sv2.scale( 3 );

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Integral scale operation of range [8,23] failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }

      // Floating point scaling of the dilatedsubvector in the range [8,23]
      sv1.scale( 0.5 );
      sv2.scale( 0.5 );

      checkSize( sv1, 21UL );
      checkSize( sv2, 21UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Floating point scale operation of range [8,23] failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DilatedSubvector subscript operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the subscript operator
// of the DilatedSubvector specialization. In case an error is detected, a \a std::runtime_error
// exception is thrown.
*/
void DenseTest::testSubscript()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::unaligned;


   test_ = "DilatedSubvector::operator[]";

   initialize();

   auto indices = generate_indices( 16UL, 21UL, 2UL );
   ASVT sv1 = blaze::elements( vec1_, indices.data( ), indices.size( ) );
   USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

   // Assignment to the element at index 1
   sv1[1] = 9;
   sv2[1] = 9;

   checkSize( sv1, 21UL );
   checkSize( sv2, 21UL );

   if( sv1 != sv2 || vec1_ != vec2_ ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Subscript operator failed\n"
          << " Details:\n"
          << "   Result:\n" << sv1 << "\n"
          << "   Expected result:\n" << sv2 << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Assignment to the element at index 2
   sv1[2] = 0;
   sv2[2] = 0;

   checkSize( sv1, 21UL );
   checkSize( sv2, 21UL );

   if( sv1 != sv2 || vec1_ != vec2_ ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Subscript operator failed\n"
          << " Details:\n"
          << "   Result:\n" << sv1 << "\n"
          << "   Expected result:\n" << sv2 << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Assignment to the element at index 3
   sv1[3] = -8;
   sv2[3] = -8;

   checkSize( sv1, 21UL );
   checkSize( sv2, 21UL );

   if( sv1 != sv2 || vec1_ != vec2_ ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Subscript operator failed\n"
          << " Details:\n"
          << "   Result:\n" << sv1 << "\n"
          << "   Expected result:\n" << sv2 << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Addition assignment to the element at index 0
   sv1[0] += -3;
   sv2[0] += -3;

   checkSize( sv1, 21UL );
   checkSize( sv2, 21UL );

   if( sv1 != sv2 || vec1_ != vec2_ ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Subscript operator failed\n"
          << " Details:\n"
          << "   Result:\n" << sv1 << "\n"
          << "   Expected result:\n" << sv2 << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Subtraction assignment to the element at index 1
   sv1[1] -= 6;
   sv2[1] -= 6;

   checkSize( sv1, 21UL );
   checkSize( sv2, 21UL );

   if( sv1 != sv2 || vec1_ != vec2_ ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Subscript operator failed\n"
          << " Details:\n"
          << "   Result:\n" << sv1 << "\n"
          << "   Expected result:\n" << sv2 << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Multiplication assignment to the element at index 1
   sv1[1] *= 3;
   sv2[1] *= 3;

   checkSize( sv1, 21UL );
   checkSize( sv2, 21UL );

   if( sv1 != sv2 || vec1_ != vec2_ ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Subscript operator failed\n"
          << " Details:\n"
          << "   Result:\n" << sv1 << "\n"
          << "   Expected result:\n" << sv2 << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Division assignment to the element at index 3
   sv1[3] /= 2;
   sv2[3] /= 2;

   checkSize( sv1, 21UL );
   checkSize( sv2, 21UL );

   if( sv1 != sv2 || vec1_ != vec2_ ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Subscript operator failed\n"
          << " Details:\n"
          << "   Result:\n" << sv1 << "\n"
          << "   Expected result:\n" << sv2 << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DilatedSubvector iterator implementation.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the iterator implementation of the DilatedSubvector specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testIterator()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::unaligned;


   initialize();

   // Testing the Iterator default constructor
   {
      test_ = "Iterator default constructor";

      ASVT::Iterator it{};

      if( it != ASVT::Iterator() ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed iterator default constructor\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Testing the ConstIterator default constructor
   {
      test_ = "ConstIterator default constructor";

      ASVT::ConstIterator it{};

      if( it != ASVT::ConstIterator() ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed iterator default constructor\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Testing conversion from Iterator to ConstIterator
   {
      test_ = "Iterator/ConstIterator conversion";

      USVT sv = dilatedsubvector( vec1_, 0UL, 16UL, 2UL );
      USVT::ConstIterator it( begin( sv ) );

      if( it == end( sv ) || *it != sv[0] ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed iterator conversion detected\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Counting the number of elements in first half of the vector via Iterator (end-begin)
   {
      test_ = "Iterator subtraction (end-begin)";

      USVT sv = dilatedsubvector( vec1_, 0UL, 16UL, 2UL );
      const ptrdiff_t number( end( sv ) - begin( sv ) );

      if( number != 16L ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid number of elements detected\n"
             << " Details:\n"
             << "   Number of elements         : " << number << "\n"
             << "   Expected number of elements: 16\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Counting the number of elements in first half of the vector via Iterator (begin-end)
   {
      test_ = "Iterator subtraction (begin-end)";

      USVT sv = dilatedsubvector( vec1_, 0UL, 16UL, 2UL );
      const ptrdiff_t number( begin( sv ) - end( sv ) );

      if( number != -16L ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid number of elements detected\n"
             << " Details:\n"
             << "   Number of elements         : " << number << "\n"
             << "   Expected number of elements: -16\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Counting the number of elements in second half of the vector via ConstIterator (end-begin)
   {
      test_ = "ConstIterator subtraction (end-begin)";

      USVT sv = dilatedsubvector( vec1_, 0UL, 31UL, 2UL );
      const ptrdiff_t number( cend( sv ) - cbegin( sv ) );

      if( number != 31L ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid number of elements detected\n"
             << " Details:\n"
             << "   Number of elements         : " << number << "\n"
             << "   Expected number of elements: 48\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Counting the number of elements in second half of the vector via ConstIterator (begin-end)
   {
      test_ = "ConstIterator subtraction (begin-end)";

      USVT sv = dilatedsubvector( vec1_, 0UL, 31UL, 2UL );
      const ptrdiff_t number( cbegin( sv ) - cend( sv ) );

      if( number != -31L ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid number of elements detected\n"
             << " Details:\n"
             << "   Number of elements         : " << number << "\n"
             << "   Expected number of elements: -48\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Testing read-only access via ConstIterator
   {
      test_ = "Read-only access via ConstIterator";

      USVT sv = dilatedsubvector( vec1_, 16UL, 8UL, 2UL );
      USVT::ConstIterator it ( cbegin( sv ) );
      USVT::ConstIterator end( cend( sv ) );

      if( it == end || *it != sv[0] ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid initial iterator detected\n";
         throw std::runtime_error( oss.str() );
      }

      ++it;

      if( it == end || *it != sv[1] ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator pre-increment failed\n";
         throw std::runtime_error( oss.str() );
      }

      --it;

      if( it == end || *it != sv[0] ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator pre-decrement failed\n";
         throw std::runtime_error( oss.str() );
      }

      it++;

      if( it == end || *it != sv[1] ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator post-increment failed\n";
         throw std::runtime_error( oss.str() );
      }

      it--;

      if( it == end || *it != sv[0] ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator post-decrement failed\n";
         throw std::runtime_error( oss.str() );
      }

      it += 2UL;

      if( it == end || *it != sv[2] ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator addition assignment failed\n";
         throw std::runtime_error( oss.str() );
      }

      it -= 2UL;

      if( it == end || *it != sv[0] ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator subtraction assignment failed\n";
         throw std::runtime_error( oss.str() );
      }

      it = it + 3UL;

      if( it == end || *it != sv[3] ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator/scalar addition failed\n";
         throw std::runtime_error( oss.str() );
      }

      it = it - 3UL;

      if( it == end || *it != sv[0] ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator/scalar subtraction failed\n";
         throw std::runtime_error( oss.str() );
      }

      it = 8UL + it;

      if( it != end ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Scalar/iterator addition failed\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Testing assignment via Iterator
   {
      test_ = "Assignment via Iterator";

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      int value = 6;

      ASVT::Iterator it1( begin( sv1 ) );
      USVT::Iterator it2( begin( sv2 ) );

      for( ; it1!=end( sv1 ); ++it1, ++it2 ) {
         *it1 = value;
         *it2 = value;
         ++value;
      }

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment via iterator failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Testing addition assignment via Iterator
   {
      test_ = "Addition assignment via Iterator";

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      int value = 6;

      ASVT::Iterator it1( begin( sv1 ) );
      USVT::Iterator it2( begin( sv2 ) );

      for( ; it1!=end( sv1 ); ++it1, ++it2 ) {
         *it1 += value;
         *it2 += value;
         ++value;
      }

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment via iterator failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Testing subtraction assignment via Iterator
   {
      test_ = "Subtraction assignment via Iterator";

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      int value = 6;

      ASVT::Iterator it1( begin( sv1 ) );
      USVT::Iterator it2( begin( sv2 ) );

      for( ; it1!=end( sv1 ); ++it1, ++it2 ) {
         *it1 -= value;
         *it2 -= value;
         ++value;
      }

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment via iterator failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Testing multiplication assignment via Iterator
   {
      test_ = "Multiplication assignment via Iterator";

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );
      int value = 1;

      ASVT::Iterator it1( begin( sv1 ) );
      USVT::Iterator it2( begin( sv2 ) );

      for( ; it1!=end( sv1 ); ++it1, ++it2 ) {
         *it1 *= value;
         *it2 *= value;
         ++value;
      }

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Multiplication assignment via iterator failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Testing division assignment via Iterator
   {
      test_ = "Division assignment via Iterator";

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

      ASVT::Iterator it1( begin( sv1 ) );
      USVT::Iterator it2( begin( sv2 ) );

      for( ; it1!=end( sv1 ); ++it1, ++it2 ) {
         *it1 /= 2;
         *it2 /= 2;
      }

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Division assignment via iterator failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c nonZeros() member function of the DilatedSubvector specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c nonZeros() member function of the DilatedSubvector
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testNonZeros()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::unaligned;


   test_ = "DilatedSubvector::nonZeros()";

   initialize();

   // Initialization check
   auto indices = generate_indices( 16UL, 21UL, 2UL );
   ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
   USVT sv2 = dilatedsubvector( vec2_, 16UL, 21UL, 2UL );

   checkSize( sv1, 21UL );
   checkSize( sv2, 21UL );

   if( sv1.nonZeros() != sv2.nonZeros() ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Initialization failed\n"
          << " Details:\n"
          << "   Result:\n" << sv1 << "\n"
          << "   Expected result:\n" << sv2 << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Changing the number of non-zeros via the dense dilatedsubvector
   sv1[3] = 0;
   sv2[3] = 0;

   checkSize( sv1, 21UL );
   checkSize( sv2, 21UL );

   if( sv1.nonZeros() != sv2.nonZeros() ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Subscript operator failed\n"
          << " Details:\n"
          << "   Result:\n" << sv1 << "\n"
          << "   Expected result:\n" << sv2 << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Changing the number of non-zeros via the dense vector
   vec1_[9UL] = 5;
   vec2_[9UL] = 5;

   checkSize( sv1, 21UL );
   checkSize( sv2, 21UL );

   if( sv1.nonZeros() != sv2.nonZeros() ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Subscript operator failed\n"
          << " Details:\n"
          << "   Result:\n" << sv1 << "\n"
          << "   Expected result:\n" << sv2 << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reset() member function of the DilatedSubvector specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reset() member function of the DilatedSubvector
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testReset()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::reset;


   test_ = "DilatedSubvector::reset()";

   // Resetting a single element in the range [0,15]
   {
      initialize();

      auto indices = generate_indices( 0UL, 16UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 0UL, 16UL, 2UL );
      reset( sv1[4] );
      reset( sv2[4] );

      checkSize( sv1, 16UL );
      checkSize( sv2, 16UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Resetting the range [0,15] (lvalue)
   {
      initialize();

      auto indices = generate_indices( 0UL, 16UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 0UL, 16UL, 2UL );
      reset( sv1 );
      reset( sv2 );

      checkSize( sv1, 16UL );
      checkSize( sv2, 16UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation of range [0,15] failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Resetting the range [16,58] (rvalue)
   {
      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      reset( blaze::elements( vec1_, indices.data(), indices.size() ) );
      reset( dilatedsubvector( vec2_, 16UL, 21UL, 2UL ) );

      if( vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation of range [16,63] failed\n"
             << " Details:\n"
             << "   Result:\n" << vec1_ << "\n"
             << "   Expected result:\n" << vec2_ << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c clear() function with the DilatedSubvector specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c clear() function with the DilatedSubvector specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testClear()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::clear;


   test_ = "DilatedSubvector::clear()";

   // Clearing a single element in the range [0,15]
   {
      initialize();

      auto indices = generate_indices( 0UL, 16UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 0UL, 16UL, 2UL );
      clear( sv1[4] );
      clear( sv2[4] );

      checkSize( sv1, 16UL );
      checkSize( sv2, 16UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Clearing the range [0,15] (lvalue)
   {
      initialize();

      auto indices = generate_indices( 0UL, 16UL, 2UL );
      ASVT sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      USVT sv2 = dilatedsubvector( vec2_, 0UL, 16UL, 2UL );
      clear( sv1 );
      clear( sv2 );

      checkSize( sv1, 16UL );
      checkSize( sv2, 16UL );

      if( sv1 != sv2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation of range [0,15] failed\n"
             << " Details:\n"
             << "   Result:\n" << sv1 << "\n"
             << "   Expected result:\n" << sv2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Clearing the range [16,63] (rvalue)
   {
      initialize();

      auto indices = generate_indices( 16UL, 21UL, 2UL );
      clear( blaze::elements( vec1_, indices.data(), indices.size() ) );
      clear( dilatedsubvector( vec2_, 16UL, 21UL, 2UL ) );

      if( vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation of range [16,63] failed\n"
             << " Details:\n"
             << "   Result:\n" << vec1_ << "\n"
             << "   Expected result:\n" << vec2_ << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDefault() function with the DilatedSubvector specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDefault() function with the DilatedSubvector specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testIsDefault()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::isDefault;


   test_ = "isDefault() function";

   initialize();

   // isDefault with default vector
   {
      VT vec( 64UL, 0 );
      USVT sv = dilatedsubvector( vec, 16UL, 21UL, 2UL );

      if( isDefault( sv[1] ) != true ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid isDefault evaluation\n"
             << " Details:\n"
             << "   DilatedSubvector element: " << sv[1] << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( isDefault( sv ) != true ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid isDefault evaluation\n"
             << " Details:\n"
             << "   DilatedSubvector:\n" << sv << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // isDefault with non-default vector
   {
      USVT sv = dilatedsubvector( vec1_, 16UL, 21UL, 2UL );

      if( isDefault( sv ) != false ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid isDefault evaluation\n"
             << " Details:\n"
             << "   DilatedSubvector:\n" << sv << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isSame() function with the DilatedSubvector specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSame() function with the DilatedSubvector specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testIsSame()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;


   //=====================================================================================
   // Vector-based tests
   //=====================================================================================

   {
      test_ = "isSame() function (vector-based)";

      // isSame with vector and matching dilatedsubvector
      {
         USVT sv = dilatedsubvector( vec1_, 0UL, 64UL, 1UL );

         if( blaze::isSame( sv, vec1_ ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Vector:\n" << vec1_ << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( vec1_, sv ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Vector:\n" << vec1_ << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with vector and non-matching dilatedsubvector (different size)
      {
         USVT sv = dilatedsubvector( vec1_, 0UL, 16UL, 2UL );

         if( blaze::isSame( sv, vec1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Vector:\n" << vec1_ << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( vec1_, sv ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Vector:\n" << vec1_ << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with vector and non-matching dilatedsubvector (different offset)
      {
         USVT sv = dilatedsubvector( vec1_, 16UL, 21UL, 2UL );

         if( blaze::isSame( sv, vec1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Vector:\n" << vec1_ << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( vec1_, sv ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Vector:\n" << vec1_ << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching dilatedsubvectors
      {
         USVT sv1 = dilatedsubvector( vec1_, 16UL, 21UL, 2UL );
         USVT sv2 = dilatedsubvector( vec1_, 16UL, 21UL, 2UL );

         if( blaze::isSame( sv1, sv2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubvector:\n" << sv1 << "\n"
                << "   Second dilatedsubvector:\n" << sv2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching dilatedsubvectors (different size)
      {
         USVT sv1 = dilatedsubvector( vec1_, 16UL, 16UL, 2UL );
         USVT sv2 = dilatedsubvector( vec1_, 16UL, 21UL, 2UL );

         if( blaze::isSame( sv1, sv2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubvector:\n" << sv1 << "\n"
                << "   Second dilatedsubvector:\n" << sv2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching dilatedsubvectors (different offset)
      {
         USVT sv1 = dilatedsubvector( vec1_, 8UL, 21UL, 2UL );
         USVT sv2 = dilatedsubvector( vec1_, 16UL, 21UL, 2UL );

         if( blaze::isSame( sv1, sv2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubvector:\n" << sv1 << "\n"
                << "   Second dilatedsubvector:\n" << sv2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching dilatedsubvectors (different dilation)
      {
         USVT sv1 = dilatedsubvector( vec1_, 8UL, 12UL, 2UL );
         USVT sv2 = dilatedsubvector( vec1_, 8UL, 12UL, 3UL );

         if( blaze::isSame( sv1, sv2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubvector:\n" << sv1 << "\n"
                << "   Second dilatedsubvector:\n" << sv2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-based tests
   //=====================================================================================

   {
      test_ = "isSame() function (row-based)";

      blaze::DynamicMatrix<int,blaze::rowMajor> mat( 64UL, 64UL );
      randomize( mat );

      // isSame with row and matching dilatedsubvector
      {
         auto indices = generate_indices( 0UL, 32UL, 2UL );
         auto r  = blaze::elements( blaze::row( mat, 8UL ), indices.data(), indices.size() );
         auto sv = dilatedsubvector( r, 0UL, 32UL, 1UL );

         if( blaze::isSame( sv, r ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row:\n" << r << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( r, sv ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row:\n" << r << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row and non-matching dilatedsubvector (different size)
      {
         auto indices = generate_indices( 0UL, 32UL, 2UL );
         auto r  = blaze::elements( blaze::row( mat, 8UL ), indices.data(), indices.size() );
         auto sv = dilatedsubvector( r, 0UL, 16UL, 1UL );

         if( blaze::isSame( sv, r ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row:\n" << r << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( r, sv ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row:\n" << r << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row and non-matching dilatedsubvector (different offset)
      {
         auto indices = generate_indices( 0UL, 16UL, 2UL );
         auto r  = blaze::elements( blaze::row( mat, 8UL ), indices.data(), indices.size() );
         auto sv = dilatedsubvector( r, 8UL, 8UL, 1UL );

         if( blaze::isSame( sv, r ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row:\n" << r << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( r, sv ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row:\n" << r << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching dilatedsubvectors
      {
         auto r   = blaze::row( mat, 8UL );
         auto sv1 = dilatedsubvector( r, 0UL, 32UL, 2UL );
         auto sv2 = dilatedsubvector( r, 0UL, 32UL, 2UL );

         if( blaze::isSame( sv1, sv2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubvector:\n" << sv1 << "\n"
                << "   Second dilatedsubvector:\n" << sv2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching dilatedsubvectors (different size)
      {
         auto r   = blaze::row( mat, 8UL );
         auto sv1 = dilatedsubvector( r, 0UL, 16UL, 2UL );
         auto sv2 = dilatedsubvector( r, 0UL, 32UL, 2UL );

         if( blaze::isSame( sv1, sv2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubvector:\n" << sv1 << "\n"
                << "   Second dilatedsubvector:\n" << sv2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching dilatedsubvectors (different offset)
      {
         auto r   = blaze::row( mat, 8UL );
         auto sv1 = dilatedsubvector( r, 16UL, 16UL, 2UL );
         auto sv2 = dilatedsubvector( r, 0UL, 16UL, 2UL );

         if( blaze::isSame( sv1, sv2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubvector:\n" << sv1 << "\n"
                << "   Second dilatedsubvector:\n" << sv2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching dilatedsubvectors (different dilation)
      {
         auto r   = blaze::row( mat, 8UL );
         auto sv1 = dilatedsubvector( r, 0UL, 8UL, 2UL );
         auto sv2 = dilatedsubvector( r, 0UL, 8UL, 3UL );

         if( blaze::isSame( sv1, sv2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubvector:\n" << sv1 << "\n"
                << "   Second dilatedsubvector:\n" << sv2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-based tests
   //=====================================================================================

   {
      test_ = "isSame() function (column-based)";

      blaze::DynamicMatrix<int,blaze::columnMajor> mat( 64UL, 64UL );
      randomize( mat );

      // isSame with column and matching dilatedsubvector
      {
         auto indices = generate_indices( 0UL, 32UL, 2UL );
         auto c  = blaze::elements( blaze::column( mat, 8UL ), indices.data(), indices.size() );
         auto sv = dilatedsubvector( c, 0UL, 32UL, 1UL );

         if( blaze::isSame( sv, c ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column:\n" << c << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( c, sv ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column:\n" << c << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column and non-matching dilatedsubvector (different size)
      {
         auto indices = generate_indices( 0UL, 32UL, 2UL );
         auto c  = blaze::elements( blaze::column( mat, 8UL ), indices.data(), indices.size() );
         auto sv = dilatedsubvector( c, 0UL, 16UL, 1UL );

         if( blaze::isSame( sv, c ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column:\n" << c << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( c, sv ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column:\n" << c << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column and non-matching dilatedsubvector (different offset)
      {
         auto indices = generate_indices( 0UL, 16UL, 2UL );
         auto c  = blaze::elements( blaze::column( mat, 8UL ), indices.data(), indices.size() );
         auto sv = dilatedsubvector( c, 8UL, 8UL, 1UL );

         if( blaze::isSame( sv, c ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column:\n" << c << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( c, sv ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column:\n" << c << "\n"
                << "   DilatedSubvector:\n" << sv << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching dilatedsubvectors
      {
         auto c   = blaze::column( mat, 8UL );
         auto sv1 = dilatedsubvector( c, 0UL, 32UL, 2UL );
         auto sv2 = dilatedsubvector( c, 0UL, 32UL, 2UL );

         if( blaze::isSame( sv1, sv2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubvector:\n" << sv1 << "\n"
                << "   Second dilatedsubvector:\n" << sv2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching dilatedsubvectors (different size)
      {
         auto c   = blaze::column( mat, 8UL );
         auto sv1 = dilatedsubvector( c, 0UL, 16UL, 2UL );
         auto sv2 = dilatedsubvector( c, 0UL, 32UL, 2UL );

         if( blaze::isSame( sv1, sv2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubvector:\n" << sv1 << "\n"
                << "   Second dilatedsubvector:\n" << sv2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching dilatedsubvectors (different offset)
      {
         auto c   = blaze::column( mat, 8UL );
         auto sv1 = dilatedsubvector( c, 16UL, 16UL, 2UL );
         auto sv2 = dilatedsubvector( c, 0UL, 16UL, 2UL );

         if( blaze::isSame( sv1, sv2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubvector:\n" << sv1 << "\n"
                << "   Second dilatedsubvector:\n" << sv2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching dilatedsubvectors (different dilation)
      {
         auto c   = blaze::column( mat, 8UL );
         auto sv1 = dilatedsubvector( c, 0UL, 8UL, 2UL );
         auto sv2 = dilatedsubvector( c, 0UL, 8UL, 3UL );

         if( blaze::isSame( sv1, sv2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubvector:\n" << sv1 << "\n"
                << "   Second dilatedsubvector:\n" << sv2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c dilatedsubvector() function with the DilatedSubvector specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c dilatedsubvector() function used with the DilatedSubvector
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testDilatedSubvector()
{
   using blaze::dilatedsubvector;
   using blaze::aligned;
   using blaze::unaligned;


   test_ = "dilatedsubvector() function";

   initialize();

   {
      auto indices1 = generate_indices( 16UL, 16UL, 2UL );
      auto indices2 = generate_indices( 8UL, 4UL, 2UL );
      auto sv1 = blaze::elements( vec1_, indices1.data(), indices1.size() );
      auto sv2 = blaze::elements( sv1, indices2.data(), indices2.size() );
      USVT sv3 = dilatedsubvector( vec2_, 16UL, 16UL, 2UL );
      USVT sv4 = dilatedsubvector( sv3  , 8UL, 4UL, 2UL );

      if( sv2 != sv4 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: DilatedSubvector function failed\n"
             << " Details:\n"
             << "   Result:\n" << sv2 << "\n"
             << "   Expected result:\n" << sv4 << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( sv2[1] != sv4[1] ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subscript operator access failed\n"
             << " Details:\n"
             << "   Result: " << sv2[1] << "\n"
             << "   Expected result: " << sv4[1] << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( *sv2.begin() != *sv4.begin() ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator access failed\n"
             << " Details:\n"
             << "   Result: " << *sv2.begin() << "\n"
             << "   Expected result: " << *sv4.begin() << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   try {
      USVT sv1 = dilatedsubvector( vec1_, 0UL, 32UL, 2UL );
      USVT sv2 = dilatedsubvector( sv1  , 32UL,  8UL, 1UL );

      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Setup of out-of-bounds dilatedsubvector succeeded\n"
          << " Details:\n"
          << "   Result:\n" << sv2 << "\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::invalid_argument& ) {}

   try {
      USVT sv1 = dilatedsubvector( vec1_, 0UL, 32UL, 2UL );
      USVT sv2 = dilatedsubvector( sv1  , 16UL, 32UL, 2UL );

      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Setup of out-of-bounds dilatedsubvector succeeded\n"
          << " Details:\n"
          << "   Result:\n" << sv2 << "\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::invalid_argument& ) {}
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c elements() function with the DilatedSubvector class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c elements() function used with the DilatedSubvector
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testElements()
{
   using blaze::dilatedsubvector;
   using blaze::elements;
   using blaze::aligned;
   using blaze::unaligned;


   test_ = "elements() function";

   initialize();

   {
      auto indices = generate_indices( 16UL, 16UL, 2UL );
      auto sv1 = blaze::elements( vec1_, indices.data(), indices.size() );
      auto e1 = elements( sv1, { 8UL, 12UL } );

      USVT sv2 = dilatedsubvector( vec2_, 16UL, 16UL, 2UL );
      auto e2 = elements( sv2, { 8UL, 12UL } );

      if( e1 != e2 || vec1_ != vec2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Elements function failed\n"
             << " Details:\n"
             << "   Result:\n" << e1 << "\n"
             << "   Expected result:\n" << e2 << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( e1[1] != e2[1] ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subscript operator access failed\n"
             << " Details:\n"
             << "   Result: " << e1[1] << "\n"
             << "   Expected result: " << e2[1] << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( *e1.begin() != *e2.begin() ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator access failed\n"
             << " Details:\n"
             << "   Result: " << *e1.begin() << "\n"
             << "   Expected result: " << *e2.begin() << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   try {
      USVT sv = dilatedsubvector( vec1_, 16UL, 16UL, 2UL );
      auto e = elements( sv, { 8UL, 32UL } );

      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Setup of out-of-bounds element selection succeeded\n"
          << " Details:\n"
          << "   Result:\n" << e << "\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::invalid_argument& ) {}
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Initialization of all member vectors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function initializes all member vectors to specific predetermined values.
*/
void DenseTest::initialize()
{
   // Initializing the dynamic row vectors
   randomize( vec1_, int(randmin), int(randmax) );
   vec2_ = vec1_;
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

} // namespace dilatedsubvector

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
   std::cout << "   Running DilatedSubvector dense aligned test..." << std::endl;

   try
   {
      RUN_DILATEDSUBVECTOR_DENSEALIGNED_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during DilatedSubvector dense aligned test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
