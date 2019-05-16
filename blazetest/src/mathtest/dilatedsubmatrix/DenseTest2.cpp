//=================================================================================================
/*!
//  \file src/mathtest/dilatedsubmatrix/DenseTest2.cpp
//  \brief Source file for the dilatedsubmatrix dense test (part 2)
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
   testScaling();
   testFunctionCall();
   testIterator();
   testNonZeros();
   testReset();
   testClear();
   testTranspose();
   testCTranspose();
   testIsDefault();
   testIsSame();
   testDilatedSubmatrix();
   testRow();
   testRows();
   testColumn();
   testColumns();
   testBand();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of all dilatedsubmatrix (self-)scaling operations.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all available ways to scale an instance of the dilatedsubmatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testScaling()
{
   using blaze::dilatedsubmatrix;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major self-scaling (M*=s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M*=s) (8x16)";

      initialize();

      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
      auto column_indices = generate_indices( 16UL, 8UL, 3UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );

      sm1 *= 3;
      sm2 *= 3;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M*=s) (16x8)";

      initialize();

      auto row_indices = generate_indices( 8UL, 16UL, 2UL );
      auto column_indices = generate_indices( 16UL, 8UL, 3UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 16UL, 8UL, 2UL, 3UL );

      sm1 *= 3;
      sm2 *= 3;

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=M*s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=M*s) (8x16)";

      initialize();

      auto row_indices = generate_indices( 16UL, 8UL, 2UL );
      auto column_indices = generate_indices( 16UL, 8UL, 3UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 16UL, 16UL, 8UL, 8UL, 2UL, 3UL );

      sm1 = sm1 * 3;
      sm2 = sm2 * 3;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M=M*s) (16x8)";

      initialize();

      auto row_indices = generate_indices( 8UL, 16UL, 2UL );
      auto column_indices = generate_indices( 16UL, 8UL, 3UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 16UL, 8UL, 2UL, 3UL );

      sm1 = sm1 * 3;
      sm2 = sm2 * 3;

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=s*M)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=s*M) (8x16)";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      sm1 = 3 * sm1;
      sm2 = 3 * sm2;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M=s*M) (16x8)";

      initialize();

      auto row_indices = generate_indices( 8UL, 16UL, 2UL );
      auto column_indices = generate_indices( 16UL, 8UL, 3UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 16UL, 8UL, 2UL, 3UL );

      sm1 = 3 * sm1;
      sm2 = 3 * sm2;

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M/=s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M/=s) (8x16)";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      sm1 /= 0.5;
      sm2 /= 0.5;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M/=s) (16x8)";

      initialize();

      auto row_indices = generate_indices( 8UL, 16UL, 2UL );
      auto column_indices = generate_indices( 16UL, 8UL, 3UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 16UL, 8UL, 2UL, 3UL );

      sm1 /= 0.5;
      sm2 /= 0.5;

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=M/s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=M/s) (8x16)";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      sm1 = sm1 / 0.5;
      sm2 = sm2 / 0.5;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M=M/s) (16x8)";

      initialize();

      auto row_indices = generate_indices( 8UL, 16UL, 2UL );
      auto column_indices = generate_indices( 16UL, 8UL, 3UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 16UL, 8UL, 2UL, 3UL );

      sm1 = sm1 / 0.5;
      sm2 = sm2 / 0.5;

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dilatedsubmatrix::scale()
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubmatrix::scale()";

      initialize();

      // Initialization check
      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      // Integral scaling of the matrix
      sm1.scale( 2 );
      sm2.scale( 2 );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Integral scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }

      // Floating point scaling of the matrix
      sm1.scale( 0.5 );
      sm2.scale( 0.5 );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Floating point scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M*=s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M*=s) (8x16)";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      sm1 *= 3;
      sm2 *= 3;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M*=s) (16x8)";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      sm1 *= 3;
      sm2 *= 3;

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M=M*s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=M*s) (8x16)";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      sm1 = sm1 * 3;
      sm2 = sm2 * 3;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M=M*s) (16x8)";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      sm1 = sm1 * 3;
      sm2 = sm2 * 3;

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M=s*M)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=s*M) (8x16)";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      sm1 = 3 * sm1;
      sm2 = 3 * sm2;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M=s*M) (16x8)";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      sm1 = 3 * sm1;
      sm2 = 3 * sm2;

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M/=s)
   //=====================================================================================

   {
      test_ = "Column-major self-scaling (M/=s) (8x16)";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      sm1 /= 0.5;
      sm2 /= 0.5;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major self-scaling (M/=s) (16x8)";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      sm1 /= 0.5;
      sm2 /= 0.5;

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M=M/s)
   //=====================================================================================

   {
      test_ = "Column-major self-scaling (M=M/s) (8x16)";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      sm1 = sm1 / 0.5;
      sm2 = sm2 / 0.5;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major self-scaling (M=M/s) (16x8)";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      sm1 = sm1 / 0.5;
      sm2 = sm2 / 0.5;

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major dilatedsubmatrix::scale()
   //=====================================================================================

   {
      test_ = "Column-major dilatedsubmatrix::scale()";

      initialize();

      // Initialization check
      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      // Integral scaling of the matrix
      sm1.scale( 2 );
      sm2.scale( 2 );

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Integral scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }

      // Floating point scaling of the matrix
      sm1.scale( 0.5 );
      sm2.scale( 0.5 );

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Floating point scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the dilatedsubmatrix function call operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the function call operator
// of the dilatedsubmatrix specialization. In case an error is detected, a \a std::runtime_error
// exception is thrown.
*/
void DenseTest::testFunctionCall()
{
   using blaze::dilatedsubmatrix;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major dilatedsubmatrix tests
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubmatrix::operator()";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      // Assignment to the element (1,4)
      {
         sm1(1,4) = 9;
         sm2(1,4) = 9;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 16UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 16UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (3,10)
      {
         sm1(3,10) = 0;
         sm2(3,10) = 0;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 16UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 16UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (6,8)
      {
         sm1(6,8) = -7;
         sm2(6,8) = -7;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 16UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 16UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Addition assignment to the element (5,7)
      {
         sm1(5,7) += 3;
         sm2(5,7) += 3;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 16UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 16UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Subtraction assignment to the element (2,14)
      {
         sm1(2,14) -= -8;
         sm2(2,14) -= -8;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 16UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 16UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Multiplication assignment to the element (1,1)
      {
         sm1(1,1) *= 3;
         sm2(1,1) *= 3;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 16UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 16UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Division assignment to the element (3,4)
      {
         sm1(3,4) /= 2;
         sm2(3,4) /= 2;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 16UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 16UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major dilatedsubmatrix tests
   //=====================================================================================

   {
      test_ = "Column-major dilatedsubmatrix::operator()";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      // Assignment to the element (4,1)
      {
         sm1(4,1) = 9;
         sm2(4,1) = 9;

         checkRows   ( sm1, 16UL );
         checkColumns( sm1,  8UL );
         checkRows   ( sm2, 16UL );
         checkColumns( sm2,  8UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (10,3)
      {
         sm1(10,3) = 0;
         sm2(10,3) = 0;

         checkRows   ( sm1, 16UL );
         checkColumns( sm1,  8UL );
         checkRows   ( sm2, 16UL );
         checkColumns( sm2,  8UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (8,6)
      {
         sm1(8,6) = -7;
         sm2(8,6) = -7;

         checkRows   ( sm1, 16UL );
         checkColumns( sm1,  8UL );
         checkRows   ( sm2, 16UL );
         checkColumns( sm2,  8UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Addition assignment to the element (7,5)
      {
         sm1(7,5) += 3;
         sm2(7,5) += 3;

         checkRows   ( sm1, 16UL );
         checkColumns( sm1,  8UL );
         checkRows   ( sm2, 16UL );
         checkColumns( sm2,  8UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Subtraction assignment to the element (14,2)
      {
         sm1(14,2) -= -8;
         sm2(14,2) -= -8;

         checkRows   ( sm1, 16UL );
         checkColumns( sm1,  8UL );
         checkRows   ( sm2, 16UL );
         checkColumns( sm2,  8UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Multiplication assignment to the element (1,1)
      {
         sm1(1,1) *= 3;
         sm2(1,1) *= 3;

         checkRows   ( sm1, 16UL );
         checkColumns( sm1,  8UL );
         checkRows   ( sm2, 16UL );
         checkColumns( sm2,  8UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Division assignment to the element (4,3)
      {
         sm1(4,3) /= 2;
         sm2(4,3) /= 2;

         checkRows   ( sm1, 16UL );
         checkColumns( sm1,  8UL );
         checkRows   ( sm2, 16UL );
         checkColumns( sm2,  8UL );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the dilatedsubmatrix iterator implementation.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the iterator implementation of the dilatedsubmatrix class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testIterator()
{
   using blaze::dilatedsubmatrix;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major dilatedsubmatrix tests
   //=====================================================================================

   {
      initialize();

      // Testing the Iterator default constructor
      {
         test_ = "Row-major Iterator default constructor";

         DSMT::Iterator it{};

         if( it != DSMT::Iterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing the ConstIterator default constructor
      {
         test_ = "Row-major ConstIterator default constructor";

         DSMT::ConstIterator it{};

         if( it != DSMT::ConstIterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing conversion from Iterator to ConstIterator
      {
         test_ = "Row-major Iterator/ConstIterator conversion";

         DSMT sm = dilatedsubmatrix( mat1_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         DSMT::ConstIterator it( begin( sm, 2UL) );

         if( it == end( sm, 2UL ) || *it != sm(2,0) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 0th row of a 8x16 dilated matrix via Iterator (end-begin)
      {
         test_ = "Row-major Iterator subtraction (end-begin)";

         DSMT sm = dilatedsubmatrix( mat1_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         const ptrdiff_t number( end( sm, 0UL ) - begin( sm, 0UL ) );

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

      // Counting the number of elements in 0th row of a 8x16 matrix via Iterator (begin-end)
      {
         test_ = "Row-major Iterator subtraction (begin-end)";

         DSMT sm = dilatedsubmatrix( mat1_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         const ptrdiff_t number( begin( sm, 0UL ) - end( sm, 0UL ) );

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

      // Counting the number of elements in 15th row of a 16x8 matrix via ConstIterator (end-begin)
      {
         test_ = "Row-major ConstIterator subtraction (end-begin)";

         DSMT sm = dilatedsubmatrix( mat2_, 8UL, 16UL, 16UL, 8UL, 2UL, 3UL );
         const ptrdiff_t number( cend( sm, 15UL ) - cbegin( sm, 15UL ) );

         if( number != 8L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 8\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 15th row of a 16x8 matrix via ConstIterator (begin-end)
      {
         test_ = "Row-major ConstIterator subtraction (begin-end)";

         DSMT sm = dilatedsubmatrix( mat2_, 8UL, 16UL, 16UL, 8UL, 2UL, 3UL );
         const ptrdiff_t number( cbegin( sm, 15UL ) - cend( sm, 15UL ) );

         if( number != -8L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: -8\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing read-only access via ConstIterator
      {
         test_ = "Row-major read-only access via ConstIterator";

         DSMT sm = dilatedsubmatrix( mat1_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         DSMT::ConstIterator it ( cbegin( sm, 2UL ) );
         DSMT::ConstIterator end( cend( sm, 2UL ) );

         if( it == end || *it != sm(2,0) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || *it != sm(2,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         --it;

         if( it == end || *it != sm(2,0) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it++;

         if( it == end || *it != sm(2,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it--;

         if( it == end || *it != sm(2,0) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it += 2UL;

         if( it == end || *it != sm(2,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator addition assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it -= 2UL;

         if( it == end || *it != sm(2,0) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator subtraction assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it + 2UL;

         if( it == end || *it != sm(2,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar addition failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it - 2UL;

         if( it == end || *it != sm(2,0) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar subtraction failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = 16UL + it;

         if( it != end ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Scalar/iterator addition failed\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing assignment via Iterator
      {
         test_ = "Row-major assignment via Iterator";

         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         RCMT sm1 = blaze::rows( blaze::columns( mat1_, column_indices.data(),
                                    column_indices.size() ),
            row_indices.data(), row_indices.size() );
         DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         int value = 7;

         RCMT::Iterator it1( begin( sm1, 2UL ) );
         DSMT::Iterator it2( begin( sm2, 2UL ) );

         for( ; it1!=end( sm1, 2UL ); ++it1, ++it2 ) {
            *it1 = value;
            *it2 = value;
            ++value;
         }

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "Row-major addition assignment via Iterator";

         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         RCMT sm1 = blaze::rows( blaze::columns( mat1_, column_indices.data(),
                                    column_indices.size() ),
            row_indices.data(), row_indices.size() );
         DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         int value = 4;

         RCMT::Iterator it1( begin( sm1, 2UL ) );
         DSMT::Iterator it2( begin( sm2, 2UL ) );

         for( ; it1!=end( sm1, 2UL ); ++it1, ++it2 ) {
            *it1 += value;
            *it2 += value;
            ++value;
         }

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "Row-major subtraction assignment via Iterator";

         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         RCMT sm1 = blaze::rows( blaze::columns( mat1_, column_indices.data(),
                                    column_indices.size() ),
            row_indices.data(), row_indices.size() );
         DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         int value = 4;

         RCMT::Iterator it1( begin( sm1, 2UL ) );
         DSMT::Iterator it2( begin( sm2, 2UL ) );

         for( ; it1!=end( sm1, 2UL ); ++it1, ++it2 ) {
            *it1 -= value;
            *it2 -= value;
            ++value;
         }

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "Row-major multiplication assignment via Iterator";

         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         RCMT sm1 = blaze::rows( blaze::columns( mat1_, column_indices.data(),
                                    column_indices.size() ),
            row_indices.data(), row_indices.size() );
         DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         int value = 2;

         RCMT::Iterator it1( begin( sm1, 2UL ) );
         DSMT::Iterator it2( begin( sm2, 2UL ) );

         for( ; it1!=end( sm1, 2UL ); ++it1, ++it2 ) {
            *it1 *= value;
            *it2 *= value;
            ++value;
         }

         if( sm1 != sm2 ||  mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "Row-major division assignment via Iterator";

         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         RCMT sm1 = blaze::rows( blaze::columns( mat1_, column_indices.data(),
                                    column_indices.size() ),
            row_indices.data(), row_indices.size() );
         DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

         RCMT::Iterator it1( begin( sm1, 2UL ) );
         DSMT::Iterator it2( begin( sm2, 2UL ) );

         for( ; it1!=end( sm1, 2UL ); ++it1, ++it2 ) {
            *it1 /= 2;
            *it2 /= 2;
         }

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major dilatedsubmatrix tests
   //=====================================================================================

   {
      initialize();

      // Testing the Iterator default constructor
      {
         test_ = "Column-major Iterator default constructor";

         ODSMT::Iterator it{};

         if( it != ODSMT::Iterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing the ConstIterator default constructor
      {
         test_ = "Column-major ConstIterator default constructor";

         ODSMT::ConstIterator it{};

         if( it != ODSMT::ConstIterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing conversion from Iterator to ConstIterator
      {
         test_ = "Column-major Iterator/ConstIterator conversion";

         ODSMT sm = dilatedsubmatrix( tmat1_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
         ODSMT::ConstIterator it( begin( sm, 2UL ) );

         if( it == end( sm, 2UL ) || *it != sm(0,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 0th column of a 16x8 matrix via Iterator (end-begin)
      {
         test_ = "Column-major Iterator subtraction (end-begin)";

         ODSMT sm = dilatedsubmatrix( tmat1_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
         const ptrdiff_t number( end( sm, 0UL ) - begin( sm, 0UL ) );

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

      // Counting the number of elements in 0th column of a 16x8 matrix via Iterator (begin-end)
      {
         test_ = "Column-major Iterator subtraction (begin-end)";

         ODSMT sm = dilatedsubmatrix( tmat1_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
         const ptrdiff_t number( begin( sm, 0UL ) - end( sm, 0UL ) );

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

      // Counting the number of elements in 15th column of a 8x16 matrix via ConstIterator (end-begin)
      {
         test_ = "Column-major ConstIterator subtraction (end-begin)";

         ODSMT sm = dilatedsubmatrix( tmat1_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL  );
         const ptrdiff_t number( cend( sm, 15UL ) - cbegin( sm, 15UL ) );

         if( number != 8L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 8\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 15th column of a 8x16 matrix via ConstIterator (begin-end)
      {
         test_ = "Column-major ConstIterator subtraction (begin-end)";

         ODSMT sm = dilatedsubmatrix( tmat1_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         const ptrdiff_t number( cbegin( sm, 15UL ) - cend( sm, 15UL ) );

         if( number != -8L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: -8\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing read-only access via ConstIterator
      {
         test_ = "Column-major read-only access via ConstIterator";

         ODSMT sm = dilatedsubmatrix( tmat1_, 8UL, 16UL, 16UL, 8UL, 2UL, 3UL );
         ODSMT::ConstIterator it ( cbegin( sm, 2UL ) );
         ODSMT::ConstIterator end( cend( sm, 2UL ) );

         if( it == end || *it != sm(0,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || *it != sm(1,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         --it;

         if( it == end || *it != sm(0,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it++;

         if( it == end || *it != sm(1,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it--;

         if( it == end || *it != sm(0,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it += 2UL;

         if( it == end || *it != sm(2,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator addition assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it -= 2UL;

         if( it == end || *it != sm(0,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator subtraction assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it + 2UL;

         if( it == end || *it != sm(2,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar addition failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it - 2UL;

         if( it == end || *it != sm(0,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar subtraction failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = 16UL + it;

         if( it != end ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Scalar/iterator addition failed\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing assignment via Iterator
      {
         test_ = "Column-major assignment via Iterator";

         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         OCRMT sm1 = blaze::columns(
            blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
            column_indices.data(), column_indices.size() );
         ODSMT sm2 = dilatedsubmatrix( tmat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         int value = 7;

         OCRMT::Iterator it1( begin( sm1, 2UL ) );
         ODSMT::Iterator it2( begin( sm2, 2UL ) );

         for( ; it1!=end( sm1, 2UL ); ++it1, ++it2 ) {
            *it1 = value;
            *it2 = value;
            ++value;
         }

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "Column-major addition assignment via Iterator";

         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         OCRMT sm1 = blaze::columns(
            blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
            column_indices.data(), column_indices.size() );
         ODSMT sm2 = dilatedsubmatrix( tmat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         int value = 4;

         OCRMT::Iterator it1( begin( sm1, 2UL ) );
         ODSMT::Iterator it2( begin( sm2, 2UL ) );

         for( ; it1!=end( sm1, 2UL ); ++it1, ++it2 ) {
            *it1 += value;
            *it2 += value;
            ++value;
         }

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "Column-major subtraction assignment via Iterator";

         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         OCRMT sm1 = blaze::columns(
            blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
            column_indices.data(), column_indices.size() );
         ODSMT sm2 = dilatedsubmatrix( tmat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         int value = 4;

         OCRMT::Iterator it1( begin( sm1, 2UL ) );
         ODSMT::Iterator it2( begin( sm2, 2UL ) );

         for( ; it1!=end( sm1, 2UL ); ++it1, ++it2 ) {
            *it1 -= value;
            *it2 -= value;
            ++value;
         }

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "Column-major multiplication assignment via Iterator";

         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         OCRMT sm1 = blaze::columns(
            blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
            column_indices.data(), column_indices.size() );
         ODSMT sm2 = dilatedsubmatrix( tmat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         int value = 2;

         OCRMT::Iterator it1( begin( sm1, 2UL ) );
         ODSMT::Iterator it2( begin( sm2, 2UL ) );

         for( ; it1!=end( sm1, 2UL ); ++it1, ++it2 ) {
            *it1 *= value;
            *it2 *= value;
            ++value;
         }

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "Column-major division assignment via Iterator";

         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         OCRMT sm1 = blaze::columns(
            blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
            column_indices.data(), column_indices.size() );
         ODSMT sm2 = dilatedsubmatrix( tmat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

         OCRMT::Iterator it1( begin( sm1, 2UL ) );
         ODSMT::Iterator it2( begin( sm2, 2UL ) );

         for( ; it1!=end( sm1, 2UL ); ++it1, ++it2 ) {
            *it1 /= 2;
            *it2 /= 2;
         }

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c nonZeros() member function of the dilatedsubmatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c nonZeros() member function of the dilatedsubmatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testNonZeros()
{
   using blaze::dilatedsubmatrix;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major dilatedsubmatrix tests
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubmatrix::nonZeros()";

      initialize();

      // Initialization check
      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1.nonZeros() != sm2.nonZeros() ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid number of non-zeros\n"
             << " Details:\n"
             << "   Result:\n" << sm1.nonZeros() << "\n"
             << "   Expected result:\n" << sm2.nonZeros() << "\n"
             << "   dilatedsubmatrix:\n" << sm1 << "\n"
             << "   Reference:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }

      for( size_t i=0UL; i<sm1.rows(); ++i ) {
         if( sm1.nonZeros(i) != sm2.nonZeros(i) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of non-zeros in row " << i << "\n"
                << " Details:\n"
                << "   Result:\n" << sm1.nonZeros(i) << "\n"
                << "   Expected result:\n" << sm2.nonZeros(i) << "\n"
                << "   Submatrix:\n" << sm1 << "\n"
                << "   Reference:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major dilatedsubmatrix tests
   //=====================================================================================

   {
      test_ = "Column-major dilatedsubmatrix::nonZeros()";

      initialize();

      // Initialization check
      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1.nonZeros() != sm2.nonZeros() ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid number of non-zeros\n"
             << " Details:\n"
             << "   Result:\n" << sm1.nonZeros() << "\n"
             << "   Expected result:\n" << sm2.nonZeros() << "\n"
             << "   dilatedsubmatrix:\n" << sm1 << "\n"
             << "   Reference:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }

      for( size_t j=0UL; j<sm1.columns(); ++j ) {
         if( sm1.nonZeros(j) != sm2.nonZeros(j) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of non-zeros in column " << j << "\n"
                << " Details:\n"
                << "   Result:\n" << sm1.nonZeros(j) << "\n"
                << "   Expected result:\n" << sm2.nonZeros(j) << "\n"
                << "   Submatrix:\n" << sm1 << "\n"
                << "   Reference:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reset() member function of the dilatedsubmatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reset() member function of the dilatedsubmatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testReset()
{
   using blaze::dilatedsubmatrix;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::reset;


   //=====================================================================================
   // Row-major single element reset
   //=====================================================================================

   {
      test_ = "Row-major reset() function";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      reset( sm1(4,4) );
      reset( sm2(4,4) );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major reset
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubmatrix::reset() (lvalue)";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      reset( sm1 );
      reset( sm2 );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( !isDefault( sm1 ) || !isDefault( sm2 ) || sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major dilatedsubmatrix::reset() (rvalue)";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );
      reset( blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() ) );
      reset( dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL ) );

      if( mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat1_ << "\n"
             << "   Expected result:\n" << mat2_ << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major row-wise reset
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubmatrix::reset( size_t )";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      for( size_t i=0UL; i<sm1.rows(); ++i )
      {
         reset( sm1, i );
         reset( sm2, i );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major single element reset
   //=====================================================================================

   {
      test_ = "Column-major reset() function";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      reset( sm1(4,4) );
      reset( sm2(4,4) );

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major reset
   //=====================================================================================

   {
      test_ = "Column-major dilatedsubmatrix::reset() (lvalue)";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      reset( sm1 );
      reset( sm2 );

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( !isDefault( sm1 ) || !isDefault( sm2 ) || sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major dilatedsubmatrix::reset() (rvalue)";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
      reset( blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() ) );
      reset( dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL ) );

      if( mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat1_ << "\n"
             << "   Expected result:\n" << mat2_ << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major row-wise reset
   //=====================================================================================

   {
      test_ = "Column-major dilatedsubmatrix::reset( size_t )";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      for( size_t j=0UL; j<sm1.columns(); ++j )
      {
         reset( sm1, j );
         reset( sm2, j );

         if( sm1 != sm2 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm1 << "\n"
                << "   Expected result:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c clear() function with the dilatedsubmatrix specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c clear() function with the dilatedsubmatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testClear()
{
   using blaze::dilatedsubmatrix;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::clear;


   //=====================================================================================
   // Row-major single element clear
   //=====================================================================================

   {
      test_ = "Row-major clear() function";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      clear( sm1(4,4) );
      clear( sm2(4,4) );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major clear
   //=====================================================================================

   {
      test_ = "Row-major clear() function (lvalue)";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

      clear( sm1 );
      clear( sm2 );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 16UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 16UL );

      if( !isDefault( sm1 ) || !isDefault( sm2 ) || sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major clear() function (rvalue)";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 16UL, 2UL );

      clear( blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() ) );
      clear( dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL ) );

      if( mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat1_ << "\n"
             << "   Expected result:\n" << mat2_ << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major single element clear
   //=====================================================================================

   {
      test_ = "Column-major clear() function";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      clear( sm1(4,4) );
      clear( sm2(4,4) );

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major clear
   //=====================================================================================

   {
      test_ = "Column-major clear() function (lvalue)";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      OCRMT sm1 = blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() );
      ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );

      clear( sm1 );
      clear( sm2 );

      checkRows   ( sm1, 16UL );
      checkColumns( sm1,  8UL );
      checkRows   ( sm2, 16UL );
      checkColumns( sm2,  8UL );

      if( !isDefault( sm1 ) || !isDefault( sm2 ) || sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major clear() function (rvalue)";

      initialize();

      auto row_indices = generate_indices( 16UL, 16UL, 2UL );
      auto column_indices = generate_indices( 8UL, 8UL, 3UL );
      clear( blaze::columns(
         blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
         column_indices.data(), column_indices.size() ) );
      clear( dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL ) );

      if( mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat1_ << "\n"
             << "   Expected result:\n" << mat2_ << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c transpose() member function of the dilatedsubmatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c transpose() member function of the dilatedsubmatrix
// specialization. Additionally, it performs a test of self-transpose via the \c trans()
// function. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testTranspose()
{
   using blaze::dilatedsubmatrix;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major dilatedsubmatrix tests
   //=====================================================================================

   {
      test_ = "Row-major self-transpose via transpose()";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 8UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 8UL, 3UL, 2UL );

      transpose( sm1 );
      transpose( sm2 );

      checkRows   ( sm1, 8UL );
      checkColumns( sm1, 8UL );
      checkRows   ( sm2, 8UL );
      checkColumns( sm2, 8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-transpose via trans()";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 8UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 8UL, 3UL, 2UL );

      sm1 = trans( sm1 );
      sm2 = trans( sm2 );

      checkRows   ( sm1, 8UL );
      checkColumns( sm1, 8UL );
      checkRows   ( sm2, 8UL );
      checkColumns( sm2, 8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


//   //=====================================================================================
//   // Column-major dilatedsubmatrix tests
//   //=====================================================================================
//
//   {
//      test_ = "Column-major self-transpose via transpose()";
//
//      initialize();
//
      //auto row_indices = generate_indices( 16UL, 8UL, 2UL );
      //auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      //OCRMT sm1 = blaze::columns(
      //   blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
      //   column_indices.data(), column_indices.size() );
      //ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 8UL, 8UL, 2UL, 3UL );
//
//      transpose( sm1 );
//      transpose( sm2 );
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Transpose operation failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major self-transpose via trans()";
//
//      initialize();
//
      //auto row_indices = generate_indices( 16UL, 8UL, 2UL );
      //auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      //OCRMT sm1 = blaze::columns(
      //   blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
      //   column_indices.data(), column_indices.size() );
      //ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 8UL, 8UL, 2UL, 3UL );
//
//      sm1 = trans( sm1 );
//      sm2 = trans( sm2 );
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Transpose operation failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c ctranspose() member function of the dilatedsubmatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c ctranspose() member function of the dilatedsubmatrix
// class template. Additionally, it performs a test of self-transpose via the \c ctrans()
// function. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testCTranspose()
{
   using blaze::dilatedsubmatrix;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major dilatedsubmatrix tests
   //=====================================================================================

   {
      test_ = "Row-major self-transpose via ctranspose()";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 8UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 8UL, 3UL, 2UL );

      ctranspose( sm1 );
      ctranspose( sm2 );

      checkRows   ( sm1, 8UL );
      checkColumns( sm1, 8UL );
      checkRows   ( sm2, 8UL );
      checkColumns( sm2, 8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-transpose via ctrans()";

      initialize();

      auto row_indices = generate_indices( 8UL, 8UL, 3UL );
      auto column_indices = generate_indices( 16UL, 8UL, 2UL );

      RCMT sm1 = blaze::rows(
         blaze::columns( mat1_, column_indices.data(), column_indices.size() ),
         row_indices.data(), row_indices.size() );
      DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 8UL, 3UL, 2UL );

      sm1 = ctrans( sm1 );
      sm2 = ctrans( sm2 );

      checkRows   ( sm1, 8UL );
      checkColumns( sm1, 8UL );
      checkRows   ( sm2, 8UL );
      checkColumns( sm2, 8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


//   //=====================================================================================
//   // Column-major dilatedsubmatrix tests
//   //=====================================================================================
//
//   {
//      test_ = "Column-major self-transpose via ctranspose()";
//
//      initialize();
//
      //auto row_indices = generate_indices( 16UL, 8UL, 2UL );
      //auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      //OCRMT sm1 = blaze::columns(
      //   blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
      //   column_indices.data(), column_indices.size() );
      //ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 8UL, 8UL, 2UL, 3UL );
//
//      ctranspose( sm1 );
//      ctranspose( sm2 );
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Transpose operation failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
//
//   {
//      test_ = "Column-major self-transpose via ctrans()";
//
//      initialize();
//
      //auto row_indices = generate_indices( 16UL, 8UL, 2UL );
      //auto column_indices = generate_indices( 8UL, 8UL, 3UL );

      //OCRMT sm1 = blaze::columns(
      //   blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
      //   column_indices.data(), column_indices.size() );
      //ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 8UL, 8UL, 2UL, 3UL );
//
//      sm1 = ctrans( sm1 );
//      sm2 = ctrans( sm2 );
//
//      checkRows   ( sm1, 8UL );
//      checkColumns( sm1, 8UL );
//      checkRows   ( sm2, 8UL );
//      checkColumns( sm2, 8UL );
//
//      if( sm1 != sm2 || mat1_ != mat2_ ) {
//         std::ostringstream oss;
//         oss << " Test: " << test_ << "\n"
//             << " Error: Transpose operation failed\n"
//             << " Details:\n"
//             << "   Result:\n" << sm1 << "\n"
//             << "   Expected result:\n" << sm2 << "\n";
//         throw std::runtime_error( oss.str() );
//      }
//   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDefault() function with the dilatedsubmatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDefault() function with the dilatedsubmatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testIsDefault()
{
   using blaze::dilatedsubmatrix;
   using blaze::aligned;
   using blaze::isDefault;


   //=====================================================================================
   // Row-major dilatedsubmatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isDefault() function";

      initialize();

      // isDefault with default dilatedsubmatrix
      {
         MT mat( 64UL, 64UL, 0 );
         DSMT sm = dilatedsubmatrix( mat, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

         if( isDefault( sm(4,4) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   dilatedsubmatrix element: " << sm(4,4) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( sm ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default dilatedsubmatrix
      {
         DSMT sm = dilatedsubmatrix( mat1_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

         if( isDefault( sm ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major dilatedsubmatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isDefault() function";

      initialize();

      // isDefault with default dilatedsubmatrix
      {
         OMT mat( 64UL, 64UL, 0 );
         ODSMT sm = dilatedsubmatrix( mat, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

         if( isDefault( sm(4,4) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   dilatedsubmatrix element: " << sm(4,4) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( sm ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default dilatedsubmatrix
      {
         ODSMT sm = dilatedsubmatrix( tmat1_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

         if( isDefault( sm ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isSame() function with the dilatedsubmatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSame() function with the dilatedsubmatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testIsSame()
{
   using blaze::dilatedsubmatrix;


   //=====================================================================================
   // Row-major matrix-based tests
   //=====================================================================================

   {
      test_ = "Row-major isSame() function (matrix-based)";

      // isSame with matrix and matching dilatedsubmatrix
      {
         DSMT sm = dilatedsubmatrix( mat1_, 0UL, 0UL, 64UL, 64UL, 1UL, 1UL );

         if( blaze::isSame( sm, mat1_ ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat1_, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching dilatedsubmatrix (different number of rows)
      {
         DSMT sm = dilatedsubmatrix( mat1_, 0UL, 0UL, 32UL, 64UL, 1UL, 1UL );

         if( blaze::isSame( sm, mat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching dilatedsubmatrix (different number of columns)
      {
         DSMT sm = dilatedsubmatrix( mat1_, 0UL, 0UL, 64UL, 32UL, 1UL, 1UL );

         if( blaze::isSame( sm, mat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching dilatedsubmatrix (different row index)
      {
         DSMT sm = dilatedsubmatrix( mat1_, 4UL, 0UL, 60UL, 64UL, 1UL, 1UL );

         if( blaze::isSame( sm, mat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching dilatedsubmatrix (different column index)
      {
         DSMT sm = dilatedsubmatrix( mat1_, 0UL, 4UL, 64UL, 60UL, 1UL, 1UL );

         if( blaze::isSame( sm, mat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching dilatedsubmatrix (different rowdilation)
      {
         DSMT sm = dilatedsubmatrix( mat1_, 0UL, 0UL, 32UL, 64UL, 2UL, 1UL );

         if( blaze::isSame( sm, mat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching dilatedsubmatrix (different columndilation)
      {
         DSMT sm = dilatedsubmatrix( mat1_, 0UL, 0UL, 64UL, 32UL, 1UL, 2UL );

         if( blaze::isSame( sm, mat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching dilatedsubmatrices
      {
         DSMT sm1 = dilatedsubmatrix( mat1_, 16UL, 0UL, 8UL, 16UL, 3UL, 2UL );
         DSMT sm2 = dilatedsubmatrix( mat1_, 16UL, 0UL, 8UL, 16UL, 3UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of rows)
      {
         DSMT sm1 = dilatedsubmatrix( mat1_, 16UL, 0UL,  8UL, 16UL, 3UL, 2UL );
         DSMT sm2 = dilatedsubmatrix( mat1_, 16UL, 0UL, 10UL, 16UL, 3UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of columns)
      {
         DSMT sm1 = dilatedsubmatrix( mat1_, 16UL, 0UL, 8UL, 24UL, 3UL, 2UL );
         DSMT sm2 = dilatedsubmatrix( mat1_, 16UL, 0UL, 8UL, 16UL, 3UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different row index)
      {
         DSMT sm1 = dilatedsubmatrix( mat1_,  8UL, 0UL, 8UL, 16UL, 3UL, 2UL );
         DSMT sm2 = dilatedsubmatrix( mat1_, 16UL, 0UL, 8UL, 16UL, 3UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different column index)
      {
         DSMT sm1 = dilatedsubmatrix( mat1_,  8UL,  0UL, 8UL, 16UL, 3UL, 2UL );
         DSMT sm2 = dilatedsubmatrix( mat1_,  8UL, 10UL, 8UL, 16UL, 3UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major rows-based tests
   //=====================================================================================

   {
      test_ = "Row-major isSame() function (rows-based)";

      // isSame with row selection and matching dilatedsubmatrix
      {
         auto rs = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( rs, 0UL, 0UL, 4UL, 64UL, 1UL, 1UL );

         if( blaze::isSame( sm, rs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching dilatedsubmatrix (different rowdilation)
      {
         auto rs = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( rs, 0UL, 0UL, 2UL, 64UL, 2UL, 1UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching dilatedsubmatrix (different number of columns)
      {
         auto rs = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( rs, 0UL, 0UL, 4UL, 32UL, 1UL, 1UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching dilatedsubmatrix (different row index)
      {
         auto rs = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( rs, 1UL, 0UL, 3UL, 64UL, 1UL, 1UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching dilatedsubmatrix (different column index)
      {
         auto rs = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( rs, 0UL, 16UL, 4UL, 48UL, 1UL, 1UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching dilatedsubmatrices
      {
         auto rs  = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( rs, 0UL, 0UL, 2UL, 8UL, 2UL, 4UL );
         auto sm2 = dilatedsubmatrix( rs, 0UL, 0UL, 2UL, 8UL, 2UL, 4UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of rows)
      {
         auto rs  = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( rs, 0UL, 0UL, 1UL, 8UL, 2UL, 4UL );
         auto sm2 = dilatedsubmatrix( rs, 0UL, 0UL, 2UL, 8UL, 2UL, 4UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of columns)
      {
         auto rs  = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( rs, 0UL, 0UL, 3UL, 32UL, 1UL, 1UL );
         auto sm2 = dilatedsubmatrix( rs, 0UL, 0UL, 3UL, 48UL, 1UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different row index)
      {
         auto rs  = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( rs, 0UL, 0UL, 3UL, 32UL, 1UL, 1UL );
         auto sm2 = dilatedsubmatrix( rs, 1UL, 0UL, 3UL, 32UL, 1UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different column index)
      {
         auto rs  = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( rs, 0UL,  0UL, 3UL, 8UL, 1UL, 2UL );
         auto sm2 = dilatedsubmatrix( rs, 0UL, 16UL, 3UL, 8UL, 1UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major columns-based tests
   //=====================================================================================

   {
      test_ = "Row-major isSame() function (columns-based)";

      // isSame with column selection and matching dilatedsubmatrix
      {
         auto cs = blaze::columns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( cs, 0UL, 0UL, 64UL, 4UL, 1UL, 1UL );

         if( blaze::isSame( sm, cs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching dilatedsubmatrix (different number of rows)
      {
         auto cs = blaze::columns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 4UL, 1UL, 1UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching dilatedsubmatrix (different number of columns)
      {
         auto cs = blaze::columns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( cs, 0UL, 0UL, 64UL, 3UL, 1UL, 1UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching dilatedsubmatrix (different row index)
      {
         auto cs = blaze::columns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( cs, 16UL, 0UL, 48UL, 4UL, 1UL, 1UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching dilatedsubmatrix (different column index)
      {
         auto cs = blaze::columns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( cs, 0UL, 1UL, 64UL, 3UL, 1UL, 1UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching submatrices
      {
         auto cs  = blaze::columns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 3UL, 2UL, 1UL );
         auto sm2 = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 3UL, 2UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of rows)
      {
         auto cs  = blaze::columns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 3UL, 2UL, 1UL );
         auto sm2 = dilatedsubmatrix( cs, 0UL, 0UL,  8UL, 3UL, 2UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of columns)
      {
         auto cs  = blaze::columns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 3UL, 2UL, 1UL );
         auto sm2 = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 2UL, 2UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different row index)
      {
         auto cs  = blaze::columns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( cs,  0UL, 0UL, 8UL, 3UL, 2UL, 1UL );
         auto sm2 = dilatedsubmatrix( cs, 16UL, 0UL, 8UL, 3UL, 2UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different column index)
      {
         auto cs  = blaze::columns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 3UL, 2UL, 1UL );
         auto sm2 = dilatedsubmatrix( cs, 0UL, 1UL, 32UL, 3UL, 2UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix-based tests
   //=====================================================================================

   {
      test_ = "Column-major isSame() function (matrix-based)";

      // isSame with matrix and matching dilatedsubmatrix
      {
         ODSMT sm = dilatedsubmatrix( tmat1_, 0UL, 0UL, 64UL, 64UL, 1UL, 1UL );

         if( blaze::isSame( sm, tmat1_ ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat1_, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching dilatedsubmatrix (different number of rows)
      {
         ODSMT sm = dilatedsubmatrix( tmat1_, 0UL, 0UL, 32UL, 64UL, 1Ul, 1UL );

         if( blaze::isSame( sm, tmat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching dilatedsubmatrix (different number of columns)
      {
         ODSMT sm = dilatedsubmatrix( tmat1_, 0UL, 0UL, 64UL, 32UL, 1UL, 1UL );

         if( blaze::isSame( sm, tmat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching dilatedsubmatrix (different row index)
      {
         ODSMT sm = dilatedsubmatrix( tmat1_, 16UL, 0UL, 48UL, 64UL, 1UL, 1UL );

         if( blaze::isSame( sm, tmat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching dilatedsubmatrix (different column index)
      {
         ODSMT sm = dilatedsubmatrix( tmat1_, 0UL, 16UL, 64UL, 48UL, 1UL, 1UL );

         if( blaze::isSame( sm, tmat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat1_ << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching submatrices
      {
         ODSMT sm1 = dilatedsubmatrix( tmat1_, 16UL, 0UL, 32UL, 16UL, 1UL, 3UL );
         ODSMT sm2 = dilatedsubmatrix( tmat1_, 16UL, 0UL, 32UL, 16UL, 1UL, 3UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of rows)
      {
         ODSMT sm1 = dilatedsubmatrix( tmat1_, 16UL, 0UL,  8UL, 16UL, 2UL, 3UL );
         ODSMT sm2 = dilatedsubmatrix( tmat1_, 16UL, 0UL, 16UL, 16UL, 2UL, 3UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of columns)
      {
         ODSMT sm1 = dilatedsubmatrix( tmat1_, 0UL, 0UL, 32UL, 16UL, 2UL, 1UL );
         ODSMT sm2 = dilatedsubmatrix( tmat1_, 0UL, 0UL, 32UL, 32UL, 2UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different row index)
      {
         ODSMT sm1 = dilatedsubmatrix( tmat1_, 16UL, 0UL, 32UL, 16UL, 1UL, 3UL );
         ODSMT sm2 = dilatedsubmatrix( tmat1_,  0UL, 0UL, 32UL, 16UL, 1UL, 3UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different column index)
      {
         ODSMT sm1 = dilatedsubmatrix( tmat1_, 16UL,  0UL, 32UL, 16UL, 1UL, 2UL );
         ODSMT sm2 = dilatedsubmatrix( tmat1_, 16UL, 16UL, 32UL, 16UL, 1UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major rows-based tests
   //=====================================================================================

   {
      test_ = "Column-major isSame() function (rows-based)";

      // isSame with row selection and matching dilatedsubmatrix
      {
         auto rs = blaze::rows( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( rs, 0UL, 0UL, 4UL, 64UL, 1UL, 1UL );

         if( blaze::isSame( sm, rs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching dilatedsubmatrix (different number of rows)
      {
         auto rs = blaze::rows( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( rs, 0UL, 0UL, 3UL, 64UL, 1UL, 1UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching dilatedsubmatrix (different number of columns)
      {
         auto rs = blaze::rows( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( rs, 0UL, 0UL, 4UL, 32UL, 1UL, 1UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching dilatedsubmatrix (different row index)
      {
         auto rs = blaze::rows( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( rs, 1UL, 0UL, 3UL, 64UL, 1UL, 1UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching dilatedsubmatrix (different column index)
      {
         auto rs = blaze::rows( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( rs, 0UL, 16UL, 4UL, 48UL, 1UL, 1UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching dilatedsubmatrix (different columndilation)
      {
         auto rs = blaze::rows( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( rs, 0UL, 0UL, 4UL, 32UL, 1UL, 2UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching submatrices
      {
         auto rs  = blaze::rows( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( rs, 0UL, 0UL, 3UL, 32UL, 1UL, 2UL );
         auto sm2 = dilatedsubmatrix( rs, 0UL, 0UL, 3UL, 32UL, 1UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of rows)
      {
         auto rs  = blaze::rows( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( rs, 0UL, 0UL, 3UL, 32UL, 1UL, 2UL );
         auto sm2 = dilatedsubmatrix( rs, 0UL, 0UL, 2UL, 32UL, 1UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of columns)
      {
         auto rs  = blaze::rows( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( rs, 0UL, 0UL, 3UL, 32UL, 1UL, 2UL );
         auto sm2 = dilatedsubmatrix( rs, 0UL, 0UL, 3UL,  8UL, 1UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different row index)
      {
         auto rs  = blaze::rows( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( rs, 0UL, 0UL, 3UL, 32UL, 1UL, 2UL );
         auto sm2 = dilatedsubmatrix( rs, 1UL, 0UL, 3UL, 32UL, 1UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different column index)
      {
         auto rs  = blaze::rows( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( rs, 0UL,  0UL, 3UL, 32UL, 1UL, 1UL );
         auto sm2 = dilatedsubmatrix( rs, 0UL, 16UL, 3UL, 48UL, 1UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major columns-based tests
   //=====================================================================================

   {
      test_ = "Column-major isSame() function (columns-based)";

      // isSame with column selection and matching dilatedsubmatrix
      {
         auto cs = blaze::columns( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( cs, 0UL, 0UL, 64UL, 4UL, 1UL, 1UL );

         if( blaze::isSame( sm, cs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching dilatedsubmatrix (different number of rows)
      {
         auto cs = blaze::columns( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 4UL, 1UL, 1UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching dilatedsubmatrix (different number of columns)
      {
         auto cs = blaze::columns( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( cs, 0UL, 0UL, 64UL, 3UL, 1UL, 1UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching dilatedsubmatrix (different row index)
      {
         auto cs = blaze::columns( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( cs, 16UL, 0UL, 48UL, 4UL, 1UL, 1UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching dilatedsubmatrix (different column index)
      {
         auto cs = blaze::columns( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm = dilatedsubmatrix( cs, 0UL, 1UL, 64UL, 3UL, 1UL, 1UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   dilatedsubmatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching submatrices
      {
         auto cs  = blaze::columns( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 3UL, 2UL, 1UL );
         auto sm2 = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 3UL, 2UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of rows)
      {
         auto cs  = blaze::columns( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 3UL, 2UL, 1UL );
         auto sm2 = dilatedsubmatrix( cs, 0UL, 0UL,  8UL, 3UL, 2UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of columns)
      {
         auto cs  = blaze::columns( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 3UL, 2UL, 1UL );
         auto sm2 = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 2UL, 2UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different row index)
      {
         auto cs  = blaze::columns( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( cs,  0UL, 0UL, 32UL, 3UL, 1UL, 1UL );
         auto sm2 = dilatedsubmatrix( cs, 16UL, 0UL, 32UL, 3UL, 1UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different column index)
      {
         auto cs  = blaze::columns( tmat1_, { 0UL, 16UL, 32UL, 48UL } );
         auto sm1 = dilatedsubmatrix( cs, 0UL, 0UL, 32UL, 3UL, 2UL, 1UL );
         auto sm2 = dilatedsubmatrix( cs, 0UL, 1UL, 32UL, 3UL, 2UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubmatrix:\n" << sm1 << "\n"
                << "   Second dilatedsubmatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c dilatedsubmatrix() function with the dilatedsubmatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c dilatedsubmatrix() function with the dilatedsubmatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testDilatedSubmatrix()
{
   using blaze::dilatedsubmatrix;


   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubmatrix() function";

      initialize();

      {
         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         RCMT sm1 = blaze::rows(
            blaze::columns( mat1_, column_indices.data(), column_indices.size()),
            row_indices.data(), row_indices.size() );

         auto sm2 = dilatedsubmatrix( sm1,   0UL,  0UL,  4UL, 8UL, 2UL, 2UL );
         DSMT sm3 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         DSMT sm4 = dilatedsubmatrix( sm3  , 0UL,  0UL,  4UL, 8UL, 2UL, 2UL );

         if( sm2 != sm4 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: dilatedsubmatrix function failed\n"
                << " Details:\n"
                << "   Result:\n" << sm2 << "\n"
                << "   Expected result:\n" << sm4 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm2(1,1) != sm4(1,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result: " << sm2(1,1) << "\n"
                << "   Expected result: " << sm4(1,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *sm2.begin(1UL) != *sm4.begin(1UL) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *sm2.begin(1UL) << "\n"
                << "   Expected result: " << *sm4.begin(1UL) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         DSMT sm1 = dilatedsubmatrix( mat1_,  8UL, 16UL, 16UL, 32UL, 2UL, 1UL );
         DSMT sm2 = dilatedsubmatrix( sm1  , 16UL,  0UL,  8UL,  8UL, 2UL, 1UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         DSMT sm1 = dilatedsubmatrix( mat1_, 8UL, 16UL, 16UL, 32UL, 2UL, 1UL );
         DSMT sm2 = dilatedsubmatrix( sm1  , 8UL, 32UL,  8UL,  8UL, 2UL, 1UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         DSMT sm1 = dilatedsubmatrix( mat1_, 8UL, 16UL, 16UL, 32UL, 1UL, 1UL );
         DSMT sm2 = dilatedsubmatrix( sm1  , 8UL,  0UL, 16UL, 24UL, 1UL, 1UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         DSMT sm1 = dilatedsubmatrix( mat1_, 8UL, 16UL, 16UL, 32UL, 1UL, 1UL );
         DSMT sm2 = dilatedsubmatrix( sm1  , 8UL,  0UL,  8UL, 40UL, 1UL, 1UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major dilatedsubmatrix() function";

      initialize();

      {
         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         OCRMT sm1 = blaze::columns(
            blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
            column_indices.data(), column_indices.size() );
         auto  sm2 = dilatedsubmatrix(    sm1,  0UL, 8UL, 8UL,  4UL, 1UL, 2UL );
         ODSMT sm3 = dilatedsubmatrix( tmat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );
         ODSMT sm4 = dilatedsubmatrix(    sm3,  0UL, 8UL, 8UL,  4UL, 1UL, 2UL );

         if( sm2 != sm4 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: dilatedsubmatrix function failed\n"
                << " Details:\n"
                << "   Result:\n" << sm2 << "\n"
                << "   Expected result:\n" << sm4 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm2(1,1) != sm4(1,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result: " << sm2(1,1) << "\n"
                << "   Expected result: " << sm4(1,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *sm2.begin(1UL) != *sm4.begin(1UL) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *sm2.begin(1UL) << "\n"
                << "   Expected result: " << *sm4.begin(1UL) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         ODSMT sm1 = dilatedsubmatrix( tmat1_, 16UL, 8UL, 32UL, 16UL, 1UL, 2UL );
         ODSMT sm2 = dilatedsubmatrix( sm1   , 32UL, 8UL,  8UL,  8UL, 1UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ODSMT sm1 = dilatedsubmatrix( tmat1_, 16UL,  8UL, 32UL, 16UL, 1UL, 2UL );
         ODSMT sm2 = dilatedsubmatrix( sm1   ,  0UL, 16UL,  8UL,  8UL, 1UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ODSMT sm1 = dilatedsubmatrix( tmat1_, 16UL, 8UL, 32UL, 16UL, 1UL, 1UL );
         ODSMT sm2 = dilatedsubmatrix( sm1   ,  0UL, 8UL, 40UL,  8UL, 1UL, 1UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ODSMT sm1 = dilatedsubmatrix( tmat1_, 16UL, 8UL, 32UL, 16UL, 1UL, 1UL );
         ODSMT sm2 = dilatedsubmatrix( sm1   ,  0UL, 8UL, 24UL, 16UL, 1UL, 1UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds dilatedsubmatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c row() function with the dilatedsubmatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c row() function with the dilatedsubmatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testRow()
{
   using blaze::dilatedsubmatrix;
   using blaze::row;


   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major row() function";

      initialize();

      {
         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         RCMT sm1 = blaze::rows(
            blaze::columns( mat1_, column_indices.data(), column_indices.size()),
            row_indices.data(), row_indices.size() );
         DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

         auto row1 = row( sm1, 1UL );
         auto row2 = row( sm2, 1UL );

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
         DSMT sm1  = dilatedsubmatrix( mat1_, 8UL, 16UL, 8UL, 16UL, 2UL, 2UL );
         auto row8 = row( sm1, 8UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row succeeded\n"
             << " Details:\n"
             << "   Result:\n" << row8 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major row() function";

      initialize();

      {
         auto row_indices = generate_indices( 16UL, 16UL, 2UL );
         auto column_indices = generate_indices( 8UL, 8UL, 3UL );

         OCRMT sm1 = blaze::columns(
            blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
            column_indices.data(), column_indices.size() );
         ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
         auto  row1 = row( sm1, 1UL );
         auto  row2 = row( sm2, 1UL );

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
         ODSMT sm1   = dilatedsubmatrix( tmat1_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
         auto  row16 = row( sm1, 16UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row succeeded\n"
             << " Details:\n"
             << "   Result:\n" << row16 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c rows() function with the dilatedsubmatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c rows() function with the dilatedsubmatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testRows()
{
   using blaze::dilatedsubmatrix;
   using blaze::rows;


   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major rows() function";

      initialize();

      {
         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         RCMT sm1 = blaze::rows(
            blaze::columns( mat1_, column_indices.data(), column_indices.size()),
            row_indices.data(), row_indices.size() );
         DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

         auto rs1 = rows( sm1, { 0UL, 2UL, 4UL, 6UL } );
         auto rs2 = rows( sm2, { 0UL, 2UL, 4UL, 6UL } );

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
         DSMT sm1 = dilatedsubmatrix( mat1_, 8UL, 16UL, 8UL, 16UL, 2UL, 2UL );
         auto rs  = rows( sm1, { 8UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major rows() function";

      initialize();

      {
         auto row_indices = generate_indices( 16UL, 16UL, 2UL );
         auto column_indices = generate_indices( 8UL, 8UL, 3UL );

         OCRMT sm1 = blaze::columns(
            blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
            column_indices.data(), column_indices.size() );
         ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
         auto  rs1 = rows( sm1, { 0UL, 2UL, 4UL, 6UL } );
         auto  rs2 = rows( sm2, { 0UL, 2UL, 4UL, 6UL } );

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
         ODSMT sm1   = dilatedsubmatrix( tmat1_, 16UL, 8UL, 16UL, 8UL, 2uL, 3UL );
         auto  row16 = rows( sm1, { 16UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << row16 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c column() function with the dilatedsubmatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c column() function with the dilatedsubmatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testColumn()
{
   using blaze::dilatedsubmatrix;
   using blaze::column;


   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major column() function";

      initialize();

      {
         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         RCMT sm1 = blaze::rows(
            blaze::columns( mat1_, column_indices.data(), column_indices.size()),
            row_indices.data(), row_indices.size() );
         DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

         auto col1 = column( sm1, 1UL );
         auto col2 = column( sm2, 1UL );

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
         DSMT sm1   = dilatedsubmatrix( mat1_, 8UL, 16UL, 8UL, 16UL, 2UL, 2UL );
         auto col16 = column( sm1, 16UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column succeeded\n"
             << " Details:\n"
             << "   Result:\n" << col16 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major column() function";

      initialize();

      {
         auto row_indices = generate_indices( 16UL, 16UL, 2UL );
         auto column_indices = generate_indices( 8UL, 8UL, 3UL );

         OCRMT sm1 = blaze::columns(
            blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
            column_indices.data(), column_indices.size() );
         ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
         auto  col1 = column( sm1, 1UL );
         auto  col2 = column( sm2, 1UL );

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
         ODSMT sm1  = dilatedsubmatrix( tmat1_, 16UL, 8UL, 16UL, 8UL, 2UL, 2UL );
         auto  col8 = column( sm1, 8UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column succeeded\n"
             << " Details:\n"
             << "   Result:\n" << col8 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c columns() function with the dilatedsubmatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c columns() function with the dilatedsubmatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testColumns()
{
   using blaze::dilatedsubmatrix;
   using blaze::rows;


   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major columns() function";

      initialize();

      {
         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         RCMT sm1 = blaze::rows(
            blaze::columns( mat1_, column_indices.data(), column_indices.size()),
            row_indices.data(), row_indices.size() );
         DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

         auto cs1 = columns( sm1, { 0UL, 2UL, 4UL, 6UL } );
         auto cs2 = columns( sm2, { 0UL, 2UL, 4UL, 6UL } );

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
         DSMT sm1 = dilatedsubmatrix( mat1_, 8UL, 16UL, 8UL, 16UL, 2Ul, 2UL );
         auto cs  = columns( sm1, { 16UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major columns() function";

      initialize();

      {
         auto row_indices = generate_indices( 16UL, 16UL, 2UL );
         auto column_indices = generate_indices( 8UL, 8UL, 3UL );

         OCRMT sm1 = blaze::columns(
            blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
            column_indices.data(), column_indices.size() );
         ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
         auto cs1 = columns( sm1, { 0UL, 2UL, 4UL, 6UL } );
         auto cs2 = columns( sm2, { 0UL, 2UL, 4UL, 6UL } );

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
         ODSMT sm1 = dilatedsubmatrix( tmat1_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
         auto cs  = columns( sm1, { 8UL } );

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
/*!\brief Test of the \c band() function with the dilatedsubmatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c band() function with the dilatedsubmatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testBand()
{
   using blaze::dilatedsubmatrix;
   using blaze::band;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major band() function";

      initialize();

      {
         auto row_indices = generate_indices( 8UL, 8UL, 3UL );
         auto column_indices = generate_indices( 16UL, 16UL, 2UL );

         RCMT sm1 = blaze::rows(
            blaze::columns( mat1_, column_indices.data(), column_indices.size()),
            row_indices.data(), row_indices.size() );
         DSMT sm2 = dilatedsubmatrix( mat2_, 8UL, 16UL, 8UL, 16UL, 3UL, 2UL );

         auto b1 = band( sm1, 1L );
         auto b2 = band( sm2, 1L );

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
         DSMT sm = dilatedsubmatrix( mat1_, 8UL, 16UL, 8UL, 16UL, 2UL, 2UL );
         auto b8 = band( sm, -8L );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds band succeeded\n"
             << " Details:\n"
             << "   Result:\n" << b8 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major band() function";

      initialize();

      {
         auto row_indices = generate_indices( 16UL, 16UL, 2UL );
         auto column_indices = generate_indices( 8UL, 8UL, 3UL );

         OCRMT sm1 = blaze::columns(
            blaze::rows( tmat1_, row_indices.data(), row_indices.size() ),
            column_indices.data(), column_indices.size() );
         ODSMT sm2 = dilatedsubmatrix( tmat2_, 16UL, 8UL, 16UL, 8UL, 2UL, 3UL );
         auto  b1  = band( sm1, 1L );
         auto  b2  = band( sm2, 1L );

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
         ODSMT sm1 = dilatedsubmatrix( tmat1_, 16UL, 8UL, 16UL, 8UL, 2UL, 1UL );
         auto  b8  = band( sm1, 8L );

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
////*************************************************************************************************
//
//


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
   std::cout << "   Running dilatedsubmatrix dense aligned test (part 2)..." << std::endl;

   try
   {
      RUN_DILATEDSUBMATRIX_DENSE_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during dilatedsubmatrix dense aligned test (part 2):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
