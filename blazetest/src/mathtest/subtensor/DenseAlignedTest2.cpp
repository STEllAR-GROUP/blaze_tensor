//=================================================================================================
/*!
//  \file src/mathtest/subtensor/DenseAlignedTest2.cpp
//  \brief Source file for the Subtensor dense aligned test (part 2)
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
#include <blaze/math/Views.h>
#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>

#include <blaze_tensor/math/Views.h>
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
   testSubtensor();
   testRowSlice();
   testRowSlices();
   testColumnSlice();
   testColumnSlices();
   testPageSlice();
   testPageSlices();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of all Subtensor (self-)scaling operations.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all available ways to scale an instance of the Subtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testScaling()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major self-scaling (M*=s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M*=s) (8x12x8)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      sm1 *= 3;
      sm2 *= 3;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  8UL );

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
      test_ = "Row-major self-scaling (M*=s) (8x8x12)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );

      sm1 *= 3;
      sm2 *= 3;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1,  8UL );
      checkPages  ( sm1, 12UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2,  8UL );
      checkPages  ( sm1, 12UL );

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
      test_ = "Row-major self-scaling (M=M*s) (8x12x8)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      sm1 = sm1 * 3;
      sm2 = sm2 * 3;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  8UL );

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
      test_ = "Row-major self-scaling (M=M*s) (8x8x12)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );

      sm1 = sm1 * 3;
      sm2 = sm2 * 3;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1,  8UL );
      checkPages  ( sm1, 12UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2,  8UL );
      checkPages  ( sm1, 12UL );

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
      test_ = "Row-major self-scaling (M=s*M) (8x12x8)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      sm1 = 3 * sm1;
      sm2 = 3 * sm2;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  8UL );

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
      test_ = "Row-major self-scaling (M=s*M) (8x8x12)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );

      sm1 = 3 * sm1;
      sm2 = 3 * sm2;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1,  8UL );
      checkPages  ( sm1, 12UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2,  8UL );
      checkPages  ( sm1, 12UL );

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
      test_ = "Row-major self-scaling (M/=s) (8x12x8)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      sm1 /= 0.5;
      sm2 /= 0.5;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  8UL );

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
      test_ = "Row-major self-scaling (M/=s) (8x8x12)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );

      sm1 /= 0.5;
      sm2 /= 0.5;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1,  8UL );
      checkPages  ( sm1, 12UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2,  8UL );
      checkPages  ( sm1, 12UL );

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
      test_ = "Row-major self-scaling (M=M/s) (8x12x8)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      sm1 = sm1 / 0.5;
      sm2 = sm2 / 0.5;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  8UL );

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
      test_ = "Row-major self-scaling (M=M/s) (8x8x12)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );

      sm1 = sm1 / 0.5;
      sm2 = sm2 / 0.5;

      checkRows   ( sm1,  8UL );
      checkColumns( sm1,  8UL );
      checkPages  ( sm1, 12UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2,  8UL );
      checkPages  ( sm1, 12UL );

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
   // Row-major Subtensor::scale()
   //=====================================================================================

   {
      test_ = "Row-major Subtensor::scale()";

      initialize();

      // Initialization check
      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  8UL );

      // Integral scaling of the tensor
      sm1.scale( 2 );
      sm2.scale( 2 );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  8UL );

      if( sm1 != sm2 || mat1_ != mat2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Integral scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm1 << "\n"
             << "   Expected result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }

      // Floating point scaling of the tensor
      sm1.scale( 0.5 );
      sm2.scale( 0.5 );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm1,  8UL );

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
/*!\brief Test of the Subtensor function call operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the function call operator
// of the Subtensor specialization. In case an error is detected, a \a std::runtime_error
// exception is thrown.
*/
void DenseAlignedTest::testFunctionCall()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major subtensor tests
   //=====================================================================================

   {
      test_ = "Row-major Subtensor::operator()";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      // Assignment to the element (1,4)
      {
         sm1(1,4,0) = 9;
         sm2(1,4,0) = 9;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 12UL );
         checkPages  ( sm1,  8UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 12UL );
         checkPages  ( sm1,  8UL );

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
         sm1(3,10,2) = 0;
         sm2(3,10,2) = 0;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 12UL );
         checkPages  ( sm1,  8UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 12UL );
         checkPages  ( sm1,  8UL );

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
         sm1(6,8,3) = -7;
         sm2(6,8,3) = -7;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 12UL );
         checkPages  ( sm1,  8UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 12UL );
         checkPages  ( sm1,  8UL );

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
         sm1(5,7,2) += 3;
         sm2(5,7,2) += 3;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 12UL );
         checkPages  ( sm1,  8UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 12UL );
         checkPages  ( sm1,  8UL );

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
         sm1(2,14,0) -= -8;
         sm2(2,14,0) -= -8;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 12UL );
         checkPages  ( sm1,  8UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 12UL );
         checkPages  ( sm1,  8UL );

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
         sm1(1,1,3) *= 3;
         sm2(1,1,3) *= 3;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 12UL );
         checkPages  ( sm1,  8UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 12UL );
         checkPages  ( sm1,  8UL );

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
         sm1(3,4,1) /= 2;
         sm2(3,4,1) /= 2;

         checkRows   ( sm1,  8UL );
         checkColumns( sm1, 12UL );
         checkPages  ( sm1,  8UL );
         checkRows   ( sm2,  8UL );
         checkColumns( sm2, 12UL );
         checkPages  ( sm1,  8UL );

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
/*!\brief Test of the Subtensor iterator implementation.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the iterator implementation of the Subtensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testIterator()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major subtensor tests
   //=====================================================================================

   {
      initialize();

      // Testing the Iterator default constructor
      {
         test_ = "Row-major Iterator default constructor";

         ASMT::Iterator it{};

         if( it != ASMT::Iterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing the ConstIterator default constructor
      {
         test_ = "Row-major ConstIterator default constructor";

         ASMT::ConstIterator it{};

         if( it != ASMT::ConstIterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing conversion from Iterator to ConstIterator
      {
         test_ = "Row-major Iterator/ConstIterator conversion";

         ASMT sm = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         ASMT::ConstIterator it( begin( sm, 2UL, 2UL ) );

         if( it == end( sm, 2UL, 2UL ) || *it != sm(2,0,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 0th row/1st page of a 8x12x8 tensor via Iterator (end-begin)
      {
         test_ = "Row-major Iterator subtraction (end-begin)";

         ASMT sm = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         const ptrdiff_t number( end( sm, 0UL, 1UL ) - begin( sm, 0UL, 1UL ) );

         if( number != 12L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 12\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 0th row/1st page of a 8x12x8 tensor via Iterator (begin-end)
      {
         test_ = "Row-major Iterator subtraction (begin-end)";

         ASMT sm = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         const ptrdiff_t number( begin( sm, 0UL, 1UL ) - end( sm, 0UL, 1UL ) );

         if( number != -12L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: -12\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 0th row/1st page of a 8x12x8 tensor via ConstIterator (begin-end)
      {
         test_ = "Row-major ConstIterator subtraction (end-begin)";

         ASMT sm = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         const ptrdiff_t number( cend( sm, 0UL, 1UL ) - cbegin( sm, 0UL, 1UL ) );

         if( number != 12L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 12\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 0th row/1st page of a 8x12x8 tensor via Iterator (begin-end)
      {
         test_ = "Row-major ConstIterator subtraction (begin-end)";

         ASMT sm = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         const ptrdiff_t number( cbegin( sm, 0UL, 1UL ) - cend( sm, 0UL, 1UL ) );

         if( number != -12L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: -12\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing read-only access via ConstIterator
      {
         test_ = "Row-major read-only access via ConstIterator";

         ASMT sm = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         ASMT::ConstIterator it ( cbegin( sm, 2UL, 4UL ) );
         ASMT::ConstIterator end( cend( sm, 2UL, 4UL ) );

         if( it == end || *it != sm(2,0,4) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || *it != sm(2,1,4) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         --it;

         if( it == end || *it != sm(2,0,4) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it++;

         if( it == end || *it != sm(2,1,4) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it--;

         if( it == end || *it != sm(2,0,4) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it += 2UL;

         if( it == end || *it != sm(2,2,4) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator addition assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it -= 2UL;

         if( it == end || *it != sm(2,0,4) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator subtraction assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it + 2UL;

         if( it == end || *it != sm(2,2,4) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar addition failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it - 2UL;

         if( it == end || *it != sm(2,0,4) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar subtraction failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = 12UL + it;

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

         ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         int value = 7;

         ASMT::Iterator it1( begin( sm1, 2UL, 6UL ) );
         USMT::Iterator it2( begin( sm2, 2UL, 6UL ) );

         for( ; it1!=end( sm1, 2UL, 6UL ); ++it1, ++it2 ) {
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
         test_ = "Row-major addition assignment via Iterator";

         ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         int value = 4;

         ASMT::Iterator it1( begin( sm1, 2UL, 6UL ) );
         USMT::Iterator it2( begin( sm2, 2UL, 6UL ) );

         for( ; it1!=end( sm1, 2UL, 6UL ); ++it1, ++it2 ) {
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

         ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         int value = 4;

         ASMT::Iterator it1( begin( sm1, 2UL, 4UL ) );
         USMT::Iterator it2( begin( sm2, 2UL, 4UL ) );

         for( ; it1!=end( sm1, 2UL, 4UL ); ++it1, ++it2 ) {
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
         test_ = "Row-major multiplication assignment via Iterator";

         ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         int value = 2;

         ASMT::Iterator it1( begin( sm1, 3UL, 5UL ) );
         USMT::Iterator it2( begin( sm2, 3UL, 5UL ) );

         for( ; it1!=end( sm1, 3UL, 5UL ); ++it1, ++it2 ) {
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
         test_ = "Row-major division assignment via Iterator";

         ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

         ASMT::Iterator it1( begin( sm1, 2UL, 3UL ) );
         USMT::Iterator it2( begin( sm2, 2UL, 3UL ) );

         for( ; it1!=end( sm1, 2UL, 3UL ); ++it1, ++it2 ) {
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
/*!\brief Test of the \c nonZeros() member function of the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c nonZeros() member function of the Subtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testNonZeros()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major subtensor tests
   //=====================================================================================

   {
      test_ = "Row-major Subtensor::nonZeros()";

      initialize();

      // Initialization check
      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm2,  8UL );

      if( sm1.nonZeros() != sm2.nonZeros() ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid number of non-zeros\n"
             << " Details:\n"
             << "   Result:\n" << sm1.nonZeros() << "\n"
             << "   Expected result:\n" << sm2.nonZeros() << "\n"
             << "   Subtensor:\n" << sm1 << "\n"
             << "   Reference:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }

      for( size_t k=0UL; k<sm1.pages(); ++k ) {
         for( size_t i=0UL; i<sm1.rows(); ++i ) {
            if( sm1.nonZeros(i, k) != sm2.nonZeros(i, k) ) {
               std::ostringstream oss;
               oss << " Test: " << test_ << "\n"
                   << " Error: Invalid number of non-zeros in row " << i << "page " << k << "\n"
                   << " Details:\n"
                   << "   Result:\n" << sm1.nonZeros(i, k) << "\n"
                   << "   Expected result:\n" << sm2.nonZeros(i, k) << "\n"
                   << "   Subtensor:\n" << sm1 << "\n"
                   << "   Reference:\n" << sm2 << "\n";
               throw std::runtime_error( oss.str() );
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reset() member function of the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reset() member function of the Subtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testReset()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::reset;


   //=====================================================================================
   // Row-major single element reset
   //=====================================================================================

   {
      test_ = "Row-major reset() function";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      reset( sm1(4,4,4) );
      reset( sm2(4,4,4) );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm2,  8UL );

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
      test_ = "Row-major Subtensor::reset() (lvalue)";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      reset( sm1 );
      reset( sm2 );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm2,  8UL );

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
      test_ = "Row-major Subtensor::reset() (rvalue)";

      initialize();

      reset( subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL ) );
      reset( subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL ) );

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
      test_ = "Row-major Subtensor::reset( size_t )";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      for( size_t k=0UL; k<sm1.pages(); ++k )
      {
         for( size_t i=0UL; i<sm1.rows(); ++i )
         {
            reset( sm1, i, k );
            reset( sm2, i, k );

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
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c clear() function with the Subtensor specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c clear() function with the Subtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testClear()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::clear;


   //=====================================================================================
   // Row-major single element clear
   //=====================================================================================

   {
      test_ = "Row-major clear() function";

      initialize();

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      clear( sm1(4,4,4) );
      clear( sm2(4,4,4) );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm2,  8UL );

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

      ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
      USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

      clear( sm1 );
      clear( sm2 );

      checkRows   ( sm1,  8UL );
      checkColumns( sm1, 12UL );
      checkPages  ( sm1,  8UL );
      checkRows   ( sm2,  8UL );
      checkColumns( sm2, 12UL );
      checkPages  ( sm2,  8UL );

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

      clear( subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL ) );
      clear( subtensor<unaligned>( mat2_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL ) );

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
/*!\brief Test of the \c transpose() member function of the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c transpose() member function of the Subtensor
// specialization. Additionally, it performs a test of self-transpose via the \c trans()
// function. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testTranspose()
{
//    using blaze::subtensor;
//    using blaze::aligned;
//    using blaze::unaligned;
//
//
//    //=====================================================================================
//    // Row-major subtensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major self-transpose via transpose()";
//
//       initialize();
//
//       ASMT sm1 = subtensor<aligned>  ( mat1_, 8UL, 16UL, 8UL, 8UL );
//       USMT sm2 = subtensor<unaligned>( mat2_, 8UL, 16UL, 8UL, 8UL );
//
//       transpose( sm1 );
//       transpose( sm2 );
//
//       checkRows   ( sm1, 8UL );
//       checkColumns( sm1, 8UL );
//       checkRows   ( sm2, 8UL );
//       checkColumns( sm2, 8UL );
//
//       if( sm1 != sm2 || mat1_ != mat2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Transpose operation failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sm1 << "\n"
//              << "   Expected result:\n" << sm2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major self-transpose via trans()";
//
//       initialize();
//
//       ASMT sm1 = subtensor<aligned>  ( mat1_, 8UL, 16UL, 8UL, 8UL );
//       USMT sm2 = subtensor<unaligned>( mat2_, 8UL, 16UL, 8UL, 8UL );
//
//       sm1 = trans( sm1 );
//       sm2 = trans( sm2 );
//
//       checkRows   ( sm1, 8UL );
//       checkColumns( sm1, 8UL );
//       checkRows   ( sm2, 8UL );
//       checkColumns( sm2, 8UL );
//
//       if( sm1 != sm2 || mat1_ != mat2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Transpose operation failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sm1 << "\n"
//              << "   Expected result:\n" << sm2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c ctranspose() member function of the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c ctranspose() member function of the Subtensor
// class template. Additionally, it performs a test of self-transpose via the \c ctrans()
// function. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testCTranspose()
{
//    using blaze::subtensor;
//    using blaze::aligned;
//    using blaze::unaligned;
//
//
//    //=====================================================================================
//    // Row-major subtensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major self-transpose via ctranspose()";
//
//       initialize();
//
//       ASMT sm1 = subtensor<aligned>  ( mat1_, 8UL, 16UL, 8UL, 8UL );
//       USMT sm2 = subtensor<unaligned>( mat2_, 8UL, 16UL, 8UL, 8UL );
//
//       ctranspose( sm1 );
//       ctranspose( sm2 );
//
//       checkRows   ( sm1, 8UL );
//       checkColumns( sm1, 8UL );
//       checkRows   ( sm2, 8UL );
//       checkColumns( sm2, 8UL );
//
//       if( sm1 != sm2 || mat1_ != mat2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Transpose operation failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sm1 << "\n"
//              << "   Expected result:\n" << sm2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major self-transpose via ctrans()";
//
//       initialize();
//
//       ASMT sm1 = subtensor<aligned>  ( mat1_, 8UL, 16UL, 8UL, 8UL );
//       USMT sm2 = subtensor<unaligned>( mat2_, 8UL, 16UL, 8UL, 8UL );
//
//       sm1 = ctrans( sm1 );
//       sm2 = ctrans( sm2 );
//
//       checkRows   ( sm1, 8UL );
//       checkColumns( sm1, 8UL );
//       checkRows   ( sm2, 8UL );
//       checkColumns( sm2, 8UL );
//
//       if( sm1 != sm2 || mat1_ != mat2_ ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Transpose operation failed\n"
//              << " Details:\n"
//              << "   Result:\n" << sm1 << "\n"
//              << "   Expected result:\n" << sm2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDefault() function with the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDefault() function with the Subtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testIsDefault()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::isDefault;


   //=====================================================================================
   // Row-major subtensor tests
   //=====================================================================================

   {
      test_ = "Row-major isDefault() function";

      initialize();

      // isDefault with default subtensor
      {
         MT mat( 16UL, 16UL, 16UL, 0 );
         ASMT sm = subtensor<aligned>  ( mat, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

         if( isDefault( sm(4,4,4) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Subtensor element: " << sm(4,4,4) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( sm ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Subtensor:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default subtensor
      {
         ASMT sm = subtensor<aligned>  ( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );;

         if( isDefault( sm ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Subtensor:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isSame() function with the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSame() function with the Subtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testIsSame()
{
   using blaze::subtensor;
   using blaze::aligned;


   //=====================================================================================
   // Row-major tensor-based tests
   //=====================================================================================

   {
      test_ = "Row-major isSame() function (tensor-based)";

      // isSame with tensor and matching subtensor
      {
         ASMT sm = subtensor<aligned>  ( mat1_, 0UL, 0UL, 0UL, 16UL, 16UL, 16UL );

         if( blaze::isSame( sm, mat1_ ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   Subtensor:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat1_, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   Subtensor:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with tensor and non-matching subtensor (different number of psumns/pages)
      {
         ASMT sm = subtensor<aligned>  ( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );

         if( blaze::isSame( sm, mat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   Subtensor:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   Subtensor:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with tensor and non-matching subtensor (different number of rows/psumns)
      {
         ASMT sm = subtensor<aligned>  ( mat1_, 4UL, 2UL, 2UL, 12UL, 8UL, 8UL );

         if( blaze::isSame( sm, mat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   Subtensor:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   Subtensor:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with tensor and non-matching subtensor (different row index)
      {
         ASMT sm = subtensor<aligned>  ( mat1_, 4UL, 4UL, 2UL, 8UL, 12UL, 8UL );

         if( blaze::isSame( sm, mat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   Subtensor:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   Subtensor:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with tensor and non-matching subtensor (different psumn index)
      {
         ASMT sm = subtensor<aligned>  ( mat1_, 2UL, 3UL, 2UL, 8UL, 12UL, 8UL );

         if( blaze::isSame( sm, mat1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   Subtensor:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat1_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat1_ << "\n"
                << "   Subtensor:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching subtensors
      {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         ASMT sm2 = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First subtensor:\n" << sm1 << "\n"
                << "   Second subtensor:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching subtensors (different number of rows)
      {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 12UL, 12UL, 8UL );
         ASMT sm2 = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First subtensor:\n" << sm1 << "\n"
                << "   Second subtensor:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching subtensors (different number of psumns)
      {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 8UL, 8UL );
         ASMT sm2 = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First subtensor:\n" << sm1 << "\n"
                << "   Second subtensor:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching subtensors (different number of pages)
      {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 12UL );
         ASMT sm2 = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First subtensor:\n" << sm1 << "\n"
                << "   Second subtensor:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching subtensors (different row index)
      {
         ASMT sm1 = subtensor<aligned>( mat1_, 4UL, 4UL, 2UL, 8UL, 12UL, 8UL );
         ASMT sm2 = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First subtensor:\n" << sm1 << "\n"
                << "   Second subtensor:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching subtensors (different psumn index)
      {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 2UL, 2UL, 8UL, 12UL, 8UL );
         ASMT sm2 = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First subtensor:\n" << sm1 << "\n"
                << "   Second subtensor:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching subtensors (different page index)
      {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 4UL, 4UL, 8UL, 12UL, 8UL );
         ASMT sm2 = subtensor<aligned>( mat1_, 2UL, 4UL, 2UL, 8UL, 12UL, 8UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First subtensor:\n" << sm1 << "\n"
                << "   Second subtensor:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


//    //=====================================================================================
//    // Row-major rows-based tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isSame() function (rows-based)";
//
//       // isSame with row selection and matching subtensor
//       {
//          auto rs = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm = subtensor<aligned>( rs, 0UL, 0UL, 4UL, 64UL );
//
//          if( blaze::isSame( sm, rs ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Row selection:\n" << rs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( rs, sm ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Row selection:\n" << rs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with row selection and non-matching subtensor (different number of rows)
//       {
//          auto rs = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm = subtensor<aligned>( rs, 0UL, 0UL, 3UL, 64UL );
//
//          if( blaze::isSame( sm, rs ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Row selection:\n" << rs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( rs, sm ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Row selection:\n" << rs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with row selection and non-matching subtensor (different number of psumns)
//       {
//          auto rs = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm = subtensor<aligned>( rs, 0UL, 0UL, 4UL, 32UL );
//
//          if( blaze::isSame( sm, rs ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Row selection:\n" << rs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( rs, sm ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Row selection:\n" << rs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with row selection and non-matching subtensor (different row index)
//       {
//          auto rs = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm = subtensor<aligned>( rs, 1UL, 0UL, 3UL, 64UL );
//
//          if( blaze::isSame( sm, rs ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Row selection:\n" << rs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( rs, sm ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Row selection:\n" << rs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with row selection and non-matching subtensor (different psumn index)
//       {
//          auto rs = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm = subtensor<aligned>( rs, 0UL, 16UL, 4UL, 48UL );
//
//          if( blaze::isSame( sm, rs ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Row selection:\n" << rs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( rs, sm ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Row selection:\n" << rs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with matching subtensors
//       {
//          auto rs  = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm1 = subtensor<aligned>( rs, 0UL, 0UL, 3UL, 32UL );
//          auto sm2 = subtensor<aligned>( rs, 0UL, 0UL, 3UL, 32UL );
//
//          if( blaze::isSame( sm1, sm2 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First subtensor:\n" << sm1 << "\n"
//                 << "   Second subtensor:\n" << sm2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with non-matching subtensors (different number of rows)
//       {
//          auto rs  = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm1 = subtensor<aligned>( rs, 0UL, 0UL, 3UL, 32UL );
//          auto sm2 = subtensor<aligned>( rs, 0UL, 0UL, 2UL, 32UL );
//
//          if( blaze::isSame( sm1, sm2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First subtensor:\n" << sm1 << "\n"
//                 << "   Second subtensor:\n" << sm2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with non-matching subtensors (different number of psumns)
//       {
//          auto rs  = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm1 = subtensor<aligned>( rs, 0UL, 0UL, 3UL, 32UL );
//          auto sm2 = subtensor<aligned>( rs, 0UL, 0UL, 3UL, 48UL );
//
//          if( blaze::isSame( sm1, sm2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First subtensor:\n" << sm1 << "\n"
//                 << "   Second subtensor:\n" << sm2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with non-matching subtensors (different row index)
//       {
//          auto rs  = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm1 = subtensor<aligned>( rs, 0UL, 0UL, 3UL, 32UL );
//          auto sm2 = subtensor<aligned>( rs, 1UL, 0UL, 3UL, 32UL );
//
//          if( blaze::isSame( sm1, sm2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First subtensor:\n" << sm1 << "\n"
//                 << "   Second subtensor:\n" << sm2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with non-matching subtensors (different psumn index)
//       {
//          auto rs  = blaze::rows( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm1 = subtensor<aligned>( rs, 0UL,  0UL, 3UL, 32UL );
//          auto sm2 = subtensor<aligned>( rs, 0UL, 16UL, 3UL, 32UL );
//
//          if( blaze::isSame( sm1, sm2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First subtensor:\n" << sm1 << "\n"
//                 << "   Second subtensor:\n" << sm2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//
//    //=====================================================================================
//    // Row-major psumns-based tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major isSame() function (psumns-based)";
//
//       // isSame with psumn selection and matching subtensor
//       {
//          auto cs = blaze::psumns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm = subtensor<aligned>( cs, 0UL, 0UL, 64UL, 4UL );
//
//          if( blaze::isSame( sm, cs ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Column selection:\n" << cs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( cs, sm ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Column selection:\n" << cs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with psumn selection and non-matching subtensor (different number of rows)
//       {
//          auto cs = blaze::psumns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm = subtensor<aligned>( cs, 0UL, 0UL, 32UL, 4UL );
//
//          if( blaze::isSame( sm, cs ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Column selection:\n" << cs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( cs, sm ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Column selection:\n" << cs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with psumn selection and non-matching subtensor (different number of psumns)
//       {
//          auto cs = blaze::psumns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm = subtensor<aligned>( cs, 0UL, 0UL, 64UL, 3UL );
//
//          if( blaze::isSame( sm, cs ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Column selection:\n" << cs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( cs, sm ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Column selection:\n" << cs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with psumn selection and non-matching subtensor (different row index)
//       {
//          auto cs = blaze::psumns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm = subtensor<aligned>( cs, 16UL, 0UL, 48UL, 4UL );
//
//          if( blaze::isSame( sm, cs ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Column selection:\n" << cs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( cs, sm ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Column selection:\n" << cs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with psumn selection and non-matching subtensor (different psumn index)
//       {
//          auto cs = blaze::psumns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm = subtensor<aligned>( cs, 0UL, 1UL, 64UL, 3UL );
//
//          if( blaze::isSame( sm, cs ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Column selection:\n" << cs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( blaze::isSame( cs, sm ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   Column selection:\n" << cs << "\n"
//                 << "   Subtensor:\n" << sm << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with matching subtensors
//       {
//          auto cs  = blaze::psumns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm1 = subtensor<aligned>( cs, 0UL, 0UL, 32UL, 3UL );
//          auto sm2 = subtensor<aligned>( cs, 0UL, 0UL, 32UL, 3UL );
//
//          if( blaze::isSame( sm1, sm2 ) == false ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First subtensor:\n" << sm1 << "\n"
//                 << "   Second subtensor:\n" << sm2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with non-matching subtensors (different number of rows)
//       {
//          auto cs  = blaze::psumns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm1 = subtensor<aligned>( cs, 0UL, 0UL, 32UL, 3UL );
//          auto sm2 = subtensor<aligned>( cs, 0UL, 0UL, 48UL, 3UL );
//
//          if( blaze::isSame( sm1, sm2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First subtensor:\n" << sm1 << "\n"
//                 << "   Second subtensor:\n" << sm2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with non-matching subtensors (different number of psumns)
//       {
//          auto cs  = blaze::psumns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm1 = subtensor<aligned>( cs, 0UL, 0UL, 32UL, 3UL );
//          auto sm2 = subtensor<aligned>( cs, 0UL, 0UL, 32UL, 2UL );
//
//          if( blaze::isSame( sm1, sm2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First subtensor:\n" << sm1 << "\n"
//                 << "   Second subtensor:\n" << sm2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with non-matching subtensors (different row index)
//       {
//          auto cs  = blaze::psumns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm1 = subtensor<aligned>( cs,  0UL, 0UL, 32UL, 3UL );
//          auto sm2 = subtensor<aligned>( cs, 16UL, 0UL, 32UL, 3UL );
//
//          if( blaze::isSame( sm1, sm2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First subtensor:\n" << sm1 << "\n"
//                 << "   Second subtensor:\n" << sm2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       // isSame with non-matching subtensors (different psumn index)
//       {
//          auto cs  = blaze::psumns( mat1_, { 0UL, 16UL, 32UL, 48UL } );
//          auto sm1 = subtensor<aligned>( cs, 0UL, 0UL, 32UL, 3UL );
//          auto sm2 = subtensor<aligned>( cs, 0UL, 1UL, 32UL, 3UL );
//
//          if( blaze::isSame( sm1, sm2 ) == true ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Invalid isSame evaluation\n"
//                 << " Details:\n"
//                 << "   First subtensor:\n" << sm1 << "\n"
//                 << "   Second subtensor:\n" << sm2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c subtensor() function with the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c subtensor() function with the Subtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testSubtensor()
{
   using blaze::subtensor;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major subtensor() function";

      initialize();

      {
         ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
         ASMT sm2 = subtensor<aligned>  ( sm1  , 2UL, 2UL, 2UL, 4UL, 4UL,  4UL );
         USMT sm3 = subtensor<unaligned>( mat2_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
         USMT sm4 = subtensor<unaligned>( sm3  , 2UL, 2UL, 2UL, 4UL, 4UL,  4UL );

         if( sm2 != sm4 || mat1_ != mat2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtensor function failed\n"
                << " Details:\n"
                << "   Result:\n" << sm2 << "\n"
                << "   Expected result:\n" << sm4 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm2(1,1,1) != sm4(1,1,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result: " << sm2(1,1,1) << "\n"
                << "   Expected result: " << sm4(1,1,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *sm2.begin(1UL, 2UL) != *sm4.begin(1UL, 2UL) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *sm2.begin(1UL, 2UL) << "\n"
                << "   Expected result: " << *sm4.begin(1UL, 2UL) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         ASMT sm1 = subtensor<aligned>( mat1_,  2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
         ASMT sm2 = subtensor<aligned>( sm1  , 16UL, 2UL, 2UL, 4UL, 4UL,  4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL,  2UL, 4UL, 8UL, 8UL, 12UL );
         ASMT sm2 = subtensor<aligned>( sm1  , 2UL, 16UL, 2UL, 4UL, 4UL,  4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 2UL,  4UL, 8UL, 8UL, 12UL );
         ASMT sm2 = subtensor<aligned>( sm1  , 2UL, 2UL, 16UL, 4UL, 4UL,  4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 2UL, 4UL,  8UL, 8UL, 12UL );
         ASMT sm2 = subtensor<aligned>( sm1  , 2UL, 2UL, 4UL, 16UL, 4UL,  4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 2UL, 4UL, 8UL,  8UL, 12UL );
         ASMT sm2 = subtensor<aligned>( sm1  , 2UL, 2UL, 4UL, 4UL, 16UL,  4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
         ASMT sm2 = subtensor<aligned>( sm1  , 2UL, 2UL, 4UL, 4UL, 4UL, 16UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds subtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c rowslice() function with the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c rowslice() function with the Subtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testRowSlice()
{
   using blaze::subtensor;
   using blaze::rowslice;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major rowslice() function";

      initialize();

      {
         ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
         USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );

         auto rowslice1 = rowslice( sm1, 1UL );
         auto rowslice2 = rowslice( sm2, 1UL );

         if( rowslice1 != rowslice2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Row function failed\n"
                << " Details:\n"
                << "   Result:\n" << rowslice1 << "\n"
                << "   Expected result:\n" << rowslice2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( rowslice1(1,1) != rowslice2(1,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result: " << rowslice1(1,1) << "\n"
                << "   Expected result: " << rowslice2(1,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *rowslice1.begin( 3UL ) != *rowslice2.begin( 3UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *rowslice1.begin( 3UL ) << "\n"
                << "   Expected result: " << *rowslice2.begin( 3UL ) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
         auto rowslice8 = rowslice( sm1, 8UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds rowslice succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rowslice8 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c rowslices() function with the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c rowslices() function with the Subtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testRowSlices()
{
//    using blaze::subtensor;
//    using blaze::rowslices;
//    using blaze::aligned;
//    using blaze::unaligned;
//
//
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major rowslices() function";
//
//       initialize();
//
//       {
//          ASMT sm1 = subtensor<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//          USMT sm2 = subtensor<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//          auto rs1 = rowslices( sm1, { 0UL, 2UL, 4UL, 6UL } );
//          auto rs2 = rowslices( sm2, { 0UL, 2UL, 4UL, 6UL } );
//
//          if( rs1 != rs2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Rows function failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << rs1 << "\n"
//                 << "   Expected result:\n" << rs2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( rs1(1,1) != rs2(1,1) ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Function call operator access failed\n"
//                 << " Details:\n"
//                 << "   Result: " << rs1(1,1) << "\n"
//                 << "   Expected result: " << rs2(1,1) << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( *rs1.begin( 1UL ) != *rs2.begin( 1UL ) ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Iterator access failed\n"
//                 << " Details:\n"
//                 << "   Result: " << *rs1.begin( 1UL ) << "\n"
//                 << "   Expected result: " << *rs2.begin( 1UL ) << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       try {
//          ASMT sm1 = subtensor<aligned>( mat1_, 8UL, 16UL, 8UL, 16UL );
//          auto rs  = rowslices( sm1, { 8UL } );
//
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Setup of out-of-bounds rowslice selection succeeded\n"
//              << " Details:\n"
//              << "   Result:\n" << rs << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//       catch( std::invalid_argument& ) {}
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c columnslice() function with the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c columnslice() function with the Subtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testColumnSlice()
{
   using blaze::subtensor;
   using blaze::columnslice;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major columnslice() function";

      initialize();

      {
         ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
         USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );

         auto ps1 = columnslice( sm1, 1UL );
         auto ps2 = columnslice( sm2, 1UL );

         if( ps1 != ps2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Column function failed\n"
                << " Details:\n"
                << "   Result:\n" << ps1 << "\n"
                << "   Expected result:\n" << ps2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( ps1(1,1) != ps2(1,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result: " << ps1(1,1) << "\n"
                << "   Expected result: " << ps2(1,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *ps1.begin( 2UL ) != *ps2.begin( 2UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *ps1.begin( 2UL ) << "\n"
                << "   Expected result: " << *ps2.begin( 2UL ) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
         auto ps16 = columnslice( sm1, 16UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds columnslice succeeded\n"
             << " Details:\n"
             << "   Result:\n" << ps16 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c columnslices() function with the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c columnslices() function with the Subtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testColumnSlices()
{
//    using blaze::subtensor;
//    using blaze::rows;
//    using blaze::aligned;
//    using blaze::unaligned;
//
//
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major columnslices() function";
//
//       initialize();
//
//       {
//          ASMT sm1 = subtensor<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//          USMT sm2 = subtensor<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//          auto cs1 = columnslices( sm1, { 0UL, 2UL, 4UL, 6UL } );
//          auto cs2 = columnslices( sm2, { 0UL, 2UL, 4UL, 6UL } );
//
//          if( cs1 != cs2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Rows function failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << cs1 << "\n"
//                 << "   Expected result:\n" << cs2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( cs1(1,1) != cs2(1,1) ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Function call operator access failed\n"
//                 << " Details:\n"
//                 << "   Result: " << cs1(1,1) << "\n"
//                 << "   Expected result: " << cs2(1,1) << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( *cs1.begin( 1UL ) != *cs2.begin( 1UL ) ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Iterator access failed\n"
//                 << " Details:\n"
//                 << "   Result: " << *cs1.begin( 1UL ) << "\n"
//                 << "   Expected result: " << *cs2.begin( 1UL ) << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       try {
//          ASMT sm1 = subtensor<aligned>( mat1_, 8UL, 16UL, 8UL, 16UL );
//          auto cs  = columnslices( sm1, { 16UL } );
//
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Setup of out-of-bounds columnslice selection succeeded\n"
//              << " Details:\n"
//              << "   Result:\n" << cs << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//       catch( std::invalid_argument& ) {}
//    }
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Test of the \c pageslice() function with the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c pageslice() function with the Subtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testPageSlice()
{
   using blaze::subtensor;
   using blaze::pageslice;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major pageslice() function";

      initialize();

      {
         ASMT sm1 = subtensor<aligned>  ( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
         USMT sm2 = subtensor<unaligned>( mat2_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );

         auto ps1 = pageslice( sm1, 1UL );
         auto ps2 = pageslice( sm2, 1UL );

         if( ps1 != ps2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Column function failed\n"
                << " Details:\n"
                << "   Result:\n" << ps1 << "\n"
                << "   Expected result:\n" << ps2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( ps1(1,1) != ps2(1,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result: " << ps1(1,1) << "\n"
                << "   Expected result: " << ps2(1,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *ps1.begin( 2UL ) != *ps2.begin( 2UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *ps1.begin( 2UL ) << "\n"
                << "   Expected result: " << *ps2.begin( 2UL ) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         ASMT sm1 = subtensor<aligned>( mat1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL );
         auto ps16 = pageslice( sm1, 16UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds pageslice succeeded\n"
             << " Details:\n"
             << "   Result:\n" << ps16 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c pageslices() function with the Subtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c pageslices() function with the Subtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseAlignedTest::testPageSlices()
{
//    using blaze::subtensor;
//    using blaze::rows;
//    using blaze::aligned;
//    using blaze::unaligned;
//
//
//    //=====================================================================================
//    // Row-major tensor tests
//    //=====================================================================================
//
//    {
//       test_ = "Row-major pageslices() function";
//
//       initialize();
//
//       {
//          ASMT sm1 = subtensor<aligned>  ( mat1_, 8UL, 16UL, 8UL, 16UL );
//          USMT sm2 = subtensor<unaligned>( mat2_, 8UL, 16UL, 8UL, 16UL );
//          auto cs1 = pageslices( sm1, { 0UL, 2UL, 4UL, 6UL } );
//          auto cs2 = pageslices( sm2, { 0UL, 2UL, 4UL, 6UL } );
//
//          if( cs1 != cs2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Rows function failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << cs1 << "\n"
//                 << "   Expected result:\n" << cs2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( cs1(1,1) != cs2(1,1) ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Function call operator access failed\n"
//                 << " Details:\n"
//                 << "   Result: " << cs1(1,1) << "\n"
//                 << "   Expected result: " << cs2(1,1) << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//
//          if( *cs1.begin( 1UL ) != *cs2.begin( 1UL ) ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Iterator access failed\n"
//                 << " Details:\n"
//                 << "   Result: " << *cs1.begin( 1UL ) << "\n"
//                 << "   Expected result: " << *cs2.begin( 1UL ) << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//
//       try {
//          ASMT sm1 = subtensor<aligned>( mat1_, 8UL, 16UL, 8UL, 16UL );
//          auto cs  = pageslices( sm1, { 16UL } );
//
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Setup of out-of-bounds pageslice selection succeeded\n"
//              << " Details:\n"
//              << "   Result:\n" << cs << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//       catch( std::invalid_argument& ) {}
//    }
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
void DenseAlignedTest::initialize()
{
   // Initializing the row-major dynamic tensors
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

//*************************************************************************************************
int main()
{
   std::cout << "   Running Subtensor dense aligned test (part 2)..." << std::endl;

   try
   {
      RUN_SUBTENSOR_DENSEALIGNED_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during Subtensor dense aligned test (part 2):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
