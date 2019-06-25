//=================================================================================================
/*!
//  \file src/mathtest/dilatedsubtensor/DenseTest2.cpp
//  \brief Source file for the dilatedsubtensor dense test (part 2)
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
//#include <blaze/math/CompressedTensor.h>

#include <blaze/system/Platform.h>
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
   testDilatedSubtensor();
   testPageslice();
   testRowslice();
   testColumnslice();
   //testPageslices();
   //testRowslices();
   //testColumnslices();

}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of all dilatedsubtensor (self-)scaling operations.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all available ways to scale an instance of the dilatedsubtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testScaling()
{
   using blaze::dilatedsubtensor;
   using blaze::pageslice;


   //=====================================================================================
   // Row-major self-scaling (M*=s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M*=s) (8x8x4)";

      initialize();


      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 8UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 4UL, 2UL, 3UL);

      st1 *= 3;
      st2 *= 3;

      checkPages  ( st1,  8UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2,  4UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M*=s) (8x16x4)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 16UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL);

      st1 *= 3;
      st2 *= 3;

      checkPages  ( st1,  8UL );
      checkRows   ( st1, 16UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2, 16UL );
      checkColumns( st2,  4UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=M*s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=M*s) (8x8x4)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 8UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 4UL, 2UL, 3UL);

      st1 = st1* 3;
      st2 = st2* 3;

      checkPages  ( st1,  8UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2,  4UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M=M*s) (8x16x4)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 16UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL);

      st1 = st1 * 3;
      st2 = st2 * 3;

      checkPages  ( st1,  8UL );
      checkRows   ( st1, 16UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2, 16UL );
      checkColumns( st2,  4UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=s*M)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=s*M) (8x8x4)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 8UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 4UL, 2UL, 3UL);

      st1 = 3*st1;
      st2 = 3*st2;

      checkPages  ( st1,  8UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2,  4UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M=s*M) (8x16x4)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 16UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL);

      st1 = 3 * st1;
      st2 = 3 * st2;

      checkPages  ( st1,  8UL );
      checkRows   ( st1, 16UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2, 16UL );
      checkColumns( st2,  4UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   //=====================================================================================
   // Row-major self-scaling (M/=s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M/=s) (8x8x4)";

      initialize();


      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 8UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 4UL, 2UL, 3UL);

      st1 /= 3;
      st2 /= 3;

      checkPages  ( st1,  8UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2,  4UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M/=s) (8x16x4)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 16UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL);

      st1 /= 3;
      st2 /= 3;

      checkPages  ( st1,  8UL );
      checkRows   ( st1, 16UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2, 16UL );
      checkColumns( st2,  4UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=M/s)
   //=====================================================================================
   {
      test_ = "Row-major self-scaling (M=M/s) (8x8x4)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 8UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 4UL, 2UL, 3UL);

      st1 = st1 / 3;
      st2 = st2 / 3;

      checkPages  ( st1,  8UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2,  4UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M=M/s) (8x16x4)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 16UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL);

      st1 = st1 / 3;
      st2 = st2 / 3;

      checkPages  ( st1,  8UL );
      checkRows   ( st1, 16UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2, 16UL );
      checkColumns( st2,  4UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
   //=====================================================================================
   // Row-major dilatedsubtensor::scale()
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubtensor::scale()";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 16UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL);


      checkPages  ( st1,  8UL );
      checkRows   ( st1, 16UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2, 16UL );
      checkColumns( st2,  4UL );

      // Integral scaling of the tensor
      st1.scale( 2 );
      st2.scale( 2 );

      checkPages  ( st1,  8UL );
      checkRows   ( st1, 16UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2, 16UL );
      checkColumns( st2,  4UL );


      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Integral scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }

      // Floating point scaling of the tensor
      st1.scale( 0.5 );
      st2.scale( 0.5 );

      checkPages  ( st1,  8UL );
      checkRows   ( st1, 16UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2, 16UL );
      checkColumns( st2,  4UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Floating point scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,0) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the dilatedsubtensor function call operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the function call operator
// of the dilatedsubtensor specialization. In case an error is detected, a \a std::runtime_error
// exception is thrown.
*/
void DenseTest::testFunctionCall()
{
   using blaze::dilatedsubtensor;
   using blaze::pageslice;

   //=====================================================================================
   // Row-major dilatedsubtensor tests
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubtensor::operator()";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 16UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL);

      // Assignment to the element (0,1,4)
      {
         st1(0,1,4) = 9;
         st2(1,4) = 9;

         checkPages  ( st1,  8UL );
         checkRows   ( st1, 16UL );
         checkColumns( st1,  4UL );
         checkRows   ( st2, 16UL );
         checkColumns( st2,  4UL );

         if( pageslice(st1,0) != st2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice(st1,0) << "\n"
                << "   Expected result:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (1,3,10)
      {
         st2 = dilatedsubmatrix(pageslice(tens2_, 8UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL);
         st1(1,3,10) = 0;
         st2(3,10) = 0;

         checkPages  ( st1,  8UL );
         checkRows   ( st1, 16UL );
         checkColumns( st1,  4UL );
         checkRows   ( st2, 16UL );
         checkColumns( st2,  4UL );

         if( pageslice(st1,1) != st2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice(st1,1) << "\n"
                << "   Expected result:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (1,6,8)
      {
         st1(1,6,8) = -7;
         st2(6,8) = -7;

         checkPages  ( st1,  8UL );
         checkRows   ( st1, 16UL );
         checkColumns( st1,  4UL );
         checkRows   ( st2, 16UL );
         checkColumns( st2,  4UL );

         if( pageslice(st1,1) != st2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice(st1,1) << "\n"
                << "   Expected result:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Addition assignment to the element (1,5,7)
      {
         st1(1,5,7) += 3;
         st2(5,7) += 3;

         checkPages  ( st1,  8UL );
         checkRows   ( st1, 16UL );
         checkColumns( st1,  4UL );
         checkRows   ( st2, 16UL );
         checkColumns( st2,  4UL );

         if( pageslice(st1,1) != st2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice(st1,1) << "\n"
                << "   Expected result:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Subtraction assignment to the element (1,2,14)
      {
         st1(1,2,14) -= -8;
         st2(2,14) -= -8;

         checkPages  ( st1,  8UL );
         checkRows   ( st1, 16UL );
         checkColumns( st1,  4UL );
         checkRows   ( st2, 16UL );
         checkColumns( st2,  4UL );

         if( pageslice(st1,1) != st2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice(st1,1) << "\n"
                << "   Expected result:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Multiplication assignment to the element (1,1,1)
      {
         st1(1,1,1) *= 3;
         st2(1,1) *= 3;

         checkPages  ( st1,  8UL );
         checkRows   ( st1, 16UL );
         checkColumns( st1,  4UL );
         checkRows   ( st2, 16UL );
         checkColumns( st2,  4UL );

         if( pageslice(st1,1) != st2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice(st1,1) << "\n"
                << "   Expected result:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Division assignment to the element (1,3,4)
      {
         st1(1,3,4) /= 2;
         st2(3,4) /= 2;

         checkPages  ( st1,  8UL );
         checkRows   ( st1, 16UL );
         checkColumns( st1,  4UL );
         checkRows   ( st2, 16UL );
         checkColumns( st2,  4UL );

         if( pageslice(st1,1) != st2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << pageslice(st1,1) << "\n"
                << "   Expected result:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the dilatedsubtensor iterator implementation.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the iterator implementation of the dilatedsubtensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testIterator()
{
   using blaze::dilatedsubtensor;


   //=====================================================================================
   // Row-major dilatedsubtensor tests
   //=====================================================================================

   {
      initialize();

      // Testing the Iterator default constructor
      {
         test_ = "Row-major Iterator default constructor";

         DSTT::Iterator it{};

         if( it != DSTT::Iterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing the ConstIterator default constructor
      {
         test_ = "Row-major ConstIterator default constructor";

         DSTT::ConstIterator it{};

         if( it != DSTT::ConstIterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing conversion from Iterator to ConstIterator
      {
         test_ = "Row-major Iterator/ConstIterator conversion";

         DSTT st = dilatedsubtensor( tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 4UL, 3UL, 2UL );
         DSTT::ConstIterator it( begin( st, 8UL, 4UL) );

         if( it == end( st, 8UL, 4UL ) || *it != st(4,8,0) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 0th row of a 8x16 dilated tensor via Iterator (end-begin)
      {
         test_ = "Row-major Iterator subtraction (end-begin)";

         DSTT st = dilatedsubtensor( tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 4UL, 3UL, 2UL  );
         const ptrdiff_t number( end( st, 0UL, 1UL  ) - begin( st, 0UL, 1UL  ) );

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

      // Counting the number of elements in 0th row of a 8x16 tensor via Iterator (begin-end)
      {
         test_ = "Row-major Iterator subtraction (begin-end)";

         DSTT st = dilatedsubtensor( tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 4UL, 3UL, 2UL );
         const ptrdiff_t number( begin( st, 0UL, 1UL  ) - end( st, 0UL, 1UL  ) );

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

      // Counting the number of elements in 15th row of a 16x8 tensor via ConstIterator (end-begin)
      {
         test_ = "Row-major ConstIterator subtraction (end-begin)";

         DSTT st = dilatedsubtensor( tens2_, 4UL, 8UL, 16UL, 4UL, 8UL, 12UL, 4UL, 3UL, 2UL );
         const ptrdiff_t number( cend( st, 15UL, 15UL ) - cbegin( st, 15UL, 15UL ) );

         if( number != 12L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 8\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 15th row of a 16x8 tensor via ConstIterator (begin-end)
      {
         test_ = "Row-major ConstIterator subtraction (begin-end)";

         DSTT st = dilatedsubtensor( tens2_, 4UL, 8UL, 16UL, 4UL, 8UL, 12UL, 4UL, 3UL, 2UL );
         const ptrdiff_t number( cbegin( st, 15UL, 15UL ) - cend( st, 15UL, 15UL ) );

         if( number != -12L ) {
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

         DSTT st = dilatedsubtensor( tens1_, 2UL, 2UL, 4UL, 8UL, 8UL, 12UL, 3UL, 2UL, 2UL );
         DSTT::ConstIterator it ( cbegin( st, 2UL, 4UL ) );
         DSTT::ConstIterator end( cend( st, 2UL, 4UL ) );

         if( it == end || *it != st(4,2,0) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || *it != st(4,2,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         --it;

         if( it == end || *it != st(4,2,0) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it++;

         if( it == end || *it != st(4,2,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it--;

         if( it == end || *it != st(4,2,0) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it += 2UL;

         if( it == end || *it != st(4,2,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator addition assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it -= 2UL;

         if( it == end || *it != st(4,2,0) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator subtraction assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it + 2UL;

         if( it == end || *it != st(4,2,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar addition failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it - 2UL;

         if( it == end || *it != st(4,2,0) ) {
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

         DSTT st1 = dilatedsubtensor( tens1_, 8UL, 16UL, 4UL, 8UL, 16UL, 4UL, 3UL, 2UL, 4UL );
         int value = 7;

         DSTT::Iterator it1( begin( st1, 2UL, 6UL ) );

         for( ; it1!=end( st1, 2UL, 6UL  ); ++it1 ) {
            *it1 = value;
            ++value;
         }

         if( st1(6,2,3) != value-1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << st1 << "\n"
                << "   Expected result:\n" << value-1 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         initialize();
         test_ = "Row-major addition assignment via Iterator";

         DSTT st1 = dilatedsubtensor( tens1_, 8UL, 16UL, 4UL, 8UL, 16UL, 4UL, 3UL, 2UL, 4UL );
         int value = 7;

         DSTT::Iterator it1( begin( st1, 4UL, 6UL ) );

         for( ; it1!=end( st1, 4UL, 6UL  ); ++it1 ) {
            *it1 += value;
            ++value;
         }

         if( st1(6,4,3) != tens2_(26,24,16)+value-1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << st1(6,4,3) << "\n"
                << "   Expected result:\n" << tens2_(26,24,16)+value-1 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

       //Testing subtraction assignment via Iterator
      {
         initialize();
         test_ = "Row-major subtraction assignment via Iterator";

         DSTT st1 = dilatedsubtensor( tens1_, 8UL, 16UL, 4UL, 8UL, 16UL, 4UL, 3UL, 2UL, 4UL );
         int value = 4;

         DSTT::Iterator it1( begin( st1, 4UL, 8UL ) );

         for( ; it1!=end( st1, 4UL, 8UL  ); ++it1 ) {
            *it1 -= value;
            ++value;
         }

         if( st1(8,4,3) != tens2_(32,24,16)-value+1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << st1(8,4,3) << "\n"
                << "   Expected result:\n" << tens2_(32,24,16)-value-1 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         initialize();
         test_ = "Row-major multiplication assignment via Iterator";

         DSTT st1 = dilatedsubtensor( tens1_, 8UL, 16UL, 4UL, 8UL, 16UL, 4UL, 3UL, 2UL, 4UL );
         int value = 4;

         DSTT::Iterator it1( begin( st1, 4UL, 8UL ) );

         for( ; it1!=end( st1, 4UL, 8UL  ); ++it1 ) {
            *it1 *= value;
            ++value;
         }

         if( st1(8,4,3) != tens2_(32,24,16)*(value-1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << st1(8,4,3) << "\n"
                << "   Expected result:\n" << tens2_(32,24,16)-value-1 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         initialize();
         test_ = "Row-major division assignment via Iterator";

         DSTT st1 = dilatedsubtensor( tens1_, 8UL, 16UL, 4UL, 8UL, 16UL, 4UL, 3UL, 2UL, 4UL );
         int value = 4;

         DSTT::Iterator it1( begin( st1, 4UL, 8UL ) );

         for( ; it1!=end( st1, 4UL, 8UL  ); ++it1 ) {
            *it1 /= value;
            ++value;
         }

         if( st1(8,4,3) != tens2_(32,24,16)/(value-1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << st1(8,4,3) << "\n"
                << "   Expected result:\n" << tens2_(32,24,16)-value-1 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c nonZeros() member function of the dilatedsubtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c nonZeros() member function of the dilatedsubtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testNonZeros()
{
   using blaze::dilatedsubtensor;
   using blaze::aligned;
   using blaze::unaligned;


   //=====================================================================================
   // Row-major dilatedsubtensor tests
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubtensor::nonZeros()";

      initialize();

      size_t page_ = blaze::rand<size_t>( 0, tens1_.pages()-1 );
      const DSTT st1 = dilatedsubtensor(tens1_, page_, 8UL, 16UL, 1UL, 16UL, 8UL ,1UL, 3UL, 2UL);
      auto st2 = dilatedsubmatrix(blaze::pageslice(tens2_, page_), 8UL, 16UL, 16UL, 8UL , 3UL, 2UL);


      checkPages  ( st1, 1UL  );
      checkRows   ( st1, 16UL );
      checkColumns( st1, 8UL  );
      checkRows   ( st2, 16UL );
      checkColumns( st2, 8UL  );

      if( st1.nonZeros() != st2.nonZeros() ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid number of non-zeros\n"
             << " Details:\n"
             << "   Result:\n" << st1.nonZeros() << "\n"
             << "   Expected result:\n" << st2.nonZeros() << "\n"
             << "   dilatedsubtensor:\n" << st1 << "\n"
             << "   Reference:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }

      for( size_t i=0UL; i<st1.rows(); ++i ) {
         if( st1.nonZeros(i,0) != st2.nonZeros(i) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
               << " Error: Invalid number of non-zeros in row " << i << "\n"
               << " Details:\n"
               << "   Result:\n" << st1.nonZeros(i, 0) << "\n"
               << "   Expected result:\n" << st2.nonZeros(i) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reset() member function of the dilatedsubtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reset() member function of the dilatedsubtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testReset()
{
   using blaze::dilatedsubtensor;
   using blaze::reset;
   using blaze::pageslice;

   //=====================================================================================
   // Row-major single element reset
   //=====================================================================================

   {
      test_ = "Row-major reset() function";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 8UL, 8UL, 16UL, 8UL, 16UL, 8UL ,4UL, 3UL, 2UL);

      reset( st1(8,4,4) );

      checkPages  ( st1, 8UL  );
      checkRows   ( st1, 16UL );
      checkColumns( st1, 8UL  );

      if( st1(8,4,4)!= 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << st1 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major reset
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubtensor::reset() (lvalue)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 16UL, 2UL, 2UL);

      reset( st1 );
      reset( st2 );

      checkPages  ( st1,  4UL );
      checkRows   ( st1,  8UL );
      checkColumns( st1, 16UL );
      checkRows   ( st2,  8UL );
      checkColumns( st2, 16UL );

      if( !isDefault( st1 ) || !isDefault( st2 ) || pageslice(st1,0) != st2  ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << st1 << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major dilatedsubtensor::reset() (rvalue)";

      initialize();

      reset( dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL) );
      reset( dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 16UL, 2UL, 2UL) );

      if( pageslice(tens1_,4UL) != pageslice(tens2_,4UL) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tens1_ << "\n"
             << "   Expected result:\n" << tens2_ << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major row-wise reset
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubtensor::reset( size_t, size_t )";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 4UL, 8UL, 16UL, 3UL, 2UL, 2UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 8UL, 16UL, 2UL, 2UL);

      for (size_t k = 0UL; k < st1.pages(); ++k)
      {
         st2 = dilatedsubmatrix(pageslice(tens2_, 4UL+k*3UL), 8UL, 16UL, 8UL, 16UL, 2UL, 2UL);
         for (size_t i = 0UL; i < st1.rows(); ++i)
         {
            reset(st1, i, k);
            reset(st2, i);
            if ( pageslice(st1,k) != st2) {
               std::ostringstream oss;
               oss << " Test: " << test_ << "\n"
                  << " Error: Reset operation failed\n"
                  << " Details:\n"
                  << "   k:\n" << k << "\n"
                  << "   Result:\n" << pageslice(st1,k) << "\n"
                  << "   Expected result:\n" << st2 << "\n";
               throw std::runtime_error(oss.str());
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c clear() function with the dilatedsubtensor specialization.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c clear() function with the dilatedsubtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testClear()
{
   using blaze::dilatedsubtensor;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::clear;


   //=====================================================================================
   // Row-major single element clear
   //=====================================================================================

   {
      test_ = "Row-major clear() function";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 16UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL);

      clear( st1(0,4,4) );
      clear( st2(4,4) );

      checkPages  ( st1,  8UL );
      checkRows   ( st1, 16UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2, 16UL );
      checkColumns( st2,  4UL );

      if( pageslice(st1,0) != st2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << st1 << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major clear
   //=====================================================================================

   {
      test_ = "Row-major clear() function (lvalue)";

      initialize();

      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 8UL, 16UL, 4UL, 4UL, 2UL, 3UL);
      DSPT st2 = dilatedsubmatrix(pageslice(tens2_, 8UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL);

      clear( st1 );
      clear( st2 );

      checkPages  ( st1,  8UL );
      checkRows   ( st1, 16UL );
      checkColumns( st1,  4UL );
      checkRows   ( st2, 16UL );
      checkColumns( st2,  4UL );

      if( !isDefault( st1 ) || !isDefault( st2 ) || pageslice(st1,1) != st2  ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << pageslice(st1,1) << "\n"
             << "   Expected result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major clear() function (rvalue)";

      initialize();

      clear( dilatedsubtensor(tens1_, 4UL, 8UL, 16UL, 4UL, 16UL, 4UL, 4UL, 2UL, 3UL) );
      clear( dilatedsubmatrix(pageslice(tens2_, 4UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL) );
      clear( dilatedsubmatrix(pageslice(tens2_, 8UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL) );
      clear( dilatedsubmatrix(pageslice(tens2_, 12UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL) );
      clear( dilatedsubmatrix(pageslice(tens2_, 16UL), 8UL, 16UL, 16UL, 4UL, 2UL, 3UL) );

      if( tens1_ != tens2_ ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tens1_ << "\n"
             << "   Expected result:\n" << tens2_ << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c transpose() member function of the dilatedsubtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c transpose() member function of the dilatedsubtensor
// specialization. Additionally, it performs a test of self-transpose via the \c trans()
// function. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testTranspose()
{
   using blaze::dilatedsubtensor;


   //=====================================================================================
   // Row-major dilatedsubtensor tests
   //=====================================================================================

   {
      test_ = "Row-major self-transpose via transpose()";

      initialize();

      int test_value1 = tens1_(8, 8, 2);
      int test_value2 = tens1_(4, 8, 5);
      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 2UL, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL);

      transpose(st1);

      checkPages  ( st1, 8UL  );
      checkRows   ( st1, 16UL );
      checkColumns( st1, 8UL  );

      if( st1(0,0,1) != test_value1 || st1(1,0,0) != test_value2   ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << st1(0,0,1) << ","<< st1(1,0,0) << "\n"
             << "   Expected result:\n" << test_value1 << "," << test_value2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-transpose via trans()";

      initialize();

      int test_value1 = tens1_(8, 8, 2);
      int test_value2 = tens1_(4, 8, 5);
      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 2UL, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL);

      st1 = trans( st1 );

      checkPages  ( st1, 8UL  );
      checkRows   ( st1, 16UL );
      checkColumns( st1, 8UL  );

      if( st1(0,0,1) != test_value1 || st1(1,0,0) != test_value2  ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << st1(0,0,1) << ","<< st1(1,0,0) << "\n"
             << "   Expected result:\n" << test_value1 << "," << test_value2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c ctranspose() member function of the dilatedsubtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c ctranspose() member function of the dilatedsubtensor
// class template. Additionally, it performs a test of self-transpose via the \c ctrans()
// function. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testCTranspose()
{
   using blaze::dilatedsubtensor;


   //=====================================================================================
   // Row-major dilatedsubtensor tests
   //=====================================================================================

   {
      test_ = "Row-major self-transpose via ctranspose()";

      initialize();

      int test_value1 = tens1_(8, 8, 2);
      int test_value2 = tens1_(4, 8, 5);
      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 2UL, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL);

      transpose(st1);

      checkPages  ( st1, 8UL  );
      checkRows   ( st1, 16UL );
      checkColumns( st1, 8UL  );

      if( st1(0,0,1) != test_value1 || st1(1,0,0) != test_value2   ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << st1(0,0,1) << ","<< st1(1,0,0) << "\n"
             << "   Expected result:\n" << test_value1 << "," << test_value2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-transpose via ctrans()";

      initialize();

      int test_value1 = tens1_(8, 8, 2);
      int test_value2 = tens1_(4, 8, 5);
      DSTT st1 = dilatedsubtensor(tens1_, 4UL, 8UL, 2UL, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL);

      st1 = trans( st1 );

      checkPages  ( st1, 8UL  );
      checkRows   ( st1, 16UL );
      checkColumns( st1, 8UL  );

      if( st1(0,0,1) != test_value1 || st1(1,0,0) != test_value2  ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << st1(0,0,1) << ","<< st1(1,0,0) << "\n"
             << "   Expected result:\n" << test_value1 << "," << test_value2 << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDefault() function with the dilatedsubtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDefault() function with the dilatedsubtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testIsDefault()
{
   using blaze::dilatedsubtensor;
   using blaze::aligned;
   using blaze::isDefault;


   //=====================================================================================
   // Row-major dilatedsubtensor tests
   //=====================================================================================

   {
      test_ = "Row-major isDefault() function";

      initialize();

      // isDefault with default dilatedsubtensor
      {
         TT tens( 64UL, 64UL, 64UL, 0 );
         DSTT st = dilatedsubtensor(tens, 4UL, 8UL, 2UL, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL);

         if( isDefault( st(2,4,4) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   dilatedsubtensor element: " << st(2,4,4) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( st ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default dilatedsubtensor
      {
         DSTT st = dilatedsubtensor(tens1_, 4UL, 8UL, 2UL, 8UL, 16UL, 8UL, 4UL, 2UL, 3UL);

         if( isDefault( st ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isSame() function with the dilatedsubtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSame() function with the dilatedsubtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testIsSame()
{
   using blaze::dilatedsubtensor;


   //=====================================================================================
   // Row-major tensor-based tests
   //=====================================================================================

   {
      test_ = "Row-major isSame() function (tensor-based)";

      // isSame with tensor and matching dilatedsubtensor
      {
         DSTT st = dilatedsubtensor( tens1_, 0UL, 0UL, 0UL, 64UL, 64UL, 64UL, 1UL, 1UL, 1UL );

         if( blaze::isSame( st, tens1_ ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tens1_, st ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with tensor and non-matching dilatedsubtensor (different number of page)
      {
         DSTT st = dilatedsubtensor( tens1_, 0UL, 0UL, 0UL, 4UL, 64UL, 64UL, 1UL, 1UL, 1UL );

         if( blaze::isSame( st, tens1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tens1_, st ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with tensor and non-matching dilatedsubtensor (different number of rows)
      {
         DSTT st = dilatedsubtensor( tens1_, 0UL, 0UL, 0UL, 64UL, 32UL, 64UL, 1UL, 1UL, 1UL );

         if( blaze::isSame( st, tens1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tens1_, st ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with tensor and non-matching dilatedsubtensor (different number of columns)
      {
         DSTT st = dilatedsubtensor( tens1_, 0UL, 0UL, 0UL, 64UL, 64UL, 60UL, 1UL, 1UL, 1UL );

         if( blaze::isSame( st, tens1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tens1_, st ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with tensor and non-matching dilatedsubtensor (different page index)
      {
         DSTT st = dilatedsubtensor( tens1_, 4UL, 0UL, 0UL, 60UL, 64UL, 64UL, 1UL, 1UL, 1UL );

         if( blaze::isSame( st, tens1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tens1_, st ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with tensor and non-matching dilatedsubtensor (different row index)
      {
         DSTT st = dilatedsubtensor( tens1_, 0UL, 4UL, 0UL, 64UL, 60UL, 64UL, 1UL, 1UL, 1UL );

         if( blaze::isSame( st, tens1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tens1_, st ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with tensor and non-matching dilatedsubtensor (different column index)
      {
         DSTT st = dilatedsubtensor( tens1_, 0UL, 0UL, 4UL, 64UL, 64UL, 60UL, 1UL, 1UL, 1UL );

         if( blaze::isSame( st, tens1_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tens1_, st ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Tensor:\n" << tens1_ << "\n"
                << "   dilatedsubtensor:\n" << st << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching dilatedsubtensors
      {
         DSTT st1 = dilatedsubtensor( tens1_, 4UL, 16UL, 0UL, 4UL, 8UL, 16UL, 4UL, 3UL, 2UL );
         DSTT st2 = dilatedsubtensor( tens1_, 4UL, 16UL, 0UL, 4UL, 8UL, 16UL, 4UL, 3UL, 2UL );

         if( blaze::isSame( st1, st2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubtensor:\n" << st1 << "\n"
                << "   Second dilatedsubtensor:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching subtensors (different number of pages)
      {
         DSTT st1 = dilatedsubtensor( tens1_, 4UL, 16UL, 0UL, 4UL, 8UL, 16UL, 4UL, 3UL, 2UL );
         DSTT st2 = dilatedsubtensor( tens1_, 4UL, 16UL, 0UL, 8UL, 8UL, 16UL, 4UL, 3UL, 2UL );

         if( blaze::isSame( st1, st2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubtensor:\n" << st1 << "\n"
                << "   Second dilatedsubtensor:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching subtensors (different number of rows)
      {
         DSTT st1 = dilatedsubtensor( tens1_, 4UL, 16UL, 0UL, 8UL, 4UL, 16UL, 4UL, 3UL, 2UL );
         DSTT st2 = dilatedsubtensor( tens1_, 4UL, 16UL, 0UL, 8UL, 8UL, 16UL, 4UL, 3UL, 2UL );

         if( blaze::isSame( st1, st2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubtensor:\n" << st1 << "\n"
                << "   Second dilatedsubtensor:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching subtensors (different number of columns)
      {
         DSTT st1 = dilatedsubtensor( tens1_, 4UL, 16UL, 0UL, 4UL, 8UL, 10UL, 4UL, 3UL, 2UL );
         DSTT st2 = dilatedsubtensor( tens1_, 4UL, 16UL, 0UL, 4UL, 8UL, 16UL, 4UL, 3UL, 2UL );

         if( blaze::isSame( st1, st2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubtensor:\n" << st1 << "\n"
                << "   Second dilatedsubtensor:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching subtensors (different page index)
      {
         DSTT st1 = dilatedsubtensor( tens1_, 0UL, 16UL, 0UL, 4UL, 8UL, 10UL, 4UL, 3UL, 2UL );
         DSTT st2 = dilatedsubtensor( tens1_, 4UL, 16UL, 0UL, 4UL, 8UL, 10UL, 4UL, 3UL, 2UL );

         if( blaze::isSame( st1, st2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubtensor:\n" << st1 << "\n"
                << "   Second dilatedsubtensor:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching subtensors (different row index)
      {
         DSTT st1 = dilatedsubtensor( tens1_, 4UL, 16UL, 0UL, 4UL, 8UL, 10UL, 4UL, 3UL, 2UL );
         DSTT st2 = dilatedsubtensor( tens1_, 4UL, 10UL, 0UL, 4UL, 8UL, 10UL, 4UL, 3UL, 2UL );

         if( blaze::isSame( st1, st2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubtensor:\n" << st1 << "\n"
                << "   Second dilatedsubtensor:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching subtensors (different column index)
      {
         DSTT st1 = dilatedsubtensor( tens1_, 4UL, 16UL, 4UL, 4UL, 8UL, 10UL, 4UL, 3UL, 2UL );
         DSTT st2 = dilatedsubtensor( tens1_, 4UL, 16UL, 0UL, 4UL, 8UL, 10UL, 4UL, 3UL, 2UL );

         if( blaze::isSame( st1, st2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First dilatedsubtensor:\n" << st1 << "\n"
                << "   Second dilatedsubtensor:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c dilatedsubtensor() function with the dilatedsubtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c dilatedsubtensor() function with the dilatedsubtensor
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testDilatedSubtensor()
{
   using blaze::dilatedsubtensor;


   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major dilatedsubtensor() function";

      initialize();

      {
         DSTT st1 = dilatedsubtensor( tens1_, 4UL, 16UL, 4UL, 8UL, 8UL, 10UL, 2UL, 3UL, 2UL );
         DSTT st2 = dilatedsubtensor( tens2_, 4UL, 16UL, 4UL, 4UL, 8UL, 10UL, 4UL, 3UL, 2UL );

         DSTT st3 = dilatedsubtensor( st1, 0UL, 0UL, 0UL, 4UL, 8UL, 10UL, 2UL, 1UL, 1UL );

         if( st2 != st3 || tens1_ != tens2_ ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: dilatedsubtensor function failed\n"
                << " Details:\n"
                << "   Result:\n" << st2 << "\n"
                << "   Expected result:\n" << st3 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( st2(1,0,1) != st3(1,0,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result: " << st2(1,0,1) << "\n"
                << "   Expected result: " << st3(1,0,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *st2.begin(1UL, 2UL) != *st3.begin(1UL, 2UL) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *st2.begin(1UL, 2UL) << "\n"
                << "   Expected result: " << *st3.begin(1UL, 2UL) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         DSTT st1 = dilatedsubtensor( tens1_,  8UL, 8UL, 16UL, 16UL, 32UL, 4UL, 2UL, 1UL, 3UL );
         DSTT st2 = dilatedsubtensor( st1   , 16UL, 0UL, 8UL,  8UL, 8UL,  4UL, 2UL, 1UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds dilatedsubtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         DSTT st1 = dilatedsubtensor( tens1_,  8UL, 8UL, 16UL, 16UL, 32UL, 4UL, 2UL, 1UL, 3UL );
         DSTT st2 = dilatedsubtensor( st1  , 0UL, 32UL, 8UL,  8UL, 8UL,  4UL, 2UL, 1UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds dilatedsubtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         DSTT st1 = dilatedsubtensor( tens1_,  8UL, 8UL, 16UL, 16UL, 32UL, 8UL, 2UL, 1UL, 3UL );
         DSTT st2 = dilatedsubtensor( st1  ,   0UL, 0UL, 18UL,  8UL, 8UL,  4UL, 2UL, 1UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds dilatedsubtensor succeeded\n"
             << " Details:\n"
             << "   Result:\n" << st2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

   }

}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c column() function with the dilatedsubtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c column() function with the dilatedsubtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testPageslice()
{
   using blaze::dilatedsubtensor;
   using blaze::pageslice;


   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major pageslice() function";

      initialize();

      {
         DSTT st1 = dilatedsubtensor( tens1_, 8UL, 8UL, 16UL, 16UL, 8UL, 4UL, 2UL, 4UL, 3UL );
         auto st2 = dilatedsubmatrix(pageslice(tens2_, 10UL), 8UL, 16UL, 8UL, 4UL, 4UL, 3UL);

         auto rs1 = pageslice( st1, 1 );

         if( rs1 != st2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Rows function failed\n"
                << " Details:\n"
                << "   Result:\n" << rs1 << "\n"
                << "   Expected result:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs1(1,1) != st2(1,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result: " << rs1(1,1) << "\n"
                << "   Expected result: " << st2(1,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *rs1.begin( 3UL ) != *st2.begin( 3UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *rs1.begin( 3UL ) << "\n"
                << "   Expected result: " << *st2.begin( 3UL ) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         DSTT st1 = dilatedsubtensor( tens1_, 8UL, 8UL, 16UL, 16UL, 8UL, 4UL, 2UL, 4UL, 3UL );
         auto rs  = pageslice( st1, 16UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds page selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c rows() function with the dilatedsubtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c rows() function with the dilatedsubtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testRowslice()
{
   using blaze::dilatedsubtensor;
   using blaze::rowslice;


   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major rowslice() function";

      initialize();

      {
         DSTT st1 = dilatedsubtensor( tens1_, 8UL, 8UL, 16UL, 16UL, 8UL, 4UL, 2UL, 4UL, 3UL );
         auto st2 = dilatedsubmatrix(rowslice(tens2_, 12UL), 16UL, 8UL, 4UL, 16UL, 3UL, 2UL);

         auto rs1 = rowslice( st1, 1 );

         if( rs1 != st2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: rowslice function failed\n"
                << " Details:\n"
                << "   Result:\n" << rs1 << "\n"
                << "   Expected result:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs1(1,1) != st2(1,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result: " << rs1(1,1) << "\n"
                << "   Expected result: " << st2(1,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *rs1.begin( 3UL ) != *st2.begin( 3UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *rs1.begin( 3UL ) << "\n"
                << "   Expected result: " << *st2.begin( 3UL ) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         DSTT st1 = dilatedsubtensor( tens1_, 8UL, 8UL, 16UL, 16UL, 8UL, 4UL, 2UL, 4UL, 3UL );
         auto rs  = rowslice( st1, 8UL );

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
/*!\brief Test of the \c column() function with the dilatedsubtensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c column() function with the dilatedsubtensor specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testColumnslice()
{
   using blaze::dilatedsubtensor;
   using blaze::columnslice;


   //=====================================================================================
   // Row-major tensor tests
   //=====================================================================================

   {
      test_ = "Row-major columnslice() function";

      initialize();

      {
         DSTT st1 = dilatedsubtensor( tens1_, 8UL, 8UL, 16UL, 16UL, 8UL, 4UL, 2UL, 4UL, 3UL );
         auto st2 = dilatedsubmatrix(columnslice(tens2_, 19UL), 8UL, 8UL, 16UL, 8UL, 2UL, 4UL);

         auto rs1 = columnslice( st1, 1 );

         if( rs1 != st2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Rows function failed\n"
                << " Details:\n"
                << "   Result:\n" << rs1 << "\n"
                << "   Expected result:\n" << st2 << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs1(1,1) != st2(1,1) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result: " << rs1(1,1) << "\n"
                << "   Expected result: " << st2(1,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( *rs1.begin( 3UL ) != *st2.begin( 3UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *rs1.begin( 3UL ) << "\n"
                << "   Expected result: " << *st2.begin( 3UL ) << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         DSTT st1 = dilatedsubtensor( tens1_, 8UL, 8UL, 16UL, 16UL, 8UL, 4UL, 2UL, 4UL, 3UL );
         auto rs  = columnslice( st1, 4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
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
   std::cout << "   Running dilatedsubtensor dense test (part 2)..." << std::endl;

   try
   {
      RUN_DILATEDSUBTENSOR_DENSE_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during dilatedsubtensor dense test (part 2):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
