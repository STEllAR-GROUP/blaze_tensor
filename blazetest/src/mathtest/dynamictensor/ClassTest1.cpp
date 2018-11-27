//=================================================================================================
/*!
//  \file src/mathtest/dynamictensor/ClassTest1.cpp
//  \brief Source file for the DynamicTensor class test (part 1)
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

#include <blaze_tensor/math/CustomTensor.h>
#include <blaze_tensor/math/DynamicTensor.h>

#include <blaze/system/Platform.h>
#include <blaze/util/Complex.h>
#include <blaze/util/Memory.h>
#include <blaze/util/policies/Deallocate.h>
#include <blaze/util/Random.h>
#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>

#include <blazetest/mathtest/dynamictensor/ClassTest.h>

namespace blazetest {

namespace mathtest {

namespace dynamictensor {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the DynamicTensor class test.
//
// \exception std::runtime_error Operation error detected.
*/
ClassTest::ClassTest()
{
   testAlignment< char           >( "char"           );
   testAlignment< signed char    >( "signed char"    );
   testAlignment< unsigned char  >( "unsigned char"  );
   testAlignment< wchar_t        >( "wchar_t"        );
   testAlignment< short          >( "short"          );
   testAlignment< unsigned short >( "unsigned short" );
   testAlignment< int            >( "int"            );
   testAlignment< unsigned int   >( "unsigned int"   );
   testAlignment< long           >( "long"           );
   testAlignment< unsigned long  >( "unsigned long"  );
   testAlignment< float          >( "float"          );
   testAlignment< double         >( "double"         );

   testAlignment< complex<char>           >( "complex<char>"           );
   testAlignment< complex<signed char>    >( "complex<signed char>"    );
   testAlignment< complex<unsigned char>  >( "complex<unsigned char>"  );
   testAlignment< complex<wchar_t>        >( "complex<wchar_t>"        );
   testAlignment< complex<short>          >( "complex<short>"          );
   testAlignment< complex<unsigned short> >( "complex<unsigned short>" );
   testAlignment< complex<int>            >( "complex<int>"            );
   testAlignment< complex<unsigned int>   >( "complex<unsigned int>"   );
   testAlignment< complex<float>          >( "complex<float>"          );
   testAlignment< complex<double>         >( "complex<double>"         );

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
/*!\brief Test of the DynamicTensor constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the DynamicTensor class template.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testConstructors()
{
   //=====================================================================================
   // default constructor
   //=====================================================================================

   // Default constructor
   {
      test_ = "DynamicTensor default constructor";

      blaze::DynamicTensor<int> tens;

      checkRows    ( tens, 0UL );
      checkColumns ( tens, 0UL );
      checkNonZeros( tens, 0UL );
   }


   //=====================================================================================
   // size constructor
   //=====================================================================================

   {
      test_ = "DynamicTensor size constructor (0x0)";

      blaze::DynamicTensor<int> tens( 0UL, 0UL, 0UL );

      checkRows    ( tens, 0UL );
      checkColumns ( tens, 0UL );
      checkPages   ( tens, 0UL );
      checkNonZeros( tens, 0UL );
   }

   {
      test_ = "DynamicTensor size constructor (0x4x2)";

      blaze::DynamicTensor<int> tens( 2UL, 0UL, 4UL );

      checkRows    ( tens, 0UL );
      checkColumns ( tens, 4UL );
      checkPages   ( tens, 2UL );
      checkNonZeros( tens, 0UL );
   }

   {
      test_ = "DynamicTensor size constructor (3x0x1)";

      blaze::DynamicTensor<int> tens( 1UL, 3UL, 0UL );

      checkRows    ( tens, 3UL );
      checkColumns ( tens, 0UL );
      checkPages   ( tens, 1UL );
      checkNonZeros( tens, 0UL );
   }

   {
      test_ = "DynamicTensor size constructor (3x1x0)";

      blaze::DynamicTensor<int> tens( 0UL, 3UL, 1UL );

      checkRows    ( tens, 3UL );
      checkColumns ( tens, 1UL );
      checkPages   ( tens, 0UL );
      checkNonZeros( tens, 0UL );
   }

   {
      test_ = "DynamicTensor size constructor (3x4x1)";

      blaze::DynamicTensor<int> tens( 1UL, 3UL, 4UL );

      checkRows    ( tens,  3UL );
      checkColumns ( tens,  4UL );
      checkPages   ( tens,  1UL );
      checkCapacity( tens, 12UL );
   }


   //=====================================================================================
   // homogeneous initialization
   //=====================================================================================

   {
      test_ = "DynamicTensor homogeneous initialization constructor (0x0x0)";

      blaze::DynamicTensor<int> tens( 0UL, 0UL, 0UL, 2 );

      checkRows    ( tens, 0UL );
      checkColumns ( tens, 0UL );
      checkPages   ( tens, 0UL );
      checkNonZeros( tens, 0UL );
   }

   {
      test_ = "DynamicTensor homogeneous initialization constructor (0x4x2)";

      blaze::DynamicTensor<int> tens( 2UL, 0UL, 4UL, 2 );

      checkRows    ( tens, 0UL );
      checkColumns ( tens, 4UL );
      checkPages   ( tens, 2UL );
      checkNonZeros( tens, 0UL );
   }

   {
      test_ = "DynamicTensor homogeneous initialization constructor (3x0x2)";

      blaze::DynamicTensor<int> tens( 2UL, 3UL, 0UL, 2 );

      checkRows    ( tens, 3UL );
      checkColumns ( tens, 0UL );
      checkPages   ( tens, 2UL );
      checkNonZeros( tens, 0UL );
   }

   {
      test_ = "DynamicTensor homogeneous initialization constructor (3x4x2)";

      blaze::DynamicTensor<int> tens( 2UL, 3UL, 4UL, 2 );

      checkRows    ( tens,  3UL );
      checkColumns ( tens,  4UL );
      checkPages   ( tens,  2UL );
      checkCapacity( tens, 24UL );
      checkNonZeros( tens, 24UL );
      checkNonZeros( tens,  0UL, 0UL, 4UL );
      checkNonZeros( tens,  1UL, 0UL, 4UL );
      checkNonZeros( tens,  2UL, 0UL, 4UL );
      checkNonZeros( tens,  0UL, 1UL, 4UL );
      checkNonZeros( tens,  1UL, 1UL, 4UL );
      checkNonZeros( tens,  2UL, 1UL, 4UL );

      if( tens(0,0,0) != 2 || tens(0,0,1) != 2 || tens(0,0,2) != 2 || tens(0,0,3) != 2 ||
          tens(0,1,0) != 2 || tens(0,1,1) != 2 || tens(0,1,2) != 2 || tens(0,1,3) != 2 ||
          tens(0,2,0) != 2 || tens(0,2,1) != 2 || tens(0,2,2) != 2 || tens(0,2,3) != 2 ||
          tens(1,0,0) != 2 || tens(1,0,1) != 2 || tens(1,0,2) != 2 || tens(1,0,3) != 2 ||
          tens(1,1,0) != 2 || tens(1,1,1) != 2 || tens(1,1,2) != 2 || tens(1,1,3) != 2 ||
          tens(1,2,0) != 2 || tens(1,2,1) != 2 || tens(1,2,2) != 2 || tens(1,2,3) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << tens << "\n"
             << "   Expected result:\n(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n)\n(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }

   //=====================================================================================
   // list initialization
   //=====================================================================================

   {
      test_ = "DynamicTensor initializer list constructor (complete list)";

      blaze::DynamicTensor<int> tens{{{1, 2, 3}, {4, 5, 6}},
                                     {{1, 2, 3}, {4, 5, 6}}};

      checkRows    ( tens,  2UL );
      checkColumns ( tens,  3UL );
      checkPages   ( tens,  2UL );
      checkCapacity( tens, 12UL );
      checkNonZeros( tens, 12UL );
      checkNonZeros( tens, 0UL, 0UL, 3UL );
      checkNonZeros( tens, 1UL, 0UL, 3UL );
      checkNonZeros( tens, 0UL, 1UL, 3UL );
      checkNonZeros( tens, 1UL, 1UL, 3UL );

      if( tens(0,0,0) != 1 || tens(0,0,1) != 2 || tens(0,0,2) != 3 ||
          tens(0,1,0) != 4 || tens(0,1,1) != 5 || tens(0,1,2) != 6 ||
          tens(1,0,0) != 1 || tens(1,0,1) != 2 || tens(1,0,2) != 3 ||
          tens(1,1,0) != 4 || tens(1,1,1) != 5 || tens(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << tens << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor initializer list constructor (incomplete list)";

      blaze::DynamicTensor<int> tens{{{1}, {4, 5, 6}}, {{1}, {4, 5, 6}}};

      checkRows    ( tens,  2UL );
      checkColumns ( tens,  3UL );
      checkPages   ( tens,  2UL );
      checkCapacity( tens, 12UL );
      checkNonZeros( tens,  8UL );
      checkNonZeros( tens, 0UL, 0UL, 1UL );
      checkNonZeros( tens, 0UL, 0UL, 1UL );
      checkNonZeros( tens, 1UL, 1UL, 3UL );
      checkNonZeros( tens, 1UL, 1UL, 3UL );

      if( tens(0,0,0) != 1 || tens(0,0,1) != 0 || tens(0,0,2) != 0 ||
          tens(0,1,0) != 4 || tens(0,1,1) != 5 || tens(0,1,2) != 6 ||
          tens(1,0,0) != 1 || tens(1,0,1) != 0 || tens(1,0,2) != 0 ||
          tens(1,1,0) != 4 || tens(1,1,1) != 5 || tens(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << tens << "\n"
             << "   Expected result:\n(( 1 0 0 )\n( 4 5 6 )\n( 1 0 0 )\n( 4 5 6 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // array initialization
   //=====================================================================================

   {
      test_ = "DynamicTensor dynamic array initialization constructor";

      std::unique_ptr<int[]> array( new int[6] );
      array[0] = 1;
      array[1] = 2;
      array[2] = 3;
      array[3] = 4;
      array[4] = 5;
      array[5] = 6;
      blaze::DynamicTensor<int> tens( 1UL, 2UL, 3UL, array.get() );

      checkRows    ( tens, 2UL );
      checkColumns ( tens, 3UL );
      checkPages   ( tens, 1UL );
      checkCapacity( tens, 6UL );
      checkNonZeros( tens, 6UL );
      checkNonZeros( tens, 0UL, 0UL, 3UL );
      checkNonZeros( tens, 1UL, 0UL, 3UL );

      if( tens(0,0,0) != 1 || tens(0,0,1) != 2 || tens(0,0,2) != 3 ||
          tens(0,1,0) != 4 || tens(0,1,1) != 5 || tens(0,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << tens << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor static array initialization constructor";

      const int array[1][2][3] = {{{1, 2, 3}, {4, 5, 6}}};
      blaze::DynamicTensor<int> tens( array );

      checkRows    ( tens, 2UL );
      checkColumns ( tens, 3UL );
      checkPages   ( tens, 1UL );
      checkCapacity( tens, 6UL );
      checkNonZeros( tens, 6UL );
      checkNonZeros( tens, 0UL, 0UL, 3UL );
      checkNonZeros( tens, 1UL, 0UL, 3UL );

      if( tens(0,0,0) != 1 || tens(0,0,1) != 2 || tens(0,0,2) != 3 ||
          tens(0,1,0) != 4 || tens(0,1,1) != 5 || tens(0,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << tens << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // copy constructor
   //=====================================================================================

   {
      test_ = "DynamicTensor copy constructor (0x0x0)";

      blaze::DynamicTensor<int> mat1( 0UL, 0UL, 0UL );
      blaze::DynamicTensor<int> mat2( mat1 );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "DynamicTensor copy constructor (0x3x1)";

      blaze::DynamicTensor<int> mat1( 1UL, 0UL, 3UL );
      blaze::DynamicTensor<int> mat2( mat1 );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 3UL );
      checkPages   ( mat2, 1UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "DynamicTensor copy constructor (2x0x1)";

      blaze::DynamicTensor<int> mat1( 1UL, 2UL, 0UL );
      blaze::DynamicTensor<int> mat2( mat1 );

      checkRows    ( mat2, 2UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 1UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "DynamicTensor copy constructor (2x1x0)";

      blaze::DynamicTensor<int> mat1( 0UL, 2UL, 1UL );
      blaze::DynamicTensor<int> mat2( mat1 );

      checkRows    ( mat2, 2UL );
      checkColumns ( mat2, 1UL );
      checkPages   ( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "DynamicTensor copy constructor (2x3x2)";

      blaze::DynamicTensor<int> mat1{{{1, 2, 3}, {4, 5, 6}},
                                     {{1, 2, 3}, {4, 5, 6}}};

      blaze::DynamicTensor<int> mat2( mat1 );

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2, 3UL, 0UL, 0UL );
      checkNonZeros( mat2, 3UL, 1UL, 0UL );
      checkNonZeros( mat2, 3UL, 0UL, 1UL );
      checkNonZeros( mat2, 3UL, 1UL, 1UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 2 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 4 || mat2(0,1,1) != 5 || mat2(0,1,2) != 6 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 2 || mat2(1,0,2) != 3 ||
          mat2(1,1,0) != 4 || mat2(1,1,1) != 5 || mat2(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   //=====================================================================================
   // move constructor
   //=====================================================================================

   {
      test_ = "DynamicTensor move constructor (0x0x0)";

      blaze::DynamicTensor<int> mat1( 0UL, 0UL, 0UL );
      blaze::DynamicTensor<int> mat2( std::move( mat1 ) );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "DynamicTensor move constructor (0x3x2)";

      blaze::DynamicTensor<int> mat1( 2UL, 0UL, 3UL );
      blaze::DynamicTensor<int> mat2( std::move( mat1 ) );

      checkRows    ( mat2, 0UL );
      checkColumns ( mat2, 3UL );
      checkPages   ( mat2, 2UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "DynamicTensor move constructor (2x0x1)";

      blaze::DynamicTensor<int> mat1( 1UL, 2UL, 0UL );
      blaze::DynamicTensor<int> mat2( std::move( mat1 ) );

      checkRows    ( mat2, 2UL );
      checkColumns ( mat2, 0UL );
      checkPages   ( mat2, 1UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "DynamicTensor move constructor (2x1x0)";

      blaze::DynamicTensor<int> mat1( 0UL, 2UL, 1UL );
      blaze::DynamicTensor<int> mat2( std::move( mat1 ) );

      checkRows    ( mat2, 2UL );
      checkColumns ( mat2, 1UL );
      checkPages   ( mat2, 0UL );
      checkNonZeros( mat2, 0UL );
   }

   {
      test_ = "DynamicTensor copy constructor (2x3x2)";

      blaze::DynamicTensor<int> mat1{{{1, 2, 3}, {4, 5, 6}},
                                     {{1, 2, 3}, {4, 5, 6}}};

      blaze::DynamicTensor<int> mat2( std::move( mat1 ) );

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2, 12UL );
      checkNonZeros( mat2, 3UL, 0UL, 0UL );
      checkNonZeros( mat2, 3UL, 1UL, 0UL );
      checkNonZeros( mat2, 3UL, 0UL, 1UL );
      checkNonZeros( mat2, 3UL, 1UL, 1UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 2 || mat2(0,0,2) != 3 ||
          mat2(0,1,0) != 4 || mat2(0,1,1) != 5 || mat2(0,1,2) != 6 ||
          mat2(1,0,0) != 1 || mat2(1,0,1) != 2 || mat2(1,0,2) != 3 ||
          mat2(1,1,0) != 4 || mat2(1,1,1) != 5 || mat2(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n( 1 2 3 )\n( 4 5 6 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   //=====================================================================================
   // dense tensor constructor
   //=====================================================================================

   {
      test_ = "DynamicTensor dense tensor constructor (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      AlignedPadded mat1( memory.get(), 2UL, 2UL, 3UL, 16UL );
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

      const blaze::DynamicTensor<int> mat2( mat1 );

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
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
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n)(( 1 2 3 )\n( 4 5 6 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor dense tensor constructor (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[13UL] );
      UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 2UL, 3UL );
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

      const blaze::DynamicTensor<int> mat2( mat1 );

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
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
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n)(( 1 2 3 )\n( 4 5 6 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DynamicTensor assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the DynamicTensor class template.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testAssignment()
{
   //=====================================================================================
   // homogeneous assignment
   //=====================================================================================

   {
      test_ = "DynamicTensor homogeneous assignment";

      blaze::DynamicTensor<int> tens( 2UL, 3UL, 4UL );
      tens = 2;

      checkRows    ( tens,  3UL );
      checkColumns ( tens,  4UL );
      checkPages   ( tens,  2UL );
      checkCapacity( tens, 24UL );
      checkNonZeros( tens, 24UL );
      checkNonZeros( tens,  0UL, 0UL, 4UL );
      checkNonZeros( tens,  1UL, 0UL, 4UL );
      checkNonZeros( tens,  2UL, 0UL, 4UL );
      checkNonZeros( tens,  0UL, 1UL, 4UL );
      checkNonZeros( tens,  1UL, 1UL, 4UL );
      checkNonZeros( tens,  2UL, 1UL, 4UL );

      if( tens(0,0,0) != 2 || tens(0,0,1) != 2 || tens(0,0,2) != 2 || tens(0,0,3) != 2 ||
          tens(0,1,0) != 2 || tens(0,1,1) != 2 || tens(0,1,2) != 2 || tens(0,1,3) != 2 ||
          tens(0,2,0) != 2 || tens(0,2,1) != 2 || tens(0,2,2) != 2 || tens(0,2,3) != 2 ||
          tens(1,0,0) != 2 || tens(1,0,1) != 2 || tens(1,0,2) != 2 || tens(1,0,3) != 2 ||
          tens(1,1,0) != 2 || tens(1,1,1) != 2 || tens(1,1,2) != 2 || tens(1,1,3) != 2 ||
          tens(1,2,0) != 2 || tens(1,2,1) != 2 || tens(1,2,2) != 2 || tens(1,2,3) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << tens << "\n"
             << "   Expected result:\n(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n)(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // list assignment
   //=====================================================================================

   {
      test_ = "DynamicTensor initializer list assignment (complete list)";

      blaze::DynamicTensor<int> tens;
      tens = {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}};

      checkRows    ( tens,  2UL );
      checkColumns ( tens,  3UL );
      checkPages   ( tens,  2UL );
      checkCapacity( tens, 12UL );
      checkNonZeros( tens, 12UL );
      checkNonZeros( tens,  0UL, 0UL, 3UL );
      checkNonZeros( tens,  1UL, 0UL, 3UL );
      checkNonZeros( tens,  0UL, 1UL, 3UL );
      checkNonZeros( tens,  1UL, 1UL, 3UL );

      if( tens(0,0,0) != 1 || tens(0,0,1) != 2 || tens(0,0,2) != 3 ||
          tens(0,1,0) != 4 || tens(0,1,1) != 5 || tens(0,1,2) != 6 ||
          tens(1,0,0) != 1 || tens(1,0,1) != 2 || tens(1,0,2) != 3 ||
          tens(1,1,0) != 4 || tens(1,1,1) != 5 || tens(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << tens << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n)(( 1 2 3 )\n( 4 5 6 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor initializer list assignment (incomplete list)";

      blaze::DynamicTensor<int> tens;
      tens = {{{1}, {4, 5, 6}}, {{1}, {4, 5, 6}}};

      checkRows    ( tens,  2UL );
      checkColumns ( tens,  3UL );
      checkPages   ( tens,  2UL );
      checkCapacity( tens, 12UL );
      checkNonZeros( tens,  8UL );
      checkNonZeros( tens, 0UL, 0UL, 1UL );
      checkNonZeros( tens, 1UL, 0UL, 3UL );
      checkNonZeros( tens, 0UL, 1UL, 1UL );
      checkNonZeros( tens, 1UL, 1UL, 3UL );

      if( tens(0,0,0) != 1 || tens(0,0,1) != 0 || tens(0,0,2) != 0 ||
          tens(0,1,0) != 4 || tens(0,1,1) != 5 || tens(0,1,2) != 6 ||
          tens(1,0,0) != 1 || tens(1,0,1) != 0 || tens(1,0,2) != 0 ||
          tens(1,1,0) != 4 || tens(1,1,1) != 5 || tens(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << tens << "\n"
             << "   Expected result:\n(( 1 0 0 )\n( 4 5 6 ))\n(( 1 0 0 )\n( 4 5 6 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // array assignment
   //=====================================================================================

   {
      test_ = "DynamicTensor array assignment";

      const int array[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}};
      blaze::DynamicTensor<int> tens;
      tens = array;

      checkRows    ( tens,  2UL );
      checkColumns ( tens,  3UL );
      checkPages   ( tens,  2UL );
      checkCapacity( tens, 12UL );
      checkNonZeros( tens, 12UL );
      checkNonZeros( tens,  0UL, 0UL, 3UL );
      checkNonZeros( tens,  1UL, 0UL, 3UL );
      checkNonZeros( tens,  0UL, 1UL, 3UL );
      checkNonZeros( tens,  1UL, 1UL, 3UL );

      if( tens(0,0,0) != 1 || tens(0,0,1) != 2 || tens(0,0,2) != 3 ||
          tens(0,1,0) != 4 || tens(0,1,1) != 5 || tens(0,1,2) != 6 ||
          tens(1,0,0) != 1 || tens(1,0,1) != 2 || tens(1,0,2) != 3 ||
          tens(1,1,0) != 4 || tens(1,1,1) != 5 || tens(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << tens << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n)(( 1 2 3 )\n( 4 5 6 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // copy assignment
   //=====================================================================================

   {
      test_ = "DynamicTensor copy assignment";

      blaze::DynamicTensor<int> mat1{{{1, 2, 3}, {4, 5, 6}},
                                     {{1, 2, 3}, {4, 5, 6}}};

      blaze::DynamicTensor<int> tens;
      tens = mat1;

      checkRows    ( tens,  2UL );
      checkColumns ( tens,  3UL );
      checkPages   ( tens,  2UL );
      checkCapacity( tens, 12UL );
      checkNonZeros( tens, 12UL );
      checkNonZeros( tens,  0UL, 0UL, 3UL );
      checkNonZeros( tens,  1UL, 0UL, 3UL );
      checkNonZeros( tens,  0UL, 1UL, 3UL );
      checkNonZeros( tens,  1UL, 1UL, 3UL );

      if( tens(0,0,0) != 1 || tens(0,0,1) != 2 || tens(0,0,2) != 3 ||
          tens(0,1,0) != 4 || tens(0,1,1) != 5 || tens(0,1,2) != 6 ||
          tens(1,0,0) != 1 || tens(1,0,1) != 2 || tens(1,0,2) != 3 ||
          tens(1,1,0) != 4 || tens(1,1,1) != 5 || tens(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << tens << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n)(( 1 2 3 )\n( 4 5 6 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor copy assignment stress test";

      using RandomMatrixType = blaze::DynamicTensor<int>;

      blaze::DynamicTensor<int> mat1;
      const int min( randmin );
      const int max( randmax );

      for( size_t i=0UL; i<100UL; ++i )
      {
         const size_t rows   ( blaze::rand<size_t>( 0UL, 10UL ) );
         const size_t columns( blaze::rand<size_t>( 0UL, 10UL ) );
         const size_t pages  ( blaze::rand<size_t>( 0UL, 10UL ) );
         const RandomMatrixType mat2( blaze::rand<RandomMatrixType>( pages, rows, columns, min, max ) );

         mat1 = mat2;

         if( mat1 != mat2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment failed\n"
                << " Details:\n"
                << "   Result:\n" << mat1 << "\n"
                << "   Expected result:\n" << mat2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // move assignment
   //=====================================================================================

   {
      test_ = "DynamicTensor move assignment";

      blaze::DynamicTensor<int> mat1{{{1, 2, 3}, {4, 5, 6}},
                                     {{1, 2, 3}, {4, 5, 6}}};

      blaze::DynamicTensor<int> tens{{{11}, {12}, {13}, {14}},
                                     {{11}, {12}, {13}, {14}}};

      tens = std::move( mat1 );

      checkRows    ( tens,  2UL );
      checkColumns ( tens,  3UL );
      checkPages   ( tens,  2UL );
      checkCapacity( tens, 12UL );
      checkNonZeros( tens, 12UL );
      checkNonZeros( tens,  0UL, 0UL, 3UL );
      checkNonZeros( tens,  1UL, 0UL, 3UL );
      checkNonZeros( tens,  0UL, 1UL, 3UL );
      checkNonZeros( tens,  1UL, 1UL, 3UL );

      if( tens(0,0,0) != 1 || tens(0,0,1) != 2 || tens(0,0,2) != 3 ||
          tens(0,1,0) != 4 || tens(0,1,1) != 5 || tens(0,1,2) != 6 ||
          tens(1,0,0) != 1 || tens(1,0,1) != 2 || tens(1,0,2) != 3 ||
          tens(1,1,0) != 4 || tens(1,1,1) != 5 || tens(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << tens << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n)(( 1 2 3 )\n( 4 5 6 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // dense tensor assignment
   //=====================================================================================

   {
      test_ = "DynamicTensor dense tensor assignment (mixed type)";

      blaze::DynamicTensor<short> mat1{{{1, 2, 3}, {4, 5, 6}},
                                       {{1, 2, 3}, {4, 5, 6}}};
      blaze::DynamicTensor<int> tens;
      tens = mat1;

      checkRows    ( tens,  2UL );
      checkColumns ( tens,  3UL );
      checkPages   ( tens,  2UL );
      checkCapacity( tens, 12UL );
      checkNonZeros( tens, 12UL );
      checkNonZeros( tens,  0UL, 0UL, 3UL );
      checkNonZeros( tens,  1UL, 0UL, 3UL );
      checkNonZeros( tens,  0UL, 1UL, 3UL );
      checkNonZeros( tens,  1UL, 1UL, 3UL );

      if( tens(0,0,0) != 1 || tens(0,0,1) != 2 || tens(0,0,2) != 3 ||
          tens(0,1,0) != 4 || tens(0,1,1) != 5 || tens(0,1,2) != 6 ||
          tens(1,0,0) != 1 || tens(1,0,1) != 2 || tens(1,0,2) != 3 ||
          tens(1,1,0) != 4 || tens(1,1,1) != 5 || tens(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << tens << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n)(( 1 2 3 )\n( 4 5 6 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor dense tensor assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      AlignedPadded mat1( memory.get(), 2UL, 2UL, 3UL, 16UL );
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

      blaze::DynamicTensor<int> mat2;
      mat2 = mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
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
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n)(( 1 2 3 )\n( 4 5 6 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor dense tensor assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[13UL] );
      UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 2UL, 3UL );
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

      blaze::DynamicTensor<int> mat2;
      mat2 = mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
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
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n)(( 1 2 3 )\n( 4 5 6 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor dense tensor assignment stress test";

      using RandomMatrixType = blaze::DynamicTensor<short>;

      blaze::DynamicTensor<int> mat1;
      const short min( randmin );
      const short max( randmax );

      for( size_t i=0UL; i<100UL; ++i )
      {
         const size_t rows   ( blaze::rand<size_t>( 0UL, 10UL ) );
         const size_t columns( blaze::rand<size_t>( 0UL, 10UL ) );
         const size_t pages  ( blaze::rand<size_t>( 0UL, 10UL ) );
         const RandomMatrixType mat2( blaze::rand<RandomMatrixType>( pages, rows, columns, min, max ) );

         mat1 = mat2;

         if( mat1 != mat2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment failed\n"
                << " Details:\n"
                << "   Result:\n" << mat1 << "\n"
                << "   Expected result:\n" << mat2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the DynamicTensor addition assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the addition assignment operators of the DynamicTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testAddAssign()
{
   //=====================================================================================
   // dense tensor addition assignment
   //=====================================================================================

   {
      test_ = "DynamicTensor dense tensor addition assignment (mixed type)";

      blaze::DynamicTensor<short> mat1{{{1, 2, 0}, {-3, 0, 4}},
                                       {{1, 2, 0}, {-3, 0, 4}}};

      blaze::DynamicTensor<int> mat2{{{0, -2, 6}, {5, 0, 0}},
                                     {{0, -2, 6}, {5, 0, 0}}};

      mat2 += mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2,  8UL );
      checkNonZeros( mat2,  0UL, 0UL, 2UL );
      checkNonZeros( mat2,  1UL, 0UL, 2UL );
      checkNonZeros( mat2,  0UL, 1UL, 2UL );
      checkNonZeros( mat2,  1UL, 1UL, 2UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 0 || mat2(0,0,2) != 6 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 0 || mat2(0,1,2) != 4 ||
          mat2(0,0,0) != 1 || mat2(0,0,1) != 0 || mat2(0,0,2) != 6 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 0 || mat2(0,1,2) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 0 6 )\n( 2 0 4 ))\n(( 1 0 6 )\n( 2 0 4 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor dense tensor addition assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      AlignedPadded mat1( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat1 = 0;
      mat1(0,0,0) =  1;
      mat1(0,0,1) =  2;
      mat1(0,1,0) = -3;
      mat1(0,1,2) =  4;
      mat1(1,0,0) =  1;
      mat1(1,0,1) =  2;
      mat1(1,1,0) = -3;
      mat1(1,1,2) =  4;

      blaze::DynamicTensor<int> mat2{{{0, -2, 6}, {5, 0, 0}},
                                     {{0, -2, 6}, {5, 0, 0}}};

      mat2 += mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
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
      test_ = "DynamicTensor dense tensor addition assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[13UL] );
      UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 2UL, 3UL );
      mat1 = 0;
      mat1(0,0,0) =  1;
      mat1(0,0,1) =  2;
      mat1(0,1,0) = -3;
      mat1(0,1,2) =  4;
      mat1(1,0,0) =  1;
      mat1(1,0,1) =  2;
      mat1(1,1,0) = -3;
      mat1(1,1,2) =  4;

      blaze::DynamicTensor<int> mat2{{{0, -2, 6}, {5, 0, 0}},
                                     {{0, -2, 6}, {5, 0, 0}}};

      mat2 += mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2,  8UL );
      checkNonZeros( mat2,  0UL, 0UL, 2UL );
      checkNonZeros( mat2,  1UL, 0UL, 2UL );
      checkNonZeros( mat2,  0UL, 1UL, 2UL );
      checkNonZeros( mat2,  1UL, 1UL, 2UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 0 || mat2(0,0,2) != 6 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 0 || mat2(0,1,2) != 4 ||
          mat2(0,0,0) != 1 || mat2(0,0,1) != 0 || mat2(0,0,2) != 6 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 0 || mat2(0,1,2) != 4 ) {
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
/*!\brief Test of the DynamicTensor subtraction assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the subtraction assignment operators of the DynamicTensor
// class template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testSubAssign()
{
   //=====================================================================================
   // dense tensor subtraction assignment
   //=====================================================================================

   {
      test_ = "DynamicTensor dense tensor subtraction assignment (mixed type)";

      blaze::DynamicTensor<short> mat1{{{-1, -2, 0}, {3, 0, -4}},
                                       {{-1, -2, 0}, {3, 0, -4}}};

      blaze::DynamicTensor<int> mat2{{{0, -2, 6}, {5, 0, 0}},
                                     {{0, -2, 6}, {5, 0, 0}}};

      mat2 -= mat1;

      checkRows    ( mat2,  2UL );
      checkColumns ( mat2,  3UL );
      checkPages   ( mat2,  2UL );
      checkCapacity( mat2, 12UL );
      checkNonZeros( mat2,  8UL );
      checkNonZeros( mat2,  0UL, 0UL, 2UL );
      checkNonZeros( mat2,  1UL, 0UL, 2UL );
      checkNonZeros( mat2,  0UL, 1UL, 2UL );
      checkNonZeros( mat2,  1UL, 1UL, 2UL );

      if( mat2(0,0,0) != 1 || mat2(0,0,1) != 0 || mat2(0,0,2) != 6 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 0 || mat2(0,1,2) != 4 ||
          mat2(0,0,0) != 1 || mat2(0,0,1) != 0 || mat2(0,0,2) != 6 ||
          mat2(0,1,0) != 2 || mat2(0,1,1) != 0 || mat2(0,1,2) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n(( 1 0 6 )\n( 2 0 4 ))\n(( 1 0 6 )\n( 2 0 4 )\n)\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "DynamicTensor dense tensor subtraction assignment (aligned/padded)";

      using blaze::aligned;
      using blaze::padded;

      using AlignedPadded = blaze::CustomTensor<int,aligned,padded>;
      std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 64UL ) );
      AlignedPadded mat1( memory.get(), 2UL, 2UL, 3UL, 16UL );
      mat1 = 0;
      mat1(0,0,0) = -1;
      mat1(0,0,1) = -2;
      mat1(0,1,0) =  3;
      mat1(0,1,2) = -4;
      mat1(1,0,0) = -1;
      mat1(1,0,1) = -2;
      mat1(1,1,0) =  3;
      mat1(1,1,2) = -4;

      blaze::DynamicTensor<int> mat2( 2UL, 2UL, 3UL, 0 );
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
      checkCapacity( mat2, 12UL );
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
      test_ = "DynamicTensor dense tensor subtraction assignment (unaligned/unpadded)";

      using blaze::unaligned;
      using blaze::unpadded;

      using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded>;
      std::unique_ptr<int[]> memory( new int[13UL] );
      UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 2UL, 3UL );
      mat1 = 0;
      mat1(0,0,0) = -1;
      mat1(0,0,1) = -2;
      mat1(0,1,0) =  3;
      mat1(0,1,2) = -4;
      mat1(1,0,0) = -1;
      mat1(1,0,1) = -2;
      mat1(1,1,0) =  3;
      mat1(1,1,2) = -4;

      blaze::DynamicTensor<int> mat2( 2UL, 2UL, 3UL, 0 );
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
      checkCapacity( mat2, 12UL );
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

} // namespace dynamictensor

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
   std::cout << "   Running DynamicTensor class test (part 1)..." << std::endl;

   try
   {
      RUN_DYNAMICTENSOR_CLASS_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during DynamicTensor class test (part 1):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
