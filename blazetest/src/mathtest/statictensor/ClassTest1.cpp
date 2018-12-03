//=================================================================================================
/*!
//  \file src/mathtest/statictensor/ClassTest1.cpp
//  \brief Source file for the StaticTensor class test (part 1)
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
#include <blaze/util/Complex.h>
#include <blaze/util/Memory.h>
#include <blaze/util/policies/Deallocate.h>
#include <blaze/util/Random.h>
#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>

// #include <blaze_tensor/math/CompressedTensor.h>
#include <blaze_tensor/math/CustomTensor.h>
// #include <blaze_tensor/math/DiagonalTensor.h>
#include <blaze_tensor/math/DynamicTensor.h>
// #include <blaze_tensor/math/LowerTensor.h>
// #include <blaze_tensor/math/UpperTensor.h>

#include <blazetest/mathtest/statictensor/ClassTest.h>

namespace blazetest {

namespace mathtest {

namespace statictensor {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the StaticTensor class test.
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
/*!\brief Test of the StaticTensor constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the StaticTensor class template.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testConstructors()
{
   //=====================================================================================
   // Row-major default constructor
   //=====================================================================================

   {
      test_ = "Row-major StaticTensor default constructor (0x0)";

      blaze::StaticTensor<int,0UL,0UL,0UL> mat;

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "Row-major StaticTensor default constructor (0x0x4)";

      blaze::StaticTensor<int,0UL,0UL,4UL> mat;

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 4UL );
      checkPages   ( mat, 0UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "Row-major StaticTensor default constructor (0x3x0)";

      blaze::StaticTensor<int,0UL,3UL,0UL> mat;

      checkRows    ( mat, 3UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "Row-major StaticTensor default constructor (2x0x0)";

      blaze::StaticTensor<int,2UL,0UL,0UL> mat;

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 2UL );
      checkCapacity( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "Row-major StaticTensor default constructor (2x3x4)";

      blaze::StaticTensor<int,2UL,3UL,4UL> mat;

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  4UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 24UL );
      checkNonZeros( mat,  0UL );
      checkNonZeros( mat,  0UL, 0UL, 0UL );
      checkNonZeros( mat,  1UL, 0UL, 0UL );
      checkNonZeros( mat,  2UL, 0UL, 0UL );
      checkNonZeros( mat,  0UL, 1UL, 0UL );
      checkNonZeros( mat,  1UL, 1UL, 0UL );
      checkNonZeros( mat,  2UL, 1UL, 0UL );

      if( mat(0,0,0) != 0 || mat(0,0,1) != 0 || mat(0,0,2) != 0 || mat(0,0,3) != 0 ||
          mat(0,1,0) != 0 || mat(0,1,1) != 0 || mat(0,1,2) != 0 || mat(0,1,3) != 0 ||
          mat(0,2,0) != 0 || mat(0,2,1) != 0 || mat(0,2,2) != 0 || mat(0,2,3) != 0 ||
          mat(1,0,0) != 0 || mat(1,0,1) != 0 || mat(1,0,2) != 0 || mat(1,0,3) != 0 ||
          mat(1,1,0) != 0 || mat(1,1,1) != 0 || mat(1,1,2) != 0 || mat(1,1,3) != 0 ||
          mat(1,2,0) != 0 || mat(1,2,1) != 0 || mat(1,2,2) != 0 || mat(1,2,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n"
                     "(( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n"
                     " ( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major homogeneous initialization
   //=====================================================================================

   {
      test_ = "Row-major StaticTensor homogeneous initialization constructor";

      blaze::StaticTensor<int,2UL,3UL,4UL> mat( 2 );

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  4UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 24UL );
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
          mat(1,0,0) != 2 || mat(1,0,1) != 2 || mat(1,0,2) != 2 || mat(1,0,3) != 2 ||
          mat(1,1,0) != 2 || mat(1,1,1) != 2 || mat(1,1,2) != 2 || mat(1,1,3) != 2 ||
          mat(1,2,0) != 2 || mat(1,2,1) != 2 || mat(1,2,2) != 2 || mat(1,2,3) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n"
                     "(( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n"
                     " ( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major list initialization
   //=====================================================================================

   {
      test_ = "Row-major StaticTensor initializer list constructor (incomplete list)";

      blaze::StaticTensor<int, 2UL, 2UL, 3UL> mat{{{1}, {4, 5, 6}},
                                                  {{1}, {4, 5, 6}}};

      checkRows    ( mat,  2UL );
      checkColumns ( mat,  3UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 12UL );
      checkNonZeros( mat,  8UL );
      checkNonZeros( mat, 0UL, 0UL, 1UL );
      checkNonZeros( mat, 0UL, 0UL, 1UL );
      checkNonZeros( mat, 1UL, 1UL, 3UL );
      checkNonZeros( mat, 1UL, 1UL, 3UL );

      if( mat(0,0,0) != 1 || mat(0,0,1) != 0 || mat(0,0,2) != 0 ||
          mat(0,1,0) != 4 || mat(0,1,1) != 5 || mat(0,1,2) != 6 ||
          mat(1,0,0) != 1 || mat(1,0,1) != 0 || mat(1,0,2) != 0 ||
          mat(1,1,0) != 4 || mat(1,1,1) != 5 || mat(1,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major StaticTensor initializer list constructor (complete list)";

      blaze::StaticTensor<int, 2UL, 2UL, 3UL> mat{{{1, 2, 3}, {4, 5, 6}},
                                                  {{1, 2, 3}, {4, 5, 6}}};

      checkRows    ( mat,  2UL );
      checkColumns ( mat,  3UL );
      checkPages   ( mat,  2UL );
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
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major array initialization
   //=====================================================================================

   {
      test_ = "Row-major StaticTensor dynamic array initialization constructor";

      std::unique_ptr<int[]> array( new int[6] );
      array[0] = 1;
      array[1] = 2;
      array[2] = 3;
      array[3] = 4;
      array[4] = 5;
      array[5] = 6;
      blaze::StaticTensor<int,1UL,2UL,3UL> mat( 1UL, 2UL, 3UL, array.get() );

      checkRows    ( mat, 2UL );
      checkColumns ( mat, 3UL );
      checkPages   ( mat, 1UL );
      checkCapacity( mat, 6UL );
      checkNonZeros( mat, 6UL );
      checkNonZeros( mat, 0UL, 0UL, 3UL );
      checkNonZeros( mat, 1UL, 0UL, 3UL );

      if( mat(0,0,0) != 1 || mat(0,0,1) != 2 || mat(0,0,2) != 3 ||
          mat(0,1,0) != 4 || mat(0,1,1) != 5 || mat(0,1,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n( 1 2 3 0 )\n( 4 5 6 0 )\n( 0 0 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major StaticTensor dynamic array initialization constructor";

      std::unique_ptr<int[]> array( new int[6] );
      array[0] = 1;
      array[1] = 2;
      array[2] = 3;
      array[3] = 4;
      array[4] = 5;
      array[5] = 6;
      blaze::StaticTensor<int,2UL,2UL,3UL> mat( 1UL, 2UL, 3UL, array.get() );

      checkRows    ( mat, 2UL );
      checkColumns ( mat, 3UL );
      checkPages   ( mat, 2UL );
      checkCapacity( mat, 6UL );
      checkNonZeros( mat, 6UL );
      checkNonZeros( mat, 0UL, 0UL, 3UL );
      checkNonZeros( mat, 1UL, 0UL, 3UL );

      if( mat(0,0,0) != 1 || mat(0,0,1) != 2 || mat(0,0,2) != 3 ||
          mat(0,1,0) != 4 || mat(0,1,1) != 5 || mat(0,1,2) != 6 ||
          mat(1,0,0) != 0 || mat(1,0,1) != 0 || mat(1,0,2) != 0 ||
          mat(1,1,0) != 0 || mat(1,1,1) != 0 || mat(1,1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n(( 1 2 3 )\n( 4 5 6 )\n( 0 0 0 )\n( 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

//    {
//       test_ = "Row-major StaticTensor static array initialization constructor";
//
//       const int array[2][3] = { { 1, 2, 3 }, { 4, 5, 6 } };
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat( array );
//
//       checkRows    ( mat, 2UL );
//       checkColumns ( mat, 3UL );
//       checkCapacity( mat, 6UL );
//       checkNonZeros( mat, 6UL );
//       checkNonZeros( mat, 0UL, 3UL );
//       checkNonZeros( mat, 1UL, 3UL );
//
//       if( mat(0,0) != 1 || mat(0,1) != 2 || mat(0,2) != 3 ||
//           mat(1,0) != 4 || mat(1,1) != 5 || mat(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Construction failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Row-major copy constructor
//    //=====================================================================================
//
//    {
//       test_ = "Row-major StaticTensor copy constructor";
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat1{ { 1, 2, 3 },
//                                                              { 4, 5, 6 } };
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2( mat1 );
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 3UL );
//       checkNonZeros( mat2, 1UL, 3UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Construction failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Row-major dense tensor constructor
//    //=====================================================================================
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor constructor (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::rowMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,rowMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(0,2) = 3;
//       mat1(1,0) = 4;
//       mat1(1,1) = 5;
//       mat1(1,2) = 6;
//
//       const blaze::StaticTensor<int,2UL,2UL,3UL> mat2( mat1 );
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 3UL );
//       checkNonZeros( mat2, 1UL, 3UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Construction failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor constructor (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::rowMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,rowMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(0,2) = 3;
//       mat1(1,0) = 4;
//       mat1(1,1) = 5;
//       mat1(1,2) = 6;
//
//       const blaze::StaticTensor<int,2UL,2UL,3UL> mat2( mat1 );
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 3UL );
//       checkNonZeros( mat2, 1UL, 3UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Construction failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor constructor (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::columnMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,columnMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 48UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(0,2) = 3;
//       mat1(1,0) = 4;
//       mat1(1,1) = 5;
//       mat1(1,2) = 6;
//
//       const blaze::StaticTensor<int,2UL,2UL,3UL> mat2( mat1 );
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 3UL );
//       checkNonZeros( mat2, 1UL, 3UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Construction failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor constructor (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::columnMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,columnMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(0,2) = 3;
//       mat1(1,0) = 4;
//       mat1(1,1) = 5;
//       mat1(1,2) = 6;
//
//       const blaze::StaticTensor<int,2UL,2UL,3UL> mat2( mat1 );
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 3UL );
//       checkNonZeros( mat2, 1UL, 3UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Construction failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Row-major sparse tensor constructor
//    //=====================================================================================
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor constructor";
//
//       blaze::CompressedTensor<int> mat1( 2UL, 3UL, 4UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(1,0) = 3;
//       mat1(1,2) = 4;
//
//       const blaze::StaticTensor<int,2UL,2UL,3UL> mat2( mat1 );
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 0 ||
//           mat2(1,0) != 3 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Construction failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 0 )\n( 3 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor constructor";
//
//       blaze::CompressedTensor<int,blaze::columnMajor> mat1( 2UL, 3UL, 4UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(1,0) = 3;
//       mat1(1,2) = 4;
//
//       const blaze::StaticTensor<int,2UL,2UL,3UL> mat2( mat1 );
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 0 ||
//           mat2(1,0) != 3 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Construction failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 0 )\n( 3 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the StaticTensor assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the StaticTensor class template.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testAssignment()
{
//    //=====================================================================================
//    // Row-major homogeneous assignment
//    //=====================================================================================
//
//    {
//       test_ = "Row-major StaticTensor homogeneous assignment";
//
//       blaze::StaticTensor<int,2UL,3UL,4UL> mat;
//       mat = 2;
//
//       checkRows    ( mat,  3UL );
//       checkColumns ( mat,  4UL );
//       checkCapacity( mat, 12UL );
//       checkNonZeros( mat, 12UL );
//       checkNonZeros( mat,  0UL, 4UL );
//       checkNonZeros( mat,  1UL, 4UL );
//       checkNonZeros( mat,  2UL, 4UL );
//
//       if( mat(0,0) != 2 || mat(0,1) != 2 || mat(0,2) != 2 || mat(0,3) != 2 ||
//           mat(1,0) != 2 || mat(1,1) != 2 || mat(1,2) != 2 || mat(1,3) != 2 ||
//           mat(2,0) != 2 || mat(2,1) != 2 || mat(2,2) != 2 || mat(2,3) != 2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat << "\n"
//              << "   Expected result:\n( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Row-major list assignment
//    //=====================================================================================
//
//    {
//       test_ = "Row-major StaticTensor initializer list assignment (complete list)";
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat;
//       mat = { { 1, 2, 3 }, { 4, 5, 6 } };
//
//       checkRows    ( mat, 2UL );
//       checkColumns ( mat, 3UL );
//       checkCapacity( mat, 6UL );
//       checkNonZeros( mat, 6UL );
//       checkNonZeros( mat, 0UL, 3UL );
//       checkNonZeros( mat, 1UL, 3UL );
//
//       if( mat(0,0) != 1 || mat(0,1) != 2 || mat(0,2) != 3 ||
//           mat(1,0) != 4 || mat(1,1) != 5 || mat(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major StaticTensor initializer list assignment (incomplete list)";
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat;
//       mat = { { 1 }, { 4, 5, 6 } };
//
//       checkRows    ( mat, 2UL );
//       checkColumns ( mat, 3UL );
//       checkCapacity( mat, 6UL );
//       checkNonZeros( mat, 4UL );
//       checkNonZeros( mat, 0UL, 1UL );
//       checkNonZeros( mat, 1UL, 3UL );
//
//       if( mat(0,0) != 1 || mat(0,1) != 0 || mat(0,2) != 0 ||
//           mat(1,0) != 4 || mat(1,1) != 5 || mat(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat << "\n"
//              << "   Expected result:\n( 1 0 0 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Row-major array assignment
//    //=====================================================================================
//
//    {
//       test_ = "Row-major StaticTensor array assignment";
//
//       const int array[2][3] = { { 1, 2, 3 }, { 4, 5, 6 } };
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat;
//       mat = array;
//
//       checkRows    ( mat, 2UL );
//       checkColumns ( mat, 3UL );
//       checkCapacity( mat, 6UL );
//       checkNonZeros( mat, 6UL );
//       checkNonZeros( mat, 0UL, 3UL );
//       checkNonZeros( mat, 1UL, 3UL );
//
//       if( mat(0,0) != 1 || mat(0,1) != 2 || mat(0,2) != 3 ||
//           mat(1,0) != 4 || mat(1,1) != 5 || mat(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Row-major copy assignment
//    //=====================================================================================
//
//    {
//       test_ = "Row-major StaticTensor copy assignment";
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat1{ { 1, 2, 3 },
//                                                              { 4, 5, 6 } };
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 3UL );
//       checkNonZeros( mat2, 1UL, 3UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major StaticTensor copy assignment stress test";
//
//       using RandomTensorType = blaze::StaticTensor<int,2UL,4UL,3UL>;
//
//       blaze::StaticTensor<int,2UL,4UL,3UL> mat1;
//       const int min( randmin );
//       const int max( randmax );
//
//       for( size_t i=0UL; i<100UL; ++i )
//       {
//          const RandomTensorType mat2( blaze::rand<RandomTensorType>( min, max ) );
//
//          mat1 = mat2;
//
//          if( mat1 != mat2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Assignment failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat1 << "\n"
//                 << "   Expected result:\n" << mat2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//
//    //=====================================================================================
//    // Row-major dense tensor assignment
//    //=====================================================================================
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor assignment (mixed type)";
//
//       blaze::StaticTensor<short,2UL,3UL> mat1{ { 1, 2, 3 }, { 4, 5, 6 } };
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 3UL );
//       checkNonZeros( mat2, 1UL, 3UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor assignment (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::rowMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,rowMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(0,2) = 3;
//       mat1(1,0) = 4;
//       mat1(1,1) = 5;
//       mat1(1,2) = 6;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 3UL );
//       checkNonZeros( mat2, 1UL, 3UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor assignment (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::rowMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,rowMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(0,2) = 3;
//       mat1(1,0) = 4;
//       mat1(1,1) = 5;
//       mat1(1,2) = 6;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 3UL );
//       checkNonZeros( mat2, 1UL, 3UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor assignment stress test";
//
//       using RandomTensorType = blaze::DynamicTensor<int>;
//
//       blaze::StaticTensor<int,2UL,4UL,3UL> mat1;
//       const int min( randmin );
//       const int max( randmax );
//
//       for( size_t i=0UL; i<100UL; ++i )
//       {
//          const RandomTensorType mat2( blaze::rand<RandomTensorType>( 4UL, 3UL, min, max ) );
//
//          mat1 = mat2;
//
//          if( mat1 != mat2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Assignment failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat1 << "\n"
//                 << "   Expected result:\n" << mat2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor assignment (mixed type)";
//
//       blaze::StaticTensor<short,2UL,3UL,blaze::columnMajor> mat1{ { 1, 2, 3 }, { 4, 5, 6 } };
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 3UL );
//       checkNonZeros( mat2, 1UL, 3UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor assignment (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::columnMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,columnMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 48UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(0,2) = 3;
//       mat1(1,0) = 4;
//       mat1(1,1) = 5;
//       mat1(1,2) = 6;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 3UL );
//       checkNonZeros( mat2, 1UL, 3UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor assignment (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::columnMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,columnMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(0,2) = 3;
//       mat1(1,0) = 4;
//       mat1(1,1) = 5;
//       mat1(1,2) = 6;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 3UL );
//       checkNonZeros( mat2, 1UL, 3UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor assignment stress test";
//
//       using RandomTensorType = blaze::DynamicTensor<int,blaze::columnMajor>;
//
//       blaze::StaticTensor<int,2UL,4UL,3UL> mat1;
//       const int min( randmin );
//       const int max( randmax );
//
//       for( size_t i=0UL; i<100UL; ++i )
//       {
//          const RandomTensorType mat2( blaze::rand<RandomTensorType>( 4UL, 3UL, min, max ) );
//
//          mat1 = mat2;
//
//          if( mat1 != mat2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Assignment failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat1 << "\n"
//                 << "   Expected result:\n" << mat2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor assignment (lower)";
//
//       blaze::LowerTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor assignment (lower)";
//
//       blaze::LowerTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor assignment (upper)";
//
//       blaze::UpperTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor assignment (upper)";
//
//       blaze::UpperTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Row-major sparse tensor assignment
//    //=====================================================================================
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor assignment";
//
//       blaze::CompressedTensor<int> mat1( 2UL, 3UL, 4UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(1,0) = 3;
//       mat1(1,2) = 4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 0 ||
//           mat2(1,0) != 3 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 0 )\n( 3 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor assignment stress test";
//
//       using RandomTensorType = blaze::CompressedTensor<int>;
//
//       blaze::StaticTensor<int,2UL,4UL,3UL> mat1;
//       const int min( randmin );
//       const int max( randmax );
//
//       for( size_t i=0UL; i<100UL; ++i )
//       {
//          const RandomTensorType mat2( blaze::rand<RandomTensorType>( 4UL, 3UL, min, max ) );
//
//          mat1 = mat2;
//
//          if( mat1 != mat2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Assignment failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat1 << "\n"
//                 << "   Expected result:\n" << mat2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor assignment";
//
//       blaze::CompressedTensor<int,blaze::columnMajor> mat1( 2UL, 3UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(1,0) = 3;
//       mat1(1,2) = 4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 0 ||
//           mat2(1,0) != 3 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 0 )\n( 3 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor assignment stress test";
//
//       using RandomTensorType = blaze::CompressedTensor<int,blaze::columnMajor>;
//
//       blaze::StaticTensor<int,2UL,4UL,3UL> mat1;
//       const int min( randmin );
//       const int max( randmax );
//
//       for( size_t i=0UL; i<100UL; ++i )
//       {
//          const RandomTensorType mat2( blaze::rand<RandomTensorType>( 4UL, 3UL, min, max ) );
//
//          mat1 = mat2;
//
//          if( mat1 != mat2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Assignment failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat1 << "\n"
//                 << "   Expected result:\n" << mat2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor assignment (upper)";
//
//       blaze::UpperTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor assignment (upper)";
//
//       blaze::UpperTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Column-major homogeneous assignment
//    //=====================================================================================
//
//    {
//       test_ = "Column-major StaticTensor homogeneous assignment";
//
//       blaze::StaticTensor<int,2UL,3UL,4UL,blaze::columnMajor> mat;
//       mat = 2;
//
//       checkRows    ( mat,  3UL );
//       checkColumns ( mat,  4UL );
//       checkCapacity( mat, 12UL );
//       checkNonZeros( mat, 12UL );
//       checkNonZeros( mat,  0UL, 3UL );
//       checkNonZeros( mat,  1UL, 3UL );
//       checkNonZeros( mat,  2UL, 3UL );
//       checkNonZeros( mat,  3UL, 3UL );
//
//       if( mat(0,0) != 2 || mat(0,1) != 2 || mat(0,2) != 2 || mat(0,3) != 2 ||
//           mat(1,0) != 2 || mat(1,1) != 2 || mat(1,2) != 2 || mat(1,3) != 2 ||
//           mat(2,0) != 2 || mat(2,1) != 2 || mat(2,2) != 2 || mat(2,3) != 2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat << "\n"
//              << "   Expected result:\n( 2 2 2 2 )\n( 2 2 2 2 )\n( 2 2 2 2 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Column-major list assignment
//    //=====================================================================================
//
//    {
//       test_ = "Column-major StaticTensor initializer list assignment (complete list)";
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat;
//       mat = { { 1, 2, 3 }, { 4, 5, 6 } };
//
//       checkRows    ( mat, 2UL );
//       checkColumns ( mat, 3UL );
//       checkCapacity( mat, 6UL );
//       checkNonZeros( mat, 6UL );
//       checkNonZeros( mat, 0UL, 2UL );
//       checkNonZeros( mat, 1UL, 2UL );
//       checkNonZeros( mat, 2UL, 2UL );
//
//       if( mat(0,0) != 1 || mat(0,1) != 2 || mat(0,2) != 3 ||
//           mat(1,0) != 4 || mat(1,1) != 5 || mat(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major StaticTensor initializer list assignment (incomplete list)";
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat;
//       mat = { { 1 }, { 4, 5, 6 } };
//
//       checkRows    ( mat, 2UL );
//       checkColumns ( mat, 3UL );
//       checkCapacity( mat, 6UL );
//       checkNonZeros( mat, 4UL );
//       checkNonZeros( mat, 0UL, 2UL );
//       checkNonZeros( mat, 1UL, 1UL );
//       checkNonZeros( mat, 2UL, 1UL );
//
//       if( mat(0,0) != 1 || mat(0,1) != 0 || mat(0,2) != 0 ||
//           mat(1,0) != 4 || mat(1,1) != 5 || mat(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat << "\n"
//              << "   Expected result:\n( 1 0 0 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Column-major array assignment
//    //=====================================================================================
//
//    {
//       test_ = "Column-major StaticTensor array assignment";
//
//       const int array[2][3] = { { 1, 2, 3 }, { 4, 5, 6 } };
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat;
//       mat = array;
//
//       checkRows    ( mat, 2UL );
//       checkColumns ( mat, 3UL );
//       checkCapacity( mat, 6UL );
//       checkNonZeros( mat, 6UL );
//       checkNonZeros( mat, 0UL, 2UL );
//       checkNonZeros( mat, 1UL, 2UL );
//       checkNonZeros( mat, 2UL, 2UL );
//
//       if( mat(0,0) != 1 || mat(0,1) != 2 || mat(0,2) != 3 ||
//           mat(1,0) != 4 || mat(1,1) != 5 || mat(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Column-major copy assignment
//    //=====================================================================================
//
//    {
//       test_ = "Column-major StaticTensor copy assignment";
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat1{ { 1, 3, 5 },
//                                                                 { 2, 4, 6 } };
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 3 || mat2(0,2) != 5 ||
//           mat2(1,0) != 2 || mat2(1,1) != 4 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 3 5 )\n( 2 4 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major StaticTensor copy assignment stress test";
//
//       using RandomTensorType = blaze::StaticTensor<int,2UL,4UL,3UL,blaze::columnMajor>;
//
//       blaze::StaticTensor<int,2UL,4UL,3UL,blaze::columnMajor> mat1;
//       const int min( randmin );
//       const int max( randmax );
//
//       for( size_t i=0UL; i<100UL; ++i )
//       {
//          const RandomTensorType mat2( blaze::rand<RandomTensorType>( min, max ) );
//
//          mat1 = mat2;
//
//          if( mat1 != mat2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Assignment failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat1 << "\n"
//                 << "   Expected result:\n" << mat2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//
//    //=====================================================================================
//    // Column-major dense tensor assignment
//    //=====================================================================================
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor assignment (mixed type)";
//
//       blaze::StaticTensor<short,2UL,3UL> mat1{ { 1, 2, 3 }, { 4, 5, 6 } };
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor assignment (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::rowMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,rowMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(0,2) = 3;
//       mat1(1,0) = 4;
//       mat1(1,1) = 5;
//       mat1(1,2) = 6;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor assignment (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::rowMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,rowMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(0,2) = 3;
//       mat1(1,0) = 4;
//       mat1(1,1) = 5;
//       mat1(1,2) = 6;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor assignment stress test";
//
//       using RandomTensorType = blaze::DynamicTensor<int>;
//
//       blaze::StaticTensor<int,2UL,4UL,3UL,blaze::columnMajor> mat1;
//       const int min( randmin );
//       const int max( randmax );
//
//       for( size_t i=0UL; i<100UL; ++i )
//       {
//          const RandomTensorType mat2( blaze::rand<RandomTensorType>( 4UL, 3UL, min, max ) );
//
//          mat1 = mat2;
//
//          if( mat1 != mat2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Assignment failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat1 << "\n"
//                 << "   Expected result:\n" << mat2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor assignment (mixed type)";
//
//       blaze::StaticTensor<short,2UL,3UL,blaze::columnMajor> mat1{ { 1, 2, 3 }, { 4, 5, 6 } };
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor assignment (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::columnMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,columnMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 48UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(0,2) = 3;
//       mat1(1,0) = 4;
//       mat1(1,1) = 5;
//       mat1(1,2) = 6;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor assignment (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::columnMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,columnMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(0,2) = 3;
//       mat1(1,0) = 4;
//       mat1(1,1) = 5;
//       mat1(1,2) = 6;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 6UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 3 ||
//           mat2(1,0) != 4 || mat2(1,1) != 5 || mat2(1,2) != 6 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 3 )\n( 4 5 6 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor assignment stress test";
//
//       using RandomTensorType = blaze::DynamicTensor<int,blaze::columnMajor>;
//
//       blaze::StaticTensor<int,2UL,4UL,3UL,blaze::columnMajor> mat1;
//       const int min( randmin );
//       const int max( randmax );
//
//       for( size_t i=0UL; i<100UL; ++i )
//       {
//          const RandomTensorType mat2( blaze::rand<RandomTensorType>( 4UL, 3UL, min, max ) );
//
//          mat1 = mat2;
//
//          if( mat1 != mat2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Assignment failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat1 << "\n"
//                 << "   Expected result:\n" << mat2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor assignment (lower)";
//
//       blaze::LowerTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor assignment (lower)";
//
//       blaze::LowerTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor assignment (upper)";
//
//       blaze::UpperTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor assignment (upper)";
//
//       blaze::UpperTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Column-major sparse tensor assignment
//    //=====================================================================================
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor assignment";
//
//       blaze::CompressedTensor<int> mat1( 2UL, 3UL, 4UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(1,0) = 3;
//       mat1(1,2) = 4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 1UL );
//       checkNonZeros( mat2, 2UL, 1UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 0 ||
//           mat2(1,0) != 3 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 0 )\n( 3 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor sparse tensor assignment stress test";
//
//       using RandomTensorType = blaze::CompressedTensor<int>;
//
//       blaze::StaticTensor<int,2UL,4UL,3UL,blaze::columnMajor> mat1;
//       const int min( randmin );
//       const int max( randmax );
//
//       for( size_t i=0UL; i<100UL; ++i )
//       {
//          const RandomTensorType mat2( blaze::rand<RandomTensorType>( 4UL, 3UL, min, max ) );
//
//          mat1 = mat2;
//
//          if( mat1 != mat2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Assignment failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat1 << "\n"
//                 << "   Expected result:\n" << mat2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor assignment";
//
//       blaze::CompressedTensor<int,blaze::columnMajor> mat1( 2UL, 3UL );
//       mat1(0,0) = 1;
//       mat1(0,1) = 2;
//       mat1(1,0) = 3;
//       mat1(1,2) = 4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2;
//       mat2 = mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 1UL );
//       checkNonZeros( mat2, 2UL, 1UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 2 || mat2(0,2) != 0 ||
//           mat2(1,0) != 3 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 2 0 )\n( 3 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor sparse tensor assignment stress test";
//
//       using RandomTensorType = blaze::CompressedTensor<int,blaze::columnMajor>;
//
//       blaze::StaticTensor<int,2UL,4UL,3UL,blaze::columnMajor> mat1;
//       const int min( randmin );
//       const int max( randmax );
//
//       for( size_t i=0UL; i<100UL; ++i )
//       {
//          const RandomTensorType mat2( blaze::rand<RandomTensorType>( 4UL, 3UL, min, max ) );
//
//          mat1 = mat2;
//
//          if( mat1 != mat2 ) {
//             std::ostringstream oss;
//             oss << " Test: " << test_ << "\n"
//                 << " Error: Assignment failed\n"
//                 << " Details:\n"
//                 << "   Result:\n" << mat1 << "\n"
//                 << "   Expected result:\n" << mat2 << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor sparse tensor assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor sparse tensor assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor sparse tensor assignment (upper)";
//
//       blaze::UpperTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor sparse tensor assignment (upper)";
//
//       blaze::UpperTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor sparse tensor assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor sparse tensor assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//       randomize( mat2 );
//
//       mat2 = mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the StaticTensor addition assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the addition assignment operators of the StaticTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testAddAssign()
{
//    //=====================================================================================
//    // Row-major dense tensor addition assignment
//    //=====================================================================================
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor addition assignment (mixed type)";
//
//       blaze::StaticTensor<short,2UL,3UL> mat1{ {  1, 2, 0 },
//                                                                { -3, 0, 4 } };
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor addition assignment (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::rowMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,rowMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1 = 0;
//       mat1(0,0) =  1;
//       mat1(0,1) =  2;
//       mat1(1,0) = -3;
//       mat1(1,2) =  4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor addition assignment (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::rowMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,rowMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1 = 0;
//       mat1(0,0) =  1;
//       mat1(0,1) =  2;
//       mat1(1,0) = -3;
//       mat1(1,2) =  4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor addition assignment (mixed type)";
//
//       blaze::StaticTensor<short,2UL,3UL,blaze::columnMajor> mat1{ {  1, 2, 0 },
//                                                                   { -3, 0, 4 } };
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor addition assignment (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::columnMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,columnMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 48UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1 = 0;
//       mat1(0,0) =  1;
//       mat1(0,1) =  2;
//       mat1(1,0) = -3;
//       mat1(1,2) =  4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor addition assignment (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::columnMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,columnMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1 = 0;
//       mat1(0,0) =  1;
//       mat1(0,1) =  2;
//       mat1(1,0) = -3;
//       mat1(1,2) =  4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor addition assignment (lower)";
//
//       blaze::LowerTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor addition assignment (lower)";
//
//       blaze::LowerTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor addition assignment (upper)";
//
//       blaze::UpperTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor addition assignment (upper)";
//
//       blaze::UpperTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor addition assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor addition assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Row-major sparse tensor addition assignment
//    //=====================================================================================
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor addition assignment";
//
//       blaze::CompressedTensor<int> mat1( 2UL, 3UL, 4UL );
//       mat1(0,0) =  1;
//       mat1(0,1) =  2;
//       mat1(1,0) = -3;
//       mat1(1,2) =  4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor addition assignment";
//
//       blaze::CompressedTensor<int,blaze::columnMajor> mat1( 2UL, 3UL, 4UL );
//       mat1(0,0) =  1;
//       mat1(0,1) =  2;
//       mat1(1,0) = -3;
//       mat1(1,2) =  4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor addition assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor addition assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor addition assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor addition assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor addition assignment (upper)";
//
//       blaze::UpperTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor addition assignment (upper)";
//
//       blaze::UpperTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor addition assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor addition assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Column-major dense tensor addition assignment
//    //=====================================================================================
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor addition assignment (mixed type)";
//
//       blaze::StaticTensor<short,2UL,3UL> mat1{ {  1, 2, 0 },
//                                                                { -3, 0, 4 } };
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor addition assignment (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::rowMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,rowMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1 = 0;
//       mat1(0,0) =  1;
//       mat1(0,1) =  2;
//       mat1(1,0) = -3;
//       mat1(1,2) =  4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor addition assignment (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::rowMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,rowMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1 = 0;
//       mat1(0,0) =  1;
//       mat1(0,1) =  2;
//       mat1(1,0) = -3;
//       mat1(1,2) =  4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor addition assignment (mixed type)";
//
//       blaze::StaticTensor<short,2UL,3UL,blaze::columnMajor> mat1{ {  1, 2, 0 },
//                                                                   { -3, 0, 4 } };
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor addition assignment (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::columnMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,columnMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 48UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1 = 0;
//       mat1(0,0) =  1;
//       mat1(0,1) =  2;
//       mat1(1,0) = -3;
//       mat1(1,2) =  4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor addition assignment (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::columnMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,columnMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1 = 0;
//       mat1(0,0) =  1;
//       mat1(0,1) =  2;
//       mat1(1,0) = -3;
//       mat1(1,2) =  4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor addition assignment (lower)";
//
//       blaze::LowerTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor addition assignment (lower)";
//
//       blaze::LowerTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor addition assignment (upper)";
//
//       blaze::UpperTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor addition assignment (upper)";
//
//       blaze::UpperTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor addition assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor addition assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Column-major sparse tensor addition assignment
//    //=====================================================================================
//
//    {
//       test_ = "Column-major/row-major StaticTensor sparse tensor addition assignment";
//
//       blaze::CompressedTensor<int> mat1( 2UL, 3UL, 4UL );
//       mat1(0,0) =  1;
//       mat1(0,1) =  2;
//       mat1(1,0) = -3;
//       mat1(1,2) =  4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor sparse tensor addition assignment";
//
//       blaze::CompressedTensor<int,blaze::columnMajor> mat1( 2UL, 3UL, 4UL );
//       mat1(0,0) =  1;
//       mat1(0,1) =  2;
//       mat1(1,0) = -3;
//       mat1(1,2) =  4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 += mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor sparse tensor addition assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor sparse tensor addition assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor sparse tensor addition assignment (upper)";
//
//       blaze::UpperTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor sparse tensor addition assignment (upper)";
//
//       blaze::UpperTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor sparse tensor addition assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor sparse tensor addition assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 += mat1;
//
//       if( mat1 != mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Addition assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the StaticTensor subtraction assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the subtraction assignment operators of the StaticTensor
// class template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testSubAssign()
{
//    //=====================================================================================
//    // Row-major dense tensor subtraction assignment
//    //=====================================================================================
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor subtraction assignment (mixed type)";
//
//       blaze::StaticTensor<short,2UL,3UL> mat1{ { -1, -2,  0 },
//                                                                {  3,  0, -4 } };
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor subtraction assignment (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::rowMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,rowMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1 = 0;
//       mat1(0,0) = -1;
//       mat1(0,1) = -2;
//       mat1(1,0) =  3;
//       mat1(1,2) = -4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor subtraction assignment (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::rowMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,rowMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1 = 0;
//       mat1(0,0) = -1;
//       mat1(0,1) = -2;
//       mat1(1,0) =  3;
//       mat1(1,2) = -4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor subtraction assignment (mixed type)";
//
//       blaze::StaticTensor<short,2UL,3UL,blaze::columnMajor> mat1{ { -1, -2,  0 },
//                                                                   {  3,  0, -4 } };
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor subtraction assignment (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::columnMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,columnMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 48UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1 = 0;
//       mat1(0,0) = -1;
//       mat1(0,1) = -2;
//       mat1(1,0) =  3;
//       mat1(1,2) = -4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor subtraction assignment (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::columnMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,columnMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1 = 0;
//       mat1(0,0) = -1;
//       mat1(0,1) = -2;
//       mat1(1,0) =  3;
//       mat1(1,2) = -4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor subtraction assignment (lower)";
//
//       blaze::LowerTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor subtraction assignment (lower)";
//
//       blaze::LowerTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor subtraction assignment (upper)";
//
//       blaze::UpperTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor subtraction assignment (upper)";
//
//       blaze::UpperTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor dense tensor subtraction assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor dense tensor subtraction assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Row-major sparse tensor subtraction assignment
//    //=====================================================================================
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor subtraction assignment";
//
//       blaze::CompressedTensor<int> mat1( 2UL, 3UL, 4UL );
//       mat1(0,0) = -1;
//       mat1(0,1) = -2;
//       mat1(1,0) =  3;
//       mat1(1,2) = -4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor subtraction assignment";
//
//       blaze::CompressedTensor<int,blaze::columnMajor> mat1( 2UL, 3UL, 4UL );
//       mat1(0,0) = -1;
//       mat1(0,1) = -2;
//       mat1(1,0) =  3;
//       mat1(1,2) = -4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL> mat2{ { 0, -2, 6 },
//                                                              { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor subtraction assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor subtraction assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor subtraction assignment (upper)";
//
//       blaze::UpperTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor subtraction assignment (upper)";
//
//       blaze::UpperTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/row-major StaticTensor sparse tensor subtraction assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Row-major/column-major StaticTensor sparse tensor subtraction assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Column-major dense tensor subtraction assignment
//    //=====================================================================================
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor subtraction assignment (mixed type)";
//
//       blaze::StaticTensor<short,2UL,3UL> mat1{ { -1, -2,  0 },
//                                                                {  3,  0, -4 } };
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor subtraction assignment (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::rowMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,rowMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 32UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1 = 0;
//       mat1(0,0) = -1;
//       mat1(0,1) = -2;
//       mat1(1,0) =  3;
//       mat1(1,2) = -4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor subtraction assignment (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::rowMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,rowMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1 = 0;
//       mat1(0,0) = -1;
//       mat1(0,1) = -2;
//       mat1(1,0) =  3;
//       mat1(1,2) = -4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor subtraction assignment (mixed type)";
//
//       blaze::StaticTensor<short,2UL,3UL,blaze::columnMajor> mat1{ { -1, -2,  0 },
//                                                                   {  3,  0, -4 } };
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor subtraction assignment (aligned/padded)";
//
//       using blaze::aligned;
//       using blaze::padded;
//       using blaze::columnMajor;
//
//       using AlignedPadded = blaze::CustomTensor<int,aligned,padded,columnMajor>;
//       std::unique_ptr<int[],blaze::Deallocate> memory( blaze::allocate<int>( 48UL ) );
//       AlignedPadded mat1( memory.get(), 2UL, 3UL, 16UL );
//       mat1 = 0;
//       mat1(0,0) = -1;
//       mat1(0,1) = -2;
//       mat1(1,0) =  3;
//       mat1(1,2) = -4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor subtraction assignment (unaligned/unpadded)";
//
//       using blaze::unaligned;
//       using blaze::unpadded;
//       using blaze::columnMajor;
//
//       using UnalignedUnpadded = blaze::CustomTensor<int,unaligned,unpadded,columnMajor>;
//       std::unique_ptr<int[]> memory( new int[7UL] );
//       UnalignedUnpadded mat1( memory.get()+1UL, 2UL, 3UL );
//       mat1 = 0;
//       mat1(0,0) = -1;
//       mat1(0,1) = -2;
//       mat1(1,0) =  3;
//       mat1(1,2) = -4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor subtraction assignment (lower)";
//
//       blaze::LowerTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor subtraction assignment (lower)";
//
//       blaze::LowerTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor subtraction assignment (upper)";
//
//       blaze::UpperTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor subtraction assignment (upper)";
//
//       blaze::UpperTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor dense tensor subtraction assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::StaticTensor<int,2UL,3UL,3UL> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor dense tensor subtraction assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> > mat1;
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Column-major sparse tensor subtraction assignment
//    //=====================================================================================
//
//    {
//       test_ = "Column-major/row-major StaticTensor sparse tensor subtraction assignment";
//
//       blaze::CompressedTensor<int> mat1( 2UL, 3UL, 4UL );
//       mat1(0,0) = -1;
//       mat1(0,1) = -2;
//       mat1(1,0) =  3;
//       mat1(1,2) = -4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor sparse tensor subtraction assignment";
//
//       blaze::CompressedTensor<int,blaze::columnMajor> mat1( 2UL, 3UL, 4UL );
//       mat1(0,0) = -1;
//       mat1(0,1) = -2;
//       mat1(1,0) =  3;
//       mat1(1,2) = -4;
//
//       blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor> mat2{ { 0, -2, 6 },
//                                                                 { 5,  0, 0 } };
//
//       mat2 -= mat1;
//
//       checkRows    ( mat2, 2UL );
//       checkColumns ( mat2, 3UL );
//       checkCapacity( mat2, 6UL );
//       checkNonZeros( mat2, 4UL );
//       checkNonZeros( mat2, 0UL, 2UL );
//       checkNonZeros( mat2, 1UL, 0UL );
//       checkNonZeros( mat2, 2UL, 2UL );
//
//       if( mat2(0,0) != 1 || mat2(0,1) != 0 || mat2(0,2) != 6 ||
//           mat2(1,0) != 2 || mat2(1,1) != 0 || mat2(1,2) != 4 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat2 << "\n"
//              << "   Expected result:\n( 1 0 6 )\n( 2 0 4 )\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor sparse tensor subtraction assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor sparse tensor subtraction assignment (lower)";
//
//       blaze::LowerTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor sparse tensor subtraction assignment (upper)";
//
//       blaze::UpperTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor sparse tensor subtraction assignment (upper)";
//
//       blaze::UpperTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/row-major StaticTensor sparse tensor subtraction assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::CompressedTensor<int> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       test_ = "Column-major/column-major StaticTensor sparse tensor subtraction assignment (diagonal)";
//
//       blaze::DiagonalTensor< blaze::CompressedTensor<int,blaze::columnMajor> > mat1( 3UL );
//       randomize( mat1 );
//
//       blaze::StaticTensor<int,2UL,3UL,3UL,blaze::columnMajor> mat2;
//
//       mat2 -= mat1;
//
//       if( mat1 != -mat2 ) {
//          std::ostringstream oss;
//          oss << " Test: " << test_ << "\n"
//              << " Error: Subtraction assignment failed\n"
//              << " Details:\n"
//              << "   Result:\n" << mat1 << "\n"
//              << "   Expected result:\n" << mat2 << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
}
//*************************************************************************************************

} // namespace statictensor

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
   std::cout << "   Running StaticTensor class test (part 1)..." << std::endl;

   try
   {
      RUN_STATICTENSOR_CLASS_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during StaticTensor class test (part 1):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
