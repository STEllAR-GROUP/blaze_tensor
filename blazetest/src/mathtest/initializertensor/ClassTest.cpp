//=================================================================================================
/*!
//  \file src/mathtest/initializertensor/ClassTest.cpp
//  \brief Source file for the InitializerTensor class test
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
#include <blazetest/mathtest/initializertensor/ClassTest.h>


namespace blazetest {

namespace mathtest {

namespace initializertensor {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the InitializerTensor class test.
//
// \exception std::runtime_error Operation error detected.
*/
ClassTest::ClassTest()
{
   testConstructors();
   testFunctionCall();
   testAt();
   testIterator();
   testNonZeros();
   testSwap();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the InitializerTensor constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the InitializerTensor class template.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testConstructors()
{
   using blaze::initializer_list;


   //=====================================================================================
   // Single argument constructor
   //=====================================================================================

   {
      test_ = "InitializerTensor single argument constructor (0x0x0)";

      initializer_list< initializer_list< initializer_list<int> > > list = {};

      blaze::InitializerTensor<int> mat( list );

      checkRows    ( mat, 0UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 0UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "InitializerTensor single argument constructor (2x3x4)";

      initializer_list< initializer_list< initializer_list< int > > > list = {
          {{1, 0, 3, 4}, {0}, {2, 0, 5}}, {{1, 0, 3, 4}, {0}, {2, 0, 5}}};

      blaze::InitializerTensor<int> mat( list );

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  4UL );
      checkPages   ( mat,  2UL );
      checkNonZeros( mat, 10UL );
   }


   //=====================================================================================
   // Two argument constructor
   //=====================================================================================

   {
      test_ = "InitializerTensor two argument constructor (2x3x0)";

      initializer_list< initializer_list< initializer_list< int > > > list = {
         {{}, {}, {}}, {{}, {}, {}}};

      blaze::InitializerTensor<int> mat( list, 3UL, 0UL );

      checkRows    ( mat, 3UL );
      checkColumns ( mat, 0UL );
      checkPages   ( mat, 2UL );
      checkNonZeros( mat, 0UL );
   }

   {
      test_ = "InitializerTensor two argument constructor (2x3x4)";

      initializer_list< initializer_list< initializer_list< int > > > list = {
          {{1, 0, 3, 4}, {0}, {2, 0, 5}}, {{1, 0, 3, 4}, {0}, {2, 0, 5}}};

      blaze::InitializerTensor<int> mat( list, 3UL, 4UL );

      checkRows    ( mat,  3UL );
      checkColumns ( mat,  4UL );
      checkPages   ( mat,  2UL );
      checkNonZeros( mat, 10UL );
   }

   {
      test_ = "InitializerTensor two argument constructor (2x4x6)";

      initializer_list< initializer_list< initializer_list< int > > > list = {
          {{1, 0, 3, 4}, {0}, {2, 0, 5}}, {{1, 0, 3, 4}, {0}, {2, 0, 5}}};

      blaze::InitializerTensor<int> mat( list, 4UL, 6UL );

      checkRows    ( mat,  4UL );
      checkColumns ( mat,  6UL );
      checkPages   ( mat,  2UL );
      checkNonZeros( mat, 10UL );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the InitializerTensor function call operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the function call operator
// of the InitializerTensor class template. In case an error is detected, a \a std::runtime_error
// exception is thrown.
*/
void ClassTest::testFunctionCall()
{
   using blaze::initializer_list;


   test_ = "InitializerTensor::operator()";

   initializer_list< initializer_list< initializer_list< int > > > list = {
         {{1, 0, 3, 4}, {0}, {2, 0, 5}}, {{1, 0, 3, 4}, {0}, {2, 0, 5}}};

   blaze::InitializerTensor<int> mat( list, 3UL, 6UL );

   // Access to the element (0,0,2)
   if( mat(0,0,2) != 3 ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Function call operator failed\n"
          << " Details:\n"
          << "   Result:\n" << mat << "\n"
          << "   Expected result:\n"
                     "(( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 )\n"
                     " ( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 ))\n";
      throw std::runtime_error( oss.str() );
   }

   // Access to the element (1,1,2)
   if( mat(1,1,2) != 0 ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Function call operator failed\n"
          << " Details:\n"
          << "   Result:\n" << mat << "\n"
          << "   Expected result:\n"
                     "(( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 )\n"
                     " ( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 ))\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c at() member function of the InitializerTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the \c at() member function
// of the InitializerTensor class template. In case an error is detected, a \a std::runtime_error
// exception is thrown.
*/
void ClassTest::testAt()
{
   using blaze::initializer_list;


   test_ = "InitializerTensor::operator()";

   initializer_list< initializer_list< initializer_list< int > > > list = {
         {{1, 0, 3, 4}, {0}, {2, 0, 5}}, {{1, 0, 3, 4}, {0}, {2, 0, 5}}};

   blaze::InitializerTensor<int> mat( list, 3UL, 6UL );

   // Access to the element (0,0,2)
   if( mat.at(0,0,2) != 3 ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Function call operator failed\n"
          << " Details:\n"
          << "   Result:\n" << mat << "\n"
          << "   Expected result:\n"
                     "(( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 )\n"
                     " ( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 ))\n";
      throw std::runtime_error( oss.str() );
   }

   // Access to the element (1,1,2)
   if( mat.at(1,1,2) != 0 ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Function call operator failed\n"
          << " Details:\n"
          << "   Result:\n" << mat << "\n"
          << "   Expected result:\n"
                     "(( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 )\n"
                     " ( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 ))\n";
      throw std::runtime_error( oss.str() );
   }

   // Attempt to access the element (1,3,0)
   try {
      mat.at(1,3,0);

      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Out-of-bound access succeeded\n"
          << " Details:\n"
          << "   Result:\n" << mat << "\n"
          << "   Expected result:\n"
                     "(( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 )\n"
                     " ( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 ))\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::out_of_range& ) {}

   // Attempt to access the element (1,2,6)
   try {
      mat.at(1,2,6);

      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Out-of-bound access succeeded\n"
          << " Details:\n"
          << "   Result:\n" << mat << "\n"
          << "   Expected result:\n"
                     "(( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 )\n"
                     " ( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 ))\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::out_of_range& ) {}

   // Attempt to access the element (3,2,0)
   try {
      mat.at(3,2,0);

      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Out-of-bound access succeeded\n"
          << " Details:\n"
          << "   Result:\n" << mat << "\n"
          << "   Expected result:\n"
                     "(( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 )\n"
                     " ( 1 0 3 4 0 0 )\n( 0 0 0 0 0 0 )\n( 2 0 5 0 0 0 ))\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::out_of_range& ) {}
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the InitializerTensor iterator implementation.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the iterator implementation of the InitializerTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testIterator()
{
   using blaze::initializer_list;


   using TensorType    = blaze::InitializerTensor<int>;
   using Iterator      = TensorType::Iterator;
   using ConstIterator = TensorType::ConstIterator;

   initializer_list< initializer_list< initializer_list< int > > > list = {
         {{1, 0, 3, 4}, {0}, {2, 0, 5}}, {{1, 0, 3, 4}, {0}, {2, 0, 5}}};

   TensorType mat( list, 3UL, 6UL );

   // Testing the Iterator default constructor
   {
      test_ = "Iterator default constructor";

      Iterator it{};

      if( it != Iterator() ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed iterator default constructor\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Testing the ConstIterator default constructor
   {
      test_ = "ConstIterator default constructor";

      ConstIterator it{};

      if( it != ConstIterator() ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed iterator default constructor\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Testing conversion from Iterator to ConstIterator
   {
      test_ = "Iterator/ConstIterator conversion";

      ConstIterator it( begin( mat, 0UL, 1UL ) );

      if( it == end( mat, 0UL, 1UL ) || *it != 1 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed iterator conversion detected\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Counting the number of elements in 0th row via Iterator (end-begin)
   {
      test_ = "Iterator subtraction (end-begin)";

      const ptrdiff_t number( end( mat, 0UL, 1UL ) - begin( mat, 0UL, 1UL ) );

      if( number != 6L ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid number of elements detected\n"
             << " Details:\n"
             << "   Number of elements         : " << number << "\n"
             << "   Expected number of elements: 6\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Counting the number of elements in 0th row via Iterator (begin-end)
   {
      test_ = "Iterator subtraction (begin-end)";

      const ptrdiff_t number( begin( mat, 0UL, 1UL ) - end( mat, 0UL, 1UL ) );

      if( number != -6L ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid number of elements detected\n"
             << " Details:\n"
             << "   Number of elements         : " << number << "\n"
             << "   Expected number of elements: -6\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Counting the number of elements in 1st row via ConstIterator (end-begin)
   {
      test_ = "ConstIterator subtraction (end-begin)";

      const ptrdiff_t number( cend( mat, 1UL, 1UL ) - cbegin( mat, 1UL, 1UL ) );

      if( number != 6L ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid number of elements detected\n"
             << " Details:\n"
             << "   Number of elements         : " << number << "\n"
             << "   Expected number of elements: 6\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Counting the number of elements in 1st row via ConstIterator (begin-end)
   {
      test_ = "ConstIterator subtraction (begin-end)";

      const ptrdiff_t number( cbegin( mat, 1UL, 1UL ) - cend( mat, 1UL, 1UL ) );

      if( number != -6L ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid number of elements detected\n"
             << " Details:\n"
             << "   Number of elements         : " << number << "\n"
             << "   Expected number of elements: -6\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Testing read-only access via ConstIterator
   {
      test_ = "Read-only access via ConstIterator";

      ConstIterator it ( cbegin( mat, 2UL, 1UL ) );
      ConstIterator end( cend( mat, 2UL, 1UL ) );

      if( it == end || *it != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Invalid initial iterator detected\n";
         throw std::runtime_error( oss.str() );
      }

      ++it;

      if( it == end || *it != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator pre-increment failed\n";
         throw std::runtime_error( oss.str() );
      }

      --it;

      if( it == end || *it != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator pre-decrement failed\n";
         throw std::runtime_error( oss.str() );
      }

      it++;

      if( it == end || *it != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator post-increment failed\n";
         throw std::runtime_error( oss.str() );
      }

      it--;

      if( it == end || *it != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator post-decrement failed\n";
         throw std::runtime_error( oss.str() );
      }

      it += 2UL;

      if( it == end || *it != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator addition assignment failed\n";
         throw std::runtime_error( oss.str() );
      }

      it -= 2UL;

      if( it == end || *it != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator subtraction assignment failed\n";
         throw std::runtime_error( oss.str() );
      }

      it = it + 2UL;

      if( it == end || *it != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator/scalar addition failed\n";
         throw std::runtime_error( oss.str() );
      }

      it = it - 2UL;

      if( it == end || *it != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Iterator/scalar subtraction failed\n";
         throw std::runtime_error( oss.str() );
      }

      it = 6UL + it;

      if( it != end ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Scalar/iterator addition failed\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c nonZeros() member function of the InitializerTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c nonZeros() member function of the InitializerTensor class
// template. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testNonZeros()
{
   using blaze::initializer_list;


   test_ = "InitializerTensor::nonZeros()";

   {
      initializer_list< initializer_list< initializer_list< int > > > list = {
            {{0, 0, 0}, {0}}, {{0, 0, 0}, {0}}};

      blaze::InitializerTensor<int> mat( list );

      checkRows    ( mat,  2UL );
      checkColumns ( mat,  3UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 12UL );
      checkNonZeros( mat,  0UL );
      checkNonZeros( mat,  0UL, 0UL, 0UL );
      checkNonZeros( mat,  1UL, 0UL, 0UL );
      checkNonZeros( mat,  0UL, 1UL, 0UL );
      checkNonZeros( mat,  1UL, 1UL, 0UL );

      if( mat(0,0,0) != 0 || mat(0,0,1) != 0 || mat(0,0,2) != 0 ||
          mat(0,1,0) != 0 || mat(0,1,1) != 0 || mat(0,1,2) != 0 ||
          mat(1,0,0) != 0 || mat(1,0,1) != 0 || mat(1,0,2) != 0 ||
          mat(1,1,0) != 0 || mat(1,1,1) != 0 || mat(1,1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n"
                        "(( 0 0 0 )\n( 0 0 0 )\n"
                        " ( 0 0 0 )\n( 0 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      initializer_list< initializer_list< initializer_list< int > > > list = {
            {{0, 1, 2}, {0, 3, 0}}, {{0, 1, 2}, {0, 3, 0}}};

      blaze::InitializerTensor<int> mat( list );

      checkRows    ( mat,  2UL );
      checkColumns ( mat,  3UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 12UL );
      checkNonZeros( mat,  6UL );
      checkNonZeros( mat,  0UL, 0UL, 2UL );
      checkNonZeros( mat,  1UL, 0UL, 1UL );
      checkNonZeros( mat,  0UL, 1UL, 2UL );
      checkNonZeros( mat,  1UL, 1UL, 1UL );

      if( mat(0,0,0) != 0 || mat(0,0,1) != 1 || mat(0,0,2) != 2 ||
          mat(0,1,0) != 0 || mat(0,1,1) != 3 || mat(0,1,2) != 0 ||
          mat(1,0,0) != 0 || mat(1,0,1) != 1 || mat(1,0,2) != 2 ||
          mat(1,1,0) != 0 || mat(1,1,1) != 3 || mat(1,1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n"
                        "(( 0 1 2 )\n( 0 3 0 )\n"
                        " ( 0 1 2 )\n( 0 3 0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      initializer_list< initializer_list< initializer_list< int > > > list = {
            {{0, 1, 2}, {0, 3, 0}}, {{0, 1, 2}, {0, 3, 0}}};

      blaze::InitializerTensor<int> mat( list, 2UL, 4UL );

      checkRows    ( mat,  2UL );
      checkColumns ( mat,  4UL );
      checkPages   ( mat,  2UL );
      checkCapacity( mat, 16UL );
      checkNonZeros( mat,  6UL );
      checkNonZeros( mat,  0UL, 0UL, 2UL );
      checkNonZeros( mat,  1UL, 0UL, 1UL );
      checkNonZeros( mat,  0UL, 1UL, 2UL );
      checkNonZeros( mat,  1UL, 1UL, 1UL );

      if( mat(0,0,0) != 0 || mat(0,0,1) != 1 || mat(0,0,2) != 2 || mat(0,0,3) != 0 ||
          mat(0,1,0) != 0 || mat(0,1,1) != 3 || mat(0,1,2) != 0 || mat(0,1,3) != 0 ||
          mat(1,0,0) != 0 || mat(1,0,1) != 1 || mat(1,0,2) != 2 || mat(1,0,3) != 0 ||
          mat(1,1,0) != 0 || mat(1,1,1) != 3 || mat(1,1,2) != 0 || mat(1,1,3) != 0) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << mat << "\n"
             << "   Expected result:\n"
                        "(( 0 1 2 0 )\n( 0 3 0 0 )\n"
                        " ( 0 1 2 0 )\n( 0 3 0 0 ))\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c swap() functionality of the InitializerTensor class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c swap() function of the InitializerTensor class template.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void ClassTest::testSwap()
{
   using blaze::initializer_list;


   test_ = "InitializerTensor swap";

   initializer_list< initializer_list< initializer_list< int > > > list1{
       {{1, 2}, {0, 3}, {4}}, {{1, 2}, {0, 3}, {4}}};
   initializer_list< initializer_list< initializer_list< int > > > list2{
       {{6, 5, 4}, {3, 2, 1}}, {{6, 5, 4}, {3, 2, 1}}};

   blaze::InitializerTensor<int> mat1( list1 );
   blaze::InitializerTensor<int> mat2( list2, 3UL, 4UL );

   swap( mat1, mat2 );

   checkRows    ( mat1,  3UL );
   checkColumns ( mat1,  4UL );
   checkPages   ( mat1,  2UL );
   checkCapacity( mat1, 24UL );
   checkNonZeros( mat1, 12UL );
   checkNonZeros( mat1,  0UL, 0UL, 3UL );
   checkNonZeros( mat1,  1UL, 0UL, 3UL );
   checkNonZeros( mat1,  2UL, 0UL, 0UL );
   checkNonZeros( mat1,  0UL, 1UL, 3UL );
   checkNonZeros( mat1,  1UL, 1UL, 3UL );
   checkNonZeros( mat1,  2UL, 1UL, 0UL );

   if( mat1(0,0,0) != 6 || mat1(0,0,1) != 5 || mat1(0,0,2) != 4 || mat1(0,0,3) != 0 ||
       mat1(0,1,0) != 3 || mat1(0,1,1) != 2 || mat1(0,1,2) != 1 || mat1(0,1,3) != 0 ||
       mat1(0,2,0) != 0 || mat1(0,2,1) != 0 || mat1(0,2,2) != 0 || mat1(0,2,3) != 0 ||
       mat1(1,0,0) != 6 || mat1(1,0,1) != 5 || mat1(1,0,2) != 4 || mat1(1,0,3) != 0 ||
       mat1(1,1,0) != 3 || mat1(1,1,1) != 2 || mat1(1,1,2) != 1 || mat1(1,1,3) != 0 ||
       mat1(1,2,0) != 0 || mat1(1,2,1) != 0 || mat1(1,2,2) != 0 || mat1(1,2,3) != 0) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Swapping the first tensor failed\n"
          << " Details:\n"
          << "   Result:\n" << mat1 << "\n"
             << "   Expected result:\n"
                        "(( 6 5 4 0 )\n( 3 2 1 0 )\n"
                        " ( 6 5 4 0 )\n( 3 2 1 0 )\n"
                        " ( 0 0 0 0 )\n( 0 0 0 0 ))\n";
      throw std::runtime_error( oss.str() );
   }

   checkRows    ( mat2,  3UL );
   checkColumns ( mat2,  2UL );
   checkPages   ( mat1,  2UL );
   checkCapacity( mat2, 12UL );
   checkNonZeros( mat2,  8UL );
   checkNonZeros( mat2,  0UL, 0UL, 2UL );
   checkNonZeros( mat2,  1UL, 0UL, 1UL );
   checkNonZeros( mat2,  2UL, 0UL, 1UL );
   checkNonZeros( mat2,  0UL, 1UL, 2UL );
   checkNonZeros( mat2,  1UL, 1UL, 1UL );
   checkNonZeros( mat2,  2UL, 1UL, 1UL );

   if( mat2(0,0,0) != 1 || mat2(0,0,1) != 2 ||
       mat2(0,1,0) != 0 || mat2(0,1,1) != 3 ||
       mat2(0,2,0) != 4 || mat2(0,2,1) != 0 ||
       mat2(1,0,0) != 1 || mat2(1,0,1) != 2 ||
       mat2(1,1,0) != 0 || mat2(1,1,1) != 3 ||
       mat2(1,2,0) != 4 || mat2(1,2,1) != 0) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Swapping the second tensor failed\n"
          << " Details:\n"
          << "   Result:\n" << mat2 << "\n"
             << "   Expected result:\n"
                        "(( 1 2 )\n( 0 3 )\n( 4 0 )\n"
                        " ( 1 2 )\n( 0 3 )\n( 4 0 ))\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************

} // namespace initializertensor

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
   std::cout << "   Running InitializerTensor class test..." << std::endl;

   try
   {
      RUN_INITIALIZERTENSOR_CLASS_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during InitializerTensor class test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
