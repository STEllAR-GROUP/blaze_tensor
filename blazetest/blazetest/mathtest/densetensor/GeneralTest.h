//=================================================================================================
/*!
//  \file blazetest/mathtest/densetensor/GeneralTest.h
//  \brief Header file for the general DenseMatrix operation test
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

#ifndef _BLAZETEST_MATHTEST_DENSETENSOR_GENERALTEST_H_
#define _BLAZETEST_MATHTEST_DENSETENSOR_GENERALTEST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <sstream>
#include <stdexcept>
#include <string>


namespace blazetest {

namespace mathtest {

namespace densetensor {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class for tests of the DenseMatrix functionality.
//
// This class represents a test suite for the DenseMatrix functionality contained in the
// <em><blaze/math/dense/DenseMatrix.h></em> header file. It performs a series of runtime
// tests with general matrices.
*/
class GeneralTest
{
 private:
   //**Type definitions****************************************************************************
   using cplx = blaze::complex<int>;  //!< Complex element type.
   //**********************************************************************************************

 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit GeneralTest();
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

 private:
   //**Test functions******************************************************************************
   /*!\name Test functions */
   //@{
   void testIsNan();
//    void testIsSquare();
//    void testIsSymmetric();
//    void testIsHermitian();
   void testIsUniform();
//    void testIsLower();
//    void testIsUniLower();
//    void testIsStrictlyLower();
//    void testIsUpper();
//    void testIsUniUpper();
//    void testIsStrictlyUpper();
//    void testIsDiagonal();
//    void testIsIdentity();
   void testMinimum();
   void testMaximum();
   void testSoftmax();
//    void testTrace();
   void testL1Norm();
   void testL2Norm();
   void testL3Norm();
   void testL4Norm();
   void testLpNorm();

   template< typename Type >
   void checkRows( const Type& tensor, size_t expectedRows ) const;

   template< typename Type >
   void checkColumns( const Type& tensor, size_t expectedColumns ) const;

   template< typename Type >
   void checkPages( const Type& tensor, size_t expectedPages ) const;

   template< typename Type >
   void checkCapacity( const Type& tensor, size_t minCapacity ) const;

   template< typename Type >
   void checkNonZeros( const Type& tensor, size_t expectedNonZeros ) const;

   template< typename Type >
   void checkNonZeros( const Type& tensor, size_t i, size_t k, size_t expectedNonZeros ) const;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::string test_;  //!< Label of the currently performed test.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Checking the number of rows of the given dense tensor.
//
// \param tensor The dense tensor to be checked.
// \param expectedRows The expected number of rows of the dense tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of rows of the given dense tensor. In case the actual number
// of rows does not correspond to the given expected number of rows, a \a std::runtime_error
// exception is thrown.
*/
template< typename Type >  // Type of the dense tensor
void GeneralTest::checkRows( const Type& tensor, size_t expectedRows ) const
{
   if( tensor.rows() != expectedRows ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of rows detected\n"
          << " Details:\n"
          << "   Number of rows         : " << tensor.rows() << "\n"
          << "   Expected number of rows: " << expectedRows << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of columns of the given dense tensor.
//
// \param tensor The dense tensor to be checked.
// \param expectedRows The expected number of columns of the dense tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of columns of the given dense tensor. In case the
// actual number of columns does not correspond to the given expected number of columns,
// a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dense tensor
void GeneralTest::checkColumns( const Type& tensor, size_t expectedColumns ) const
{
   if( tensor.columns() != expectedColumns ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of columns detected\n"
          << " Details:\n"
          << "   Number of columns         : " << tensor.columns() << "\n"
          << "   Expected number of columns: " << expectedColumns << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of pages of the given dense tensor.
//
// \param tensor The dense tensor to be checked.
// \param expectedRows The expected number of pages of the dense tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of pages of the given dense tensor. In case the
// actual number of pages does not correspond to the given expected number of pages,
// a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dense tensor
void GeneralTest::checkPages( const Type& tensor, size_t expectedPages ) const
{
   if( tensor.pages() != expectedPages ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of pages detected\n"
          << " Details:\n"
          << "   Number of pages         : " << tensor.pages() << "\n"
          << "   Expected number of pages: " << expectedPages << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the capacity of the given dense tensor.
//
// \param tensor The dense tensor to be checked.
// \param minCapacity The expected minimum capacity of the dense tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the capacity of the given dense tensor. In case the actual capacity
// is smaller than the given expected minimum capacity, a \a std::runtime_error exception is
// thrown.
*/
template< typename Type >  // Type of the dense tensor
void GeneralTest::checkCapacity( const Type& tensor, size_t minCapacity ) const
{
   if( tensor.capacity() < minCapacity ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected\n"
          << " Details:\n"
          << "   Capacity                 : " << tensor.capacity() << "\n"
          << "   Expected minimum capacity: " << minCapacity << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of non-zero elements of the given dense tensor.
//
// \param tensor The dense tensor to be checked.
// \param expectedNonZeros The expected number of non-zero elements of the dense tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of non-zero elements of the given dense tensor. In
// case the actual number of non-zero elements does not correspond to the given expected
// number, a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dense tensor
void GeneralTest::checkNonZeros( const Type& tensor, size_t expectedNonZeros ) const
{
   if( tensor.nonZeros() != expectedNonZeros ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of non-zero elements\n"
          << " Details:\n"
          << "   Number of non-zeros         : " << tensor.nonZeros() << "\n"
          << "   Expected number of non-zeros: " << expectedNonZeros << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( tensor.capacity() < tensor.nonZeros() ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected\n"
          << " Details:\n"
          << "   Number of non-zeros: " << tensor.nonZeros() << "\n"
          << "   Capacity           : " << tensor.capacity() << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of non-zero elements in a specific row/column of the given dense tensor.
//
// \param tensor The dense tensor to be checked.
// \param index The row/column to be checked.
// \param expectedNonZeros The expected number of non-zero elements in the specified row/column.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of non-zero elements in the specified row/column of the
// given dense tensor. In case the actual number of non-zero elements does not correspond
// to the given expected number, a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dense tensor
void GeneralTest::checkNonZeros( const Type& tensor, size_t i, size_t k, size_t expectedNonZeros ) const
{
   if( tensor.nonZeros( i, k ) != expectedNonZeros ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of non-zero elements in row " << i << " page " << k << "\n"
          << " Details:\n"
          << "   Number of non-zeros         : " << tensor.nonZeros( i, k ) << "\n"
          << "   Expected number of non-zeros: " << expectedNonZeros << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( tensor.capacity( i, k ) < tensor.nonZeros( i, k ) ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected in row " << i << " page " << k << "\n"
          << " Details:\n"
          << "   Number of non-zeros: " << nonZeros( tensor, i, k ) << "\n"
          << "   Capacity           : " << capacity( tensor, i, k ) << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Testing the functionality of the DenseMatrix class template.
//
// \return void
*/
void runTest()
{
   GeneralTest();
}
//*************************************************************************************************




//=================================================================================================
//
//  MACRO DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of the general DenseMatrix operation test.
*/
#define RUN_DENSETENSOR_GENERAL_TEST \
   blazetest::mathtest::densetensor::runTest()
/*! \endcond */
//*************************************************************************************************

} // namespace densetensor

} // namespace mathtest

} // namespace blazetest

#endif
