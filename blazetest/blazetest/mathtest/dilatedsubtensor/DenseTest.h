//=================================================================================================
/*!
//  \file blazetest/mathtest/dilatedsubtensor/DenseTest.h
//  \brief Header file for the DilatedSubtensor dense aligned test
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

#ifndef _BLAZETEST_MATHTEST_DILATEDSUBTENSOR_DENSETEST_H_
#define _BLAZETEST_MATHTEST_DILATEDSUBTENSOR_DENSETEST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <sstream>
#include <stdexcept>
#include <string>
#include <blazetest/system/Types.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/RowMajorTensor.h>
#include <blaze_tensor/math/DilatedSubmatrix.h>
#include <blaze_tensor/math/DilatedSubtensor.h>
#include <blaze_tensor/math/DilatedSubvector.h>
#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/typetraits/IsRowMajorTensor.h>


namespace blazetest {

namespace mathtest {

namespace dilatedsubtensor {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class for all tests of the dense aligned DilatedSubtensor specialization.
//
// This class represents a test suite for the blaze::DilatedSubtensor class template specialization for
// dense aligned subtensors. It performs a series of both compile time as well as runtime tests.
*/
class DenseTest
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit DenseTest();
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
   void testConstructors();
   void testAssignment();
   void testAddAssign();
   void testSubAssign();
   void testSchurAssign();
   void testScaling();
   void testFunctionCall();
   void testIterator();
   void testNonZeros();
   void testReset();
   void testClear();
   void testTranspose();
   void testCTranspose();
   void testIsDefault();
   void testIsSame();
   void testDilatedSubtensor();
   void testPageslice();
   void testRowslice();
   void testColumnslice();


   template< typename Type >
   void checkPages( const Type& tensor, size_t expectedPages ) const;

   template< typename Type >
   void checkRows( const Type& tensor, size_t expectedRows ) const;

   template< typename Type >
   void checkColumns( const Type& tensor, size_t expectedColumns ) const;

   template< typename Type >
   void checkNonZeros( const Type& tensor, size_t expectedNonZeros ) const;

   template< typename Type >
   void checkNonZeros( const Type& tensor, size_t index, size_t expectedNonZeros ) const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   void initialize();
   std::vector<size_t> generate_indices( size_t i, size_t n, size_t dilation );
   //@}
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   using TT    = blaze::DynamicTensor<int>;                                 //!< Row-major dynamic tensor type
   using DSTT  = blaze::DilatedSubtensor<TT, true>;                         //!< Dense dilated subtensor type for row-major tensors.
   using DSPT  = blaze::DilatedSubmatrix<blaze::PageSlice<TT>,false,true>;  //!< Dense row-major dilated submatrix on a page of the original tensor

   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   TT  tens1_;   //!< First row-major dynamic tensor.
                /*!< The \f$ 64 \times 64 \f$ row-major dense tensor is randomly initialized. */
   TT  tens2_;   //!< Second row-major dynamic tensor.
                /*!< The \f$ 64 \times 64 \f$ row-major dense tensor is randomly initialized. */

   std::string test_;  //!< Label of the currently performed test.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( TT    );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( DSTT  );
   //BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RCTT  );
   /*! \endcond */
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
// This function checks the number of rows of the given dense tensor. In case the
// actual number of rows does not correspond to the given expected number of rows, a
// \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dense tensor
void DenseTest::checkPages( const Type& tensor, size_t expectedPages ) const
{
   if( pages( tensor ) != expectedPages ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of pages detected\n"
          << " Details:\n"
          << "   Number of pages         : " << pages( tensor ) << "\n"
          << "   Expected number of pages: " << expectedPages << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of rows of the given dense tensor.
//
// \param tensor The dense tensor to be checked.
// \param expectedRows The expected number of rows of the dense tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of rows of the given dense tensor. In case the
// actual number of rows does not correspond to the given expected number of rows, a
// \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dense tensor
void DenseTest::checkRows( const Type& tensor, size_t expectedRows ) const
{
   if( rows( tensor ) != expectedRows ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of rows detected\n"
          << " Details:\n"
          << "   Number of rows         : " << rows( tensor ) << "\n"
          << "   Expected number of rows: " << expectedRows << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of columns of the given dense tensor.
//
// \param tensor The dense tensor to be checked.
// \param expectedColumns The expected number of columns of the dense tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of columns of the given dense tensor. In case the
// actual number of columns does not correspond to the given expected number of columns,
// a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dense tensor
void DenseTest::checkColumns( const Type& tensor, size_t expectedColumns ) const
{
   if( columns( tensor ) != expectedColumns ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of columns detected\n"
          << " Details:\n"
          << "   Number of columns         : " << columns( tensor ) << "\n"
          << "   Expected number of columns: " << expectedColumns << "\n";
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
void DenseTest::checkNonZeros( const Type& tensor, size_t expectedNonZeros ) const
{
   if( nonZeros( tensor ) != expectedNonZeros ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of non-zero elements\n"
          << " Details:\n"
          << "   Number of non-zeros         : " << nonZeros( tensor ) << "\n"
          << "   Expected number of non-zeros: " << expectedNonZeros << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( capacity( tensor ) < nonZeros( tensor ) ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected\n"
          << " Details:\n"
          << "   Number of non-zeros: " << nonZeros( tensor ) << "\n"
          << "   Capacity           : " << capacity( tensor ) << "\n";
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
void DenseTest::checkNonZeros( const Type& tensor, size_t index, size_t expectedNonZeros ) const
{
   if( nonZeros( tensor, index ) != expectedNonZeros ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of non-zero elements in "
          << ( blaze::IsRowMajorTensor<Type>::value ? "row " : "non row " ) << index << "\n"
          << " Details:\n"
          << "   Number of non-zeros         : " << nonZeros( tensor, index ) << "\n"
          << "   Expected number of non-zeros: " << expectedNonZeros << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( capacity( tensor, index ) < nonZeros( tensor, index ) ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected in "
          << ( blaze::IsRowMajorTensor<Type>::value ? "row " : "non row " ) << index << "\n"
          << " Details:\n"
          << "   Number of non-zeros: " << nonZeros( tensor, index ) << "\n"
          << "   Capacity           : " << capacity( tensor, index ) << "\n";
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
/*!\brief Testing the functionality of the dense aligned DilatedSubtensor specialization.
//
// \return void
*/
void runTest()
{
   DenseTest();
}
//*************************************************************************************************




//=================================================================================================
//
//  MACRO DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of the DilatedSubtensor dense aligned test.
*/
#define RUN_DILATEDSUBTENSOR_DENSE_TEST \
   blazetest::mathtest::dilatedsubtensor::runTest()
/*! \endcond */
//*************************************************************************************************

} // namespace dilatedsubtensor

} // namespace mathtest

} // namespace blazetest

#endif
