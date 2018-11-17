//=================================================================================================
/*!
//  \file blazetest/mathtest/pageslice/DenseGeneralTest.h
//  \brief Header file for the PageSlice dense general test
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

#ifndef _BLAZETEST_MATHTEST_PAGESLICE_DENSEGENERALTEST_H_
#define _BLAZETEST_MATHTEST_PAGESLICE_DENSEGENERALTEST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <sstream>
#include <stdexcept>
#include <string>
#include <blazetest/system/Types.h>

#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/PageSlice.h>
#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/PageSliceMatrix.h>
#include <blaze_tensor/math/typetraits/IsPageSliceMatrix.h>

namespace blazetest {

namespace mathtest {

namespace pageslice {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class for all tests of the dense general PageSlice specialization.
//
// This class represents a test suite for the blaze::PageSlice class template specialization for
// dense general matrices. It performs a series of both compile time as well as runtime tests.
*/
class DenseGeneralTest
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit DenseGeneralTest();
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
   void testMultAssign();
   void testSchurAssign();
   void testScaling();
   void testFunctionCall();
   void testAt();
   void testIterator();
   void testNonZeros();
   void testReset();
   void testClear();
   void testIsDefault();
   void testIsSame();
   void testSubmatrix();
   void testRow();
   void testRows();
   void testColumn();
   void testColumns();
   void testBand();

   template< typename Type >
   void checkSize( const Type& pageslice, size_t expectedSize ) const;

   template< typename Type >
   void checkRows( const Type& tensor, size_t expectedRows ) const;

   template< typename Type >
   void checkColumns( const Type& tensor, size_t expectedColumns ) const;

   template< typename Type >
   void checkPages( const Type& tensor, size_t expectedPages ) const;

   template< typename Type >
   void checkCapacity( const Type& object, size_t minCapacity ) const;

   template< typename Type >
   void checkNonZeros( const Type& object, size_t expectedNonZeros ) const;

   template< typename Type >
   void checkNonZeros( const Type& tensor, size_t index, size_t expectedNonZeros ) const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   void initialize();
   //@}
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   using MT  = blaze::DynamicTensor<int>;    //!< Dynamic tensor type.
   using RT  = blaze::PageSlice<MT>;         //!< Dense pageslice type for tensors.
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   MT  mat_;   //!< dynamic tensor.
   std::string test_;  //!< Label of the currently performed test.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE    ( MT  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE    ( RT  );
   BLAZE_CONSTRAINT_MUST_BE_PAGESLICE_MATRIX_TYPE( RT  );
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
/*!\brief Checking the size of the given dense pageslice.
//
// \param pageslice The dense pageslice to be checked.
// \param expectedSize The expected size of the dense pageslice.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the size of the given dense pageslice. In case the actual size does not
// correspond to the given expected size, a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dense pageslice
void DenseGeneralTest::checkSize( const Type& pageslice, size_t expectedSize ) const
{
   if( size( pageslice ) != expectedSize ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid size detected\n"
          << " Details:\n"
          << "   Size         : " << size( pageslice ) << "\n"
          << "   Expected size: " << expectedSize << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of pageslices of the given dynamic tensor.
//
// \param tensor The dynamic tensor to be checked.
// \param expectedPageSlices The expected number of rows of the dynamic tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of rows of the given dynamic tensor. In case the actual number
// of pageslices does not correspond to the given expected number of pageslices, a \a std::runtime_error
// exception is thrown.
*/
template< typename Type >  // Type of the dynamic tensor
void DenseGeneralTest::checkRows( const Type& tensor, size_t expectedRows ) const
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
/*!\brief Checking the number of columns of the given dynamic tensor.
//
// \param tensor The dynamic tensor to be checked.
// \param expectedColumns The expected number of columns of the dynamic tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of columns of the given dynamic tensor. In case the
// actual number of columns does not correspond to the given expected number of columns,
// a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dynamic tensor
void DenseGeneralTest::checkColumns( const Type& tensor, size_t expectedColumns ) const
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
/*!\brief Checking the number of pages of the given dynamic tensor.
//
// \param tensor The dynamic tensor to be checked.
// \param expectedPages The expected number of columns of the dynamic tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of pages of the given dynamic tensor. In case the
// actual number of pages does not correspond to the given expected number of pages,
// a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dynamic tensor
void DenseGeneralTest::checkPages( const Type& tensor, size_t expectedPages ) const
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
/*!\brief Checking the capacity of the given dense pageslice or dynamic tensor.
//
// \param object The dense pageslice or dynamic tensor to be checked.
// \param minCapacity The expected minimum capacity.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the capacity of the given dense pageslice or dynamic tensor. In case the actual
// capacity is smaller than the given expected minimum capacity, a \a std::runtime_error exception
// is thrown.
*/
template< typename Type >  // Type of the dense pageslice or dynamic tensor
void DenseGeneralTest::checkCapacity( const Type& object, size_t minCapacity ) const
{
   if( capacity( object ) < minCapacity ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected\n"
          << " Details:\n"
          << "   Capacity                 : " << capacity( object ) << "\n"
          << "   Expected minimum capacity: " << minCapacity << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of non-zero elements of the given dense pageslice or dynamic tensor.
//
// \param object The dense pageslice or dynamic tensor to be checked.
// \param expectedNonZeros The expected number of non-zero elements.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of non-zero elements of the given dense pageslice. In case the
// actual number of non-zero elements does not correspond to the given expected number, a
// \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dense pageslice or dynamic tensor
void DenseGeneralTest::checkNonZeros( const Type& object, size_t expectedNonZeros ) const
{
   if( nonZeros( object ) != expectedNonZeros ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of non-zero elements\n"
          << " Details:\n"
          << "   Number of non-zeros         : " << nonZeros( object ) << "\n"
          << "   Expected number of non-zeros: " << expectedNonZeros << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( capacity( object ) < nonZeros( object ) ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected\n"
          << " Details:\n"
          << "   Number of non-zeros: " << nonZeros( object ) << "\n"
          << "   Capacity           : " << capacity( object ) << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of non-zero elements in a specific pageslice/column of the given dynamic tensor.
//
// \param tensor The dynamic tensor to be checked.
// \param index The pageslice/column to be checked.
// \param expectedNonZeros The expected number of non-zero elements in the specified pageslice/column.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of non-zero elements in the specified pageslice/column of the
// given dynamic tensor. In case the actual number of non-zero elements does not correspond
// to the given expected number, a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dynamic tensor
void DenseGeneralTest::checkNonZeros( const Type& tensor, size_t index, size_t expectedNonZeros ) const
{
   if( nonZeros( tensor, index ) != expectedNonZeros ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of non-zero elements in "
          << ( blaze::IsPageSliceMajorMatrix<Type>::value ? "pageslice " : "column " ) << index << "\n"
          << " Details:\n"
          << "   Number of non-zeros         : " << nonZeros( tensor, index ) << "\n"
          << "   Expected number of non-zeros: " << expectedNonZeros << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( capacity( tensor, index ) < nonZeros( tensor, index ) ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected in "
          << ( blaze::IsPageSliceMajorMatrix<Type>::value ? "pageslice " : "column " ) << index << "\n"
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
/*!\brief Testing the functionality of the dense general PageSlice specialization.
//
// \return void
*/
void runTest()
{
   DenseGeneralTest();
}
//*************************************************************************************************




//=================================================================================================
//
//  MACRO DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of the PageSlice dense general test.
*/
#define RUN_PAGESLICE_DENSEGENERAL_TEST \
   blazetest::mathtest::pageslice::runTest()
/*! \endcond */
//*************************************************************************************************

} // namespace pageslice

} // namespace mathtest

} // namespace blazetest

#endif
