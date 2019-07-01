//=================================================================================================
/*!
//  \file blazetest/mathtest/quatslice/DenseGeneralTest.h
//  \brief Header file for the QuatSlice dense general test
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZETEST_MATHTEST_QUATSLICE_DENSEGENERALTEST_H_
#define _BLAZETEST_MATHTEST_QUATSLICE_DENSEGENERALTEST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <sstream>
#include <stdexcept>
#include <string>
#include <blazetest/system/Types.h>

#include <blaze_tensor/math/DynamicArray.h>
#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/QuatSlice.h>
#include <blaze_tensor/math/constraints/DenseArray.h>
#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/QuatSliceTensor.h>
#include <blaze_tensor/math/typetraits/IsQuatSliceTensor.h>

namespace blazetest {

namespace mathtest {

namespace quatslice {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class for all tests of the dense general QuatSlice specialization.
//
// This class represents a test suite for the blaze::QuatSlice class template specialization for
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
   void testSubtensor();
   //void testRow();
   //void testRows();
   //void testColumn();
   //void testColumns();
   //void testBand();

   template< typename Type >
   void checkSize( const Type& quatslice, size_t expectedSize ) const;

   template< typename Type >
   void checkPages( const Type& quaternion, size_t expectedPages ) const;

   template< typename Type >
   void checkRows( const Type& quaternion, size_t expectedRows ) const;

   template< typename Type >
   void checkColumns( const Type& quaternion, size_t expectedColumns ) const;

   template< typename Type >
   void checkQuats( const Type& quaternion, size_t expectedQuats ) const;

   template< typename Type >
   void checkCapacity( const Type& object, size_t minCapacity ) const;

   template< typename Type >
   void checkNonZeros( const Type& object, size_t expectedNonZeros ) const;

   template< typename Type >
   void checkNonZeros( const Type& quaternion, size_t i, size_t k, size_t expectedNonZeros ) const;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   void initialize();
   //@}
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   using AT  = blaze::DynamicArray<4, int>;    //!< Dynamic quaternion type.
   using RT  = blaze::QuatSlice<AT>;           //!< Dense quatslice type for quaternions.
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   AT  quat_;          //!< dynamic quaternion.
   std::string test_;  //!< Label of the currently performed test.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE     ( AT  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE    ( RT  );
   BLAZE_CONSTRAINT_MUST_BE_QUATSLICE_TENSOR_TYPE( RT  );
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
/*!\brief Checking the size of the given dense quatslice.
//
// \param quatslice The dense quatslice to be checked.
// \param expectedSize The expected size of the dense quatslice.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the size of the given dense quatslice. In case the actual size does not
// correspond to the given expected size, a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dense quatslice
void DenseGeneralTest::checkSize( const Type& quatslice, size_t expectedSize ) const
{
   if( size( quatslice ) != expectedSize ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid size detected\n"
          << " Details:\n"
          << "   Size         : " << size( quatslice ) << "\n"
          << "   Expected size: " << expectedSize << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of quatslices of the given dynamic quaternion.
//
// \param quaternion The dynamic quaternion to be checked.
// \param expectedQuatSlices The expected number of rows of the dynamic quaternion.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of rows of the given dynamic quaternion. In case the actual number
// of quatslices does not correspond to the given expected number of quatslices, a \a std::runtime_error
// exception is thrown.
*/
template< typename Type >  // Type of the dynamic quaternion
void DenseGeneralTest::checkPages( const Type& quaternion, size_t expectedRows ) const
{
   if( pages( quaternion ) != expectedRows ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of pages detected\n"
          << " Details:\n"
          << "   Number of pages         : " << pages( quaternion ) << "\n"
          << "   Expected number of pages: " << expectedRows << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of quatslices of the given dynamic quaternion.
//
// \param quaternion The dynamic quaternion to be checked.
// \param expectedQuatSlices The expected number of rows of the dynamic quaternion.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of rows of the given dynamic quaternion. In case the actual number
// of quatslices does not correspond to the given expected number of quatslices, a \a std::runtime_error
// exception is thrown.
*/
template< typename Type >  // Type of the dynamic quaternion
void DenseGeneralTest::checkRows( const Type& quaternion, size_t expectedRows ) const
{
   if( rows( quaternion ) != expectedRows ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of rows detected\n"
          << " Details:\n"
          << "   Number of rows         : " << rows( quaternion ) << "\n"
          << "   Expected number of rows: " << expectedRows << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of columns of the given dynamic quaternion.
//
// \param quaternion The dynamic quaternion to be checked.
// \param expectedColumns The expected number of columns of the dynamic quaternion.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of columns of the given dynamic quaternion. In case the
// actual number of columns does not correspond to the given expected number of columns,
// a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dynamic quaternion
void DenseGeneralTest::checkColumns( const Type& quaternion, size_t expectedColumns ) const
{
   if( columns( quaternion ) != expectedColumns ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of columns detected\n"
          << " Details:\n"
          << "   Number of columns         : " << columns( quaternion ) << "\n"
          << "   Expected number of columns: " << expectedColumns << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of quats of the given dynamic quaternion.
//
// \param quaternion The dynamic quaternion to be checked.
// \param expectedQuats The expected number of columns of the dynamic quaternion.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of quats of the given dynamic quaternion. In case the
// actual number of quats does not correspond to the given expected number of quats,
// a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dynamic quaternion
void DenseGeneralTest::checkQuats( const Type& quaternion, size_t expectedQuats ) const
{
   if( quats( quaternion ) != expectedQuats ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of quats detected\n"
          << " Details:\n"
          << "   Number of quats         : " << quats( quaternion ) << "\n"
          << "   Expected number of quats: " << expectedQuats << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the capacity of the given dense quatslice or dynamic quaternion.
//
// \param object The dense quatslice or dynamic quaternion to be checked.
// \param minCapacity The expected minimum capacity.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the capacity of the given dense quatslice or dynamic quaternion. In case the actual
// capacity is smaller than the given expected minimum capacity, a \a std::runtime_error exception
// is thrown.
*/
template< typename Type >  // Type of the dense quatslice or dynamic quaternion
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
/*!\brief Checking the number of non-zero elements of the given dense quatslice or dynamic quaternion.
//
// \param object The dense quatslice or dynamic quaternion to be checked.
// \param expectedNonZeros The expected number of non-zero elements.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of non-zero elements of the given dense quatslice. In case the
// actual number of non-zero elements does not correspond to the given expected number, a
// \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dense quatslice or dynamic quaternion
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
/*!\brief Checking the number of non-zero elements in a specific quatslice/column of the given dynamic quaternion.
//
// \param quaternion The dynamic quaternion to be checked.
// \param index The quatslice/column to be checked.
// \param expectedNonZeros The expected number of non-zero elements in the specified quatslice/column.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of non-zero elements in the specified quatslice/column of the
// given dynamic quaternion. In case the actual number of non-zero elements does not correspond
// to the given expected number, a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dynamic quaternion
void DenseGeneralTest::checkNonZeros( const Type& quaternion, size_t i, size_t k, size_t expectedNonZeros ) const
{
   if( nonZeros( quaternion, i, k ) != expectedNonZeros ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of non-zero elements in row " << i << " quat " << k << "\n"
          << " Details:\n"
          << "   Number of non-zeros         : " << nonZeros( quaternion, i, k ) << "\n"
          << "   Expected number of non-zeros: " << expectedNonZeros << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( capacity( quaternion, i, k ) < nonZeros( quaternion, i, k ) ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected in row " << i << " quat " << k << "\n"
          << " Details:\n"
          << "   Number of non-zeros: " << nonZeros( quaternion, i, k ) << "\n"
          << "   Capacity           : " << capacity( quaternion, i, k ) << "\n";
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
/*!\brief Testing the functionality of the dense general QuatSlice specialization.
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
/*!\brief Macro for the execution of the QuatSlice dense general test.
*/
#define RUN_QUATSLICE_DENSEGENERAL_TEST \
   blazetest::mathtest::quatslice::runTest()
/*! \endcond */
//*************************************************************************************************

} // namespace quatslice

} // namespace mathtest

} // namespace blazetest

#endif
