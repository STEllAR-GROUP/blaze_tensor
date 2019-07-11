//=================================================================================================
/*!
//  \file blazetest/mathtest/customarray/AlignedPaddedTest.h
//  \brief Header file for the aligned/padded CustomArray class test
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018-2019 Hartmut Kaiser - All Rights Reserved
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

#ifndef _BLAZETEST_MATHTEST_CUSTOMARRAY_ALIGNEDPADDEDTEST_H_
#define _BLAZETEST_MATHTEST_CUSTOMARRAY_ALIGNEDPADDEDTEST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <sstream>
#include <stdexcept>
#include <string>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/util/constraints/SameType.h>
#include <blazetest/system/Types.h>

#include <blaze_tensor/math/CustomArray.h>
#include <blaze_tensor/math/constraints/DenseArray.h>

namespace blazetest {

namespace mathtest {

namespace customarray {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class for all tests of the aligned/padded CustomArray class template.
//
// This class represents a test suite for the specialization of the blaze::CustomArray class
// template for aligned and padded custom matrices. It performs a series of both compile time
// as well as runtime tests.
*/
class AlignedPaddedTest
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit AlignedPaddedTest();
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
   void testAssignment  ();
   void testAddAssign   ();
   void testSubAssign   ();
   void testSchurAssign ();
   void testMultAssign  ();
   void testScaling     ();
   void testFunctionCall();
   void testAt          ();
   void testIterator    ();
   void testNonZeros    ();
   void testReset       ();
   void testClear       ();
   void testSwap        ();
   void testTranspose   ();
   void testCTranspose  ();
   void testIsDefault   ();

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

   //**Type definitions****************************************************************************
   //! Type of the row-major custom tensor.
   using MT = blaze::CustomArray<3,int,blaze::aligned,blaze::padded>;

   using RMT  = MT::Rebind<const double>::Other;   //!< Rebound row-major custom tensor type.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( MT                  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( MT::ResultType      );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( MT::OppositeType    );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( MT::TransposeType   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( RMT                 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( RMT::ResultType     );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( RMT::OppositeType   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( RMT::TransposeType  );

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT::ResultType      );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT::OppositeType    );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT::TransposeType   );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RMT::ResultType     );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RMT::OppositeType   );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RMT::TransposeType  );

   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT::ElementType,   MT::ResultType::ElementType      );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT::ElementType,   MT::OppositeType::ElementType    );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT::ElementType,   MT::TransposeType::ElementType   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( RMT::ElementType,  RMT::ResultType::ElementType     );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( RMT::ElementType,  RMT::OppositeType::ElementType   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( RMT::ElementType,  RMT::TransposeType::ElementType  );
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
/*!\brief Checking the number of rows of the given custom tensor.
//
// \param tensor The custom tensor to be checked.
// \param expectedRows The expected number of rows of the custom tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of rows of the given custom tensor. In case the actual number
// of rows does not correspond to the given expected number of rows, a \a std::runtime_error
// exception is thrown.
*/
template< typename Type >  // Type of the custom tensor
void AlignedPaddedTest::checkRows( const Type& tensor, size_t expectedRows ) const
{
   if( tensor.template dimension<1>() != expectedRows ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of rows detected\n"
          << " Details:\n"
          << "   Number of rows         : " << tensor.template dimension<1>() << "\n"
          << "   Expected number of rows: " << expectedRows << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of columns of the given custom tensor.
//
// \param tensor The custom tensor to be checked.
// \param expectedRows The expected number of columns of the custom tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of columns of the given custom tensor. In case the
// actual number of columns does not correspond to the given expected number of columns,
// a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the custom tensor
void AlignedPaddedTest::checkColumns( const Type& tensor, size_t expectedColumns ) const
{
   if( tensor.template dimension<0>() != expectedColumns ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of columns detected\n"
          << " Details:\n"
          << "   Number of columns         : " << tensor.template dimension<0>() << "\n"
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
void AlignedPaddedTest::checkPages( const Type& tensor, size_t expectedPages ) const
{
   if( tensor.template dimension<2>() != expectedPages ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of pages detected\n"
          << " Details:\n"
          << "   Number of pages         : " << tensor.template dimension<2>() << "\n"
          << "   Expected number of pages: " << expectedPages << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the capacity of the given custom tensor.
//
// \param tensor The custom tensor to be checked.
// \param minCapacity The expected minimum capacity of the custom tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the capacity of the given custom tensor. In case the actual capacity
// is smaller than the given expected minimum capacity, a \a std::runtime_error exception is
// thrown.
*/
template< typename Type >  // Type of the custom tensor
void AlignedPaddedTest::checkCapacity( const Type& tensor, size_t minCapacity ) const
{
   if( capacity( tensor ) < minCapacity ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected\n"
          << " Details:\n"
          << "   Capacity                 : " << capacity( tensor ) << "\n"
          << "   Expected minimum capacity: " << minCapacity << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of non-zero elements of the given custom tensor.
//
// \param tensor The custom tensor to be checked.
// \param expectedNonZeros The expected number of non-zero elements of the custom tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of non-zero elements of the given custom tensor. In
// case the actual number of non-zero elements does not correspond to the given expected
// number, a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the custom tensor
void AlignedPaddedTest::checkNonZeros( const Type& tensor, size_t expectedNonZeros ) const
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
/*!\brief Checking the number of non-zero elements in a specific row/column of the given custom tensor.
//
// \param tensor The custom tensor to be checked.
// \param index The row/column to be checked.
// \param expectedNonZeros The expected number of non-zero elements in the specified row/column.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of non-zero elements in the specified row/column of the
// given custom tensor. In case the actual number of non-zero elements does not correspond
// to the given expected number, a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the custom tensor
void AlignedPaddedTest::checkNonZeros( const Type& tensor, size_t i, size_t k, size_t expectedNonZeros ) const
{
   if( nonZeros( tensor, i, k ) != expectedNonZeros ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of non-zero elements in row " << i << " page " << k << "\n"
          << " Details:\n"
          << "   Number of non-zeros         : " << nonZeros( tensor, i, k ) << "\n"
          << "   Expected number of non-zeros: " << expectedNonZeros << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( capacity( tensor, i, k ) < nonZeros( tensor, i, k ) ) {
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
/*!\brief Testing the functionality of the aligned/padded CustomArray class template.
//
// \return void
*/
void runTest()
{
   AlignedPaddedTest();
}
//*************************************************************************************************




//=================================================================================================
//
//  MACRO DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of the aligned/padded CustomArray class test.
*/
#define RUN_CUSTOMARRAY_ALIGNED_PADDED_TEST \
   blazetest::mathtest::customarray::runTest()
/*! \endcond */
//*************************************************************************************************

} // namespace customarray

} // namespace mathtest

} // namespace blazetest

#endif
