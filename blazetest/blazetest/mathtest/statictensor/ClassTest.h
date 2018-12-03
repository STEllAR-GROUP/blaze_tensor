//=================================================================================================
/*!
//  \file blazetest/mathtest/statictensor/ClassTest.h
//  \brief Header file for the StaticTensor class test
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZETEST_MATHTEST_STATICTENSOR_CLASSTEST_H_
#define _BLAZETEST_MATHTEST_STATICTENSOR_CLASSTEST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/util/AlignedAllocator.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blazetest/system/Types.h>

#include <blaze_tensor/math/StaticTensor.h>
#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/typetraits/IsRowMajorTensor.h>

namespace blazetest {

namespace mathtest {

namespace statictensor {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class for all tests of the StaticTensor class template.
//
// This class represents a test suite for the blaze::StaticTensor class template. It performs
// a series of both compile time as well as runtime tests.
*/
class ClassTest
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit ClassTest();
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
   template< typename Type >
   void testAlignment( const std::string& type );

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
   void checkPages( const Type& tensor, size_t expectedColumns ) const;

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
   using MT  = blaze::StaticTensor<int,2UL,2UL,3UL>;     //!< Type of the row-major static tensor.
//    using OMT = blaze::StaticTensor<int,2UL,2UL,3UL,blaze::columnMajor>;  //!< Type of the column-major static tensor.

   using RMT  = MT::Rebind<double>::Other;   //!< Rebound row-major static tensor type.
//    using ORMT = OMT::Rebind<double>::Other;  //!< Rebound column-major static tensor type.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT                  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT::ResultType      );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT::OppositeType    );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT::TransposeType   );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( OMT                 );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( OMT::ResultType     );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( OMT::OppositeType   );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( OMT::TransposeType  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RMT                 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RMT::ResultType     );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RMT::OppositeType   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RMT::TransposeType  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( ORMT                );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( ORMT::ResultType    );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( ORMT::OppositeType  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( ORMT::TransposeType );

//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( MT                  );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( MT::ResultType      );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( MT::OppositeType    );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( MT::TransposeType   );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( OMT                 );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( OMT::ResultType     );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( OMT::OppositeType   );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( OMT::TransposeType  );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( RMT                 );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( RMT::ResultType     );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( RMT::OppositeType   );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( RMT::TransposeType  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( ORMT                );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( ORMT::ResultType    );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( ORMT::OppositeType  );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( ORMT::TransposeType );

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT::ResultType      );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT::OppositeType    );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT::TransposeType   );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( OMT::ResultType     );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( OMT::OppositeType   );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( OMT::TransposeType  );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RMT::ResultType     );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RMT::OppositeType   );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RMT::TransposeType  );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ORMT::ResultType    );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ORMT::OppositeType  );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ORMT::TransposeType );

   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT::ElementType,   MT::ResultType::ElementType      );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT::ElementType,   MT::OppositeType::ElementType    );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT::ElementType,   MT::TransposeType::ElementType   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( OMT::ElementType,  OMT::ResultType::ElementType     );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( OMT::ElementType,  OMT::OppositeType::ElementType   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( OMT::ElementType,  OMT::TransposeType::ElementType  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( RMT::ElementType,  RMT::ResultType::ElementType     );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( RMT::ElementType,  RMT::OppositeType::ElementType   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( RMT::ElementType,  RMT::TransposeType::ElementType  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ORMT::ElementType, ORMT::ResultType::ElementType    );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ORMT::ElementType, ORMT::OppositeType::ElementType  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ORMT::ElementType, ORMT::TransposeType::ElementType );
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
/*!\brief Test of the alignment of different StaticTensor instances.
//
// \return void
// \param type The string representation of the given template type.
// \exception std::runtime_error Error detected.
//
// This function performs a test of the alignment of both a row-major and a column-major
// StaticTensor instance of the given element type. In case an error is detected, a
// \a std::runtime_error exception is thrown.
*/
template< typename Type >
void ClassTest::testAlignment( const std::string& type )
{
   using RowMajorTensorType    = blaze::StaticTensor<Type,2UL,7UL,5UL>;
//    using ColumnMajorTensorType = blaze::StaticTensor<Type,2UL,7UL,5UL,blaze::columnMajor>;

   using RowMajorAllocatorType    = blaze::AlignedAllocator<RowMajorTensorType>;
//    using ColumnMajorAllocatorType = blaze::AlignedAllocator<ColumnMajorTensorType>;

   constexpr size_t alignment( blaze::AlignmentOf<Type>::value );


   //=====================================================================================
   // Single tensor alignment test
   //=====================================================================================

   if( blaze::IsAligned_v<RowMajorTensorType> )
   {
      const RowMajorTensorType mat;

      const size_t pages( mat.pages() );
      const size_t rows( blaze::usePadding ? mat.rows() : 1UL );

      for( size_t k=0UL; k<pages; ++k )
      {
         for( size_t i=0UL; i<rows; ++i )
         {
            const size_t deviation( reinterpret_cast<size_t>( &mat(k,i,0UL) ) % alignment );

            if( deviation != 0UL ) {
               std::ostringstream oss;
               oss << " Test: Single tensor alignment test (row-major)\n"
                   << " Error: Invalid alignment in row " << i << " page " << k << " detected\n"
                   << " Details:\n"
                   << "   Element type      : " << type << "\n"
                   << "   Expected alignment: " << alignment << "\n"
                   << "   Deviation         : " << deviation << "\n";
               throw std::runtime_error( oss.str() );
            }
         }
      }
   }

//    if( blaze::IsAligned_v<ColumnMajorTensorType> )
//    {
//       const ColumnMajorTensorType mat;
//
//       const size_t columns( blaze::usePadding ? mat.columns() : 1UL );
//
//       for( size_t j=0UL; j<columns; ++j )
//       {
//          const size_t deviation( reinterpret_cast<size_t>( &mat(0UL,j) ) % alignment );
//
//          if( deviation != 0UL ) {
//             std::ostringstream oss;
//             oss << " Test: Single tensor alignment test (column-major)\n"
//                 << " Error: Invalid alignment in column " << j << " detected\n"
//                 << " Details:\n"
//                 << "   Element type      : " << type << "\n"
//                 << "   Expected alignment: " << alignment << "\n"
//                 << "   Deviation         : " << deviation << "\n";
//             throw std::runtime_error( oss.str() );
//          }
//       }
//    }


   //=====================================================================================
   // Static array alignment test
   //=====================================================================================

   if( blaze::IsAligned_v<RowMajorTensorType> )
   {
      const RowMajorTensorType init;
      const std::array<RowMajorTensorType,7UL> mats{ init, init, init, init, init, init, init };

      for( size_t i=0UL; i<mats.size(); ++i )
      {
         const size_t pages( mats[i].pages() );
         const size_t rows( blaze::usePadding ? mats[i].rows() : 1UL );

         for( size_t k=0UL; k<pages; ++k )
         {
            for( size_t j=0UL; j<rows; ++j )
            {
               const size_t deviation( reinterpret_cast<size_t>( &mats[i](k,j,0UL) ) % alignment );

               if( deviation != 0UL ) {
                  std::ostringstream oss;
                  oss << " Test: Static array alignment test (row-major)\n"
                      << " Error: Invalid alignment at index " << i << " in row " << j << " page " << k << " detected\n"
                      << " Details:\n"
                      << "   Element type      : " << type << "\n"
                      << "   Expected alignment: " << alignment << "\n"
                      << "   Deviation         : " << deviation << "\n";
                  throw std::runtime_error( oss.str() );
               }
            }
         }
      }
   }

//    if( blaze::IsAligned_v<ColumnMajorTensorType> )
//    {
//       const ColumnMajorTensorType init;
//       const std::array<ColumnMajorTensorType,7UL> mats{ init, init, init, init, init, init, init };
//
//       for( size_t i=0UL; i<mats.size(); ++i )
//       {
//          const size_t columns( blaze::usePadding ? mats[i].columns() : 1UL );
//
//          for( size_t j=0UL; j<columns; ++j )
//          {
//             const size_t deviation( reinterpret_cast<size_t>( &mats[i](0UL,j) ) % alignment );
//
//             if( deviation != 0UL ) {
//                std::ostringstream oss;
//                oss << " Test: Static array alignment test (column-major)\n"
//                    << " Error: Invalid alignment at index " << i << " in column " << j << " detected\n"
//                    << " Details:\n"
//                    << "   Element type      : " << type << "\n"
//                    << "   Expected alignment: " << alignment << "\n"
//                    << "   Deviation         : " << deviation << "\n";
//                throw std::runtime_error( oss.str() );
//             }
//          }
//       }
//    }


   //=====================================================================================
   // Dynamic array alignment test
   //=====================================================================================

   if( blaze::IsAligned_v<RowMajorTensorType> )
   {
      const RowMajorTensorType init;
      const std::vector<RowMajorTensorType,RowMajorAllocatorType> mats( 7UL, init );

      for( size_t i=0UL; i<mats.size(); ++i )
      {
         const size_t pages( mats[i].pages() );
         const size_t rows( blaze::usePadding ? mats[i].rows() : 1UL );

         for( size_t k=0UL; k<pages; ++k )
         {
            for( size_t j=0UL; j<rows; ++j )
            {
               const size_t deviation( reinterpret_cast<size_t>( &mats[i](k,j,0UL) ) % alignment );

               if( deviation != 0UL ) {
                  std::ostringstream oss;
                  oss << " Test: Dynamic array alignment test (row-major)\n"
                      << " Error: Invalid alignment at index " << i << " in row " << j << " page " << k << " detected\n"
                      << " Details:\n"
                      << "   Element type      : " << type << "\n"
                      << "   Expected alignment: " << alignment << "\n"
                      << "   Deviation         : " << deviation << "\n";
                  throw std::runtime_error( oss.str() );
               }
            }
         }
      }
   }

//    if( blaze::IsAligned_v<ColumnMajorTensorType> )
//    {
//       const ColumnMajorTensorType init;
//       const std::vector<ColumnMajorTensorType,ColumnMajorAllocatorType> mats( 7UL, init );
//
//       for( size_t i=0UL; i<mats.size(); ++i )
//       {
//          const size_t columns( blaze::usePadding ? mats[i].columns() : 1UL );
//
//          for( size_t j=0UL; j<columns; ++j )
//          {
//             const size_t deviation( reinterpret_cast<size_t>( &mats[i](0UL,j) ) % alignment );
//
//             if( deviation != 0UL ) {
//                std::ostringstream oss;
//                oss << " Test: Dynamic array alignment test (column-major)\n"
//                    << " Error: Invalid alignment at index " << i << " in column " << j << " detected\n"
//                    << " Details:\n"
//                    << "   Element type      : " << type << "\n"
//                    << "   Expected alignment: " << alignment << "\n"
//                    << "   Deviation         : " << deviation << "\n";
//                throw std::runtime_error( oss.str() );
//             }
//          }
//       }
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of rows of the given static tensor.
//
// \param tensor The static tensor to be checked.
// \param expectedRows The expected number of rows of the static tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of rows of the given static tensor. In case the actual number
// of rows does not correspond to the given expected number of rows, a \a std::runtime_error
// exception is thrown.
*/
template< typename Type >  // Type of the static tensor
void ClassTest::checkRows( const Type& tensor, size_t expectedRows ) const
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
/*!\brief Checking the number of columns of the given static tensor.
//
// \param tensor The static tensor to be checked.
// \param expectedRows The expected number of columns of the static tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of columns of the given static tensor. In case the
// actual number of columns does not correspond to the given expected number of columns,
// a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the static tensor
void ClassTest::checkColumns( const Type& tensor, size_t expectedColumns ) const
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
void ClassTest::checkPages( const Type& tensor, size_t expectedPages ) const
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
/*!\brief Checking the capacity of the given static tensor.
//
// \param tensor The static tensor to be checked.
// \param minCapacity The expected minimum capacity of the static tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the capacity of the given static tensor. In case the actual capacity
// is smaller than the given expected minimum capacity, a \a std::runtime_error exception is
// thrown.
*/
template< typename Type >  // Type of the static tensor
void ClassTest::checkCapacity( const Type& tensor, size_t minCapacity ) const
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
/*!\brief Checking the total number of non-zero elements of the given static tensor.
//
// \param tensor The static tensor to be checked.
// \param expectedNonZeros The expected number of non-zero elements of the static tensor.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the total number of non-zero elements of the given static tensor.
// In case the actual number of non-zero elements does not correspond to the given expected
// number, a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the static tensor
void ClassTest::checkNonZeros( const Type& tensor, size_t expectedNonZeros ) const
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
/*!\brief Checking the number of non-zero elements in a specific row/column of the given static tensor.
//
// \param tensor The static tensor to be checked.
// \param index The row/column to be checked.
// \param expectedNonZeros The expected number of non-zero elements in the specified row/column.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of non-zero elements in the specified row/column of the
// given static tensor. In case the actual number of non-zero elements does not correspond
// to the given expected number, a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dynamic tensor
void ClassTest::checkNonZeros( const Type& tensor, size_t index, size_t page, size_t expectedNonZeros ) const
{
   if( nonZeros( tensor, index, page ) != expectedNonZeros ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of non-zero elements in row " << index << " page " << page << "\n"
          << " Details:\n"
          << "   Number of non-zeros         : " << nonZeros( tensor, index, page ) << "\n"
          << "   Expected number of non-zeros: " << expectedNonZeros << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( capacity( tensor, index, page ) < nonZeros( tensor, index, page ) ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected in row " << index << " page " << page << "\n"
          << " Details:\n"
          << "   Number of non-zeros: " << nonZeros( tensor, index, page ) << "\n"
          << "   Capacity           : " << capacity( tensor, index, page ) << "\n";
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
/*!\brief Testing the functionality of the StaticTensor class template.
//
// \return void
*/
void runTest()
{
   ClassTest();
}
//*************************************************************************************************




//=================================================================================================
//
//  MACRO DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of the StaticTensor class test.
*/
#define RUN_STATICTENSOR_CLASS_TEST \
   blazetest::mathtest::statictensor::runTest()
/*! \endcond */
//*************************************************************************************************

} // namespace statictensor

} // namespace mathtest

} // namespace blazetest

#endif
