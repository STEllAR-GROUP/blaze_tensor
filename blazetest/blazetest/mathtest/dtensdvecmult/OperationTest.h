//=================================================================================================
/*!
//  \file blazetest/mathtest/dtensdvecmult/OperationTest.h
//  \brief Header file for the dense tensor/dense vector multiplication operation test
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

#ifndef _BLAZETEST_MATHTEST_DTENSDVECMULT_OPERATIONTEST_H_
#define _BLAZETEST_MATHTEST_DTENSDVECMULT_OPERATIONTEST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>
#include <blaze/math/Aliases.h>
#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/constraints/ColumnMajorMatrix.h>
#include <blaze/math/constraints/ColumnVector.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/RowMajorMatrix.h>
#include <blaze/math/constraints/RowVector.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/constraints/SparseVector.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/Functors.h>
#include <blaze/math/shims/Equal.h>
#include <blaze/math/shims/IsDivisor.h>
#include <blaze/math/StaticMatrix.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsUniform.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/math/typetraits/UnderlyingNumeric.h>
#include <blaze/math/Views.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/util/FalseType.h>
#include <blaze/util/mpl/Not.h>
#include <blaze/util/Random.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/typetraits/Decay.h>
#include <blazetest/system/MathTest.h>
#include <blazetest/mathtest/Creator.h>
#include <blazetest/mathtest/IsEqual.h>
#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/StaticTensor.h>
#include <blaze_tensor/math/traits/MultTrait.h>

namespace blazetest {

namespace mathtest {

namespace dtensdvecmult {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class template for the dense tensor/dense vector multiplication operation test.
//
// This class template represents one particular tensor/vector multiplication test between a
// tensor and a vector of particular types. The two template arguments \a TT and \a VT represent
// the types of the left-hand side tensor and right-hand side vector, respectively.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
class OperationTest
{
 private:
   //**Type definitions****************************************************************************
   using TET = blaze::ElementType_t<TT>;  //!< Element type of the tensor type
   using VET = blaze::ElementType_t<VT>;  //!< Element type of the vector type

   using TTT  = blaze::TransposeType_t<TT>;   //!< Transpose tensor type
   using TVT  = blaze::TransposeType_t<VT>;   //!< Transpose vector type

   //! Dense result type
   using DRE = blaze::MultTrait_t<TT,VT>;

   using DET  = blaze::ElementType_t<DRE>;    //!< Element type of the dense result
   using TDRE = blaze::TransposeType_t<DRE>;  //!< Transpose dense result type

   using TRT  = blaze::DynamicTensor<TET>;           //!< Tensor reference type
   using VRT  = blaze::DynamicVector<VET,false>;     //!< Vector reference type
   using RRE  = blaze::MultTrait_t<TRT,VRT>;         //!< Reference result type
   using TRRE = blaze::TransposeType_t<RRE>;         //!< Transpose reference result type

   //! Type of the tensor/vector multiplication expression
   using TensVecMultExprType = blaze::Decay_t< decltype( std::declval<TT>() * std::declval<VT>() ) >;

   ////! Type of the transpose tensor/vector multiplication expression
   //using TTensVecMultExprType = blaze::Decay_t< decltype( std::declval<OTT>() * std::declval<VT>() ) >;
   //**********************************************************************************************

 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit OperationTest( const Creator<TT>& creator1, const Creator<VT>& creator2 );
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
                          void testInitialStatus     ();
                          void testAssignment        ();
                          void testEvaluation        ();
                          void testElementAccess     ();
                          void testBasicOperation    ();
                          void testNegatedOperation  ();
   template< typename T > void testScaledOperation   ( T scalar );
                          void testTransOperation    ();
                          void testCTransOperation   ();
                          void testAbsOperation      ();
                          void testConjOperation     ();
                          void testRealOperation     ();
                          void testImagOperation     ();
                          void testEvalOperation     ();
                          void testSerialOperation   ();
                          void testSubmatrixOperation( blaze::TrueType  );
                          void testSubmatrixOperation( blaze::FalseType );
                          void testRowOperation      ( blaze::TrueType  );
                          void testRowOperation      ( blaze::FalseType );
                          void testColumnOperation   ( blaze::TrueType  );
                          void testColumnOperation   ( blaze::FalseType );

   template< typename OP > void testCustomOperation( OP op, const std::string& name );
   //@}
   //**********************************************************************************************

   //**Error detection functions*******************************************************************
   /*!\name Error detection functions */
   //@{
   template< typename LT > void checkResults();
   template< typename LT > void checkTransposeResults();
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   void initResults();
   void initTransposeResults();
   template< typename LT > void convertException( const std::exception& ex );
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   TT   lhs_;      //!< The left-hand side dense tensor.
   VT   rhs_;      //!< The right-hand side dense vector.
   DRE  dres_;     //!< The dense result matrix.
   TRT  reflhs_;   //!< The reference left-hand side tensor.
   VRT  refrhs_;   //!< The reference right-hand side vector.
   RRE  refres_;   //!< The reference result.
   TDRE tdres_;    //!< The transpose dense result matrix.
   TRRE trefres_;  //!< The transpose reference result.

   std::string test_;   //!< Label of the currently performed test.
   std::string error_;  //!< Description of the current error type.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TT   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( VT   );

   BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE      ( VT   );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( DRE  );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE      ( VRT  );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( TDRE );

   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( TET, blaze::ElementType_t<TTT>    );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( VET, blaze::ElementType_t<TVT>    );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<DRE>    );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<TDRE>   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( TT , blaze::TransposeType_t<TTT>  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( VT , blaze::TransposeType_t<TVT>  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DRE, blaze::TransposeType_t<TDRE> );

   BLAZE_CONSTRAINT_MATRICES_MUST_HAVE_SAME_STORAGE_ORDER     ( TensVecMultExprType, blaze::ResultType_t<TensVecMultExprType>    );
   BLAZE_CONSTRAINT_MATRICES_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TensVecMultExprType, blaze::TransposeType_t<TensVecMultExprType> );

   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the dense tensor/dense vector multiplication operation test.
//
// \param creator1 The creator for the left-hand side dense tensor of the multiplication.
// \param creator2 The creator for the right-hand side dense vector of the multiplication.
// \exception std::runtime_error Operation error detected.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
OperationTest<TT,VT>::OperationTest( const Creator<TT>& creator1, const Creator<VT>& creator2 )
   : lhs_ ( creator1() )  // The left-hand side dense tensor
   , rhs_ ( creator2() )  // The right-hand side dense vector
   , dres_()              // The dense result matrix
   , reflhs_( lhs_ )      // The reference left-hand side tensor
   , refrhs_( rhs_ )      // The reference right-hand side vector
   , refres_()            // The reference result
   , tdres_()             // The transpose dense result matrix.
   , trefres_()           // The transpose reference result.
   , test_()              // Label of the currently performed test
   , error_()             // Description of the current error type
{
   using namespace blaze;

   using Scalar = UnderlyingNumeric_t<DET>;

   testInitialStatus();
   testAssignment();
   testEvaluation();
   testElementAccess();
   testBasicOperation();
   testNegatedOperation();
   testScaledOperation( 2 );
   testScaledOperation( 2UL );
   testScaledOperation( 2.0F );
   testScaledOperation( 2.0 );
   testScaledOperation( Scalar( 2 ) );
   testTransOperation();
   testCTransOperation();
   testAbsOperation();
   testConjOperation();
   testRealOperation();
   testImagOperation();
   testEvalOperation();
   testSerialOperation();
   testSubmatrixOperation( Not< IsUniform<DRE> >() );
   testRowOperation      ( Not< IsUniform<DRE> >() );
   testColumnOperation   ( Not< IsUniform<DRE> >() );
   //testRowsOperation      ( Not< IsUniform<DRE> >() ); // needs pageslices
   //testColumnsOperation   ( Not< IsUniform<DRE> >() ); // needs rowslices
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Tests on the initial status of the operands.
//
// \return void
// \exception std::runtime_error Initialization error detected.
//
// This function runs tests on the initial status of the operands. In case any initialization
// error is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testInitialStatus()
{
   //=====================================================================================
   // Performing initial tests with the given types
   //=====================================================================================

   // Checking the number of pages of the left-hand side operand
   if( lhs_.pages() != reflhs_.pages() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of left-hand side dense operand\n"
          << " Error: Invalid number of rows\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Detected number of pages = " << lhs_.pages() << "\n"
          << "   Expected number of pages = " << reflhs_.pages() << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( lhs_.rows() != reflhs_.rows() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of left-hand side dense operand\n"
          << " Error: Invalid number of rows\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Detected number of rows = " << lhs_.rows() << "\n"
          << "   Expected number of rows = " << reflhs_.rows() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the number of columns of the left-hand side operand
   if( lhs_.columns() != reflhs_.columns() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of left-hand side dense operand\n"
          << " Error: Invalid number of columns\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Detected number of columns = " << lhs_.columns() << "\n"
          << "   Expected number of columns = " << reflhs_.columns() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the size of the right-hand side operand
   if( rhs_.size() != refrhs_.size() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of right-hand side dense operand\n"
          << " Error: Invalid vector size\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Dense vector type:\n"
          << "     " << typeid( VT ).name() << "\n"
          << "   Detected size = " << rhs_.size() << "\n"
          << "   Expected size = " << refrhs_.size() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the initialization of the left-hand side operand
   if( !isEqual( lhs_, reflhs_ ) ) {
      std::ostringstream oss;
      oss << " Test: Initial test of initialization of left-hand side dense operand\n"
          << " Error: Invalid tensor initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Current initialization:\n" << lhs_ << "\n"
          << "   Expected initialization:\n" << reflhs_ << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the initialization of the right-hand side operand
   if( !isEqual( rhs_, refrhs_ ) ) {
      std::ostringstream oss;
      oss << " Test: Initial test of initialization of right-hand side dense operand\n"
          << " Error: Invalid vector initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Dense vector type:\n"
          << "     " << typeid( VT ).name() << "\n"
          << "   Current initialization:\n" << rhs_ << "\n"
          << "   Expected initialization:\n" << refrhs_ << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the vector assignment.
//
// \return void
// \exception std::runtime_error Assignment error detected.
//
// This function tests the vector assignment. In case any error is detected, a
// \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testAssignment()
{
   //=====================================================================================
   // Performing an assignment with the given types
   //=====================================================================================

   try {
      lhs_ = reflhs_;
      rhs_ = refrhs_;
   }
   catch( std::exception& ex ) {
      std::ostringstream oss;
      oss << " Test: Assignment with the given types\n"
          << " Error: Failed assignment\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Right-hand side dense vector type:\n"
          << "     " << typeid( VT ).name() << "\n"
          << "   Error message: " << ex.what() << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( !isEqual( lhs_, reflhs_ ) ) {
      std::ostringstream oss;
      oss << " Test: Checking the assignment result of left-hand side dense operand\n"
          << " Error: Invalid tensor initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Current initialization:\n" << lhs_ << "\n"
          << "   Expected initialization:\n" << reflhs_ << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( !isEqual( rhs_, refrhs_ ) ) {
      std::ostringstream oss;
      oss << " Test: Checking the assignment result of right-hand side dense operand\n"
          << " Error: Invalid vector initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Dense vector type:\n"
          << "     " << typeid( VT ).name() << "\n"
          << "   Current initialization:\n" << rhs_ << "\n"
          << "   Expected initialization:\n" << refrhs_ << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the explicit evaluation.
//
// \return void
// \exception std::runtime_error Evaluation error detected.
//
// This function tests the explicit evaluation. In case any error is detected, a
// \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testEvaluation()
{
   using blaze::IsRowMajorTensor;


   //=====================================================================================
   // Testing the evaluation with the given types
   //=====================================================================================

   {
      const auto res   ( evaluate( lhs_    * rhs_    ) );
      const auto refres( evaluate( reflhs_ * refrhs_ ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with the given tensor/vector\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side " << ( IsRowMajorTensor<TT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
             << "     " << typeid( lhs_ ).name() << "\n"
             << "   Right-hand side dense vector type:\n"
             << "     " << typeid( rhs_ ).name() << "\n"
             << "   Deduced result type:\n"
             << "     " << typeid( res ).name() << "\n"
             << "   Deduced reference result type:\n"
             << "     " << typeid( refres ).name() << "\n"
             << "   Result:\n" << res << "\n"
             << "   Expected result:\n" << refres << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      const auto res   ( evaluate( eval( lhs_ )    * eval( rhs_ )    ) );
      const auto refres( evaluate( eval( reflhs_ ) * eval( refrhs_ ) ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with evaluated tensor/vector\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side " << ( IsRowMajorTensor<TT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
             << "     " << typeid( lhs_ ).name() << "\n"
             << "   Right-hand side dense vector type:\n"
             << "     " << typeid( rhs_ ).name() << "\n"
             << "   Deduced result type:\n"
             << "     " << typeid( res ).name() << "\n"
             << "   Deduced reference result type:\n"
             << "     " << typeid( refres ).name() << "\n"
             << "   Result:\n" << res << "\n"
             << "   Expected result:\n" << refres << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the vector element access.
//
// \return void
// \exception std::runtime_error Element access error detected.
//
// This function tests the element access via the subscript operator. In case any
// error is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testElementAccess()
{
   using blaze::equal;


   //=====================================================================================
   // Testing the element access with the given types
   //=====================================================================================

   if( lhs_.pages() > 0UL && lhs_.rows() > 0UL )
   {
      const size_t m( lhs_.pages() - 1UL );
      const size_t n( lhs_.rows() - 1UL );

      if( !equal( ( lhs_ * rhs_ )(m,n), ( reflhs_ * refrhs_ )(m,n) ) ||
          !equal( ( lhs_ * rhs_ ).at(m,n), ( reflhs_ * refrhs_ ).at(m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of multiplication expression\n"
             << " Error: Unequal resulting elements at index " << n << " detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side row-major dense tensor type:\n"
             << "     " << typeid( TT ).name() << "\n"
             << "   Right-hand side dense vector type:\n"
             << "     " << typeid( VT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( ( lhs_ * eval( rhs_ ) )(m,n), ( reflhs_ * eval( refrhs_ ) )(m,n) ) ||
          !equal( ( lhs_ * eval( rhs_ ) ).at(m,n), ( reflhs_ * eval( refrhs_ ) ).at(m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of right evaluated multiplication expression\n"
             << " Error: Unequal resulting elements at index " << n << " detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side row-major dense tensor type:\n"
             << "     " << typeid( TT ).name() << "\n"
             << "   Right-hand side dense vector type:\n"
             << "     " << typeid( VT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( ( eval( lhs_ ) * rhs_ )(m,n), ( eval( reflhs_ ) * refrhs_ )(m,n) ) ||
          !equal( ( eval( lhs_ ) * rhs_ ).at(m,n), ( eval( reflhs_ ) * refrhs_ ).at(m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of left evaluated multiplication expression\n"
             << " Error: Unequal resulting elements at index " << n << " detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side row-major dense tensor type:\n"
             << "     " << typeid( TT ).name() << "\n"
             << "   Right-hand side dense vector type:\n"
             << "     " << typeid( VT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( ( eval( lhs_ ) * eval( rhs_ ) )(m,n), ( eval( reflhs_ ) * eval( refrhs_ ) )(m,n) ) ||
          !equal( ( eval( lhs_ ) * eval( rhs_ ) ).at(m,n), ( eval( reflhs_ ) * eval( refrhs_ ) ).at(m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of fully evaluated multiplication expression\n"
             << " Error: Unequal resulting elements at index " << n << " detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side row-major dense tensor type:\n"
             << "     " << typeid( TT ).name() << "\n"
             << "   Right-hand side dense vector type:\n"
             << "     " << typeid( VT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   try {
      ( lhs_ * rhs_ ).at( lhs_.rows(),lhs_.columns() );

      std::ostringstream oss;
      oss << " Test : Checked element access of multiplication expression\n"
          << " Error: Out-of-bound access succeeded\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Right-hand side dense vector type:\n"
          << "     " << typeid( VT ).name() << "\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::out_of_range& ) {}
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the plain dense tensor/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the plain tensor/vector multiplication with plain assignment, addition
// assignment, subtraction assignment, multiplication assignment, and division assignment. In
// case any error resulting from the multiplication or the subsequent assignment is detected,
// a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testBasicOperation()
{
#if BLAZETEST_MATHTEST_TEST_BASIC_OPERATION
   if( BLAZETEST_MATHTEST_TEST_BASIC_OPERATION > 1 )
   {
      //=====================================================================================
      // Multiplication
      //=====================================================================================

      // Multiplication with the given tensor/vector
      {
         test_  = "Multiplication with the given tensor/vector";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            dres_   = lhs_ * rhs_;
            refres_ = reflhs_ * refrhs_;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Multiplication with evaluated tensor/vector
      {
         test_  = "Multiplication with evaluated tensor/vector";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            dres_   = eval( lhs_ ) * eval( rhs_ );
            refres_ = eval( reflhs_ ) * eval( refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      //=====================================================================================
      // Multiplication with addition assignment
      //=====================================================================================

      // Multiplication with addition assignment with the given tensor/vector
      {
         test_  = "Multiplication with addition assignment with the given tensor/vector";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += lhs_ * rhs_;
            refres_ += reflhs_ * refrhs_;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Multiplication with addition assignment with evaluated tensor/vector
      {
         test_  = "Multiplication with addition assignment with evaluated tensor/vector";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += eval( lhs_ ) * eval( rhs_ );
            refres_ += eval( reflhs_ ) * eval( refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      //=====================================================================================
      // Multiplication with subtraction assignment
      //=====================================================================================

      // Multiplication with subtraction assignment with the given tensor/vector
      {
         test_  = "Multiplication with subtraction assignment with the given tensor/vector";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= lhs_ * rhs_;
            refres_ -= reflhs_ * refrhs_;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Multiplication with subtraction assignment with evaluated tensor/vector
      {
         test_  = "Multiplication with subtraction assignment with evaluated tensor/vector";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= eval( lhs_ ) * eval( rhs_ );
            refres_ -= eval( reflhs_ ) * eval( refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Multiplication with schur assignment
      //=====================================================================================

      // Multiplication with schur assignment with the given tensor/vector
      {
         test_  = "Multiplication with schur assignment with the given tensor/vector";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            dres_   %= lhs_ * rhs_;
            refres_ %= reflhs_ * refrhs_;

         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Multiplication with schur assignment with evaluated tensor/vector
      {
         test_  = "Multiplication with schur assignment with evaluated tensor/vector";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            dres_   %= eval( lhs_ ) * eval( rhs_ );
            refres_ %= eval( reflhs_ ) * eval( refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the negated dense tensor/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the negated tensor/vector multiplication with plain assignment, addition
// assignment, subtraction assignment, multiplication assignment, and division assignment. In
// case any error resulting from the multiplication or the subsequent assignment is detected,
// a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testNegatedOperation()
{
#if BLAZETEST_MATHTEST_TEST_NEGATED_OPERATION
   if( BLAZETEST_MATHTEST_TEST_NEGATED_OPERATION > 1 )
   {
      //=====================================================================================
      // Negated multiplication
      //=====================================================================================

      // Negated multiplication with the given tensor/vector
      {
         test_  = "Negated multiplication with the given tensor/vector";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            dres_   = -( lhs_ * rhs_ );
            refres_ = -( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Negated multiplication with evaluated tensor/vector
      {
         test_  = "Negated multiplication with evaluated tensor/vector";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            dres_   = -( eval( lhs_ ) * eval( rhs_ ) );
            refres_ = -( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

      }


      //=====================================================================================
      // Negated multiplication with addition assignment
      //=====================================================================================

      // Negated multiplication with addition assignment with the given tensor/vector
      {
         test_  = "Negated multiplication with addition assignment with the given tensor/vector";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += -( lhs_ * rhs_ );
            refres_ += -( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Negated multiplication with addition assignment with evaluated tensor/vector
      {
         test_  = "Negated multiplication with addition assignment with evaluated tensor/vector";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += -( eval( lhs_ ) * eval( rhs_ ) );
            refres_ += -( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Negated multiplication with subtraction assignment
      //=====================================================================================

      // Negated multiplication with subtraction assignment with the given tensor/vector
      {
         test_  = "Negated multiplication with subtraction assignment with the given tensor/vector";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= -( lhs_ * rhs_ );
            refres_ -= -( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Negated multiplication with subtraction assignment with evaluated tensor/vector
      {
         test_  = "Negated multiplication with subtraction assignment with evaluated tensor/vector";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= -( eval( lhs_ ) * eval( rhs_ ) );
            refres_ -= -( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Negated multiplication with schur assignment
      //=====================================================================================

      // Negated multiplication with schur assignment with the given tensor/vector
      {
         test_  = "Negated multiplication with schur assignment with the given tensor/vector";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            dres_   %= -( lhs_ * rhs_ );
            refres_ %= -( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Negated multiplication with schur assignment with evaluated tensor/vector
      {
         test_  = "Negated multiplication with schur assignment with evaluated tensor/vector";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            dres_   %= -( eval( lhs_ ) * eval( rhs_ ) );
            refres_ %= -( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the scaled dense tensor/dense vector multiplication.
//
// \param scalar The scalar value.
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the scaled tensor/vector multiplication with plain assignment, addition
// assignment, subtraction assignment, multiplication assignment, and division assignment. In
// case any error resulting from the multiplication or the subsequent assignment is detected,
// a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
template< typename T >   // Type of the scalar
void OperationTest<TT,VT>::testScaledOperation( T scalar )
{
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( T );

   if( scalar == T(0) )
      throw std::invalid_argument( "Invalid scalar parameter" );


//#if BLAZETEST_MATHTEST_TEST_SCALED_OPERATION
//   if( BLAZETEST_MATHTEST_TEST_SCALED_OPERATION > 1 )
   {
      //=====================================================================================
      // Self-scaling (v*=s)
      //=====================================================================================

      // Self-scaling (v*=s)
      {
         test_ = "Self-scaling (v*=s)";

         try {
            dres_   = lhs_ * rhs_;
            refres_ = dres_;

            dres_   *= scalar;
            refres_ *= scalar;
         }
         catch( std::exception& ex ) {
            std::ostringstream oss;
            oss << " Test : " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Random seed = " << blaze::getSeed() << "\n"
                << "   Scalar = " << scalar << "\n"
                << "   Error message: " << ex.what() << "\n";
            throw std::runtime_error( oss.str() );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Self-scaling (v=v*s)
      //=====================================================================================

      // Self-scaling (v=v*s)
      {
         test_ = "Self-scaling (v=v*s)";

         try {
            dres_   = lhs_ * rhs_;
            refres_ = dres_;

            dres_   = dres_   * scalar;
            refres_ = refres_ * scalar;
         }
         catch( std::exception& ex ) {
            std::ostringstream oss;
            oss << " Test : " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Random seed = " << blaze::getSeed() << "\n"
                << "   Scalar = " << scalar << "\n"
                << "   Error message: " << ex.what() << "\n";
            throw std::runtime_error( oss.str() );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Self-scaling (v=s*v)
      //=====================================================================================

      // Self-scaling (v=s*v)
      {
         test_ = "Self-scaling (v=s*v)";

         try {
            dres_   = lhs_ * rhs_;
            refres_ = dres_;

            dres_   = scalar * dres_;
            refres_ = scalar * refres_;
         }
         catch( std::exception& ex ) {
            std::ostringstream oss;
            oss << " Test : " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Random seed = " << blaze::getSeed() << "\n"
                << "   Scalar = " << scalar << "\n"
                << "   Error message: " << ex.what() << "\n";
            throw std::runtime_error( oss.str() );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Self-scaling (v/=s)
      //=====================================================================================

      // Self-scaling (v/=s)
      {
         test_ = "Self-scaling (v/=s)";

         try {
            dres_   = lhs_ * rhs_;
            refres_ = dres_;

            dres_   /= scalar;
            refres_ /= scalar;
         }
         catch( std::exception& ex ) {
            std::ostringstream oss;
            oss << " Test : " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Random seed = " << blaze::getSeed() << "\n"
                << "   Scalar = " << scalar << "\n"
                << "   Error message: " << ex.what() << "\n";
            throw std::runtime_error( oss.str() );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Self-scaling (v=v/s)
      //=====================================================================================

      // Self-scaling (v=v/s)
      {
         test_ = "Self-scaling (v=v/s)";

         try {
            dres_   = lhs_ * rhs_;
            refres_ = dres_;

            dres_   = dres_   / scalar;
            refres_ = refres_ / scalar;
         }
         catch( std::exception& ex ) {
            std::ostringstream oss;
            oss << " Test : " << test_ << "\n"
                << " Error: Failed self-scaling operation\n"
                << " Details:\n"
                << "   Random seed = " << blaze::getSeed() << "\n"
                << "   Scalar = " << scalar << "\n"
                << "   Error message: " << ex.what() << "\n";
            throw std::runtime_error( oss.str() );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Scaled multiplication (s*OP)
      //=====================================================================================

      // Scaled multiplication with the given tensor/vector
      {
         test_  = "Scaled multiplication with the given tensor/vector (s*OP)";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            dres_   = scalar * ( lhs_ * rhs_ );
            refres_ = scalar * ( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Scaled multiplication with evaluated tensor/vector
      {
         test_  = "Scaled multiplication with evaluated tensor/vector (s*OP)";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            dres_   = scalar * ( eval( lhs_ ) * eval( rhs_ ) );
            refres_ = scalar * ( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Scaled multiplication (OP*s)
      //=====================================================================================

      // Scaled multiplication with the given tensor/vector
      {
         test_  = "Scaled multiplication with the given tensor/vector (OP*s)";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            dres_   = ( lhs_ * rhs_ ) * scalar;
            refres_ = ( reflhs_ * refrhs_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Scaled multiplication with evaluated tensor/vector
      {
         test_  = "Scaled multiplication with evaluated tensor/vector (OP*s)";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            dres_   = ( eval( lhs_ ) * eval( rhs_ ) ) * scalar;
            refres_ = ( eval( reflhs_ ) * eval( refrhs_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Scaled multiplication (OP/s)
      //=====================================================================================

      // Scaled multiplication with the given tensor/vector
      {
         test_  = "Scaled multiplication with the given tensor/vector (OP/s)";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            dres_   = ( lhs_ * rhs_ ) / scalar;
            refres_ = ( reflhs_ * refrhs_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Scaled multiplication with evaluated tensor/vector
      {
         test_  = "Scaled multiplication with evaluated tensor/vector (OP/s)";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            dres_   = ( eval( lhs_ ) * eval( rhs_ ) ) / scalar;
            refres_ = ( eval( reflhs_ ) * eval( refrhs_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Scaled multiplication with addition assignment (s*OP)
      //=====================================================================================

      // Scaled multiplication with addition assignment with the given tensor/vector
      {
         test_  = "Scaled multiplication with addition assignment with the given tensor/vector (s*OP)";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += scalar * ( lhs_ * rhs_ );
            refres_ += scalar * ( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Scaled multiplication with addition assignment with evaluated tensor/vector
      {
         test_  = "Scaled multiplication with addition assignment with evaluated tensor/vector (s*OP)";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += scalar * ( eval( lhs_ ) * eval( rhs_ ) );
            refres_ += scalar * ( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Scaled multiplication with addition assignment (OP*s)
      //=====================================================================================

      // Scaled multiplication with addition assignment with the given tensor/vector
      {
         test_  = "Scaled multiplication with addition assignment with the given tensor/vector (OP*s)";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += ( lhs_ * rhs_ ) * scalar;
            refres_ += ( reflhs_ * refrhs_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Scaled multiplication with addition assignment with evaluated tensor/vector
      {
         test_  = "Scaled multiplication with addition assignment with evaluated tensor/vector (OP*s)";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += ( eval( lhs_ ) * eval( rhs_ ) ) * scalar;
            refres_ += ( eval( reflhs_ ) * eval( refrhs_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Scaled multiplication with addition assignment (OP/s)
      //=====================================================================================

      // Scaled multiplication with addition assignment with the given tensor/vector
      {
         test_  = "Scaled multiplication with addition assignment with the given tensor/vector (OP/s)";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += ( lhs_ * rhs_ ) / scalar;
            refres_ += ( reflhs_ * refrhs_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Scaled multiplication with addition assignment with evaluated tensor/vector
      {
         test_  = "Scaled multiplication with addition assignment with evaluated tensor/vector (OP/s)";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += ( eval( lhs_ ) * eval( rhs_ ) ) / scalar;
            refres_ += ( eval( reflhs_ ) * eval( refrhs_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Scaled multiplication with subtraction assignment (s*OP)
      //=====================================================================================

      // Scaled multiplication with subtraction assignment with the given tensor/vector
      {
         test_  = "Scaled multiplication with subtraction assignment with the given tensor/vector (s*OP)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= scalar * ( lhs_ * rhs_ );
            refres_ -= scalar * ( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Scaled multiplication with subtraction assignment with evaluated tensor/vector
      {
         test_  = "Scaled multiplication with subtraction assignment with evaluated tensor/vector (s*OP)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= scalar * ( eval( lhs_ ) * eval( rhs_ ) );
            refres_ -= scalar * ( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Scaled multiplication with subtraction assignment (OP*s)
      //=====================================================================================

      // Scaled multiplication with subtraction assignment with the given tensor/vector
      {
         test_  = "Scaled multiplication with subtraction assignment with the given tensor/vector (OP*s)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ( lhs_ * rhs_ ) * scalar;
            refres_ -= ( reflhs_ * refrhs_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Scaled multiplication with subtraction assignment with evaluated tensor/vector
      {
         test_  = "Scaled multiplication with subtraction assignment with evaluated tensor/vector (OP*s)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ( eval( lhs_ ) * eval( rhs_ ) ) * scalar;
            refres_ -= ( eval( reflhs_ ) * eval( refrhs_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Scaled multiplication with subtraction assignment (OP/s)
      //=====================================================================================

      // Scaled multiplication with subtraction assignment with the given tensor/vector
      {
         test_  = "Scaled multiplication with subtraction assignment with the given tensor/vector (OP/s)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ( lhs_ * rhs_ ) / scalar;
            refres_ -= ( reflhs_ * refrhs_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Scaled multiplication with subtraction assignment with evaluated tensor/vector
      {
         test_  = "Scaled multiplication with subtraction assignment with evaluated tensor/vector (OP/s)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ( eval( lhs_ ) * eval( rhs_ ) ) / scalar;
            refres_ -= ( eval( reflhs_ ) * eval( refrhs_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Scaled multiplication with schur assignment (s*OP)
      //=====================================================================================

      // Scaled multiplication with schur assignment with the given tensor/vector
      {
         test_  = "Scaled multiplication with schur assignment with the given tensor/vector (s*OP)";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            dres_   %= scalar * ( lhs_ * rhs_ );
            refres_ %= scalar * ( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Scaled multiplication with schur assignment with evaluated tensor/vector
      {
         test_  = "Scaled multiplication with schur assignment with evaluated tensor/vector (s*OP)";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            dres_   %= scalar * ( eval( lhs_ ) * eval( rhs_ ) );
            refres_ %= scalar * ( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Scaled multiplication with schur assignment (OP/s)
      //=====================================================================================

      // Scaled multiplication with schur assignment with the given tensor/vector
      {
         test_  = "Scaled multiplication with schur assignment with the given tensor/vector (OP/s)";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            dres_   %= ( lhs_ * rhs_ ) / scalar;
            refres_ %= ( reflhs_ * refrhs_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Scaled multiplication with schur assignment with evaluated tensor/vector
      {
         test_  = "Scaled multiplication with schur assignment with evaluated tensor/vector (OP/s)";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            dres_   %= ( eval( lhs_ ) * eval( rhs_ ) ) / scalar;
            refres_ %= ( eval( reflhs_ ) * eval( refrhs_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }
   }
//#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the transpose dense tensor/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the transpose tensor/vector multiplication with plain assignment,
// addition assignment, subtraction assignment, multiplication assignment, and division
// assignment. In case any error resulting from the multiplication or the subsequent
// assignment is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testTransOperation()
{
#if BLAZETEST_MATHTEST_TEST_TRANS_OPERATION
   if( BLAZETEST_MATHTEST_TEST_TRANS_OPERATION > 1 )
   {
      //=====================================================================================
      // Transpose multiplication
      //=====================================================================================

      // Transpose multiplication with the given tensor/vector
      {
         test_  = "Transpose multiplication with the given tensor/vector";
         error_ = "Failed multiplication operation";

         try {
            initTransposeResults();
            tdres_   = trans( lhs_ * rhs_ );
            trefres_ = trans( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }

      // Transpose multiplication with evaluated tensor/vector
      {
         test_  = "Transpose multiplication with evaluated tensor/vector";
         error_ = "Failed multiplication operation";

         try {
            initTransposeResults();
            tdres_   = trans( eval( lhs_ ) * eval( rhs_ ) );
            trefres_ = trans( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }


      //=====================================================================================
      // Transpose multiplication with addition assignment
      //=====================================================================================

      // Transpose multiplication with addition assignment with the given tensor/vector
      {
         test_  = "Transpose multiplication with addition assignment with the given tensor/vector";
         error_ = "Failed addition assignment operation";

         try {
            initTransposeResults();
            tdres_   += trans( lhs_ * rhs_ );
            trefres_ += trans( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }

      // Transpose multiplication with addition assignment with evaluated tensor/vector
      {
         test_ = "Transpose multiplication with addition assignment with evaluated tensor/vector";
         error_ = "Failed addition assignment operation";

         try {
            initTransposeResults();
            tdres_ += trans(eval(lhs_) * eval(rhs_));
            trefres_ += trans(eval(reflhs_) * eval(refrhs_));
         }
         catch (std::exception& ex) {
            convertException<TT>(ex);
         }

         checkTransposeResults<TT>();
      }

      //=====================================================================================
      // Transpose multiplication with subtraction assignment
      //=====================================================================================

      // Transpose multiplication with subtraction assignment with the given tensor/vector
      {
         test_  = "Transpose multiplication with subtraction assignment with the given tensor/vector";
         error_ = "Failed subtraction assignment operation";

         try {
            initTransposeResults();
            tdres_   -= trans( lhs_ * rhs_ );
            trefres_ -= trans( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }

      // Transpose multiplication with subtraction assignment with evaluated tensor/vector
      {
         test_  = "Transpose multiplication with subtraction assignment with evaluated tensor/vector";
         error_ = "Failed subtraction assignment operation";

         try {
            initTransposeResults();
            tdres_   -= trans( eval( lhs_ ) * eval( rhs_ ) );
            trefres_ -= trans( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }


      //=====================================================================================
      // Transpose multiplication with schur assignment
      //=====================================================================================

      // Transpose multiplication with schur assignment with the given tensor/vector
      {
         test_  = "Transpose multiplication with schur assignment with the given tensor/vector";
         error_ = "Failed multiplication assignment operation";

         try {
            initTransposeResults();
            tdres_   %= trans( lhs_ * rhs_ );
            trefres_ %= trans( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }

      // Transpose multiplication with multiplication assignment with evaluated tensor/vector
      {
         test_  = "Transpose multiplication with schur assignment with evaluated tensor/vector";
         error_ = "Failed multiplication assignment operation";

         try {
            initTransposeResults();
            tdres_   %= trans( eval( lhs_ ) * eval( rhs_ ) );
            trefres_ %= trans( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the conjugate transpose dense tensor/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the conjugate transpose tensor/vector multiplication with plain
// assignment, addition assignment, subtraction assignment, multiplication assignment,
// and division assignment. In case any error resulting from the multiplication or the
// subsequent assignment is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testCTransOperation()
{
#if BLAZETEST_MATHTEST_TEST_CTRANS_OPERATION
   if( BLAZETEST_MATHTEST_TEST_CTRANS_OPERATION > 1 )
   {
      //=====================================================================================
      // Conjugate transpose multiplication
      //=====================================================================================

      // Conjugate transpose multiplication with the given tensor/vector
      {
         test_  = "Conjugate transpose multiplication with the given tensor/vector";
         error_ = "Failed multiplication operation";

         try {
            initTransposeResults();
            tdres_   = ctrans( lhs_ * rhs_ );
            trefres_ = ctrans( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }

      // Conjugate transpose multiplication with evaluated tensor/vector
      {
         test_  = "Conjugate transpose multiplication with evaluated tensor/vector";
         error_ = "Failed multiplication operation";

         try {
            initTransposeResults();
            tdres_   = ctrans( eval( lhs_ ) * eval( rhs_ ) );
            trefres_ = ctrans( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }


      //=====================================================================================
      // Conjugate transpose multiplication with addition assignment
      //=====================================================================================

      // Conjugate transpose multiplication with addition assignment with the given tensor/vector
      {
         test_  = "Conjugate transpose multiplication with addition assignment with the given tensor/vector";
         error_ = "Failed addition assignment operation";

         try {
            initTransposeResults();
            tdres_   += ctrans( lhs_ * rhs_ );
            trefres_ += ctrans( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }

      // Conjugate transpose multiplication with addition assignment with evaluated tensor/vector
      {
         test_  = "Conjugate transpose multiplication with addition assignment with evaluated tensor/vector";
         error_ = "Failed addition assignment operation";

         try {
            initTransposeResults();
            tdres_   += ctrans( eval( lhs_ ) * eval( rhs_ ) );
            trefres_ += ctrans( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }


      //=====================================================================================
      // Conjugate transpose multiplication with subtraction assignment
      //=====================================================================================

      // Conjugate transpose multiplication with subtraction assignment with the given tensor/vector
      {
         test_  = "Conjugate transpose multiplication with subtraction assignment with the given tensor/vector";
         error_ = "Failed subtraction assignment operation";

         try {
            initTransposeResults();
            tdres_   -= ctrans( lhs_ * rhs_ );
            trefres_ -= ctrans( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }

      // Conjugate transpose multiplication with subtraction assignment with evaluated tensor/vector
      {
         test_  = "Conjugate transpose multiplication with subtraction assignment with evaluated tensor/vector";
         error_ = "Failed subtraction assignment operation";

         try {
            initTransposeResults();
            tdres_   -= ctrans( eval( lhs_ ) * eval( rhs_ ) );
            trefres_ -= ctrans( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }


      //=====================================================================================
      // Conjugate transpose multiplication with multiplication assignment
      //=====================================================================================

      // Conjugate transpose multiplication with multiplication assignment with the given tensor/vector
      {
         test_  = "Conjugate transpose multiplication with multiplication assignment with the given tensor/vector";
         error_ = "Failed multiplication assignment operation";

         try {
            initTransposeResults();
            tdres_   %= ctrans( lhs_ * rhs_ );
            trefres_ %= ctrans( reflhs_ * refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }

      // Conjugate transpose multiplication with multiplication assignment with evaluated tensor/vector
      {
         test_  = "Conjugate transpose multiplication with multiplication assignment with evaluated tensor/vector";
         error_ = "Failed multiplication assignment operation";

         try {
            initTransposeResults();
            tdres_   %= ctrans( eval( lhs_ ) * eval( rhs_ ) );
            trefres_ %= ctrans( eval( reflhs_ ) * eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the abs dense tensor/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the abs tensor/vector multiplication with plain assignment, addition
// assignment, subtraction assignment, multiplication assignment, and division assignment. In
// case any error resulting from the multiplication or the subsequent assignment is detected,
// a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testAbsOperation()
{
#if BLAZETEST_MATHTEST_TEST_ABS_OPERATION
   if( BLAZETEST_MATHTEST_TEST_ABS_OPERATION > 1 )
   {
      testCustomOperation( blaze::Abs(), "abs" );
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the conjugate dense tensor/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the conjugate tensor/vector multiplication with plain assignment,
// addition assignment, subtraction assignment, multiplication assignment, and division
// assignment. In case any error resulting from the multiplication or the subsequent
// assignment is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testConjOperation()
{
#if BLAZETEST_MATHTEST_TEST_CONJ_OPERATION
   if( BLAZETEST_MATHTEST_TEST_CONJ_OPERATION > 1 )
   {
      testCustomOperation( blaze::Conj(), "conj" );
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the \a real dense tensor/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the \a real tensor/vector multiplication with plain assignment, addition
// assignment, subtraction assignment, multiplication assignment, and division assignment. In
// case any error resulting from the multiplication or the subsequent assignment is detected,
// a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testRealOperation()
{
#if BLAZETEST_MATHTEST_TEST_REAL_OPERATION
   if( BLAZETEST_MATHTEST_TEST_REAL_OPERATION > 1 )
   {
      testCustomOperation( blaze::Real(), "real" );
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the \a imag dense tensor/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the \a imag tensor/vector multiplication with plain assignment, addition
// assignment, subtraction assignment, multiplication assignment, and division assignment. In
// case any error resulting from the multiplication or the subsequent assignment is detected,
// a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testImagOperation()
{
#if BLAZETEST_MATHTEST_TEST_IMAG_OPERATION
   if( BLAZETEST_MATHTEST_TEST_IMAG_OPERATION > 1 )
   {
      testCustomOperation( blaze::Imag(), "imag" );
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the evaluated dense tensor/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the evaluated tensor/vector multiplication with plain assignment,
// addition assignment, subtraction assignment, multiplication assignment, and division
// assignment. In case any error resulting from the multiplication or the subsequent
// assignment is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testEvalOperation()
{
#if BLAZETEST_MATHTEST_TEST_EVAL_OPERATION
   if( BLAZETEST_MATHTEST_TEST_EVAL_OPERATION > 1 )
   {
      testCustomOperation( blaze::Eval(), "eval" );
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the serialized dense tensor/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the serialized tensor/vector multiplication with plain assignment,
// addition assignment, subtraction assignment, multiplication assignment, and division
// assignment. In case any error resulting from the multiplication or the subsequent
// assignment is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testSerialOperation()
{
#if BLAZETEST_MATHTEST_TEST_SERIAL_OPERATION
   if( BLAZETEST_MATHTEST_TEST_SERIAL_OPERATION > 1 )
   {
      testCustomOperation( blaze::Serial(), "serial" );
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the subvector-wise dense tensor/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the subvector-wise tensor/vector multiplication with plain assignment,
// addition assignment, subtraction assignment, multiplication assignment, and division
// assignment. In case any error resulting from the multiplication or the subsequent
// assignment is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testSubmatrixOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_SUBMATRIX_OPERATION
   if( BLAZETEST_MATHTEST_TEST_SUBMATRIX_OPERATION > 1 )
   {
      if( lhs_.rows() == 0UL )
         return;


      //=====================================================================================
      // Submatrix-wise multiplication
      //=====================================================================================

      // Submatrix-wise multiplication with the given tensor/vector
      {
         test_  = "Submatrix-wise multiplication with the given tensor/vector";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            for (size_t i = 0UL, rows = 0UL; i < lhs_.pages(); i++) {
               rows = blaze::rand<size_t>(1UL, lhs_.pages() - i );
               for( size_t j = 0UL, columns = 0UL; j < lhs_.rows();j++ )
               {
                  columns = blaze::rand<size_t>(1UL, lhs_.rows() - j );
                  submatrix(dres_,   i, j, rows, columns) = submatrix(lhs_ * rhs_,       i, j, rows, columns);
                  submatrix(refres_, i, j, rows, columns) = submatrix(reflhs_ * refrhs_, i, j, rows, columns);
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Submatrix-wise multiplication with evaluated tensor/vector
      {
         test_  = "Submatrix-wise multiplication with evaluated tensor/vector";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            for (size_t i = 0UL, rows = 0UL; i < lhs_.pages(); i++) {
               rows = blaze::rand<size_t>(1UL, lhs_.pages() - i );
               for (size_t j = 0UL, columns = 0UL; j < lhs_.rows(); j++)
               {
                  columns = blaze::rand<size_t>(1UL, lhs_.rows() - j );
                  submatrix(dres_,   i, j, rows, columns) = submatrix(eval(lhs_) * eval(rhs_),       i, j, rows, columns);
                  submatrix(refres_, i, j, rows, columns) = submatrix(eval(reflhs_) * eval(refrhs_), i, j, rows, columns);
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Submatrix-wise multiplication with addition assignment
      //=====================================================================================

      // Submatrix-wise multiplication with addition assignment with the given tensor/vector
      {
         test_  = "Submatrix-wise multiplication with addition assignment the given tensor/vector";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for (size_t i = 0UL, rows = 0UL; i < lhs_.pages(); i++) {
               rows = blaze::rand<size_t>(1UL, lhs_.pages() - i );
               for (size_t j = 0UL, columns = 0UL; j < lhs_.rows(); j++)
               {
                  columns = blaze::rand<size_t>(1UL, lhs_.rows() - j);
                  submatrix(dres_,   i, j, rows, columns) += submatrix(lhs_ * rhs_,       i, j, rows, columns);
                  submatrix(refres_, i, j, rows, columns) += submatrix(reflhs_ * refrhs_, i, j, rows, columns);
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Submatrix-wise multiplication with addition assignment with evaluated tensor/vector
      {
         test_  = "Submatrix-wise multiplication with addition assignment with evaluated tensor/vector";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for (size_t i = 0UL, rows = 0UL; i < lhs_.pages(); i++) {
               rows = blaze::rand<size_t>(1UL, lhs_.pages() - i );
               for (size_t j = 0UL, columns = 0UL; j < lhs_.rows(); j++)
               {
                  columns = blaze::rand<size_t>(1UL, lhs_.rows() - j);
                  submatrix(dres_,   i, j, rows, columns) += submatrix(eval(lhs_) * eval(rhs_),       i, j, rows, columns);
                  submatrix(refres_, i, j, rows, columns) += submatrix(eval(reflhs_) * eval(refrhs_), i, j, rows, columns);
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      //=====================================================================================
      // Submatrix-wise multiplication with subtraction assignment
      //=====================================================================================

      // Submatrix-wise multiplication with subtraction assignment with the given tensor/vector
      {
         test_  = "Submatrix-wise multiplication with subtraction assignment the given tensor/vector";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for (size_t i = 0UL, rows = 0UL; i < lhs_.pages(); i++) {
               rows = blaze::rand<size_t>(1UL, lhs_.pages() - i );
               for (size_t j = 0UL, columns = 0UL; j < lhs_.rows(); j++)
               {
                  columns = blaze::rand<size_t>(1UL, lhs_.rows() - j);
                  submatrix(dres_,   i, j, rows, columns) -= submatrix(lhs_ * rhs_,       i, j, rows, columns);
                  submatrix(refres_, i, j, rows, columns) -= submatrix(reflhs_ * refrhs_, i, j, rows, columns);
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Submatrix-wise multiplication with subtraction assignment with evaluated tensor/vector
      {
         test_  = "Submatrix-wise multiplication with subtraction assignment with evaluated tensor/vector";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for (size_t i = 0UL, rows = 0UL; i < lhs_.pages(); i++) {
               rows = blaze::rand<size_t>(1UL, lhs_.pages() - i );
               for (size_t j = 0UL, columns = 0UL; j < lhs_.rows(); j++)
               {
                  columns = blaze::rand<size_t>(1UL, lhs_.rows() - j);
                  submatrix(dres_,   i, j, rows, columns) -= submatrix(eval(lhs_) * eval(rhs_),       i, j, rows, columns);
                  submatrix(refres_, i, j, rows, columns) -= submatrix(eval(reflhs_) * eval(refrhs_), i, j, rows, columns);
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      //=====================================================================================
      // Submatrix-wise multiplication with schur assignment
      //=====================================================================================

      // Submatrix-wise multiplication with schur assignment with the given tensor/vector
      {
         test_  = "Submatrix-wise multiplication with schur assignment the given tensor/vector";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            for (size_t i = 0UL, rows = 0UL; i < lhs_.pages(); i++) {
               rows = blaze::rand<size_t>(1UL, lhs_.pages() - i );
               for (size_t j = 0UL, columns = 0UL; j < lhs_.rows(); j++)
               {
                  columns = blaze::rand<size_t>(1UL, lhs_.rows() - j);
                  submatrix(dres_,   i, j, rows, columns) %= submatrix(lhs_ * rhs_,       i, j, rows, columns);
                  submatrix(refres_, i, j, rows, columns) %= submatrix(reflhs_ * refrhs_, i, j, rows, columns);
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Submatrix-wise multiplication with schur assignment with evaluated tensor/vector
      {
         test_  = "Submatrix-wise multiplication with schur assignment with evaluated tensor/vector";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            for (size_t i = 0UL, rows = 0UL; i < lhs_.pages(); i++) {
               rows = blaze::rand<size_t>(1UL, lhs_.pages() - i );
               for (size_t j = 0UL, columns = 0UL; j < lhs_.rows(); j++)
               {
                  columns = blaze::rand<size_t>(1UL, lhs_.rows() - j);
                  submatrix(dres_,   i, j, rows, columns) %= submatrix(eval(lhs_) * eval(rhs_),       i, j, rows, columns);
                  submatrix(refres_, i, j, rows, columns) %= submatrix(eval(reflhs_) * eval(refrhs_), i, j, rows, columns);
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the subvector-wise dense tensor/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function is called in case the subvector-wise tensor/vector multiplication operation is
// not available for the given types \a TT and \a VT.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::testSubmatrixOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the row-wise dense matrix/dense matrix multiplication.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the row-wise matrix multiplication with plain assignment, addition
// assignment, subtraction assignment, and multiplication assignment. In case any error resulting
// from the multiplication or the subsequent assignment is detected, a \a std::runtime_error
// exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vactor
void OperationTest<TT,VT>::testRowOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_ROW_OPERATION
   if( BLAZETEST_MATHTEST_TEST_ROW_OPERATION > 1 )
   {
      if( lhs_.pages() == 0UL || lhs_.rows() == 0UL )
         return;


      //=====================================================================================
      // Row-wise multiplication
      //=====================================================================================

      // Row-wise multiplication with the given matrices
      {
         test_  = "Row-wise multiplication with the given matrices";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               row( dres_  , i ) = row( lhs_ * rhs_, i );
               row( refres_, i ) = row( reflhs_ * refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Row-wise multiplication with evaluated matrices
      {
         test_  = "Row-wise multiplication with evaluated matrices";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               row( dres_  , i ) = row( eval( lhs_ ) * eval( rhs_ ), i );
               row( refres_, i ) = row( eval( reflhs_ ) * eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Row-wise multiplication with addition assignment
      //=====================================================================================

      // Row-wise multiplication with addition assignment with the given matrices
      {
         test_  = "Row-wise multiplication with addition assignment with the given matrices";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               row( dres_  , i ) += row( lhs_ * rhs_, i );
               row( refres_, i ) += row( reflhs_ * refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Row-wise multiplication with addition assignment with evaluated matrices
      {
         test_  = "Row-wise multiplication with addition assignment with evaluated matrices";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               row( dres_  , i ) += row( eval( lhs_ ) * eval( rhs_ ), i );
               row( refres_, i ) += row( eval( reflhs_ ) * eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Row-wise multiplication with subtraction assignment
      //=====================================================================================

      // Row-wise multiplication with subtraction assignment with the given matrices
      {
         test_ = "Row-wise multiplication with subtraction assignment with the given matrices";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for (size_t i = 0UL; i < lhs_.pages(); ++i) {
               row(dres_, i) -= row(lhs_ * rhs_, i);
               row(refres_, i) -= row(reflhs_ * refrhs_, i);
            }
         }
         catch (std::exception& ex) {
            convertException<TT>(ex);
         }

         checkResults<TT>();
      }

      // Row-wise multiplication with subtraction assignment with evaluated matrices
      {
         test_  = "Row-wise multiplication with subtraction assignment with evaluated matrices";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               row( dres_  , i ) -= row( eval( lhs_ ) * eval( rhs_ ), i );
               row( refres_, i ) -= row( eval( reflhs_ ) * eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Row-wise multiplication with multiplication assignment
      //=====================================================================================

      // Row-wise multiplication with multiplication assignment with the given matrices
      {
         test_  = "Row-wise multiplication with multiplication assignment with the given matrices";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               row( dres_  , i ) *= row( lhs_ * rhs_, i );
               row( refres_, i ) *= row( reflhs_ * refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Row-wise multiplication with multiplication assignment with evaluated matrices
      {
         test_  = "Row-wise multiplication with multiplication assignment with evaluated matrices";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               row( dres_  , i ) *= row( eval( lhs_ ) * eval( rhs_ ), i );
               row( refres_, i ) *= row( eval( reflhs_ ) * eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the row-wise dense matrix/dense matrix multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function is called in case the row-wise matrix/matrix multiplication operation is not
// available for the given matrix types \a MT1 and \a MT2.
*/
template< typename TT    // Type of the left-hand side dense matrix
        , typename VT >  // Type of the right-hand side dense matrix
void OperationTest<TT,VT>::testRowOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the column-wise dense matrix/dense matrix multiplication.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the column-wise matrix multiplication with plain assignment, addition
// assignment, subtraction assignment, and multiplication assignment. In case any error resulting
// from the multiplication or the subsequent assignment is detected, a \a std::runtime_error
// exception is thcolumnn.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vactor
void OperationTest<TT,VT>::testColumnOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_COLUMN_OPERATION
   if( BLAZETEST_MATHTEST_TEST_COLUMN_OPERATION > 1 )
   {
      if( lhs_.pages() == 0UL || lhs_.rows() == 0UL )
         return;


      //=====================================================================================
      // Column-wise multiplication
      //=====================================================================================

      // Column-wise multiplication with the given tensor and vector
      {
         test_  = "Column-wise multiplication with the given tensor and vector";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               column( dres_  , i ) = column( lhs_ * rhs_, i );
               column( refres_, i ) = column( reflhs_ * refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Column-wise multiplication with evaluated tensor and vector
      {
         test_  = "Column-wise multiplication with evaluated tensor and vector";
         error_ = "Failed multiplication operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               column( dres_  , i ) = column( eval( lhs_ ) * eval( rhs_ ), i );
               column( refres_, i ) = column( eval( reflhs_ ) * eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Column-wise multiplication with addition assignment
      //=====================================================================================

      // Column-wise multiplication with addition assignment with the given tensor and vector
      {
         test_  = "Column-wise multiplication with addition assignment with the given tensor and vector";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               column( dres_  , i ) += column( lhs_ * rhs_, i );
               column( refres_, i ) += column( reflhs_ * refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Column-wise multiplication with addition assignment with evaluated tensor and vector
      {
         test_  = "Column-wise multiplication with addition assignment with evaluated tensor and vector";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               column( dres_  , i ) += column( eval( lhs_ ) * eval( rhs_ ), i );
               column( refres_, i ) += column( eval( reflhs_ ) * eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Column-wise multiplication with subtraction assignment
      //=====================================================================================

      // Column-wise multiplication with subtraction assignment with the given tensor and vector
      {
         test_ = "Column-wise multiplication with subtraction assignment with the given tensor and vector";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for (size_t i = 0UL; i < lhs_.rows(); ++i) {
               column(dres_, i) -= column(lhs_ * rhs_, i);
               column(refres_, i) -= column(reflhs_ * refrhs_, i);
            }
         }
         catch (std::exception& ex) {
            convertException<TT>(ex);
         }

         checkResults<TT>();
      }

      // Column-wise multiplication with subtraction assignment with evaluated tensor and vector
      {
         test_  = "Column-wise multiplication with subtraction assignment with evaluated tensor and vector";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               column( dres_  , i ) -= column( eval( lhs_ ) * eval( rhs_ ), i );
               column( refres_, i ) -= column( eval( reflhs_ ) * eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }


      //=====================================================================================
      // Column-wise multiplication with multiplication assignment
      //=====================================================================================

      // Column-wise multiplication with multiplication assignment with the given tensor and vector
      {
         test_  = "Column-wise multiplication with multiplication assignment with the given tensor and vector";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               column( dres_  , i ) *= column( lhs_ * rhs_, i );
               column( refres_, i ) *= column( reflhs_ * refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }

      // Column-wise multiplication with multiplication assignment with evaluated tensor and vector
      {
         test_  = "Column-wise multiplication with multiplication assignment with evaluated tensor and vector";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               column( dres_  , i ) *= column( eval( lhs_ ) * eval( rhs_ ), i );
               column( refres_, i ) *= column( eval( reflhs_ ) * eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the column-wise dense matrix/dense matrix multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function is called in case the column-wise matrix/matrix multiplication operation is not
// available for the given matrix types \a MT1 and \a MT2.
*/
template< typename TT    // Type of the left-hand side dense matrix
        , typename VT >  // Type of the right-hand side dense matrix
void OperationTest<TT,VT>::testColumnOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the customized dense tensor/dense vector multiplication.
//
// \param op The custom operation to be tested.
// \param name The human-readable name of the operation.
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the tensor/vector multiplication with plain assignment, addition
// assignment, subtraction assignment, multiplication assignment, and division assignment
// in combination with a custom operation. In case any error resulting from the multiplication
// or the subsequent assignment is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
template< typename OP >  // Type of the custom operation
void OperationTest<TT,VT>::testCustomOperation( OP op, const std::string& name )
{
   //=====================================================================================
   // Customized multiplication
   //=====================================================================================

   // Customized multiplication with the given tensor/vector
   {
      test_  = "Customized multiplication with the given tensor/vector (" + name + ")";
      error_ = "Failed multiplication operation";

      try {
         initResults();
         dres_   = op( lhs_ * rhs_ );
         refres_ = op( reflhs_ * refrhs_ );
      }
      catch( std::exception& ex ) {
         convertException<TT>( ex );
      }

      checkResults<TT>();
   }

   // Customized multiplication with evaluated tensor/vector
   {
      test_  = "Customized multiplication with evaluated tensor/vector (" + name + ")";
      error_ = "Failed multiplication operation";

      try {
         initResults();
         dres_   = op( eval( lhs_ ) * eval( rhs_ ) );
         refres_ = op( eval( reflhs_ ) * eval( refrhs_ ) );
      }
      catch( std::exception& ex ) {
         convertException<TT>( ex );
      }

      checkResults<TT>();
   }


   //=====================================================================================
   // Customized multiplication with addition assignment
   //=====================================================================================

   // Customized multiplication with addition assignment with the given tensor/vector
   {
      test_  = "Customized multiplication with addition assignment with the given tensor/vector (" + name + ")";
      error_ = "Failed addition assignment operation";

      try {
         initResults();
         dres_   += op( lhs_ * rhs_ );
         refres_ += op( reflhs_ * refrhs_ );
      }
      catch( std::exception& ex ) {
         convertException<TT>( ex );
      }

      checkResults<TT>();
   }

   // Customized multiplication with addition assignment with evaluated tensor/vector
   {
      test_  = "Customized multiplication with addition assignment with evaluated tensor/vector (" + name + ")";
      error_ = "Failed addition assignment operation";

      try {
         initResults();
         dres_   += op( eval( lhs_ ) * eval( rhs_ ) );
         refres_ += op( eval( reflhs_ ) * eval( refrhs_ ) );
      }
      catch( std::exception& ex ) {
         convertException<TT>( ex );
      }

      checkResults<TT>();
   }


   //=====================================================================================
   // Customized multiplication with subtraction assignment
   //=====================================================================================

   // Customized multiplication with subtraction assignment with the given tensor/vector
   {
      test_  = "Customized multiplication with subtraction assignment with the given tensor/vector (" + name + ")";
      error_ = "Failed subtraction assignment operation";

      try {
         initResults();
         dres_   -= op( lhs_ * rhs_ );
         refres_ -= op( reflhs_ * refrhs_ );
      }
      catch( std::exception& ex ) {
         convertException<TT>( ex );
      }

      checkResults<TT>();
   }

   // Customized multiplication with subtraction assignment with evaluated tensor/vector
   {
      test_  = "Customized multiplication with subtraction assignment with evaluated tensor/vector (" + name + ")";
      error_ = "Failed subtraction assignment operation";

      try {
         initResults();
         dres_   -= op( eval( lhs_ ) * eval( rhs_ ) );
         refres_ -= op( eval( reflhs_ ) * eval( refrhs_ ) );
      }
      catch( std::exception& ex ) {
         convertException<TT>( ex );
      }

      checkResults<TT>();
   }


   //=====================================================================================
   // Customized multiplication with schur assignment
   //=====================================================================================

   // Customized multiplication with schur assignment with the given tensor/vector
   {
      test_  = "Customized multiplication with schur assignment with the given tensor/vector (" + name + ")";
      error_ = "Failed schur assignment operation";

      try {
         initResults();
         dres_   %= op( lhs_ * rhs_ );
         refres_ %= op( reflhs_ * refrhs_ );
      }
      catch( std::exception& ex ) {
         convertException<TT>( ex );
      }

      checkResults<TT>();
   }

   // Customized multiplication with multiplication assignment with evaluated tensor/vector
   {
      test_  = "Customized multiplication with multiplication assignment with evaluated tensor/vector (" + name + ")";
      error_ = "Failed multiplication assignment operation";

      try {
         initResults();
         dres_   %= op( eval( lhs_ ) * eval( rhs_ ) );
         refres_ %= op( eval( reflhs_ ) * eval( refrhs_ ) );
      }
      catch( std::exception& ex ) {
         convertException<TT>( ex );
      }

      checkResults<TT>();
   }

}
//*************************************************************************************************




//=================================================================================================
//
//  ERROR DETECTION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Checking and comparing the computed results.
//
// \return void
// \exception std::runtime_error Incorrect dense result detected.
// \exception std::runtime_error Incorrect sparse result detected.
//
// This function is called after each test case to check and compare the computed results.
// The template argument \a LT indicates the types of the left-hand side operand used for
// the computations.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
template< typename LT >  // Type of the left-hand side operand
void OperationTest<TT,VT>::checkResults()
{
   using blaze::IsRowMajorTensor;

   if( !isEqual( dres_, refres_ ) ) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect dense result detected\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side " << ( IsRowMajorTensor<LT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
          << "     " << typeid( LT ).name() << "\n"
          << "   Right-hand side dense vector type:\n"
          << "     " << typeid( VT ).name() << "\n"
          << "   Result:\n" << dres_ << "\n"
          << "   Expected result:\n" << refres_ << "\n";
      throw std::runtime_error( oss.str() );
   }

}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking and comparing the computed transpose results.
//
// \return void
// \exception std::runtime_error Incorrect dense result detected.
// \exception std::runtime_error Incorrect sparse result detected.
//
// This function is called after each test case to check and compare the computed transpose
// results. The template argument \a LT indicates the types of the left-hand side operand
// used for the computations.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
template< typename LT >  // Type of the left-hand side operand
void OperationTest<TT,VT>::checkTransposeResults()
{
   using blaze::IsRowMajorTensor;

   if( !isEqual( tdres_, trefres_ ) ) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect dense result detected\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side " << ( IsRowMajorTensor<LT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
          << "     " << typeid( LT ).name() << "\n"
          << "   Right-hand side dense vector type:\n"
          << "     " << typeid( VT ).name() << "\n"
          << "   Transpose result:\n" << tdres_ << "\n"
          << "   Expected transpose result:\n" << trefres_ << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Initializing the non-transpose result vectors.
//
// \return void
//
// This function is called before each non-transpose test case to initialize the according result
// vectors to random values.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::initResults()
{
   const blaze::UnderlyingBuiltin_t<DRE> min( randmin );
   const blaze::UnderlyingBuiltin_t<DRE> max( randmax );

   resize( dres_, pages( lhs_ ), rows( lhs_ ) );
   randomize( dres_, min, max );

   refres_ = dres_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initializing the transpose result vectors.
//
// \return void
//
// This function is called before each transpose test case to initialize the according result
// vectors to random values.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void OperationTest<TT,VT>::initTransposeResults()
{
   const blaze::UnderlyingBuiltin_t<TDRE> min( randmin );
   const blaze::UnderlyingBuiltin_t<TDRE> max( randmax );

   resize( tdres_, rows( lhs_ ), pages( lhs_ ) );
   randomize( tdres_, min, max );

   trefres_ = tdres_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Convert the given exception into a \a std::runtime_error exception.
//
// \param ex The \a std::exception to be extended.
// \return void
// \exception std::runtime_error The converted exception.
//
// This function converts the given exception to a \a std::runtime_error exception. Additionally,
// the function extends the given exception message by all available infortension for the failed
// test. The template argument \a LT indicates the types of the left-hand side operand used for
// used for the computations.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
template< typename LT >  // Type of the left-hand side operand
void OperationTest<TT,VT>::convertException( const std::exception& ex )
{
   using blaze::IsRowMajorTensor;

   std::ostringstream oss;
   oss << " Test : " << test_ << "\n"
       << " Error: " << error_ << "\n"
       << " Details:\n"
       << "   Random seed = " << blaze::getSeed() << "\n"
       << "   Left-hand side " << ( IsRowMajorTensor<LT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
       << "     " << typeid( LT ).name() << "\n"
       << "   Right-hand side dense vector type:\n"
       << "     " << typeid( VT ).name() << "\n"
       << "   Error message: " << ex.what() << "\n";
   throw std::runtime_error( oss.str() );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Testing the tensor/vector multiplication between two specific types.
//
// \param creator1 The creator for the left-hand side tensor.
// \param creator2 The creator for the right-hand side vector.
// \return void
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename VT >  // Type of the right-hand side dense vector
void runTest( const Creator<TT>& creator1, const Creator<VT>& creator2 )
{
#if BLAZETEST_MATHTEST_TEST_MULTIPLICATION
   if( BLAZETEST_MATHTEST_TEST_MULTIPLICATION > 1 )
   {
      for( size_t rep=0UL; rep<repetitions; ++rep ) {
         OperationTest<TT,VT>( creator1, creator2 );
      }
   }
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  MACROS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of a dense tensor/dense vector multiplication test case.
*/
#define RUN_DTENSDVECMULT_OPERATION_TEST( C1, C2 ) \
   blazetest::mathtest::dtensdvecmult::runTest( C1, C2 )
/*! \endcond */
//*************************************************************************************************

} // namespace dtensdvecmult

} // namespace mathtest

} // namespace blazetest

#endif
