//=================================================================================================
/*!
//  \file blazetest/mathtest/dtensdmatschur/OperationTest.h
//  \brief Header file for the dense tensor/dense tensor schur operation test
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

#ifndef _BLAZETEST_MATHTEST_DTENSDMATSCHUR_OPERATIONTEST_H_
#define _BLAZETEST_MATHTEST_DTENSDMATSCHUR_OPERATIONTEST_H_


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
#include <blaze/math/constraints/ColumnMajorMatrix.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/RowMajorMatrix.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/Functors.h>
#include <blaze/math/shims/Equal.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsSquare.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsTriangular.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/math/typetraits/UnderlyingNumeric.h>
#include <blaze/math/Views.h>
#include <blaze/util/algorithms/Min.h>
#include <blaze/util/constraints/Numeric.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Nor.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/Random.h>
#include <blaze/util/typetraits/Decay.h>
#include <blaze/util/typetraits/IsComplex.h>
#include <blazetest/system/LAPACK.h>
#include <blazetest/system/MathTest.h>
#include <blazetest/mathtest/Creator.h>
#include <blazetest/mathtest/IsEqual.h>
#include <blazetest/mathtest/MatchAdaptor.h>
#include <blazetest/mathtest/MatchSymmetry.h>
#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>

#include <blazetest/config/TensorMathTest.h>

#include <blaze_tensor/math/constraints/StorageOrder.h>
// #include <blaze_tensor/math/CompressedTensor.h>
#include <blaze_tensor/math/constraints/Tensor.h>
#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/RowMajorTensor.h>
// #include <blaze_tensor/math/constraints/SparseTensor.h>
#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/typetraits/IsRowMajorTensor.h>
#include <blaze_tensor/math/typetraits/StorageOrder.h>
#include <blaze_tensor/math/Views.h>

namespace blazetest {

namespace mathtest {

namespace dtensdmatschur {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class template for the dense tensor/dense tensor schur operation test.
//
// This class template represents one particular tensor schur test between two tensors of
// a particular type. The two template arguments \a TT and \a MT represent the types of the
// left-hand side and right-hand side tensor, respectively.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense matrix
class OperationTest
{
 private:
   //**Type definitions****************************************************************************
   using ET1 = blaze::ElementType_t<TT>;  //!< Element type 1
   using ET2 = blaze::ElementType_t<MT>;  //!< Element type 2

//    using OTT  = blaze::OppositeType_t<TT>;    //!< Tensor type 1 with opposite storage order
   using OMT  = blaze::OppositeType_t<MT>;    //!< Matrix type with opposite storage order
   using TTT  = blaze::TransposeType_t<TT>;   //!< Transpose tensor type 1
   using TMT  = blaze::TransposeType_t<MT>;   //!< Transpose tensor type 2
//    using TOTT = blaze::TransposeType_t<OTT>;  //!< Transpose tensor type 1 with opposite storage order
//    using TOMT = blaze::TransposeType_t<OMT>;  //!< Transpose tensor type 2 with opposite storage order

   //! Dense result type
   using DRE = blaze::SchurTrait_t<TT,MT>;

   using DET   = blaze::ElementType_t<DRE>;     //!< Element type of the dense result
   using TDRE  = blaze::TransposeType_t<DRE>;   //!< Transpose dense result type

   //! Sparse result type
//    using SRE = MatchAdaptor_t< DRE, blaze::CompressedTensor<DET,false> >;
//
//    using SET   = blaze::ElementType_t<SRE>;     //!< Element type of the sparse result
//    using OSRE  = blaze::OppositeType_t<SRE>;    //!< Sparse result type with opposite storage order
//    using TSRE  = blaze::TransposeType_t<SRE>;   //!< Transpose sparse result type
//    using TOSRE = blaze::TransposeType_t<OSRE>;  //!< Transpose sparse result type with opposite storage order

   using RT1 = blaze::DynamicTensor<ET1>;       //!< Reference type 1
   using RT2 = blaze::DynamicMatrix<ET2,false>;       //!< Reference type 2
//    using RT2 = blaze::CompressedTensor<ET2,false>;  //!< Reference type 2

   //! Reference result type
   using RRE = blaze::SchurTrait_t<RT1, RT2>;

   //! Type of the tensor/tensor schur expression
   using TensMatSchurExprType = blaze::Decay_t< decltype( std::declval<TT>() % std::declval<MT>() ) >;

//    //! Type of the tensor/transpose tensor schur expression
//    using TensTTensAddExprType = blaze::Decay_t< decltype( std::declval<TT>() + std::declval<OMT>() ) >;
//
//    //! Type of the transpose tensor/tensor schur expression
//    using TTensMatSchurExprType = blaze::Decay_t< decltype( std::declval<OTT>() + std::declval<MT>() ) >;
//
//    //! Type of the transpose tensor/transpose tensor schur expression
//    using TTensTTensAddExprType = blaze::Decay_t< decltype( std::declval<OTT>() + std::declval<OMT>() ) >;
   //**********************************************************************************************

 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit OperationTest( const Creator<TT>& creator1, const Creator<MT>& creator2 );
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
                          void testInvOperation      ();
                          void testEvalOperation     ();
                          void testSerialOperation   ();
//                           void testDeclSymOperation  ( blaze::TrueType  );
//                           void testDeclSymOperation  ( blaze::FalseType );
//                           void testDeclHermOperation ( blaze::TrueType  );
//                           void testDeclHermOperation ( blaze::FalseType );
//                           void testDeclLowOperation  ( blaze::TrueType  );
//                           void testDeclLowOperation  ( blaze::FalseType );
//                           void testDeclUppOperation  ( blaze::TrueType  );
//                           void testDeclUppOperation  ( blaze::FalseType );
//                           void testDeclDiagOperation ( blaze::TrueType  );
//                           void testDeclDiagOperation ( blaze::FalseType );
                          void testSubtensorOperation( blaze::TrueType );
                          void testSubtensorOperation( blaze::FalseType );
                          void testRowSliceOperation    ( blaze::TrueType  );
                          void testRowSliceOperation    ( blaze::FalseType );
//                           void testRowSlicesOperation     ( blaze::TrueType  );
//                           void testRowSlicesOperation     ( blaze::FalseType );
                          void testColumnSliceOperation ( blaze::TrueType  );
                          void testColumnSliceOperation ( blaze::FalseType );
//                           void testColumnSlicesOperation  ( blaze::TrueType  );
//                           void testColumnSlicesOperation  ( blaze::FalseType );
                          void testPageSliceOperation   ( blaze::TrueType  );
                          void testPageSliceOperation   ( blaze::FalseType );
//                           void testPageSlicesOperation  ( blaze::TrueType  );
//                           void testPageSlicesOperation  ( blaze::FalseType );
//                           void testBandOperation     ();

   template< typename OP > void testCustomOperation( OP op, const std::string& name );
   //@}
   //**********************************************************************************************

   //**Error detection functions*******************************************************************
   /*!\name Error detection functions */
   //@{
   template< typename LT, typename RT > void checkResults();
   template< typename LT, typename RT > void checkTransposeResults();
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   void initResults();
   void initTransposeResults();
   template< typename LT, typename RT > void convertException( const std::exception& ex );
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   TT    lhs_;     //!< The left-hand side dense tensor.
   MT    rhs_;     //!< The right-hand side dense tensor.
   OMT   orhs_;    //!< The right-hand side dense tensor with opposite storage order.
   DRE   dres_;    //!< The dense result tensor.
   TDRE  tdres_;   //!< The transpose dense result tensor.
   RT1   reflhs_;  //!< The reference left-hand side tensor.
   RT2   refrhs_;  //!< The reference right-hand side tensor.
   RRE   refres_;  //!< The reference result.

   std::string test_;   //!< Label of the currently performed test.
   std::string error_;  //!< Description of the current error type.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TT   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( MT   );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( OTT  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( OMT  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TTT  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( TMT  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TOTT );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TOMT );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( RT1   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( RT2   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( RRE   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( DRE   );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( SRE   );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( ODRE  );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( OSRE  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TDRE  );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( TSRE  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TODRE );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( TOSRE );

    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( TT   );
    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( MT   );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( OTT  );
    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( OMT  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( TTT  );
    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( TMT  );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( TOTT );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( TOMT );
    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( RT1   );
    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( RT2   );
    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( DRE   );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( SRE   );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( ODRE  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( OSRE  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( TDRE  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( TSRE  );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( TODRE );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( TOSRE );

//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET1, blaze::ElementType_t<OTT>   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET2, blaze::ElementType_t<OMT>   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET1, blaze::ElementType_t<TTT>   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET2, blaze::ElementType_t<TMT>   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET1, blaze::ElementType_t<TOTT>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET2, blaze::ElementType_t<TOMT>  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<DRE>    );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<ODRE>   );
   //BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<TDRE>   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<TODRE>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<SRE>    );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<SRE>    );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<OSRE>   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<TSRE>   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<TOSRE>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<DRE>    );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( TT, blaze::OppositeType_t<OTT>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT, blaze::OppositeType_t<OMT>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( TT, blaze::TransposeType_t<TTT> );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT, blaze::TransposeType_t<TMT> );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DRE, blaze::OppositeType_t<ODRE>  );
    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DRE, blaze::TransposeType_t<TDRE> );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SRE, blaze::OppositeType_t<OSRE>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SRE, blaze::TransposeType_t<TSRE> );

   //BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_SAME_STORAGE_ORDER     ( TensMatSchurExprType, blaze::ResultType_t<TensMatSchurExprType>    );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TensMatSchurExprType, blaze::OppositeType_t<TensMatSchurExprType>  );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TensMatSchurExprType, blaze::TransposeType_t<TensMatSchurExprType> );

//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_SAME_STORAGE_ORDER     ( TensTTensAddExprType, blaze::ResultType_t<TensTTensAddExprType>    );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TensTTensAddExprType, blaze::OppositeType_t<TensTTensAddExprType>  );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TensTTensAddExprType, blaze::TransposeType_t<TensTTensAddExprType> );
//
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_SAME_STORAGE_ORDER     ( TTensMatSchurExprType, blaze::ResultType_t<TTensMatSchurExprType>    );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TTensMatSchurExprType, blaze::OppositeType_t<TTensMatSchurExprType>  );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TTensMatSchurExprType, blaze::TransposeType_t<TTensMatSchurExprType> );
//
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_SAME_STORAGE_ORDER     ( TTensTTensAddExprType, blaze::ResultType_t<TTensTTensAddExprType>    );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TTensTTensAddExprType, blaze::OppositeType_t<TTensTTensAddExprType>  );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TTensTTensAddExprType, blaze::TransposeType_t<TTensTTensAddExprType> );
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
/*!\brief Constructor for the dense tensor/dense tensor schur operation test.
//
// \param creator1 The creator for the left-hand side dense tensor of the tensor schur.
// \param creator2 The creator for the right-hand side dense tensor of the tensor schur.
// \exception std::runtime_error Operation error detected.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense matrix
OperationTest<TT,MT>::OperationTest( const Creator<TT>& creator1, const Creator<MT>& creator2 )
   : lhs_( creator1() )  // The left-hand side dense tensor
   , rhs_( creator2() )  // The right-hand side dense matrix
   , orhs_( rhs_ )       // The right-hand side dense matrix with opposite storage order
   , dres_()             // The dense result tensor
   , tdres_()            // The transpose dense result tensor
   , reflhs_( lhs_ )     // The reference left-hand side tensor
   , refrhs_( rhs_ )     // The reference right-hand side matrix
   , refres_()           // The reference result
   , test_()             // Label of the currently performed test
   , error_()            // Description of the current error type
{
   using blaze::Or_t;
   using blaze::Nor_t;

   using Scalar = blaze::UnderlyingNumeric_t<DET>;

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
   testInvOperation();
   testEvalOperation();
   testSerialOperation();
   testSubtensorOperation  ( blaze::Not_t< blaze::IsUniform<DRE> >() );
   //testRowSliceOperation   ( blaze::Not_t< blaze::IsUniform<DRE> >() );
   //testColumnSliceOperation( blaze::Not_t< blaze::IsUniform<DRE> >() );
   testPageSliceOperation  ( blaze::Not_t< blaze::IsUniform<DRE> >() );
//    testRowSlicesOperation( Nor_t< blaze::IsSymmetric<DRE>, blaze::IsHermitian<DRE> >() );
//    testColumnSlicesOperation( Nor_t< blaze::IsSymmetric<DRE>, blaze::IsHermitian<DRE> >() );
//    testPageSlicesOperation( Nor_t< blaze::IsSymmetric<DRE>, blaze::IsHermitian<DRE> >() );
}
//*************************************************************************************************






//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Tests on the initial status of the tensors.
//
// \return void
// \exception std::runtime_error Initialization error detected.
//
// This function runs tests on the initial status of the tensors. In case any initialization
// error is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testInitialStatus()
{
   //=====================================================================================
   // Performing initial tests with the row-major types
   //=====================================================================================

   // Checking the number of rows of the left-hand side operand
   if( lhs_.rows() != reflhs_.rows() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of left-hand side row-major dense operand\n"
          << " Error: Invalid number of rows\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Detected number of rows = " << lhs_.rows() << "\n"
          << "   Expected number of rows = " << reflhs_.rows() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the number of columns of the left-hand side operand
   if( lhs_.columns() != reflhs_.columns() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of left-hand side row-major dense operand\n"
          << " Error: Invalid number of columns\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Detected number of columns = " << lhs_.columns() << "\n"
          << "   Expected number of columns = " << reflhs_.columns() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the number of pages of the left-hand side operand
   if( lhs_.pages() != reflhs_.pages() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of left-hand side row-major dense operand\n"
          << " Error: Invalid number of pages\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Detected number of pages = " << lhs_.pages() << "\n"
          << "   Expected number of pages = " << reflhs_.pages() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the number of rows of the right-hand side operand
   if( rhs_.rows() != refrhs_.rows() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of right-hand side row-major dense operand\n"
          << " Error: Invalid number of rows\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Detected number of rows = " << rhs_.rows() << "\n"
          << "   Expected number of rows = " << refrhs_.rows() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the number of columns of the right-hand side operand
   if( rhs_.columns() != refrhs_.columns() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of right-hand side row-major dense operand\n"
          << " Error: Invalid number of columns\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Detected number of columns = " << rhs_.columns() << "\n"
          << "   Expected number of columns = " << refrhs_.columns() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the initialization of the left-hand side operand
   if( !isEqual( lhs_, reflhs_ ) ) {
      std::ostringstream oss;
      oss << " Test: Initial test of initialization of left-hand side row-major dense operand\n"
          << " Error: Invalid tensor initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Current initialization:\n" << lhs_ << "\n"
          << "   Expected initialization:\n" << reflhs_ << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the initialization of the right-hand side operand
   if( !isEqual( rhs_, refrhs_ ) ) {
      std::ostringstream oss;
      oss << " Test: Initial test of initialization of right-hand side row-major dense operand\n"
          << " Error: Invalid tensor initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Current initialization:\n" << rhs_ << "\n"
          << "   Expected initialization:\n" << refrhs_ << "\n";
      throw std::runtime_error( oss.str() );
   }

}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Testing the tensor assignment.
//
// \return void
// \exception std::runtime_error Assignment error detected.
//
// This function tests the tensor assignment. In case any error is detected, a
// \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testAssignment()
{
   //=====================================================================================
   // Performing an assignment with the row-major types
   //=====================================================================================

   try {
      lhs_ = reflhs_;
      rhs_ = refrhs_;
   }
   catch( std::exception& ex ) {
      std::ostringstream oss;
      oss << " Test: Assignment with the row-major types\n"
          << " Error: Failed assignment\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Right-hand side row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Error message: " << ex.what() << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( !isEqual( lhs_, reflhs_ ) ) {
      std::ostringstream oss;
      oss << " Test: Checking the assignment result of left-hand side row-major dense operand\n"
          << " Error: Invalid tensor initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Current initialization:\n" << lhs_ << "\n"
          << "   Expected initialization:\n" << reflhs_ << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( !isEqual( rhs_, refrhs_ ) ) {
      std::ostringstream oss;
      oss << " Test: Checking the assignment result of right-hand side row-major dense operand\n"
          << " Error: Invalid tensor initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Current initialization:\n" << rhs_ << "\n"
          << "   Expected initialization:\n" << refrhs_ << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************

template <typename TT>
using IsRowMajorTensor = blaze::IsRowMajorTensor<TT>;

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
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testEvaluation()
{
   //=====================================================================================
   // Testing the evaluation with two row-major tensors
   //=====================================================================================

   {
      const auto res   ( evaluate( lhs_    % rhs_    ) );
      const auto refres( evaluate( reflhs_ % refrhs_ ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with the given tensor and matrix\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side " << ( IsRowMajorTensor<TT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
             << "     " << typeid( lhs_ ).name() << "\n"
             << "   Right-hand side " << ( blaze::IsRowMajorMatrix<MT>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense matrix type:\n"
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
      const auto res   ( evaluate( eval( lhs_ )    % eval( rhs_ )    ) );
      const auto refres( evaluate( eval( reflhs_ ) % eval( refrhs_ ) ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with evaluated tensor and matrix\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side " << ( IsRowMajorTensor<TT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
             << "     " << typeid( lhs_ ).name() << "\n"
             << "   Right-hand side " << ( blaze::IsRowMajorMatrix<MT>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense matrix type:\n"
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


    //=====================================================================================
    // Testing the evaluation with a row-major tensor and a column-major matrix
    //=====================================================================================

    {
       const auto res   ( evaluate( lhs_    % orhs_   ) );
       const auto refres( evaluate( reflhs_ % refrhs_ ) );

       if( !isEqual( res, refres ) ) {
          std::ostringstream oss;
          oss << " Test: Evaluation with the given tensor and matrix\n"
              << " Error: Failed evaluation\n"
              << " Details:\n"
              << "   Random seed = " << blaze::getSeed() << "\n"
              << "   Left-hand side " << ( IsRowMajorTensor<TT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
              << "     " << typeid( lhs_ ).name() << "\n"
              << "   Right-hand side " << ( blaze::IsRowMajorMatrix<OMT>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense matrix type:\n"
              << "     " << typeid( orhs_ ).name() << "\n"
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
       const auto res   ( evaluate( eval( lhs_ )    % eval( orhs_ )   ) );
       const auto refres( evaluate( eval( reflhs_ ) % eval( refrhs_ ) ) );

       if( !isEqual( res, refres ) ) {
          std::ostringstream oss;
          oss << " Test: Evaluation with the given tensor and matrix\n"
              << " Error: Failed evaluation\n"
              << " Details:\n"
              << "   Random seed = " << blaze::getSeed() << "\n"
              << "   Left-hand side " << ( IsRowMajorTensor<TT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
              << "     " << typeid( lhs_ ).name() << "\n"
              << "   Right-hand side " << ( blaze::IsRowMajorMatrix<OMT>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense matrix type:\n"
              << "     " << typeid( orhs_ ).name() << "\n"
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
/*!\brief Testing the tensor element access.
//
// \return void
// \exception std::runtime_error Element access error detected.
//
// This function tests the element access via the subscript operator. In case any
// error is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testElementAccess()
{
   using blaze::equal;


   //=====================================================================================
   // Testing the element access with two row-major tensors
   //=====================================================================================

   if( lhs_.rows() > 0UL && lhs_.columns() > 0UL && lhs_.pages() > 0UL )
   {
      const size_t o( lhs_.pages()   - 1UL );
      const size_t m( lhs_.rows()    - 1UL );
      const size_t n( lhs_.columns() - 1UL );

      if( !equal( ( lhs_ % rhs_ )(o,m,n),    ( reflhs_ % refrhs_ )(o,m,n) ) ||
          !equal( ( lhs_ % rhs_ ).at(o,m,n), ( reflhs_ % refrhs_ ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of schur expression\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side row-major dense tensor type:\n"
             << "     " << typeid( TT ).name() << "\n"
             << "   Right-hand side row-major dense matrix type:\n"
             << "     " << typeid( MT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( ( lhs_ % eval( rhs_ ) )(o,m,n),    ( reflhs_ % eval( refrhs_ ) )(o,m,n) ) ||
          !equal( ( lhs_ % eval( rhs_ ) ).at(o,m,n), ( reflhs_ % eval( refrhs_ ) ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of right evaluated schur expression\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side row-major dense tensor type:\n"
             << "     " << typeid( TT ).name() << "\n"
             << "   Right-hand side row-major dense matrix type:\n"
             << "     " << typeid( MT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( ( eval( lhs_ ) % rhs_ )(o,m,n),    ( eval( reflhs_ ) % refrhs_ )(o,m,n) ) ||
          !equal( ( eval( lhs_ ) % rhs_ ).at(o,m,n), ( eval( reflhs_ ) % refrhs_ ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of left evaluated schur expression\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side row-major dense tensor type:\n"
             << "     " << typeid( TT ).name() << "\n"
             << "   Right-hand side row-major dense matrix type:\n"
             << "     " << typeid( MT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( ( eval( lhs_ ) % eval( rhs_ ) )(o,m,n),    ( eval( reflhs_ ) % eval( refrhs_ ) )(o,m,n) ) ||
          !equal( ( eval( lhs_ ) % eval( rhs_ ) ).at(o,m,n), ( eval( reflhs_ ) % eval( refrhs_ ) ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of fully evaluated schur expression\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side row-major dense tensor type:\n"
             << "     " << typeid( TT ).name() << "\n"
             << "   Right-hand side row-major dense matrix type:\n"
             << "     " << typeid( MT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   try {
      ( lhs_ % rhs_ ).at( 0UL, 0UL, lhs_.columns() );

      std::ostringstream oss;
      oss << " Test : Checked element access of schur expression\n"
          << " Error: Out-of-bound access succeeded\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Right-hand side row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::out_of_range& ) {}

   try {
      ( lhs_ % rhs_ ).at( 0UL, lhs_.rows(), 0UL );

      std::ostringstream oss;
      oss << " Test : Checked element access of schur expression\n"
          << " Error: Out-of-bound access succeeded\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Right-hand side row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::out_of_range& ) {}

   try {
      ( lhs_ % rhs_ ).at( lhs_.pages(), 0UL, 0UL );

      std::ostringstream oss;
      oss << " Test : Checked element access of schur expression\n"
          << " Error: Out-of-bound access succeeded\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Right-hand side row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::out_of_range& ) {}


    //=====================================================================================
    // Testing the element access with a row-major tensor and a column-major matrix
    //=====================================================================================

    if( lhs_.rows() > 0UL && lhs_.columns() > 0UL )
    {
       const size_t o( lhs_.pages()   - 1UL );
       const size_t m( lhs_.rows()    - 1UL );
       const size_t n( lhs_.columns() - 1UL );

       if( !equal( ( lhs_ % orhs_ )(o,m,n),    ( reflhs_ % refrhs_ )(o,m,n) ) ||
           !equal( ( lhs_ % orhs_ ).at(o,m,n), ( reflhs_ % refrhs_ ).at(o,m,n) ) ) {
          std::ostringstream oss;
          oss << " Test : Element access of schur expression\n"
              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
              << " Details:\n"
              << "   Random seed = " << blaze::getSeed() << "\n"
              << "   Left-hand side row-major dense tensor type:\n"
              << "     " << typeid( TT ).name() << "\n"
              << "   Right-hand side column-major dense tensor type:\n"
              << "     " << typeid( OMT ).name() << "\n";
          throw std::runtime_error( oss.str() );
       }

       if( !equal( ( lhs_ % eval( orhs_ ) )(o,m,n),    ( reflhs_ % eval( refrhs_ ) )(o,m,n) ) ||
           !equal( ( lhs_ % eval( orhs_ ) ).at(o,m,n), ( reflhs_ % eval( refrhs_ ) ).at(o,m,n) ) ) {
          std::ostringstream oss;
          oss << " Test : Element access of right evaluated schur expression\n"
              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
              << " Details:\n"
              << "   Random seed = " << blaze::getSeed() << "\n"
              << "   Left-hand side row-major dense tensor type:\n"
              << "     " << typeid( TT ).name() << "\n"
              << "   Right-hand side column-major dense tensor type:\n"
              << "     " << typeid( OMT ).name() << "\n";
          throw std::runtime_error( oss.str() );
       }

       if( !equal( ( eval( lhs_ ) % orhs_ )(o,m,n),    ( eval( reflhs_ ) % refrhs_ )(o,m,n) ) ||
           !equal( ( eval( lhs_ ) % orhs_ ).at(o,m,n), ( eval( reflhs_ ) % refrhs_ ).at(o,m,n) ) ) {
          std::ostringstream oss;
          oss << " Test : Element access of left evaluated schur expression\n"
              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
              << " Details:\n"
              << "   Random seed = " << blaze::getSeed() << "\n"
              << "   Left-hand side row-major dense tensor type:\n"
              << "     " << typeid( TT ).name() << "\n"
              << "   Right-hand side column-major dense tensor type:\n"
              << "     " << typeid( OMT ).name() << "\n";
          throw std::runtime_error( oss.str() );
       }

       if( !equal( ( eval( lhs_ ) % eval( orhs_ ) )(o,m,n),    ( eval( reflhs_ ) % eval( refrhs_ ) )(o,m,n) ) ||
           !equal( ( eval( lhs_ ) % eval( orhs_ ) ).at(o,m,n), ( eval( reflhs_ ) % eval( refrhs_ ) ).at(o,m,n) ) ) {
          std::ostringstream oss;
          oss << " Test : Element access of fully evaluated schur expression\n"
              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
              << " Details:\n"
              << "   Random seed = " << blaze::getSeed() << "\n"
              << "   Left-hand side row-major dense tensor type:\n"
              << "     " << typeid( TT ).name() << "\n"
              << "   Right-hand side column-major dense tensor type:\n"
              << "     " << typeid( OMT ).name() << "\n";
          throw std::runtime_error( oss.str() );
       }
    }

    try {
       ( lhs_ % orhs_ ).at( 0UL, 0UL, lhs_.columns() );

       std::ostringstream oss;
       oss << " Test : Checked element access of schur expression\n"
           << " Error: Out-of-bound access succeeded\n"
           << " Details:\n"
           << "   Random seed = " << blaze::getSeed() << "\n"
           << "   Left-hand side row-major dense tensor type:\n"
           << "     " << typeid( TT ).name() << "\n"
           << "   Right-hand side column-major dense tensor type:\n"
           << "     " << typeid( OMT ).name() << "\n";
       throw std::runtime_error( oss.str() );
    }
    catch( std::out_of_range& ) {}

    try {
       ( lhs_ % orhs_ ).at( 0UL, lhs_.rows(), 0UL );

       std::ostringstream oss;
       oss << " Test : Checked element access of schur expression\n"
           << " Error: Out-of-bound access succeeded\n"
           << " Details:\n"
           << "   Random seed = " << blaze::getSeed() << "\n"
           << "   Left-hand side row-major dense tensor type:\n"
           << "     " << typeid( TT ).name() << "\n"
           << "   Right-hand side column-major dense tensor type:\n"
           << "     " << typeid( OMT ).name() << "\n";
       throw std::runtime_error( oss.str() );
    }
    catch( std::out_of_range& ) {}

    try {
       ( lhs_ % orhs_ ).at( lhs_.pages(), 0UL, 0UL );

       std::ostringstream oss;
       oss << " Test : Checked element access of schur expression\n"
           << " Error: Out-of-bound access succeeded\n"
           << " Details:\n"
           << "   Random seed = " << blaze::getSeed() << "\n"
           << "   Left-hand side row-major dense tensor type:\n"
           << "     " << typeid( TT ).name() << "\n"
           << "   Right-hand side column-major dense tensor type:\n"
           << "     " << typeid( OMT ).name() << "\n";
       throw std::runtime_error( oss.str() );
    }
    catch( std::out_of_range& ) {}
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the plain dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the plain tensor schur with plain assignment, schur assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testBasicOperation()
{
#if BLAZETEST_MATHTEST_TEST_BASIC_OPERATION
   if( BLAZETEST_MATHTEST_TEST_BASIC_OPERATION > 1 )
   {
      //=====================================================================================
      // Schur product
      //=====================================================================================

      // Schur product with the given tensor and matrix
      {
         test_  = "Schur product with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            dres_   = lhs_ % rhs_;
            refres_ = reflhs_ % refrhs_;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Schur product with evaluated tensor and matrix
      {
         test_  = "Schur product with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            dres_   = eval( lhs_ ) % eval( rhs_ );
            refres_ = eval( reflhs_ ) % eval( refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Schur product with schur assignment
      //=====================================================================================

      // Schur product with schur assignment with the given tensor and matrix
      {
         test_  = "Schur product with addition assignment with the given tensor and matrix";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += lhs_ % rhs_;
            refres_ += reflhs_ % refrhs_;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Schur product with schur assignment with evaluated tensor and matrix
      {
         test_  = "Schur product with addition assignment with evaluated tensor and matrix";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += eval( lhs_ ) % eval( rhs_ );
            refres_ += eval( reflhs_ ) % eval( refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Schur product with subtraction assignment with the given tensor and matrix
      //=====================================================================================

      // Schur product with subtraction assignment with the given tensor and matrix
      {
         test_  = "Schur product with subtraction assignment with the given tensor and matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= lhs_ % rhs_;
            refres_ -= reflhs_ % refrhs_;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Schur product with subtraction assignment with evaluated tensor and matrix
      {
         test_  = "Schur product with subtraction assignment with evaluated tensor and matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= eval( lhs_ ) % eval( rhs_ );
            refres_ -= eval( reflhs_ ) % eval( refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //=====================================================================================
      // Schur product with Schur product assignment
      //=====================================================================================

      // Schur product with Schur product assignment with the given tensor and matrix
      {
         test_  = "Schur product with schur assignment with the given tensor and matrix";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            dres_   %= lhs_ % rhs_;
            refres_ %= reflhs_ % refrhs_;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Schur product with schur assignment with evaluated tensor and matrix
      {
         test_  = "Schur product with schur assignment with evaluated tensor and matrix";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            dres_   %= eval( lhs_ ) % eval( rhs_ );
            refres_ %= eval( reflhs_ ) % eval( refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the negated dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the negated tensor schur with plain assignment, schur assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testNegatedOperation()
{
#if BLAZETEST_MATHTEST_TEST_NEGATED_OPERATION
   if( BLAZETEST_MATHTEST_TEST_NEGATED_OPERATION > 1 )
   {
      //=====================================================================================
      // Negated schur product
      //=====================================================================================

      // Negated schur product with the given tensor and matrix
      {
         test_ = "Negated schur product with the given tensor and matrix";
         error_ = "Failed schur product operation";

         try {
            initResults();
            dres_ = -(lhs_ % rhs_);
            refres_ = -(reflhs_ % refrhs_);
         }
         catch (std::exception& ex) {
            convertException<TT, MT>(ex);
         }

         checkResults<TT, MT>();
      }

      // Negated schur with evaluated tensor and matrix
      {
         test_ = "Negated schur product with evaluated tensor and matrix";
         error_ = "Failed schur product operation";

         try {
            initResults();
            dres_ = -(eval(lhs_) % eval(rhs_));
            refres_ = -(eval(reflhs_) % eval(refrhs_));
         }
         catch (std::exception& ex) {
            convertException<TT, MT>(ex);
         }

         checkResults<TT, MT>();
      }

      //=====================================================================================
      // Negated schur with schur assignment
      //=====================================================================================

      // Negated schur product with schur assignment with the given tensor and matrix
      {
         test_ = "Negated schur product with addition assignment with the given tensor and matrix";
         error_ = "Failed schur product operation";

         try {
            initResults();
            dres_   += -(lhs_ % rhs_);
            refres_ += -(reflhs_ % refrhs_);
         }
         catch (std::exception& ex) {
            convertException<TT, MT>(ex);
         }

         checkResults<TT, MT>();
      }

      // Negated schur with evaluated tensor and matrix
      {
         test_ = "Negated schur product with addition assignment with evaluated tensor and matrix";
         error_ = "Failed schur product operation";

         try {
            initResults();
            dres_   += -(eval(lhs_) % eval(rhs_));
            refres_ += -(eval(reflhs_) % eval(refrhs_));
         }
         catch (std::exception& ex) {
            convertException<TT, MT>(ex);
         }

         checkResults<TT, MT>();
      }

      //=====================================================================================
      // Negated schur product with subtraction assignment
      //=====================================================================================

      // Negated schur product with the subtraction assignment with given tensor and matrix
      {
         test_ = "Negated schur product with subtraction assignment with the given tensor and matrix";
         error_ = "Failed schur product operation";

         try {
            initResults();
            dres_   -= -(lhs_ % rhs_);
            refres_ -= -(reflhs_ % refrhs_);
         }
         catch (std::exception& ex) {
            convertException<TT, MT>(ex);
         }

         checkResults<TT, MT>();
      }

      // Negated schur with evaluated tensor and matrix
      {
         test_ = "Negated schur product with subtraction assignment with evaluated tensor and matrix";
         error_ = "Failed schur product operation";

         try {
            initResults();
            dres_   -= -(eval(lhs_) % eval(rhs_));
            refres_ -= -(eval(reflhs_) % eval(refrhs_));
         }
         catch (std::exception& ex) {
            convertException<TT, MT>(ex);
         }

         checkResults<TT, MT>();
      }


      //=====================================================================================
      // Negated schur with Schur product assignment
      //=====================================================================================

      // Negated schur product with the subtraction assignment with given tensor and matrix
      {
         test_ = "Negated schur product with schur assignment with the given tensor and matrix";
         error_ = "Failed schur product operation";

         try {
            initResults();
            dres_   %= -(lhs_ % rhs_);
            refres_ %= -(reflhs_ % refrhs_);
         }
         catch (std::exception& ex) {
            convertException<TT, MT>(ex);
         }

         checkResults<TT, MT>();
      }

      // Negated schur with evaluated tensor and matrix
      {
         test_ = "Negated schur product with schur assignment with evaluated tensor and matrix";
         error_ = "Failed schur product operation";

         try {
            initResults();
            dres_   %= -(eval(lhs_) % eval(rhs_));
            refres_ %= -(eval(reflhs_) % eval(refrhs_));
         }
         catch (std::exception& ex) {
            convertException<TT, MT>(ex);
         }

         checkResults<TT, MT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the scaled dense tensor/dense tensor schur.
//
// \param scalar The scalar value.
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the scaled tensor schur with plain assignment, schur assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
template< typename T >    // Type of the scalar
void OperationTest<TT,MT>::testScaledOperation( T scalar )
{
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( T );

   if( scalar == T(0) )
      throw std::invalid_argument( "Invalid scalar parameter" );


#if BLAZETEST_MATHTEST_TEST_SCALED_OPERATION
   if( BLAZETEST_MATHTEST_TEST_SCALED_OPERATION > 1 )
   {
      //=====================================================================================
      // Self-scaling (M*=s)
      //=====================================================================================

      // Self-scaling (M*=s)
      {
         test_ = "Self-scaling (M*=s)";

         try {
            dres_   = lhs_ % rhs_;
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

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Self-scaling (M=M*s)
      //=====================================================================================

      // Self-scaling (M=M*s)
      {
         test_ = "Self-scaling (M=M*s)";

         try {
            dres_   = lhs_ % rhs_;
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

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Self-scaling (M=s*M)
      //=====================================================================================

      // Self-scaling (M=s*M)
      {
         test_ = "Self-scaling (M=s*M)";

         try {
            dres_   = lhs_ % rhs_;
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

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Self-scaling (M/=s)
      //=====================================================================================

      // Self-scaling (M/=s)
      {
         test_ = "Self-scaling (M/=s)";

         try {
            dres_   = lhs_ % rhs_;
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

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Self-scaling (M=M/s)
      //=====================================================================================

      // Self-scaling (M=M/s)
      {
         test_ = "Self-scaling (M=M/s)";

         try {
            dres_   = lhs_ % rhs_;
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

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Scaled schur (s*OP)
      //=====================================================================================

      // Scaled schur with the given tensor and matrix
      {
         test_  = "Scaled schur with the given tensor and matrix (s*OP)";
         error_ = "Failed schur operation";

         try {
            initResults();
            dres_   = scalar * ( lhs_ % rhs_ );
            refres_ = scalar * ( reflhs_ % refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Scaled schur with evaluated tensor and matrix
      {
         test_  = "Scaled schur with evaluated tensor and matrix (s*OP)";
         error_ = "Failed schur operation";

         try {
            initResults();
            dres_   = scalar * ( eval( lhs_ ) % eval( rhs_ ) );
            refres_ = scalar * ( eval( reflhs_ ) % eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Scaled schur (OP*s)
      //=====================================================================================

      // Scaled schur with the given tensor and matrix
      {
         test_  = "Scaled schur with the given tensor and matrix (OP*s)";
         error_ = "Failed schur operation";

         try {
            initResults();
            dres_   = ( lhs_ % rhs_ ) * scalar;
            refres_ = ( reflhs_ % refrhs_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Scaled schur with evaluated tensor and matrix
      {
         test_  = "Scaled schur with evaluated tensor and matrix (OP*s)";
         error_ = "Failed schur operation";

         try {
            initResults();
            dres_   = ( eval( lhs_ ) % eval( rhs_ ) ) * scalar;
            refres_ = ( eval( reflhs_ ) % eval( refrhs_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Scaled schur (OP/s)
      //=====================================================================================

      // Scaled schur with the given tensor and matrix
      {
         test_  = "Scaled schur with the given tensor and matrix (OP/s)";
         error_ = "Failed schur operation";

         try {
            initResults();
            dres_   = ( lhs_ % rhs_ ) / scalar;
            refres_ = ( reflhs_ % refrhs_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Scaled schur with evaluated tensor and matrix
      {
         test_  = "Scaled schur with evaluated tensor and matrix (OP/s)";
         error_ = "Failed schur operation";

         try {
            initResults();
            dres_   = ( eval( lhs_ ) % eval( rhs_ ) ) / scalar;
            refres_ = ( eval( reflhs_ ) % eval( refrhs_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Scaled schur with schur assignment (s*OP)
      //=====================================================================================

      // Scaled schur with schur assignment with the given tensor and matrix
      {
         test_  = "Scaled schur with addition assignment with the given tensor and matrix (s*OP)";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            dres_   += scalar * ( lhs_ % rhs_ );
            refres_ += scalar * ( reflhs_ % refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Scaled schur with schur assignment with evaluated tensor and matrix
      {
         test_  = "Scaled schur with addition assignment with evaluated tensor and matrix (s*OP)";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            dres_   += scalar * ( eval( lhs_ ) % eval( rhs_ ) );
            refres_ += scalar * ( eval( reflhs_ ) % eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Scaled schur with schur assignment (OP*s)
      //=====================================================================================

      // Scaled schur with schur assignment with the given tensor and matrix
      {
         test_  = "Scaled schur with addition assignment with the given tensor and matrix (OP*s)";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            dres_   += ( lhs_ % rhs_ ) * scalar;
            refres_ += ( reflhs_ % refrhs_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Scaled schur with schur assignment with evaluated tensor and matrix
      {
         test_  = "Scaled schur with addition assignment with evaluated tensor and matrix (OP*s)";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            dres_   += ( eval( lhs_ ) % eval( rhs_ ) ) * scalar;
            refres_ += ( eval( reflhs_ ) % eval( refrhs_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Scaled schur with schur assignment (OP/s)
      //=====================================================================================

      // Scaled schur with schur assignment with the given tensor and matrix
      {
         test_  = "Scaled schur with addition assignment with the given tensor and matrix (OP/s)";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            dres_   += ( lhs_ % rhs_ ) / scalar;
            refres_ += ( reflhs_ % refrhs_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Scaled schur with schur assignment with evaluated tensor and matrix
      {
         test_  = "Scaled schur with addition assignment with evaluated tensor and matrix (OP/s)";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            dres_   += ( eval( lhs_ ) % eval( rhs_ ) ) / scalar;
            refres_ += ( eval( reflhs_ ) % eval( refrhs_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Scaled schur with subtraction assignment (s*OP)
      //=====================================================================================

      // Scaled schur with subtraction assignment with the given tensor and matrix
      {
         test_  = "Scaled schur with subtraction assignment with the given tensor and matrix (s*OP)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= scalar * ( lhs_ % rhs_ );
            refres_ -= scalar * ( reflhs_ % refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Scaled schur with subtraction assignment with evaluated tensor and matrix
      {
         test_  = "Scaled schur with subtraction assignment with evaluated tensor and matrix (s*OP)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= scalar * ( eval( lhs_ ) % eval( rhs_ ) );
            refres_ -= scalar * ( eval( reflhs_ ) % eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Scaled schur with subtraction assignment (OP*s)
      //=====================================================================================

      // Scaled schur with subtraction assignment with the given tensor and matrix
      {
         test_  = "Scaled schur with subtraction assignment with the given tensor and matrix (OP*s)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ( lhs_ % rhs_ ) * scalar;
            refres_ -= ( reflhs_ % refrhs_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Scaled schur with subtraction assignment with evaluated tensor and matrix
      {
         test_  = "Scaled schur with subtraction assignment with evaluated tensor and matrix (OP*s)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ( eval( lhs_ ) % eval( rhs_ ) ) * scalar;
            refres_ -= ( eval( reflhs_ ) % eval( refrhs_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Scaled schur with subtraction assignment (OP/s)
      //=====================================================================================

      // Scaled schur with subtraction assignment with the given tensor and matrix
      {
         test_  = "Scaled schur with subtraction assignment with the given tensor and matrix (OP/s)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ( lhs_ % rhs_ ) / scalar;
            refres_ -= ( reflhs_ % refrhs_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Scaled schur with subtraction assignment with evaluated tensor and matrix
      {
         test_  = "Scaled schur with subtraction assignment with evaluated tensor and matrix (OP/s)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ( eval( lhs_ ) % eval( rhs_ ) ) / scalar;
            refres_ -= ( eval( reflhs_ ) % eval( refrhs_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Scaled schur with Schur product assignment (s*OP)
      //=====================================================================================

      // Scaled schur with Schur product assignment with the given tensor and matrix
      {
         test_  = "Scaled schur with Schur product assignment with the given tensor and matrix (s*OP)";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= scalar * ( lhs_ % rhs_ );
            refres_ %= scalar * ( reflhs_ % refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Scaled schur with Schur product assignment with evaluated tensor and matrix
      {
         test_  = "Scaled schur with Schur product assignment with evaluated tensor and matrix (s*OP)";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= scalar * ( eval( lhs_ ) % eval( rhs_ ) );
            refres_ %= scalar * ( eval( reflhs_ ) % eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Scaled schur with Schur product assignment (OP*s)
      //=====================================================================================

      // Scaled schur with Schur product assignment with the given tensor and matrix
      {
         test_  = "Scaled schur with Schur product assignment with the given tensor and matrix (OP*s)";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= ( lhs_ % rhs_ ) * scalar;
            refres_ %= ( reflhs_ % refrhs_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Scaled schur with Schur product assignment with evaluated tensor and matrix
      {
         test_  = "Scaled schur with Schur product assignment with evaluated tensor and matrix (OP*s)";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= ( eval( lhs_ ) % eval( rhs_ ) ) * scalar;
            refres_ %= ( eval( reflhs_ ) % eval( refrhs_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Scaled schur with Schur product assignment (OP/s)
      //=====================================================================================

      // Scaled schur with Schur product assignment with the given tensor and matrix
      {
         test_  = "Scaled schur with Schur product assignment with the given tensor and matrix (OP/s)";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= ( lhs_ % rhs_ ) / scalar;
            refres_ %= ( reflhs_ % refrhs_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // Scaled schur with Schur product assignment with evaluated tensor and matrix
      {
         test_  = "Scaled schur with Schur product assignment with evaluated tensor and matrix (OP/s)";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= ( eval( lhs_ ) % eval( rhs_ ) ) / scalar;
            refres_ %= ( eval( reflhs_ ) % eval( refrhs_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the transpose dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the transpose tensor schur with plain assignment. In case any error
// resulting from the schur or the subsequent assignment is detected, a \a std::runtime_error
// exception is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testTransOperation()
{
#if BLAZETEST_MATHTEST_TEST_TRANS_OPERATION
   if( BLAZETEST_MATHTEST_TEST_TRANS_OPERATION > 1 )
   {
      //=====================================================================================
      // Transpose schur
      //=====================================================================================

      // Transpose schur with the given tensor and matrix
      {
         test_  = "Transpose schur with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initTransposeResults();
            tdres_  = trans( lhs_ % rhs_ );
            refres_ = trans( reflhs_ % refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkTransposeResults<TT,MT>();
      }

      // Transpose schur with evaluated tensor and matrix
      {
         test_  = "Transpose schur with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initTransposeResults();
            tdres_  = trans( eval( lhs_ ) % eval( rhs_ ) );
            refres_ = trans( eval( reflhs_ ) % eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkTransposeResults<TT,MT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the conjugate transpose dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the conjugate transpose tensor schur with plain assignment. In
// case any error resulting from the schur or the subsequent assignment is detected, a
// \a std::runtime_error exception is thrown.
*/
 template< typename TT    // Type of the left-hand side dense tensor
         , typename MT >  // Type of the right-hand side dense tensor
    void OperationTest<TT, MT>::testCTransOperation()
{
    #if BLAZETEST_MATHTEST_TEST_CTRANS_OPERATION
       if( BLAZETEST_MATHTEST_TEST_CTRANS_OPERATION > 1 )
   {
       //=====================================================================================
       // Conjugate transpose schur
       //=====================================================================================

       // Conjugate transpose schur with the given tensor and matrix
       {
          test_ = "Conjugate transpose schur with the given tensor and matrix";
          error_ = "Failed schur operation";

          try {
             initTransposeResults();
             tdres_ = ctrans(lhs_ % rhs_);
             refres_ = ctrans(reflhs_ % refrhs_);
          }
          catch (std::exception& ex) {
             convertException<TT, MT>(ex);
          }

          checkTransposeResults<TT,MT>();
       }

       // Conjugate transpose schur with evaluated tensor and matrix
       {
          test_  = "Conjugate transpose schur with evaluated tensor and matrix";
          error_ = "Failed schur operation";

          try {
             initTransposeResults();
             tdres_  = ctrans( eval( lhs_ ) % eval( rhs_ ) );
             refres_ = ctrans( eval( reflhs_ ) % eval( refrhs_ ) );
          }
          catch( std::exception& ex ) {
             convertException<TT,MT>( ex );
          }

          checkTransposeResults<TT,MT>();
       }
    }
 #endif
 }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the abs dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the abs tensor schur with plain assignment, schur assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testAbsOperation()
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
/*!\brief Testing the conjugate dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the conjugate tensor schur with plain assignment, schur assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testConjOperation()
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
/*!\brief Testing the \a real dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the \a real tensor schur with plain assignment, schur assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testRealOperation()
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
/*!\brief Testing the \a imag dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the \a imag tensor schur with plain assignment, schur assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testImagOperation()
{
#if BLAZETEST_MATHTEST_TEST_IMAG_OPERATION
   if( BLAZETEST_MATHTEST_TEST_IMAG_OPERATION > 1
   {
      testCustomOperation( blaze::Imag(), "imag" );
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the \a inv dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the \a inv tensor schur with plain assignment, schur assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testInvOperation()
{
#if BLAZETEST_MATHTEST_TEST_INV_OPERATION && BLAZETEST_MATHTEST_LAPACK_MODE
   if( BLAZETEST_MATHTEST_TEST_INV_OPERATION > 1 )
   {
      if( !isSquare( lhs_ + rhs_ ) || blaze::isDefault( det( lhs_ + rhs_ ) ) )
         return;

      testCustomOperation( blaze::Inv(), "inv" );
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the evaluated dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the evaluated tensor schur with plain assignment, schur assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testEvalOperation()
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
/*!\brief Testing the serialized dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the serialized tensor schur with plain assignment, schur assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testSerialOperation()
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
/*!\brief Testing the subtensor-wise dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the subtensor-wise tensor schur with plain assignment, schur
// assignment, subtraction assignment, and Schur product assignment. In case any error resulting
// from the schur or the subsequent assignment is detected, a \a std::runtime_error exception
// is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testSubtensorOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_SUBTENSOR_OPERATION
   if( BLAZETEST_MATHTEST_TEST_SUBTENSOR_OPERATION > 1 )
   {
      if( lhs_.rows() == 0UL || lhs_.columns() == 0UL || lhs_.pages() == 0 )
         return;


      //=====================================================================================
      // Subtensor-wise schur
      //=====================================================================================

      // Subtensor-wise schur with the given tensor and matrix
      {
         test_  = "Subtensor-wise schur with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) = subtensor( lhs_ % rhs_      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) = subtensor( reflhs_ % refrhs_, page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //Subtensor-wise schur with evaluated tensor and matrix
      {
         test_  = "Subtensor-wise schur with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) = subtensor( eval( lhs_ ) % eval( rhs_ )      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) = subtensor( eval( reflhs_ ) % eval( refrhs_ ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // Subtensor-wise schur with addition assignment
      //=====================================================================================

      // Subtensor-wise schur with addition assignment with the given tensor and matrix
      {
         test_  = "Subtensor-wise schur with addition assignment with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) += subtensor( lhs_ % rhs_      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) += subtensor( reflhs_ % refrhs_, page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //Subtensor-wise schur with addition assignment with the evaluated tensor and matrix
      {
         test_  = "Subtensor-wise schur with addition assignment with the evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) += subtensor( eval( lhs_ ) % eval( rhs_ )      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) += subtensor( eval( reflhs_ ) % eval( refrhs_ ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //=====================================================================================
      // Subtensor-wise schur with subtraction assignment
      //=====================================================================================

      // Subtensor-wise schur with subtraction assignment with the given tensor and matrix
      {
         test_  = "Subtensor-wise schur with subtraction assignment with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( lhs_ % rhs_      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) -= subtensor( reflhs_ % refrhs_, page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //Subtensor-wise schur with subtraction assignment with the evaluated tensor and matrix
      {
         test_  = "Subtensor-wise schur with subtraction assignment with the evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( eval( lhs_ ) % eval( rhs_ )      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) -= subtensor( eval( reflhs_ ) % eval( refrhs_ ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //=====================================================================================
      // Subtensor-wise schur with Schur product assignment
      //=====================================================================================

      // Subtensor-wise schur with Schur product assignment with the given tensor and matrix
      {
         test_  = "Subtensor-wise schur with schur assignment with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( lhs_ % rhs_      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) %= subtensor( reflhs_ % refrhs_, page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //Subtensor-wise schur with subtraction assignment with the evaluated tensor and matrix
      {
         test_  = "Subtensor-wise schur with schur assignment with the evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( eval( lhs_ ) % eval( rhs_ )      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) %= subtensor( eval( reflhs_ ) % eval( refrhs_ ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the subtensor-wise dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function is called in case the submatrix-wise tensor/tensor schur operation is not
// available for the given tensor types \a TT and \a MT.
*/
template< typename TT    // Type of the left-hand side dense matrix
        , typename MT >  // Type of the right-hand side dense matrix
void OperationTest<TT,MT>::testSubtensorOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the row-wise dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the row-wise tensor schur with plain assignment, schur assignment,
// subtraction assignment, and multiplication assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testRowSliceOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_ROWSLICE_OPERATION
   if( BLAZETEST_MATHTEST_TEST_ROWSLICE_OPERATION > 1 )
   {
      if( lhs_.rows() == 0UL )
         return;


      //=====================================================================================
      // RowSlice-wise schur
      //=====================================================================================

      // RowSlice-wise schur with the given tensor and matrix
      {
         test_  = "RowSlice-wise schur with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) = rowslice( lhs_ % rhs_, i );
               rowslice( refres_, i ) = rowslice( reflhs_ % refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // RowSlice-wise schur with evaluated tensor and matrix
      {
         test_  = "RowSlice-wise schur with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) = rowslice( eval( lhs_ ) % eval( rhs_ ), i );
               rowslice( refres_, i ) = rowslice( eval( reflhs_ ) % eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // RowSlice-wise schur with addition assignment
      //=====================================================================================

      // RowSlice-wise schur with addition assignment with the given tensor and matrix
      {
         test_  = "RowSlice-wise schur with addition assignment with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) += rowslice( lhs_ % rhs_, i );
               rowslice( refres_, i ) += rowslice( reflhs_ % refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // RowSlice-wise schur with evaluated tensor and matrix
      {
         test_  = "RowSlice-wise schur with addition assignment with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) += rowslice( eval( lhs_ ) % eval( rhs_ ), i );
               rowslice( refres_, i ) += rowslice( eval( reflhs_ ) % eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //=====================================================================================
      // RowSlice-wise schur with subtraction assignment
      //=====================================================================================

      // RowSlice-wise schur with subtraction assignment with the given tensor and matrix
      {
         test_  = "RowSlice-wise schur with subtraction assignment with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) -= rowslice( lhs_ % rhs_, i );
               rowslice( refres_, i ) -= rowslice( reflhs_ % refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // RowSlice-wise schur with evaluated tensor and matrix
      {
         test_  = "RowSlice-wise schur with subtraction assignment with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) -= rowslice( eval( lhs_ ) % eval( rhs_ ), i );
               rowslice( refres_, i ) -= rowslice( eval( reflhs_ ) % eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //=====================================================================================
      // RowSlice-wise schur with schur assignment
      //=====================================================================================

      // RowSlice-wise schur with schur assignment with the given tensor and matrix
      {
         test_  = "RowSlice-wise schur with schur assignment with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) %= rowslice( lhs_ % rhs_, i );
               rowslice( refres_, i ) %= rowslice( reflhs_ % refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // RowSlice-wise schur with evaluated tensor and matrix
      {
         test_  = "RowSlice-wise schur with schur assignment with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) %= rowslice( eval( lhs_ ) % eval( rhs_ ), i );
               rowslice( refres_, i ) %= rowslice( eval( reflhs_ ) % eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the rowslice-wise dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function is called in case the rowslice-wise tensor/tensor schur operation is not
// available for the given matrix types \a TT and \a MT.
*/
template< typename TT    // Type of the left-hand side dense matrix
        , typename MT >  // Type of the right-hand side dense matrix
void OperationTest<TT,MT>::testRowSliceOperation( blaze::FalseType )
{}
//*************************************************************************************************


#if 0
//*************************************************************************************************
/*!\brief Testing the rows-wise dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the rows-wise tensor schur with plain assignment, schur assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
// template< typename TT    // Type of the left-hand side dense tensor
//         , typename MT >  // Type of the right-hand side dense tensor
// void OperationTest<TT,MT>::testRowSlicesOperation( blaze::TrueType )
// {
// #if BLAZETEST_MATHTEST_TEST_ROWS_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_ROWS_OPERATION > 1 )
//    {
//       if( lhs_.rows() == 0UL )
//          return;
//
//
//       std::vector<size_t> indices( lhs_.rows() );
//       std::iota( indices.begin(), indices.end(), 0UL );
//       std::random_shuffle( indices.begin(), indices.end() );
//
//
//       //=====================================================================================
//       // Rows-wise schur
//       //=====================================================================================
//
//       // Rows-wise schur with the given tensor and matrix
//       {
//          test_  = "Rows-wise schur with the given tensor and matrix";
//          error_ = "Failed schur operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) = rows( lhs_ + rhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) = rows( lhs_ + rhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) = rows( lhs_ + rhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) = rows( lhs_ + rhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) = rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,MT>( ex );
//          }
//
//          checkResults<TT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) = rows( lhs_ + orhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) = rows( lhs_ + orhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) = rows( lhs_ + orhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) = rows( lhs_ + orhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) = rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,OMT>( ex );
//          }
//
//          checkResults<TT,OMT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) = rows( olhs_ + rhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) = rows( olhs_ + rhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) = rows( olhs_ + rhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) = rows( olhs_ + rhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) = rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,MT>( ex );
//          }
//
//          checkResults<OTT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) = rows( olhs_ + orhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) = rows( olhs_ + orhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) = rows( olhs_ + orhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) = rows( olhs_ + orhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) = rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,OMT>( ex );
//          }
//
//          checkResults<OTT,OMT>();
//       }
//
//       // Rows-wise schur with evaluated tensor and matrix
//       {
//          test_  = "Rows-wise schur with evaluated tensor and matrix";
//          error_ = "Failed schur operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) = rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) = rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) = rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) = rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) = rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,MT>( ex );
//          }
//
//          checkResults<TT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) = rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) = rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) = rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) = rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) = rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,OMT>( ex );
//          }
//
//          checkResults<TT,OMT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) = rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) = rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) = rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) = rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) = rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,MT>( ex );
//          }
//
//          checkResults<OTT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) = rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) = rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) = rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) = rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) = rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,OMT>( ex );
//          }
//
//          checkResults<OTT,OMT>();
//       }
//
//
//       //=====================================================================================
//       // Rows-wise schur with schur assignment
//       //=====================================================================================
//
//       // Rows-wise schur with schur assignment with the given tensor and matrix
//       {
//          test_  = "Rows-wise schur with schur assignment with the given tensor and matrix";
//          error_ = "Failed schur assignment operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) += rows( lhs_ + rhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) += rows( lhs_ + rhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) += rows( lhs_ + rhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) += rows( lhs_ + rhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) += rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,MT>( ex );
//          }
//
//          checkResults<TT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) += rows( lhs_ + orhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) += rows( lhs_ + orhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) += rows( lhs_ + orhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) += rows( lhs_ + orhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) += rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,OMT>( ex );
//          }
//
//          checkResults<TT,OMT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) += rows( olhs_ + rhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) += rows( olhs_ + rhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) += rows( olhs_ + rhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) += rows( olhs_ + rhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) += rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,MT>( ex );
//          }
//
//          checkResults<OTT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) += rows( olhs_ + orhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) += rows( olhs_ + orhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) += rows( olhs_ + orhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) += rows( olhs_ + orhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) += rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,OMT>( ex );
//          }
//
//          checkResults<OTT,OMT>();
//       }
//
//       // Rows-wise schur with schur assignment with evaluated tensor and matrix
//       {
//          test_  = "Rows-wise schur with schur assignment with evaluated tensor and matrix";
//          error_ = "Failed schur assignment operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) += rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) += rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) += rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) += rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) += rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,MT>( ex );
//          }
//
//          checkResults<TT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) += rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) += rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) += rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) += rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) += rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,OMT>( ex );
//          }
//
//          checkResults<TT,OMT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) += rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) += rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) += rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) += rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) += rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,MT>( ex );
//          }
//
//          checkResults<OTT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) += rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) += rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) += rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) += rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) += rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,OMT>( ex );
//          }
//
//          checkResults<OTT,OMT>();
//       }
//
//
//       //=====================================================================================
//       // Rows-wise schur with subtraction assignment
//       //=====================================================================================
//
//       // Rows-wise schur with subtraction assignment with the given tensor and matrix
//       {
//          test_  = "Rows-wise schur with subtraction assignment with the given tensor and matrix";
//          error_ = "Failed subtraction assignment operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) -= rows( lhs_ + rhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) -= rows( lhs_ + rhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) -= rows( lhs_ + rhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) -= rows( lhs_ + rhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) -= rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,MT>( ex );
//          }
//
//          checkResults<TT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) -= rows( lhs_ + orhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) -= rows( lhs_ + orhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) -= rows( lhs_ + orhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) -= rows( lhs_ + orhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) -= rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,OMT>( ex );
//          }
//
//          checkResults<TT,OMT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) -= rows( olhs_ + rhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) -= rows( olhs_ + rhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) -= rows( olhs_ + rhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) -= rows( olhs_ + rhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) -= rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,MT>( ex );
//          }
//
//          checkResults<OTT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) -= rows( olhs_ + orhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) -= rows( olhs_ + orhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) -= rows( olhs_ + orhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) -= rows( olhs_ + orhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) -= rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,OMT>( ex );
//          }
//
//          checkResults<OTT,OMT>();
//       }
//
//       // Rows-wise schur with subtraction assignment with evaluated tensor and matrix
//       {
//          test_  = "Rows-wise schur with subtraction assignment with evaluated tensor and matrix";
//          error_ = "Failed subtraction assignment operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) -= rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) -= rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) -= rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) -= rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) -= rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,MT>( ex );
//          }
//
//          checkResults<TT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) -= rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) -= rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) -= rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) -= rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) -= rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,OMT>( ex );
//          }
//
//          checkResults<TT,OMT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) -= rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) -= rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) -= rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) -= rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) -= rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,MT>( ex );
//          }
//
//          checkResults<OTT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) -= rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) -= rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) -= rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) -= rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) -= rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,OMT>( ex );
//          }
//
//          checkResults<OTT,OMT>();
//       }
//
//
//       //=====================================================================================
//       // Rows-wise schur with Schur product assignment
//       //=====================================================================================
//
//       // Rows-wise schur with Schur product assignment with the given tensor and matrix
//       {
//          test_  = "Rows-wise schur with Schur product assignment with the given tensor and matrix";
//          error_ = "Failed Schur product assignment operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) %= rows( lhs_ + rhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) %= rows( lhs_ + rhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) %= rows( lhs_ + rhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) %= rows( lhs_ + rhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) %= rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,MT>( ex );
//          }
//
//          checkResults<TT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) %= rows( lhs_ + orhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) %= rows( lhs_ + orhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) %= rows( lhs_ + orhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) %= rows( lhs_ + orhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) %= rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,OMT>( ex );
//          }
//
//          checkResults<TT,OMT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) %= rows( olhs_ + rhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) %= rows( olhs_ + rhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) %= rows( olhs_ + rhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) %= rows( olhs_ + rhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) %= rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,MT>( ex );
//          }
//
//          checkResults<OTT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) %= rows( olhs_ + orhs_, &indices[index], n );
//                rows( odres_ , &indices[index], n ) %= rows( olhs_ + orhs_, &indices[index], n );
//                rows( sres_  , &indices[index], n ) %= rows( olhs_ + orhs_, &indices[index], n );
//                rows( osres_ , &indices[index], n ) %= rows( olhs_ + orhs_, &indices[index], n );
//                rows( refres_, &indices[index], n ) %= rows( reflhs_ + refrhs_, &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,OMT>( ex );
//          }
//
//          checkResults<OTT,OMT>();
//       }
//
//       // Rows-wise schur with Schur product assignment with evaluated tensor and matrix
//       {
//          test_  = "Rows-wise schur with Schur product assignment with evaluated tensor and matrix";
//          error_ = "Failed Schur product assignment operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) %= rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) %= rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) %= rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) %= rows( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) %= rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,MT>( ex );
//          }
//
//          checkResults<TT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) %= rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) %= rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) %= rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) %= rows( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) %= rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TT,OMT>( ex );
//          }
//
//          checkResults<TT,OMT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) %= rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) %= rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) %= rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) %= rows( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) %= rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,MT>( ex );
//          }
//
//          checkResults<OTT,MT>();
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) %= rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) %= rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) %= rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) %= rows( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) %= rows( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT,OMT>( ex );
//          }
//
//          checkResults<OTT,OMT>();
//       }
//    }
// #endif
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the rows-wise dense tensor/dense tensor schur.
//
// \return void
//
// This function is called in case the rows-wise tensor/tensor schur operation is not
// available for the given tensor types \a TT and \a MT.
*/
// template< typename TT    // Type of the left-hand side dense tensor
//         , typename MT >  // Type of the right-hand side dense tensor
// void OperationTest<TT,MT>::testRowSlicesOperation( blaze::FalseType )
// {}
//*************************************************************************************************
#endif


//*************************************************************************************************
/*!\brief Testing the row-wise dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the row-wise tensor schur with plain assignment, schur assignment,
// subtraction assignment, and multiplication assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testColumnSliceOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_COLUMNSLICE_OPERATION
   if( BLAZETEST_MATHTEST_TEST_COLUMNSLICE_OPERATION > 1 )
   {
      if( lhs_.columns() == 0UL )
         return;


      //=====================================================================================
      // ColumnSlice-wise schur
      //=====================================================================================

      // ColumnSlice-wise schur with the given tensor and matrix
      {
         test_  = "ColumnSlice-wise schur with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) = columnslice( lhs_ % rhs_, i );
               columnslice( refres_, i ) = columnslice( reflhs_ % refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // ColumnSlice-wise schur with evaluated tensor and matrix
      {
         test_  = "ColumnSlice-wise schur with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) = columnslice( eval( lhs_ ) % eval( rhs_ ), i );
               columnslice( refres_, i ) = columnslice( eval( reflhs_ ) % eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // ColumnSlice-wise schur with addition assignment
      //=====================================================================================

      // ColumnSlice-wise schur with addition assignment with the given tensor and matrix
      {
         test_  = "ColumnSlice-wise schur with addition assignment with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) += columnslice( lhs_ % rhs_, i );
               columnslice( refres_, i ) += columnslice( reflhs_ % refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // ColumnSlice-wise schur with evaluated tensor and matrix
      {
         test_  = "ColumnSlice-wise schur with addition assignment with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) += columnslice( eval( lhs_ ) % eval( rhs_ ), i );
               columnslice( refres_, i ) += columnslice( eval( reflhs_ ) % eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //=====================================================================================
      // ColumnSlice-wise schur with subtraction assignment
      //=====================================================================================

      // ColumnSlice-wise schur with subtraction assignment with the given tensor and matrix
      {
         test_  = "ColumnSlice-wise schur with subtraction assignment with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) -= columnslice( lhs_ % rhs_, i );
               columnslice( refres_, i ) -= columnslice( reflhs_ % refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // ColumnSlice-wise schur with evaluated tensor and matrix
      {
         test_  = "ColumnSlice-wise schur with subtraction assignment with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) -= columnslice( eval( lhs_ ) % eval( rhs_ ), i );
               columnslice( refres_, i ) -= columnslice( eval( reflhs_ ) % eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //=====================================================================================
      // ColumnSlice-wise schur with schur assignment
      //=====================================================================================

      // ColumnSlice-wise schur with schur assignment with the given tensor and matrix
      {
         test_  = "ColumnSlice-wise schur with schur assignment with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) %= columnslice( lhs_ % rhs_, i );
               columnslice( refres_, i ) %= columnslice( reflhs_ % refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // ColumnSlice-wise schur with evaluated tensor and matrix
      {
         test_  = "ColumnSlice-wise schur with schur assignment with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) %= columnslice( eval( lhs_ ) % eval( rhs_ ), i );
               columnslice( refres_, i ) %= columnslice( eval( reflhs_ ) % eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the columnslice-wise dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function is called in case the columnslice-wise tensor/tensor schur operation is not
// available for the given matrix types \a TT and \a MT.
*/
template< typename TT    // Type of the left-hand side dense matrix
        , typename MT >  // Type of the right-hand side dense matrix
void OperationTest<TT,MT>::testColumnSliceOperation( blaze::FalseType )
{}
//*************************************************************************************************



//*************************************************************************************************
/*!\brief Testing the row-wise dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the row-wise tensor schur with plain assignment, schur assignment,
// subtraction assignment, and multiplication assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testPageSliceOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_PAGESLICE_OPERATION
   if( BLAZETEST_MATHTEST_TEST_PAGESLICE_OPERATION > 1 )
   {
      if( lhs_.pages() == 0UL )
         return;

      //=====================================================================================
      // PageSlice-wise schur
      //=====================================================================================

      // PageSlice-wise schur with the given tensor and matrix
      {
         test_  = "PageSlice-wise schur with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) = pageslice( lhs_ % rhs_, i );
               pageslice( refres_, i ) = pageslice( reflhs_ % refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // PageSlice-wise schur with evaluated tensor and matrix
      {
         test_  = "PageSlice-wise schur with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) = pageslice( eval( lhs_ ) % eval( rhs_ ), i );
               pageslice( refres_, i ) = pageslice( eval( reflhs_ ) % eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }


      //=====================================================================================
      // PageSlice-wise schur with addition assignment
      //=====================================================================================

      // PageSlice-wise schur with addition assignment with the given tensor and matrix
      {
         test_  = "PageSlice-wise schur with addition assignment with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) += pageslice( lhs_ % rhs_, i );
               pageslice( refres_, i ) += pageslice( reflhs_ % refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // PageSlice-wise schur with evaluated tensor and matrix
      {
         test_  = "PageSlice-wise schur with addition assignment with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) += pageslice( eval( lhs_ ) % eval( rhs_ ), i );
               pageslice( refres_, i ) += pageslice( eval( reflhs_ ) % eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //=====================================================================================
      // PageSlice-wise schur with subtraction assignment
      //=====================================================================================

      // PageSlice-wise schur with subtraction assignment with the given tensor and matrix
      {
         test_  = "PageSlice-wise schur with subtraction assignment with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) -= pageslice( lhs_ % rhs_, i );
               pageslice( refres_, i ) -= pageslice( reflhs_ % refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // PageSlice-wise schur with evaluated tensor and matrix
      {
         test_  = "PageSlice-wise schur with subtraction assignment with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) -= pageslice( eval( lhs_ ) % eval( rhs_ ), i );
               pageslice( refres_, i ) -= pageslice( eval( reflhs_ ) % eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      //=====================================================================================
      // PageSlice-wise schur with schur assignment
      //=====================================================================================

      // PageSlice-wise schur with schur assignment with the given tensor and matrix
      {
         test_  = "PageSlice-wise schur with schur assignment with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) %= pageslice( lhs_ % rhs_, i );
               pageslice( refres_, i ) %= pageslice( reflhs_ % refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }

      // PageSlice-wise schur with evaluated tensor and matrix
      {
         test_  = "PageSlice-wise schur with schur assignment with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) %= pageslice( eval( lhs_ ) % eval( rhs_ ), i );
               pageslice( refres_, i ) %= pageslice( eval( reflhs_ ) % eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();
      }
   }
#endif
}//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the pageslice-wise dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function is called in case the pageslice-wise tensor/tensor schur operation is not
// available for the given matrix types \a TT and \a MT.
*/
template< typename TT    // Type of the left-hand side dense matrix
        , typename MT >  // Type of the right-hand side dense matrix
void OperationTest<TT,MT>::testPageSliceOperation( blaze::FalseType )
{}
//*************************************************************************************************


#if 0
//*************************************************************************************************
/*!\brief Testing the column-wise dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the column-wise tensor schur with plain assignment, schur assignment,
// subtraction assignment, and multiplication assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testColumnOperation()
{
#if BLAZETEST_MATHTEST_TEST_COLUMN_OPERATION
   if( BLAZETEST_MATHTEST_TEST_COLUMN_OPERATION > 1 )
   {
      if( lhs_.columns() == 0UL )
         return;


      //=====================================================================================
      // Column-wise schur
      //=====================================================================================

      // Column-wise schur with the given tensor and matrix
      {
         test_  = "Column-wise schur with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) = column( lhs_ + rhs_, j );
               column( odres_ , j ) = column( lhs_ + rhs_, j );
               column( sres_  , j ) = column( lhs_ + rhs_, j );
               column( osres_ , j ) = column( lhs_ + rhs_, j );
               column( refres_, j ) = column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) = column( lhs_ + orhs_, j );
               column( odres_ , j ) = column( lhs_ + orhs_, j );
               column( sres_  , j ) = column( lhs_ + orhs_, j );
               column( osres_ , j ) = column( lhs_ + orhs_, j );
               column( refres_, j ) = column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) = column( olhs_ + rhs_, j );
               column( odres_ , j ) = column( olhs_ + rhs_, j );
               column( sres_  , j ) = column( olhs_ + rhs_, j );
               column( osres_ , j ) = column( olhs_ + rhs_, j );
               column( refres_, j ) = column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) = column( olhs_ + orhs_, j );
               column( odres_ , j ) = column( olhs_ + orhs_, j );
               column( sres_  , j ) = column( olhs_ + orhs_, j );
               column( osres_ , j ) = column( olhs_ + orhs_, j );
               column( refres_, j ) = column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }

      // Column-wise schur with evaluated tensor and matrix
      {
         test_  = "Column-wise schur with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) = column( eval( lhs_ ) + eval( rhs_ ), j );
               column( odres_ , j ) = column( eval( lhs_ ) + eval( rhs_ ), j );
               column( sres_  , j ) = column( eval( lhs_ ) + eval( rhs_ ), j );
               column( osres_ , j ) = column( eval( lhs_ ) + eval( rhs_ ), j );
               column( refres_, j ) = column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) = column( eval( lhs_ ) + eval( orhs_ ), j );
               column( odres_ , j ) = column( eval( lhs_ ) + eval( orhs_ ), j );
               column( sres_  , j ) = column( eval( lhs_ ) + eval( orhs_ ), j );
               column( osres_ , j ) = column( eval( lhs_ ) + eval( orhs_ ), j );
               column( refres_, j ) = column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) = column( eval( olhs_ ) + eval( rhs_ ), j );
               column( odres_ , j ) = column( eval( olhs_ ) + eval( rhs_ ), j );
               column( sres_  , j ) = column( eval( olhs_ ) + eval( rhs_ ), j );
               column( osres_ , j ) = column( eval( olhs_ ) + eval( rhs_ ), j );
               column( refres_, j ) = column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) = column( eval( olhs_ ) + eval( orhs_ ), j );
               column( odres_ , j ) = column( eval( olhs_ ) + eval( orhs_ ), j );
               column( sres_  , j ) = column( eval( olhs_ ) + eval( orhs_ ), j );
               column( osres_ , j ) = column( eval( olhs_ ) + eval( orhs_ ), j );
               column( refres_, j ) = column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }


      //=====================================================================================
      // Column-wise schur with schur assignment
      //=====================================================================================

      // Column-wise schur with schur assignment with the given tensor and matrix
      {
         test_  = "Column-wise schur with schur assignment with the given tensor and matrix";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) += column( lhs_ + rhs_, j );
               column( odres_ , j ) += column( lhs_ + rhs_, j );
               column( sres_  , j ) += column( lhs_ + rhs_, j );
               column( osres_ , j ) += column( lhs_ + rhs_, j );
               column( refres_, j ) += column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) += column( lhs_ + orhs_, j );
               column( odres_ , j ) += column( lhs_ + orhs_, j );
               column( sres_  , j ) += column( lhs_ + orhs_, j );
               column( osres_ , j ) += column( lhs_ + orhs_, j );
               column( refres_, j ) += column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) += column( olhs_ + rhs_, j );
               column( odres_ , j ) += column( olhs_ + rhs_, j );
               column( sres_  , j ) += column( olhs_ + rhs_, j );
               column( osres_ , j ) += column( olhs_ + rhs_, j );
               column( refres_, j ) += column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) += column( olhs_ + orhs_, j );
               column( odres_ , j ) += column( olhs_ + orhs_, j );
               column( sres_  , j ) += column( olhs_ + orhs_, j );
               column( osres_ , j ) += column( olhs_ + orhs_, j );
               column( refres_, j ) += column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }

      // Column-wise schur with schur assignment with evaluated tensor and matrix
      {
         test_  = "Column-wise schur with schur assignment with evaluated tensor and matrix";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) += column( eval( lhs_ ) + eval( rhs_ ), j );
               column( odres_ , j ) += column( eval( lhs_ ) + eval( rhs_ ), j );
               column( sres_  , j ) += column( eval( lhs_ ) + eval( rhs_ ), j );
               column( osres_ , j ) += column( eval( lhs_ ) + eval( rhs_ ), j );
               column( refres_, j ) += column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) += column( eval( lhs_ ) + eval( orhs_ ), j );
               column( odres_ , j ) += column( eval( lhs_ ) + eval( orhs_ ), j );
               column( sres_  , j ) += column( eval( lhs_ ) + eval( orhs_ ), j );
               column( osres_ , j ) += column( eval( lhs_ ) + eval( orhs_ ), j );
               column( refres_, j ) += column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) += column( eval( olhs_ ) + eval( rhs_ ), j );
               column( odres_ , j ) += column( eval( olhs_ ) + eval( rhs_ ), j );
               column( sres_  , j ) += column( eval( olhs_ ) + eval( rhs_ ), j );
               column( osres_ , j ) += column( eval( olhs_ ) + eval( rhs_ ), j );
               column( refres_, j ) += column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) += column( eval( olhs_ ) + eval( orhs_ ), j );
               column( odres_ , j ) += column( eval( olhs_ ) + eval( orhs_ ), j );
               column( sres_  , j ) += column( eval( olhs_ ) + eval( orhs_ ), j );
               column( osres_ , j ) += column( eval( olhs_ ) + eval( orhs_ ), j );
               column( refres_, j ) += column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }


      //=====================================================================================
      // Column-wise schur with subtraction assignment
      //=====================================================================================

      // Column-wise schur with subtraction assignment with the given tensor and matrix
      {
         test_  = "Column-wise schur with subtraction assignment with the given tensor and matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) -= column( lhs_ + rhs_, j );
               column( odres_ , j ) -= column( lhs_ + rhs_, j );
               column( sres_  , j ) -= column( lhs_ + rhs_, j );
               column( osres_ , j ) -= column( lhs_ + rhs_, j );
               column( refres_, j ) -= column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) -= column( lhs_ + orhs_, j );
               column( odres_ , j ) -= column( lhs_ + orhs_, j );
               column( sres_  , j ) -= column( lhs_ + orhs_, j );
               column( osres_ , j ) -= column( lhs_ + orhs_, j );
               column( refres_, j ) -= column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) -= column( olhs_ + rhs_, j );
               column( odres_ , j ) -= column( olhs_ + rhs_, j );
               column( sres_  , j ) -= column( olhs_ + rhs_, j );
               column( osres_ , j ) -= column( olhs_ + rhs_, j );
               column( refres_, j ) -= column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) -= column( olhs_ + orhs_, j );
               column( odres_ , j ) -= column( olhs_ + orhs_, j );
               column( sres_  , j ) -= column( olhs_ + orhs_, j );
               column( osres_ , j ) -= column( olhs_ + orhs_, j );
               column( refres_, j ) -= column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }

      // Column-wise schur with subtraction assignment with evaluated tensor and matrix
      {
         test_  = "Column-wise schur with subtraction assignment with evaluated tensor and matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) -= column( eval( lhs_ ) + eval( rhs_ ), j );
               column( odres_ , j ) -= column( eval( lhs_ ) + eval( rhs_ ), j );
               column( sres_  , j ) -= column( eval( lhs_ ) + eval( rhs_ ), j );
               column( osres_ , j ) -= column( eval( lhs_ ) + eval( rhs_ ), j );
               column( refres_, j ) -= column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) -= column( eval( lhs_ ) + eval( orhs_ ), j );
               column( odres_ , j ) -= column( eval( lhs_ ) + eval( orhs_ ), j );
               column( sres_  , j ) -= column( eval( lhs_ ) + eval( orhs_ ), j );
               column( osres_ , j ) -= column( eval( lhs_ ) + eval( orhs_ ), j );
               column( refres_, j ) -= column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) -= column( eval( olhs_ ) + eval( rhs_ ), j );
               column( odres_ , j ) -= column( eval( olhs_ ) + eval( rhs_ ), j );
               column( sres_  , j ) -= column( eval( olhs_ ) + eval( rhs_ ), j );
               column( osres_ , j ) -= column( eval( olhs_ ) + eval( rhs_ ), j );
               column( refres_, j ) -= column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) -= column( eval( olhs_ ) + eval( orhs_ ), j );
               column( odres_ , j ) -= column( eval( olhs_ ) + eval( orhs_ ), j );
               column( sres_  , j ) -= column( eval( olhs_ ) + eval( orhs_ ), j );
               column( osres_ , j ) -= column( eval( olhs_ ) + eval( orhs_ ), j );
               column( refres_, j ) -= column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }


      //=====================================================================================
      // Column-wise schur with multiplication assignment
      //=====================================================================================

      // Column-wise schur with multiplication assignment with the given tensor and matrix
      {
         test_  = "Column-wise schur with multiplication assignment with the given tensor and matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) *= column( lhs_ + rhs_, j );
               column( odres_ , j ) *= column( lhs_ + rhs_, j );
               column( sres_  , j ) *= column( lhs_ + rhs_, j );
               column( osres_ , j ) *= column( lhs_ + rhs_, j );
               column( refres_, j ) *= column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) *= column( lhs_ + orhs_, j );
               column( odres_ , j ) *= column( lhs_ + orhs_, j );
               column( sres_  , j ) *= column( lhs_ + orhs_, j );
               column( osres_ , j ) *= column( lhs_ + orhs_, j );
               column( refres_, j ) *= column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) *= column( olhs_ + rhs_, j );
               column( odres_ , j ) *= column( olhs_ + rhs_, j );
               column( sres_  , j ) *= column( olhs_ + rhs_, j );
               column( osres_ , j ) *= column( olhs_ + rhs_, j );
               column( refres_, j ) *= column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) *= column( olhs_ + orhs_, j );
               column( odres_ , j ) *= column( olhs_ + orhs_, j );
               column( sres_  , j ) *= column( olhs_ + orhs_, j );
               column( osres_ , j ) *= column( olhs_ + orhs_, j );
               column( refres_, j ) *= column( reflhs_ + refrhs_, j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }

      // Column-wise schur with multiplication assignment with evaluated tensor and matrix
      {
         test_  = "Column-wise schur with multiplication assignment with evaluated tensor and matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) *= column( eval( lhs_ ) + eval( rhs_ ), j );
               column( odres_ , j ) *= column( eval( lhs_ ) + eval( rhs_ ), j );
               column( sres_  , j ) *= column( eval( lhs_ ) + eval( rhs_ ), j );
               column( osres_ , j ) *= column( eval( lhs_ ) + eval( rhs_ ), j );
               column( refres_, j ) *= column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) *= column( eval( lhs_ ) + eval( orhs_ ), j );
               column( odres_ , j ) *= column( eval( lhs_ ) + eval( orhs_ ), j );
               column( sres_  , j ) *= column( eval( lhs_ ) + eval( orhs_ ), j );
               column( osres_ , j ) *= column( eval( lhs_ ) + eval( orhs_ ), j );
               column( refres_, j ) *= column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) *= column( eval( olhs_ ) + eval( rhs_ ), j );
               column( odres_ , j ) *= column( eval( olhs_ ) + eval( rhs_ ), j );
               column( sres_  , j ) *= column( eval( olhs_ ) + eval( rhs_ ), j );
               column( osres_ , j ) *= column( eval( olhs_ ) + eval( rhs_ ), j );
               column( refres_, j ) *= column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t j=0UL; j<lhs_.columns(); ++j ) {
               column( dres_  , j ) *= column( eval( olhs_ ) + eval( orhs_ ), j );
               column( odres_ , j ) *= column( eval( olhs_ ) + eval( orhs_ ), j );
               column( sres_  , j ) *= column( eval( olhs_ ) + eval( orhs_ ), j );
               column( osres_ , j ) *= column( eval( olhs_ ) + eval( orhs_ ), j );
               column( refres_, j ) *= column( eval( reflhs_ ) + eval( refrhs_ ), j );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the columns-wise dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the columns-wise tensor schur with plain assignment, schur assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testColumnsOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_COLUMNS_OPERATION
   if( BLAZETEST_MATHTEST_TEST_COLUMNS_OPERATION > 1 )
   {
      if( lhs_.columns() == 0UL )
         return;


      std::vector<size_t> indices( lhs_.columns() );
      std::iota( indices.begin(), indices.end(), 0UL );
      std::random_shuffle( indices.begin(), indices.end() );


      //=====================================================================================
      // Columns-wise schur
      //=====================================================================================

      // Columns-wise schur with the given tensor and matrix
      {
         test_  = "Columns-wise schur with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) = columns( lhs_ + rhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) = columns( lhs_ + rhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) = columns( lhs_ + rhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) = columns( lhs_ + rhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) = columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) = columns( lhs_ + orhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) = columns( lhs_ + orhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) = columns( lhs_ + orhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) = columns( lhs_ + orhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) = columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) = columns( olhs_ + rhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) = columns( olhs_ + rhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) = columns( olhs_ + rhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) = columns( olhs_ + rhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) = columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) = columns( olhs_ + orhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) = columns( olhs_ + orhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) = columns( olhs_ + orhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) = columns( olhs_ + orhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) = columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }

      // Columns-wise schur with evaluated tensor and matrix
      {
         test_  = "Columns-wise schur with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) = columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) = columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) = columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) = columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) = columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) = columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) = columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) = columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) = columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) = columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) = columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) = columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) = columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) = columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) = columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) = columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) = columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) = columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) = columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) = columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }


      //=====================================================================================
      // Columns-wise schur with schur assignment
      //=====================================================================================

      // Columns-wise schur with schur assignment with the given tensor and matrix
      {
         test_  = "Columns-wise schur with schur assignment with the given tensor and matrix";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) += columns( lhs_ + rhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) += columns( lhs_ + rhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) += columns( lhs_ + rhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) += columns( lhs_ + rhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) += columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) += columns( lhs_ + orhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) += columns( lhs_ + orhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) += columns( lhs_ + orhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) += columns( lhs_ + orhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) += columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) += columns( olhs_ + rhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) += columns( olhs_ + rhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) += columns( olhs_ + rhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) += columns( olhs_ + rhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) += columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) += columns( olhs_ + orhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) += columns( olhs_ + orhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) += columns( olhs_ + orhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) += columns( olhs_ + orhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) += columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }

      // Columns-wise schur with schur assignment with evaluated tensor and matrix
      {
         test_  = "Columns-wise schur with schur assignment with evaluated tensor and matrix";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) += columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) += columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) += columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) += columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) += columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) += columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) += columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) += columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) += columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) += columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) += columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) += columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) += columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) += columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) += columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) += columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) += columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) += columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) += columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) += columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }


      //=====================================================================================
      // Columns-wise schur with subtraction assignment
      //=====================================================================================

      // Columns-wise schur with subtraction assignment with the given tensor and matrix
      {
         test_  = "Columns-wise schur with subtraction assignment with the given tensor and matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) -= columns( lhs_ + rhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) -= columns( lhs_ + rhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) -= columns( lhs_ + rhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) -= columns( lhs_ + rhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) -= columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) -= columns( lhs_ + orhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) -= columns( lhs_ + orhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) -= columns( lhs_ + orhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) -= columns( lhs_ + orhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) -= columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) -= columns( olhs_ + rhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) -= columns( olhs_ + rhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) -= columns( olhs_ + rhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) -= columns( olhs_ + rhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) -= columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) -= columns( olhs_ + orhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) -= columns( olhs_ + orhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) -= columns( olhs_ + orhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) -= columns( olhs_ + orhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) -= columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }

      // Columns-wise schur with subtraction assignment with evaluated tensor and matrix
      {
         test_  = "Columns-wise schur with subtraction assignment with evaluated tensor and matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) -= columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) -= columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) -= columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) -= columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) -= columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) -= columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) -= columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) -= columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) -= columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) -= columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) -= columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) -= columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) -= columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) -= columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) -= columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) -= columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) -= columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) -= columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) -= columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) -= columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }


      //=====================================================================================
      // Columns-wise schur with Schur product assignment
      //=====================================================================================

      // Columns-wise schur with Schur product assignment with the given tensor and matrix
      {
         test_  = "Columns-wise schur with Schur product assignment with the given tensor and matrix";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) %= columns( lhs_ + rhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) %= columns( lhs_ + rhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) %= columns( lhs_ + rhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) %= columns( lhs_ + rhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) %= columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) %= columns( lhs_ + orhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) %= columns( lhs_ + orhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) %= columns( lhs_ + orhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) %= columns( lhs_ + orhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) %= columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) %= columns( olhs_ + rhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) %= columns( olhs_ + rhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) %= columns( olhs_ + rhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) %= columns( olhs_ + rhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) %= columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) %= columns( olhs_ + orhs_, &indices[index], n );
               columns( odres_ , &indices[index], n ) %= columns( olhs_ + orhs_, &indices[index], n );
               columns( sres_  , &indices[index], n ) %= columns( olhs_ + orhs_, &indices[index], n );
               columns( osres_ , &indices[index], n ) %= columns( olhs_ + orhs_, &indices[index], n );
               columns( refres_, &indices[index], n ) %= columns( reflhs_ + refrhs_, &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }

      // Columns-wise schur with Schur product assignment with evaluated tensor and matrix
      {
         test_  = "Columns-wise schur with Schur product assignment with evaluated tensor and matrix";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) %= columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) %= columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) %= columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) %= columns( eval( lhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) %= columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) %= columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) %= columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) %= columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) %= columns( eval( lhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) %= columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) %= columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) %= columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) %= columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) %= columns( eval( olhs_ ) + eval( rhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) %= columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               columns( dres_  , &indices[index], n ) %= columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( odres_ , &indices[index], n ) %= columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( sres_  , &indices[index], n ) %= columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( osres_ , &indices[index], n ) %= columns( eval( olhs_ ) + eval( orhs_ ), &indices[index], n );
               columns( refres_, &indices[index], n ) %= columns( eval( reflhs_ ) + eval( refrhs_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the columns-wise dense tensor/dense tensor schur.
//
// \return void
//
// This function is called in case the columns-wise tensor/tensor schur operation is not
// available for the given tensor types \a TT and \a MT.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testColumnsOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the band-wise dense tensor/dense tensor schur.
//
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the band-wise tensor schur with plain assignment, schur assignment,
// subtraction assignment, and multiplication assignment. In case any error resulting from the
// schur or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::testBandOperation()
{
#if BLAZETEST_MATHTEST_TEST_BAND_OPERATION
   if( BLAZETEST_MATHTEST_TEST_BAND_OPERATION > 1 )
   {
      if( lhs_.rows() == 0UL || lhs_.columns() == 0UL )
         return;


      const ptrdiff_t ibegin( 1UL - lhs_.rows() );
      const ptrdiff_t iend  ( lhs_.columns() );


      //=====================================================================================
      // Band-wise schur
      //=====================================================================================

      // Band-wise schur with the given tensor and matrix
      {
         test_  = "Band-wise schur with the given tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) = band( lhs_ + rhs_, i );
               band( odres_ , i ) = band( lhs_ + rhs_, i );
               band( sres_  , i ) = band( lhs_ + rhs_, i );
               band( osres_ , i ) = band( lhs_ + rhs_, i );
               band( refres_, i ) = band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) = band( lhs_ + orhs_, i );
               band( odres_ , i ) = band( lhs_ + orhs_, i );
               band( sres_  , i ) = band( lhs_ + orhs_, i );
               band( osres_ , i ) = band( lhs_ + orhs_, i );
               band( refres_, i ) = band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) = band( olhs_ + rhs_, i );
               band( odres_ , i ) = band( olhs_ + rhs_, i );
               band( sres_  , i ) = band( olhs_ + rhs_, i );
               band( osres_ , i ) = band( olhs_ + rhs_, i );
               band( refres_, i ) = band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( size_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) = band( olhs_ + orhs_, i );
               band( odres_ , i ) = band( olhs_ + orhs_, i );
               band( sres_  , i ) = band( olhs_ + orhs_, i );
               band( osres_ , i ) = band( olhs_ + orhs_, i );
               band( refres_, i ) = band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }

      // Band-wise schur with evaluated tensor and matrix
      {
         test_  = "Band-wise schur with evaluated tensor and matrix";
         error_ = "Failed schur operation";

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) = band( eval( lhs_ ) + eval( rhs_ ), i );
               band( odres_ , i ) = band( eval( lhs_ ) + eval( rhs_ ), i );
               band( sres_  , i ) = band( eval( lhs_ ) + eval( rhs_ ), i );
               band( osres_ , i ) = band( eval( lhs_ ) + eval( rhs_ ), i );
               band( refres_, i ) = band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) = band( eval( lhs_ ) + eval( orhs_ ), i );
               band( odres_ , i ) = band( eval( lhs_ ) + eval( orhs_ ), i );
               band( sres_  , i ) = band( eval( lhs_ ) + eval( orhs_ ), i );
               band( osres_ , i ) = band( eval( lhs_ ) + eval( orhs_ ), i );
               band( refres_, i ) = band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) = band( eval( olhs_ ) + eval( rhs_ ), i );
               band( odres_ , i ) = band( eval( olhs_ ) + eval( rhs_ ), i );
               band( sres_  , i ) = band( eval( olhs_ ) + eval( rhs_ ), i );
               band( osres_ , i ) = band( eval( olhs_ ) + eval( rhs_ ), i );
               band( refres_, i ) = band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) = band( eval( olhs_ ) + eval( orhs_ ), i );
               band( odres_ , i ) = band( eval( olhs_ ) + eval( orhs_ ), i );
               band( sres_  , i ) = band( eval( olhs_ ) + eval( orhs_ ), i );
               band( osres_ , i ) = band( eval( olhs_ ) + eval( orhs_ ), i );
               band( refres_, i ) = band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }


      //=====================================================================================
      // Band-wise schur with schur assignment
      //=====================================================================================

      // Band-wise schur with schur assignment with the given tensor and matrix
      {
         test_  = "Band-wise schur with schur assignment with the given tensor and matrix";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) += band( lhs_ + rhs_, i );
               band( odres_ , i ) += band( lhs_ + rhs_, i );
               band( sres_  , i ) += band( lhs_ + rhs_, i );
               band( osres_ , i ) += band( lhs_ + rhs_, i );
               band( refres_, i ) += band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) += band( lhs_ + orhs_, i );
               band( odres_ , i ) += band( lhs_ + orhs_, i );
               band( sres_  , i ) += band( lhs_ + orhs_, i );
               band( osres_ , i ) += band( lhs_ + orhs_, i );
               band( refres_, i ) += band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) += band( olhs_ + rhs_, i );
               band( odres_ , i ) += band( olhs_ + rhs_, i );
               band( sres_  , i ) += band( olhs_ + rhs_, i );
               band( osres_ , i ) += band( olhs_ + rhs_, i );
               band( refres_, i ) += band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) += band( olhs_ + orhs_, i );
               band( odres_ , i ) += band( olhs_ + orhs_, i );
               band( sres_  , i ) += band( olhs_ + orhs_, i );
               band( osres_ , i ) += band( olhs_ + orhs_, i );
               band( refres_, i ) += band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }

      // Band-wise schur with schur assignment with evaluated tensor and matrix
      {
         test_  = "Band-wise schur with schur assignment with evaluated tensor and matrix";
         error_ = "Failed schur assignment operation";

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) += band( eval( lhs_ ) + eval( rhs_ ), i );
               band( odres_ , i ) += band( eval( lhs_ ) + eval( rhs_ ), i );
               band( sres_  , i ) += band( eval( lhs_ ) + eval( rhs_ ), i );
               band( osres_ , i ) += band( eval( lhs_ ) + eval( rhs_ ), i );
               band( refres_, i ) += band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) += band( eval( lhs_ ) + eval( orhs_ ), i );
               band( odres_ , i ) += band( eval( lhs_ ) + eval( orhs_ ), i );
               band( sres_  , i ) += band( eval( lhs_ ) + eval( orhs_ ), i );
               band( osres_ , i ) += band( eval( lhs_ ) + eval( orhs_ ), i );
               band( refres_, i ) += band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) += band( eval( olhs_ ) + eval( rhs_ ), i );
               band( odres_ , i ) += band( eval( olhs_ ) + eval( rhs_ ), i );
               band( sres_  , i ) += band( eval( olhs_ ) + eval( rhs_ ), i );
               band( osres_ , i ) += band( eval( olhs_ ) + eval( rhs_ ), i );
               band( refres_, i ) += band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) += band( eval( olhs_ ) + eval( orhs_ ), i );
               band( odres_ , i ) += band( eval( olhs_ ) + eval( orhs_ ), i );
               band( sres_  , i ) += band( eval( olhs_ ) + eval( orhs_ ), i );
               band( osres_ , i ) += band( eval( olhs_ ) + eval( orhs_ ), i );
               band( refres_, i ) += band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }


      //=====================================================================================
      // Band-wise schur with subtraction assignment
      //=====================================================================================

      // Band-wise schur with subtraction assignment with the given tensor and matrix
      {
         test_  = "Band-wise schur with subtraction assignment with the given tensor and matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) -= band( lhs_ + rhs_, i );
               band( odres_ , i ) -= band( lhs_ + rhs_, i );
               band( sres_  , i ) -= band( lhs_ + rhs_, i );
               band( osres_ , i ) -= band( lhs_ + rhs_, i );
               band( refres_, i ) -= band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) -= band( lhs_ + orhs_, i );
               band( odres_ , i ) -= band( lhs_ + orhs_, i );
               band( sres_  , i ) -= band( lhs_ + orhs_, i );
               band( osres_ , i ) -= band( lhs_ + orhs_, i );
               band( refres_, i ) -= band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) -= band( olhs_ + rhs_, i );
               band( odres_ , i ) -= band( olhs_ + rhs_, i );
               band( sres_  , i ) -= band( olhs_ + rhs_, i );
               band( osres_ , i ) -= band( olhs_ + rhs_, i );
               band( refres_, i ) -= band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) -= band( olhs_ + orhs_, i );
               band( odres_ , i ) -= band( olhs_ + orhs_, i );
               band( sres_  , i ) -= band( olhs_ + orhs_, i );
               band( osres_ , i ) -= band( olhs_ + orhs_, i );
               band( refres_, i ) -= band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }

      // Band-wise schur with subtraction assignment with evaluated tensor and matrix
      {
         test_  = "Band-wise schur with subtraction assignment with evaluated tensor and matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) -= band( eval( lhs_ ) + eval( rhs_ ), i );
               band( odres_ , i ) -= band( eval( lhs_ ) + eval( rhs_ ), i );
               band( sres_  , i ) -= band( eval( lhs_ ) + eval( rhs_ ), i );
               band( osres_ , i ) -= band( eval( lhs_ ) + eval( rhs_ ), i );
               band( refres_, i ) -= band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) -= band( eval( lhs_ ) + eval( orhs_ ), i );
               band( odres_ , i ) -= band( eval( lhs_ ) + eval( orhs_ ), i );
               band( sres_  , i ) -= band( eval( lhs_ ) + eval( orhs_ ), i );
               band( osres_ , i ) -= band( eval( lhs_ ) + eval( orhs_ ), i );
               band( refres_, i ) -= band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) -= band( eval( olhs_ ) + eval( rhs_ ), i );
               band( odres_ , i ) -= band( eval( olhs_ ) + eval( rhs_ ), i );
               band( sres_  , i ) -= band( eval( olhs_ ) + eval( rhs_ ), i );
               band( osres_ , i ) -= band( eval( olhs_ ) + eval( rhs_ ), i );
               band( refres_, i ) -= band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) -= band( eval( olhs_ ) + eval( orhs_ ), i );
               band( odres_ , i ) -= band( eval( olhs_ ) + eval( orhs_ ), i );
               band( sres_  , i ) -= band( eval( olhs_ ) + eval( orhs_ ), i );
               band( osres_ , i ) -= band( eval( olhs_ ) + eval( orhs_ ), i );
               band( refres_, i ) -= band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }


      //=====================================================================================
      // Band-wise schur with multiplication assignment
      //=====================================================================================

      // Band-wise schur with multiplication assignment with the given tensor and matrix
      {
         test_  = "Band-wise schur with multiplication assignment with the given tensor and matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) *= band( lhs_ + rhs_, i );
               band( odres_ , i ) *= band( lhs_ + rhs_, i );
               band( sres_  , i ) *= band( lhs_ + rhs_, i );
               band( osres_ , i ) *= band( lhs_ + rhs_, i );
               band( refres_, i ) *= band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) *= band( lhs_ + orhs_, i );
               band( odres_ , i ) *= band( lhs_ + orhs_, i );
               band( sres_  , i ) *= band( lhs_ + orhs_, i );
               band( osres_ , i ) *= band( lhs_ + orhs_, i );
               band( refres_, i ) *= band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) *= band( olhs_ + rhs_, i );
               band( odres_ , i ) *= band( olhs_ + rhs_, i );
               band( sres_  , i ) *= band( olhs_ + rhs_, i );
               band( osres_ , i ) *= band( olhs_ + rhs_, i );
               band( refres_, i ) *= band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) *= band( olhs_ + orhs_, i );
               band( odres_ , i ) *= band( olhs_ + orhs_, i );
               band( sres_  , i ) *= band( olhs_ + orhs_, i );
               band( osres_ , i ) *= band( olhs_ + orhs_, i );
               band( refres_, i ) *= band( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }

      // Band-wise schur with multiplication assignment with evaluated tensor and matrix
      {
         test_  = "Band-wise schur with multiplication assignment with evaluated tensor and matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) *= band( eval( lhs_ ) + eval( rhs_ ), i );
               band( odres_ , i ) *= band( eval( lhs_ ) + eval( rhs_ ), i );
               band( sres_  , i ) *= band( eval( lhs_ ) + eval( rhs_ ), i );
               band( osres_ , i ) *= band( eval( lhs_ ) + eval( rhs_ ), i );
               band( refres_, i ) *= band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,MT>( ex );
         }

         checkResults<TT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) *= band( eval( lhs_ ) + eval( orhs_ ), i );
               band( odres_ , i ) *= band( eval( lhs_ ) + eval( orhs_ ), i );
               band( sres_  , i ) *= band( eval( lhs_ ) + eval( orhs_ ), i );
               band( osres_ , i ) *= band( eval( lhs_ ) + eval( orhs_ ), i );
               band( refres_, i ) *= band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT,OMT>( ex );
         }

         checkResults<TT,OMT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) *= band( eval( olhs_ ) + eval( rhs_ ), i );
               band( odres_ , i ) *= band( eval( olhs_ ) + eval( rhs_ ), i );
               band( sres_  , i ) *= band( eval( olhs_ ) + eval( rhs_ ), i );
               band( osres_ , i ) *= band( eval( olhs_ ) + eval( rhs_ ), i );
               band( refres_, i ) *= band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,MT>( ex );
         }

         checkResults<OTT,MT>();

         try {
            initResults();
            for( ptrdiff_t i=ibegin; i<iend; ++i ) {
               band( dres_  , i ) *= band( eval( olhs_ ) + eval( orhs_ ), i );
               band( odres_ , i ) *= band( eval( olhs_ ) + eval( orhs_ ), i );
               band( sres_  , i ) *= band( eval( olhs_ ) + eval( orhs_ ), i );
               band( osres_ , i ) *= band( eval( olhs_ ) + eval( orhs_ ), i );
               band( refres_, i ) *= band( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<OTT,OMT>( ex );
         }

         checkResults<OTT,OMT>();
      }
   }
#endif
}
//*************************************************************************************************
#endif


//*************************************************************************************************
/*!\brief Testing the customized dense tensor/dense tensor schur.
//
// \param op The custom operation to be tested.
// \param name The human-readable name of the operation.
// \return void
// \exception std::runtime_error Schur product error detected.
//
// This function tests the tensor schur with plain assignment, schur assignment, and
// subtraction assignment in combination with a custom operation. In case any error resulting
// from the schur or the subsequent assignment is detected, a \a std::runtime_error exception
// is thrown.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
template< typename OP >   // Type of the custom operation
void OperationTest<TT,MT>::testCustomOperation( OP op, const std::string& name )
{
   //=====================================================================================
   // Customized schur
   //=====================================================================================

   // Customized schur with the given tensor and matrix
   {
      test_  = "Customized schur with the given tensor and matrix (" + name + ")";
      error_ = "Failed schur operation";

      try {
         initResults();
         dres_   = op( lhs_ % rhs_ );
         refres_ = op( reflhs_ % refrhs_ );
      }
      catch( std::exception& ex ) {
         convertException<TT,MT>( ex );
      }

      checkResults<TT,MT>();
   }

   // Customized schur with evaluated tensor and matrix
   {
      test_  = "Customized schur with evaluated tensor and matrix (" + name + ")";
      error_ = "Failed schur operation";

      try {
         initResults();
         dres_   = op( eval( lhs_ ) % eval( rhs_ ) );
         refres_ = op( eval( reflhs_ ) % eval( refrhs_ ) );
      }
      catch( std::exception& ex ) {
         convertException<TT,MT>( ex );
      }

      checkResults<TT,MT>();
   }


   //=====================================================================================
   // Customized schur with schur assignment
   //=====================================================================================

   // Customized schur with schur assignment with the given tensor and matrix
   {
      test_  = "Customized schur with addition assignment with the given tensor and matrix (" + name + ")";
      error_ = "Failed addition assignment operation";

      try {
         initResults();
         dres_   += op( lhs_ % rhs_ );
         refres_ += op( reflhs_ % refrhs_ );
      }
      catch( std::exception& ex ) {
         convertException<TT,MT>( ex );
      }

      checkResults<TT,MT>();
   }

   // Customized schur with schur assignment with evaluated tensor and matrix
   {
      test_  = "Customized schur with addition assignment with evaluated tensor and matrix (" + name + ")";
      error_ = "Failed addition assignment operation";

      try {
         initResults();
         dres_   += op( eval( lhs_ ) % eval( rhs_ ) );
         refres_ += op( eval( reflhs_ ) % eval( refrhs_ ) );
      }
      catch( std::exception& ex ) {
         convertException<TT,MT>( ex );
      }

      checkResults<TT,MT>();
   }


   //=====================================================================================
   // Customized schur with subtraction assignment
   //=====================================================================================

   // Customized schur with subtraction assignment with the given tensor and matrix
    {
      test_  = "Customized schur with subtraction assignment with the given tensor and matrix (" + name + ")";
      error_ = "Failed subtraction assignment operation";

      try {
         initResults();
         dres_   -= op( lhs_ % rhs_ );
         refres_ -= op( reflhs_ % refrhs_ );
      }
      catch( std::exception& ex ) {
         convertException<TT,MT>( ex );
      }

      checkResults<TT,MT>();
   }

   // Customized schur with schur assignment with evaluated tensor and matrix
   {
      test_  = "Customized schur with subtraction assignment with evaluated tensor and matrix (" + name + ")";
      error_ = "Failed subtraction assignment operation";

      try {
         initResults();
         dres_   -= op( eval( lhs_ ) % eval( rhs_ ) );
         refres_ -= op( eval( reflhs_ ) % eval( refrhs_ ) );
      }
      catch( std::exception& ex ) {
         convertException<TT,MT>( ex );
      }

      checkResults<TT,MT>();
   }


   //=====================================================================================
   // Customized schur with Schur product assignment
   //=====================================================================================

   // Customized schur with Schur product assignment with the given tensor and matrix
    {
      test_  = "Customized schur with schur assignment with the given tensor and matrix (" + name + ")";
      error_ = "Failed schur assignment operation";

      try {
         initResults();
         dres_   %= op( lhs_ % rhs_ );
         refres_ %= op( reflhs_ % refrhs_ );
      }
      catch( std::exception& ex ) {
         convertException<TT,MT>( ex );
      }

      checkResults<TT,MT>();
   }

   // Customized schur with schur assignment with evaluated tensor and matrix
   {
      test_  = "Customized schur with schur assignment with evaluated tensor and matrix (" + name + ")";
      error_ = "Failed schur assignment operation";

      try {
         initResults();
         dres_   %= op( eval( lhs_ ) % eval( rhs_ ) );
         refres_ %= op( eval( reflhs_ ) % eval( refrhs_ ) );
      }
      catch( std::exception& ex ) {
         convertException<TT,MT>( ex );
      }

      checkResults<TT,MT>();
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
// This function is called after each test case to check and compare the computed results. The
// two template arguments \a LT and \a RT indicate the types of the left-hand side and right-hand
// side operands used for the computations.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
template< typename LT     // Type of the left-hand side operand
        , typename RT >   // Type of the right-hand side operand
void OperationTest<TT,MT>::checkResults()
{
//    template <typename MT>
//    using IsRowMajorTensor = blaze::IsTensor<MT>;

   if( !isEqual( dres_, refres_ ) /*|| !isEqual( odres_, refres_ )*/ ) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect dense result detected\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side " << ( IsRowMajorTensor<LT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
          << "     " << typeid( LT ).name() << "\n"
          << "   Right-hand side " << ( IsRowMajorTensor<RT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
          << "     " << typeid( RT ).name() << "\n"
          << "   Result:\n" << dres_ << "\n"
//           << "   Result with opposite storage order:\n" << odres_ << "\n"
          << "   Expected result:\n" << refres_ << "\n";
      throw std::runtime_error( oss.str() );
   }

//    if( !isEqual( sres_, refres_ ) || !isEqual( osres_, refres_ ) ) {
//       std::ostringstream oss;
//       oss.precision( 20 );
//       oss << " Test : " << test_ << "\n"
//           << " Error: Incorrect sparse result detected\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Left-hand side " << ( IsRowMajorTensor<LT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
//           << "     " << typeid( LT ).name() << "\n"
//           << "   Right-hand side " << ( IsRowMajorTensor<RT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
//           << "     " << typeid( RT ).name() << "\n"
//           << "   Result:\n" << sres_ << "\n"
//           << "   Result with opposite storage order:\n" << osres_ << "\n"
//           << "   Expected result:\n" << refres_ << "\n";
//       throw std::runtime_error( oss.str() );
//    }
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
// results. The two template arguments \a LT and \a RT indicate the types of the left-hand
// side and right-hand side operands used for the computations.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense matrix
template< typename LT     // Type of the left-hand side operand
        , typename RT >   // Type of the right-hand side operand
void OperationTest<TT,MT>::checkTransposeResults()
{

   if( !isEqual( tdres_, refres_ )) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect dense result detected\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side " << " dense tensor type:\n"
          << "     " << typeid( LT ).name() << "\n"
          << "   Right-hand side " << " dense matrix type:\n"
          << "     " << typeid( RT ).name() << "\n"
          << "   Transpose result:\n" << tdres_ << "\n"
          << "   Expected result:\n" << refres_ << "\n";
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
/*!\brief Initializing the non-transpose result tensors.
//
// \return void
//
// This function is called before each non-transpose test case to initialize the according result
// tensors to random values.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::initResults()
{
   const blaze::UnderlyingBuiltin_t<DRE> min( randmin );
   const blaze::UnderlyingBuiltin_t<DRE> max( randmax );

   resize( dres_, pages( lhs_ ), rows( lhs_ ), columns( lhs_ ) );
   randomize( dres_, min, max );

   refres_ = dres_;
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Initializing the transpose result tensors.
//
// \return void
//
// This function is called before each transpose test case to initialize the according result
// tensors to random values.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void OperationTest<TT,MT>::initTransposeResults()
{
   const blaze::UnderlyingBuiltin_t<TDRE> min( randmin );
   const blaze::UnderlyingBuiltin_t<TDRE> max( randmax );

   resize( tdres_, columns( lhs_ ), rows( lhs_ ), pages( lhs_ ) );
   randomize( tdres_, min, max );

   refres_ = tdres_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Convert the given exception into a \a std::runtime_error exception.
//
// \param ex The \a std::exception to be extended.
// \return void
// \exception std::runtime_error The converted exception.
//
// This function converts the given exception to a \a std::runtime_error exception. Schur productally,
// the function extends the given exception message by all available information for the failed
// test. The two template arguments \a LT and \a RT indicate the types of the left-hand side and
// right-hand side operands used for the computations.
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
template< typename LT     // Type of the left-hand side operand
        , typename RT >   // Type of the right-hand side operand
void OperationTest<TT,MT>::convertException( const std::exception& ex )
{
//    template <typename MT>
//    using IsRowMajorTensor = blaze::IsTensor<MT>;

   std::ostringstream oss;
   oss << " Test : " << test_ << "\n"
       << " Error: " << error_ << "\n"
       << " Details:\n"
       << "   Random seed = " << blaze::getSeed() << "\n"
       << "   Left-hand side " << ( IsRowMajorTensor<LT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
       << "     " << typeid( LT ).name() << "\n"
       << "   Right-hand side " << ( IsRowMajorTensor<LT>::value ? ( "row-major" ) : ( "not row-major" ) ) << " dense tensor type:\n"
       << "     " << typeid( RT ).name() << "\n"
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
/*!\brief Testing the tensor schur between two specific tensor types.
//
// \param creator1 The creator for the left-hand side tensor.
// \param creator2 The creator for the right-hand side tensor.
// \return void
*/
template< typename TT    // Type of the left-hand side dense tensor
        , typename MT >  // Type of the right-hand side dense tensor
void runTest( const Creator<TT>& creator1, const Creator<MT>& creator2 )
{
#if BLAZETEST_MATHTEST_TEST_MULTIPLICATION
   if( BLAZETEST_MATHTEST_TEST_MULTIPLICATION > 1 )
   {
      for( size_t rep=0UL; rep<BLAZETEST_REPETITIONS; ++rep ) {
         OperationTest<TT,MT>( creator1, creator2 );
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
/*!\brief Macro for the definition of a dense tensor/dense tensor schur test case.
*/
#define DEFINE_DTENSDMATSCHUR_OPERATION_TEST( TT, MT ) \
   extern template class blazetest::mathtest::dtensdmatschur::OperationTest<TT,MT>
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of a dense tensor/dense tensor schur test case.
*/
#define RUN_DTENSDMATSCHUR_OPERATION_TEST( C1, C2 ) \
   blazetest::mathtest::dtensdmatschur::runTest( C1, C2 )
/*! \endcond */
//*************************************************************************************************

} // namespace dtensdmatschur

} // namespace mathtest

} // namespace blazetest

#endif
