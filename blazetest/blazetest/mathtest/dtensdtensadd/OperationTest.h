//=================================================================================================
/*!
//  \file blazetest/mathtest/dtensdtensadd/OperationTest.h
//  \brief Header file for the dense tensor/dense tensor addition operation test
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

#ifndef _BLAZETEST_MATHTEST_DTENSDTENSADD_OPERATIONTEST_H_
#define _BLAZETEST_MATHTEST_DTENSDTENSADD_OPERATIONTEST_H_


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
#include <blaze/math/Functors.h>
#include <blaze/math/shims/Equal.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsResizable.h>
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
// #include <blaze_tensor/math/constraints/SparseTensor.h>
#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/typetraits/StorageOrder.h>
#include <blaze_tensor/math/Views.h>

namespace blazetest {

namespace mathtest {

namespace dtensdtensadd {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class template for the dense tensor/dense tensor addition operation test.
//
// This class template represents one particular tensor addition test between two tensors of
// a particular type. The two template arguments \a MT1 and \a MT2 represent the types of the
// left-hand side and right-hand side tensor, respectively.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
class OperationTest
{
 private:
   //**Type definitions****************************************************************************
   using ET1 = blaze::ElementType_t<MT1>;  //!< Element type 1
   using ET2 = blaze::ElementType_t<MT2>;  //!< Element type 2

//    using OMT1  = blaze::OppositeType_t<MT1>;    //!< Tensor type 1 with opposite storage order
//    using OMT2  = blaze::OppositeType_t<MT2>;    //!< Tensor type 2 with opposite storage order
   using TMT1  = blaze::TransposeType_t<MT1>;   //!< Transpose tensor type 1
   using TMT2  = blaze::TransposeType_t<MT2>;   //!< Transpose tensor type 2
//    using TOMT1 = blaze::TransposeType_t<OMT1>;  //!< Transpose tensor type 1 with opposite storage order
//    using TOMT2 = blaze::TransposeType_t<OMT2>;  //!< Transpose tensor type 2 with opposite storage order

   //! Dense result type
   using DRE = blaze::AddTrait_t<MT1,MT2>;

   using DET   = blaze::ElementType_t<DRE>;     //!< Element type of the dense result
//    using ODRE  = blaze::OppositeType_t<DRE>;    //!< Dense result type with opposite storage order
   using TDRE  = blaze::TransposeType_t<DRE>;   //!< Transpose dense result type
//    using TODRE = blaze::TransposeType_t<ODRE>;  //!< Transpose dense result type with opposite storage order

   //! Sparse result type
//    using SRE = MatchAdaptor_t< DRE, blaze::CompressedTensor<DET,false> >;
//
//    using SET   = blaze::ElementType_t<SRE>;     //!< Element type of the sparse result
//    using OSRE  = blaze::OppositeType_t<SRE>;    //!< Sparse result type with opposite storage order
//    using TSRE  = blaze::TransposeType_t<SRE>;   //!< Transpose sparse result type
//    using TOSRE = blaze::TransposeType_t<OSRE>;  //!< Transpose sparse result type with opposite storage order

   using RT1 = blaze::DynamicTensor<ET1>;       //!< Reference type 1
   using RT2 = blaze::DynamicTensor<ET2>;       //!< Reference type 2
//    using RT2 = blaze::CompressedTensor<ET2,false>;  //!< Reference type 2

   //! Reference result type
   using RRE = blaze::AddTrait_t<RT1, RT2>; //MatchSymmetry_t< DRE, blaze::AddTrait_t<RT1, RT2> >;

   //! Type of the tensor/tensor addition expression
   using TensTensAddExprType = blaze::Decay_t< decltype( std::declval<MT1>() + std::declval<MT2>() ) >;

//    //! Type of the tensor/transpose tensor addition expression
//    using TensTTensAddExprType = blaze::Decay_t< decltype( std::declval<MT1>() + std::declval<OMT2>() ) >;
//
//    //! Type of the transpose tensor/tensor addition expression
//    using TTensTensAddExprType = blaze::Decay_t< decltype( std::declval<OMT1>() + std::declval<MT2>() ) >;
//
//    //! Type of the transpose tensor/transpose tensor addition expression
//    using TTensTTensAddExprType = blaze::Decay_t< decltype( std::declval<OMT1>() + std::declval<OMT2>() ) >;
   //**********************************************************************************************

 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit OperationTest( const Creator<MT1>& creator1, const Creator<MT2>& creator2 );
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
//                           void testCTransOperation   ();
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
   MT1   lhs_;     //!< The left-hand side dense tensor.
   MT2   rhs_;     //!< The right-hand side dense tensor.
//    OMT1  olhs_;    //!< The left-hand side dense tensor with opposite storage order.
//    OMT2  orhs_;    //!< The right-hand side dense tensor with opposite storage order.
   DRE   dres_;    //!< The dense result tensor.
//    SRE   sres_;    //!< The sparse result tensor.
//    ODRE  odres_;   //!< The dense result tensor with opposite storage order.
//    OSRE  osres_;   //!< The sparse result tensor with opposite storage order.
   TDRE  tdres_;   //!< The transpose dense result tensor.
//    TSRE  tsres_;   //!< The transpose sparse result tensor.
//    TODRE todres_;  //!< The transpose dense result tensor with opposite storage order.
//    TOSRE tosres_;  //!< The transpose sparse result tensor with opposite storage order.
   RT1   reflhs_;  //!< The reference left-hand side tensor.
   RT2   refrhs_;  //!< The reference right-hand side tensor.
   RRE   refres_;  //!< The reference result.

   std::string test_;   //!< Label of the currently performed test.
   std::string error_;  //!< Description of the current error type.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( MT1   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( MT2   );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( OMT1  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( OMT2  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TMT1  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TMT2  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TOMT1 );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TOMT2 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( RT1   );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( RT2   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( RRE   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( DRE   );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( SRE   );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( ODRE  );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( OSRE  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TDRE  );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( TSRE  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TODRE );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( TOSRE );

//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( MT1   );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( MT2   );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( OMT1  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( OMT2  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( TMT1  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( TMT2  );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( TOMT1 );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( TOMT2 );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( RT1   );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( RT2   );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( DRE   );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( SRE   );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( ODRE  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( OSRE  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( TDRE  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( TSRE  );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( TODRE );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( TOSRE );

//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET1, blaze::ElementType_t<OMT1>   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET2, blaze::ElementType_t<OMT2>   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET1, blaze::ElementType_t<TMT1>   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET2, blaze::ElementType_t<TMT2>   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET1, blaze::ElementType_t<TOMT1>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET2, blaze::ElementType_t<TOMT2>  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<DRE>    );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<ODRE>   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<TDRE>   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<TODRE>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<SRE>    );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<SRE>    );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<OSRE>   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<TSRE>   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<TOSRE>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<DRE>    );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT1, blaze::OppositeType_t<OMT1>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT2, blaze::OppositeType_t<OMT2>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT1, blaze::TransposeType_t<TMT1> );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT2, blaze::TransposeType_t<TMT2> );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DRE, blaze::OppositeType_t<ODRE>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DRE, blaze::TransposeType_t<TDRE> );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SRE, blaze::OppositeType_t<OSRE>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SRE, blaze::TransposeType_t<TSRE> );

   BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_SAME_STORAGE_ORDER     ( TensTensAddExprType, blaze::ResultType_t<TensTensAddExprType>    );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TensTensAddExprType, blaze::OppositeType_t<TensTensAddExprType>  );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TensTensAddExprType, blaze::TransposeType_t<TensTensAddExprType> );

//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_SAME_STORAGE_ORDER     ( TensTTensAddExprType, blaze::ResultType_t<TensTTensAddExprType>    );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TensTTensAddExprType, blaze::OppositeType_t<TensTTensAddExprType>  );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TensTTensAddExprType, blaze::TransposeType_t<TensTTensAddExprType> );
//
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_SAME_STORAGE_ORDER     ( TTensTensAddExprType, blaze::ResultType_t<TTensTensAddExprType>    );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TTensTensAddExprType, blaze::OppositeType_t<TTensTensAddExprType>  );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TTensTensAddExprType, blaze::TransposeType_t<TTensTensAddExprType> );
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
/*!\brief Constructor for the dense tensor/dense tensor addition operation test.
//
// \param creator1 The creator for the left-hand side dense tensor of the tensor addition.
// \param creator2 The creator for the right-hand side dense tensor of the tensor addition.
// \exception std::runtime_error Operation error detected.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
OperationTest<MT1,MT2>::OperationTest( const Creator<MT1>& creator1, const Creator<MT2>& creator2 )
   : lhs_( creator1() )  // The left-hand side dense tensor
   , rhs_( creator2() )  // The right-hand side dense tensor
//    , olhs_( lhs_ )       // The left-hand side dense tensor with opposite storage order
//    , orhs_( rhs_ )       // The right-hand side dense tensor with opposite storage order
   , dres_()             // The dense result tensor
//    , sres_()             // The sparse result tensor
//    , odres_()            // The dense result tensor with opposite storage order
//    , osres_()            // The sparse result tensor with opposite storage order
   , tdres_()            // The transpose dense result tensor
//    , tsres_()            // The transpose sparse result tensor
//    , todres_()           // The transpose dense result tensor with opposite storage order
//    , tosres_()           // The transpose sparse result tensor with opposite storage order
   , reflhs_( lhs_ )     // The reference left-hand side tensor
   , refrhs_( rhs_ )     // The reference right-hand side tensor
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
//    testCTransOperation();
   testAbsOperation();
   testConjOperation();
   testRealOperation();
   testImagOperation();
   testInvOperation();
   testEvalOperation();
   testSerialOperation();
//    testDeclSymOperation( Or_t< blaze::IsSquare<DRE>, blaze::IsResizable<DRE> >() );
//    testDeclHermOperation( Or_t< blaze::IsSquare<DRE>, blaze::IsResizable<DRE> >() );
//    testDeclLowOperation( Or_t< blaze::IsSquare<DRE>, blaze::IsResizable<DRE> >() );
//    testDeclUppOperation( Or_t< blaze::IsSquare<DRE>, blaze::IsResizable<DRE> >() );
//    testDeclDiagOperation( Or_t< blaze::IsSquare<DRE>, blaze::IsResizable<DRE> >() );
   testSubtensorOperation( blaze::Not_t< blaze::IsUniform<DRE> >() );
   testRowSliceOperation( blaze::Not_t< blaze::IsUniform<DRE> >() );
//    testRowSlicesOperation( Nor_t< blaze::IsSymmetric<DRE>, blaze::IsHermitian<DRE> >() );
   testColumnSliceOperation( blaze::Not_t< blaze::IsUniform<DRE> >() );
//    testColumnSlicesOperation( Nor_t< blaze::IsSymmetric<DRE>, blaze::IsHermitian<DRE> >() );
   testPageSliceOperation( blaze::Not_t< blaze::IsUniform<DRE> >() );
//    testPageSlicesOperation( Nor_t< blaze::IsSymmetric<DRE>, blaze::IsHermitian<DRE> >() );
//    testBandOperation();
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
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testInitialStatus()
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
          << "     " << typeid( MT1 ).name() << "\n"
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
          << "     " << typeid( MT1 ).name() << "\n"
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
          << "     " << typeid( MT1 ).name() << "\n"
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
          << "     " << typeid( MT2 ).name() << "\n"
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
          << "     " << typeid( MT2 ).name() << "\n"
          << "   Detected number of columns = " << rhs_.columns() << "\n"
          << "   Expected number of columns = " << refrhs_.columns() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the number of pages of the right-hand side operand
   if( rhs_.pages() != refrhs_.pages() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of right-hand side row-major dense operand\n"
          << " Error: Invalid number of pages\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( MT2 ).name() << "\n"
          << "   Detected number of pages = " << rhs_.pages() << "\n"
          << "   Expected number of pages = " << refrhs_.pages() << "\n";
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
          << "     " << typeid( MT1 ).name() << "\n"
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
          << "     " << typeid( MT2 ).name() << "\n"
          << "   Current initialization:\n" << rhs_ << "\n"
          << "   Expected initialization:\n" << refrhs_ << "\n";
      throw std::runtime_error( oss.str() );
   }


//    //=====================================================================================
//    // Performing initial tests with the column-major types
//    //=====================================================================================
//
//    // Checking the number of rows of the left-hand side operand
//    if( olhs_.rows() != reflhs_.rows() ) {
//       std::ostringstream oss;
//       oss << " Test: Initial size comparison of left-hand side column-major dense operand\n"
//           << " Error: Invalid number of rows\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Column-major dense tensor type:\n"
//           << "     " << typeid( OMT1 ).name() << "\n"
//           << "   Detected number of rows = " << olhs_.rows() << "\n"
//           << "   Expected number of rows = " << reflhs_.rows() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//
//    // Checking the number of columns of the left-hand side operand
//    if( olhs_.columns() != reflhs_.columns() ) {
//       std::ostringstream oss;
//       oss << " Test: Initial size comparison of left-hand side column-major dense operand\n"
//           << " Error: Invalid number of columns\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Column-major dense tensor type:\n"
//           << "     " << typeid( OMT1 ).name() << "\n"
//           << "   Detected number of columns = " << olhs_.columns() << "\n"
//           << "   Expected number of columns = " << reflhs_.columns() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//
//    // Checking the number of rows of the right-hand side operand
//    if( orhs_.rows() != refrhs_.rows() ) {
//       std::ostringstream oss;
//       oss << " Test: Initial size comparison of right-hand side column-major dense operand\n"
//           << " Error: Invalid number of rows\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Column-major dense tensor type:\n"
//           << "     " << typeid( OMT2 ).name() << "\n"
//           << "   Detected number of rows = " << orhs_.rows() << "\n"
//           << "   Expected number of rows = " << refrhs_.rows() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//
//    // Checking the number of columns of the right-hand side operand
//    if( orhs_.columns() != refrhs_.columns() ) {
//       std::ostringstream oss;
//       oss << " Test: Initial size comparison of right-hand side column-major dense operand\n"
//           << " Error: Invalid number of columns\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Column-major dense tensor type:\n"
//           << "     " << typeid( OMT2 ).name() << "\n"
//           << "   Detected number of columns = " << orhs_.columns() << "\n"
//           << "   Expected number of columns = " << refrhs_.columns() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//
//    // Checking the initialization of the left-hand side operand
//    if( !isEqual( olhs_, reflhs_ ) ) {
//       std::ostringstream oss;
//       oss << " Test: Initial test of initialization of left-hand side column-major dense operand\n"
//           << " Error: Invalid tensor initialization\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Column-major dense tensor type:\n"
//           << "     " << typeid( OMT1 ).name() << "\n"
//           << "   Current initialization:\n" << olhs_ << "\n"
//           << "   Expected initialization:\n" << reflhs_ << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//
//    // Checking the initialization of the right-hand side operand
//    if( !isEqual( orhs_, refrhs_ ) ) {
//       std::ostringstream oss;
//       oss << " Test: Initial test of initialization of right-hand side column-major dense operand\n"
//           << " Error: Invalid tensor initialization\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Column-major dense tensor type:\n"
//           << "     " << typeid( OMT2 ).name() << "\n"
//           << "   Current initialization:\n" << orhs_ << "\n"
//           << "   Expected initialization:\n" << refrhs_ << "\n";
//       throw std::runtime_error( oss.str() );
//    }
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
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testAssignment()
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
          << "     " << typeid( MT1 ).name() << "\n"
          << "   Right-hand side row-major dense tensor type:\n"
          << "     " << typeid( MT2 ).name() << "\n"
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
          << "     " << typeid( MT1 ).name() << "\n"
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
          << "     " << typeid( MT2 ).name() << "\n"
          << "   Current initialization:\n" << rhs_ << "\n"
          << "   Expected initialization:\n" << refrhs_ << "\n";
      throw std::runtime_error( oss.str() );
   }


//    //=====================================================================================
//    // Performing an assignment with the column-major types
//    //=====================================================================================
//
//    try {
//       olhs_ = reflhs_;
//       orhs_ = refrhs_;
//    }
//    catch( std::exception& ex ) {
//       std::ostringstream oss;
//       oss << " Test: Assignment with the column-major types\n"
//           << " Error: Failed assignment\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Left-hand side column-major dense tensor type:\n"
//           << "     " << typeid( OMT1 ).name() << "\n"
//           << "   Right-hand side column-major dense tensor type:\n"
//           << "     "  << typeid( OMT2 ).name() << "\n"
//           << "   Error message: " << ex.what() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//
//    if( !isEqual( olhs_, reflhs_ ) ) {
//       std::ostringstream oss;
//       oss << " Test: Checking the assignment result of left-hand side column-major dense operand\n"
//           << " Error: Invalid tensor initialization\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Column-major dense tensor type:\n"
//           << "     " << typeid( OMT1 ).name() << "\n"
//           << "   Current initialization:\n" << olhs_ << "\n"
//           << "   Expected initialization:\n" << reflhs_ << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//
//    if( !isEqual( orhs_, refrhs_ ) ) {
//       std::ostringstream oss;
//       oss << " Test: Checking the assignment result of right-hand side column-major dense operand\n"
//           << " Error: Invalid tensor initialization\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Column-major dense tensor type:\n"
//           << "     " << typeid( OMT2 ).name() << "\n"
//           << "   Current initialization:\n" << orhs_ << "\n"
//           << "   Expected initialization:\n" << refrhs_ << "\n";
//       throw std::runtime_error( oss.str() );
//    }
}
//*************************************************************************************************

template <typename MT>
using IsRowMajorTensor = blaze::IsRowMajorTensor<MT>;


//*************************************************************************************************
/*!\brief Testing the explicit evaluation.
//
// \return void
// \exception std::runtime_error Evaluation error detected.
//
// This function tests the explicit evaluation. In case any error is detected, a
// \a std::runtime_error exception is thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testEvaluation()
{
   //=====================================================================================
   // Testing the evaluation with two row-major tensors
   //=====================================================================================

   {
      const auto res   ( evaluate( lhs_    + rhs_    ) );
      const auto refres( evaluate( reflhs_ + refrhs_ ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with the given tensors\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side " << ( IsRowMajorTensor<MT1>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
             << "     " << typeid( lhs_ ).name() << "\n"
             << "   Right-hand side " << ( IsRowMajorTensor<MT2>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
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
      const auto res   ( evaluate( eval( lhs_ )    + eval( rhs_ )    ) );
      const auto refres( evaluate( eval( reflhs_ ) + eval( refrhs_ ) ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with evaluated tensors\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side " << ( IsRowMajorTensor<MT1>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
             << "     " << typeid( lhs_ ).name() << "\n"
             << "   Right-hand side " << ( IsRowMajorTensor<MT2>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
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


//    //=====================================================================================
//    // Testing the evaluation with a row-major tensor and a column-major tensor
//    //=====================================================================================
//
//    {
//       const auto res   ( evaluate( lhs_    + orhs_   ) );
//       const auto refres( evaluate( reflhs_ + refrhs_ ) );
//
//       if( !isEqual( res, refres ) ) {
//          std::ostringstream oss;
//          oss << " Test: Evaluation with the given tensors\n"
//              << " Error: Failed evaluation\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side " << ( IsRowMajorTensor<MT1>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//              << "     " << typeid( lhs_ ).name() << "\n"
//              << "   Right-hand side " << ( IsRowMajorTensor<OMT2>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//              << "     " << typeid( orhs_ ).name() << "\n"
//              << "   Deduced result type:\n"
//              << "     " << typeid( res ).name() << "\n"
//              << "   Deduced reference result type:\n"
//              << "     " << typeid( refres ).name() << "\n"
//              << "   Result:\n" << res << "\n"
//              << "   Expected result:\n" << refres << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       const auto res   ( evaluate( eval( lhs_ )    + eval( orhs_ )   ) );
//       const auto refres( evaluate( eval( reflhs_ ) + eval( refrhs_ ) ) );
//
//       if( !isEqual( res, refres ) ) {
//          std::ostringstream oss;
//          oss << " Test: Evaluation with the given tensors\n"
//              << " Error: Failed evaluation\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side " << ( IsRowMajorTensor<MT1>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//              << "     " << typeid( lhs_ ).name() << "\n"
//              << "   Right-hand side " << ( IsRowMajorTensor<OMT2>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//              << "     " << typeid( orhs_ ).name() << "\n"
//              << "   Deduced result type:\n"
//              << "     " << typeid( res ).name() << "\n"
//              << "   Deduced reference result type:\n"
//              << "     " << typeid( refres ).name() << "\n"
//              << "   Result:\n" << res << "\n"
//              << "   Expected result:\n" << refres << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Testing the evaluation with a column-major tensor and a row-major tensor
//    //=====================================================================================
//
//    {
//       const auto res   ( evaluate( olhs_   + rhs_    ) );
//       const auto refres( evaluate( reflhs_ + refrhs_ ) );
//
//       if( !isEqual( res, refres ) ) {
//          std::ostringstream oss;
//          oss << " Test: Evaluation with the given tensors\n"
//              << " Error: Failed evaluation\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side " << ( IsRowMajorTensor<OMT1>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//              << "     " << typeid( olhs_ ).name() << "\n"
//              << "   Right-hand side " << ( IsRowMajorTensor<MT2>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//              << "     " << typeid( rhs_ ).name() << "\n"
//              << "   Deduced result type:\n"
//              << "     " << typeid( res ).name() << "\n"
//              << "   Deduced reference result type:\n"
//              << "     " << typeid( refres ).name() << "\n"
//              << "   Result:\n" << res << "\n"
//              << "   Expected result:\n" << refres << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       const auto res   ( evaluate( eval( olhs_ )   + eval( rhs_ )    ) );
//       const auto refres( evaluate( eval( reflhs_ ) + eval( refrhs_ ) ) );
//
//       if( !isEqual( res, refres ) ) {
//          std::ostringstream oss;
//          oss << " Test: Evaluation with the given tensors\n"
//              << " Error: Failed evaluation\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side " << ( IsRowMajorTensor<OMT1>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//              << "     " << typeid( olhs_ ).name() << "\n"
//              << "   Right-hand side " << ( IsRowMajorTensor<MT2>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//              << "     " << typeid( rhs_ ).name() << "\n"
//              << "   Deduced result type:\n"
//              << "     " << typeid( res ).name() << "\n"
//              << "   Deduced reference result type:\n"
//              << "     " << typeid( refres ).name() << "\n"
//              << "   Result:\n" << res << "\n"
//              << "   Expected result:\n" << refres << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//
//    //=====================================================================================
//    // Testing the evaluation with two column-major tensors
//    //=====================================================================================
//
//    {
//       const auto res   ( evaluate( olhs_   + orhs_   ) );
//       const auto refres( evaluate( reflhs_ + refrhs_ ) );
//
//       if( !isEqual( res, refres ) ) {
//          std::ostringstream oss;
//          oss << " Test: Evaluation with the given tensors\n"
//              << " Error: Failed evaluation\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side " << ( IsRowMajorTensor<OMT1>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//              << "     " << typeid( olhs_ ).name() << "\n"
//              << "   Right-hand side " << ( IsRowMajorTensor<OMT2>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//              << "     " << typeid( orhs_ ).name() << "\n"
//              << "   Deduced result type:\n"
//              << "     " << typeid( res ).name() << "\n"
//              << "   Deduced reference result type:\n"
//              << "     " << typeid( refres ).name() << "\n"
//              << "   Result:\n" << res << "\n"
//              << "   Expected result:\n" << refres << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    {
//       const auto res   ( evaluate( eval( olhs_ )   + eval( orhs_ )   ) );
//       const auto refres( evaluate( eval( reflhs_ ) + eval( refrhs_ ) ) );
//
//       if( !isEqual( res, refres ) ) {
//          std::ostringstream oss;
//          oss << " Test: Evaluation with the given tensors\n"
//              << " Error: Failed evaluation\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed()} << "\n"
//              << "   Left-hand side " << ( IsRowMajorTensor<OMT1>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//              << "     " << typeid( olhs_ ).name() << "\n"
//              << "   Right-hand side " << ( IsRowMajorTensor<OMT2>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//              << "     " << typeid( orhs_ ).name() << "\n"
//              << "   Deduced result type:\n"
//              << "     " << typeid( res ).name() << "\n"
//              << "   Deduced reference result type:\n"
//              << "     " << typeid( refres ).name() << "\n"
//              << "   Result:\n" << res << "\n"
//              << "   Expected result:\n" << refres << "\n";
//          throw std::runtime_error( oss.str() );
//       }
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
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testElementAccess()
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

      if( !equal( ( lhs_ + rhs_ )(o,m,n), ( reflhs_ + refrhs_ )(o,m,n) ) ||
          !equal( ( lhs_ + rhs_ ).at(o,m,n), ( reflhs_ + refrhs_ ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of addition expression\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side row-major dense tensor type:\n"
             << "     " << typeid( MT1 ).name() << "\n"
             << "   Right-hand side row-major dense tensor type:\n"
             << "     " << typeid( MT2 ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( ( lhs_ + eval( rhs_ ) )(o,m,n), ( reflhs_ + eval( refrhs_ ) )(o,m,n) ) ||
          !equal( ( lhs_ + eval( rhs_ ) ).at(o,m,n), ( reflhs_ + eval( refrhs_ ) ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of right evaluated addition expression\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side row-major dense tensor type:\n"
             << "     " << typeid( MT1 ).name() << "\n"
             << "   Right-hand side row-major dense tensor type:\n"
             << "     " << typeid( MT2 ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( ( eval( lhs_ ) + rhs_ )(o,m,n), ( eval( reflhs_ ) + refrhs_ )(o,m,n) ) ||
          !equal( ( eval( lhs_ ) + rhs_ ).at(o,m,n), ( eval( reflhs_ ) + refrhs_ ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of left evaluated addition expression\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side row-major dense tensor type:\n"
             << "     " << typeid( MT1 ).name() << "\n"
             << "   Right-hand side row-major dense tensor type:\n"
             << "     " << typeid( MT2 ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( ( eval( lhs_ ) + eval( rhs_ ) )(o,m,n), ( eval( reflhs_ ) + eval( refrhs_ ) )(o,m,n) ) ||
          !equal( ( eval( lhs_ ) + eval( rhs_ ) ).at(o,m,n), ( eval( reflhs_ ) + eval( refrhs_ ) ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of fully evaluated addition expression\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Left-hand side row-major dense tensor type:\n"
             << "     " << typeid( MT1 ).name() << "\n"
             << "   Right-hand side row-major dense tensor type:\n"
             << "     " << typeid( MT2 ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }
   }

   try {
      ( lhs_ + rhs_ ).at( 0UL, 0UL, lhs_.columns() );

      std::ostringstream oss;
      oss << " Test : Checked element access of addition expression\n"
          << " Error: Out-of-bound access succeeded\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side row-major dense tensor type:\n"
          << "     " << typeid( MT1 ).name() << "\n"
          << "   Right-hand side row-major dense tensor type:\n"
          << "     " << typeid( MT2 ).name() << "\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::out_of_range& ) {}

   try {
      ( lhs_ + rhs_ ).at( 0UL, lhs_.rows(), 0UL );

      std::ostringstream oss;
      oss << " Test : Checked element access of addition expression\n"
          << " Error: Out-of-bound access succeeded\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side row-major dense tensor type:\n"
          << "     " << typeid( MT1 ).name() << "\n"
          << "   Right-hand side row-major dense tensor type:\n"
          << "     " << typeid( MT2 ).name() << "\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::out_of_range& ) {}

   try {
      ( lhs_ + rhs_ ).at( lhs_.pages(), 0UL, 0UL );

      std::ostringstream oss;
      oss << " Test : Checked element access of addition expression\n"
          << " Error: Out-of-bound access succeeded\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side row-major dense tensor type:\n"
          << "     " << typeid( MT1 ).name() << "\n"
          << "   Right-hand side row-major dense tensor type:\n"
          << "     " << typeid( MT2 ).name() << "\n";
      throw std::runtime_error( oss.str() );
   }
   catch( std::out_of_range& ) {}


//    //=====================================================================================
//    // Testing the element access with a row-major tensor and a column-major tensor
//    //=====================================================================================
//
//    if( lhs_.rows() > 0UL && lhs_.columns() > 0UL )
//    {
//       const size_t m( lhs_.rows()    - 1UL );
//       const size_t n( lhs_.columns() - 1UL );
//
//       if( !equal( ( lhs_ + orhs_ )(o,m,n), ( reflhs_ + refrhs_ )(o,m,n) ) ||
//           !equal( ( lhs_ + orhs_ ).at(o,m,n), ( reflhs_ + refrhs_ ).at(o,m,n) ) ) {
//          std::ostringstream oss;
//          oss << " Test : Element access of addition expression\n"
//              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side row-major dense tensor type:\n"
//              << "     " << typeid( MT1 ).name() << "\n"
//              << "   Right-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT2 ).name() << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//
//       if( !equal( ( lhs_ + eval( orhs_ ) )(o,m,n), ( reflhs_ + eval( refrhs_ ) )(o,m,n) ) ||
//           !equal( ( lhs_ + eval( orhs_ ) ).at(o,m,n), ( reflhs_ + eval( refrhs_ ) ).at(o,m,n) ) ) {
//          std::ostringstream oss;
//          oss << " Test : Element access of right evaluated addition expression\n"
//              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side row-major dense tensor type:\n"
//              << "     " << typeid( MT1 ).name() << "\n"
//              << "   Right-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT2 ).name() << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//
//       if( !equal( ( eval( lhs_ ) + orhs_ )(o,m,n), ( eval( reflhs_ ) + refrhs_ )(o,m,n) ) ||
//           !equal( ( eval( lhs_ ) + orhs_ ).at(o,m,n), ( eval( reflhs_ ) + refrhs_ ).at(o,m,n) ) ) {
//          std::ostringstream oss;
//          oss << " Test : Element access of left evaluated addition expression\n"
//              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side row-major dense tensor type:\n"
//              << "     " << typeid( MT1 ).name() << "\n"
//              << "   Right-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT2 ).name() << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//
//       if( !equal( ( eval( lhs_ ) + eval( orhs_ ) )(o,m,n), ( eval( reflhs_ ) + eval( refrhs_ ) )(o,m,n) ) ||
//           !equal( ( eval( lhs_ ) + eval( orhs_ ) ).at(o,m,n), ( eval( reflhs_ ) + eval( refrhs_ ) ).at(o,m,n) ) ) {
//          std::ostringstream oss;
//          oss << " Test : Element access of fully evaluated addition expression\n"
//              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side row-major dense tensor type:\n"
//              << "     " << typeid( MT1 ).name() << "\n"
//              << "   Right-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT2 ).name() << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    try {
//       ( lhs_ + orhs_ ).at( 0UL, lhs_.columns() );
//
//       std::ostringstream oss;
//       oss << " Test : Checked element access of addition expression\n"
//           << " Error: Out-of-bound access succeeded\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Left-hand side row-major dense tensor type:\n"
//           << "     " << typeid( MT1 ).name() << "\n"
//           << "   Right-hand side column-major dense tensor type:\n"
//           << "     " << typeid( OMT2 ).name() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//    catch( std::out_of_range& ) {}
//
//    try {
//       ( lhs_ + orhs_ ).at( lhs_.rows(), 0UL );
//
//       std::ostringstream oss;
//       oss << " Test : Checked element access of addition expression\n"
//           << " Error: Out-of-bound access succeeded\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Left-hand side row-major dense tensor type:\n"
//           << "     " << typeid( MT1 ).name() << "\n"
//           << "   Right-hand side column-major dense tensor type:\n"
//           << "     " << typeid( OMT2 ).name() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//    catch( std::out_of_range& ) {}
//
//
//    //=====================================================================================
//    // Testing the element access with a column-major tensor and a row-major tensor
//    //=====================================================================================
//
//    if( olhs_.rows() > 0UL && olhs_.columns() > 0UL )
//    {
//       const size_t m( olhs_.rows()    - 1UL );
//       const size_t n( olhs_.columns() - 1UL );
//
//       if( !equal( ( olhs_ + rhs_ )(o,m,n), ( reflhs_ + refrhs_ )(o,m,n) ) ||
//           !equal( ( olhs_ + rhs_ ).at(o,m,n), ( reflhs_ + refrhs_ ).at(o,m,n) ) ) {
//          std::ostringstream oss;
//          oss << " Test : Element access of addition expression\n"
//              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT1 ).name() << "\n"
//              << "   Right-hand side row-major dense tensor type:\n"
//              << "     " << typeid( MT2 ).name() << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//
//       if( !equal( ( olhs_ + eval( rhs_ ) )(o,m,n), ( reflhs_ + eval( refrhs_ ) )(o,m,n) ) ||
//           !equal( ( olhs_ + eval( rhs_ ) ).at(o,m,n), ( reflhs_ + eval( refrhs_ ) ).at(o,m,n) ) ) {
//          std::ostringstream oss;
//          oss << " Test : Element access of right evaluated addition expression\n"
//              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT1 ).name() << "\n"
//              << "   Right-hand side row-major dense tensor type:\n"
//              << "     " << typeid( MT2 ).name() << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//
//       if( !equal( ( eval( olhs_ ) + rhs_ )(o,m,n), ( eval( reflhs_ ) + refrhs_ )(o,m,n) ) ||
//           !equal( ( eval( olhs_ ) + rhs_ ).at(o,m,n), ( eval( reflhs_ ) + refrhs_ ).at(o,m,n) ) ) {
//          std::ostringstream oss;
//          oss << " Test : Element access of left evaluated addition expression\n"
//              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT1 ).name() << "\n"
//              << "   Right-hand side row-major dense tensor type:\n"
//              << "     " << typeid( MT2 ).name() << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//
//       if( !equal( ( eval( olhs_ ) + eval( rhs_ ) )(o,m,n), ( eval( reflhs_ ) + eval( refrhs_ ) )(o,m,n) ) ||
//           !equal( ( eval( olhs_ ) + eval( rhs_ ) ).at(o,m,n), ( eval( reflhs_ ) + eval( refrhs_ ) ).at(o,m,n) ) ) {
//          std::ostringstream oss;
//          oss << " Test : Element access of fully evaluated addition expression\n"
//              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT1 ).name() << "\n"
//              << "   Right-hand side row-major dense tensor type:\n"
//              << "     " << typeid( MT2 ).name() << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    try {
//       ( olhs_ + rhs_ ).at( 0UL, lhs_.columns() );
//
//       std::ostringstream oss;
//       oss << " Test : Checked element access of addition expression\n"
//           << " Error: Out-of-bound access succeeded\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Left-hand side column-major dense tensor type:\n"
//           << "     " << typeid( OMT1 ).name() << "\n"
//           << "   Right-hand side row-major dense tensor type:\n"
//           << "     " << typeid( MT2 ).name() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//    catch( std::out_of_range& ) {}
//
//    try {
//       ( olhs_ + rhs_ ).at( lhs_.rows(), 0UL );
//
//       std::ostringstream oss;
//       oss << " Test : Checked element access of addition expression\n"
//           << " Error: Out-of-bound access succeeded\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Left-hand side column-major dense tensor type:\n"
//           << "     " << typeid( OMT1 ).name() << "\n"
//           << "   Right-hand side row-major dense tensor type:\n"
//           << "     " << typeid( MT2 ).name() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//    catch( std::out_of_range& ) {}
//
//
//    //=====================================================================================
//    // Testing the element access with two column-major tensors
//    //=====================================================================================
//
//    if( olhs_.rows() > 0UL && olhs_.columns() > 0UL )
//    {
//       const size_t m( olhs_.rows()    - 1UL );
//       const size_t n( olhs_.columns() - 1UL );
//
//       if( !equal( ( olhs_ + orhs_ )(o,m,n), ( reflhs_ + refrhs_ )(o,m,n) ) ||
//           !equal( ( olhs_ + orhs_ ).at(o,m,n), ( reflhs_ + refrhs_ ).at(o,m,n) ) ) {
//          std::ostringstream oss;
//          oss << " Test : Element access of addition expression\n"
//              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT1 ).name() << "\n"
//              << "   Right-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT2 ).name() << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//
//       if( !equal( ( olhs_ + eval( orhs_ ) )(o,m,n), ( reflhs_ + eval( refrhs_ ) )(o,m,n) ) ||
//           !equal( ( olhs_ + eval( orhs_ ) ).at(o,m,n), ( reflhs_ + eval( refrhs_ ) ).at(o,m,n) ) ) {
//          std::ostringstream oss;
//          oss << " Test : Element access of right evaluated addition expression\n"
//              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT1 ).name() << "\n"
//              << "   Right-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT2 ).name() << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//
//       if( !equal( ( eval( olhs_ ) + orhs_ )(o,m,n), ( eval( reflhs_ ) + refrhs_ )(o,m,n) ) ||
//           !equal( ( eval( olhs_ ) + orhs_ ).at(o,m,n), ( eval( reflhs_ ) + refrhs_ ).at(o,m,n) ) ) {
//          std::ostringstream oss;
//          oss << " Test : Element access of left evaluated addition expression\n"
//              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT1 ).name() << "\n"
//              << "   Right-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT2 ).name() << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//
//       if( !equal( ( eval( olhs_ ) + eval( orhs_ ) )(o,m,n), ( eval( reflhs_ ) + eval( refrhs_ ) )(o,m,n) ) ||
//           !equal( ( eval( olhs_ ) + eval( orhs_ ) ).at(o,m,n), ( eval( reflhs_ ) + eval( refrhs_ ) ).at(o,m,n) ) ) {
//          std::ostringstream oss;
//          oss << " Test : Element access of fully evaluated addition expression\n"
//              << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
//              << " Details:\n"
//              << "   Random seed = " << blaze::getSeed() << "\n"
//              << "   Left-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT1 ).name() << "\n"
//              << "   Right-hand side column-major dense tensor type:\n"
//              << "     " << typeid( OMT2 ).name() << "\n";
//          throw std::runtime_error( oss.str() );
//       }
//    }
//
//    try {
//       ( olhs_ + orhs_ ).at( 0UL, lhs_.columns() );
//
//       std::ostringstream oss;
//       oss << " Test : Checked element access of addition expression\n"
//           << " Error: Out-of-bound access succeeded\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Left-hand side column-major dense tensor type:\n"
//           << "     " << typeid( OMT1 ).name() << "\n"
//           << "   Right-hand side column-major dense tensor type:\n"
//           << "     " << typeid( OMT2 ).name() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//    catch( std::out_of_range& ) {}
//
//    try {
//       ( olhs_ + orhs_ ).at( lhs_.rows(), 0UL );
//
//       std::ostringstream oss;
//       oss << " Test : Checked element access of addition expression\n"
//           << " Error: Out-of-bound access succeeded\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Left-hand side column-major dense tensor type:\n"
//           << "     " << typeid( OMT1 ).name() << "\n"
//           << "   Right-hand side column-major dense tensor type:\n"
//           << "     " << typeid( OMT2 ).name() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//    catch( std::out_of_range& ) {}
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the plain dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the plain tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testBasicOperation()
{
#if BLAZETEST_MATHTEST_TEST_BASIC_OPERATION
   if( BLAZETEST_MATHTEST_TEST_BASIC_OPERATION > 1 )
   {
      //=====================================================================================
      // Addition
      //=====================================================================================

      // Addition with the given tensors
      {
         test_  = "Addition with the given tensors";
         error_ = "Failed addition operation";

         try {
            initResults();
            dres_   = lhs_ + rhs_;
//             odres_  = lhs_ + rhs_;
//             sres_   = lhs_ + rhs_;
//             osres_  = lhs_ + rhs_;
            refres_ = reflhs_ + refrhs_;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   = lhs_ + orhs_;
//             odres_  = lhs_ + orhs_;
//             sres_   = lhs_ + orhs_;
//             osres_  = lhs_ + orhs_;
//             refres_ = reflhs_ + refrhs_;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = olhs_ + rhs_;
//             odres_  = olhs_ + rhs_;
//             sres_   = olhs_ + rhs_;
//             osres_  = olhs_ + rhs_;
//             refres_ = reflhs_ + refrhs_;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = olhs_ + orhs_;
//             odres_  = olhs_ + orhs_;
//             sres_   = olhs_ + orhs_;
//             osres_  = olhs_ + orhs_;
//             refres_ = reflhs_ + refrhs_;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Addition with evaluated tensors
      {
         test_  = "Addition with evaluated tensors";
         error_ = "Failed addition operation";

         try {
            initResults();
            dres_   = eval( lhs_ ) + eval( rhs_ );
//             odres_  = eval( lhs_ ) + eval( rhs_ );
//             sres_   = eval( lhs_ ) + eval( rhs_ );
//             osres_  = eval( lhs_ ) + eval( rhs_ );
            refres_ = eval( reflhs_ ) + eval( refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   = eval( lhs_ ) + eval( orhs_ );
//             odres_  = eval( lhs_ ) + eval( orhs_ );
//             sres_   = eval( lhs_ ) + eval( orhs_ );
//             osres_  = eval( lhs_ ) + eval( orhs_ );
//             refres_ = eval( reflhs_ ) + eval( refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = eval( olhs_ ) + eval( rhs_ );
//             odres_  = eval( olhs_ ) + eval( rhs_ );
//             sres_   = eval( olhs_ ) + eval( rhs_ );
//             osres_  = eval( olhs_ ) + eval( rhs_ );
//             refres_ = eval( reflhs_ ) + eval( refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = eval( olhs_ ) + eval( orhs_ );
//             odres_  = eval( olhs_ ) + eval( orhs_ );
//             sres_   = eval( olhs_ ) + eval( orhs_ );
//             osres_  = eval( olhs_ ) + eval( orhs_ );
//             refres_ = eval( reflhs_ ) + eval( refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Addition with addition assignment
      //=====================================================================================

      // Addition with addition assignment with the given tensors
      {
         test_  = "Addition with addition assignment with the given tensors";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += lhs_ + rhs_;
//             odres_  += lhs_ + rhs_;
//             sres_   += lhs_ + rhs_;
//             osres_  += lhs_ + rhs_;
            refres_ += reflhs_ + refrhs_;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   += lhs_ + orhs_;
//             odres_  += lhs_ + orhs_;
//             sres_   += lhs_ + orhs_;
//             osres_  += lhs_ + orhs_;
//             refres_ += reflhs_ + refrhs_;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += olhs_ + rhs_;
//             odres_  += olhs_ + rhs_;
//             sres_   += olhs_ + rhs_;
//             osres_  += olhs_ + rhs_;
//             refres_ += reflhs_ + refrhs_;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += olhs_ + orhs_;
//             odres_  += olhs_ + orhs_;
//             sres_   += olhs_ + orhs_;
//             osres_  += olhs_ + orhs_;
//             refres_ += reflhs_ + refrhs_;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Addition with addition assignment with evaluated tensors
      {
         test_  = "Addition with addition assignment with evaluated tensors";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += eval( lhs_ ) + eval( rhs_ );
//             odres_  += eval( lhs_ ) + eval( rhs_ );
//             sres_   += eval( lhs_ ) + eval( rhs_ );
//             osres_  += eval( lhs_ ) + eval( rhs_ );
            refres_ += eval( reflhs_ ) + eval( refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   += eval( lhs_ ) + eval( orhs_ );
//             odres_  += eval( lhs_ ) + eval( orhs_ );
//             sres_   += eval( lhs_ ) + eval( orhs_ );
//             osres_  += eval( lhs_ ) + eval( orhs_ );
//             refres_ += eval( reflhs_ ) + eval( refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += eval( olhs_ ) + eval( rhs_ );
//             odres_  += eval( olhs_ ) + eval( rhs_ );
//             sres_   += eval( olhs_ ) + eval( rhs_ );
//             osres_  += eval( olhs_ ) + eval( rhs_ );
//             refres_ += eval( reflhs_ ) + eval( refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += eval( olhs_ ) + eval( orhs_ );
//             odres_  += eval( olhs_ ) + eval( orhs_ );
//             sres_   += eval( olhs_ ) + eval( orhs_ );
//             osres_  += eval( olhs_ ) + eval( orhs_ );
//             refres_ += eval( reflhs_ ) + eval( refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Addition with subtraction assignment with the given tensors
      //=====================================================================================

      // Addition with subtraction assignment with the given tensors
      {
         test_  = "Addition with subtraction assignment with the given tensors";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= lhs_ + rhs_;
//             odres_  -= lhs_ + rhs_;
//             sres_   -= lhs_ + rhs_;
//             osres_  -= lhs_ + rhs_;
            refres_ -= reflhs_ + refrhs_;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   -= lhs_ + orhs_;
//             odres_  -= lhs_ + orhs_;
//             sres_   -= lhs_ + orhs_;
//             osres_  -= lhs_ + orhs_;
//             refres_ -= reflhs_ + refrhs_;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= olhs_ + rhs_;
//             odres_  -= olhs_ + rhs_;
//             sres_   -= olhs_ + rhs_;
//             osres_  -= olhs_ + rhs_;
//             refres_ -= reflhs_ + refrhs_;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= olhs_ + orhs_;
//             odres_  -= olhs_ + orhs_;
//             sres_   -= olhs_ + orhs_;
//             osres_  -= olhs_ + orhs_;
//             refres_ -= reflhs_ + refrhs_;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Addition with subtraction assignment with evaluated tensors
      {
         test_  = "Addition with subtraction assignment with evaluated tensors";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= eval( lhs_ ) + eval( rhs_ );
//             odres_  -= eval( lhs_ ) + eval( rhs_ );
//             sres_   -= eval( lhs_ ) + eval( rhs_ );
//             osres_  -= eval( lhs_ ) + eval( rhs_ );
            refres_ -= eval( reflhs_ ) + eval( refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   -= eval( lhs_ ) + eval( orhs_ );
//             odres_  -= eval( lhs_ ) + eval( orhs_ );
//             sres_   -= eval( lhs_ ) + eval( orhs_ );
//             osres_  -= eval( lhs_ ) + eval( orhs_ );
//             refres_ -= eval( reflhs_ ) + eval( refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= eval( olhs_ ) + eval( rhs_ );
//             odres_  -= eval( olhs_ ) + eval( rhs_ );
//             sres_   -= eval( olhs_ ) + eval( rhs_ );
//             osres_  -= eval( olhs_ ) + eval( rhs_ );
//             refres_ -= eval( reflhs_ ) + eval( refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= eval( olhs_ ) + eval( orhs_ );
//             odres_  -= eval( olhs_ ) + eval( orhs_ );
//             sres_   -= eval( olhs_ ) + eval( orhs_ );
//             osres_  -= eval( olhs_ ) + eval( orhs_ );
//             refres_ -= eval( reflhs_ ) + eval( refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Addition with Schur product assignment
      //=====================================================================================

      // Addition with Schur product assignment with the given tensors
      {
         test_  = "Addition with Schur product assignment with the given tensors";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= lhs_ + rhs_;
//             odres_  %= lhs_ + rhs_;
//             sres_   %= lhs_ + rhs_;
//             osres_  %= lhs_ + rhs_;
            refres_ %= reflhs_ + refrhs_;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   %= lhs_ + orhs_;
//             odres_  %= lhs_ + orhs_;
//             sres_   %= lhs_ + orhs_;
//             osres_  %= lhs_ + orhs_;
//             refres_ %= reflhs_ + refrhs_;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= olhs_ + rhs_;
//             odres_  %= olhs_ + rhs_;
//             sres_   %= olhs_ + rhs_;
//             osres_  %= olhs_ + rhs_;
//             refres_ %= reflhs_ + refrhs_;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= olhs_ + orhs_;
//             odres_  %= olhs_ + orhs_;
//             sres_   %= olhs_ + orhs_;
//             osres_  %= olhs_ + orhs_;
//             refres_ %= reflhs_ + refrhs_;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Addition with Schur product assignment with evaluated tensors
      {
         test_  = "Addition with Schur product assignment with evaluated tensors";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= eval( lhs_ ) + eval( rhs_ );
//             odres_  %= eval( lhs_ ) + eval( rhs_ );
//             sres_   %= eval( lhs_ ) + eval( rhs_ );
//             osres_  %= eval( lhs_ ) + eval( rhs_ );
            refres_ %= eval( reflhs_ ) + eval( refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   %= eval( lhs_ ) + eval( orhs_ );
//             odres_  %= eval( lhs_ ) + eval( orhs_ );
//             sres_   %= eval( lhs_ ) + eval( orhs_ );
//             osres_  %= eval( lhs_ ) + eval( orhs_ );
//             refres_ %= eval( reflhs_ ) + eval( refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= eval( olhs_ ) + eval( rhs_ );
//             odres_  %= eval( olhs_ ) + eval( rhs_ );
//             sres_   %= eval( olhs_ ) + eval( rhs_ );
//             osres_  %= eval( olhs_ ) + eval( rhs_ );
//             refres_ %= eval( reflhs_ ) + eval( refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= eval( olhs_ ) + eval( orhs_ );
//             odres_  %= eval( olhs_ ) + eval( orhs_ );
//             sres_   %= eval( olhs_ ) + eval( orhs_ );
//             osres_  %= eval( olhs_ ) + eval( orhs_ );
//             refres_ %= eval( reflhs_ ) + eval( refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the negated dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the negated tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testNegatedOperation()
{
#if BLAZETEST_MATHTEST_TEST_NEGATED_OPERATION
   if( BLAZETEST_MATHTEST_TEST_NEGATED_OPERATION > 1 )
   {
      //=====================================================================================
      // Negated addition
      //=====================================================================================

      // Negated addition with the given tensors
      {
         test_  = "Negated addition with the given tensors";
         error_ = "Failed addition operation";

         try {
            initResults();
            dres_   = -( lhs_ + rhs_ );
//             odres_  = -( lhs_ + rhs_ );
//             sres_   = -( lhs_ + rhs_ );
//             osres_  = -( lhs_ + rhs_ );
            refres_ = -( reflhs_ + refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   = -( lhs_ + orhs_ );
//             odres_  = -( lhs_ + orhs_ );
//             sres_   = -( lhs_ + orhs_ );
//             osres_  = -( lhs_ + orhs_ );
//             refres_ = -( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = -( olhs_ + rhs_ );
//             odres_  = -( olhs_ + rhs_ );
//             sres_   = -( olhs_ + rhs_ );
//             osres_  = -( olhs_ + rhs_ );
//             refres_ = -( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = -( olhs_ + orhs_ );
//             odres_  = -( olhs_ + orhs_ );
//             sres_   = -( olhs_ + orhs_ );
//             osres_  = -( olhs_ + orhs_ );
//             refres_ = -( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Negated addition with evaluated tensors
      {
         test_  = "Negated addition with evaluated tensors";
         error_ = "Failed addition operation";

         try {
            initResults();
            dres_   = -( eval( lhs_ ) + eval( rhs_ ) );
//             odres_  = -( eval( lhs_ ) + eval( rhs_ ) );
//             sres_   = -( eval( lhs_ ) + eval( rhs_ ) );
//             osres_  = -( eval( lhs_ ) + eval( rhs_ ) );
            refres_ = -( eval( reflhs_ ) + eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   = -( eval( lhs_ ) + eval( orhs_ ) );
//             odres_  = -( eval( lhs_ ) + eval( orhs_ ) );
//             sres_   = -( eval( lhs_ ) + eval( orhs_ ) );
//             osres_  = -( eval( lhs_ ) + eval( orhs_ ) );
//             refres_ = -( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = -( eval( olhs_ ) + eval( rhs_ ) );
//             odres_  = -( eval( olhs_ ) + eval( rhs_ ) );
//             sres_   = -( eval( olhs_ ) + eval( rhs_ ) );
//             osres_  = -( eval( olhs_ ) + eval( rhs_ ) );
//             refres_ = -( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = -( eval( olhs_ ) + eval( orhs_ ) );
//             odres_  = -( eval( olhs_ ) + eval( orhs_ ) );
//             sres_   = -( eval( olhs_ ) + eval( orhs_ ) );
//             osres_  = -( eval( olhs_ ) + eval( orhs_ ) );
//             refres_ = -( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Negated addition with addition assignment
      //=====================================================================================

      // Negated addition with addition assignment with the given tensors
      {
         test_  = "Negated addition with addition assignment with the given tensors";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += -( lhs_ + rhs_ );
//             odres_  += -( lhs_ + rhs_ );
//             sres_   += -( lhs_ + rhs_ );
//             osres_  += -( lhs_ + rhs_ );
            refres_ += -( reflhs_ + refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   += -( lhs_ + orhs_ );
//             odres_  += -( lhs_ + orhs_ );
//             sres_   += -( lhs_ + orhs_ );
//             osres_  += -( lhs_ + orhs_ );
//             refres_ += -( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += -( olhs_ + rhs_ );
//             odres_  += -( olhs_ + rhs_ );
//             sres_   += -( olhs_ + rhs_ );
//             osres_  += -( olhs_ + rhs_ );
//             refres_ += -( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += -( olhs_ + orhs_ );
//             odres_  += -( olhs_ + orhs_ );
//             sres_   += -( olhs_ + orhs_ );
//             osres_  += -( olhs_ + orhs_ );
//             refres_ += -( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Negated addition with addition assignment with the given tensors
      {
         test_  = "Negated addition with addition assignment with evaluated tensors";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += -( eval( lhs_ ) + eval( rhs_ ) );
//             odres_  += -( eval( lhs_ ) + eval( rhs_ ) );
//             sres_   += -( eval( lhs_ ) + eval( rhs_ ) );
//             osres_  += -( eval( lhs_ ) + eval( rhs_ ) );
            refres_ += -( eval( reflhs_ ) + eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   += -( eval( lhs_ ) + eval( orhs_ ) );
//             odres_  += -( eval( lhs_ ) + eval( orhs_ ) );
//             sres_   += -( eval( lhs_ ) + eval( orhs_ ) );
//             osres_  += -( eval( lhs_ ) + eval( orhs_ ) );
//             refres_ += -( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += -( eval( olhs_ ) + eval( rhs_ ) );
//             odres_  += -( eval( olhs_ ) + eval( rhs_ ) );
//             sres_   += -( eval( olhs_ ) + eval( rhs_ ) );
//             osres_  += -( eval( olhs_ ) + eval( rhs_ ) );
//             refres_ += -( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += -( eval( olhs_ ) + eval( orhs_ ) );
//             odres_  += -( eval( olhs_ ) + eval( orhs_ ) );
//             sres_   += -( eval( olhs_ ) + eval( orhs_ ) );
//             osres_  += -( eval( olhs_ ) + eval( orhs_ ) );
//             refres_ += -( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Negated addition with subtraction assignment
      //=====================================================================================

      // Negated addition with subtraction assignment with the given tensors
      {
         test_  = "Negated addition with subtraction assignment with the given tensors";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= -( lhs_ + rhs_ );
//             odres_  -= -( lhs_ + rhs_ );
//             sres_   -= -( lhs_ + rhs_ );
//             osres_  -= -( lhs_ + rhs_ );
            refres_ -= -( reflhs_ + refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   -= -( lhs_ + orhs_ );
//             odres_  -= -( lhs_ + orhs_ );
//             sres_   -= -( lhs_ + orhs_ );
//             osres_  -= -( lhs_ + orhs_ );
//             refres_ -= -( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= -( olhs_ + rhs_ );
//             odres_  -= -( olhs_ + rhs_ );
//             sres_   -= -( olhs_ + rhs_ );
//             osres_  -= -( olhs_ + rhs_ );
//             refres_ -= -( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= -( olhs_ + orhs_ );
//             odres_  -= -( olhs_ + orhs_ );
//             sres_   -= -( olhs_ + orhs_ );
//             osres_  -= -( olhs_ + orhs_ );
//             refres_ -= -( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Negated addition with subtraction assignment with evaluated tensors
      {
         test_  = "Negated addition with subtraction assignment with evaluated tensors";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= -( eval( lhs_ ) + eval( rhs_ ) );
//             odres_  -= -( eval( lhs_ ) + eval( rhs_ ) );
//             sres_   -= -( eval( lhs_ ) + eval( rhs_ ) );
//             osres_  -= -( eval( lhs_ ) + eval( rhs_ ) );
            refres_ -= -( eval( reflhs_ ) + eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   -= -( eval( lhs_ ) + eval( orhs_ ) );
//             odres_  -= -( eval( lhs_ ) + eval( orhs_ ) );
//             sres_   -= -( eval( lhs_ ) + eval( orhs_ ) );
//             osres_  -= -( eval( lhs_ ) + eval( orhs_ ) );
//             refres_ -= -( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= -( eval( olhs_ ) + eval( rhs_ ) );
//             odres_  -= -( eval( olhs_ ) + eval( rhs_ ) );
//             sres_   -= -( eval( olhs_ ) + eval( rhs_ ) );
//             osres_  -= -( eval( olhs_ ) + eval( rhs_ ) );
//             refres_ -= -( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= -( eval( olhs_ ) + eval( orhs_ ) );
//             odres_  -= -( eval( olhs_ ) + eval( orhs_ ) );
//             sres_   -= -( eval( olhs_ ) + eval( orhs_ ) );
//             osres_  -= -( eval( olhs_ ) + eval( orhs_ ) );
//             refres_ -= -( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Negated addition with Schur product assignment
      //=====================================================================================

      // Negated addition with Schur product assignment with the given tensors
      {
         test_  = "Negated addition with Schur product assignment with the given tensors";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= -( lhs_ + rhs_ );
//             odres_  %= -( lhs_ + rhs_ );
//             sres_   %= -( lhs_ + rhs_ );
//             osres_  %= -( lhs_ + rhs_ );
            refres_ %= -( reflhs_ + refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   %= -( lhs_ + orhs_ );
//             odres_  %= -( lhs_ + orhs_ );
//             sres_   %= -( lhs_ + orhs_ );
//             osres_  %= -( lhs_ + orhs_ );
//             refres_ %= -( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= -( olhs_ + rhs_ );
//             odres_  %= -( olhs_ + rhs_ );
//             sres_   %= -( olhs_ + rhs_ );
//             osres_  %= -( olhs_ + rhs_ );
//             refres_ %= -( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= -( olhs_ + orhs_ );
//             odres_  %= -( olhs_ + orhs_ );
//             sres_   %= -( olhs_ + orhs_ );
//             osres_  %= -( olhs_ + orhs_ );
//             refres_ %= -( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Negated addition with Schur product assignment with the given tensors
      {
         test_  = "Negated addition with Schur product assignment with evaluated tensors";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= -( eval( lhs_ ) + eval( rhs_ ) );
//             odres_  %= -( eval( lhs_ ) + eval( rhs_ ) );
//             sres_   %= -( eval( lhs_ ) + eval( rhs_ ) );
//             osres_  %= -( eval( lhs_ ) + eval( rhs_ ) );
            refres_ %= -( eval( reflhs_ ) + eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   %= -( eval( lhs_ ) + eval( orhs_ ) );
//             odres_  %= -( eval( lhs_ ) + eval( orhs_ ) );
//             sres_   %= -( eval( lhs_ ) + eval( orhs_ ) );
//             osres_  %= -( eval( lhs_ ) + eval( orhs_ ) );
//             refres_ %= -( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= -( eval( olhs_ ) + eval( rhs_ ) );
//             odres_  %= -( eval( olhs_ ) + eval( rhs_ ) );
//             sres_   %= -( eval( olhs_ ) + eval( rhs_ ) );
//             osres_  %= -( eval( olhs_ ) + eval( rhs_ ) );
//             refres_ %= -( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= -( eval( olhs_ ) + eval( orhs_ ) );
//             odres_  %= -( eval( olhs_ ) + eval( orhs_ ) );
//             sres_   %= -( eval( olhs_ ) + eval( orhs_ ) );
//             osres_  %= -( eval( olhs_ ) + eval( orhs_ ) );
//             refres_ %= -( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the scaled dense tensor/dense tensor addition.
//
// \param scalar The scalar value.
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the scaled tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
template< typename T >    // Type of the scalar
void OperationTest<MT1,MT2>::testScaledOperation( T scalar )
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
            dres_   = lhs_ + rhs_;
//             odres_  = dres_;
//             sres_   = dres_;
//             osres_  = dres_;
            refres_ = dres_;

            dres_   *= scalar;
//             odres_  *= scalar;
//             sres_   *= scalar;
//             osres_  *= scalar;
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

         checkResults<MT1,MT2>();
      }


      //=====================================================================================
      // Self-scaling (M=M*s)
      //=====================================================================================

      // Self-scaling (M=M*s)
      {
         test_ = "Self-scaling (M=M*s)";

         try {
            dres_   = lhs_ + rhs_;
//             odres_  = dres_;
//             sres_   = dres_;
//             osres_  = dres_;
            refres_ = dres_;

            dres_   = dres_   * scalar;
//             odres_  = odres_  * scalar;
//             sres_   = sres_   * scalar;
//             osres_  = osres_  * scalar;
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

         checkResults<MT1,MT2>();
      }


      //=====================================================================================
      // Self-scaling (M=s*M)
      //=====================================================================================

      // Self-scaling (M=s*M)
      {
         test_ = "Self-scaling (M=s*M)";

         try {
            dres_   = lhs_ + rhs_;
//             odres_  = dres_;
//             sres_   = dres_;
//             osres_  = dres_;
            refres_ = dres_;

            dres_   = scalar * dres_;
//             odres_  = scalar * odres_;
//             sres_   = scalar * sres_;
//             osres_  = scalar * osres_;
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

         checkResults<MT1,MT2>();
      }


      //=====================================================================================
      // Self-scaling (M/=s)
      //=====================================================================================

      // Self-scaling (M/=s)
      {
         test_ = "Self-scaling (M/=s)";

         try {
            dres_   = lhs_ + rhs_;
//             odres_  = dres_;
//             sres_   = dres_;
//             osres_  = dres_;
            refres_ = dres_;

            dres_   /= scalar;
//             odres_  /= scalar;
//             sres_   /= scalar;
//             osres_  /= scalar;
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

         checkResults<MT1,MT2>();
      }


      //=====================================================================================
      // Self-scaling (M=M/s)
      //=====================================================================================

      // Self-scaling (M=M/s)
      {
         test_ = "Self-scaling (M=M/s)";

         try {
            dres_   = lhs_ + rhs_;
//             odres_  = dres_;
//             sres_   = dres_;
//             osres_  = dres_;
            refres_ = dres_;

            dres_   = dres_   / scalar;
//             odres_  = odres_  / scalar;
//             sres_   = sres_   / scalar;
//             osres_  = osres_  / scalar;
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

         checkResults<MT1,MT2>();
      }


      //=====================================================================================
      // Scaled addition (s*OP)
      //=====================================================================================

      // Scaled addition with the given tensors
      {
         test_  = "Scaled addition with the given tensors (s*OP)";
         error_ = "Failed addition operation";

         try {
            initResults();
            dres_   = scalar * ( lhs_ + rhs_ );
//             odres_  = scalar * ( lhs_ + rhs_ );
//             sres_   = scalar * ( lhs_ + rhs_ );
//             osres_  = scalar * ( lhs_ + rhs_ );
            refres_ = scalar * ( reflhs_ + refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   = scalar * ( lhs_ + orhs_ );
//             odres_  = scalar * ( lhs_ + orhs_ );
//             sres_   = scalar * ( lhs_ + orhs_ );
//             osres_  = scalar * ( lhs_ + orhs_ );
//             refres_ = scalar * ( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = scalar * ( olhs_ + rhs_ );
//             odres_  = scalar * ( olhs_ + rhs_ );
//             sres_   = scalar * ( olhs_ + rhs_ );
//             osres_  = scalar * ( olhs_ + rhs_ );
//             refres_ = scalar * ( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = scalar * ( olhs_ + orhs_ );
//             odres_  = scalar * ( olhs_ + orhs_ );
//             sres_   = scalar * ( olhs_ + orhs_ );
//             osres_  = scalar * ( olhs_ + orhs_ );
//             refres_ = scalar * ( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Scaled addition with evaluated tensors
      {
         test_  = "Scaled addition with evaluated tensors (s*OP)";
         error_ = "Failed addition operation";

         try {
            initResults();
            dres_   = scalar * ( eval( lhs_ ) + eval( rhs_ ) );
//             odres_  = scalar * ( eval( lhs_ ) + eval( rhs_ ) );
//             sres_   = scalar * ( eval( lhs_ ) + eval( rhs_ ) );
//             osres_  = scalar * ( eval( lhs_ ) + eval( rhs_ ) );
            refres_ = scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   = scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             odres_  = scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             sres_   = scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             osres_  = scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             refres_ = scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             odres_  = scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             sres_   = scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             osres_  = scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             refres_ = scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             odres_  = scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             sres_   = scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             osres_  = scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             refres_ = scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Scaled addition (OP*s)
      //=====================================================================================

      // Scaled addition with the given tensors
      {
         test_  = "Scaled addition with the given tensors (OP*s)";
         error_ = "Failed addition operation";

         try {
            initResults();
            dres_   = ( lhs_ + rhs_ ) * scalar;
//             odres_  = ( lhs_ + rhs_ ) * scalar;
//             sres_   = ( lhs_ + rhs_ ) * scalar;
//             osres_  = ( lhs_ + rhs_ ) * scalar;
            refres_ = ( reflhs_ + refrhs_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   = ( lhs_ + orhs_ ) * scalar;
//             odres_  = ( lhs_ + orhs_ ) * scalar;
//             sres_   = ( lhs_ + orhs_ ) * scalar;
//             osres_  = ( lhs_ + orhs_ ) * scalar;
//             refres_ = ( reflhs_ + refrhs_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = ( olhs_ + rhs_ ) * scalar;
//             odres_  = ( olhs_ + rhs_ ) * scalar;
//             sres_   = ( olhs_ + rhs_ ) * scalar;
//             osres_  = ( olhs_ + rhs_ ) * scalar;
//             refres_ = ( reflhs_ + refrhs_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = ( olhs_ + orhs_ ) * scalar;
//             odres_  = ( olhs_ + orhs_ ) * scalar;
//             sres_   = ( olhs_ + orhs_ ) * scalar;
//             osres_  = ( olhs_ + orhs_ ) * scalar;
//             refres_ = ( reflhs_ + refrhs_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Scaled addition with evaluated tensors
      {
         test_  = "Scaled addition with evaluated tensors (OP*s)";
         error_ = "Failed addition operation";

         try {
            initResults();
            dres_   = ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
//             odres_  = ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
//             sres_   = ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
//             osres_  = ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
            refres_ = ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   = ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             odres_  = ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             sres_   = ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             osres_  = ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             refres_ = ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             odres_  = ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             sres_   = ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             osres_  = ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             refres_ = ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             odres_  = ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             sres_   = ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             osres_  = ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             refres_ = ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Scaled addition (OP/s)
      //=====================================================================================

      // Scaled addition with the given tensors
      {
         test_  = "Scaled addition with the given tensors (OP/s)";
         error_ = "Failed addition operation";

         try {
            initResults();
            dres_   = ( lhs_ + rhs_ ) / scalar;
//             odres_  = ( lhs_ + rhs_ ) / scalar;
//             sres_   = ( lhs_ + rhs_ ) / scalar;
//             osres_  = ( lhs_ + rhs_ ) / scalar;
            refres_ = ( reflhs_ + refrhs_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   = ( lhs_ + orhs_ ) / scalar;
//             odres_  = ( lhs_ + orhs_ ) / scalar;
//             sres_   = ( lhs_ + orhs_ ) / scalar;
//             osres_  = ( lhs_ + orhs_ ) / scalar;
//             refres_ = ( reflhs_ + refrhs_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = ( olhs_ + rhs_ ) / scalar;
//             odres_  = ( olhs_ + rhs_ ) / scalar;
//             sres_   = ( olhs_ + rhs_ ) / scalar;
//             osres_  = ( olhs_ + rhs_ ) / scalar;
//             refres_ = ( reflhs_ + refrhs_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = ( olhs_ + orhs_ ) / scalar;
//             odres_  = ( olhs_ + orhs_ ) / scalar;
//             sres_   = ( olhs_ + orhs_ ) / scalar;
//             osres_  = ( olhs_ + orhs_ ) / scalar;
//             refres_ = ( reflhs_ + refrhs_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Scaled addition with evaluated tensors
      {
         test_  = "Scaled addition with evaluated tensors (OP/s)";
         error_ = "Failed addition operation";

         try {
            initResults();
            dres_   = ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
//             odres_  = ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
//             sres_   = ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
//             osres_  = ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
            refres_ = ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   = ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             odres_  = ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             sres_   = ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             osres_  = ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             refres_ = ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             odres_  = ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             sres_   = ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             osres_  = ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             refres_ = ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             odres_  = ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             sres_   = ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             osres_  = ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             refres_ = ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Scaled addition with addition assignment (s*OP)
      //=====================================================================================

      // Scaled addition with addition assignment with the given tensors
      {
         test_  = "Scaled addition with addition assignment with the given tensors (s*OP)";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += scalar * ( lhs_ + rhs_ );
//             odres_  += scalar * ( lhs_ + rhs_ );
//             sres_   += scalar * ( lhs_ + rhs_ );
//             osres_  += scalar * ( lhs_ + rhs_ );
            refres_ += scalar * ( reflhs_ + refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   += scalar * ( lhs_ + orhs_ );
//             odres_  += scalar * ( lhs_ + orhs_ );
//             sres_   += scalar * ( lhs_ + orhs_ );
//             osres_  += scalar * ( lhs_ + orhs_ );
//             refres_ += scalar * ( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += scalar * ( olhs_ + rhs_ );
//             odres_  += scalar * ( olhs_ + rhs_ );
//             sres_   += scalar * ( olhs_ + rhs_ );
//             osres_  += scalar * ( olhs_ + rhs_ );
//             refres_ += scalar * ( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += scalar * ( olhs_ + orhs_ );
//             odres_  += scalar * ( olhs_ + orhs_ );
//             sres_   += scalar * ( olhs_ + orhs_ );
//             osres_  += scalar * ( olhs_ + orhs_ );
//             refres_ += scalar * ( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Scaled addition with addition assignment with evaluated tensors
      {
         test_  = "Scaled addition with addition assignment with evaluated tensors (s*OP)";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += scalar * ( eval( lhs_ ) + eval( rhs_ ) );
//             odres_  += scalar * ( eval( lhs_ ) + eval( rhs_ ) );
//             sres_   += scalar * ( eval( lhs_ ) + eval( rhs_ ) );
//             osres_  += scalar * ( eval( lhs_ ) + eval( rhs_ ) );
            refres_ += scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   += scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             odres_  += scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             sres_   += scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             osres_  += scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             refres_ += scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             odres_  += scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             sres_   += scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             osres_  += scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             refres_ += scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             odres_  += scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             sres_   += scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             osres_  += scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             refres_ += scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Scaled addition with addition assignment (OP*s)
      //=====================================================================================

      // Scaled addition with addition assignment with the given tensors
      {
         test_  = "Scaled addition with addition assignment with the given tensors (OP*s)";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += ( lhs_ + rhs_ ) * scalar;
//             odres_  += ( lhs_ + rhs_ ) * scalar;
//             sres_   += ( lhs_ + rhs_ ) * scalar;
//             osres_  += ( lhs_ + rhs_ ) * scalar;
            refres_ += ( reflhs_ + refrhs_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   += ( lhs_ + orhs_ ) * scalar;
//             odres_  += ( lhs_ + orhs_ ) * scalar;
//             sres_   += ( lhs_ + orhs_ ) * scalar;
//             osres_  += ( lhs_ + orhs_ ) * scalar;
//             refres_ += ( reflhs_ + refrhs_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += ( olhs_ + rhs_ ) * scalar;
//             odres_  += ( olhs_ + rhs_ ) * scalar;
//             sres_   += ( olhs_ + rhs_ ) * scalar;
//             osres_  += ( olhs_ + rhs_ ) * scalar;
//             refres_ += ( reflhs_ + refrhs_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += ( olhs_ + orhs_ ) * scalar;
//             odres_  += ( olhs_ + orhs_ ) * scalar;
//             sres_   += ( olhs_ + orhs_ ) * scalar;
//             osres_  += ( olhs_ + orhs_ ) * scalar;
//             refres_ += ( reflhs_ + refrhs_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Scaled addition with addition assignment with evaluated tensors
      {
         test_  = "Scaled addition with addition assignment with evaluated tensors (OP*s)";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
//             odres_  += ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
//             sres_   += ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
//             osres_  += ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
            refres_ += ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   += ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             odres_  += ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             sres_   += ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             osres_  += ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             refres_ += ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             odres_  += ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             sres_   += ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             osres_  += ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             refres_ += ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             odres_  += ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             sres_   += ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             osres_  += ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             refres_ += ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Scaled addition with addition assignment (OP/s)
      //=====================================================================================

      // Scaled addition with addition assignment with the given tensors
      {
         test_  = "Scaled addition with addition assignment with the given tensors (OP/s)";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += ( lhs_ + rhs_ ) / scalar;
//             odres_  += ( lhs_ + rhs_ ) / scalar;
//             sres_   += ( lhs_ + rhs_ ) / scalar;
//             osres_  += ( lhs_ + rhs_ ) / scalar;
            refres_ += ( reflhs_ + refrhs_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   += ( lhs_ + orhs_ ) / scalar;
//             odres_  += ( lhs_ + orhs_ ) / scalar;
//             sres_   += ( lhs_ + orhs_ ) / scalar;
//             osres_  += ( lhs_ + orhs_ ) / scalar;
//             refres_ += ( reflhs_ + refrhs_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += ( olhs_ + rhs_ ) / scalar;
//             odres_  += ( olhs_ + rhs_ ) / scalar;
//             sres_   += ( olhs_ + rhs_ ) / scalar;
//             osres_  += ( olhs_ + rhs_ ) / scalar;
//             refres_ += ( reflhs_ + refrhs_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += ( olhs_ + orhs_ ) / scalar;
//             odres_  += ( olhs_ + orhs_ ) / scalar;
//             sres_   += ( olhs_ + orhs_ ) / scalar;
//             osres_  += ( olhs_ + orhs_ ) / scalar;
//             refres_ += ( reflhs_ + refrhs_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Scaled addition with addition assignment with evaluated tensors
      {
         test_  = "Scaled addition with addition assignment with evaluated tensors (OP/s)";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
//             odres_  += ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
//             sres_   += ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
//             osres_  += ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
            refres_ += ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   += ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             odres_  += ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             sres_   += ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             osres_  += ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             refres_ += ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             odres_  += ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             sres_   += ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             osres_  += ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             refres_ += ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             odres_  += ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             sres_   += ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             osres_  += ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             refres_ += ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Scaled addition with subtraction assignment (s*OP)
      //=====================================================================================

      // Scaled addition with subtraction assignment with the given tensors
      {
         test_  = "Scaled addition with subtraction assignment with the given tensors (s*OP)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= scalar * ( lhs_ + rhs_ );
//             odres_  -= scalar * ( lhs_ + rhs_ );
//             sres_   -= scalar * ( lhs_ + rhs_ );
//             osres_  -= scalar * ( lhs_ + rhs_ );
            refres_ -= scalar * ( reflhs_ + refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   -= scalar * ( lhs_ + orhs_ );
//             odres_  -= scalar * ( lhs_ + orhs_ );
//             sres_   -= scalar * ( lhs_ + orhs_ );
//             osres_  -= scalar * ( lhs_ + orhs_ );
//             refres_ -= scalar * ( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= scalar * ( olhs_ + rhs_ );
//             odres_  -= scalar * ( olhs_ + rhs_ );
//             sres_   -= scalar * ( olhs_ + rhs_ );
//             osres_  -= scalar * ( olhs_ + rhs_ );
//             refres_ -= scalar * ( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= scalar * ( olhs_ + orhs_ );
//             odres_  -= scalar * ( olhs_ + orhs_ );
//             sres_   -= scalar * ( olhs_ + orhs_ );
//             osres_  -= scalar * ( olhs_ + orhs_ );
//             refres_ -= scalar * ( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Scaled addition with subtraction assignment with evaluated tensors
      {
         test_  = "Scaled addition with subtraction assignment with evaluated tensors (s*OP)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= scalar * ( eval( lhs_ ) + eval( rhs_ ) );
//             odres_  -= scalar * ( eval( lhs_ ) + eval( rhs_ ) );
//             sres_   -= scalar * ( eval( lhs_ ) + eval( rhs_ ) );
//             osres_  -= scalar * ( eval( lhs_ ) + eval( rhs_ ) );
            refres_ -= scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   -= scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             odres_  -= scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             sres_   -= scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             osres_  -= scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             refres_ -= scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             odres_  -= scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             sres_   -= scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             osres_  -= scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             refres_ -= scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             odres_  -= scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             sres_   -= scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             osres_  -= scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             refres_ -= scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Scaled addition with subtraction assignment (OP*s)
      //=====================================================================================

      // Scaled addition with subtraction assignment with the given tensors
      {
         test_  = "Scaled addition with subtraction assignment with the given tensors (OP*s)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ( lhs_ + rhs_ ) * scalar;
//             odres_  -= ( lhs_ + rhs_ ) * scalar;
//             sres_   -= ( lhs_ + rhs_ ) * scalar;
//             osres_  -= ( lhs_ + rhs_ ) * scalar;
            refres_ -= ( reflhs_ + refrhs_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   -= ( lhs_ + orhs_ ) * scalar;
//             odres_  -= ( lhs_ + orhs_ ) * scalar;
//             sres_   -= ( lhs_ + orhs_ ) * scalar;
//             osres_  -= ( lhs_ + orhs_ ) * scalar;
//             refres_ -= ( reflhs_ + refrhs_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= ( olhs_ + rhs_ ) * scalar;
//             odres_  -= ( olhs_ + rhs_ ) * scalar;
//             sres_   -= ( olhs_ + rhs_ ) * scalar;
//             osres_  -= ( olhs_ + rhs_ ) * scalar;
//             refres_ -= ( reflhs_ + refrhs_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= ( olhs_ + orhs_ ) * scalar;
//             odres_  -= ( olhs_ + orhs_ ) * scalar;
//             sres_   -= ( olhs_ + orhs_ ) * scalar;
//             osres_  -= ( olhs_ + orhs_ ) * scalar;
//             refres_ -= ( reflhs_ + refrhs_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Scaled addition with subtraction assignment with evaluated tensors
      {
         test_  = "Scaled addition with subtraction assignment with evaluated tensors (OP*s)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
//             odres_  -= ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
//             sres_   -= ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
//             osres_  -= ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
            refres_ -= ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   -= ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             odres_  -= ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             sres_   -= ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             osres_  -= ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             refres_ -= ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             odres_  -= ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             sres_   -= ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             osres_  -= ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             refres_ -= ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             odres_  -= ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             sres_   -= ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             osres_  -= ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             refres_ -= ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Scaled addition with subtraction assignment (OP/s)
      //=====================================================================================

      // Scaled addition with subtraction assignment with the given tensors
      {
         test_  = "Scaled addition with subtraction assignment with the given tensors (OP/s)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ( lhs_ + rhs_ ) / scalar;
//             odres_  -= ( lhs_ + rhs_ ) / scalar;
//             sres_   -= ( lhs_ + rhs_ ) / scalar;
//             osres_  -= ( lhs_ + rhs_ ) / scalar;
            refres_ -= ( reflhs_ + refrhs_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   -= ( lhs_ + orhs_ ) / scalar;
//             odres_  -= ( lhs_ + orhs_ ) / scalar;
//             sres_   -= ( lhs_ + orhs_ ) / scalar;
//             osres_  -= ( lhs_ + orhs_ ) / scalar;
//             refres_ -= ( reflhs_ + refrhs_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= ( olhs_ + rhs_ ) / scalar;
//             odres_  -= ( olhs_ + rhs_ ) / scalar;
//             sres_   -= ( olhs_ + rhs_ ) / scalar;
//             osres_  -= ( olhs_ + rhs_ ) / scalar;
//             refres_ -= ( reflhs_ + refrhs_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= ( olhs_ + orhs_ ) / scalar;
//             odres_  -= ( olhs_ + orhs_ ) / scalar;
//             sres_   -= ( olhs_ + orhs_ ) / scalar;
//             osres_  -= ( olhs_ + orhs_ ) / scalar;
//             refres_ -= ( reflhs_ + refrhs_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Scaled addition with subtraction assignment with evaluated tensors
      {
         test_  = "Scaled addition with subtraction assignment with evaluated tensors (OP/s)";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
//             odres_  -= ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
//             sres_   -= ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
//             osres_  -= ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
            refres_ -= ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   -= ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             odres_  -= ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             sres_   -= ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             osres_  -= ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             refres_ -= ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             odres_  -= ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             sres_   -= ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             osres_  -= ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             refres_ -= ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             odres_  -= ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             sres_   -= ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             osres_  -= ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             refres_ -= ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Scaled addition with Schur product assignment (s*OP)
      //=====================================================================================

      // Scaled addition with Schur product assignment with the given tensors
      {
         test_  = "Scaled addition with Schur product assignment with the given tensors (s*OP)";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= scalar * ( lhs_ + rhs_ );
//             odres_  %= scalar * ( lhs_ + rhs_ );
//             sres_   %= scalar * ( lhs_ + rhs_ );
//             osres_  %= scalar * ( lhs_ + rhs_ );
            refres_ %= scalar * ( reflhs_ + refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   %= scalar * ( lhs_ + orhs_ );
//             odres_  %= scalar * ( lhs_ + orhs_ );
//             sres_   %= scalar * ( lhs_ + orhs_ );
//             osres_  %= scalar * ( lhs_ + orhs_ );
//             refres_ %= scalar * ( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= scalar * ( olhs_ + rhs_ );
//             odres_  %= scalar * ( olhs_ + rhs_ );
//             sres_   %= scalar * ( olhs_ + rhs_ );
//             osres_  %= scalar * ( olhs_ + rhs_ );
//             refres_ %= scalar * ( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= scalar * ( olhs_ + orhs_ );
//             odres_  %= scalar * ( olhs_ + orhs_ );
//             sres_   %= scalar * ( olhs_ + orhs_ );
//             osres_  %= scalar * ( olhs_ + orhs_ );
//             refres_ %= scalar * ( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Scaled addition with Schur product assignment with evaluated tensors
      {
         test_  = "Scaled addition with Schur product assignment with evaluated tensors (s*OP)";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= scalar * ( eval( lhs_ ) + eval( rhs_ ) );
//             odres_  %= scalar * ( eval( lhs_ ) + eval( rhs_ ) );
//             sres_   %= scalar * ( eval( lhs_ ) + eval( rhs_ ) );
//             osres_  %= scalar * ( eval( lhs_ ) + eval( rhs_ ) );
            refres_ %= scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   %= scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             odres_  %= scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             sres_   %= scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             osres_  %= scalar * ( eval( lhs_ ) + eval( orhs_ ) );
//             refres_ %= scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             odres_  %= scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             sres_   %= scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             osres_  %= scalar * ( eval( olhs_ ) + eval( rhs_ ) );
//             refres_ %= scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             odres_  %= scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             sres_   %= scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             osres_  %= scalar * ( eval( olhs_ ) + eval( orhs_ ) );
//             refres_ %= scalar * ( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Scaled addition with Schur product assignment (OP*s)
      //=====================================================================================

      // Scaled addition with Schur product assignment with the given tensors
      {
         test_  = "Scaled addition with Schur product assignment with the given tensors (OP*s)";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= ( lhs_ + rhs_ ) * scalar;
//             odres_  %= ( lhs_ + rhs_ ) * scalar;
//             sres_   %= ( lhs_ + rhs_ ) * scalar;
//             osres_  %= ( lhs_ + rhs_ ) * scalar;
            refres_ %= ( reflhs_ + refrhs_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   %= ( lhs_ + orhs_ ) * scalar;
//             odres_  %= ( lhs_ + orhs_ ) * scalar;
//             sres_   %= ( lhs_ + orhs_ ) * scalar;
//             osres_  %= ( lhs_ + orhs_ ) * scalar;
//             refres_ %= ( reflhs_ + refrhs_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= ( olhs_ + rhs_ ) * scalar;
//             odres_  %= ( olhs_ + rhs_ ) * scalar;
//             sres_   %= ( olhs_ + rhs_ ) * scalar;
//             osres_  %= ( olhs_ + rhs_ ) * scalar;
//             refres_ %= ( reflhs_ + refrhs_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= ( olhs_ + orhs_ ) * scalar;
//             odres_  %= ( olhs_ + orhs_ ) * scalar;
//             sres_   %= ( olhs_ + orhs_ ) * scalar;
//             osres_  %= ( olhs_ + orhs_ ) * scalar;
//             refres_ %= ( reflhs_ + refrhs_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Scaled addition with Schur product assignment with evaluated tensors
      {
         test_  = "Scaled addition with Schur product assignment with evaluated tensors (OP*s)";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
//             odres_  %= ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
//             sres_   %= ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
//             osres_  %= ( eval( lhs_ ) + eval( rhs_ ) ) * scalar;
            refres_ %= ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   %= ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             odres_  %= ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             sres_   %= ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             osres_  %= ( eval( lhs_ ) + eval( orhs_ ) ) * scalar;
//             refres_ %= ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             odres_  %= ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             sres_   %= ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             osres_  %= ( eval( olhs_ ) + eval( rhs_ ) ) * scalar;
//             refres_ %= ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             odres_  %= ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             sres_   %= ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             osres_  %= ( eval( olhs_ ) + eval( orhs_ ) ) * scalar;
//             refres_ %= ( eval( reflhs_ ) + eval( refrhs_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Scaled addition with Schur product assignment (OP/s)
      //=====================================================================================

      // Scaled addition with Schur product assignment with the given tensors
      {
         test_  = "Scaled addition with Schur product assignment with the given tensors (OP/s)";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= ( lhs_ + rhs_ ) / scalar;
//             odres_  %= ( lhs_ + rhs_ ) / scalar;
//             sres_   %= ( lhs_ + rhs_ ) / scalar;
//             osres_  %= ( lhs_ + rhs_ ) / scalar;
            refres_ %= ( reflhs_ + refrhs_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   %= ( lhs_ + orhs_ ) / scalar;
//             odres_  %= ( lhs_ + orhs_ ) / scalar;
//             sres_   %= ( lhs_ + orhs_ ) / scalar;
//             osres_  %= ( lhs_ + orhs_ ) / scalar;
//             refres_ %= ( reflhs_ + refrhs_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= ( olhs_ + rhs_ ) / scalar;
//             odres_  %= ( olhs_ + rhs_ ) / scalar;
//             sres_   %= ( olhs_ + rhs_ ) / scalar;
//             osres_  %= ( olhs_ + rhs_ ) / scalar;
//             refres_ %= ( reflhs_ + refrhs_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= ( olhs_ + orhs_ ) / scalar;
//             odres_  %= ( olhs_ + orhs_ ) / scalar;
//             sres_   %= ( olhs_ + orhs_ ) / scalar;
//             osres_  %= ( olhs_ + orhs_ ) / scalar;
//             refres_ %= ( reflhs_ + refrhs_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Scaled addition with Schur product assignment with evaluated tensors
      {
         test_  = "Scaled addition with Schur product assignment with evaluated tensors (OP/s)";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            dres_   %= ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
//             odres_  %= ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
//             sres_   %= ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
//             osres_  %= ( eval( lhs_ ) + eval( rhs_ ) ) / scalar;
            refres_ %= ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             dres_   %= ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             odres_  %= ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             sres_   %= ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             osres_  %= ( eval( lhs_ ) + eval( orhs_ ) ) / scalar;
//             refres_ %= ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             odres_  %= ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             sres_   %= ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             osres_  %= ( eval( olhs_ ) + eval( rhs_ ) ) / scalar;
//             refres_ %= ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             odres_  %= ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             sres_   %= ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             osres_  %= ( eval( olhs_ ) + eval( orhs_ ) ) / scalar;
//             refres_ %= ( eval( reflhs_ ) + eval( refrhs_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the transpose dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the transpose tensor addition with plain assignment. In case any error
// resulting from the addition or the subsequent assignment is detected, a \a std::runtime_error
// exception is thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testTransOperation()
{
#if BLAZETEST_MATHTEST_TEST_TRANS_OPERATION
   if( BLAZETEST_MATHTEST_TEST_TRANS_OPERATION > 1 )
   {
      //=====================================================================================
      // Transpose addition
      //=====================================================================================

      // Transpose addition with the given tensors
      {
         test_  = "Transpose addition with the given tensors";
         error_ = "Failed addition operation";

         try {
            initTransposeResults();
            tdres_  = trans(lhs_ + rhs_ );
//             todres_ = trans( lhs_ + rhs_ );
//             tsres_  = trans( lhs_ + rhs_ );
//             tosres_ = trans( lhs_ + rhs_ );
            refres_ = trans( reflhs_ + refrhs_ );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkTransposeResults<MT1,MT2>();

//          try {
//             initTransposeResults();
//             tdres_  = trans( lhs_ + orhs_ );
//             todres_ = trans( lhs_ + orhs_ );
//             tsres_  = trans( lhs_ + orhs_ );
//             tosres_ = trans( lhs_ + orhs_ );
//             refres_ = trans( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkTransposeResults<MT1,OMT2>();
//
//          try {
//             initTransposeResults();
//             tdres_  = trans( olhs_ + rhs_ );
//             todres_ = trans( olhs_ + rhs_ );
//             tsres_  = trans( olhs_ + rhs_ );
//             tosres_ = trans( olhs_ + rhs_ );
//             refres_ = trans( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkTransposeResults<OMT1,MT2>();
//
//          try {
//             initTransposeResults();
//             tdres_  = trans( olhs_ + orhs_ );
//             todres_ = trans( olhs_ + orhs_ );
//             tsres_  = trans( olhs_ + orhs_ );
//             tosres_ = trans( olhs_ + orhs_ );
//             refres_ = trans( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkTransposeResults<OMT1,OMT2>();
      }

      // Transpose addition with evaluated tensors
      {
         test_  = "Transpose addition with evaluated tensors";
         error_ = "Failed addition operation";

         try {
            initTransposeResults();
            tdres_  = trans( eval( lhs_ ) + eval( rhs_ ) );
//             todres_ = trans( eval( lhs_ ) + eval( rhs_ ) );
//             tsres_  = trans( eval( lhs_ ) + eval( rhs_ ) );
//             tosres_ = trans( eval( lhs_ ) + eval( rhs_ ) );
            refres_ = trans( eval( reflhs_ ) + eval( refrhs_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkTransposeResults<MT1,MT2>();

//          try {
//             initTransposeResults();
//             tdres_  = trans( eval( lhs_ ) + eval( orhs_ ) );
//             todres_ = trans( eval( lhs_ ) + eval( orhs_ ) );
//             tsres_  = trans( eval( lhs_ ) + eval( orhs_ ) );
//             tosres_ = trans( eval( lhs_ ) + eval( orhs_ ) );
//             refres_ = trans( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkTransposeResults<MT1,OMT2>();
//
//          try {
//             initTransposeResults();
//             tdres_  = trans( eval( olhs_ ) + eval( rhs_ ) );
//             todres_ = trans( eval( olhs_ ) + eval( rhs_ ) );
//             tsres_  = trans( eval( olhs_ ) + eval( rhs_ ) );
//             tosres_ = trans( eval( olhs_ ) + eval( rhs_ ) );
//             refres_ = trans( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkTransposeResults<OMT1,MT2>();
//
//          try {
//             initTransposeResults();
//             tdres_  = trans( eval( olhs_ ) + eval( orhs_ ) );
//             todres_ = trans( eval( olhs_ ) + eval( orhs_ ) );
//             tsres_  = trans( eval( olhs_ ) + eval( orhs_ ) );
//             tosres_ = trans( eval( olhs_ ) + eval( orhs_ ) );
//             refres_ = trans( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkTransposeResults<OMT1,OMT2>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the conjugate transpose dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the conjugate transpose tensor addition with plain assignment. In
// case any error resulting from the addition or the subsequent assignment is detected, a
// \a std::runtime_error exception is thrown.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testCTransOperation()
// {
// #if BLAZETEST_MATHTEST_TEST_CTRANS_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_CTRANS_OPERATION > 1 )
//    {
//       //=====================================================================================
//       // Conjugate transpose addition
//       //=====================================================================================
//
//       // Conjugate transpose addition with the given tensors
//       {
//          test_  = "Conjugate transpose addition with the given tensors";
//          error_ = "Failed addition operation";
//
//          try {
//             initTransposeResults();
//             tdres_  = ctrans( lhs_ + rhs_ );
//             todres_ = ctrans( lhs_ + rhs_ );
//             tsres_  = ctrans( lhs_ + rhs_ );
//             tosres_ = ctrans( lhs_ + rhs_ );
//             refres_ = ctrans( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkTransposeResults<MT1,MT2>();
//
//          try {
//             initTransposeResults();
//             tdres_  = ctrans( lhs_ + orhs_ );
//             todres_ = ctrans( lhs_ + orhs_ );
//             tsres_  = ctrans( lhs_ + orhs_ );
//             tosres_ = ctrans( lhs_ + orhs_ );
//             refres_ = ctrans( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkTransposeResults<MT1,OMT2>();
//
//          try {
//             initTransposeResults();
//             tdres_  = ctrans( olhs_ + rhs_ );
//             todres_ = ctrans( olhs_ + rhs_ );
//             tsres_  = ctrans( olhs_ + rhs_ );
//             tosres_ = ctrans( olhs_ + rhs_ );
//             refres_ = ctrans( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkTransposeResults<OMT1,MT2>();
//
//          try {
//             initTransposeResults();
//             tdres_  = ctrans( olhs_ + orhs_ );
//             todres_ = ctrans( olhs_ + orhs_ );
//             tsres_  = ctrans( olhs_ + orhs_ );
//             tosres_ = ctrans( olhs_ + orhs_ );
//             refres_ = ctrans( reflhs_ + refrhs_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkTransposeResults<OMT1,OMT2>();
//       }
//
//       // Conjugate transpose addition with evaluated tensors
//       {
//          test_  = "Conjugate transpose addition with evaluated tensors";
//          error_ = "Failed addition operation";
//
//          try {
//             initTransposeResults();
//             tdres_  = ctrans( eval( lhs_ ) + eval( rhs_ ) );
//             todres_ = ctrans( eval( lhs_ ) + eval( rhs_ ) );
//             tsres_  = ctrans( eval( lhs_ ) + eval( rhs_ ) );
//             tosres_ = ctrans( eval( lhs_ ) + eval( rhs_ ) );
//             refres_ = ctrans( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkTransposeResults<MT1,MT2>();
//
//          try {
//             initTransposeResults();
//             tdres_  = ctrans( eval( lhs_ ) + eval( orhs_ ) );
//             todres_ = ctrans( eval( lhs_ ) + eval( orhs_ ) );
//             tsres_  = ctrans( eval( lhs_ ) + eval( orhs_ ) );
//             tosres_ = ctrans( eval( lhs_ ) + eval( orhs_ ) );
//             refres_ = ctrans( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkTransposeResults<MT1,OMT2>();
//
//          try {
//             initTransposeResults();
//             tdres_  = ctrans( eval( olhs_ ) + eval( rhs_ ) );
//             todres_ = ctrans( eval( olhs_ ) + eval( rhs_ ) );
//             tsres_  = ctrans( eval( olhs_ ) + eval( rhs_ ) );
//             tosres_ = ctrans( eval( olhs_ ) + eval( rhs_ ) );
//             refres_ = ctrans( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkTransposeResults<OMT1,MT2>();
//
//          try {
//             initTransposeResults();
//             tdres_  = ctrans( eval( olhs_ ) + eval( orhs_ ) );
//             todres_ = ctrans( eval( olhs_ ) + eval( orhs_ ) );
//             tsres_  = ctrans( eval( olhs_ ) + eval( orhs_ ) );
//             tosres_ = ctrans( eval( olhs_ ) + eval( orhs_ ) );
//             refres_ = ctrans( eval( reflhs_ ) + eval( refrhs_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkTransposeResults<OMT1,OMT2>();
//       }
//    }
// #endif
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the abs dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the abs tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testAbsOperation()
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
/*!\brief Testing the conjugate dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the conjugate tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testConjOperation()
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
/*!\brief Testing the \a real dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the \a real tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testRealOperation()
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
/*!\brief Testing the \a imag dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the \a imag tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testImagOperation()
{
#if BLAZETEST_MATHTEST_TEST_IMAG_OPERATION
   if( BLAZETEST_MATHTEST_TEST_IMAG_OPERATION > 1 &&
       ( !blaze::IsHermitian<DRE>::value || blaze::isSymmetric( imag( lhs_ + rhs_ ) ) ) )
   {
      testCustomOperation( blaze::Imag(), "imag" );
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the \a inv dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the \a inv tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testInvOperation()
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
/*!\brief Testing the evaluated dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the evaluated tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testEvalOperation()
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
/*!\brief Testing the serialized dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the serialized tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testSerialOperation()
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
/*!\brief Testing the symmetric dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the symmetric tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testDeclSymOperation( blaze::TrueType )
// {
// #if BLAZETEST_MATHTEST_TEST_DECLSYM_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_DECLSYM_OPERATION > 1 )
//    {
//       if( ( !blaze::IsDiagonal<MT1>::value && blaze::IsTriangular<MT1>::value ) ||
//           ( !blaze::IsDiagonal<MT2>::value && blaze::IsTriangular<MT2>::value ) ||
//           ( !blaze::IsDiagonal<MT1>::value && blaze::IsHermitian<MT1>::value && blaze::IsComplex<ET1>::value ) ||
//           ( !blaze::IsDiagonal<MT2>::value && blaze::IsHermitian<MT2>::value && blaze::IsComplex<ET2>::value ) ||
//           ( lhs_.rows() != lhs_.columns() ) )
//          return;
//
//
//       //=====================================================================================
//       // Test-specific setup of the left-hand side operand
//       //=====================================================================================
//
//       MT1  lhs   ( lhs_ * trans( lhs_ ) );
//       OMT1 olhs  ( lhs );
//       RT1  reflhs( lhs );
//
//
//       //=====================================================================================
//       // Test-specific setup of the right-hand side operand
//       //=====================================================================================
//
//       MT2  rhs   ( rhs_ * trans( rhs_ ) );
//       OMT2 orhs  ( rhs );
//       RT2  refrhs( rhs );
//
//
//       //=====================================================================================
//       // Declsym addition
//       //=====================================================================================
//
//       // Declsym addition with the given tensors
//       {
//          test_  = "Declsym addition with the given tensors";
//          error_ = "Failed addition operation";
//
//          try {
//             initResults();
//             dres_   = declsym( lhs + rhs );
//             odres_  = declsym( lhs + rhs );
//             sres_   = declsym( lhs + rhs );
//             osres_  = declsym( lhs + rhs );
//             refres_ = declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = declsym( lhs + orhs );
//             odres_  = declsym( lhs + orhs );
//             sres_   = declsym( lhs + orhs );
//             osres_  = declsym( lhs + orhs );
//             refres_ = declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = declsym( olhs + rhs );
//             odres_  = declsym( olhs + rhs );
//             sres_   = declsym( olhs + rhs );
//             osres_  = declsym( olhs + rhs );
//             refres_ = declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = declsym( olhs + orhs );
//             odres_  = declsym( olhs + orhs );
//             sres_   = declsym( olhs + orhs );
//             osres_  = declsym( olhs + orhs );
//             refres_ = declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Declsym addition with evaluated tensors
//       {
//          test_  = "Declsym addition with evaluated left-hand side tensor";
//          error_ = "Failed addition operation";
//
//          try {
//             initResults();
//             dres_   = declsym( eval( lhs ) + eval( rhs ) );
//             odres_  = declsym( eval( lhs ) + eval( rhs ) );
//             sres_   = declsym( eval( lhs ) + eval( rhs ) );
//             osres_  = declsym( eval( lhs ) + eval( rhs ) );
//             refres_ = declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = declsym( eval( lhs ) + eval( orhs ) );
//             odres_  = declsym( eval( lhs ) + eval( orhs ) );
//             sres_   = declsym( eval( lhs ) + eval( orhs ) );
//             osres_  = declsym( eval( lhs ) + eval( orhs ) );
//             refres_ = declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = declsym( eval( olhs ) + eval( rhs ) );
//             odres_  = declsym( eval( olhs ) + eval( rhs ) );
//             sres_   = declsym( eval( olhs ) + eval( rhs ) );
//             osres_  = declsym( eval( olhs ) + eval( rhs ) );
//             refres_ = declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = declsym( eval( olhs ) + eval( orhs ) );
//             odres_  = declsym( eval( olhs ) + eval( orhs ) );
//             sres_   = declsym( eval( olhs ) + eval( orhs ) );
//             osres_  = declsym( eval( olhs ) + eval( orhs ) );
//             refres_ = declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Declsym addition with addition assignment
//       //=====================================================================================
//
//       // Declsym addition with addition assignment with the given tensors
//       {
//          test_  = "Declsym addition with addition assignment with the given tensors";
//          error_ = "Failed addition assignment operation";
//
//          try {
//             initResults();
//             dres_   += declsym( lhs + rhs );
//             odres_  += declsym( lhs + rhs );
//             sres_   += declsym( lhs + rhs );
//             osres_  += declsym( lhs + rhs );
//             refres_ += declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += declsym( lhs + orhs );
//             odres_  += declsym( lhs + orhs );
//             sres_   += declsym( lhs + orhs );
//             osres_  += declsym( lhs + orhs );
//             refres_ += declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += declsym( olhs + rhs );
//             odres_  += declsym( olhs + rhs );
//             sres_   += declsym( olhs + rhs );
//             osres_  += declsym( olhs + rhs );
//             refres_ += declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += declsym( olhs + orhs );
//             odres_  += declsym( olhs + orhs );
//             sres_   += declsym( olhs + orhs );
//             osres_  += declsym( olhs + orhs );
//             refres_ += declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Declsym addition with addition assignment with evaluated tensors
//       {
//          test_  = "Declsym addition with addition assignment with evaluated tensors";
//          error_ = "Failed addition assignment operation";
//
//          try {
//             initResults();
//             dres_   += declsym( eval( lhs ) + eval( rhs ) );
//             odres_  += declsym( eval( lhs ) + eval( rhs ) );
//             sres_   += declsym( eval( lhs ) + eval( rhs ) );
//             osres_  += declsym( eval( lhs ) + eval( rhs ) );
//             refres_ += declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += declsym( eval( lhs ) + eval( orhs ) );
//             odres_  += declsym( eval( lhs ) + eval( orhs ) );
//             sres_   += declsym( eval( lhs ) + eval( orhs ) );
//             osres_  += declsym( eval( lhs ) + eval( orhs ) );
//             refres_ += declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += declsym( eval( olhs ) + eval( rhs ) );
//             odres_  += declsym( eval( olhs ) + eval( rhs ) );
//             sres_   += declsym( eval( olhs ) + eval( rhs ) );
//             osres_  += declsym( eval( olhs ) + eval( rhs ) );
//             refres_ += declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += declsym( eval( olhs ) + eval( orhs ) );
//             odres_  += declsym( eval( olhs ) + eval( orhs ) );
//             sres_   += declsym( eval( olhs ) + eval( orhs ) );
//             osres_  += declsym( eval( olhs ) + eval( orhs ) );
//             refres_ += declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Declsym addition with subtraction assignment
//       //=====================================================================================
//
//       // Declsym addition with subtraction assignment with the given tensors
//       {
//          test_  = "Declsym addition with subtraction assignment with the given tensors";
//          error_ = "Failed subtraction assignment operation";
//
//          try {
//             initResults();
//             dres_   -= declsym( lhs + rhs );
//             odres_  -= declsym( lhs + rhs );
//             sres_   -= declsym( lhs + rhs );
//             osres_  -= declsym( lhs + rhs );
//             refres_ -= declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= declsym( lhs + orhs );
//             odres_  -= declsym( lhs + orhs );
//             sres_   -= declsym( lhs + orhs );
//             osres_  -= declsym( lhs + orhs );
//             refres_ -= declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= declsym( olhs + rhs );
//             odres_  -= declsym( olhs + rhs );
//             sres_   -= declsym( olhs + rhs );
//             osres_  -= declsym( olhs + rhs );
//             refres_ -= declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= declsym( olhs + orhs );
//             odres_  -= declsym( olhs + orhs );
//             sres_   -= declsym( olhs + orhs );
//             osres_  -= declsym( olhs + orhs );
//             refres_ -= declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Declsym addition with subtraction assignment with evaluated tensors
//       {
//          test_  = "Declsym addition with subtraction assignment with evaluated tensors";
//          error_ = "Failed subtraction assignment operation";
//
//          try {
//             initResults();
//             dres_   -= declsym( eval( lhs ) + eval( rhs ) );
//             odres_  -= declsym( eval( lhs ) + eval( rhs ) );
//             sres_   -= declsym( eval( lhs ) + eval( rhs ) );
//             osres_  -= declsym( eval( lhs ) + eval( rhs ) );
//             refres_ -= declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= declsym( eval( lhs ) + eval( orhs ) );
//             odres_  -= declsym( eval( lhs ) + eval( orhs ) );
//             sres_   -= declsym( eval( lhs ) + eval( orhs ) );
//             osres_  -= declsym( eval( lhs ) + eval( orhs ) );
//             refres_ -= declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= declsym( eval( olhs ) + eval( rhs ) );
//             odres_  -= declsym( eval( olhs ) + eval( rhs ) );
//             sres_   -= declsym( eval( olhs ) + eval( rhs ) );
//             osres_  -= declsym( eval( olhs ) + eval( rhs ) );
//             refres_ -= declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= declsym( eval( olhs ) + eval( orhs ) );
//             odres_  -= declsym( eval( olhs ) + eval( orhs ) );
//             sres_   -= declsym( eval( olhs ) + eval( orhs ) );
//             osres_  -= declsym( eval( olhs ) + eval( orhs ) );
//             refres_ -= declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Declsym addition with Schur product assignment
//       //=====================================================================================
//
//       // Declsym addition with Schur product assignment with the given tensors
//       {
//          test_  = "Declsym addition with Schur product assignment with the given tensors";
//          error_ = "Failed Schur product assignment operation";
//
//          try {
//             initResults();
//             dres_   %= declsym( lhs + rhs );
//             odres_  %= declsym( lhs + rhs );
//             sres_   %= declsym( lhs + rhs );
//             osres_  %= declsym( lhs + rhs );
//             refres_ %= declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= declsym( lhs + orhs );
//             odres_  %= declsym( lhs + orhs );
//             sres_   %= declsym( lhs + orhs );
//             osres_  %= declsym( lhs + orhs );
//             refres_ %= declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= declsym( olhs + rhs );
//             odres_  %= declsym( olhs + rhs );
//             sres_   %= declsym( olhs + rhs );
//             osres_  %= declsym( olhs + rhs );
//             refres_ %= declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= declsym( olhs + orhs );
//             odres_  %= declsym( olhs + orhs );
//             sres_   %= declsym( olhs + orhs );
//             osres_  %= declsym( olhs + orhs );
//             refres_ %= declsym( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Declsym addition with Schur product assignment with evaluated tensors
//       {
//          test_  = "Declsym addition with Schur product assignment with evaluated tensors";
//          error_ = "Failed Schur product assignment operation";
//
//          try {
//             initResults();
//             dres_   %= declsym( eval( lhs ) + eval( rhs ) );
//             odres_  %= declsym( eval( lhs ) + eval( rhs ) );
//             sres_   %= declsym( eval( lhs ) + eval( rhs ) );
//             osres_  %= declsym( eval( lhs ) + eval( rhs ) );
//             refres_ %= declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= declsym( eval( lhs ) + eval( orhs ) );
//             odres_  %= declsym( eval( lhs ) + eval( orhs ) );
//             sres_   %= declsym( eval( lhs ) + eval( orhs ) );
//             osres_  %= declsym( eval( lhs ) + eval( orhs ) );
//             refres_ %= declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= declsym( eval( olhs ) + eval( rhs ) );
//             odres_  %= declsym( eval( olhs ) + eval( rhs ) );
//             sres_   %= declsym( eval( olhs ) + eval( rhs ) );
//             osres_  %= declsym( eval( olhs ) + eval( rhs ) );
//             refres_ %= declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= declsym( eval( olhs ) + eval( orhs ) );
//             odres_  %= declsym( eval( olhs ) + eval( orhs ) );
//             sres_   %= declsym( eval( olhs ) + eval( orhs ) );
//             osres_  %= declsym( eval( olhs ) + eval( orhs ) );
//             refres_ %= declsym( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//    }
// #endif
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the symmetric dense tensor/dense tensor addition.
//
// \return void
//
// This function is called in case the symmetric tensor/tensor addition operation is not
// available for the given tensor types \a MT1 and \a MT2.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testDeclSymOperation( blaze::FalseType )
// {}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the Hermitian dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the Hermitian tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testDeclHermOperation( blaze::TrueType )
// {
// #if BLAZETEST_MATHTEST_TEST_DECLHERM_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_DECLHERM_OPERATION > 1 )
//    {
//       if( ( !blaze::IsDiagonal<MT1>::value && blaze::IsTriangular<MT1>::value ) ||
//           ( !blaze::IsDiagonal<MT2>::value && blaze::IsTriangular<MT2>::value ) ||
//           ( !blaze::IsDiagonal<MT1>::value && blaze::IsSymmetric<MT1>::value && blaze::IsComplex<ET1>::value ) ||
//           ( !blaze::IsDiagonal<MT2>::value && blaze::IsSymmetric<MT2>::value && blaze::IsComplex<ET2>::value ) ||
//           ( lhs_.rows() != lhs_.columns() ) )
//          return;
//
//
//       //=====================================================================================
//       // Test-specific setup of the left-hand side operand
//       //=====================================================================================
//
//       MT1  lhs   ( lhs_ * ctrans( lhs_ ) );
//       OMT1 olhs  ( lhs );
//       RT1  reflhs( lhs );
//
//
//       //=====================================================================================
//       // Test-specific setup of the right-hand side operand
//       //=====================================================================================
//
//       MT2  rhs   ( rhs_ * ctrans( rhs_ ) );
//       OMT2 orhs  ( rhs );
//       RT2  refrhs( rhs );
//
//
//       //=====================================================================================
//       // Declherm addition
//       //=====================================================================================
//
//       // Declherm addition with the given tensors
//       {
//          test_  = "Declherm addition with the given tensors";
//          error_ = "Failed addition operation";
//
//          try {
//             initResults();
//             dres_   = declherm( lhs + rhs );
//             odres_  = declherm( lhs + rhs );
//             sres_   = declherm( lhs + rhs );
//             osres_  = declherm( lhs + rhs );
//             refres_ = declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = declherm( lhs + orhs );
//             odres_  = declherm( lhs + orhs );
//             sres_   = declherm( lhs + orhs );
//             osres_  = declherm( lhs + orhs );
//             refres_ = declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = declherm( olhs + rhs );
//             odres_  = declherm( olhs + rhs );
//             sres_   = declherm( olhs + rhs );
//             osres_  = declherm( olhs + rhs );
//             refres_ = declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = declherm( olhs + orhs );
//             odres_  = declherm( olhs + orhs );
//             sres_   = declherm( olhs + orhs );
//             osres_  = declherm( olhs + orhs );
//             refres_ = declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Declherm addition with evaluated tensors
//       {
//          test_  = "Declherm addition with evaluated left-hand side tensor";
//          error_ = "Failed addition operation";
//
//          try {
//             initResults();
//             dres_   = declherm( eval( lhs ) + eval( rhs ) );
//             odres_  = declherm( eval( lhs ) + eval( rhs ) );
//             sres_   = declherm( eval( lhs ) + eval( rhs ) );
//             osres_  = declherm( eval( lhs ) + eval( rhs ) );
//             refres_ = declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = declherm( eval( lhs ) + eval( orhs ) );
//             odres_  = declherm( eval( lhs ) + eval( orhs ) );
//             sres_   = declherm( eval( lhs ) + eval( orhs ) );
//             osres_  = declherm( eval( lhs ) + eval( orhs ) );
//             refres_ = declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = declherm( eval( olhs ) + eval( rhs ) );
//             odres_  = declherm( eval( olhs ) + eval( rhs ) );
//             sres_   = declherm( eval( olhs ) + eval( rhs ) );
//             osres_  = declherm( eval( olhs ) + eval( rhs ) );
//             refres_ = declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = declherm( eval( olhs ) + eval( orhs ) );
//             odres_  = declherm( eval( olhs ) + eval( orhs ) );
//             sres_   = declherm( eval( olhs ) + eval( orhs ) );
//             osres_  = declherm( eval( olhs ) + eval( orhs ) );
//             refres_ = declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Declherm addition with addition assignment
//       //=====================================================================================
//
//       // Declherm addition with addition assignment with the given tensors
//       {
//          test_  = "Declherm addition with addition assignment with the given tensors";
//          error_ = "Failed addition assignment operation";
//
//          try {
//             initResults();
//             dres_   += declherm( lhs + rhs );
//             odres_  += declherm( lhs + rhs );
//             sres_   += declherm( lhs + rhs );
//             osres_  += declherm( lhs + rhs );
//             refres_ += declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += declherm( lhs + orhs );
//             odres_  += declherm( lhs + orhs );
//             sres_   += declherm( lhs + orhs );
//             osres_  += declherm( lhs + orhs );
//             refres_ += declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += declherm( olhs + rhs );
//             odres_  += declherm( olhs + rhs );
//             sres_   += declherm( olhs + rhs );
//             osres_  += declherm( olhs + rhs );
//             refres_ += declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += declherm( olhs + orhs );
//             odres_  += declherm( olhs + orhs );
//             sres_   += declherm( olhs + orhs );
//             osres_  += declherm( olhs + orhs );
//             refres_ += declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Declherm addition with addition assignment with evaluated tensors
//       {
//          test_  = "Declherm addition with addition assignment with evaluated tensors";
//          error_ = "Failed addition assignment operation";
//
//          try {
//             initResults();
//             dres_   += declherm( eval( lhs ) + eval( rhs ) );
//             odres_  += declherm( eval( lhs ) + eval( rhs ) );
//             sres_   += declherm( eval( lhs ) + eval( rhs ) );
//             osres_  += declherm( eval( lhs ) + eval( rhs ) );
//             refres_ += declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += declherm( eval( lhs ) + eval( orhs ) );
//             odres_  += declherm( eval( lhs ) + eval( orhs ) );
//             sres_   += declherm( eval( lhs ) + eval( orhs ) );
//             osres_  += declherm( eval( lhs ) + eval( orhs ) );
//             refres_ += declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += declherm( eval( olhs ) + eval( rhs ) );
//             odres_  += declherm( eval( olhs ) + eval( rhs ) );
//             sres_   += declherm( eval( olhs ) + eval( rhs ) );
//             osres_  += declherm( eval( olhs ) + eval( rhs ) );
//             refres_ += declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += declherm( eval( olhs ) + eval( orhs ) );
//             odres_  += declherm( eval( olhs ) + eval( orhs ) );
//             sres_   += declherm( eval( olhs ) + eval( orhs ) );
//             osres_  += declherm( eval( olhs ) + eval( orhs ) );
//             refres_ += declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Declherm addition with subtraction assignment
//       //=====================================================================================
//
//       // Declherm addition with subtraction assignment with the given tensors
//       {
//          test_  = "Declherm addition with subtraction assignment with the given tensors";
//          error_ = "Failed subtraction assignment operation";
//
//          try {
//             initResults();
//             dres_   -= declherm( lhs + rhs );
//             odres_  -= declherm( lhs + rhs );
//             sres_   -= declherm( lhs + rhs );
//             osres_  -= declherm( lhs + rhs );
//             refres_ -= declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= declherm( lhs + orhs );
//             odres_  -= declherm( lhs + orhs );
//             sres_   -= declherm( lhs + orhs );
//             osres_  -= declherm( lhs + orhs );
//             refres_ -= declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= declherm( olhs + rhs );
//             odres_  -= declherm( olhs + rhs );
//             sres_   -= declherm( olhs + rhs );
//             osres_  -= declherm( olhs + rhs );
//             refres_ -= declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= declherm( olhs + orhs );
//             odres_  -= declherm( olhs + orhs );
//             sres_   -= declherm( olhs + orhs );
//             osres_  -= declherm( olhs + orhs );
//             refres_ -= declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Declherm addition with subtraction assignment with evaluated tensors
//       {
//          test_  = "Declherm addition with subtraction assignment with evaluated tensors";
//          error_ = "Failed subtraction assignment operation";
//
//          try {
//             initResults();
//             dres_   -= declherm( eval( lhs ) + eval( rhs ) );
//             odres_  -= declherm( eval( lhs ) + eval( rhs ) );
//             sres_   -= declherm( eval( lhs ) + eval( rhs ) );
//             osres_  -= declherm( eval( lhs ) + eval( rhs ) );
//             refres_ -= declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= declherm( eval( lhs ) + eval( orhs ) );
//             odres_  -= declherm( eval( lhs ) + eval( orhs ) );
//             sres_   -= declherm( eval( lhs ) + eval( orhs ) );
//             osres_  -= declherm( eval( lhs ) + eval( orhs ) );
//             refres_ -= declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= declherm( eval( olhs ) + eval( rhs ) );
//             odres_  -= declherm( eval( olhs ) + eval( rhs ) );
//             sres_   -= declherm( eval( olhs ) + eval( rhs ) );
//             osres_  -= declherm( eval( olhs ) + eval( rhs ) );
//             refres_ -= declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= declherm( eval( olhs ) + eval( orhs ) );
//             odres_  -= declherm( eval( olhs ) + eval( orhs ) );
//             sres_   -= declherm( eval( olhs ) + eval( orhs ) );
//             osres_  -= declherm( eval( olhs ) + eval( orhs ) );
//             refres_ -= declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Declherm addition with Schur product assignment
//       //=====================================================================================
//
//       // Declherm addition with Schur product assignment with the given tensors
//       {
//          test_  = "Declherm addition with Schur product assignment with the given tensors";
//          error_ = "Failed Schur product assignment operation";
//
//          try {
//             initResults();
//             dres_   %= declherm( lhs + rhs );
//             odres_  %= declherm( lhs + rhs );
//             sres_   %= declherm( lhs + rhs );
//             osres_  %= declherm( lhs + rhs );
//             refres_ %= declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= declherm( lhs + orhs );
//             odres_  %= declherm( lhs + orhs );
//             sres_   %= declherm( lhs + orhs );
//             osres_  %= declherm( lhs + orhs );
//             refres_ %= declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= declherm( olhs + rhs );
//             odres_  %= declherm( olhs + rhs );
//             sres_   %= declherm( olhs + rhs );
//             osres_  %= declherm( olhs + rhs );
//             refres_ %= declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= declherm( olhs + orhs );
//             odres_  %= declherm( olhs + orhs );
//             sres_   %= declherm( olhs + orhs );
//             osres_  %= declherm( olhs + orhs );
//             refres_ %= declherm( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Declherm addition with Schur product assignment with evaluated tensors
//       {
//          test_  = "Declherm addition with Schur product assignment with evaluated tensors";
//          error_ = "Failed Schur product assignment operation";
//
//          try {
//             initResults();
//             dres_   %= declherm( eval( lhs ) + eval( rhs ) );
//             odres_  %= declherm( eval( lhs ) + eval( rhs ) );
//             sres_   %= declherm( eval( lhs ) + eval( rhs ) );
//             osres_  %= declherm( eval( lhs ) + eval( rhs ) );
//             refres_ %= declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= declherm( eval( lhs ) + eval( orhs ) );
//             odres_  %= declherm( eval( lhs ) + eval( orhs ) );
//             sres_   %= declherm( eval( lhs ) + eval( orhs ) );
//             osres_  %= declherm( eval( lhs ) + eval( orhs ) );
//             refres_ %= declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= declherm( eval( olhs ) + eval( rhs ) );
//             odres_  %= declherm( eval( olhs ) + eval( rhs ) );
//             sres_   %= declherm( eval( olhs ) + eval( rhs ) );
//             osres_  %= declherm( eval( olhs ) + eval( rhs ) );
//             refres_ %= declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= declherm( eval( olhs ) + eval( orhs ) );
//             odres_  %= declherm( eval( olhs ) + eval( orhs ) );
//             sres_   %= declherm( eval( olhs ) + eval( orhs ) );
//             osres_  %= declherm( eval( olhs ) + eval( orhs ) );
//             refres_ %= declherm( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//    }
// #endif
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the Hermitian dense tensor/dense tensor addition.
//
// \return void
//
// This function is called in case the Hermitian tensor/tensor addition operation is not
// available for the given tensor types \a MT1 and \a MT2.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testDeclHermOperation( blaze::FalseType )
// {}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the lower dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the lower tensor addition with plain assignment, addition assignment,
// and subtraction assignment. In case any error resulting from the addition or the subsequent
// assignment is detected, a \a std::runtime_error exception is thrown.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testDeclLowOperation( blaze::TrueType )
// {
// #if BLAZETEST_MATHTEST_TEST_DECLLOW_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_DECLLOW_OPERATION > 1 )
//    {
//       if( lhs_.rows() != lhs_.columns() )
//          return;
//
//
//       //=====================================================================================
//       // Test-specific setup of the left-hand side operand
//       //=====================================================================================
//
//       MT1 lhs( lhs_ );
//
//       blaze::resetUpper( lhs );
//
//       OMT1 olhs  ( lhs );
//       RT1  reflhs( lhs );
//
//
//       //=====================================================================================
//       // Test-specific setup of the right-hand side operand
//       //=====================================================================================
//
//       MT2 rhs( rhs_ );
//
//       blaze::resetUpper( rhs );
//
//       OMT2 orhs  ( rhs );
//       RT2  refrhs( rhs );
//
//
//       //=====================================================================================
//       // Decllow addition
//       //=====================================================================================
//
//       // Decllow addition with the given tensors
//       {
//          test_  = "Decllow addition with the given tensors";
//          error_ = "Failed addition operation";
//
//          try {
//             initResults();
//             dres_   = decllow( lhs + rhs );
//             odres_  = decllow( lhs + rhs );
//             sres_   = decllow( lhs + rhs );
//             osres_  = decllow( lhs + rhs );
//             refres_ = decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = decllow( lhs + orhs );
//             odres_  = decllow( lhs + orhs );
//             sres_   = decllow( lhs + orhs );
//             osres_  = decllow( lhs + orhs );
//             refres_ = decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = decllow( olhs + rhs );
//             odres_  = decllow( olhs + rhs );
//             sres_   = decllow( olhs + rhs );
//             osres_  = decllow( olhs + rhs );
//             refres_ = decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = decllow( olhs + orhs );
//             odres_  = decllow( olhs + orhs );
//             sres_   = decllow( olhs + orhs );
//             osres_  = decllow( olhs + orhs );
//             refres_ = decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Decllow addition with evaluated tensors
//       {
//          test_  = "Decllow addition with evaluated left-hand side tensor";
//          error_ = "Failed addition operation";
//
//          try {
//             initResults();
//             dres_   = decllow( eval( lhs ) + eval( rhs ) );
//             odres_  = decllow( eval( lhs ) + eval( rhs ) );
//             sres_   = decllow( eval( lhs ) + eval( rhs ) );
//             osres_  = decllow( eval( lhs ) + eval( rhs ) );
//             refres_ = decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = decllow( eval( lhs ) + eval( orhs ) );
//             odres_  = decllow( eval( lhs ) + eval( orhs ) );
//             sres_   = decllow( eval( lhs ) + eval( orhs ) );
//             osres_  = decllow( eval( lhs ) + eval( orhs ) );
//             refres_ = decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = decllow( eval( olhs ) + eval( rhs ) );
//             odres_  = decllow( eval( olhs ) + eval( rhs ) );
//             sres_   = decllow( eval( olhs ) + eval( rhs ) );
//             osres_  = decllow( eval( olhs ) + eval( rhs ) );
//             refres_ = decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = decllow( eval( olhs ) + eval( orhs ) );
//             odres_  = decllow( eval( olhs ) + eval( orhs ) );
//             sres_   = decllow( eval( olhs ) + eval( orhs ) );
//             osres_  = decllow( eval( olhs ) + eval( orhs ) );
//             refres_ = decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Decllow addition with addition assignment
//       //=====================================================================================
//
//       // Decllow addition with addition assignment with the given tensors
//       {
//          test_  = "Decllow addition with addition assignment with the given tensors";
//          error_ = "Failed addition assignment operation";
//
//          try {
//             initResults();
//             dres_   += decllow( lhs + rhs );
//             odres_  += decllow( lhs + rhs );
//             sres_   += decllow( lhs + rhs );
//             osres_  += decllow( lhs + rhs );
//             refres_ += decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += decllow( lhs + orhs );
//             odres_  += decllow( lhs + orhs );
//             sres_   += decllow( lhs + orhs );
//             osres_  += decllow( lhs + orhs );
//             refres_ += decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += decllow( olhs + rhs );
//             odres_  += decllow( olhs + rhs );
//             sres_   += decllow( olhs + rhs );
//             osres_  += decllow( olhs + rhs );
//             refres_ += decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += decllow( olhs + orhs );
//             odres_  += decllow( olhs + orhs );
//             sres_   += decllow( olhs + orhs );
//             osres_  += decllow( olhs + orhs );
//             refres_ += decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Decllow addition with addition assignment with evaluated tensors
//       {
//          test_  = "Decllow addition with addition assignment with evaluated tensors";
//          error_ = "Failed addition assignment operation";
//
//          try {
//             initResults();
//             dres_   += decllow( eval( lhs ) + eval( rhs ) );
//             odres_  += decllow( eval( lhs ) + eval( rhs ) );
//             sres_   += decllow( eval( lhs ) + eval( rhs ) );
//             osres_  += decllow( eval( lhs ) + eval( rhs ) );
//             refres_ += decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += decllow( eval( lhs ) + eval( orhs ) );
//             odres_  += decllow( eval( lhs ) + eval( orhs ) );
//             sres_   += decllow( eval( lhs ) + eval( orhs ) );
//             osres_  += decllow( eval( lhs ) + eval( orhs ) );
//             refres_ += decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += decllow( eval( olhs ) + eval( rhs ) );
//             odres_  += decllow( eval( olhs ) + eval( rhs ) );
//             sres_   += decllow( eval( olhs ) + eval( rhs ) );
//             osres_  += decllow( eval( olhs ) + eval( rhs ) );
//             refres_ += decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += decllow( eval( olhs ) + eval( orhs ) );
//             odres_  += decllow( eval( olhs ) + eval( orhs ) );
//             sres_   += decllow( eval( olhs ) + eval( orhs ) );
//             osres_  += decllow( eval( olhs ) + eval( orhs ) );
//             refres_ += decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Decllow addition with subtraction assignment
//       //=====================================================================================
//
//       // Decllow addition with subtraction assignment with the given tensors
//       {
//          test_  = "Decllow addition with subtraction assignment with the given tensors";
//          error_ = "Failed subtraction assignment operation";
//
//          try {
//             initResults();
//             dres_   -= decllow( lhs + rhs );
//             odres_  -= decllow( lhs + rhs );
//             sres_   -= decllow( lhs + rhs );
//             osres_  -= decllow( lhs + rhs );
//             refres_ -= decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= decllow( lhs + orhs );
//             odres_  -= decllow( lhs + orhs );
//             sres_   -= decllow( lhs + orhs );
//             osres_  -= decllow( lhs + orhs );
//             refres_ -= decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= decllow( olhs + rhs );
//             odres_  -= decllow( olhs + rhs );
//             sres_   -= decllow( olhs + rhs );
//             osres_  -= decllow( olhs + rhs );
//             refres_ -= decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= decllow( olhs + orhs );
//             odres_  -= decllow( olhs + orhs );
//             sres_   -= decllow( olhs + orhs );
//             osres_  -= decllow( olhs + orhs );
//             refres_ -= decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Decllow addition with subtraction assignment with evaluated tensors
//       {
//          test_  = "Decllow addition with subtraction assignment with evaluated tensors";
//          error_ = "Failed subtraction assignment operation";
//
//          try {
//             initResults();
//             dres_   -= decllow( eval( lhs ) + eval( rhs ) );
//             odres_  -= decllow( eval( lhs ) + eval( rhs ) );
//             sres_   -= decllow( eval( lhs ) + eval( rhs ) );
//             osres_  -= decllow( eval( lhs ) + eval( rhs ) );
//             refres_ -= decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= decllow( eval( lhs ) + eval( orhs ) );
//             odres_  -= decllow( eval( lhs ) + eval( orhs ) );
//             sres_   -= decllow( eval( lhs ) + eval( orhs ) );
//             osres_  -= decllow( eval( lhs ) + eval( orhs ) );
//             refres_ -= decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= decllow( eval( olhs ) + eval( rhs ) );
//             odres_  -= decllow( eval( olhs ) + eval( rhs ) );
//             sres_   -= decllow( eval( olhs ) + eval( rhs ) );
//             osres_  -= decllow( eval( olhs ) + eval( rhs ) );
//             refres_ -= decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= decllow( eval( olhs ) + eval( orhs ) );
//             odres_  -= decllow( eval( olhs ) + eval( orhs ) );
//             sres_   -= decllow( eval( olhs ) + eval( orhs ) );
//             osres_  -= decllow( eval( olhs ) + eval( orhs ) );
//             refres_ -= decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Decllow addition with Schur product assignment
//       //=====================================================================================
//
//       // Decllow addition with Schur product assignment with the given tensors
//       {
//          test_  = "Decllow addition with Schur product assignment with the given tensors";
//          error_ = "Failed Schur product assignment operation";
//
//          try {
//             initResults();
//             dres_   %= decllow( lhs + rhs );
//             odres_  %= decllow( lhs + rhs );
//             sres_   %= decllow( lhs + rhs );
//             osres_  %= decllow( lhs + rhs );
//             refres_ %= decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= decllow( lhs + orhs );
//             odres_  %= decllow( lhs + orhs );
//             sres_   %= decllow( lhs + orhs );
//             osres_  %= decllow( lhs + orhs );
//             refres_ %= decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= decllow( olhs + rhs );
//             odres_  %= decllow( olhs + rhs );
//             sres_   %= decllow( olhs + rhs );
//             osres_  %= decllow( olhs + rhs );
//             refres_ %= decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= decllow( olhs + orhs );
//             odres_  %= decllow( olhs + orhs );
//             sres_   %= decllow( olhs + orhs );
//             osres_  %= decllow( olhs + orhs );
//             refres_ %= decllow( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Decllow addition with Schur product assignment with evaluated tensors
//       {
//          test_  = "Decllow addition with Schur product assignment with evaluated tensors";
//          error_ = "Failed Schur product assignment operation";
//
//          try {
//             initResults();
//             dres_   %= decllow( eval( lhs ) + eval( rhs ) );
//             odres_  %= decllow( eval( lhs ) + eval( rhs ) );
//             sres_   %= decllow( eval( lhs ) + eval( rhs ) );
//             osres_  %= decllow( eval( lhs ) + eval( rhs ) );
//             refres_ %= decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= decllow( eval( lhs ) + eval( orhs ) );
//             odres_  %= decllow( eval( lhs ) + eval( orhs ) );
//             sres_   %= decllow( eval( lhs ) + eval( orhs ) );
//             osres_  %= decllow( eval( lhs ) + eval( orhs ) );
//             refres_ %= decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= decllow( eval( olhs ) + eval( rhs ) );
//             odres_  %= decllow( eval( olhs ) + eval( rhs ) );
//             sres_   %= decllow( eval( olhs ) + eval( rhs ) );
//             osres_  %= decllow( eval( olhs ) + eval( rhs ) );
//             refres_ %= decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= decllow( eval( olhs ) + eval( orhs ) );
//             odres_  %= decllow( eval( olhs ) + eval( orhs ) );
//             sres_   %= decllow( eval( olhs ) + eval( orhs ) );
//             osres_  %= decllow( eval( olhs ) + eval( orhs ) );
//             refres_ %= decllow( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//    }
// #endif
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the lower dense tensor/dense tensor addition.
//
// \return void
//
// This function is called in case the lower tensor/tensor addition operation is not
// available for the given tensor types \a MT1 and \a MT2.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testDeclLowOperation( blaze::FalseType )
// {}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the upper dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the upper tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testDeclUppOperation( blaze::TrueType )
// {
// #if BLAZETEST_MATHTEST_TEST_DECLUPP_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_DECLUPP_OPERATION > 1 )
//    {
//       if( lhs_.rows() != lhs_.columns() )
//          return;
//
//
//       //=====================================================================================
//       // Test-specific setup of the left-hand side operand
//       //=====================================================================================
//
//       MT1 lhs( lhs_ );
//
//       blaze::resetLower( lhs );
//
//       OMT1 olhs  ( lhs );
//       RT1  reflhs( lhs );
//
//
//       //=====================================================================================
//       // Test-specific setup of the right-hand side operand
//       //=====================================================================================
//
//       MT2 rhs( rhs_ );
//
//       blaze::resetLower( rhs );
//
//       OMT2 orhs  ( rhs );
//       RT2  refrhs( rhs );
//
//
//       //=====================================================================================
//       // Declupp addition
//       //=====================================================================================
//
//       // Declupp addition with the given tensors
//       {
//          test_  = "Declupp addition with the given tensors";
//          error_ = "Failed addition operation";
//
//          try {
//             initResults();
//             dres_   = declupp( lhs + rhs );
//             odres_  = declupp( lhs + rhs );
//             sres_   = declupp( lhs + rhs );
//             osres_  = declupp( lhs + rhs );
//             refres_ = declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = declupp( lhs + orhs );
//             odres_  = declupp( lhs + orhs );
//             sres_   = declupp( lhs + orhs );
//             osres_  = declupp( lhs + orhs );
//             refres_ = declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = declupp( olhs + rhs );
//             odres_  = declupp( olhs + rhs );
//             sres_   = declupp( olhs + rhs );
//             osres_  = declupp( olhs + rhs );
//             refres_ = declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = declupp( olhs + orhs );
//             odres_  = declupp( olhs + orhs );
//             sres_   = declupp( olhs + orhs );
//             osres_  = declupp( olhs + orhs );
//             refres_ = declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Declupp addition with evaluated tensors
//       {
//          test_  = "Declupp addition with evaluated left-hand side tensor";
//          error_ = "Failed addition operation";
//
//          try {
//             initResults();
//             dres_   = declupp( eval( lhs ) + eval( rhs ) );
//             odres_  = declupp( eval( lhs ) + eval( rhs ) );
//             sres_   = declupp( eval( lhs ) + eval( rhs ) );
//             osres_  = declupp( eval( lhs ) + eval( rhs ) );
//             refres_ = declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = declupp( eval( lhs ) + eval( orhs ) );
//             odres_  = declupp( eval( lhs ) + eval( orhs ) );
//             sres_   = declupp( eval( lhs ) + eval( orhs ) );
//             osres_  = declupp( eval( lhs ) + eval( orhs ) );
//             refres_ = declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = declupp( eval( olhs ) + eval( rhs ) );
//             odres_  = declupp( eval( olhs ) + eval( rhs ) );
//             sres_   = declupp( eval( olhs ) + eval( rhs ) );
//             osres_  = declupp( eval( olhs ) + eval( rhs ) );
//             refres_ = declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = declupp( eval( olhs ) + eval( orhs ) );
//             odres_  = declupp( eval( olhs ) + eval( orhs ) );
//             sres_   = declupp( eval( olhs ) + eval( orhs ) );
//             osres_  = declupp( eval( olhs ) + eval( orhs ) );
//             refres_ = declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Declupp addition with addition assignment
//       //=====================================================================================
//
//       // Declupp addition with addition assignment with the given tensors
//       {
//          test_  = "Declupp addition with addition assignment with the given tensors";
//          error_ = "Failed addition assignment operation";
//
//          try {
//             initResults();
//             dres_   += declupp( lhs + rhs );
//             odres_  += declupp( lhs + rhs );
//             sres_   += declupp( lhs + rhs );
//             osres_  += declupp( lhs + rhs );
//             refres_ += declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += declupp( lhs + orhs );
//             odres_  += declupp( lhs + orhs );
//             sres_   += declupp( lhs + orhs );
//             osres_  += declupp( lhs + orhs );
//             refres_ += declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += declupp( olhs + rhs );
//             odres_  += declupp( olhs + rhs );
//             sres_   += declupp( olhs + rhs );
//             osres_  += declupp( olhs + rhs );
//             refres_ += declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += declupp( olhs + orhs );
//             odres_  += declupp( olhs + orhs );
//             sres_   += declupp( olhs + orhs );
//             osres_  += declupp( olhs + orhs );
//             refres_ += declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Declupp addition with addition assignment with evaluated tensors
//       {
//          test_  = "Declupp addition with addition assignment with evaluated tensors";
//          error_ = "Failed addition assignment operation";
//
//          try {
//             initResults();
//             dres_   += declupp( eval( lhs ) + eval( rhs ) );
//             odres_  += declupp( eval( lhs ) + eval( rhs ) );
//             sres_   += declupp( eval( lhs ) + eval( rhs ) );
//             osres_  += declupp( eval( lhs ) + eval( rhs ) );
//             refres_ += declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += declupp( eval( lhs ) + eval( orhs ) );
//             odres_  += declupp( eval( lhs ) + eval( orhs ) );
//             sres_   += declupp( eval( lhs ) + eval( orhs ) );
//             osres_  += declupp( eval( lhs ) + eval( orhs ) );
//             refres_ += declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += declupp( eval( olhs ) + eval( rhs ) );
//             odres_  += declupp( eval( olhs ) + eval( rhs ) );
//             sres_   += declupp( eval( olhs ) + eval( rhs ) );
//             osres_  += declupp( eval( olhs ) + eval( rhs ) );
//             refres_ += declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += declupp( eval( olhs ) + eval( orhs ) );
//             odres_  += declupp( eval( olhs ) + eval( orhs ) );
//             sres_   += declupp( eval( olhs ) + eval( orhs ) );
//             osres_  += declupp( eval( olhs ) + eval( orhs ) );
//             refres_ += declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Declupp addition with subtraction assignment
//       //=====================================================================================
//
//       // Declupp addition with subtraction assignment with the given tensors
//       {
//          test_  = "Declupp addition with subtraction assignment with the given tensors";
//          error_ = "Failed subtraction assignment operation";
//
//          try {
//             initResults();
//             dres_   -= declupp( lhs + rhs );
//             odres_  -= declupp( lhs + rhs );
//             sres_   -= declupp( lhs + rhs );
//             osres_  -= declupp( lhs + rhs );
//             refres_ -= declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= declupp( lhs + orhs );
//             odres_  -= declupp( lhs + orhs );
//             sres_   -= declupp( lhs + orhs );
//             osres_  -= declupp( lhs + orhs );
//             refres_ -= declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= declupp( olhs + rhs );
//             odres_  -= declupp( olhs + rhs );
//             sres_   -= declupp( olhs + rhs );
//             osres_  -= declupp( olhs + rhs );
//             refres_ -= declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= declupp( olhs + orhs );
//             odres_  -= declupp( olhs + orhs );
//             sres_   -= declupp( olhs + orhs );
//             osres_  -= declupp( olhs + orhs );
//             refres_ -= declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Declupp addition with subtraction assignment with evaluated tensors
//       {
//          test_  = "Declupp addition with subtraction assignment with evaluated tensors";
//          error_ = "Failed subtraction assignment operation";
//
//          try {
//             initResults();
//             dres_   -= declupp( eval( lhs ) + eval( rhs ) );
//             odres_  -= declupp( eval( lhs ) + eval( rhs ) );
//             sres_   -= declupp( eval( lhs ) + eval( rhs ) );
//             osres_  -= declupp( eval( lhs ) + eval( rhs ) );
//             refres_ -= declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= declupp( eval( lhs ) + eval( orhs ) );
//             odres_  -= declupp( eval( lhs ) + eval( orhs ) );
//             sres_   -= declupp( eval( lhs ) + eval( orhs ) );
//             osres_  -= declupp( eval( lhs ) + eval( orhs ) );
//             refres_ -= declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= declupp( eval( olhs ) + eval( rhs ) );
//             odres_  -= declupp( eval( olhs ) + eval( rhs ) );
//             sres_   -= declupp( eval( olhs ) + eval( rhs ) );
//             osres_  -= declupp( eval( olhs ) + eval( rhs ) );
//             refres_ -= declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= declupp( eval( olhs ) + eval( orhs ) );
//             odres_  -= declupp( eval( olhs ) + eval( orhs ) );
//             sres_   -= declupp( eval( olhs ) + eval( orhs ) );
//             osres_  -= declupp( eval( olhs ) + eval( orhs ) );
//             refres_ -= declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Declupp addition with Schur product assignment
//       //=====================================================================================
//
//       // Declupp addition with Schur product assignment with the given tensors
//       {
//          test_  = "Declupp addition with Schur product assignment with the given tensors";
//          error_ = "Failed Schur product assignment operation";
//
//          try {
//             initResults();
//             dres_   %= declupp( lhs + rhs );
//             odres_  %= declupp( lhs + rhs );
//             sres_   %= declupp( lhs + rhs );
//             osres_  %= declupp( lhs + rhs );
//             refres_ %= declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= declupp( lhs + orhs );
//             odres_  %= declupp( lhs + orhs );
//             sres_   %= declupp( lhs + orhs );
//             osres_  %= declupp( lhs + orhs );
//             refres_ %= declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= declupp( olhs + rhs );
//             odres_  %= declupp( olhs + rhs );
//             sres_   %= declupp( olhs + rhs );
//             osres_  %= declupp( olhs + rhs );
//             refres_ %= declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= declupp( olhs + orhs );
//             odres_  %= declupp( olhs + orhs );
//             sres_   %= declupp( olhs + orhs );
//             osres_  %= declupp( olhs + orhs );
//             refres_ %= declupp( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Declupp addition with Schur product assignment with evaluated tensors
//       {
//          test_  = "Declupp addition with Schur product assignment with evaluated tensors";
//          error_ = "Failed Schur product assignment operation";
//
//          try {
//             initResults();
//             dres_   %= declupp( eval( lhs ) + eval( rhs ) );
//             odres_  %= declupp( eval( lhs ) + eval( rhs ) );
//             sres_   %= declupp( eval( lhs ) + eval( rhs ) );
//             osres_  %= declupp( eval( lhs ) + eval( rhs ) );
//             refres_ %= declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= declupp( eval( lhs ) + eval( orhs ) );
//             odres_  %= declupp( eval( lhs ) + eval( orhs ) );
//             sres_   %= declupp( eval( lhs ) + eval( orhs ) );
//             osres_  %= declupp( eval( lhs ) + eval( orhs ) );
//             refres_ %= declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= declupp( eval( olhs ) + eval( rhs ) );
//             odres_  %= declupp( eval( olhs ) + eval( rhs ) );
//             sres_   %= declupp( eval( olhs ) + eval( rhs ) );
//             osres_  %= declupp( eval( olhs ) + eval( rhs ) );
//             refres_ %= declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= declupp( eval( olhs ) + eval( orhs ) );
//             odres_  %= declupp( eval( olhs ) + eval( orhs ) );
//             sres_   %= declupp( eval( olhs ) + eval( orhs ) );
//             osres_  %= declupp( eval( olhs ) + eval( orhs ) );
//             refres_ %= declupp( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//    }
// #endif
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the upper dense tensor/dense tensor addition.
//
// \return void
//
// This function is called in case the upper tensor/tensor addition operation is not
// available for the given tensor types \a MT1 and \a MT2.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testDeclUppOperation( blaze::FalseType )
// {}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the diagonal dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the diagonal tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testDeclDiagOperation( blaze::TrueType )
// {
// #if BLAZETEST_MATHTEST_TEST_DECLDIAG_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_DECLDIAG_OPERATION > 1 )
//    {
//       if( lhs_.rows() != lhs_.columns() )
//          return;
//
//
//       //=====================================================================================
//       // Test-specific setup of the left-hand side operand
//       //=====================================================================================
//
//       MT1 lhs( lhs_ );
//
//       blaze::resetLower( lhs );
//       blaze::resetUpper( lhs );
//
//       OMT1 olhs  ( lhs );
//       RT1  reflhs( lhs );
//
//
//       //=====================================================================================
//       // Test-specific setup of the right-hand side operand
//       //=====================================================================================
//
//       MT2 rhs( rhs_ );
//
//       blaze::resetLower( rhs );
//       blaze::resetUpper( rhs );
//
//       OMT2 orhs  ( rhs );
//       RT2  refrhs( rhs );
//
//
//       //=====================================================================================
//       // Decldiag addition
//       //=====================================================================================
//
//       // Decldiag addition with the given tensors
//       {
//          test_  = "Decldiag addition with the given tensors";
//          error_ = "Failed addition operation";
//
//          try {
//             initResults();
//             dres_   = decldiag( lhs + rhs );
//             odres_  = decldiag( lhs + rhs );
//             sres_   = decldiag( lhs + rhs );
//             osres_  = decldiag( lhs + rhs );
//             refres_ = decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = decldiag( lhs + orhs );
//             odres_  = decldiag( lhs + orhs );
//             sres_   = decldiag( lhs + orhs );
//             osres_  = decldiag( lhs + orhs );
//             refres_ = decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = decldiag( olhs + rhs );
//             odres_  = decldiag( olhs + rhs );
//             sres_   = decldiag( olhs + rhs );
//             osres_  = decldiag( olhs + rhs );
//             refres_ = decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = decldiag( olhs + orhs );
//             odres_  = decldiag( olhs + orhs );
//             sres_   = decldiag( olhs + orhs );
//             osres_  = decldiag( olhs + orhs );
//             refres_ = decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Decldiag addition with evaluated tensors
//       {
//          test_  = "Decldiag addition with evaluated left-hand side tensor";
//          error_ = "Failed addition operation";
//
//          try {
//             initResults();
//             dres_   = decldiag( eval( lhs ) + eval( rhs ) );
//             odres_  = decldiag( eval( lhs ) + eval( rhs ) );
//             sres_   = decldiag( eval( lhs ) + eval( rhs ) );
//             osres_  = decldiag( eval( lhs ) + eval( rhs ) );
//             refres_ = decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = decldiag( eval( lhs ) + eval( orhs ) );
//             odres_  = decldiag( eval( lhs ) + eval( orhs ) );
//             sres_   = decldiag( eval( lhs ) + eval( orhs ) );
//             osres_  = decldiag( eval( lhs ) + eval( orhs ) );
//             refres_ = decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   = decldiag( eval( olhs ) + eval( rhs ) );
//             odres_  = decldiag( eval( olhs ) + eval( rhs ) );
//             sres_   = decldiag( eval( olhs ) + eval( rhs ) );
//             osres_  = decldiag( eval( olhs ) + eval( rhs ) );
//             refres_ = decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   = decldiag( eval( olhs ) + eval( orhs ) );
//             odres_  = decldiag( eval( olhs ) + eval( orhs ) );
//             sres_   = decldiag( eval( olhs ) + eval( orhs ) );
//             osres_  = decldiag( eval( olhs ) + eval( orhs ) );
//             refres_ = decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Decldiag addition with addition assignment
//       //=====================================================================================
//
//       // Decldiag addition with addition assignment with the given tensors
//       {
//          test_  = "Decldiag addition with addition assignment with the given tensors";
//          error_ = "Failed addition assignment operation";
//
//          try {
//             initResults();
//             dres_   += decldiag( lhs + rhs );
//             odres_  += decldiag( lhs + rhs );
//             sres_   += decldiag( lhs + rhs );
//             osres_  += decldiag( lhs + rhs );
//             refres_ += decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += decldiag( lhs + orhs );
//             odres_  += decldiag( lhs + orhs );
//             sres_   += decldiag( lhs + orhs );
//             osres_  += decldiag( lhs + orhs );
//             refres_ += decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += decldiag( olhs + rhs );
//             odres_  += decldiag( olhs + rhs );
//             sres_   += decldiag( olhs + rhs );
//             osres_  += decldiag( olhs + rhs );
//             refres_ += decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += decldiag( olhs + orhs );
//             odres_  += decldiag( olhs + orhs );
//             sres_   += decldiag( olhs + orhs );
//             osres_  += decldiag( olhs + orhs );
//             refres_ += decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Decldiag addition with addition assignment with evaluated tensors
//       {
//          test_  = "Decldiag addition with addition assignment with evaluated tensors";
//          error_ = "Failed addition assignment operation";
//
//          try {
//             initResults();
//             dres_   += decldiag( eval( lhs ) + eval( rhs ) );
//             odres_  += decldiag( eval( lhs ) + eval( rhs ) );
//             sres_   += decldiag( eval( lhs ) + eval( rhs ) );
//             osres_  += decldiag( eval( lhs ) + eval( rhs ) );
//             refres_ += decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += decldiag( eval( lhs ) + eval( orhs ) );
//             odres_  += decldiag( eval( lhs ) + eval( orhs ) );
//             sres_   += decldiag( eval( lhs ) + eval( orhs ) );
//             osres_  += decldiag( eval( lhs ) + eval( orhs ) );
//             refres_ += decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   += decldiag( eval( olhs ) + eval( rhs ) );
//             odres_  += decldiag( eval( olhs ) + eval( rhs ) );
//             sres_   += decldiag( eval( olhs ) + eval( rhs ) );
//             osres_  += decldiag( eval( olhs ) + eval( rhs ) );
//             refres_ += decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   += decldiag( eval( olhs ) + eval( orhs ) );
//             odres_  += decldiag( eval( olhs ) + eval( orhs ) );
//             sres_   += decldiag( eval( olhs ) + eval( orhs ) );
//             osres_  += decldiag( eval( olhs ) + eval( orhs ) );
//             refres_ += decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Decldiag addition with subtraction assignment
//       //=====================================================================================
//
//       // Decldiag addition with subtraction assignment with the given tensors
//       {
//          test_  = "Decldiag addition with subtraction assignment with the given tensors";
//          error_ = "Failed subtraction assignment operation";
//
//          try {
//             initResults();
//             dres_   -= decldiag( lhs + rhs );
//             odres_  -= decldiag( lhs + rhs );
//             sres_   -= decldiag( lhs + rhs );
//             osres_  -= decldiag( lhs + rhs );
//             refres_ -= decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= decldiag( lhs + orhs );
//             odres_  -= decldiag( lhs + orhs );
//             sres_   -= decldiag( lhs + orhs );
//             osres_  -= decldiag( lhs + orhs );
//             refres_ -= decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= decldiag( olhs + rhs );
//             odres_  -= decldiag( olhs + rhs );
//             sres_   -= decldiag( olhs + rhs );
//             osres_  -= decldiag( olhs + rhs );
//             refres_ -= decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= decldiag( olhs + orhs );
//             odres_  -= decldiag( olhs + orhs );
//             sres_   -= decldiag( olhs + orhs );
//             osres_  -= decldiag( olhs + orhs );
//             refres_ -= decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Decldiag addition with subtraction assignment with evaluated tensors
//       {
//          test_  = "Decldiag addition with subtraction assignment with evaluated tensors";
//          error_ = "Failed subtraction assignment operation";
//
//          try {
//             initResults();
//             dres_   -= decldiag( eval( lhs ) + eval( rhs ) );
//             odres_  -= decldiag( eval( lhs ) + eval( rhs ) );
//             sres_   -= decldiag( eval( lhs ) + eval( rhs ) );
//             osres_  -= decldiag( eval( lhs ) + eval( rhs ) );
//             refres_ -= decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= decldiag( eval( lhs ) + eval( orhs ) );
//             odres_  -= decldiag( eval( lhs ) + eval( orhs ) );
//             sres_   -= decldiag( eval( lhs ) + eval( orhs ) );
//             osres_  -= decldiag( eval( lhs ) + eval( orhs ) );
//             refres_ -= decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   -= decldiag( eval( olhs ) + eval( rhs ) );
//             odres_  -= decldiag( eval( olhs ) + eval( rhs ) );
//             sres_   -= decldiag( eval( olhs ) + eval( rhs ) );
//             osres_  -= decldiag( eval( olhs ) + eval( rhs ) );
//             refres_ -= decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   -= decldiag( eval( olhs ) + eval( orhs ) );
//             odres_  -= decldiag( eval( olhs ) + eval( orhs ) );
//             sres_   -= decldiag( eval( olhs ) + eval( orhs ) );
//             osres_  -= decldiag( eval( olhs ) + eval( orhs ) );
//             refres_ -= decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Decldiag addition with Schur product assignment
//       //=====================================================================================
//
//       // Decldiag addition with Schur product assignment with the given tensors
//       {
//          test_  = "Decldiag addition with Schur product assignment with the given tensors";
//          error_ = "Failed Schur product assignment operation";
//
//          try {
//             initResults();
//             dres_   %= decldiag( lhs + rhs );
//             odres_  %= decldiag( lhs + rhs );
//             sres_   %= decldiag( lhs + rhs );
//             osres_  %= decldiag( lhs + rhs );
//             refres_ %= decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= decldiag( lhs + orhs );
//             odres_  %= decldiag( lhs + orhs );
//             sres_   %= decldiag( lhs + orhs );
//             osres_  %= decldiag( lhs + orhs );
//             refres_ %= decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= decldiag( olhs + rhs );
//             odres_  %= decldiag( olhs + rhs );
//             sres_   %= decldiag( olhs + rhs );
//             osres_  %= decldiag( olhs + rhs );
//             refres_ %= decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= decldiag( olhs + orhs );
//             odres_  %= decldiag( olhs + orhs );
//             sres_   %= decldiag( olhs + orhs );
//             osres_  %= decldiag( olhs + orhs );
//             refres_ %= decldiag( reflhs + refrhs );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Decldiag addition with Schur product assignment with evaluated tensors
//       {
//          test_  = "Decldiag addition with Schur product assignment with evaluated tensors";
//          error_ = "Failed Schur product assignment operation";
//
//          try {
//             initResults();
//             dres_   %= decldiag( eval( lhs ) + eval( rhs ) );
//             odres_  %= decldiag( eval( lhs ) + eval( rhs ) );
//             sres_   %= decldiag( eval( lhs ) + eval( rhs ) );
//             osres_  %= decldiag( eval( lhs ) + eval( rhs ) );
//             refres_ %= decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= decldiag( eval( lhs ) + eval( orhs ) );
//             odres_  %= decldiag( eval( lhs ) + eval( orhs ) );
//             sres_   %= decldiag( eval( lhs ) + eval( orhs ) );
//             osres_  %= decldiag( eval( lhs ) + eval( orhs ) );
//             refres_ %= decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             dres_   %= decldiag( eval( olhs ) + eval( rhs ) );
//             odres_  %= decldiag( eval( olhs ) + eval( rhs ) );
//             sres_   %= decldiag( eval( olhs ) + eval( rhs ) );
//             osres_  %= decldiag( eval( olhs ) + eval( rhs ) );
//             refres_ %= decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             dres_   %= decldiag( eval( olhs ) + eval( orhs ) );
//             odres_  %= decldiag( eval( olhs ) + eval( orhs ) );
//             sres_   %= decldiag( eval( olhs ) + eval( orhs ) );
//             osres_  %= decldiag( eval( olhs ) + eval( orhs ) );
//             refres_ %= decldiag( eval( reflhs ) + eval( refrhs ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//    }
// #endif
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the diagonal dense tensor/dense tensor addition.
//
// \return void
//
// This function is called in case the diagonal tensor/tensor addition operation is not
// available for the given tensor types \a MT1 and \a MT2.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testDeclDiagOperation( blaze::FalseType )
// {}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the subtensor-wise dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the subtensor-wise tensor addition with plain assignment, addition
// assignment, subtraction assignment, and Schur product assignment. In case any error resulting
// from the addition or the subsequent assignment is detected, a \a std::runtime_error exception
// is thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testSubtensorOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_SUBTENSOR_OPERATION
   if( BLAZETEST_MATHTEST_TEST_SUBTENSOR_OPERATION > 1 )
   {
      if( lhs_.rows() == 0UL || lhs_.columns() == 0UL || lhs_.pages() == 0 )
         return;


      //=====================================================================================
      // Subtensor-wise addition
      //=====================================================================================

      // Subtensor-wise addition with the given tensors
      {
         test_  = "Subtensor-wise addition with the given tensors";
         error_ = "Failed addition operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) = subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) = subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) = subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) = subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) = subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) = subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) = subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) = subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) = subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) = subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) = subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) = subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) = subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) = subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) = subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) = subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) = subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) = subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) = subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) = subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Subtensor-wise addition with evaluated tensors
      {
         test_  = "Subtensor-wise addition with evaluated tensors";
         error_ = "Failed addition operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) = subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) = subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) = subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) = subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) = subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) = subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) = subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) = subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) = subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) = subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) = subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) = subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) = subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) = subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) = subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) = subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) = subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) = subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) = subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) = subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Subtensor-wise addition with addition assignment
      //=====================================================================================

      // Subtensor-wise addition with addition assignment with the given tensors
      {
         test_  = "Subtensor-wise addition with addition assignment with the given tensors";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) += subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) += subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) += subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) += subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) += subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) += subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) += subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) += subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) += subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) += subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) += subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) += subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) += subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) += subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) += subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) += subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) += subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) += subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) += subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) += subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Subtensor-wise addition with addition assignment with evaluated tensors
      {
         test_  = "Subtensor-wise addition with addition assignment with evaluated tensors";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) += subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) += subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) += subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) += subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) += subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) += subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) += subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) += subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) += subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) += subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) += subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) += subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) += subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) += subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) += subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) += subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) += subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) += subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) += subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) += subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Subtensor-wise addition with subtraction assignment
      //=====================================================================================

      // Subtensor-wise addition with subtraction assignment with the given tensors
      {
         test_  = "Subtensor-wise addition with subtraction assignment with the given tensors";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) -= subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) -= subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) -= subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) -= subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) -= subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) -= subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) -= subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) -= subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) -= subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) -= subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) -= subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) -= subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) -= subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) -= subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) -= subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) -= subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Subtensor-wise addition with subtraction assignment with evaluated tensors
      {
         test_  = "Subtensor-wise addition with subtraction assignment with evaluated tensors";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) -= subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) -= subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) -= subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) -= subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) -= subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) -= subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) -= subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) -= subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) -= subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) -= subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) -= subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) -= subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) -= subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) -= subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) -= subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) -= subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Subtensor-wise addition with Schur product assignment
      //=====================================================================================

      // Subtensor-wise addition with Schur product assignment with the given tensors
      {
         test_  = "Subtensor-wise addition with Schur product assignment with the given tensors";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) %= subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) %= subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) %= subtensor( lhs_ + rhs_      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) %= subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) %= subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) %= subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) %= subtensor( lhs_ + orhs_     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) %= subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) %= subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) %= subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) %= subtensor( olhs_ + rhs_     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) %= subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) %= subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) %= subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) %= subtensor( olhs_ + orhs_    , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) %= subtensor( reflhs_ + refrhs_, page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // Subtensor-wise addition with Schur product assignment with evaluated tensors
      {
         test_  = "Subtensor-wise addition with Schur product assignment with evaluated tensors";
         error_ = "Failed Schur product assignment operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
               o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
               for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) %= subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) %= subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) %= subtensor( eval( lhs_ ) + eval( rhs_ )      , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) %= subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) %= subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) %= subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) %= subtensor( eval( lhs_ ) + eval( orhs_ )     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) %= subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<rhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, rhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) %= subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) %= subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) %= subtensor( eval( olhs_ ) + eval( rhs_ )     , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) %= subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t page=0UL, o=0UL; page<lhs_.pages(); page+=o ) {
//                o = blaze::rand<size_t>( 1UL, lhs_.pages() - page );
//                for( size_t row=0UL, m=0UL; row<lhs_.rows(); row+=m ) {
//                   m = blaze::rand<size_t>( 1UL, lhs_.rows() - row );
//                   for( size_t column=0UL, n=0UL; column<orhs_.columns(); column+=n ) {
//                      n = blaze::rand<size_t>( 1UL, orhs_.columns() - column );
//                      subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) %= subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) %= subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) %= subtensor( eval( olhs_ ) + eval( orhs_ )    , page, row, column, o, m, n );
//                      subtensor( refres_, page, row, column, o, m, n ) %= subtensor( eval( reflhs_ ) + eval( refrhs_ ), page, row, column, o, m, n );
//                   }
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the subtensor-wise dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function is called in case the submatrix-wise tensor/tensor addition operation is not
// available for the given tensor types \a MT1 and \a MT2.
*/
template< typename MT1    // Type of the left-hand side dense matrix
        , typename MT2 >  // Type of the right-hand side dense matrix
void OperationTest<MT1,MT2>::testSubtensorOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the row-wise dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the row-wise tensor addition with plain assignment, addition assignment,
// subtraction assignment, and multiplication assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testRowSliceOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_ROWSLICE_OPERATION
   if( BLAZETEST_MATHTEST_TEST_ROWSLICE_OPERATION > 1 )
   {
      if( lhs_.rows() == 0UL )
         return;


      //=====================================================================================
      // RowSlice-wise addition
      //=====================================================================================

      // RowSlice-wise addition with the given tensors
      {
         test_  = "RowSlice-wise addition with the given tensors";
         error_ = "Failed addition operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) = rowslice( lhs_ + rhs_, i );
//                rowslice( odres_ , i ) = rowslice( lhs_ + rhs_, i );
//                rowslice( sres_  , i ) = rowslice( lhs_ + rhs_, i );
//                rowslice( osres_ , i ) = rowslice( lhs_ + rhs_, i );
               rowslice( refres_, i ) = rowslice( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) = rowslice( lhs_ + orhs_, i );
//                rowslice( odres_ , i ) = rowslice( lhs_ + orhs_, i );
//                rowslice( sres_  , i ) = rowslice( lhs_ + orhs_, i );
//                rowslice( osres_ , i ) = rowslice( lhs_ + orhs_, i );
//                rowslice( refres_, i ) = rowslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) = rowslice( olhs_ + rhs_, i );
//                rowslice( odres_ , i ) = rowslice( olhs_ + rhs_, i );
//                rowslice( sres_  , i ) = rowslice( olhs_ + rhs_, i );
//                rowslice( osres_ , i ) = rowslice( olhs_ + rhs_, i );
//                rowslice( refres_, i ) = rowslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) = rowslice( olhs_ + orhs_, i );
//                rowslice( odres_ , i ) = rowslice( olhs_ + orhs_, i );
//                rowslice( sres_  , i ) = rowslice( olhs_ + orhs_, i );
//                rowslice( osres_ , i ) = rowslice( olhs_ + orhs_, i );
//                rowslice( refres_, i ) = rowslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // RowSlice-wise addition with evaluated tensors
      {
         test_  = "RowSlice-wise addition with evaluated tensors";
         error_ = "Failed addition operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) = rowslice( eval( lhs_ ) + eval( rhs_ ), i );
//                rowslice( odres_ , i ) = rowslice( eval( lhs_ ) + eval( rhs_ ), i );
//                rowslice( sres_  , i ) = rowslice( eval( lhs_ ) + eval( rhs_ ), i );
//                rowslice( osres_ , i ) = rowslice( eval( lhs_ ) + eval( rhs_ ), i );
               rowslice( refres_, i ) = rowslice( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) = rowslice( eval( lhs_ ) + eval( orhs_ ), i );
//                rowslice( odres_ , i ) = rowslice( eval( lhs_ ) + eval( orhs_ ), i );
//                rowslice( sres_  , i ) = rowslice( eval( lhs_ ) + eval( orhs_ ), i );
//                rowslice( osres_ , i ) = rowslice( eval( lhs_ ) + eval( orhs_ ), i );
//                rowslice( refres_, i ) = rowslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) = rowslice( eval( olhs_ ) + eval( rhs_ ), i );
//                rowslice( odres_ , i ) = rowslice( eval( olhs_ ) + eval( rhs_ ), i );
//                rowslice( sres_  , i ) = rowslice( eval( olhs_ ) + eval( rhs_ ), i );
//                rowslice( osres_ , i ) = rowslice( eval( olhs_ ) + eval( rhs_ ), i );
//                rowslice( refres_, i ) = rowslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) = rowslice( eval( olhs_ ) + eval( orhs_ ), i );
//                rowslice( odres_ , i ) = rowslice( eval( olhs_ ) + eval( orhs_ ), i );
//                rowslice( sres_  , i ) = rowslice( eval( olhs_ ) + eval( orhs_ ), i );
//                rowslice( osres_ , i ) = rowslice( eval( olhs_ ) + eval( orhs_ ), i );
//                rowslice( refres_, i ) = rowslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // RowSlice-wise addition with addition assignment
      //=====================================================================================

      // RowSlice-wise addition with addition assignment with the given tensors
      {
         test_  = "RowSlice-wise addition with addition assignment with the given tensors";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) += rowslice( lhs_ + rhs_, i );
//                rowslice( odres_ , i ) += rowslice( lhs_ + rhs_, i );
//                rowslice( sres_  , i ) += rowslice( lhs_ + rhs_, i );
//                rowslice( osres_ , i ) += rowslice( lhs_ + rhs_, i );
               rowslice( refres_, i ) += rowslice( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) += rowslice( lhs_ + orhs_, i );
//                rowslice( odres_ , i ) += rowslice( lhs_ + orhs_, i );
//                rowslice( sres_  , i ) += rowslice( lhs_ + orhs_, i );
//                rowslice( osres_ , i ) += rowslice( lhs_ + orhs_, i );
//                rowslice( refres_, i ) += rowslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) += rowslice( olhs_ + rhs_, i );
//                rowslice( odres_ , i ) += rowslice( olhs_ + rhs_, i );
//                rowslice( sres_  , i ) += rowslice( olhs_ + rhs_, i );
//                rowslice( osres_ , i ) += rowslice( olhs_ + rhs_, i );
//                rowslice( refres_, i ) += rowslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) += rowslice( olhs_ + orhs_, i );
//                rowslice( odres_ , i ) += rowslice( olhs_ + orhs_, i );
//                rowslice( sres_  , i ) += rowslice( olhs_ + orhs_, i );
//                rowslice( osres_ , i ) += rowslice( olhs_ + orhs_, i );
//                rowslice( refres_, i ) += rowslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // RowSlice-wise addition with addition assignment with evaluated tensors
      {
         test_  = "RowSlice-wise addition with addition assignment with evaluated tensors";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) += rowslice( eval( lhs_ ) + eval( rhs_ ), i );
//                rowslice( odres_ , i ) += rowslice( eval( lhs_ ) + eval( rhs_ ), i );
//                rowslice( sres_  , i ) += rowslice( eval( lhs_ ) + eval( rhs_ ), i );
//                rowslice( osres_ , i ) += rowslice( eval( lhs_ ) + eval( rhs_ ), i );
               rowslice( refres_, i ) += rowslice( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) += rowslice( eval( lhs_ ) + eval( orhs_ ), i );
//                rowslice( odres_ , i ) += rowslice( eval( lhs_ ) + eval( orhs_ ), i );
//                rowslice( sres_  , i ) += rowslice( eval( lhs_ ) + eval( orhs_ ), i );
//                rowslice( osres_ , i ) += rowslice( eval( lhs_ ) + eval( orhs_ ), i );
//                rowslice( refres_, i ) += rowslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) += rowslice( eval( olhs_ ) + eval( rhs_ ), i );
//                rowslice( odres_ , i ) += rowslice( eval( olhs_ ) + eval( rhs_ ), i );
//                rowslice( sres_  , i ) += rowslice( eval( olhs_ ) + eval( rhs_ ), i );
//                rowslice( osres_ , i ) += rowslice( eval( olhs_ ) + eval( rhs_ ), i );
//                rowslice( refres_, i ) += rowslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) += rowslice( eval( olhs_ ) + eval( orhs_ ), i );
//                rowslice( odres_ , i ) += rowslice( eval( olhs_ ) + eval( orhs_ ), i );
//                rowslice( sres_  , i ) += rowslice( eval( olhs_ ) + eval( orhs_ ), i );
//                rowslice( osres_ , i ) += rowslice( eval( olhs_ ) + eval( orhs_ ), i );
//                rowslice( refres_, i ) += rowslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // RowSlice-wise addition with subtraction assignment
      //=====================================================================================

      // RowSlice-wise addition with subtraction assignment with the given tensors
      {
         test_  = "RowSlice-wise addition with subtraction assignment with the given tensors";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) -= rowslice( lhs_ + rhs_, i );
//                rowslice( odres_ , i ) -= rowslice( lhs_ + rhs_, i );
//                rowslice( sres_  , i ) -= rowslice( lhs_ + rhs_, i );
//                rowslice( osres_ , i ) -= rowslice( lhs_ + rhs_, i );
               rowslice( refres_, i ) -= rowslice( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) -= rowslice( lhs_ + orhs_, i );
//                rowslice( odres_ , i ) -= rowslice( lhs_ + orhs_, i );
//                rowslice( sres_  , i ) -= rowslice( lhs_ + orhs_, i );
//                rowslice( osres_ , i ) -= rowslice( lhs_ + orhs_, i );
//                rowslice( refres_, i ) -= rowslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) -= rowslice( olhs_ + rhs_, i );
//                rowslice( odres_ , i ) -= rowslice( olhs_ + rhs_, i );
//                rowslice( sres_  , i ) -= rowslice( olhs_ + rhs_, i );
//                rowslice( osres_ , i ) -= rowslice( olhs_ + rhs_, i );
//                rowslice( refres_, i ) -= rowslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) -= rowslice( olhs_ + orhs_, i );
//                rowslice( odres_ , i ) -= rowslice( olhs_ + orhs_, i );
//                rowslice( sres_  , i ) -= rowslice( olhs_ + orhs_, i );
//                rowslice( osres_ , i ) -= rowslice( olhs_ + orhs_, i );
//                rowslice( refres_, i ) -= rowslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // RowSlice-wise addition with subtraction assignment with evaluated tensors
      {
         test_  = "RowSlice-wise addition with subtraction assignment with evaluated tensors";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.rows(); ++i ) {
               rowslice( dres_  , i ) -= rowslice( eval( lhs_ ) + eval( rhs_ ), i );
//                rowslice( odres_ , i ) -= rowslice( eval( lhs_ ) + eval( rhs_ ), i );
//                rowslice( sres_  , i ) -= rowslice( eval( lhs_ ) + eval( rhs_ ), i );
//                rowslice( osres_ , i ) -= rowslice( eval( lhs_ ) + eval( rhs_ ), i );
               rowslice( refres_, i ) -= rowslice( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) -= rowslice( eval( lhs_ ) + eval( orhs_ ), i );
//                rowslice( odres_ , i ) -= rowslice( eval( lhs_ ) + eval( orhs_ ), i );
//                rowslice( sres_  , i ) -= rowslice( eval( lhs_ ) + eval( orhs_ ), i );
//                rowslice( osres_ , i ) -= rowslice( eval( lhs_ ) + eval( orhs_ ), i );
//                rowslice( refres_, i ) -= rowslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) -= rowslice( eval( olhs_ ) + eval( rhs_ ), i );
//                rowslice( odres_ , i ) -= rowslice( eval( olhs_ ) + eval( rhs_ ), i );
//                rowslice( sres_  , i ) -= rowslice( eval( olhs_ ) + eval( rhs_ ), i );
//                rowslice( osres_ , i ) -= rowslice( eval( olhs_ ) + eval( rhs_ ), i );
//                rowslice( refres_, i ) -= rowslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.rows(); ++i ) {
//                rowslice( dres_  , i ) -= rowslice( eval( olhs_ ) + eval( orhs_ ), i );
//                rowslice( odres_ , i ) -= rowslice( eval( olhs_ ) + eval( orhs_ ), i );
//                rowslice( sres_  , i ) -= rowslice( eval( olhs_ ) + eval( orhs_ ), i );
//                rowslice( osres_ , i ) -= rowslice( eval( olhs_ ) + eval( orhs_ ), i );
//                rowslice( refres_, i ) -= rowslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the rowslice-wise dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function is called in case the rowslice-wise tensor/tensor addition operation is not
// available for the given matrix types \a MT1 and \a MT2.
*/
template< typename MT1    // Type of the left-hand side dense matrix
        , typename MT2 >  // Type of the right-hand side dense matrix
void OperationTest<MT1,MT2>::testRowSliceOperation( blaze::FalseType )
{}
//*************************************************************************************************


#if 0
//*************************************************************************************************
/*!\brief Testing the rows-wise dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the rows-wise tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testRowSlicesOperation( blaze::TrueType )
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
//       // Rows-wise addition
//       //=====================================================================================
//
//       // Rows-wise addition with the given tensors
//       {
//          test_  = "Rows-wise addition with the given tensors";
//          error_ = "Failed addition operation";
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
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
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
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
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
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
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
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Rows-wise addition with evaluated tensors
//       {
//          test_  = "Rows-wise addition with evaluated tensors";
//          error_ = "Failed addition operation";
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
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
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
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
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
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
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
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Rows-wise addition with addition assignment
//       //=====================================================================================
//
//       // Rows-wise addition with addition assignment with the given tensors
//       {
//          test_  = "Rows-wise addition with addition assignment with the given tensors";
//          error_ = "Failed addition assignment operation";
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
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
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
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
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
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
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
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Rows-wise addition with addition assignment with evaluated tensors
//       {
//          test_  = "Rows-wise addition with addition assignment with evaluated tensors";
//          error_ = "Failed addition assignment operation";
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
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
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
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
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
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
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
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Rows-wise addition with subtraction assignment
//       //=====================================================================================
//
//       // Rows-wise addition with subtraction assignment with the given tensors
//       {
//          test_  = "Rows-wise addition with subtraction assignment with the given tensors";
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
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
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
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
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
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
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
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Rows-wise addition with subtraction assignment with evaluated tensors
//       {
//          test_  = "Rows-wise addition with subtraction assignment with evaluated tensors";
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
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
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
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
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
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
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
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//
//       //=====================================================================================
//       // Rows-wise addition with Schur product assignment
//       //=====================================================================================
//
//       // Rows-wise addition with Schur product assignment with the given tensors
//       {
//          test_  = "Rows-wise addition with Schur product assignment with the given tensors";
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
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
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
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
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
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
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
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//
//       // Rows-wise addition with Schur product assignment with evaluated tensors
//       {
//          test_  = "Rows-wise addition with Schur product assignment with evaluated tensors";
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
//             convertException<MT1,MT2>( ex );
//          }
//
//          checkResults<MT1,MT2>();
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
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
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
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
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
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
//       }
//    }
// #endif
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the rows-wise dense tensor/dense tensor addition.
//
// \return void
//
// This function is called in case the rows-wise tensor/tensor addition operation is not
// available for the given tensor types \a MT1 and \a MT2.
*/
// template< typename MT1    // Type of the left-hand side dense tensor
//         , typename MT2 >  // Type of the right-hand side dense tensor
// void OperationTest<MT1,MT2>::testRowSlicesOperation( blaze::FalseType )
// {}
//*************************************************************************************************
#endif


//*************************************************************************************************
/*!\brief Testing the row-wise dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the row-wise tensor addition with plain assignment, addition assignment,
// subtraction assignment, and multiplication assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testColumnSliceOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_COLUMNSLICE_OPERATION
   if( BLAZETEST_MATHTEST_TEST_COLUMNSLICE_OPERATION > 1 )
   {
      if( lhs_.columns() == 0UL )
         return;


      //=====================================================================================
      // ColumnSlice-wise addition
      //=====================================================================================

      // ColumnSlice-wise addition with the given tensors
      {
         test_  = "ColumnSlice-wise addition with the given tensors";
         error_ = "Failed addition operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) = columnslice( lhs_ + rhs_, i );
//                columnslice( odres_ , i ) = columnslice( lhs_ + rhs_, i );
//                columnslice( sres_  , i ) = columnslice( lhs_ + rhs_, i );
//                columnslice( osres_ , i ) = columnslice( lhs_ + rhs_, i );
               columnslice( refres_, i ) = columnslice( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) = columnslice( lhs_ + orhs_, i );
//                columnslice( odres_ , i ) = columnslice( lhs_ + orhs_, i );
//                columnslice( sres_  , i ) = columnslice( lhs_ + orhs_, i );
//                columnslice( osres_ , i ) = columnslice( lhs_ + orhs_, i );
//                columnslice( refres_, i ) = columnslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) = columnslice( olhs_ + rhs_, i );
//                columnslice( odres_ , i ) = columnslice( olhs_ + rhs_, i );
//                columnslice( sres_  , i ) = columnslice( olhs_ + rhs_, i );
//                columnslice( osres_ , i ) = columnslice( olhs_ + rhs_, i );
//                columnslice( refres_, i ) = columnslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) = columnslice( olhs_ + orhs_, i );
//                columnslice( odres_ , i ) = columnslice( olhs_ + orhs_, i );
//                columnslice( sres_  , i ) = columnslice( olhs_ + orhs_, i );
//                columnslice( osres_ , i ) = columnslice( olhs_ + orhs_, i );
//                columnslice( refres_, i ) = columnslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // ColumnSlice-wise addition with evaluated tensors
      {
         test_  = "ColumnSlice-wise addition with evaluated tensors";
         error_ = "Failed addition operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) = columnslice( eval( lhs_ ) + eval( rhs_ ), i );
//                columnslice( odres_ , i ) = columnslice( eval( lhs_ ) + eval( rhs_ ), i );
//                columnslice( sres_  , i ) = columnslice( eval( lhs_ ) + eval( rhs_ ), i );
//                columnslice( osres_ , i ) = columnslice( eval( lhs_ ) + eval( rhs_ ), i );
               columnslice( refres_, i ) = columnslice( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) = columnslice( eval( lhs_ ) + eval( orhs_ ), i );
//                columnslice( odres_ , i ) = columnslice( eval( lhs_ ) + eval( orhs_ ), i );
//                columnslice( sres_  , i ) = columnslice( eval( lhs_ ) + eval( orhs_ ), i );
//                columnslice( osres_ , i ) = columnslice( eval( lhs_ ) + eval( orhs_ ), i );
//                columnslice( refres_, i ) = columnslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) = columnslice( eval( olhs_ ) + eval( rhs_ ), i );
//                columnslice( odres_ , i ) = columnslice( eval( olhs_ ) + eval( rhs_ ), i );
//                columnslice( sres_  , i ) = columnslice( eval( olhs_ ) + eval( rhs_ ), i );
//                columnslice( osres_ , i ) = columnslice( eval( olhs_ ) + eval( rhs_ ), i );
//                columnslice( refres_, i ) = columnslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) = columnslice( eval( olhs_ ) + eval( orhs_ ), i );
//                columnslice( odres_ , i ) = columnslice( eval( olhs_ ) + eval( orhs_ ), i );
//                columnslice( sres_  , i ) = columnslice( eval( olhs_ ) + eval( orhs_ ), i );
//                columnslice( osres_ , i ) = columnslice( eval( olhs_ ) + eval( orhs_ ), i );
//                columnslice( refres_, i ) = columnslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // ColumnSlice-wise addition with addition assignment
      //=====================================================================================

      // ColumnSlice-wise addition with addition assignment with the given tensors
      {
         test_  = "ColumnSlice-wise addition with addition assignment with the given tensors";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) += columnslice( lhs_ + rhs_, i );
//                columnslice( odres_ , i ) += columnslice( lhs_ + rhs_, i );
//                columnslice( sres_  , i ) += columnslice( lhs_ + rhs_, i );
//                columnslice( osres_ , i ) += columnslice( lhs_ + rhs_, i );
               columnslice( refres_, i ) += columnslice( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) += columnslice( lhs_ + orhs_, i );
//                columnslice( odres_ , i ) += columnslice( lhs_ + orhs_, i );
//                columnslice( sres_  , i ) += columnslice( lhs_ + orhs_, i );
//                columnslice( osres_ , i ) += columnslice( lhs_ + orhs_, i );
//                columnslice( refres_, i ) += columnslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) += columnslice( olhs_ + rhs_, i );
//                columnslice( odres_ , i ) += columnslice( olhs_ + rhs_, i );
//                columnslice( sres_  , i ) += columnslice( olhs_ + rhs_, i );
//                columnslice( osres_ , i ) += columnslice( olhs_ + rhs_, i );
//                columnslice( refres_, i ) += columnslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) += columnslice( olhs_ + orhs_, i );
//                columnslice( odres_ , i ) += columnslice( olhs_ + orhs_, i );
//                columnslice( sres_  , i ) += columnslice( olhs_ + orhs_, i );
//                columnslice( osres_ , i ) += columnslice( olhs_ + orhs_, i );
//                columnslice( refres_, i ) += columnslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // ColumnSlice-wise addition with addition assignment with evaluated tensors
      {
         test_  = "ColumnSlice-wise addition with addition assignment with evaluated tensors";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) += columnslice( eval( lhs_ ) + eval( rhs_ ), i );
//                columnslice( odres_ , i ) += columnslice( eval( lhs_ ) + eval( rhs_ ), i );
//                columnslice( sres_  , i ) += columnslice( eval( lhs_ ) + eval( rhs_ ), i );
//                columnslice( osres_ , i ) += columnslice( eval( lhs_ ) + eval( rhs_ ), i );
               columnslice( refres_, i ) += columnslice( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) += columnslice( eval( lhs_ ) + eval( orhs_ ), i );
//                columnslice( odres_ , i ) += columnslice( eval( lhs_ ) + eval( orhs_ ), i );
//                columnslice( sres_  , i ) += columnslice( eval( lhs_ ) + eval( orhs_ ), i );
//                columnslice( osres_ , i ) += columnslice( eval( lhs_ ) + eval( orhs_ ), i );
//                columnslice( refres_, i ) += columnslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) += columnslice( eval( olhs_ ) + eval( rhs_ ), i );
//                columnslice( odres_ , i ) += columnslice( eval( olhs_ ) + eval( rhs_ ), i );
//                columnslice( sres_  , i ) += columnslice( eval( olhs_ ) + eval( rhs_ ), i );
//                columnslice( osres_ , i ) += columnslice( eval( olhs_ ) + eval( rhs_ ), i );
//                columnslice( refres_, i ) += columnslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) += columnslice( eval( olhs_ ) + eval( orhs_ ), i );
//                columnslice( odres_ , i ) += columnslice( eval( olhs_ ) + eval( orhs_ ), i );
//                columnslice( sres_  , i ) += columnslice( eval( olhs_ ) + eval( orhs_ ), i );
//                columnslice( osres_ , i ) += columnslice( eval( olhs_ ) + eval( orhs_ ), i );
//                columnslice( refres_, i ) += columnslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // ColumnSlice-wise addition with subtraction assignment
      //=====================================================================================

      // ColumnSlice-wise addition with subtraction assignment with the given tensors
      {
         test_  = "ColumnSlice-wise addition with subtraction assignment with the given tensors";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) -= columnslice( lhs_ + rhs_, i );
//                columnslice( odres_ , i ) -= columnslice( lhs_ + rhs_, i );
//                columnslice( sres_  , i ) -= columnslice( lhs_ + rhs_, i );
//                columnslice( osres_ , i ) -= columnslice( lhs_ + rhs_, i );
               columnslice( refres_, i ) -= columnslice( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) -= columnslice( lhs_ + orhs_, i );
//                columnslice( odres_ , i ) -= columnslice( lhs_ + orhs_, i );
//                columnslice( sres_  , i ) -= columnslice( lhs_ + orhs_, i );
//                columnslice( osres_ , i ) -= columnslice( lhs_ + orhs_, i );
//                columnslice( refres_, i ) -= columnslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) -= columnslice( olhs_ + rhs_, i );
//                columnslice( odres_ , i ) -= columnslice( olhs_ + rhs_, i );
//                columnslice( sres_  , i ) -= columnslice( olhs_ + rhs_, i );
//                columnslice( osres_ , i ) -= columnslice( olhs_ + rhs_, i );
//                columnslice( refres_, i ) -= columnslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) -= columnslice( olhs_ + orhs_, i );
//                columnslice( odres_ , i ) -= columnslice( olhs_ + orhs_, i );
//                columnslice( sres_  , i ) -= columnslice( olhs_ + orhs_, i );
//                columnslice( osres_ , i ) -= columnslice( olhs_ + orhs_, i );
//                columnslice( refres_, i ) -= columnslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // ColumnSlice-wise addition with subtraction assignment with evaluated tensors
      {
         test_  = "ColumnSlice-wise addition with subtraction assignment with evaluated tensors";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.columns(); ++i ) {
               columnslice( dres_  , i ) -= columnslice( eval( lhs_ ) + eval( rhs_ ), i );
//                columnslice( odres_ , i ) -= columnslice( eval( lhs_ ) + eval( rhs_ ), i );
//                columnslice( sres_  , i ) -= columnslice( eval( lhs_ ) + eval( rhs_ ), i );
//                columnslice( osres_ , i ) -= columnslice( eval( lhs_ ) + eval( rhs_ ), i );
               columnslice( refres_, i ) -= columnslice( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) -= columnslice( eval( lhs_ ) + eval( orhs_ ), i );
//                columnslice( odres_ , i ) -= columnslice( eval( lhs_ ) + eval( orhs_ ), i );
//                columnslice( sres_  , i ) -= columnslice( eval( lhs_ ) + eval( orhs_ ), i );
//                columnslice( osres_ , i ) -= columnslice( eval( lhs_ ) + eval( orhs_ ), i );
//                columnslice( refres_, i ) -= columnslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) -= columnslice( eval( olhs_ ) + eval( rhs_ ), i );
//                columnslice( odres_ , i ) -= columnslice( eval( olhs_ ) + eval( rhs_ ), i );
//                columnslice( sres_  , i ) -= columnslice( eval( olhs_ ) + eval( rhs_ ), i );
//                columnslice( osres_ , i ) -= columnslice( eval( olhs_ ) + eval( rhs_ ), i );
//                columnslice( refres_, i ) -= columnslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.columns(); ++i ) {
//                columnslice( dres_  , i ) -= columnslice( eval( olhs_ ) + eval( orhs_ ), i );
//                columnslice( odres_ , i ) -= columnslice( eval( olhs_ ) + eval( orhs_ ), i );
//                columnslice( sres_  , i ) -= columnslice( eval( olhs_ ) + eval( orhs_ ), i );
//                columnslice( osres_ , i ) -= columnslice( eval( olhs_ ) + eval( orhs_ ), i );
//                columnslice( refres_, i ) -= columnslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the columnslice-wise dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function is called in case the columnslice-wise tensor/tensor addition operation is not
// available for the given matrix types \a MT1 and \a MT2.
*/
template< typename MT1    // Type of the left-hand side dense matrix
        , typename MT2 >  // Type of the right-hand side dense matrix
void OperationTest<MT1,MT2>::testColumnSliceOperation( blaze::FalseType )
{}
//*************************************************************************************************



//*************************************************************************************************
/*!\brief Testing the row-wise dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the row-wise tensor addition with plain assignment, addition assignment,
// subtraction assignment, and multiplication assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testPageSliceOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_PAGESLICE_OPERATION
   if( BLAZETEST_MATHTEST_TEST_PAGESLICE_OPERATION > 1 )
   {
      if( lhs_.pages() == 0UL )
         return;


      //=====================================================================================
      // PageSlice-wise addition
      //=====================================================================================

      // PageSlice-wise addition with the given tensors
      {
         test_  = "PageSlice-wise addition with the given tensors";
         error_ = "Failed addition operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) = pageslice( lhs_ + rhs_, i );
//                pageslice( odres_ , i ) = pageslice( lhs_ + rhs_, i );
//                pageslice( sres_  , i ) = pageslice( lhs_ + rhs_, i );
//                pageslice( osres_ , i ) = pageslice( lhs_ + rhs_, i );
               pageslice( refres_, i ) = pageslice( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) = pageslice( lhs_ + orhs_, i );
//                pageslice( odres_ , i ) = pageslice( lhs_ + orhs_, i );
//                pageslice( sres_  , i ) = pageslice( lhs_ + orhs_, i );
//                pageslice( osres_ , i ) = pageslice( lhs_ + orhs_, i );
//                pageslice( refres_, i ) = pageslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) = pageslice( olhs_ + rhs_, i );
//                pageslice( odres_ , i ) = pageslice( olhs_ + rhs_, i );
//                pageslice( sres_  , i ) = pageslice( olhs_ + rhs_, i );
//                pageslice( osres_ , i ) = pageslice( olhs_ + rhs_, i );
//                pageslice( refres_, i ) = pageslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) = pageslice( olhs_ + orhs_, i );
//                pageslice( odres_ , i ) = pageslice( olhs_ + orhs_, i );
//                pageslice( sres_  , i ) = pageslice( olhs_ + orhs_, i );
//                pageslice( osres_ , i ) = pageslice( olhs_ + orhs_, i );
//                pageslice( refres_, i ) = pageslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // PageSlice-wise addition with evaluated tensors
      {
         test_  = "PageSlice-wise addition with evaluated tensors";
         error_ = "Failed addition operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) = pageslice( eval( lhs_ ) + eval( rhs_ ), i );
//                pageslice( odres_ , i ) = pageslice( eval( lhs_ ) + eval( rhs_ ), i );
//                pageslice( sres_  , i ) = pageslice( eval( lhs_ ) + eval( rhs_ ), i );
//                pageslice( osres_ , i ) = pageslice( eval( lhs_ ) + eval( rhs_ ), i );
               pageslice( refres_, i ) = pageslice( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) = pageslice( eval( lhs_ ) + eval( orhs_ ), i );
//                pageslice( odres_ , i ) = pageslice( eval( lhs_ ) + eval( orhs_ ), i );
//                pageslice( sres_  , i ) = pageslice( eval( lhs_ ) + eval( orhs_ ), i );
//                pageslice( osres_ , i ) = pageslice( eval( lhs_ ) + eval( orhs_ ), i );
//                pageslice( refres_, i ) = pageslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) = pageslice( eval( olhs_ ) + eval( rhs_ ), i );
//                pageslice( odres_ , i ) = pageslice( eval( olhs_ ) + eval( rhs_ ), i );
//                pageslice( sres_  , i ) = pageslice( eval( olhs_ ) + eval( rhs_ ), i );
//                pageslice( osres_ , i ) = pageslice( eval( olhs_ ) + eval( rhs_ ), i );
//                pageslice( refres_, i ) = pageslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) = pageslice( eval( olhs_ ) + eval( orhs_ ), i );
//                pageslice( odres_ , i ) = pageslice( eval( olhs_ ) + eval( orhs_ ), i );
//                pageslice( sres_  , i ) = pageslice( eval( olhs_ ) + eval( orhs_ ), i );
//                pageslice( osres_ , i ) = pageslice( eval( olhs_ ) + eval( orhs_ ), i );
//                pageslice( refres_, i ) = pageslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // PageSlice-wise addition with addition assignment
      //=====================================================================================

      // PageSlice-wise addition with addition assignment with the given tensors
      {
         test_  = "PageSlice-wise addition with addition assignment with the given tensors";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) += pageslice( lhs_ + rhs_, i );
//                pageslice( odres_ , i ) += pageslice( lhs_ + rhs_, i );
//                pageslice( sres_  , i ) += pageslice( lhs_ + rhs_, i );
//                pageslice( osres_ , i ) += pageslice( lhs_ + rhs_, i );
               pageslice( refres_, i ) += pageslice( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) += pageslice( lhs_ + orhs_, i );
//                pageslice( odres_ , i ) += pageslice( lhs_ + orhs_, i );
//                pageslice( sres_  , i ) += pageslice( lhs_ + orhs_, i );
//                pageslice( osres_ , i ) += pageslice( lhs_ + orhs_, i );
//                pageslice( refres_, i ) += pageslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) += pageslice( olhs_ + rhs_, i );
//                pageslice( odres_ , i ) += pageslice( olhs_ + rhs_, i );
//                pageslice( sres_  , i ) += pageslice( olhs_ + rhs_, i );
//                pageslice( osres_ , i ) += pageslice( olhs_ + rhs_, i );
//                pageslice( refres_, i ) += pageslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) += pageslice( olhs_ + orhs_, i );
//                pageslice( odres_ , i ) += pageslice( olhs_ + orhs_, i );
//                pageslice( sres_  , i ) += pageslice( olhs_ + orhs_, i );
//                pageslice( osres_ , i ) += pageslice( olhs_ + orhs_, i );
//                pageslice( refres_, i ) += pageslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

// PageSlice-wise addition with addition assignment with evaluated tensors
      {
         test_  = "PageSlice-wise addition with addition assignment with evaluated tensors";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) += pageslice( eval( lhs_ ) + eval( rhs_ ), i );
//                pageslice( odres_ , i ) += pageslice( eval( lhs_ ) + eval( rhs_ ), i );
//                pageslice( sres_  , i ) += pageslice( eval( lhs_ ) + eval( rhs_ ), i );
//                pageslice( osres_ , i ) += pageslice( eval( lhs_ ) + eval( rhs_ ), i );
               pageslice( refres_, i ) += pageslice( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) += pageslice( eval( lhs_ ) + eval( orhs_ ), i );
//                pageslice( odres_ , i ) += pageslice( eval( lhs_ ) + eval( orhs_ ), i );
//                pageslice( sres_  , i ) += pageslice( eval( lhs_ ) + eval( orhs_ ), i );
//                pageslice( osres_ , i ) += pageslice( eval( lhs_ ) + eval( orhs_ ), i );
//                pageslice( refres_, i ) += pageslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) += pageslice( eval( olhs_ ) + eval( rhs_ ), i );
//                pageslice( odres_ , i ) += pageslice( eval( olhs_ ) + eval( rhs_ ), i );
//                pageslice( sres_  , i ) += pageslice( eval( olhs_ ) + eval( rhs_ ), i );
//                pageslice( osres_ , i ) += pageslice( eval( olhs_ ) + eval( rhs_ ), i );
//                pageslice( refres_, i ) += pageslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) += pageslice( eval( olhs_ ) + eval( orhs_ ), i );
//                pageslice( odres_ , i ) += pageslice( eval( olhs_ ) + eval( orhs_ ), i );
//                pageslice( sres_  , i ) += pageslice( eval( olhs_ ) + eval( orhs_ ), i );
//                pageslice( osres_ , i ) += pageslice( eval( olhs_ ) + eval( orhs_ ), i );
//                pageslice( refres_, i ) += pageslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // PageSlice-wise addition with subtraction assignment
      //=====================================================================================

      // PageSlice-wise addition with subtraction assignment with the given tensors
      {
         test_  = "PageSlice-wise addition with subtraction assignment with the given tensors";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) -= pageslice( lhs_ + rhs_, i );
//                pageslice( odres_ , i ) -= pageslice( lhs_ + rhs_, i );
//                pageslice( sres_  , i ) -= pageslice( lhs_ + rhs_, i );
//                pageslice( osres_ , i ) -= pageslice( lhs_ + rhs_, i );
               pageslice( refres_, i ) -= pageslice( reflhs_ + refrhs_, i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) -= pageslice( lhs_ + orhs_, i );
//                pageslice( odres_ , i ) -= pageslice( lhs_ + orhs_, i );
//                pageslice( sres_  , i ) -= pageslice( lhs_ + orhs_, i );
//                pageslice( osres_ , i ) -= pageslice( lhs_ + orhs_, i );
//                pageslice( refres_, i ) -= pageslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) -= pageslice( olhs_ + rhs_, i );
//                pageslice( odres_ , i ) -= pageslice( olhs_ + rhs_, i );
//                pageslice( sres_  , i ) -= pageslice( olhs_ + rhs_, i );
//                pageslice( osres_ , i ) -= pageslice( olhs_ + rhs_, i );
//                pageslice( refres_, i ) -= pageslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) -= pageslice( olhs_ + orhs_, i );
//                pageslice( odres_ , i ) -= pageslice( olhs_ + orhs_, i );
//                pageslice( sres_  , i ) -= pageslice( olhs_ + orhs_, i );
//                pageslice( osres_ , i ) -= pageslice( olhs_ + orhs_, i );
//                pageslice( refres_, i ) -= pageslice( reflhs_ + refrhs_, i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }

      // PageSlice-wise addition with subtraction assignment with evaluated tensors
      {
         test_  = "PageSlice-wise addition with subtraction assignment with evaluated tensors";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t i=0UL; i<lhs_.pages(); ++i ) {
               pageslice( dres_  , i ) -= pageslice( eval( lhs_ ) + eval( rhs_ ), i );
//                pageslice( odres_ , i ) -= pageslice( eval( lhs_ ) + eval( rhs_ ), i );
//                pageslice( sres_  , i ) -= pageslice( eval( lhs_ ) + eval( rhs_ ), i );
//                pageslice( osres_ , i ) -= pageslice( eval( lhs_ ) + eval( rhs_ ), i );
               pageslice( refres_, i ) -= pageslice( eval( reflhs_ ) + eval( refrhs_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) -= pageslice( eval( lhs_ ) + eval( orhs_ ), i );
//                pageslice( odres_ , i ) -= pageslice( eval( lhs_ ) + eval( orhs_ ), i );
//                pageslice( sres_  , i ) -= pageslice( eval( lhs_ ) + eval( orhs_ ), i );
//                pageslice( osres_ , i ) -= pageslice( eval( lhs_ ) + eval( orhs_ ), i );
//                pageslice( refres_, i ) -= pageslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT1,OMT2>( ex );
//          }
//
//          checkResults<MT1,OMT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) -= pageslice( eval( olhs_ ) + eval( rhs_ ), i );
//                pageslice( odres_ , i ) -= pageslice( eval( olhs_ ) + eval( rhs_ ), i );
//                pageslice( sres_  , i ) -= pageslice( eval( olhs_ ) + eval( rhs_ ), i );
//                pageslice( osres_ , i ) -= pageslice( eval( olhs_ ) + eval( rhs_ ), i );
//                pageslice( refres_, i ) -= pageslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,MT2>( ex );
//          }
//
//          checkResults<OMT1,MT2>();
//
//          try {
//             initResults();
//             for( size_t i=0UL; i<lhs_.pages(); ++i ) {
//                pageslice( dres_  , i ) -= pageslice( eval( olhs_ ) + eval( orhs_ ), i );
//                pageslice( odres_ , i ) -= pageslice( eval( olhs_ ) + eval( orhs_ ), i );
//                pageslice( sres_  , i ) -= pageslice( eval( olhs_ ) + eval( orhs_ ), i );
//                pageslice( osres_ , i ) -= pageslice( eval( olhs_ ) + eval( orhs_ ), i );
//                pageslice( refres_, i ) -= pageslice( eval( reflhs_ ) + eval( refrhs_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<OMT1,OMT2>( ex );
//          }
//
//          checkResults<OMT1,OMT2>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the pageslice-wise dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function is called in case the pageslice-wise tensor/tensor addition operation is not
// available for the given matrix types \a MT1 and \a MT2.
*/
template< typename MT1    // Type of the left-hand side dense matrix
        , typename MT2 >  // Type of the right-hand side dense matrix
void OperationTest<MT1,MT2>::testPageSliceOperation( blaze::FalseType )
{}
//*************************************************************************************************


#if 0
//*************************************************************************************************
/*!\brief Testing the column-wise dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the column-wise tensor addition with plain assignment, addition assignment,
// subtraction assignment, and multiplication assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testColumnOperation()
{
#if BLAZETEST_MATHTEST_TEST_COLUMN_OPERATION
   if( BLAZETEST_MATHTEST_TEST_COLUMN_OPERATION > 1 )
   {
      if( lhs_.columns() == 0UL )
         return;


      //=====================================================================================
      // Column-wise addition
      //=====================================================================================

      // Column-wise addition with the given tensors
      {
         test_  = "Column-wise addition with the given tensors";
         error_ = "Failed addition operation";

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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }

      // Column-wise addition with evaluated tensors
      {
         test_  = "Column-wise addition with evaluated tensors";
         error_ = "Failed addition operation";

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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Column-wise addition with addition assignment
      //=====================================================================================

      // Column-wise addition with addition assignment with the given tensors
      {
         test_  = "Column-wise addition with addition assignment with the given tensors";
         error_ = "Failed addition assignment operation";

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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }

      // Column-wise addition with addition assignment with evaluated tensors
      {
         test_  = "Column-wise addition with addition assignment with evaluated tensors";
         error_ = "Failed addition assignment operation";

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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Column-wise addition with subtraction assignment
      //=====================================================================================

      // Column-wise addition with subtraction assignment with the given tensors
      {
         test_  = "Column-wise addition with subtraction assignment with the given tensors";
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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }

      // Column-wise addition with subtraction assignment with evaluated tensors
      {
         test_  = "Column-wise addition with subtraction assignment with evaluated tensors";
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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Column-wise addition with multiplication assignment
      //=====================================================================================

      // Column-wise addition with multiplication assignment with the given tensors
      {
         test_  = "Column-wise addition with multiplication assignment with the given tensors";
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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }

      // Column-wise addition with multiplication assignment with evaluated tensors
      {
         test_  = "Column-wise addition with multiplication assignment with evaluated tensors";
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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the columns-wise dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the columns-wise tensor addition with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testColumnsOperation( blaze::TrueType )
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
      // Columns-wise addition
      //=====================================================================================

      // Columns-wise addition with the given tensors
      {
         test_  = "Columns-wise addition with the given tensors";
         error_ = "Failed addition operation";

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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }

      // Columns-wise addition with evaluated tensors
      {
         test_  = "Columns-wise addition with evaluated tensors";
         error_ = "Failed addition operation";

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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Columns-wise addition with addition assignment
      //=====================================================================================

      // Columns-wise addition with addition assignment with the given tensors
      {
         test_  = "Columns-wise addition with addition assignment with the given tensors";
         error_ = "Failed addition assignment operation";

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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }

      // Columns-wise addition with addition assignment with evaluated tensors
      {
         test_  = "Columns-wise addition with addition assignment with evaluated tensors";
         error_ = "Failed addition assignment operation";

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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Columns-wise addition with subtraction assignment
      //=====================================================================================

      // Columns-wise addition with subtraction assignment with the given tensors
      {
         test_  = "Columns-wise addition with subtraction assignment with the given tensors";
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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }

      // Columns-wise addition with subtraction assignment with evaluated tensors
      {
         test_  = "Columns-wise addition with subtraction assignment with evaluated tensors";
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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Columns-wise addition with Schur product assignment
      //=====================================================================================

      // Columns-wise addition with Schur product assignment with the given tensors
      {
         test_  = "Columns-wise addition with Schur product assignment with the given tensors";
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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }

      // Columns-wise addition with Schur product assignment with evaluated tensors
      {
         test_  = "Columns-wise addition with Schur product assignment with evaluated tensors";
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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the columns-wise dense tensor/dense tensor addition.
//
// \return void
//
// This function is called in case the columns-wise tensor/tensor addition operation is not
// available for the given tensor types \a MT1 and \a MT2.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testColumnsOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the band-wise dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the band-wise tensor addition with plain assignment, addition assignment,
// subtraction assignment, and multiplication assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::testBandOperation()
{
#if BLAZETEST_MATHTEST_TEST_BAND_OPERATION
   if( BLAZETEST_MATHTEST_TEST_BAND_OPERATION > 1 )
   {
      if( lhs_.rows() == 0UL || lhs_.columns() == 0UL )
         return;


      const ptrdiff_t ibegin( 1UL - lhs_.rows() );
      const ptrdiff_t iend  ( lhs_.columns() );


      //=====================================================================================
      // Band-wise addition
      //=====================================================================================

      // Band-wise addition with the given tensors
      {
         test_  = "Band-wise addition with the given tensors";
         error_ = "Failed addition operation";

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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }

      // Band-wise addition with evaluated tensors
      {
         test_  = "Band-wise addition with evaluated tensors";
         error_ = "Failed addition operation";

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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Band-wise addition with addition assignment
      //=====================================================================================

      // Band-wise addition with addition assignment with the given tensors
      {
         test_  = "Band-wise addition with addition assignment with the given tensors";
         error_ = "Failed addition assignment operation";

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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }

      // Band-wise addition with addition assignment with evaluated tensors
      {
         test_  = "Band-wise addition with addition assignment with evaluated tensors";
         error_ = "Failed addition assignment operation";

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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Band-wise addition with subtraction assignment
      //=====================================================================================

      // Band-wise addition with subtraction assignment with the given tensors
      {
         test_  = "Band-wise addition with subtraction assignment with the given tensors";
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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }

      // Band-wise addition with subtraction assignment with evaluated tensors
      {
         test_  = "Band-wise addition with subtraction assignment with evaluated tensors";
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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }


      //=====================================================================================
      // Band-wise addition with multiplication assignment
      //=====================================================================================

      // Band-wise addition with multiplication assignment with the given tensors
      {
         test_  = "Band-wise addition with multiplication assignment with the given tensors";
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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }

      // Band-wise addition with multiplication assignment with evaluated tensors
      {
         test_  = "Band-wise addition with multiplication assignment with evaluated tensors";
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
            convertException<MT1,MT2>( ex );
         }

         checkResults<MT1,MT2>();

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
            convertException<MT1,OMT2>( ex );
         }

         checkResults<MT1,OMT2>();

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
            convertException<OMT1,MT2>( ex );
         }

         checkResults<OMT1,MT2>();

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
            convertException<OMT1,OMT2>( ex );
         }

         checkResults<OMT1,OMT2>();
      }
   }
#endif
}
//*************************************************************************************************
#endif


//*************************************************************************************************
/*!\brief Testing the customized dense tensor/dense tensor addition.
//
// \param op The custom operation to be tested.
// \param name The human-readable name of the operation.
// \return void
// \exception std::runtime_error Addition error detected.
//
// This function tests the tensor addition with plain assignment, addition assignment, and
// subtraction assignment in combination with a custom operation. In case any error resulting
// from the addition or the subsequent assignment is detected, a \a std::runtime_error exception
// is thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
template< typename OP >   // Type of the custom operation
void OperationTest<MT1,MT2>::testCustomOperation( OP op, const std::string& name )
{
   //=====================================================================================
   // Customized addition
   //=====================================================================================

   // Customized addition with the given tensors
   {
      test_  = "Customized addition with the given tensors (" + name + ")";
      error_ = "Failed addition operation";

      try {
         initResults();
         dres_   = op( lhs_ + rhs_ );
//          odres_  = op( lhs_ + rhs_ );
//          sres_   = op( lhs_ + rhs_ );
//          osres_  = op( lhs_ + rhs_ );
         refres_ = op( reflhs_ + refrhs_ );
      }
      catch( std::exception& ex ) {
         convertException<MT1,MT2>( ex );
      }

      checkResults<MT1,MT2>();

//       try {
//          initResults();
//          dres_   = op( lhs_ + orhs_ );
//          odres_  = op( lhs_ + orhs_ );
//          sres_   = op( lhs_ + orhs_ );
//          osres_  = op( lhs_ + orhs_ );
//          refres_ = op( reflhs_ + refrhs_ );
//       }
//       catch( std::exception& ex ) {
//          convertException<MT1,OMT2>( ex );
//       }
//
//       checkResults<MT1,OMT2>();
//
//       try {
//          initResults();
//          dres_   = op( olhs_ + rhs_ );
//          odres_  = op( olhs_ + rhs_ );
//          sres_   = op( olhs_ + rhs_ );
//          osres_  = op( olhs_ + rhs_ );
//          refres_ = op( reflhs_ + refrhs_ );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,MT2>( ex );
//       }
//
//       checkResults<OMT1,MT2>();
//
//       try {
//          initResults();
//          dres_   = op( olhs_ + orhs_ );
//          odres_  = op( olhs_ + orhs_ );
//          sres_   = op( olhs_ + orhs_ );
//          osres_  = op( olhs_ + orhs_ );
//          refres_ = op( reflhs_ + refrhs_ );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,OMT2>( ex );
//       }
//
//       checkResults<OMT1,OMT2>();
   }

   // Customized addition with evaluated tensors
   {
      test_  = "Customized addition with evaluated tensors (" + name + ")";
      error_ = "Failed addition operation";

      try {
         initResults();
         dres_   = op( eval( lhs_ ) + eval( rhs_ ) );
//          odres_  = op( eval( lhs_ ) + eval( rhs_ ) );
//          sres_   = op( eval( lhs_ ) + eval( rhs_ ) );
//          osres_  = op( eval( lhs_ ) + eval( rhs_ ) );
         refres_ = op( eval( reflhs_ ) + eval( refrhs_ ) );
      }
      catch( std::exception& ex ) {
         convertException<MT1,MT2>( ex );
      }

      checkResults<MT1,MT2>();

//       try {
//          initResults();
//          dres_   = op( eval( lhs_ ) + eval( orhs_ ) );
//          odres_  = op( eval( lhs_ ) + eval( orhs_ ) );
//          sres_   = op( eval( lhs_ ) + eval( orhs_ ) );
//          osres_  = op( eval( lhs_ ) + eval( orhs_ ) );
//          refres_ = op( eval( reflhs_ ) + eval( refrhs_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<MT1,OMT2>( ex );
//       }
//
//       checkResults<MT1,OMT2>();
//
//       try {
//          initResults();
//          dres_   = op( eval( olhs_ ) + eval( rhs_ ) );
//          odres_  = op( eval( olhs_ ) + eval( rhs_ ) );
//          sres_   = op( eval( olhs_ ) + eval( rhs_ ) );
//          osres_  = op( eval( olhs_ ) + eval( rhs_ ) );
//          refres_ = op( eval( reflhs_ ) + eval( refrhs_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,MT2>( ex );
//       }
//
//       checkResults<OMT1,MT2>();
//
//       try {
//          initResults();
//          dres_   = op( eval( olhs_ ) + eval( orhs_ ) );
//          odres_  = op( eval( olhs_ ) + eval( orhs_ ) );
//          sres_   = op( eval( olhs_ ) + eval( orhs_ ) );
//          osres_  = op( eval( olhs_ ) + eval( orhs_ ) );
//          refres_ = op( eval( reflhs_ ) + eval( refrhs_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,OMT2>( ex );
//       }
//
//       checkResults<OMT1,OMT2>();
   }


   //=====================================================================================
   // Customized addition with addition assignment
   //=====================================================================================

   // Customized addition with addition assignment with the given tensors
   {
      test_  = "Customized addition with addition assignment with the given tensors (" + name + ")";
      error_ = "Failed addition assignment operation";

      try {
         initResults();
         dres_   += op( lhs_ + rhs_ );
//          odres_  += op( lhs_ + rhs_ );
//          sres_   += op( lhs_ + rhs_ );
//          osres_  += op( lhs_ + rhs_ );
         refres_ += op( reflhs_ + refrhs_ );
      }
      catch( std::exception& ex ) {
         convertException<MT1,MT2>( ex );
      }

      checkResults<MT1,MT2>();

//       try {
//          initResults();
//          dres_   += op( lhs_ + orhs_ );
//          odres_  += op( lhs_ + orhs_ );
//          sres_   += op( lhs_ + orhs_ );
//          osres_  += op( lhs_ + orhs_ );
//          refres_ += op( reflhs_ + refrhs_ );
//       }
//       catch( std::exception& ex ) {
//          convertException<MT1,OMT2>( ex );
//       }
//
//       checkResults<MT1,OMT2>();
//
//       try {
//          initResults();
//          dres_   += op( olhs_ + rhs_ );
//          odres_  += op( olhs_ + rhs_ );
//          sres_   += op( olhs_ + rhs_ );
//          osres_  += op( olhs_ + rhs_ );
//          refres_ += op( reflhs_ + refrhs_ );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,MT2>( ex );
//       }
//
//       checkResults<OMT1,MT2>();
//
//       try {
//          initResults();
//          dres_   += op( olhs_ + orhs_ );
//          odres_  += op( olhs_ + orhs_ );
//          sres_   += op( olhs_ + orhs_ );
//          osres_  += op( olhs_ + orhs_ );
//          refres_ += op( reflhs_ + refrhs_ );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,OMT2>( ex );
//       }
//
//       checkResults<OMT1,OMT2>();
   }

   // Customized addition with addition assignment with evaluated tensors
   {
      test_  = "Customized addition with addition assignment with evaluated tensors (" + name + ")";
      error_ = "Failed addition assignment operation";

      try {
         initResults();
         dres_   += op( eval( lhs_ ) + eval( rhs_ ) );
//          odres_  += op( eval( lhs_ ) + eval( rhs_ ) );
//          sres_   += op( eval( lhs_ ) + eval( rhs_ ) );
//          osres_  += op( eval( lhs_ ) + eval( rhs_ ) );
         refres_ += op( eval( reflhs_ ) + eval( refrhs_ ) );
      }
      catch( std::exception& ex ) {
         convertException<MT1,MT2>( ex );
      }

      checkResults<MT1,MT2>();

//       try {
//          initResults();
//          dres_   += op( eval( lhs_ ) + eval( orhs_ ) );
//          odres_  += op( eval( lhs_ ) + eval( orhs_ ) );
//          sres_   += op( eval( lhs_ ) + eval( orhs_ ) );
//          osres_  += op( eval( lhs_ ) + eval( orhs_ ) );
//          refres_ += op( eval( reflhs_ ) + eval( refrhs_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<MT1,OMT2>( ex );
//       }
//
//       checkResults<MT1,OMT2>();
//
//       try {
//          initResults();
//          dres_   += op( eval( olhs_ ) + eval( rhs_ ) );
//          odres_  += op( eval( olhs_ ) + eval( rhs_ ) );
//          sres_   += op( eval( olhs_ ) + eval( rhs_ ) );
//          osres_  += op( eval( olhs_ ) + eval( rhs_ ) );
//          refres_ += op( eval( reflhs_ ) + eval( refrhs_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,MT2>( ex );
//       }
//
//       checkResults<OMT1,MT2>();
//
//       try {
//          initResults();
//          dres_   += op( eval( olhs_ ) + eval( orhs_ ) );
//          odres_  += op( eval( olhs_ ) + eval( orhs_ ) );
//          sres_   += op( eval( olhs_ ) + eval( orhs_ ) );
//          osres_  += op( eval( olhs_ ) + eval( orhs_ ) );
//          refres_ += op( eval( reflhs_ ) + eval( refrhs_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,OMT2>( ex );
//       }
//
//       checkResults<OMT1,OMT2>();
   }


   //=====================================================================================
   // Customized addition with subtraction assignment
   //=====================================================================================

   // Customized addition with subtraction assignment with the given tensors
   {
      test_  = "Customized addition with subtraction assignment with the given tensors (" + name + ")";
      error_ = "Failed subtraction assignment operation";

      try {
         initResults();
         dres_   -= op( lhs_ + rhs_ );
//          odres_  -= op( lhs_ + rhs_ );
//          sres_   -= op( lhs_ + rhs_ );
//          osres_  -= op( lhs_ + rhs_ );
         refres_ -= op( reflhs_ + refrhs_ );
      }
      catch( std::exception& ex ) {
         convertException<MT1,MT2>( ex );
      }

      checkResults<MT1,MT2>();

//       try {
//          initResults();
//          dres_   -= op( lhs_ + orhs_ );
//          odres_  -= op( lhs_ + orhs_ );
//          sres_   -= op( lhs_ + orhs_ );
//          osres_  -= op( lhs_ + orhs_ );
//          refres_ -= op( reflhs_ + refrhs_ );
//       }
//       catch( std::exception& ex ) {
//          convertException<MT1,OMT2>( ex );
//       }
//
//       checkResults<MT1,OMT2>();
//
//       try {
//          initResults();
//          dres_   -= op( olhs_ + rhs_ );
//          odres_  -= op( olhs_ + rhs_ );
//          sres_   -= op( olhs_ + rhs_ );
//          osres_  -= op( olhs_ + rhs_ );
//          refres_ -= op( reflhs_ + refrhs_ );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,MT2>( ex );
//       }
//
//       checkResults<OMT1,MT2>();
//
//       try {
//          initResults();
//          dres_   -= op( olhs_ + orhs_ );
//          odres_  -= op( olhs_ + orhs_ );
//          sres_   -= op( olhs_ + orhs_ );
//          osres_  -= op( olhs_ + orhs_ );
//          refres_ -= op( reflhs_ + refrhs_ );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,OMT2>( ex );
//       }
//
//       checkResults<OMT1,OMT2>();
   }

   // Customized addition with subtraction assignment with evaluated tensors
   {
      test_  = "Customized addition with subtraction assignment with evaluated tensors (" + name + ")";
      error_ = "Failed subtraction assignment operation";

      try {
         initResults();
         dres_   -= op( eval( lhs_ ) + eval( rhs_ ) );
//          odres_  -= op( eval( lhs_ ) + eval( rhs_ ) );
//          sres_   -= op( eval( lhs_ ) + eval( rhs_ ) );
//          osres_  -= op( eval( lhs_ ) + eval( rhs_ ) );
         refres_ -= op( eval( reflhs_ ) + eval( refrhs_ ) );
      }
      catch( std::exception& ex ) {
         convertException<MT1,MT2>( ex );
      }

      checkResults<MT1,MT2>();

//       try {
//          initResults();
//          dres_   -= op( eval( lhs_ ) + eval( orhs_ ) );
//          odres_  -= op( eval( lhs_ ) + eval( orhs_ ) );
//          sres_   -= op( eval( lhs_ ) + eval( orhs_ ) );
//          osres_  -= op( eval( lhs_ ) + eval( orhs_ ) );
//          refres_ -= op( eval( reflhs_ ) + eval( refrhs_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<MT1,OMT2>( ex );
//       }
//
//       checkResults<MT1,OMT2>();
//
//       try {
//          initResults();
//          dres_   -= op( eval( olhs_ ) + eval( rhs_ ) );
//          odres_  -= op( eval( olhs_ ) + eval( rhs_ ) );
//          sres_   -= op( eval( olhs_ ) + eval( rhs_ ) );
//          osres_  -= op( eval( olhs_ ) + eval( rhs_ ) );
//          refres_ -= op( eval( reflhs_ ) + eval( refrhs_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,MT2>( ex );
//       }
//
//       checkResults<OMT1,MT2>();
//
//       try {
//          initResults();
//          dres_   -= op( eval( olhs_ ) + eval( orhs_ ) );
//          odres_  -= op( eval( olhs_ ) + eval( orhs_ ) );
//          sres_   -= op( eval( olhs_ ) + eval( orhs_ ) );
//          osres_  -= op( eval( olhs_ ) + eval( orhs_ ) );
//          refres_ -= op( eval( reflhs_ ) + eval( refrhs_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,OMT2>( ex );
//       }
//
//       checkResults<OMT1,OMT2>();
   }


   //=====================================================================================
   // Customized addition with Schur product assignment
   //=====================================================================================

   // Customized addition with Schur product assignment with the given tensors
   {
      test_  = "Customized addition with Schur product assignment with the given tensors (" + name + ")";
      error_ = "Failed Schur product assignment operation";

      try {
         initResults();
         dres_   %= op( lhs_ + rhs_ );
//          odres_  %= op( lhs_ + rhs_ );
//          sres_   %= op( lhs_ + rhs_ );
//          osres_  %= op( lhs_ + rhs_ );
         refres_ %= op( reflhs_ + refrhs_ );
      }
      catch( std::exception& ex ) {
         convertException<MT1,MT2>( ex );
      }

      checkResults<MT1,MT2>();

//       try {
//          initResults();
//          dres_   %= op( lhs_ + orhs_ );
//          odres_  %= op( lhs_ + orhs_ );
//          sres_   %= op( lhs_ + orhs_ );
//          osres_  %= op( lhs_ + orhs_ );
//          refres_ %= op( reflhs_ + refrhs_ );
//       }
//       catch( std::exception& ex ) {
//          convertException<MT1,OMT2>( ex );
//       }
//
//       checkResults<MT1,OMT2>();
//
//       try {
//          initResults();
//          dres_   %= op( olhs_ + rhs_ );
//          odres_  %= op( olhs_ + rhs_ );
//          sres_   %= op( olhs_ + rhs_ );
//          osres_  %= op( olhs_ + rhs_ );
//          refres_ %= op( reflhs_ + refrhs_ );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,MT2>( ex );
//       }
//
//       checkResults<OMT1,MT2>();
//
//       try {
//          initResults();
//          dres_   %= op( olhs_ + orhs_ );
//          odres_  %= op( olhs_ + orhs_ );
//          sres_   %= op( olhs_ + orhs_ );
//          osres_  %= op( olhs_ + orhs_ );
//          refres_ %= op( reflhs_ + refrhs_ );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,OMT2>( ex );
//       }
//
//       checkResults<OMT1,OMT2>();
   }

   // Customized addition with Schur product assignment with evaluated tensors
   {
      test_  = "Customized addition with Schur product assignment with evaluated tensors (" + name + ")";
      error_ = "Failed Schur product assignment operation";

      try {
         initResults();
         dres_   %= op( eval( lhs_ ) + eval( rhs_ ) );
//          odres_  %= op( eval( lhs_ ) + eval( rhs_ ) );
//          sres_   %= op( eval( lhs_ ) + eval( rhs_ ) );
//          osres_  %= op( eval( lhs_ ) + eval( rhs_ ) );
         refres_ %= op( eval( reflhs_ ) + eval( refrhs_ ) );
      }
      catch( std::exception& ex ) {
         convertException<MT1,MT2>( ex );
      }

      checkResults<MT1,MT2>();

//       try {
//          initResults();
//          dres_   %= op( eval( lhs_ ) + eval( orhs_ ) );
//          odres_  %= op( eval( lhs_ ) + eval( orhs_ ) );
//          sres_   %= op( eval( lhs_ ) + eval( orhs_ ) );
//          osres_  %= op( eval( lhs_ ) + eval( orhs_ ) );
//          refres_ %= op( eval( reflhs_ ) + eval( refrhs_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<MT1,OMT2>( ex );
//       }
//
//       checkResults<MT1,OMT2>();
//
//       try {
//          initResults();
//          dres_   %= op( eval( olhs_ ) + eval( rhs_ ) );
//          odres_  %= op( eval( olhs_ ) + eval( rhs_ ) );
//          sres_   %= op( eval( olhs_ ) + eval( rhs_ ) );
//          osres_  %= op( eval( olhs_ ) + eval( rhs_ ) );
//          refres_ %= op( eval( reflhs_ ) + eval( refrhs_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,MT2>( ex );
//       }
//
//       checkResults<OMT1,MT2>();
//
//       try {
//          initResults();
//          dres_   %= op( eval( olhs_ ) + eval( orhs_ ) );
//          odres_  %= op( eval( olhs_ ) + eval( orhs_ ) );
//          sres_   %= op( eval( olhs_ ) + eval( orhs_ ) );
//          osres_  %= op( eval( olhs_ ) + eval( orhs_ ) );
//          refres_ %= op( eval( reflhs_ ) + eval( refrhs_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<OMT1,OMT2>( ex );
//       }
//
//       checkResults<OMT1,OMT2>();
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
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
template< typename LT     // Type of the left-hand side operand
        , typename RT >   // Type of the right-hand side operand
void OperationTest<MT1,MT2>::checkResults()
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
          << "   Left-hand side " << ( IsRowMajorTensor<LT>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
          << "     " << typeid( LT ).name() << "\n"
          << "   Right-hand side " << ( IsRowMajorTensor<RT>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
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
//           << "   Left-hand side " << ( IsRowMajorTensor<LT>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//           << "     " << typeid( LT ).name() << "\n"
//           << "   Right-hand side " << ( IsRowMajorTensor<RT>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
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
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
template< typename LT     // Type of the left-hand side operand
        , typename RT >   // Type of the right-hand side operand
void OperationTest<MT1,MT2>::checkTransposeResults()
{
//    template <typename MT>
//    using IsRowMajorTensor = blaze::IsTensor<MT>;

   if( !isEqual( tdres_, refres_ ) /*|| !isEqual( todres_, refres_ )*/ ) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect dense result detected\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Left-hand side " << " dense tensor type:\n"
          << "     " << typeid( LT ).name() << "\n"
          << "   Right-hand side " << " dense tensor type:\n"
          << "     " << typeid( RT ).name() << "\n"
          << "   Transpose result:\n" << tdres_ << "\n"
//           << "   Transpose result with opposite storage order:\n" << todres_ << "\n"
          << "   Expected result:\n" << refres_ << "\n";
      throw std::runtime_error( oss.str() );
   }

//    if( !isEqual( tsres_, refres_ ) || !isEqual( tosres_, refres_ ) ) {
//       std::ostringstream oss;
//       oss.precision( 20 );
//       oss << " Test : " << test_ << "\n"
//           << " Error: Incorrect sparse result detected\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Left-hand side " << ( IsRowMajorTensor<LT>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//           << "     " << typeid( LT ).name() << "\n"
//           << "   Right-hand side " << ( IsRowMajorTensor<RT>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
//           << "     " << typeid( RT ).name() << "\n"
//           << "   Transpose result:\n" << tsres_ << "\n"
//           << "   Transpose result with opposite storage order:\n" << tosres_ << "\n"
//           << "   Expected result:\n" << refres_ << "\n";
//       throw std::runtime_error( oss.str() );
//    }
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
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::initResults()
{
   const blaze::UnderlyingBuiltin_t<DRE> min( randmin );
   const blaze::UnderlyingBuiltin_t<DRE> max( randmax );

   resize( dres_, pages( lhs_ ), rows( lhs_ ), columns( lhs_ ) );
   randomize( dres_, min, max );

//    odres_  = dres_;
//    sres_   = dres_;
//    osres_  = dres_;
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
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void OperationTest<MT1,MT2>::initTransposeResults()
{
   const blaze::UnderlyingBuiltin_t<TDRE> min( randmin );
   const blaze::UnderlyingBuiltin_t<TDRE> max( randmax );

   resize( tdres_, pages( lhs_ ), columns( lhs_ ), rows( lhs_ ) );
   randomize( tdres_, min, max );

//    todres_ = tdres_;
//    tsres_  = tdres_;
//    tosres_ = tdres_;
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
// This function converts the given exception to a \a std::runtime_error exception. Additionally,
// the function extends the given exception message by all available information for the failed
// test. The two template arguments \a LT and \a RT indicate the types of the left-hand side and
// right-hand side operands used for the computations.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
template< typename LT     // Type of the left-hand side operand
        , typename RT >   // Type of the right-hand side operand
void OperationTest<MT1,MT2>::convertException( const std::exception& ex )
{
//    template <typename MT>
//    using IsRowMajorTensor = blaze::IsTensor<MT>;

   std::ostringstream oss;
   oss << " Test : " << test_ << "\n"
       << " Error: " << error_ << "\n"
       << " Details:\n"
       << "   Random seed = " << blaze::getSeed() << "\n"
       << "   Left-hand side " << ( IsRowMajorTensor<LT>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
       << "     " << typeid( LT ).name() << "\n"
       << "   Right-hand side " << ( IsRowMajorTensor<LT>::value ? ( "row-major" ) : ( "column-major" ) ) << " dense tensor type:\n"
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
/*!\brief Testing the tensor addition between two specific tensor types.
//
// \param creator1 The creator for the left-hand side tensor.
// \param creator2 The creator for the right-hand side tensor.
// \return void
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
void runTest( const Creator<MT1>& creator1, const Creator<MT2>& creator2 )
{
#if BLAZETEST_MATHTEST_TEST_ADDITION
   if( BLAZETEST_MATHTEST_TEST_ADDITION > 1 )
   {
      for( size_t rep=0UL; rep<BLAZETEST_REPETITIONS; ++rep ) {
         OperationTest<MT1,MT2>( creator1, creator2 );
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
/*!\brief Macro for the definition of a dense tensor/dense tensor addition test case.
*/
#define DEFINE_DTENSDTENSADD_OPERATION_TEST( MT1, MT2 ) \
   extern template class blazetest::mathtest::dtensdtensadd::OperationTest<MT1,MT2>
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of a dense tensor/dense tensor addition test case.
*/
#define RUN_DTENSDTENSADD_OPERATION_TEST( C1, C2 ) \
   blazetest::mathtest::dtensdtensadd::runTest( C1, C2 )
/*! \endcond */
//*************************************************************************************************

} // namespace dtensdtensadd

} // namespace mathtest

} // namespace blazetest

#endif
