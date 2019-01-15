//=================================================================================================
/*!
//  \file blazetest/mathtest/dmatexpand/OperationTest.h
//  \brief Header file for the dense Matrix expansion operation test
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

#ifndef _BLAZETEST_MATHTEST_DMATEXPAND_OPERATIONTEST_H_
#define _BLAZETEST_MATHTEST_DMATEXPAND_OPERATIONTEST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/Views.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/constraints/StorageOrder.h>
#include <blaze/math/shims/Equal.h>
#include <blaze/math/traits/ExpandTrait.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsUniform.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/math/typetraits/UnderlyingNumeric.h>
#include <blaze/util/FalseType.h>
#include <blaze/util/Random.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/util/mpl/Not.h>
#include <blaze/util/typetraits/Decay.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/StorageOrder.h>
#include <blaze_tensor/math/Views.h>

#include <blazetest/config/TensorMathTest.h>
#include <blazetest/mathtest/Creator.h>
#include <blazetest/mathtest/IsEqual.h>
#include <blazetest/mathtest/RandomMaximum.h>
#include <blazetest/mathtest/RandomMinimum.h>
#include <blazetest/system/MathTest.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace blazetest {

namespace mathtest {

namespace dmatexpand {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class template for the dense Matrix expansion operation test.
//
// This class template represents one particular test of an expansion operation on a Matrix of
// a particular type. The template argument \a MT represents the type of the Matrix operand.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
class OperationTest
{
 private:
   //**Type definitions****************************************************************************
   using ET = blaze::ElementType_t<MT>;  //!< Element type.

   using TMT = blaze::TransposeType_t<MT>;  //!< Transpose Matrix type

   using DRE   = blaze::ExpandTrait_t<MT,E>;    //!< Dense result type.
   using DET   = blaze::ElementType_t<DRE>;     //!< Element type of the dense result
//    using ODRE  = blaze::OppositeType_t<DRE>;   //!< Dense result type with opposite storage order
//    using TDRE  = blaze::TransposeType_t<DRE>;  //!< Transpose dense result type with opposite storage order
//    using TODRE = blaze::OppositeType_t<ODRE>;  //!< Transpose dense result type with opposite storage order

//    using SRE   = blaze::CompressedTensor<DET,true>;  //!< Sparse result type
//    using SET   = blaze::ElementType_t<SRE>;          //!< Element type of the sparse result
//    using OSRE  = blaze::OppositeType_t<SRE>;         //!< Sparse result type with opposite storage order
//    using TSRE  = blaze::TransposeType_t<SRE>;        //!< Transpose sparse result type
//    using TOSRE = blaze::TransposeType_t<OSRE>;       //!< Transpose sparse result type with opposite storage order

   using RT  = blaze::DynamicMatrix<ET,false>;     //!< Reference type.
   using RRE = blaze::ExpandTrait_t<RT,E>;         //!< Reference result type.

   using TRT  = blaze::TransposeType_t<RT>;        //!< Transpose reference type.
//    using TRRE = blaze::ExpandTrait_t<TRT,E>;       //!< Transpose reference result type.

   //! Type of the Matrix expand expression (runtime argument)
   using MatExpandExprType1 = blaze::Decay_t< decltype( blaze::expand( std::declval<MT>(), E ) ) >;

   //! Type of the Matrix expand expression (compile time argument)
   using MatExpandExprType2 = blaze::Decay_t< decltype( blaze::expand<E>( std::declval<MT>() ) ) >;

   //! Type of the transpose Matrix expand expression (runtime argument)
   using TMatExpandExprType1 = blaze::Decay_t< decltype( blaze::expand( std::declval<TMT>(), E ) ) >;

   //! Type of the transpose Matrix expand expression (compile time argument)
   using TMatExpandExprType2 = blaze::Decay_t< decltype( blaze::expand<E>( std::declval<TMT>() ) ) >;
   //**********************************************************************************************

 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit OperationTest( const Creator<MT>& creator );
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
                          void testSubtensorOperation( blaze::TrueType  );
                          void testSubtensorOperation( blaze::FalseType );
                          void testRowSliceOperation      ( blaze::TrueType  );
                          void testRowSliceOperation      ( blaze::FalseType );
                          void testRowSlicesOperation     ( blaze::TrueType  );
                          void testRowSlicesOperation     ( blaze::FalseType );
                          void testColumnSliceOperation   ( blaze::TrueType  );
                          void testColumnSliceOperation   ( blaze::FalseType );
                          void testColumnSlicesOperation  ( blaze::TrueType  );
                          void testColumnSlicesOperation  ( blaze::FalseType );
                          void testPageSliceOperation     ( blaze::TrueType  );
                          void testPageSliceOperation     ( blaze::FalseType );
                          void testPageSlicesOperation    ( blaze::TrueType  );
                          void testPageSlicesOperation    ( blaze::FalseType );
                          void testBandOperation     ( blaze::TrueType  );
                          void testBandOperation     ( blaze::FalseType );

   template< typename OP > void testCustomOperation( OP op, const std::string& name );
   //@}
   //**********************************************************************************************

   //**Error detection functions*******************************************************************
   /*!\name Error detection functions */
   //@{
   template< typename T > void checkResults();
   template< typename T > void checkTransposeResults();
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   void initResults();
   void initTransposeResults();
   template< typename T > void convertException( const std::exception& ex );
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   MT    mat_;      //!< The dense Matrix operand.
   DRE   dres_;     //!< The dense result tensor.
//    SRE   sres_;     //!< The sparse result tensor.
//    ODRE  odres_;    //!< The dense result tensor with opposite storage order.
//    OSRE  osres_;    //!< The sparse result tensor with opposite storage order.
   RT    refmat_;   //!< The reference Matrix.
   RRE   refres_;   //!< The reference result.
   TMT   tmat_;     //!< The transpose dense Matrix operand.
//    TDRE  tdres_;    //!< The transpose dense result tensor.
//    TSRE  tsres_;    //!< The transpose sparse result tensor.
//    TODRE todres_;   //!< The transpose dense result tensor with opposite storage order.
//    TOSRE tosres_;   //!< The transpose sparse result tensor with opposite storage order.
   TRT   trefmat_;  //!< The transpose reference Matrix.
//    TRRE  trefres_;  //!< The transpose reference result.

   std::string test_;   //!< Label of the currently performed test.
   std::string error_;  //!< Description of the current error type.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( MT    );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( TMT   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( DRE   );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( ODRE  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TDRE  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TODRE );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( SRE   );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( OSRE  );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( TSRE  );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( TOSRE );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( RT    );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE( TRT   );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( RRE   );
//    BLAZE_CONSTRAINT_MUST_BE_SPARSE_TENSOR_TYPE( TRRE  );

   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE     ( MT    );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE  ( TMT   );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( DRE   );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( ODRE  );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( TDRE  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( TODRE );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( SRE   );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( OSRE  );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( TSRE  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( TOSRE );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MATRIX_TYPE      ( RT    );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MATRIX_TYPE         ( TRT   );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_TENSOR_TYPE( RRE   );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE   ( TRRE  );

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( DRE   );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ODRE  );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( TDRE  );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( TODRE );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SRE   );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( OSRE  );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( TSRE  );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( TOSRE );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RRE   );
//    BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( TRRE  );

   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET , blaze::ElementType_t<TMT>   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<DRE>   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<ODRE>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<TDRE>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<TODRE> );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<SRE>   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<SRE>   );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<OSRE>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<TSRE>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<TOSRE> );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<DRE>   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT , blaze::TransposeType_t<TMT> );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( RT , blaze::TransposeType_t<TRT> );

   BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_SAME_STORAGE_ORDER     ( MatExpandExprType1, blaze::ResultType_t<MatExpandExprType1>    );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( MatExpandExprType1, blaze::TransposeType_t<MatExpandExprType1> );
   BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_SAME_STORAGE_ORDER     ( MatExpandExprType2, blaze::ResultType_t<MatExpandExprType2>    );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( MatExpandExprType2, blaze::TransposeType_t<MatExpandExprType2> );

   BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_SAME_STORAGE_ORDER     ( TMatExpandExprType1, blaze::ResultType_t<TMatExpandExprType1>    );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TMatExpandExprType1, blaze::TransposeType_t<TMatExpandExprType1> );
   BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_SAME_STORAGE_ORDER     ( TMatExpandExprType2, blaze::ResultType_t<TMatExpandExprType2>    );
//    BLAZE_CONSTRAINT_TENSORS_MUST_HAVE_DIFFERENT_STORAGE_ORDER( TMatExpandExprType2, blaze::TransposeType_t<TMatExpandExprType2> );
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
/*!\brief Constructor for the dense Matrix expansion operation test.
//
// \param creator The creator for dense Matrix operand.
// \exception std::runtime_error Operation error detected.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
OperationTest<MT,E>::OperationTest( const Creator<MT>& creator )
   : mat_( creator() )     // The dense Matrix operand
   , dres_()               // The dense result tensor
//    , sres_()               // The sparse result tensor
//    , odres_()              // The dense result tensor with opposite storage order
//    , osres_()              // The sparse result tensor with opposite storage order
   , refmat_( mat_ )       // The reference Matrix
   , refres_()             // The reference result
   , tmat_( trans(mat_) )  // The transpose dense Matrix operand
//    , tdres_()              // The transpose dense result tensor
//    , tsres_()              // The transpose sparse result tensor
//    , todres_()             // The transpose dense result tensor with opposite storage order
//    , tosres_()             // The transpose dense result tensor with opposite storage order
   , trefmat_( tmat_ )     // The transpose reference Matrix
//    , trefres_()            // The transpose reference result
   , test_()               // Label of the currently performed test
   , error_()              // Description of the current error type
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
   testSubtensorOperation( Not< IsUniform<DRE> >() );
   testRowSliceOperation( Not< IsUniform<DRE> >() );
   testRowSlicesOperation( Not< IsUniform<DRE> >() );
   testColumnSliceOperation( Not< IsUniform<DRE> >() );
   testColumnSlicesOperation( Not< IsUniform<DRE> >() );
   testPageSliceOperation( Not< IsUniform<DRE> >() );
   testPageSlicesOperation( Not< IsUniform<DRE> >() );
   testBandOperation( Not< IsUniform<DRE> >() );
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Tests on the initial status of the Matrix.
//
// \return void
// \exception std::runtime_error Initialization error detected.
//
// This function runs tests on the initial status of the Matrix. In case any initialization
// error is detected, a \a std::runtime_error exception is thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testInitialStatus()
{
   //=====================================================================================
   // Performing initial tests with the given Matrix
   //=====================================================================================

   // Checking the number of rows of the dense operand
   if( mat_.rows() != refmat_.rows() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of dense Matrix operand\n"
          << " Error: Invalid Matrix size\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Dense Matrix type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Detected number of rows = " << mat_.rows() << "\n"
          << "   Expected number of rows = " << refmat_.rows() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the number of columns of the dense operand
   if( mat_.columns() != refmat_.columns() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of row-major dense operand\n"
          << " Error: Invalid number of columns\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Detected number of columns = " << mat_.columns() << "\n"
          << "   Expected number of columns = " << refmat_.columns() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the initialization of the dense operand
   if( !isEqual( mat_, refmat_ ) ) {
      std::ostringstream oss;
      oss << " Test: Initial test of initialization of row-major dense operand\n"
          << " Error: Invalid matrix initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Current initialization:\n" << mat_ << "\n"
          << "   Expected initialization:\n" << refmat_ << "\n";
      throw std::runtime_error( oss.str() );
   }


   //=====================================================================================
   // Performing initial tests with the column-major types
   //=====================================================================================

   // Checking the number of rows of the dense operand
   if( tmat_.rows() != trefmat_.rows() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of column-major dense operand\n"
          << " Error: Invalid number of rows\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Detected number of rows = " << tmat_.rows() << "\n"
          << "   Expected number of rows = " << refmat_.rows() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the number of columns of the dense operand
   if( tmat_.columns() != trefmat_.columns() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of column-major dense operand\n"
          << " Error: Invalid number of columns\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Detected number of columns = " << tmat_.columns() << "\n"
          << "   Expected number of columns = " << refmat_.columns() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the initialization of the dense operand
   if( !isEqual( tmat_, trefmat_ ) ) {
      std::ostringstream oss;
      oss << " Test: Initial test of initialization of column-major dense operand\n"
          << " Error: Invalid matrix initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Current initialization:\n" << tmat_ << "\n"
          << "   Expected initialization:\n" << refmat_ << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the Matrix assignment.
//
// \return void
// \exception std::runtime_error Assignment error detected.
//
// This function tests the Matrix assignment. In case any error is detected, a
// \a std::runtime_error exception is thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testAssignment()
{
   //=====================================================================================
   // Performing an assignment with the row-major types
   //=====================================================================================

   try {
      mat_ = refmat_;
   }
   catch( std::exception& ex ) {
      std::ostringstream oss;
      oss << " Test: Assignment with the row-major types\n"
          << " Error: Failed assignment\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Error message: " << ex.what() << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( !isEqual( mat_, refmat_ ) ) {
      std::ostringstream oss;
      oss << " Test: Checking the assignment result of row-major dense operand\n"
          << " Error: Invalid matrix initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Current initialization:\n" << mat_ << "\n"
          << "   Expected initialization:\n" << refmat_ << "\n";
      throw std::runtime_error( oss.str() );
   }


   //=====================================================================================
   // Performing an assignment with the column-major types
   //=====================================================================================

//    try {
//       tmat_ = refmat_;
//    }
//    catch( std::exception& ex ) {
//       std::ostringstream oss;
//       oss << " Test: Assignment with the column-major types\n"
//           << " Error: Failed assignment\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Column-major dense matrix type:\n"
//           << "     " << typeid( OMT ).name() << "\n"
//           << "   Error message: " << ex.what() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//
//    if( !isEqual( tmat_, refmat_ ) ) {
//       std::ostringstream oss;
//       oss << " Test: Checking the assignment result of column-major dense operand\n"
//           << " Error: Invalid matrix initialization\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Column-major dense matrix type:\n"
//           << "     " << typeid( OMT ).name() << "\n"
//           << "   Current initialization:\n" << tmat_ << "\n"
//           << "   Expected initialization:\n" << refmat_ << "\n";
//       throw std::runtime_error( oss.str() );
//    }
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
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testEvaluation()
{
   using blaze::expand;


   //=====================================================================================
   // Testing the evaluation with a row-major Matrix
   //=====================================================================================

   {
      const auto res   ( evaluate( expand( mat_, E ) ) );
      const auto refres( evaluate( expand( refmat_, E ) ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with the given Matrix (runtime)\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense row-major Matrix type:\n"
             << "     " << typeid( mat_ ).name() << "\n"
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
      const auto res   ( evaluate( expand<E>( mat_ ) ) );
      const auto refres( evaluate( expand<E>( refmat_ ) ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with the given Matrix (compile time)\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense row-major Matrix type:\n"
             << "     " << typeid( mat_ ).name() << "\n"
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
      const auto res   ( evaluate( expand( eval( mat_ ), E ) ) );
      const auto refres( evaluate( expand( eval( refmat_ ), E ) ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with evaluated Matrix (runtime)\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense row-major Matrix type:\n"
             << "     " << typeid( mat_ ).name() << "\n"
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
      const auto res   ( evaluate( expand<E>( eval( mat_ ) ) ) );
      const auto refres( evaluate( expand<E>( eval( refmat_ ) ) ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with evaluated Matrix (compile time)\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense row-major Matrix type:\n"
             << "     " << typeid( mat_ ).name() << "\n"
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
   // Testing the evaluation with a transposed Matrix
   //=====================================================================================

   {
      const auto res   ( evaluate( expand( tmat_, E ) ) );
      const auto refres( evaluate( expand( trefmat_, E ) ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with the given Matrix (runtime)\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense column-major Matrix type:\n"
             << "     " << typeid( mat_ ).name() << "\n"
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
      const auto res   ( evaluate( expand<E>( tmat_ ) ) );
      const auto refres( evaluate( expand<E>( trefmat_ ) ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with the given Matrix (compile time)\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense column-major Matrix type:\n"
             << "     " << typeid( mat_ ).name() << "\n"
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
      const auto res   ( evaluate( expand( eval( tmat_ ), E ) ) );
      const auto refres( evaluate( expand( eval( trefmat_ ), E ) ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with evaluated Matrix (runtime)\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense column-major Matrix type:\n"
             << "     " << typeid( mat_ ).name() << "\n"
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
      const auto res   ( evaluate( expand<E>( eval( tmat_ ) ) ) );
      const auto refres( evaluate( expand<E>( eval( trefmat_ ) ) ) );

      if( !isEqual( res, refres ) ) {
         std::ostringstream oss;
         oss << " Test: Evaluation with evaluated Matrix (compile time)\n"
             << " Error: Failed evaluation\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense column-major Matrix type:\n"
             << "     " << typeid( mat_ ).name() << "\n"
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
// This function tests the element access via the function call operator. In case any error
// is detected, a \a std::runtime_error exception is thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testElementAccess()
{
   using blaze::equal;
   using blaze::expand;


   //=====================================================================================
   // Testing the element access with a row-major Matrix
   //=====================================================================================

   if( mat_.rows() > 0UL && mat_.columns() > 0UL && E > 0UL )
   {
      const size_t o( E-1UL );
      const size_t m( mat_.rows()    - 1UL );
      const size_t n( mat_.columns() - 1UL );

      if( !equal( expand( mat_, E )(o,m,n), expand( refmat_, E )(o,m,n) ) ||
          !equal( expand( mat_, E ).at(o,m,n), expand( refmat_, E ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of expansion expression (runtime)\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense row-major Matrix type:\n"
             << "     " << typeid( MT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( expand<E>( mat_ )(o,m,n), expand<E>( refmat_ )(o,m,n) ) ||
          !equal( expand<E>( mat_ ).at(o,m,n), expand<E>( refmat_ ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of expansion expression (compile time)\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense row-major Matrix type:\n"
             << "     " << typeid( MT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( expand( eval( mat_ ), E )(o,m,n), expand( eval( refmat_ ), E )(o,m,n) ) ||
          !equal( expand( eval( mat_ ), E ).at(o,m,n), expand( eval( refmat_ ), E ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of evaluated expansion expression (runtime)\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense row-major Matrix type:\n"
             << "     " << typeid( MT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( expand<E>( eval( mat_ ) )(o,m,n), expand<E>( eval( refmat_ ) )(o,m,n) ) ||
          !equal( expand<E>( eval( mat_ ) ).at(o,m,n), expand<E>( eval( refmat_ ) ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of evaluated expansion expression (compile time)\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense row-major Matrix type:\n"
             << "     " << typeid( MT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Testing the element access with a column-major Matrix
   //=====================================================================================

   if( tmat_.rows() > 0UL && tmat_.columns() > 0UL && E > 0UL )
   {
      const size_t o( E-1UL );
      const size_t m( tmat_.rows()    - 1UL );
      const size_t n( tmat_.columns() - 1UL );

      if( !equal( expand( tmat_, E )(o,m,n), expand( trefmat_, E )(o,m,n) ) ||
          !equal( expand( tmat_, E ).at(o,m,n), expand( trefmat_, E ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of expansion expression (runtime)\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense column-major Matrix type:\n"
             << "     " << typeid( TMT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( expand<E>( tmat_ )(o,m,n), expand<E>( trefmat_ )(o,m,n) ) ||
          !equal( expand<E>( tmat_ ).at(o,m,n), expand<E>( trefmat_ ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of expansion expression (compile time)\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense column-major Matrix type:\n"
             << "     " << typeid( TMT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( expand( eval( tmat_ ), E )(o,m,n), expand( eval( trefmat_ ), E )(o,m,n) ) ||
          !equal( expand( eval( tmat_ ), E ).at(o,m,n), expand( eval( trefmat_ ), E ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of evaluated expansion expression (runtime)\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense column-major Matrix type:\n"
             << "     " << typeid( TMT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }

      if( !equal( expand<E>( eval( tmat_ ) )(o,m,n), expand<E>( eval( trefmat_ ) )(o,m,n) ) ||
          !equal( expand<E>( eval( tmat_ ) ).at(o,m,n), expand<E>( eval( trefmat_ ) ).at(o,m,n) ) ) {
         std::ostringstream oss;
         oss << " Test : Element access of evaluated expansion expression (compile time)\n"
             << " Error: Unequal resulting elements at element (" << m << "," << n << ") detected\n"
             << " Details:\n"
             << "   Random seed = " << blaze::getSeed() << "\n"
             << "   Dense column-major Matrix type:\n"
             << "     " << typeid( TMT ).name() << "\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the plain dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the plain Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// expansion or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testBasicOperation()
{
#if BLAZETEST_MATHTEST_TEST_BASIC_OPERATION
   if( BLAZETEST_MATHTEST_TEST_BASIC_OPERATION > 1 )
   {
      using blaze::expand;


      //=====================================================================================
      // Expansion operation
      //=====================================================================================

      // Expansion operation with the given Matrix (runtime)
      {
         test_  = "Expansion operation with the given Matrix (runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = expand( mat_, E );
//             odres_  = expand( mat_, E );
//             sres_   = expand( mat_, E );
//             osres_  = expand( mat_, E );
            refres_ = expand( refmat_, E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = expand( tmat_, E );
//             todres_  = expand( tmat_, E );
//             tsres_   = expand( tmat_, E );
//             tosres_  = expand( tmat_, E );
//             trefres_ = expand( trefmat_, E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Expansion operation with the given Matrix (compile time)
      {
         test_  = "Expansion operation with the given Matrix (compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = expand<E>( mat_ );
//             odres_  = expand<E>( mat_ );
//             sres_   = expand<E>( mat_ );
//             osres_  = expand<E>( mat_ );
            refres_ = expand<E>( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = expand<E>( tmat_ );
//             todres_  = expand<E>( tmat_ );
//             tsres_   = expand<E>( tmat_ );
//             tosres_  = expand<E>( tmat_ );
//             trefres_ = expand<E>( trefmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Expansion operation with evaluated Matrix (runtime)
      {
         test_  = "Expansion operation with evaluated Matrix (runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = expand( eval( mat_ ), E );
//             odres_  = expand( eval( mat_ ), E );
//             sres_   = expand( eval( mat_ ), E );
//             osres_  = expand( eval( mat_ ), E );
            refres_ = expand( eval( refmat_ ), E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = expand( eval( tmat_ ), E );
//             todres_  = expand( eval( tmat_ ), E );
//             tsres_   = expand( eval( tmat_ ), E );
//             tosres_  = expand( eval( tmat_ ), E );
//             trefres_ = expand( eval( trefmat_ ), E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Expansion operation with evaluated Matrix (compile time)
      {
         test_  = "Expansion operation with evaluated Matrix (compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = expand<E>( eval( mat_ ) );
//             odres_  = expand<E>( eval( mat_ ) );
//             sres_   = expand<E>( eval( mat_ ) );
//             osres_  = expand<E>( eval( mat_ ) );
            refres_ = expand<E>( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = expand<E>( eval( tmat_ ) );
//             todres_  = expand<E>( eval( tmat_ ) );
//             tsres_   = expand<E>( eval( tmat_ ) );
//             tosres_  = expand<E>( eval( tmat_ ) );
//             trefres_ = expand<E>( eval( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Expansion with addition assignment
      //=====================================================================================

      // Expansion with addition assignment with the given Matrix (runtime)
      {
         test_  = "Expansion with addition assignment with the given Matrix (runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += expand( mat_, E );
//             odres_  += expand( mat_, E );
//             sres_   += expand( mat_, E );
//             osres_  += expand( mat_, E );
            refres_ += expand( refmat_, E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += expand( tmat_, E );
//             todres_  += expand( tmat_, E );
//             tsres_   += expand( tmat_, E );
//             tosres_  += expand( tmat_, E );
//             trefres_ += expand( trefmat_, E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Expansion with addition assignment with the given Matrix (compile time)
      {
         test_  = "Expansion with addition assignment with the given Matrix (compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += expand<E>( mat_ );
//             odres_  += expand<E>( mat_ );
//             sres_   += expand<E>( mat_ );
//             osres_  += expand<E>( mat_ );
            refres_ += expand<E>( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += expand<E>( tmat_ );
//             todres_  += expand<E>( tmat_ );
//             tsres_   += expand<E>( tmat_ );
//             tosres_  += expand<E>( tmat_ );
//             trefres_ += expand<E>( trefmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Expansion with addition assignment with evaluated Matrix (runtime)
      {
         test_  = "Expansion with addition assignment with evaluated Matrix (runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += expand( eval( mat_ ), E );
//             odres_  += expand( eval( mat_ ), E );
//             sres_   += expand( eval( mat_ ), E );
//             osres_  += expand( eval( mat_ ), E );
            refres_ += expand( eval( refmat_ ), E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += expand( eval( tmat_ ), E );
//             todres_  += expand( eval( tmat_ ), E );
//             tsres_   += expand( eval( tmat_ ), E );
//             tosres_  += expand( eval( tmat_ ), E );
//             trefres_ += expand( eval( trefmat_ ), E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Expansion with addition assignment with evaluated Matrix (compile time)
      {
         test_  = "Expansion with addition assignment with evaluated Matrix (compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += expand<E>( eval( mat_ ) );
//             odres_  += expand<E>( eval( mat_ ) );
//             sres_   += expand<E>( eval( mat_ ) );
//             osres_  += expand<E>( eval( mat_ ) );
            refres_ += expand<E>( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += expand<E>( eval( tmat_ ) );
//             todres_  += expand<E>( eval( tmat_ ) );
//             tsres_   += expand<E>( eval( tmat_ ) );
//             tosres_  += expand<E>( eval( tmat_ ) );
//             trefres_ += expand<E>( eval( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Expansion with subtraction assignment
      //=====================================================================================

      // Expansion with subtraction assignment with the given Matrix (runtime)
      {
         test_  = "Expansion with subtraction assignment with the given Matrix (runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= expand( mat_, E );
//             odres_  -= expand( mat_, E );
//             sres_   -= expand( mat_, E );
//             osres_  -= expand( mat_, E );
            refres_ -= expand( refmat_, E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= expand( tmat_, E );
//             todres_  -= expand( tmat_, E );
//             tsres_   -= expand( tmat_, E );
//             tosres_  -= expand( tmat_, E );
//             trefres_ -= expand( trefmat_, E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Expansion with subtraction assignment with the given Matrix (compile time)
      {
         test_  = "Expansion with subtraction assignment with the given Matrix (compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= expand<E>( mat_ );
//             odres_  -= expand<E>( mat_ );
//             sres_   -= expand<E>( mat_ );
//             osres_  -= expand<E>( mat_ );
            refres_ -= expand<E>( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= expand<E>( tmat_ );
//             todres_  -= expand<E>( tmat_ );
//             tsres_   -= expand<E>( tmat_ );
//             tosres_  -= expand<E>( tmat_ );
//             trefres_ -= expand<E>( trefmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Expansion with subtraction assignment with evaluated Matrix (runtime)
      {
         test_  = "Expansion with subtraction assignment with evaluated Matrix (runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= expand( eval( mat_ ), E );
//             odres_  -= expand( eval( mat_ ), E );
//             sres_   -= expand( eval( mat_ ), E );
//             osres_  -= expand( eval( mat_ ), E );
            refres_ -= expand( eval( refmat_ ), E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= expand( eval( tmat_ ), E );
//             todres_  -= expand( eval( tmat_ ), E );
//             tsres_   -= expand( eval( tmat_ ), E );
//             tosres_  -= expand( eval( tmat_ ), E );
//             trefres_ -= expand( eval( trefmat_ ), E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Expansion with subtraction assignment with evaluated Matrix (compile time)
      {
         test_  = "Expansion with subtraction assignment with evaluated Matrix (compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= expand<E>( eval( mat_ ) );
//             odres_  -= expand<E>( eval( mat_ ) );
//             sres_   -= expand<E>( eval( mat_ ) );
//             osres_  -= expand<E>( eval( mat_ ) );
            refres_ -= expand<E>( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= expand<E>( eval( tmat_ ) );
//             todres_  -= expand<E>( eval( tmat_ ) );
//             tsres_   -= expand<E>( eval( tmat_ ) );
//             tosres_  -= expand<E>( eval( tmat_ ) );
//             trefres_ -= expand<E>( eval( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Expansion with Schur product assignment
      //=====================================================================================

      // Expansion with Schur product assignment with the given Matrix (runtime)
      {
         test_  = "Expansion with Schur product assignment with the given Matrix (runtime)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= expand( mat_, E );
//             odres_  %= expand( mat_, E );
//             sres_   %= expand( mat_, E );
//             osres_  %= expand( mat_, E );
            refres_ %= expand( refmat_, E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= expand( tmat_, E );
//             todres_  %= expand( tmat_, E );
//             tsres_   %= expand( tmat_, E );
//             tosres_  %= expand( tmat_, E );
//             trefres_ %= expand( trefmat_, E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Expansion with Schur product assignment with the given Matrix (compile time)
      {
         test_  = "Expansion with Schur product assignment with the given Matrix (compile time)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= expand<E>( mat_ );
//             odres_  %= expand<E>( mat_ );
//             sres_   %= expand<E>( mat_ );
//             osres_  %= expand<E>( mat_ );
            refres_ %= expand<E>( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= expand<E>( tmat_ );
//             todres_  %= expand<E>( tmat_ );
//             tsres_   %= expand<E>( tmat_ );
//             tosres_  %= expand<E>( tmat_ );
//             trefres_ %= expand<E>( trefmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Expansion with Schur product assignment with evaluated Matrix (runtime)
      {
         test_  = "Expansion with Schur product assignment with evaluated Matrix (runtime)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= expand( eval( mat_ ), E );
//             odres_  %= expand( eval( mat_ ), E );
//             sres_   %= expand( eval( mat_ ), E );
//             osres_  %= expand( eval( mat_ ), E );
            refres_ %= expand( eval( refmat_ ), E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= expand( eval( tmat_ ), E );
//             todres_  %= expand( eval( tmat_ ), E );
//             tsres_   %= expand( eval( tmat_ ), E );
//             tosres_  %= expand( eval( tmat_ ), E );
//             trefres_ %= expand( eval( trefmat_ ), E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Expansion with Schur product assignment with evaluated Matrix (compile time)
      {
         test_  = "Expansion with Schur product assignment with evaluated Matrix (compile time)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= expand<E>( eval( mat_ ) );
//             odres_  %= expand<E>( eval( mat_ ) );
//             sres_   %= expand<E>( eval( mat_ ) );
//             osres_  %= expand<E>( eval( mat_ ) );
            refres_ %= expand<E>( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= expand<E>( eval( tmat_ ) );
//             todres_  %= expand<E>( eval( tmat_ ) );
//             tsres_   %= expand<E>( eval( tmat_ ) );
//             tosres_  %= expand<E>( eval( tmat_ ) );
//             trefres_ %= expand<E>( eval( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the negated dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the negated Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// expansion or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testNegatedOperation()
{
#if BLAZETEST_MATHTEST_TEST_NEGATED_OPERATION
   if( BLAZETEST_MATHTEST_TEST_NEGATED_OPERATION > 1 )
   {
      using blaze::expand;


      //=====================================================================================
      // Negated expansion operation
      //=====================================================================================

      // Negated expansion operation with the given Matrix (runtime)
      {
         test_  = "Negated expansion operation with the given Matrix (runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = -expand( mat_, E );
//             odres_  = -expand( mat_, E );
//             sres_   = -expand( mat_, E );
//             osres_  = -expand( mat_, E );
            refres_ = -expand( refmat_, E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = -expand( tmat_, E );
//             todres_  = -expand( tmat_, E );
//             tsres_   = -expand( tmat_, E );
//             tosres_  = -expand( tmat_, E );
//             trefres_ = -expand( trefmat_, E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Negated expansion operation with the given Matrix (compile time)
      {
         test_  = "Negated expansion operation with the given Matrix (compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = -expand<E>( mat_ );
//             odres_  = -expand<E>( mat_ );
//             sres_   = -expand<E>( mat_ );
//             osres_  = -expand<E>( mat_ );
            refres_ = -expand<E>( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = -expand<E>( tmat_ );
//             todres_  = -expand<E>( tmat_ );
//             tsres_   = -expand<E>( tmat_ );
//             tosres_  = -expand<E>( tmat_ );
//             trefres_ = -expand<E>( trefmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Negated expansion operation with evaluated Matrix (runtime)
      {
         test_  = "Negated expansion operation with evaluated Matrix (runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = -expand( eval( mat_ ), E );
//             odres_  = -expand( eval( mat_ ), E );
//             sres_   = -expand( eval( mat_ ), E );
//             osres_  = -expand( eval( mat_ ), E );
            refres_ = -expand( eval( refmat_ ), E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = -expand( eval( tmat_ ), E );
//             todres_  = -expand( eval( tmat_ ), E );
//             tsres_   = -expand( eval( tmat_ ), E );
//             tosres_  = -expand( eval( tmat_ ), E );
//             trefres_ = -expand( eval( trefmat_ ), E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Negated expansion operation with evaluated Matrix (compile time)
      {
         test_  = "Negated expansion operation with evaluated Matrix (compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = -expand<E>( eval( mat_ ) );
//             odres_  = -expand<E>( eval( mat_ ) );
//             sres_   = -expand<E>( eval( mat_ ) );
//             osres_  = -expand<E>( eval( mat_ ) );
            refres_ = -expand<E>( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = -expand<E>( eval( tmat_ ) );
//             todres_  = -expand<E>( eval( tmat_ ) );
//             tsres_   = -expand<E>( eval( tmat_ ) );
//             tosres_  = -expand<E>( eval( tmat_ ) );
//             trefres_ = -expand<E>( eval( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Negated expansion with addition assignment
      //=====================================================================================

      // Negated expansion with addition assignment with the given Matrix (runtime)
      {
         test_  = "Negated expansion with addition assignment with the given Matrix (runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += -expand( mat_, E );
//             odres_  += -expand( mat_, E );
//             sres_   += -expand( mat_, E );
//             osres_  += -expand( mat_, E );
            refres_ += -expand( refmat_, E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += -expand( tmat_, E );
//             todres_  += -expand( tmat_, E );
//             tsres_   += -expand( tmat_, E );
//             tosres_  += -expand( tmat_, E );
//             trefres_ += -expand( trefmat_, E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Negated expansion with addition assignment with the given Matrix (compile time)
      {
         test_  = "Negated expansion with addition assignment with the given Matrix (compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += -expand<E>( mat_ );
//             odres_  += -expand<E>( mat_ );
//             sres_   += -expand<E>( mat_ );
//             osres_  += -expand<E>( mat_ );
            refres_ += -expand<E>( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += -expand<E>( tmat_ );
//             todres_  += -expand<E>( tmat_ );
//             tsres_   += -expand<E>( tmat_ );
//             tosres_  += -expand<E>( tmat_ );
//             trefres_ += -expand<E>( trefmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Negated expansion with addition assignment with evaluated Matrix (runtime)
      {
         test_  = "Negated expansion with addition assignment with evaluated Matrix (runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += -expand( eval( mat_ ), E );
//             odres_  += -expand( eval( mat_ ), E );
//             sres_   += -expand( eval( mat_ ), E );
//             osres_  += -expand( eval( mat_ ), E );
            refres_ += -expand( eval( refmat_ ), E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += -expand( eval( tmat_ ), E );
//             todres_  += -expand( eval( tmat_ ), E );
//             tsres_   += -expand( eval( tmat_ ), E );
//             tosres_  += -expand( eval( tmat_ ), E );
//             trefres_ += -expand( eval( trefmat_ ), E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Negated expansion with addition assignment with evaluated Matrix (compile time)
      {
         test_  = "Negated expansion with addition assignment with evaluated Matrix (compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += -expand<E>( eval( mat_ ) );
//             odres_  += -expand<E>( eval( mat_ ) );
//             sres_   += -expand<E>( eval( mat_ ) );
//             osres_  += -expand<E>( eval( mat_ ) );
            refres_ += -expand<E>( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += -expand<E>( eval( tmat_ ) );
//             todres_  += -expand<E>( eval( tmat_ ) );
//             tsres_   += -expand<E>( eval( tmat_ ) );
//             tosres_  += -expand<E>( eval( tmat_ ) );
//             trefres_ += -expand<E>( eval( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Negated expansion with subtraction assignment
      //=====================================================================================

      // Negated expansion with subtraction assignment with the given Matrix (runtime)
      {
         test_  = "Negated expansion with subtraction assignment with the given Matrix (runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= -expand( mat_, E );
//             odres_  -= -expand( mat_, E );
//             sres_   -= -expand( mat_, E );
//             osres_  -= -expand( mat_, E );
            refres_ -= -expand( refmat_, E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= -expand( tmat_, E );
//             todres_  -= -expand( tmat_, E );
//             tsres_   -= -expand( tmat_, E );
//             tosres_  -= -expand( tmat_, E );
//             trefres_ -= -expand( trefmat_, E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Negated expansion with subtraction assignment with the given Matrix (compile time)
      {
         test_  = "Negated expansion with subtraction assignment with the given Matrix (compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= -expand<E>( mat_ );
//             odres_  -= -expand<E>( mat_ );
//             sres_   -= -expand<E>( mat_ );
//             osres_  -= -expand<E>( mat_ );
            refres_ -= -expand<E>( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= -expand<E>( tmat_ );
//             todres_  -= -expand<E>( tmat_ );
//             tsres_   -= -expand<E>( tmat_ );
//             tosres_  -= -expand<E>( tmat_ );
//             trefres_ -= -expand<E>( trefmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Negated expansion with subtraction assignment with evaluated Matrix (runtime)
      {
         test_  = "Negated expansion with subtraction assignment with evaluated Matrix (runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= -expand( eval( mat_ ), E );
//             odres_  -= -expand( eval( mat_ ), E );
//             sres_   -= -expand( eval( mat_ ), E );
//             osres_  -= -expand( eval( mat_ ), E );
            refres_ -= -expand( eval( refmat_ ), E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= -expand( eval( tmat_ ), E );
//             todres_  -= -expand( eval( tmat_ ), E );
//             tsres_   -= -expand( eval( tmat_ ), E );
//             tosres_  -= -expand( eval( tmat_ ), E );
//             trefres_ -= -expand( eval( trefmat_ ), E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Negated expansion with subtraction assignment with evaluated Matrix (compile time)
      {
         test_  = "Negated expansion with subtraction assignment with evaluated Matrix (compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= -expand<E>( eval( mat_ ) );
//             odres_  -= -expand<E>( eval( mat_ ) );
//             sres_   -= -expand<E>( eval( mat_ ) );
//             osres_  -= -expand<E>( eval( mat_ ) );
            refres_ -= -expand<E>( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= -expand<E>( eval( tmat_ ) );
//             todres_  -= -expand<E>( eval( tmat_ ) );
//             tsres_   -= -expand<E>( eval( tmat_ ) );
//             tosres_  -= -expand<E>( eval( tmat_ ) );
//             trefres_ -= -expand<E>( eval( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Negated expansion with Schur product assignment
      //=====================================================================================

      // Negated expansion with Schur product assignment with the given Matrix (runtime)
      {
         test_  = "Negated expansion with Schur product assignment with the given Matrix (runtime)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= -expand( mat_, E );
//             odres_  %= -expand( mat_, E );
//             sres_   %= -expand( mat_, E );
//             osres_  %= -expand( mat_, E );
            refres_ %= -expand( refmat_, E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= -expand( tmat_, E );
//             todres_  %= -expand( tmat_, E );
//             tsres_   %= -expand( tmat_, E );
//             tosres_  %= -expand( tmat_, E );
//             trefres_ %= -expand( trefmat_, E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Negated expansion with Schur product assignment with the given Matrix (compile time)
      {
         test_  = "Negated expansion with Schur product assignment with the given Matrix (compile time)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= -expand<E>( mat_ );
//             odres_  %= -expand<E>( mat_ );
//             sres_   %= -expand<E>( mat_ );
//             osres_  %= -expand<E>( mat_ );
            refres_ %= -expand<E>( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= -expand<E>( tmat_ );
//             todres_  %= -expand<E>( tmat_ );
//             tsres_   %= -expand<E>( tmat_ );
//             tosres_  %= -expand<E>( tmat_ );
//             trefres_ %= -expand<E>( trefmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Negated expansion with Schur product assignment with evaluated Matrix (runtime)
      {
         test_  = "Negated expansion with Schur product assignment with evaluated Matrix (runtime)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= -expand( eval( mat_ ), E );
//             odres_  %= -expand( eval( mat_ ), E );
//             sres_   %= -expand( eval( mat_ ), E );
//             osres_  %= -expand( eval( mat_ ), E );
            refres_ %= -expand( eval( refmat_ ), E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= -expand( eval( tmat_ ), E );
//             todres_  %= -expand( eval( tmat_ ), E );
//             tsres_   %= -expand( eval( tmat_ ), E );
//             tosres_  %= -expand( eval( tmat_ ), E );
//             trefres_ %= -expand( eval( trefmat_ ), E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Negated expansion with Schur product assignment with evaluated Matrix (compile time)
      {
         test_  = "Negated expansion with Schur product assignment with evaluated Matrix (compile time)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= -expand<E>( eval( mat_ ) );
//             odres_  %= -expand<E>( eval( mat_ ) );
//             sres_   %= -expand<E>( eval( mat_ ) );
//             osres_  %= -expand<E>( eval( mat_ ) );
            refres_ %= -expand<E>( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= -expand<E>( eval( tmat_ ) );
//             todres_  %= -expand<E>( eval( tmat_ ) );
//             tsres_   %= -expand<E>( eval( tmat_ ) );
//             tosres_  %= -expand<E>( eval( tmat_ ) );
//             trefres_ %= -expand<E>( eval( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the scaled dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the scaled Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// expansion or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
template< typename T >    // Type of the scalar
void OperationTest<MT,E>::testScaledOperation( T scalar )
{
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( T );

   if( scalar == T(0) )
      throw std::invalid_argument( "Invalid scalar parameter" );


#if BLAZETEST_MATHTEST_TEST_SCALED_OPERATION
   if( BLAZETEST_MATHTEST_TEST_SCALED_OPERATION > 1 )
   {
      using blaze::expand;


      //=====================================================================================
      // Scaled expansion (s*OP)
      //=====================================================================================

      // Scaled expansion operation with the given Matrix (s*OP, runtime)
      {
         test_  = "Scaled expansion operation with the given Matrix (s*OP, runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = scalar * expand( mat_, E );
//             odres_  = scalar * expand( mat_, E );
//             sres_   = scalar * expand( mat_, E );
//             osres_  = scalar * expand( mat_, E );
            refres_ = scalar * expand( refmat_, E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = scalar * expand( tmat_, E );
//             todres_  = scalar * expand( tmat_, E );
//             tsres_   = scalar * expand( tmat_, E );
//             tosres_  = scalar * expand( tmat_, E );
//             trefres_ = scalar * expand( trefmat_, E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with the given Matrix (s*OP, compile time)
      {
         test_  = "Scaled expansion operation with the given Matrix (s*OP, compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = scalar * expand<E>( mat_ );
//             odres_  = scalar * expand<E>( mat_ );
//             sres_   = scalar * expand<E>( mat_ );
//             osres_  = scalar * expand<E>( mat_ );
            refres_ = scalar * expand<E>( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = scalar * expand<E>( tmat_ );
//             todres_  = scalar * expand<E>( tmat_ );
//             tsres_   = scalar * expand<E>( tmat_ );
//             tosres_  = scalar * expand<E>( tmat_ );
//             trefres_ = scalar * expand<E>( trefmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with evaluated Matrix (s*OP, runtime)
      {
         test_  = "Scaled expansion operation with evaluated Matrix (s*OP, runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = scalar * expand( eval( mat_ ), E );
//             odres_  = scalar * expand( eval( mat_ ), E );
//             sres_   = scalar * expand( eval( mat_ ), E );
//             osres_  = scalar * expand( eval( mat_ ), E );
            refres_ = scalar * expand( eval( refmat_ ), E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = scalar * expand( eval( tmat_ ), E );
//             todres_  = scalar * expand( eval( tmat_ ), E );
//             tsres_   = scalar * expand( eval( tmat_ ), E );
//             tosres_  = scalar * expand( eval( tmat_ ), E );
//             trefres_ = scalar * expand( eval( trefmat_ ), E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with evaluated Matrix (s*OP, compile time)
      {
         test_  = "Scaled expansion operation with evaluated Matrix (s*OP, compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = scalar * expand<E>( eval( mat_ ) );
//             odres_  = scalar * expand<E>( eval( mat_ ) );
//             sres_   = scalar * expand<E>( eval( mat_ ) );
//             osres_  = scalar * expand<E>( eval( mat_ ) );
            refres_ = scalar * expand<E>( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = scalar * expand<E>( eval( tmat_ ) );
//             todres_  = scalar * expand<E>( eval( tmat_ ) );
//             tsres_   = scalar * expand<E>( eval( tmat_ ) );
//             tosres_  = scalar * expand<E>( eval( tmat_ ) );
//             trefres_ = scalar * expand<E>( eval( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Scaled expansion (OP*s)
      //=====================================================================================

      // Scaled expansion operation with the given Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with the given Matrix (OP*s, runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = expand( mat_, E ) * scalar;
//             odres_  = expand( mat_, E ) * scalar;
//             sres_   = expand( mat_, E ) * scalar;
//             osres_  = expand( mat_, E ) * scalar;
            refres_ = expand( refmat_, E ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = expand( tmat_, E ) * scalar;
//             todres_  = expand( tmat_, E ) * scalar;
//             tsres_   = expand( tmat_, E ) * scalar;
//             tosres_  = expand( tmat_, E ) * scalar;
//             trefres_ = expand( trefmat_, E ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with the given Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with the given Matrix (OP*s, compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = expand<E>( mat_ ) * scalar;
//             odres_  = expand<E>( mat_ ) * scalar;
//             sres_   = expand<E>( mat_ ) * scalar;
//             osres_  = expand<E>( mat_ ) * scalar;
            refres_ = expand<E>( refmat_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = expand<E>( tmat_ ) * scalar;
//             todres_  = expand<E>( tmat_ ) * scalar;
//             tsres_   = expand<E>( tmat_ ) * scalar;
//             tosres_  = expand<E>( tmat_ ) * scalar;
//             trefres_ = expand<E>( trefmat_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with evaluated Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with evaluated Matrix (OP*s, runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = expand( eval( mat_ ), E ) * scalar;
//             odres_  = expand( eval( mat_ ), E ) * scalar;
//             sres_   = expand( eval( mat_ ), E ) * scalar;
//             osres_  = expand( eval( mat_ ), E ) * scalar;
            refres_ = expand( eval( refmat_ ), E ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = expand( eval( tmat_ ), E ) * scalar;
//             todres_  = expand( eval( tmat_ ), E ) * scalar;
//             tsres_   = expand( eval( tmat_ ), E ) * scalar;
//             tosres_  = expand( eval( tmat_ ), E ) * scalar;
//             trefres_ = expand( eval( trefmat_ ), E ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with evaluated Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with evaluated Matrix (OP*s, compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = expand<E>( eval( mat_ ) ) * scalar;
//             odres_  = expand<E>( eval( mat_ ) ) * scalar;
//             sres_   = expand<E>( eval( mat_ ) ) * scalar;
//             osres_  = expand<E>( eval( mat_ ) ) * scalar;
            refres_ = expand<E>( eval( refmat_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = expand<E>( eval( tmat_ ) ) * scalar;
//             todres_  = expand<E>( eval( tmat_ ) ) * scalar;
//             tsres_   = expand<E>( eval( tmat_ ) ) * scalar;
//             tosres_  = expand<E>( eval( tmat_ ) ) * scalar;
//             trefres_ = expand<E>( eval( trefmat_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Scaled expansion (OP/s)
      //=====================================================================================

      // Scaled expansion operation with the given Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with the given Matrix (OP*s, runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = expand( mat_, E ) / scalar;
//             odres_  = expand( mat_, E ) / scalar;
//             sres_   = expand( mat_, E ) / scalar;
//             osres_  = expand( mat_, E ) / scalar;
            refres_ = expand( refmat_, E ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = expand( tmat_, E ) / scalar;
//             todres_  = expand( tmat_, E ) / scalar;
//             tsres_   = expand( tmat_, E ) / scalar;
//             tosres_  = expand( tmat_, E ) / scalar;
//             trefres_ = expand( trefmat_, E ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with the given Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with the given Matrix (OP*s, compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = expand<E>( mat_ ) / scalar;
//             odres_  = expand<E>( mat_ ) / scalar;
//             sres_   = expand<E>( mat_ ) / scalar;
//             osres_  = expand<E>( mat_ ) / scalar;
            refres_ = expand<E>( refmat_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = expand<E>( tmat_ ) / scalar;
//             todres_  = expand<E>( tmat_ ) / scalar;
//             tsres_   = expand<E>( tmat_ ) / scalar;
//             tosres_  = expand<E>( tmat_ ) / scalar;
//             trefres_ = expand<E>( trefmat_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with evaluated Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with evaluated Matrix (OP*s, runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = expand( eval( mat_ ), E ) / scalar;
//             odres_  = expand( eval( mat_ ), E ) / scalar;
//             sres_   = expand( eval( mat_ ), E ) / scalar;
//             osres_  = expand( eval( mat_ ), E ) / scalar;
            refres_ = expand( eval( refmat_ ), E ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = expand( eval( tmat_ ), E ) / scalar;
//             todres_  = expand( eval( tmat_ ), E ) / scalar;
//             tsres_   = expand( eval( tmat_ ), E ) / scalar;
//             tosres_  = expand( eval( tmat_ ), E ) / scalar;
//             trefres_ = expand( eval( trefmat_ ), E ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with evaluated Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with evaluated Matrix (OP*s, compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            dres_   = expand<E>( eval( mat_ ) ) / scalar;
//             odres_  = expand<E>( eval( mat_ ) ) / scalar;
//             sres_   = expand<E>( eval( mat_ ) ) / scalar;
//             osres_  = expand<E>( eval( mat_ ) ) / scalar;
            refres_ = expand<E>( eval( refmat_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   = expand<E>( eval( tmat_ ) ) / scalar;
//             todres_  = expand<E>( eval( tmat_ ) ) / scalar;
//             tsres_   = expand<E>( eval( tmat_ ) ) / scalar;
//             tosres_  = expand<E>( eval( tmat_ ) ) / scalar;
//             trefres_ = expand<E>( eval( trefmat_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Scaled expansion with addition assignment (s*OP)
      //=====================================================================================

      // Scaled expansion operation with addition assignment with the given Matrix (s*OP, runtime)
      {
         test_  = "Scaled expansion operation with addition assignment with the given Matrix (s*OP, runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += scalar * expand( mat_, E );
//             odres_  += scalar * expand( mat_, E );
//             sres_   += scalar * expand( mat_, E );
//             osres_  += scalar * expand( mat_, E );
            refres_ += scalar * expand( refmat_, E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += scalar * expand( tmat_, E );
//             todres_  += scalar * expand( tmat_, E );
//             tsres_   += scalar * expand( tmat_, E );
//             tosres_  += scalar * expand( tmat_, E );
//             trefres_ += scalar * expand( trefmat_, E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with addition assignment with the given Matrix (s*OP, compile time)
      {
         test_  = "Scaled expansion operation with addition assignment with the given Matrix (s*OP, compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += scalar * expand<E>( mat_ );
//             odres_  += scalar * expand<E>( mat_ );
//             sres_   += scalar * expand<E>( mat_ );
//             osres_  += scalar * expand<E>( mat_ );
            refres_ += scalar * expand<E>( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += scalar * expand<E>( tmat_ );
//             todres_  += scalar * expand<E>( tmat_ );
//             tsres_   += scalar * expand<E>( tmat_ );
//             tosres_  += scalar * expand<E>( tmat_ );
//             trefres_ += scalar * expand<E>( trefmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with addition assignment with evaluated Matrix (s*OP, runtime)
      {
         test_  = "Scaled expansion operation with addition assignment with evaluated Matrix (s*OP, runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += scalar * expand( eval( mat_ ), E );
//             odres_  += scalar * expand( eval( mat_ ), E );
//             sres_   += scalar * expand( eval( mat_ ), E );
//             osres_  += scalar * expand( eval( mat_ ), E );
            refres_ += scalar * expand( eval( refmat_ ), E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += scalar * expand( eval( tmat_ ), E );
//             todres_  += scalar * expand( eval( tmat_ ), E );
//             tsres_   += scalar * expand( eval( tmat_ ), E );
//             tosres_  += scalar * expand( eval( tmat_ ), E );
//             trefres_ += scalar * expand( eval( trefmat_ ), E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with addition assignment with evaluated Matrix (s*OP, compile time)
      {
         test_  = "Scaled expansion operation with addition assignment with evaluated Matrix (s*OP, compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += scalar * expand<E>( eval( mat_ ) );
//             odres_  += scalar * expand<E>( eval( mat_ ) );
//             sres_   += scalar * expand<E>( eval( mat_ ) );
//             osres_  += scalar * expand<E>( eval( mat_ ) );
            refres_ += scalar * expand<E>( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += scalar * expand<E>( eval( tmat_ ) );
//             todres_  += scalar * expand<E>( eval( tmat_ ) );
//             tsres_   += scalar * expand<E>( eval( tmat_ ) );
//             tosres_  += scalar * expand<E>( eval( tmat_ ) );
//             trefres_ += scalar * expand<E>( eval( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Scaled expansion with addition assignment (OP*s)
      //=====================================================================================

      // Scaled expansion operation with addition assignment with the given Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with addition assignment with the given Matrix (OP*s, runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += expand( mat_, E ) * scalar;
//             odres_  += expand( mat_, E ) * scalar;
//             sres_   += expand( mat_, E ) * scalar;
//             osres_  += expand( mat_, E ) * scalar;
            refres_ += expand( refmat_, E ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += expand( tmat_, E ) * scalar;
//             todres_  += expand( tmat_, E ) * scalar;
//             tsres_   += expand( tmat_, E ) * scalar;
//             tosres_  += expand( tmat_, E ) * scalar;
//             trefres_ += expand( trefmat_, E ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with addition assignment with the given Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with addition assignment with the given Matrix (OP*s, compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += expand<E>( mat_ ) * scalar;
//             odres_  += expand<E>( mat_ ) * scalar;
//             sres_   += expand<E>( mat_ ) * scalar;
//             osres_  += expand<E>( mat_ ) * scalar;
            refres_ += expand<E>( refmat_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += expand<E>( tmat_ ) * scalar;
//             todres_  += expand<E>( tmat_ ) * scalar;
//             tsres_   += expand<E>( tmat_ ) * scalar;
//             tosres_  += expand<E>( tmat_ ) * scalar;
//             trefres_ += expand<E>( trefmat_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with addition assignment with evaluated Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with addition assignment with evaluated Matrix (OP*s, runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += expand( eval( mat_ ), E ) * scalar;
//             odres_  += expand( eval( mat_ ), E ) * scalar;
//             sres_   += expand( eval( mat_ ), E ) * scalar;
//             osres_  += expand( eval( mat_ ), E ) * scalar;
            refres_ += expand( eval( refmat_ ), E ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += expand( eval( tmat_ ), E ) * scalar;
//             todres_  += expand( eval( tmat_ ), E ) * scalar;
//             tsres_   += expand( eval( tmat_ ), E ) * scalar;
//             tosres_  += expand( eval( tmat_ ), E ) * scalar;
//             trefres_ += expand( eval( trefmat_ ), E ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with addition assignment with evaluated Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with addition assignment with evaluated Matrix (OP*s, compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += expand<E>( eval( mat_ ) ) * scalar;
//             odres_  += expand<E>( eval( mat_ ) ) * scalar;
//             sres_   += expand<E>( eval( mat_ ) ) * scalar;
//             osres_  += expand<E>( eval( mat_ ) ) * scalar;
            refres_ += expand<E>( eval( refmat_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += expand<E>( eval( tmat_ ) ) * scalar;
//             todres_  += expand<E>( eval( tmat_ ) ) * scalar;
//             tsres_   += expand<E>( eval( tmat_ ) ) * scalar;
//             tosres_  += expand<E>( eval( tmat_ ) ) * scalar;
//             trefres_ += expand<E>( eval( trefmat_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Scaled expansion (OP/s)
      //=====================================================================================

      // Scaled expansion operation with addition assignment with the given Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with addition assignment with the given Matrix (OP*s, runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += expand( mat_, E ) / scalar;
//             odres_  += expand( mat_, E ) / scalar;
//             sres_   += expand( mat_, E ) / scalar;
//             osres_  += expand( mat_, E ) / scalar;
            refres_ += expand( refmat_, E ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += expand( tmat_, E ) / scalar;
//             todres_  += expand( tmat_, E ) / scalar;
//             tsres_   += expand( tmat_, E ) / scalar;
//             tosres_  += expand( tmat_, E ) / scalar;
//             trefres_ += expand( trefmat_, E ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with addition assignment with the given Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with addition assignment with the given Matrix (OP*s, compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += expand<E>( mat_ ) / scalar;
//             odres_  += expand<E>( mat_ ) / scalar;
//             sres_   += expand<E>( mat_ ) / scalar;
//             osres_  += expand<E>( mat_ ) / scalar;
            refres_ += expand<E>( refmat_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += expand<E>( tmat_ ) / scalar;
//             todres_  += expand<E>( tmat_ ) / scalar;
//             tsres_   += expand<E>( tmat_ ) / scalar;
//             tosres_  += expand<E>( tmat_ ) / scalar;
//             trefres_ += expand<E>( trefmat_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with addition assignment with evaluated Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with addition assignment with evaluated Matrix (OP*s, runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += expand( eval( mat_ ), E ) / scalar;
//             odres_  += expand( eval( mat_ ), E ) / scalar;
//             sres_   += expand( eval( mat_ ), E ) / scalar;
//             osres_  += expand( eval( mat_ ), E ) / scalar;
            refres_ += expand( eval( refmat_ ), E ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += expand( eval( tmat_ ), E ) / scalar;
//             todres_  += expand( eval( tmat_ ), E ) / scalar;
//             tsres_   += expand( eval( tmat_ ), E ) / scalar;
//             tosres_  += expand( eval( tmat_ ), E ) / scalar;
//             trefres_ += expand( eval( trefmat_ ), E ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with addition assignment with evaluated Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with addition assignment with evaluated Matrix (OP*s, compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            dres_   += expand<E>( eval( mat_ ) ) / scalar;
//             odres_  += expand<E>( eval( mat_ ) ) / scalar;
//             sres_   += expand<E>( eval( mat_ ) ) / scalar;
//             osres_  += expand<E>( eval( mat_ ) ) / scalar;
            refres_ += expand<E>( eval( refmat_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   += expand<E>( eval( tmat_ ) ) / scalar;
//             todres_  += expand<E>( eval( tmat_ ) ) / scalar;
//             tsres_   += expand<E>( eval( tmat_ ) ) / scalar;
//             tosres_  += expand<E>( eval( tmat_ ) ) / scalar;
//             trefres_ += expand<E>( eval( trefmat_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Scaled expansion with subtraction assignment (s*OP)
      //=====================================================================================

      // Scaled expansion operation with subtraction assignment with the given Matrix (s*OP, runtime)
      {
         test_  = "Scaled expansion operation with subtraction assignment with the given Matrix (s*OP, runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= scalar * expand( mat_, E );
//             odres_  -= scalar * expand( mat_, E );
//             sres_   -= scalar * expand( mat_, E );
//             osres_  -= scalar * expand( mat_, E );
            refres_ -= scalar * expand( refmat_, E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= scalar * expand( tmat_, E );
//             todres_  -= scalar * expand( tmat_, E );
//             tsres_   -= scalar * expand( tmat_, E );
//             tosres_  -= scalar * expand( tmat_, E );
//             trefres_ -= scalar * expand( trefmat_, E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with subtraction assignment with the given Matrix (s*OP, compile time)
      {
         test_  = "Scaled expansion operation with subtraction assignment with the given Matrix (s*OP, compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= scalar * expand<E>( mat_ );
//             odres_  -= scalar * expand<E>( mat_ );
//             sres_   -= scalar * expand<E>( mat_ );
//             osres_  -= scalar * expand<E>( mat_ );
            refres_ -= scalar * expand<E>( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= scalar * expand<E>( tmat_ );
//             todres_  -= scalar * expand<E>( tmat_ );
//             tsres_   -= scalar * expand<E>( tmat_ );
//             tosres_  -= scalar * expand<E>( tmat_ );
//             trefres_ -= scalar * expand<E>( trefmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with subtraction assignment with evaluated Matrix (s*OP, runtime)
      {
         test_  = "Scaled expansion operation with subtraction assignment with evaluated Matrix (s*OP, runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= scalar * expand( eval( mat_ ), E );
//             odres_  -= scalar * expand( eval( mat_ ), E );
//             sres_   -= scalar * expand( eval( mat_ ), E );
//             osres_  -= scalar * expand( eval( mat_ ), E );
            refres_ -= scalar * expand( eval( refmat_ ), E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= scalar * expand( eval( tmat_ ), E );
//             todres_  -= scalar * expand( eval( tmat_ ), E );
//             tsres_   -= scalar * expand( eval( tmat_ ), E );
//             tosres_  -= scalar * expand( eval( tmat_ ), E );
//             trefres_ -= scalar * expand( eval( trefmat_ ), E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with subtraction assignment with evaluated Matrix (s*OP, compile time)
      {
         test_  = "Scaled expansion operation with subtraction assignment with evaluated Matrix (s*OP, compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= scalar * expand<E>( eval( mat_ ) );
//             odres_  -= scalar * expand<E>( eval( mat_ ) );
//             sres_   -= scalar * expand<E>( eval( mat_ ) );
//             osres_  -= scalar * expand<E>( eval( mat_ ) );
            refres_ -= scalar * expand<E>( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= scalar * expand<E>( eval( tmat_ ) );
//             todres_  -= scalar * expand<E>( eval( tmat_ ) );
//             tsres_   -= scalar * expand<E>( eval( tmat_ ) );
//             tosres_  -= scalar * expand<E>( eval( tmat_ ) );
//             trefres_ -= scalar * expand<E>( eval( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Scaled expansion with subtraction assignment (OP*s)
      //=====================================================================================

      // Scaled expansion operation with subtraction assignment with the given Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with subtraction assignment with the given Matrix (OP*s, runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= expand( mat_, E ) * scalar;
//             odres_  -= expand( mat_, E ) * scalar;
//             sres_   -= expand( mat_, E ) * scalar;
//             osres_  -= expand( mat_, E ) * scalar;
            refres_ -= expand( refmat_, E ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= expand( tmat_, E ) * scalar;
//             todres_  -= expand( tmat_, E ) * scalar;
//             tsres_   -= expand( tmat_, E ) * scalar;
//             tosres_  -= expand( tmat_, E ) * scalar;
//             trefres_ -= expand( trefmat_, E ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with subtraction assignment with the given Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with subtraction assignment with the given Matrix (OP*s, compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= expand<E>( mat_ ) * scalar;
//             odres_  -= expand<E>( mat_ ) * scalar;
//             sres_   -= expand<E>( mat_ ) * scalar;
//             osres_  -= expand<E>( mat_ ) * scalar;
            refres_ -= expand<E>( refmat_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= expand<E>( tmat_ ) * scalar;
//             todres_  -= expand<E>( tmat_ ) * scalar;
//             tsres_   -= expand<E>( tmat_ ) * scalar;
//             tosres_  -= expand<E>( tmat_ ) * scalar;
//             trefres_ -= expand<E>( trefmat_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with subtraction assignment with evaluated Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with subtraction assignment with evaluated Matrix (OP*s, runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= expand( eval( mat_ ), E ) * scalar;
//             odres_  -= expand( eval( mat_ ), E ) * scalar;
//             sres_   -= expand( eval( mat_ ), E ) * scalar;
//             osres_  -= expand( eval( mat_ ), E ) * scalar;
            refres_ -= expand( eval( refmat_ ), E ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= expand( eval( tmat_ ), E ) * scalar;
//             todres_  -= expand( eval( tmat_ ), E ) * scalar;
//             tsres_   -= expand( eval( tmat_ ), E ) * scalar;
//             tosres_  -= expand( eval( tmat_ ), E ) * scalar;
//             trefres_ -= expand( eval( trefmat_ ), E ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with subtraction assignment with evaluated Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with subtraction assignment with evaluated Matrix (OP*s, compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= expand<E>( eval( mat_ ) ) * scalar;
//             odres_  -= expand<E>( eval( mat_ ) ) * scalar;
//             sres_   -= expand<E>( eval( mat_ ) ) * scalar;
//             osres_  -= expand<E>( eval( mat_ ) ) * scalar;
            refres_ -= expand<E>( eval( refmat_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= expand<E>( eval( tmat_ ) ) * scalar;
//             todres_  -= expand<E>( eval( tmat_ ) ) * scalar;
//             tsres_   -= expand<E>( eval( tmat_ ) ) * scalar;
//             tosres_  -= expand<E>( eval( tmat_ ) ) * scalar;
//             trefres_ -= expand<E>( eval( trefmat_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Scaled expansion (OP/s)
      //=====================================================================================

      // Scaled expansion operation with subtraction assignment with the given Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with subtraction assignment with the given Matrix (OP*s, runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= expand( mat_, E ) / scalar;
//             odres_  -= expand( mat_, E ) / scalar;
//             sres_   -= expand( mat_, E ) / scalar;
//             osres_  -= expand( mat_, E ) / scalar;
            refres_ -= expand( refmat_, E ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= expand( tmat_, E ) / scalar;
//             todres_  -= expand( tmat_, E ) / scalar;
//             tsres_   -= expand( tmat_, E ) / scalar;
//             tosres_  -= expand( tmat_, E ) / scalar;
//             trefres_ -= expand( trefmat_, E ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with subtraction assignment with the given Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with subtraction assignment with the given Matrix (OP*s, compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= expand<E>( mat_ ) / scalar;
//             odres_  -= expand<E>( mat_ ) / scalar;
//             sres_   -= expand<E>( mat_ ) / scalar;
//             osres_  -= expand<E>( mat_ ) / scalar;
            refres_ -= expand<E>( refmat_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= expand<E>( tmat_ ) / scalar;
//             todres_  -= expand<E>( tmat_ ) / scalar;
//             tsres_   -= expand<E>( tmat_ ) / scalar;
//             tosres_  -= expand<E>( tmat_ ) / scalar;
//             trefres_ -= expand<E>( trefmat_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with subtraction assignment with evaluated Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with subtraction assignment with evaluated Matrix (OP*s, runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= expand( eval( mat_ ), E ) / scalar;
//             odres_  -= expand( eval( mat_ ), E ) / scalar;
//             sres_   -= expand( eval( mat_ ), E ) / scalar;
//             osres_  -= expand( eval( mat_ ), E ) / scalar;
            refres_ -= expand( eval( refmat_ ), E ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= expand( eval( tmat_ ), E ) / scalar;
//             todres_  -= expand( eval( tmat_ ), E ) / scalar;
//             tsres_   -= expand( eval( tmat_ ), E ) / scalar;
//             tosres_  -= expand( eval( tmat_ ), E ) / scalar;
//             trefres_ -= expand( eval( trefmat_ ), E ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with subtraction assignment with evaluated Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with subtraction assignment with evaluated Matrix (OP*s, compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            dres_   -= expand<E>( eval( mat_ ) ) / scalar;
//             odres_  -= expand<E>( eval( mat_ ) ) / scalar;
//             sres_   -= expand<E>( eval( mat_ ) ) / scalar;
//             osres_  -= expand<E>( eval( mat_ ) ) / scalar;
            refres_ -= expand<E>( eval( refmat_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   -= expand<E>( eval( tmat_ ) ) / scalar;
//             todres_  -= expand<E>( eval( tmat_ ) ) / scalar;
//             tsres_   -= expand<E>( eval( tmat_ ) ) / scalar;
//             tosres_  -= expand<E>( eval( tmat_ ) ) / scalar;
//             trefres_ -= expand<E>( eval( trefmat_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Scaled expansion with Schur product assignment (s*OP)
      //=====================================================================================

      // Scaled expansion operation with Schur product assignment with the given Matrix (s*OP, runtime)
      {
         test_  = "Scaled expansion operation with Schur product assignment with the given Matrix (s*OP, runtime)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= scalar * expand( mat_, E );
//             odres_  %= scalar * expand( mat_, E );
//             sres_   %= scalar * expand( mat_, E );
//             osres_  %= scalar * expand( mat_, E );
            refres_ %= scalar * expand( refmat_, E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= scalar * expand( tmat_, E );
//             todres_  %= scalar * expand( tmat_, E );
//             tsres_   %= scalar * expand( tmat_, E );
//             tosres_  %= scalar * expand( tmat_, E );
//             trefres_ %= scalar * expand( trefmat_, E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with Schur product assignment with the given Matrix (s*OP, compile time)
      {
         test_  = "Scaled expansion operation with Schur product assignment with the given Matrix (s*OP, compile time)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= scalar * expand<E>( mat_ );
//             odres_  %= scalar * expand<E>( mat_ );
//             sres_   %= scalar * expand<E>( mat_ );
//             osres_  %= scalar * expand<E>( mat_ );
            refres_ %= scalar * expand<E>( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= scalar * expand<E>( tmat_ );
//             todres_  %= scalar * expand<E>( tmat_ );
//             tsres_   %= scalar * expand<E>( tmat_ );
//             tosres_  %= scalar * expand<E>( tmat_ );
//             trefres_ %= scalar * expand<E>( trefmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with Schur product assignment with evaluated Matrix (s*OP, runtime)
      {
         test_  = "Scaled expansion operation with Schur product assignment with evaluated Matrix (s*OP, runtime)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= scalar * expand( eval( mat_ ), E );
//             odres_  %= scalar * expand( eval( mat_ ), E );
//             sres_   %= scalar * expand( eval( mat_ ), E );
//             osres_  %= scalar * expand( eval( mat_ ), E );
            refres_ %= scalar * expand( eval( refmat_ ), E );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= scalar * expand( eval( tmat_ ), E );
//             todres_  %= scalar * expand( eval( tmat_ ), E );
//             tsres_   %= scalar * expand( eval( tmat_ ), E );
//             tosres_  %= scalar * expand( eval( tmat_ ), E );
//             trefres_ %= scalar * expand( eval( trefmat_ ), E );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with Schur product assignment with evaluated Matrix (s*OP, compile time)
      {
         test_  = "Scaled expansion operation with Schur product assignment with evaluated Matrix (s*OP, compile time)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= scalar * expand<E>( eval( mat_ ) );
//             odres_  %= scalar * expand<E>( eval( mat_ ) );
//             sres_   %= scalar * expand<E>( eval( mat_ ) );
//             osres_  %= scalar * expand<E>( eval( mat_ ) );
            refres_ %= scalar * expand<E>( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= scalar * expand<E>( eval( tmat_ ) );
//             todres_  %= scalar * expand<E>( eval( tmat_ ) );
//             tsres_   %= scalar * expand<E>( eval( tmat_ ) );
//             tosres_  %= scalar * expand<E>( eval( tmat_ ) );
//             trefres_ %= scalar * expand<E>( eval( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Scaled expansion with Schur product assignment (OP*s)
      //=====================================================================================

      // Scaled expansion operation with Schur product assignment with the given Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with Schur product assignment with the given Matrix (OP*s, runtime)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= expand( mat_, E ) * scalar;
//             odres_  %= expand( mat_, E ) * scalar;
//             sres_   %= expand( mat_, E ) * scalar;
//             osres_  %= expand( mat_, E ) * scalar;
            refres_ %= expand( refmat_, E ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= expand( tmat_, E ) * scalar;
//             todres_  %= expand( tmat_, E ) * scalar;
//             tsres_   %= expand( tmat_, E ) * scalar;
//             tosres_  %= expand( tmat_, E ) * scalar;
//             trefres_ %= expand( trefmat_, E ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with Schur product assignment with the given Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with Schur product assignment with the given Matrix (OP*s, compile time)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= expand<E>( mat_ ) * scalar;
//             odres_  %= expand<E>( mat_ ) * scalar;
//             sres_   %= expand<E>( mat_ ) * scalar;
//             osres_  %= expand<E>( mat_ ) * scalar;
            refres_ %= expand<E>( refmat_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= expand<E>( tmat_ ) * scalar;
//             todres_  %= expand<E>( tmat_ ) * scalar;
//             tsres_   %= expand<E>( tmat_ ) * scalar;
//             tosres_  %= expand<E>( tmat_ ) * scalar;
//             trefres_ %= expand<E>( trefmat_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with Schur product assignment with evaluated Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with Schur product assignment with evaluated Matrix (OP*s, runtime)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= expand( eval( mat_ ), E ) * scalar;
//             odres_  %= expand( eval( mat_ ), E ) * scalar;
//             sres_   %= expand( eval( mat_ ), E ) * scalar;
//             osres_  %= expand( eval( mat_ ), E ) * scalar;
            refres_ %= expand( eval( refmat_ ), E ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= expand( eval( tmat_ ), E ) * scalar;
//             todres_  %= expand( eval( tmat_ ), E ) * scalar;
//             tsres_   %= expand( eval( tmat_ ), E ) * scalar;
//             tosres_  %= expand( eval( tmat_ ), E ) * scalar;
//             trefres_ %= expand( eval( trefmat_ ), E ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with Schur product assignment with evaluated Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with Schur product assignment with evaluated Matrix (OP*s, compile time)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= expand<E>( eval( mat_ ) ) * scalar;
//             odres_  %= expand<E>( eval( mat_ ) ) * scalar;
//             sres_   %= expand<E>( eval( mat_ ) ) * scalar;
//             osres_  %= expand<E>( eval( mat_ ) ) * scalar;
            refres_ %= expand<E>( eval( refmat_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= expand<E>( eval( tmat_ ) ) * scalar;
//             todres_  %= expand<E>( eval( tmat_ ) ) * scalar;
//             tsres_   %= expand<E>( eval( tmat_ ) ) * scalar;
//             tosres_  %= expand<E>( eval( tmat_ ) ) * scalar;
//             trefres_ %= expand<E>( eval( trefmat_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Scaled expansion (OP/s)
      //=====================================================================================

      // Scaled expansion operation with Schur product assignment with the given Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with Schur product assignment with the given Matrix (OP*s, runtime)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= expand( mat_, E ) / scalar;
//             odres_  %= expand( mat_, E ) / scalar;
//             sres_   %= expand( mat_, E ) / scalar;
//             osres_  %= expand( mat_, E ) / scalar;
            refres_ %= expand( refmat_, E ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= expand( tmat_, E ) / scalar;
//             todres_  %= expand( tmat_, E ) / scalar;
//             tsres_   %= expand( tmat_, E ) / scalar;
//             tosres_  %= expand( tmat_, E ) / scalar;
//             trefres_ %= expand( trefmat_, E ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with Schur product assignment with the given Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with Schur product assignment with the given Matrix (OP*s, compile time)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= expand<E>( mat_ ) / scalar;
//             odres_  %= expand<E>( mat_ ) / scalar;
//             sres_   %= expand<E>( mat_ ) / scalar;
//             osres_  %= expand<E>( mat_ ) / scalar;
            refres_ %= expand<E>( refmat_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= expand<E>( tmat_ ) / scalar;
//             todres_  %= expand<E>( tmat_ ) / scalar;
//             tsres_   %= expand<E>( tmat_ ) / scalar;
//             tosres_  %= expand<E>( tmat_ ) / scalar;
//             trefres_ %= expand<E>( trefmat_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with Schur product assignment with evaluated Matrix (OP*s, runtime)
      {
         test_  = "Scaled expansion operation with Schur product assignment with evaluated Matrix (OP*s, runtime)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= expand( eval( mat_ ), E ) / scalar;
//             odres_  %= expand( eval( mat_ ), E ) / scalar;
//             sres_   %= expand( eval( mat_ ), E ) / scalar;
//             osres_  %= expand( eval( mat_ ), E ) / scalar;
            refres_ %= expand( eval( refmat_ ), E ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= expand( eval( tmat_ ), E ) / scalar;
//             todres_  %= expand( eval( tmat_ ), E ) / scalar;
//             tsres_   %= expand( eval( tmat_ ), E ) / scalar;
//             tosres_  %= expand( eval( tmat_ ), E ) / scalar;
//             trefres_ %= expand( eval( trefmat_ ), E ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Scaled expansion operation with Schur product assignment with evaluated Matrix (OP*s, compile time)
      {
         test_  = "Scaled expansion operation with Schur product assignment with evaluated Matrix (OP*s, compile time)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            dres_   %= expand<E>( eval( mat_ ) ) / scalar;
//             odres_  %= expand<E>( eval( mat_ ) ) / scalar;
//             sres_   %= expand<E>( eval( mat_ ) ) / scalar;
//             osres_  %= expand<E>( eval( mat_ ) ) / scalar;
            refres_ %= expand<E>( eval( refmat_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             tdres_   %= expand<E>( eval( tmat_ ) ) / scalar;
//             todres_  %= expand<E>( eval( tmat_ ) ) / scalar;
//             tsres_   %= expand<E>( eval( tmat_ ) ) / scalar;
//             tosres_  %= expand<E>( eval( tmat_ ) ) / scalar;
//             trefres_ %= expand<E>( eval( trefmat_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the transpose dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the transpose Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// expansion or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testTransOperation()
{
// #if BLAZETEST_MATHTEST_TEST_TRANS_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_TRANS_OPERATION > 1 )
//    {
//       using blaze::expand;
//
//
//       //=====================================================================================
//       // Transpose expansion operation
//       //=====================================================================================
//
//       // Transpose expansion operation with the given Matrix (runtime)
//       {
//          test_  = "Transpose expansion operation with the given Matrix (runtime)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initTransposeResults();
//             tdres_   = trans( expand( mat_, E ) );
//             todres_  = trans( expand( mat_, E ) );
//             tsres_   = trans( expand( mat_, E ) );
//             tosres_  = trans( expand( mat_, E ) );
//             trefres_ = trans( expand( refmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   = trans( expand( tmat_, E ) );
//             odres_  = trans( expand( tmat_, E ) );
//             sres_   = trans( expand( tmat_, E ) );
//             osres_  = trans( expand( tmat_, E ) );
//             refres_ = trans( expand( trefmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Transpose expansion operation with the given Matrix (compile time)
//       {
//          test_  = "Transpose expansion operation with the given Matrix (compile time)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initTransposeResults();
//             tdres_   = trans( expand<E>( mat_ ) );
//             todres_  = trans( expand<E>( mat_ ) );
//             tsres_   = trans( expand<E>( mat_ ) );
//             tosres_  = trans( expand<E>( mat_ ) );
//             trefres_ = trans( expand<E>( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   = trans( expand<E>( tmat_ ) );
//             odres_  = trans( expand<E>( tmat_ ) );
//             sres_   = trans( expand<E>( tmat_ ) );
//             osres_  = trans( expand<E>( tmat_ ) );
//             refres_ = trans( expand<E>( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Transpose expansion operation with evaluated Matrix (runtime)
//       {
//          test_  = "Transpose expansion operation with evaluated Matrix (runtime)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initTransposeResults();
//             tdres_   = trans( expand( eval( mat_ ), E ) );
//             todres_  = trans( expand( eval( mat_ ), E ) );
//             tsres_   = trans( expand( eval( mat_ ), E ) );
//             tosres_  = trans( expand( eval( mat_ ), E ) );
//             trefres_ = trans( expand( eval( refmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   = trans( expand( eval( tmat_ ), E ) );
//             odres_  = trans( expand( eval( tmat_ ), E ) );
//             sres_   = trans( expand( eval( tmat_ ), E ) );
//             osres_  = trans( expand( eval( tmat_ ), E ) );
//             refres_ = trans( expand( eval( trefmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Transpose expansion operation with evaluated Matrix (compile time)
//       {
//          test_  = "Transpose expansion operation with evaluated Matrix (compile time)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initTransposeResults();
//             tdres_   = trans( expand<E>( eval( mat_ ) ) );
//             todres_  = trans( expand<E>( eval( mat_ ) ) );
//             tsres_   = trans( expand<E>( eval( mat_ ) ) );
//             tosres_  = trans( expand<E>( eval( mat_ ) ) );
//             trefres_ = trans( expand<E>( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   = trans( expand<E>( eval( tmat_ ) ) );
//             odres_  = trans( expand<E>( eval( tmat_ ) ) );
//             sres_   = trans( expand<E>( eval( tmat_ ) ) );
//             osres_  = trans( expand<E>( eval( tmat_ ) ) );
//             refres_ = trans( expand<E>( eval( trefmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Transpose expansion with addition assignment
//       //=====================================================================================
//
//       // Transpose expansion with addition assignment with the given Matrix (runtime)
//       {
//          test_  = "Transpose expansion with addition assignment with the given Matrix (runtime)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   += trans( expand( mat_, E ) );
//             todres_  += trans( expand( mat_, E ) );
//             tsres_   += trans( expand( mat_, E ) );
//             tosres_  += trans( expand( mat_, E ) );
//             trefres_ += trans( expand( refmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   += trans( expand( tmat_, E ) );
//             odres_  += trans( expand( tmat_, E ) );
//             sres_   += trans( expand( tmat_, E ) );
//             osres_  += trans( expand( tmat_, E ) );
//             refres_ += trans( expand( trefmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Transpose expansion with addition assignment with the given Matrix (compile time)
//       {
//          test_  = "Transpose expansion with addition assignment with the given Matrix (compile time)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   += trans( expand<E>( mat_ ) );
//             todres_  += trans( expand<E>( mat_ ) );
//             tsres_   += trans( expand<E>( mat_ ) );
//             tosres_  += trans( expand<E>( mat_ ) );
//             trefres_ += trans( expand<E>( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   += trans( expand<E>( tmat_ ) );
//             odres_  += trans( expand<E>( tmat_ ) );
//             sres_   += trans( expand<E>( tmat_ ) );
//             osres_  += trans( expand<E>( tmat_ ) );
//             refres_ += trans( expand<E>( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Transpose expansion with addition assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Transpose expansion with addition assignment with evaluated Matrix (runtime)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   += trans( expand( eval( mat_ ), E ) );
//             todres_  += trans( expand( eval( mat_ ), E ) );
//             tsres_   += trans( expand( eval( mat_ ), E ) );
//             tosres_  += trans( expand( eval( mat_ ), E ) );
//             trefres_ += trans( expand( eval( refmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   += trans( expand( eval( tmat_ ), E ) );
//             odres_  += trans( expand( eval( tmat_ ), E ) );
//             sres_   += trans( expand( eval( tmat_ ), E ) );
//             osres_  += trans( expand( eval( tmat_ ), E ) );
//             refres_ += trans( expand( eval( trefmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Transpose expansion with addition assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Transpose expansion with addition assignment with evaluated Matrix (compile time)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   += trans( expand<E>( eval( mat_ ) ) );
//             todres_  += trans( expand<E>( eval( mat_ ) ) );
//             tsres_   += trans( expand<E>( eval( mat_ ) ) );
//             tosres_  += trans( expand<E>( eval( mat_ ) ) );
//             trefres_ += trans( expand<E>( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   += trans( expand<E>( eval( tmat_ ) ) );
//             odres_  += trans( expand<E>( eval( tmat_ ) ) );
//             sres_   += trans( expand<E>( eval( tmat_ ) ) );
//             osres_  += trans( expand<E>( eval( tmat_ ) ) );
//             refres_ += trans( expand<E>( eval( trefmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Transpose expansion with subtraction assignment
//       //=====================================================================================
//
//       // Transpose expansion with subtraction assignment with the given Matrix (runtime)
//       {
//          test_  = "Transpose expansion with subtraction assignment with the given Matrix (runtime)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   -= trans( expand( mat_, E ) );
//             todres_  -= trans( expand( mat_, E ) );
//             tsres_   -= trans( expand( mat_, E ) );
//             tosres_  -= trans( expand( mat_, E ) );
//             trefres_ -= trans( expand( refmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   -= trans( expand( tmat_, E ) );
//             odres_  -= trans( expand( tmat_, E ) );
//             sres_   -= trans( expand( tmat_, E ) );
//             osres_  -= trans( expand( tmat_, E ) );
//             refres_ -= trans( expand( trefmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Transpose expansion with subtraction assignment with the given Matrix (compile time)
//       {
//          test_  = "Transpose expansion with subtraction assignment with the given Matrix (compile time)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   -= trans( expand<E>( mat_ ) );
//             todres_  -= trans( expand<E>( mat_ ) );
//             tsres_   -= trans( expand<E>( mat_ ) );
//             tosres_  -= trans( expand<E>( mat_ ) );
//             trefres_ -= trans( expand<E>( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   -= trans( expand<E>( tmat_ ) );
//             odres_  -= trans( expand<E>( tmat_ ) );
//             sres_   -= trans( expand<E>( tmat_ ) );
//             osres_  -= trans( expand<E>( tmat_ ) );
//             refres_ -= trans( expand<E>( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Transpose expansion with subtraction assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Transpose expansion with subtraction assignment with evaluated Matrix (runtime)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   -= trans( expand( eval( mat_ ), E ) );
//             todres_  -= trans( expand( eval( mat_ ), E ) );
//             tsres_   -= trans( expand( eval( mat_ ), E ) );
//             tosres_  -= trans( expand( eval( mat_ ), E ) );
//             trefres_ -= trans( expand( eval( refmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   -= trans( expand( eval( tmat_ ), E ) );
//             odres_  -= trans( expand( eval( tmat_ ), E ) );
//             sres_   -= trans( expand( eval( tmat_ ), E ) );
//             osres_  -= trans( expand( eval( tmat_ ), E ) );
//             refres_ -= trans( expand( eval( trefmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Transpose expansion with subtraction assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Transpose expansion with subtraction assignment with evaluated Matrix (compile time)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   -= trans( expand<E>( eval( mat_ ) ) );
//             todres_  -= trans( expand<E>( eval( mat_ ) ) );
//             tsres_   -= trans( expand<E>( eval( mat_ ) ) );
//             tosres_  -= trans( expand<E>( eval( mat_ ) ) );
//             trefres_ -= trans( expand<E>( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   -= trans( expand<E>( eval( tmat_ ) ) );
//             odres_  -= trans( expand<E>( eval( tmat_ ) ) );
//             sres_   -= trans( expand<E>( eval( tmat_ ) ) );
//             osres_  -= trans( expand<E>( eval( tmat_ ) ) );
//             refres_ -= trans( expand<E>( eval( trefmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Transpose expansion with Schur product assignment
//       //=====================================================================================
//
//       // Transpose expansion with Schur product assignment with the given Matrix (runtime)
//       {
//          test_  = "Transpose expansion with Schur product assignment with the given Matrix (runtime)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   %= trans( expand( mat_, E ) );
//             todres_  %= trans( expand( mat_, E ) );
//             tsres_   %= trans( expand( mat_, E ) );
//             tosres_  %= trans( expand( mat_, E ) );
//             trefres_ %= trans( expand( refmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   %= trans( expand( tmat_, E ) );
//             odres_  %= trans( expand( tmat_, E ) );
//             sres_   %= trans( expand( tmat_, E ) );
//             osres_  %= trans( expand( tmat_, E ) );
//             refres_ %= trans( expand( trefmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Transpose expansion with Schur product assignment with the given Matrix (compile time)
//       {
//          test_  = "Transpose expansion with Schur product assignment with the given Matrix (compile time)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   %= trans( expand<E>( mat_ ) );
//             todres_  %= trans( expand<E>( mat_ ) );
//             tsres_   %= trans( expand<E>( mat_ ) );
//             tosres_  %= trans( expand<E>( mat_ ) );
//             trefres_ %= trans( expand<E>( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   %= trans( expand<E>( tmat_ ) );
//             odres_  %= trans( expand<E>( tmat_ ) );
//             sres_   %= trans( expand<E>( tmat_ ) );
//             osres_  %= trans( expand<E>( tmat_ ) );
//             refres_ %= trans( expand<E>( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Transpose expansion with Schur product assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Transpose expansion with Schur product assignment with evaluated Matrix (runtime)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   %= trans( expand( eval( mat_ ), E ) );
//             todres_  %= trans( expand( eval( mat_ ), E ) );
//             tsres_   %= trans( expand( eval( mat_ ), E ) );
//             tosres_  %= trans( expand( eval( mat_ ), E ) );
//             trefres_ %= trans( expand( eval( refmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   %= trans( expand( eval( tmat_ ), E ) );
//             odres_  %= trans( expand( eval( tmat_ ), E ) );
//             sres_   %= trans( expand( eval( tmat_ ), E ) );
//             osres_  %= trans( expand( eval( tmat_ ), E ) );
//             refres_ %= trans( expand( eval( trefmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Transpose expansion with Schur product assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Transpose expansion with Schur product assignment with evaluated Matrix (compile time)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   %= trans( expand<E>( eval( mat_ ) ) );
//             todres_  %= trans( expand<E>( eval( mat_ ) ) );
//             tsres_   %= trans( expand<E>( eval( mat_ ) ) );
//             tosres_  %= trans( expand<E>( eval( mat_ ) ) );
//             trefres_ %= trans( expand<E>( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   %= trans( expand<E>( eval( tmat_ ) ) );
//             odres_  %= trans( expand<E>( eval( tmat_ ) ) );
//             sres_   %= trans( expand<E>( eval( tmat_ ) ) );
//             osres_  %= trans( expand<E>( eval( tmat_ ) ) );
//             refres_ %= trans( expand<E>( eval( trefmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//    }
// #endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the conjugate transpose dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the conjugate transpose Matrix expansion with plain assignment, addition
// assignment, subtraction assignment, and Schur product assignment. In case any error resulting
// from the expansion or the subsequent assignment is detected, a \a std::runtime_error exception
// is thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testCTransOperation()
{
// #if BLAZETEST_MATHTEST_TEST_CTRANS_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_CTRANS_OPERATION > 1 )
//    {
//       using blaze::expand;
//
//
//       //=====================================================================================
//       // Conjugate transpose expansion operation
//       //=====================================================================================
//
//       // Conjugate transpose expansion operation with the given Matrix (runtime)
//       {
//          test_  = "Conjugate transpose expansion operation with the given Matrix (runtime)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initTransposeResults();
//             tdres_   = ctrans( expand( mat_, E ) );
//             todres_  = ctrans( expand( mat_, E ) );
//             tsres_   = ctrans( expand( mat_, E ) );
//             tosres_  = ctrans( expand( mat_, E ) );
//             trefres_ = ctrans( expand( refmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   = ctrans( expand( tmat_, E ) );
//             odres_  = ctrans( expand( tmat_, E ) );
//             sres_   = ctrans( expand( tmat_, E ) );
//             osres_  = ctrans( expand( tmat_, E ) );
//             refres_ = ctrans( expand( trefmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Conjugate transpose expansion operation with the given Matrix (compile time)
//       {
//          test_  = "Conjugate transpose expansion operation with the given Matrix (compile time)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initTransposeResults();
//             tdres_   = ctrans( expand<E>( mat_ ) );
//             todres_  = ctrans( expand<E>( mat_ ) );
//             tsres_   = ctrans( expand<E>( mat_ ) );
//             tosres_  = ctrans( expand<E>( mat_ ) );
//             trefres_ = ctrans( expand<E>( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   = ctrans( expand<E>( tmat_ ) );
//             odres_  = ctrans( expand<E>( tmat_ ) );
//             sres_   = ctrans( expand<E>( tmat_ ) );
//             osres_  = ctrans( expand<E>( tmat_ ) );
//             refres_ = ctrans( expand<E>( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Conjugate transpose expansion operation with evaluated Matrix (runtime)
//       {
//          test_  = "Conjugate transpose expansion operation with evaluated Matrix (runtime)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initTransposeResults();
//             tdres_   = ctrans( expand( eval( mat_ ), E ) );
//             todres_  = ctrans( expand( eval( mat_ ), E ) );
//             tsres_   = ctrans( expand( eval( mat_ ), E ) );
//             tosres_  = ctrans( expand( eval( mat_ ), E ) );
//             trefres_ = ctrans( expand( eval( refmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   = ctrans( expand( eval( tmat_ ), E ) );
//             odres_  = ctrans( expand( eval( tmat_ ), E ) );
//             sres_   = ctrans( expand( eval( tmat_ ), E ) );
//             osres_  = ctrans( expand( eval( tmat_ ), E ) );
//             refres_ = ctrans( expand( eval( trefmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Conjugate transpose expansion operation with evaluated Matrix (compile time)
//       {
//          test_  = "Conjugate transpose expansion operation with evaluated Matrix (compile time)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initTransposeResults();
//             tdres_   = ctrans( expand<E>( eval( mat_ ) ) );
//             todres_  = ctrans( expand<E>( eval( mat_ ) ) );
//             tsres_   = ctrans( expand<E>( eval( mat_ ) ) );
//             tosres_  = ctrans( expand<E>( eval( mat_ ) ) );
//             trefres_ = ctrans( expand<E>( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   = ctrans( expand<E>( eval( tmat_ ) ) );
//             odres_  = ctrans( expand<E>( eval( tmat_ ) ) );
//             sres_   = ctrans( expand<E>( eval( tmat_ ) ) );
//             osres_  = ctrans( expand<E>( eval( tmat_ ) ) );
//             refres_ = ctrans( expand<E>( eval( trefmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Conjugate transpose expansion with addition assignment
//       //=====================================================================================
//
//       // Conjugate transpose expansion with addition assignment with the given Matrix (runtime)
//       {
//          test_  = "Conjugate transpose expansion with addition assignment with the given Matrix (runtime)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   += ctrans( expand( mat_, E ) );
//             todres_  += ctrans( expand( mat_, E ) );
//             tsres_   += ctrans( expand( mat_, E ) );
//             tosres_  += ctrans( expand( mat_, E ) );
//             trefres_ += ctrans( expand( refmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   += ctrans( expand( tmat_, E ) );
//             odres_  += ctrans( expand( tmat_, E ) );
//             sres_   += ctrans( expand( tmat_, E ) );
//             osres_  += ctrans( expand( tmat_, E ) );
//             refres_ += ctrans( expand( trefmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Conjugate transpose expansion with addition assignment with the given Matrix (compile time)
//       {
//          test_  = "Conjugate transpose expansion with addition assignment with the given Matrix (compile time)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   += ctrans( expand<E>( mat_ ) );
//             todres_  += ctrans( expand<E>( mat_ ) );
//             tsres_   += ctrans( expand<E>( mat_ ) );
//             tosres_  += ctrans( expand<E>( mat_ ) );
//             trefres_ += ctrans( expand<E>( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   += ctrans( expand<E>( tmat_ ) );
//             odres_  += ctrans( expand<E>( tmat_ ) );
//             sres_   += ctrans( expand<E>( tmat_ ) );
//             osres_  += ctrans( expand<E>( tmat_ ) );
//             refres_ += ctrans( expand<E>( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Conjugate transpose expansion with addition assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Conjugate transpose expansion with addition assignment with evaluated Matrix (runtime)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   += ctrans( expand( eval( mat_ ), E ) );
//             todres_  += ctrans( expand( eval( mat_ ), E ) );
//             tsres_   += ctrans( expand( eval( mat_ ), E ) );
//             tosres_  += ctrans( expand( eval( mat_ ), E ) );
//             trefres_ += ctrans( expand( eval( refmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   += ctrans( expand( eval( tmat_ ), E ) );
//             odres_  += ctrans( expand( eval( tmat_ ), E ) );
//             sres_   += ctrans( expand( eval( tmat_ ), E ) );
//             osres_  += ctrans( expand( eval( tmat_ ), E ) );
//             refres_ += ctrans( expand( eval( trefmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Conjugate transpose expansion with addition assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Conjugate transpose expansion with addition assignment with evaluated Matrix (compile time)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   += ctrans( expand<E>( eval( mat_ ) ) );
//             todres_  += ctrans( expand<E>( eval( mat_ ) ) );
//             tsres_   += ctrans( expand<E>( eval( mat_ ) ) );
//             tosres_  += ctrans( expand<E>( eval( mat_ ) ) );
//             trefres_ += ctrans( expand<E>( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   += ctrans( expand<E>( eval( tmat_ ) ) );
//             odres_  += ctrans( expand<E>( eval( tmat_ ) ) );
//             sres_   += ctrans( expand<E>( eval( tmat_ ) ) );
//             osres_  += ctrans( expand<E>( eval( tmat_ ) ) );
//             refres_ += ctrans( expand<E>( eval( trefmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Conjugate transpose expansion with subtraction assignment
//       //=====================================================================================
//
//       // Conjugate transpose expansion with subtraction assignment with the given Matrix (runtime)
//       {
//          test_  = "Conjugate transpose expansion with subtraction assignment with the given Matrix (runtime)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   -= ctrans( expand( mat_, E ) );
//             todres_  -= ctrans( expand( mat_, E ) );
//             tsres_   -= ctrans( expand( mat_, E ) );
//             tosres_  -= ctrans( expand( mat_, E ) );
//             trefres_ -= ctrans( expand( refmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   -= ctrans( expand( tmat_, E ) );
//             odres_  -= ctrans( expand( tmat_, E ) );
//             sres_   -= ctrans( expand( tmat_, E ) );
//             osres_  -= ctrans( expand( tmat_, E ) );
//             refres_ -= ctrans( expand( trefmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Conjugate transpose expansion with subtraction assignment with the given Matrix (compile time)
//       {
//          test_  = "Conjugate transpose expansion with subtraction assignment with the given Matrix (compile time)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   -= ctrans( expand<E>( mat_ ) );
//             todres_  -= ctrans( expand<E>( mat_ ) );
//             tsres_   -= ctrans( expand<E>( mat_ ) );
//             tosres_  -= ctrans( expand<E>( mat_ ) );
//             trefres_ -= ctrans( expand<E>( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   -= ctrans( expand<E>( tmat_ ) );
//             odres_  -= ctrans( expand<E>( tmat_ ) );
//             sres_   -= ctrans( expand<E>( tmat_ ) );
//             osres_  -= ctrans( expand<E>( tmat_ ) );
//             refres_ -= ctrans( expand<E>( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Conjugate transpose expansion with subtraction assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Conjugate transpose expansion with subtraction assignment with evaluated Matrix (runtime)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   -= ctrans( expand( eval( mat_ ), E ) );
//             todres_  -= ctrans( expand( eval( mat_ ), E ) );
//             tsres_   -= ctrans( expand( eval( mat_ ), E ) );
//             tosres_  -= ctrans( expand( eval( mat_ ), E ) );
//             trefres_ -= ctrans( expand( eval( refmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   -= ctrans( expand( eval( tmat_ ), E ) );
//             odres_  -= ctrans( expand( eval( tmat_ ), E ) );
//             sres_   -= ctrans( expand( eval( tmat_ ), E ) );
//             osres_  -= ctrans( expand( eval( tmat_ ), E ) );
//             refres_ -= ctrans( expand( eval( trefmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Conjugate transpose expansion with subtraction assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Conjugate transpose expansion with subtraction assignment with evaluated Matrix (compile time)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   -= ctrans( expand<E>( eval( mat_ ) ) );
//             todres_  -= ctrans( expand<E>( eval( mat_ ) ) );
//             tsres_   -= ctrans( expand<E>( eval( mat_ ) ) );
//             tosres_  -= ctrans( expand<E>( eval( mat_ ) ) );
//             trefres_ -= ctrans( expand<E>( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   -= ctrans( expand<E>( eval( tmat_ ) ) );
//             odres_  -= ctrans( expand<E>( eval( tmat_ ) ) );
//             sres_   -= ctrans( expand<E>( eval( tmat_ ) ) );
//             osres_  -= ctrans( expand<E>( eval( tmat_ ) ) );
//             refres_ -= ctrans( expand<E>( eval( trefmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Conjugate transpose expansion with Schur product assignment
//       //=====================================================================================
//
//       // Conjugate transpose expansion with Schur product assignment with the given Matrix (runtime)
//       {
//          test_  = "Conjugate transpose expansion with Schur product assignment with the given Matrix (runtime)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   %= ctrans( expand( mat_, E ) );
//             todres_  %= ctrans( expand( mat_, E ) );
//             tsres_   %= ctrans( expand( mat_, E ) );
//             tosres_  %= ctrans( expand( mat_, E ) );
//             trefres_ %= ctrans( expand( refmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   %= ctrans( expand( tmat_, E ) );
//             odres_  %= ctrans( expand( tmat_, E ) );
//             sres_   %= ctrans( expand( tmat_, E ) );
//             osres_  %= ctrans( expand( tmat_, E ) );
//             refres_ %= ctrans( expand( trefmat_, E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Conjugate transpose expansion with Schur product assignment with the given Matrix (compile time)
//       {
//          test_  = "Conjugate transpose expansion with Schur product assignment with the given Matrix (compile time)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   %= ctrans( expand<E>( mat_ ) );
//             todres_  %= ctrans( expand<E>( mat_ ) );
//             tsres_   %= ctrans( expand<E>( mat_ ) );
//             tosres_  %= ctrans( expand<E>( mat_ ) );
//             trefres_ %= ctrans( expand<E>( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   %= ctrans( expand<E>( tmat_ ) );
//             odres_  %= ctrans( expand<E>( tmat_ ) );
//             sres_   %= ctrans( expand<E>( tmat_ ) );
//             osres_  %= ctrans( expand<E>( tmat_ ) );
//             refres_ %= ctrans( expand<E>( trefmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Conjugate transpose expansion with Schur product assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Conjugate transpose expansion with Schur product assignment with evaluated Matrix (runtime)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   %= ctrans( expand( eval( mat_ ), E ) );
//             todres_  %= ctrans( expand( eval( mat_ ), E ) );
//             tsres_   %= ctrans( expand( eval( mat_ ), E ) );
//             tosres_  %= ctrans( expand( eval( mat_ ), E ) );
//             trefres_ %= ctrans( expand( eval( refmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   %= ctrans( expand( eval( tmat_ ), E ) );
//             odres_  %= ctrans( expand( eval( tmat_ ), E ) );
//             sres_   %= ctrans( expand( eval( tmat_ ), E ) );
//             osres_  %= ctrans( expand( eval( tmat_ ), E ) );
//             refres_ %= ctrans( expand( eval( trefmat_ ), E ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//
//       // Conjugate transpose expansion with Schur product assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Conjugate transpose expansion with Schur product assignment with evaluated Matrix (compile time)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initTransposeResults();
//             tdres_   %= ctrans( expand<E>( eval( mat_ ) ) );
//             todres_  %= ctrans( expand<E>( eval( mat_ ) ) );
//             tsres_   %= ctrans( expand<E>( eval( mat_ ) ) );
//             tosres_  %= ctrans( expand<E>( eval( mat_ ) ) );
//             trefres_ %= ctrans( expand<E>( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkTransposeResults<MT>();
//
//          try {
//             initResults();
//             dres_   %= ctrans( expand<E>( eval( tmat_ ) ) );
//             odres_  %= ctrans( expand<E>( eval( tmat_ ) ) );
//             sres_   %= ctrans( expand<E>( eval( tmat_ ) ) );
//             osres_  %= ctrans( expand<E>( eval( tmat_ ) ) );
//             refres_ %= ctrans( expand<E>( eval( trefmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkResults<TMT>();
//       }
//    }
// #endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the abs dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the abs Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// expansion or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testAbsOperation()
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
/*!\brief Testing the conjugate dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the conjugate Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// expansion or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testConjOperation()
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
/*!\brief Testing the \a real dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the \a real Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// expansion or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testRealOperation()
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
/*!\brief Testing the \a imag dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the \a imag Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// expansion or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testImagOperation()
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
/*!\brief Testing the evaluated dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the evalated Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// expansion or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testEvalOperation()
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
/*!\brief Testing the serialized dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the serialized Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// expansion or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testSerialOperation()
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
/*!\brief Testing the subtensor-wise dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the subtensor-wise Matrix expansion with plain assignment, addition
// assignment, subtraction assignment, and Schur product assignment. In case any error resulting
// from the addition or the subsequent assignment is detected, a \a std::runtime_error exception
// is thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testSubtensorOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_SUBTENSOR_OPERATION
   if( BLAZETEST_MATHTEST_TEST_SUBTENSOR_OPERATION > 1 )
   {
      using blaze::expand;

      if( mat_.rows() == 0UL || mat_.columns() == 0UL || E == 0UL )
         return;


      //=====================================================================================
      // Subtensor-wise expansion
      //=====================================================================================

      // Subtensor-wise expansion with the given Matrix (runtime)
      {
         test_  = "Subtensor-wise expansion with the given Matrix (runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) = subtensor( expand( mat_, E )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) = subtensor( expand( mat_, E )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) = subtensor( expand( mat_, E )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) = subtensor( expand( mat_, E )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) = subtensor( expand( refmat_, E ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) = subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) = subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) = subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) = subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) = subtensor( expand( trefmat_, E ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Subtensor-wise expansion with the given Matrix (compile time)
      {
         test_  = "Subtensor-wise expansion with the given Matrix (compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) = subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
//                   subtensor( odres_ , page, row, column, o, m, n ) = subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
//                   subtensor( sres_  , page, row, column, o, m, n ) = subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
//                   subtensor( osres_ , page, row, column, o, m, n ) = subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) = subtensor( expand<E>( refmat_ ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) = subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) = subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) = subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) = subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) = subtensor( expand<E>( trefmat_ ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Subtensor-wise expansion with evaluated Matrix (runtime)
      {
         test_  = "Subtensor-wise expansion with evaluated Matrix (runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) = subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) = subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) = subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) = subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) = subtensor( expand( eval( refmat_ ), E ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) = subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) = subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) = subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) = subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) = subtensor( expand( eval( trefmat_ ), E ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Subtensor-wise expansion with evaluated Matrix (compile time)
      {
         test_  = "Subtensor-wise expansion with evaluated Matrix (compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) = subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) = subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) = subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) = subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) = subtensor( expand<E>( eval( refmat_ ) ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) = subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) = subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) = subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) = subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) = subtensor( expand<E>( eval( trefmat_ ) ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Subtensor-wise expansion with addition assignment
      //=====================================================================================

      // Subtensor-wise expansion with addition assignment with the given Matrix (runtime)
      {
         test_  = "Subtensor-wise expansion with addition assignment with the given Matrix (runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) += subtensor( expand( mat_, E )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) += subtensor( expand( mat_, E )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) += subtensor( expand( mat_, E )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) += subtensor( expand( mat_, E )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) += subtensor( expand( refmat_, E ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) += subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) += subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) += subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) += subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) += subtensor( expand( trefmat_, E ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Subtensor-wise expansion with addition assignment with the given Matrix (compile time)
      {
         test_  = "Subtensor-wise expansion with addition assignment with the given Matrix (compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) += subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) += subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) += subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) += subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) += subtensor( expand<E>( refmat_ ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) += subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) += subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) += subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) += subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) += subtensor( expand<E>( trefmat_ ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Subtensor-wise expansion with addition assignment with evaluated Matrix (runtime)
      {
         test_  = "Subtensor-wise expansion with addition assignment with evaluated Matrix (runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) += subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) += subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) += subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) += subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) += subtensor( expand( eval( refmat_ ), E ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) += subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) += subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) += subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) += subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) += subtensor( expand( eval( trefmat_ ), E ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Subtensor-wise expansion with addition assignment with evaluated Matrix (compile time)
      {
         test_  = "Subtensor-wise expansion with addition assignment with evaluated Matrix (compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) += subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) += subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) += subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) += subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) += subtensor( expand<E>( eval( refmat_ ) ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) += subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) += subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) += subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) += subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) += subtensor( expand<E>( eval( trefmat_ ) ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Subtensor-wise expansion with subtraction assignment
      //=====================================================================================

      // Subtensor-wise expansion with subtraction assignment with the given Matrix (runtime)
      {
         test_  = "Subtensor-wise expansion with subtraction assignment with the given Matrix (runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( expand( mat_, E )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) -= subtensor( expand( mat_, E )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) -= subtensor( expand( mat_, E )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) -= subtensor( expand( mat_, E )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) -= subtensor( expand( refmat_, E ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) -= subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) -= subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) -= subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) -= subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) -= subtensor( expand( trefmat_, E ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Subtensor-wise expansion with subtraction assignment with the given Matrix (compile time)
      {
         test_  = "Subtensor-wise expansion with subtraction assignment with the given Matrix (compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) -= subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) -= subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) -= subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) -= subtensor( expand<E>( refmat_ ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) -= subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) -= subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) -= subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) -= subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) -= subtensor( expand<E>( trefmat_ ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Subtensor-wise expansion with subtraction assignment with evaluated Matrix (runtime)
      {
         test_  = "Subtensor-wise expansion with subtraction assignment with evaluated Matrix (runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) -= subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) -= subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) -= subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) -= subtensor( expand( eval( refmat_ ), E ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) -= subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) -= subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) -= subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) -= subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) -= subtensor( expand( eval( trefmat_ ), E ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Subtensor-wise expansion with subtraction assignment with evaluated Matrix (compile time)
      {
         test_  = "Subtensor-wise expansion with subtraction assignment with evaluated Matrix (compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) -= subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) -= subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) -= subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) -= subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) -= subtensor( expand<E>( eval( refmat_ ) ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) -= subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) -= subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) -= subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) -= subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) -= subtensor( expand<E>( eval( trefmat_ ) ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // Subtensor-wise expansion with Schur product assignment
      //=====================================================================================

      // Subtensor-wise expansion with Schur product assignment with the given Matrix (runtime)
      {
         test_  = "Subtensor-wise expansion with Schur product assignment with the given Matrix (runtime)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( expand( mat_, E )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) %= subtensor( expand( mat_, E )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) %= subtensor( expand( mat_, E )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) %= subtensor( expand( mat_, E )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) %= subtensor( expand( refmat_, E ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) %= subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) %= subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) %= subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) %= subtensor( expand( tmat_, E )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) %= subtensor( expand( trefmat_, E ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Subtensor-wise expansion with Schur product assignment with the given Matrix (compile time)
      {
         test_  = "Subtensor-wise expansion with Schur product assignment with the given Matrix (compile time)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) %= subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) %= subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) %= subtensor( expand<E>( mat_ )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) %= subtensor( expand<E>( refmat_ ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) %= subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) %= subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) %= subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) %= subtensor( expand<E>( tmat_ )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) %= subtensor( expand<E>( trefmat_ ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Subtensor-wise expansion with Schur product assignment with evaluated Matrix (runtime)
      {
         test_  = "Subtensor-wise expansion with Schur product assignment with evaluated Matrix (runtime)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                        n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) %= subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) %= subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) %= subtensor( expand( eval( mat_ ), E )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) %= subtensor( expand( eval( refmat_ ), E ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) %= subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) %= subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) %= subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) %= subtensor( expand( eval( tmat_ ), E )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) %= subtensor( expand( eval( trefmat_ ), E ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // Subtensor-wise expansion with Schur product assignment with evaluated Matrix (compile time)
      {
         test_  = "Subtensor-wise expansion with Schur product assignment with evaluated Matrix (compile time)";
         error_ = "Failed Schur product assignment";

         try {
            initResults();
            for( size_t page=0UL, o=0UL; page<E; page+=o ) {
               o = blaze::rand<size_t>( 1UL, E - page );
               for( size_t row=0UL, m=0UL; row<mat_.rows(); row+=m ) {
                  m = blaze::rand<size_t>( 1UL, mat_.rows() - row );
                  for( size_t column=0UL, n=0UL; column<mat_.columns(); column+=n ) {
                     n = blaze::rand<size_t>( 1UL, mat_.columns() - column );
                     subtensor( dres_  , page, row, column, o, m, n ) %= subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
//                      subtensor( odres_ , page, row, column, o, m, n ) %= subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
//                      subtensor( sres_  , page, row, column, o, m, n ) %= subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
//                      subtensor( osres_ , page, row, column, o, m, n ) %= subtensor( expand<E>( eval( mat_ ) )   , page, row, column, o, m, n );
                     subtensor( refres_, page, row, column, o, m, n ) %= subtensor( expand<E>( eval( refmat_ ) ), page, row, column, o, m, n );
                  }
               }
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t row=0UL, m=0UL; row<E; row+=m ) {
//                m = blaze::rand<size_t>( 1UL, E - row );
//                for( size_t column=0UL, n=0UL; column<tmat_.size(); column+=n ) {
//                   n = blaze::rand<size_t>( 1UL, tmat_.size() - column );
//                   subtensor( tdres_  , page, row, column, o, m, n ) %= subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( todres_ , page, row, column, o, m, n ) %= subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( tsres_  , page, row, column, o, m, n ) %= subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( tosres_ , page, row, column, o, m, n ) %= subtensor( expand<E>( eval( tmat_ ) )   , page, row, column, o, m, n );
//                   subtensor( trefres_, page, row, column, o, m, n ) %= subtensor( expand<E>( eval( trefmat_ ) ), page, row, column, o, m, n );
//                }
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the subtensor-wise dense Matrix expansion operation.
//
// \return void
//
// This function is called in case the subtensor-wise dense Matrix expansion operation is not
// available for the given Matrix type \a MT.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testSubtensorOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the row-wise dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the row-wise Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testRowSliceOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_ROWSLICE_OPERATION
   if( BLAZETEST_MATHTEST_TEST_ROWSLICE_OPERATION > 1 )
   {
      using blaze::expand;

      if( mat_.rows() == 0UL || mat_.columns() == 0UL || E == 0UL )
         return;


      //=====================================================================================
      // rowslice-wise expansion
      //=====================================================================================

      // rowslice-wise expansion with the given Matrix (runtime)
      {
         test_  = "rowslice-wise expansion with the given Matrix (runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.rows(); ++i ) {
               rowslice( dres_  , i ) = rowslice( expand( mat_, E ), i );
//                rowslice( odres_ , i ) = rowslice( expand( mat_, E ), i );
//                rowslice( sres_  , i ) = rowslice( expand( mat_, E ), i );
//                rowslice( osres_ , i ) = rowslice( expand( mat_, E ), i );
               rowslice( refres_, i ) = rowslice( expand( refmat_, E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<E; ++i ) {
//                rowslice( tdres_  , i ) = rowslice( expand( tmat_, E ), i );
//                rowslice( todres_ , i ) = rowslice( expand( tmat_, E ), i );
//                rowslice( tsres_  , i ) = rowslice( expand( tmat_, E ), i );
//                rowslice( tosres_ , i ) = rowslice( expand( tmat_, E ), i );
//                rowslice( trefres_, i ) = rowslice( expand( trefmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // rowslice-wise expansion with the given Matrix (compile time)
      {
         test_  = "rowslice-wise expansion with the given Matrix (compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.rows(); ++i ) {
               rowslice( dres_  , i ) = rowslice( expand<E>( mat_ ), i );
//                rowslice( odres_ , i ) = rowslice( expand<E>( mat_ ), i );
//                rowslice( sres_  , i ) = rowslice( expand<E>( mat_ ), i );
//                rowslice( osres_ , i ) = rowslice( expand<E>( mat_ ), i );
               rowslice( refres_, i ) = rowslice( expand<E>( refmat_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<E; ++i ) {
//                rowslice( tdres_  , i ) = rowslice( expand<E>( tmat_ ), i );
//                rowslice( todres_ , i ) = rowslice( expand<E>( tmat_ ), i );
//                rowslice( tsres_  , i ) = rowslice( expand<E>( tmat_ ), i );
//                rowslice( tosres_ , i ) = rowslice( expand<E>( tmat_ ), i );
//                rowslice( trefres_, i ) = rowslice( expand<E>( trefmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // rowslice-wise expansion with evaluated Matrix (runtime)
      {
         test_  = "rowslice-wise expansion with evaluated Matrix (runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.rows(); ++i ) {
               rowslice( dres_  , i ) = rowslice( expand( eval( mat_ ), E ), i );
//                rowslice( odres_ , i ) = rowslice( expand( eval( mat_ ), E ), i );
//                rowslice( sres_  , i ) = rowslice( expand( eval( mat_ ), E ), i );
//                rowslice( osres_ , i ) = rowslice( expand( eval( mat_ ), E ), i );
               rowslice( refres_, i ) = rowslice( expand( eval( refmat_ ), E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<E; ++i ) {
//                rowslice( tdres_  , i ) = rowslice( expand( eval( tmat_ ), E ), i );
//                rowslice( todres_ , i ) = rowslice( expand( eval( tmat_ ), E ), i );
//                rowslice( tsres_  , i ) = rowslice( expand( eval( tmat_ ), E ), i );
//                rowslice( tosres_ , i ) = rowslice( expand( eval( tmat_ ), E ), i );
//                rowslice( trefres_, i ) = rowslice( expand( eval( trefmat_ ), E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // rowslice-wise expansion with evaluated Matrix (compile time)
      {
         test_  = "rowslice-wise expansion with evaluated Matrix (compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.rows(); ++i ) {
               rowslice( dres_  , i ) = rowslice( expand<E>( eval( mat_ ) ), i );
//                rowslice( odres_ , i ) = rowslice( expand<E>( eval( mat_ ) ), i );
//                rowslice( sres_  , i ) = rowslice( expand<E>( eval( mat_ ) ), i );
//                rowslice( osres_ , i ) = rowslice( expand<E>( eval( mat_ ) ), i );
               rowslice( refres_, i ) = rowslice( expand<E>( eval( refmat_ ) ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<E; ++i ) {
//                rowslice( tdres_  , i ) = rowslice( expand<E>( eval( tmat_ ) ), i );
//                rowslice( todres_ , i ) = rowslice( expand<E>( eval( tmat_ ) ), i );
//                rowslice( tsres_  , i ) = rowslice( expand<E>( eval( tmat_ ) ), i );
//                rowslice( tosres_ , i ) = rowslice( expand<E>( eval( tmat_ ) ), i );
//                rowslice( trefres_, i ) = rowslice( expand<E>( eval( trefmat_ ) ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // rowslice-wise expansion with addition assignment
      //=====================================================================================

      // rowslice-wise expansion with addition assignment with the given Matrix (runtime)
      {
         test_  = "rowslice-wise expansion with addition assignment with the given Matrix (runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.rows(); ++i ) {
               rowslice( dres_  , i ) += rowslice( expand( mat_, E ), i );
//                rowslice( odres_ , i ) += rowslice( expand( mat_, E ), i );
//                rowslice( sres_  , i ) += rowslice( expand( mat_, E ), i );
//                rowslice( osres_ , i ) += rowslice( expand( mat_, E ), i );
               rowslice( refres_, i ) += rowslice( expand( refmat_, E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<E; ++i ) {
//                rowslice( tdres_  , i ) += rowslice( expand( tmat_, E ), i );
//                rowslice( todres_ , i ) += rowslice( expand( tmat_, E ), i );
//                rowslice( tsres_  , i ) += rowslice( expand( tmat_, E ), i );
//                rowslice( tosres_ , i ) += rowslice( expand( tmat_, E ), i );
//                rowslice( trefres_, i ) += rowslice( expand( trefmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // rowslice-wise expansion with addition assignment with the given Matrix (compile time)
      {
         test_  = "rowslice-wise expansion with addition assignment with the given Matrix (compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.rows(); ++i ) {
               rowslice( dres_  , i ) += rowslice( expand<E>( mat_ ), i );
//                rowslice( odres_ , i ) += rowslice( expand<E>( mat_ ), i );
//                rowslice( sres_  , i ) += rowslice( expand<E>( mat_ ), i );
//                rowslice( osres_ , i ) += rowslice( expand<E>( mat_ ), i );
               rowslice( refres_, i ) += rowslice( expand<E>( refmat_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<E; ++i ) {
//                rowslice( tdres_  , i ) += rowslice( expand<E>( tmat_ ), i );
//                rowslice( todres_ , i ) += rowslice( expand<E>( tmat_ ), i );
//                rowslice( tsres_  , i ) += rowslice( expand<E>( tmat_ ), i );
//                rowslice( tosres_ , i ) += rowslice( expand<E>( tmat_ ), i );
//                rowslice( trefres_, i ) += rowslice( expand<E>( trefmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // rowslice-wise expansion with addition assignment with evaluated Matrix (runtime)
      {
         test_  = "rowslice-wise expansion with addition assignment with evaluated Matrix (runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.rows(); ++i ) {
               rowslice( dres_  , i ) += rowslice( expand( eval( mat_ ), E ), i );
//                rowslice( odres_ , i ) += rowslice( expand( eval( mat_ ), E ), i );
//                rowslice( sres_  , i ) += rowslice( expand( eval( mat_ ), E ), i );
//                rowslice( osres_ , i ) += rowslice( expand( eval( mat_ ), E ), i );
               rowslice( refres_, i ) += rowslice( expand( eval( refmat_ ), E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<E; ++i ) {
//                rowslice( tdres_  , i ) += rowslice( expand( eval( tmat_ ), E ), i );
//                rowslice( todres_ , i ) += rowslice( expand( eval( tmat_ ), E ), i );
//                rowslice( tsres_  , i ) += rowslice( expand( eval( tmat_ ), E ), i );
//                rowslice( tosres_ , i ) += rowslice( expand( eval( tmat_ ), E ), i );
//                rowslice( trefres_, i ) += rowslice( expand( eval( trefmat_ ), E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // rowslice-wise expansion with addition assignment with evaluated Matrix (compile time)
      {
         test_  = "rowslice-wise expansion with addition assignment with evaluated Matrix (compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.rows(); ++i ) {
               rowslice( dres_  , i ) += rowslice( expand<E>( eval( mat_ ) ), i );
//                rowslice( odres_ , i ) += rowslice( expand<E>( eval( mat_ ) ), i );
//                rowslice( sres_  , i ) += rowslice( expand<E>( eval( mat_ ) ), i );
//                rowslice( osres_ , i ) += rowslice( expand<E>( eval( mat_ ) ), i );
               rowslice( refres_, i ) += rowslice( expand<E>( eval( refmat_ ) ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<E; ++i ) {
//                rowslice( tdres_  , i ) += rowslice( expand<E>( eval( tmat_ ) ), i );
//                rowslice( todres_ , i ) += rowslice( expand<E>( eval( tmat_ ) ), i );
//                rowslice( tsres_  , i ) += rowslice( expand<E>( eval( tmat_ ) ), i );
//                rowslice( tosres_ , i ) += rowslice( expand<E>( eval( tmat_ ) ), i );
//                rowslice( trefres_, i ) += rowslice( expand<E>( eval( trefmat_ ) ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // rowslice-wise expansion with subtraction assignment
      //=====================================================================================

      // rowslice-wise expansion with subtraction assignment with the given Matrix (runtime)
      {
         test_  = "rowslice-wise expansion with subtraction assignment with the given Matrix (runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.rows(); ++i ) {
               rowslice( dres_  , i ) -= rowslice( expand( mat_, E ), i );
//                rowslice( odres_ , i ) -= rowslice( expand( mat_, E ), i );
//                rowslice( sres_  , i ) -= rowslice( expand( mat_, E ), i );
//                rowslice( osres_ , i ) -= rowslice( expand( mat_, E ), i );
               rowslice( refres_, i ) -= rowslice( expand( refmat_, E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<E; ++i ) {
//                rowslice( tdres_  , i ) -= rowslice( expand( tmat_, E ), i );
//                rowslice( todres_ , i ) -= rowslice( expand( tmat_, E ), i );
//                rowslice( tsres_  , i ) -= rowslice( expand( tmat_, E ), i );
//                rowslice( tosres_ , i ) -= rowslice( expand( tmat_, E ), i );
//                rowslice( trefres_, i ) -= rowslice( expand( trefmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // rowslice-wise expansion with subtraction assignment with the given Matrix (compile time)
      {
         test_  = "rowslice-wise expansion with subtraction assignment with the given Matrix (compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.rows(); ++i ) {
               rowslice( dres_  , i ) -= rowslice( expand<E>( mat_ ), i );
//                rowslice( odres_ , i ) -= rowslice( expand<E>( mat_ ), i );
//                rowslice( sres_  , i ) -= rowslice( expand<E>( mat_ ), i );
//                rowslice( osres_ , i ) -= rowslice( expand<E>( mat_ ), i );
               rowslice( refres_, i ) -= rowslice( expand<E>( refmat_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<E; ++i ) {
//                rowslice( tdres_  , i ) -= rowslice( expand<E>( tmat_ ), i );
//                rowslice( todres_ , i ) -= rowslice( expand<E>( tmat_ ), i );
//                rowslice( tsres_  , i ) -= rowslice( expand<E>( tmat_ ), i );
//                rowslice( tosres_ , i ) -= rowslice( expand<E>( tmat_ ), i );
//                rowslice( trefres_, i ) -= rowslice( expand<E>( trefmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // rowslice-wise expansion with subtraction assignment with evaluated Matrix (runtime)
      {
         test_  = "rowslice-wise expansion with subtraction assignment with evaluated Matrix (runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.rows(); ++i ) {
               rowslice( dres_  , i ) -= rowslice( expand( eval( mat_ ), E ), i );
//                rowslice( odres_ , i ) -= rowslice( expand( eval( mat_ ), E ), i );
//                rowslice( sres_  , i ) -= rowslice( expand( eval( mat_ ), E ), i );
//                rowslice( osres_ , i ) -= rowslice( expand( eval( mat_ ), E ), i );
               rowslice( refres_, i ) -= rowslice( expand( eval( refmat_ ), E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<E; ++i ) {
//                rowslice( tdres_  , i ) -= rowslice( expand( eval( tmat_ ), E ), i );
//                rowslice( todres_ , i ) -= rowslice( expand( eval( tmat_ ), E ), i );
//                rowslice( tsres_  , i ) -= rowslice( expand( eval( tmat_ ), E ), i );
//                rowslice( tosres_ , i ) -= rowslice( expand( eval( tmat_ ), E ), i );
//                rowslice( trefres_, i ) -= rowslice( expand( eval( trefmat_ ), E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // rowslice-wise expansion with subtraction assignment with evaluated Matrix (compile time)
      {
         test_  = "rowslice-wise expansion with subtraction assignment with evaluated Matrix (compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.rows(); ++i ) {
               rowslice( dres_  , i ) -= rowslice( expand<E>( eval( mat_ ) ), i );
//                rowslice( odres_ , i ) -= rowslice( expand<E>( eval( mat_ ) ), i );
//                rowslice( sres_  , i ) -= rowslice( expand<E>( eval( mat_ ) ), i );
//                rowslice( osres_ , i ) -= rowslice( expand<E>( eval( mat_ ) ), i );
               rowslice( refres_, i ) -= rowslice( expand<E>( eval( refmat_ ) ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<E; ++i ) {
//                rowslice( tdres_  , i ) -= rowslice( expand<E>( eval( tmat_ ) ), i );
//                rowslice( todres_ , i ) -= rowslice( expand<E>( eval( tmat_ ) ), i );
//                rowslice( tsres_  , i ) -= rowslice( expand<E>( eval( tmat_ ) ), i );
//                rowslice( tosres_ , i ) -= rowslice( expand<E>( eval( tmat_ ) ), i );
//                rowslice( trefres_, i ) -= rowslice( expand<E>( eval( trefmat_ ) ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the row-wise dense Matrix expansion operation.
//
// \return void
//
// This function is called in case the row-wise dense Matrix expansion operation is not
// available for the given Matrix type \a MT.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testRowSliceOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the rows-wise dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the rows-wise Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testRowSlicesOperation( blaze::TrueType )
{
// #if BLAZETEST_MATHTEST_TEST_ROWS_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_ROWS_OPERATION > 1 )
//    {
//       using blaze::expand;
//
//       if( mat_.size() == 0UL || E == 0UL )
//          return;
//
//
//       std::Matrix<size_t> indices( mat_.size() );
//       std::iota( indices.begin(), indices.end(), 0UL );
//       std::random_shuffle( indices.begin(), indices.end() );
//
//       std::Matrix<size_t> tindices( E );
//       std::iota( tindices.begin(), tindices.end(), 0UL );
//       std::random_shuffle( tindices.begin(), tindices.end() );
//
//
//       //=====================================================================================
//       // Rows-wise expansion
//       //=====================================================================================
//
//       // Rows-wise expansion with the given Matrix (runtime)
//       {
//          test_  = "Rows-wise expansion with the given Matrix (runtime)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) = rows( expand( mat_, E ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) = rows( expand( mat_, E ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) = rows( expand( mat_, E ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) = rows( expand( mat_, E ), &indices[index], n );
//                rows( refres_, &indices[index], n ) = rows( expand( refmat_, E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) = rows( expand( tmat_, E ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) = rows( expand( tmat_, E ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) = rows( expand( tmat_, E ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) = rows( expand( tmat_, E ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) = rows( expand( trefmat_, E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Rows-wise expansion with the given Matrix (compile time)
//       {
//          test_  = "Rows-wise expansion with the given Matrix (compile time)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) = rows( expand<E>( mat_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) = rows( expand<E>( mat_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) = rows( expand<E>( mat_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) = rows( expand<E>( mat_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) = rows( expand<E>( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) = rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) = rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) = rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) = rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) = rows( expand<E>( trefmat_ ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Rows-wise expansion with evaluated Matrix (runtime)
//       {
//          test_  = "Rows-wise expansion with evaluated Matrix (runtime)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) = rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) = rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) = rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) = rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( refres_, &indices[index], n ) = rows( expand( eval( refmat_ ), E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) = rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) = rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) = rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) = rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) = rows( expand( eval( trefmat_ ), E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Rows-wise expansion with evaluated Matrix (compile time)
//       {
//          test_  = "Rows-wise expansion with evaluated Matrix (compile time)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) = rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) = rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) = rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) = rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( refres_, &indices[index], n ) = rows( expand<E>( eval( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) = rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) = rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) = rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) = rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) = rows( expand<E>( eval( trefmat_ ) ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Rows-wise expansion with addition assignment
//       //=====================================================================================
//
//       // Rows-wise expansion with addition assignment with the given Matrix (runtime)
//       {
//          test_  = "Rows-wise expansion with addition assignment with the given Matrix (runtime)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) += rows( expand( mat_, E ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) += rows( expand( mat_, E ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) += rows( expand( mat_, E ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) += rows( expand( mat_, E ), &indices[index], n );
//                rows( refres_, &indices[index], n ) += rows( expand( refmat_, E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) += rows( expand( tmat_, E ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) += rows( expand( tmat_, E ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) += rows( expand( tmat_, E ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) += rows( expand( tmat_, E ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) += rows( expand( trefmat_, E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Rows-wise expansion with addition assignment with the given Matrix (compile time)
//       {
//          test_  = "Rows-wise expansion with addition assignment with the given Matrix (compile time)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) += rows( expand<E>( mat_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) += rows( expand<E>( mat_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) += rows( expand<E>( mat_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) += rows( expand<E>( mat_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) += rows( expand<E>( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) += rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) += rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) += rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) += rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) += rows( expand<E>( trefmat_ ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Rows-wise expansion with addition assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Rows-wise expansion with addition assignment with evaluated Matrix (runtime)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) += rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) += rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) += rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) += rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( refres_, &indices[index], n ) += rows( expand( eval( refmat_ ), E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) += rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) += rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) += rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) += rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) += rows( expand( eval( trefmat_ ), E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Rows-wise expansion with addition assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Rows-wise expansion with addition assignment with evaluated Matrix (compile time)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) += rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) += rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) += rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) += rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( refres_, &indices[index], n ) += rows( expand<E>( eval( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) += rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) += rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) += rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) += rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) += rows( expand<E>( eval( trefmat_ ) ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Rows-wise expansion with subtraction assignment
//       //=====================================================================================
//
//       // Rows-wise expansion with subtraction assignment with the given Matrix (runtime)
//       {
//          test_  = "Rows-wise expansion with subtraction assignment with the given Matrix (runtime)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) -= rows( expand( mat_, E ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) -= rows( expand( mat_, E ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) -= rows( expand( mat_, E ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) -= rows( expand( mat_, E ), &indices[index], n );
//                rows( refres_, &indices[index], n ) -= rows( expand( refmat_, E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) -= rows( expand( tmat_, E ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) -= rows( expand( tmat_, E ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) -= rows( expand( tmat_, E ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) -= rows( expand( tmat_, E ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) -= rows( expand( trefmat_, E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Rows-wise expansion with subtraction assignment with the given Matrix (compile time)
//       {
//          test_  = "Rows-wise expansion with subtraction assignment with the given Matrix (compile time)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) -= rows( expand<E>( mat_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) -= rows( expand<E>( mat_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) -= rows( expand<E>( mat_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) -= rows( expand<E>( mat_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) -= rows( expand<E>( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) -= rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) -= rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) -= rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) -= rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) -= rows( expand<E>( trefmat_ ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Rows-wise expansion with subtraction assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Rows-wise expansion with subtraction assignment with evaluated Matrix (runtime)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) -= rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) -= rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) -= rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) -= rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( refres_, &indices[index], n ) -= rows( expand( eval( refmat_ ), E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) -= rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) -= rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) -= rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) -= rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) -= rows( expand( eval( trefmat_ ), E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Rows-wise expansion with subtraction assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Rows-wise expansion with subtraction assignment with evaluated Matrix (compile time)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) -= rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) -= rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) -= rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) -= rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( refres_, &indices[index], n ) -= rows( expand<E>( eval( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) -= rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) -= rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) -= rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) -= rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) -= rows( expand<E>( eval( trefmat_ ) ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Rows-wise expansion with Schur product assignment
//       //=====================================================================================
//
//       // Rows-wise expansion with Schur product assignment with the given Matrix (runtime)
//       {
//          test_  = "Rows-wise expansion with Schur product assignment with the given Matrix (runtime)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) %= rows( expand( mat_, E ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) %= rows( expand( mat_, E ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) %= rows( expand( mat_, E ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) %= rows( expand( mat_, E ), &indices[index], n );
//                rows( refres_, &indices[index], n ) %= rows( expand( refmat_, E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) %= rows( expand( tmat_, E ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) %= rows( expand( tmat_, E ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) %= rows( expand( tmat_, E ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) %= rows( expand( tmat_, E ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) %= rows( expand( trefmat_, E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Rows-wise expansion with Schur product assignment with the given Matrix (compile time)
//       {
//          test_  = "Rows-wise expansion with Schur product assignment with the given Matrix (compile time)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) %= rows( expand<E>( mat_ ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) %= rows( expand<E>( mat_ ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) %= rows( expand<E>( mat_ ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) %= rows( expand<E>( mat_ ), &indices[index], n );
//                rows( refres_, &indices[index], n ) %= rows( expand<E>( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) %= rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) %= rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) %= rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) %= rows( expand<E>( tmat_ ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) %= rows( expand<E>( trefmat_ ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Rows-wise expansion with Schur product assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Rows-wise expansion with Schur product assignment with evaluated Matrix (runtime)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) %= rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) %= rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) %= rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) %= rows( expand( eval( mat_ ), E ), &indices[index], n );
//                rows( refres_, &indices[index], n ) %= rows( expand( eval( refmat_ ), E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) %= rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) %= rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) %= rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) %= rows( expand( eval( tmat_ ), E ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) %= rows( expand( eval( trefmat_ ), E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Rows-wise expansion with Schur product assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Rows-wise expansion with Schur product assignment with evaluated Matrix (compile time)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                rows( dres_  , &indices[index], n ) %= rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( odres_ , &indices[index], n ) %= rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( sres_  , &indices[index], n ) %= rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( osres_ , &indices[index], n ) %= rows( expand<E>( eval( mat_ ) ), &indices[index], n );
//                rows( refres_, &indices[index], n ) %= rows( expand<E>( eval( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                rows( tdres_  , &tindices[index], n ) %= rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( todres_ , &tindices[index], n ) %= rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( tsres_  , &tindices[index], n ) %= rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( tosres_ , &tindices[index], n ) %= rows( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                rows( trefres_, &tindices[index], n ) %= rows( expand<E>( eval( trefmat_ ) ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//    }
// #endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the rows-wise dense Matrix expansion operation.
//
// \return void
//
// This function is called in case the rows-wise dense Matrix expansion operation is not
// available for the given Matrix type \a MT.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testRowSlicesOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the column-wise dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the column-wise Matrix expansion with plain assignment, addition
// assignment, subtraction assignment, and Schur product assignment. In case any error resulting
// from the addition or the subsequent assignment is detected, a \a std::runtime_error exception
// is thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testColumnSliceOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_COLUMNSLICE_OPERATION
   if( BLAZETEST_MATHTEST_TEST_COLUMNSLICE_OPERATION > 1 )
   {
      using blaze::expand;

      if( mat_.rows() == 0UL || mat_.columns() == 0UL || E == 0UL )
         return;


      //=====================================================================================
      // columnslice-wise expansion
      //=====================================================================================

      // columnslice-wise expansion with the given Matrix (runtime)
      {
         test_  = "columnslice-wise expansion with the given Matrix (runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.columns(); ++i ) {
               columnslice( dres_  , i ) = columnslice( expand( mat_, E ), i );
//                columnslice( odres_ , i ) = columnslice( expand( mat_, E ), i );
//                columnslice( sres_  , i ) = columnslice( expand( mat_, E ), i );
//                columnslice( osres_ , i ) = columnslice( expand( mat_, E ), i );
               columnslice( refres_, i ) = columnslice( expand( refmat_, E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.rows(); ++i ) {
//                columnslice( tdres_  , i ) = columnslice( expand( tmat_, E ), i );
//                columnslice( todres_ , i ) = columnslice( expand( tmat_, E ), i );
//                columnslice( tsres_  , i ) = columnslice( expand( tmat_, E ), i );
//                columnslice( tosres_ , i ) = columnslice( expand( tmat_, E ), i );
//                columnslice( trefres_, i ) = columnslice( expand( trefmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // columnslice-wise expansion with the given Matrix (compile time)
      {
         test_  = "columnslice-wise expansion with the given Matrix (compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.columns(); ++i ) {
               columnslice( dres_  , i ) = columnslice( expand<E>( mat_ ), i );
//                columnslice( odres_ , i ) = columnslice( expand<E>( mat_ ), i );
//                columnslice( sres_  , i ) = columnslice( expand<E>( mat_ ), i );
//                columnslice( osres_ , i ) = columnslice( expand<E>( mat_ ), i );
               columnslice( refres_, i ) = columnslice( expand<E>( refmat_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.rows(); ++i ) {
//                columnslice( tdres_  , i ) = columnslice( expand<E>( tmat_ ), i );
//                columnslice( todres_ , i ) = columnslice( expand<E>( tmat_ ), i );
//                columnslice( tsres_  , i ) = columnslice( expand<E>( tmat_ ), i );
//                columnslice( tosres_ , i ) = columnslice( expand<E>( tmat_ ), i );
//                columnslice( trefres_, i ) = columnslice( expand<E>( trefmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // columnslice-wise expansion with evaluated Matrix (runtime)
      {
         test_  = "columnslice-wise expansion with evaluated Matrix (runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.columns(); ++i ) {
               columnslice( dres_  , i ) = columnslice( expand( eval( mat_ ), E ), i );
//                columnslice( odres_ , i ) = columnslice( expand( eval( mat_ ), E ), i );
//                columnslice( sres_  , i ) = columnslice( expand( eval( mat_ ), E ), i );
//                columnslice( osres_ , i ) = columnslice( expand( eval( mat_ ), E ), i );
               columnslice( refres_, i ) = columnslice( expand( eval( refmat_ ), E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.rows(); ++i ) {
//                columnslice( tdres_  , i ) = columnslice( expand( eval( tmat_ ), E ), i );
//                columnslice( todres_ , i ) = columnslice( expand( eval( tmat_ ), E ), i );
//                columnslice( tsres_  , i ) = columnslice( expand( eval( tmat_ ), E ), i );
//                columnslice( tosres_ , i ) = columnslice( expand( eval( tmat_ ), E ), i );
//                columnslice( trefres_, i ) = columnslice( expand( eval( trefmat_ ), E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // columnslice-wise expansion with evaluated Matrix (compile time)
      {
         test_  = "columnslice-wise expansion with evaluated Matrix (compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.columns(); ++i ) {
               columnslice( dres_  , i ) = columnslice( expand<E>( eval( mat_ ) ), i );
//                columnslice( odres_ , i ) = columnslice( expand<E>( eval( mat_ ) ), i );
//                columnslice( sres_  , i ) = columnslice( expand<E>( eval( mat_ ) ), i );
//                columnslice( osres_ , i ) = columnslice( expand<E>( eval( mat_ ) ), i );
               columnslice( refres_, i ) = columnslice( expand<E>( eval( refmat_ ) ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.rows(); ++i ) {
//                columnslice( tdres_  , i ) = columnslice( expand<E>( eval( tmat_ ) ), i );
//                columnslice( todres_ , i ) = columnslice( expand<E>( eval( tmat_ ) ), i );
//                columnslice( tsres_  , i ) = columnslice( expand<E>( eval( tmat_ ) ), i );
//                columnslice( tosres_ , i ) = columnslice( expand<E>( eval( tmat_ ) ), i );
//                columnslice( trefres_, i ) = columnslice( expand<E>( eval( trefmat_ ) ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // columnslice-wise expansion with addition assignment
      //=====================================================================================

      // columnslice-wise expansion with addition assignment with the given Matrix (runtime)
      {
         test_  = "columnslice-wise expansion with addition assignment with the given Matrix (runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.columns(); ++i ) {
               columnslice( dres_  , i ) += columnslice( expand( mat_, E ), i );
//                columnslice( odres_ , i ) += columnslice( expand( mat_, E ), i );
//                columnslice( sres_  , i ) += columnslice( expand( mat_, E ), i );
//                columnslice( osres_ , i ) += columnslice( expand( mat_, E ), i );
               columnslice( refres_, i ) += columnslice( expand( refmat_, E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.rows(); ++i ) {
//                columnslice( tdres_  , i ) += columnslice( expand( tmat_, E ), i );
//                columnslice( todres_ , i ) += columnslice( expand( tmat_, E ), i );
//                columnslice( tsres_  , i ) += columnslice( expand( tmat_, E ), i );
//                columnslice( tosres_ , i ) += columnslice( expand( tmat_, E ), i );
//                columnslice( trefres_, i ) += columnslice( expand( trefmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // columnslice-wise expansion with addition assignment with the given Matrix (compile time)
      {
         test_  = "columnslice-wise expansion with addition assignment with the given Matrix (compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.columns(); ++i ) {
               columnslice( dres_  , i ) += columnslice( expand<E>( mat_ ), i );
//                columnslice( odres_ , i ) += columnslice( expand<E>( mat_ ), i );
//                columnslice( sres_  , i ) += columnslice( expand<E>( mat_ ), i );
//                columnslice( osres_ , i ) += columnslice( expand<E>( mat_ ), i );
               columnslice( refres_, i ) += columnslice( expand<E>( refmat_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.rows(); ++i ) {
//                columnslice( tdres_  , i ) += columnslice( expand<E>( tmat_ ), i );
//                columnslice( todres_ , i ) += columnslice( expand<E>( tmat_ ), i );
//                columnslice( tsres_  , i ) += columnslice( expand<E>( tmat_ ), i );
//                columnslice( tosres_ , i ) += columnslice( expand<E>( tmat_ ), i );
//                columnslice( trefres_, i ) += columnslice( expand<E>( trefmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // columnslice-wise expansion with addition assignment with evaluated Matrix (runtime)
      {
         test_  = "columnslice-wise expansion with addition assignment with evaluated Matrix (runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.columns(); ++i ) {
               columnslice( dres_  , i ) += columnslice( expand( eval( mat_ ), E ), i );
//                columnslice( odres_ , i ) += columnslice( expand( eval( mat_ ), E ), i );
//                columnslice( sres_  , i ) += columnslice( expand( eval( mat_ ), E ), i );
//                columnslice( osres_ , i ) += columnslice( expand( eval( mat_ ), E ), i );
               columnslice( refres_, i ) += columnslice( expand( eval( refmat_ ), E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.rows(); ++i ) {
//                columnslice( tdres_  , i ) += columnslice( expand( eval( tmat_ ), E ), i );
//                columnslice( todres_ , i ) += columnslice( expand( eval( tmat_ ), E ), i );
//                columnslice( tsres_  , i ) += columnslice( expand( eval( tmat_ ), E ), i );
//                columnslice( tosres_ , i ) += columnslice( expand( eval( tmat_ ), E ), i );
//                columnslice( trefres_, i ) += columnslice( expand( eval( trefmat_ ), E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // columnslice-wise expansion with addition assignment with evaluated Matrix (compile time)
      {
         test_  = "columnslice-wise expansion with addition assignment with evaluated Matrix (compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.columns(); ++i ) {
               columnslice( dres_  , i ) += columnslice( expand<E>( eval( mat_ ) ), i );
//                columnslice( odres_ , i ) += columnslice( expand<E>( eval( mat_ ) ), i );
//                columnslice( sres_  , i ) += columnslice( expand<E>( eval( mat_ ) ), i );
//                columnslice( osres_ , i ) += columnslice( expand<E>( eval( mat_ ) ), i );
               columnslice( refres_, i ) += columnslice( expand<E>( eval( refmat_ ) ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.rows(); ++i ) {
//                columnslice( tdres_  , i ) += columnslice( expand<E>( eval( tmat_ ) ), i );
//                columnslice( todres_ , i ) += columnslice( expand<E>( eval( tmat_ ) ), i );
//                columnslice( tsres_  , i ) += columnslice( expand<E>( eval( tmat_ ) ), i );
//                columnslice( tosres_ , i ) += columnslice( expand<E>( eval( tmat_ ) ), i );
//                columnslice( trefres_, i ) += columnslice( expand<E>( eval( trefmat_ ) ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // columnslice-wise expansion with subtraction assignment
      //=====================================================================================

      // columnslice-wise expansion with subtraction assignment with the given Matrix (runtime)
      {
         test_  = "columnslice-wise expansion with subtraction assignment with the given Matrix (runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.columns(); ++i ) {
               columnslice( dres_  , i ) -= columnslice( expand( mat_, E ), i );
//                columnslice( odres_ , i ) -= columnslice( expand( mat_, E ), i );
//                columnslice( sres_  , i ) -= columnslice( expand( mat_, E ), i );
//                columnslice( osres_ , i ) -= columnslice( expand( mat_, E ), i );
               columnslice( refres_, i ) -= columnslice( expand( refmat_, E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.rows(); ++i ) {
//                columnslice( tdres_  , i ) -= columnslice( expand( tmat_, E ), i );
//                columnslice( todres_ , i ) -= columnslice( expand( tmat_, E ), i );
//                columnslice( tsres_  , i ) -= columnslice( expand( tmat_, E ), i );
//                columnslice( tosres_ , i ) -= columnslice( expand( tmat_, E ), i );
//                columnslice( trefres_, i ) -= columnslice( expand( trefmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // columnslice-wise expansion with subtraction assignment with the given Matrix (compile time)
      {
         test_  = "columnslice-wise expansion with subtraction assignment with the given Matrix (compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.columns(); ++i ) {
               columnslice( dres_  , i ) -= columnslice( expand<E>( mat_ ), i );
//                columnslice( odres_ , i ) -= columnslice( expand<E>( mat_ ), i );
//                columnslice( sres_  , i ) -= columnslice( expand<E>( mat_ ), i );
//                columnslice( osres_ , i ) -= columnslice( expand<E>( mat_ ), i );
               columnslice( refres_, i ) -= columnslice( expand<E>( refmat_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.rows(); ++i ) {
//                columnslice( tdres_  , i ) -= columnslice( expand<E>( tmat_ ), i );
//                columnslice( todres_ , i ) -= columnslice( expand<E>( tmat_ ), i );
//                columnslice( tsres_  , i ) -= columnslice( expand<E>( tmat_ ), i );
//                columnslice( tosres_ , i ) -= columnslice( expand<E>( tmat_ ), i );
//                columnslice( trefres_, i ) -= columnslice( expand<E>( trefmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // columnslice-wise expansion with subtraction assignment with evaluated Matrix (runtime)
      {
         test_  = "columnslice-wise expansion with subtraction assignment with evaluated Matrix (runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.columns(); ++i ) {
               columnslice( dres_  , i ) -= columnslice( expand( eval( mat_ ), E ), i );
//                columnslice( odres_ , i ) -= columnslice( expand( eval( mat_ ), E ), i );
//                columnslice( sres_  , i ) -= columnslice( expand( eval( mat_ ), E ), i );
//                columnslice( osres_ , i ) -= columnslice( expand( eval( mat_ ), E ), i );
               columnslice( refres_, i ) -= columnslice( expand( eval( refmat_ ), E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.rows(); ++i ) {
//                columnslice( tdres_  , i ) -= columnslice( expand( eval( tmat_ ), E ), i );
//                columnslice( todres_ , i ) -= columnslice( expand( eval( tmat_ ), E ), i );
//                columnslice( tsres_  , i ) -= columnslice( expand( eval( tmat_ ), E ), i );
//                columnslice( tosres_ , i ) -= columnslice( expand( eval( tmat_ ), E ), i );
//                columnslice( trefres_, i ) -= columnslice( expand( eval( trefmat_ ), E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // columnslice-wise expansion with subtraction assignment with evaluated Matrix (compile time)
      {
         test_  = "columnslice-wise expansion with subtraction assignment with evaluated Matrix (compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t i=0UL; i<mat_.columns(); ++i ) {
               columnslice( dres_  , i ) -= columnslice( expand<E>( eval( mat_ ) ), i );
//                columnslice( odres_ , i ) -= columnslice( expand<E>( eval( mat_ ) ), i );
//                columnslice( sres_  , i ) -= columnslice( expand<E>( eval( mat_ ) ), i );
//                columnslice( osres_ , i ) -= columnslice( expand<E>( eval( mat_ ) ), i );
               columnslice( refres_, i ) -= columnslice( expand<E>( eval( refmat_ ) ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.rows(); ++i ) {
//                columnslice( tdres_  , i ) -= columnslice( expand<E>( eval( tmat_ ) ), i );
//                columnslice( todres_ , i ) -= columnslice( expand<E>( eval( tmat_ ) ), i );
//                columnslice( tsres_  , i ) -= columnslice( expand<E>( eval( tmat_ ) ), i );
//                columnslice( tosres_ , i ) -= columnslice( expand<E>( eval( tmat_ ) ), i );
//                columnslice( trefres_, i ) -= columnslice( expand<E>( eval( trefmat_ ) ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the column-wise dense Matrix expansion operation.
//
// \return void
//
// This function is called in case the column-wise dense Matrix expansion operation is not
// available for the given Matrix type \a MT.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testColumnSliceOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the columns-wise dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the columns-wise Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testColumnSlicesOperation( blaze::TrueType )
{
// #if BLAZETEST_MATHTEST_TEST_COLUMNS_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_COLUMNS_OPERATION > 1 )
//    {
//       using blaze::expand;
//
//       if( mat_.size() == 0UL || E == 0UL )
//          return;
//
//
//       std::Matrix<size_t> indices( E );
//       std::iota( indices.begin(), indices.end(), 0UL );
//       std::random_shuffle( indices.begin(), indices.end() );
//
//       std::Matrix<size_t> tindices( mat_.size() );
//       std::iota( tindices.begin(), tindices.end(), 0UL );
//       std::random_shuffle( tindices.begin(), tindices.end() );
//
//
//       //=====================================================================================
//       // Columns-wise expansion
//       //=====================================================================================
//
//       // Columns-wise expansion with the given Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with the given Matrix (runtime)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) = columns( expand( mat_, E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) = columns( expand( mat_, E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) = columns( expand( mat_, E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) = columns( expand( mat_, E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) = columns( expand( refmat_, E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) = columns( expand( tmat_, E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) = columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) = columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) = columns( expand( tmat_, E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) = columns( expand( trefmat_, E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with the given Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with the given Matrix (compile time)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) = columns( expand<E>( mat_ ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) = columns( expand<E>( mat_ ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) = columns( expand<E>( mat_ ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) = columns( expand<E>( mat_ ), &indices[index], n );
//                columns( refres_, &indices[index], n ) = columns( expand<E>( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) = columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) = columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) = columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) = columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) = columns( expand<E>( trefmat_ ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with evaluated Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with evaluated Matrix (runtime)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) = columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) = columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) = columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) = columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) = columns( expand( eval( refmat_ ), E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) = columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) = columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) = columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) = columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) = columns( expand( eval( trefmat_ ), E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with evaluated Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with evaluated Matrix (compile time)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) = columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) = columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) = columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) = columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( refres_, &indices[index], n ) = columns( expand<E>( eval( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) = columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) = columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) = columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) = columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) = columns( expand<E>( eval( trefmat_ ) ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Columns-wise expansion with addition assignment
//       //=====================================================================================
//
//       // Columns-wise expansion with addition assignment with the given Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with addition assignment with the given Matrix (runtime)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) += columns( expand( mat_, E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) += columns( expand( mat_, E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) += columns( expand( mat_, E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) += columns( expand( mat_, E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) += columns( expand( refmat_, E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) += columns( expand( tmat_, E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) += columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) += columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) += columns( expand( tmat_, E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) += columns( expand( trefmat_, E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with addition assignment with the given Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with addition assignment with the given Matrix (compile time)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) += columns( expand<E>( mat_ ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) += columns( expand<E>( mat_ ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) += columns( expand<E>( mat_ ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) += columns( expand<E>( mat_ ), &indices[index], n );
//                columns( refres_, &indices[index], n ) += columns( expand<E>( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) += columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) += columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) += columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) += columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) += columns( expand<E>( trefmat_ ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with addition assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with addition assignment with evaluated Matrix (runtime)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) += columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) += columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) += columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) += columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) += columns( expand( eval( refmat_ ), E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) += columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) += columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) += columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) += columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) += columns( expand( eval( trefmat_ ), E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with addition assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with addition assignment with evaluated Matrix (compile time)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) += columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) += columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) += columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) += columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( refres_, &indices[index], n ) += columns( expand<E>( eval( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) += columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) += columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) += columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) += columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) += columns( expand<E>( eval( trefmat_ ) ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Columns-wise expansion with subtraction assignment
//       //=====================================================================================
//
//       // Columns-wise expansion with subtraction assignment with the given Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with subtraction assignment with the given Matrix (runtime)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) -= columns( expand( mat_, E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) -= columns( expand( mat_, E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) -= columns( expand( mat_, E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) -= columns( expand( mat_, E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) -= columns( expand( refmat_, E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) -= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) -= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) -= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) -= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) -= columns( expand( trefmat_, E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with subtraction assignment with the given Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with subtraction assignment with the given Matrix (compile time)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) -= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) -= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) -= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) -= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( refres_, &indices[index], n ) -= columns( expand<E>( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) -= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) -= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) -= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) -= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) -= columns( expand<E>( trefmat_ ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with subtraction assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with subtraction assignment with evaluated Matrix (runtime)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) -= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) -= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) -= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) -= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) -= columns( expand( eval( refmat_ ), E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) -= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) -= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) -= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) -= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) -= columns( expand( eval( trefmat_ ), E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with subtraction assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with subtraction assignment with evaluated Matrix (compile time)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) -= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) -= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) -= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) -= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( refres_, &indices[index], n ) -= columns( expand<E>( eval( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) -= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) -= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) -= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) -= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) -= columns( expand<E>( eval( trefmat_ ) ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Columns-wise expansion with Schur product assignment
//       //=====================================================================================
//
//       // Columns-wise expansion with Schur product assignment with the given Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with Schur product assignment with the given Matrix (runtime)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) %= columns( expand( mat_, E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) %= columns( expand( mat_, E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) %= columns( expand( mat_, E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) %= columns( expand( mat_, E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) %= columns( expand( refmat_, E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) %= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) %= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) %= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) %= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) %= columns( expand( trefmat_, E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with Schur product assignment with the given Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with Schur product assignment with the given Matrix (compile time)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) %= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) %= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) %= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) %= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( refres_, &indices[index], n ) %= columns( expand<E>( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) %= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) %= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) %= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) %= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) %= columns( expand<E>( trefmat_ ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with Schur product assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with Schur product assignment with evaluated Matrix (runtime)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) %= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) %= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) %= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) %= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) %= columns( expand( eval( refmat_ ), E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) %= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) %= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) %= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) %= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) %= columns( expand( eval( trefmat_ ), E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with Schur product assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with Schur product assignment with evaluated Matrix (compile time)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) %= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) %= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) %= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) %= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( refres_, &indices[index], n ) %= columns( expand<E>( eval( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) %= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) %= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) %= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) %= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) %= columns( expand<E>( eval( trefmat_ ) ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//    }
// #endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the columns-wise dense Matrix expansion operation.
//
// \return void
//
// This function is called in case the columns-wise dense Matrix expansion operation is not
// available for the given Matrix type \a MT.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testColumnSlicesOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the column-wise dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the column-wise Matrix expansion with plain assignment, addition
// assignment, subtraction assignment, and Schur product assignment. In case any error resulting
// from the addition or the subsequent assignment is detected, a \a std::runtime_error exception
// is thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testPageSliceOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_PAGESLICE_OPERATION
   if( BLAZETEST_MATHTEST_TEST_PAGESLICE_OPERATION > 1 )
   {
      using blaze::expand;

      if( mat_.rows() == 0UL || mat_.columns() == 0UL || E == 0UL )
         return;


      //=====================================================================================
      // pageslice-wise expansion
      //=====================================================================================

      // pageslice-wise expansion with the given Matrix (runtime)
      {
         test_  = "pageslice-wise expansion with the given Matrix (runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t i=0UL; i<E; ++i ) {
               pageslice( dres_  , i ) = pageslice( expand( mat_, E ), i );
//                pageslice( odres_ , i ) = pageslice( expand( mat_, E ), i );
//                pageslice( sres_  , i ) = pageslice( expand( mat_, E ), i );
//                pageslice( osres_ , i ) = pageslice( expand( mat_, E ), i );
               pageslice( refres_, i ) = pageslice( expand( refmat_, E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.size(); ++i ) {
//                pageslice( tdres_  , i ) = pageslice( expand( tmat_, E ), i );
//                pageslice( todres_ , i ) = pageslice( expand( tmat_, E ), i );
//                pageslice( tsres_  , i ) = pageslice( expand( tmat_, E ), i );
//                pageslice( tosres_ , i ) = pageslice( expand( tmat_, E ), i );
//                pageslice( trefres_, i ) = pageslice( expand( trefmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // pageslice-wise expansion with the given Matrix (compile time)
      {
         test_  = "pageslice-wise expansion with the given Matrix (compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t i=0UL; i<E; ++i ) {
               pageslice( dres_  , i ) = pageslice( expand<E>( mat_ ), i );
//                pageslice( odres_ , i ) = pageslice( expand<E>( mat_ ), i );
//                pageslice( sres_  , i ) = pageslice( expand<E>( mat_ ), i );
//                pageslice( osres_ , i ) = pageslice( expand<E>( mat_ ), i );
               pageslice( refres_, i ) = pageslice( expand<E>( refmat_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.size(); ++i ) {
//                pageslice( tdres_  , i ) = pageslice( expand<E>( tmat_ ), i );
//                pageslice( todres_ , i ) = pageslice( expand<E>( tmat_ ), i );
//                pageslice( tsres_  , i ) = pageslice( expand<E>( tmat_ ), i );
//                pageslice( tosres_ , i ) = pageslice( expand<E>( tmat_ ), i );
//                pageslice( trefres_, i ) = pageslice( expand<E>( trefmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // pageslice-wise expansion with evaluated Matrix (runtime)
      {
         test_  = "pageslice-wise expansion with evaluated Matrix (runtime)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t i=0UL; i<E; ++i ) {
               pageslice( dres_  , i ) = pageslice( expand( eval( mat_ ), E ), i );
//                pageslice( odres_ , i ) = pageslice( expand( eval( mat_ ), E ), i );
//                pageslice( sres_  , i ) = pageslice( expand( eval( mat_ ), E ), i );
//                pageslice( osres_ , i ) = pageslice( expand( eval( mat_ ), E ), i );
               pageslice( refres_, i ) = pageslice( expand( eval( refmat_ ), E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.size(); ++i ) {
//                pageslice( tdres_  , i ) = pageslice( expand( eval( tmat_ ), E ), i );
//                pageslice( todres_ , i ) = pageslice( expand( eval( tmat_ ), E ), i );
//                pageslice( tsres_  , i ) = pageslice( expand( eval( tmat_ ), E ), i );
//                pageslice( tosres_ , i ) = pageslice( expand( eval( tmat_ ), E ), i );
//                pageslice( trefres_, i ) = pageslice( expand( eval( trefmat_ ), E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // pageslice-wise expansion with evaluated Matrix (compile time)
      {
         test_  = "pageslice-wise expansion with evaluated Matrix (compile time)";
         error_ = "Failed expansion operation";

         try {
            initResults();
            for( size_t i=0UL; i<E; ++i ) {
               pageslice( dres_  , i ) = pageslice( expand<E>( eval( mat_ ) ), i );
//                pageslice( odres_ , i ) = pageslice( expand<E>( eval( mat_ ) ), i );
//                pageslice( sres_  , i ) = pageslice( expand<E>( eval( mat_ ) ), i );
//                pageslice( osres_ , i ) = pageslice( expand<E>( eval( mat_ ) ), i );
               pageslice( refres_, i ) = pageslice( expand<E>( eval( refmat_ ) ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.size(); ++i ) {
//                pageslice( tdres_  , i ) = pageslice( expand<E>( eval( tmat_ ) ), i );
//                pageslice( todres_ , i ) = pageslice( expand<E>( eval( tmat_ ) ), i );
//                pageslice( tsres_  , i ) = pageslice( expand<E>( eval( tmat_ ) ), i );
//                pageslice( tosres_ , i ) = pageslice( expand<E>( eval( tmat_ ) ), i );
//                pageslice( trefres_, i ) = pageslice( expand<E>( eval( trefmat_ ) ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // pageslice-wise expansion with addition assignment
      //=====================================================================================

      // pageslice-wise expansion with addition assignment with the given Matrix (runtime)
      {
         test_  = "pageslice-wise expansion with addition assignment with the given Matrix (runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t i=0UL; i<E; ++i ) {
               pageslice( dres_  , i ) += pageslice( expand( mat_, E ), i );
//                pageslice( odres_ , i ) += pageslice( expand( mat_, E ), i );
//                pageslice( sres_  , i ) += pageslice( expand( mat_, E ), i );
//                pageslice( osres_ , i ) += pageslice( expand( mat_, E ), i );
               pageslice( refres_, i ) += pageslice( expand( refmat_, E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.size(); ++i ) {
//                pageslice( tdres_  , i ) += pageslice( expand( tmat_, E ), i );
//                pageslice( todres_ , i ) += pageslice( expand( tmat_, E ), i );
//                pageslice( tsres_  , i ) += pageslice( expand( tmat_, E ), i );
//                pageslice( tosres_ , i ) += pageslice( expand( tmat_, E ), i );
//                pageslice( trefres_, i ) += pageslice( expand( trefmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // pageslice-wise expansion with addition assignment with the given Matrix (compile time)
      {
         test_  = "pageslice-wise expansion with addition assignment with the given Matrix (compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t i=0UL; i<E; ++i ) {
               pageslice( dres_  , i ) += pageslice( expand<E>( mat_ ), i );
//                pageslice( odres_ , i ) += pageslice( expand<E>( mat_ ), i );
//                pageslice( sres_  , i ) += pageslice( expand<E>( mat_ ), i );
//                pageslice( osres_ , i ) += pageslice( expand<E>( mat_ ), i );
               pageslice( refres_, i ) += pageslice( expand<E>( refmat_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.size(); ++i ) {
//                pageslice( tdres_  , i ) += pageslice( expand<E>( tmat_ ), i );
//                pageslice( todres_ , i ) += pageslice( expand<E>( tmat_ ), i );
//                pageslice( tsres_  , i ) += pageslice( expand<E>( tmat_ ), i );
//                pageslice( tosres_ , i ) += pageslice( expand<E>( tmat_ ), i );
//                pageslice( trefres_, i ) += pageslice( expand<E>( trefmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // pageslice-wise expansion with addition assignment with evaluated Matrix (runtime)
      {
         test_  = "pageslice-wise expansion with addition assignment with evaluated Matrix (runtime)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t i=0UL; i<E; ++i ) {
               pageslice( dres_  , i ) += pageslice( expand( eval( mat_ ), E ), i );
//                pageslice( odres_ , i ) += pageslice( expand( eval( mat_ ), E ), i );
//                pageslice( sres_  , i ) += pageslice( expand( eval( mat_ ), E ), i );
//                pageslice( osres_ , i ) += pageslice( expand( eval( mat_ ), E ), i );
               pageslice( refres_, i ) += pageslice( expand( eval( refmat_ ), E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.size(); ++i ) {
//                pageslice( tdres_  , i ) += pageslice( expand( eval( tmat_ ), E ), i );
//                pageslice( todres_ , i ) += pageslice( expand( eval( tmat_ ), E ), i );
//                pageslice( tsres_  , i ) += pageslice( expand( eval( tmat_ ), E ), i );
//                pageslice( tosres_ , i ) += pageslice( expand( eval( tmat_ ), E ), i );
//                pageslice( trefres_, i ) += pageslice( expand( eval( trefmat_ ), E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // pageslice-wise expansion with addition assignment with evaluated Matrix (compile time)
      {
         test_  = "pageslice-wise expansion with addition assignment with evaluated Matrix (compile time)";
         error_ = "Failed addition assignment";

         try {
            initResults();
            for( size_t i=0UL; i<E; ++i ) {
               pageslice( dres_  , i ) += pageslice( expand<E>( eval( mat_ ) ), i );
//                pageslice( odres_ , i ) += pageslice( expand<E>( eval( mat_ ) ), i );
//                pageslice( sres_  , i ) += pageslice( expand<E>( eval( mat_ ) ), i );
//                pageslice( osres_ , i ) += pageslice( expand<E>( eval( mat_ ) ), i );
               pageslice( refres_, i ) += pageslice( expand<E>( eval( refmat_ ) ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.size(); ++i ) {
//                pageslice( tdres_  , i ) += pageslice( expand<E>( eval( tmat_ ) ), i );
//                pageslice( todres_ , i ) += pageslice( expand<E>( eval( tmat_ ) ), i );
//                pageslice( tsres_  , i ) += pageslice( expand<E>( eval( tmat_ ) ), i );
//                pageslice( tosres_ , i ) += pageslice( expand<E>( eval( tmat_ ) ), i );
//                pageslice( trefres_, i ) += pageslice( expand<E>( eval( trefmat_ ) ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }


      //=====================================================================================
      // pageslice-wise expansion with subtraction assignment
      //=====================================================================================

      // pageslice-wise expansion with subtraction assignment with the given Matrix (runtime)
      {
         test_  = "pageslice-wise expansion with subtraction assignment with the given Matrix (runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t i=0UL; i<E; ++i ) {
               pageslice( dres_  , i ) -= pageslice( expand( mat_, E ), i );
//                pageslice( odres_ , i ) -= pageslice( expand( mat_, E ), i );
//                pageslice( sres_  , i ) -= pageslice( expand( mat_, E ), i );
//                pageslice( osres_ , i ) -= pageslice( expand( mat_, E ), i );
               pageslice( refres_, i ) -= pageslice( expand( refmat_, E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.size(); ++i ) {
//                pageslice( tdres_  , i ) -= pageslice( expand( tmat_, E ), i );
//                pageslice( todres_ , i ) -= pageslice( expand( tmat_, E ), i );
//                pageslice( tsres_  , i ) -= pageslice( expand( tmat_, E ), i );
//                pageslice( tosres_ , i ) -= pageslice( expand( tmat_, E ), i );
//                pageslice( trefres_, i ) -= pageslice( expand( trefmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // pageslice-wise expansion with subtraction assignment with the given Matrix (compile time)
      {
         test_  = "pageslice-wise expansion with subtraction assignment with the given Matrix (compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t i=0UL; i<E; ++i ) {
               pageslice( dres_  , i ) -= pageslice( expand<E>( mat_ ), i );
//                pageslice( odres_ , i ) -= pageslice( expand<E>( mat_ ), i );
//                pageslice( sres_  , i ) -= pageslice( expand<E>( mat_ ), i );
//                pageslice( osres_ , i ) -= pageslice( expand<E>( mat_ ), i );
               pageslice( refres_, i ) -= pageslice( expand<E>( refmat_ ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.size(); ++i ) {
//                pageslice( tdres_  , i ) -= pageslice( expand<E>( tmat_ ), i );
//                pageslice( todres_ , i ) -= pageslice( expand<E>( tmat_ ), i );
//                pageslice( tsres_  , i ) -= pageslice( expand<E>( tmat_ ), i );
//                pageslice( tosres_ , i ) -= pageslice( expand<E>( tmat_ ), i );
//                pageslice( trefres_, i ) -= pageslice( expand<E>( trefmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // pageslice-wise expansion with subtraction assignment with evaluated Matrix (runtime)
      {
         test_  = "pageslice-wise expansion with subtraction assignment with evaluated Matrix (runtime)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t i=0UL; i<E; ++i ) {
               pageslice( dres_  , i ) -= pageslice( expand( eval( mat_ ), E ), i );
//                pageslice( odres_ , i ) -= pageslice( expand( eval( mat_ ), E ), i );
//                pageslice( sres_  , i ) -= pageslice( expand( eval( mat_ ), E ), i );
//                pageslice( osres_ , i ) -= pageslice( expand( eval( mat_ ), E ), i );
               pageslice( refres_, i ) -= pageslice( expand( eval( refmat_ ), E ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.size(); ++i ) {
//                pageslice( tdres_  , i ) -= pageslice( expand( eval( tmat_ ), E ), i );
//                pageslice( todres_ , i ) -= pageslice( expand( eval( tmat_ ), E ), i );
//                pageslice( tsres_  , i ) -= pageslice( expand( eval( tmat_ ), E ), i );
//                pageslice( tosres_ , i ) -= pageslice( expand( eval( tmat_ ), E ), i );
//                pageslice( trefres_, i ) -= pageslice( expand( eval( trefmat_ ), E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }

      // pageslice-wise expansion with subtraction assignment with evaluated Matrix (compile time)
      {
         test_  = "pageslice-wise expansion with subtraction assignment with evaluated Matrix (compile time)";
         error_ = "Failed subtraction assignment";

         try {
            initResults();
            for( size_t i=0UL; i<E; ++i ) {
               pageslice( dres_  , i ) -= pageslice( expand<E>( eval( mat_ ) ), i );
//                pageslice( odres_ , i ) -= pageslice( expand<E>( eval( mat_ ) ), i );
//                pageslice( sres_  , i ) -= pageslice( expand<E>( eval( mat_ ) ), i );
//                pageslice( osres_ , i ) -= pageslice( expand<E>( eval( mat_ ) ), i );
               pageslice( refres_, i ) -= pageslice( expand<E>( eval( refmat_ ) ), i );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

//          try {
//             initTransposeResults();
//             for( size_t i=0UL; i<mat_.size(); ++i ) {
//                pageslice( tdres_  , i ) -= pageslice( expand<E>( eval( tmat_ ) ), i );
//                pageslice( todres_ , i ) -= pageslice( expand<E>( eval( tmat_ ) ), i );
//                pageslice( tsres_  , i ) -= pageslice( expand<E>( eval( tmat_ ) ), i );
//                pageslice( tosres_ , i ) -= pageslice( expand<E>( eval( tmat_ ) ), i );
//                pageslice( trefres_, i ) -= pageslice( expand<E>( eval( trefmat_ ) ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the page-wise dense Matrix expansion operation.
//
// \return void
//
// This function is called in case the column-wise dense Matrix expansion operation is not
// available for the given Matrix type \a MT.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testPageSliceOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the pageslices-wise dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the pageslices-wise Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and Schur product assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testPageSlicesOperation( blaze::TrueType )
{
// #if BLAZETEST_MATHTEST_TEST_COLUMNS_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_COLUMNS_OPERATION > 1 )
//    {
//       using blaze::expand;
//
//       if( mat_.size() == 0UL || E == 0UL )
//          return;
//
//
//       std::Matrix<size_t> indices( E );
//       std::iota( indices.begin(), indices.end(), 0UL );
//       std::random_shuffle( indices.begin(), indices.end() );
//
//       std::Matrix<size_t> tindices( mat_.size() );
//       std::iota( tindices.begin(), tindices.end(), 0UL );
//       std::random_shuffle( tindices.begin(), tindices.end() );
//
//
//       //=====================================================================================
//       // Columns-wise expansion
//       //=====================================================================================
//
//       // Columns-wise expansion with the given Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with the given Matrix (runtime)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) = columns( expand( mat_, E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) = columns( expand( mat_, E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) = columns( expand( mat_, E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) = columns( expand( mat_, E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) = columns( expand( refmat_, E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) = columns( expand( tmat_, E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) = columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) = columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) = columns( expand( tmat_, E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) = columns( expand( trefmat_, E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with the given Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with the given Matrix (compile time)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) = columns( expand<E>( mat_ ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) = columns( expand<E>( mat_ ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) = columns( expand<E>( mat_ ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) = columns( expand<E>( mat_ ), &indices[index], n );
//                columns( refres_, &indices[index], n ) = columns( expand<E>( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) = columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) = columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) = columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) = columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) = columns( expand<E>( trefmat_ ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with evaluated Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with evaluated Matrix (runtime)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) = columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) = columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) = columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) = columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) = columns( expand( eval( refmat_ ), E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) = columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) = columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) = columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) = columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) = columns( expand( eval( trefmat_ ), E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with evaluated Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with evaluated Matrix (compile time)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) = columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) = columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) = columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) = columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( refres_, &indices[index], n ) = columns( expand<E>( eval( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) = columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) = columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) = columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) = columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) = columns( expand<E>( eval( trefmat_ ) ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Columns-wise expansion with addition assignment
//       //=====================================================================================
//
//       // Columns-wise expansion with addition assignment with the given Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with addition assignment with the given Matrix (runtime)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) += columns( expand( mat_, E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) += columns( expand( mat_, E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) += columns( expand( mat_, E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) += columns( expand( mat_, E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) += columns( expand( refmat_, E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) += columns( expand( tmat_, E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) += columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) += columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) += columns( expand( tmat_, E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) += columns( expand( trefmat_, E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with addition assignment with the given Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with addition assignment with the given Matrix (compile time)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) += columns( expand<E>( mat_ ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) += columns( expand<E>( mat_ ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) += columns( expand<E>( mat_ ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) += columns( expand<E>( mat_ ), &indices[index], n );
//                columns( refres_, &indices[index], n ) += columns( expand<E>( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) += columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) += columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) += columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) += columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) += columns( expand<E>( trefmat_ ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with addition assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with addition assignment with evaluated Matrix (runtime)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) += columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) += columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) += columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) += columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) += columns( expand( eval( refmat_ ), E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) += columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) += columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) += columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) += columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) += columns( expand( eval( trefmat_ ), E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with addition assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with addition assignment with evaluated Matrix (compile time)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) += columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) += columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) += columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) += columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( refres_, &indices[index], n ) += columns( expand<E>( eval( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) += columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) += columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) += columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) += columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) += columns( expand<E>( eval( trefmat_ ) ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Columns-wise expansion with subtraction assignment
//       //=====================================================================================
//
//       // Columns-wise expansion with subtraction assignment with the given Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with subtraction assignment with the given Matrix (runtime)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) -= columns( expand( mat_, E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) -= columns( expand( mat_, E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) -= columns( expand( mat_, E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) -= columns( expand( mat_, E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) -= columns( expand( refmat_, E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) -= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) -= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) -= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) -= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) -= columns( expand( trefmat_, E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with subtraction assignment with the given Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with subtraction assignment with the given Matrix (compile time)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) -= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) -= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) -= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) -= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( refres_, &indices[index], n ) -= columns( expand<E>( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) -= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) -= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) -= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) -= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) -= columns( expand<E>( trefmat_ ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with subtraction assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with subtraction assignment with evaluated Matrix (runtime)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) -= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) -= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) -= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) -= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) -= columns( expand( eval( refmat_ ), E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) -= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) -= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) -= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) -= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) -= columns( expand( eval( trefmat_ ), E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with subtraction assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with subtraction assignment with evaluated Matrix (compile time)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) -= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) -= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) -= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) -= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( refres_, &indices[index], n ) -= columns( expand<E>( eval( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) -= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) -= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) -= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) -= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) -= columns( expand<E>( eval( trefmat_ ) ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Columns-wise expansion with Schur product assignment
//       //=====================================================================================
//
//       // Columns-wise expansion with Schur product assignment with the given Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with Schur product assignment with the given Matrix (runtime)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) %= columns( expand( mat_, E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) %= columns( expand( mat_, E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) %= columns( expand( mat_, E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) %= columns( expand( mat_, E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) %= columns( expand( refmat_, E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) %= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) %= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) %= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) %= columns( expand( tmat_, E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) %= columns( expand( trefmat_, E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with Schur product assignment with the given Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with Schur product assignment with the given Matrix (compile time)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) %= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) %= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) %= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) %= columns( expand<E>( mat_ ), &indices[index], n );
//                columns( refres_, &indices[index], n ) %= columns( expand<E>( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) %= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) %= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) %= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) %= columns( expand<E>( tmat_ ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) %= columns( expand<E>( trefmat_ ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with Schur product assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Columns-wise expansion with Schur product assignment with evaluated Matrix (runtime)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) %= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) %= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) %= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) %= columns( expand( eval( mat_ ), E ), &indices[index], n );
//                columns( refres_, &indices[index], n ) %= columns( expand( eval( refmat_ ), E ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) %= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) %= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) %= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) %= columns( expand( eval( tmat_ ), E ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) %= columns( expand( eval( trefmat_ ), E ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Columns-wise expansion with Schur product assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Columns-wise expansion with Schur product assignment with evaluated Matrix (compile time)";
//          error_ = "Failed Schur product assignment";
//
//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                columns( dres_  , &indices[index], n ) %= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( odres_ , &indices[index], n ) %= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( sres_  , &indices[index], n ) %= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( osres_ , &indices[index], n ) %= columns( expand<E>( eval( mat_ ) ), &indices[index], n );
//                columns( refres_, &indices[index], n ) %= columns( expand<E>( eval( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( size_t index=0UL, n=0UL; index<tindices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, tindices.size() - index );
//                columns( tdres_  , &tindices[index], n ) %= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( todres_ , &tindices[index], n ) %= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tsres_  , &tindices[index], n ) %= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( tosres_ , &tindices[index], n ) %= columns( expand<E>( eval( tmat_ ) ), &tindices[index], n );
//                columns( trefres_, &tindices[index], n ) %= columns( expand<E>( eval( trefmat_ ) ), &tindices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//    }
// #endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the pageslices-wise dense Matrix expansion operation.
//
// \return void
//
// This function is called in case the pageslices-wise dense Matrix expansion operation is not
// available for the given Matrix type \a MT.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testPageSlicesOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the band-wise dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function tests the band-wise Matrix expansion with plain assignment, addition assignment,
// subtraction assignment, and multiplication assignment. In case any error resulting from the
// addition or the subsequent assignment is detected, a \a std::runtime_error exception is
// thrown.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testBandOperation( blaze::TrueType )
{
// #if BLAZETEST_MATHTEST_TEST_BAND_OPERATION
//    if( BLAZETEST_MATHTEST_TEST_BAND_OPERATION > 1 )
//    {
//       using blaze::expand;
//
//       if( mat_.size() == 0UL || E == 0UL )
//          return;
//
//
//       //=====================================================================================
//       // Band-wise expansion
//       //=====================================================================================
//
//       // Band-wise expansion with the given Matrix (runtime)
//       {
//          test_  = "Band-wise expansion with the given Matrix (runtime)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) = band( expand( mat_, E )   , i );
//                band( odres_ , i ) = band( expand( mat_, E )   , i );
//                band( sres_  , i ) = band( expand( mat_, E )   , i );
//                band( osres_ , i ) = band( expand( mat_, E )   , i );
//                band( refres_, i ) = band( expand( refmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) = band( expand( tmat_, E )   , j );
//                band( todres_ , j ) = band( expand( tmat_, E )   , j );
//                band( tsres_  , j ) = band( expand( tmat_, E )   , j );
//                band( tosres_ , j ) = band( expand( tmat_, E )   , j );
//                band( trefres_, j ) = band( expand( trefmat_, E ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Band-wise expansion with the given Matrix (compile time)
//       {
//          test_  = "Band-wise expansion with the given Matrix (compile time)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) = band( expand<E>( mat_ )   , i );
//                band( odres_ , i ) = band( expand<E>( mat_ )   , i );
//                band( sres_  , i ) = band( expand<E>( mat_ )   , i );
//                band( osres_ , i ) = band( expand<E>( mat_ )   , i );
//                band( refres_, i ) = band( expand<E>( refmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) = band( expand<E>( tmat_ )   , j );
//                band( todres_ , j ) = band( expand<E>( tmat_ )   , j );
//                band( tsres_  , j ) = band( expand<E>( tmat_ )   , j );
//                band( tosres_ , j ) = band( expand<E>( tmat_ )   , j );
//                band( trefres_, j ) = band( expand<E>( trefmat_ ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Band-wise expansion with evaluated Matrix (runtime)
//       {
//          test_  = "Band-wise expansion with evaluated Matrix (runtime)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) = band( expand( mat_, E )   , i );
//                band( odres_ , i ) = band( expand( mat_, E )   , i );
//                band( sres_  , i ) = band( expand( mat_, E )   , i );
//                band( osres_ , i ) = band( expand( mat_, E )   , i );
//                band( refres_, i ) = band( expand( refmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) = band( expand( tmat_, E )   , j );
//                band( todres_ , j ) = band( expand( tmat_, E )   , j );
//                band( tsres_  , j ) = band( expand( tmat_, E )   , j );
//                band( tosres_ , j ) = band( expand( tmat_, E )   , j );
//                band( trefres_, j ) = band( expand( trefmat_, E ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Band-wise expansion with evaluated Matrix (compile time)
//       {
//          test_  = "Band-wise expansion with evaluated Matrix (compile time)";
//          error_ = "Failed expansion operation";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) = band( expand<E>( mat_ )   , i );
//                band( odres_ , i ) = band( expand<E>( mat_ )   , i );
//                band( sres_  , i ) = band( expand<E>( mat_ )   , i );
//                band( osres_ , i ) = band( expand<E>( mat_ )   , i );
//                band( refres_, i ) = band( expand<E>( refmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) = band( expand<E>( tmat_ )   , j );
//                band( todres_ , j ) = band( expand<E>( tmat_ )   , j );
//                band( tsres_  , j ) = band( expand<E>( tmat_ )   , j );
//                band( tosres_ , j ) = band( expand<E>( tmat_ )   , j );
//                band( trefres_, j ) = band( expand<E>( trefmat_ ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Band-wise expansion with addition assignment
//       //=====================================================================================
//
//       // Band-wise expansion with addition assignment with the given Matrix (runtime)
//       {
//          test_  = "Band-wise expansion with addition assignment with the given Matrix (runtime)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) += band( expand( mat_, E )   , i );
//                band( odres_ , i ) += band( expand( mat_, E )   , i );
//                band( sres_  , i ) += band( expand( mat_, E )   , i );
//                band( osres_ , i ) += band( expand( mat_, E )   , i );
//                band( refres_, i ) += band( expand( refmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) += band( expand( tmat_, E )   , j );
//                band( todres_ , j ) += band( expand( tmat_, E )   , j );
//                band( tsres_  , j ) += band( expand( tmat_, E )   , j );
//                band( tosres_ , j ) += band( expand( tmat_, E )   , j );
//                band( trefres_, j ) += band( expand( trefmat_, E ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Band-wise expansion with addition assignment with the given Matrix (compile time)
//       {
//          test_  = "Band-wise expansion with addition assignment with the given Matrix (compile time)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) += band( expand<E>( mat_ )   , i );
//                band( odres_ , i ) += band( expand<E>( mat_ )   , i );
//                band( sres_  , i ) += band( expand<E>( mat_ )   , i );
//                band( osres_ , i ) += band( expand<E>( mat_ )   , i );
//                band( refres_, i ) += band( expand<E>( refmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) += band( expand<E>( tmat_ )   , j );
//                band( todres_ , j ) += band( expand<E>( tmat_ )   , j );
//                band( tsres_  , j ) += band( expand<E>( tmat_ )   , j );
//                band( tosres_ , j ) += band( expand<E>( tmat_ )   , j );
//                band( trefres_, j ) += band( expand<E>( trefmat_ ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Band-wise expansion with addition assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Band-wise expansion with addition assignment with evaluated Matrix (runtime)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) += band( expand( mat_, E )   , i );
//                band( odres_ , i ) += band( expand( mat_, E )   , i );
//                band( sres_  , i ) += band( expand( mat_, E )   , i );
//                band( osres_ , i ) += band( expand( mat_, E )   , i );
//                band( refres_, i ) += band( expand( refmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) += band( expand( tmat_, E )   , j );
//                band( todres_ , j ) += band( expand( tmat_, E )   , j );
//                band( tsres_  , j ) += band( expand( tmat_, E )   , j );
//                band( tosres_ , j ) += band( expand( tmat_, E )   , j );
//                band( trefres_, j ) += band( expand( trefmat_, E ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Band-wise expansion with addition assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Band-wise expansion with addition assignment with evaluated Matrix (compile time)";
//          error_ = "Failed addition assignment";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) += band( expand<E>( mat_ )   , i );
//                band( odres_ , i ) += band( expand<E>( mat_ )   , i );
//                band( sres_  , i ) += band( expand<E>( mat_ )   , i );
//                band( osres_ , i ) += band( expand<E>( mat_ )   , i );
//                band( refres_, i ) += band( expand<E>( refmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) += band( expand<E>( tmat_ )   , j );
//                band( todres_ , j ) += band( expand<E>( tmat_ )   , j );
//                band( tsres_  , j ) += band( expand<E>( tmat_ )   , j );
//                band( tosres_ , j ) += band( expand<E>( tmat_ )   , j );
//                band( trefres_, j ) += band( expand<E>( trefmat_ ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Band-wise expansion with subtraction assignment
//       //=====================================================================================
//
//       // Band-wise expansion with subtraction assignment with the given Matrix (runtime)
//       {
//          test_  = "Band-wise expansion with subtraction assignment with the given Matrix (runtime)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) -= band( expand( mat_, E )   , i );
//                band( odres_ , i ) -= band( expand( mat_, E )   , i );
//                band( sres_  , i ) -= band( expand( mat_, E )   , i );
//                band( osres_ , i ) -= band( expand( mat_, E )   , i );
//                band( refres_, i ) -= band( expand( refmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) -= band( expand( tmat_, E )   , j );
//                band( todres_ , j ) -= band( expand( tmat_, E )   , j );
//                band( tsres_  , j ) -= band( expand( tmat_, E )   , j );
//                band( tosres_ , j ) -= band( expand( tmat_, E )   , j );
//                band( trefres_, j ) -= band( expand( trefmat_, E ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Band-wise expansion with subtraction assignment with the given Matrix (compile time)
//       {
//          test_  = "Band-wise expansion with subtraction assignment with the given Matrix (compile time)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) -= band( expand<E>( mat_ )   , i );
//                band( odres_ , i ) -= band( expand<E>( mat_ )   , i );
//                band( sres_  , i ) -= band( expand<E>( mat_ )   , i );
//                band( osres_ , i ) -= band( expand<E>( mat_ )   , i );
//                band( refres_, i ) -= band( expand<E>( refmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) -= band( expand<E>( tmat_ )   , j );
//                band( todres_ , j ) -= band( expand<E>( tmat_ )   , j );
//                band( tsres_  , j ) -= band( expand<E>( tmat_ )   , j );
//                band( tosres_ , j ) -= band( expand<E>( tmat_ )   , j );
//                band( trefres_, j ) -= band( expand<E>( trefmat_ ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Band-wise expansion with subtraction assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Band-wise expansion with subtraction assignment with evaluated Matrix (runtime)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) -= band( expand( mat_, E )   , i );
//                band( odres_ , i ) -= band( expand( mat_, E )   , i );
//                band( sres_  , i ) -= band( expand( mat_, E )   , i );
//                band( osres_ , i ) -= band( expand( mat_, E )   , i );
//                band( refres_, i ) -= band( expand( refmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) -= band( expand( tmat_, E )   , j );
//                band( todres_ , j ) -= band( expand( tmat_, E )   , j );
//                band( tsres_  , j ) -= band( expand( tmat_, E )   , j );
//                band( tosres_ , j ) -= band( expand( tmat_, E )   , j );
//                band( trefres_, j ) -= band( expand( trefmat_, E ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Band-wise expansion with subtraction assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Band-wise expansion with subtraction assignment with evaluated Matrix (compile time)";
//          error_ = "Failed subtraction assignment";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) -= band( expand<E>( mat_ )   , i );
//                band( odres_ , i ) -= band( expand<E>( mat_ )   , i );
//                band( sres_  , i ) -= band( expand<E>( mat_ )   , i );
//                band( osres_ , i ) -= band( expand<E>( mat_ )   , i );
//                band( refres_, i ) -= band( expand<E>( refmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) -= band( expand<E>( tmat_ )   , j );
//                band( todres_ , j ) -= band( expand<E>( tmat_ )   , j );
//                band( tsres_  , j ) -= band( expand<E>( tmat_ )   , j );
//                band( tosres_ , j ) -= band( expand<E>( tmat_ )   , j );
//                band( trefres_, j ) -= band( expand<E>( trefmat_ ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//
//       //=====================================================================================
//       // Band-wise expansion with multiplication assignment
//       //=====================================================================================
//
//       // Band-wise expansion with multiplication assignment with the given Matrix (runtime)
//       {
//          test_  = "Band-wise expansion with multiplication assignment with the given Matrix (runtime)";
//          error_ = "Failed multiplication assignment";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) *= band( expand( mat_, E )   , i );
//                band( odres_ , i ) *= band( expand( mat_, E )   , i );
//                band( sres_  , i ) *= band( expand( mat_, E )   , i );
//                band( osres_ , i ) *= band( expand( mat_, E )   , i );
//                band( refres_, i ) *= band( expand( refmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) *= band( expand( tmat_, E )   , j );
//                band( todres_ , j ) *= band( expand( tmat_, E )   , j );
//                band( tsres_  , j ) *= band( expand( tmat_, E )   , j );
//                band( tosres_ , j ) *= band( expand( tmat_, E )   , j );
//                band( trefres_, j ) *= band( expand( trefmat_, E ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Band-wise expansion with multiplication assignment with the given Matrix (compile time)
//       {
//          test_  = "Band-wise expansion with multiplication assignment with the given Matrix (compile time)";
//          error_ = "Failed multiplication assignment";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) *= band( expand<E>( mat_ )   , i );
//                band( odres_ , i ) *= band( expand<E>( mat_ )   , i );
//                band( sres_  , i ) *= band( expand<E>( mat_ )   , i );
//                band( osres_ , i ) *= band( expand<E>( mat_ )   , i );
//                band( refres_, i ) *= band( expand<E>( refmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) *= band( expand<E>( tmat_ )   , j );
//                band( todres_ , j ) *= band( expand<E>( tmat_ )   , j );
//                band( tsres_  , j ) *= band( expand<E>( tmat_ )   , j );
//                band( tosres_ , j ) *= band( expand<E>( tmat_ )   , j );
//                band( trefres_, j ) *= band( expand<E>( trefmat_ ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Band-wise expansion with multiplication assignment with evaluated Matrix (runtime)
//       {
//          test_  = "Band-wise expansion with multiplication assignment with evaluated Matrix (runtime)";
//          error_ = "Failed multiplication assignment";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) *= band( expand( mat_, E )   , i );
//                band( odres_ , i ) *= band( expand( mat_, E )   , i );
//                band( sres_  , i ) *= band( expand( mat_, E )   , i );
//                band( osres_ , i ) *= band( expand( mat_, E )   , i );
//                band( refres_, i ) *= band( expand( refmat_, E ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) *= band( expand( tmat_, E )   , j );
//                band( todres_ , j ) *= band( expand( tmat_, E )   , j );
//                band( tsres_  , j ) *= band( expand( tmat_, E )   , j );
//                band( tosres_ , j ) *= band( expand( tmat_, E )   , j );
//                band( trefres_, j ) *= band( expand( trefmat_, E ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//
//       // Band-wise expansion with multiplication assignment with evaluated Matrix (compile time)
//       {
//          test_  = "Band-wise expansion with multiplication assignment with evaluated Matrix (compile time)";
//          error_ = "Failed multiplication assignment";
//
//          try {
//             initResults();
//             for( ptrdiff_t i=1UL-mat_.size(); i<E; ++i ) {
//                band( dres_  , i ) *= band( expand<E>( mat_ )   , i );
//                band( odres_ , i ) *= band( expand<E>( mat_ )   , i );
//                band( sres_  , i ) *= band( expand<E>( mat_ )   , i );
//                band( osres_ , i ) *= band( expand<E>( mat_ )   , i );
//                band( refres_, i ) *= band( expand<E>( refmat_ ), i );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<MT>( ex );
//          }
//
//          checkResults<MT>();
//
//          try {
//             initTransposeResults();
//             for( ptrdiff_t j=1UL-E; j<tmat_.size(); ++j ) {
//                band( tdres_  , j ) *= band( expand<E>( tmat_ )   , j );
//                band( todres_ , j ) *= band( expand<E>( tmat_ )   , j );
//                band( tsres_  , j ) *= band( expand<E>( tmat_ )   , j );
//                band( tosres_ , j ) *= band( expand<E>( tmat_ )   , j );
//                band( trefres_, j ) *= band( expand<E>( trefmat_ ), j );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TMT>( ex );
//          }
//
//          checkTransposeResults<TMT>();
//       }
//    }
// #endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the band-wise dense Matrix expansion operation.
//
// \return void
// \exception std::runtime_error Expansion error detected.
//
// This function is called in case the band-wise dense Matrix expansion operation is not available
// for the given Matrix type \a MT.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::testBandOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the customized dense Matrix expansion operation.
//
// \param op The custom operation to be tested.
// \param name The human-readable name of the operation.
// \return void
// \exception std::runtime_error Operation error detected.
//
// This function tests the Matrix expansion operation with plain assignment, addition assignment,
// subtraction assignment, multiplication assignment, and division assignment in combination with
// a custom operation. In case any error resulting from the expansion or the subsequent assignment
// is detected, a \a std::runtime_error exception is thrown.
*/
template< typename MT    // Type of the dense Matrix
        , size_t E >     // Compile time expansion
template< typename OP >  // Type of the custom operation
void OperationTest<MT,E>::testCustomOperation( OP op, const std::string& name )
{
   using blaze::expand;


   //=====================================================================================
   // Customized expansion operation
   //=====================================================================================

   // Customized expansion operation with the given Matrix (runtime)
   {
      test_  = "Customized expansion operation with the given Matrix (runtime)";
      error_ = "Failed expansion operation";

      try {
         initResults();
         dres_   = op( expand( mat_, E ) );
//          odres_  = op( expand( mat_, E ) );
//          sres_   = op( expand( mat_, E ) );
//          osres_  = op( expand( mat_, E ) );
         refres_ = op( expand( refmat_, E ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   = op( expand( tmat_, E ) );
//          todres_  = op( expand( tmat_, E ) );
//          tsres_   = op( expand( tmat_, E ) );
//          tosres_  = op( expand( tmat_, E ) );
//          trefres_ = op( expand( trefmat_, E ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }

   // Customized expansion operation with the given Matrix (compile time)
   {
      test_  = "Customized expansion operation with the given Matrix (compile time)";
      error_ = "Failed expansion operation";

      try {
         initResults();
         dres_   = op( expand<E>( mat_ ) );
//          odres_  = op( expand<E>( mat_ ) );
//          sres_   = op( expand<E>( mat_ ) );
//          osres_  = op( expand<E>( mat_ ) );
         refres_ = op( expand<E>( refmat_ ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   = op( expand<E>( tmat_ ) );
//          todres_  = op( expand<E>( tmat_ ) );
//          tsres_   = op( expand<E>( tmat_ ) );
//          tosres_  = op( expand<E>( tmat_ ) );
//          trefres_ = op( expand<E>( trefmat_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }

   // Customized expansion operation with evaluated Matrix (runtime)
   {
      test_  = "Customized expansion operation with evaluated Matrix (runtime)";
      error_ = "Failed expansion operation";

      try {
         initResults();
         dres_   = op( expand( eval( mat_ ), E ) );
//          odres_  = op( expand( eval( mat_ ), E ) );
//          sres_   = op( expand( eval( mat_ ), E ) );
//          osres_  = op( expand( eval( mat_ ), E ) );
         refres_ = op( expand( eval( refmat_ ), E ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   = op( expand( eval( tmat_ ), E ) );
//          todres_  = op( expand( eval( tmat_ ), E ) );
//          tsres_   = op( expand( eval( tmat_ ), E ) );
//          tosres_  = op( expand( eval( tmat_ ), E ) );
//          trefres_ = op( expand( eval( trefmat_ ), E ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }

   // Customized expansion operation with evaluated Matrix (compile time)
   {
      test_  = "Customized expansion operation with evaluated Matrix (compile time)";
      error_ = "Failed expansion operation";

      try {
         initResults();
         dres_   = op( expand<E>( eval( mat_ ) ) );
//          odres_  = op( expand<E>( eval( mat_ ) ) );
//          sres_   = op( expand<E>( eval( mat_ ) ) );
//          osres_  = op( expand<E>( eval( mat_ ) ) );
         refres_ = op( expand<E>( eval( refmat_ ) ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   = op( expand<E>( eval( tmat_ ) ) );
//          todres_  = op( expand<E>( eval( tmat_ ) ) );
//          tsres_   = op( expand<E>( eval( tmat_ ) ) );
//          tosres_  = op( expand<E>( eval( tmat_ ) ) );
//          trefres_ = op( expand<E>( eval( trefmat_ ) ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }


   //=====================================================================================
   // Customized expansion with addition assignment
   //=====================================================================================

   // Customized expansion with addition assignment with the given Matrix (runtime)
   {
      test_  = "Customized expansion with addition assignment with the given Matrix (runtime)";
      error_ = "Failed addition assignment";

      try {
         initResults();
         dres_   += op( expand( mat_, E ) );
//          odres_  += op( expand( mat_, E ) );
//          sres_   += op( expand( mat_, E ) );
//          osres_  += op( expand( mat_, E ) );
         refres_ += op( expand( refmat_, E ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   += op( expand( tmat_, E ) );
//          todres_  += op( expand( tmat_, E ) );
//          tsres_   += op( expand( tmat_, E ) );
//          tosres_  += op( expand( tmat_, E ) );
//          trefres_ += op( expand( trefmat_, E ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }

   // Customized expansion with addition assignment with the given Matrix (compile time)
   {
      test_  = "Customized expansion with addition assignment with the given Matrix (compile time)";
      error_ = "Failed addition assignment";

      try {
         initResults();
         dres_   += op( expand<E>( mat_ ) );
//          odres_  += op( expand<E>( mat_ ) );
//          sres_   += op( expand<E>( mat_ ) );
//          osres_  += op( expand<E>( mat_ ) );
         refres_ += op( expand<E>( refmat_ ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   += op( expand<E>( tmat_ ) );
//          todres_  += op( expand<E>( tmat_ ) );
//          tsres_   += op( expand<E>( tmat_ ) );
//          tosres_  += op( expand<E>( tmat_ ) );
//          trefres_ += op( expand<E>( trefmat_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }

   // Customized expansion with addition assignment with evaluated Matrix (runtime)
   {
      test_  = "Customized expansion with addition assignment with evaluated Matrix (runtime)";
      error_ = "Failed addition assignment";

      try {
         initResults();
         dres_   += op( expand( eval( mat_ ), E ) );
//          odres_  += op( expand( eval( mat_ ), E ) );
//          sres_   += op( expand( eval( mat_ ), E ) );
//          osres_  += op( expand( eval( mat_ ), E ) );
         refres_ += op( expand( eval( refmat_ ), E ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   += op( expand( eval( tmat_ ), E ) );
//          todres_  += op( expand( eval( tmat_ ), E ) );
//          tsres_   += op( expand( eval( tmat_ ), E ) );
//          tosres_  += op( expand( eval( tmat_ ), E ) );
//          trefres_ += op( expand( eval( trefmat_ ), E ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }

   // Customized expansion with addition assignment with evaluated Matrix (compile time)
   {
      test_  = "Customized expansion with addition assignment with evaluated Matrix (compile time)";
      error_ = "Failed addition assignment";

      try {
         initResults();
         dres_   += op( expand<E>( eval( mat_ ) ) );
//          odres_  += op( expand<E>( eval( mat_ ) ) );
//          sres_   += op( expand<E>( eval( mat_ ) ) );
//          osres_  += op( expand<E>( eval( mat_ ) ) );
         refres_ += op( expand<E>( eval( refmat_ ) ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   += op( expand<E>( eval( tmat_ ) ) );
//          todres_  += op( expand<E>( eval( tmat_ ) ) );
//          tsres_   += op( expand<E>( eval( tmat_ ) ) );
//          tosres_  += op( expand<E>( eval( tmat_ ) ) );
//          trefres_ += op( expand<E>( eval( trefmat_ ) ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }


   //=====================================================================================
   // Customized expansion with subtraction assignment
   //=====================================================================================

   // Customized expansion with subtraction assignment with the given Matrix (runtime)
   {
      test_  = "Customized expansion with subtraction assignment with the given Matrix (runtime)";
      error_ = "Failed subtraction assignment";

      try {
         initResults();
         dres_   -= op( expand( mat_, E ) );
//          odres_  -= op( expand( mat_, E ) );
//          sres_   -= op( expand( mat_, E ) );
//          osres_  -= op( expand( mat_, E ) );
         refres_ -= op( expand( refmat_, E ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   -= op( expand( tmat_, E ) );
//          todres_  -= op( expand( tmat_, E ) );
//          tsres_   -= op( expand( tmat_, E ) );
//          tosres_  -= op( expand( tmat_, E ) );
//          trefres_ -= op( expand( trefmat_, E ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }

   // Customized expansion with subtraction assignment with the given Matrix (compile time)
   {
      test_  = "Customized expansion with subtraction assignment with the given Matrix (compile time)";
      error_ = "Failed subtraction assignment";

      try {
         initResults();
         dres_   -= op( expand<E>( mat_ ) );
//          odres_  -= op( expand<E>( mat_ ) );
//          sres_   -= op( expand<E>( mat_ ) );
//          osres_  -= op( expand<E>( mat_ ) );
         refres_ -= op( expand<E>( refmat_ ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   -= op( expand<E>( tmat_ ) );
//          todres_  -= op( expand<E>( tmat_ ) );
//          tsres_   -= op( expand<E>( tmat_ ) );
//          tosres_  -= op( expand<E>( tmat_ ) );
//          trefres_ -= op( expand<E>( trefmat_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }

   // Customized expansion with subtraction assignment with evaluated Matrix (runtime)
   {
      test_  = "Customized expansion with subtraction assignment with evaluated Matrix (runtime)";
      error_ = "Failed subtraction assignment";

      try {
         initResults();
         dres_   -= op( expand( eval( mat_ ), E ) );
//          odres_  -= op( expand( eval( mat_ ), E ) );
//          sres_   -= op( expand( eval( mat_ ), E ) );
//          osres_  -= op( expand( eval( mat_ ), E ) );
         refres_ -= op( expand( eval( refmat_ ), E ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   -= op( expand( eval( tmat_ ), E ) );
//          todres_  -= op( expand( eval( tmat_ ), E ) );
//          tsres_   -= op( expand( eval( tmat_ ), E ) );
//          tosres_  -= op( expand( eval( tmat_ ), E ) );
//          trefres_ -= op( expand( eval( trefmat_ ), E ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }

   // Customized expansion with subtraction assignment with evaluated Matrix (compile time)
   {
      test_  = "Customized expansion with subtraction assignment with evaluated Matrix (compile time)";
      error_ = "Failed subtraction assignment";

      try {
         initResults();
         dres_   -= op( expand<E>( eval( mat_ ) ) );
//          odres_  -= op( expand<E>( eval( mat_ ) ) );
//          sres_   -= op( expand<E>( eval( mat_ ) ) );
//          osres_  -= op( expand<E>( eval( mat_ ) ) );
         refres_ -= op( expand<E>( eval( refmat_ ) ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   -= op( expand<E>( eval( tmat_ ) ) );
//          todres_  -= op( expand<E>( eval( tmat_ ) ) );
//          tsres_   -= op( expand<E>( eval( tmat_ ) ) );
//          tosres_  -= op( expand<E>( eval( tmat_ ) ) );
//          trefres_ -= op( expand<E>( eval( trefmat_ ) ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }


   //=====================================================================================
   // Customized expansion with Schur product assignment
   //=====================================================================================

   // Customized expansion with Schur product assignment with the given Matrix (runtime)
   {
      test_  = "Customized expansion with Schur product assignment with the given Matrix (runtime)";
      error_ = "Failed Schur product assignment";

      try {
         initResults();
         dres_   %= op( expand( mat_, E ) );
//          odres_  %= op( expand( mat_, E ) );
//          sres_   %= op( expand( mat_, E ) );
//          osres_  %= op( expand( mat_, E ) );
         refres_ %= op( expand( refmat_, E ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   %= op( expand( tmat_, E ) );
//          todres_  %= op( expand( tmat_, E ) );
//          tsres_   %= op( expand( tmat_, E ) );
//          tosres_  %= op( expand( tmat_, E ) );
//          trefres_ %= op( expand( trefmat_, E ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }

   // Customized expansion with Schur product assignment with the given Matrix (compile time)
   {
      test_  = "Customized expansion with Schur product assignment with the given Matrix (compile time)";
      error_ = "Failed Schur product assignment";

      try {
         initResults();
         dres_   %= op( expand<E>( mat_ ) );
//          odres_  %= op( expand<E>( mat_ ) );
//          sres_   %= op( expand<E>( mat_ ) );
//          osres_  %= op( expand<E>( mat_ ) );
         refres_ %= op( expand<E>( refmat_ ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   %= op( expand<E>( tmat_ ) );
//          todres_  %= op( expand<E>( tmat_ ) );
//          tsres_   %= op( expand<E>( tmat_ ) );
//          tosres_  %= op( expand<E>( tmat_ ) );
//          trefres_ %= op( expand<E>( trefmat_ ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }

   // Customized expansion with Schur product assignment with evaluated Matrix (runtime)
   {
      test_  = "Customized expansion with Schur product assignment with evaluated Matrix (runtime)";
      error_ = "Failed Schur product assignment";

      try {
         initResults();
         dres_   %= op( expand( eval( mat_ ), E ) );
//          odres_  %= op( expand( eval( mat_ ), E ) );
//          sres_   %= op( expand( eval( mat_ ), E ) );
//          osres_  %= op( expand( eval( mat_ ), E ) );
         refres_ %= op( expand( eval( refmat_ ), E ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   %= op( expand( eval( tmat_ ), E ) );
//          todres_  %= op( expand( eval( tmat_ ), E ) );
//          tsres_   %= op( expand( eval( tmat_ ), E ) );
//          tosres_  %= op( expand( eval( tmat_ ), E ) );
//          trefres_ %= op( expand( eval( trefmat_ ), E ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
   }

   // Customized expansion with Schur product assignment with evaluated Matrix (compile time)
   {
      test_  = "Customized expansion with Schur product assignment with evaluated Matrix (compile time)";
      error_ = "Failed Schur product assignment";

      try {
         initResults();
         dres_   %= op( expand<E>( eval( mat_ ) ) );
//          odres_  %= op( expand<E>( eval( mat_ ) ) );
//          sres_   %= op( expand<E>( eval( mat_ ) ) );
//          osres_  %= op( expand<E>( eval( mat_ ) ) );
         refres_ %= op( expand<E>( eval( refmat_ ) ) );
      }
      catch( std::exception& ex ) {
         convertException<MT>( ex );
      }

      checkResults<MT>();

//       try {
//          initTransposeResults();
//          tdres_   %= op( expand<E>( eval( tmat_ ) ) );
//          todres_  %= op( expand<E>( eval( tmat_ ) ) );
//          tsres_   %= op( expand<E>( eval( tmat_ ) ) );
//          tosres_  %= op( expand<E>( eval( tmat_ ) ) );
//          trefres_ %= op( expand<E>( eval( trefmat_ ) ) );
//       }
//       catch( std::exception& ex ) {
//          convertException<TMT>( ex );
//       }
//
//       checkTransposeResults<TMT>();
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
// template argument \a T indicates the type of the Matrix operand used for the computations.
*/
template< typename MT   // Type of the dense Matrix
        , size_t E >    // Compile time expansion
template< typename T >  // Type of the Matrix operand
void OperationTest<MT,E>::checkResults()
{
   using blaze::IsRowMajorMatrix;

   if( !isEqual( dres_, refres_ ) /*|| !isEqual( odres_, refres_ )*/ ) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect dense result tensor detected\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Dense " << ( IsRowMajorMatrix<T>::value ? ( "row-major" ) : ( "column-major" ) ) << " Matrix type:\n"
          << "     " << typeid( T ).name() << "\n"
          << "   Result:\n" << dres_ << "\n"
//           << "   Result with opposite storage order:\n" << odres_ << "\n"
          << "   Expected result:\n" << refres_ << "\n";
      throw std::runtime_error( oss.str() );
   }

//    if( !isEqual( sres_, refres_ ) /*|| !isEqual( osres_, refres_ )*/ ) {
//       std::ostringstream oss;
//       oss.precision( 20 );
//       oss << " Test : " << test_ << "\n"
//           << " Error: Incorrect sparse result tensor detected\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Dense " << ( IsRowMajorMatrix<T>::value ? ( "row-major" ) : ( "column-major" ) ) << " Matrix type:\n"
//           << "     " << typeid( T ).name() << "\n"
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
// results. The template argument \a T indicates the type of the Matrix operand used for the
// computations.
*/
template< typename MT   // Type of the dense Matrix
        , size_t E >    // Compile time expansion
template< typename T >  // Type of the Matrix operand
void OperationTest<MT,E>::checkTransposeResults()
{
//    using blaze::IsRowMatrix;
//
//    if( !isEqual( tdres_, trefres_ ) || !isEqual( todres_, trefres_ ) ) {
//       std::ostringstream oss;
//       oss.precision( 20 );
//       oss << " Test : " << test_ << "\n"
//           << " Error: Incorrect dense result tensor detected\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Dense " << ( IsRowMatrix<T>::value ? ( "row" ) : ( "column" ) ) << " Matrix type:\n"
//           << "     " << typeid( T ).name() << "\n"
//           << "   Transpose result:\n" << tdres_ << "\n"
//           << "   Transpose result with opposite storage order:\n" << todres_ << "\n"
//           << "   Expected result:\n" << trefres_ << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//
//    if( !isEqual( tsres_, trefres_ ) || !isEqual( tosres_, trefres_ ) ) {
//       std::ostringstream oss;
//       oss.precision( 20 );
//       oss << " Test : " << test_ << "\n"
//           << " Error: Incorrect sparse result tensor detected\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Dense " << ( IsRowMatrix<T>::value ? ( "row" ) : ( "column" ) ) << " Matrix type:\n"
//           << "     " << typeid( T ).name() << "\n"
//           << "   Transpose result:\n" << tsres_ << "\n"
//           << "   Transpose result with opposite storage order:\n" << tosres_ << "\n"
//           << "   Expected result:\n" << trefres_ << "\n";
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
/*!\brief Initializing the non-transpose result matrices.
//
// \return void
//
// This function is called before each non-transpose test case to initialize the according result
// matrices to random values.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::initResults()
{
   const blaze::UnderlyingBuiltin_t<DRE> min( randmin );
   const blaze::UnderlyingBuiltin_t<DRE> max( randmax );

   resize( dres_, E, rows( mat_ ), columns( mat_ ) );
   randomize( dres_, min, max );

//    odres_  = dres_;
//    sres_   = dres_;
//    osres_  = dres_;
   refres_ = dres_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initializing the transpose result matrices.
//
// \return void
//
// This function is called before each transpose test case to initialize the according result
// matrices to random values.
*/
template< typename MT  // Type of the dense Matrix
        , size_t E >   // Compile time expansion
void OperationTest<MT,E>::initTransposeResults()
{
//    const blaze::UnderlyingBuiltin_t<DRE> min( randmin );
//    const blaze::UnderlyingBuiltin_t<DRE> max( randmax );
//
//    resize( tdres_, E, size( tmat_ ) );
//    randomize( tdres_, min, max );
//
//    todres_  = tdres_;
//    tsres_   = tdres_;
//    tosres_  = tdres_;
//    trefres_ = tdres_;
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
// test. The template argument \a T indicates the type of Matrix operand used for the computations.
*/
template< typename MT   // Type of the dense Matrix
        , size_t E >    // Compile time expansion
template< typename T >  // Type of the Matrix operand
void OperationTest<MT,E>::convertException( const std::exception& ex )
{
   using blaze::IsRowMajorMatrix;

   std::ostringstream oss;
   oss << " Test : " << test_ << "\n"
       << " Error: " << error_ << "\n"
       << " Details:\n"
       << "   Random seed = " << blaze::getSeed() << "\n"
       << "   Dense " << ( IsRowMajorMatrix<T>::value ? ( "row-major" ) : ( "column-major" ) ) << " Matrix type:\n"
       << "     " << typeid( T ).name() << "\n"
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
/*!\brief Testing the expansion operation for a specific Matrix type.
//
// \param creator The creator for the dense Matrix.
// \return void
*/
template< typename MT >  // Type of the dense Matrix
void runTest( const Creator<MT>& creator )
{
   for( size_t rep=0UL; rep<repetitions; ++rep ) {
      OperationTest<MT,3UL>{ creator };
      OperationTest<MT,6UL>{ creator };
      OperationTest<MT,7UL>{ creator };
      OperationTest<MT,16UL>{ creator };
      OperationTest<MT,17UL>{ creator };
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  MACRO DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the definition of a dense Matrix expansion operation test case.
*/
#define DEFINE_DMATEXPAND_OPERATION_TEST( MT ) \
   extern template class blazetest::mathtest::dmatexpand::OperationTest<MT>
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of a dense Matrix expansion operation test case.
*/
#define RUN_DMATEXPAND_OPERATION_TEST( C ) \
   blazetest::mathtest::dmatexpand::runTest( C )
/*! \endcond */
//*************************************************************************************************

} // namespace dmatexpand

} // namespace mathtest

} // namespace blazetest

#endif
