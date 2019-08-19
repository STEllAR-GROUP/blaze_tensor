//=================================================================================================
/*!
//  \file blazetest/mathtest/dtensravel/OperationTest.h
//  \brief Header file for the dense tensor ravel operation test
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

#ifndef _BLAZETEST_MATHTEST_DTENSRAVEL_OPERATIONTEST_H_
#define _BLAZETEST_MATHTEST_DTENSRAVEL_OPERATIONTEST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <blaze/math/Aliases.h>
// #include <blaze/math/CompressedTensor.h>
#include <blaze/math/CompressedVector.h>
#include <blaze/math/Views.h>
// #include <blaze/math/constraints/SparseTensor.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/functors/Add.h>
#include <blaze/math/typetraits/IsUniform.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/math/typetraits/UnderlyingNumeric.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/Random.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/util/mpl/Not.h>
#include <blaze/util/typetraits/Decay.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/traits/RavelTrait.h>

#include <blazetest/mathtest/Creator.h>
#include <blazetest/mathtest/IsEqual.h>
#include <blazetest/system/MathTest.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace blazetest {

namespace mathtest {

namespace dtensravel {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class template for the dense tensor ravel operation test.
//
// This class template represents one particular test of a ravel operation on a
// tensor of a particular type. The template argument \a TT represents the type of the tensor
// operand.
*/
template< typename TT >  // Type of the dense tensor
class OperationTest
{
 private:
   //**Type definitions****************************************************************************
   using ET = blaze::ElementType_t<TT>;  //!< Element type.

//    using OTT  = blaze::OppositeType_t<TT>;    //!< Tensor type with opposite storage order.
//    using TTT  = blaze::TransposeType_t<TT>;   //!< Transpose tensor type.
//    using TOTT = blaze::TransposeType_t<OTT>;  //!< Transpose tensor type with opposite storage order.

   //! Dense vector result type of the ravel operation.
   using DRE = blaze::RavelTrait_t<TT>;

   using DET  = blaze::ElementType_t<DRE>;    //!< Element type of the dense result.
   using TDRE = blaze::TransposeType_t<DRE>;  //!< Transpose dense result type.

   //! Sparse vector result type of the ravel operation.
   using SRE = blaze::CompressedVector<DET,true>;

   using SET  = blaze::ElementType_t<SRE>;    //!< Element type of the sparse result.
   using TSRE = blaze::TransposeType_t<SRE>;  //!< Transpose sparse result type.

   using RT = blaze::DynamicTensor<ET>;  //!< Reference type.

   //! Reference result type for ravel operations
   using RRE = blaze::CompressedVector<DET,true>;

   //! Transpose reference result type for ravel operations
   using TRRE = blaze::TransposeType_t<RRE>;

   //! Type of the tensor ravel expression
   using MatRavelExprType = blaze::Decay_t< decltype( blaze::ravel( std::declval<TT>() ) ) >;

   //! Type of the transpose tensor ravel expression
//    using TMatRavelExprType = blaze::Decay_t< decltype( blaze::ravel( std::declval<OTT>() ) ) >;
   //**********************************************************************************************

 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit OperationTest( const Creator<TT>& creator );
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
                          void testBasicOperation    ();
                          void testNegatedOperation  ();
   template< typename T > void testScaledOperation   ( T scalar );
                          void testTransOperation    ();
                          void testCTransOperation   ();
                          void testSubvectorOperation( blaze::TrueType  );
                          void testSubvectorOperation( blaze::FalseType );
                          void testElementsOperation ( blaze::TrueType  );
                          void testElementsOperation ( blaze::FalseType );
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
   TT   tens_;      //!< The dense tensor operand.
//    OTT  omat_;     //!< The dense tensor with opposite storage order.
   DRE  dres_;     //!< The dense result vector.
   SRE  sres_;     //!< The sparse result vector.
   RT   reftens_;   //!< The reference tensor.
   RRE  refres_;   //!< The reference result.
   TDRE tdres_;    //!< The transpose dense result vector.
   TSRE tsres_;    //!< The transpose sparse result vector.
   TRRE trefres_;  //!< The transpose reference result.

   std::string test_;   //!< Label of the currently performed test.
   std::string error_;  //!< Description of the current error type.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( TT   );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( OTT  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( TTT  );
//    BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( TOTT );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE ( RT   );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( RRE  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( DRE  );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( SRE  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( TDRE );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( TSRE );

   BLAZE_CONSTRAINT_MUST_BE_TENSOR_TYPE   ( TT   );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( OTT  );
//    BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( TTT  );
//    BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( TOTT );
   BLAZE_CONSTRAINT_MUST_BE_TENSOR_TYPE   ( RT   );
   BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE         ( RRE  );
   BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE         ( DRE  );
   BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE         ( SRE  );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE      ( TDRE );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE      ( TSRE );

//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET , blaze::ElementType_t<OTT>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET , blaze::ElementType_t<TTT>  );
//    BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET , blaze::ElementType_t<TOTT> );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<RRE>  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<DRE>  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<SRE>  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<TDRE> );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<TSRE> );

   BLAZE_CONSTRAINT_VECTORS_MUST_HAVE_SAME_TRANSPOSE_FLAG     ( MatRavelExprType, blaze::ResultType_t<MatRavelExprType>    );
   BLAZE_CONSTRAINT_VECTORS_MUST_HAVE_DIFFERENT_TRANSPOSE_FLAG( MatRavelExprType, blaze::TransposeType_t<MatRavelExprType> );

//    BLAZE_CONSTRAINT_VECTORS_MUST_HAVE_SAME_TRANSPOSE_FLAG     ( TMatRavelExprType, blaze::ResultType_t<TMatRavelExprType>    );
//    BLAZE_CONSTRAINT_VECTORS_MUST_HAVE_DIFFERENT_TRANSPOSE_FLAG( TMatRavelExprType, blaze::TransposeType_t<TMatRavelExprType> );
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
/*!\brief Constructor for the dense tensor ravel operation test.
//
// \param creator The creator for dense tensor operand.
// \param op The ravel operation.
// \exception std::runtime_error Operation error detected.
*/
template< typename TT >  // Type of the dense tensor
OperationTest<TT>::OperationTest( const Creator<TT>& creator )
   : tens_( creator( NoZeros() ) )  // The dense tensor operand
//    , omat_( mat_ )                 // The dense tensor with opposite storage order
   , dres_()                       // The dense result vector
   , sres_()                       // The sparse result vector
   , reftens_( tens_ )               // The reference tensor
   , refres_()                     // The reference result
   , tdres_()                      // The transpose dense result vector
   , tsres_()                      // The transpose sparse result vector
   , trefres_()                    // The transpose reference result
   , test_()                       // Label of the currently performed test
   , error_()                      // Description of the current error type
{
   using namespace blaze;

   using Scalar = UnderlyingNumeric_t<DET>;

   testInitialStatus();
   testAssignment();
   testBasicOperation();
   testScaledOperation( 2 );
   testScaledOperation( 2UL );
   testScaledOperation( 2.0F );
   testScaledOperation( 2.0 );
   testScaledOperation( Scalar( 2 ) );
   testTransOperation();
   testCTransOperation();
   testSubvectorOperation( Not_t< IsUniform<DRE> >() );
   testElementsOperation( Not_t< IsUniform<DRE> >() );
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Tests on the initial status of the tensor.
//
// \return void
// \exception std::runtime_error Initialization error detected.
//
// This function runs tests on the initial status of the tensor. In case any initialization
// error is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT >  // Type of the dense tensor
void OperationTest<TT>::testInitialStatus()
{
   //=====================================================================================
   // Performing initial tests with the row-major types
   //=====================================================================================

   // Checking the number of rows of the dense operand
   if( tens_.rows() != reftens_.rows() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of row-major dense operand\n"
          << " Error: Invalid number of rows\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Detected number of rows = " << tens_.rows() << "\n"
          << "   Expected number of rows = " << reftens_.rows() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the number of columns of the dense operand
   if( tens_.columns() != reftens_.columns() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of row-major dense operand\n"
          << " Error: Invalid number of columns\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Detected number of columns = " << tens_.columns() << "\n"
          << "   Expected number of columns = " << reftens_.columns() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the initialization of the dense operand
   if( !isEqual( tens_, reftens_ ) ) {
      std::ostringstream oss;
      oss << " Test: Initial test of initialization of row-major dense operand\n"
          << " Error: Invalid tensor initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Current initialization:\n" << tens_ << "\n"
          << "   Expected initialization:\n" << reftens_ << "\n";
      throw std::runtime_error( oss.str() );
   }


//    //=====================================================================================
//    // Performing initial tests with the column-major types
//    //=====================================================================================
//
//    // Checking the number of rows of the dense operand
//    if( omat_.rows() != refmat_.rows() ) {
//       std::ostringstream oss;
//       oss << " Test: Initial size comparison of column-major dense operand\n"
//           << " Error: Invalid number of rows\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Row-major dense tensor type:\n"
//           << "     " << typeid( TT ).name() << "\n"
//           << "   Detected number of rows = " << omat_.rows() << "\n"
//           << "   Expected number of rows = " << refmat_.rows() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//
//    // Checking the number of columns of the dense operand
//    if( omat_.columns() != refmat_.columns() ) {
//       std::ostringstream oss;
//       oss << " Test: Initial size comparison of column-major dense operand\n"
//           << " Error: Invalid number of columns\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Row-major dense tensor type:\n"
//           << "     " << typeid( TT ).name() << "\n"
//           << "   Detected number of columns = " << omat_.columns() << "\n"
//           << "   Expected number of columns = " << refmat_.columns() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//
//    // Checking the initialization of the dense operand
//    if( !isEqual( omat_, refmat_ ) ) {
//       std::ostringstream oss;
//       oss << " Test: Initial test of initialization of column-major dense operand\n"
//           << " Error: Invalid tensor initialization\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Row-major dense tensor type:\n"
//           << "     " << typeid( TT ).name() << "\n"
//           << "   Current initialization:\n" << omat_ << "\n"
//           << "   Expected initialization:\n" << refmat_ << "\n";
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
template< typename TT >  // Type of the dense tensor
void OperationTest<TT>::testAssignment()
{
   //=====================================================================================
   // Performing an assignment with the row-major types
   //=====================================================================================

   try {
      tens_ = reftens_;
   }
   catch( std::exception& ex ) {
      std::ostringstream oss;
      oss << " Test: Assignment with the row-major types\n"
          << " Error: Failed assignment\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Error message: " << ex.what() << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( !isEqual( tens_, reftens_ ) ) {
      std::ostringstream oss;
      oss << " Test: Checking the assignment result of row-major dense operand\n"
          << " Error: Invalid tensor initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense tensor type:\n"
          << "     " << typeid( TT ).name() << "\n"
          << "   Current initialization:\n" << tens_ << "\n"
          << "   Expected initialization:\n" << reftens_ << "\n";
      throw std::runtime_error( oss.str() );
   }


//    //=====================================================================================
//    // Performing an assignment with the column-major types
//    //=====================================================================================
//
//    try {
//       omat_ = refmat_;
//    }
//    catch( std::exception& ex ) {
//       std::ostringstream oss;
//       oss << " Test: Assignment with the column-major types\n"
//           << " Error: Failed assignment\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Column-major dense tensor type:\n"
//           << "     " << typeid( OTT ).name() << "\n"
//           << "   Error message: " << ex.what() << "\n";
//       throw std::runtime_error( oss.str() );
//    }
//
//    if( !isEqual( omat_, refmat_ ) ) {
//       std::ostringstream oss;
//       oss << " Test: Checking the assignment result of column-major dense operand\n"
//           << " Error: Invalid tensor initialization\n"
//           << " Details:\n"
//           << "   Random seed = " << blaze::getSeed() << "\n"
//           << "   Column-major dense tensor type:\n"
//           << "     " << typeid( OTT ).name() << "\n"
//           << "   Current initialization:\n" << omat_ << "\n"
//           << "   Expected initialization:\n" << refmat_ << "\n";
//       throw std::runtime_error( oss.str() );
//    }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the plain dense tensor ravel operation.
//
// \param op The ravel operation.
// \return void
// \exception std::runtime_error Reduction error detected.
//
// This function tests the plain ravel operation with plain assignment, addition assignment,
// subtraction assignment, multiplication assignment, and division assignment. In case any error
// resulting from the ravel or the subsequent assignment is detected, a \a std::runtime_error
// exception is thrown.
*/
template< typename TT >  // Type of the dense tensor
void OperationTest<TT>::testBasicOperation()
{
#if BLAZETEST_MATHTEST_TEST_BASIC_OPERATION
   if( BLAZETEST_MATHTEST_TEST_BASIC_OPERATION > 1 )
   {
      using blaze::ravel;


      //=====================================================================================
      // Reduction operation
      //=====================================================================================

      // Reduction operation with the given tensor
      {
         test_  = "Reduction operation with the given tensor";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = ravel( tens_ );
            sres_   = ravel( tens_ );
            refres_ = ravel( reftens_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   = ravel( omat_ );
//             sres_   = ravel( omat_ );
//             refres_ = ravel( refmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Reduction operation with evaluated tensor
      {
         test_  = "Reduction operation with evaluated matrices";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = ravel( eval( tens_ ) );
            sres_   = ravel( eval( tens_ ) );
            refres_ = ravel( eval( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   = ravel( eval( omat_ ) );
//             sres_   = ravel( eval( omat_ ) );
//             refres_ = ravel( eval( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Reduction operation with addition assignment
      //=====================================================================================

      // Reduction operation with addition assignment with the given tensor
      {
         test_  = "Reduction operation with addition assignment with the given tensor";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += ravel( tens_ );
            sres_   += ravel( tens_ );
            refres_ += ravel( reftens_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   += ravel( omat_ );
//             sres_   += ravel( omat_ );
//             refres_ += ravel( refmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Reduction operation with addition assignment with evaluated tensor
      {
         test_  = "Reduction operation with addition assignment with evaluated tensor";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += ravel( eval( tens_ ) );
            sres_   += ravel( eval( tens_ ) );
            refres_ += ravel( eval( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   += ravel( eval( omat_ ) );
//             sres_   += ravel( eval( omat_ ) );
//             refres_ += ravel( eval( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Reduction operation with subtraction assignment
      //=====================================================================================

      // Reduction operation with subtraction assignment with the given tensor
      {
         test_  = "Reduction operation with subtraction assignment with the given tensor";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ravel( tens_ );
            sres_   -= ravel( tens_ );
            refres_ -= ravel( reftens_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   -= ravel( omat_ );
//             sres_   -= ravel( omat_ );
//             refres_ -= ravel( refmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Reduction operation with subtraction assignment with evaluated tensor
      {
         test_  = "Reduction operation with subtraction assignment with evaluated tensor";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ravel( eval( tens_ ) );
            sres_   -= ravel( eval( tens_ ) );
            refres_ -= ravel( eval( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   -= ravel( eval( omat_ ) );
//             sres_   -= ravel( eval( omat_ ) );
//             refres_ -= ravel( eval( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Reduction operation with multiplication assignment
      //=====================================================================================

      // Reduction operation with multiplication assignment with the given tensor
      {
         test_  = "Reduction operation with multiplication assignment with the given tensor";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            dres_   *= ravel( tens_ );
            sres_   *= ravel( tens_ );
            refres_ *= ravel( reftens_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   *= ravel( omat_ );
//             sres_   *= ravel( omat_ );
//             refres_ *= ravel( refmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Reduction operation with multiplication assignment with evaluated tensor
      {
         test_  = "Reduction operation with multiplication assignment with evaluated tensor";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            dres_   *= ravel( eval( tens_ ) );
            sres_   *= ravel( eval( tens_ ) );
            refres_ *= ravel( eval( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   *= ravel( eval( omat_ ) );
//             sres_   *= ravel( eval( omat_ ) );
//             refres_ *= ravel( eval( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Reduction operation with division assignment
      //=====================================================================================

      if( blaze::isDivisor( ravel( tens_ ) ) )
      {
         // Reduction operation with division assignment with the given tensor
         {
            test_  = "Reduction operation with division assignment with the given tensor";
            error_ = "Failed division assignment operation";

            try {
               initResults();
               dres_   /= ravel( tens_ );
               sres_   /= ravel( tens_ );
               refres_ /= ravel( reftens_ );
            }
            catch( std::exception& ex ) {
               convertException<TT>( ex );
            }

            checkResults<TT>();

//             try {
//                initResults();
//                dres_   /= ravel( omat_ );
//                sres_   /= ravel( omat_ );
//                refres_ /= ravel( refmat_ );
//             }
//             catch( std::exception& ex ) {
//                convertException<OTT>( ex );
//             }
//
//             checkResults<OTT>();
         }

         // Reduction operation with division assignment with evaluated tensor
         {
            test_  = "Reduction operation with division assignment with evaluated tensor";
            error_ = "Failed division assignment operation";

            try {
               initResults();
               dres_   /= ravel( eval( tens_ ) );
               sres_   /= ravel( eval( tens_ ) );
               refres_ /= ravel( eval( reftens_ ) );
            }
            catch( std::exception& ex ) {
               convertException<TT>( ex );
            }

            checkResults<TT>();

//             try {
//                initResults();
//                dres_   /= ravel( eval( omat_ ) );
//                sres_   /= ravel( eval( omat_ ) );
//                refres_ /= ravel( eval( refmat_ ) );
//             }
//             catch( std::exception& ex ) {
//                convertException<OTT>( ex );
//             }
//
//             checkResults<OTT>();
         }
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the scaled dense tensor ravel operation.
//
// \param scalar The scalar value.
// \return void
// \exception std::runtime_error Reduction error detected.
//
// This function tests the scaled tensor ravel operation with plain assignment, addition
// assignment, subtraction assignment, multiplication assignment, and division assignment. In
// case any error resulting from the multiplication or the subsequent assignment is detected,
// a \a std::runtime_error exception is thrown.
*/
template< typename TT >  // Type of the dense tensor
template< typename T >   // Type of the scalar
void OperationTest<TT>::testScaledOperation( T scalar )
{
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( T );

   if( scalar == T(0) )
      throw std::invalid_argument( "Invalid scalar parameter" );


#if BLAZETEST_MATHTEST_TEST_SCALED_OPERATION
   if( BLAZETEST_MATHTEST_TEST_SCALED_OPERATION > 1 )
   {
      using blaze::ravel;


      //=====================================================================================
      // Self-scaling (v*=s)
      //=====================================================================================

      // Self-scaling (v*=s)
      {
         test_ = "Self-scaling (v*=s)";

         try {
            dres_   = ravel( tens_ );
            sres_   = dres_;
            refres_ = dres_;

            dres_   *= scalar;
            sres_   *= scalar;
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
            dres_   = ravel( tens_ );
            sres_   = dres_;
            refres_ = dres_;

            dres_   = dres_   * scalar;
            sres_   = sres_   * scalar;
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
            dres_   = ravel( tens_ );
            sres_   = dres_;
            refres_ = dres_;

            dres_   = scalar * dres_;
            sres_   = scalar * sres_;
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
            dres_   = ravel( tens_ );
            sres_   = dres_;
            refres_ = dres_;

            dres_   /= scalar;
            sres_   /= scalar;
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
            dres_   = ravel( tens_ );
            sres_   = dres_;
            refres_ = dres_;

            dres_   = dres_   / scalar;
            sres_   = sres_   / scalar;
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
      // Scaled ravel operation (s*OP)
      //=====================================================================================

      // Scaled ravel operation with the given tensor
      {
         test_  = "Scaled ravel operation with the given tensor (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = scalar * ravel( tens_ );
            sres_   = scalar * ravel( tens_ );
            refres_ = scalar * ravel( reftens_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   = scalar * ravel( omat_ );
//             sres_   = scalar * ravel( omat_ );
//             refres_ = scalar * ravel( refmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Scaled ravel operation with evaluated tensor
      {
         test_  = "Scaled ravel operation with evaluated tensor (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = scalar * ravel( eval( tens_ ) );
            sres_   = scalar * ravel( eval( tens_ ) );
            refres_ = scalar * ravel( eval( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   = scalar * ravel( eval( omat_ ) );
//             sres_   = scalar * ravel( eval( omat_ ) );
//             refres_ = scalar * ravel( eval( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Scaled ravel operation (OP*s)
      //=====================================================================================

      // Scaled ravel operation with the given tensor
      {
         test_  = "Scaled ravel operation with the given tensor (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = ravel( tens_ ) * scalar;
            sres_   = ravel( tens_ ) * scalar;
            refres_ = ravel( reftens_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   = ravel( omat_ ) * scalar;
//             sres_   = ravel( omat_ ) * scalar;
//             refres_ = ravel( refmat_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Scaled ravel operation with evaluated tensor
      {
         test_  = "Scaled ravel operation with evaluated tensor (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = ravel( eval( tens_ ) ) * scalar;
            sres_   = ravel( eval( tens_ ) ) * scalar;
            refres_ = ravel( eval( reftens_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   = ravel( eval( omat_ ) ) * scalar;
//             sres_   = ravel( eval( omat_ ) ) * scalar;
//             refres_ = ravel( eval( refmat_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Scaled ravel operation (OP/s)
      //=====================================================================================

      // Scaled ravel operation with the given tensor
      {
         test_  = "Scaled ravel operation with the given tensor (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = ravel( tens_ ) / scalar;
            sres_   = ravel( tens_ ) / scalar;
            refres_ = ravel( reftens_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   = ravel( omat_ ) / scalar;
//             sres_   = ravel( omat_ ) / scalar;
//             refres_ = ravel( refmat_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Scaled ravel operation with evaluated tensor
      {
         test_  = "Scaled ravel operation with evaluated tensor (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = ravel( eval( tens_ ) ) / scalar;
            sres_   = ravel( eval( tens_ ) ) / scalar;
            refres_ = ravel( eval( reftens_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   = ravel( eval( omat_ ) ) / scalar;
//             sres_   = ravel( eval( omat_ ) ) / scalar;
//             refres_ = ravel( eval( refmat_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Scaled ravel operation with addition assignment (s*OP)
      //=====================================================================================

      // Scaled ravel operation with addition assignment with the given tensor
      {
         test_  = "Scaled ravel operation with addition assignment with the given tensor (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   += scalar * ravel( tens_ );
            sres_   += scalar * ravel( tens_ );
            refres_ += scalar * ravel( reftens_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   += scalar * ravel( omat_ );
//             sres_   += scalar * ravel( omat_ );
//             refres_ += scalar * ravel( refmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Scaled ravel operation with addition assignment with evaluated tensor
      {
         test_  = "Scaled ravel operation with addition assignment with evaluated tensor (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   += scalar * ravel( eval( tens_ ) );
            sres_   += scalar * ravel( eval( tens_ ) );
            refres_ += scalar * ravel( eval( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   += scalar * ravel( eval( omat_ ) );
//             sres_   += scalar * ravel( eval( omat_ ) );
//             refres_ += scalar * ravel( eval( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Scaled ravel operation with addition assignment (OP*s)
      //=====================================================================================

      // Scaled ravel operation with addition assignment with the given tensor
      {
         test_  = "Scaled ravel operation with addition assignment with the given tensor (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   += ravel( tens_ ) * scalar;
            sres_   += ravel( tens_ ) * scalar;
            refres_ += ravel( reftens_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   += ravel( omat_ ) * scalar;
//             sres_   += ravel( omat_ ) * scalar;
//             refres_ += ravel( refmat_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Scaled ravel operation with addition assignment with evaluated tensor
      {
         test_  = "Scaled ravel operation with addition assignment with evaluated tensor (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   += ravel( eval( tens_ ) ) * scalar;
            sres_   += ravel( eval( tens_ ) ) * scalar;
            refres_ += ravel( eval( reftens_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   += ravel( eval( omat_ ) ) * scalar;
//             sres_   += ravel( eval( omat_ ) ) * scalar;
//             refres_ += ravel( eval( refmat_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Scaled ravel operation with addition assignment (OP/s)
      //=====================================================================================

      // Scaled ravel operation with addition assignment with the given tensor
      {
         test_  = "Scaled ravel operation with addition assignment with the given tensor (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   += ravel( tens_ ) / scalar;
            sres_   += ravel( tens_ ) / scalar;
            refres_ += ravel( reftens_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   += ravel( omat_ ) / scalar;
//             sres_   += ravel( omat_ ) / scalar;
//             refres_ += ravel( refmat_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Scaled ravel operation with addition assignment with evaluated tensor
      {
         test_  = "Scaled ravel operation with addition assignment with evaluated tensor (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   += ravel( eval( tens_ ) ) / scalar;
            sres_   += ravel( eval( tens_ ) ) / scalar;
            refres_ += ravel( eval( reftens_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   += ravel( eval( omat_ ) ) / scalar;
//             sres_   += ravel( eval( omat_ ) ) / scalar;
//             refres_ += ravel( eval( refmat_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Scaled ravel operation with subtraction assignment (s*OP)
      //=====================================================================================

      // Scaled ravel operation with subtraction assignment with the given tensor
      {
         test_  = "Scaled ravel operation with subtraction assignment with the given tensor (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   -= scalar * ravel( tens_ );
            sres_   -= scalar * ravel( tens_ );
            refres_ -= scalar * ravel( reftens_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   -= scalar * ravel( omat_ );
//             sres_   -= scalar * ravel( omat_ );
//             refres_ -= scalar * ravel( refmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Scaled ravel operation with subtraction assignment with evaluated tensor
      {
         test_  = "Scaled ravel operation with subtraction assignment with evaluated tensor (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   -= scalar * ravel( eval( tens_ ) );
            sres_   -= scalar * ravel( eval( tens_ ) );
            refres_ -= scalar * ravel( eval( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   -= scalar * ravel( eval( omat_ ) );
//             sres_   -= scalar * ravel( eval( omat_ ) );
//             refres_ -= scalar * ravel( eval( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Scaled ravel operation with subtraction assignment (OP*s)
      //=====================================================================================

      // Scaled ravel operation with subtraction assignment with the given tensor
      {
         test_  = "Scaled ravel operation with subtraction assignment with the given tensor (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   -= ravel( tens_ ) * scalar;
            sres_   -= ravel( tens_ ) * scalar;
            refres_ -= ravel( reftens_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   -= ravel( omat_ ) * scalar;
//             sres_   -= ravel( omat_ ) * scalar;
//             refres_ -= ravel( refmat_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Scaled ravel operation with subtraction assignment with evaluated tensor
      {
         test_  = "Scaled ravel operation with subtraction assignment with evaluated tensor (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   -= ravel( eval( tens_ ) ) * scalar;
            sres_   -= ravel( eval( tens_ ) ) * scalar;
            refres_ -= ravel( eval( reftens_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   -= ravel( eval( omat_ ) ) * scalar;
//             sres_   -= ravel( eval( omat_ ) ) * scalar;
//             refres_ -= ravel( eval( refmat_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Scaled ravel operation with subtraction assignment (OP/s)
      //=====================================================================================

      // Scaled ravel operation with subtraction assignment with the given tensor
      {
         test_  = "Scaled ravel operation with subtraction assignment with the given tensor (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   -= ravel( tens_ ) / scalar;
            sres_   -= ravel( tens_ ) / scalar;
            refres_ -= ravel( reftens_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   -= ravel( omat_ ) / scalar;
//             sres_   -= ravel( omat_ ) / scalar;
//             refres_ -= ravel( refmat_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Scaled ravel operation with subtraction assignment with evaluated tensor
      {
         test_  = "Scaled ravel operation with subtraction assignment with evaluated tensor (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   -= ravel( eval( tens_ ) ) / scalar;
            sres_   -= ravel( eval( tens_ ) ) / scalar;
            refres_ -= ravel( eval( reftens_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   -= ravel( eval( omat_ ) ) / scalar;
//             sres_   -= ravel( eval( omat_ ) ) / scalar;
//             refres_ -= ravel( eval( refmat_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Scaled ravel operation with multiplication assignment (s*OP)
      //=====================================================================================

      // Scaled ravel operation with multiplication assignment with the given tensor
      {
         test_  = "Scaled ravel operation with multiplication assignment with the given tensor (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   *= scalar * ravel( tens_ );
            sres_   *= scalar * ravel( tens_ );
            refres_ *= scalar * ravel( reftens_ );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   *= scalar * ravel( omat_ );
//             sres_   *= scalar * ravel( omat_ );
//             refres_ *= scalar * ravel( refmat_ );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Scaled ravel operation with multiplication assignment with evaluated tensor
      {
         test_  = "Scaled ravel operation with multiplication assignment with evaluated tensor (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   *= scalar * ravel( eval( tens_ ) );
            sres_   *= scalar * ravel( eval( tens_ ) );
            refres_ *= scalar * ravel( eval( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   *= scalar * ravel( eval( omat_ ) );
//             sres_   *= scalar * ravel( eval( omat_ ) );
//             refres_ *= scalar * ravel( eval( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Scaled ravel operation with multiplication assignment (OP*s)
      //=====================================================================================

      // Scaled ravel operation with multiplication assignment with the given tensor
      {
         test_  = "Scaled ravel operation with multiplication assignment with the given tensor (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   *= ravel( tens_ ) * scalar;
            sres_   *= ravel( tens_ ) * scalar;
            refres_ *= ravel( reftens_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   *= ravel( omat_ ) * scalar;
//             sres_   *= ravel( omat_ ) * scalar;
//             refres_ *= ravel( refmat_ ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Scaled ravel operation with multiplication assignment with evaluated tensor
      {
         test_  = "Scaled ravel operation with multiplication assignment with evaluated tensor (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   *= ravel( eval( tens_ ) ) * scalar;
            sres_   *= ravel( eval( tens_ ) ) * scalar;
            refres_ *= ravel( eval( reftens_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   *= ravel( eval( omat_ ) ) * scalar;
//             sres_   *= ravel( eval( omat_ ) ) * scalar;
//             refres_ *= ravel( eval( refmat_ ) ) * scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Scaled ravel operation with multiplication assignment (OP/s)
      //=====================================================================================

      // Scaled ravel operation with multiplication assignment with the given tensor
      {
         test_  = "Scaled ravel operation with multiplication assignment with the given tensor (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   *= ravel( tens_ ) / scalar;
            sres_   *= ravel( tens_ ) / scalar;
            refres_ *= ravel( reftens_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   *= ravel( omat_ ) / scalar;
//             sres_   *= ravel( omat_ ) / scalar;
//             refres_ *= ravel( refmat_ ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }

      // Scaled ravel operation with multiplication assignment with evaluated tensor
      {
         test_  = "Scaled ravel operation with multiplication assignment with evaluated tensor (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   *= ravel( eval( tens_ ) ) / scalar;
            sres_   *= ravel( eval( tens_ ) ) / scalar;
            refres_ *= ravel( eval( reftens_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             dres_   *= ravel( eval( omat_ ) ) / scalar;
//             sres_   *= ravel( eval( omat_ ) ) / scalar;
//             refres_ *= ravel( eval( refmat_ ) ) / scalar;
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkResults<OTT>();
      }


      //=====================================================================================
      // Scaled ravel operation with division assignment (s*OP)
      //=====================================================================================

      if( blaze::isDivisor( ravel( tens_ ) ) )
      {
         // Scaled ravel operation with division assignment with the given tensor
         {
            test_  = "Scaled ravel operation with division assignment with the given tensor (s*OP)";
            error_ = "Failed ravel operation";

            try {
               initResults();
               dres_   /= scalar * ravel( tens_ );
               sres_   /= scalar * ravel( tens_ );
               refres_ /= scalar * ravel( reftens_ );
            }
            catch( std::exception& ex ) {
               convertException<TT>( ex );
            }

            checkResults<TT>();

//             try {
//                initResults();
//                dres_   /= scalar * ravel( omat_ );
//                sres_   /= scalar * ravel( omat_ );
//                refres_ /= scalar * ravel( refmat_ );
//             }
//             catch( std::exception& ex ) {
//                convertException<OTT>( ex );
//             }
//
//             checkResults<OTT>();
         }

         // Scaled ravel operation with division assignment with evaluated tensor
         {
            test_  = "Scaled ravel operation with division assignment with evaluated tensor (s*OP)";
            error_ = "Failed ravel operation";

            try {
               initResults();
               dres_   /= scalar * ravel( eval( tens_ ) );
               sres_   /= scalar * ravel( eval( tens_ ) );
               refres_ /= scalar * ravel( eval( reftens_ ) );
            }
            catch( std::exception& ex ) {
               convertException<TT>( ex );
            }

            checkResults<TT>();

//             try {
//                initResults();
//                dres_   /= scalar * ravel( eval( omat_ ) );
//                sres_   /= scalar * ravel( eval( omat_ ) );
//                refres_ /= scalar * ravel( eval( refmat_ ) );
//             }
//             catch( std::exception& ex ) {
//                convertException<OTT>( ex );
//             }
//
//             checkResults<OTT>();
         }
      }


      //=====================================================================================
      // Scaled ravel operation with division assignment (OP*s)
      //=====================================================================================

      if( blaze::isDivisor( ravel( tens_ ) ) )
      {
         // Scaled ravel operation with division assignment with the given tensor
         {
            test_  = "Scaled ravel operation with division assignment with the given tensor (OP*s)";
            error_ = "Failed ravel operation";

            try {
               initResults();
               dres_   /= ravel( tens_ ) * scalar;
               sres_   /= ravel( tens_ ) * scalar;
               refres_ /= ravel( reftens_ ) * scalar;
            }
            catch( std::exception& ex ) {
               convertException<TT>( ex );
            }

            checkResults<TT>();

//             try {
//                initResults();
//                dres_   /= ravel( omat_ ) * scalar;
//                sres_   /= ravel( omat_ ) * scalar;
//                refres_ /= ravel( refmat_ ) * scalar;
//             }
//             catch( std::exception& ex ) {
//                convertException<OTT>( ex );
//             }
//
//             checkResults<OTT>();
         }

         // Scaled ravel operation with division assignment with evaluated tensor
         {
            test_  = "Scaled ravel operation with division assignment with evaluated tensor (OP*s)";
            error_ = "Failed ravel operation";

            try {
               initResults();
               dres_   /= ravel( eval( tens_ ) ) * scalar;
               sres_   /= ravel( eval( tens_ ) ) * scalar;
               refres_ /= ravel( eval( reftens_ ) ) * scalar;
            }
            catch( std::exception& ex ) {
               convertException<TT>( ex );
            }

            checkResults<TT>();

//             try {
//                initResults();
//                dres_   /= ravel( eval( omat_ ) ) * scalar;
//                sres_   /= ravel( eval( omat_ ) ) * scalar;
//                refres_ /= ravel( eval( refmat_ ) ) * scalar;
//             }
//             catch( std::exception& ex ) {
//                convertException<OTT>( ex );
//             }
//
//             checkResults<OTT>();
         }
      }


      //=====================================================================================
      // Scaled ravel operation with division assignment (OP/s)
      //=====================================================================================

      if( blaze::isDivisor( ravel( tens_ ) / scalar ) )
      {
         // Scaled ravel operation with division assignment with the given tensor
         {
            test_  = "Scaled ravel operation with division assignment with the given tensor (OP/s)";
            error_ = "Failed ravel operation";

            try {
               initResults();
               dres_   /= ravel( tens_ ) / scalar;
               sres_   /= ravel( tens_ ) / scalar;
               refres_ /= ravel( reftens_ ) / scalar;
            }
            catch( std::exception& ex ) {
               convertException<TT>( ex );
            }

            checkResults<TT>();

//             try {
//                initResults();
//                dres_   /= ravel( omat_ ) / scalar;
//                sres_   /= ravel( omat_ ) / scalar;
//                refres_ /= ravel( refmat_ ) / scalar;
//             }
//             catch( std::exception& ex ) {
//                convertException<OTT>( ex );
//             }
//
//             checkResults<OTT>();
         }

         // Scaled ravel operation with division assignment with evaluated tensor
         {
            test_  = "Scaled ravel operation with division assignment with evaluated tensor (OP/s)";
            error_ = "Failed ravel operation";

            try {
               initResults();
               dres_   /= ravel( eval( tens_ ) ) / scalar;
               sres_   /= ravel( eval( tens_ ) ) / scalar;
               refres_ /= ravel( eval( reftens_ ) ) / scalar;
            }
            catch( std::exception& ex ) {
               convertException<TT>( ex );
            }

            checkResults<TT>();

//             try {
//                initResults();
//                dres_   /= ravel( eval( omat_ ) ) / scalar;
//                sres_   /= ravel( eval( omat_ ) ) / scalar;
//                refres_ /= ravel( eval( refmat_ ) ) / scalar;
//             }
//             catch( std::exception& ex ) {
//                convertException<OTT>( ex );
//             }
//
//             checkResults<OTT>();
         }
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the transpose dense tensor ravel operation.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the transpose tensor ravel operation with plain assignment,
// addition assignment, subtraction assignment, multiplication assignment, and division
// assignment. In case any error resulting from the multiplication or the subsequent
// assignment is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT >  // Type of the dense tensor
void OperationTest<TT>::testTransOperation()
{
#if BLAZETEST_MATHTEST_TEST_TRANS_OPERATION
   if( BLAZETEST_MATHTEST_TEST_TRANS_OPERATION > 1 )
   {
      using blaze::ravel;


      //=====================================================================================
      // Transpose ravel operation
      //=====================================================================================

      // Transpose ravel operation with the given tensor
      {
         test_  = "Transpose ravel operation with the given tensor";
         error_ = "Failed ravel operation";

         try {
            initTransposeResults();
            tdres_   = trans( ravel( tens_ ) );
            tsres_   = trans( ravel( tens_ ) );
            trefres_ = trans( ravel( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   = trans( ravel( omat_ ) );
//             tsres_   = trans( ravel( omat_ ) );
//             trefres_ = trans( ravel( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }

      // Transpose ravel operation with evaluated tensor
      {
         test_  = "Transpose ravel operation with evaluated tensor";
         error_ = "Failed ravel operation";

         try {
            initTransposeResults();
            tdres_   = trans( ravel( eval( tens_ ) ) );
            tsres_   = trans( ravel( eval( tens_ ) ) );
            trefres_ = trans( ravel( eval( reftens_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   = trans( ravel( eval( omat_ ) ) );
//             tsres_   = trans( ravel( eval( omat_ ) ) );
//             trefres_ = trans( ravel( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }


      //=====================================================================================
      // Transpose ravel operation with addition assignment
      //=====================================================================================

      // Transpose ravel operation with addition assignment with the given tensor
      {
         test_  = "Transpose ravel operation with addition assignment with the given tensor";
         error_ = "Failed addition assignment operation";

         try {
            initTransposeResults();
            tdres_   += trans( ravel( tens_ ) );
            tsres_   += trans( ravel( tens_ ) );
            trefres_ += trans( ravel( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   += trans( ravel( omat_ ) );
//             tsres_   += trans( ravel( omat_ ) );
//             trefres_ += trans( ravel( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }

      // Transpose ravel operation with addition assignment with evaluated tensor
      {
         test_  = "Transpose ravel operation with addition assignment with evaluated tensor";
         error_ = "Failed addition assignment operation";

         try {
            initTransposeResults();
            tdres_   += trans( ravel( eval( tens_ ) ) );
            tsres_   += trans( ravel( eval( tens_ ) ) );
            trefres_ += trans( ravel( eval( reftens_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   += trans( ravel( eval( omat_ ) ) );
//             tsres_   += trans( ravel( eval( omat_ ) ) );
//             trefres_ += trans( ravel( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }


      //=====================================================================================
      // Transpose ravel operation with subtraction assignment
      //=====================================================================================

      // Transpose ravel operation with subtraction assignment with the given tensor
      {
         test_  = "Transpose ravel operation with subtraction assignment with the given tensor";
         error_ = "Failed subtraction assignment operation";

         try {
            initTransposeResults();
            tdres_   -= trans( ravel( tens_ ) );
            tsres_   -= trans( ravel( tens_ ) );
            trefres_ -= trans( ravel( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   -= trans( ravel( omat_ ) );
//             tsres_   -= trans( ravel( omat_ ) );
//             trefres_ -= trans( ravel( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }

      // Transpose ravel operation with subtraction assignment with evaluated tensor
      {
         test_  = "Transpose ravel operation with subtraction assignment with evaluated tensor";
         error_ = "Failed subtraction assignment operation";

         try {
            initTransposeResults();
            tdres_   -= trans( ravel( eval( tens_ ) ) );
            tsres_   -= trans( ravel( eval( tens_ ) ) );
            trefres_ -= trans( ravel( eval( reftens_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   -= trans( ravel( eval( omat_ ) ) );
//             tsres_   -= trans( ravel( eval( omat_ ) ) );
//             trefres_ -= trans( ravel( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }


      //=====================================================================================
      // Transpose ravel operation with multiplication assignment
      //=====================================================================================

      // Transpose ravel operation with multiplication assignment with the given tensor
      {
         test_  = "Transpose ravel operation with multiplication assignment with the given tensor";
         error_ = "Failed multiplication assignment operation";

         try {
            initTransposeResults();
            tdres_   *= trans( ravel( tens_ ) );
            tsres_   *= trans( ravel( tens_ ) );
            trefres_ *= trans( ravel( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   *= trans( ravel( omat_ ) );
//             tsres_   *= trans( ravel( omat_ ) );
//             trefres_ *= trans( ravel( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }

      // Transpose ravel operation with multiplication assignment with evaluated tensor
      {
         test_  = "Transpose ravel operation with multiplication assignment with evaluated tensor";
         error_ = "Failed multiplication assignment operation";

         try {
            initTransposeResults();
            tdres_   *= trans( ravel( eval( tens_ ) ) );
            tsres_   *= trans( ravel( eval( tens_ ) ) );
            trefres_ *= trans( ravel( eval( reftens_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   *= trans( ravel( eval( omat_ ) ) );
//             tsres_   *= trans( ravel( eval( omat_ ) ) );
//             trefres_ *= trans( ravel( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }


      //=====================================================================================
      // Transpose ravel operation with division assignment
      //=====================================================================================

      if( blaze::isDivisor( ravel( tens_ ) ) )
      {
         // Transpose ravel operation with division assignment with the given tensor
         {
            test_  = "Transpose ravel operation with division assignment with the given tensor";
            error_ = "Failed division assignment operation";

            try {
               initTransposeResults();
               tdres_   /= trans( ravel( tens_ ) );
               tsres_   /= trans( ravel( tens_ ) );
               trefres_ /= trans( ravel( reftens_ ) );
            }
            catch( std::exception& ex ) {
               convertException<TT>( ex );
            }

            checkTransposeResults<TT>();

//             try {
//                initTransposeResults();
//                tdres_   /= trans( ravel( omat_ ) );
//                tsres_   /= trans( ravel( omat_ ) );
//                trefres_ /= trans( ravel( refmat_ ) );
//             }
//             catch( std::exception& ex ) {
//                convertException<OTT>( ex );
//             }
//
//             checkTransposeResults<OTT>();
         }

         // Transpose ravel operation with division assignment with evaluated tensor
         {
            test_  = "Transpose ravel operation with division assignment with evaluated tensor";
            error_ = "Failed division assignment operation";

            try {
               initTransposeResults();
               tdres_   /= trans( ravel( eval( tens_ ) ) );
               tsres_   /= trans( ravel( eval( tens_ ) ) );
               trefres_ /= trans( ravel( eval( reftens_ ) ) );
            }
            catch( std::exception& ex ) {
               convertException<TT>( ex );
            }

            checkTransposeResults<TT>();

//             try {
//                initTransposeResults();
//                tdres_   /= trans( ravel( eval( omat_ ) ) );
//                tsres_   /= trans( ravel( eval( omat_ ) ) );
//                trefres_ /= trans( ravel( eval( refmat_ ) ) );
//             }
//             catch( std::exception& ex ) {
//                convertException<OTT>( ex );
//             }
//
//             checkTransposeResults<OTT>();
         }
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
// This function tests the conjugate transpose tensor ravel operation with plain
// assignment, addition assignment, subtraction assignment, multiplication assignment,
// and division assignment. In case any error resulting from the multiplication or the
// subsequent assignment is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT >  // Type of the dense tensor
void OperationTest<TT>::testCTransOperation()
{
#if BLAZETEST_MATHTEST_TEST_CTRANS_OPERATION
   if( BLAZETEST_MATHTEST_TEST_CTRANS_OPERATION > 1 )
   {
      using blaze::ravel;


      //=====================================================================================
      // Conjugate transpose ravel operation
      //=====================================================================================

      // Conjugate transpose ravel operation with the given tensor
      {
         test_  = "Conjugate transpose ravel operation with the given tensor";
         error_ = "Failed ravel operation";

         try {
            initTransposeResults();
            tdres_   = ctrans( ravel( tens_ ) );
            tsres_   = ctrans( ravel( tens_ ) );
            trefres_ = ctrans( ravel( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   = ctrans( ravel( omat_ ) );
//             tsres_   = ctrans( ravel( omat_ ) );
//             trefres_ = ctrans( ravel( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }

      // Conjugate transpose ravel operation with evaluated tensor
      {
         test_  = "Conjugate transpose ravel operation with evaluated tensor";
         error_ = "Failed ravel operation";

         try {
            initTransposeResults();
            tdres_   = ctrans( ravel( eval( tens_ ) ) );
            tsres_   = ctrans( ravel( eval( tens_ ) ) );
            trefres_ = ctrans( ravel( eval( reftens_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   = ctrans( ravel( eval( omat_ ) ) );
//             tsres_   = ctrans( ravel( eval( omat_ ) ) );
//             trefres_ = ctrans( ravel( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }


      //=====================================================================================
      // Conjugate transpose ravel operation with addition assignment
      //=====================================================================================

      // Conjugate transpose ravel operation with addition assignment with the given tensor
      {
         test_  = "Conjugate transpose ravel operation with addition assignment with the given tensor";
         error_ = "Failed addition assignment operation";

         try {
            initTransposeResults();
            tdres_   += ctrans( ravel( tens_ ) );
            tsres_   += ctrans( ravel( tens_ ) );
            trefres_ += ctrans( ravel( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   += ctrans( ravel( omat_ ) );
//             tsres_   += ctrans( ravel( omat_ ) );
//             trefres_ += ctrans( ravel( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }

      // Conjugate transpose ravel operation with addition assignment with evaluated tensor
      {
         test_  = "Conjugate transpose ravel operation with addition assignment with evaluated tensor";
         error_ = "Failed addition assignment operation";

         try {
            initTransposeResults();
            tdres_   += ctrans( ravel( eval( tens_ ) ) );
            tsres_   += ctrans( ravel( eval( tens_ ) ) );
            trefres_ += ctrans( ravel( eval( reftens_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   += ctrans( ravel( eval( omat_ ) ) );
//             tsres_   += ctrans( ravel( eval( omat_ ) ) );
//             trefres_ += ctrans( ravel( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }


      //=====================================================================================
      // Conjugate transpose ravel operation with subtraction assignment
      //=====================================================================================

      // Conjugate transpose ravel operation with subtraction assignment with the given tensor
      {
         test_  = "Conjugate transpose ravel operation with subtraction assignment with the given tensor";
         error_ = "Failed subtraction assignment operation";

         try {
            initTransposeResults();
            tdres_   -= ctrans( ravel( tens_ ) );
            tsres_   -= ctrans( ravel( tens_ ) );
            trefres_ -= ctrans( ravel( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   -= ctrans( ravel( omat_ ) );
//             tsres_   -= ctrans( ravel( omat_ ) );
//             trefres_ -= ctrans( ravel( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }

      // Conjugate transpose ravel operation with subtraction assignment with evaluated tensor
      {
         test_  = "Conjugate transpose ravel operation with subtraction assignment with evaluated tensor";
         error_ = "Failed subtraction assignment operation";

         try {
            initTransposeResults();
            tdres_   -= ctrans( ravel( eval( tens_ ) ) );
            tsres_   -= ctrans( ravel( eval( tens_ ) ) );
            trefres_ -= ctrans( ravel( eval( reftens_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   -= ctrans( ravel( eval( omat_ ) ) );
//             tsres_   -= ctrans( ravel( eval( omat_ ) ) );
//             trefres_ -= ctrans( ravel( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }


      //=====================================================================================
      // Conjugate transpose ravel operation with multiplication assignment
      //=====================================================================================

      // Conjugate transpose ravel operation with multiplication assignment with the given tensor
      {
         test_  = "Conjugate transpose ravel operation with multiplication assignment with the given tensor";
         error_ = "Failed multiplication assignment operation";

         try {
            initTransposeResults();
            tdres_   *= ctrans( ravel( tens_ ) );
            tsres_   *= ctrans( ravel( tens_ ) );
            trefres_ *= ctrans( ravel( reftens_ ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   *= ctrans( ravel( omat_ ) );
//             tsres_   *= ctrans( ravel( omat_ ) );
//             trefres_ *= ctrans( ravel( refmat_ ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }

      // Conjugate transpose ravel operation with multiplication assignment with evaluated tensor
      {
         test_  = "Conjugate transpose ravel operation with multiplication assignment with evaluated tensor";
         error_ = "Failed multiplication assignment operation";

         try {
            initTransposeResults();
            tdres_   *= ctrans( ravel( eval( tens_ ) ) );
            tsres_   *= ctrans( ravel( eval( tens_ ) ) );
            trefres_ *= ctrans( ravel( eval( reftens_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkTransposeResults<TT>();

//          try {
//             initTransposeResults();
//             tdres_   *= ctrans( ravel( eval( omat_ ) ) );
//             tsres_   *= ctrans( ravel( eval( omat_ ) ) );
//             trefres_ *= ctrans( ravel( eval( refmat_ ) ) );
//          }
//          catch( std::exception& ex ) {
//             convertException<OTT>( ex );
//          }
//
//          checkTransposeResults<OTT>();
      }


      //=====================================================================================
      // Conjugate transpose ravel operation with division assignment
      //=====================================================================================

      if( blaze::isDivisor( ravel( tens_ ) ) )
      {
         // Conjugate transpose ravel operation with division assignment with the given tensor
         {
            test_  = "Conjugate transpose ravel operation with division assignment with the given tensor";
            error_ = "Failed division assignment operation";

            try {
               initTransposeResults();
               tdres_   /= ctrans( ravel( tens_ ) );
               tsres_   /= ctrans( ravel( tens_ ) );
               trefres_ /= ctrans( ravel( reftens_ ) );
            }
            catch( std::exception& ex ) {
               convertException<TT>( ex );
            }

            checkTransposeResults<TT>();

//             try {
//                initTransposeResults();
//                tdres_   /= ctrans( ravel( omat_ ) );
//                tsres_   /= ctrans( ravel( omat_ ) );
//                trefres_ /= ctrans( ravel( refmat_ ) );
//             }
//             catch( std::exception& ex ) {
//                convertException<OTT>( ex );
//             }
//
//             checkTransposeResults<OTT>();
         }

         // Conjugate transpose ravel operation with division assignment with evaluated tensor
         {
            test_  = "Conjugate transpose ravel operation with division assignment with evaluated tensor";
            error_ = "Failed division assignment operation";

            try {
               initTransposeResults();
               tdres_   /= ctrans( ravel( eval( tens_ ) ) );
               tsres_   /= ctrans( ravel( eval( tens_ ) ) );
               trefres_ /= ctrans( ravel( eval( reftens_ ) ) );
            }
            catch( std::exception& ex ) {
               convertException<TT>( ex );
            }

            checkTransposeResults<TT>();

//             try {
//                initTransposeResults();
//                tdres_   /= ctrans( ravel( eval( omat_ ) ) );
//                tsres_   /= ctrans( ravel( eval( omat_ ) ) );
//                trefres_ /= ctrans( ravel( eval( refmat_ ) ) );
//             }
//             catch( std::exception& ex ) {
//                convertException<OTT>( ex );
//             }
//
//             checkTransposeResults<OTT>();
         }
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the subvector-wise dense tensor ravel operation.
//
// \param op The ravel operation.
// \return void
// \exception std::runtime_error Reduction error detected.
//
// This function tests the subvector-wise tensor ravel operation with plain assignment,
// addition assignment, subtraction assignment, multiplication assignment, and division
// assignment. In case any error resulting from the ravel or the subsequent assignment
// is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT >  // Type of the dense tensor
void OperationTest<TT>::testSubvectorOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_SUBVECTOR_OPERATION
   if( BLAZETEST_MATHTEST_TEST_SUBVECTOR_OPERATION > 1 )
   {
      using blaze::ravel;


      if( tens_.rows() == 0UL )
         return;


      //=====================================================================================
      // Subvector-wise ravel operation
      //=====================================================================================

      // Subvector-wise ravel operation with the given tensor
      {
         test_  = "Subvector-wise ravel operation with the given tensor";
         error_ = "Failed ravel operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<tens_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, tens_.rows() - index );
               subvector( dres_  , index, size ) = subvector( ravel( tens_ )   , index, size );
               subvector( sres_  , index, size ) = subvector( ravel( tens_ )   , index, size );
               subvector( refres_, index, size ) = subvector( ravel( reftens_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
//                size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
//                subvector( dres_  , index, size ) = subvector( ravel( omat_ )  , index, size );
//                subvector( sres_  , index, size ) = subvector( ravel( omat_ )  , index, size );
//                subvector( refres_, index, size ) = subvector( ravel( refmat_ ), index, size );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }

      // Subvector-wise ravel operation with evaluated tensor
      {
         test_  = "Subvector-wise ravel operation with evaluated tensor";
         error_ = "Failed ravel operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<tens_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, tens_.rows() - index );
               subvector( dres_  , index, size ) = subvector( ravel( eval( tens_ ) )   , index, size );
               subvector( sres_  , index, size ) = subvector( ravel( eval( tens_ ) )   , index, size );
               subvector( refres_, index, size ) = subvector( ravel( eval( reftens_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
//                size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
//                subvector( dres_  , index, size ) = subvector( ravel( eval( omat_ ) )  , index, size );
//                subvector( sres_  , index, size ) = subvector( ravel( eval( omat_ ) )  , index, size );
//                subvector( refres_, index, size ) = subvector( ravel( eval( refmat_ ) ), index, size );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }


      //=====================================================================================
      // Subvector-wise ravel operation with addition assignment
      //=====================================================================================

      // Subvector-wise ravel operation with addition assignment with the given tensor
      {
         test_  = "Subvector-wise ravel operation with addition assignment with the given tensor";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<tens_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, tens_.rows() - index );
               subvector( dres_  , index, size ) += subvector( ravel( tens_ )   , index, size );
               subvector( sres_  , index, size ) += subvector( ravel( tens_ )   , index, size );
               subvector( refres_, index, size ) += subvector( ravel( reftens_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
//                size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
//                subvector( dres_  , index, size ) += subvector( ravel( omat_ )  , index, size );
//                subvector( sres_  , index, size ) += subvector( ravel( omat_ )  , index, size );
//                subvector( refres_, index, size ) += subvector( ravel( refmat_ ), index, size );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }

      // Subvector-wise ravel operation with addition assignment with evaluated tensor
      {
         test_  = "Subvector-wise ravel operation with addition assignment with evaluated tensor";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<tens_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, tens_.rows() - index );
               subvector( dres_  , index, size ) += subvector( ravel( eval( tens_ ) )   , index, size );
               subvector( sres_  , index, size ) += subvector( ravel( eval( tens_ ) )   , index, size );
               subvector( refres_, index, size ) += subvector( ravel( eval( reftens_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
//                size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
//                subvector( dres_  , index, size ) += subvector( ravel( eval( omat_ ) )  , index, size );
//                subvector( sres_  , index, size ) += subvector( ravel( eval( omat_ ) )  , index, size );
//                subvector( refres_, index, size ) += subvector( ravel( eval( refmat_ ) ), index, size );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }


      //=====================================================================================
      // Subvector-wise ravel operation with subtraction assignment
      //=====================================================================================

      // Subvector-wise ravel operation with subtraction assignment with the given tensor
      {
         test_  = "Subvector-wise ravel operation with subtraction assignment with the given tensor";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<tens_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, tens_.rows() - index );
               subvector( dres_  , index, size ) -= subvector( ravel( tens_ )   , index, size );
               subvector( sres_  , index, size ) -= subvector( ravel( tens_ )   , index, size );
               subvector( refres_, index, size ) -= subvector( ravel( reftens_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
//                size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
//                subvector( dres_  , index, size ) -= subvector( ravel( omat_ )  , index, size );
//                subvector( sres_  , index, size ) -= subvector( ravel( omat_ )  , index, size );
//                subvector( refres_, index, size ) -= subvector( ravel( refmat_ ), index, size );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }

      // Subvector-wise ravel operation with subtraction assignment with evaluated tensor
      {
         test_  = "Subvector-wise ravel operation with subtraction assignment with evaluated tensor";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<tens_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, tens_.rows() - index );
               subvector( dres_  , index, size ) -= subvector( ravel( eval( tens_ ) )   , index, size );
               subvector( sres_  , index, size ) -= subvector( ravel( eval( tens_ ) )   , index, size );
               subvector( refres_, index, size ) -= subvector( ravel( eval( reftens_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
//                size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
//                subvector( dres_  , index, size ) -= subvector( ravel( eval( omat_ ) )  , index, size );
//                subvector( sres_  , index, size ) -= subvector( ravel( eval( omat_ ) )  , index, size );
//                subvector( refres_, index, size ) -= subvector( ravel( eval( refmat_ ) ), index, size );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }


      //=====================================================================================
      // Subvector-wise ravel operation with multiplication assignment
      //=====================================================================================

      // Subvector-wise ravel operation with multiplication assignment with the given tensor
      {
         test_  = "Subvector-wise ravel operation with multiplication assignment with the given tensor";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<tens_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, tens_.rows() - index );
               subvector( dres_  , index, size ) *= subvector( ravel( tens_ )   , index, size );
               subvector( sres_  , index, size ) *= subvector( ravel( tens_ )   , index, size );
               subvector( refres_, index, size ) *= subvector( ravel( reftens_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
//                size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
//                subvector( dres_  , index, size ) *= subvector( ravel( omat_ )  , index, size );
//                subvector( sres_  , index, size ) *= subvector( ravel( omat_ )  , index, size );
//                subvector( refres_, index, size ) *= subvector( ravel( refmat_ ), index, size );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }

      // Subvector-wise ravel operation with multiplication assignment with evaluated tensor
      {
         test_  = "Subvector-wise ravel operation with multiplication assignment with evaluated tensor";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<tens_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, tens_.rows() - index );
               subvector( dres_  , index, size ) *= subvector( ravel( eval( tens_ ) )   , index, size );
               subvector( sres_  , index, size ) *= subvector( ravel( eval( tens_ ) )   , index, size );
               subvector( refres_, index, size ) *= subvector( ravel( eval( reftens_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
//                size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
//                subvector( dres_  , index, size ) *= subvector( ravel( eval( omat_ ) )  , index, size );
//                subvector( sres_  , index, size ) *= subvector( ravel( eval( omat_ ) )  , index, size );
//                subvector( refres_, index, size ) *= subvector( ravel( eval( refmat_ ) ), index, size );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }


      //=====================================================================================
      // Subvector-wise ravel operation with division assignment
      //=====================================================================================

      // Subvector-wise ravel operation with division assignment with the given tensor
      {
         test_  = "Subvector-wise ravel operation with division assignment with the given tensor";
         error_ = "Failed division assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<tens_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, tens_.rows() - index );
               if( !blaze::isDivisor( subvector( ravel( tens_ ), index, size ) ) ) continue;
               subvector( dres_  , index, size ) /= subvector( ravel( tens_ )   , index, size );
               subvector( sres_  , index, size ) /= subvector( ravel( tens_ )   , index, size );
               subvector( refres_, index, size ) /= subvector( ravel( reftens_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
//                size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
//                if( !blaze::isDivisor( subvector( ravel( omat_ ), index, size ) ) ) continue;
//                subvector( dres_  , index, size ) /= subvector( ravel( omat_ )  , index, size );
//                subvector( sres_  , index, size ) /= subvector( ravel( omat_ )  , index, size );
//                subvector( refres_, index, size ) /= subvector( ravel( refmat_ ), index, size );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }

      // Subvector-wise ravel operation with division assignment with evaluated tensor
      {
         test_  = "Subvector-wise ravel operation with division assignment with evaluated tensor";
         error_ = "Failed division assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<tens_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, tens_.rows() - index );
               if( !blaze::isDivisor( subvector( ravel( tens_ ), index, size ) ) ) continue;
               subvector( dres_  , index, size ) /= subvector( ravel( eval( tens_ ) )   , index, size );
               subvector( sres_  , index, size ) /= subvector( ravel( eval( tens_ ) )   , index, size );
               subvector( refres_, index, size ) /= subvector( ravel( eval( reftens_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
//                size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
//                if( !blaze::isDivisor( subvector( ravel( omat_ ), index, size ) ) ) continue;
//                subvector( dres_  , index, size ) /= subvector( ravel( eval( omat_ ) )  , index, size );
//                subvector( sres_  , index, size ) /= subvector( ravel( eval( omat_ ) )  , index, size );
//                subvector( refres_, index, size ) /= subvector( ravel( eval( refmat_ ) ), index, size );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the subvector-wise dense tensor ravel operation.
//
// \param op The ravel operation.
// \return void
// \exception std::runtime_error Reduction error detected.
//
// This function is called in case the subvector-wise tensor ravel operation is not
// available for the given tensor type \a TT.
*/
template< typename TT >  // Type of the dense tensor
void OperationTest<TT>::testSubvectorOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the elements-wise dense tensor ravel operation.
//
// \return void
// \exception std::runtime_error Reduction error detected.
//
// This function tests the elements-wise tensor ravel operation with plain assignment,
// addition assignment, subtraction assignment, multiplication assignment, and division
// assignment. In case any error resulting from the ravel or the subsequent assignment
// is detected, a \a std::runtime_error exception is thrown.
*/
template< typename TT >  // Type of the dense tensor
void OperationTest<TT>::testElementsOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_ELEMENTS_OPERATION
   if( BLAZETEST_MATHTEST_TEST_ELEMENTS_OPERATION > 1 )
   {
      using blaze::ravel;


      if( tens_.rows() == 0UL )
         return;


      std::vector<size_t> indices( tens_.rows() );
      std::iota( indices.begin(), indices.end(), 0UL );
      std::random_shuffle( indices.begin(), indices.end() );


      //=====================================================================================
      // Elements-wise ravel operation
      //=====================================================================================

      // Elements-wise ravel operation with the given tensor
      {
         test_  = "Elements-wise ravel operation with the given tensor";
         error_ = "Failed ravel operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) = elements( ravel( tens_ )   , &indices[index], n );
               elements( sres_  , &indices[index], n ) = elements( ravel( tens_ )   , &indices[index], n );
               elements( refres_, &indices[index], n ) = elements( ravel( reftens_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                elements( dres_  , &indices[index], n ) = elements( ravel( omat_ )  , &indices[index], n );
//                elements( sres_  , &indices[index], n ) = elements( ravel( omat_ )  , &indices[index], n );
//                elements( refres_, &indices[index], n ) = elements( ravel( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }

      // Elements-wise ravel operation with evaluated tensor
      {
         test_  = "Elements-wise ravel operation with evaluated tensor";
         error_ = "Failed ravel operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) = elements( eval( ravel( tens_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) = elements( eval( ravel( tens_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) = elements( eval( ravel( reftens_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                elements( dres_  , &indices[index], n ) = elements( eval( ravel( omat_ ) ), &indices[index], n );
//                elements( sres_  , &indices[index], n ) = elements( eval( ravel( omat_ ) ), &indices[index], n );
//                elements( refres_, &indices[index], n ) = elements( eval( ravel( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }


      //=====================================================================================
      // Elements-wise ravel operation with addition assignment
      //=====================================================================================

      // Elements-wise ravel operation with addition assignment with the given tensor
      {
         test_  = "Elements-wise ravel operation with addition assignment with the given tensor";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) += elements( ravel( tens_ )   , &indices[index], n );
               elements( sres_  , &indices[index], n ) += elements( ravel( tens_ )   , &indices[index], n );
               elements( refres_, &indices[index], n ) += elements( ravel( reftens_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                elements( dres_  , &indices[index], n ) += elements( ravel( omat_ )  , &indices[index], n );
//                elements( sres_  , &indices[index], n ) += elements( ravel( omat_ )  , &indices[index], n );
//                elements( refres_, &indices[index], n ) += elements( ravel( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }

      // Elements-wise ravel operation with addition assignment with evaluated tensor
      {
         test_  = "Elements-wise ravel operation with addition assignment with evaluated tensor";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) += elements( eval( ravel( tens_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) += elements( eval( ravel( tens_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) += elements( eval( ravel( reftens_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                elements( dres_  , &indices[index], n ) += elements( eval( ravel( omat_ ) ), &indices[index], n );
//                elements( sres_  , &indices[index], n ) += elements( eval( ravel( omat_ ) ), &indices[index], n );
//                elements( refres_, &indices[index], n ) += elements( eval( ravel( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }


      //=====================================================================================
      // Elements-wise ravel operation with subtraction assignment
      //=====================================================================================

      // Elements-wise ravel operation with subtraction assignment with the given tensor
      {
         test_  = "Elements-wise ravel operation with subtraction assignment with the given tensor";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) -= elements( ravel( tens_ )   , &indices[index], n );
               elements( sres_  , &indices[index], n ) -= elements( ravel( tens_ )   , &indices[index], n );
               elements( refres_, &indices[index], n ) -= elements( ravel( reftens_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                elements( dres_  , &indices[index], n ) -= elements( ravel( omat_ )  , &indices[index], n );
//                elements( sres_  , &indices[index], n ) -= elements( ravel( omat_ )  , &indices[index], n );
//                elements( refres_, &indices[index], n ) -= elements( ravel( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }

      // Elements-wise ravel operation with subtraction assignment with evaluated tensor
      {
         test_  = "Elements-wise ravel operation with subtraction assignment with evaluated tensor";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) -= elements( eval( ravel( tens_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) -= elements( eval( ravel( tens_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) -= elements( eval( ravel( reftens_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                elements( dres_  , &indices[index], n ) -= elements( eval( ravel( omat_ ) ), &indices[index], n );
//                elements( sres_  , &indices[index], n ) -= elements( eval( ravel( omat_ ) ), &indices[index], n );
//                elements( refres_, &indices[index], n ) -= elements( eval( ravel( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }


      //=====================================================================================
      // Elements-wise ravel operation with multiplication assignment
      //=====================================================================================

      // Elements-wise ravel operation with multiplication assignment with the given tensor
      {
         test_  = "Elements-wise ravel operation with multiplication assignment with the given tensor";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) *= elements( ravel( tens_ )   , &indices[index], n );
               elements( sres_  , &indices[index], n ) *= elements( ravel( tens_ )   , &indices[index], n );
               elements( refres_, &indices[index], n ) *= elements( ravel( reftens_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                elements( dres_  , &indices[index], n ) *= elements( ravel( omat_ )  , &indices[index], n );
//                elements( sres_  , &indices[index], n ) *= elements( ravel( omat_ )  , &indices[index], n );
//                elements( refres_, &indices[index], n ) *= elements( ravel( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }

      // Elements-wise ravel operation with multiplication assignment with evaluated tensor
      {
         test_  = "Elements-wise ravel operation with multiplication assignment with evaluated tensor";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) *= elements( eval( ravel( tens_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) *= elements( eval( ravel( tens_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) *= elements( eval( ravel( reftens_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                elements( dres_  , &indices[index], n ) *= elements( eval( ravel( omat_ ) ), &indices[index], n );
//                elements( sres_  , &indices[index], n ) *= elements( eval( ravel( omat_ ) ), &indices[index], n );
//                elements( refres_, &indices[index], n ) *= elements( eval( ravel( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }


      //=====================================================================================
      // Elements-wise ravel operation with division assignment
      //=====================================================================================

      // Elements-wise ravel operation with division assignment with the given tensor
      {
         test_  = "Elements-wise ravel operation with division assignment with the given tensor";
         error_ = "Failed division assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               if( !blaze::isDivisor( elements( ravel( tens_ ), &indices[index], n ) ) ) continue;
               elements( dres_  , &indices[index], n ) /= elements( ravel( tens_ )   , &indices[index], n );
               elements( sres_  , &indices[index], n ) /= elements( ravel( tens_ )   , &indices[index], n );
               elements( refres_, &indices[index], n ) /= elements( ravel( reftens_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                if( !blaze::isDivisor( elements( ravel( omat_ ), &indices[index], n ) ) ) continue;
//                elements( dres_  , &indices[index], n ) /= elements( ravel( omat_ )  , &indices[index], n );
//                elements( sres_  , &indices[index], n ) /= elements( ravel( omat_ )  , &indices[index], n );
//                elements( refres_, &indices[index], n ) /= elements( ravel( refmat_ ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }

      // Elements-wise ravel operation with division assignment with evaluated tensor
      {
         test_  = "Elements-wise ravel operation with division assignment with evaluated tensor";
         error_ = "Failed division assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               if( !blaze::isDivisor( elements( ravel( tens_ ), &indices[index], n ) ) ) continue;
               elements( dres_  , &indices[index], n ) /= elements( eval( ravel( tens_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) /= elements( eval( ravel( tens_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) /= elements( eval( ravel( reftens_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TT>( ex );
         }

         checkResults<TT>();

//          try {
//             initResults();
//             for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
//                n = blaze::rand<size_t>( 1UL, indices.size() - index );
//                if( !blaze::isDivisor( elements( ravel( omat_ ), &indices[index], n ) ) ) continue;
//                elements( dres_  , &indices[index], n ) /= elements( eval( ravel( omat_ ) ), &indices[index], n );
//                elements( sres_  , &indices[index], n ) /= elements( eval( ravel( omat_ ) ), &indices[index], n );
//                elements( refres_, &indices[index], n ) /= elements( eval( ravel( refmat_ ) ), &indices[index], n );
//             }
//          }
//          catch( std::exception& ex ) {
//             convertException<TTT>( ex );
//          }
//
//          checkResults<TTT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the elements-wise dense tensor ravel operation.
//
// \param op The ravel operation.
// \return void
// \exception std::runtime_error Reduction error detected.
//
// This function is called in case the elements-wise tensor ravel operation is not
// available for the given tensor type \a TT.
*/
template< typename TT >  // Type of the dense tensor
void OperationTest<TT>::testElementsOperation( blaze::FalseType )
{}
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
// \exception std::runtime_error Incorrect result detected.
//
// This function is called after each test case to check and compare the computed results.
*/
template< typename TT >  // Type of the dense tensor
template< typename T >   // Type of the operand
void OperationTest<TT>::checkResults()
{
   using blaze::IsRowMajorTensor;

   if( !isEqual( dres_, refres_ ) ) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect dense result detected\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   " << ( IsRowMajorTensor<T>::value ? ( "Row-major" ) : ( "Column-major" ) ) << " dense tensor type:\n"
          << "     " << typeid( T ).name() << "\n"
          << "   Result:\n" << dres_ << "\n"
          << "   Expected result:\n" << refres_ << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( !isEqual( sres_, refres_ ) ) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect sparse result detected\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   " << ( IsRowMajorTensor<T>::value ? ( "Row-major" ) : ( "Column-major" ) ) << " dense tensor type:\n"
          << "     " << typeid( T ).name() << "\n"
          << "   Result:\n" << sres_ << "\n"
          << "   Expected result:\n" << refres_ << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking and comparing the computed transpose results.
//
// \return void
// \exception std::runtime_error Incorrect result detected.
//
// This function is called after each test case to check and compare the computed transpose
// results.
*/
template< typename TT >  // Type of the dense tensor
template< typename T >   // Type of the operand
void OperationTest<TT>::checkTransposeResults()
{
   using blaze::IsRowMajorTensor;

   if( !isEqual( tdres_, trefres_ ) ) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect dense result detected\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   " << ( IsRowMajorTensor<T>::value ? ( "Row-major" ) : ( "Column-major" ) ) << " dense tensor type:\n"
          << "     " << typeid( T ).name() << "\n"
          << "   Transpose result:\n" << tdres_ << "\n"
          << "   Expected transpose result:\n" << trefres_ << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( !isEqual( tsres_, trefres_ ) ) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect sparse result detected\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   " << ( IsRowMajorTensor<T>::value ? ( "Row-major" ) : ( "Column-major" ) ) << " dense tensor type:\n"
          << "     " << typeid( T ).name() << "\n"
          << "   Transpose result:\n" << tsres_ << "\n"
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
/*!\brief Initializing the results.
//
// \return void
//
// This function is called before each non-transpose test case to initialize the according result
// vectors to random values.
*/
template< typename TT >  // Type of the dense tensor
void OperationTest<TT>::initResults()
{
   const blaze::UnderlyingBuiltin_t<DRE> min( randmin );
   const blaze::UnderlyingBuiltin_t<DRE> max( randmax );

   resize( dres_, pages( tens_ ) * rows( tens_ ) * columns( tens_ ) );
   randomize( dres_, min, max );

   sres_   = dres_;
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
template< typename TT >  // Type of the dense tensor
void OperationTest<TT>::initTransposeResults()
{
   const blaze::UnderlyingBuiltin_t<TDRE> min( randmin );
   const blaze::UnderlyingBuiltin_t<TDRE> max( randmax );

   resize( dres_, pages( tens_ ) * rows( tens_ ) * columns( tens_ ) );
   randomize( tdres_, min, max );

   tsres_   = tdres_;
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
// the function extends the given exception message by all available information for the failed
// test.
*/
template< typename TT >  // Type of the dense tensor
template< typename T >   // Type of the operand
void OperationTest<TT>::convertException( const std::exception& ex )
{
   using blaze::IsRowMajorTensor;

   std::ostringstream oss;
   oss << " Test : " << test_ << "\n"
       << " Error: " << error_ << "\n"
       << " Details:\n"
       << "   Random seed = " << blaze::getSeed() << "\n"
       << "   " << ( IsRowMajorTensor<T>::value ? ( "Row-major" ) : ( "Column-major" ) ) << " dense tensor type:\n"
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
/*!\brief Testing the ravel operation for a specific tensor type.
//
// \param creator The creator for the dense tensor.
// \return void
*/
template< typename TT >  // Type of the dense tensor
void runTest( const Creator<TT>& creator )
{
   for( size_t rep=0UL; rep<repetitions; ++rep ) {
      OperationTest<TT>{ creator };
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  MACROS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the definition of a dense tensor ravel operation test case.
*/
#define DEFINE_DTENSRAVEL_OPERATION_TEST( TT ) \
   extern template class blazetest::mathtest::dtensravel::OperationTest<TT>
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of a dense tensor ravel operation test case.
*/
#define RUN_DTENSRAVEL_OPERATION_TEST( C ) \
   blazetest::mathtest::dtensravel::runTest( C )
/*! \endcond */
//*************************************************************************************************

} // namespace dtensravel

} // namespace mathtest

} // namespace blazetest

#endif
