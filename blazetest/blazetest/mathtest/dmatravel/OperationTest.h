//=================================================================================================
/*!
//  \file blazetest/mathtest/dmatravel/OperationTest.h
//  \brief Header file for the dense matrix ravel operation test
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

#ifndef _BLAZETEST_MATHTEST_DMATRAVEL_OPERATIONTEST_H_
#define _BLAZETEST_MATHTEST_DMATRAVEL_OPERATIONTEST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <blaze/math/Aliases.h>
#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/CompressedVector.h>
#include <blaze/math/Views.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/SparseMatrix.h>
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

#include <blaze_tensor/math/expressions/Forward.h>
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

namespace dmatravel {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class template for the dense matrix ravel operation test.
//
// This class template represents one particular test of a ravel operation on a
// matrix of a particular type. The template argument \a MT represents the type of the matrix
// operand.
*/
template< typename MT >  // Type of the dense matrix
class OperationTest
{
 private:
   //**Type definitions****************************************************************************
   using ET = blaze::ElementType_t<MT>;  //!< Element type.

   using OMT  = blaze::OppositeType_t<MT>;    //!< Matrix type with opposite storage order.
   using TMT  = blaze::TransposeType_t<MT>;   //!< Transpose matrix type.
   using TOMT = blaze::TransposeType_t<OMT>;  //!< Transpose matrix type with opposite storage order.

   //! Dense vector result type of the ravel operation.
   using DRE = blaze::RavelTrait_t<MT>;

   using DET  = blaze::ElementType_t<DRE>;    //!< Element type of the dense result.
   using TDRE = blaze::TransposeType_t<DRE>;  //!< Transpose dense result type.

   //! Sparse vector result type of the ravel operation.
   using SRE = blaze::CompressedVector<DET,true>;

   using SET  = blaze::ElementType_t<SRE>;    //!< Element type of the sparse result.
   using TSRE = blaze::TransposeType_t<SRE>;  //!< Transpose sparse result type.

   using RT = blaze::DynamicMatrix<ET,false>;  //!< Reference type.

   //! Reference result type for ravel operations
   using RRE = blaze::CompressedVector<DET,true>;

   //! Transpose reference result type for column-wise ravel operations
   using TRRE = blaze::TransposeType_t<RRE>;

   //! Type of the vector map expression
   using MatRavelExprType = blaze::Decay_t< decltype( blaze::ravel( std::declval<MT>() ) ) >;

   //! Type of the transpose vector map expression
   using TMatRavelExprType = blaze::Decay_t< decltype( blaze::ravel( std::declval<OMT>() ) ) >;
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
   MT   mat_;      //!< The dense matrix operand.
   OMT  omat_;     //!< The dense matrix with opposite storage order.
   DRE  dres_;     //!< The dense result vector.
   SRE  sres_;     //!< The sparse result vector.
   RT   refmat_;   //!< The reference matrix.
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
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( MT   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( OMT  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( TMT  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( TOMT );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE ( RT   );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( RRE  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( DRE  );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( SRE  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( TDRE );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( TSRE );

   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( MT   );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( OMT  );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( TMT  );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( TOMT );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( RT   );
   BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE         ( RRE  );
   BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE         ( DRE  );
   BLAZE_CONSTRAINT_MUST_BE_ROW_VECTOR_TYPE         ( SRE  );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE      ( TDRE );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_VECTOR_TYPE      ( TSRE );

   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET , blaze::ElementType_t<OMT>  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET , blaze::ElementType_t<TMT>  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ET , blaze::ElementType_t<TOMT> );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<RRE>  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<DRE>  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<SRE>  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( DET, blaze::ElementType_t<TDRE> );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( SET, blaze::ElementType_t<TSRE> );

   BLAZE_CONSTRAINT_VECTORS_MUST_HAVE_SAME_TRANSPOSE_FLAG     ( MatRavelExprType, blaze::ResultType_t<MatRavelExprType>    );
   BLAZE_CONSTRAINT_VECTORS_MUST_HAVE_DIFFERENT_TRANSPOSE_FLAG( MatRavelExprType, blaze::TransposeType_t<MatRavelExprType> );

   BLAZE_CONSTRAINT_VECTORS_MUST_HAVE_SAME_TRANSPOSE_FLAG     ( TMatRavelExprType, blaze::ResultType_t<TMatRavelExprType>    );
   BLAZE_CONSTRAINT_VECTORS_MUST_HAVE_DIFFERENT_TRANSPOSE_FLAG( TMatRavelExprType, blaze::TransposeType_t<TMatRavelExprType> );
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
/*!\brief Constructor for the dense matrix ravel operation test.
//
// \param creator The creator for dense matrix operand.
// \param op The ravel operation.
// \exception std::runtime_error Operation error detected.
*/
template< typename MT >  // Type of the dense matrix
OperationTest<MT>::OperationTest( const Creator<MT>& creator )
   : mat_( creator( NoZeros() ) )  // The dense matrix operand
   , omat_( mat_ )                 // The dense matrix with opposite storage order
   , dres_()                       // The dense result vector
   , sres_()                       // The sparse result vector
   , refmat_( mat_ )               // The reference matrix
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
/*!\brief Tests on the initial status of the matrix.
//
// \return void
// \exception std::runtime_error Initialization error detected.
//
// This function runs tests on the initial status of the matrix. In case any initialization
// error is detected, a \a std::runtime_error exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
void OperationTest<MT>::testInitialStatus()
{
   //=====================================================================================
   // Performing initial tests with the row-major types
   //=====================================================================================

   // Checking the number of rows of the dense operand
   if( mat_.rows() != refmat_.rows() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of row-major dense operand\n"
          << " Error: Invalid number of rows\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense matrix type:\n"
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
   if( omat_.rows() != refmat_.rows() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of column-major dense operand\n"
          << " Error: Invalid number of rows\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Detected number of rows = " << omat_.rows() << "\n"
          << "   Expected number of rows = " << refmat_.rows() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the number of columns of the dense operand
   if( omat_.columns() != refmat_.columns() ) {
      std::ostringstream oss;
      oss << " Test: Initial size comparison of column-major dense operand\n"
          << " Error: Invalid number of columns\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Detected number of columns = " << omat_.columns() << "\n"
          << "   Expected number of columns = " << refmat_.columns() << "\n";
      throw std::runtime_error( oss.str() );
   }

   // Checking the initialization of the dense operand
   if( !isEqual( omat_, refmat_ ) ) {
      std::ostringstream oss;
      oss << " Test: Initial test of initialization of column-major dense operand\n"
          << " Error: Invalid matrix initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Row-major dense matrix type:\n"
          << "     " << typeid( MT ).name() << "\n"
          << "   Current initialization:\n" << omat_ << "\n"
          << "   Expected initialization:\n" << refmat_ << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the matrix assignment.
//
// \return void
// \exception std::runtime_error Assignment error detected.
//
// This function tests the matrix assignment. In case any error is detected, a
// \a std::runtime_error exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
void OperationTest<MT>::testAssignment()
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

   try {
      omat_ = refmat_;
   }
   catch( std::exception& ex ) {
      std::ostringstream oss;
      oss << " Test: Assignment with the column-major types\n"
          << " Error: Failed assignment\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Column-major dense matrix type:\n"
          << "     " << typeid( OMT ).name() << "\n"
          << "   Error message: " << ex.what() << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( !isEqual( omat_, refmat_ ) ) {
      std::ostringstream oss;
      oss << " Test: Checking the assignment result of column-major dense operand\n"
          << " Error: Invalid matrix initialization\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   Column-major dense matrix type:\n"
          << "     " << typeid( OMT ).name() << "\n"
          << "   Current initialization:\n" << omat_ << "\n"
          << "   Expected initialization:\n" << refmat_ << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the plain dense matrix ravel operation.
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
template< typename MT >  // Type of the dense matrix
void OperationTest<MT>::testBasicOperation()
{
#if BLAZETEST_MATHTEST_TEST_BASIC_OPERATION
   if( BLAZETEST_MATHTEST_TEST_BASIC_OPERATION > 1 )
   {
      using blaze::ravel;


      //=====================================================================================
      // Reduction operation
      //=====================================================================================

      // Reduction operation with the given matrix
      {
         test_  = "Reduction operation with the given matrix";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = ravel( mat_ );
            sres_   = ravel( mat_ );
            refres_ = ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   = ravel( omat_ );
            sres_   = ravel( omat_ );
            refres_ = ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Reduction operation with evaluated matrix
      {
         test_  = "Reduction operation with evaluated matrices";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = ravel( eval( mat_ ) );
            sres_   = ravel( eval( mat_ ) );
            refres_ = ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   = ravel( eval( omat_ ) );
            sres_   = ravel( eval( omat_ ) );
            refres_ = ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Reduction operation with addition assignment
      //=====================================================================================

      // Reduction operation with addition assignment with the given matrix
      {
         test_  = "Reduction operation with addition assignment with the given matrix";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += ravel( mat_ );
            sres_   += ravel( mat_ );
            refres_ += ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   += ravel( omat_ );
            sres_   += ravel( omat_ );
            refres_ += ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Reduction operation with addition assignment with evaluated matrix
      {
         test_  = "Reduction operation with addition assignment with evaluated matrix";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            dres_   += ravel( eval( mat_ ) );
            sres_   += ravel( eval( mat_ ) );
            refres_ += ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   += ravel( eval( omat_ ) );
            sres_   += ravel( eval( omat_ ) );
            refres_ += ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Reduction operation with subtraction assignment
      //=====================================================================================

      // Reduction operation with subtraction assignment with the given matrix
      {
         test_  = "Reduction operation with subtraction assignment with the given matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ravel( mat_ );
            sres_   -= ravel( mat_ );
            refres_ -= ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   -= ravel( omat_ );
            sres_   -= ravel( omat_ );
            refres_ -= ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Reduction operation with subtraction assignment with evaluated matrix
      {
         test_  = "Reduction operation with subtraction assignment with evaluated matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            dres_   -= ravel( eval( mat_ ) );
            sres_   -= ravel( eval( mat_ ) );
            refres_ -= ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   -= ravel( eval( omat_ ) );
            sres_   -= ravel( eval( omat_ ) );
            refres_ -= ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Reduction operation with multiplication assignment
      //=====================================================================================

      // Reduction operation with multiplication assignment with the given matrix
      {
         test_  = "Reduction operation with multiplication assignment with the given matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            dres_   *= ravel( mat_ );
            sres_   *= ravel( mat_ );
            refres_ *= ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   *= ravel( omat_ );
            sres_   *= ravel( omat_ );
            refres_ *= ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Reduction operation with multiplication assignment with evaluated matrix
      {
         test_  = "Reduction operation with multiplication assignment with evaluated matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            dres_   *= ravel( eval( mat_ ) );
            sres_   *= ravel( eval( mat_ ) );
            refres_ *= ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   *= ravel( eval( omat_ ) );
            sres_   *= ravel( eval( omat_ ) );
            refres_ *= ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Reduction operation with division assignment
      //=====================================================================================

      if( blaze::isDivisor( ravel( mat_ ) ) )
      {
         // Reduction operation with division assignment with the given matrix
         {
            test_  = "Reduction operation with division assignment with the given matrix";
            error_ = "Failed division assignment operation";

            try {
               initResults();
               dres_   /= ravel( mat_ );
               sres_   /= ravel( mat_ );
               refres_ /= ravel( refmat_ );
            }
            catch( std::exception& ex ) {
               convertException<MT>( ex );
            }

            checkResults<MT>();

            try {
               initResults();
               dres_   /= ravel( omat_ );
               sres_   /= ravel( omat_ );
               refres_ /= ravel( refmat_ );
            }
            catch( std::exception& ex ) {
               convertException<OMT>( ex );
            }

            checkResults<OMT>();
         }

         // Reduction operation with division assignment with evaluated matrix
         {
            test_  = "Reduction operation with division assignment with evaluated matrix";
            error_ = "Failed division assignment operation";

            try {
               initResults();
               dres_   /= ravel( eval( mat_ ) );
               sres_   /= ravel( eval( mat_ ) );
               refres_ /= ravel( eval( refmat_ ) );
            }
            catch( std::exception& ex ) {
               convertException<MT>( ex );
            }

            checkResults<MT>();

            try {
               initResults();
               dres_   /= ravel( eval( omat_ ) );
               sres_   /= ravel( eval( omat_ ) );
               refres_ /= ravel( eval( refmat_ ) );
            }
            catch( std::exception& ex ) {
               convertException<OMT>( ex );
            }

            checkResults<OMT>();
         }
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the scaled dense matrix ravel operation.
//
// \param scalar The scalar value.
// \return void
// \exception std::runtime_error Reduction error detected.
//
// This function tests the scaled matrix ravel operation with plain assignment, addition
// assignment, subtraction assignment, multiplication assignment, and division assignment. In
// case any error resulting from the multiplication or the subsequent assignment is detected,
// a \a std::runtime_error exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
template< typename T >   // Type of the scalar
void OperationTest<MT>::testScaledOperation( T scalar )
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
            dres_   = ravel( mat_ );
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

         checkResults<MT>();
      }


      //=====================================================================================
      // Self-scaling (v=v*s)
      //=====================================================================================

      // Self-scaling (v=v*s)
      {
         test_ = "Self-scaling (v=v*s)";

         try {
            dres_   = ravel( mat_ );
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

         checkResults<MT>();
      }


      //=====================================================================================
      // Self-scaling (v=s*v)
      //=====================================================================================

      // Self-scaling (v=s*v)
      {
         test_ = "Self-scaling (v=s*v)";

         try {
            dres_   = ravel( mat_ );
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

         checkResults<MT>();
      }


      //=====================================================================================
      // Self-scaling (v/=s)
      //=====================================================================================

      // Self-scaling (v/=s)
      {
         test_ = "Self-scaling (v/=s)";

         try {
            dres_   = ravel( mat_ );
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

         checkResults<MT>();
      }


      //=====================================================================================
      // Self-scaling (v=v/s)
      //=====================================================================================

      // Self-scaling (v=v/s)
      {
         test_ = "Self-scaling (v=v/s)";

         try {
            dres_   = ravel( mat_ );
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

         checkResults<MT>();
      }


      //=====================================================================================
      // Scaled ravel operation (s*OP)
      //=====================================================================================

      // Scaled ravel operation with the given matrix
      {
         test_  = "Scaled ravel operation with the given matrix (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = scalar * ravel( mat_ );
            sres_   = scalar * ravel( mat_ );
            refres_ = scalar * ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   = scalar * ravel( omat_ );
            sres_   = scalar * ravel( omat_ );
            refres_ = scalar * ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Scaled ravel operation with evaluated matrix
      {
         test_  = "Scaled ravel operation with evaluated matrix (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = scalar * ravel( eval( mat_ ) );
            sres_   = scalar * ravel( eval( mat_ ) );
            refres_ = scalar * ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   = scalar * ravel( eval( omat_ ) );
            sres_   = scalar * ravel( eval( omat_ ) );
            refres_ = scalar * ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Scaled ravel operation (OP*s)
      //=====================================================================================

      // Scaled ravel operation with the given matrix
      {
         test_  = "Scaled ravel operation with the given matrix (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = ravel( mat_ ) * scalar;
            sres_   = ravel( mat_ ) * scalar;
            refres_ = ravel( refmat_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   = ravel( omat_ ) * scalar;
            sres_   = ravel( omat_ ) * scalar;
            refres_ = ravel( refmat_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Scaled ravel operation with evaluated matrix
      {
         test_  = "Scaled ravel operation with evaluated matrix (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = ravel( eval( mat_ ) ) * scalar;
            sres_   = ravel( eval( mat_ ) ) * scalar;
            refres_ = ravel( eval( refmat_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   = ravel( eval( omat_ ) ) * scalar;
            sres_   = ravel( eval( omat_ ) ) * scalar;
            refres_ = ravel( eval( refmat_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Scaled ravel operation (OP/s)
      //=====================================================================================

      // Scaled ravel operation with the given matrix
      {
         test_  = "Scaled ravel operation with the given matrix (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = ravel( mat_ ) / scalar;
            sres_   = ravel( mat_ ) / scalar;
            refres_ = ravel( refmat_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   = ravel( omat_ ) / scalar;
            sres_   = ravel( omat_ ) / scalar;
            refres_ = ravel( refmat_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Scaled ravel operation with evaluated matrix
      {
         test_  = "Scaled ravel operation with evaluated matrix (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   = ravel( eval( mat_ ) ) / scalar;
            sres_   = ravel( eval( mat_ ) ) / scalar;
            refres_ = ravel( eval( refmat_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   = ravel( eval( omat_ ) ) / scalar;
            sres_   = ravel( eval( omat_ ) ) / scalar;
            refres_ = ravel( eval( refmat_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Scaled ravel operation with addition assignment (s*OP)
      //=====================================================================================

      // Scaled ravel operation with addition assignment with the given matrix
      {
         test_  = "Scaled ravel operation with addition assignment with the given matrix (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   += scalar * ravel( mat_ );
            sres_   += scalar * ravel( mat_ );
            refres_ += scalar * ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   += scalar * ravel( omat_ );
            sres_   += scalar * ravel( omat_ );
            refres_ += scalar * ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Scaled ravel operation with addition assignment with evaluated matrix
      {
         test_  = "Scaled ravel operation with addition assignment with evaluated matrix (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   += scalar * ravel( eval( mat_ ) );
            sres_   += scalar * ravel( eval( mat_ ) );
            refres_ += scalar * ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   += scalar * ravel( eval( omat_ ) );
            sres_   += scalar * ravel( eval( omat_ ) );
            refres_ += scalar * ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Scaled ravel operation with addition assignment (OP*s)
      //=====================================================================================

      // Scaled ravel operation with addition assignment with the given matrix
      {
         test_  = "Scaled ravel operation with addition assignment with the given matrix (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   += ravel( mat_ ) * scalar;
            sres_   += ravel( mat_ ) * scalar;
            refres_ += ravel( refmat_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   += ravel( omat_ ) * scalar;
            sres_   += ravel( omat_ ) * scalar;
            refres_ += ravel( refmat_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Scaled ravel operation with addition assignment with evaluated matrix
      {
         test_  = "Scaled ravel operation with addition assignment with evaluated matrix (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   += ravel( eval( mat_ ) ) * scalar;
            sres_   += ravel( eval( mat_ ) ) * scalar;
            refres_ += ravel( eval( refmat_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   += ravel( eval( omat_ ) ) * scalar;
            sres_   += ravel( eval( omat_ ) ) * scalar;
            refres_ += ravel( eval( refmat_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Scaled ravel operation with addition assignment (OP/s)
      //=====================================================================================

      // Scaled ravel operation with addition assignment with the given matrix
      {
         test_  = "Scaled ravel operation with addition assignment with the given matrix (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   += ravel( mat_ ) / scalar;
            sres_   += ravel( mat_ ) / scalar;
            refres_ += ravel( refmat_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   += ravel( omat_ ) / scalar;
            sres_   += ravel( omat_ ) / scalar;
            refres_ += ravel( refmat_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Scaled ravel operation with addition assignment with evaluated matrix
      {
         test_  = "Scaled ravel operation with addition assignment with evaluated matrix (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   += ravel( eval( mat_ ) ) / scalar;
            sres_   += ravel( eval( mat_ ) ) / scalar;
            refres_ += ravel( eval( refmat_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   += ravel( eval( omat_ ) ) / scalar;
            sres_   += ravel( eval( omat_ ) ) / scalar;
            refres_ += ravel( eval( refmat_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Scaled ravel operation with subtraction assignment (s*OP)
      //=====================================================================================

      // Scaled ravel operation with subtraction assignment with the given matrix
      {
         test_  = "Scaled ravel operation with subtraction assignment with the given matrix (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   -= scalar * ravel( mat_ );
            sres_   -= scalar * ravel( mat_ );
            refres_ -= scalar * ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   -= scalar * ravel( omat_ );
            sres_   -= scalar * ravel( omat_ );
            refres_ -= scalar * ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Scaled ravel operation with subtraction assignment with evaluated matrix
      {
         test_  = "Scaled ravel operation with subtraction assignment with evaluated matrix (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   -= scalar * ravel( eval( mat_ ) );
            sres_   -= scalar * ravel( eval( mat_ ) );
            refres_ -= scalar * ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   -= scalar * ravel( eval( omat_ ) );
            sres_   -= scalar * ravel( eval( omat_ ) );
            refres_ -= scalar * ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Scaled ravel operation with subtraction assignment (OP*s)
      //=====================================================================================

      // Scaled ravel operation with subtraction assignment with the given matrix
      {
         test_  = "Scaled ravel operation with subtraction assignment with the given matrix (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   -= ravel( mat_ ) * scalar;
            sres_   -= ravel( mat_ ) * scalar;
            refres_ -= ravel( refmat_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   -= ravel( omat_ ) * scalar;
            sres_   -= ravel( omat_ ) * scalar;
            refres_ -= ravel( refmat_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Scaled ravel operation with subtraction assignment with evaluated matrix
      {
         test_  = "Scaled ravel operation with subtraction assignment with evaluated matrix (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   -= ravel( eval( mat_ ) ) * scalar;
            sres_   -= ravel( eval( mat_ ) ) * scalar;
            refres_ -= ravel( eval( refmat_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   -= ravel( eval( omat_ ) ) * scalar;
            sres_   -= ravel( eval( omat_ ) ) * scalar;
            refres_ -= ravel( eval( refmat_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Scaled ravel operation with subtraction assignment (OP/s)
      //=====================================================================================

      // Scaled ravel operation with subtraction assignment with the given matrix
      {
         test_  = "Scaled ravel operation with subtraction assignment with the given matrix (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   -= ravel( mat_ ) / scalar;
            sres_   -= ravel( mat_ ) / scalar;
            refres_ -= ravel( refmat_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   -= ravel( omat_ ) / scalar;
            sres_   -= ravel( omat_ ) / scalar;
            refres_ -= ravel( refmat_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Scaled ravel operation with subtraction assignment with evaluated matrix
      {
         test_  = "Scaled ravel operation with subtraction assignment with evaluated matrix (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   -= ravel( eval( mat_ ) ) / scalar;
            sres_   -= ravel( eval( mat_ ) ) / scalar;
            refres_ -= ravel( eval( refmat_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   -= ravel( eval( omat_ ) ) / scalar;
            sres_   -= ravel( eval( omat_ ) ) / scalar;
            refres_ -= ravel( eval( refmat_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Scaled ravel operation with multiplication assignment (s*OP)
      //=====================================================================================

      // Scaled ravel operation with multiplication assignment with the given matrix
      {
         test_  = "Scaled ravel operation with multiplication assignment with the given matrix (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   *= scalar * ravel( mat_ );
            sres_   *= scalar * ravel( mat_ );
            refres_ *= scalar * ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   *= scalar * ravel( omat_ );
            sres_   *= scalar * ravel( omat_ );
            refres_ *= scalar * ravel( refmat_ );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Scaled ravel operation with multiplication assignment with evaluated matrix
      {
         test_  = "Scaled ravel operation with multiplication assignment with evaluated matrix (s*OP)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   *= scalar * ravel( eval( mat_ ) );
            sres_   *= scalar * ravel( eval( mat_ ) );
            refres_ *= scalar * ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   *= scalar * ravel( eval( omat_ ) );
            sres_   *= scalar * ravel( eval( omat_ ) );
            refres_ *= scalar * ravel( eval( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Scaled ravel operation with multiplication assignment (OP*s)
      //=====================================================================================

      // Scaled ravel operation with multiplication assignment with the given matrix
      {
         test_  = "Scaled ravel operation with multiplication assignment with the given matrix (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   *= ravel( mat_ ) * scalar;
            sres_   *= ravel( mat_ ) * scalar;
            refres_ *= ravel( refmat_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   *= ravel( omat_ ) * scalar;
            sres_   *= ravel( omat_ ) * scalar;
            refres_ *= ravel( refmat_ ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Scaled ravel operation with multiplication assignment with evaluated matrix
      {
         test_  = "Scaled ravel operation with multiplication assignment with evaluated matrix (OP*s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   *= ravel( eval( mat_ ) ) * scalar;
            sres_   *= ravel( eval( mat_ ) ) * scalar;
            refres_ *= ravel( eval( refmat_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   *= ravel( eval( omat_ ) ) * scalar;
            sres_   *= ravel( eval( omat_ ) ) * scalar;
            refres_ *= ravel( eval( refmat_ ) ) * scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Scaled ravel operation with multiplication assignment (OP/s)
      //=====================================================================================

      // Scaled ravel operation with multiplication assignment with the given matrix
      {
         test_  = "Scaled ravel operation with multiplication assignment with the given matrix (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   *= ravel( mat_ ) / scalar;
            sres_   *= ravel( mat_ ) / scalar;
            refres_ *= ravel( refmat_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   *= ravel( omat_ ) / scalar;
            sres_   *= ravel( omat_ ) / scalar;
            refres_ *= ravel( refmat_ ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }

      // Scaled ravel operation with multiplication assignment with evaluated matrix
      {
         test_  = "Scaled ravel operation with multiplication assignment with evaluated matrix (OP/s)";
         error_ = "Failed ravel operation";

         try {
            initResults();
            dres_   *= ravel( eval( mat_ ) ) / scalar;
            sres_   *= ravel( eval( mat_ ) ) / scalar;
            refres_ *= ravel( eval( refmat_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            dres_   *= ravel( eval( omat_ ) ) / scalar;
            sres_   *= ravel( eval( omat_ ) ) / scalar;
            refres_ *= ravel( eval( refmat_ ) ) / scalar;
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkResults<OMT>();
      }


      //=====================================================================================
      // Scaled ravel operation with division assignment (s*OP)
      //=====================================================================================

      if( blaze::isDivisor( ravel( mat_ ) ) )
      {
         // Scaled ravel operation with division assignment with the given matrix
         {
            test_  = "Scaled ravel operation with division assignment with the given matrix (s*OP)";
            error_ = "Failed ravel operation";

            try {
               initResults();
               dres_   /= scalar * ravel( mat_ );
               sres_   /= scalar * ravel( mat_ );
               refres_ /= scalar * ravel( refmat_ );
            }
            catch( std::exception& ex ) {
               convertException<MT>( ex );
            }

            checkResults<MT>();

            try {
               initResults();
               dres_   /= scalar * ravel( omat_ );
               sres_   /= scalar * ravel( omat_ );
               refres_ /= scalar * ravel( refmat_ );
            }
            catch( std::exception& ex ) {
               convertException<OMT>( ex );
            }

            checkResults<OMT>();
         }

         // Scaled ravel operation with division assignment with evaluated matrix
         {
            test_  = "Scaled ravel operation with division assignment with evaluated matrix (s*OP)";
            error_ = "Failed ravel operation";

            try {
               initResults();
               dres_   /= scalar * ravel( eval( mat_ ) );
               sres_   /= scalar * ravel( eval( mat_ ) );
               refres_ /= scalar * ravel( eval( refmat_ ) );
            }
            catch( std::exception& ex ) {
               convertException<MT>( ex );
            }

            checkResults<MT>();

            try {
               initResults();
               dres_   /= scalar * ravel( eval( omat_ ) );
               sres_   /= scalar * ravel( eval( omat_ ) );
               refres_ /= scalar * ravel( eval( refmat_ ) );
            }
            catch( std::exception& ex ) {
               convertException<OMT>( ex );
            }

            checkResults<OMT>();
         }
      }


      //=====================================================================================
      // Scaled ravel operation with division assignment (OP*s)
      //=====================================================================================

      if( blaze::isDivisor( ravel( mat_ ) ) )
      {
         // Scaled ravel operation with division assignment with the given matrix
         {
            test_  = "Scaled ravel operation with division assignment with the given matrix (OP*s)";
            error_ = "Failed ravel operation";

            try {
               initResults();
               dres_   /= ravel( mat_ ) * scalar;
               sres_   /= ravel( mat_ ) * scalar;
               refres_ /= ravel( refmat_ ) * scalar;
            }
            catch( std::exception& ex ) {
               convertException<MT>( ex );
            }

            checkResults<MT>();

            try {
               initResults();
               dres_   /= ravel( omat_ ) * scalar;
               sres_   /= ravel( omat_ ) * scalar;
               refres_ /= ravel( refmat_ ) * scalar;
            }
            catch( std::exception& ex ) {
               convertException<OMT>( ex );
            }

            checkResults<OMT>();
         }

         // Scaled ravel operation with division assignment with evaluated matrix
         {
            test_  = "Scaled ravel operation with division assignment with evaluated matrix (OP*s)";
            error_ = "Failed ravel operation";

            try {
               initResults();
               dres_   /= ravel( eval( mat_ ) ) * scalar;
               sres_   /= ravel( eval( mat_ ) ) * scalar;
               refres_ /= ravel( eval( refmat_ ) ) * scalar;
            }
            catch( std::exception& ex ) {
               convertException<MT>( ex );
            }

            checkResults<MT>();

            try {
               initResults();
               dres_   /= ravel( eval( omat_ ) ) * scalar;
               sres_   /= ravel( eval( omat_ ) ) * scalar;
               refres_ /= ravel( eval( refmat_ ) ) * scalar;
            }
            catch( std::exception& ex ) {
               convertException<OMT>( ex );
            }

            checkResults<OMT>();
         }
      }


      //=====================================================================================
      // Scaled ravel operation with division assignment (OP/s)
      //=====================================================================================

      if( blaze::isDivisor( ravel( mat_ ) / scalar ) )
      {
         // Scaled ravel operation with division assignment with the given matrix
         {
            test_  = "Scaled ravel operation with division assignment with the given matrix (OP/s)";
            error_ = "Failed ravel operation";

            try {
               initResults();
               dres_   /= ravel( mat_ ) / scalar;
               sres_   /= ravel( mat_ ) / scalar;
               refres_ /= ravel( refmat_ ) / scalar;
            }
            catch( std::exception& ex ) {
               convertException<MT>( ex );
            }

            checkResults<MT>();

            try {
               initResults();
               dres_   /= ravel( omat_ ) / scalar;
               sres_   /= ravel( omat_ ) / scalar;
               refres_ /= ravel( refmat_ ) / scalar;
            }
            catch( std::exception& ex ) {
               convertException<OMT>( ex );
            }

            checkResults<OMT>();
         }

         // Scaled ravel operation with division assignment with evaluated matrix
         {
            test_  = "Scaled ravel operation with division assignment with evaluated matrix (OP/s)";
            error_ = "Failed ravel operation";

            try {
               initResults();
               dres_   /= ravel( eval( mat_ ) ) / scalar;
               sres_   /= ravel( eval( mat_ ) ) / scalar;
               refres_ /= ravel( eval( refmat_ ) ) / scalar;
            }
            catch( std::exception& ex ) {
               convertException<MT>( ex );
            }

            checkResults<MT>();

            try {
               initResults();
               dres_   /= ravel( eval( omat_ ) ) / scalar;
               sres_   /= ravel( eval( omat_ ) ) / scalar;
               refres_ /= ravel( eval( refmat_ ) ) / scalar;
            }
            catch( std::exception& ex ) {
               convertException<OMT>( ex );
            }

            checkResults<OMT>();
         }
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the transpose dense matrix ravel operation.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the transpose matrix ravel operation with plain assignment,
// addition assignment, subtraction assignment, multiplication assignment, and division
// assignment. In case any error resulting from the multiplication or the subsequent
// assignment is detected, a \a std::runtime_error exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
void OperationTest<MT>::testTransOperation()
{
#if BLAZETEST_MATHTEST_TEST_TRANS_OPERATION
   if( BLAZETEST_MATHTEST_TEST_TRANS_OPERATION > 1 )
   {
      using blaze::ravel;


      //=====================================================================================
      // Transpose ravel operation
      //=====================================================================================

      // Transpose ravel operation with the given matrix
      {
         test_  = "Transpose ravel operation with the given matrix";
         error_ = "Failed ravel operation";

         try {
            initTransposeResults();
            tdres_   = trans( ravel( mat_ ) );
            tsres_   = trans( ravel( mat_ ) );
            trefres_ = trans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   = trans( ravel( omat_ ) );
            tsres_   = trans( ravel( omat_ ) );
            trefres_ = trans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }

      // Transpose ravel operation with evaluated matrix
      {
         test_  = "Transpose ravel operation with evaluated matrix";
         error_ = "Failed ravel operation";

         try {
            initTransposeResults();
            tdres_   = trans( ravel( eval( mat_ ) ) );
            tsres_   = trans( ravel( eval( mat_ ) ) );
            trefres_ = trans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   = trans( ravel( eval( omat_ ) ) );
            tsres_   = trans( ravel( eval( omat_ ) ) );
            trefres_ = trans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }


      //=====================================================================================
      // Transpose ravel operation with addition assignment
      //=====================================================================================

      // Transpose ravel operation with addition assignment with the given matrix
      {
         test_  = "Transpose ravel operation with addition assignment with the given matrix";
         error_ = "Failed addition assignment operation";

         try {
            initTransposeResults();
            tdres_   += trans( ravel( mat_ ) );
            tsres_   += trans( ravel( mat_ ) );
            trefres_ += trans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   += trans( ravel( omat_ ) );
            tsres_   += trans( ravel( omat_ ) );
            trefres_ += trans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }

      // Transpose ravel operation with addition assignment with evaluated matrix
      {
         test_  = "Transpose ravel operation with addition assignment with evaluated matrix";
         error_ = "Failed addition assignment operation";

         try {
            initTransposeResults();
            tdres_   += trans( ravel( eval( mat_ ) ) );
            tsres_   += trans( ravel( eval( mat_ ) ) );
            trefres_ += trans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   += trans( ravel( eval( omat_ ) ) );
            tsres_   += trans( ravel( eval( omat_ ) ) );
            trefres_ += trans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }


      //=====================================================================================
      // Transpose ravel operation with subtraction assignment
      //=====================================================================================

      // Transpose ravel operation with subtraction assignment with the given matrix
      {
         test_  = "Transpose ravel operation with subtraction assignment with the given matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initTransposeResults();
            tdres_   -= trans( ravel( mat_ ) );
            tsres_   -= trans( ravel( mat_ ) );
            trefres_ -= trans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   -= trans( ravel( omat_ ) );
            tsres_   -= trans( ravel( omat_ ) );
            trefres_ -= trans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }

      // Transpose ravel operation with subtraction assignment with evaluated matrix
      {
         test_  = "Transpose ravel operation with subtraction assignment with evaluated matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initTransposeResults();
            tdres_   -= trans( ravel( eval( mat_ ) ) );
            tsres_   -= trans( ravel( eval( mat_ ) ) );
            trefres_ -= trans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   -= trans( ravel( eval( omat_ ) ) );
            tsres_   -= trans( ravel( eval( omat_ ) ) );
            trefres_ -= trans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }


      //=====================================================================================
      // Transpose ravel operation with multiplication assignment
      //=====================================================================================

      // Transpose ravel operation with multiplication assignment with the given matrix
      {
         test_  = "Transpose ravel operation with multiplication assignment with the given matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initTransposeResults();
            tdres_   *= trans( ravel( mat_ ) );
            tsres_   *= trans( ravel( mat_ ) );
            trefres_ *= trans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   *= trans( ravel( omat_ ) );
            tsres_   *= trans( ravel( omat_ ) );
            trefres_ *= trans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }

      // Transpose ravel operation with multiplication assignment with evaluated matrix
      {
         test_  = "Transpose ravel operation with multiplication assignment with evaluated matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initTransposeResults();
            tdres_   *= trans( ravel( eval( mat_ ) ) );
            tsres_   *= trans( ravel( eval( mat_ ) ) );
            trefres_ *= trans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   *= trans( ravel( eval( omat_ ) ) );
            tsres_   *= trans( ravel( eval( omat_ ) ) );
            trefres_ *= trans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }


      //=====================================================================================
      // Transpose ravel operation with division assignment
      //=====================================================================================

      if( blaze::isDivisor( ravel( mat_ ) ) )
      {
         // Transpose ravel operation with division assignment with the given matrix
         {
            test_  = "Transpose ravel operation with division assignment with the given matrix";
            error_ = "Failed division assignment operation";

            try {
               initTransposeResults();
               tdres_   /= trans( ravel( mat_ ) );
               tsres_   /= trans( ravel( mat_ ) );
               trefres_ /= trans( ravel( refmat_ ) );
            }
            catch( std::exception& ex ) {
               convertException<MT>( ex );
            }

            checkTransposeResults<MT>();

            try {
               initTransposeResults();
               tdres_   /= trans( ravel( omat_ ) );
               tsres_   /= trans( ravel( omat_ ) );
               trefres_ /= trans( ravel( refmat_ ) );
            }
            catch( std::exception& ex ) {
               convertException<OMT>( ex );
            }

            checkTransposeResults<OMT>();
         }

         // Transpose ravel operation with division assignment with evaluated matrix
         {
            test_  = "Transpose ravel operation with division assignment with evaluated matrix";
            error_ = "Failed division assignment operation";

            try {
               initTransposeResults();
               tdres_   /= trans( ravel( eval( mat_ ) ) );
               tsres_   /= trans( ravel( eval( mat_ ) ) );
               trefres_ /= trans( ravel( eval( refmat_ ) ) );
            }
            catch( std::exception& ex ) {
               convertException<MT>( ex );
            }

            checkTransposeResults<MT>();

            try {
               initTransposeResults();
               tdres_   /= trans( ravel( eval( omat_ ) ) );
               tsres_   /= trans( ravel( eval( omat_ ) ) );
               trefres_ /= trans( ravel( eval( refmat_ ) ) );
            }
            catch( std::exception& ex ) {
               convertException<OMT>( ex );
            }

            checkTransposeResults<OMT>();
         }
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the conjugate transpose dense matrix/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Multiplication error detected.
//
// This function tests the conjugate transpose matrix ravel operation with plain
// assignment, addition assignment, subtraction assignment, multiplication assignment,
// and division assignment. In case any error resulting from the multiplication or the
// subsequent assignment is detected, a \a std::runtime_error exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
void OperationTest<MT>::testCTransOperation()
{
#if BLAZETEST_MATHTEST_TEST_CTRANS_OPERATION
   if( BLAZETEST_MATHTEST_TEST_CTRANS_OPERATION > 1 )
   {
      using blaze::ravel;


      //=====================================================================================
      // Conjugate transpose ravel operation
      //=====================================================================================

      // Conjugate transpose ravel operation with the given matrix
      {
         test_  = "Conjugate transpose ravel operation with the given matrix";
         error_ = "Failed ravel operation";

         try {
            initTransposeResults();
            tdres_   = ctrans( ravel( mat_ ) );
            tsres_   = ctrans( ravel( mat_ ) );
            trefres_ = ctrans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   = ctrans( ravel( omat_ ) );
            tsres_   = ctrans( ravel( omat_ ) );
            trefres_ = ctrans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }

      // Conjugate transpose ravel operation with evaluated matrix
      {
         test_  = "Conjugate transpose ravel operation with evaluated matrix";
         error_ = "Failed ravel operation";

         try {
            initTransposeResults();
            tdres_   = ctrans( ravel( eval( mat_ ) ) );
            tsres_   = ctrans( ravel( eval( mat_ ) ) );
            trefres_ = ctrans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   = ctrans( ravel( eval( omat_ ) ) );
            tsres_   = ctrans( ravel( eval( omat_ ) ) );
            trefres_ = ctrans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }


      //=====================================================================================
      // Conjugate transpose ravel operation with addition assignment
      //=====================================================================================

      // Conjugate transpose ravel operation with addition assignment with the given matrix
      {
         test_  = "Conjugate transpose ravel operation with addition assignment with the given matrix";
         error_ = "Failed addition assignment operation";

         try {
            initTransposeResults();
            tdres_   += ctrans( ravel( mat_ ) );
            tsres_   += ctrans( ravel( mat_ ) );
            trefres_ += ctrans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   += ctrans( ravel( omat_ ) );
            tsres_   += ctrans( ravel( omat_ ) );
            trefres_ += ctrans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }

      // Conjugate transpose ravel operation with addition assignment with evaluated matrix
      {
         test_  = "Conjugate transpose ravel operation with addition assignment with evaluated matrix";
         error_ = "Failed addition assignment operation";

         try {
            initTransposeResults();
            tdres_   += ctrans( ravel( eval( mat_ ) ) );
            tsres_   += ctrans( ravel( eval( mat_ ) ) );
            trefres_ += ctrans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   += ctrans( ravel( eval( omat_ ) ) );
            tsres_   += ctrans( ravel( eval( omat_ ) ) );
            trefres_ += ctrans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }


      //=====================================================================================
      // Conjugate transpose ravel operation with subtraction assignment
      //=====================================================================================

      // Conjugate transpose ravel operation with subtraction assignment with the given matrix
      {
         test_  = "Conjugate transpose ravel operation with subtraction assignment with the given matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initTransposeResults();
            tdres_   -= ctrans( ravel( mat_ ) );
            tsres_   -= ctrans( ravel( mat_ ) );
            trefres_ -= ctrans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   -= ctrans( ravel( omat_ ) );
            tsres_   -= ctrans( ravel( omat_ ) );
            trefres_ -= ctrans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }

      // Conjugate transpose ravel operation with subtraction assignment with evaluated matrix
      {
         test_  = "Conjugate transpose ravel operation with subtraction assignment with evaluated matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initTransposeResults();
            tdres_   -= ctrans( ravel( eval( mat_ ) ) );
            tsres_   -= ctrans( ravel( eval( mat_ ) ) );
            trefres_ -= ctrans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   -= ctrans( ravel( eval( omat_ ) ) );
            tsres_   -= ctrans( ravel( eval( omat_ ) ) );
            trefres_ -= ctrans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }


      //=====================================================================================
      // Conjugate transpose ravel operation with multiplication assignment
      //=====================================================================================

      // Conjugate transpose ravel operation with multiplication assignment with the given matrix
      {
         test_  = "Conjugate transpose ravel operation with multiplication assignment with the given matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initTransposeResults();
            tdres_   *= ctrans( ravel( mat_ ) );
            tsres_   *= ctrans( ravel( mat_ ) );
            trefres_ *= ctrans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   *= ctrans( ravel( omat_ ) );
            tsres_   *= ctrans( ravel( omat_ ) );
            trefres_ *= ctrans( ravel( refmat_ ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }

      // Conjugate transpose ravel operation with multiplication assignment with evaluated matrix
      {
         test_  = "Conjugate transpose ravel operation with multiplication assignment with evaluated matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initTransposeResults();
            tdres_   *= ctrans( ravel( eval( mat_ ) ) );
            tsres_   *= ctrans( ravel( eval( mat_ ) ) );
            trefres_ *= ctrans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkTransposeResults<MT>();

         try {
            initTransposeResults();
            tdres_   *= ctrans( ravel( eval( omat_ ) ) );
            tsres_   *= ctrans( ravel( eval( omat_ ) ) );
            trefres_ *= ctrans( ravel( eval( refmat_ ) ) );
         }
         catch( std::exception& ex ) {
            convertException<OMT>( ex );
         }

         checkTransposeResults<OMT>();
      }


      //=====================================================================================
      // Conjugate transpose ravel operation with division assignment
      //=====================================================================================

      if( blaze::isDivisor( ravel( mat_ ) ) )
      {
         // Conjugate transpose ravel operation with division assignment with the given matrix
         {
            test_  = "Conjugate transpose ravel operation with division assignment with the given matrix";
            error_ = "Failed division assignment operation";

            try {
               initTransposeResults();
               tdres_   /= ctrans( ravel( mat_ ) );
               tsres_   /= ctrans( ravel( mat_ ) );
               trefres_ /= ctrans( ravel( refmat_ ) );
            }
            catch( std::exception& ex ) {
               convertException<MT>( ex );
            }

            checkTransposeResults<MT>();

            try {
               initTransposeResults();
               tdres_   /= ctrans( ravel( omat_ ) );
               tsres_   /= ctrans( ravel( omat_ ) );
               trefres_ /= ctrans( ravel( refmat_ ) );
            }
            catch( std::exception& ex ) {
               convertException<OMT>( ex );
            }

            checkTransposeResults<OMT>();
         }

         // Conjugate transpose ravel operation with division assignment with evaluated matrix
         {
            test_  = "Conjugate transpose ravel operation with division assignment with evaluated matrix";
            error_ = "Failed division assignment operation";

            try {
               initTransposeResults();
               tdres_   /= ctrans( ravel( eval( mat_ ) ) );
               tsres_   /= ctrans( ravel( eval( mat_ ) ) );
               trefres_ /= ctrans( ravel( eval( refmat_ ) ) );
            }
            catch( std::exception& ex ) {
               convertException<MT>( ex );
            }

            checkTransposeResults<MT>();

            try {
               initTransposeResults();
               tdres_   /= ctrans( ravel( eval( omat_ ) ) );
               tsres_   /= ctrans( ravel( eval( omat_ ) ) );
               trefres_ /= ctrans( ravel( eval( refmat_ ) ) );
            }
            catch( std::exception& ex ) {
               convertException<OMT>( ex );
            }

            checkTransposeResults<OMT>();
         }
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the subvector-wise dense matrix ravel operation.
//
// \param op The ravel operation.
// \return void
// \exception std::runtime_error Reduction error detected.
//
// This function tests the subvector-wise matrix ravel operation with plain assignment,
// addition assignment, subtraction assignment, multiplication assignment, and division
// assignment. In case any error resulting from the ravel or the subsequent assignment
// is detected, a \a std::runtime_error exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
void OperationTest<MT>::testSubvectorOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_SUBVECTOR_OPERATION
   if( BLAZETEST_MATHTEST_TEST_SUBVECTOR_OPERATION > 1 )
   {
      using blaze::ravel;


      if( mat_.rows() == 0UL )
         return;


      //=====================================================================================
      // Subvector-wise ravel operation
      //=====================================================================================

      // Subvector-wise ravel operation with the given matrix
      {
         test_  = "Subvector-wise ravel operation with the given matrix";
         error_ = "Failed ravel operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<mat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, mat_.rows() - index );
               subvector( dres_  , index, size ) = subvector( ravel( mat_ )   , index, size );
               subvector( sres_  , index, size ) = subvector( ravel( mat_ )   , index, size );
               subvector( refres_, index, size ) = subvector( ravel( refmat_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
               subvector( dres_  , index, size ) = subvector( ravel( omat_ )  , index, size );
               subvector( sres_  , index, size ) = subvector( ravel( omat_ )  , index, size );
               subvector( refres_, index, size ) = subvector( ravel( refmat_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }

      // Subvector-wise ravel operation with evaluated matrix
      {
         test_  = "Subvector-wise ravel operation with evaluated matrix";
         error_ = "Failed ravel operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<mat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, mat_.rows() - index );
               subvector( dres_  , index, size ) = subvector( ravel( eval( mat_ ) )   , index, size );
               subvector( sres_  , index, size ) = subvector( ravel( eval( mat_ ) )   , index, size );
               subvector( refres_, index, size ) = subvector( ravel( eval( refmat_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
               subvector( dres_  , index, size ) = subvector( ravel( eval( omat_ ) )  , index, size );
               subvector( sres_  , index, size ) = subvector( ravel( eval( omat_ ) )  , index, size );
               subvector( refres_, index, size ) = subvector( ravel( eval( refmat_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }


      //=====================================================================================
      // Subvector-wise ravel operation with addition assignment
      //=====================================================================================

      // Subvector-wise ravel operation with addition assignment with the given matrix
      {
         test_  = "Subvector-wise ravel operation with addition assignment with the given matrix";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<mat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, mat_.rows() - index );
               subvector( dres_  , index, size ) += subvector( ravel( mat_ )   , index, size );
               subvector( sres_  , index, size ) += subvector( ravel( mat_ )   , index, size );
               subvector( refres_, index, size ) += subvector( ravel( refmat_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
               subvector( dres_  , index, size ) += subvector( ravel( omat_ )  , index, size );
               subvector( sres_  , index, size ) += subvector( ravel( omat_ )  , index, size );
               subvector( refres_, index, size ) += subvector( ravel( refmat_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }

      // Subvector-wise ravel operation with addition assignment with evaluated matrix
      {
         test_  = "Subvector-wise ravel operation with addition assignment with evaluated matrix";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<mat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, mat_.rows() - index );
               subvector( dres_  , index, size ) += subvector( ravel( eval( mat_ ) )   , index, size );
               subvector( sres_  , index, size ) += subvector( ravel( eval( mat_ ) )   , index, size );
               subvector( refres_, index, size ) += subvector( ravel( eval( refmat_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
               subvector( dres_  , index, size ) += subvector( ravel( eval( omat_ ) )  , index, size );
               subvector( sres_  , index, size ) += subvector( ravel( eval( omat_ ) )  , index, size );
               subvector( refres_, index, size ) += subvector( ravel( eval( refmat_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }


      //=====================================================================================
      // Subvector-wise ravel operation with subtraction assignment
      //=====================================================================================

      // Subvector-wise ravel operation with subtraction assignment with the given matrix
      {
         test_  = "Subvector-wise ravel operation with subtraction assignment with the given matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<mat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, mat_.rows() - index );
               subvector( dres_  , index, size ) -= subvector( ravel( mat_ )   , index, size );
               subvector( sres_  , index, size ) -= subvector( ravel( mat_ )   , index, size );
               subvector( refres_, index, size ) -= subvector( ravel( refmat_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
               subvector( dres_  , index, size ) -= subvector( ravel( omat_ )  , index, size );
               subvector( sres_  , index, size ) -= subvector( ravel( omat_ )  , index, size );
               subvector( refres_, index, size ) -= subvector( ravel( refmat_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }

      // Subvector-wise ravel operation with subtraction assignment with evaluated matrix
      {
         test_  = "Subvector-wise ravel operation with subtraction assignment with evaluated matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<mat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, mat_.rows() - index );
               subvector( dres_  , index, size ) -= subvector( ravel( eval( mat_ ) )   , index, size );
               subvector( sres_  , index, size ) -= subvector( ravel( eval( mat_ ) )   , index, size );
               subvector( refres_, index, size ) -= subvector( ravel( eval( refmat_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
               subvector( dres_  , index, size ) -= subvector( ravel( eval( omat_ ) )  , index, size );
               subvector( sres_  , index, size ) -= subvector( ravel( eval( omat_ ) )  , index, size );
               subvector( refres_, index, size ) -= subvector( ravel( eval( refmat_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }


      //=====================================================================================
      // Subvector-wise ravel operation with multiplication assignment
      //=====================================================================================

      // Subvector-wise ravel operation with multiplication assignment with the given matrix
      {
         test_  = "Subvector-wise ravel operation with multiplication assignment with the given matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<mat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, mat_.rows() - index );
               subvector( dres_  , index, size ) *= subvector( ravel( mat_ )   , index, size );
               subvector( sres_  , index, size ) *= subvector( ravel( mat_ )   , index, size );
               subvector( refres_, index, size ) *= subvector( ravel( refmat_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
               subvector( dres_  , index, size ) *= subvector( ravel( omat_ )  , index, size );
               subvector( sres_  , index, size ) *= subvector( ravel( omat_ )  , index, size );
               subvector( refres_, index, size ) *= subvector( ravel( refmat_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }

      // Subvector-wise ravel operation with multiplication assignment with evaluated matrix
      {
         test_  = "Subvector-wise ravel operation with multiplication assignment with evaluated matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<mat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, mat_.rows() - index );
               subvector( dres_  , index, size ) *= subvector( ravel( eval( mat_ ) )   , index, size );
               subvector( sres_  , index, size ) *= subvector( ravel( eval( mat_ ) )   , index, size );
               subvector( refres_, index, size ) *= subvector( ravel( eval( refmat_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
               subvector( dres_  , index, size ) *= subvector( ravel( eval( omat_ ) )  , index, size );
               subvector( sres_  , index, size ) *= subvector( ravel( eval( omat_ ) )  , index, size );
               subvector( refres_, index, size ) *= subvector( ravel( eval( refmat_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }


      //=====================================================================================
      // Subvector-wise ravel operation with division assignment
      //=====================================================================================

      // Subvector-wise ravel operation with division assignment with the given matrix
      {
         test_  = "Subvector-wise ravel operation with division assignment with the given matrix";
         error_ = "Failed division assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<mat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, mat_.rows() - index );
               if( !blaze::isDivisor( subvector( ravel( mat_ ), index, size ) ) ) continue;
               subvector( dres_  , index, size ) /= subvector( ravel( mat_ )   , index, size );
               subvector( sres_  , index, size ) /= subvector( ravel( mat_ )   , index, size );
               subvector( refres_, index, size ) /= subvector( ravel( refmat_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
               if( !blaze::isDivisor( subvector( ravel( omat_ ), index, size ) ) ) continue;
               subvector( dres_  , index, size ) /= subvector( ravel( omat_ )  , index, size );
               subvector( sres_  , index, size ) /= subvector( ravel( omat_ )  , index, size );
               subvector( refres_, index, size ) /= subvector( ravel( refmat_ ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }

      // Subvector-wise ravel operation with division assignment with evaluated matrix
      {
         test_  = "Subvector-wise ravel operation with division assignment with evaluated matrix";
         error_ = "Failed division assignment operation";

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<mat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, mat_.rows() - index );
               if( !blaze::isDivisor( subvector( ravel( mat_ ), index, size ) ) ) continue;
               subvector( dres_  , index, size ) /= subvector( ravel( eval( mat_ ) )   , index, size );
               subvector( sres_  , index, size ) /= subvector( ravel( eval( mat_ ) )   , index, size );
               subvector( refres_, index, size ) /= subvector( ravel( eval( refmat_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, size=0UL; index<omat_.rows(); index+=size ) {
               size = blaze::rand<size_t>( 1UL, omat_.rows() - index );
               if( !blaze::isDivisor( subvector( ravel( omat_ ), index, size ) ) ) continue;
               subvector( dres_  , index, size ) /= subvector( ravel( eval( omat_ ) )  , index, size );
               subvector( sres_  , index, size ) /= subvector( ravel( eval( omat_ ) )  , index, size );
               subvector( refres_, index, size ) /= subvector( ravel( eval( refmat_ ) ), index, size );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the subvector-wise dense matrix ravel operation.
//
// \param op The ravel operation.
// \return void
// \exception std::runtime_error Reduction error detected.
//
// This function is called in case the subvector-wise matrix ravel operation is not
// available for the given matrix type \a MT.
*/
template< typename MT >  // Type of the dense matrix
void OperationTest<MT>::testSubvectorOperation( blaze::FalseType )
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Testing the elements-wise dense matrix ravel operation.
//
// \return void
// \exception std::runtime_error Reduction error detected.
//
// This function tests the elements-wise matrix ravel operation with plain assignment,
// addition assignment, subtraction assignment, multiplication assignment, and division
// assignment. In case any error resulting from the ravel or the subsequent assignment
// is detected, a \a std::runtime_error exception is thrown.
*/
template< typename MT >  // Type of the dense matrix
void OperationTest<MT>::testElementsOperation( blaze::TrueType )
{
#if BLAZETEST_MATHTEST_TEST_ELEMENTS_OPERATION
   if( BLAZETEST_MATHTEST_TEST_ELEMENTS_OPERATION > 1 )
   {
      using blaze::ravel;


      if( mat_.rows() == 0UL )
         return;


      std::vector<size_t> indices( mat_.rows() );
      std::iota( indices.begin(), indices.end(), 0UL );
      std::random_shuffle( indices.begin(), indices.end() );


      //=====================================================================================
      // Elements-wise ravel operation
      //=====================================================================================

      // Elements-wise ravel operation with the given matrix
      {
         test_  = "Elements-wise ravel operation with the given matrix";
         error_ = "Failed ravel operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) = elements( ravel( mat_ )   , &indices[index], n );
               elements( sres_  , &indices[index], n ) = elements( ravel( mat_ )   , &indices[index], n );
               elements( refres_, &indices[index], n ) = elements( ravel( refmat_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) = elements( ravel( omat_ )  , &indices[index], n );
               elements( sres_  , &indices[index], n ) = elements( ravel( omat_ )  , &indices[index], n );
               elements( refres_, &indices[index], n ) = elements( ravel( refmat_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }

      // Elements-wise ravel operation with evaluated matrix
      {
         test_  = "Elements-wise ravel operation with evaluated matrix";
         error_ = "Failed ravel operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) = elements( eval( ravel( mat_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) = elements( eval( ravel( mat_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) = elements( eval( ravel( refmat_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) = elements( eval( ravel( omat_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) = elements( eval( ravel( omat_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) = elements( eval( ravel( refmat_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }


      //=====================================================================================
      // Elements-wise ravel operation with addition assignment
      //=====================================================================================

      // Elements-wise ravel operation with addition assignment with the given matrix
      {
         test_  = "Elements-wise ravel operation with addition assignment with the given matrix";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) += elements( ravel( mat_ )   , &indices[index], n );
               elements( sres_  , &indices[index], n ) += elements( ravel( mat_ )   , &indices[index], n );
               elements( refres_, &indices[index], n ) += elements( ravel( refmat_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) += elements( ravel( omat_ )  , &indices[index], n );
               elements( sres_  , &indices[index], n ) += elements( ravel( omat_ )  , &indices[index], n );
               elements( refres_, &indices[index], n ) += elements( ravel( refmat_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }

      // Elements-wise ravel operation with addition assignment with evaluated matrix
      {
         test_  = "Elements-wise ravel operation with addition assignment with evaluated matrix";
         error_ = "Failed addition assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) += elements( eval( ravel( mat_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) += elements( eval( ravel( mat_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) += elements( eval( ravel( refmat_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) += elements( eval( ravel( omat_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) += elements( eval( ravel( omat_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) += elements( eval( ravel( refmat_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }


      //=====================================================================================
      // Elements-wise ravel operation with subtraction assignment
      //=====================================================================================

      // Elements-wise ravel operation with subtraction assignment with the given matrix
      {
         test_  = "Elements-wise ravel operation with subtraction assignment with the given matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) -= elements( ravel( mat_ )   , &indices[index], n );
               elements( sres_  , &indices[index], n ) -= elements( ravel( mat_ )   , &indices[index], n );
               elements( refres_, &indices[index], n ) -= elements( ravel( refmat_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) -= elements( ravel( omat_ )  , &indices[index], n );
               elements( sres_  , &indices[index], n ) -= elements( ravel( omat_ )  , &indices[index], n );
               elements( refres_, &indices[index], n ) -= elements( ravel( refmat_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }

      // Elements-wise ravel operation with subtraction assignment with evaluated matrix
      {
         test_  = "Elements-wise ravel operation with subtraction assignment with evaluated matrix";
         error_ = "Failed subtraction assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) -= elements( eval( ravel( mat_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) -= elements( eval( ravel( mat_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) -= elements( eval( ravel( refmat_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) -= elements( eval( ravel( omat_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) -= elements( eval( ravel( omat_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) -= elements( eval( ravel( refmat_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }


      //=====================================================================================
      // Elements-wise ravel operation with multiplication assignment
      //=====================================================================================

      // Elements-wise ravel operation with multiplication assignment with the given matrix
      {
         test_  = "Elements-wise ravel operation with multiplication assignment with the given matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) *= elements( ravel( mat_ )   , &indices[index], n );
               elements( sres_  , &indices[index], n ) *= elements( ravel( mat_ )   , &indices[index], n );
               elements( refres_, &indices[index], n ) *= elements( ravel( refmat_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) *= elements( ravel( omat_ )  , &indices[index], n );
               elements( sres_  , &indices[index], n ) *= elements( ravel( omat_ )  , &indices[index], n );
               elements( refres_, &indices[index], n ) *= elements( ravel( refmat_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }

      // Elements-wise ravel operation with multiplication assignment with evaluated matrix
      {
         test_  = "Elements-wise ravel operation with multiplication assignment with evaluated matrix";
         error_ = "Failed multiplication assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) *= elements( eval( ravel( mat_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) *= elements( eval( ravel( mat_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) *= elements( eval( ravel( refmat_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               elements( dres_  , &indices[index], n ) *= elements( eval( ravel( omat_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) *= elements( eval( ravel( omat_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) *= elements( eval( ravel( refmat_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }


      //=====================================================================================
      // Elements-wise ravel operation with division assignment
      //=====================================================================================

      // Elements-wise ravel operation with division assignment with the given matrix
      {
         test_  = "Elements-wise ravel operation with division assignment with the given matrix";
         error_ = "Failed division assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               if( !blaze::isDivisor( elements( ravel( mat_ ), &indices[index], n ) ) ) continue;
               elements( dres_  , &indices[index], n ) /= elements( ravel( mat_ )   , &indices[index], n );
               elements( sres_  , &indices[index], n ) /= elements( ravel( mat_ )   , &indices[index], n );
               elements( refres_, &indices[index], n ) /= elements( ravel( refmat_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               if( !blaze::isDivisor( elements( ravel( omat_ ), &indices[index], n ) ) ) continue;
               elements( dres_  , &indices[index], n ) /= elements( ravel( omat_ )  , &indices[index], n );
               elements( sres_  , &indices[index], n ) /= elements( ravel( omat_ )  , &indices[index], n );
               elements( refres_, &indices[index], n ) /= elements( ravel( refmat_ ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }

      // Elements-wise ravel operation with division assignment with evaluated matrix
      {
         test_  = "Elements-wise ravel operation with division assignment with evaluated matrix";
         error_ = "Failed division assignment operation";

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               if( !blaze::isDivisor( elements( ravel( mat_ ), &indices[index], n ) ) ) continue;
               elements( dres_  , &indices[index], n ) /= elements( eval( ravel( mat_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) /= elements( eval( ravel( mat_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) /= elements( eval( ravel( refmat_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<MT>( ex );
         }

         checkResults<MT>();

         try {
            initResults();
            for( size_t index=0UL, n=0UL; index<indices.size(); index+=n ) {
               n = blaze::rand<size_t>( 1UL, indices.size() - index );
               if( !blaze::isDivisor( elements( ravel( omat_ ), &indices[index], n ) ) ) continue;
               elements( dres_  , &indices[index], n ) /= elements( eval( ravel( omat_ ) ), &indices[index], n );
               elements( sres_  , &indices[index], n ) /= elements( eval( ravel( omat_ ) ), &indices[index], n );
               elements( refres_, &indices[index], n ) /= elements( eval( ravel( refmat_ ) ), &indices[index], n );
            }
         }
         catch( std::exception& ex ) {
            convertException<TMT>( ex );
         }

         checkResults<TMT>();
      }
   }
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Skipping the elements-wise dense matrix ravel operation.
//
// \param op The ravel operation.
// \return void
// \exception std::runtime_error Reduction error detected.
//
// This function is called in case the elements-wise matrix ravel operation is not
// available for the given matrix type \a MT.
*/
template< typename MT >  // Type of the dense matrix
void OperationTest<MT>::testElementsOperation( blaze::FalseType )
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
template< typename MT >  // Type of the dense matrix
template< typename T >   // Type of the operand
void OperationTest<MT>::checkResults()
{
   using blaze::IsRowMajorMatrix;

   if( !isEqual( dres_, refres_ ) ) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect dense result detected\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   " << ( IsRowMajorMatrix<T>::value ? ( "Row-major" ) : ( "Column-major" ) ) << " dense matrix type:\n"
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
          << "   " << ( IsRowMajorMatrix<T>::value ? ( "Row-major" ) : ( "Column-major" ) ) << " dense matrix type:\n"
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
template< typename MT >  // Type of the dense matrix
template< typename T >   // Type of the operand
void OperationTest<MT>::checkTransposeResults()
{
   using blaze::IsRowMajorMatrix;

   if( !isEqual( tdres_, trefres_ ) ) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect dense result detected\n"
          << " Details:\n"
          << "   Random seed = " << blaze::getSeed() << "\n"
          << "   " << ( IsRowMajorMatrix<T>::value ? ( "Row-major" ) : ( "Column-major" ) ) << " dense matrix type:\n"
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
          << "   " << ( IsRowMajorMatrix<T>::value ? ( "Row-major" ) : ( "Column-major" ) ) << " dense matrix type:\n"
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
template< typename MT >  // Type of the dense matrix
void OperationTest<MT>::initResults()
{
   const blaze::UnderlyingBuiltin_t<DRE> min( randmin );
   const blaze::UnderlyingBuiltin_t<DRE> max( randmax );

   resize( dres_, rows( mat_ ) * columns( mat_ ) );
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
template< typename MT >  // Type of the dense matrix
void OperationTest<MT>::initTransposeResults()
{
   const blaze::UnderlyingBuiltin_t<TDRE> min( randmin );
   const blaze::UnderlyingBuiltin_t<TDRE> max( randmax );

   resize( dres_, rows( mat_ ) * columns( mat_ ) );
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
template< typename MT >  // Type of the dense matrix
template< typename T >   // Type of the operand
void OperationTest<MT>::convertException( const std::exception& ex )
{
   using blaze::IsRowMajorMatrix;

   std::ostringstream oss;
   oss << " Test : " << test_ << "\n"
       << " Error: " << error_ << "\n"
       << " Details:\n"
       << "   Random seed = " << blaze::getSeed() << "\n"
       << "   " << ( IsRowMajorMatrix<T>::value ? ( "Row-major" ) : ( "Column-major" ) ) << " dense matrix type:\n"
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
/*!\brief Testing the ravel operation for a specific matrix type.
//
// \param creator The creator for the dense matrix.
// \return void
*/
template< typename MT >  // Type of the dense matrix
void runTest( const Creator<MT>& creator )
{
   for( size_t rep=0UL; rep<repetitions; ++rep ) {
      OperationTest<MT>{ creator };
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
/*!\brief Macro for the definition of a dense matrix ravel operation test case.
*/
#define DEFINE_DMATRAVEL_OPERATION_TEST( MT ) \
   extern template class blazetest::mathtest::dmatravel::OperationTest<MT>
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of a dense matrix ravel operation test case.
*/
#define RUN_DMATRAVEL_OPERATION_TEST( C ) \
   blazetest::mathtest::dmatravel::runTest( C )
/*! \endcond */
//*************************************************************************************************

} // namespace dmatravel

} // namespace mathtest

} // namespace blazetest

#endif
