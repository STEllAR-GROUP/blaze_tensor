//=================================================================================================
/*!
//  \file blaze_tensor/math/expressions/DTensTransExpr.h
//  \brief Header file for the dense tensor transpose expression
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSTRANSEXPR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSTRANSEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>

#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/StorageOrder.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/Transformation.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/simd/SIMDTrait.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/IsUniUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/Types.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/typetraits/GetMemberType.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/expressions/DTensTransExprData.h>
#include <blaze_tensor/math/expressions/DTensTransposer.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>
#include <blaze_tensor/math/expressions/TensTransExpr.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DTENSTRANSEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for dense tensor transpositions.
// \ingroup dense_tensor_expression
//
// The DTensTransExpr class represents the compile time expression for transpositions of
// dense tensors.
*/
template< typename MT,        // Type of the dense tensor
          size_t... CTAs >    // Compile time arguments
class DTensTransExpr
   : public TensTransExpr< DenseTensor< DTensTransExpr<MT> > >
   , public DTensTransExprData<CTAs...>
   , private If< IsComputation_v<MT>, Computation, Transformation >::Type
{
 private:
   //**Type definitions****************************************************************************
   using DataType = DTensTransExprData<CTAs...>;  //!< The type of the DTensTransExprData base class.
   using RT = ResultType_t<MT>;     //!< Result type of the dense tensor expression.
   using CT = CompositeType_t<MT>;  //!< Composite type of the dense tensor expression.

   //! Definition of the GetConstIterator type trait.
   BLAZE_CREATE_GET_TYPE_MEMBER_TYPE_TRAIT( GetConstIterator, ConstIterator, INVALID_TYPE );
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the transposition expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the serial evaluation strategy of the transposition expression. In case the given dense
       tensor expression of type \a MT requires an intermediate evaluation, \a useAssign will
       be set to 1 and the transposition expression will be evaluated via the \a assign function
       family. Otherwise \a useAssign will be set to 0 and the expression will be evaluated
       via the subscript operator. */
   static constexpr bool useAssign = RequiresEvaluation_v<MT>;

   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT2 >
   static constexpr bool UseAssign_v = useAssign;
   /*! \endcond */
   //**********************************************************************************************

   //**Parallel evaluation strategy****************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! This variable template is a helper for the selection of the parallel evaluation strategy.
       In case the target tensor is SMP assignable and the dense tensor operand requires an
       intermediate evaluation, the variable is set to 1 and the expression specific evaluation
       strategy is selected. Otherwise the variable is set to 0 and the default strategy is
       chosen. */
   template< typename MT2 >
   static constexpr bool UseSMPAssign_v = ( MT2::smpAssignable && useAssign );
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This          = DTensTransExpr<MT>;          //!< Type of this DTensTransExpr instance.
   using ResultType    = TransposeType_t<MT>;         //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;  //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = ResultType_t<MT>;            //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<MT>;           //!< Resulting element type.
   using ReturnType    = ReturnType_t<MT>;            //!< Return type for expression template evaluations.

   //! Data type for composite expression templates.
   using CompositeType = If_t< useAssign, const ResultType, const DTensTransExpr& >;

   //! Iterator over the elements of the dense tensor.
   using ConstIterator = GetConstIterator_t<MT>;

   //! Composite data type of the dense tensor expression.
   using Operand = If_t< IsExpression_v<MT>, const MT, const MT& >;
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   using DataType::idces;
   using DataType::page;
   using DataType::row;
   using DataType::column;
   using DataType::reverse_page;
   using DataType::reverse_row;
   using DataType::reverse_column;
   //@}
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = MT::simdEnabled;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = MT::smpAssignable;
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   static constexpr size_t SIMDSIZE = SIMDTrait<ElementType>::size;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DTensTransExpr class.
   //
   // \param dm The dense tensor operand of the transposition expression.
   */
   template< typename... RTAs >
   explicit inline DTensTransExpr( const MT& dm, RTAs... args ) noexcept
      : DataType( args... )   // Base class initialization
      , dm_( dm )             // Dense tensor of the transposition expression
   {}
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 3D-access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator()( size_t k, size_t i, size_t j ) const {
      BLAZE_INTERNAL_ASSERT( k < pages()  , "Invalid page access index" );
      BLAZE_INTERNAL_ASSERT( i < columns(), "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < rows()   , "Invalid column access index" );
      return dm_( reverse_page(k, i, j), reverse_row(k, i, j), reverse_column(k, i, j) );
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid tensor access index.
   */
   inline ReturnType at( size_t k, size_t i, size_t j ) const {
      if( k >= pages() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
      }
      if( i >= rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      return (*this)(k,i,j);
   }
   //**********************************************************************************************

   //**Load function*******************************************************************************
   /*!\brief Access to the SIMD elements of the tensor.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return Reference to the accessed values.
   */
   BLAZE_ALWAYS_INLINE auto load( size_t k, size_t i, size_t j ) const noexcept {
      BLAZE_INTERNAL_ASSERT( k < pages()  , "Invalid page access index" );
      BLAZE_INTERNAL_ASSERT( i < rows()   , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
      BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL , "Invalid column access index" );
      return dm_.load( reverse_page(k, i, j), reverse_row(k, i, j), reverse_column(k, i, j) );
   }
   //**********************************************************************************************

   //**Low-level data access***********************************************************************
   /*!\brief Low-level data access to the tensor elements.
   //
   // \return Pointer to the internal element storage.
   */
   inline const ElementType* data() const noexcept {
      return dm_.data();
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator to the first non-zero element of row/column \a i.
   */
   inline ConstIterator begin( size_t i, size_t k ) const {
      return ConstIterator( dm_.begin( reverse_row(k, i, 0), reverse_page(k, i, 0) ) );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator just past the last non-zero element of row/column \a i.
   */
   inline ConstIterator end( size_t i, size_t k ) const {
      return ConstIterator( dm_.end( reverse_row(k, i, 0), reverse_page(k, i, 0) ) );
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the tensor.
   //
   // \return The number of rows of the tensor.
   */
   inline size_t rows() const noexcept {
      return row( dm_.pages(), dm_.rows(), dm_.columns() );
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the tensor.
   //
   // \return The number of columns of the tensor.
   */
   inline size_t columns() const noexcept {
      return column( dm_.pages(), dm_.rows(), dm_.columns() );
   }
   //**********************************************************************************************

   //**Pages function****************************************************************************
   /*!\brief Returns the current number of pages of the tensor.
   //
   // \return The number of pages of the tensor.
   */
   inline size_t pages() const noexcept {
      return page( dm_.pages(), dm_.rows(), dm_.columns() );
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the dense tensor.
   //
   // \return The number of non-zero elements in the dense tensor.
   */
   inline size_t nonZeros() const {
      return dm_.nonZeros();
   }
   //**********************************************************************************************

   //**NonZeros function***************************************************************************
   /*!\brief Returns the number of non-zero elements in the specified row/column.
   //
   // \param i The index of the row/column.
   // \return The number of non-zero elements of row/column \a i.
   */
   inline size_t nonZeros( size_t i, size_t k ) const {
      return dm_.nonZeros( reverse_row(k, i, 0), reverse_page(k, i, 0) );
   }
   //**********************************************************************************************

   //**Operand access******************************************************************************
   /*!\brief Returns the dense tensor operand.
   //
   // \return The dense tensor operand.
   */
   inline Operand operand() const noexcept {
      return dm_;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the expression can alias, \a false otherwise.
   */
   template< typename T >
   inline bool canAlias( const T* alias ) const noexcept {
      return dm_.isAliased( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case an alias effect is detected, \a false otherwise.
   */
   template< typename T >
   inline bool isAliased( const T* alias ) const noexcept {
      return dm_.isAliased( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const noexcept {
      return dm_.isAligned();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return dm_.canSMPAssign();
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   Operand dm_;  //!< Dense tensor of the transposition expression.
   //**********************************************************************************************

   //**Assignment to dense tensors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense tensor transposition expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side transposition expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense tensor
   // transposition expression to a dense tensor. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case
   // the operand requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT2> >
      assign( DenseTensor<MT2>& lhs, const DTensTransExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid number of pages"   );
      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      DTensTransposer<MT2> tmp( ~lhs, rhs.pages(), rhs.rows(), rhs.columns() );
      assign( tmp, rhs.dm_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to sparse tensors***************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense tensor transposition expression to a sparse tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side sparse tensor.
   // \param rhs The right-hand side transposition expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense tensor
   // transposition expression to a sparse tensor. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case
   // the operand requires an intermediate evaluation.
   */
//    template< typename MT2 > // Type of the target dense tensor
//    friend inline EnableIf_t< UseAssign_v<MT2> >
//       assign( SparseTensor<MT2>& lhs, const DTensTransExpr& rhs )
//    {
//       BLAZE_FUNCTION_TRACE;
//
//       using TmpType = If_t< SO == SO2, ResultType, OppositeType >;
//
//       BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
//       BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( OppositeType );
//       BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( ResultType, SO );
//       BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( OppositeType, !SO );
//       BLAZE_CONSTRAINT_MATRICES_MUST_HAVE_SAME_STORAGE_ORDER( MT2, TmpType );
//       BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( TmpType );
//
//       BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
//       BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
//
//       const TmpType tmp( serial( rhs ) );
//       assign( ~lhs, tmp );
//    }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense tensors*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a dense tensor transposition expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side transposition expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a dense
   // tensor transposition expression to a dense tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case
   // the operand requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT2> >
      addAssign( DenseTensor<MT2>& lhs, const DTensTransExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid number of pages"   );
      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      DTensTransposer<MT2> tmp( ~lhs, rhs.pages(), rhs.rows(), rhs.columns() );
      addAssign( tmp, rhs.dm_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to sparse tensors******************************************************
   // No special implementation for the addition assignment to sparse tensors.
   //**********************************************************************************************

   //**Subtraction assignment to dense tensors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a dense tensor transposition expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side transposition expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a dense
   // tensor transposition expression to a dense tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case
   // the operand requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT2> >
      subAssign( DenseTensor<MT2>& lhs, const DTensTransExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid number of pages"   );
      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      DTensTransposer<MT2> tmp( ~lhs, rhs.pages(), rhs.rows(), rhs.columns() );
      subAssign( tmp, rhs.dm_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to sparse tensors***************************************************
   // No special implementation for the subtraction assignment to sparse tensors.
   //**********************************************************************************************

   //**Schur product assignment to dense tensors**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Schur product assignment of a dense tensor transposition expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side transposition expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized Schur product assignment of a dense
   // tensor transposition expression to a dense tensor. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case the
   // operand requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT2> >
      schurAssign( DenseTensor<MT2>& lhs, const DTensTransExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid number of pages"   );
      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      DTensTransposer<MT2> tmp( ~lhs, rhs.pages(), rhs.rows(), rhs.columns() );
      schurAssign( tmp, rhs.dm_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Schur product assignment to sparse tensors*************************************************
   // No special implementation for the Schur product assignment to sparse tensors.
   //**********************************************************************************************

   //**Multiplication assignment to dense tensors*************************************************
   // No special implementation for the multiplication assignment to dense tensors.
   //**********************************************************************************************

   //**Multiplication assignment to sparse tensors************************************************
   // No special implementation for the multiplication assignment to sparse tensors.
   //**********************************************************************************************

   //**SMP assignment to dense tensors************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense tensor transposition expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side transposition expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense tensor
   // transposition expression to a dense tensor. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpAssign( DenseTensor<MT2>& lhs, const DTensTransExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid number of pages"   );
      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      DTensTransposer<MT2> tmp( ~lhs, rhs.pages(), rhs.rows(), rhs.columns() );
      smpAssign( tmp, rhs.dm_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to sparse tensors***********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense tensor transposition expression to a sparse tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side sparse tensor.
   // \param rhs The right-hand side transposition expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense tensor
   // transposition expression to a sparse tensor. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
//    template< typename MT2 > // Type of the target dense tensor
//    friend inline EnableIf_t< UseSMPAssign_v<MT2> >
//       smpAssign( SparseTensor<MT2>& lhs, const DTensTransExpr& rhs )
//    {
//       BLAZE_FUNCTION_TRACE;
//
//       using TmpType = If_t< SO == SO2, ResultType, OppositeType >;
//
//       BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
//       BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( OppositeType );
//       BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( ResultType, SO );
//       BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( OppositeType, !SO );
//       BLAZE_CONSTRAINT_MATRICES_MUST_HAVE_SAME_STORAGE_ORDER( MT2, TmpType );
//       BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( TmpType );
//
//       BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
//       BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
//
//       const TmpType tmp( rhs );
//       smpAssign( ~lhs, tmp );
//    }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to dense tensors***************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a dense tensor transposition expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side transposition expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a dense
   // tensor transposition expression to a dense tensor. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpAddAssign( DenseTensor<MT2>& lhs, const DTensTransExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid number of pages"   );
      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      DTensTransposer<MT2> tmp( ~lhs, rhs.pages(), rhs.rows(), rhs.columns() );
      smpAddAssign( tmp, rhs.dm_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to sparse tensors**************************************************
   // No special implementation for the SMP addition assignment to sparse tensors.
   //**********************************************************************************************

   //**SMP subtraction assignment to dense tensors************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a dense tensor transposition expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side transposition expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a dense
   // tensor transposition expression to a dense tensor. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpSubAssign( DenseTensor<MT2>& lhs, const DTensTransExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid number of pages"   );
      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      DTensTransposer<MT2> tmp( ~lhs, rhs.pages(), rhs.rows(), rhs.columns() );
      smpSubAssign( tmp, rhs.dm_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to sparse tensors***********************************************
   // No special implementation for the SMP subtraction assignment to sparse tensors.
   //**********************************************************************************************

   //**SMP Schur product assignment to dense tensors**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP Schur product assignment of a dense tensor transposition expression to a dense
   //        tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side transposition expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized SMP Schur product assignment of a dense
   // tensor transposition expression to a dense tensor. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpSchurAssign( DenseTensor<MT2>& lhs, const DTensTransExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid number of pages"   );
      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      DTensTransposer<MT2> tmp( ~lhs, rhs.pages(), rhs.rows(), rhs.columns() );
      smpSchurAssign( tmp, rhs.dm_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP Schur product assignment to sparse tensors*********************************************
   // No special implementation for the SMP Schur product assignment to sparse tensors.
   //**********************************************************************************************

   //**SMP multiplication assignment to dense tensors*********************************************
   // No special implementation for the SMP multiplication assignment to dense tensors.
   //**********************************************************************************************

   //**SMP multiplication assignment to sparse tensors********************************************
   // No special implementation for the SMP multiplication assignment to sparse tensors.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT );
//    BLAZE_CONSTRAINT_MUST_BE_TENSOR_WITH_STORAGE_ORDER( MT, !SO );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Calculation of the transpose of the given dense tensor.
// \ingroup dense_tensor
//
// \param dm The dense tensor to be transposed.
// \return The transpose of the tensor.
//
// This function returns an expression representing the transpose of the given dense tensor:

   \code
   blaze::DynamicTensor<double> A;
   blaze::DynamicTensor<double,> B;
   // ... Resizing and initialization
   B = trans<2UL, 0UL, 1UL>( A );      // rotate tensor
   \endcode
*/
template< size_t O            // Mapping index for page dimension
        , size_t M            // Mapping index for row dimension
        , size_t N            // Mapping index for column dimension
        , typename MT         // Type of the target dense tensor
        , typename ... RTAs>  // Runtime arguments
inline decltype(auto) trans( const DenseTensor<MT>& dm, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensTransExpr<MT, O, M, N>;
   return ReturnType( ~dm, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculation of the transpose of the given dense tensor.
// \ingroup dense_tensor
//
// \param dm The dense tensor to be transposed.
// \return The transpose of the tensor.
//
// This function returns an expression representing the transpose of the given dense tensor:

   \code
   blaze::DynamicTensor<double> A;
   blaze::DynamicTensor<double,> B;
   // ... Resizing and initialization
   B = trans( A, { 2UL, 0UL, 1UL } );     // rotate tensor
   \endcode
*/
template< typename MT         // Type of the target dense tensor
        , typename ... RTAs>  // Runtime arguments
inline decltype(auto) trans( const DenseTensor<MT>& dm, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensTransExpr<MT>;
   return ReturnType( ~dm, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculation of the transpose of the given dense tensor.
// \ingroup dense_tensor
//
// \param dm The dense tensor to be transposed.
// \return The transpose of the tensor.
//
// This function returns an expression representing the transpose of the given dense tensor:

   \code
   blaze::DynamicTensor<double> A;
   blaze::DynamicTensor<double,> B;
   // ... Resizing and initialization
   B = trans( A, { 2UL, 0UL, 1UL } );     // rotate tensor
   \endcode
*/
template< typename MT         // Type of the target dense tensor
        , typename T          // Type of the element indices
        , typename ... RTAs>  // Runtime arguments
inline decltype(auto) trans( const DenseTensor<MT>& dm, const T* indices, size_t n, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensTransExpr<MT>;
   return ReturnType( ~dm, indices, n, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculation of the transpose of the given dense tensor.
// \ingroup dense_tensor
//
// \param dm The dense tensor to be transposed.
// \return The transpose of the tensor.
//
// This function returns an expression representing the transpose of the given dense tensor:

   \code
   blaze::DynamicTensor<double> A;
   blaze::DynamicTensor<double,> B;
   // ... Resizing and initialization
   B = trans( A, { 2UL, 0UL, 1UL } );     // rotate tensor
   \endcode
*/
template< typename MT         // Type of the target dense tensor
        , size_t... Is        // Element indices
        , typename ... RTAs>  // Runtime arguments
inline decltype(auto) trans( const DenseTensor<MT>& dm, std::index_sequence<Is...> indices, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return trans<Is...>( ~dm, args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculation of the transpose of the given dense tensor.
// \ingroup dense_tensor
//
// \param dm The dense tensor to be transposed.
// \return The transpose of the tensor.
//
// This function returns an expression representing the transpose of the given dense tensor:

   \code
   blaze::DynamicTensor<double> A;
   blaze::DynamicTensor<double,> B;
   // ... Resizing and initialization
   B = trans( A, { 2UL, 0UL, 1UL } );     // rotate tensor
   \endcode
*/
template< typename MT         // Type of the target dense tensor
        , typename T          // Type of the element indices
        , typename ... RTAs>  // Runtime arguments
inline decltype(auto) trans( const DenseTensor<MT>& dm, std::initializer_list<T> indices, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return trans( ~dm, indices.begin(), indices.size(), args... );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS
//
//=================================================================================================


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Calculating the transpose of a transpose dense tensor.
// \ingroup dense_tensor
//
// \param dm The dense tensor to be (re-)transposed.
// \return The transpose of the transpose tensor.
//
// This function implements a performance optimized treatment of the transpose operation on a
// dense tensor transpose expression. It returns an expression representing the transpose of a
// transpose dense tensor:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = trans<2UL, 0UL, 1UL>( trans( A ) );
   \endcode
*/
template< size_t O               // Mapping index for page dimension
        , size_t M               // Mapping index for row dimension
        , size_t N               // Mapping index for column dimension
        , typename MT            // Type of the target dense tensor
        , size_t... CTAs         // Compile time arguments of source
        , typename ... RTAs>     // Runtime arguments
inline decltype(auto) trans( const DTensTransExpr<MT, CTAs...>& dm, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensTransExpr<MT, O, M, N>;
   return ReturnType( ~dm, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Calculating the transpose of a transpose dense tensor.
// \ingroup dense_tensor
//
// \param dm The dense tensor to be (re-)transposed.
// \return The transpose of the transpose tensor.
//
// This function implements a performance optimized treatment of the transpose operation on a
// dense tensor transpose expression. It returns an expression representing the transpose of a
// transpose dense tensor:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = trans( trans( A ), { 2UL, 0UL, 1UL } );
   \endcode
*/
template< typename MT            // Type of the target dense tensor
        , size_t... CTAs         // Compile time arguments of source
        , typename ... RTAs >     // Runtime arguments
inline decltype(auto) trans( const DTensTransExpr<MT, CTAs...>& dm, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensTransExpr<MT, CTAs...>;
   return ReturnType( ~dm, args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Calculation of the transpose of the given dense tensor-scalar multiplication.
// \ingroup dense_tensor
//
// \param dm The dense tensor-scalar multiplication expression to be transposed.
// \return The transpose of the expression.
//
// This operator implements the performance optimized treatment of the transpose of a dense
// tensor-scalar multiplication. It restructures the expression \f$ A=trans(B*s1) \f$ to the
// expression \f$ A=trans(B)*s1 \f$.
*/
template< typename MT  // Type of the left-hand side dense tensor
        , typename ST  // Type of the right-hand side scalar value
        , typename ... RTAs >     // Runtime arguments
inline decltype(auto) trans( const DTensScalarMultExpr<MT,ST>& dm, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return trans( dm.leftOperand(), args... ) * dm.rightOperand();
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HASCONSTDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT > // Type of the target dense tensor
struct HasConstDataAccess< DTensTransExpr<MT> >
   : public HasConstDataAccess<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISALIGNED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT > // Type of the target dense tensor
struct IsAligned< DTensTransExpr<MT> >
   : public IsAligned<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISPADDED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT > // Type of the target dense tensor
struct IsPadded< DTensTransExpr<MT> >
   : public IsPadded<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSYMMETRIC SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT > // Type of the target dense tensor
struct IsSymmetric< DTensTransExpr<MT> >
   : public IsSymmetric<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISHERMITIAN SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT > // Type of the target dense tensor
struct IsHermitian< DTensTransExpr<MT> >
   : public IsHermitian<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISLOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT > // Type of the target dense tensor
struct IsLower< DTensTransExpr<MT> >
   : public IsUpper<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUNILOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT > // Type of the target dense tensor
struct IsUniLower< DTensTransExpr<MT> >
   : public IsUniUpper<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSTRICTLYLOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT > // Type of the target dense tensor
struct IsStrictlyLower< DTensTransExpr<MT> >
   : public IsStrictlyUpper<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUPPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT > // Type of the target dense tensor
struct IsUpper< DTensTransExpr<MT> >
   : public IsLower<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUNIUPPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT > // Type of the target dense tensor
struct IsUniUpper< DTensTransExpr<MT> >
   : public IsUniLower<MT>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSTRICTLYUPPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT > // Type of the target dense tensor
struct IsStrictlyUpper< DTensTransExpr<MT> >
   : public IsStrictlyLower<MT>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
