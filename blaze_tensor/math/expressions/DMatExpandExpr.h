//=================================================================================================
/*!
//  \file blaze_tensor/math/expressions/DMatExpandExpr.h
//  \brief Header file for the dense matrix expansion expression
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DMATEXPANDEXPR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DMATEXPANDEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/ExpandExprData.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/Transformation.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/simd/SIMDTrait.h>
#include <blaze/math/traits/ExpandTrait.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/math/typetraits/StorageOrder.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/Types.h>
#include <blaze/util/MaybeUnused.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/typetraits/GetMemberType.h>

#include <blaze_tensor/math/expressions/MatExpandExpr.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DMATEXPANDEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for dense matrix expansion.
// \ingroup dense_matrix_expression
//
// The DMatExpandExpr class represents the compile time expression for expansions of
// dense matrices.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CEAs >  // Compile time expansion arguments
class DMatExpandExpr
   : public MatExpandExpr< DenseTensor< DMatExpandExpr<MT,CEAs...> >, CEAs... >
   , private Transformation
   , private ExpandExprData<CEAs...>
{
 private:
   //**Type definitions****************************************************************************
   using CT = CompositeType_t<MT>;  //!< Composite type of the dense matrix expression.

   using DataType = ExpandExprData<CEAs...>;  //!< The type of the ExpandExprData base class.

   //! Definition of the GetIterator and GetConstIterator type traits.
   BLAZE_CREATE_GET_TYPE_MEMBER_TYPE_TRAIT( GetIterator, Iterator, INVALID_TYPE );
   BLAZE_CREATE_GET_TYPE_MEMBER_TYPE_TRAIT( GetConstIterator, ConstIterator, INVALID_TYPE );
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the expansion expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the serial evaluation strategy of the expansion expression. In case the dense matrix
       operand requires an intermediate evaluation, \a useAssign will be set to 1 and the
       expansion expression will be evaluated via the \a assign function family. Otherwise
       \a useAssign will be set to 0 and the expression will be evaluated via the function
       call operator. */
   static constexpr bool useAssign = RequiresEvaluation_v<MT>;

   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT1 >
   static constexpr bool UseAssign_v = useAssign;
   /*! \endcond */
   //**********************************************************************************************

   //**Parallel evaluation strategy****************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! This variable template is a helper for the selection of the parallel evaluation strategy.
       In case the matrix operand is not SMP assignable and requires an intermediate evaluation,
       the variable is set to 1 and the expression specific evaluation strategy is selected.
       Otherwise the variable is set to 0 and the default strategy is chosen. */
   template< typename MT1 >
   static constexpr bool UseSMPAssign_v = ( !MT1::smpAssignable && useAssign );
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This          = DMatExpandExpr<MT,CEAs...>;     //!< Type of this DMatExpandExpr instance.
   using BaseType      = DenseTensor<This>;              //!< Base type of this DMatExpandExpr instance.
   using ResultType    = ExpandTrait_t<MT,CEAs...>;      //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;     //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;    //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<MT>;              //!< Resulting element type.
   using ReturnType    = ReturnType_t<MT>;               //!< Return type for expression template evaluations.

   //! Data type for composite expression templates.
   using CompositeType = If_t< useAssign, const ResultType, const DMatExpandExpr& >;

   using Reference      = ElementType&;        //!< Reference to a non-constant tensor value.
   using ConstReference = const ElementType&;  //!< Reference to a constant tensor value.
   using Pointer        = ElementType*;        //!< Pointer to a non-constant tensor value.
   using ConstPointer   = const ElementType*;  //!< Pointer to a constant tensor value.

   //! Iterator over the elements of the dense matrix.
   using Iterator = GetIterator_t<MT>;
   using ConstIterator = GetConstIterator_t<MT>;

   //! Composite data type of the dense matrix expression.
   using Operand = If_t< IsExpression_v<MT>, const MT, const MT& >;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = (StorageOrder_v<MT> == rowMajor) && MT::simdEnabled;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = (StorageOrder_v<MT> == rowMajor) && MT::smpAssignable;
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   static constexpr size_t SIMDSIZE = SIMDTrait<ElementType>::size;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DMatExpandExpr class.
   //
   // \param dm The dense matrix operand of the expansion expression.
   // \param args The runtime expansion expression arguments.
   */
   template< typename... REAs >  // Runtime expansion arguments
   explicit inline DMatExpandExpr( const MT& dm, REAs... args ) noexcept
      : DataType( args... )  // Base class initialization
      , dm_     ( dm )       // Dense matrix of the expansion expression
   {}
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 3D-access to the tensor elements.
   //
   // \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator()( size_t k, size_t i, size_t j ) const {

      BLAZE_INTERNAL_ASSERT( k < expansion()   , "Invalid page access index"    );
      BLAZE_INTERNAL_ASSERT( i < dm_.rows()    , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < dm_.columns() , "Invalid column access index" );

      return dm_(i, j);
   }

   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the matrix elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid matrix access index.
   */
   inline ReturnType at( size_t k, size_t i, size_t j ) const {
      if( k >= expansion() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
      }
      if( i >= dm_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= dm_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      return (*this)(k,i,j);
   }
   //**********************************************************************************************

   //**Load function*******************************************************************************
   /*!\brief Access to the SIMD elements of the matrix.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return Reference to the accessed values.
   */
   BLAZE_ALWAYS_INLINE auto load( size_t k, size_t i, size_t j ) const noexcept {

      BLAZE_INTERNAL_ASSERT( k < expansion()     , "Invalid page access index"    );
      BLAZE_INTERNAL_ASSERT( i < dm_.rows()      , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < dm_.columns()   , "Invalid column access index" );
      BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL , "Invalid column access index" );

      return dm_.load(i, j);
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator to the first element of row/column \a i.
   */
   inline ConstIterator begin( size_t i, size_t k ) const {
      MAYBE_UNUSED( i, k );
      return ConstIterator( dm_.begin( i ) );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator just past the last element of row/column \a i.
   */
   inline ConstIterator end( size_t i, size_t k ) const {
      MAYBE_UNUSED( i, k );
      return ConstIterator( dm_.end( i ) );
   }
   //**********************************************************************************************

   //**Pages function*******************************************************************************
   /*!\brief Returns the current number of pages of the tensor.
   //
   // \return The number of pages of the tensor.
   */
   inline size_t pages() const noexcept {
      return expansion();
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the tensor.
   //
   // \return The number of rows of the tensor.
   */
   inline size_t rows() const noexcept {
      return dm_.rows();
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the tensor.
   //
   // \return The number of columns of the tensor.
   */
   inline size_t columns() const noexcept {
      return dm_.columns();
   }
   //**********************************************************************************************

   //**Operand access******************************************************************************
   /*!\brief Returns the dense matrix operand.
   //
   // \return The dense matrix operand.
   */
   inline Operand operand() const noexcept {
      return dm_;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   using DataType::expansion;
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the expression can alias, \a false otherwise.
   */
   template< typename T >
   inline bool canAlias( const T* alias ) const noexcept {
      return dm_.canAlias( alias );
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
   Operand dm_;  //!< Dense matrix of the expansion expression.
   //**********************************************************************************************

   //**Assignment to matrices**********************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense matrix expansion expression to a tensor.
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side matrix.
   // \param rhs The right-hand side expansion expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense matrix expansion
   // expression to a tensor. Due to the explicit application of the SFINAE principle, this
   // function can only be selected by the compiler in case the operand requires an intermediate
   // evaluation.
   */
   template< typename MT1 > // Type of the target tensor
   friend inline EnableIf_t< UseAssign_v<MT1> >
      assign( Tensor<MT1>& lhs, const DMatExpandExpr& rhs )
   {
      using blaze::expand;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );

      CT tmp( serial( *rhs.dm_ ) );

      assign( *lhs, expand<CEAs...>( tmp, rhs.expansion() ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to matrices*************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a dense matrix expansion expression to a matrix.
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side matrix.
   // \param rhs The right-hand side expansion expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a dense matrix
   // expansion expression to a matrix. Due to the explicit application of the SFINAE principle,
   // this function can only be selected by the compiler in case the operand requires an
   // intermediate evaluation.
   */
   template< typename MT1 > // Type of the target tensor
   friend inline EnableIf_t< UseAssign_v<MT1> >
      addAssign( Tensor<MT1>& lhs, const DMatExpandExpr& rhs )
   {
      using blaze::expand;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );

      CT tmp( serial( *rhs.dm_ ) );

      addAssign( *lhs, expand<CEAs...>( tmp, rhs.expansion() ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to matrices**********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a dense matrix expansion expression to a matrix.
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side matrix.
   // \param rhs The right-hand side expansion expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a dense matrix
   // expansion expression to a matrix. Due to the explicit application of the SFINAE principle,
   // this function can only be selected by the compiler in case the operand requires an
   // intermediate evaluation.
   */
   template< typename MT1 > // Type of the target tensor
   friend inline EnableIf_t< UseAssign_v<MT1> >
      subAssign( Tensor<MT1>& lhs, const DMatExpandExpr& rhs )
   {
      using blaze::expand;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );

      CT tmp( serial( *rhs.dm_ ) );

      subAssign( *lhs, expand<CEAs...>( tmp, rhs.expansion() ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Schur product assignment to matrices********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Schur product assignment of a dense matrix expansion expression to a matrix.
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side matrix.
   // \param rhs The right-hand side expansion expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized Schur product assignment of a dense
   // matrix expansion expression to a matrix. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the operand requires
   // an intermediate evaluation.
   */
   template< typename MT1 > // Type of the target tensor
   friend inline EnableIf_t< UseAssign_v<MT1> >
      schurAssign( Tensor<MT1>& lhs, const DMatExpandExpr& rhs )
   {
      using blaze::expand;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );

      CT tmp( serial( *rhs.dm_ ) );

      schurAssign( *lhs, expand<CEAs...>( tmp, rhs.expansion() ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Multiplication assignment to matrices*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Multiplication assignment of a dense matrix expansion expression to a matrix.
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side matrix.
   // \param rhs The right-hand side expansion expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a dense
   // matrix expansion expression to a matrix. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the operand requires
   // an intermediate evaluation.
   */
   template< typename MT1 > // Type of the target tensor
   friend inline EnableIf_t< UseAssign_v<MT1> >
      multAssign( Tensor<MT1>& lhs, const DMatExpandExpr& rhs )
   {
      using blaze::expand;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );

      CT tmp( serial( *rhs.dm_ ) );

      multAssign( *lhs, expand<CEAs...>( tmp, rhs.expansion() ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to matrices******************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense matrix expansion expression to a matrix.
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side matrix.
   // \param rhs The right-hand side expansion expansion to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense matrix
   // expansion expansion to a matrix. Due to the explicit application of the SFINAE principle,
   // this function can only be selected by the compiler in case the expression specific parallel
   // evaluation strategy is selected.
   */
   template< typename MT1 > // Type of the target tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT1> >
      smpAssign( Tensor<MT1>& lhs, const DMatExpandExpr& rhs )
   {
      using blaze::expand;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );

      CT tmp( *rhs.dm_ );

      smpAssign( *lhs, expand<CEAs...>( tmp, rhs.expansion() ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to matrices*********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a dense matrix expansion expression to a matrix.
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side matrix.
   // \param rhs The right-hand side expansion expansion to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a dense
   // matrix expansion expansion to a matrix. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename MT1 > // Type of the target tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT1> >
      smpAddAssign( Tensor<MT1>& lhs, const DMatExpandExpr& rhs )
   {
      using blaze::expand;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );

      CT tmp( *rhs.dm_ );

      smpAddAssign( *lhs, expand<CEAs...>( tmp, rhs.expansion() ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to matrices******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a dense matrix expansion expression to a matrix.
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side matrix.
   // \param rhs The right-hand side expansion expansion to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a dense
   // matrix expansion expansion to a matrix. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename MT1 > // Type of the target tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT1> >
      smpSubAssign( Tensor<MT1>& lhs, const DMatExpandExpr& rhs )
   {
      using blaze::expand;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );

      CT tmp( *rhs.dm_ );

      smpSubAssign( *lhs, expand<CEAs...>( tmp, rhs.expansion() ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP Schur product assignment to matrices****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP Schur product assignment of a dense matrix expansion expression to a matrix.
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side matrix.
   // \param rhs The right-hand side expansion expansion for the Schur product.
   // \return void
   //
   // This function implements the performance optimized SMP Schur product assignment of a dense
   // matrix expansion expansion to a matrix. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename MT1 > // Type of the target tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT1> >
      smpSchurAssign( Tensor<MT1>& lhs, const DMatExpandExpr& rhs )
   {
      using blaze::expand;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );

      CT tmp( *rhs.dm_ );

      smpSchurAssign( *lhs, expand<CEAs...>( tmp, rhs.expansion() ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP multiplication assignment to matrices***************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP multiplication assignment of a dense matrix expansion expression to a matrix.
   // \ingroup dense_matrix
   //
   // \param lhs The target left-hand side matrix.
   // \param rhs The right-hand side expansion expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a dense
   // matrix expansion expression to a matrix. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename MT1 > // Type of the target tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT1> >
      smpMultAssign( Tensor<MT1>& lhs, const DMatExpandExpr& rhs )
   {
      using blaze::expand;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );

      CT tmp( *rhs.dm_ );

      smpMultAssign( *lhs, expand<CEAs...>( tmp, rhs.expansion() ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT );
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
/*!\brief Expansion of the given dense matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be expanded.
// \param expansion The expansion.
// \return The expansion of the matrix.
//
// This function returns an expression representing the expansion of the given dense matrix:

   \code
   using blaze::rowMajor;
   using blaze::columnMajor;

   blaze::DynamicMatrix<int,columnMatrix> a{ 1, 5, -2, 4 }
   blaze::DynamicMatrix<int,rowMatrix> b{ 3, -1, 7, 0 }

   blaze::DynamicMatrix<double,columnMajor> A;
   blaze::DynamicMatrix<double,rowMajor> B;
   // ... Resizing and initialization

   // Expansion of the column matrix 'a' to 4x3 column-major matrix
   //
   //    (  1  1  1 )
   //    (  5  5  5 )
   //    ( -2 -2 -2 )
   //    (  4  4  4 )
   //
   A = expand( a, 3UL );

   // Expansion of the row matrix 'b' to a 3x4 row-major matrix
   //
   //    ( 3, -1, 7, 0 )
   //    ( 3, -1, 7, 0 )
   //    ( 3, -1, 7, 0 )
   //
   B = expand( b, 3UL );
   \endcode
*/
template< typename MT, bool SO > // Type of the target tensor
inline decltype(auto) expand( const DenseMatrix<MT, SO>& dm, size_t expansion )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DMatExpandExpr<MT>;
   return ReturnType( *dm, expansion );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Expansion of the given dense matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be expanded.
// \return The expansion of the matrix.
//
// This function returns an expression representing the expansion of the given dense matrix:

   \code
   using blaze::rowMajor;
   using blaze::columnMajor;

   blaze::DynamicMatrix<int,columnMatrix> a{ 1, 5, -2, 4 }
   blaze::DynamicMatrix<int,rowMatrix> b{ 3, -1, 7, 0 }

   blaze::DynamicTensor<double> A;
   blaze::DynamicTensor<double> B;
   // ... Resizing and initialization

   // Expansion of the column matrix 'a' to 4x3 column-major matrix
   //
   //    (  1  1  1 )
   //    (  5  5  5 )
   //    ( -2 -2 -2 )
   //    (  4  4  4 )
   //
   A = expand<3UL>( a );

   // Expansion of the row matrix 'b' to a 3x4 row-major matrix
   //
   //    ( 3, -1, 7, 0 )
   //    ( 3, -1, 7, 0 )
   //    ( 3, -1, 7, 0 )
   //
   B = expand<3UL>( b );
   \endcode
*/
template< size_t E     // Compile time expansion argument
        , typename MT  // Type of the dense tensor
        , bool SO >    // Storage order
inline decltype(auto) expand( const DenseMatrix<MT, SO>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DMatExpandExpr<MT,E>;
   return ReturnType( *dm );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Expansion of the given dense matrix.
// \ingroup dense_matrix
//
// \param dm The dense matrix to be expanded.
// \param expansion The expansion.
// \return The expansion of the matrix.
//
// This auxiliary overload of the \c expand() function accepts both a compile time and a runtime
// expansion. The runtime argument is discarded in favor of the compile time argument.
*/
template< size_t E     // Compile time expansion argument
        , typename MT  // Type of the dense tensor
        , bool SO >    // Storage order
inline decltype(auto) expand( const DenseMatrix<MT, SO>& dm, size_t expansion )
{
   MAYBE_UNUSED( expansion );

   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DMatExpandExpr<MT,E>;
   return ReturnType( *dm );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISALIGNED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT >
struct IsAligned< DMatExpandExpr<MT> >
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
template< typename MT >
struct IsPadded< DMatExpandExpr<MT> >
   : public IsPadded<MT>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
