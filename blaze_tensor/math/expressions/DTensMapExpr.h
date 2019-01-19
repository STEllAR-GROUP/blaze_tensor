//=================================================================================================
/*!
//  \file blaze/math/expressions/DTensMapExpr.h
//  \brief Header file for the dense tensor map expression
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018 Hartmut Kaiser - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSMAPEXPR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSMAPEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <utility>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/DMatMapExpr.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>
#include <blaze_tensor/math/expressions/Forward.h>
#include <blaze_tensor/math/expressions/TensMapExpr.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DTENSMAPEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for the dense tensor map() function.
// \ingroup dense_tensor_expression
//
// The DTensMapExpr class represents the compile time expression for the evaluation of a custom
// operation on each element of a dense tensor via the map() function.
*/
template< typename MT  // Type of the dense tensor
        , typename OP > // Type of the custom operation
class DTensMapExpr
   : public TensMapExpr< DenseTensor< DTensMapExpr<MT,OP> > >
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   using RT = ResultType_t<MT>;    //!< Result type of the dense tensor expression.
   using OT = OppositeType_t<MT>;  //!< Opposite type of the dense tensor expression.
   using ET = ElementType_t<MT>;   //!< Element type of the dense tensor expression.
   using RN = ReturnType_t<MT>;    //!< Return type of the dense tensor expression.

   //! Definition of the HasSIMDEnabled type trait.
   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasSIMDEnabled, simdEnabled );

   //! Definition of the HasLoad type trait.
   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasLoad, load );
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the map expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the serial evaluation strategy of the map expression. In case the given dense tensor
       expression of type \a MT requires an intermediate evaluation, \a useAssign will be
       set to 1 and the map expression will be evaluated via the \a assign function family.
       Otherwise \a useAssign will be set to 0 and the expression will be evaluated via the
       subscript operator. */
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
       In case either the target tensor or the dense tensor operand is not SMP assignable and
       the tensor operand requires an intermediate evaluation, the variable is set to 1 and the
       expression specific evaluation strategy is selected. Otherwise the variable is set to 0
       and the default strategy is chosen. */
   template< typename MT2 >
   static constexpr bool UseSMPAssign_v =
      ( ( !MT2::smpAssignable || !MT::smpAssignable ) && useAssign );
   /*! \endcond */
   //**********************************************************************************************

   //**SIMD support detection**********************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper structure for the detection of the SIMD capabilities of the given custom operation.
   struct UseSIMDEnabledFlag {
      static constexpr bool test( bool (*fnc)() ) { return fnc(); }
      static constexpr bool test( bool b ) { return b; }
      static constexpr bool value = test( OP::BLAZE_TEMPLATE simdEnabled<ET> );
   };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This          = DTensMapExpr<MT,OP>;          //!< Type of this DTensMapExpr instance.
   using ResultType    = MapTrait_t<RT,OP>;            //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;   //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;  //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<ResultType>;    //!< Resulting element type.

   //! Return type for expression template evaluations.
   using ReturnType = decltype( std::declval<OP>()( std::declval<RN>() ) );

   //! Data type for composite expression templates.
   using CompositeType = If_t< useAssign, const ResultType, const DTensMapExpr& >;

   //! Composite data type of the dense tensor expression.
   using Operand = If_t< IsExpression_v<MT>, const MT, const MT& >;

   //! Data type of the custom unary operation.
   using Operation = OP;
   //**********************************************************************************************

   //**ConstIterator class definition**************************************************************
   /*!\brief Iterator over the elements of the dense tensor map expression.
   */
   class ConstIterator
   {
    public:
      //**Type definitions*************************************************************************
      using IteratorCategory = std::random_access_iterator_tag;  //!< The iterator category.
      using ValueType        = ElementType;                      //!< Type of the underlying elements.
      using PointerType      = ElementType*;                     //!< Pointer return type.
      using ReferenceType    = ElementType&;                     //!< Reference return type.
      using DifferenceType   = ptrdiff_t;                        //!< Difference between two iterators.

      // STL iterator requirements
      using iterator_category = IteratorCategory;  //!< The iterator category.
      using value_type        = ValueType;         //!< Type of the underlying elements.
      using pointer           = PointerType;       //!< Pointer return type.
      using reference         = ReferenceType;     //!< Reference return type.
      using difference_type   = DifferenceType;    //!< Difference between two iterators.

      //! ConstIterator type of the dense tensor expression.
      using IteratorType = ConstIterator_t<MT>;
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the ConstIterator class.
      //
      // \param it Iterator to the initial tensor element.
      // \param op The custom unary operation.
      */
      explicit inline ConstIterator( IteratorType it, OP op )
         : it_( it )  // Iterator to the current tensor element
         , op_( op )  // The custom unary operation
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline ConstIterator& operator+=( size_t inc ) {
         it_ += inc;
         return *this;
      }
      //*******************************************************************************************

      //**Subtraction assignment operator**********************************************************
      /*!\brief Subtraction assignment operator.
      //
      // \param dec The decrement of the iterator.
      // \return The decremented iterator.
      */
      inline ConstIterator& operator-=( size_t dec ) {
         it_ -= dec;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline ConstIterator& operator++() {
         ++it_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator++( int ) {
         return ConstIterator( it_++, op_ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline ConstIterator& operator--() {
         --it_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator--( int ) {
         return ConstIterator( it_--, op_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReturnType operator*() const {
         return op_( *it_ );
      }
      //*******************************************************************************************

      //**Load function****************************************************************************
      /*!\brief Access to the SIMD elements of the tensor.
      //
      // \return The resulting SIMD element.
      */
      inline auto load() const noexcept {
         return op_.load( it_.load() );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const ConstIterator& rhs ) const {
         return it_ == rhs.it_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const ConstIterator& rhs ) const {
         return it_ != rhs.it_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const ConstIterator& rhs ) const {
         return it_ < rhs.it_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const ConstIterator& rhs ) const {
         return it_ > rhs.it_;
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const ConstIterator& rhs ) const {
         return it_ <= rhs.it_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const ConstIterator& rhs ) const {
         return it_ >= rhs.it_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const ConstIterator& rhs ) const {
         return it_ - rhs.it_;
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between a ConstIterator and an integral value.
      //
      // \param it The iterator to be incremented.
      // \param inc The number of elements the iterator is incremented.
      // \return The incremented iterator.
      */
      friend inline const ConstIterator operator+( const ConstIterator& it, size_t inc ) {
         return ConstIterator( it.it_ + inc, it.op_ );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between an integral value and a ConstIterator.
      //
      // \param inc The number of elements the iterator is incremented.
      // \param it The iterator to be incremented.
      // \return The incremented iterator.
      */
      friend inline const ConstIterator operator+( size_t inc, const ConstIterator& it ) {
         return ConstIterator( it.it_ + inc, it.op_ );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Subtraction between a ConstIterator and an integral value.
      //
      // \param it The iterator to be decremented.
      // \param dec The number of elements the iterator is decremented.
      // \return The decremented iterator.
      */
      friend inline const ConstIterator operator-( const ConstIterator& it, size_t dec ) {
         return ConstIterator( it.it_ - dec, it.op_ );
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType it_;  //!< Iterator to the current tensor element.
      OP           op_;  //!< The custom unary operation.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled =
      ( MT::simdEnabled &&
        If_t< HasSIMDEnabled_v<OP>, UseSIMDEnabledFlag, HasLoad<OP> >::value );

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = MT::smpAssignable;
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   static constexpr size_t SIMDSIZE = SIMDTrait<ElementType>::size;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DTensMapExpr class.
   //
   // \param dm The dense tensor operand of the map expression.
   // \param op The custom unary operation.
   */
   explicit inline DTensMapExpr( const MT& dm, OP op ) noexcept
      : dm_( dm )  // Dense tensor of the map expression
      , op_( op )  // The custom unary operation
   {}
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 2D-access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator()( size_t k, size_t i, size_t j ) const noexcept {
      BLAZE_INTERNAL_ASSERT( i < dm_.rows()   , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < dm_.columns(), "Invalid column access index" );
      BLAZE_INTERNAL_ASSERT( k < dm_.pages()  , "Invalid page access index" );
      return op_( dm_(k,i,j) );
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
      if( i >= dm_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= dm_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      if( k >= dm_.pages() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
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
   BLAZE_ALWAYS_INLINE auto load( size_t i, size_t j ) const noexcept {
      BLAZE_INTERNAL_ASSERT( i < dm_.rows()   , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < dm_.columns(), "Invalid column access index" );
      BLAZE_INTERNAL_ASSERT( k < dm_.pages()  , "Invalid page access index" );
      BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );
      BLAZE_INTERNAL_ASSERT( k % SIMDSIZE == 0UL, "Invalid page access index" );
      return op_.load( dm_.load(i,j) );
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of row \a i.
   //
   // \param i The row index.
   // \return Iterator to the first non-zero element of row \a i.
   */
   inline ConstIterator begin( size_t i, size_t k ) const {
      return ConstIterator( dm_.begin(i, k), op_ );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row \a i.
   //
   // \param i The row index.
   // \return Iterator just past the last non-zero element of row \a i.
   */
   inline ConstIterator end( size_t i, size_t k ) const {
      return ConstIterator( dm_.end(i, k), op_ );
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

   //**Pages function****************************************************************************
   /*!\brief Returns the current number of pages of the tensor.
   //
   // \return The number of pages of the tensor.
   */
   inline size_t pages() const noexcept {
      return dm_.pages();
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

   //**Operation access****************************************************************************
   /*!\brief Returns a copy of the custom operation.
   //
   // \return A copy of the custom operation.
   */
   inline Operation operation() const {
      return op_;
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
      return IsExpression_v<MT> && dm_.canAlias( alias );
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
   Operand   dm_;  //!< Dense tensor of the map expression.
   Operation op_;  //!< The custom unary operation.
   //**********************************************************************************************

   //**Assignment to dense tensors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense tensor map
   // expression to a dense tensor. Due to the explicit application of the SFINAE principle,
   // this function can only be selected by the compiler in case the operand requires an
   // intermediate evaluation and the underlying numeric data type of the operand and the
   // target tensor are identical.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT2> && IsSame_v< UnderlyingNumeric_t<MT>
                           , UnderlyingNumeric_t<MT2> > >
      assign( DenseTensor<MT2>& lhs, const DTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid pages of columns" );

      assign( ~lhs, rhs.dm_ );
      assign( ~lhs, rhs.op_( ~lhs ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to dense tensors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense tensor map
   // expression to a dense tensor. Due to the explicit application of the SFINAE principle,
   // this function can only be selected by the compiler in case the operand requires an
   // intermediate evaluation and the underlying numeric data type of the operand and the
   // target vector differ.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT2> && !IsSame_v< UnderlyingNumeric_t<MT>
                           , UnderlyingNumeric_t<MT2> > >
      assign( DenseTensor<MT2>& lhs, const DTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RT );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid pages of columns" );

      const RT tmp( serial( rhs.dm_ ) );
      assign( ~lhs, map( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense tensors*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a dense
   // tensor map expression to a dense tensor. Due to the explicit application of the
   // SFINAE principle, this operator can only be selected by the compiler in case the
   // operand requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT2> >
      addAssign( DenseTensor<MT2>& lhs, const DTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RT );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid pages of columns" );

      const RT tmp( serial( rhs.dm_ ) );
      addAssign( ~lhs, map( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to dense tensors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a dense
   // tensor map expression to a dense tensor. Due to the explicit application of the SFINAE
   // principle, this operator can only be selected by the compiler in case the operand
   // requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT2> >
      subAssign( DenseTensor<MT2>& lhs, const DTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RT );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid pages of columns" );

      const RT tmp( serial( rhs.dm_ ) );
      subAssign( ~lhs, map( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Schur product assignment to dense tensors**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Schur product assignment of a dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized Schur product assignment of a dense
   // tensor map expression to a dense tensor. Due to the explicit application of the SFINAE
   // principle, this operator can only be selected by the compiler in case the operand
   // requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT2> >
      schurAssign( DenseTensor<MT2>& lhs, const DTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RT );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid pages of columns" );

      const RT tmp( serial( rhs.dm_ ) );
      schurAssign( ~lhs, map( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to dense tensors************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense tensor map expression to a row-major dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense tensor
   // map expression to a row-major dense tensor. Due to the explicit application of the
   // SFINAE principle, this operator can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected and the underlying
   // numeric data type of the operand and the target tensor are identical.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT2> && IsSame_v< UnderlyingNumeric_t<MT>
                           , UnderlyingNumeric_t<MT2> > >
      smpAssign( DenseTensor<MT2>& lhs, const DTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid pages of columns" );

      smpAssign( ~lhs, rhs.dm_ );
      smpAssign( ~lhs, rhs.op_( ~lhs ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to dense tensors************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense tensor map expression to a row-major dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense tensor map
   // expression to a row-major dense tensor. Due to the explicit application of the SFINAE
   // principle, this operator can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected and the underlying numeric data type
   // of the operand and the target vector differ.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT2> && !IsSame_v< UnderlyingNumeric_t<MT>
                           , UnderlyingNumeric_t<MT2> > >
      smpAssign( DenseTensor<MT2>& lhs, const DTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RT );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid pages of columns" );

      const RT tmp( rhs.dm_ );
      smpAssign( ~lhs, map( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to dense tensors***************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a dense
   // tensor map expression to a dense tensor. Due to the explicit application of the SFINAE
   // principle, this operator can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpAddAssign( DenseTensor<MT2>& lhs, const DTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RT );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid pages of columns" );

      const RT tmp( rhs.dm_ );
      smpAddAssign( ~lhs, map( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to dense tensors************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a
   // dense tensor map expression to a dense tensor. Due to the explicit application of
   // the SFINAE principle, this operator can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpSubAssign( DenseTensor<MT2>& lhs, const DTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RT );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid pages of columns" );

      const RT tmp( rhs.dm_ );
      smpSubAssign( ~lhs, map( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP Schur product assignment to dense tensors**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP Schur product assignment of a dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized SMP Schur product assignment of a
   // dense tensor map expression to a dense tensor. Due to the explicit application of the
   // SFINAE principle, this operator can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpSchurAssign( DenseTensor<MT2>& lhs, const DTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( RT );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RT );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).pages()   == rhs.pages()  , "Invalid pages of columns" );

      const RT tmp( rhs.dm_ );
      smpSchurAssign( ~lhs, map( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Evaluates the given custom operation on each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \param op The custom operation.
// \return The custom operation applied to each single element of \a dm.
//
// The \a map() function evaluates the given custom operation on each element of the input
// tensor \a dm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a map() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = map( A, []( double a ){ return std::sqrt( a ); } );
   \endcode
*/
template< typename MT    // Type of the dense tensor
        , typename OP >  // Type of the custom operation
inline decltype(auto) map( const DenseTensor<MT>& dm, OP op )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,OP>;
   return ReturnType( ~dm, op );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Evaluates the given custom operation on each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \param op The custom operation.
// \return The custom operation applied to each single element of \a dm.
//
// The \a forEach() function evaluates the given custom operation on each element of the input
// tensor \a dm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a forEach() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = forEach( A, []( double a ){ return std::sqrt( a ); } );
   \endcode
*/
template< typename MT    // Type of the dense tensor
        , typename OP >  // Type of the custom operation
inline decltype(auto) forEach( const DenseTensor<MT>& dm, OP op )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,OP>;
   return ReturnType( ~dm, op );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Applies the \a abs() function to each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The resulting dense tensor.
//
// This function applies the \a abs() function to each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a abs() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = abs( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) abs( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Abs>;
   return ReturnType( ~dm, Abs() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Applies the \a sign() function to each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The resulting dense tensor.
//
// This function applies the \a sign() function to each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a sign() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = sign( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) sign( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Sign>;
   return ReturnType( ~dm, Sign() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Applies the \a floor() function to each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The resulting dense tensor.
//
// This function applies the \a floor() function to each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a floor() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = floor( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) floor( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Floor>;
   return ReturnType( ~dm, Floor() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Applies the \a ceil() function to each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The resulting dense tensor.
//
// This function applies the \a ceil() function to each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a ceil() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = ceil( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) ceil( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Ceil>;
   return ReturnType( ~dm, Ceil() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Applies the \a trunc() function to each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The resulting dense tensor.
//
// This function applies the \a trunc() function to each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a trunc() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = trunc( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) trunc( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Trunc>;
   return ReturnType( ~dm, Trunc() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Applies the \a round() function to each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The resulting dense tensor.
//
// This function applies the \a round() function to each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a round() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = round( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) round( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Round>;
   return ReturnType( ~dm, Round() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a tensor containing the complex conjugate of each single element of \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The conjugate complex of each single element of \a dm.
//
// The \a conj function calculates the complex conjugate of each element of the input tensor
// \a dm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a conj function:

   \code
   blaze::DynamicTensor< complex<double> > A, B;
   // ... Resizing and initialization
   B = conj( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) conj( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Conj>;
   return ReturnType( ~dm, Conj() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the conjugate transpose tensor of \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The conjugate transpose of \a dm.
//
// The \a ctrans function returns an expression representing the conjugate transpose (also called
// adjoint tensor, Hermitian conjugate tensor or transjugate tensor) of the given input tensor
// \a dm.\n
// The following example demonstrates the use of the \a ctrans function:

   \code
   blaze::DynamicTensor< complex<double> > A, B;
   // ... Resizing and initialization
   B = ctrans( A );
   \endcode

// Note that the \a ctrans function has the same effect as manually applying the \a conj and
// \a trans function in any order:

   \code
   B = trans( conj( A ) );  // Computing the conjugate transpose tensor
   B = conj( trans( A ) );  // Computing the conjugate transpose tensor
   \endcode
*/
template< typename MT         // Type of the dense tensor
        , typename ... RTAs>  // Runtime arguments
inline decltype(auto) ctrans( const DenseTensor<MT>& dm, RTAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return trans( conj( ~dm ), args... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a tensor containing the real part of each single element of \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The real part of each single element of \a dm.
//
// The \a real function calculates the real part of each element of the input tensor \a dm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a real function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = real( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) real( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Real>;
   return ReturnType( ~dm, Real() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a tensor containing the imaginary part of each single element of \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The imaginary part of each single element of \a dm.
//
// The \a imag function calculates the imaginary part of each element of the input tensor \a dm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a imag function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = imag( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) imag( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Imag>;
   return ReturnType( ~dm, Imag() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the square root of each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor; all elements must be in the range \f$[0..\infty)\f$.
// \return The square root of each single element of \a dm.
//
// The \a sqrt() function computes the square root of each element of the input tensor \a dm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a sqrt() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = sqrt( A );
   \endcode

// \note All elements are expected to be in the range \f$[0..\infty)\f$. No runtime checks are
// performed to assert this precondition!
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) sqrt( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Sqrt>;
   return ReturnType( ~dm, Sqrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse square root of each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor; all elements must be in the range \f$(0..\infty)\f$.
// \return The inverse square root of each single element of \a dm.
//
// The \a invsqrt() function computes the inverse square root of each element of the input tensor
// \a dm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a invsqrt() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = invsqrt( A );
   \endcode

// \note All elements are expected to be in the range \f$(0..\infty)\f$. No runtime checks are
// performed to assert this precondition!
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) invsqrt( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,InvSqrt>;
   return ReturnType( ~dm, InvSqrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the cubic root of each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor; all elements must be in the range \f$[0..\infty)\f$.
// \return The cubic root of each single element of \a dm.
//
// The \a cbrt() function computes the cubic root of each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a cbrt() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = cbrt( A );
   \endcode

// \note All elements are expected to be in the range \f$[0..\infty)\f$. No runtime checks are
// performed to assert this precondition!
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) cbrt( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Cbrt>;
   return ReturnType( ~dm, Cbrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse cubic root of each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor; all elements must be in the range \f$(0..\infty)\f$.
// \return The inverse cubic root of each single element of \a dm.
//
// The \a invcbrt() function computes the inverse cubic root of each element of the input tensor
// \a dm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a invcbrt() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = invcbrt( A );
   \endcode

// \note All elements are expected to be in the range \f$(0..\infty)\f$. No runtime checks are
// performed to assert this precondition!
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) invcbrt( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,InvCbrt>;
   return ReturnType( ~dm, InvCbrt() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Restricts each single element of the dense tensor \a dm to the range \f$[min..max]\f$.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \param min The lower delimiter.
// \param max The upper delimiter.
// \return The tensor with restricted elements.
//
// The \a clamp() function restricts each element of the input tensor \a dm to the range
// \f$[min..max]\f$. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a clamp() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = clamp( A, -1.0, 1.0 );
   \endcode
*/
template< typename MT    // Type of the dense tensor
        , typename DT >  // Type of the delimiters
inline decltype(auto) clamp( const DenseTensor<MT>& dm, const DT& min, const DT& max )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Clamp<DT>>;
   return ReturnType( ~dm, Clamp<DT>( min, max ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the exponential value for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \param exp The scalar exponent.
// \return The exponential value of each single element of \a dm.
//
// The \a pow() function computes the exponential value for each element of the input tensor
// \a dm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a pow() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = pow( A, 4.2 );
   \endcode
*/
template< typename MT  // Type of the dense tensor
        , typename ST  // Type of the scalar exponent
        , EnableIf_t< IsNumeric_v<ST> >* = nullptr >
inline decltype(auto) pow( const DenseTensor<MT>& dm, ST exp )
{
   BLAZE_FUNCTION_TRACE;

   using ScalarType = MultTrait_t< UnderlyingBuiltin_t<MT>, ST >;
   using ReturnType = const DTensMapExpr<MT,UnaryPow<ScalarType>>;
   return ReturnType( ~dm, UnaryPow<ScalarType>( exp ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes \f$ e^x \f$ for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The resulting dense tensor.
//
// The \a exp() function computes \f$ e^x \f$ for each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a exp() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = exp( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) exp( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Exp>;
   return ReturnType( ~dm, Exp() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes \f$ 2^x \f$ for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The resulting dense tensor.
//
// The \a exp2() function computes \f$ 2^x \f$ for each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a exp2() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = exp2( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) exp2( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Exp2>;
   return ReturnType( ~dm, Exp2() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes \f$ 10^x \f$ for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The resulting dense tensor.
//
// The \a exp10() function computes \f$ 10^x \f$ for each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a exp10() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = exp10( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) exp10( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Exp10>;
   return ReturnType( ~dm, Exp10() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the natural logarithm for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor; all elements must be in the range \f$[0..\infty)\f$.
// \return The natural logarithm of each single element of \a dm.
//
// The \a log() function computes natural logarithm for each element of the input tensor \a dm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a log() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = log( A );
   \endcode

// \note All elements are expected to be in the range \f$[0..\infty)\f$. No runtime checks are
// performed to assert this precondition!
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) log( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Log>;
   return ReturnType( ~dm, Log() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the binary logarithm for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor; all elements must be in the range \f$[0..\infty)\f$.
// \return The binary logarithm of each single element of \a dm.
//
// The \a log2() function computes binary logarithm for each element of the input tensor \a dm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a log2() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = log2( A );
   \endcode

// \note All elements are expected to be in the range \f$[0..\infty)\f$. No runtime checks are
// performed to assert this precondition!
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) log2( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Log2>;
   return ReturnType( ~dm, Log2() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the common logarithm for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor; all elements must be in the range \f$[0..\infty)\f$.
// \return The common logarithm of each single element of \a dm.
//
// The \a log10() function computes common logarithm for each element of the input tensor \a dm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a log10() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = log10( A );
   \endcode

// \note All elements are expected to be in the range \f$[0..\infty)\f$. No runtime checks are
// performed to assert this precondition!
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) log10( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Log10>;
   return ReturnType( ~dm, Log10() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the sine for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The sine of each single element of \a dm.
//
// The \a sin() function computes the sine for each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a sin() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = sin( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) sin( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Sin>;
   return ReturnType( ~dm, Sin() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse sine for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor; all elements must be in the range \f$[-1..1]\f$.
// \return The inverse sine of each single element of \a dm.
//
// The \a asin() function computes the inverse sine for each element of the input tensor \a dm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a asin() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = asin( A );
   \endcode

// \note All elements are expected to be in the range \f$[-1..1]\f$. No runtime checks are
// performed to assert this precondition!
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) asin( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Asin>;
   return ReturnType( ~dm, Asin() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the hyperbolic sine for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The hyperbolic sine of each single element of \a dm.
//
// The \a sinh() function computes the hyperbolic sine for each element of the input tensor \a dm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a sinh() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = sinh( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) sinh( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Sinh>;
   return ReturnType( ~dm, Sinh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse hyperbolic sine for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The inverse hyperbolic sine of each single element of \a dm.
//
// The \a asinh() function computes the inverse hyperbolic sine for each element of the input
// tensor \a dm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a asinh() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = asinh( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) asinh( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Asinh>;
   return ReturnType( ~dm, Asinh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the cosine for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The cosine of each single element of \a dm.
//
// The \a cos() function computes the cosine for each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a cos() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = cos( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) cos( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Cos>;
   return ReturnType( ~dm, Cos() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse cosine for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor; all elements must be in the range \f$[-1..1]\f$.
// \return The inverse cosine of each single element of \a dm.
//
// The \a acos() function computes the inverse cosine for each element of the input tensor \a dm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a acos() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = acos( A );
   \endcode

// \note All elements are expected to be in the range \f$[-1..1]\f$. No runtime checks are
// performed to assert this precondition!
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) acos( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Acos>;
   return ReturnType( ~dm, Acos() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the hyperbolic cosine for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The hyperbolic cosine of each single element of \a dm.
//
// The \a cosh() function computes the hyperbolic cosine for each element of the input tensor
// \a dm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a cosh() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = cosh( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) cosh( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Cosh>;
   return ReturnType( ~dm, Cosh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse hyperbolic cosine for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor; all elements must be in the range \f$[1..\infty)\f$.
// \return The inverse hyperbolic cosine of each single element of \a dm.
//
// The \a acosh() function computes the inverse hyperbolic cosine for each element of the input
// tensor \a dm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a acosh() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = acosh( A );
   \endcode

// \note All elements are expected to be in the range \f$[1..\infty)\f$. No runtime checks are
// performed to assert this precondition!
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) acosh( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Acosh>;
   return ReturnType( ~dm, Acosh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the tangent for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The tangent of each single element of \a dm.
//
// The \a tan() function computes the tangent for each element of the input tensor \a dm. The
// function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a tan() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = tan( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) tan( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Tan>;
   return ReturnType( ~dm, Tan() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse tangent for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The inverse tangent of each single element of \a dm.
//
// The \a atan() function computes the inverse tangent for each element of the input tensor \a dm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a atan() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = atan( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) atan( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Atan>;
   return ReturnType( ~dm, Atan() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the hyperbolic tangent for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor; all elements must be in the range \f$[-1..1]\f$.
// \return The hyperbolic tangent of each single element of \a dm.
//
// The \a tanh() function computes the hyperbolic tangent for each element of the input tensor
// \a dm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a tanh() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = tanh( A );
   \endcode

// \note All elements are expected to be in the range \f$[-1..1]\f$. No runtime checks are
// performed to assert this precondition!
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) tanh( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Tanh>;
   return ReturnType( ~dm, Tanh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the inverse hyperbolic tangent for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor; all elements must be in the range \f$[-1..1]\f$.
// \return The inverse hyperbolic tangent of each single element of \a dm.
//
// The \a atanh() function computes the inverse hyperbolic tangent for each element of the input
// tensor \a dm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a atanh() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = atanh( A );
   \endcode

// \note All elements are expected to be in the range \f$[-1..1]\f$. No runtime checks are
// performed to assert this precondition!
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) atanh( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Atanh>;
   return ReturnType( ~dm, Atanh() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the error function for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The error function of each single element of \a dm.
//
// The \a erf() function computes the error function for each element of the input tensor \a dm.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a erf() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = erf( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) erf( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Erf>;
   return ReturnType( ~dm, Erf() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the complementary error function for each single element of the dense tensor \a dm.
// \ingroup dense_tensor
//
// \param dm The input tensor.
// \return The complementary error function of each single element of \a dm.
//
// The \a erfc() function computes the complementary error function for each element of the input
// tensor \a dm. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a erfc() function:

   \code
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization
   B = erfc( A );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) erfc( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensMapExpr<MT,Erfc>;
   return ReturnType( ~dm, Erfc() );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Absolute value function for dense tensor absolute value expressions.
// \ingroup dense_tensor
//
// \param dm The absolute value dense tensor expression.
// \return The absolute value of each single element of \a dm.
//
// This function implements a performance optimized treatment of the absolute value operation
// on a dense tensor absolute value expression.
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) abs( const DTensMapExpr<MT,Abs>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return dm;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Applies the \a sign() function for dense tensor \a sign() expressions.
// \ingroup dense_tensor
//
// \param dm The dense tensor \a sign() expression.
// \return The sign of each single element of \a dm.
//
// This function implements a performance optimized treatment of the \a sign() operation on a
// dense tensor \a sign() expression.
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) sign( const DTensMapExpr<MT,Sign>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return dm;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Applies the \a floor() function to a dense tensor \a floor() expressions.
// \ingroup dense_tensor
//
// \param dm The dense tensor \a floor() expression.
// \return The resulting dense tensor.
//
// This function implements a performance optimized treatment of the \a floor() operation on
// a dense tensor \a floor() expression.
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) floor( const DTensMapExpr<MT,Floor>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return dm;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Applies the \a ceil() function to a dense tensor \a ceil() expressions.
// \ingroup dense_tensor
//
// \param dm The dense tensor \a ceil() expression.
// \return The resulting dense tensor.
//
// This function implements a performance optimized treatment of the \a ceil() operation on
// a dense tensor \a ceil() expression.
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) ceil( const DTensMapExpr<MT,Ceil>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return dm;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Applies the \a trunc() function to a dense tensor \a trunc() expressions.
// \ingroup dense_tensor
//
// \param dm The dense tensor \a trunc() expression.
// \return The resulting dense tensor.
//
// This function implements a performance optimized treatment of the \a trunc() operation on
// a dense tensor \a trunc() expression.
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) trunc( const DTensMapExpr<MT,Trunc>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return dm;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Applies the \a round() function to a dense tensor \a round() expressions.
// \ingroup dense_tensor
//
// \param dm The dense tensor \a round() expression.
// \return The resulting dense tensor.
//
// This function implements a performance optimized treatment of the \a round() operation on
// a dense tensor \a round() expression.
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) round( const DTensMapExpr<MT,Round>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return dm;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Complex conjugate function for complex conjugate dense tensor expressions.
// \ingroup dense_tensor
//
// \param dm The complex conjugate dense tensor expression.
// \return The original dense tensor.
//
// This function implements a performance optimized treatment of the complex conjugate operation
// on a dense tensor complex conjugate expression. It returns an expression representing the
// original dense tensor:

   \code
   blaze::DynamicTensor< complex<double> > A, B;
   // ... Resizing and initialization
   B = conj( conj( A ) );
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) conj( const DTensMapExpr<MT,Conj>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return dm.operand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Complex conjugate function for conjugate transpose dense tensor expressions.
// \ingroup dense_tensor
//
// \param dm The conjugate transpose dense tensor expression.
// \return The transpose dense tensor.
//
// This function implements a performance optimized treatment of the complex conjugate operation
// on a dense tensor conjugate transpose expression. It returns an expression representing the
// transpose of the dense tensor:

   \code
   blaze::DynamicTensor< complex<double> > A, B;
   // ... Resizing and initialization
   B = conj( ctrans( A ) );
   \endcode
*/
template< typename MT         // Type of the dense tensor
        , typename Conj       // Type of the custom operation
        , size_t ... CTAs>  // Compile time arguments
inline decltype(auto) conj( const DTensTransExpr<DTensMapExpr<MT,Conj>, CTAs... >& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensTransExpr<MT,CTAs...>;
   return ReturnType( dm.operand().operand(), (~dm).idces() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Real part function for real part dense tensor expressions.
// \ingroup dense_tensor
//
// \param dm The real part dense tensor expression.
// \return The real part of each single element of \a dm.
//
// This function implements a performance optimized treatment of the real part operation on
// a dense tensor real part expression.
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) real( const DTensMapExpr<MT,Real>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return dm;
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
template< typename MT, typename OP >
struct IsAligned< DTensMapExpr<MT,OP> >
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
template< typename MT, typename OP >
struct IsPadded< DTensMapExpr<MT,OP> >
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
template< typename MT, typename OP >
struct IsSymmetric< DTensMapExpr<MT,OP> >
   : public YieldsSymmetric<OP,MT>
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
template< typename MT, typename OP >
struct IsHermitian< DTensMapExpr<MT,OP> >
   : public YieldsHermitian<OP,MT>
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
template< typename MT, typename OP >
struct IsLower< DTensMapExpr<MT,OP> >
   : public YieldsLower<OP,MT>
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
template< typename MT, typename OP >
struct IsUniLower< DTensMapExpr<MT,OP> >
   : public YieldsUniLower<OP,MT>
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
template< typename MT, typename OP >
struct IsStrictlyLower< DTensMapExpr<MT,OP> >
   : public YieldsStrictlyLower<OP,MT>
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
template< typename MT, typename OP >
struct IsUpper< DTensMapExpr<MT,OP> >
   : public YieldsUpper<OP,MT>
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
template< typename MT, typename OP >
struct IsUniUpper< DTensMapExpr<MT,OP> >
   : public YieldsUniUpper<OP,MT>
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
template< typename MT, typename OP >
struct IsStrictlyUpper< DTensMapExpr<MT,OP> >
   : public YieldsStrictlyUpper<OP,MT>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
