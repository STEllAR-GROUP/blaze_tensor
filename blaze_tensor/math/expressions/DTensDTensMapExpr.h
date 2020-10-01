//=================================================================================================
/*!
//  \file blaze_tensor/math/expressions/DTensDTensMapExpr.h
//  \brief Header file for the dense tensor/dense tensor map expression
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSDTENSMAPEXPR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSDTENSMAPEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <utility>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/DMatDMatMapExpr.h>
#include <blaze/math/expressions/MatMatMapExpr.h>
#include <blaze/math/typetraits/IsSIMDEnabled.h>
#include <blaze/math/typetraits/YieldsHermitian.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>
#include <blaze_tensor/math/expressions/TensTensMapExpr.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DMATDMATMAPEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for the dense tensor-dense tensor map() function.
// \ingroup dense_tensor_expression
//
// The DTensDTensMapExpr class represents the compile time expression for the pairwise evaluation
// of a binary custom operation on the elements of two dense tensors with identical storage order
// via the map() function.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2  // Type of the right-hand side dense tensor
        , typename OP >  // Type of the custom operation
class DTensDTensMapExpr
   : public TensTensMapExpr< DenseTensor< DTensDTensMapExpr<MT1,MT2,OP> > >
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   using RT1 = ResultType_t<MT1>;     //!< Result type of the left-hand side dense tensor expression.
   using RT2 = ResultType_t<MT2>;     //!< Result type of the right-hand side dense tensor expression.
   using ET1 = ElementType_t<MT1>;    //!< Element type of the left-hand side dense tensor expression.
   using ET2 = ElementType_t<MT2>;    //!< Element type of the right-hand side dense tensor expression.
   using RN1 = ReturnType_t<MT1>;     //!< Return type of the left-hand side dense tensor expression.
   using RN2 = ReturnType_t<MT2>;     //!< Return type of the right-hand side dense tensor expression.
   using CT1 = CompositeType_t<MT1>;  //!< Composite type of the left-hand side dense tensor expression.
   using CT2 = CompositeType_t<MT2>;  //!< Composite type of the right-hand side dense tensor expression.

   //! Definition of the HasSIMDEnabled type trait.
   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasSIMDEnabled, simdEnabled );

   //! Definition of the HasLoad type trait.
   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasLoad, load );
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the map expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the serial evaluation strategy of the map expression. In case either of the two dense
       tensor operands requires an intermediate evaluation, \a useAssign will be set to 1 and
       the addition expression will be evaluated via the \a assign function family. Otherwise
       \a useAssign will be set to 0 and the expression will be evaluated via the subscript
       operator. */
   static constexpr bool useAssign = ( RequiresEvaluation_v<MT1> || RequiresEvaluation_v<MT2> );

   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool UseAssign_v = useAssign;
   /*! \endcond */
   //**********************************************************************************************

   //**Parallel evaluation strategy****************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! This variable template is a helper for the selection of the parallel evaluation strategy.
       In case at least one of the two dense tensor operands is not SMP assignable and at least
       one of the two operands requires an intermediate evaluation, the variable is set to 1 and
       the expression specific evaluation strategy is selected. Otherwise the variable is set to
       0 and the default strategy is chosen. */
   template< typename MT >
   static constexpr bool UseSMPAssign_v =
      ( ( !MT1::smpAssignable || !MT2::smpAssignable ) && useAssign );
   /*! \endcond */
   //**********************************************************************************************

public:
   //**Type definitions****************************************************************************
   using This          = DTensDTensMapExpr<MT1,MT2,OP>;   //!< Type of this DTensDTensMapExpr instance.
   using ResultType    = MapTrait_t<RT1,RT2,OP>;          //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;      //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;     //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<ResultType>;       //!< Resulting element type.

   //! Return type for expression template evaluations.
   using ReturnType = decltype( std::declval< OP >()( std::declval< RN1 >(), std::declval< RN2 >() ) );

   //! Data type for composite expression templates.
   using CompositeType = If_t< useAssign, const ResultType, const DTensDTensMapExpr& >;

   //! Composite type of the left-hand side dense tensor expression.
   using LeftOperand = If_t< IsExpression_v<MT1>, const MT1, const MT1& >;

   //! Composite type of the right-hand side dense tensor expression.
   using RightOperand = If_t< IsExpression_v<MT2>, const MT2, const MT2& >;

   //! Data type of the custom unary operation.
   using Operation = OP;

   //! Type for the assignment of the left-hand side dense tensor operand.
   using LT = If_t< RequiresEvaluation_v<MT1>, const RT1, CT1 >;

   //! Type for the assignment of the right-hand side dense tensor operand.
   using RT = If_t< RequiresEvaluation_v<MT2>, const RT2, CT2 >;
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

      //! ConstIterator type of the left-hand side dense tensor expression.
      using LeftIteratorType = ConstIterator_t<MT1>;

      //! ConstIterator type of the right-hand side dense tensor expression.
      using RightIteratorType = ConstIterator_t<MT2>;
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the ConstIterator class.
      //
      // \param left Iterator to the initial left-hand side element.
      // \param right Iterator to the initial right-hand side element.
      // \param op The custom unary operation.
      */
      explicit inline ConstIterator( LeftIteratorType left, RightIteratorType right, OP op )
         : left_ ( left  )  // Iterator to the current left-hand side element
         , right_( right )  // Iterator to the current right-hand side element
         , op_   ( op    )  // The custom unary operation
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline ConstIterator& operator+=( size_t inc ) {
         left_  += inc;
         right_ += inc;
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
         left_  -= dec;
         right_ -= dec;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline ConstIterator& operator++() {
         ++left_;
         ++right_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator++( int ) {
         return ConstIterator( left_++, right_++, op_ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline ConstIterator& operator--() {
         --left_;
         --right_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator--( int ) {
         return ConstIterator( left_--, right_--, op_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReturnType operator*() const {
         return op_( *left_, *right_ );
      }
      //*******************************************************************************************

      //**Load function****************************************************************************
      /*!\brief Access to the SIMD elements of the tensor.
      //
      // \return The resulting SIMD element.
      */
      inline auto load() const noexcept {
         return op_.load( left_.load(), right_.load() );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const ConstIterator& rhs ) const {
         return left_ == rhs.left_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const ConstIterator& rhs ) const {
         return left_ != rhs.left_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const ConstIterator& rhs ) const {
         return left_ < rhs.left_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const ConstIterator& rhs ) const {
         return left_ > rhs.left_;
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const ConstIterator& rhs ) const {
         return left_ <= rhs.left_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const ConstIterator& rhs ) const {
         return left_ >= rhs.left_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const ConstIterator& rhs ) const {
         return left_ - rhs.left_;
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
         return ConstIterator( it.left_ + inc, it.right_ + inc, it.op_ );
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
         return ConstIterator( it.left_ + inc, it.right_ + inc, it.op_ );
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
         return ConstIterator( it.left_ - dec, it.right_ - dec, it.op_ );
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      LeftIteratorType  left_;   //!< Iterator to the current left-hand side element.
      RightIteratorType right_;  //!< Iterator to the current right-hand side element.
      OP                op_;     //!< The custom unary operation.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled =
      ( MT1::simdEnabled && MT2::simdEnabled &&
        If_t< HasSIMDEnabled_v<OP>, GetSIMDEnabled<OP,ET1,ET2>, HasLoad<OP> >::value );

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = ( MT1::smpAssignable && MT2::smpAssignable );
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   static constexpr size_t SIMDSIZE = SIMDTrait<ElementType>::size;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DTensDTensMapExpr class.
   //
   // \param lhs The left-hand side dense tensor operand of the map expression.
   // \param rhs The right-hand side dense tensor operand of the map expression.
   // \param op The custom unary operation.
   */
   explicit inline DTensDTensMapExpr( const MT1& lhs, const MT2& rhs, OP op ) noexcept
      : lhs_( lhs )  // Left-hand side dense tensor of the map expression
      , rhs_( rhs )  // Right-hand side dense tensor of the map expression
      , op_ ( op  )  // The custom unary operation
   {}
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 2D-access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator()( size_t k, size_t i, size_t j ) const {
      BLAZE_INTERNAL_ASSERT( i < lhs_.rows()   , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < lhs_.columns(), "Invalid column access index" );
      BLAZE_INTERNAL_ASSERT( k < lhs_.pages(),   "Invalid page access index" );
      return op_( lhs_(k,i,j), rhs_(k,i,j) );
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
      if( i >= lhs_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= lhs_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      if( k >= lhs_.pages() ) {
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
   BLAZE_ALWAYS_INLINE auto load( size_t k, size_t i, size_t j ) const noexcept {
      BLAZE_INTERNAL_ASSERT( i < lhs_.rows()   , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < lhs_.columns(), "Invalid column access index" );
      BLAZE_INTERNAL_ASSERT( k < lhs_.pages(),   "Invalid page access index" );
      BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );
      BLAZE_INTERNAL_ASSERT( k % SIMDSIZE == 0UL, "Invalid page access index" );
      return op_.load( lhs_.load(k,i,j), rhs_.load(k,i,j) );
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of row \a i.
   //
   // \param i The row index.
   // \return Iterator to the first non-zero element of row \a i.
   */
   inline ConstIterator begin( size_t i, size_t k ) const {
      return ConstIterator( lhs_.begin(i,k), rhs_.begin(i,k), op_ );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row \a i.
   //
   // \param i The row index.
   // \return Iterator just past the last non-zero element of row \a i.
   */
   inline ConstIterator end( size_t i, size_t k ) const {
      return ConstIterator( lhs_.end(i,k), rhs_.end(i,k), op_ );
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the tensor.
   //
   // \return The number of rows of the tensor.
   */
   inline size_t rows() const noexcept {
      return lhs_.rows();
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the tensor.
   //
   // \return The number of columns of the tensor.
   */
   inline size_t columns() const noexcept {
      return lhs_.columns();
   }
   //**********************************************************************************************

   //**Pages function****************************************************************************
   /*!\brief Returns the current number of pages of the tensor.
   //
   // \return The number of pages of the tensor.
   */
   inline size_t pages() const noexcept {
      return lhs_.pages();
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side dense tensor operand.
   //
   // \return The left-hand side dense tensor operand.
   */
   inline LeftOperand leftOperand() const noexcept {
      return lhs_;
   }
   //**********************************************************************************************

   //**Right operand access************************************************************************
   /*!\brief Returns the right-hand side dense tensor operand.
   //
   // \return The right-hand side dense tensor operand.
   */
   inline RightOperand rightOperand() const noexcept {
      return rhs_;
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
      return ( IsExpression_v<MT1> && lhs_.canAlias( alias ) ) ||
             ( IsExpression_v<MT2> && rhs_.canAlias( alias ) );
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
      return ( lhs_.isAliased( alias ) || rhs_.isAliased( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const noexcept {
      return lhs_.isAligned() && rhs_.isAligned();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return lhs_.canSMPAssign() && rhs_.canSMPAssign();
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  lhs_;  //!< Left-hand side dense tensor of the map expression.
   RightOperand rhs_;  //!< Right-hand side dense tensor of the map expression.
   Operation    op_;   //!< The custom unary operation.
   //**********************************************************************************************

   //**Assignment to dense tensors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense tensor-dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense tensor-dense
   // tensor map expression to a dense tensor. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case either of the two
   // operands requires an intermediate evaluation.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT> >
      assign( DenseTensor<MT2>& lhs, const DTensDTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages(),   "Invalid number of pages" );

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense tensor operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.pages()   == rhs.lhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == rhs.rhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (*lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (*lhs).columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == (*lhs).pages()    , "Invalid number of columns" );

      assign( *lhs, map( A, B, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense tensors*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a dense tensor-dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a dense
   // tensor-dense tensor map expression to a dense tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case
   // either of the two operands requires an intermediate evaluation.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT> >
      addAssign( DenseTensor<MT2>& lhs, const DTensDTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages(),   "Invalid number of pages" );

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense tensor operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.pages()   == rhs.lhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == rhs.rhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (*lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (*lhs).columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == (*lhs).pages()    , "Invalid number of columns" );

      addAssign( *lhs, map( A, B, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to dense tensors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a dense tensor-dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a dense
   // tensor-dense tensor map expression to a dense tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case
   // either of the two operands requires an intermediate evaluation.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT> >
      subAssign( DenseTensor<MT2>& lhs, const DTensDTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages(),   "Invalid number of pages" );

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense tensor operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.pages()   == rhs.lhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == rhs.rhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (*lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (*lhs).columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == (*lhs).pages()    , "Invalid number of columns" );

      subAssign( *lhs, map( A, B, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Schur product assignment to dense tensors**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Schur product assignment of a dense tensor-dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized Schur product assignment of a dense
   // tensor-dense tensor map expression to a dense tensor. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case either
   // of the two operands requires an intermediate evaluation.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT> >
      schurAssign( DenseTensor<MT2>& lhs, const DTensDTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages(),   "Invalid number of pages" );

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense tensor operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.pages()   == rhs.lhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == rhs.rhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (*lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (*lhs).columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == (*lhs).pages()    , "Invalid number of columns" );

      schurAssign( *lhs, map( A, B, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to dense tensors************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense tensor-dense tensor map expression to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense tensor-dense
   // tensor map expression to a dense tensor. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT> >
      smpAssign( DenseTensor<MT2>& lhs, const DTensDTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages(),   "Invalid number of pages" );

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side dense tensor operand
      RT B( rhs.rhs_ );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.pages()   == rhs.lhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == rhs.rhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (*lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (*lhs).columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == (*lhs).pages()    , "Invalid number of columns" );

      smpAssign( *lhs, map( A, B, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to dense tensors***************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a dense tensor-dense tensor map expression to a dense
   //        tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a dense
   // tensor-dense tensor map expression to a dense tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT> >
      smpAddAssign( DenseTensor<MT2>& lhs, const DTensDTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages(),   "Invalid number of pages" );

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side dense tensor operand
      RT B( rhs.rhs_ );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.pages()   == rhs.lhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == rhs.rhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (*lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (*lhs).columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == (*lhs).pages()    , "Invalid number of columns" );

      smpAddAssign( *lhs, map( A, B, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to dense tensors************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a dense tensor-dense tensor map expression to a dense
   //        tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a dense
   // tensor-dense tensor map expression to a dense tensor. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT> >
      smpSubAssign( DenseTensor<MT2>& lhs, const DTensDTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages(),   "Invalid number of pages" );

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side dense tensor operand
      RT B( rhs.rhs_ );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.pages()   == rhs.lhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == rhs.rhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (*lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (*lhs).columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == (*lhs).pages()    , "Invalid number of columns" );

      smpSubAssign( *lhs, map( A, B, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP Schur product assignment to dense tensors**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP Schur product assignment of a dense tensor-dense tensor map expression to a
   //        dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side map expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized SMP Schur product assignment of a
   // dense tensor-dense tensor map expression to a dense tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT> >
      smpSchurAssign( DenseTensor<MT2>& lhs, const DTensDTensMapExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages(),   "Invalid number of pages" );

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side dense tensor operand
      RT B( rhs.rhs_ );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.pages()   == rhs.lhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == rhs.rhs_.pages()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (*lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (*lhs).columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages()   == (*lhs).pages()    , "Invalid number of columns" );

      smpSchurAssign( *lhs, map( A, B, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT2 );
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
/*!\brief Evaluates the given binary operation on each single element of the dense tensors
//        \a lhs and \a rhs.
// \ingroup dense_tensor
//
// \param lhs The left-hand side dense tensor operand.
// \param rhs The right-hand side dense tensor operand.
// \param op The custom, binary operation.
// \return The binary operation applied to each single element of \a lhs and \a rhs.
// \exception std::invalid_argument Tensor sizes do not match.
//
// The \a map() function evaluates the given binary operation on each element of the input
// tensors \a lhs and \a rhs. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a map() function:

   \code
   blaze::DynamicTensor<double> A, B, C;
   // ... Resizing and initialization
   C = map( A, B, []( double x, double y ){ return std::min( x, y ); } );
   \endcode
*/
template< typename MT1   // Type of the left-hand side dense tensor
        , typename MT2   // Type of the right-hand side dense tensor
        , typename OP >  // Type of the custom operation
inline decltype(auto)
   map( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs, OP op )
{
   BLAZE_FUNCTION_TRACE;

   if( (*lhs).rows() != (*rhs).rows() || (*lhs).columns() != (*rhs).columns()  || (*lhs).pages() != (*rhs).pages()) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   using ReturnType = const DTensDTensMapExpr<MT1,MT2,OP>;
   return ReturnType( *lhs, *rhs, op );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the componentwise minimum of the dense tensors \a lhs and \a rhs.
// \ingroup dense_tensor
//
// \param lhs The left-hand side dense tensor operand.
// \param rhs The right-hand side dense tensor operand.
// \return The resulting dense tensor.
//
// This function computes the componentwise minimum of the two dense tensors \a lhs and \a rhs.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a min() function:

   \code
   blaze::DynamicTensor<double> A, B, C;
   // ... Resizing and initialization
   C = min( A, B );
   \endcode
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side dense tensor
inline decltype(auto)
   min( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return map( *lhs, *rhs, Min() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the componentwise maximum of the dense tensors \a lhs and \a rhs.
// \ingroup dense_tensor
//
// \param lhs The left-hand side dense tensor operand.
// \param rhs The right-hand side dense tensor operand.
// \return The resulting dense tensor.
//
// This function computes the componentwise maximum of the two dense tensors \a lhs and \a rhs.
// The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a max() function:

   \code
   blaze::DynamicTensor<double> A, B, C;
   // ... Resizing and initialization
   C = max( A, B );
   \endcode
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side dense tensor
inline decltype(auto)
   max( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return map( *lhs, *rhs, Max() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the componentwise hypotenous for the dense tensors \a lhs and \a rhs.
// \ingroup dense_tensor
//
// \param lhs The left-hand side dense tensor operand.
// \param rhs The right-hand side dense tensor operand.
// \return The resulting dense tensor.
//
// The \a hypot() function computes the componentwise hypotenous for the two dense tensors
// \a lhs and \a rhs. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a hypot() function:

   \code
   blaze::DynamicTensor<double> A, B, C;
   // ... Resizing and initialization
   C = hypot( A, B );
   \endcode
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side dense tensor
inline decltype(auto)
   hypot( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return map( *lhs, *rhs, Hypot() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the componentwise exponential value for the dense tensors \a lhs and \a rhs.
// \ingroup dense_tensor
//
// \param lhs The left-hand side dense tensor operand.
// \param rhs The right-hand side dense tensor operand.
// \return The resulting dense tensor.
//
// The \a pow() function computes the componentwise exponential value for the two dense tensors
// \a lhs and \a rhs. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a pow() function:

   \code
   blaze::DynamicTensor<double> A, B, C;
   // ... Resizing and initialization
   C = pow( A, B );
   \endcode
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side dense tensor
inline decltype(auto)
   pow( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return map( *lhs, *rhs, Pow() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Computes the multi-valued inverse tangent of the dense tensors \a lhs and \a rhs.
// \ingroup dense_tensor
//
// \param lhs The left-hand side dense tensor operand.
// \param rhs The right-hand side dense tensor operand.
// \return The resulting dense tensor.
//
// This function computes the multi-valued inverse tangent of the two dense tensor \a lhs and
// \a rhs. The function returns an expression representing this operation.\n
// The following example demonstrates the use of the \a max() function:

   \code
   blaze::DynamicTensor<double> A, B, C;
   // ... Resizing and initialization
   C = atan2( A, B );
   \endcode
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side dense tensor
inline decltype(auto)
   atan2( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return map( *lhs, *rhs, Atan2() );
}
//*************************************************************************************************




//=================================================================================================
//
//  ISALIGNED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2, typename OP >
struct IsAligned< DTensDTensMapExpr<MT1,MT2,OP> >
   : public BoolConstant< IsAligned_v<MT1> && IsAligned_v<MT2> >
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
template< typename MT1, typename MT2, typename OP >
struct IsPadded< DTensDTensMapExpr<MT1,MT2,OP> >
   : public BoolConstant< IsPadded_v<MT1> && IsPadded_v<MT2> >
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
template< typename MT1, typename MT2, typename OP >
struct IsSymmetric< DTensDTensMapExpr<MT1,MT2,OP> >
   : public YieldsSymmetric<OP,MT1,MT2>
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
template< typename MT1, typename MT2, typename OP >
struct IsHermitian< DTensDTensMapExpr<MT1,MT2,OP> >
   : public YieldsHermitian<OP,MT1,MT2>
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
template< typename MT1, typename MT2, typename OP >
struct IsLower< DTensDTensMapExpr<MT1,MT2,OP> >
   : public YieldsLower<OP,MT1,MT2>
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
template< typename MT1, typename MT2, typename OP >
struct IsUniLower< DTensDTensMapExpr<MT1,MT2,OP> >
   : public YieldsUniLower<OP,MT1,MT2>
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
template< typename MT1, typename MT2, typename OP >
struct IsStrictlyLower< DTensDTensMapExpr<MT1,MT2,OP> >
   : public YieldsStrictlyLower<OP,MT1,MT2>
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
template< typename MT1, typename MT2, typename OP >
struct IsUpper< DTensDTensMapExpr<MT1,MT2,OP> >
   : public YieldsUpper<OP,MT1,MT2>
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
template< typename MT1, typename MT2, typename OP >
struct IsUniUpper< DTensDTensMapExpr<MT1,MT2,OP> >
   : public YieldsUniUpper<OP,MT1,MT2>
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
template< typename MT1, typename MT2, typename OP >
struct IsStrictlyUpper< DTensDTensMapExpr<MT1,MT2,OP> >
   : public YieldsStrictlyUpper<OP,MT1,MT2>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
