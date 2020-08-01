//=================================================================================================
/*!
//  \file blaze_tensor/math/expressions/DTensDTensSchurExpr.h
//  \brief Header file for the dense tensor/dense tensor Schur product expression
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSDTENSSCHUREXPR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSDTENSSCHUREXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <utility>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/SchurExpr.h>
#include <blaze/math/constraints/StorageOrder.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/SchurExpr.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/SchurTrait.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsCommutative.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsOperation.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsTemporary.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/IsUniUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/Types.h>
#include <blaze/util/MaybeUnused.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/SchurExpr.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DTENSDTENSSCHUREXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for dense tensor-dense tensor Schur products.
// \ingroup dense_tensor_expression
//
// The DTensDTensSchurExpr class represents the compile time expression for Schur products between
// dense tensors with identical storage order.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side dense tensor
class DTensDTensSchurExpr
   : public SchurExpr< DenseTensor< DTensDTensSchurExpr<MT1,MT2> > >
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   using RT1 = ResultType_t<MT1>;     //!< Result type of the left-hand side dense tensor expression.
   using RT2 = ResultType_t<MT2>;     //!< Result type of the right-hand side dense tensor expression.
   using RN1 = ReturnType_t<MT1>;     //!< Return type of the left-hand side dense tensor expression.
   using RN2 = ReturnType_t<MT2>;     //!< Return type of the right-hand side dense tensor expression.
   using CT1 = CompositeType_t<MT1>;  //!< Composite type of the left-hand side dense tensor expression.
   using CT2 = CompositeType_t<MT2>;  //!< Composite type of the right-hand side dense tensor expression.
   using ET1 = ElementType_t<MT1>;    //!< Element type of the left-hand side dense tensor expression.
   using ET2 = ElementType_t<MT2>;    //!< Element type of the right-hand side dense tensor expression.
   //**********************************************************************************************

   //**Return type evaluation**********************************************************************
   //! Compilation switch for the selection of the subscript operator return type.
   /*! The \a returnExpr compile time constant expression is a compilation switch for the
       selection of the \a ReturnType. If either tensor operand returns a temporary vector
       or tensor, \a returnExpr will be set to \a false and the subscript operator will
       return it's result by value. Otherwise \a returnExpr will be set to \a true and
       the subscript operator may return it's result as an expression. */
   static constexpr bool returnExpr = ( !IsTemporary_v<RN1> && !IsTemporary_v<RN2> );

   //! Expression return type for the subscript operator.
   using ExprReturnType = decltype( std::declval<RN1>() * std::declval<RN2>() );
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the Schur product expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for the
       serial evaluation strategy of the Schur product expression. In case either of the two dense
       tensor operands requires an intermediate evaluation or the subscript operator can only
       return by value, \a useAssign will be set to 1 and the Schur product expression will be
       evaluated via the \a assign function family. Otherwise \a useAssign will be set to 0 and
       the expression will be evaluated via the function call operator. */
   static constexpr bool useAssign =
      ( RequiresEvaluation_v<MT1> || RequiresEvaluation_v<MT2> || !returnExpr );

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
       In case at least one of the two tensor operands is not SMP assignable and at least one of
       the two operands requires an intermediate evaluation, the variable is set to 1 and the
       expression specific evaluation strategy is selected. Otherwise the variable is set to 0
       and the default strategy is chosen. */
   template< typename MT >
   static constexpr bool UseSMPAssign_v =
      ( ( !MT1::smpAssignable || !MT2::smpAssignable ) && useAssign );
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This          = DTensDTensSchurExpr<MT1,MT2>;   //!< Type of this DTensDTensSchurExpr instance.
   using ResultType    = SchurTrait_t<RT1,RT2>;          //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;     //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;    //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<ResultType>;      //!< Resulting element type.

   //! Return type for expression template evaluations.
   using ReturnType = const If_t< returnExpr, ExprReturnType, ElementType >;

   //! Data type for composite expression templates.
   using CompositeType = If_t< useAssign, const ResultType, const DTensDTensSchurExpr& >;

   //! Composite type of the left-hand side dense tensor expression.
   using LeftOperand = If_t< IsExpression_v<MT1>, const MT1, const MT1& >;

   //! Composite type of the right-hand side dense tensor expression.
   using RightOperand = If_t< IsExpression_v<MT2>, const MT2, const MT2& >;
   //**********************************************************************************************

   //**ConstIterator class definition**************************************************************
   /*!\brief Iterator over the elements of the dense tensor.
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
      */
      explicit inline ConstIterator( LeftIteratorType left, RightIteratorType right )
         : left_ ( left  )  // Iterator to the current left-hand side element
         , right_( right )  // Iterator to the current right-hand side element
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
         return ConstIterator( left_++, right_++ );
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
         return ConstIterator( left_--, right_-- );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReturnType operator*() const {
         return (*left_) * (*right_);
      }
      //*******************************************************************************************

      //**Load function****************************************************************************
      /*!\brief Access to the SIMD elements of the tensor.
      //
      // \return The resulting SIMD element.
      */
      inline auto load() const noexcept {
         return left_.load() * right_.load();
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
         return ConstIterator( it.left_ + inc, it.right_ + inc );
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
         return ConstIterator( it.left_ + inc, it.right_ + inc );
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
         return ConstIterator( it.left_ - dec, it.right_ - dec );
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      LeftIteratorType  left_;   //!< Iterator to the current left-hand side element.
      RightIteratorType right_;  //!< Iterator to the current right-hand side element.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled =
      ( MT1::simdEnabled && MT2::simdEnabled && HasSIMDMult_v<ET1,ET2> );

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = ( MT1::smpAssignable && MT2::smpAssignable );
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   static constexpr size_t SIMDSIZE = SIMDTrait<ElementType>::size;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DTensDTensSchurExpr class.
   //
   // \param lhs The left-hand side operand of the Schur product expression.
   // \param rhs The right-hand side operand of the Schur product expression.
   */
   explicit inline DTensDTensSchurExpr( const MT1& lhs, const MT2& rhs ) noexcept
      : lhs_( lhs )  // Left-hand side dense tensor of the Schur product expression
      , rhs_( rhs )  // Right-hand side dense tensor of the Schur product expression
   {
      BLAZE_INTERNAL_ASSERT( lhs.rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( lhs.columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( lhs.pages()   == rhs.pages()  , "Invalid number of pages" );
   }
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 3D-access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator()( size_t k, size_t i, size_t j ) const {
      BLAZE_INTERNAL_ASSERT( i < lhs_.rows()   , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < lhs_.columns(), "Invalid column access index" );
      BLAZE_INTERNAL_ASSERT( k < lhs_.pages()  , "Invalid page access index" );
      return lhs_(k,i,j) * rhs_(k,i,j);
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
      BLAZE_INTERNAL_ASSERT( k < lhs_.pages()  , "Invalid page access index" );
      BLAZE_INTERNAL_ASSERT( i % SIMDSIZE == 0UL, "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );
      return lhs_.load(k,i,j) * rhs_.load(k,i,j);
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of row \a i.
   //
   // \param i The row index.
   // \return Iterator to the first non-zero element of row \a i.
   */
   inline ConstIterator begin( size_t i, size_t k ) const {
      return ConstIterator( lhs_.begin(i, k), rhs_.begin(i, k) );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row \a i.
   //
   // \param i The row index.
   // \return Iterator just past the last non-zero element of row \a i.
   */
   inline ConstIterator end( size_t i, size_t k ) const {
      return ConstIterator( lhs_.end(i, k), rhs_.end(i, k) );
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

   //**********************************************************************************************
   /*!\brief Returns whether the expression can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the expression can alias, \a false otherwise.
   */
   template< typename T >
   inline bool canAlias( const T* alias ) const noexcept {
      return ( IsExpression_v<MT1> && ( RequiresEvaluation_v<MT1> ? lhs_.isAliased( alias ) : lhs_.canAlias( alias ) ) ) ||
             ( IsExpression_v<MT2> && ( RequiresEvaluation_v<MT2> ? rhs_.isAliased( alias ) : rhs_.canAlias( alias ) ) );
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
      return lhs_.canSMPAssign() || rhs_.canSMPAssign() ||
             ( rows() * columns() * pages() >= SMP_DMATDMATSCHUR_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  lhs_;  //!< Left-hand side dense tensor of the Schur product expression.
   RightOperand rhs_;  //!< Right-hand side dense tensor of the Schur product expression.
   //**********************************************************************************************

   //**Assignment to dense tensors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a non-commutative dense tensor-dense tensor Schur product to a
   //        dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side Schur product expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a non-commutative dense
   // tensor-dense tensor Schur product expression to a dense tensor. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case either of the two operands requires an intermediate evaluation.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT> && !IsCommutative_v<MT1,MT2> >
      assign( DenseTensor<MT2>& lhs, const DTensDTensSchurExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages" );

      if( !IsOperation_v<MT1> && isSame( *lhs, rhs.lhs_ ) ) {
         schurAssign( *lhs, rhs.rhs_ );
      }
      else {
         CT1 A( serial( rhs.lhs_ ) );
         CT2 B( serial( rhs.rhs_ ) );
         assign( *lhs, A % B );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to dense tensors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a commutative dense tensor-dense tensor Schur product to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side Schur product expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a commutative dense
   // tensor-dense tensor Schur product expression to a dense tensor. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case either of the two operands requires an intermediate evaluation.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT> && IsCommutative_v<MT1,MT2> >
      assign( DenseTensor<MT2>& lhs, const DTensDTensSchurExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages" );

      if( !IsOperation_v<MT1> && isSame( *lhs, rhs.lhs_ ) ) {
         schurAssign( *lhs, rhs.rhs_ );
      }
      else if( !IsOperation_v<MT2> && isSame( *lhs, rhs.rhs_ ) ) {
         schurAssign( *lhs, rhs.lhs_ );
      }
      else if( !RequiresEvaluation_v<MT2> ) {
         assign     ( *lhs, rhs.rhs_ );
         schurAssign( *lhs, rhs.lhs_ );
      }
      else {
         assign     ( *lhs, rhs.lhs_ );
         schurAssign( *lhs, rhs.rhs_ );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense tensors*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a dense tensor-dense tensor Schur product to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side Schur product expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a dense tensor-
   // dense tensor Schur product expression to a dense tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case either
   // of the operands requires an intermediate evaluation.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT> >
      addAssign( DenseTensor<MT2>& lhs, const DTensDTensSchurExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages" );

      const ResultType tmp( serial( rhs ) );
      addAssign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to sparse tensors******************************************************
   // No special implementation for the addition assignment to sparse tensors.
   //**********************************************************************************************

   //**Subtraction assignment to dense tensors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a dense tensor-dense tensor Schur product to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side Schur product expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a dense tensor-
   // dense tensor Schur product expression to a dense tensor. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case either
   // of the operands requires an intermediate evaluation.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT> >
      subAssign( DenseTensor<MT2>& lhs, const DTensDTensSchurExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages" );

      const ResultType tmp( serial( rhs ) );
      subAssign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Schur product assignment to dense tensors**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Schur product assignment of a non-commutative dense tensor-dense tensor Schur
   //        product to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side Schur product expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized Schur product assignment of a
   // non-commutative dense tensor-dense tensor Schur product expression to a dense tensor. Due
   // to the explicit application of the SFINAE principle, this function can only be selected by
   // the compiler in case either of the operands requires an intermediate evaluation.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT> && !IsCommutative_v<MT1,MT2> >
      schurAssign( DenseTensor<MT2>& lhs, const DTensDTensSchurExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages" );

      const ResultType tmp( serial( rhs ) );
      schurAssign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Schur product assignment to dense tensors**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Schur product assignment of a commutative dense tensor-dense tensor Schur product
   //        to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side Schur product expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized Schur product assignment of a
   // commutative dense tensor-dense tensor Schur product expression to a dense tensor. Due
   // to the explicit application of the SFINAE principle, this function can only be selected
   // by the compiler in case either of the operands requires an intermediate evaluation.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT> && IsCommutative_v<MT1,MT2> >
      schurAssign( DenseTensor<MT2>& lhs, const DTensDTensSchurExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages" );

      if( !RequiresEvaluation_v<MT2> ) {
         schurAssign( *lhs, rhs.rhs_ );
         schurAssign( *lhs, rhs.lhs_ );
      }
      else {
         schurAssign( *lhs, rhs.lhs_ );
         schurAssign( *lhs, rhs.rhs_ );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to dense tensors************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of non-commutative a dense tensor-dense tensor Schur product to a
   //        dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side Schur product expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a non-commutative
   // dense tensor-dense tensor Schur product expression to a dense tensor. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT> && !IsCommutative_v<MT1,MT2> >
      smpAssign( DenseTensor<MT2>& lhs, const DTensDTensSchurExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages" );

      if( !IsOperation_v<MT1> && isSame( *lhs, rhs.lhs_ ) ) {
         smpSchurAssign( *lhs, rhs.rhs_ );
      }
      else {
         CT1 A( rhs.lhs_ );
         CT2 B( rhs.rhs_ );
         smpAssign( *lhs, A % B );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to dense tensors************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of commutative a dense tensor-dense tensor Schur product to a
   //        dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side Schur product expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a commutative dense
   // tensor-dense tensor Schur product expression to a dense tensor. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT> && IsCommutative_v<MT1,MT2> >
      smpAssign( DenseTensor<MT2>& lhs, const DTensDTensSchurExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages" );

      if( !IsOperation_v<MT1> && isSame( *lhs, rhs.lhs_ ) ) {
         smpSchurAssign( *lhs, rhs.rhs_ );
      }
      else if( !IsOperation_v<MT2> && isSame( *lhs, rhs.rhs_ ) ) {
         smpSchurAssign( *lhs, rhs.lhs_ );
      }
      else if( !RequiresEvaluation_v<MT2> ) {
         smpAssign     ( *lhs, rhs.rhs_ );
         smpSchurAssign( *lhs, rhs.lhs_ );
      }
      else {
         smpAssign     ( *lhs, rhs.lhs_ );
         smpSchurAssign( *lhs, rhs.rhs_ );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to dense tensors***************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a dense tensor-dense tensor Schur product to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side Schur product expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a dense
   // tensor-dense tensor Schur product expression to a dense tensor. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseAssign_v<MT> >
      smpAddAssign( DenseTensor<MT2>& lhs, const DTensDTensSchurExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages" );

      const ResultType tmp( rhs );
      smpAddAssign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to dense tensors************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a dense tensor-dense tensor Schur product to a
   //        dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side Schur product expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a dense
   // tensor-dense tensor Schur product expression to a dense tensor. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT> >
      smpSubAssign( DenseTensor<MT2>& lhs, const DTensDTensSchurExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages" );

      const ResultType tmp( rhs );
      smpSubAssign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP Schur product assignment to dense tensors**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP Schur product assignment of a non-commutative dense tensor-dense tensor Schur
   //        product to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side Schur product expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized SMP Schur product assignment of a
   // non-commutative dense tensor-dense tensor Schur product expression to a dense tensor. Due
   // to the explicit application of the SFINAE principle, this function can only be selected by
   // the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT> && !IsCommutative_v<MT1,MT2> >
      smpSchurAssign( DenseTensor<MT2>& lhs, const DTensDTensSchurExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages" );

      const ResultType tmp( rhs );
      smpSchurAssign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP Schur product assignment to dense tensors**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP Schur product assignment of a commutative dense tensor-dense tensor Schur
   //        product to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side Schur product expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized SMP Schur product assignment of a
   // commutative dense tensor-dense tensor Schur product expression to a dense tensor. Due to
   // the explicit application of the SFINAE principle, this function can only be selected by
   // the compiler in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline EnableIf_t< UseSMPAssign_v<MT> && IsCommutative_v<MT1,MT2> >
      smpSchurAssign( DenseTensor<MT2>& lhs, const DTensDTensSchurExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (*lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == rhs.pages()  , "Invalid number of pages" );

      if( !RequiresEvaluation_v<MT2> ) {
         smpSchurAssign( *lhs, rhs.rhs_ );
         smpSchurAssign( *lhs, rhs.lhs_ );
      }
      else {
         smpSchurAssign( *lhs, rhs.lhs_ );
         smpSchurAssign( *lhs, rhs.rhs_ );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_FORM_VALID_TENSOR_SCHUREXPR( MT1, MT2 );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the Schur product between two dense tensors with identical
//        storage order (\f$ A=B \circ C \f$).
// \ingroup dense_tensor
//
// \param lhs The left-hand side dense tensor for the Schur product.
// \param rhs The right-hand side dense tensor for the Schur product.
// \return The Schur product of the two tensors.
//
// This function implements a performance optimized treatment of the Schur product of two
// dense tensors with identical storage order.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2  // Type of the right-hand side dense tensor
        , EnableIf_t< !( IsUniLower_v<MT1> && IsUniUpper_v<MT2> ) ||
                       ( IsUniUpper_v<MT1> && IsUniLower_v<MT2> ) >* = nullptr >
inline const DTensDTensSchurExpr<MT1,MT2>
   dtensdtensschur( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == (*rhs).pages()  , "Invalid number of pages" );

   return DTensDTensSchurExpr<MT1,MT2>( *lhs, *rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the Schur product between two unitriangular dense tensors
//        with identical storage order (\f$ A=B \circ C \f$).
// \ingroup dense_tensor
//
// \param lhs The left-hand side dense tensor for the Schur product.
// \param rhs The right-hand side dense tensor for the Schur product.
// \return The Schur product of the two tensors.
//
// This function implements a performance optimized treatment of the Schur product between two
// unitriangular dense tensors with identical storage order.
*/
// template< typename MT1  // Type of the left-hand side dense tensor
//         , typename MT2  // Type of the right-hand side dense tensor
//         , EnableIf_t< ( IsUniLower_v<MT1> && IsUniUpper_v<MT2> ) ||
//                       ( IsUniUpper_v<MT1> && IsUniLower_v<MT2> ) >* = nullptr >
// inline const IdentityTensor< MultTrait_t< ElementType_t<MT1>, ElementType_t<MT2> > >
//    dmatdmatschur( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    MAYBE_UNUSED( rhs );
//
//    BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
//    BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );
//    BLAZE_INTERNAL_ASSERT( (*lhs).pages()   == (*rhs).pages()  , "Invalid number of pages" );
//
//    return IdentityTensor< MultTrait_t< ElementType_t<MT1>, ElementType_t<MT2> > >( (*lhs).rows() );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Operator for the Schur product of two dense tensors with identical storage order
//        (\f$ A=B \circ C \f$).
// \ingroup dense_tensor
//
// \param lhs The left-hand side dense tensor for the Schur product.
// \param rhs The right-hand side dense tensor for the Schur product.
// \return The Schur product of the two tensors.
// \exception std::invalid_argument Tensor sizes do not match.
//
// This operator represents the Schur product of two dense tensors with identical storage order:

   \code
   blaze::DynamicTensor<double> A, B, C;
   // ... Resizing and initialization
   C = A % B;
   \endcode

// The operator returns an expression representing a dense tensor of the higher-order element
// type of the two involved tensor element types \a MT1::ElementType and \a MT2::ElementType.
// Both tensor types \a MT1 and \a MT2 as well as the two element types \a MT1::ElementType
// and \a MT2::ElementType have to be supported by the MultTrait class template.\n
// In case the current number of rows and columns of the two given  tensors don't match, a
// \a std::invalid_argument is thrown.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side dense tensor
inline decltype(auto)
   operator%( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   if( (*lhs).rows() != (*rhs).rows() || (*lhs).columns() != (*rhs).columns() || (*lhs).pages() != (*rhs).pages() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   return dtensdtensschur( *lhs, *rhs );
}
//*************************************************************************************************




//=================================================================================================
//
//  ISALIGNED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct IsAligned< DTensDTensSchurExpr<MT1,MT2> >
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
template< typename MT1, typename MT2 >
struct IsPadded< DTensDTensSchurExpr<MT1,MT2> >
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
template< typename MT1, typename MT2 >
struct IsSymmetric< DTensDTensSchurExpr<MT1,MT2> >
   : public BoolConstant< IsSymmetric_v<MT1> && IsSymmetric_v<MT2> >
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
template< typename MT1, typename MT2 >
struct IsHermitian< DTensDTensSchurExpr<MT1,MT2> >
   : public BoolConstant< IsHermitian_v<MT1> && IsHermitian_v<MT2> >
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
template< typename MT1, typename MT2 >
struct IsLower< DTensDTensSchurExpr<MT1,MT2> >
   : public BoolConstant< IsLower_v<MT1> || IsLower_v<MT2> >
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
template< typename MT1, typename MT2 >
struct IsUniLower< DTensDTensSchurExpr<MT1,MT2> >
   : public BoolConstant< IsUniLower_v<MT1> && IsUniLower_v<MT2> >
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
template< typename MT1, typename MT2 >
struct IsStrictlyLower< DTensDTensSchurExpr<MT1,MT2> >
   : public BoolConstant< IsStrictlyLower_v<MT1> || IsStrictlyLower_v<MT2> >
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
template< typename MT1, typename MT2 >
struct IsUpper< DTensDTensSchurExpr<MT1,MT2> >
   : public BoolConstant< IsUpper_v<MT1> || IsUpper_v<MT2> >
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
template< typename MT1, typename MT2 >
struct IsUniUpper< DTensDTensSchurExpr<MT1,MT2> >
   : public BoolConstant< IsUniUpper_v<MT1> && IsUniUpper_v<MT2> >
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
template< typename MT1, typename MT2 >
struct IsStrictlyUpper< DTensDTensSchurExpr<MT1,MT2> >
   : public BoolConstant< IsStrictlyUpper_v<MT1> || IsStrictlyUpper_v<MT2> >
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
