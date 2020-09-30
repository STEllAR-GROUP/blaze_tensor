//=================================================================================================
/*!
//  \file blaze_array/math/expressions/DArrScalarDivExpr.h
//  \brief Header file for the dense array/scalar division expression
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

#ifndef _BLAZE_ARRAY_MATH_EXPRESSIONS_DARRSCALARDIVEXPR_H_
#define _BLAZE_ARRAY_MATH_EXPRESSIONS_DARRSCALARDIVEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze_tensor/math/expressions/DTensScalarDivExpr.h>
#include <blaze_tensor/math/constraints/DenseArray.h>
#include <blaze_tensor/math/expressions/Forward.h>
#include <blaze_tensor/math/expressions/DenseArray.h>
#include <blaze_tensor/math/expressions/ArrScalarDivExpr.h>
#include <blaze_tensor/math/typetraits/IsTemporaryEx.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DARRSCALARDIVEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for divisions of a dense array by a scalar.
// \ingroup dense_array_expression
//
// The DArrScalarDivExpr class represents the compile time expression for divisions of dense
// matrices and by scalar values.
*/
template< typename MT  // Type of the left-hand side dense array
        , typename ST > // Type of the right-hand side scalar value
class DArrScalarDivExpr
   : public ArrScalarDivExpr< DenseArray< DArrScalarDivExpr<MT,ST> > >
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   using RT = ResultType_t<MT>;     //!< Result type of the dense array expression.
   using RN = ReturnType_t<MT>;     //!< Return type of the dense array expression.
   using ET = ElementType_t<MT>;    //!< Element type of the dense array expression.
   using CT = CompositeType_t<MT>;  //!< Composite type of the dense array expression.
   //**********************************************************************************************

   //**Return type evaluation**********************************************************************
   //! Compilation switch for the selection of the subscript operator return type.
   /*! The \a returnExpr compile time constant expression is a compilation switch for the
       selection of the \a ReturnType. If the array operand returns a temporary vector
       or array, \a returnExpr will be set to \a false and the subscript operator will
       return it's result by value. Otherwise \a returnExpr will be set to \a true and
       the subscript operator may return it's result as an expression. */
   static constexpr bool returnExpr = !IsTemporaryEx_v<RN>;

   //! Expression return type for the subscript operator.
   using ExprReturnType = decltype( std::declval<RN>() / std::declval<ST>() );
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the division expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the serial evaluation strategy of the division expression. In case the given dense
       array expression of type \a MT is a computation expression and requires an intermediate
       evaluation, \a useAssign will be set to 1 and the division expression will be evaluated
       via the \a assign function family. Otherwise \a useAssign will be set to 0 and the
       expression will be evaluated via the subscript operator. */
   static constexpr bool useAssign = IsComputation_v<MT> && RequiresEvaluation_v<MT>;

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
       In case either the target array or the dense array operand is not SMP assignable and the
       array operand is a computation expression that requires an intermediate evaluation, the
       variable is set to 1 and the expression specific evaluation strategy is selected. Otherwise
       the variable is set to 0 and the default strategy is chosen. */
   template< typename MT2 >
   static constexpr bool UseSMPAssign_v =
      ( ( !MT2::smpAssignable || !MT::smpAssignable ) && useAssign );
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This          = DArrScalarDivExpr<MT,ST>;  //!< Type of this DArrScalarDivExpr instance.
   using ResultType    = DivTrait_t<RT,ST>;            //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;   //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;  //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<ResultType>;    //!< Resulting element type.

   //! Return type for expression template evaluations.
   using ReturnType = const If_t< returnExpr, ExprReturnType, ElementType >;

   //! Data type for composite expression templates.
   using CompositeType = If_t< useAssign, const ResultType, const DArrScalarDivExpr& >;

   //! Composite type of the left-hand side dense array expression.
   using LeftOperand = If_t< IsExpression_v<MT>, const MT, const MT& >;

   //! Composite type of the right-hand side scalar value.
   using RightOperand = ST;
   //**********************************************************************************************

   //**ConstIterator class definition**************************************************************
   /*!\brief Iterator over the elements of the dense array.
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

      //! ConstIterator type of the dense array expression.
      using IteratorType = ConstIterator_t<MT>;
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the ConstIterator class.
      //
      // \param iterator Iterator to the initial element.
      // \param scalar Scalar of the multiplication expression.
      */
      explicit inline ConstIterator( IteratorType iterator, RightOperand scalar )
         : iterator_( iterator )  // Iterator to the current element
         , scalar_  ( scalar   )  // Scalar of the multiplication expression
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline ConstIterator& operator+=( size_t inc ) {
         iterator_ += inc;
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
         iterator_ -= dec;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline ConstIterator& operator++() {
         ++iterator_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator++( int ) {
         return ConstIterator( iterator_++ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline ConstIterator& operator--() {
         --iterator_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator--( int ) {
         return ConstIterator( iterator_-- );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReturnType operator*() const {
         return *iterator_ / scalar_;
      }
      //*******************************************************************************************

      //**Load function****************************************************************************
      /*!\brief Access to the SIMD elements of the array.
      //
      // \return The resulting SIMD element.
      */
      inline auto load() const noexcept {
         return iterator_.load() / set( scalar_ );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const ConstIterator& rhs ) const {
         return iterator_ == rhs.iterator_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const ConstIterator& rhs ) const {
         return iterator_ != rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const ConstIterator& rhs ) const {
         return iterator_ < rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const ConstIterator& rhs ) const {
         return iterator_ > rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const ConstIterator& rhs ) const {
         return iterator_ <= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const ConstIterator& rhs ) const {
         return iterator_ >= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const ConstIterator& rhs ) const {
         return iterator_ - rhs.iterator_;
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
         return ConstIterator( it.iterator_ + inc, it.scalar_ );
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
         return ConstIterator( it.iterator_ + inc, it.scalar_ );
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
         return ConstIterator( it.iterator_ - dec, it.scalar_ );
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType iterator_;  //!< Iterator to the current element.
      RightOperand scalar_;    //!< Scalar of the multiplication expression.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled =
      ( MT::simdEnabled && IsNumeric_v<ET> &&
        ( HasSIMDDiv_v<ET,ST> || HasSIMDDiv_v<UnderlyingElement_t<ET>,ST> ) );

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = MT::smpAssignable;
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   static constexpr size_t SIMDSIZE = SIMDTrait<ElementType>::size;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DArrScalarDivExpr class.
   //
   // \param array The left-hand side dense array of the division expression.
   // \param scalar The right-hand side scalar of the division expression.
   */
   explicit inline DArrScalarDivExpr( const MT& array, ST scalar ) noexcept
      : array_( array )  // Left-hand side dense array of the division expression
      , scalar_( scalar )  // Right-hand side scalar of the division expression
   {}
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 2D-access to the array elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   template< typename... Dims >
   inline ReturnType operator()( Dims... dims ) const {
      return array_( dims... ) / scalar_;
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the array elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid array access index.
   */
   template< typename... Dims >
   inline ReturnType at( Dims... dims ) const {
      return ( *this )( dims... );
   }
   //**********************************************************************************************

   //**Load function*******************************************************************************
   /*!\brief Access to the SIMD elements of the array.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return Reference to the accessed values.
   */
   template< typename... Dims >
   BLAZE_ALWAYS_INLINE auto load( Dims... dims ) const noexcept {
      return array_.load( dims... ) / set( scalar_ );
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of row \a i.
   //
   // \param i The row index.
   // \return Iterator to the first non-zero element of row \a i.
   */
   template< typename... Dims >
   inline ConstIterator begin( size_t i, Dims... dims ) const {
      return ConstIterator( array_.begin( i, dims... ), scalar_ );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row \a i.
   //
   // \param i The row index.
   // \return Iterator just past the last non-zero element of row \a i.
   */
   template< typename... Dims >
   inline ConstIterator end( size_t i, Dims... dims ) const {
      return ConstIterator( array_.end(i, dims...), scalar_ );
   }
   //**********************************************************************************************

   //**Num_dimensions function*******************************************************************************
   /*!\brief Returns the current number of dimensions of the array.
   //
   // \return The number of rows of the array.
   */
   static constexpr size_t num_dimensions =
      RemoveCV_t< RemoveReference_t< LeftOperand > >::num_dimensions;

   //**********************************************************************************************

   //**Dimensions function****************************************************************************
   /*!\brief Returns the current dimensions of the array.
   //
   // \return The dimensions of the array.
   */
   inline decltype(auto) dimensions() const noexcept {
      return array_.dimensions();
   }
   //**********************************************************************************************

   //**Dimension function****************************************************************************
   /*!\brief Returns the current number of columns of the array.
   //
   // \return The number of columns of the array.
   */
   template< size_t Dim >
   inline size_t dimension() const noexcept {
      return array_.template dimension<Dim>();
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side dense array operand.
   //
   // \return The left-hand side dense array operand.
   */
   inline LeftOperand leftOperand() const noexcept {
      return array_;
   }
   //**********************************************************************************************

   //**Right operand access************************************************************************
   /*!\brief Returns the right-hand side scalar operand.
   //
   // \return The right-hand side scalar operand.
   */
   inline RightOperand rightOperand() const noexcept {
      return scalar_;
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
      return IsExpression_v<MT> && array_.canAlias( alias );
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
      return array_.isAliased( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const noexcept {
      return array_.isAligned();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return array_.canSMPAssign() ||
             ( dimension<0>() * dimension<1>() >= SMP_DMATSCALARMULT_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  array_;  //!< Left-hand side dense array of the division expression.
   RightOperand scalar_;  //!< Right-hand side scalar of the division expression.
   //**********************************************************************************************

   //**Assignment to dense matrices****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense array-scalar division to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side division expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense array-scalar
   // division expression to a dense array. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the array
   // operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseAssign_v<MT2> >
      assign( DenseArray<MT2>& lhs, const DArrScalarDivExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( ( *lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      assign( *lhs, rhs.array_ );
      assign( *lhs, (*lhs) / rhs.scalar_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense matrices*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a dense array-scalar division to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side division expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a dense array-
   // scalar division expression to a dense array. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the array
   // operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseAssign_v<MT2> >
      addAssign( DenseArray<MT2>& lhs, const DArrScalarDivExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( ( *lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const ResultType tmp( serial( rhs ) );
      addAssign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to sparse matrices******************************************************
   // No special implementation for the addition assignment to sparse matrices.
   //**********************************************************************************************

   //**Subtraction assignment to dense matrices****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a dense array-scalar division to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side division expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a dense array-
   // scalar division expression to a dense array. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the array operand is
   // a computation expression and requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseAssign_v<MT2> >
      subAssign( DenseArray<MT2>& lhs, const DArrScalarDivExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( ( *lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const ResultType tmp( serial( rhs ) );
      subAssign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to sparse matrices***************************************************
   // No special implementation for the subtraction assignment to sparse matrices.
   //**********************************************************************************************

   //**Schur product assignment to dense matrices**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Schur product assignment of a dense array-scalar division to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side division expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized Schur product assignment of a dense
   // array-scalar division expression to a dense array. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case the
   // array operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseAssign_v<MT2> >
      schurAssign( DenseArray<MT2>& lhs, const DArrScalarDivExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( ( *lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const ResultType tmp( serial( rhs ) );
      schurAssign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to dense matrices************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense array-scalar division to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side division expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense array-scalar
   // division expression to a dense array. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpAssign( DenseArray<MT2>& lhs, const DArrScalarDivExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( ( *lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      smpAssign( *lhs, rhs.array_ );
      smpAssign( *lhs, (*lhs) / rhs.scalar_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to dense matrices***************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a dense array-scalar division to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side division expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a dense
   // array-scalar division expression to a dense array. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case
   // the expression specific parallel evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpAddAssign( DenseArray<MT2>& lhs, const DArrScalarDivExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( ( *lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const ResultType tmp( rhs );
      smpAddAssign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to dense matrices************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a dense array-scalar division to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side division expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a dense
   // array-scalar division expression to a dense array. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpSubAssign( DenseArray<MT2>& lhs, const DArrScalarDivExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( ( *lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const ResultType tmp( rhs );
      smpSubAssign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP Schur product assignment to dense matrices**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP Schur product assignment of a dense array-scalar division to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side division expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized SMP Schur product assignment of a dense
   // array-scalar division expression to a dense array. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpSchurAssign( DenseArray<MT2>& lhs, const DArrScalarDivExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( ( *lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const ResultType tmp( rhs );
      smpSchurAssign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( ST );
   BLAZE_CONSTRAINT_MUST_NOT_BE_FLOATING_POINT_TYPE( ST );
   BLAZE_CONSTRAINT_MUST_NOT_BE_FLOATING_POINT_TYPE( ElementType );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ST, RightOperand );
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
/*!\brief Auxiliary helper struct for the dense array/scalar division operator.
// \ingroup math_traits
*/
template< typename MT  // Type of the left-hand side dense array
        , typename ST > // Type of the right-hand side scalar
struct DArrScalarDivExprHelper
{
 private:
   //**********************************************************************************************
   using ScalarType = If_t< IsFloatingPoint_v< UnderlyingBuiltin_t<MT> > ||
                            IsFloatingPoint_v< UnderlyingBuiltin_t<ST> >
                          , If_t< IsComplex_v< UnderlyingNumeric_t<MT> > && IsBuiltin_v<ST>
                                , DivTrait_t< UnderlyingBuiltin_t<MT>, ST >
                                , DivTrait_t< UnderlyingNumeric_t<MT>, ST > >
                          , ST >;
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   using Type = If_t< IsInvertible_v<ScalarType>
                    , DArrScalarMultExpr<MT,ScalarType>
                    , DArrScalarDivExpr<MT,ScalarType> >;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division operator for the division of a dense array by a scalar value (\f$ A=B/s \f$).
// \ingroup dense_array
//
// \param mat The left-hand side dense array for the division.
// \param scalar The right-hand side scalar value for the division.
// \return The scaled result array.
//
// This operator represents the division of a dense array by a scalar value:

   \code
   blaze::DynamicMatrix<double> A, B;
   // ... Resizing and initialization
   B = A / 0.24;
   \endcode

// The operator returns an expression representing a dense array of the higher-order element
// type of the involved data types \a MT::ElementType and \a ST. Note that this operator only
// works for scalar values of built-in data type.
//
// \note A division by zero is only checked by an user assert.
*/
template< typename MT  // Type of the left-hand side dense array
        , typename ST  // Type of the right-hand side scalar
        , EnableIf_t< IsNumeric_v<ST> >* = nullptr >
inline decltype(auto) operator/( const DenseArray<MT>& mat, ST scalar )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_USER_ASSERT( scalar != ST(0), "Division by zero detected" );

   using ReturnType = typename DArrScalarDivExprHelper<MT,ST>::Type;
   using ScalarType = RightOperand_t<ReturnType>;

   if( IsMultExpr_v<ReturnType> ) {
      return ReturnType( *mat, ScalarType(1)/ScalarType(scalar) );
   }
   else {
      return ReturnType( *mat, scalar );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense array-scalar division
//        expression and a scalar value (\f$ A=(B/s1)*s2 \f$).
// \ingroup dense_array
//
// \param mat The left-hand side dense array-scalar division.
// \param scalar The right-hand side scalar value for the multiplication.
// \return The scaled result array.
//
// This operator implements a performance optimized treatment of the multiplication of a
// dense array-scalar division expression and a scalar value.
*/
template< typename MT   // Type of the dense array of the left-hand side expression
        , typename ST1  // Type of the scalar of the left-hand side expression
        , typename ST2  // Type of the right-hand side scalar
        , EnableIf_t< IsNumeric_v<ST2> && ( IsInvertible_v<ST1> || IsInvertible_v<ST2> ) >* = nullptr >
inline decltype(auto) operator*( const DArrScalarDivExpr<MT,ST1>& mat, ST2 scalar )
{
   BLAZE_FUNCTION_TRACE;

   return mat.leftOperand() * ( scalar / mat.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a scalar value and a dense array-
//        scalar division expression (\f$ A=s2*(B/s1) \f$).
// \ingroup dense_array
//
// \param scalar The left-hand side scalar value for the multiplication.
// \param mat The right-hand side dense array-scalar division.
// \return The scaled result array.
//
// This operator implements a performance optimized treatment of the multiplication of a
// scalar value and a dense array-scalar division expression.
*/
template< typename ST1  // Type of the left-hand side scalar
        , typename MT   // Type of the dense array of the right-hand side expression
        , typename ST2  // Type of the scalar of the right-hand side expression
        , EnableIf_t< IsNumeric_v<ST1> && ( IsInvertible_v<ST1> || IsInvertible_v<ST2> ) >* = nullptr >
inline decltype(auto) operator*( ST1 scalar, const DArrScalarDivExpr<MT,ST2>& mat )
{
   BLAZE_FUNCTION_TRACE;

   return mat.leftOperand() * ( scalar / mat.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division operator for the division of a dense array-scalar division expression
//        and a scalar value (\f$ A=(B/s1)/s2 \f$).
// \ingroup dense_array
//
// \param mat The left-hand side dense array-scalar division.
// \param scalar The right-hand side scalar value for the division.
// \return The scaled result array.
//
// This operator implements a performance optimized treatment of the division of a dense
// array-scalar division expression and a scalar value.
*/
template< typename MT   // Type of the dense array of the left-hand side expression
        , typename ST1  // Type of the scalar of the left-hand side expression
        , typename ST2  // Type of the right-hand side scalar
        , EnableIf_t< IsNumeric_v<ST2> >* = nullptr >
inline decltype(auto) operator/( const DArrScalarDivExpr<MT,ST1>& mat, ST2 scalar )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_USER_ASSERT( scalar != ST2(0), "Division by zero detected" );

   using MultType   = MultTrait_t<ST1,ST2>;
   using ReturnType = typename DArrScalarDivExprHelper<MT,MultType>::Type;
   using ScalarType = RightOperand_t<ReturnType>;

   if( IsMultExpr_v<ReturnType> ) {
      return ReturnType( mat.leftOperand(), ScalarType(1)/( mat.rightOperand() * scalar ) );
   }
   else {
      return ReturnType( mat.leftOperand(), mat.rightOperand() * scalar );
   }
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
template< typename MT, typename ST >
struct IsAligned< DArrScalarDivExpr<MT,ST> >
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
template< typename MT, typename ST >
struct IsPadded< DArrScalarDivExpr<MT,ST> >
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
template< typename MT, typename ST >
struct IsSymmetric< DArrScalarDivExpr<MT,ST> >
   : public FalseType
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
template< typename MT, typename ST >
struct IsHermitian< DArrScalarDivExpr<MT,ST> >
   : public FalseType
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
template< typename MT, typename ST >
struct IsLower< DArrScalarDivExpr<MT,ST> >
   : public FalseType
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
template< typename MT, typename ST >
struct IsStrictlyLower< DArrScalarDivExpr<MT,ST> >
   : public FalseType
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
template< typename MT, typename ST >
struct IsUpper< DArrScalarDivExpr<MT,ST> >
   : public FalseType
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
template< typename MT, typename ST >
struct IsStrictlyUpper< DArrScalarDivExpr<MT,ST> >
   : public FalseType
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
