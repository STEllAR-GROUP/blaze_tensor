//=================================================================================================
/*!
//  \file blaze_array/math/expressions/DArrScalarMultExpr.h
//  \brief Header file for the dense array/scalar multiplication expression
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

#ifndef _BLAZE_ARRAY_MATH_EXPRESSIONS_DARRSCALARMULTEXPR_H_
#define _BLAZE_ARRAY_MATH_EXPRESSIONS_DARRSCALARMULTEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze_tensor/math/expressions/DTensScalarMultExpr.h>
#include <blaze_tensor/math/constraints/DenseArray.h>
#include <blaze_tensor/math/expressions/DenseArray.h>
#include <blaze_tensor/math/expressions/Forward.h>
#include <blaze_tensor/math/expressions/ArrScalarMultExpr.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DARRSCALARMULTEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for dense array-scalar multiplications.
// \ingroup dense_array_expression
//
// The DArrScalarMultExpr class represents the compile time expression for multiplications between
// a dense array and a scalar value.
*/
template< typename MT  // Type of the left-hand side dense array
        , typename ST > // Type of the right-hand side scalar value
class DArrScalarMultExpr
   : public ArrScalarMultExpr< DenseArray< DArrScalarMultExpr<MT,ST> > >
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
   static constexpr bool returnExpr = !IsTemporary_v<RN>;

   //! Expression return type for the subscript operator.
   using ExprReturnType = decltype( std::declval<RN>() * std::declval<ST>() );
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the multiplication expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the serial evaluation strategy of the multiplication expression. In case the given dense
       array expression of type \a MT is a computation expression and requires an intermediate
       evaluation, \a useAssign will be set to 1 and the multiplication expression will be
       evaluated via the \a assign function family. Otherwise \a useAssign will be set to 0
       and the expression will be evaluated via the subscript operator. */
   static constexpr bool useAssign = ( IsComputation_v<MT> && RequiresEvaluation_v<MT> );

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
   using This          = DArrScalarMultExpr<MT,ST>;  //!< Type of this DArrScalarMultExpr instance.
   using ResultType    = MultTrait_t<RT,ST>;            //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;    //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;   //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<ResultType>;     //!< Resulting element type.

   //! Return type for expression template evaluations.
   using ReturnType = const If_t< returnExpr, ExprReturnType, ElementType >;

   //! Data type for composite expression templates.
   using CompositeType = If_t< useAssign, const ResultType, const DArrScalarMultExpr& >;

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
         return *iterator_ * scalar_;
      }
      //*******************************************************************************************

      //**Load function****************************************************************************
      /*!\brief Access to the SIMD elements of the array.
      //
      // \return The resulting SIMD element.
      */
      inline auto load() const noexcept {
         return iterator_.load() * set( scalar_ );
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
        ( HasSIMDMult_v<ET,ST> || HasSIMDMult_v<UnderlyingElement_t<ET>,ST> ) );

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = MT::smpAssignable;
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   static constexpr size_t SIMDSIZE = SIMDTrait<ElementType>::size;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DArrScalarMultExpr class.
   //
   // \param array The left-hand side dense array of the multiplication expression.
   // \param scalar The right-hand side scalar of the multiplication expression.
   */
   explicit inline DArrScalarMultExpr( const MT& array, ST scalar ) noexcept
      : array_( array )  // Left-hand side dense array of the multiplication expression
      , scalar_( scalar )  // Right-hand side scalar of the multiplication expression
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
      return array_( dims... ) * scalar_;
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
      return array_.load( dims... ) * set( scalar_ );
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
   inline static constexpr size_t num_dimensions() noexcept {
      return RemoveCV_t<RemoveReference_t<LeftOperand>>::num_dimensions();
   }
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
   LeftOperand  array_;  //!< Left-hand side dense array of the multiplication expression.
   RightOperand scalar_;  //!< Right-hand side scalar of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense matrices****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense array-scalar multiplication to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense array-scalar
   // multiplication expression to a dense array. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the array
   // operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseAssign_v<MT2> >
      assign( DenseArray<MT2>& lhs, const DArrScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( ( ~lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      assign( ~lhs, rhs.array_ );
      assign( ~lhs, (~lhs) * rhs.scalar_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense matrices*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a dense array-scalar multiplication to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a dense array-
   // scalar multiplication expression to a dense array. Due to the explicit application of
   // the SFINAE principle, this function can only be selected by the compiler in case the
   // array operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseAssign_v<MT2> >
      addAssign( DenseArray<MT2>& lhs, const DArrScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( ( ~lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const ResultType tmp( serial( rhs ) );
      addAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to dense matrices****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a dense array-scalar multiplication to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a dense array-
   // scalar multiplication expression to a dense array. Due to the explicit application of the
   // SFINAE principle, this function can only be selected by the compiler in case the array
   // operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseAssign_v<MT2> >
      subAssign( DenseArray<MT2>& lhs, const DArrScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( ( ~lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const ResultType tmp( serial( rhs ) );
      subAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Schur product assignment to dense matrices**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Schur product assignment of a dense array-scalar multiplication to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side multiplication expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized Schur product assignment of a dense
   // array-scalar multiplication expression to a dense array. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // array operand is a computation expression and requires an intermediate evaluation.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseAssign_v<MT2> >
      schurAssign( DenseArray<MT2>& lhs, const DArrScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( ( ~lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const ResultType tmp( serial( rhs ) );
      schurAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to dense matrices************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense array-scalar multiplication to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense array-scalar
   // multiplication expression to a dense array. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpAssign( DenseArray<MT2>& lhs, const DArrScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( ( ~lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      smpAssign( ~lhs, rhs.array_ );
      smpAssign( ~lhs, (~lhs) * rhs.scalar_ );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to dense matrices***************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a dense array-scalar multiplication to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a dense
   // array-scalar multiplication expression to a dense array. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpAddAssign( DenseArray<MT2>& lhs, const DArrScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( ( ~lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const ResultType tmp( rhs );
      smpAddAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to dense matrices************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a dense array-scalar multiplication to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a dense
   // array-scalar multiplication expression to a dense array. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpSubAssign( DenseArray<MT2>& lhs, const DArrScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( ( ~lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const ResultType tmp( rhs );
      smpSubAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP Schur product assignment to dense matrices**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP Schur product assignment of a dense array-scalar multiplication to a dense array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side dense array.
   // \param rhs The right-hand side multiplication expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized SMP Schur product assignment of a dense
   // array-scalar multiplication expression to a dense array. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific evaluation strategy is selected.
   */
   template< typename MT2 > // Type of the target dense array
   friend inline EnableIf_t< UseSMPAssign_v<MT2> >
      smpSchurAssign( DenseArray<MT2>& lhs, const DArrScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( ( ~lhs ).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const ResultType tmp( rhs );
      smpSchurAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( ST );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ST, RightOperand );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL UNARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Unary minus operator for the negation of a dense array (\f$ A = -B \f$).
// \ingroup dense_array
//
// \param dm The dense array to be negated.
// \return The negation of the array.
//
// This operator represents the negation of a dense array:

   \code
   blaze::DynamicArray<double> A, B;
   // ... Resizing and initialization
   B = -A;
   \endcode

// The operator returns an expression representing the negation of the given dense array.
*/
template< typename MT > // Type of the target dense array
inline decltype(auto) operator-( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ScalarType = UnderlyingBuiltin_t<MT>;
   using ReturnType = const DArrScalarMultExpr<MT,ScalarType>;
   return ReturnType( ~dm, ScalarType(-1) );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Multiplication operator for the multiplication of a dense array and a scalar value
//        (\f$ A=B*s \f$).
// \ingroup dense_array
//
// \param mat The left-hand side dense array for the multiplication.
// \param scalar The right-hand side scalar value for the multiplication.
// \return The scaled result array.
//
// This operator represents the multiplication between a dense array and a scalar value:

   \code
   blaze::DynamicArray<double> A, B;
   // ... Resizing and initialization
   B = A * 1.25;
   \endcode

// The operator returns an expression representing a dense array of the higher-order element
// type of the involved data types \a MT::ElementType and \a ST. Note that this operator only
// works for scalar values of built-in data type.
*/
template< typename MT  // Type of the left-hand side dense array
        , typename ST  // Type of the right-hand side scalar
        , EnableIf_t< IsNumeric_v<ST> >* = nullptr >
inline decltype(auto) operator*( const DenseArray<MT>& mat, ST scalar )
{
   BLAZE_FUNCTION_TRACE;

   using ScalarType = MultTrait_t< UnderlyingBuiltin_t<MT>, ST >;
   using ReturnType = const DArrScalarMultExpr<MT,ScalarType>;
   return ReturnType( ~mat, scalar );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication operator for the multiplication of a scalar value and a dense array
//        (\f$ A=s*B \f$).
// \ingroup dense_array
//
// \param scalar The left-hand side scalar value for the multiplication.
// \param mat The right-hand side dense array for the multiplication.
// \return The scaled result array.
//
// This operator represents the multiplication between a a scalar value and dense array:

   \code
   blaze::DynamicArray<double> A, B;
   // ... Resizing and initialization
   B = 1.25 * A;
   \endcode

// The operator returns an expression representing a dense array of the higher-order element
// type of the involved data types \a ST and \a MT::ElementType. Note that this operator only
// works for scalar values of built-in data type.
*/
template< typename ST  // Type of the left-hand side scalar
        , typename MT  // Type of the right-hand side dense array
        , EnableIf_t< IsNumeric_v<ST> >* = nullptr >
inline decltype(auto) operator*( ST scalar, const DenseArray<MT>& mat )
{
   BLAZE_FUNCTION_TRACE;

   using ScalarType = MultTrait_t< ST, UnderlyingBuiltin_t<MT> >;
   using ReturnType = const DArrScalarMultExpr<MT,ScalarType>;
   return ReturnType( ~mat, scalar );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING UNARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unary minus operator for the negation of a dense array-scalar multiplication
//        (\f$ A = -(B*s) \f$).
// \ingroup dense_array
//
// \param dm The dense array-scalar multiplication to be negated.
// \return The negation of the dense array-scalar multiplication.
//
// This operator implements a performance optimized treatment of the negation of a dense array-
// scalar multiplication expression.
*/
template< typename MT  // Type of the dense array
        , typename ST > // Type of the scalar
inline decltype(auto) operator-( const DArrScalarMultExpr<MT,ST>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DArrScalarMultExpr<MT,ST>;
   return ReturnType( dm.leftOperand(), -dm.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense array-scalar multiplication
//        expression and a scalar value (\f$ A=(B*s1)*s2 \f$).
// \ingroup dense_array
//
// \param mat The left-hand side dense array-scalar multiplication.
// \param scalar The right-hand side scalar value for the multiplication.
// \return The scaled result array.
//
// This operator implements a performance optimized treatment of the multiplication of a
// dense array-scalar multiplication expression and a scalar value.
*/
template< typename MT   // Type of the dense array of the left-hand side expression
        , typename ST1  // Type of the scalar of the left-hand side expression
        , typename ST2  // Type of the right-hand side scalar
        , EnableIf_t< IsNumeric_v<ST2> >* = nullptr >
inline decltype(auto) operator*( const DArrScalarMultExpr<MT,ST1>& mat, ST2 scalar )
{
   BLAZE_FUNCTION_TRACE;

   return mat.leftOperand() * ( mat.rightOperand() * scalar );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a scalar value and a dense array-
//        scalar multiplication expression (\f$ A=s2*(B*s1) \f$).
// \ingroup dense_array
//
// \param scalar The left-hand side scalar value for the multiplication.
// \param mat The right-hand side dense array-scalar multiplication.
// \return The scaled result array.
//
// This operator implements a performance optimized treatment of the multiplication of a
// scalar value and a dense array-scalar multiplication expression.
*/
template< typename ST1  // Type of the left-hand side scalar
        , typename MT   // Type of the dense array of the right-hand side expression
        , typename ST2  // Type of the scalar of the right-hand side expression
        , EnableIf_t< IsNumeric_v<ST1> >* = nullptr >
inline decltype(auto) operator*( ST1 scalar, const DArrScalarMultExpr<MT,ST2>& mat )
{
   BLAZE_FUNCTION_TRACE;

   return mat.leftOperand() * ( scalar * mat.rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division operator for the division of a dense array-scalar multiplication
//        expression by a scalar value (\f$ A=(B*s1)/s2 \f$).
// \ingroup dense_array
//
// \param mat The left-hand side dense array-scalar multiplication.
// \param scalar The right-hand side scalar value for the division.
// \return The scaled result array.
//
// This operator implements a performance optimized treatment of the division of a
// dense array-scalar multiplication expression by a scalar value.
*/
template< typename MT   // Type of the dense array of the left-hand side expression
        , typename ST1  // Type of the scalar of the left-hand side expression
        , typename ST2  // Type of the right-hand side scalar
        , EnableIf_t< IsNumeric_v<ST2> && ( IsInvertible_v<ST1> || IsInvertible_v<ST2> ) >* = nullptr >
inline decltype(auto) operator/( const DArrScalarMultExpr<MT,ST1>& mat, ST2 scalar )
{
   BLAZE_FUNCTION_TRACE;

   return mat.leftOperand() * ( mat.rightOperand() / scalar );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense array-scalar
//        multiplication expression and a dense vector (\f$ \vec{a}=(B*s1)*\vec{c} \f$).
// \ingroup dense_vector
//
// \param mat The left-hand side dense array-scalar multiplication.
// \param vec The right-hand side dense vector.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// dense array-scalar multiplication and a dense vector. It restructures the expression
// \f$ \vec{a}=(B*s1)*\vec{c} \f$ to the expression \f$ \vec{a}=(B*\vec{c})*s1 \f$.
*/
// template< typename MT    // Type of the dense array of the left-hand side expression
//         , typename ST    // Type of the scalar of the left-hand side expression
//         , typename VT >  // Type of the right-hand side dense vector
// inline decltype(auto)
//    operator*( const DArrScalarMultExpr<MT,ST>& mat, const DenseVector<VT,false>& vec )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return ( mat.leftOperand() * (~vec) ) * mat.rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense vector and a dense
//        array-scalar multiplication expression (\f$ \vec{a}^T=\vec{c}^T*(B*s1) \f$).
// \ingroup dense_vector
//
// \param vec The left-hand side dense vector.
// \param mat The right-hand side dense array-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication of a
// dense vector and a dense array-scalar multiplication. It restructures the expression
// \f$ \vec{a}=\vec{c}^T*(B*s1) \f$ to the expression \f$ \vec{a}^T=(\vec{c}^T*B)*s1 \f$.
*/
// template< typename VT  // Type of the left-hand side dense vector
//         , typename MT  // Type of the dense array of the right-hand side expression
//         , typename ST > // Type of the scalar of the right-hand side expression
// inline decltype(auto)
//    operator*( const DenseVector<VT,true>& vec, const DArrScalarMultExpr<MT,ST>& mat )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return ( (~vec) * mat.leftOperand() ) * mat.rightOperand();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense array-scalar
//        multiplication expression and a dense vector-scalar multiplication expression
//        (\f$ \vec{a}=(B*s1)*(\vec{c}*s2) \f$).
// \ingroup dense_vector
//
// \param mat The left-hand side dense array-scalar multiplication.
// \param vec The right-hand side dense vector-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication
// of a dense array-scalar multiplication and a dense vector-scalar multiplication. It
// restructures the expression \f$ \vec{a}=(B*s1)*(\vec{c}*s2) \f$ to the expression
// \f$ \vec{a}=(B*\vec{c})*(s1*s2) \f$.
*/
// template< typename MT     // Type of the dense array of the left-hand side expression
//         , typename ST1    // Type of the scalar of the left-hand side expression
//         , typename VT     // Type of the dense vector of the right-hand side expression
//         , typename ST2 >  // Type of the scalar of the right-hand side expression
// inline decltype(auto)
//    operator*( const DArrScalarMultExpr<MT,ST1>& mat, const DVecScalarMultExpr<VT,ST2,false>& vec )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return ( mat.leftOperand() * vec.leftOperand() ) * ( mat.rightOperand() * vec.rightOperand() );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense vector-scalar
//        multiplication expression and a dense array-scalar multiplication expression
//        (\f$ \vec{a}^T=\vec{b}^T*(C*s1) \f$).
// \ingroup dense_vector
//
// \param vec The left-hand side dense vector-scalar multiplication.
// \param mat The right-hand side dense array-scalar multiplication.
// \return The scaled result vector.
//
// This operator implements the performance optimized treatment of the multiplication
// of a dense vector-scalar multiplication and a dense array-scalar multiplication. It
// restructures the expression \f$ \vec{a}=(\vec{b}^T*s1)*(C*s2) \f$ to the expression
// \f$ \vec{a}^T=(\vec{b}^T*C)*(s1*s2) \f$.
*/
// template< typename VT   // Type of the dense vector of the left-hand side expression
//         , typename ST1  // Type of the scalar of the left-hand side expression
//         , typename MT   // Type of the dense array of the right-hand side expression
//         , typename ST2 > // Type of the scalar of the right-hand side expression
// inline decltype(auto)
//    operator*( const DVecScalarMultExpr<VT,ST1,true>& vec, const DArrScalarMultExpr<MT,ST2>& mat )
// {
//    BLAZE_FUNCTION_TRACE;
//
//    return ( vec.leftOperand() * mat.leftOperand() ) * ( vec.rightOperand() * mat.rightOperand() );
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense array-scalar multiplication
//        expression and a dense array (\f$ A=(B*s1)*C \f$).
// \ingroup dense_array
//
// \param lhs The left-hand side dense array-scalar multiplication.
// \param rhs The right-hand side dense array.
// \return The scaled result array.
//
// This operator implements the performance optimized treatment of the multiplication of a
// dense array-scalar multiplication and a dense array. It restructures the expression
// \f$ A=(B*s1)*C \f$ to the expression \f$ A=(B*C)*s1 \f$.
*/
template< typename MT1  // Type of the dense array of the left-hand side expression
        , typename ST   // Type of the scalar of the left-hand side expression
        , typename MT2 > // Type of the right-hand side dense array
inline decltype(auto)
   operator*( const DArrScalarMultExpr<MT1,ST>& lhs, const DenseArray<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * (~rhs) ) * lhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a dense array and a dense array-
//        scalar multiplication expression (\f$ A=(B*s1)*C \f$).
// \ingroup dense_array
//
// \param lhs The left-hand side dense array.
// \param rhs The right-hand side dense array-scalar multiplication.
// \return The scaled result array.
//
// This operator implements the performance optimized treatment of the multiplication of a
// dense array and a dense array-scalar multiplication. It restructures the expression
// \f$ A=B*(C*s1) \f$ to the expression \f$ A=(B*C)*s1 \f$.
*/
template< typename MT1  // Type of the left-hand side dense array
        , typename MT2  // Type of the dense array of the right-hand side expression
        , typename ST >  // Type of the scalar of the right-hand side expression
inline decltype(auto)
   operator*( const DenseArray<MT1>& lhs, const DArrScalarMultExpr<MT2,ST>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( (~lhs) * rhs.leftOperand() ) * rhs.rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of two dense array-scalar multiplication
//        expressions (\f$ A=(B*s1)*(C*s2) \f$).
// \ingroup dense_array
//
// \param lhs The left-hand side dense array-scalar multiplication.
// \param rhs The right-hand side dense array-scalar multiplication.
// \return The scaled result array.
//
// This operator implements the performance optimized treatment of the multiplication of
// two dense array-scalar multiplication expressions. It restructures the expression
// \f$ A=(B*s1)*(C*s2) \f$ to the expression \f$ A=(B*C)*(s1*s2) \f$.
*/
template< typename MT1  // Type of the dense array of the left-hand side expression
        , typename ST1  // Type of the scalar of the left-hand side expression
        , typename MT2  // Type of the right-hand side dense array
        , typename ST2 > // Type of the scalar of the right-hand side expression
inline decltype(auto)
   operator*( const DArrScalarMultExpr<MT1,ST1>& lhs, const DArrScalarMultExpr<MT2,ST2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   return ( lhs.leftOperand() * rhs.leftOperand() ) * ( lhs.rightOperand() * rhs.rightOperand() );
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
struct IsAligned< DArrScalarMultExpr<MT,ST> >
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
struct IsPadded< DArrScalarMultExpr<MT,ST> >
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
struct IsSymmetric< DArrScalarMultExpr<MT,ST> >
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
template< typename MT, typename ST >
struct IsHermitian< DArrScalarMultExpr<MT,ST> >
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
template< typename MT, typename ST >
struct IsLower< DArrScalarMultExpr<MT,ST> >
   : public IsLower<MT>
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
struct IsStrictlyLower< DArrScalarMultExpr<MT,ST> >
   : public IsStrictlyLower<MT>
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
struct IsUpper< DArrScalarMultExpr<MT,ST> >
   : public IsUpper<MT>
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
struct IsStrictlyUpper< DArrScalarMultExpr<MT,ST> >
   : public IsStrictlyUpper<MT>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
