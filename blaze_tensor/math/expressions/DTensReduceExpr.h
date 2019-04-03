//=================================================================================================
/*!
//  \file blaze/math/expressions/DTensReduceExpr.h
//  \brief Header file for the dense tensor reduce expression
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSREDUCEEXPR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSREDUCEEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/functors/Add.h>
#include <blaze/math/functors/Max.h>
#include <blaze/math/functors/Min.h>
#include <blaze/math/functors/Mult.h>
#include <blaze/math/ReductionFlag.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/ReduceTrait.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsSIMDEnabled.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/math/views/Check.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/Assert.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Template.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/HasMember.h>
#include <blaze/util/typetraits/IsSame.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/ReductionFlag.h>
#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/Tensor.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>
#include <blaze_tensor/math/expressions/TensReduceExpr.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Base template for dense tensor reduction operations.
// \ingroup dense_tensor_expression
//
// The DTensReduceExpr class represents the compile time expression for partial reduction operations
// of dense matrices.
*/
template< typename MT  // Type of the dense tensor
        , typename OP  // Type of the reduction operation
        , size_t RF >  // Reduction flag
class DTensReduceExpr
{};
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR COLUMN-WISE REDUCTION OPERATIONS OF TENSORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for column-wise dense tensor reduction operations.
// \ingroup dense_tensor_expression
//
// This specialization of the DTensReduceExpr class template represents the compile time expression
// for column-wise reduction operations of dense matrices.
*/
template< typename MT    // Type of the dense matrix
        , typename OP >  // Type of the reduction operation
class DTensReduceExpr<MT,OP,columnwise>
   : public TensReduceExpr< DenseMatrix< DTensReduceExpr<MT,OP,columnwise>, rowMajor >, columnwise >
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   using RT = ResultType_t<MT>;     //!< Result type of the dense tensor expression.
   using ET = ElementType_t<MT>;    //!< Element type of the dense tensor expression.
   using CT = CompositeType_t<MT>;  //!< Composite type of the dense tensor expression.
   //**********************************************************************************************

   //**Parallel evaluation strategy****************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! This variable template is a helper for the selection of the parallel evaluation strategy.
       In case the dense tensor operand is not SMP assignable and requires an intermediate
       evaluation, the variable is set to 1 and the expression specific evaluation strategy is
       selected. Otherwise the variable is set to 0 and the default strategy is chosen. */
   template< typename VT >
   static constexpr bool UseSMPAssign_v = ( !MT::smpAssignable && RequiresEvaluation_v<MT> );
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This          = DTensReduceExpr<MT,OP,columnwise>;  //!< Type of this DTensReduceExpr instance.
   using ResultType    = ReduceTrait_t<RT,OP,columnwise>;   //!< Result type for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;       //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<ResultType>;         //!< Resulting element type.
   using SIMDType      = SIMDTrait_t<ElementType>;          //!< Resulting SIMD element type.
   using ReturnType    = const ElementType;                 //!< Return type for expression template evaluations.
   using CompositeType = const ResultType;                  //!< Data type for composite expression templates.

   //! Composite type of the left-hand side dense tensor expression.
   using Operand = If_t< IsExpression_v<MT>, const MT, const MT& >;

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
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the ConstIterator class.
      //
      // \param dm The dense tensor operand of the reduction expression.
      // \param j Index to the initial tensor column.
      // \param k Index to the tensor page.
      // \param op The reduction operation.
      */
      explicit inline ConstIterator( Operand dm, size_t j, size_t k, OP op )
         : dm_   ( dm    )  // Dense tensor of the reduction expression
         , j_    ( j     )  // Index to the current tensor column
         , k_    ( k     )  // Index to the tensor page
         , op_   ( op    )  // The reduction operation
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline ConstIterator& operator+=( size_t inc ) {
         j_ += inc;
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
         j_ -= dec;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline ConstIterator& operator++() {
         ++j_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator++( int ) {
         return ConstIterator( j_++, k_ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline ConstIterator& operator--() {
         --j_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator--( int ) {
         return ConstIterator( j_--, k_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReturnType operator*() const {
         return reduce( column( pageslice(dm_, k_, unchecked), j_, unchecked ), op_ );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const ConstIterator& rhs ) const {
         return j_ == rhs.j_ && k_ == rhs.k_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const ConstIterator& rhs ) const {
         return j_ != rhs.j_ || k_ != rhs.k_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const ConstIterator& rhs ) const {
         return j_ < rhs.j_ ? true : k_ < rhs.k_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const ConstIterator& rhs ) const {
         return !(*this <= rhs);
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const ConstIterator& rhs ) const {
         return j_ <= rhs.j_ ? true : k_ <= rhs.k_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const ConstIterator& rhs ) const {
         return !(*this < rhs);
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const ConstIterator& rhs ) const {
         return j_ - rhs.j_;
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
         return ConstIterator( it.j_ + inc, it.k_ );
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
         return ConstIterator( it.j_ + inc, it.k_ );
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
         return ConstIterator( it.j_ - dec, it.k_ );
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      Operand dm_;     //!< Dense tensor of the reduction expression.
      size_t  j_;      //!< Index to the current tensor row.
      size_t  k_;      //!< Index to the tensor page.
      OP      op_;     //!< The reduction operation.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //! Data type of the custom unary operation.
   using Operation = OP;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = false;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = MT::smpAssignable;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DTensReduceExpr class.
   //
   // \param dm The tensor operand of the reduction expression.
   // \param op The reduction operation.
   */
   explicit inline DTensReduceExpr( const MT& dm, OP op ) noexcept
      : dm_( dm )  // Dense tensor of the reduction expression
      , op_( op )  // The reduction operation
   {}
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the tensor elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator()( size_t j, size_t k ) const {
      BLAZE_INTERNAL_ASSERT( j < dm_.columns(), "Invalid tensor access index" );
      BLAZE_INTERNAL_ASSERT( k < dm_.pages(), "Invalid tensor access index" );
      return reduce( column( pageslice(dm_, k, unchecked ), j, unchecked), op_ );
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the tensor elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid tensor access index.
   */
   inline ReturnType at( size_t j, size_t k ) const {
      if( j >= dm_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid tensor access index" );
      }
      if( k >= dm_.pages() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid tensor access index" );
      }
      return (*this)(j, k);
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first element of the dense tensor.
   //
   // \return Iterator to the first element of the dense tensor.
   */
   inline ConstIterator begin( size_t k ) const {
      return ConstIterator( dm_, 0UL, k, op_ );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of the dense tensor.
   //
   // \return Iterator just past the last non-zero element of the dense tensor.
   */
   inline ConstIterator end( size_t k ) const {
      return ConstIterator( dm_, dm_.columns(), k, op_ );
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current size/dimension of the tensor.
   //
   // \return The size of the tensor.
   */
   inline size_t rows() const noexcept {
      return dm_.pages();
   }
   //**********************************************************************************************

   //**Pages function*******************************************************************************
   /*!\brief Returns the current size/dimension of the tensor.
   //
   // \return The size of the tensor.
   */
   inline size_t columns() const noexcept {
      return dm_.columns();
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
   /*!\brief Returns a copy of the reduction operation.
   //
   // \return A copy of the reduction operation.
   */
   inline Operation operation() const {
      return op_;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case an aliasing effect is possible, \a false if not.
   */
   template< typename T >
   inline bool canAlias( const T* alias ) const noexcept {
      return ( dm_.isAliased( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the given alias is contained in this expression, \a false if not.
   */
   template< typename T >
   inline bool isAliased( const T* alias ) const noexcept {
      return ( dm_.isAliased( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const noexcept {
      return false;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return dm_.canSMPAssign() || ( rows() * columns() > SMP_DMATREDUCE_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   Operand   dm_;  //!< Dense tensor of the reduction expression.
   Operation op_;  //!< The reduction operation.
   //**********************************************************************************************

   //**Assignment to dense tensors*****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a column-wise dense tensor reduction operation to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side reduction expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a column-wise row-major
   // dense tensor reduction expression to a dense tensor.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline void assign( DenseMatrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const size_t M( rhs.dm_.rows() );

      if( M == 0UL ) {
         reset( ~lhs );
         return;
      }

      CT tmp( serial( rhs.dm_ ) );

      assign( ~lhs, rowslice( tmp, 0UL, unchecked ) );
      for( size_t i=1UL; i<M; ++i ) {
         assign( ~lhs, map( ~lhs, rowslice( tmp, i, unchecked ), rhs.op_ ) );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to sparse tensors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a column-wise dense tensor reduction operation to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side reduction expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a column-wise row-major
   // dense tensor reduction expression to a dense tensor.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline void assign( SparseMatrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const ResultType tmp( serial( rhs ) );
      assign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense tensors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a column-wise dense tensor reduction operation to a
   //        dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side reduction expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a column-wise
   // dense tensor reduction expression to a dense tensor.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline void addAssign( DenseMatrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      if( rhs.dm_.rows() == 0UL ) {
         return;
      }
      else if( IsSame_v<OP,Add> ) {
         CT tmp( serial( rhs.dm_ ) );
         const size_t M( tmp.rows() );
         for( size_t i=0UL; i<M; ++i ) {
            addAssign( (~lhs), rowslice( tmp, i, unchecked ) );
         }
      }
      else {
         const ResultType tmp( serial( rhs ) );
         addAssign( ~lhs, tmp );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to sparse tensors*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a column-wise dense tensor reduction operation to a
   //        dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side reduction expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a column-wise
   // dense tensor reduction expression to a dense tensor.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline void addAssign( SparseMatrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const ResultType tmp( serial( rhs ) );
      addAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to dense tensors*****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a column-wise dense tensor reduction operation
   //        to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side reduction expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a column-wise
   // dense tensor reduction expression to a dense tensor.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline void subAssign( DenseMatrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      if( rhs.dm_.rows() == 0UL ) {
         return;
      }
      else if( IsSame_v<OP,Add> ) {
         CT tmp( serial( rhs.dm_ ) );
         const size_t M( tmp.rows() );
         for( size_t i=0UL; i<M; ++i ) {
            subAssign( (~lhs), rowslice( tmp, i, unchecked ) );
         }
      }
      else {
         const ResultType tmp( serial( rhs ) );
         subAssign( ~lhs, tmp );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to sparse tensors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a column-wise dense tensor reduction operation
   //        to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side reduction expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a column-wise
   // dense tensor reduction expression to a dense tensor.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline void subAssign( SparseMatrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const ResultType tmp( serial( rhs ) );
      subAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Multiplication assignment to dense tensors**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Multiplication assignment of a column-wise dense tensor reduction operation
   //        to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side reduction expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a column-wise
   // dense tensor reduction expression to a dense tensor.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline void multAssign( DenseMatrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      if( rhs.dm_.rows() == 0UL ) {
         reset( ~lhs );
      }
      else if( IsSame_v<OP,Mult> ) {
         CT tmp( serial( rhs.dm_ ) );
         const size_t M( tmp.rows() );
         for( size_t i=0UL; i<M; ++i ) {
            multAssign( (~lhs), rowslice( tmp, i, unchecked ) );
         }
      }
      else {
         const ResultType tmp( serial( rhs ) );
         multAssign( ~lhs, tmp );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Multiplication assignment to sparse tensors*************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Multiplication assignment of a column-wise dense tensor reduction operation
   //        to a dense tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side reduction expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a column-wise
   // dense tensor reduction expression to a dense tensor.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline void multAssign( SparseMatrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const ResultType tmp( serial( rhs ) );
      multAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Division assignment to tensors**************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Division assignment of a column-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a column-wise
   // dense tensor reduction expression to a tensor.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline void divAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const ResultType tmp( serial( rhs ) );
      divAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to tensors*******************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a column-wise dense tensor reduction operation to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a column-wise row-major
   // dense tensor reduction expression to a tensor. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression specific
   // parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpAssign( ~lhs, reduce<columnwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to tensors**********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a column-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a column-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpAddAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpAddAssign( ~lhs, reduce<columnwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to tensors*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a column-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a column-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpSubAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpSubAssign( ~lhs, reduce<columnwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP multiplication assignment to tensors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP multiplication assignment of a column-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a
   // column-wise dense tensor reduction expression to a tensor. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpMultAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpMultAssign( ~lhs, reduce<columnwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP division assignment to tensors**********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP division assignment of a column-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression divisor.
   // \return void
   //
   // This function implements the performance optimized SMP division assignment of a column-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpDivAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpDivAssign( ~lhs, reduce<columnwise>( tmp, rhs.op_ ) );
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
//  CLASS TEMPLATE SPECIALIZATION FOR ROW-WISE REDUCTION OPERATIONS OF TENSORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for row-wise dense tensor reduction operations.
// \ingroup dense_tensor_expression
//
// This specialization of the DTensReduceExpr class template represents the compile time expression
// for row-wise reduction operations of dense matrices.
*/
template< typename MT    // Type of the dense matrix
        , typename OP >  // Type of the reduction operation
class DTensReduceExpr<MT,OP,rowwise>
   : public TensReduceExpr< DenseMatrix< DTensReduceExpr<MT,OP,rowwise>, rowMajor >, rowwise >
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   using RT = ResultType_t<MT>;   //!< Result type of the dense tensor expression.
   using ET = ElementType_t<MT>;  //!< Element type of the dense tensor expression.
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the reduction expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the serial evaluation strategy of the reduction expression. In case the dense tensor
       operand requires an intermediate evaluation, \a useAssign will be set to 1 and the
       reduction expression will be evaluated via the \a assign function family. Otherwise
       \a useAssign will be set to 0 and the expression will be evaluated via the subscript
       operator. */
   static constexpr bool useAssign = RequiresEvaluation_v<MT>;

   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename VT >
   static constexpr bool UseAssign_v = useAssign;
   /*! \endcond */
   //**********************************************************************************************

   //**Parallel evaluation strategy****************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! This variable template is a helper for the selection of the parallel evaluation strategy.
       In case the dense tensor operand is not SMP assignable and requires an intermediate
       evaluation, the variable is set to 1 and the expression specific evaluation strategy
       is selected. Otherwise the variable is set to 0 and the default strategy is chosen. */
   template< typename VT >
   static constexpr bool UseSMPAssign_v = ( !MT::smpAssignable && useAssign );
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This          = DTensReduceExpr<MT,OP,rowwise>; //!< Type of this DTensReduceExpr instance.
   using ResultType    = ReduceTrait_t<RT,OP,rowwise>;   //!< Result type for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;    //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<ResultType>;      //!< Resulting element type.
   using SIMDType      = SIMDTrait_t<ElementType>;       //!< Resulting SIMD element type.
   using ReturnType    = const ElementType;              //!< Return type for expression template evaluations.

   //! Data type for composite expression templates.
   using CompositeType = If_t< useAssign, const ResultType, const DTensReduceExpr& >;

   //! Composite type of the left-hand side dense tensor expression.
   using Operand = If_t< IsExpression_v<MT>, const MT, const MT& >;

   //! Data type of the custom unary operation.
   using Operation = OP;
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
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the ConstIterator class.
      //
      // \param dm The dense tensor operand of the reduction expression.
      // \param i Index to the initial tensor row.
      // \param i Index to the tensor column.
      // \param op The reduction operation.
      */
      explicit inline ConstIterator( Operand dm, size_t j, size_t i, OP op )
         : dm_   ( dm    )  // Dense tensor of the reduction expression
         , j_    ( j )      // Index to the current tensor column
         , i_    ( i )      // Index to the tensor row
         , op_   ( op    )  // The reduction operation
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline ConstIterator& operator+=( size_t inc ) {
         j_ += inc;
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
         j_ -= dec;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline ConstIterator& operator++() {
         ++j_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator++( int ) {
         return ConstIterator( j_++, i_ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline ConstIterator& operator--() {
         --j_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator--( int ) {
         return ConstIterator( j_--, i_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReturnType operator*() const {
         return reduce( row( columnslice(dm_, j_, unchecked), i_, unchecked ), op_ );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const ConstIterator& rhs ) const {
         return j_ == rhs.j_ && i_ == rhs.i_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const ConstIterator& rhs ) const {
         return j_ != rhs.j_ || i_ != rhs.i_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const ConstIterator& rhs ) const {
         return j_ < rhs.j_ ? true : i_ < rhs.i_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const ConstIterator& rhs ) const {
         return !(*this <= rhs);
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const ConstIterator& rhs ) const {
         return j_ <= rhs.j_ ? true : i_ <= rhs.i_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const ConstIterator& rhs ) const {
         return !(*this < rhs);
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const ConstIterator& rhs ) const {
         return j_ - rhs.j_;
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
         return ConstIterator( it.j_ + inc, it.i_ );
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
         return ConstIterator( it.j_ + inc, it.i_ );
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
         return ConstIterator( it.j_ - dec, it.i_ );
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      Operand dm_;     //!< Dense tensor of the reduction expression.
      size_t  j_;      //!< Index to the current tensor column.
      size_t  i_;      //!< Index to the tensor row.
      OP      op_;     //!< The reduction operation.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = false;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = MT::smpAssignable;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DTensReduceExpr class.
   //
   // \param dm The tensor operand of the reduction expression.
   // \param op The reduction operation.
   */
   explicit inline DTensReduceExpr( const MT& dm, OP op ) noexcept
      : dm_( dm )  // Dense tensor of the reduction expression
      , op_( op )  // The reduction operation
   {}
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the tensor elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator()( size_t i, size_t j ) const {
      BLAZE_INTERNAL_ASSERT( i < dm_.rows(), "Invalid tensor access index" );
      BLAZE_INTERNAL_ASSERT( j < dm_.columns(), "Invalid tensor access index" );
      return reduce( column( rowslice(dm_, i, unchecked ), j, unchecked ), op_ );
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the tensor elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid tensor access index.
   */
   inline ReturnType at( size_t i, size_t j ) const {
      if( i >= dm_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid tensor access index" );
      }
      if( j >= dm_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid tensor access index" );
      }
      return (*this)(i, j);
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first element of the dense tensor.
   //
   // \return Iterator to the first element of the dense tensor.
   */
   inline ConstIterator begin( size_t i ) const {
      return ConstIterator( dm_, 0UL, i, op_ );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of the dense tensor.
   //
   // \return Iterator just past the last non-zero element of the dense tensor.
   */
   inline ConstIterator end( size_t i ) const {
      return ConstIterator( dm_, dm_.columns(), i, op_ );
   }
   //**********************************************************************************************

   //**Size function*******************************************************************************
   /*!\brief Returns the current size/dimension of the tensor.
   //
   // \return The size of the tensor.
   */
   inline size_t rows() const noexcept {
      return dm_.rows();
   }
   //**********************************************************************************************

   //**Pages function*******************************************************************************
   /*!\brief Returns the current size/dimension of the tensor.
   //
   // \return The size of the tensor.
   */
   inline size_t columns() const noexcept {
      return dm_.columns();
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
   /*!\brief Returns a copy of the reduction operation.
   //
   // \return A copy of the reduction operation.
   */
   inline Operation operation() const {
      return op_;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case an aliasing effect is possible, \a false if not.
   */
   template< typename T >
   inline bool canAlias( const T* alias ) const noexcept {
      return ( dm_.isAliased( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the given alias is contained in this expression, \a false if not.
   */
   template< typename T >
   inline bool isAliased( const T* alias ) const noexcept {
      return ( dm_.isAliased( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const noexcept {
      return false;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return dm_.canSMPAssign() || ( rows() * columns() > SMP_DMATREDUCE_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   Operand   dm_;  //!< Dense tensor of the reduction expression.
   Operation op_;  //!< The reduction operation.
   //**********************************************************************************************

   //**Assignment to tensors***********************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a row-wise dense tensor reduction operation to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a row-wise row-major
   // dense tensor reduction expression to a tensor. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression specific
   // parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto assign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense tensor operand
      assign( ~lhs, reduce<rowwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to tensors**************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a row-wise dense tensor reduction operation to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto addAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense tensor operand
      addAssign( ~lhs, reduce<rowwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to tensors***********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a row-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto subAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense tensor operand
      subAssign( ~lhs, reduce<rowwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Multiplication assignment to tensors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Multiplication assignment of a row-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto multAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense tensor operand
      multAssign( ~lhs, reduce<rowwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Division assignment to tensors**************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Division assignment of a row-wise dense tensor reduction operation to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto divAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense tensor operand
      divAssign( ~lhs, reduce<rowwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to tensors*******************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a row-wise dense tensor reduction operation to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a row-wise row-major
   // dense tensor reduction expression to a tensor. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression specific
   // parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpAssign( ~lhs, reduce<rowwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to tensors**********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a row-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpAddAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpAddAssign( ~lhs, reduce<rowwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to tensors*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a row-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpSubAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpSubAssign( ~lhs, reduce<rowwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP multiplication assignment to tensors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP multiplication assignment of a row-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a
   // row-wise dense tensor reduction expression to a tensor. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpMultAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpMultAssign( ~lhs, reduce<rowwise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP division assignment to tensors**********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP division assignment of a row-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression divisor.
   // \return void
   //
   // This function implements the performance optimized SMP division assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpDivAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpDivAssign( ~lhs, reduce<rowwise>( tmp, rhs.op_ ) );
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
//  CLASS TEMPLATE SPECIALIZATION FOR PAGE-WISE REDUCTION OPERATIONS OF TENSORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for row-wise dense tensor reduction operations.
// \ingroup dense_tensor_expression
//
// This specialization of the DTensReduceExpr class template represents the compile time expression
// for row-wise reduction operations of dense matrices.
*/
template< typename MT    // Type of the dense matrix
        , typename OP >  // Type of the reduction operation
class DTensReduceExpr<MT,OP,pagewise>
   : public TensReduceExpr< DenseMatrix< DTensReduceExpr<MT,OP,pagewise>, true >, pagewise >
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   using RT = ResultType_t<MT>;   //!< Result type of the dense tensor expression.
   using ET = ElementType_t<MT>;  //!< Element type of the dense tensor expression.
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the reduction expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the serial evaluation strategy of the reduction expression. In case the dense tensor
       operand requires an intermediate evaluation, \a useAssign will be set to 1 and the
       reduction expression will be evaluated via the \a assign function family. Otherwise
       \a useAssign will be set to 0 and the expression will be evaluated via the subscript
       operator. */
   static constexpr bool useAssign = RequiresEvaluation_v<MT>;

   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename VT >
   static constexpr bool UseAssign_v = useAssign;
   /*! \endcond */
   //**********************************************************************************************

   //**Parallel evaluation strategy****************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! This variable template is a helper for the selection of the parallel evaluation strategy.
       In case the dense tensor operand is not SMP assignable and requires an intermediate
       evaluation, the variable is set to 1 and the expression specific evaluation strategy
       is selected. Otherwise the variable is set to 0 and the default strategy is chosen. */
   template< typename VT >
   static constexpr bool UseSMPAssign_v = ( !MT::smpAssignable && useAssign );
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This          = DTensReduceExpr<MT,OP,pagewise>; //!< Type of this DTensReduceExpr instance.
   using ResultType    = ReduceTrait_t<RT,OP,pagewise>;   //!< Result type for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;    //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<ResultType>;      //!< Resulting element type.
   using SIMDType      = SIMDTrait_t<ElementType>;       //!< Resulting SIMD element type.
   using ReturnType    = const ElementType;              //!< Return type for expression template evaluations.

   //! Data type for composite expression templates.
   using CompositeType = If_t< useAssign, const ResultType, const DTensReduceExpr& >;

   //! Composite type of the left-hand side dense tensor expression.
   using Operand = If_t< IsExpression_v<MT>, const MT, const MT& >;

   //! Data type of the custom unary operation.
   using Operation = OP;
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
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the ConstIterator class.
      //
      // \param dm The dense tensor operand of the reduction expression.
      // \param i Index to the initial tensor page.
      // \param i Index to the tensor row.
      // \param op The reduction operation.
      */
      explicit inline ConstIterator( Operand dm, size_t k, size_t i, OP op )
         : dm_   ( dm    )  // Dense tensor of the reduction expression
         , k_    ( k )      // Index to the current tensor page
         , i_    ( i )      // Index to the tensor row
         , op_   ( op    )  // The reduction operation
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline ConstIterator& operator+=( size_t inc ) {
         k_ += inc;
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
         k_ -= dec;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline ConstIterator& operator++() {
         ++k_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator++( int ) {
         return ConstIterator( k_++, i_ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline ConstIterator& operator--() {
         --k_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const ConstIterator operator--( int ) {
         return ConstIterator( k_--, i_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReturnType operator*() const {
         return reduce( column( rowslice(dm_, i_, unchecked), k_, unchecked ), op_ );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const ConstIterator& rhs ) const {
         return k_ == rhs.k_ && i_ == rhs.i_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const ConstIterator& rhs ) const {
         return k_ != rhs.k_ || i_ != rhs.i_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const ConstIterator& rhs ) const {
         return k_ < rhs.k_ ? true : i_ < rhs.i_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const ConstIterator& rhs ) const {
         return !(*this <= rhs);
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const ConstIterator& rhs ) const {
         return k_ <= rhs.k_ ? true : i_ <= rhs.i_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const ConstIterator& rhs ) const {
         return !(*this < rhs);
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const ConstIterator& rhs ) const {
         return k_ - rhs.k_;
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
         return ConstIterator( it.k_ + inc, it.i_ );
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
         return ConstIterator( it.k_ + inc, it.i_ );
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
         return ConstIterator( it.k_ - dec, it.i_ );
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      Operand dm_;     //!< Dense tensor of the reduction expression.
      size_t  k_;      //!< Index to the current tensor page.
      size_t  i_;      //!< Index to the tensor row.
      OP      op_;     //!< The reduction operation.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = false;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = MT::smpAssignable;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DTensReduceExpr class.
   //
   // \param dm The tensor operand of the reduction expression.
   // \param op The reduction operation.
   */
   explicit inline DTensReduceExpr( const MT& dm, OP op ) noexcept
      : dm_( dm )  // Dense tensor of the reduction expression
      , op_( op )  // The reduction operation
   {}
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the tensor elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator()( size_t k, size_t i ) const {
      BLAZE_INTERNAL_ASSERT( k < dm_.pages(), "Invalid tensor access index" );
      BLAZE_INTERNAL_ASSERT( i < dm_.rows(), "Invalid tensor access index" );
      return reduce( column( rowslice(dm_, i, unchecked ), k, unchecked ), op_ );
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the tensor elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid tensor access index.
   */
   inline ReturnType at( size_t k, size_t i ) const {
      if( k >= dm_.pages() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid tensor access index" );
      }
      if( i >= dm_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid tensor access index" );
      }
      return (*this)(k, i);
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first element of the dense tensor.
   //
   // \return Iterator to the first element of the dense tensor.
   */
   inline ConstIterator begin( size_t i ) const {
      return ConstIterator( dm_, 0UL, i, op_ );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of the dense tensor.
   //
   // \return Iterator just past the last non-zero element of the dense tensor.
   */
   inline ConstIterator end( size_t i ) const {
      return ConstIterator( dm_, dm_.pages(), i, op_ );
   }
   //**********************************************************************************************

   //**Size function*******************************************************************************
   /*!\brief Returns the current size/dimension of the tensor.
   //
   // \return The size of the tensor.
   */
   inline size_t rows() const noexcept {
      return dm_.rows();
   }
   //**********************************************************************************************

   //**Pages function*******************************************************************************
   /*!\brief Returns the current size/dimension of the tensor.
   //
   // \return The size of the tensor.
   */
   inline size_t columns() const noexcept {
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
   /*!\brief Returns a copy of the reduction operation.
   //
   // \return A copy of the reduction operation.
   */
   inline Operation operation() const {
      return op_;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case an aliasing effect is possible, \a false if not.
   */
   template< typename T >
   inline bool canAlias( const T* alias ) const noexcept {
      return ( dm_.isAliased( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the given alias is contained in this expression, \a false if not.
   */
   template< typename T >
   inline bool isAliased( const T* alias ) const noexcept {
      return ( dm_.isAliased( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const noexcept {
      return false;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return dm_.canSMPAssign() || ( rows() * columns() > SMP_DMATREDUCE_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   Operand   dm_;  //!< Dense tensor of the reduction expression.
   Operation op_;  //!< The reduction operation.
   //**********************************************************************************************

   //**Assignment to tensors***********************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a row-wise dense tensor reduction operation to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a row-wise row-major
   // dense tensor reduction expression to a tensor. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression specific
   // parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto assign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense tensor operand
      assign( ~lhs, reduce<pagewise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to tensors**************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a row-wise dense tensor reduction operation to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto addAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense tensor operand
      addAssign( ~lhs, reduce<pagewise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to tensors***********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a row-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto subAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense tensor operand
      subAssign( ~lhs, reduce<pagewise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Multiplication assignment to tensors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Multiplication assignment of a row-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto multAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense tensor operand
      multAssign( ~lhs, reduce<pagewise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Division assignment to tensors**************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Division assignment of a row-wise dense tensor reduction operation to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto divAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense tensor operand
      divAssign( ~lhs, reduce<pagewise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to tensors*******************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a row-wise dense tensor reduction operation to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a row-wise row-major
   // dense tensor reduction expression to a tensor. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression specific
   // parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpAssign( ~lhs, reduce<pagewise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to tensors**********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a row-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpAddAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpAddAssign( ~lhs, reduce<pagewise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to tensors*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a row-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpSubAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpSubAssign( ~lhs, reduce<pagewise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP multiplication assignment to tensors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP multiplication assignment of a row-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a
   // row-wise dense tensor reduction expression to a tensor. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpMultAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpMultAssign( ~lhs, reduce<pagewise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP division assignment to tensors**********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP division assignment of a row-wise dense tensor reduction operation
   //        to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side tensor.
   // \param rhs The right-hand side reduction expression divisor.
   // \return void
   //
   // This function implements the performance optimized SMP division assignment of a row-wise
   // dense tensor reduction expression to a tensor. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1   // Type of the target dense tensor
           , bool SO >      // Storage order of destination matrix
   friend inline auto smpDivAssign( Matrix<VT1,SO>& lhs, const DTensReduceExpr& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows() == rhs.rows(), "Invalid tensor sizes" );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid tensor sizes" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense tensor operand
      smpDivAssign( ~lhs, reduce<pagewise>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT );
   /*! \endcond */
   //**********************************************************************************************
};
//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the dense tensor reduction operation.
// \ingroup dense_tensor
*/
template< typename MT    // Type of the dense tensor
        , typename OP >  // Type of the reduction operation
struct DTensReduceExprHelper
{
   //**Type definitions****************************************************************************
   //! Composite type of the dense tensor expression.
   using CT = RemoveReference_t< CompositeType_t<MT> >;

   //! Element type of the dense tensor expression.
   using ET = ElementType_t<CT>;

   //! Definition of the HasSIMDEnabled type trait.
   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasSIMDEnabled, simdEnabled );

   //! Definition of the HasLoad type trait.
   BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasLoad, load );
   //**********************************************************************************************

   //**********************************************************************************************
   static constexpr bool value =
      ( CT::simdEnabled &&
        If_t< HasSIMDEnabled_v<OP>, GetSIMDEnabled<OP,ET,ET>, HasLoad<OP> >::value );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default backend implementation of the reduction of a dense tensor.
// \ingroup dense_tensor
//
// \param dm The given dense tensor for the reduction computation.
// \param op The reduction operation.
// \return The result of the reduction operation.
//
// This function implements the performance optimized reduction operation for a dense
// tensor. Due to the explicit application of the SFINAE principle, this function can only be
// selected by the compiler in case vectorization cannot be applied.
*/
template< typename MT    // Type of the dense tensor
        , typename OP >  // Type of the reduction operation
inline auto dtensreduce( const DenseTensor<MT>& dm, OP op )
   -> DisableIf_t< DTensReduceExprHelper<MT,OP>::value, ElementType_t<MT> >
{
   using CT = CompositeType_t<MT>;
   using ET = ElementType_t<MT>;

   const size_t M( (~dm).rows()    );
   const size_t N( (~dm).columns() );
   const size_t O( (~dm).pages()   );

   if( M == 0UL || N == 0UL || O == 0UL ) return ET{};
   if( M == 1UL && N == 1UL && O == 1UL ) return (~dm)(0UL,0UL,0UL);

   CT tmp( ~dm );

   BLAZE_INTERNAL_ASSERT( tmp.rows()    == M, "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( tmp.columns() == N, "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( tmp.pages()   == O, "Invalid number of pages" );

   ET redux0{};

   {
      redux0 = tmp(0UL,0UL,0UL);

      size_t j(1UL);
      for( size_t k=0UL; k<O; ++k )
      {
         for( ; j<N; ++j )
         {
            redux0 = op( redux0, tmp(k,0UL,j) );
         }
         j = 0UL;
      }
   }

   size_t i( 1UL );
   for( size_t k=0UL; k<O; ++k )
   {
      for( ; (i+2UL) <= M; i+=2UL )
      {
         ET redux1( tmp(k,i    ,0UL) );
         ET redux2( tmp(k,i+1UL,0UL) );

         for( size_t j=1UL; j<N; ++j ) {
            redux1 = op( redux1, tmp(k,i    ,j) );
            redux2 = op( redux2, tmp(k,i+1UL,j) );
         }

         redux1 = op( redux1, redux2 );
         redux0 = op( redux0, redux1 );
      }

      if( i < M )
      {
         ET redux1( tmp(k,i,0UL) );

         for( size_t j=1UL; j<N; ++j ) {
            redux1 = op( redux1, tmp(k,i,j) );
         }

         redux0 = op( redux0, redux1 );
      }

      i = 1UL;
   }

   return redux0;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized backend implementation of the reduction of a dense tensor.
// \ingroup dense_tensor
//
// \param dm The given dense tensor for the reduction computation.
// \param op The reduction operation.
// \return The result of the reduction operation.
//
// This function implements the performance optimized reduction operation for a dense
// tensor. Due to the explicit application of the SFINAE principle, this function can only be
// selected by the compiler in case vectorization can be applied.
*/
template< typename MT    // Type of the dense tensor
        , typename OP >  // Type of the reduction operation
inline auto dtensreduce( const DenseTensor<MT>& dm, OP op )
   -> EnableIf_t< DTensReduceExprHelper<MT,OP>::value, ElementType_t<MT> >
{
   using CT = CompositeType_t<MT>;
   using ET = ElementType_t<MT>;

   const size_t M( (~dm).rows()    );
   const size_t N( (~dm).columns() );
   const size_t O( (~dm).pages()   );

   if( M == 0UL || N == 0UL || O == 0UL ) return ET{};

   CT tmp( ~dm );

   BLAZE_INTERNAL_ASSERT( tmp.rows()    == M, "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( tmp.columns() == N, "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( tmp.pages()   == O, "Invalid number of pages" );

   constexpr size_t SIMDSIZE = SIMDTrait<ET>::size;

   alignas( AlignmentOf_v<ET> ) ET array1[SIMDSIZE];
   alignas( AlignmentOf_v<ET> ) ET array2[SIMDSIZE];
   alignas( AlignmentOf_v<ET> ) ET array3[SIMDSIZE];
   alignas( AlignmentOf_v<ET> ) ET array4[SIMDSIZE];

   ET redux{};

   if( N >= SIMDSIZE )
   {
      const size_t jpos( N & size_t(-SIMDSIZE) );
      BLAZE_INTERNAL_ASSERT( ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      SIMDTrait_t<ET> xmm1 = tmp.load(0UL,0UL,0UL);

      for( size_t k=0UL; k<O; ++k )
      {
         {
            if ( k != 0UL ) {
               xmm1 = op( xmm1, tmp.load(k,0UL,0UL) );
            }

            size_t j( SIMDSIZE );

            for( ; j<jpos; j+=SIMDSIZE ) {
               xmm1 = op( xmm1, tmp.load(k,0UL,j) );
            }

            if( jpos < N )
            {
               storea( array1, xmm1 );

               for( ; j<N; ++j ) {
                  array1[0UL] = op( array1[0UL], tmp(k,0UL,j) );
               }

               xmm1 = loada( array1 );
            }
         }

         size_t i( 1UL );

         for( ; (i+4UL) <= M; i+=4UL )
         {
            xmm1 = op( xmm1, tmp.load(k,i,0UL) );
            SIMDTrait_t<ET> xmm2( tmp.load(k,i+1UL,0UL) );
            SIMDTrait_t<ET> xmm3( tmp.load(k,i+2UL,0UL) );
            SIMDTrait_t<ET> xmm4( tmp.load(k,i+3UL,0UL) );
            size_t j( SIMDSIZE );

            for( ; j<jpos; j+=SIMDSIZE ) {
               xmm1 = op( xmm1, tmp.load(k,i    ,j) );
               xmm2 = op( xmm2, tmp.load(k,i+1UL,j) );
               xmm3 = op( xmm3, tmp.load(k,i+2UL,j) );
               xmm4 = op( xmm4, tmp.load(k,i+3UL,j) );
            }

            if( jpos < N )
            {
               storea( array1, xmm1 );
               storea( array2, xmm2 );
               storea( array3, xmm3 );
               storea( array4, xmm4 );

               for( ; j<N; ++j ) {
                  array1[0UL] = op( array1[0UL], tmp(k,i    ,j) );
                  array2[0UL] = op( array2[0UL], tmp(k,i+1UL,j) );
                  array3[0UL] = op( array3[0UL], tmp(k,i+2UL,j) );
                  array4[0UL] = op( array4[0UL], tmp(k,i+3UL,j) );
               }

               xmm1 = loada( array1 );
               xmm2 = loada( array2 );
               xmm3 = loada( array3 );
               xmm4 = loada( array4 );
            }

            xmm1 = op( xmm1, xmm2 );
            xmm3 = op( xmm3, xmm4 );
            xmm1 = op( xmm1, xmm3 );
         }

         if( i+2UL <= M )
         {
            xmm1 = op( xmm1, tmp.load(k,i,0UL) );
            SIMDTrait_t<ET> xmm2( tmp.load(k,i+1UL,0UL) );
            size_t j( SIMDSIZE );

            for( ; j<jpos; j+=SIMDSIZE ) {
               xmm1 = op( xmm1, tmp.load(k,i    ,j) );
               xmm2 = op( xmm2, tmp.load(k,i+1UL,j) );
            }

            if( jpos < N )
            {
               storea( array1, xmm1 );
               storea( array2, xmm2 );

               for( ; j<N; ++j ) {
                  array1[0UL] = op( array1[0UL], tmp(k,i    ,j) );
                  array2[0UL] = op( array2[0UL], tmp(k,i+1UL,j) );
               }

               xmm1 = loada( array1 );
               xmm2 = loada( array2 );
            }

            xmm1 = op( xmm1, xmm2 );

            i += 2UL;
         }

         if( i < M )
         {
            xmm1 = op( xmm1, tmp.load(k,i,0UL) );
            size_t j( SIMDSIZE );

            for( ; j<jpos; j+=SIMDSIZE ) {
               xmm1 = op( xmm1, tmp.load(k,i,j) );
            }

            if( jpos < N )
            {
               storea( array1, xmm1 );

               for( ; j<N; ++j ) {
                  array1[0] = op( array1[0], tmp(k,i,j) );
               }

               xmm1 = loada( array1 );
            }
         }

         redux = reduce( xmm1, op );
      }
   }
   else
   {
      for( size_t k=0UL; k<O; ++k )
      {
         if ( k == 0UL ) {
            redux = tmp(k,0UL,0UL);
         }
         else {
            redux = op( redux, tmp(k,0UL,0UL) );
         }
         for( size_t j=1UL; j<N; ++j ) {
            redux = op( redux, tmp(k,0UL,j) );
         }

         for( size_t i=1UL; i<M; ++i ) {
            for( size_t j=0UL; j<N; ++j ) {
               redux = op( redux, tmp(k,i,j) );
            }
         }
      }
   }

   return redux;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized backend implementation of the summation of a dense tensor.
// \ingroup dense_tensor
//
// \param dm The given dense tensor for the summation.
// \return The result of the summation.
//
// This function implements the performance optimized summation for a dense tensor.
// Due to the explicit application of the SFINAE principle, this function can only be selected
// by the compiler in case vectorization can be applied.
*/
template< typename MT >  // Type of the dense tensor
inline auto dtensreduce( const DenseTensor<MT>& dm, Add /*op*/ )
   -> EnableIf_t< DTensReduceExprHelper<MT,Add>::value, ElementType_t<MT> >
{
   using CT = CompositeType_t<MT>;
   using ET = ElementType_t<MT>;

   const size_t M( (~dm).rows()    );
   const size_t N( (~dm).columns() );
   const size_t O( (~dm).pages()   );

   if( M == 0UL || N == 0UL || O == 0UL ) return ET{};

   CT tmp( ~dm );

   BLAZE_INTERNAL_ASSERT( tmp.rows()    == M, "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( tmp.columns() == N, "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( tmp.pages()   == O, "Invalid number of pages" );

   constexpr bool remainder( !usePadding || !IsPadded_v< RemoveReference_t<CT> > );
   constexpr size_t SIMDSIZE = SIMDTrait<ET>::size;

   ET redux{};

   if( !remainder || N >= SIMDSIZE )
   {
      const size_t jpos( ( remainder )?( N & size_t(-SIMDSIZE) ):( N ) );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      SIMDTrait_t<ET> xmm1;

      for( size_t k=0UL; k<O; ++k )
      {
         size_t i( 0UL );
         for( ; (i+4UL) <= M; i+=4UL )
         {
            xmm1 = tmp.load(k,i,0UL);
            SIMDTrait_t<ET> xmm2( tmp.load(k,i+1UL,0UL) );
            SIMDTrait_t<ET> xmm3( tmp.load(k,i+2UL,0UL) );
            SIMDTrait_t<ET> xmm4( tmp.load(k,i+3UL,0UL) );
            size_t j( SIMDSIZE );

            for( ; j<jpos; j+=SIMDSIZE ) {
               xmm1 += tmp.load(k,i    ,j);
               xmm2 += tmp.load(k,i+1UL,j);
               xmm3 += tmp.load(k,i+2UL,j);
               xmm4 += tmp.load(k,i+3UL,j);
            }
            for( ; remainder && j<N; ++j ) {
               redux += tmp(k,i    ,j);
               redux += tmp(k,i+1UL,j);
               redux += tmp(k,i+2UL,j);
               redux += tmp(k,i+3UL,j);
            }

            xmm1 += xmm2;
            xmm3 += xmm4;
            xmm1 += xmm3;
         }

         if( i+2UL <= M )
         {
            xmm1 += tmp.load(k,i,0UL);
            SIMDTrait_t<ET> xmm2( tmp.load(k,i+1UL,0UL) );
            size_t j( SIMDSIZE );

            for( ; j<jpos; j+=SIMDSIZE ) {
               xmm1 += tmp.load(k,i    ,j);
               xmm2 += tmp.load(k,i+1UL,j);
            }
            for( ; remainder && j<N; ++j ) {
               redux += tmp(k,i    ,j);
               redux += tmp(k,i+1UL,j);
            }

            xmm1 += xmm2;

            i += 2UL;
         }

         if( i < M )
         {
            xmm1 += tmp.load(k,i,0UL);
            size_t j( SIMDSIZE );

            for( ; j<jpos; j+=SIMDSIZE ) {
               xmm1 += tmp.load(k,i,j);
            }
            for( ; remainder && j<N; ++j ) {
               redux += tmp(k,i,j);
            }
         }

         redux += sum( xmm1 );
      }
   }
   else
   {
      for( size_t k=0UL; k<O; ++k ) {
         for( size_t i=0UL; i<M; ++i ) {
            for( size_t j=0UL; j<N; ++j ) {
               redux += tmp(k,i,j);
            }
         }
      }
   }

   return redux;

}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Performs a custom reduction operation on the given dense tensor.
// \ingroup dense_tensor
//
// \param dm The given dense tensor for the reduction computation.
// \param op The reduction operation.
// \return The result of the reduction operation.
//
// This function reduces the given dense tensor \a dm by means of the given reduction operation
// \a op:

   \code
   blaze::DynamicTensor<double> A;
   // ... Resizing and initialization

   const double totalsum1 = reduce( A, blaze::Add() );
   const double totalsum2 = reduce( A, []( double a, double b ){ return a + b; } );
   \endcode

// As demonstrated in the example it is possible to pass any binary callable as custom reduction
// operation. However, for instance in the case of lambdas the vectorization of the reduction
// operation is compiler dependent and might not perform at peak performance. However, it is also
// possible to create vectorized custom operations. See \ref custom_operations for a detailed
// overview of the possibilities of custom operations.
//
// Please note that the evaluation order of the reduction operation is unspecified. Thus the
// behavior is non-deterministic if \a op is not associative or not commutative. Also, the
// operation is undefined if the given reduction operation modifies the values.
*/
template< typename MT    // Type of the dense tensor
        , typename OP >  // Type of the reduction operation
inline decltype(auto) reduce( const DenseTensor<MT>& dm, OP op )
{
   BLAZE_FUNCTION_TRACE;

   return dtensreduce( ~dm, op );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation for custom reduction operations on dense matrices.
// \ingroup dense_tensor
//
// \param dm The given dense tensor for the reduction computation.
// \param op The reduction operation.
// \return The result of the reduction operation.
*/
template< size_t RF      // Reduction flag
        , typename MT    // Type of the dense tensor
        , typename OP >  // Type of the reduction operation
inline const DTensReduceExpr<MT,OP,RF> reduce_backend( const DenseTensor<MT>& dm, OP op )
{
   return DTensReduceExpr<MT,OP,RF>( ~dm, op );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Performs a custom reduction operation on the given dense tensor.
// \ingroup dense_tensor
//
// \param dm The given dense tensor for the reduction computation.
// \param op The reduction operation.
// \return The result of the reduction operation.
//
// This function reduces the rows or columns of the given dense tensor \a dm by means of the
// given reduction operation \a op. In case the reduction flag \a RF is set to \a blaze::columnwise,
// the elements of the tensor are reduced column-wise and the result is a row tensor. In case
// \a RF is set to \a blaze::rowwise, the elements of the tensor are reduced row-wise and the
// result is a column tensor:

   \code
   using blaze::columnwise;

   blaze::DynamicTensor<double> A;
   blaze::DynamicMatrix<double,rowMatrix> colsum1, colsum2;
   // ... Resizing and initialization

   colsum1 = reduce<columnwise>( A, blaze::Add() );
   colsum2 = reduce<columnwise>( A, []( double a, double b ){ return a + b; } );
   \endcode

   \code
   using blaze::rowwise;

   blaze::DynamicTensor<double> A;
   blaze::DynamicMatrix<double,columnMatrix> rowsum1, rowsum2;
   // ... Resizing and initialization

   rowsum1 = reduce<rowwise>( A, blaze::Add() );
   rowsum2 = reduce<rowwise>( A, []( double a, double b ){ return a + b; } );
   \endcode

// As demonstrated in the examples it is possible to pass any binary callable as custom reduction
// operation. However, for instance in the case of lambdas the vectorization of the reduction
// operation is compiler dependent and might not perform at peak performance. However, it is also
// possible to create vectorized custom operations. See \ref custom_operations for a detailed
// overview of the possibilities of custom operations.
//
// Please note that the evaluation order of the reduction operation is unspecified. Thus the
// behavior is non-deterministic if \a op is not associative or not commutative. Also, the
// operation is undefined if the given reduction operation modifies the values.
*/
template< size_t RF      // Reduction flag
        , typename MT    // Type of the dense tensor
        , typename OP >  // Type of the reduction operation
inline decltype(auto) reduce( const DenseTensor<MT>& dm, OP op )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_STATIC_ASSERT_MSG( RF < 3UL, "Invalid reduction flag" );

   return reduce_backend<RF>( ~dm, op );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reduces the given dense tensor by means of addition.
// \ingroup dense_tensor
//
// \param dm The given dense tensor for the reduction operation.
// \return The result of the reduction operation.
//
// This function reduces the given dense tensor \a dm by means of addition:

   \code
   blaze::DynamicTensor<int> A{ { 1, 2 }, { 3, 4 } };

   const int totalsum = sum( A );  // Results in 10
   \endcode

// Please note that the evaluation order of the reduction operation is unspecified.
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) sum( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce( ~dm, Add() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reduces the given dense tensor by means of addition.
// \ingroup dense_tensor
//
// \param dm The given dense tensor for the reduction operation.
// \return The result of the reduction operation.
//
// This function reduces the rows or columns of the given dense tensor \a dm by means of
// addition. In case the reduction flag \a RF is set to \a blaze::columnwise, the elements of
// the tensor are reduced column-wise and the result is a row tensor. In case \a RF is set to
// \a blaze::rowwise, the elements of the tensor are reduced row-wise and the result is a
// column tensor:

   \code
   using blaze::columnwise;

   blaze::DynamicTensor<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,rowMatrix> colsum;

   colsum = sum<columnwise>( A );  // Results in ( 2, 3, 6 )
   \endcode

   \code
   using blaze::rowwise;

   blaze::DynamicTensor<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,columnMatrix> rowsum;

   rowsum = sum<rowwise>( A );  // Results in ( 3, 8 )
   \endcode

// Please note that the evaluation order of the reduction operation is unspecified.
*/
template< size_t RF     // Reduction flag
        , typename MT > // Type of the dense tensor
inline decltype(auto) sum( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce<RF>( ~dm, Add() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reduces the given dense tensor by means of multiplication.
// \ingroup dense_tensor
//
// \param dm The given dense tensor for the reduction operation.
// \return The result of the reduction operation.
//
// This function reduces the given dense tensor \a dm by means of multiplication:

   \code
   blaze::DynamicTensor<int> A{ { 1, 2 }, { 3, 4 } };

   const int totalprod = prod( A );  // Results in 24
   \endcode

// Please note that the evaluation order of the reduction operation is unspecified.
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) prod( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce( ~dm, Mult() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reduces the given dense tensor by means of multiplication.
// \ingroup dense_tensor
//
// \param dm The given dense tensor for the reduction operation.
// \return The result of the reduction operation.
//
// This function reduces the rows or columns of the given dense tensor \a dm by means of
// multiplication. In case the reduction flag \a RF is set to \a blaze::columnwise, the elements
// of the tensor are reduced column-wise and the result is a row tensor. In case \a RF is set to
// \a blaze::rowwise, the elements of the tensor are reduced row-wise and the result is a column
// tensor:

   \code
   using blaze::columnwise;

   blaze::DynamicTensor<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,rowMatrix> colprod;

   colprod = prod<columnwise>( A );  // Results in ( 1, 0, 8 )
   \endcode

   \code
   using blaze::rowwise;

   blaze::DynamicTensor<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,columnMatrix> rowprod;

   rowprod = prod<rowwise>( A );  // Results in ( 0, 12 )
   \endcode

// Please note that the evaluation order of the reduction operation is unspecified.
*/
template< size_t RF    // Reduction flag
        , typename MT > // Type of the dense tensor
inline decltype(auto) prod( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce<RF>( ~dm, Mult() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the smallest element of the dense tensor.
// \ingroup dense_tensor
//
// \param dm The given dense tensor.
// \return The smallest dense tensor element.
//
// This function returns the smallest element of the given dense tensor. This function can only
// be used for element types that support the smaller-than relationship. In case the given tensor
// currently has either 0 rows or 0 columns, the returned value is the default value (e.g. 0 in
// case of fundamental data types).

   \code
   blaze::DynamicTensor<int> A{ { 1, 2 }, { 3, 4 } };

   const int totalmin = min( A );  // Results in 1
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) min( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce( ~dm, Min() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the smallest element of each row/columns of the dense tensor.
// \ingroup dense_tensor
//
// \param dm The given dense tensor.
// \return The smallest elements in each row/column.
//
// This function returns the smallest element of each row/column of the given dense tensor \a dm.
// In case the reduction flag \a RF is set to \a blaze::columnwise, a row tensor containing the
// smallest element of each column is returned. In case \a RF is set to \a blaze::rowwise, a
// column tensor containing the smallest element of each row is returned.

   \code
   using blaze::columnwise;

   blaze::DynamicTensor<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,rowMatrix> colmin;

   colmin = min<columnwise>( A );  // Results in ( 1, 0, 2 )
   \endcode

   \code
   using blaze::rowwise;

   blaze::DynamicTensor<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,columnMatrix> rowmin;

   rowmin = min<rowwise>( A );  // Results in ( 0, 1 )
   \endcode
*/
template< size_t RF    // Reduction flag
        , typename MT > // Type of the dense tensor
inline decltype(auto) min( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce<RF>( ~dm, Min() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the largest element of the dense tensor.
// \ingroup dense_tensor
//
// \param dm The given dense tensor.
// \return The largest dense tensor element.
//
// This function returns the largest element of the given dense tensor. This function can only
// be used for element types that support the smaller-than relationship. In case the given martix
// currently has either 0 rows or 0 columns, the returned value is the default value (e.g. 0 in
// case of fundamental data types).

   \code
   blaze::DynamicTensor<int> A{ { 1, 2 }, { 3, 4 } };

   const int totalmax = max( A );  // Results in 4
   \endcode
*/
template< typename MT > // Type of the dense tensor
inline decltype(auto) max( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce( ~dm, Max() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the largest element of each row/columns of the dense tensor.
// \ingroup dense_tensor
//
// \param dm The given dense tensor.
// \return The largest elements in each row/column.
//
// This function returns the largest element of each row/column of the given dense tensor \a dm.
// In case the reduction flag \a RF is set to \a blaze::columnwise, a row tensor containing the
// largest element of each column is returned. In case \a RF is set to \a blaze::rowwise, a
// column tensor containing the largest element of each row is returned.

   \code
   using blaze::columnwise;

   blaze::DynamicTensor<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,rowMatrix> colmax;

   colmax = max<columnwise>( A );  // Results in ( 1, 3, 4 )
   \endcode

   \code
   using blaze::rowwise;

   blaze::DynamicTensor<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,columnMatrix> rowmax;

   rowmax = max<rowwise>( A );  // Results in ( 2, 4 )
   \endcode
*/
template< size_t RF    // Reduction flag
        , typename MT > // Type of the dense tensor
inline decltype(auto) max( const DenseTensor<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce<RF>( ~dm, Max() );
}
//*************************************************************************************************

} // namespace blaze

#endif
