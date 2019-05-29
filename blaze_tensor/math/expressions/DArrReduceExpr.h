//=================================================================================================
/*!
//  \file blaze/math/expressions/DArrReduceExpr.h
//  \brief Header file for the dense array reduce expression
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DARRREDUCEEXPR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DARRREDUCEEXPR_H_


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
#include <blaze/util/typetraits/RemoveCV.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/ReductionFlag.h>
#include <blaze_tensor/math/constraints/DenseArray.h>
#include <blaze_tensor/math/constraints/Array.h>
#include <blaze_tensor/math/expressions/DenseArray.h>
#include <blaze_tensor/math/expressions/ArrReduceExpr.h>
#include <blaze_tensor/util/ArrayForEach.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Base template for dense array reduction operations.
// \ingroup dense_array_expression
//
// The ReducedArray class represents the compile time expression for partial reduction operations
// of dense matrices.
*/
template< typename MT  // Type of the dense array
        , typename OP  // Type of the reduction operation
        , size_t RF >  // Reduction flag
class ReducedArray;
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ARBITRARY REDUCTION OPERATIONS OF ARRAYS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for arbitrary dense array reduction operations.
// \ingroup dense_array_expression
//
// This specialization of the ReducedArray class template represents the compile time expression
// for row-wise reduction operations of dense matrices.
*/
template< typename MT    // Type of the dense array
        , typename OP    // Type of the reduction operation
        , size_t R >     // DImension along which to perform reduction
class ReducedArray
   : public ArrReduceExpr< DenseArray< ReducedArray<MT,OP,R> >, R >
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   using RT = ResultType_t<MT>;   //!< Result type of the dense array expression.
   using ET = ElementType_t<MT>;  //!< Element type of the dense array expression.
   //**********************************************************************************************

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the reduction expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the serial evaluation strategy of the reduction expression. In case the dense array
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
       In case the dense array operand is not SMP assignable and requires an intermediate
       evaluation, the variable is set to 1 and the expression specific evaluation strategy
       is selected. Otherwise the variable is set to 0 and the default strategy is chosen. */
   template< typename VT >
   static constexpr bool UseSMPAssign_v = ( !MT::smpAssignable && useAssign );
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This          = ReducedArray<MT,OP,R>;          //!< Type of this ReducedArray instance.
   using ResultType    = ReduceTrait_t<RT,OP,R>;         //!< Result type for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;    //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<ResultType>;      //!< Resulting element type.
   using SIMDType      = SIMDTrait_t<ElementType>;       //!< Resulting SIMD element type.
   using ReturnType    = const ElementType;              //!< Return type for expression template evaluations.

   //! Data type for composite expression templates.
   using CompositeType = If_t< useAssign, const ResultType, const ReducedArray& >;

   //! Composite type of the left-hand side dense array expression.
   using Operand = If_t< IsExpression_v<MT>, const MT, const MT& >;

   //! Data type of the custom unary operation.
   using Operation = OP;
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
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the ConstIterator class.
      //
      // \param dm The dense array operand of the reduction expression.
      // \param i Index to the initial array page.
      // \param i Index to the array row.
      // \param op The reduction operation.
      */
      explicit inline ConstIterator( Operand dm, size_t k, OP op )
         : dm_   ( dm    )  // Dense array of the reduction expression
         , k_    ( k )      // Index to the current array page
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
         return ConstIterator( k_++ );
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
         return ConstIterator( k_-- );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReturnType operator*() const {
         return reduce( arrayslice< R >( dm_, k_, unchecked ), op_ );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const ConstIterator& rhs ) const {
         return k_ == rhs.k_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const ConstIterator& rhs ) const {
         return k_ != rhs.k_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two ConstIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const ConstIterator& rhs ) const {
         return k_ < rhs.k_;
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
         return k_ <= rhs.k_;
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
         return ConstIterator( it.k_ + inc );
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
         return ConstIterator( it.k_ + inc );
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
         return ConstIterator( it.k_ - dec );
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      Operand dm_;     //!< Dense array of the reduction expression.
      size_t  k_;      //!< Index to the current array page.
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
   /*!\brief Constructor for the ReducedArray class.
   //
   // \param dm The array operand of the reduction expression.
   // \param op The reduction operation.
   */
   explicit inline ReducedArray( const MT& dm, OP op ) noexcept
      : dm_( dm )  // Dense array of the reduction expression
      , op_( op )  // The reduction operation
   {}
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the array elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   template< typename... Dims >
   inline ReturnType operator()( Dims... dims ) const {
      return reduce( arrayslice<R>( dm_, dims..., unchecked ), op_ );
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the array elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid array access index.
   */
   template< typename... Dims >
   inline ReturnType at( Dims... dims ) const {
      constexpr size_t indices[] = {dims...};

      ArrayDimForEach( dm_.dimensions(), [&]( size_t i ) {
         if( indices[i] >= dm_.dimensions()[i] ) {
            BLAZE_THROW_OUT_OF_RANGE( "Invalid array access index" );
         }
      } );
      return (*this)(dims...);
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first element of the dense array.
   //
   // \return Iterator to the first element of the dense array.
   */
   inline ConstIterator begin( size_t i ) const {
      return ConstIterator( dm_, i, op_ );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of the dense array.
   //
   // \return Iterator just past the last non-zero element of the dense array.
   */
   inline ConstIterator end( size_t i ) const {
      return ConstIterator( dm_, i, op_ );
   }
   //**********************************************************************************************

   //**Num_dimensions function*******************************************************************************
   /*!\brief Returns the current number of dimensions of the array.
   //
   // \return The size of the array.
   */
   inline static constexpr size_t num_dimensions() noexcept {
      return RemoveCV_t<RemoveReference_t<Operand>>::num_dimensions();
   }
   //**********************************************************************************************

   //**Dimensions function****************************************************************************
   /*!\brief Returns the current dimensions of the array.
   //
   // \return The dimensions of the array.
   */
   inline decltype(auto) dimensions() const noexcept {
      return dm_.dimensions();
   }
   //**********************************************************************************************

   //**Pages function*******************************************************************************
   /*!\brief Returns the current size/dimension of the array.
   //
   // \return The size of the array.
   */
   template< size_t Dim >
   inline size_t dimension() const noexcept {
      return dm_.template dimension<Dim>();
   }
   //**********************************************************************************************

   //**Operand access******************************************************************************
   /*!\brief Returns the dense array operand.
   //
   // \return The dense array operand.
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
      return dm_.canSMPAssign() || ( dimensions<1>() * dimensions<0>() > SMP_DMATREDUCE_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   Operand   dm_;  //!< Dense array of the reduction expression.
   Operation op_;  //!< The reduction operation.
   //**********************************************************************************************

   //**Assignment to tensors***********************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a row-wise dense array reduction operation to a array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side array.
   // \param rhs The right-hand side reduction expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a row-wise row-major
   // dense array reduction expression to a array. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression specific
   // parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense array
   friend inline auto assign( Array<VT1>& lhs, const ReducedArray& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense array operand
      assign( ~lhs, reduce<R>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to tensors**************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a row-wise dense array reduction operation to a array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side array.
   // \param rhs The right-hand side reduction expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a row-wise
   // dense array reduction expression to a array. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense array
   friend inline auto addAssign( Array<VT1>& lhs, const ReducedArray& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense array operand
      addAssign( ~lhs, reduce<R>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to tensors***********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a row-wise dense array reduction operation
   //        to a array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side array.
   // \param rhs The right-hand side reduction expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a row-wise
   // dense array reduction expression to a array. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense array
   friend inline auto subAssign( Array<VT1>& lhs, const ReducedArray& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense array operand
      subAssign( ~lhs, reduce<R>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Multiplication assignment to tensors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Multiplication assignment of a row-wise dense array reduction operation
   //        to a array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side array.
   // \param rhs The right-hand side reduction expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a row-wise
   // dense array reduction expression to a array. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense array
   friend inline auto multAssign( Array<VT1>& lhs, const ReducedArray& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense array operand
      multAssign( ~lhs, reduce<R>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Division assignment to tensors**************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Division assignment of a row-wise dense array reduction operation to a array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side array.
   // \param rhs The right-hand side reduction expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a row-wise
   // dense array reduction expression to a array. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense array
   friend inline auto divAssign( Array<VT1>& lhs, const ReducedArray& rhs )
      -> EnableIf_t< UseAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const RT tmp( serial( rhs.dm_ ) );  // Evaluation of the dense array operand
      divAssign( ~lhs, reduce<R>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to tensors*******************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a row-wise dense array reduction operation to a array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side array.
   // \param rhs The right-hand side reduction expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a row-wise row-major
   // dense array reduction expression to a array. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression specific
   // parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense array
   friend inline auto smpAssign( Array<VT1>& lhs, const ReducedArray& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense array operand
      smpAssign( ~lhs, reduce<R>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to tensors**********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a row-wise dense array reduction operation
   //        to a array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side array.
   // \param rhs The right-hand side reduction expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a row-wise
   // dense array reduction expression to a array. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense array
   friend inline auto smpAddAssign( Array<VT1>& lhs, const ReducedArray& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense array operand
      smpAddAssign( ~lhs, reduce<R>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to tensors*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a row-wise dense array reduction operation
   //        to a array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side array.
   // \param rhs The right-hand side reduction expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a row-wise
   // dense array reduction expression to a array. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense array
   friend inline auto smpSubAssign( Array<VT1>& lhs, const ReducedArray& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense array operand
      smpSubAssign( ~lhs, reduce<R>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP multiplication assignment to tensors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP multiplication assignment of a row-wise dense array reduction operation
   //        to a array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side array.
   // \param rhs The right-hand side reduction expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a
   // row-wise dense array reduction expression to a array. Due to the explicit
   // application of the SFINAE principle, this function can only be selected by the compiler
   // in case the expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense array
   friend inline auto smpMultAssign( Array<VT1>& lhs, const ReducedArray& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense array operand
      smpMultAssign( ~lhs, reduce<R>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP division assignment to tensors**********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP division assignment of a row-wise dense array reduction operation
   //        to a array.
   // \ingroup dense_array
   //
   // \param lhs The target left-hand side array.
   // \param rhs The right-hand side reduction expression divisor.
   // \return void
   //
   // This function implements the performance optimized SMP division assignment of a row-wise
   // dense array reduction expression to a array. Due to the explicit application
   // of the SFINAE principle, this function can only be selected by the compiler in case the
   // expression specific parallel evaluation strategy is selected.
   */
   template< typename VT1 >  // Type of the target dense array
   friend inline auto smpDivAssign( Array<VT1>& lhs, const ReducedArray& rhs )
      -> EnableIf_t< UseSMPAssign_v<VT1> >
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).dimensions() == rhs.dimensions(), "Invalid number of elements" );

      const RT tmp( rhs.dm_ );  // Evaluation of the dense array operand
      smpDivAssign( ~lhs, reduce<R>( tmp, rhs.op_ ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE( MT );
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
/*!\brief Auxiliary helper struct for the dense array reduction operation.
// \ingroup dense_array
*/
template< typename MT    // Type of the dense array
        , typename OP >  // Type of the reduction operation
struct ArrayHelper
{
   //**Type definitions****************************************************************************
   //! Composite type of the dense array expression.
   using CT = RemoveReference_t< CompositeType_t<MT> >;

   //! Element type of the dense array expression.
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
/*!\brief Default backend implementation of the reduction of a dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the reduction computation.
// \param op The reduction operation.
// \return The result of the reduction operation.
//
// This function implements the performance optimized reduction operation for a dense
// array. Due to the explicit application of the SFINAE principle, this function can only be
// selected by the compiler in case vectorization cannot be applied.
*/
template< typename MT    // Type of the dense array
        , typename OP >  // Type of the reduction operation
inline ElementType_t<MT> darrayreduce( const DenseArray<MT>& dm, OP op )
{
   using CT = CompositeType_t<MT>;
   using ET = ElementType_t<MT>;

   constexpr size_t N =
      RemoveCV_t< RemoveReference_t< decltype( ~dm ) > >::num_dimensions();

   std::array< size_t, N > dims{};

   if( ArrayDimAnyOf( ( ~dm ).dimensions(), []( size_t i ) { return i == 0; } ) ) return ET{};
   if( ArrayDimAllOf( ( ~dm ).dimensions(), []( size_t i ) { return i == 1; } ) ) return ( ~dm )( dims );

   CT tmp( ~dm );

   BLAZE_INTERNAL_ASSERT( tmp.dimensions() == (~dm).dimensions(), "Invalid number of elements" );

   ET redux{};

   ArrayForEachGrouped( ( ~dm ).dimensions(),
      [&]( std::array< size_t, N > const& dims ) {
         redux = op( redux, tmp( dims ) );
      } );

   return redux;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Performs a custom reduction operation on the given dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the reduction computation.
// \param op The reduction operation.
// \return The result of the reduction operation.
//
// This function reduces the given dense array \a dm by means of the given reduction operation
// \a op:

   \code
   blaze::DynamicArray<double> A;
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
template< typename MT    // Type of the dense array
        , typename OP >  // Type of the reduction operation
inline decltype(auto) reduce( const DenseArray<MT>& dm, OP op )
{
   BLAZE_FUNCTION_TRACE;

   return darrayreduce( ~dm, op );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation for custom reduction operations on dense matrices.
// \ingroup dense_array
//
// \param dm The given dense array for the reduction computation.
// \param op The reduction operation.
// \return The result of the reduction operation.
*/
template< size_t RF      // Reduction flag
        , typename MT    // Type of the dense array
        , typename OP >  // Type of the reduction operation
inline const ReducedArray<MT,OP,RF> reduce_backend( const DenseArray<MT>& dm, OP op )
{
   return ReducedArray<MT,OP,RF>( ~dm, op );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Performs a custom reduction operation on the given dense array.
// \ingroup dense_array
//
// \param dm The given dense array for the reduction computation.
// \param op The reduction operation.
// \return The result of the reduction operation.
//
// This function reduces the rows or columns of the given dense array \a dm by means of the
// given reduction operation \a op. In case the reduction flag \a RF is set to \a blaze::columnwise,
// the elements of the array are reduced column-wise and the result is a row array. In case
// \a RF is set to \a blaze::rowwise, the elements of the array are reduced row-wise and the
// result is a column array:

   \code
   using blaze::columnwise;

   blaze::DynamicArray<double> A;
   blaze::DynamicMatrix<double,rowMatrix> colsum1, colsum2;
   // ... Resizing and initialization

   colsum1 = reduce<columnwise>( A, blaze::Add() );
   colsum2 = reduce<columnwise>( A, []( double a, double b ){ return a + b; } );
   \endcode

   \code
   using blaze::rowwise;

   blaze::DynamicArray<double> A;
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
        , typename MT    // Type of the dense array
        , typename OP >  // Type of the reduction operation
inline decltype(auto) reduce( const DenseArray<MT>& dm, OP op )
{
   BLAZE_FUNCTION_TRACE;

   return reduce_backend<RF>( ~dm, op );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reduces the given dense array by means of addition.
// \ingroup dense_array
//
// \param dm The given dense array for the reduction operation.
// \return The result of the reduction operation.
//
// This function reduces the given dense array \a dm by means of addition:

   \code
   blaze::DynamicArray<int> A{ { 1, 2 }, { 3, 4 } };

   const int totalsum = sum( A );  // Results in 10
   \endcode

// Please note that the evaluation order of the reduction operation is unspecified.
*/
template< typename MT > // Type of the dense array
inline decltype(auto) sum( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce( ~dm, Add() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reduces the given dense array by means of addition.
// \ingroup dense_array
//
// \param dm The given dense array for the reduction operation.
// \return The result of the reduction operation.
//
// This function reduces the rows or columns of the given dense array \a dm by means of
// addition. In case the reduction flag \a RF is set to \a blaze::columnwise, the elements of
// the array are reduced column-wise and the result is a row array. In case \a RF is set to
// \a blaze::rowwise, the elements of the array are reduced row-wise and the result is a
// column array:

   \code
   using blaze::columnwise;

   blaze::DynamicArray<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,rowMatrix> colsum;

   colsum = sum<columnwise>( A );  // Results in ( 2, 3, 6 )
   \endcode

   \code
   using blaze::rowwise;

   blaze::DynamicArray<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,columnMatrix> rowsum;

   rowsum = sum<rowwise>( A );  // Results in ( 3, 8 )
   \endcode

// Please note that the evaluation order of the reduction operation is unspecified.
*/
template< size_t RF     // Reduction flag
        , typename MT > // Type of the dense array
inline decltype(auto) sum( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce<RF>( ~dm, Add() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reduces the given dense array by means of multiplication.
// \ingroup dense_array
//
// \param dm The given dense array for the reduction operation.
// \return The result of the reduction operation.
//
// This function reduces the given dense array \a dm by means of multiplication:

   \code
   blaze::DynamicArray<int> A{ { 1, 2 }, { 3, 4 } };

   const int totalprod = prod( A );  // Results in 24
   \endcode

// Please note that the evaluation order of the reduction operation is unspecified.
*/
template< typename MT > // Type of the dense array
inline decltype(auto) prod( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce( ~dm, Mult() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reduces the given dense array by means of multiplication.
// \ingroup dense_array
//
// \param dm The given dense array for the reduction operation.
// \return The result of the reduction operation.
//
// This function reduces the rows or columns of the given dense array \a dm by means of
// multiplication. In case the reduction flag \a RF is set to \a blaze::columnwise, the elements
// of the array are reduced column-wise and the result is a row array. In case \a RF is set to
// \a blaze::rowwise, the elements of the array are reduced row-wise and the result is a column
// array:

   \code
   using blaze::columnwise;

   blaze::DynamicArray<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,rowMatrix> colprod;

   colprod = prod<columnwise>( A );  // Results in ( 1, 0, 8 )
   \endcode

   \code
   using blaze::rowwise;

   blaze::DynamicArray<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,columnMatrix> rowprod;

   rowprod = prod<rowwise>( A );  // Results in ( 0, 12 )
   \endcode

// Please note that the evaluation order of the reduction operation is unspecified.
*/
template< size_t RF    // Reduction flag
        , typename MT > // Type of the dense array
inline decltype(auto) prod( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce<RF>( ~dm, Mult() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the smallest element of the dense array.
// \ingroup dense_array
//
// \param dm The given dense array.
// \return The smallest dense array element.
//
// This function returns the smallest element of the given dense array. This function can only
// be used for element types that support the smaller-than relationship. In case the given array
// currently has either 0 rows or 0 columns, the returned value is the default value (e.g. 0 in
// case of fundamental data types).

   \code
   blaze::DynamicArray<int> A{ { 1, 2 }, { 3, 4 } };

   const int totalmin = min( A );  // Results in 1
   \endcode
*/
template< typename MT > // Type of the dense array
inline decltype(auto) min( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce( ~dm, Min() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the smallest element of each row/columns of the dense array.
// \ingroup dense_array
//
// \param dm The given dense array.
// \return The smallest elements in each row/column.
//
// This function returns the smallest element of each row/column of the given dense array \a dm.
// In case the reduction flag \a RF is set to \a blaze::columnwise, a row array containing the
// smallest element of each column is returned. In case \a RF is set to \a blaze::rowwise, a
// column array containing the smallest element of each row is returned.

   \code
   using blaze::columnwise;

   blaze::DynamicArray<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,rowMatrix> colmin;

   colmin = min<columnwise>( A );  // Results in ( 1, 0, 2 )
   \endcode

   \code
   using blaze::rowwise;

   blaze::DynamicArray<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,columnMatrix> rowmin;

   rowmin = min<rowwise>( A );  // Results in ( 0, 1 )
   \endcode
*/
template< size_t RF    // Reduction flag
        , typename MT > // Type of the dense array
inline decltype(auto) min( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce<RF>( ~dm, Min() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the largest element of the dense array.
// \ingroup dense_array
//
// \param dm The given dense array.
// \return The largest dense array element.
//
// This function returns the largest element of the given dense array. This function can only
// be used for element types that support the smaller-than relationship. In case the given martix
// currently has either 0 rows or 0 columns, the returned value is the default value (e.g. 0 in
// case of fundamental data types).

   \code
   blaze::DynamicArray<int> A{ { 1, 2 }, { 3, 4 } };

   const int totalmax = max( A );  // Results in 4
   \endcode
*/
template< typename MT > // Type of the dense array
inline decltype(auto) max( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce( ~dm, Max() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the largest element of each row/columns of the dense array.
// \ingroup dense_array
//
// \param dm The given dense array.
// \return The largest elements in each row/column.
//
// This function returns the largest element of each row/column of the given dense array \a dm.
// In case the reduction flag \a RF is set to \a blaze::columnwise, a row array containing the
// largest element of each column is returned. In case \a RF is set to \a blaze::rowwise, a
// column array containing the largest element of each row is returned.

   \code
   using blaze::columnwise;

   blaze::DynamicArray<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,rowMatrix> colmax;

   colmax = max<columnwise>( A );  // Results in ( 1, 3, 4 )
   \endcode

   \code
   using blaze::rowwise;

   blaze::DynamicArray<int> A{ { 1, 0, 2 }, { 1, 3, 4 } };
   blaze::DynamicMatrix<int,columnMatrix> rowmax;

   rowmax = max<rowwise>( A );  // Results in ( 2, 4 )
   \endcode
*/
template< size_t RF    // Reduction flag
        , typename MT > // Type of the dense array
inline decltype(auto) max( const DenseArray<MT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return reduce<RF>( ~dm, Max() );
}
//*************************************************************************************************

} // namespace blaze

#endif
