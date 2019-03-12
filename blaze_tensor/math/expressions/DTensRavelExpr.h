//=================================================================================================
/*!
//  \file blaze_tensor/math/expressions/DTensRavelExpr.h
//  \brief Header file for the dense tensor ravel expression
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSRAVELEXPR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSRAVELEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/Transformation.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/simd/SIMDTrait.h>
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

#include <blaze_tensor/math/expressions/TensRavelExpr.h>
#include <blaze_tensor/math/traits/RavelTrait.h>

#include <cstdlib>

namespace blaze {

//=================================================================================================
//
//  CLASS DTensRavelEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for dense tensor ravel.
// \ingroup dense_tensor_expression
//
// The DTensRavelExpr class represents the compile time expression for ravels of
// dense matrices.
*/
template< typename TT >     // Tensor base type of the expression
class DTensRavelExpr
   : public TensRavelExpr< DenseVector< DTensRavelExpr<TT>, rowVector > >
   , private Transformation
{
 private:
   //**Type definitions****************************************************************************
   using CT = CompositeType_t<TT>;  //!< Composite type of the dense tensor expression.

   //! Definition of the Get[Const]Iterator type trait.
   BLAZE_CREATE_GET_TYPE_MEMBER_TYPE_TRAIT( GetIterator, Iterator, INVALID_TYPE );
   BLAZE_CREATE_GET_TYPE_MEMBER_TYPE_TRAIT( GetConstIterator, ConstIterator, INVALID_TYPE );
   //**********************************************************************************************

   //**Transpose flag of the produced result*******************************************************
   static constexpr bool TF = rowVector;

   //**Serial evaluation strategy******************************************************************
   //! Compilation switch for the serial evaluation strategy of the ravel expression.
   /*! The \a useAssign compile time constant expression represents a compilation switch for
       the serial evaluation strategy of the ravel expression. In case the dense tensor
       operand requires an intermediate evaluation, \a useAssign will be set to 1 and the
       ravel expression will be evaluated via the \a assign function family. Otherwise
       \a useAssign will be set to 0 and the expression will be evaluated via the function
       call operator. */
   static constexpr bool useAssign = RequiresEvaluation_v<TT>;

   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename TT1 >
   static constexpr bool UseAssign_v = useAssign;
   /*! \endcond */
   //**********************************************************************************************

   //**Parallel evaluation strategy****************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! This variable template is a helper for the selection of the parallel evaluation strategy.
       In case the tensor operand is not SMP assignable and requires an intermediate evaluation,
       the variable is set to 1 and the expression specific evaluation strategy is selected.
       Otherwise the variable is set to 0 and the default strategy is chosen. */
   template< typename TT1 >
   static constexpr bool UseSMPAssign_v = ( !TT1::smpAssignable && useAssign );
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This          = DTensRavelExpr<TT>;              //!< Type of this DTensRavelExpr instance.
   using BaseType      = DenseVector<This,TF>;           //!< Base type of this DTensRavelExpr instance.
   using ResultType    = RavelTrait_t<TT>;               //!< Result type for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;    //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<TT>;              //!< Resulting element type.
   using ReturnType    = ReturnType_t<TT>;               //!< Return type for expression template evaluations.

   //! Data type for composite expression templates.
   using CompositeType = If_t< useAssign, const ResultType, const DTensRavelExpr& >;

   using Reference      = ElementType&;        //!< Reference to a non-constant tensor value.
   using ConstReference = const ElementType&;  //!< Reference to a constant tensor value.
   using Pointer        = ElementType*;        //!< Pointer to a non-constant tensor value.
   using ConstPointer   = const ElementType*;  //!< Pointer to a constant tensor value.

   //! Iterator over the elements of the dense vector.
   //**RavelIterator class definition**************************************************************
   /*!\brief Iterator over the elements of the dense tensor map expression.
   */
   template< typename TensorType      // Type of the dense tensor
           , typename IteratorType >  // Type of the dense tensor iterator
   class RavelIterator
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
      /*!\brief Constructor for the RavelIterator class.
      //
      // \param it Iterator to the initial tensor element.
      // \param op The custom unary operation.
      */
      explicit inline RavelIterator( TensorType& tensor, size_t pos )
         : tensor_( &tensor )     //!< The dense tensor being ravel-ed
         , page_  ( ( pos / tensor_->columns() ) / tensor_->rows() )    //!< The current page index
         , row_   ( ( pos / tensor_->columns() ) % tensor_->rows() )    //!< The current row index
         , column_( ( pos % tensor_->columns() ) )                      //!< The current column index
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline RavelIterator& operator+=( size_t inc ) {
         column_ += inc;
         if ( column_ >= tensor_->columns() )
         {
            column_ -= tensor_->columns();
            ++row_;
            if ( row_ >= tensor_->rows() )
            {
               ++page_;
               row_ = 0;
            }
         }
         return *this;
      }
      //*******************************************************************************************

      //**Subtraction assignment operator**********************************************************
      /*!\brief Subtraction assignment operator.
      //
      // \param dec The decrement of the iterator.
      // \return The decremented iterator.
      */
      inline RavelIterator& operator-=( size_t dec ) {
         if ( column_ < dec )
         {
            column_ += tensor_->columns() - dec;
            if ( row_ == 0 )
            {
               --page_;
               row_ = tensor_->rows() - 1;
            }
            else
            {
               --row_;
            }
         }
         else
         {
            column_ -= dec;
         }
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline RavelIterator& operator++() {
         ++column_;
         if ( column_ >= tensor_->columns() )
         {
            column_ -= tensor_->columns();
            ++row_;
            if ( row_ >= tensor_->rows() )
            {
               ++page_;
               row_ = 0;
            }
         }
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const RavelIterator operator++( int ) {
         return RavelIterator( *tensor_, position() + 1 );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline RavelIterator& operator--() {
         if ( column_ == 0 )
         {
            column_ = tensor_->columns() - 1;
            if ( row_ == 0 )
            {
               --page_;
               row_ = tensor_->rows() - 1;
            }
            else
            {
               --row_;
            }
         }
         else
         {
            --column_;
         }
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const RavelIterator operator--( int ) {
         return RavelIterator( *tensor_, position() - 1 );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReturnType operator*() const {
         return (*tensor_)( page_, row_, column_ );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two RavelIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const RavelIterator& rhs ) const {
         return tensor_ == rhs.tensor_ && page_ == rhs.page_ && row_ == rhs.row_ && column_ == rhs.column_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two RavelIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const RavelIterator& rhs ) const {
         return tensor_ != rhs.tensor_ || page_ != rhs.page_ || row_ != rhs.row_ || column_ != rhs.column_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two RavelIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const RavelIterator& rhs ) const {
         return position() < rhs.position();
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two RavelIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const RavelIterator& rhs ) const {
         return position() > rhs.position();
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two RavelIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const RavelIterator& rhs ) const {
         return position() <= rhs.position();
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two RavelIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const RavelIterator& rhs ) const {
         return position() >= rhs.position();
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const RavelIterator& rhs ) const {
         return position() - rhs.position();
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between a RavelIterator and an integral value.
      //
      // \param it The iterator to be incremented.
      // \param inc The number of elements the iterator is incremented.
      // \return The incremented iterator.
      */
      friend inline const RavelIterator operator+( const RavelIterator& it, size_t inc ) {
         return RavelIterator( *it.tensor_, it.position() + inc );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between an integral value and a RavelIterator.
      //
      // \param inc The number of elements the iterator is incremented.
      // \param it The iterator to be incremented.
      // \return The incremented iterator.
      */
      friend inline const RavelIterator operator+( size_t inc, const RavelIterator& it ) {
         return RavelIterator( *it.tensor_, it.position() + inc );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Subtraction between a RavelIterator and an integral value.
      //
      // \param it The iterator to be decremented.
      // \param dec The number of elements the iterator is decremented.
      // \return The decremented iterator.
      */
      friend inline const RavelIterator operator-( const RavelIterator& it, size_t dec ) {
         return RavelIterator( *it.tensor_, it.pos_ - dec );
      }
      //*******************************************************************************************

    private:
      size_t position() const
      {
         return ( page_ * tensor_->rows() + row_ ) * tensor_->columns() + column_;
      }

      //**Member variables*************************************************************************
      TensorType*  tensor_;
      size_t       page_;     //!< The current page index.
      size_t       row_;      //!< The current row index.
      size_t       column_;   //!< The current column index.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //! Iterator over constant elements.
   using ConstIterator = RavelIterator< const TT, GetConstIterator_t<TT> >;

   //! Iterator over non-constant elements.
   using Iterator = If_t< IsConst_v<TT>, ConstIterator, RavelIterator< TT, GetIterator_t<TT> > >;
   //**********************************************************************************************

   //! Composite data type of the dense tensor expression.
   using Operand = If_t< IsExpression_v<TT>, const TT, const TT& >;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = false;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = TT::smpAssignable;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DTensRavelExpr class.
   //
   // \param dm The dense tensor operand of the ravel expression.
   // \param args The runtime ravel expression arguments.
   */
   explicit inline DTensRavelExpr( const TT& dm ) noexcept
      : dm_     ( dm )       // Dense tensor of the ravel expression
   {}
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 1D-access to the vector elements.
   //
   // \param index Access index for the elements.
   // \return The resulting value.
   */
   inline ReturnType operator[]( size_t index ) const {

      BLAZE_INTERNAL_ASSERT( index < size(), "Invalid access index");

//       if (TF == blaze::columnVector)
//       {
//          auto div = std::div(index, dm_.rows());
//          return dm_(div.rem, div.quot);
//       }

      auto divc = std::div(int64_t(index), int64_t(dm_.columns()));
      auto divr = std::div(divc.quot, int64_t(dm_.rows()));
      return dm_( divr.quot, divr.rem, divc.rem );
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
   inline ReturnType at( size_t index ) const {
      if( index >= size() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid access index" );
      }
      return (*this)[index];
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first element of the underlying tensor.
   //
   // \return Iterator to the first element of the underlying tensor.
   */
   inline ConstIterator begin( ) const {
      return ConstIterator( dm_, 0 );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last element of the underlying tensor.
   //
   // \return Iterator just past the last element of the underlying tensor.
   */
   inline ConstIterator end( ) const {
      return ConstIterator( dm_, size() );
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first element of the underlying tensor.
   //
   // \return Iterator to the first element of the underlying tensor.
   */
   inline ConstIterator cbegin( ) const {
      return ConstIterator( dm_, 0 );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last element of the underlying tensor.
   //
   // \return Iterator just past the last element of the underlying tensor.
   */
   inline ConstIterator cend( ) const {
      return ConstIterator( dm_, size() );
   }
   //**********************************************************************************************

   //**Size function*******************************************************************************
   /*!\brief Returns the current number of elements of the generated vector.
   //
   // \return The number of elements of the generated vector.
   */
   inline size_t size() const noexcept {
      return dm_.pages() * dm_.rows() * dm_.columns();
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
   Operand dm_;  //!< Dense tensor of the ravel expression.
   //**********************************************************************************************

   //**Assignment to vectors**********************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense tensor ravel expression to a vector.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side vector.
   // \param rhs The right-hand side ravel expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense tensor ravel
   // expression to a vector. Due to the explicit application of the SFINAE principle, this
   // function can only be selected by the compiler in case the operand requires an intermediate
   // evaluation.
   */
   template< typename VT1  // Type of the target tensor
           , bool TF1 >    // Transpose flag
   friend inline EnableIf_t< UseAssign_v<VT1> >
      assign( Vector<VT1,TF1>& lhs, const DTensRavelExpr& rhs )
   {
      using blaze::ravel;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid number of elements");

      CT tmp( serial( ~rhs.dm_ ) );

      assign( ~lhs, ravel( tmp ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to vector*s**********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a dense tensor ravel expression to a vector.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side vector.
   // \param rhs The right-hand side ravel expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a dense tensor
   // ravel expression to a vector. Due to the explicit application of the SFINAE principle,
   // this function can only be selected by the compiler in case the operand requires an
   // intermediate evaluation.
   */
   template< typename VT1  // Type of the target tensor
           , bool TF1 >    // Transpose flag
   friend inline EnableIf_t< UseAssign_v<VT1> >
      subAssign( Vector<VT1,TF1>& lhs, const DTensRavelExpr& rhs )
   {
      using blaze::ravel;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid number of elements");

      CT tmp( serial( ~rhs.dm_ ) );

      subAssign( ~lhs, ravel( tmp ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Schur product assignment to vectors*********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Schur product assignment of a dense tensor ravel expression to a tensor.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side vector.
   // \param rhs The right-hand side ravel expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized Schur product assignment of a dense
   // tensor ravel expression to a vector. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the operand requires
   // an intermediate evaluation.
   */
   template< typename VT1  // Type of the target tensor
           , bool TF1 >    // Transpose flag
   friend inline EnableIf_t< UseAssign_v<VT1> >
      schurAssign( Vector<VT1,TF1>& lhs, const DTensRavelExpr& rhs )
   {
      using blaze::ravel;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid number of elements");

      CT tmp( serial( ~rhs.dm_ ) );

      schurAssign( ~lhs, ravel( tmp ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Multiplication assignment to vectors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Multiplication assignment of a dense tensor ravel expression to a vector.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side vector.
   // \param rhs The right-hand side ravel expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a dense
   // tensor ravel expression to a vector. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the operand requires
   // an intermediate evaluation.
   */
   template< typename VT1  // Type of the target tensor
           , bool TF1 >    // Transpose flag
   friend inline EnableIf_t< UseAssign_v<VT1> >
      multAssign( Vector<VT1,TF1>& lhs, const DTensRavelExpr& rhs )
   {
      using blaze::ravel;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid number of elements");

      CT tmp( serial( ~rhs.dm_ ) );

      multAssign( ~lhs, ravel( tmp ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to vectors*******************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense tensor ravel expression to a vector.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side vector.
   // \param rhs The right-hand side ravel ravel to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense tensor
   // ravel ravel to a vector. Due to the explicit application of the SFINAE principle,
   // this function can only be selected by the compiler in case the expression specific parallel
   // evaluation strategy is selected.
   */
   template< typename VT1  // Type of the target tensor
           , bool TF1 >    // Transpose flag
   friend inline EnableIf_t< UseSMPAssign_v<VT1> >
      smpAssign( Vector<VT1,TF1>& lhs, const DTensRavelExpr& rhs )
   {
      using blaze::ravel;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid number of elements");

      CT tmp( ~rhs.dm_ );

      smpAssign( ~lhs, ravel( tmp ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to vectors**********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a dense tensor ravel expression to a vector.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side vector.
   // \param rhs The right-hand side ravel ravel to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a dense
   // tensor ravel ravel to a vector. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename VT1  // Type of the target tensor
           , bool TF1 >    // Transpose flag
   friend inline EnableIf_t< UseSMPAssign_v<VT1> >
      smpAddAssign( Vector<VT1,TF1>& lhs, const DTensRavelExpr& rhs )
   {
      using blaze::ravel;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid number of elements");

      CT tmp( ~rhs.dm_ );

      smpAddAssign( ~lhs, ravel( tmp ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to vectors*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a dense tensor ravel expression to a vector.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side vector.
   // \param rhs The right-hand side ravel ravel to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a dense
   // tensor ravel ravel to a vector. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename VT1  // Type of the target tensor
           , bool TF1 >    // Transpose flag
   friend inline EnableIf_t< UseSMPAssign_v<VT1> >
      smpSubAssign( Vector<VT1,TF1>& lhs, const DTensRavelExpr& rhs )
   {
      using blaze::ravel;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid number of elements");

      CT tmp( ~rhs.dm_ );

      smpSubAssign( ~lhs, ravel( tmp ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP Schur product assignment to vectors*****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP Schur product assignment of a dense tensor ravel expression to a vector.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side vector.
   // \param rhs The right-hand side ravel ravel for the Schur product.
   // \return void
   //
   // This function implements the performance optimized SMP Schur product assignment of a dense
   // tensor ravel ravel to a vector. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename VT1  // Type of the target tensor
           , bool TF1 >    // Transpose flag
   friend inline EnableIf_t< UseSMPAssign_v<VT1> >
      smpSchurAssign( Vector<VT1,TF1>& lhs, const DTensRavelExpr& rhs )
   {
      using blaze::ravel;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid number of elements");

      CT tmp( ~rhs.dm_ );

      smpSchurAssign( ~lhs, ravel( tmp ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP multiplication assignment to vectors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP multiplication assignment of a dense tensor ravel expression to a vector.
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side vector.
   // \param rhs The right-hand side ravel expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized SMP multiplication assignment of a dense
   // tensor ravel expression to a vector. Due to the explicit application of the SFINAE
   // principle, this function can only be selected by the compiler in case the expression
   // specific parallel evaluation strategy is selected.
   */
   template< typename VT1  // Type of the target tensor
           , bool TF1 >    // Transpose flag
   friend inline EnableIf_t< UseSMPAssign_v<VT1> >
      smpMultAssign( Vector<VT1,TF1>& lhs, const DTensRavelExpr& rhs )
   {
      using blaze::ravel;

      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).size() == rhs.size(), "Invalid number of elements");

      CT tmp( ~rhs.dm_ );

      smpMultAssign( ~lhs, ravel( tmp ) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( TT );
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
/*!\brief Ravel of the given dense tensor.
// \ingroup dense_tensor
//
// \param dm The dense tensor to be raveled.
// \param ravel The ravel.
// \return The ravel of the tensor.
//
// This function returns an expression representing the ravel of the given dense tensor:

   \code
   using blaze::rowMajor;
   using blaze::columnMajor;

   blaze::DynamicTensor<int> a{ {1, 5}, {-2, 4} }
   blaze::DynamicTensor<int> b{ {3, -1}, {7, 0} }

   blaze::DynamicVector<double,columnVector> A;
   blaze::DynamicVector<double,rowVector> B;
   // ... Resizing and initialization

   // Ravel of the column tensor 'a' to 4x1 column vector
   //
   //    (  1 -2  5  4 )
   //
   A = trans( ravel( a ) );

   // Ravel of the row tensor 'b' to a 1x4 row vector
   //
   //    ( 3 -1  7  0 )
   //
   B = ravel( b );
   \endcode
*/
template< typename TT > // Type of the target tensor
inline decltype(auto) ravel( const DenseTensor<TT>& dm )
{
   BLAZE_FUNCTION_TRACE;

   using ReturnType = const DTensRavelExpr<TT>;
   return ReturnType( ~dm );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Creating a view on a specific subvector of the given tensor ravel operation.
// \ingroup subvector
//
// \param vector The constant vector transpose operation.
// \param args The runtime subvector arguments.
// \return View on the specified subvector of the ravel operation.
//
// This function returns an expression representing the specified subvector of the given tensor
// ravel operation.
*/
template< AlignmentFlag AF    // Alignment flag
        , size_t... CSAs      // Compile time subvector arguments
        , typename TT         // Matrix base type of the expression
        , typename... RSAs >  // Runtime subvector arguments
inline decltype(auto) subvector( const DTensRavelExpr<TT>& tensor, RSAs... args )
{
   BLAZE_FUNCTION_TRACE;

   return subvector<AF,CSAs...>( evaluate( ~tensor ), args... );
}
/*! \endcond */
//*************************************************************************************************


} // namespace blaze

#endif
