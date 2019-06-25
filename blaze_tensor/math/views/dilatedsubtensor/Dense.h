//=================================================================================================
/*!
//  \file blaze_tensor/math/views/DilatedSubtensor/Dense.h
//  \brief DilatedSubtensor specialization for dense matrices
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018-2019 Hartmut Kaiser - All Rights Reserved
//  Copyright (C) 2019 Bita Hasheminezhad - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBTENSOR_DENSE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBTENSOR_DENSE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <iterator>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/TransExpr.h>
#include <blaze/math/constraints/UniTriangular.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/View.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/StorageOrder.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/SchurTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/HasSIMDSub.h>
#include <blaze/math/typetraits/IsContiguous.h>
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsRestricted.h>
#include <blaze/math/typetraits/IsSIMDCombinable.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsTriangular.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/IsUniUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/math/views/Check.h>
#include <blaze/system/Blocking.h>
#include <blaze/system/CacheSize.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/algorithms/Max.h>
#include <blaze/util/algorithms/Min.h>
#include <blaze/util/AlignmentCheck.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/MaybeUnused.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/TypeList.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsReference.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/DilatedSubtensor.h>
#include <blaze_tensor/math/constraints/RowMajorTensor.h>
#include <blaze_tensor/math/dense/InitializerTensor.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>
#include <blaze_tensor/math/traits/DilatedSubtensorTrait.h>
#include <blaze_tensor/math/views/dilatedsubtensor/BaseTemplate.h>
#include <blaze_tensor/math/views/dilatedsubtensor/DilatedSubtensorData.h>
#include <blaze_tensor/system/Thresholds.h>

namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR UNALIGNED ROW-MAJOR DENSE DILATEDSUBMATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of DilatedSubtensor for unaligned row-major dense dilatedsubmatrices.
// \ingroup DilatedSubtensor
//
// This Specialization of DilatedSubtensor adapts the class template to the requirements of unaligned
// row-major dense dilatedsubmatrices.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
class DilatedSubtensor<TT,true,CSAs...>
   : public View< DenseTensor< DilatedSubtensor<TT,true,CSAs...> > >
   , private DilatedSubtensorData<CSAs...>
{
 private:
   //**Type definitions****************************************************************************
   using DataType = DilatedSubtensorData<CSAs...>;        //!< The type of the DilatedSubtensorData base class.
   using Operand  = If_t< IsExpression_v<TT>, TT, TT& >;  //!< Composite data type of the tensor expression.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename TT1, typename TT2 >
   static constexpr bool EnforceEvaluation_v =
      ( IsRestricted_v<TT1> && RequiresEvaluation_v<TT2> );
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this DilatedSubtensor instance.
   using This = DilatedSubtensor<TT,true,CSAs...>;

   using BaseType      = DenseTensor<This>;                    //!< Base type of this DilatedSubtensor instance.
   using ViewedType    = TT;                                   //!< The type viewed by this DilatedSubtensor instance.
   using ResultType    = DilatedSubtensorTrait_t<TT,CSAs...>;  //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;           //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;          //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<TT>;                    //!< Type of the DilatedSubtensor elements.
   using SIMDType      = SIMDTrait_t<ElementType>;             //!< SIMD type of the DilatedSubtensor elements.
   using ReturnType    = ReturnType_t<TT>;                     //!< Return type for expression template evaluations
   using CompositeType = const DilatedSubtensor&;              //!< Data type for composite expression templates.

   //! Reference to a constant dilatedsubtensor value.
   using ConstReference = ConstReference_t<TT>;

   //! Reference to a non-constant dilatedsubtensor value.
   using Reference = If_t< IsConst_v<TT>, ConstReference, Reference_t<TT> >;

   //! Pointer to a constant dilatedsubtensor value.
   using ConstPointer = ConstPointer_t<TT>;

   //! Pointer to a non-constant dilatedsubtensor value.
   using Pointer = If_t< IsConst_v<TT> || !HasMutableDataAccess_v<TT>, ConstPointer, Pointer_t<TT> >;
   ////**********************************************************************************************

   //**DilatedSubtensorIterator class definition****************************************************
   /*!\brief Iterator over the elements of the sparse DilatedSubtensor.
   */
   template< typename IteratorType >  // Type of the dense tensor iterator
   class DilatedSubtensorIterator
   {
    public:
      //**Type definitions*************************************************************************
      //! The iterator category.
      using IteratorCategory = typename std::iterator_traits<IteratorType>::iterator_category;

      //! Type of the underlying elements.
      using ValueType = typename std::iterator_traits<IteratorType>::value_type;

      //! Pointer return type.
      using PointerType = typename std::iterator_traits<IteratorType>::pointer;

      //! Reference return type.
      using ReferenceType = typename std::iterator_traits<IteratorType>::reference;

      //! Difference between two iterators.
      using DifferenceType = typename std::iterator_traits<IteratorType>::difference_type;

      // STL iterator requirements
      using iterator_category = IteratorCategory;  //!< The iterator category.
      using value_type        = ValueType;         //!< Type of the underlying elements.
      using pointer           = PointerType;       //!< Pointer return type.
      using reference         = ReferenceType;     //!< Reference return type.
      using difference_type   = DifferenceType;    //!< Difference between two iterators.
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Default constructor of the DilatedSubtensorIterator class.
      */
      inline DilatedSubtensorIterator()
         : iterator_ (       )  // Iterator to the current DilatedSubtensor element
         , pagedilation_ ( 1     )   // page step-size of the underlying dilated subtensor
         , rowdilation_  ( 1     )   // row step-size of the underlying dilated subtensor
         , columndilation_ ( 1     ) // column step-size of the underlying dilated subtensor
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor of the DilatedSubtensorIterator class.
      //
      // \param iterator Iterator to the initial element.
      // \param isMemoryAligned Memory alignment flag.
      */
      inline DilatedSubtensorIterator( IteratorType iterator, size_t pagedilation, size_t rowdilation, size_t columndilation)
         : iterator_ ( iterator        )        // Iterator to the current DilatedSubtensor element
         , pagedilation_ ( pagedilation )       // page step-size of the underlying dilated subtensor
         , rowdilation_  ( rowdilation )        // row step-size of the underlying dilated subtensor
         , columndilation_ ( columndilation )   // column step-size of the underlying dilated subtensor
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Conversion constructor from different DilatedSubtensorIterator instances.
      //
      // \param it The DilatedSubtensor iterator to be copied.
      */
      template< typename IteratorType2 >
      inline DilatedSubtensorIterator( const DilatedSubtensorIterator<IteratorType2>& it )
         : iterator_ ( it.base()      )  // Iterator to the current DilatedSubtensor element
         , pagedilation_ ( it.pagedilation() )   // page step-size of the underlying dilated subtensor
         , rowdilation_  ( it.rowdilation() )    // row step-size of the underlying dilated subtensor
         , columndilation_ ( it.columndilation() )   // column step-size of the underlying dilated subtensor
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline DilatedSubtensorIterator& operator+=( size_t inc ) {
         iterator_ += inc * columndilation_;
         return *this;
      }
      //*******************************************************************************************

      //**Subtraction assignment operator**********************************************************
      /*!\brief Subtraction assignment operator.
      //
      // \param dec The decrement of the iterator.
      // \return The decremented iterator.
      */
      inline DilatedSubtensorIterator& operator-=( size_t dec ) {
         iterator_ -= dec * columndilation_;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline DilatedSubtensorIterator& operator++() {
         iterator_+= columndilation_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const DilatedSubtensorIterator operator++( int ) {
         return DilatedSubtensorIterator( iterator_+=columndilation_, pagedilation_, rowdilation_, columndilation_ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline DilatedSubtensorIterator& operator--() {
         iterator_-= columndilation_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const DilatedSubtensorIterator operator--( int ) {
         return DilatedSubtensorIterator( iterator_-=columndilation_, pagedilation_, rowdilation_, columndilation_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReferenceType operator*() const {
         return *iterator_;
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return Pointer to the element at the current iterator position.
      */
      inline IteratorType operator->() const {
         return iterator_;
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two DilatedSubtensorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const DilatedSubtensorIterator& rhs ) const {
         return iterator_ == rhs.iterator_  && pagedilation_ == rhs.pagedilation_ && rowdilation_ == rhs.rowdilation_ &&
            columndilation_ == rhs.columndilation_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two DilatedSubtensorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const DilatedSubtensorIterator& rhs ) const {
         return iterator_ != rhs.iterator_  || pagedilation_ != rhs.pagedilation_ || rowdilation_ != rhs.rowdilation_ ||
            columndilation_ != rhs.columndilation_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two DilatedSubtensorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const DilatedSubtensorIterator& rhs ) const {
         return iterator_ < rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two DilatedSubtensorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const DilatedSubtensorIterator& rhs ) const {
         return iterator_ > rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two DilatedSubtensorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const DilatedSubtensorIterator& rhs ) const {
         return iterator_ <= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two DilatedSubtensorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const DilatedSubtensorIterator& rhs ) const {
         return iterator_ >= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const DilatedSubtensorIterator& rhs ) const {
         return (iterator_ - rhs.iterator_)/ptrdiff_t(columndilation_);
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between a DilatedSubtensorIterator and an integral value.
      //
      // \param it The iterator to be incremented.
      // \param inc The number of elements the iterator is incremented.
      // \return The incremented iterator.
      */
      friend inline const DilatedSubtensorIterator operator+( const DilatedSubtensorIterator& it, size_t inc ) {
         return DilatedSubtensorIterator( it.iterator_ + inc*it.columndilation_, it.pagedilation_, it.rowdilation_, it.columndilation_ );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between an integral value and a DilatedSubtensorIterator.
      //
      // \param inc The number of elements the iterator is incremented.
      // \param it The iterator to be incremented.
      // \return The incremented iterator.
      */
      friend inline const DilatedSubtensorIterator operator+( size_t inc, const DilatedSubtensorIterator& it ) {
         return DilatedSubtensorIterator( it.iterator_ + inc*it.columndilation_, it.pagedilation_, it.rowdilation_, it.columndilation_ );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Subtraction between a DilatedSubtensorIterator and an integral value.
      //
      // \param it The iterator to be decremented.
      // \param dec The number of elements the iterator is decremented.
      // \return The decremented iterator.
      */
      friend inline const DilatedSubtensorIterator operator-( const DilatedSubtensorIterator& it, size_t dec ) {
         return DilatedSubtensorIterator( it.iterator_ - dec*it.columndilation_, it.pagedilation_, it.rowdilation_, it.columndilation_  );
      }
      //*******************************************************************************************

      //**Base function****************************************************************************
      /*!\brief Access to the current position of the DilatedSubtensor iterator.
      //
      // \return The current position of the DilatedSubtensor iterator.
      */
      inline IteratorType base() const {
         return iterator_;
      }
      //*******************************************************************************************

      //**PageDilation function********************************************************************
      /*!\briefAccess to the current rowdilation of the dilatedsubtensor iterator.
      //
      // \return The page dilation of the dilatedsubtensor iterator.
      */
      inline size_t pagedilation() const noexcept {
         return pagedilation_;
      }
      //*******************************************************************************************

      //**RowDilation function*********************************************************************
      /*!\briefAccess to the current rowdilation of the dilatedsubtensor iterator.
      //
      // \return The row dilation of the dilatedsubtensor iterator.
      */
      inline size_t rowdilation() const noexcept {
         return rowdilation_;
      }
      //*******************************************************************************************

      //**ColumnDilation function******************************************************************
      /*!\brief Access to the current columndilation of the dilatedsubtensor iterator.
      //
      // \return The row dilation of the dilatedsubtensor iterator.
      */
      inline size_t columndilation() const noexcept {
         return columndilation_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType iterator_;       //!< Iterator to the current DilatedSubtensor element.
      size_t pagedilation_;         //!< Row step-size of the underlying dilated subtensor
      size_t rowdilation_;          //!< Row step-size of the underlying dilated subtensor
      size_t columndilation_;       //!< Column step-size of the underlying dilated subtensor
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   //! Iterator over constant elements.
   using ConstIterator = DilatedSubtensorIterator< ConstIterator_t<TT> >;

   //! Iterator over non-constant elements.
   using Iterator = If_t< IsConst_v<TT>, ConstIterator, DilatedSubtensorIterator< Iterator_t<TT> > >;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = false;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = TT::smpAssignable;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RSAs >
   explicit inline DilatedSubtensor( TT& tensor, RSAs... args );

   DilatedSubtensor( const DilatedSubtensor& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DilatedSubtensor() = default;
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator()( size_t k, size_t i, size_t j );
   inline ConstReference operator()( size_t k, size_t i, size_t j ) const;
   inline Reference      at( size_t k, size_t i, size_t j );
   inline ConstReference at( size_t k, size_t i, size_t j ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Pointer        data  ( size_t i, size_t k ) noexcept;
   inline ConstPointer   data  ( size_t i, size_t k ) const noexcept;
   inline Iterator       begin ( size_t i, size_t k );
   inline ConstIterator  begin ( size_t i, size_t k ) const;
   inline ConstIterator  cbegin( size_t i, size_t k ) const;
   inline Iterator       end   ( size_t i, size_t k );
   inline ConstIterator  end   ( size_t i, size_t k ) const;
   inline ConstIterator  cend  ( size_t i, size_t k ) const;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline DilatedSubtensor& operator=( const ElementType& rhs );
   inline DilatedSubtensor& operator=( initializer_list< initializer_list< initializer_list<ElementType> > > list );
   inline DilatedSubtensor& operator=( const DilatedSubtensor& rhs );

   template< typename TT2 >
   inline DilatedSubtensor& operator=( const Tensor<TT2>& rhs );

   template< typename TT2 >
   inline auto operator+=( const Tensor<TT2>& rhs )
      -> DisableIf_t< EnforceEvaluation_v<TT,TT2>, DilatedSubtensor& >;

   template< typename TT2 >
   inline auto operator+=( const Tensor<TT2>& rhs )
      -> EnableIf_t< EnforceEvaluation_v<TT,TT2>, DilatedSubtensor& >;

   template< typename TT2 >
   inline auto operator-=( const Tensor<TT2>& rhs )
      -> DisableIf_t< EnforceEvaluation_v<TT,TT2>, DilatedSubtensor& >;

   template< typename TT2 >
   inline auto operator-=( const Tensor<TT2>& rhs )
      -> EnableIf_t< EnforceEvaluation_v<TT,TT2>, DilatedSubtensor& >;

   template< typename TT2 >
   inline auto operator%=( const Tensor<TT2>& rhs )
      -> DisableIf_t< EnforceEvaluation_v<TT,TT2>, DilatedSubtensor& >;

   template< typename TT2 >
   inline auto operator%=( const Tensor<TT2>& rhs )
      -> EnableIf_t< EnforceEvaluation_v<TT,TT2>, DilatedSubtensor& >;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   using DataType::page;
   using DataType::row;
   using DataType::column;
   using DataType::pages;
   using DataType::rows;
   using DataType::columns;
   using DataType::pagedilation;
   using DataType::rowdilation;
   using DataType::columndilation;

   inline TT&       operand() noexcept;
   inline const TT& operand() const noexcept;

   inline size_t spacing() const noexcept;
   inline size_t capacity() const noexcept;
   inline size_t capacity( size_t i, size_t k ) const noexcept;
   inline size_t nonZeros() const;
   inline size_t nonZeros( size_t i, size_t k ) const;
   inline void   reset();
   inline void   reset( size_t i, size_t k );
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   inline DilatedSubtensor& transpose();
   inline DilatedSubtensor& ctranspose();

   template< typename Other > inline DilatedSubtensor& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 public:
   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other >
   inline bool canAlias( const Other* alias ) const noexcept;

   template< typename TT2, size_t... CSAs2 >
   inline bool canAlias( const DilatedSubtensor<TT2,true,CSAs2...>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename TT2, size_t... CSAs2 >
   inline bool isAliased( const DilatedSubtensor<TT2,true,CSAs2...>* alias ) const noexcept;

   inline bool isAligned() const noexcept { return false; }
   inline bool canSMPAssign() const noexcept;


   template< typename TT2 > inline auto assign( const DenseTensor<TT2>& rhs );
   template< typename TT2 > inline auto addAssign( const DenseTensor<TT2>& rhs );
   template< typename TT2 > inline auto subAssign( const DenseTensor<TT2>& rhs );
   template< typename TT2 > inline auto schurAssign( const DenseTensor<TT2>& rhs );


   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline bool hasOverlap() const noexcept;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand tensor_;        //!< The tensor containing the DilatedSubtensor.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename TT2, bool DF2, size_t... CSAs2 > friend class DilatedSubtensor;
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE    ( TT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE ( TT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE   ( TT );
   //BLAZE_CONSTRAINT_MUST_NOT_BE_SUBTENSOR_TYPE   ( TT );
   //BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_TENSOR_TYPE( TT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE     ( TT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE   ( TT );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for unaligned row-major dense submatrices.
//
// \param tensor The dense tensor containing the DilatedSubtensor.
// \param args The runtime DilatedSubtensor arguments.
// \exception std::invalid_argument Invalid DilatedSubtensor specification.
//
// By default, the provided DilatedSubtensor arguments are checked at runtime. In case the DilatedSubtensor is
// not properly specified (i.e. if the specified DilatedSubtensor is not contained in the given dense
// tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped by providing
// the optional \a blaze::unchecked argument.
*/
template< typename TT           // Type of the dense tensor
        , size_t... CSAs >      // Compile time DilatedSubtensor arguments
template< typename... RSAs >    // Runtime DilatedSubtensor arguments
inline DilatedSubtensor<TT,true,CSAs...>::DilatedSubtensor( TT& tensor, RSAs... args )
   : DataType  ( args... )      // Base class initialization
   , tensor_   ( tensor  )      // The tensor containing the DilatedSubtensor
{
   if( !Contains_v< TypeList<RSAs...>, Unchecked > ) {
      if(( page() + ( pages() - 1 ) * pagedilation() + 1 > tensor_.pages() ) ||
         ( row() + ( rows() - 1 ) * rowdilation() + 1 > tensor_.rows() ) ||
         ( column() + ( columns() - 1 ) * columndilation() + 1 >  tensor_.columns() ) )
      {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT(
         page() + ( pages() - 1 ) * pagedilation() + 1 <= tensor_.pages(),
         "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( row() + ( rows() - 1 ) * rowdilation() + 1 <= tensor_.rows(),
         "Invalid dilatedsubtensor specification" );
      BLAZE_USER_ASSERT( column() + ( columns() - 1 ) * columndilation() + 1 <= tensor_.columns(),
         "Invalid dilatedsubtensor specification" );
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the dense DilatedSubtensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::Reference
   DilatedSubtensor<TT,true,CSAs...>::operator()( size_t k, size_t i, size_t j )
{
   BLAZE_USER_ASSERT( k < pages(),   "Invalid page access index"   );
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return tensor_( page() + k * pagedilation(), row() + i * rowdilation(), column() + j * columndilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 3D-access to the dense DilatedSubtensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::ConstReference
   DilatedSubtensor<TT,true,CSAs...>::operator()( size_t k, size_t i, size_t j ) const
{
   BLAZE_USER_ASSERT( k < pages(),   "Invalid page access index"   );
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return const_cast< const TT& >( tensor_ )( page() + k * pagedilation(), row() + i * rowdilation(), column() + j * columndilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the DilatedSubtensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid tensor access index.
//
// In contrast to the function call operator this function always performs a check of the given
// access indices.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::Reference
   DilatedSubtensor<TT,true,CSAs...>::at( size_t k, size_t i, size_t j )
{
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
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the DilatedSubtensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid tensor access index.
//
// In contrast to the function call operator this function always performs a check of the given
// access indices.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::ConstReference
   DilatedSubtensor<TT,true,CSAs...>::at( size_t k, size_t i, size_t j ) const
{
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
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the DilatedSubtensor elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense DilatedSubtensor. Note that
// you can NOT assume that all tensor elements lie adjacent to each other! The dense DilatedSubtensor
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::Pointer
   DilatedSubtensor<TT,true,CSAs...>::data() noexcept
{
   return tensor_.data() + ( page()*tensor_.rows() + row() ) * spacing() + column();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the DilatedSubtensor elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense DilatedSubtensor. Note that
// you can NOT assume that all tensor elements lie adjacent to each other! The dense DilatedSubtensor
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::ConstPointer
   DilatedSubtensor<TT,true,CSAs...>::data() const noexcept
{
   return tensor_.data() + ( page()*tensor_.rows() + row() ) * spacing() + column();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the subtensor elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::Pointer
   DilatedSubtensor<TT,true,CSAs...>::data( size_t i, size_t k ) noexcept
{
   return tensor_.data() + ( ( page()+k ) * tensor_.rows() + ( row()+i ) ) * spacing() + column();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the subtensor elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::ConstPointer
   DilatedSubtensor<TT,true,CSAs...>::data( size_t i, size_t k ) const noexcept
{
   return tensor_.data() + ( ( page()+k ) * tensor_.rows() + ( row()+i ) ) * spacing() + column();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first non-zero element of row/column \a i.
//
// This function returns a row/column iterator to the first non-zero element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator to the first
// non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator to the first non-zero element of column \a i.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::Iterator
   DilatedSubtensor<TT,true,CSAs...>::begin( size_t i, size_t k )
{
   BLAZE_USER_ASSERT( k < pages(), "Invalid dense dilatedsubtensor page access index" );
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense dilatedsubtensor row access index" );
   return Iterator( tensor_.begin( row() + i * rowdilation() , page() + k * pagedilation() ) + column(),
      pagedilation(), rowdilation(), columndilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first non-zero element of row/column \a i.
//
// This function returns a row/column iterator to the first non-zero element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator to the first
// non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator to the first non-zero element of column \a i.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::ConstIterator
   DilatedSubtensor<TT,true,CSAs...>::begin( size_t i, size_t k ) const
{
   BLAZE_USER_ASSERT( k < pages(),"Invalid dense dilatedsubtensor page access index" );
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense dilatedsubtensor row access index" );
   return ConstIterator( tensor_.cbegin( row() + i * rowdilation() , page() + k * pagedilation() ) + column(),
      pagedilation(), rowdilation(), columndilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first non-zero element of row/column \a i.
//
// This function returns a row/column iterator to the first non-zero element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator to the first
// non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator to the first non-zero element of column \a i.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::ConstIterator
   DilatedSubtensor<TT,true,CSAs...>::cbegin( size_t i, size_t k ) const
{
   BLAZE_USER_ASSERT( k < pages(),"Invalid dense dilatedsubtensor page access index" );
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense dilatedsubtensor row access index" );
   return ConstIterator( tensor_.cbegin( row() + i * rowdilation() , page() + k * pagedilation()) + column(),
      pagedilation(), rowdilation(), columndilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last non-zero element of row/column \a i.
//
// This function returns an row/column iterator just past the last non-zero element of row/column
// \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
// past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
// the function returns an iterator just past the last non-zero element of column \a i.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::Iterator
   DilatedSubtensor<TT,true,CSAs...>::end( size_t i, size_t k )
{
   BLAZE_USER_ASSERT( k < pages(),"Invalid dense dilatedsubtensor page access index" );
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense dilatedsubtensor row access index" );
   return Iterator( tensor_.begin( row() + i * rowdilation() , page() + k * pagedilation() ) + column() + columns() * columndilation(),
      pagedilation(), rowdilation(), columndilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last non-zero element of row/column \a i.
//
// This function returns an row/column iterator just past the last non-zero element of row/column
// \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
// past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
// the function returns an iterator just past the last non-zero element of column \a i.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::ConstIterator
   DilatedSubtensor<TT,true,CSAs...>::end( size_t i, size_t k ) const
{
   BLAZE_USER_ASSERT( k < pages(),"Invalid dense dilatedsubtensor page access index" );
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense dilatedsubtensor row access index" );
   return ConstIterator( tensor_.cbegin( row() + i * rowdilation() , page() + k * pagedilation() ) + column() + columns() * columndilation(),
      pagedilation(), rowdilation(), columndilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last non-zero element of row/column \a i.
//
// This function returns an row/column iterator just past the last non-zero element of row/column
// \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
// past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
// the function returns an iterator just past the last non-zero element of column \a i.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline typename DilatedSubtensor<TT,true,CSAs...>::ConstIterator
   DilatedSubtensor<TT,true,CSAs...>::cend( size_t i, size_t k ) const
{
   BLAZE_USER_ASSERT( k < pages(),"Invalid dense dilatedsubtensor page access index" );
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense dilatedsubtensor row access index" );
   return ConstIterator( tensor_.cbegin( row() + i * rowdilation() , page() + k * pagedilation() ) + column() + columns() * columndilation(),
      pagedilation(), rowdilation(), columndilation() );;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Homogenous assignment to all DilatedSubtensor elements.
//
// \param rhs Scalar value to be assigned to all DilatedSubtensor elements.
// \return Reference to the assigned DilatedSubtensor.
//
// This function homogeneously assigns the given value to all dense tensor elements. Note that in
// case the underlying dense tensor is a lower/upper tensor only lower/upper and diagonal elements
// of the underlying tensor are modified.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline DilatedSubtensor<TT,true,CSAs...>&
   DilatedSubtensor<TT,true,CSAs...>::operator=( const ElementType& rhs )
{
   decltype(auto) left( derestrict( tensor_ ) );

   const size_t kend( page() + pages() * pagedilation() );
   for( size_t k=page(); k<kend; k += pagedilation() )
   {
       const size_t iend( row() + rows() * rowdilation() );
       for( size_t i=row(); i<iend; i += rowdilation() )
       {
          const size_t jbegin( column() );
          const size_t jend  ( column() + columns() * columndilation() );

          for( size_t j=jbegin; j<jend; j += columndilation() ) {
             if( !IsRestricted_v<TT> || IsTriangular_v<TT> || trySet( tensor_, i, j, k, rhs ) )
                left(k,i,j) = rhs;
          }
       }
   }
   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all DilatedSubtensor elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid initializer list dimension.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// This assignment operator offers the option to directly assign to all elements of the DilatedSubtensor
// by means of an initializer list. The DilatedSubtensor elements are assigned the values from the given
// initializer list. Missing values are initialized as default. Note that in case the size of the
// top-level initializer list does not match the number of rows of the DilatedSubtensor or the size of
// any nested list exceeds the number of columns, a \a std::invalid_argument exception is thrown.
// Also, if the underlying tensor \a TT is restricted and the assignment would violate an
// invariant of the tensor, a \a std::invalid_argument exception is thrown.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline DilatedSubtensor<TT,true,CSAs...>&
   DilatedSubtensor<TT,true,CSAs...>::operator=( initializer_list< initializer_list<initializer_list<ElementType> > > list )
{
   if( list.size() != pages() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to dilatedsubtensor" );
   }

   if( IsRestricted_v<TT> ) {
      const InitializerTensor<ElementType> tmp( list, rows(), columns() );
      if( !tryAssign( tensor_, tmp, row(), column(), page() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
      }
   }

   decltype(auto) left( derestrict( *this ) );

   size_t k( 0UL );
   for( const auto& colList : list ) {
      size_t i( 0UL );
      for( const auto& rowList : colList ) {
         std::fill( std::copy( rowList.begin(), rowList.end(), left.begin(i, k) ), left.end(i, k), ElementType() );
         ++i;
      }
      ++k;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for DilatedSubtensor.
//
// \param rhs Dense DilatedSubtensor to be copied.
// \return Reference to the assigned DilatedSubtensor.
// \exception std::invalid_argument DilatedSubtensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// The dense DilatedSubtensor is initialized as a copy of the given dense DilatedSubtensor. In case the current
// sizes of the two submatrices don't match, a \a std::invalid_argument exception is thrown. Also,
// if the underlying tensor \a TT is a lower triangular, upper triangular, or symmetric tensor
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline DilatedSubtensor<TT,true,CSAs...>&
   DilatedSubtensor<TT,true,CSAs...>::operator=( const DilatedSubtensor& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( this == &rhs || ( &tensor_ == &rhs.tensor_ && page() == rhs.page() && row() == rhs.row() && column() == rhs.column() &&
      pagedilation() == rhs.pagedilation() && rowdilation() == rhs.rowdilation() && columndilation() == rhs.columndilation() ) )
      return *this;

   if( pages() != rhs.pages() || rows() != rhs.rows() || columns() != rhs.columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "DilatedSubtensor sizes do not match" );
   }

   if( !tryAssign( tensor_, rhs, row(), column(), page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( rhs.canAlias( &tensor_ ) ) {
      const ResultType tmp( rhs );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( tensor_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different matrices.
//
// \param rhs Tensor to be assigned.
// \return Reference to the assigned DilatedSubtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// The dense DilatedSubtensor is initialized as a copy of the given tensor. In case the current sizes
// of the two matrices don't match, a \a std::invalid_argument exception is thrown. Also, if
// the underlying tensor \a TT is a lower triangular, upper triangular, or symmetric tensor
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename TT2 >    // Type of the right-hand side tensor
inline DilatedSubtensor<TT,true,CSAs...>&
   DilatedSubtensor<TT,true,CSAs...>::operator=( const Tensor<TT2>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<TT2> );

   if( pages() != (~rhs).pages() || rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<TT>, CompositeType_t<TT2>, const TT2& >;
   Right right( ~rhs );

   if( !tryAssign( tensor_, right, row(), column(), page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &tensor_ ) ) {
      const ResultType_t<TT2> tmp( right );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( tensor_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a tensor (\f$ A+=B \f$).
//
// \param rhs The right-hand side tensor to be added to the DilatedSubtensor.
// \return Reference to the dense DilatedSubtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a TT is a lower triangular, upper triangular, or
// symmetric tensor and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename TT2 >    // Type of the right-hand side tensor
inline auto DilatedSubtensor<TT,true,CSAs...>::operator+=( const Tensor<TT2>& rhs )
   -> DisableIf_t< EnforceEvaluation_v<TT,TT2>, DilatedSubtensor& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<TT2> );

   using AddType = AddTrait_t< ResultType, ResultType_t<TT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( pages() != (~rhs).pages() || rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   if( !tryAddAssign( tensor_, ~rhs, row(), column(), page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( (~rhs).canAlias( &tensor_ ) ) {
      const AddType tmp( *this + (~rhs) );
      smpAssign( left, tmp );
   }
   else {
      smpAddAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( tensor_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a tensor (\f$ A+=B \f$).
//
// \param rhs The right-hand side tensor to be added to the DilatedSubtensor.
// \return Reference to the dense DilatedSubtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a TT is a lower triangular, upper triangular, or
// symmetric tensor and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename TT2 >    // Type of the right-hand side tensor
inline auto DilatedSubtensor<TT,true,CSAs...>::operator+=( const Tensor<TT2>& rhs )
   -> EnableIf_t< EnforceEvaluation_v<TT,TT2>, DilatedSubtensor& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<TT2> );

   using AddType = AddTrait_t< ResultType, ResultType_t<TT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( pages() != (~rhs).pages() || rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   const AddType tmp( *this + (~rhs) );

   if( !tryAssign( tensor_, tmp, row(), column(), page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( tensor_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a tensor (\f$ A-=B \f$).
//
// \param rhs The right-hand side tensor to be subtracted from the DilatedSubtensor.
// \return Reference to the dense DilatedSubtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a TT is a lower triangular, upper triangular, or
// symmetric tensor and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename TT2 >    // Type of the right-hand side tensor
inline auto DilatedSubtensor<TT,true,CSAs...>::operator-=( const Tensor<TT2>& rhs )
   -> DisableIf_t< EnforceEvaluation_v<TT,TT2>, DilatedSubtensor& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<TT2> );

   using SubType = SubTrait_t< ResultType, ResultType_t<TT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( pages() != (~rhs).pages() || rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   if( !trySubAssign( tensor_, ~rhs, row(), column(), page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( (~rhs).canAlias( &tensor_ ) ) {
      const SubType tmp( *this - (~rhs ) );
      smpAssign( left, tmp );
   }
   else {
      smpSubAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( tensor_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a tensor (\f$ A-=B \f$).
//
// \param rhs The right-hand side tensor to be subtracted from the DilatedSubtensor.
// \return Reference to the dense DilatedSubtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a TT is a lower triangular, upper triangular, or
// symmetric tensor and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename TT2 >    // Type of the right-hand side tensor
inline auto DilatedSubtensor<TT,true,CSAs...>::operator-=( const Tensor<TT2>& rhs )
   -> EnableIf_t< EnforceEvaluation_v<TT,TT2>, DilatedSubtensor& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<TT2> );

   using SubType = SubTrait_t< ResultType, ResultType_t<TT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( pages() != (~rhs).pages() || rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   const SubType tmp( *this - (~rhs) );

   if( !tryAssign( tensor_, tmp, row(), column(), page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( tensor_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Schur product assignment operator for the multiplication of a tensor (\f$ A=B \f$).
//
// \param rhs The right-hand side tensor for the Schur product.
// \return Reference to the dense DilatedSubtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a TT is a lower triangular, upper triangular, or
// symmetric tensor and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename TT2 >    // Type of the right-hand side tensor
inline auto DilatedSubtensor<TT,true,CSAs...>::operator%=( const Tensor<TT2>& rhs )
   -> DisableIf_t< EnforceEvaluation_v<TT,TT2>, DilatedSubtensor& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<TT2> );

   using SchurType = SchurTrait_t< ResultType, ResultType_t<TT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SchurType );

   if( pages() != (~rhs).pages() || rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   if( !trySchurAssign( tensor_, ~rhs, row(), column(), page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( (~rhs).canAlias( &tensor_ ) ) {
      const SchurType tmp( *this % (~rhs) );
      smpAssign( left, tmp );
   }
   else {
      smpSchurAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( tensor_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Schur product assignment operator for the multiplication of a tensor (\f$ A=B \f$).
//
// \param rhs The right-hand side tensor for the Schur product.
// \return Reference to the dense DilatedSubtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a TT is a lower triangular, upper triangular, or
// symmetric tensor and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename TT2 >    // Type of the right-hand side tensor
inline auto DilatedSubtensor<TT,true,CSAs...>::operator%=( const Tensor<TT2>& rhs )
   -> EnableIf_t< EnforceEvaluation_v<TT,TT2>, DilatedSubtensor& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<TT2> );

   using SchurType = SchurTrait_t< ResultType, ResultType_t<TT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SchurType );

   if( pages() != (~rhs).pages() || rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   const SchurType tmp( *this % (~rhs) );

   if( !tryAssign( tensor_, tmp, row(), column(), page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( tensor_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the tensor containing the DilatedSubtensor.
//
// \return The tensor containing the DilatedSubtensor.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline TT& DilatedSubtensor<TT,true,CSAs...>::operand() noexcept
{
   return tensor_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the tensor containing the DilatedSubtensor.
//
// \return The tensor containing the DilatedSubtensor.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline const TT& DilatedSubtensor<TT,true,CSAs...>::operand() const noexcept
{
   return tensor_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the spacing between the beginning of two rows/columns.
//
// \return The spacing between the beginning of two rows/columns.
//
// This function returns the spacing between the beginning of two rows/columns, i.e. the
// total number of elements of a row/column. In case the storage order is set to \a rowMajor
// the function returns the spacing between two rows, in case the storage flag is set to
// \a columnMajor the function returns the spacing between two columns.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline size_t DilatedSubtensor<TT,true,CSAs...>::spacing() const noexcept
{
   return tensor_.spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense DilatedSubtensor.
//
// \return The capacity of the dense DilatedSubtensor.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline size_t DilatedSubtensor<TT,true,CSAs...>::capacity() const noexcept
{
   return pages() * rows() * columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current capacity of the specified row/column.
//
// \param i The index of the row/column.
// \return The current capacity of row/column \a i.
//
// This function returns the current capacity of the specified row/column. In case the
// storage order is set to \a rowMajor the function returns the capacity of row \a i,
// in case the storage flag is set to \a columnMajor the function returns the capacity
// of column \a i.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline size_t DilatedSubtensor<TT,true,CSAs...>::capacity( size_t i, size_t k ) const noexcept
{
   MAYBE_UNUSED( i, k );

   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index" );
   BLAZE_USER_ASSERT( i < rows(),  "Invalid row access index"  );

   return columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the dense DilatedSubtensor
//
// \return The number of non-zero elements in the dense DilatedSubtensor.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline size_t DilatedSubtensor<TT,true,CSAs...>::nonZeros() const
{
   const size_t kend( page() + pages() * pagedilation() );
   const size_t iend( row() + rows() * rowdilation() );
   const size_t jend( column() + columns() * columndilation() );
   size_t nonzeros( 0UL );

   for( size_t k=page(); k<kend; k+=pagedilation() )
      for( size_t i=row(); i<iend; i+=rowdilation() )
         for( size_t j=column(); j<jend; j+=columndilation() )
            if( !isDefault( tensor_(k,i,j) ) )
               ++nonzeros;

   return nonzeros;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the specified row/column.
//
// \param i The index of the row/column.
// \return The number of non-zero elements of row/column \a i.
//
// This function returns the current number of non-zero elements in the specified row/column.
// In case the storage order is set to \a rowMajor the function returns the number of non-zero
// elements in row \a i, in case the storage flag is set to \a columnMajor the function returns
// the number of non-zero elements in column \a i.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline size_t DilatedSubtensor<TT,true,CSAs...>::nonZeros( size_t i, size_t k ) const
{
   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index");
   BLAZE_USER_ASSERT( i < rows(),  "Invalid row access index" );

   const size_t jend( column() + columns() * columndilation() );
   size_t nonzeros( 0UL );

   for( size_t j=column(); j<jend; j+=columndilation() )
      if (!isDefault(tensor_( page() + k * pagedilation(), row() + i * rowdilation(), j)))
         ++nonzeros;

   return nonzeros;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline void DilatedSubtensor<TT,true,CSAs...>::reset()
{
   using blaze::clear;

   for ( size_t k = page(); k < page() + pages() * pagedilation(); k+=pagedilation() )
   {
      for ( size_t i = row(); i < row() + rows() * rowdilation(); i += rowdilation() )
      {
         const size_t jbegin( column() );
         const size_t jend( column() + columns() *columndilation() );
         for (size_t j = jbegin; j < jend; j+=columndilation())
            clear(tensor_(k, i, j));
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset the specified row/column to the default initial values.
//
// \param i The index of the row/column.
// \return void
//
// This function resets the values in the specified row/column to their default value. In case
// the storage order is set to \a rowMajor the function resets the values in row \a i, in case
// the storage order is set to \a columnMajor the function resets the values in column \a i.
// Note that the capacity of the row/column remains unchanged.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline void DilatedSubtensor<TT,true,CSAs...>::reset( size_t i, size_t k )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index" );
   BLAZE_USER_ASSERT( i < rows(),  "Invalid row access index"  );

   const size_t jend( column() + columns()*columndilation() );

   for( size_t j=column(); j<jend; j+=columndilation() )
      clear( tensor_( page() + k * pagedilation(), row() + i * rowdilation(), j ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checking whether there exists an overlap in the context of a symmetric tensor.
//
// \return \a true in case an overlap exists, \a false if not.
//
// This function checks if in the context of a symmetric tensor the DilatedSubtensor has an overlap with
// its counterpart. In case an overlap exists, the function return \a true, otherwise it returns
// \a false.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool DilatedSubtensor<TT,true,CSAs...>::hasOverlap() const noexcept
{
   //BLAZE_INTERNAL_ASSERT( IsSymmetric_v<TT> || IsHermitian_v<TT>, "Invalid tensor detected" );

   if( ( row() + rows()*rowdilation() <= column() ) || ( column() + columns()*columndilation() <= row() ) )
      return false;
   else return true;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  NUMERIC FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transpose of the DilatedSubtensor.
//
// \return Reference to the transposed DilatedSubtensor.
// \exception std::logic_error Invalid transpose of a non-quadratic DilatedSubtensor.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense DilatedSubtensor in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the DilatedSubtensor contains elements from the upper part of the underlying lower tensor;
//  - ... the DilatedSubtensor contains elements from the lower part of the underlying upper tensor;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian tensor.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline DilatedSubtensor<TT,true,CSAs...>&
   DilatedSubtensor<TT,true,CSAs...>::transpose()
{
   if( pages() != columns() ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic dilatedsubtensor" );
   }

   if( !tryAssign( tensor_, trans( *this ), row(), column(), page() ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   decltype(auto) left( derestrict( *this ) );
   const ResultType tmp( trans( *this ) );

   smpAssign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place conjugate transpose of the DilatedSubtensor.
//
// \return Reference to the transposed DilatedSubtensor.
// \exception std::logic_error Invalid transpose of a non-quadratic DilatedSubtensor.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense DilatedSubtensor in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the DilatedSubtensor contains elements from the upper part of the underlying lower tensor;
//  - ... the DilatedSubtensor contains elements from the lower part of the underlying upper tensor;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian tensor.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline DilatedSubtensor<TT,true,CSAs...>&
   DilatedSubtensor<TT,true,CSAs...>::ctranspose()
{
   if( pages() != columns() ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic dilatedsubtensor" );
   }

   if( !tryAssign( tensor_, ctrans( *this ), row(), column(), page() ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   decltype(auto) left( derestrict( *this ) );
   const ResultType tmp( ctrans( *this ) );

   smpAssign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the dense DilatedSubtensor by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the DilatedSubtensor scaling.
// \return Reference to the dense DilatedSubtensor.
//
// This function scales the DilatedSubtensor by applying the given scalar value \a scalar to each
// element of the DilatedSubtensor. For built-in and \c complex data types it has the same effect
// as using the multiplication assignment operator. Note that the function cannot be used
// to scale a DilatedSubtensor on a lower or upper unitriangular tensor. The attempt to scale
// such a DilatedSubtensor results in a compile time error!
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename Other >  // Data type of the scalar value
inline DilatedSubtensor<TT,true,CSAs...>&
   DilatedSubtensor<TT,true,CSAs...>::scale( const Other& scalar )
{
   const size_t kend( page() + pages() * pagedilation() );
   for( size_t k=page(); k<kend; k+=pagedilation() )
   {
      const size_t iend( row() + rows() * rowdilation() );
      for( size_t i=row(); i<iend; i+=rowdilation() )
      {
         const size_t jbegin( column() );
         const size_t jend  ( column() + columns()*columndilation() );

         for( size_t j=jbegin; j<jend; j+=columndilation()  )
            tensor_(k,i,j) *= scalar;
      }
   }
   return *this;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  EXPRESSION TEMPLATE EVALUATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the DilatedSubtensor can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this DilatedSubtensor, \a false if not.
//
// This function returns whether the given address can alias with the DilatedSubtensor. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename Other >  // Data type of the foreign expression
inline bool DilatedSubtensor<TT,true,CSAs...>::canAlias( const Other* alias ) const noexcept
{
   return tensor_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the DilatedSubtensor can alias with the given dense DilatedSubtensor \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this DilatedSubtensor, \a false if not.
//
// This function returns whether the given address can alias with the DilatedSubtensor. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename TT        // Type of the dense tensor
        , size_t... CSAs >   // Compile time DilatedSubtensor arguments
template< typename TT2       // Data type of the foreign dense DilatedSubtensor
        , size_t... CSAs2 >  // Compile time DilatedSubtensor arguments of the foreign dense DilatedSubtensor
inline bool
   DilatedSubtensor<TT,true,CSAs...>::canAlias( const DilatedSubtensor<TT2,true,CSAs2...>* alias ) const noexcept
{
   return ( tensor_.isAliased( &alias->tensor_ ) &&
          ( row() + rows() * rowdilation() > alias->row() ) &&
          ( row() < alias->row() + ( alias->rows() - 1 ) * alias->rowdilation() + 1 ) &&
          ( column() + columns() > alias->column() ) &&
          ( column() < alias->column() + (alias->columns() - 1) * alias->columndilation() + 1 ) &&
          ( page() + pages() > alias->page() ) &&
          ( page() < alias->page() + (alias->pages() - 1) * alias->pagedilation() + 1 ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the DilatedSubtensor is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this DilatedSubtensor, \a false if not.
//
// This function returns whether the given address is aliased with the DilatedSubtensor. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename Other >  // Data type of the foreign expression
inline bool DilatedSubtensor<TT,true,CSAs...>::isAliased( const Other* alias ) const noexcept
{
   return tensor_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the DilatedSubtensor is aliased with the given dense DilatedSubtensor \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this DilatedSubtensor, \a false if not.
//
// This function returns whether the given address is aliased with the DilatedSubtensor. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename TT        // Type of the dense tensor
        , size_t... CSAs >   // Compile time DilatedSubtensor arguments
template< typename TT2       // Data type of the foreign dense DilatedSubtensor
        , size_t... CSAs2 >  // Compile time DilatedSubtensor arguments of the foreign dense DilatedSubtensor
inline bool
   DilatedSubtensor<TT,true,CSAs...>::isAliased( const DilatedSubtensor<TT2,true,CSAs2...>* alias ) const noexcept
{
   return ( tensor_.isAliased( &alias->tensor_ ) &&
      ( row() + rows() * rowdilation() > alias->row() ) &&
      ( row() < alias->row() + ( alias->rows() - 1 ) * alias->rowdilation() + 1 ) &&
      ( column() + columns() * columndilation() > alias->column() ) &&
      ( column() < alias->column() + ( alias->columns() - 1 ) * alias->columndilation() + 1 ) &&
      ( page() + pages() > alias->page() ) &&
      ( page() < alias->page() + (alias->pages() - 1) * alias->pagedilation() + 1 ) );
}
/*! \endcond */
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the DilatedSubtensor can be used in SMP assignments.
//
// \return \a true in case the DilatedSubtensor can be used in SMP assignments, \a false if not.
//
// This function returns whether the DilatedSubtensor can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the DilatedSubtensor).
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
inline bool DilatedSubtensor<TT,true,CSAs...>::canSMPAssign() const noexcept
{
   return ( pages() * rows() * columns() >= SMP_DTENSASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major dense tensor.
//
// \param rhs The right-hand side dense tensor to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename TT2 >    // Type of the right-hand side dense tensor
inline auto DilatedSubtensor<TT,true,CSAs...>::assign( const DenseTensor<TT2>& rhs )
{
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages"  );
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"   );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns");

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for (size_t k = 0UL; k < pages(); ++k) {
      for (size_t i = 0UL; i < rows(); ++i) {
         for (size_t j = 0UL; j < jpos; j += 2UL) {
            (*this)(k, i, j) = (~rhs)(k, i, j);
            (*this)(k, i, j + 1UL) = (~rhs)(k, i, j + 1UL);
         }
         if (jpos < columns()) {
            (*this)(k, i, jpos) = (~rhs)(k, i, jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a row-major dense tensor.
//
// \param rhs The right-hand side dense tensor to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename TT2 >    // Type of the right-hand side dense tensor
inline auto DilatedSubtensor<TT,true,CSAs...>::addAssign( const DenseTensor<TT2>& rhs )
{
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages"  );
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"   );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns");

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for (size_t k = 0UL; k < pages(); ++k)
   {
      for (size_t i = 0UL; i < rows(); ++i)
      {
         for (size_t j = 0UL; j < jpos; j += 2UL) {
            (*this)(k, i, j) += (~rhs)(k, i, j);
            (*this)(k, i, j + 1UL) += (~rhs)(k, i, j + 1UL);
         }
         if (jpos < columns()) {
            (*this)(k, i, jpos) += (~rhs)(k, i, jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a row-major dense tensor.
//
// \param rhs The right-hand side dense tensor to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename TT2 >    // Type of the right-hand side dense tensor
inline auto DilatedSubtensor<TT,true,CSAs...>::subAssign( const DenseTensor<TT2>& rhs )
{
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages"  );
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"   );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns");

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for (size_t k = 0UL; k < pages(); ++k)
   {
      for (size_t i = 0UL; i < rows(); ++i)
      {
         for (size_t j = 0UL; j < jpos; j += 2UL) {
            (*this)(k, i, j) -= (~rhs)(k, i, j);
            (*this)(k, i, j + 1UL) -= (~rhs)(k, i, j + 1UL);
         }
         if (jpos < columns()) {
            (*this)(k, i, jpos) -= (~rhs)(k, i, jpos);
         }

      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the Schur product assignment of a row-major dense tensor.
//
// \param rhs The right-hand side dense tensor for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename TT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time DilatedSubtensor arguments
template< typename TT2 >    // Type of the right-hand side dense tensor
inline auto DilatedSubtensor<TT,true,CSAs...>::schurAssign( const DenseTensor<TT2>& rhs )
{
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages"  );
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"   );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns");

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for (size_t k = 0UL; k < pages(); ++k)
   {
      for (size_t i = 0UL; i < rows(); ++i) {
         for (size_t j = 0UL; j < jpos; j += 2UL) {
            (*this)(k, i, j) *= (~rhs)(k, i, j);
            (*this)(k, i, j + 1UL) *= (~rhs)(k, i, j + 1UL);
         }
         if (jpos < columns()) {
            (*this)(k, i, jpos) *= (~rhs)(k, i, jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
