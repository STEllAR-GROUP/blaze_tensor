//=================================================================================================
/*!
//  \file blaze_tensor/math/views/subtensor/DenseUnaligned.h
//  \brief Subtensor specialization for dense matrices
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_SUBTENSOR_DENSEUNALIGNED_H_
#define _BLAZE_TENSOR_MATH_VIEWS_SUBTENSOR_DENSEUNALIGNED_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/Exception.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/StorageOrder.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/TransExpr.h>
#include <blaze/math/expressions/View.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/SchurTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/HasSIMDSub.h>
#include <blaze/math/typetraits/RequiresEvaluation.h>
#include <blaze/math/views/Check.h>
#include <blaze/system/Blocking.h>
#include <blaze/system/CacheSize.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/AlignmentCheck.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/TypeList.h>
#include <blaze/util/Types.h>
#include <blaze/util/MaybeUnused.h>
#include <blaze/util/algorithms/Max.h>
#include <blaze/util/algorithms/Min.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsReference.h>
#include <iterator>

#include <blaze_tensor/math/InitializerList.h>
#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/Subtensor.h>
#include <blaze_tensor/math/dense/InitializerTensor.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>
#include <blaze_tensor/math/traits/SubtensorTrait.h>
#include <blaze_tensor/math/views/subtensor/BaseTemplate.h>
#include <blaze_tensor/math/views/subtensor/SubtensorData.h>

namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR UNALIGNED ROW-MAJOR DENSE SUBTENSORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Subtensor for unaligned row-major dense subtensors.
// \ingroup subtensor
//
// This Specialization of Subtensor adapts the class template to the requirements of unaligned
// row-major dense subtensors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
class Subtensor<MT,unaligned,CSAs...>
   : public View< DenseTensor< Subtensor<MT,unaligned,CSAs...> > >
   , private SubtensorData<CSAs...>
{
 private:
   //**Type definitions****************************************************************************
   using DataType = SubtensorData<CSAs...>;               //!< The type of the SubtensorData base class.
   using Operand  = If_t< IsExpression_v<MT>, MT, MT& >;  //!< Composite data type of the tensor expression.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT1, typename MT2 >
   static constexpr bool EnforceEvaluation_v =
      ( IsRestricted_v<MT1> && RequiresEvaluation_v<MT2> );
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this Subtensor instance.
   using This = Subtensor<MT,unaligned,CSAs...>;

   using BaseType      = DenseTensor<This>;             //!< Base type of this Subtensor instance.
   using ViewedType    = MT;                            //!< The type viewed by this Subtensor instance.
   using ResultType    = SubtensorTrait_t<MT,CSAs...>;  //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;    //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;   //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<MT>;             //!< Type of the subtensor elements.
   using SIMDType      = SIMDTrait_t<ElementType>;      //!< SIMD type of the subtensor elements.
   using ReturnType    = ReturnType_t<MT>;              //!< Return type for expression template evaluations
   using CompositeType = const Subtensor&;              //!< Data type for composite expression templates.

   //! Reference to a constant subtensor value.
   using ConstReference = ConstReference_t<MT>;

   //! Reference to a non-constant subtensor value.
   using Reference = If_t< IsConst_v<MT>, ConstReference, Reference_t<MT> >;

   //! Pointer to a constant subtensor value.
   using ConstPointer = ConstPointer_t<MT>;

   //! Pointer to a non-constant subtensor value.
   using Pointer = If_t< IsConst_v<MT> || !HasMutableDataAccess_v<MT>, ConstPointer, Pointer_t<MT> >;
   //**********************************************************************************************

   //**SubtensorIterator class definition**********************************************************
   /*!\brief Iterator over the elements of the dense subtensor.
   */
   template< typename IteratorType >  // Type of the dense tensor iterator
   class SubtensorIterator
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
      /*!\brief Default constructor of the SubtensorIterator class.
      */
      inline SubtensorIterator()
         : iterator_ (       )  // Iterator to the current subtensor element
         , isAligned_( false )  // Memory alignment flag
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor of the SubtensorIterator class.
      //
      // \param iterator Iterator to the initial element.
      // \param isMemoryAligned Memory alignment flag.
      */
      inline SubtensorIterator( IteratorType iterator, bool isMemoryAligned )
         : iterator_ ( iterator        )  // Iterator to the current subtensor element
         , isAligned_( isMemoryAligned )  // Memory alignment flag
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Conversion constructor from different SubtensorIterator instances.
      //
      // \param it The subtensor iterator to be copied.
      */
      template< typename IteratorType2 >
      inline SubtensorIterator( const SubtensorIterator<IteratorType2>& it )
         : iterator_ ( it.base()      )  // Iterator to the current subtensor element
         , isAligned_( it.isAligned() )  // Memory alignment flag
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline SubtensorIterator& operator+=( size_t inc ) {
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
      inline SubtensorIterator& operator-=( size_t dec ) {
         iterator_ -= dec;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline SubtensorIterator& operator++() {
         ++iterator_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const SubtensorIterator operator++( int ) {
         return SubtensorIterator( iterator_++, isAligned_ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline SubtensorIterator& operator--() {
         --iterator_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const SubtensorIterator operator--( int ) {
         return SubtensorIterator( iterator_--, isAligned_ );
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

      //**Load function****************************************************************************
      /*!\brief Load of a SIMD element of the dense subtensor.
      //
      // \return The loaded SIMD element.
      //
      // This function performs a load of the current SIMD element of the subtensor iterator.
      // This function must \b NOT be called explicitly! It is used internally for the performance
      // optimized evaluation of expression templates. Calling this function explicitly might
      // result in erroneous results and/or in compilation errors.
      */
      inline SIMDType load() const noexcept {
         if( isAligned_ )
            return loada();
         else
            return loadu();
      }
      //*******************************************************************************************

      //**Loada function***************************************************************************
      /*!\brief Aligned load of a SIMD element of the dense subtensor.
      //
      // \return The loaded SIMD element.
      //
      // This function performs an aligned load of the current SIMD element of the subtensor
      // iterator. This function must \b NOT be called explicitly! It is used internally for
      // the performance optimized evaluation of expression templates. Calling this function
      // explicitly might result in erroneous results and/or in compilation errors.
      */
      inline SIMDType loada() const noexcept {
         return iterator_.loada();
      }
      //*******************************************************************************************

      //**Loadu function***************************************************************************
      /*!\brief Unaligned load of a SIMD element of the dense subtensor.
      //
      // \return The loaded SIMD element.
      //
      // This function performs an unaligned load of the current SIMD element of the subtensor
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline SIMDType loadu() const noexcept {
         return iterator_.loadu();
      }
      //*******************************************************************************************

      //**Store function***************************************************************************
      /*!\brief Store of a SIMD element of the dense subtensor.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs a store of the current SIMD element of the subtensor iterator.
      // This function must \b NOT be called explicitly! It is used internally for the performance
      // optimized evaluation of expression templates. Calling this function explicitly might
      // result in erroneous results and/or in compilation errors.
      */
      inline void store( const SIMDType& value ) const {
         storeu( value );
      }
      //*******************************************************************************************

      //**Storea function**************************************************************************
      /*!\brief Aligned store of a SIMD element of the dense subtensor.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs an aligned store of the current SIMD element of the subtensor
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline void storea( const SIMDType& value ) const {
         iterator_.storea( value );
      }
      //*******************************************************************************************

      //**Storeu function**************************************************************************
      /*!\brief Unaligned store of a SIMD element of the dense subtensor.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs an unaligned store of the current SIMD element of the subtensor
      // iterator. This function must \b NOT be called explicitly! It is used internally for the
      // performance optimized evaluation of expression templates. Calling this function explicitly
      // might result in erroneous results and/or in compilation errors.
      */
      inline void storeu( const SIMDType& value ) const {
         if( isAligned_ ) {
            iterator_.storea( value );
         }
         else {
            iterator_.storeu( value );
         }
      }
      //*******************************************************************************************

      //**Stream function**************************************************************************
      /*!\brief Aligned, non-temporal store of a SIMD element of the dense subtensor.
      //
      // \param value The SIMD element to be stored.
      // \return void
      //
      // This function performs an aligned, non-temporal store of the current SIMD element of the
      // subtensor iterator. This function must \b NOT be called explicitly! It is used internally
      // for the performance optimized evaluation of expression templates. Calling this function
      // explicitly might result in erroneous results and/or in compilation errors.
      */
      inline void stream( const SIMDType& value ) const {
         iterator_.stream( value );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two SubtensorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const SubtensorIterator& rhs ) const {
         return iterator_ == rhs.iterator_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two SubtensorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const SubtensorIterator& rhs ) const {
         return iterator_ != rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two SubtensorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const SubtensorIterator& rhs ) const {
         return iterator_ < rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two SubtensorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const SubtensorIterator& rhs ) const {
         return iterator_ > rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two SubtensorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const SubtensorIterator& rhs ) const {
         return iterator_ <= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two SubtensorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const SubtensorIterator& rhs ) const {
         return iterator_ >= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const SubtensorIterator& rhs ) const {
         return iterator_ - rhs.iterator_;
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between a SubtensorIterator and an integral value.
      //
      // \param it The iterator to be incremented.
      // \param inc The number of elements the iterator is incremented.
      // \return The incremented iterator.
      */
      friend inline const SubtensorIterator operator+( const SubtensorIterator& it, size_t inc ) {
         return SubtensorIterator( it.iterator_ + inc, it.isAligned_ );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between an integral value and a SubtensorIterator.
      //
      // \param inc The number of elements the iterator is incremented.
      // \param it The iterator to be incremented.
      // \return The incremented iterator.
      */
      friend inline const SubtensorIterator operator+( size_t inc, const SubtensorIterator& it ) {
         return SubtensorIterator( it.iterator_ + inc, it.isAligned_ );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Subtraction between a SubtensorIterator and an integral value.
      //
      // \param it The iterator to be decremented.
      // \param dec The number of elements the iterator is decremented.
      // \return The decremented iterator.
      */
      friend inline const SubtensorIterator operator-( const SubtensorIterator& it, size_t dec ) {
         return SubtensorIterator( it.iterator_ - dec, it.isAligned_ );
      }
      //*******************************************************************************************

      //**Base function****************************************************************************
      /*!\brief Access to the current position of the subtensor iterator.
      //
      // \return The current position of the subtensor iterator.
      */
      inline IteratorType base() const {
         return iterator_;
      }
      //*******************************************************************************************

      //**IsAligned function***********************************************************************
      /*!\brief Access to the iterator's memory alignment flag.
      //
      // \return \a true in case the iterator is aligned, \a false if it is not.
      */
      inline bool isAligned() const noexcept {
         return isAligned_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType iterator_;   //!< Iterator to the current subtensor element.
      bool         isAligned_;  //!< Memory alignment flag.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   //! Iterator over constant elements.
   using ConstIterator = SubtensorIterator< ConstIterator_t<MT> >;

   //! Iterator over non-constant elements.
   using Iterator = If_t< IsConst_v<MT>, ConstIterator, SubtensorIterator< Iterator_t<MT> > >;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = MT::simdEnabled;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = MT::smpAssignable;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RSAs >
   explicit inline Subtensor( MT& tensor, RSAs... args );

   Subtensor( const Subtensor& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~Subtensor() = default;
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
   inline Subtensor& operator=( const ElementType& rhs );
   inline Subtensor& operator=( initializer_list< initializer_list< initializer_list<ElementType> > > list );
   inline Subtensor& operator=( const Subtensor& rhs );

   template< typename MT2 >
   inline Subtensor& operator=( const Tensor<MT2>& rhs );

   template< typename MT2 >
   inline auto operator+=( const Tensor<MT2>& rhs )
      -> EnableIf_t< !EnforceEvaluation_v<MT,MT2>, Subtensor& >;

   template< typename MT2 >
   inline auto operator+=( const Tensor<MT2>& rhs )
      -> EnableIf_t< EnforceEvaluation_v<MT,MT2>, Subtensor& >;

   template< typename MT2 >
   inline auto operator-=( const Tensor<MT2>& rhs )
      -> EnableIf_t< !EnforceEvaluation_v<MT,MT2>, Subtensor& >;

   template< typename MT2 >
   inline auto operator-=( const Tensor<MT2>& rhs )
      -> EnableIf_t< EnforceEvaluation_v<MT,MT2>, Subtensor& >;

   template< typename MT2 >
   inline auto operator%=( const Tensor<MT2>& rhs )
      -> EnableIf_t< !EnforceEvaluation_v<MT,MT2>, Subtensor& >;

   template< typename MT2 >
   inline auto operator%=( const Tensor<MT2>& rhs )
      -> EnableIf_t< EnforceEvaluation_v<MT,MT2>, Subtensor& >;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   using DataType::row;
   using DataType::column;
   using DataType::page;
   using DataType::rows;
   using DataType::columns;
   using DataType::pages;

   inline MT&       operand() noexcept;
   inline const MT& operand() const noexcept;

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
   inline Subtensor& transpose();
   inline Subtensor& ctranspose();

   template< typename T >
   inline Subtensor& transpose( const T* indices, size_t n );
   template< typename T >
   inline Subtensor& ctranspose( const T* indices, size_t n );

   template< typename Other > inline Subtensor& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT2 >
   static constexpr bool VectorizedAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && MT2::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<MT2> > );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT2 >
   static constexpr bool VectorizedAddAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && MT2::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<MT2> > &&
        HasSIMDAdd_v< ElementType, ElementType_t<MT2> > &&
        !IsDiagonal_v<MT2> );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT2 >
   static constexpr bool VectorizedSubAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && MT2::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<MT2> > &&
        HasSIMDSub_v< ElementType, ElementType_t<MT2> > &&
        !IsDiagonal_v<MT2> );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT2 >
   static constexpr bool VectorizedSchurAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && MT2::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<MT2> > &&
        HasSIMDMult_v< ElementType, ElementType_t<MT2> > );
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   static constexpr size_t SIMDSIZE = SIMDTrait<ElementType>::size;
   //**********************************************************************************************

 public:
   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other >
   inline bool canAlias( const Other* alias ) const noexcept;

   template< typename MT2, AlignmentFlag AF2, size_t... CSAs2 >
   inline bool canAlias( const Subtensor<MT2,AF2,CSAs2...>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename MT2, AlignmentFlag AF2, size_t... CSAs2 >
   inline bool isAliased( const Subtensor<MT2,AF2,CSAs2...>* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t k, size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t k, size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t k, size_t i, size_t j ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept;

   template< typename MT2 >
   inline auto assign( const DenseTensor<MT2>& rhs ) -> EnableIf_t< !VectorizedAssign_v<MT2> >;

   template< typename MT2 >
   inline auto assign( const DenseTensor<MT2>& rhs ) -> EnableIf_t< VectorizedAssign_v<MT2> >;

   template< typename MT2 >
   inline auto addAssign( const DenseTensor<MT2>& rhs ) -> EnableIf_t< !VectorizedAddAssign_v<MT2> >;

   template< typename MT2 >
   inline auto addAssign( const DenseTensor<MT2>& rhs ) -> EnableIf_t< VectorizedAddAssign_v<MT2> >;

   template< typename MT2 >
   inline auto subAssign( const DenseTensor<MT2>& rhs ) -> EnableIf_t< !VectorizedSubAssign_v<MT2> >;

   template< typename MT2 >
   inline auto subAssign( const DenseTensor<MT2>& rhs ) -> EnableIf_t< VectorizedSubAssign_v<MT2> >;

   template< typename MT2 >
   inline auto schurAssign( const DenseTensor<MT2>& rhs ) -> EnableIf_t< !VectorizedSchurAssign_v<MT2> >;

   template< typename MT2 >
   inline auto schurAssign( const DenseTensor<MT2>& rhs ) -> EnableIf_t< VectorizedSchurAssign_v<MT2> >;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand tensor_;        //!< The tensor containing the subtensor.
   const bool isAligned_;  //!< Memory alignment flag.
                           /*!< The alignment flag indicates whether the subtensor is fully aligned
                                with respect to the given element type and the available instruction
                                set. In case the subtensor is fully aligned it is possible to use
                                aligned loads and stores instead of unaligned loads and stores. In
                                order to be aligned, the first element of each row/column must be
                                aligned. */
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename MT2, AlignmentFlag AF2, size_t... CSAs2 > friend class Subtensor;
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE   ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SUBTENSOR_TYPE   ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE     ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE   ( MT );
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
/*!\brief Constructor for unaligned row-major dense subtensors.
//
// \param tensor The dense tensor containing the subtensor.
// \param args The runtime subtensor arguments.
// \exception std::invalid_argument Invalid subtensor specification.
//
// By default, the provided subtensor arguments are checked at runtime. In case the subtensor is
// not properly specified (i.e. if the specified subtensor is not contained in the given dense
// tensor) a \a std::invalid_argument exception is thrown. The checks can be skipped by providing
// the optional \a blaze::unchecked argument.
*/
template< typename MT         // Type of the dense tensor
        , size_t... CSAs >    // Compile time subtensor arguments
template< typename... RSAs >  // Runtime subtensor arguments
inline Subtensor<MT,unaligned,CSAs...>::Subtensor( MT& tensor, RSAs... args )
   : DataType  ( args... )  // Base class initialization
   , tensor_   ( tensor  )  // The tensor containing the subtensor
   , isAligned_( simdEnabled && tensor.data() != nullptr && checkAlignment( data() ) &&
                 ( rows() < 2UL || ( tensor.spacing() & size_t(-SIMDSIZE) ) == 0UL ) )
{
   if( !Contains_v< TypeList<RSAs...>, Unchecked > ) {
      if( ( row() + rows() > tensor_.rows() ) || ( column() + columns() > tensor_.columns() )  || ( page() + pages() > tensor_.pages() )) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid subtensor specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( row()    + rows()    <= tensor_.rows()   , "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( column() + columns() <= tensor_.columns(), "Invalid subtensor specification" );
      BLAZE_USER_ASSERT( page()   + pages()   <= tensor_.pages()  , "Invalid subtensor specification" );
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
/*!\brief 2D-access to the dense subtensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::Reference
   Subtensor<MT,unaligned,CSAs...>::operator()( size_t k, size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_USER_ASSERT( k < pages()  , "Invalid page access index" );

   return tensor_(page()+k,row()+i,column()+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the dense subtensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::ConstReference
   Subtensor<MT,unaligned,CSAs...>::operator()( size_t k, size_t i, size_t j ) const
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_USER_ASSERT( k < pages()  , "Invalid page access index" );

   return const_cast<const MT&>( tensor_ )(page()+k,row()+i,column()+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the subtensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid tensor access index.
//
// In contrast to the function call operator this function always performs a check of the given
// access indices.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::Reference
   Subtensor<MT,unaligned,CSAs...>::at( size_t k, size_t i, size_t j )
{
   if( i >= rows() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= columns() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   if( k >= pages() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
   }
   return (*this)(k,i,j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the subtensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid tensor access index.
//
// In contrast to the function call operator this function always performs a check of the given
// access indices.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::ConstReference
   Subtensor<MT,unaligned,CSAs...>::at( size_t k, size_t i, size_t j ) const
{
   if( i >= rows() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= columns() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   if( k >= pages() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
   }
   return (*this)(k,i,j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the subtensor elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense subtensor. Note that
// you can NOT assume that all tensor elements lie adjacent to each other! The dense subtensor
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::Pointer
   Subtensor<MT,unaligned,CSAs...>::data() noexcept
{
   return tensor_.data() + ( page()*tensor_.rows() + row() ) * spacing() + column();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the subtensor elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense subtensor. Note that
// you can NOT assume that all tensor elements lie adjacent to each other! The dense subtensor
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::ConstPointer
   Subtensor<MT,unaligned,CSAs...>::data() const noexcept
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::Pointer
   Subtensor<MT,unaligned,CSAs...>::data( size_t i, size_t k ) noexcept
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::ConstPointer
   Subtensor<MT,unaligned,CSAs...>::data( size_t i, size_t k ) const noexcept
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::Iterator
   Subtensor<MT,unaligned,CSAs...>::begin( size_t i, size_t k )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense subtensor row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid dense subtensor page access index" );
   return Iterator( tensor_.begin( row() + i, page() + k ) + column(), isAligned_ );
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::ConstIterator
   Subtensor<MT,unaligned,CSAs...>::begin( size_t i, size_t k ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense subtensor row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid dense subtensor page access index" );
   return ConstIterator( tensor_.cbegin( row() + i, page() + k ) + column(), isAligned_ );
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::ConstIterator
   Subtensor<MT,unaligned,CSAs...>::cbegin( size_t i, size_t k ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense subtensor row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid dense subtensor page access index" );
   return ConstIterator( tensor_.cbegin( row() + i, page() + k ) + column(), isAligned_ );
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::Iterator
   Subtensor<MT,unaligned,CSAs...>::end( size_t i, size_t k )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense subtensor row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid dense subtensor page access index" );
   return Iterator( tensor_.begin( row() + i, page() + k ) + column() + columns(), isAligned_ );
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::ConstIterator
   Subtensor<MT,unaligned,CSAs...>::end( size_t i, size_t k ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense subtensor row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid dense subtensor page access index" );
   return ConstIterator( tensor_.cbegin( row() + i, page() + k ) + column() + columns(), isAligned_ );
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline typename Subtensor<MT,unaligned,CSAs...>::ConstIterator
   Subtensor<MT,unaligned,CSAs...>::cend( size_t i, size_t k ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense subtensor row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid dense subtensor page access index" );
   return ConstIterator( tensor_.cbegin( row() + i, page() + k ) + column() + columns(), isAligned_ );
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
/*!\brief Homogenous assignment to all subtensor elements.
//
// \param rhs Scalar value to be assigned to all subtensor elements.
// \return Reference to the assigned subtensor.
//
// This function homogeneously assigns the given value to all dense tensor elements. Note that in
// case the underlying dense tensor is a lower/upper tensor only lower/upper and diagonal elements
// of the underlying tensor are modified.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline Subtensor<MT,unaligned,CSAs...>&
   Subtensor<MT,unaligned,CSAs...>::operator=( const ElementType& rhs )
{
   decltype(auto) left( derestrict( tensor_ ) );

   const size_t kend( page() + pages() );
   for( size_t k=page(); k<kend; ++k ) {
      const size_t iend( row() + rows() );

      for( size_t i=row(); i<iend; ++i )
      {
          const size_t jbegin( column() );
          const size_t jend  ( column() + columns() );

         for( size_t j=jbegin; j<jend; ++j ) {
            if( !IsRestricted_v<MT> || IsTriangular_v<MT> || trySet( tensor_, i, j, k, rhs ) )
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
/*!\brief List assignment to all subtensor elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid initializer list dimension.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// This assignment operator offers the option to directly assign to all elements of the subtensor
// by means of an initializer list. The subtensor elements are assigned the values from the given
// initializer list. Missing values are initialized as default. Note that in case the size of the
// top-level initializer list does not match the number of rows of the subtensor or the size of
// any nested list exceeds the number of columns, a \a std::invalid_argument exception is thrown.
// Also, if the underlying tensor \a MT is restricted and the assignment would violate an
// invariant of the tensor, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline Subtensor<MT,unaligned,CSAs...>&
   Subtensor<MT,unaligned,CSAs...>::operator=( initializer_list<initializer_list< initializer_list<ElementType> > > list )
{
   if( list.size() != pages() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to subtensor" );
   }

   if( IsRestricted_v<MT> ) {
      const InitializerTensor<ElementType> tmp( list, row(), columns() );
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
/*!\brief Copy assignment operator for Subtensor.
//
// \param rhs Dense subtensor to be copied.
// \return Reference to the assigned subtensor.
// \exception std::invalid_argument Subtensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// The dense subtensor is initialized as a copy of the given dense subtensor. In case the current
// sizes of the two subtensors don't match, a \a std::invalid_argument exception is thrown. Also,
// if the underlying tensor \a MT is a lower triangular, upper triangular, or symmetric tensor
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline Subtensor<MT,unaligned,CSAs...>&
   Subtensor<MT,unaligned,CSAs...>::operator=( const Subtensor& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( this == &rhs || ( &tensor_ == &rhs.tensor_ && row() == rhs.row() && column() == rhs.column() && page() == rhs.page()) )
      return *this;

   if( rows() != rhs.rows() || columns() != rhs.columns() || pages() != rhs.pages() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Subtensor sizes do not match" );
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
// \return Reference to the assigned subtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// The dense subtensor is initialized as a copy of the given tensor. In case the current sizes
// of the two matrices don't match, a \a std::invalid_argument exception is thrown. Also, if
// the underlying tensor \a MT is a lower triangular, upper triangular, or symmetric tensor
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >     // Type of the right-hand side tensor
inline Subtensor<MT,unaligned,CSAs...>&
   Subtensor<MT,unaligned,CSAs...>::operator=( const Tensor<MT2>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() || pages() != (~rhs).pages() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<MT>, CompositeType_t<MT2>, const MT2& >;
   Right right( ~rhs );

   if( !tryAssign( tensor_, right, row(), column(), page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &tensor_ ) ) {
      const ResultType_t<MT2> tmp( right );
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
// \param rhs The right-hand side tensor to be added to the subtensor.
// \return Reference to the dense subtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower triangular, upper triangular, or
// symmetric tensor and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >     // Type of the right-hand side tensor
inline auto Subtensor<MT,unaligned,CSAs...>::operator+=( const Tensor<MT2>& rhs )
   -> EnableIf_t< !EnforceEvaluation_v<MT,MT2>, Subtensor& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using AddType = AddTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() || pages() != (~rhs).pages() ) {
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
// \param rhs The right-hand side tensor to be added to the subtensor.
// \return Reference to the dense subtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower triangular, upper triangular, or
// symmetric tensor and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >     // Type of the right-hand side tensor
inline auto Subtensor<MT,unaligned,CSAs...>::operator+=( const Tensor<MT2>& rhs )
   -> EnableIf_t< EnforceEvaluation_v<MT,MT2>, Subtensor& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using AddType = AddTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() || pages() != (~rhs).pages() ) {
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
// \param rhs The right-hand side tensor to be subtracted from the subtensor.
// \return Reference to the dense subtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower triangular, upper triangular, or
// symmetric tensor and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >     // Type of the right-hand side tensor
inline auto Subtensor<MT,unaligned,CSAs...>::operator-=( const Tensor<MT2>& rhs )
   -> EnableIf_t< !EnforceEvaluation_v<MT,MT2>, Subtensor& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using SubType = SubTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() || pages() != (~rhs).pages() ) {
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
// \param rhs The right-hand side tensor to be subtracted from the subtensor.
// \return Reference to the dense subtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower triangular, upper triangular, or
// symmetric tensor and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >     // Type of the right-hand side tensor
inline auto Subtensor<MT,unaligned,CSAs...>::operator-=( const Tensor<MT2>& rhs )
   -> EnableIf_t< EnforceEvaluation_v<MT,MT2>, Subtensor& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using SubType = SubTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() || pages() != (~rhs).pages() ) {
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
// \return Reference to the dense subtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower triangular, upper triangular, or
// symmetric tensor and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >     // Type of the right-hand side tensor
inline auto Subtensor<MT,unaligned,CSAs...>::operator%=( const Tensor<MT2>& rhs )
   -> EnableIf_t< !EnforceEvaluation_v<MT,MT2>, Subtensor& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using SchurType = SchurTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SchurType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() || pages() != (~rhs).pages() ) {
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
// \return Reference to the dense subtensor.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower triangular, upper triangular, or
// symmetric tensor and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >     // Type of the right-hand side tensor
inline auto Subtensor<MT,unaligned,CSAs...>::operator%=( const Tensor<MT2>& rhs )
   -> EnableIf_t< EnforceEvaluation_v<MT,MT2>, Subtensor& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using SchurType = SchurTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SchurType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() || pages() != (~rhs).pages() ) {
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
/*!\brief Returns the tensor containing the subtensor.
//
// \return The tensor containing the subtensor.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline MT& Subtensor<MT,unaligned,CSAs...>::operand() noexcept
{
   return tensor_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the tensor containing the subtensor.
//
// \return The tensor containing the subtensor.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline const MT& Subtensor<MT,unaligned,CSAs...>::operand() const noexcept
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline size_t Subtensor<MT,unaligned,CSAs...>::spacing() const noexcept
{
   return tensor_.spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense subtensor.
//
// \return The capacity of the dense subtensor.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline size_t Subtensor<MT,unaligned,CSAs...>::capacity() const noexcept
{
   return rows() * columns() * pages();
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline size_t Subtensor<MT,unaligned,CSAs...>::capacity( size_t i, size_t k ) const noexcept
{
   MAYBE_UNUSED( i, k );

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index" );

   return columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the dense subtensor
//
// \return The number of non-zero elements in the dense subtensor.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline size_t Subtensor<MT,unaligned,CSAs...>::nonZeros() const
{
   const size_t iend( row() + rows() );
   const size_t jend( column() + columns() );
   const size_t kend( page() + pages() );
   size_t nonzeros( 0UL );

   for( size_t k=page(); k<kend; ++k )
      for( size_t i=row(); i<iend; ++i )
         for( size_t j=column(); j<jend; ++j )
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline size_t Subtensor<MT,unaligned,CSAs...>::nonZeros( size_t i, size_t k ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index" );

   const size_t jend( column() + columns() );
   size_t nonzeros( 0UL );

   for( size_t j=column(); j<jend; ++j )
      if( !isDefault( tensor_(page()+k,row()+i,j) ) )
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline void Subtensor<MT,unaligned,CSAs...>::reset()
{
   using blaze::clear;

   for( size_t k=page(); k<page()+pages(); ++k )
      for( size_t i=row(); i<row()+rows(); ++i )
      {
         const size_t jbegin( column() );
         const size_t jend  ( column()+columns() );
         for( size_t j=jbegin; j<jend; ++j )
            clear( tensor_(k,i,j) );
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline void Subtensor<MT,unaligned,CSAs...>::reset( size_t i, size_t k )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index" );

   const size_t jend( column() + columns() );

   for( size_t j=column(); j<jend; j++ )
      clear( tensor_( page() + k, i + row(), j ) );
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
/*!\brief In-place transpose of the subtensor.
//
// \return Reference to the transposed subtensor.
// \exception std::logic_error Invalid transpose of a non-quadratic subtensor.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense subtensor in-place. Note that this function can only be used
// for quadratic subtensors, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the subtensor contains elements from the upper part of the underlying lower tensor;
//  - ... the subtensor contains elements from the lower part of the underlying upper tensor;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian tensor.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline Subtensor<MT,unaligned,CSAs...>&
   Subtensor<MT,unaligned,CSAs...>::transpose()
{
   if( pages() != columns() ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic subtensor" );
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
/*!\brief In-place transpose of the subtensor.
//
// \return Reference to the transposed subtensor.
// \exception std::logic_error Invalid transpose of a non-quadratic subtensor.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense subtensor in-place. Note that this function can only be used
// for quadratic subtensors, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the subtensor contains elements from the upper part of the underlying lower tensor;
//  - ... the subtensor contains elements from the lower part of the underlying upper tensor;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian tensor.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename T >      // Type of the mapping indices
inline Subtensor<MT,unaligned,CSAs...>&
   Subtensor<MT,unaligned,CSAs...>::transpose( const T* indices, size_t n )
{
//    if( rows() != columns() ) {
//       BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic subtensor" );
//    }

   if( !tryAssign( tensor_, trans( *this, indices, n ), row(), column(), page() ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   decltype(auto) left( derestrict( *this ) );
   const ResultType tmp( trans( *this, indices, n ) );

   smpAssign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place conjugate transpose of the subtensor.
//
// \return Reference to the transposed subtensor.
// \exception std::logic_error Invalid transpose of a non-quadratic subtensor.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense subtensor in-place. Note that this function can only be used
// for quadratic subtensors, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the subtensor contains elements from the upper part of the underlying lower tensor;
//  - ... the subtensor contains elements from the lower part of the underlying upper tensor;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian tensor.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline Subtensor<MT,unaligned,CSAs...>&
   Subtensor<MT,unaligned,CSAs...>::ctranspose()
{
   if( pages() != columns() ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic subtensor" );
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
/*!\brief In-place conjugate transpose of the subtensor.
//
// \return Reference to the transposed subtensor.
// \exception std::logic_error Invalid transpose of a non-quadratic subtensor.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense subtensor in-place. Note that this function can only be used
// for quadratic subtensors, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the subtensor contains elements from the upper part of the underlying lower tensor;
//  - ... the subtensor contains elements from the lower part of the underlying upper tensor;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian tensor.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename T >      // Type of the mapping indices
inline Subtensor<MT,unaligned,CSAs...>&
   Subtensor<MT,unaligned,CSAs...>::ctranspose( const T* indices, size_t n )
{
   if( rows() != columns() ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic subtensor" );
   }

   if( !tryAssign( tensor_, ctrans( *this, indices, n ), row(), column(), page() ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   decltype(auto) left( derestrict( *this ) );
   const ResultType tmp( ctrans( *this, indices, n ) );

   smpAssign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the dense subtensor by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the subtensor scaling.
// \return Reference to the dense subtensor.
//
// This function scales the subtensor by applying the given scalar value \a scalar to each
// element of the subtensor. For built-in and \c complex data types it has the same effect
// as using the multiplication assignment operator. Note that the function cannot be used
// to scale a subtensor on a lower or upper unitriangular tensor. The attempt to scale
// such a subtensor results in a compile time error!
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename Other >  // Data type of the scalar value
inline Subtensor<MT,unaligned,CSAs...>&
   Subtensor<MT,unaligned,CSAs...>::scale( const Other& scalar )
{
   const size_t kend( page() + pages() );
   for( size_t k=page(); k<kend; ++k )
   {
      const size_t iend( row() + rows() );
      for( size_t i=row(); i<iend; ++i )
      {
         const size_t jbegin( column() );
         const size_t jend  ( column()+columns() );

         for( size_t j=jbegin; j<jend; ++j )
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
/*!\brief Returns whether the subtensor can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this subtensor, \a false if not.
//
// This function returns whether the given address can alias with the subtensor. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename Other >  // Data type of the foreign expression
inline bool Subtensor<MT,unaligned,CSAs...>::canAlias( const Other* alias ) const noexcept
{
   return tensor_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the subtensor can alias with the given dense subtensor \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this subtensor, \a false if not.
//
// This function returns whether the given address can alias with the subtensor. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT        // Type of the dense tensor
        , size_t... CSAs >   // Compile time subtensor arguments
template< typename MT2       // Data type of the foreign dense subtensor
        , AlignmentFlag AF2  // Alignment flag of the foreign dense subtensor
        , size_t... CSAs2 >  // Compile time subtensor arguments of the foreign dense subtensor
inline bool
   Subtensor<MT,unaligned,CSAs...>::canAlias( const Subtensor<MT2,AF2,CSAs2...>* alias ) const noexcept
{
   return ( tensor_.isAliased( &alias->tensor_ ) &&
            ( row() + rows() > alias->row() ) &&
            ( row() < alias->row() + alias->rows() ) &&
            ( column() + columns() > alias->column() ) &&
            ( column() < alias->column() + alias->columns() ) &&
            ( page() + pages() > alias->page() ) &&
            ( page() < alias->page() + alias->pages() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the subtensor is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this subtensor, \a false if not.
//
// This function returns whether the given address is aliased with the subtensor. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename Other >  // Data type of the foreign expression
inline bool Subtensor<MT,unaligned,CSAs...>::isAliased( const Other* alias ) const noexcept
{
   return tensor_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the subtensor is aliased with the given dense subtensor \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this subtensor, \a false if not.
//
// This function returns whether the given address is aliased with the subtensor. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT        // Type of the dense tensor
        , size_t... CSAs >   // Compile time subtensor arguments
template< typename MT2       // Data type of the foreign dense subtensor
        , AlignmentFlag AF2  // Alignment flag of the foreign dense subtensor
        , size_t... CSAs2 >  // Compile time subtensor arguments of the foreign dense subtensor
inline bool
   Subtensor<MT,unaligned,CSAs...>::isAliased( const Subtensor<MT2,AF2,CSAs2...>* alias ) const noexcept
{
   return ( tensor_.isAliased( &alias->tensor_ ) &&
            ( row() + rows() > alias->row() ) &&
            ( row() < alias->row() + alias->rows() ) &&
            ( column() + columns() > alias->column() ) &&
            ( column() < alias->column() + alias->columns() ) &&
            ( page() + pages() > alias->page() ) &&
            ( page() < alias->page() + alias->pages() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the subtensor is properly aligned in memory.
//
// \return \a true in case the subtensor is aligned, \a false if not.
//
// This function returns whether the subtensor is guaranteed to be properly aligned in memory,
// i.e. whether the beginning and the end of each row/column of the subtensor are guaranteed to
// conform to the alignment restrictions of the underlying element type.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool Subtensor<MT,unaligned,CSAs...>::isAligned() const noexcept
{
   return isAligned_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the subtensor can be used in SMP assignments.
//
// \return \a true in case the subtensor can be used in SMP assignments, \a false if not.
//
// This function returns whether the subtensor can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the subtensor).
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
inline bool Subtensor<MT,unaligned,CSAs...>::canSMPAssign() const noexcept
{
   return ( rows() * columns() * pages() >= SMP_DMATASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Load of a SIMD element of the subtensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense subtensor. The row
// index must be smaller than the number of rows and the column index must be smaller than
// the number of columns. Additionally, the column index (in case of a row-major tensor) or
// the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is
// used internally for the performance optimized evaluation of expression templates. Calling
// this function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
BLAZE_ALWAYS_INLINE typename Subtensor<MT,unaligned,CSAs...>::SIMDType
   Subtensor<MT,unaligned,CSAs...>::load( size_t k, size_t i, size_t j ) const noexcept
{
   if( isAligned_ )
      return loada( k, i, j );
   else
      return loadu( k, i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the subtensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense subtensor.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
BLAZE_ALWAYS_INLINE typename Subtensor<MT,unaligned,CSAs...>::SIMDType
   Subtensor<MT,unaligned,CSAs...>::loada( size_t k, size_t i, size_t j ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   return tensor_.loada( page()+k, row()+i, column()+j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the subtensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense subtensor.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
BLAZE_ALWAYS_INLINE typename Subtensor<MT,unaligned,CSAs...>::SIMDType
   Subtensor<MT,unaligned,CSAs...>::loadu( size_t k, size_t i, size_t j ) const noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   return tensor_.loadu( page()+k, row()+i, column()+j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Store of a SIMD element of the subtensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store of a specific SIMD element of the dense subtensor. The row
// index must be smaller than the number of rows and the column index must be smaller than the
// number of columns. Additionally, the column index (in case of a row-major tensor) or the row
// index (in case of a column-major tensor) must be a multiple of the number of values inside
// the SIMD element. This function must \b NOT be called explicitly! It is used internally
// for the performance optimized evaluation of expression templates. Calling this function
// explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
BLAZE_ALWAYS_INLINE void
   Subtensor<MT,unaligned,CSAs...>::store( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   if( isAligned_ )
      storea( k, i, j, value );
   else
      storeu( k, i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the subtensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store of a specific SIMD element of the dense subtensor.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
BLAZE_ALWAYS_INLINE void
   Subtensor<MT,unaligned,CSAs...>::storea( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   tensor_.storea( page()+k, row()+i, column()+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned store of a SIMD element of the subtensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store of a specific SIMD element of the dense subtensor.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
BLAZE_ALWAYS_INLINE void
   Subtensor<MT,unaligned,CSAs...>::storeu( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   tensor_.storeu( page()+k, row()+i, column()+j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned, non-temporal store of a SIMD element of the subtensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store of a specific SIMD element of the dense
// subtensor. The row index must be smaller than the number of rows and the column index must be
// smaller than the number of columns. Additionally, the column index (in case of a row-major
// tensor) or the row index (in case of a column-major tensor) must be a multiple of the number
// of values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
BLAZE_ALWAYS_INLINE void
   Subtensor<MT,unaligned,CSAs...>::stream( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < pages(), "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= columns(), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j % SIMDSIZE == 0UL, "Invalid column access index" );

   if( isAligned_ )
      tensor_.stream( page()+k, row()+i, column()+j, value );
   else
      tensor_.storeu( page()+k, row()+i, column()+j, value );
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >    // Type of the right-hand side dense tensor
inline auto Subtensor<MT,unaligned,CSAs...>::assign( const DenseTensor<MT2>& rhs )
   -> EnableIf_t< !VectorizedAssign_v<MT2> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages" );

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t k=0UL; k<pages(); ++k ) {
      for( size_t i=0UL; i<rows(); ++i ) {
         for( size_t j=0UL; j<jpos; j+=2UL ) {
            tensor_(page()+k,row()+i,column()+j) = (~rhs)(k,i,j);
            tensor_(page()+k,row()+i,column()+j+1UL) = (~rhs)(k,i,j+1UL);
         }
         if( jpos < columns() ) {
            tensor_(page()+k,row()+i,column()+jpos) = (~rhs)(k,i,jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the assignment of a row-major dense tensor.
//
// \param rhs The right-hand side dense tensor to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >    // Type of the right-hand side dense tensor
inline auto Subtensor<MT,unaligned,CSAs...>::assign( const DenseTensor<MT2>& rhs )
   -> EnableIf_t< VectorizedAssign_v<MT2> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages" );

   const size_t jpos( columns() & size_t(-SIMDSIZE) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

   if( useStreaming && isAligned_ &&
       rows()*columns() > ( cacheSize / ( sizeof(ElementType) * 3UL ) ) &&
       !(~rhs).isAliased( &tensor_ ) )
   {
      for( size_t k=0UL; k<pages(); ++k )
      {
         for( size_t i=0UL; i<rows(); ++i )
         {
            size_t j( 0UL );
            Iterator left( begin(i, k) );
            ConstIterator_t<MT2> right( (~rhs).begin(i, k) );

            for( ; j<jpos; j+=SIMDSIZE ) {
               left.stream( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            }
            for( ; j<columns(); ++j ) {
               *left = *right; ++left; ++right;
            }
         }
      }
   }
   else
   {
      for( size_t k=0UL; k<pages(); ++k )
      {
         for( size_t i=0UL; i<rows(); ++i )
         {
            size_t j( 0UL );
            Iterator left( begin(i, k) );
            ConstIterator_t<MT2> right( (~rhs).begin(i, k) );

            for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
               left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
               left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
               left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
               left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            }
            for( ; j<jpos; j+=SIMDSIZE ) {
               left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            }
            for( ; j<columns(); ++j ) {
               *left = *right; ++left; ++right;
            }
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >    // Type of the right-hand side dense tensor
inline auto Subtensor<MT,unaligned,CSAs...>::addAssign( const DenseTensor<MT2>& rhs )
   -> EnableIf_t< !VectorizedAddAssign_v<MT2> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages" );

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t k=0UL; k<pages(); ++k )
   {
      for( size_t i=0UL; i<rows(); ++i )
      {
         for( size_t j=0UL; j<jpos; j+=2UL ) {
            tensor_(page()+k,row()+i,column()+j) += (~rhs)(k,i,j);
            tensor_(page()+k,row()+i,column()+j+1UL) += (~rhs)(k,i,j+1UL);
         }
         if( jpos < columns() ) {
            tensor_(page()+k,row()+i,column()+jpos) += (~rhs)(k,i,jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the addition assignment of a row-major dense tensor.
//
// \param rhs The right-hand side dense tensor to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >    // Type of the right-hand side dense tensor
inline auto Subtensor<MT,unaligned,CSAs...>::addAssign( const DenseTensor<MT2>& rhs )
   -> EnableIf_t< VectorizedAddAssign_v<MT2> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages" );

   for( size_t k=0UL; k<pages(); ++k )
   {
      for( size_t i=0UL; i<rows(); ++i )
      {
         const size_t jbegin( 0UL );
         const size_t jend  ( columns() );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( jend & size_t(-SIMDSIZE) );
         BLAZE_INTERNAL_ASSERT( ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );
         Iterator left( begin(i, k) + jbegin );
         ConstIterator_t<MT2> right( (~rhs).begin(i, k) + jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<jpos; j+=SIMDSIZE ) {
            left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<jend; ++j ) {
            *left += *right; ++left; ++right;
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >    // Type of the right-hand side dense tensor
inline auto Subtensor<MT,unaligned,CSAs...>::subAssign( const DenseTensor<MT2>& rhs )
   -> EnableIf_t< !VectorizedSubAssign_v<MT2> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages" );

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t k=0UL; k<pages(); ++k )
   {
      for( size_t i=0UL; i<rows(); ++i )
      {
         for( size_t j=0UL; j<jpos; j+=2UL ) {
            tensor_(page()+k,row()+i,column()+j) -= (~rhs)(k,i,j);
            tensor_(page()+k,row()+i,column()+j+1UL) -= (~rhs)(k,i,j+1UL);
         }
         if( jpos < columns() ) {
            tensor_(page()+k,row()+i,column()+jpos) -= (~rhs)(k,i,jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the subtraction assignment of a row-major dense tensor.
//
// \param rhs The right-hand side dense tensor to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >    // Type of the right-hand side dense tensor
inline auto Subtensor<MT,unaligned,CSAs...>::subAssign( const DenseTensor<MT2>& rhs )
   -> EnableIf_t< VectorizedSubAssign_v<MT2> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages" );

   for( size_t k=0UL; k<pages(); ++k )
   {
      for( size_t i=0UL; i<rows(); ++i )
      {
         const size_t jbegin( 0UL );
         const size_t jend  ( columns() );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( jend & size_t(-SIMDSIZE) );
         BLAZE_INTERNAL_ASSERT( ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );
         Iterator left( begin(i, k) + jbegin );
         ConstIterator_t<MT2> right( (~rhs).begin(i, k) + jbegin );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<jpos; j+=SIMDSIZE ) {
            left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<jend; ++j ) {
            *left -= *right; ++left; ++right;
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
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >    // Type of the right-hand side dense tensor
inline auto Subtensor<MT,unaligned,CSAs...>::schurAssign( const DenseTensor<MT2>& rhs )
   -> EnableIf_t< !VectorizedSchurAssign_v<MT2> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages" );

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t k=0UL; k<pages(); ++k )
   {
      for( size_t i=0UL; i<rows(); ++i )
      {
         for( size_t j=0UL; j<jpos; j+=2UL ) {
            tensor_(page()+k,row()+i,column()+j) *= (~rhs)(k,i,j);
            tensor_(page()+k,row()+i,column()+j+1UL) *= (~rhs)(k,i,j+1UL);
         }
         if( jpos < columns() ) {
            tensor_(page()+k,row()+i,column()+jpos) *= (~rhs)(k,i,jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the Schur product assignment of a row-major dense tensor.
//
// \param rhs The right-hand side dense tensor for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CSAs >  // Compile time subtensor arguments
template< typename MT2 >    // Type of the right-hand side dense tensor
inline auto Subtensor<MT,unaligned,CSAs...>::schurAssign( const DenseTensor<MT2>& rhs )
   -> EnableIf_t< VectorizedSchurAssign_v<MT2> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages" );

   for( size_t k=0UL; k<pages(); ++k )
   {
      for( size_t i=0UL; i<rows(); ++i )
      {
         const size_t jpos( columns() & size_t(-SIMDSIZE) );
         BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( 0UL );
         Iterator left( begin(i, k) );
         ConstIterator_t<MT2> right( (~rhs).begin(i, k) );

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<jpos; j+=SIMDSIZE ) {
            left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<columns(); ++j ) {
            *left *= *right; ++left; ++right;
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
