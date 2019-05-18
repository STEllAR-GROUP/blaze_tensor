//=================================================================================================
/*!
//  \file blaze_tensor/math/views/dilatedsubvector/Dense.h
//  \brief DilatedSubvector specialization for dense vectors
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBVECTOR_DENSE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBVECTOR_DENSE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/Exception.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/TransExpr.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/dense/InitializerVector.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/CrossExpr.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/expressions/View.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/traits/CrossTrait.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDDiv.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/HasSIMDSub.h>
#include <blaze/math/typetraits/IsContiguous.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsRestricted.h>
#include <blaze/math/typetraits/IsSIMDCombinable.h>
#include <blaze/math/typetraits/IsSparseVector.h>
#include <blaze/math/views/Check.h>
#include <blaze/system/CacheSize.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/AlignmentCheck.h>
#include <blaze/util/Assert.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/TypeList.h>
#include <blaze/util/Types.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/typetraits/IsConst.h>
#include <iterator>

#include <blaze_tensor/math/constraints/DilatedSubvector.h>
#include <blaze_tensor/math/traits/DilatedSubvectorTrait.h>
#include <blaze_tensor/math/views/dilatedsubvector/BaseTemplate.h>
#include <blaze_tensor/math/views/dilatedsubvector/DilatedSubvectorData.h>

namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR UNALIGNED DENSE DILATEDSUBVECTORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of DilatedSubvector for unaligned dense dilatedsubvectors.
// \ingroup dilatedsubvector
//
// This specialization of DilatedSubvector adapts the class template to the requirements of unaligned
// dense dilatedsubvectors.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
class DilatedSubvector<VT,TF,true,CSAs...>
   : public View< DenseVector< DilatedSubvector<VT,TF,true,CSAs...>, TF > >
   , private DilatedSubvectorData<CSAs...>
{
 private:
   //**Type definitions****************************************************************************
   using DataType = DilatedSubvectorData<CSAs...>;               //!< The type of the DilatedSubvectorData base class.
   using Operand  = If_t< IsExpression_v<VT>, VT, VT& >;  //!< Composite data type of the vector expression.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this DilatedSubvector instance.
   using This = DilatedSubvector<VT,TF,true,CSAs...>;

   using BaseType      = DenseVector<This,TF>;          //!< Base type of this DilatedSubvector instance.
   using ViewedType    = VT;                            //!< The type viewed by this DilatedSubvector instance.
   using ResultType    = DilatedSubvectorTrait_t<VT,CSAs...>;  //!< Result type for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;   //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<VT>;             //!< Type of the dilatedsubvector elements.
   using SIMDType      = SIMDTrait_t<ElementType>;      //!< SIMD type of the dilatedsubvector elements.
   using ReturnType    = ReturnType_t<VT>;              //!< Return type for expression template evaluations
   using CompositeType = const DilatedSubvector&;              //!< Data type for composite expression templates.

   //! Reference to a constant dilatedsubvector value.
   using ConstReference = ConstReference_t<VT>;

   //! Reference to a non-constant dilatedsubvector value.
   using Reference = If_t< IsConst_v<VT>, ConstReference, Reference_t<VT> >;

   //! Pointer to a constant dilatedsubvector value.
   using ConstPointer = ConstPointer_t<VT>;

   //! Pointer to a non-constant dilatedsubvector value.
   using Pointer = If_t< IsConst_v<VT> || !HasMutableDataAccess_v<VT>, ConstPointer, Pointer_t<VT> >;
   //**********************************************************************************************

   //**DilatedSubvectorIterator class definition**********************************************************
   /*!\brief Iterator over the elements of the dense dilatedsubvector.
   */
   template< typename IteratorType >  // Type of the dense vector iterator
   class DilatedSubvectorIterator
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
      /*!\brief Default constructor of the DilatedSubvectorIterator class.
      */
      inline DilatedSubvectorIterator()
         : iterator_ (       )   // Iterator to the current dilatedsubvector element
         , dilation_ ( 0     )   // step-size of the underlying dilated subvector
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor of the DilatedSubvectorIterator class.
      //
      // \param iterator Iterator to the initial element.
      // \param isMemoryAligned Memory alignment flag.
      */
      inline DilatedSubvectorIterator( IteratorType iterator, size_t dilation )
         : iterator_ ( iterator )   // Iterator to the current dilatedsubvector element
         , dilation_ ( dilation )   // step-size of the underlying dilated subvector
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Conversion constructor from different DilatedSubvectorIterator instances.
      //
      // \param it The dilatedsubvector iterator to be copied
      */
      template< typename IteratorType2 >
      inline DilatedSubvectorIterator( const DilatedSubvectorIterator<IteratorType2>& it )
         : iterator_ ( it.base()      )  // Iterator to the current dilatedsubvector element
         , dilation_ ( it.dilation()  )  // step-size of the underlying dilated subvector
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline DilatedSubvectorIterator& operator+=( size_t inc ) {
         iterator_ += inc * dilation_;
         return *this;
      }
      //*******************************************************************************************

      //**Subtraction assignment operator**********************************************************
      /*!\brief Subtraction assignment operator.
      //
      // \param dec The decrement of the iterator.
      // \return The decremented iterator.
      */
      inline DilatedSubvectorIterator& operator-=( size_t dec ) {
         iterator_ -= dec * dilation_;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline DilatedSubvectorIterator& operator++() {
         iterator_ += dilation_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const DilatedSubvectorIterator operator++( int ) {
         return DilatedSubvectorIterator( iterator_ += dilation_, dilation_ );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline DilatedSubvectorIterator& operator--() {
         iterator_ -= dilation_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const DilatedSubvectorIterator operator--( int ) {
         return DilatedSubvectorIterator( iterator_ -= dilation_, dilation_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return Reference to the current value.
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
      /*!\brief Equality comparison between two DilatedSubvectorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const DilatedSubvectorIterator& rhs ) const {
         return iterator_ == rhs.iterator_ && dilation_ == rhs.dilation_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two DilatedSubvectorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const DilatedSubvectorIterator& rhs ) const {
         return iterator_ != rhs.iterator_ || dilation_ != rhs.dilation_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two DilatedSubvectorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const DilatedSubvectorIterator& rhs ) const {
         return iterator_ < rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two DilatedSubvectorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const DilatedSubvectorIterator& rhs ) const {
         return iterator_ > rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two DilatedSubvectorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const DilatedSubvectorIterator& rhs ) const {
         return iterator_ <= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two DilatedSubvectorIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const DilatedSubvectorIterator& rhs ) const {
         return iterator_ >= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const DilatedSubvectorIterator& rhs ) const {
         return ( iterator_ - rhs.iterator_ ) / ptrdiff_t(dilation_);
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between a DilatedSubvectorIterator and an integral value.
      //
      // \param it The iterator to be incremented.
      // \param inc The number of elements the iterator is incremented.
      // \return The incremented iterator.
      */
      friend inline const DilatedSubvectorIterator operator+( const DilatedSubvectorIterator& it, size_t inc ) {
         return DilatedSubvectorIterator( it.iterator_ + inc * it.dilation_, it.dilation_ );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between an integral value and a DilatedSubvectorIterator.
      //
      // \param inc The number of elements the iterator is incremented.
      // \param it The iterator to be incremented.
      // \return The incremented iterator.
      */
      friend inline const DilatedSubvectorIterator operator+( size_t inc, const DilatedSubvectorIterator& it ) {
         return DilatedSubvectorIterator( it.iterator_ + inc * it.dilation_, it.dilation_ );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Subtraction between a DilatedSubvectorIterator and an integral value.
      //
      // \param it The iterator to be decremented.
      // \param dec The number of elements the iterator is decremented.
      // \return The decremented iterator.
      */
      friend inline const DilatedSubvectorIterator operator-( const DilatedSubvectorIterator& it, size_t dec ) {
         return DilatedSubvectorIterator( it.iterator_ - dec * it.dilation_, it.dilation_ );
      }
      //*******************************************************************************************

      //**Base function****************************************************************************
      /*!\brief Access to the current position of the dilatedsubvector iterator.
      //
      // \return The current position of the dilatedsubvector iterator.
      */
      inline IteratorType base() const {
         return iterator_;
      }
      //*******************************************************************************************

      //**Dilation function****************************************************************************
      /*!\brief Access to the current dilation of the dilatedsubvector iterator.
      //
      // \return The dilation of the dilatedsubvector iterator.
      */
      inline size_t dilation() const {
         return dilation_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType iterator_;   //!< Iterator to the current dilatedsubvector element.
      size_t dilation_;         //!< Step-size of the underlying dilated subvector
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   //! Iterator over constant elements.
   using ConstIterator = DilatedSubvectorIterator< ConstIterator_t<VT> >;

   //! Iterator over non-constant elements.
   using Iterator = If_t< IsConst_v<VT>, ConstIterator, DilatedSubvectorIterator< Iterator_t<VT> > >;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = false;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = VT::smpAssignable;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RSAs >
   explicit inline DilatedSubvector( VT& vector, RSAs... args );

   DilatedSubvector( const DilatedSubvector& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DilatedSubvector() = default;
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator[]( size_t index );
   inline ConstReference operator[]( size_t index ) const;
   inline Reference      at( size_t index );
   inline ConstReference at( size_t index ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Iterator       begin ();
   inline ConstIterator  begin () const;
   inline ConstIterator  cbegin() const;
   inline Iterator       end   ();
   inline ConstIterator  end   () const;
   inline ConstIterator  cend  () const;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
                            inline DilatedSubvector& operator= ( const ElementType& rhs );
                            inline DilatedSubvector& operator= ( initializer_list<ElementType> list );
                            inline DilatedSubvector& operator= ( const DilatedSubvector& rhs );
   template< typename VT2 > inline DilatedSubvector& operator= ( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline DilatedSubvector& operator+=( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline DilatedSubvector& operator-=( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline DilatedSubvector& operator*=( const Vector<VT2,TF>& rhs );
   template< typename VT2 > inline DilatedSubvector& operator/=( const DenseVector<VT2,TF>& rhs );
   template< typename VT2 > inline DilatedSubvector& operator%=( const Vector<VT2,TF>& rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   using DataType::offset;
   using DataType::size;
   using DataType::dilation;

   inline VT&       operand() noexcept;
   inline const VT& operand() const noexcept;

   inline size_t spacing() const noexcept;
   inline size_t capacity() const noexcept;
   inline size_t nonZeros() const;
   inline void   reset();
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   template< typename Other > inline DilatedSubvector& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 public:
   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other >
   inline bool canAlias( const Other* alias ) const noexcept;

   template< typename VT2, bool TF2, size_t... CSAs2 >
   inline bool canAlias( const DilatedSubvector<VT2,TF2,true,CSAs2...>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename VT2, bool TF2, size_t... CSAs2 >
   inline bool isAliased( const DilatedSubvector<VT2,TF2,true,CSAs2...>* alias ) const noexcept;

   inline bool isAligned   () const noexcept { return false; }
   inline bool canSMPAssign() const noexcept;

   template< typename VT2 > inline void assign( const DenseVector <VT2,TF>& rhs );
//    template< typename VT2 > inline void assign( const SparseVector<VT2,TF>& rhs );

   template< typename VT2 > inline void addAssign( const DenseVector <VT2,TF>& rhs );
//    template< typename VT2 > inline void addAssign( const SparseVector<VT2,TF>& rhs );

   template< typename VT2 > inline void subAssign( const DenseVector <VT2,TF>& rhs );
//    template< typename VT2 > inline void subAssign( const SparseVector<VT2,TF>& rhs );

   template< typename VT2 > inline void multAssign( const DenseVector <VT2,TF>& rhs );
//    template< typename VT2 > inline void multAssign( const SparseVector<VT2,TF>& rhs );

   template< typename VT2 > inline void divAssign( const DenseVector <VT2,TF>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand vector_;        //!< The vector containing the dilatedsubvector.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename VT2, bool TF2, bool DF2, size_t... CSAs2 > friend class DilatedSubvector;
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE   ( VT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE  ( VT );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT, TF );
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
/*!\brief Constructor for unaligned dense dilatedsubvectors.
//
// \param vector The dense vector containing the dilatedsubvector.
// \param args The runtime dilatedsubvector arguments.
// \exception std::invalid_argument Invalid dilatedsubvector specification.
//
// By default, the provided dilatedsubvector arguments are checked at runtime. In case the dilatedsubvector is
// not properly specified (i.e. if the specified offset is greater than the size of the given
// vector or the dilatedsubvector is specified beyond the size of the vector) a \a std::invalid_argument
// exception is thrown. The checks can be skipped by providing the optional \a blaze::unchecked
// argument.
*/
template< typename VT         // Type of the dense vector
        , bool TF             // Transpose flag
        , size_t... CSAs >    // Compile time dilatedsubvector arguments
template< typename... RSAs >  // Runtime dilatedsubvector arguments
inline DilatedSubvector<VT,TF,true,CSAs...>::DilatedSubvector( VT& vector, RSAs... args )
   : DataType  ( args... )  // Base class initialization
   , vector_   ( vector  )  // The vector containing the dilatedsubvector
{
   if( !Contains_v< TypeList<RSAs...>, Unchecked > ) {
      if( offset() + ( size() - 1 ) * dilation() + 1 > vector.size() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid dilatedsubvector specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT(
         offset() + ( size() - 1 ) * dilation() + 1 <= vector.size(),
         "Invalid dilatedsubvector specification" );
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
/*!\brief Subscript operator for the direct access to the dilatedsubvector elements.
//
// \param index Access index. The index must be smaller than the number of dilatedsubvector elements.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline typename DilatedSubvector<VT,TF,true,CSAs...>::Reference
   DilatedSubvector<VT,TF,true,CSAs...>::operator[]( size_t index )
{
   BLAZE_USER_ASSERT( index < size(), "Invalid dilatedsubvector access index" );
   return vector_[offset()+index*dilation()];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subscript operator for the direct access to the dilatedsubvector elements.
//
// \param index Access index. The index must be smaller than the number of dilatedsubvector elements.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline typename DilatedSubvector<VT,TF,true,CSAs...>::ConstReference
   DilatedSubvector<VT,TF,true,CSAs...>::operator[]( size_t index ) const
{
   BLAZE_USER_ASSERT( index < size(), "Invalid dilatedsubvector access index" );
   return const_cast<const VT&>( vector_ )[offset()+index*dilation()];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the dilatedsubvector elements.
//
// \param index Access index. The index must be smaller than the number of dilatedsubvector elements.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid dilatedsubvector access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline typename DilatedSubvector<VT,TF,true,CSAs...>::Reference
   DilatedSubvector<VT,TF,true,CSAs...>::at( size_t index )
{
   if( index >= size() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid dilatedsubvector access index" );
   }
   return (*this)[index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the dilatedsubvector elements.
//
// \param index Access index. The index must be smaller than the number of dilatedsubvector elements.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid dilatedsubvector access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline typename DilatedSubvector<VT,TF,true,CSAs...>::ConstReference
   DilatedSubvector<VT,TF,true,CSAs...>::at( size_t index ) const
{
   if( index >= size() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid dilatedsubvector access index" );
   }
   return (*this)[index];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the dilatedsubvector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline typename DilatedSubvector<VT,TF,true,CSAs...>::Pointer
   DilatedSubvector<VT,TF,true,CSAs...>::data() noexcept
{
   return vector_.data() + offset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the dilatedsubvector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline typename DilatedSubvector<VT,TF,true,CSAs...>::ConstPointer
   DilatedSubvector<VT,TF,true,CSAs...>::data() const noexcept
{
   return vector_.data() + offset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the dilatedsubvector.
//
// \return Iterator to the first element of the dilatedsubvector.
//
// This function returns an iterator to the first element of the dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline typename DilatedSubvector<VT,TF,true,CSAs...>::Iterator
   DilatedSubvector<VT,TF,true,CSAs...>::begin()
{
   return Iterator( vector_.begin() + offset(), dilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the dilatedsubvector.
//
// \return Iterator to the first element of the dilatedsubvector.
//
// This function returns an iterator to the first element of the dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline typename DilatedSubvector<VT,TF,true,CSAs...>::ConstIterator
   DilatedSubvector<VT,TF,true,CSAs...>::begin() const
{
   return ConstIterator( vector_.cbegin() + offset(), dilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the dilatedsubvector.
//
// \return Iterator to the first element of the dilatedsubvector.
//
// This function returns an iterator to the first element of the dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline typename DilatedSubvector<VT,TF,true,CSAs...>::ConstIterator
   DilatedSubvector<VT,TF,true,CSAs...>::cbegin() const
{
   return ConstIterator( vector_.cbegin() + offset(), dilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the dilatedsubvector.
//
// \return Iterator just past the last element of the dilatedsubvector.
//
// This function returns an iterator just past the last element of the dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline typename DilatedSubvector<VT,TF,true,CSAs...>::Iterator
   DilatedSubvector<VT,TF,true,CSAs...>::end()
{
   return Iterator( vector_.begin() + offset() + size() * dilation(), dilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the dilatedsubvector.
//
// \return Iterator just past the last element of the dilatedsubvector.
//
// This function returns an iterator just past the last element of the dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline typename DilatedSubvector<VT,TF,true,CSAs...>::ConstIterator
   DilatedSubvector<VT,TF,true,CSAs...>::end() const
{
   return ConstIterator( vector_.cbegin() + offset() + size() * dilation(), dilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the dilatedsubvector.
//
// \return Iterator just past the last element of the dilatedsubvector.
//
// This function returns an iterator just past the last element of the dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline typename DilatedSubvector<VT,TF,true,CSAs...>::ConstIterator
   DilatedSubvector<VT,TF,true,CSAs...>::cend() const
{
   return ConstIterator( vector_.cbegin() + offset() + size() * dilation(), dilation() );
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
/*!\brief Homogenous assignment to all dilatedsubvector elements.
//
// \param rhs Scalar value to be assigned to all dilatedsubvector elements.
// \return Reference to the assigned dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline DilatedSubvector<VT,TF,true,CSAs...>&
   DilatedSubvector<VT,TF,true,CSAs...>::operator=( const ElementType& rhs )
{
   const size_t iend( offset() + size() * dilation() );
   decltype(auto) left( derestrict( vector_ ) );

   for( size_t i=offset(); i<iend; i+=dilation() ) {
      if( !IsRestricted_v<VT> || trySet( vector_, i, rhs ) )
         left[i] = rhs;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all dilatedsubvector elements.
//
// \param list The initializer list.
// \return Reference to the assigned dilatedsubvector.
// \exception std::invalid_argument Invalid assignment to dilatedsubvector.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// This assignment operator offers the option to directly assign to all elements of the dilatedsubvector
// by means of an initializer list. The dilatedsubvector elements are assigned the values from the given
// initializer list. Missing values are reset to their default state. Note that in case the size
// of the initializer list exceeds the size of the dilatedsubvector, a \a std::invalid_argument exception
// is thrown. Also, if the underlying vector \a VT is restricted and the assignment would violate
// an invariant of the vector, a \a std::invalid_argument exception is thrown.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline DilatedSubvector<VT,TF,true,CSAs...>&
   DilatedSubvector<VT,TF,true,CSAs...>::operator=( initializer_list<ElementType> list )
{
   if( list.size() > size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to dilatedsubvector" );
   }

   if( IsRestricted_v<VT> ) {
      const InitializerVector<ElementType,TF> tmp( list, size() );
      if( !tryAssign( vector_, tmp, offset() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
      }
   }

   decltype(auto) left( derestrict( *this ) );

   std::fill( std::copy( list.begin(), list.end(), left.begin() ), left.end(), ElementType() );

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for DilatedSubvector.
//
// \param rhs Dense dilatedsubvector to be copied.
// \return Reference to the assigned dilatedsubvector.
// \exception std::invalid_argument DilatedSubvector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two dilatedsubvectors don't match, a \a std::invalid_argument
// exception is thrown.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline DilatedSubvector<VT,TF,true,CSAs...>&
   DilatedSubvector<VT,TF,true,CSAs...>::operator=( const DilatedSubvector& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( &rhs == this || ( &vector_ == &rhs.vector_ && offset() == rhs.offset() && dilation() == rhs.dilation() ) )
      return *this;

   if( size() != rhs.size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "DilatedSubvector sizes do not match" );
   }

   if( !tryAssign( vector_, rhs, offset() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( rhs.canAlias( &vector_ ) ) {
      const ResultType tmp( rhs );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different vectors.
//
// \param rhs Vector to be assigned.
// \return Reference to the assigned dilatedsubvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument
// exception is thrown.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename VT2 >    // Type of the right-hand side vector
inline DilatedSubvector<VT,TF,true,CSAs...>&
   DilatedSubvector<VT,TF,true,CSAs...>::operator=( const Vector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_t<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<VT>, CompositeType_t<VT2>, const VT2& >;
   Right right( ~rhs );

   if( !tryAssign( vector_, right, offset() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &vector_ ) ) {
      const ResultType_t<VT2> tmp( right );
      smpAssign( left, tmp );
   }
   else {
      if( IsSparseVector_v<VT2> )
         reset();
      smpAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a vector (\f$ \vec{a}+=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be added to the dense dilatedsubvector.
// \return Reference to the assigned dilatedsubvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename VT2 >    // Type of the right-hand side vector
inline DilatedSubvector<VT,TF,true,CSAs...>&
   DilatedSubvector<VT,TF,true,CSAs...>::operator+=( const Vector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_t<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<VT>, CompositeType_t<VT2>, const VT2& >;
   Right right( ~rhs );

   if( !tryAddAssign( vector_, right, offset() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &vector_ ) ) {
      const ResultType_t<VT2> tmp( right );
      smpAddAssign( left, tmp );
   }
   else {
      smpAddAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a vector (\f$ \vec{a}-=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be subtracted from the dense dilatedsubvector.
// \return Reference to the assigned dilatedsubvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename VT2 >    // Type of the right-hand side vector
inline DilatedSubvector<VT,TF,true,CSAs...>&
   DilatedSubvector<VT,TF,true,CSAs...>::operator-=( const Vector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_t<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<VT>, CompositeType_t<VT2>, const VT2& >;
   Right right( ~rhs );

   if( !trySubAssign( vector_, right, offset() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &vector_ ) ) {
      const ResultType_t<VT2> tmp( right );
      smpSubAssign( left, tmp );
   }
   else {
      smpSubAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication of a vector
//        (\f$ \vec{a}*=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be multiplied with the dense dilatedsubvector.
// \return Reference to the assigned dilatedsubvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename VT2 >    // Type of the right-hand side vector
inline DilatedSubvector<VT,TF,true,CSAs...>&
   DilatedSubvector<VT,TF,true,CSAs...>::operator*=( const Vector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_t<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<VT>, CompositeType_t<VT2>, const VT2& >;
   Right right( ~rhs );

   if( !tryMultAssign( vector_, right, offset() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &vector_ ) ) {
      const ResultType_t<VT2> tmp( right );
      smpMultAssign( left, tmp );
   }
   else {
      smpMultAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a dense vector (\f$ \vec{a}/=\vec{b} \f$).
//
// \param rhs The right-hand side dense vector divisor.
// \return Reference to the assigned dilatedsubvector.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename VT2 >    // Type of the right-hand side dense vector
inline DilatedSubvector<VT,TF,true,CSAs...>&
   DilatedSubvector<VT,TF,true,CSAs...>::operator/=( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_t<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<VT2> );

   if( size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<VT>, CompositeType_t<VT2>, const VT2& >;
   Right right( ~rhs );

   if( !tryDivAssign( vector_, right, offset() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &vector_ ) ) {
      const ResultType_t<VT2> tmp( right );
      smpDivAssign( left, tmp );
   }
   else {
      smpDivAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Cross product assignment operator for the multiplication of a vector
//        (\f$ \vec{a}\times=\vec{b} \f$).
//
// \param rhs The right-hand side vector for the cross product.
// \return Reference to the assigned dilatedsubvector.
// \exception std::invalid_argument Invalid vector size for cross product.
// \exception std::invalid_argument Invalid assignment to restricted vector.
//
// In case the current size of any of the two vectors is not equal to 3, a \a std::invalid_argument
// exception is thrown.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename VT2 >    // Type of the right-hand side vector
inline DilatedSubvector<VT,TF,true,CSAs...>&
   DilatedSubvector<VT,TF,true,CSAs...>::operator%=( const Vector<VT2,TF>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_t<VT2>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<VT2> );

   using CrossType = CrossTrait_t< ResultType, ResultType_t<VT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( CrossType );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( CrossType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( CrossType );

   if( size() != 3UL || (~rhs).size() != 3UL ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid vector size for cross product" );
   }

   const CrossType tmp( *this % (~rhs) );

   if( !tryAssign( vector_, tmp, offset() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted vector" );
   }

   decltype(auto) left( derestrict( *this ) );

   assign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( vector_ ), "Invariant violation detected" );

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
/*!\brief Returns the vector containing the dilatedsubvector.
//
// \return The vector containing the dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline VT& DilatedSubvector<VT,TF,true,CSAs...>::operand() noexcept
{
   return vector_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the vector containing the dilatedsubvector.
//
// \return The vector containing the dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline const VT& DilatedSubvector<VT,TF,true,CSAs...>::operand() const noexcept
{
   return vector_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the minimum capacity of the dense dilatedsubvector.
//
// \return The minimum capacity of the dense dilatedsubvector.
//
// This function returns the minimum capacity of the dense dilatedsubvector, which corresponds to the
// current size plus padding.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline size_t DilatedSubvector<VT,TF,true,CSAs...>::spacing() const noexcept
{
   return vector_.spacing() - offset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense dilatedsubvector.
//
// \return The maximum capacity of the dense dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline size_t DilatedSubvector<VT,TF,true,CSAs...>::capacity() const noexcept
{
   return vector_.capacity() - offset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the dilatedsubvector.
//
// \return The number of non-zero elements in the dilatedsubvector.
//
// Note that the number of non-zero elements is always less than or equal to the current size
// of the dilatedsubvector.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline size_t DilatedSubvector<VT,TF,true,CSAs...>::nonZeros() const
{
   size_t nonzeros( 0 );

   const size_t iend( offset() + size()*dilation() );
   for( size_t i=offset(); i<iend; i+=dilation() ) {
      if( !isDefault( vector_[i] ) )
         ++nonzeros;
   }

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
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline void DilatedSubvector<VT,TF,true,CSAs...>::reset()
{
   using blaze::clear;

   const size_t iend( offset() + size()*dilation() );
   for( size_t i=offset(); i<iend; i+=dilation() )
      clear( vector_[i] );
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
/*!\brief Scaling of the dense dilatedsubvector by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the dilatedsubvector scaling.
// \return Reference to the dense dilatedsubvector.
//
// This function scales the dilatedsubvector by applying the given scalar value \a scalar to each
// element of the dilatedsubvector. For built-in and \c complex data types it has the same effect
// as using the multiplication assignment operator.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename Other >  // Data type of the scalar value
inline DilatedSubvector<VT,TF,true,CSAs...>&
   DilatedSubvector<VT,TF,true,CSAs...>::scale( const Other& scalar )
{
   const size_t iend( offset() + size()*dilation() );
   for( size_t i=offset(); i<iend; i+=dilation() )
      vector_[i] *= scalar;
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
/*!\brief Returns whether the dense dilatedsubvector can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense dilatedsubvector, \a false if not.
//
// This function returns whether the given address can alias with the dense dilatedsubvector.
// In contrast to the isAliased() function this function is allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename Other >  // Data type of the foreign expression
inline bool DilatedSubvector<VT,TF,true,CSAs...>::canAlias( const Other* alias ) const noexcept
{
   return vector_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense dilatedsubvector can alias with the given dense dilatedsubvector \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense dilatedsubvector, \a false if not.
//
// This function returns whether the given address can alias with the dense dilatedsubvector.
// In contrast to the isAliased() function this function is allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT        // Type of the dense vector
        , bool TF            // Transpose flag
        , size_t... CSAs >   // Compile time dilatedsubvector arguments
template< typename VT2       // Data type of the foreign dense dilatedsubvector
        , bool TF2           // Transpose flag of the foreign dense dilatedsubvector
        , size_t... CSAs2 >  // Compile time dilatedsubvector arguments of the foreign dense dilatedsubvector
inline bool
   DilatedSubvector<VT,TF,true,CSAs...>::canAlias( const DilatedSubvector<VT2,TF2,true,CSAs2...>* alias ) const noexcept
{
   return ( vector_.isAliased( &alias->vector_ ) &&
            ( offset() + size()*dilation() > alias->offset() ) &&
            ( offset() < alias->offset() + (alias->size() - 1) * alias->dilation() + 1) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense dilatedsubvector is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense dilatedsubvector, \a false if not.
//
// This function returns whether the given address is aliased with the dense dilatedsubvector.
// In contrast to the canAlias() function this function is not allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename Other >  // Data type of the foreign expression
inline bool DilatedSubvector<VT,TF,true,CSAs...>::isAliased( const Other* alias ) const noexcept
{
   return vector_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense dilatedsubvector is aliased with the given dense dilatedsubvector \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense dilatedsubvector, \a false if not.
//
// This function returns whether the given address is aliased with the dense dilatedsubvector.
// In contrast to the canAlias() function this function is not allowed to use compile time
// expressions to optimize the evaluation.
*/
template< typename VT        // Type of the dense vector
        , bool TF            // Transpose flag
        , size_t... CSAs >   // Compile time dilatedsubvector arguments
template< typename VT2       // Data type of the foreign dense dilatedsubvector
        , bool TF2           // Transpose flag of the foreign dense dilatedsubvector
        , size_t... CSAs2 >  // Compile time dilatedsubvector arguments of the foreign dense dilatedsubvector
inline bool
   DilatedSubvector<VT,TF,true,CSAs...>::isAliased( const DilatedSubvector<VT2,TF2,true,CSAs2...>* alias ) const noexcept
{
   return ( vector_.isAliased( &alias->vector_ ) &&
            ( offset() + size()*dilation() > alias->offset() ) &&
            ( offset() < alias->offset() + (alias->size() - 1) * alias->dilation() + 1) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dilatedsubvector can be used in SMP assignments.
//
// \return \a true in case the dilatedsubvector can be used in SMP assignments, \a false if not.
//
// This function returns whether the dilatedsubvector can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current size of the
// dilatedsubvector).
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
inline bool DilatedSubvector<VT,TF,true,CSAs...>::canSMPAssign() const noexcept
{
   return ( size() > SMP_DVECASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************



//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename VT2 >    // Type of the right-hand side dense vector
inline void DilatedSubvector<VT,TF,true,CSAs...>::assign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2 ) {
      (*this)[i  ] = (~rhs)[i    ];
      (*this)[i+1] = (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      (*this)[ipos] = (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************



//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
// template< typename VT       // Type of the dense vector
//         , bool TF           // Transpose flag
//         , size_t... CSAs >  // Compile time dilatedsubvector arguments
// template< typename VT2 >    // Type of the right-hand side sparse vector
// inline void DilatedSubvector<VT,TF,true,CSAs...>::assign( const SparseVector<VT2,TF>& rhs )
// {
//    BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );
//
//    for( ConstIterator_t<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
//       vector_[offset()+element->index()] = element->value();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename VT2 >    // Type of the right-hand side dense vector
inline void DilatedSubvector<VT,TF,true,CSAs...>::addAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      (*this)[i  ] += (~rhs)[i    ];
      (*this)[i+1] += (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      (*this)[ipos] += (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
// template< typename VT       // Type of the dense vector
//         , bool TF           // Transpose flag
//         , size_t... CSAs >  // Compile time dilatedsubvector arguments
// template< typename VT2 >    // Type of the right-hand side sparse vector
// inline void DilatedSubvector<VT,TF,true,CSAs...>::addAssign( const SparseVector<VT2,TF>& rhs )
// {
//    BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );
//
//    for( ConstIterator_t<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
//       vector_[offset()+element->index()] += element->value();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename VT2 >    // Type of the right-hand side dense vector
inline void DilatedSubvector<VT,TF,true,CSAs...>::subAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      (*this)[i  ] -= (~rhs)[i    ];
      (*this)[i+1] -= (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      (*this)[ipos] -= (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
// template< typename VT       // Type of the dense vector
//         , bool TF           // Transpose flag
//         , size_t... CSAs >  // Compile time dilatedsubvector arguments
// template< typename VT2 >    // Type of the right-hand side sparse vector
// inline void DilatedSubvector<VT,TF,true,CSAs...>::subAssign( const SparseVector<VT2,TF>& rhs )
// {
//    BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );
//
//    for( ConstIterator_t<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element )
//       vector_[offset()+element->index()] -= element->value();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the multiplication assignment of a dense vector.
//
// \param rhs The right-hand side dense vector to be multiplied.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename VT2 >    // Type of the right-hand side dense vector
inline void DilatedSubvector<VT,TF,true,CSAs...>::multAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      (*this)[i  ] *= (~rhs)[i    ];
      (*this)[i+1] *= (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      (*this)[ipos] *= (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the multiplication assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be multiplied.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
// template< typename VT       // Type of the dense vector
//         , bool TF           // Transpose flag
//         , size_t... CSAs >  // Compile time dilatedsubvector arguments
// template< typename VT2 >    // Type of the right-hand side sparse vector
// inline void DilatedSubvector<VT,TF,true,CSAs...>::multAssign( const SparseVector<VT2,TF>& rhs )
// {
//    using blaze::reset;
//
//    BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );
//
//    size_t i( 0UL );
//
//    for( ConstIterator_t<VT2> element=(~rhs).begin(); element!=(~rhs).end(); ++element ) {
//       const size_t index( element->index() );
//       for( ; i<index; ++i )
//          reset( vector_[offset()+i] );
//       vector_[offset()+i] *= element->value();
//       ++i;
//    }
//
//    for( ; i<size(); ++i ) {
//       reset( vector_[offset()+i] );
//    }
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the division assignment of a dense vector.
//
// \param rhs The right-hand side dense vector divisor.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT       // Type of the dense vector
        , bool TF           // Transpose flag
        , size_t... CSAs >  // Compile time dilatedsubvector arguments
template< typename VT2 >    // Type of the right-hand side dense vector
inline void DilatedSubvector<VT,TF,true,CSAs...>::divAssign( const DenseVector<VT2,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size() == (~rhs).size(), "Invalid vector sizes" );

   const size_t ipos( size() & size_t(-2) );
   for( size_t i=0UL; i<ipos; i+=2UL ) {
      (*this)[i  ] /= (~rhs)[i    ];
      (*this)[i+1] /= (~rhs)[i+1UL];
   }
   if( ipos < size() ) {
      (*this)[ipos] /= (~rhs)[ipos];
   }
}
/*! \endcond */
//*************************************************************************************************
} // namespace blaze

#endif
