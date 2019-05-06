//=================================================================================================
/*!
//  \file blaze_tensor/math/views/DilatedSubmatrix/Dense.h
//  \brief DilatedSubmatrix specialization for dense matrices
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBMATRIX_DENSE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBMATRIX_DENSE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <algorithm>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/ColumnMajorMatrix.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/RowMajorMatrix.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/TransExpr.h>
#include <blaze/math/constraints/UniTriangular.h>
#include <blaze/math/dense/InitializerMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
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
#include <blaze/math/typetraits/IsSparseMatrix.h>
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

#include <blaze_tensor/math/constraints/DilatedSubmatrix.h>
#include <blaze_tensor/math/traits/DilatedSubmatrixTrait.h>
#include <blaze_tensor/math/views/dilatedsubmatrix/BaseTemplate.h>
#include <blaze_tensor/math/views/dilatedsubmatrix/DilatedSubmatrixData.h>

namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR UNALIGNED ROW-MAJOR DENSE DILATEDSUBMATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of DilatedSubmatrix for unaligned row-major dense dilatedsubmatrices.
// \ingroup DilatedSubmatrix
//
// This Specialization of DilatedSubmatrix adapts the class template to the requirements of unaligned
// row-major dense dilatedsubmatrices.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
class DilatedSubmatrix<MT,false,true,CSAs...>
   : public View< DenseMatrix< DilatedSubmatrix<MT,false,true,CSAs...>, false > >
   , private DilatedSubmatrixData<CSAs...>
{
 private:
   //**Type definitions****************************************************************************
   using DataType = DilatedSubmatrixData<CSAs...>;               //!< The type of the DilatedSubmatrixData base class.
   using Operand  = If_t< IsExpression_v<MT>, MT, MT& >;  //!< Composite data type of the matrix expression.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT1, typename MT2 >
   static constexpr bool EnforceEvaluation_v =
      ( IsRestricted_v<MT1> && RequiresEvaluation_v<MT2> );
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this DilatedSubmatrix instance.
   using This = DilatedSubmatrix<MT,false,true,CSAs...>;

   using BaseType      = DenseMatrix<This,false>;       //!< Base type of this DilatedSubmatrix instance.
   using ViewedType    = MT;                            //!< The type viewed by this DilatedSubmatrix instance.
   using ResultType    = DilatedSubmatrixTrait_t<MT,CSAs...>;  //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;    //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;   //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<MT>;             //!< Type of the DilatedSubmatrix elements.
   using SIMDType      = SIMDTrait_t<ElementType>;      //!< SIMD type of the DilatedSubmatrix elements.
   using ReturnType    = ReturnType_t<MT>;              //!< Return type for expression template evaluations
   using CompositeType = const DilatedSubmatrix&;              //!< Data type for composite expression templates.

   //! Reference to a constant DilatedSubmatrix value.
   using ConstReference = ConstReference_t<MT>;

   //! Reference to a non-constant DilatedSubmatrix value.
   using Reference = If_t< IsConst_v<MT>, ConstReference, Reference_t<MT> >;

   //! Pointer to a constant DilatedSubmatrix value.
   using ConstPointer = ConstPointer_t<MT>;

   //! Pointer to a non-constant DilatedSubmatrix value.
   using Pointer = If_t< IsConst_v<MT> || !HasMutableDataAccess_v<MT>, ConstPointer, Pointer_t<MT> >;
   //**********************************************************************************************

   //**DilatedSubmatrixIterator class definition**********************************************************
   /*!\brief Iterator over the elements of the sparse DilatedSubmatrix.
   */
   template< typename IteratorType >  // Type of the dense matrix iterator
   class DilatedSubmatrixIterator
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
      /*!\brief Default constructor of the DilatedSubmatrixIterator class.
      */
      inline DilatedSubmatrixIterator()
         : iterator_ (       )  // Iterator to the current DilatedSubmatrix element
         , rowdilation_ ( 1     ) // row step-size of the underlying dilated submatrix
         , columndilation_ ( 1     ) // column step-size of the underlying dilated submatrix
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor of the DilatedSubmatrixIterator class.
      //
      // \param iterator Iterator to the initial element.
      // \param isMemoryAligned Memory alignment flag.
      */
      inline DilatedSubmatrixIterator( IteratorType iterator, size_t rowdilation, size_t columndilation)
         : iterator_ ( iterator        )  // Iterator to the current DilatedSubmatrix element
         , rowdilation_ ( rowdilation )   // row step-size of the underlying dilated submatrix
         , columndilation_ ( columndilation )   // column step-size of the underlying dilated submatrix
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Conversion constructor from different DilatedSubmatrixIterator instances.
      //
      // \param it The DilatedSubmatrix iterator to be copied.
      */
      template< typename IteratorType2 >
      inline DilatedSubmatrixIterator( const DilatedSubmatrixIterator<IteratorType2>& it )
         : iterator_ ( it.base()      )  // Iterator to the current DilatedSubmatrix element
         , rowdilation_ ( it.rowdilation() )   // row step-size of the underlying dilated submatrix
         , columndilation_ ( it.columndilation() )   // column step-size of the underlying dilated submatrix
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline DilatedSubmatrixIterator& operator+=( size_t inc ) {
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
      inline DilatedSubmatrixIterator& operator-=( size_t dec ) {
         iterator_ -= dec * columndilation_;
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline DilatedSubmatrixIterator& operator++() {
         iterator_+= columndilation_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const DilatedSubmatrixIterator operator++( int ) {
         return DilatedSubmatrixIterator( iterator_+=columndilation_, rowdilation, columndilation );
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline DilatedSubmatrixIterator& operator--() {
         iterator_-= columndilation_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const DilatedSubmatrixIterator operator--( int ) {
         return DilatedSubmatrixIterator( iterator_-=columndilation_, rowdilation, columndilation );
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
      /*!\brief Equality comparison between two DilatedSubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      inline bool operator==( const DilatedSubmatrixIterator& rhs ) const {
         return iterator_ == rhs.iterator_  && rowdilation_ == rhs.rowdilation_ &&
            columndilation_ == rhs.columndilation_;
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two DilatedSubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      inline bool operator!=( const DilatedSubmatrixIterator& rhs ) const {
         return iterator_ != rhs.iterator_  || rowdilation_ != rhs.rowdilation_ ||
            columndilation_ != rhs.columndilation_;
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two DilatedSubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      inline bool operator<( const DilatedSubmatrixIterator& rhs ) const {
         return iterator_ < rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two DilatedSubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      inline bool operator>( const DilatedSubmatrixIterator& rhs ) const {
         return iterator_ > rhs.iterator_;
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two DilatedSubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      inline bool operator<=( const DilatedSubmatrixIterator& rhs ) const {
         return iterator_ <= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two DilatedSubmatrixIterator objects.
      //
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      inline bool operator>=( const DilatedSubmatrixIterator& rhs ) const {
         return iterator_ >= rhs.iterator_;
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const DilatedSubmatrixIterator& rhs ) const {
         return (iterator_ - rhs.iterator_)/ptrdiff_t(columndilation_);
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between a DilatedSubmatrixIterator and an integral value.
      //
      // \param it The iterator to be incremented.
      // \param inc The number of elements the iterator is incremented.
      // \return The incremented iterator.
      */
      friend inline const DilatedSubmatrixIterator operator+( const DilatedSubmatrixIterator& it, size_t inc ) {
         return DilatedSubmatrixIterator( it.iterator_ + inc*it.columndilation_, it.rowdilation_, it.columndilation_ );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between an integral value and a DilatedSubmatrixIterator.
      //
      // \param inc The number of elements the iterator is incremented.
      // \param it The iterator to be incremented.
      // \return The incremented iterator.
      */
      friend inline const DilatedSubmatrixIterator operator+( size_t inc, const DilatedSubmatrixIterator& it ) {
         return DilatedSubmatrixIterator( it.iterator_ + inc*it.columndilation_, it.rowdilation_, it.columndilation_ );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Subtraction between a DilatedSubmatrixIterator and an integral value.
      //
      // \param it The iterator to be decremented.
      // \param dec The number of elements the iterator is decremented.
      // \return The decremented iterator.
      */
      friend inline const DilatedSubmatrixIterator operator-( const DilatedSubmatrixIterator& it, size_t dec ) {
         return DilatedSubmatrixIterator( it.iterator_ - dec*it.columndilation_, it.rowdilation_, it.columndilation_  );
      }
      //*******************************************************************************************

      //**Base function****************************************************************************
      /*!\brief Access to the current position of the DilatedSubmatrix iterator.
      //
      // \return The current position of the DilatedSubmatrix iterator.
      */
      inline IteratorType base() const {
         return iterator_;
      }
      //*******************************************************************************************

      //**RowDilation function*********************************************************************
      /*!\brief Access to the iterator's memory alignment flag.
      //
      // \return The row dilation of the dilatedsubvector iterator.
      */
      inline size_t rowdilation() const noexcept {
         return rowdilation_;
      }
      //*******************************************************************************************

      //**ColumnDilation function******************************************************************
      /*!\brief Access to the iterator's memory alignment flag.
      //
      // \return The row dilation of the dilatedsubvector iterator.
      */
      inline size_t columndilation() const noexcept {
         return columndilation_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType iterator_;   //!< Iterator to the current DilatedSubmatrix element.
      size_t rowdilation_;         //!< Row step-size of the underlying dilated submatrix
      size_t columndilation_;         //!< Column step-size of the underlying dilated submatrix
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   //! Iterator over constant elements.
   using ConstIterator = DilatedSubmatrixIterator< ConstIterator_t<MT> >;

   //! Iterator over non-constant elements.
   using Iterator = If_t< IsConst_v<MT>, ConstIterator, DilatedSubmatrixIterator< Iterator_t<MT> > >;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = false;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = MT::smpAssignable;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RSAs >
   explicit inline DilatedSubmatrix( MT& matrix, RSAs... args );

   DilatedSubmatrix( const DilatedSubmatrix& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DilatedSubmatrix() = default;
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator()( size_t i, size_t j );
   inline ConstReference operator()( size_t i, size_t j ) const;
   inline Reference      at( size_t i, size_t j );
   inline ConstReference at( size_t i, size_t j ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   //inline Pointer        data  ( size_t i ) noexcept;
   //inline ConstPointer   data  ( size_t i ) const noexcept;
   inline Iterator       begin ( size_t i );
   inline ConstIterator  begin ( size_t i ) const;
   inline ConstIterator  cbegin( size_t i ) const;
   inline Iterator       end   ( size_t i );
   inline ConstIterator  end   ( size_t i ) const;
   inline ConstIterator  cend  ( size_t i ) const;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline DilatedSubmatrix& operator=( const ElementType& rhs );
   inline DilatedSubmatrix& operator=( initializer_list< initializer_list<ElementType> > list );
   inline DilatedSubmatrix& operator=( const DilatedSubmatrix& rhs );

   template< typename MT2, bool SO2 >
   inline DilatedSubmatrix& operator=( const Matrix<MT2,SO2>& rhs );

   template< typename MT2, bool SO2 >
   inline auto operator+=( const Matrix<MT2,SO2>& rhs )
      -> DisableIf_t< EnforceEvaluation_v<MT,MT2>, DilatedSubmatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator+=( const Matrix<MT2,SO2>& rhs )
      -> EnableIf_t< EnforceEvaluation_v<MT,MT2>, DilatedSubmatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator-=( const Matrix<MT2,SO2>& rhs )
      -> DisableIf_t< EnforceEvaluation_v<MT,MT2>, DilatedSubmatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator-=( const Matrix<MT2,SO2>& rhs )
      -> EnableIf_t< EnforceEvaluation_v<MT,MT2>, DilatedSubmatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator%=( const Matrix<MT2,SO2>& rhs )
      -> DisableIf_t< EnforceEvaluation_v<MT,MT2>, DilatedSubmatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator%=( const Matrix<MT2,SO2>& rhs )
      -> EnableIf_t< EnforceEvaluation_v<MT,MT2>, DilatedSubmatrix& >;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   using DataType::row;
   using DataType::column;
   using DataType::rows;
   using DataType::columns;
   using DataType::rowdilation;
   using DataType::columndilation;

   inline MT&       operand() noexcept;
   inline const MT& operand() const noexcept;

   inline size_t spacing() const noexcept;
   inline size_t capacity() const noexcept;
   inline size_t capacity( size_t i ) const noexcept;
   inline size_t nonZeros() const;
   inline size_t nonZeros( size_t i ) const;
   inline void   reset();
   inline void   reset( size_t i );
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   inline DilatedSubmatrix& transpose();
   inline DilatedSubmatrix& ctranspose();

   template< typename Other > inline DilatedSubmatrix& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 public:
   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other >
   inline bool canAlias( const Other* alias ) const noexcept;

   template< typename MT2, bool SO2, size_t... CSAs2 >
   inline bool canAlias( const DilatedSubmatrix<MT2,SO2,true,CSAs2...>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename MT2, bool SO2, size_t... CSAs2 >
   inline bool isAliased( const DilatedSubmatrix<MT2,SO2,true,CSAs2...>* alias ) const noexcept;

   inline bool canSMPAssign() const noexcept;


   template< typename MT2 > inline void assign( const DenseMatrix<MT2,false>& rhs );
   template< typename MT2 > inline void assign( const DenseMatrix<MT2,true>& rhs );
   template< typename MT2 > inline void addAssign( const DenseMatrix<MT2,false>& rhs );
   template< typename MT2 > inline void addAssign( const DenseMatrix<MT2,true>& rhs );
   template< typename MT2 > inline void subAssign( const DenseMatrix<MT2,false>& rhs );
   template< typename MT2 > inline void subAssign( const DenseMatrix<MT2,true>& rhs );
   template< typename MT2 > inline void schurAssign( const DenseMatrix<MT2,false>& rhs );
   template< typename MT2 > inline void schurAssign( const DenseMatrix<MT2,true>& rhs );


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
   Operand matrix_;        //!< The matrix containing the DilatedSubmatrix.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename MT2, bool SO2, bool DF2, size_t... CSAs2 > friend class DilatedSubmatrix;
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE   ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SUBMATRIX_TYPE   ( MT );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT );
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
/*!\brief Constructor for unaligned row-major dense submatrices.
//
// \param matrix The dense matrix containing the DilatedSubmatrix.
// \param args The runtime DilatedSubmatrix arguments.
// \exception std::invalid_argument Invalid DilatedSubmatrix specification.
//
// By default, the provided DilatedSubmatrix arguments are checked at runtime. In case the DilatedSubmatrix is
// not properly specified (i.e. if the specified DilatedSubmatrix is not contained in the given dense
// matrix) a \a std::invalid_argument exception is thrown. The checks can be skipped by providing
// the optional \a blaze::unchecked argument.
*/
template< typename MT         // Type of the dense matrix
        , size_t... CSAs >    // Compile time DilatedSubmatrix arguments
template< typename... RSAs >  // Runtime DilatedSubmatrix arguments
inline DilatedSubmatrix<MT,false,true,CSAs...>::DilatedSubmatrix( MT& matrix, RSAs... args )
   : DataType  ( args... )  // Base class initialization
   , matrix_   ( matrix  )  // The matrix containing the DilatedSubmatrix
{
   if( !Contains_v< TypeList<RSAs...>, Unchecked > ) {
      if( ( row( ) + ( rows( ) - 1 ) * rowdilation( ) + 1 > matrix_.rows( ) ) ||
         ( column( ) + ( columns( ) - 1 ) * columndilation( ) + 1 >  matrix_.columns( ) ) )
      {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid DilatedSubmatrix specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( row( ) + ( rows( ) - 1 ) * rowdilation( ) + 1    <= matrix_.rows()   , "Invalid DilatedSubmatrix specification" );
      BLAZE_USER_ASSERT( column( ) + ( columns( ) - 1 ) * columndilation( ) + 1 <= matrix_.columns(), "Invalid DilatedSubmatrix specification" );
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
/*!\brief 2D-access to the dense DilatedSubmatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline typename DilatedSubmatrix<MT,false,true,CSAs...>::Reference
   DilatedSubmatrix<MT,false,true,CSAs...>::operator()( size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return matrix_( row( ) + i * rowdilation( ), column( ) + j * columndilation( ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the dense DilatedSubmatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline typename DilatedSubmatrix<MT,false,true,CSAs...>::ConstReference
   DilatedSubmatrix<MT,false,true,CSAs...>::operator()( size_t i, size_t j ) const
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return const_cast< const MT& >( matrix_ )( row() + i * rowdilation(), column() + j * columndilation() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the DilatedSubmatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the function call operator this function always performs a check of the given
// access indices.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline typename DilatedSubmatrix<MT,false,true,CSAs...>::Reference
   DilatedSubmatrix<MT,false,true,CSAs...>::at( size_t i, size_t j )
{
   if( i >= rows() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= columns() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i,j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the DilatedSubmatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the function call operator this function always performs a check of the given
// access indices.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline typename DilatedSubmatrix<MT,false,true,CSAs...>::ConstReference
   DilatedSubmatrix<MT,false,true,CSAs...>::at( size_t i, size_t j ) const
{
   if( i >= rows() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= columns() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i,j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the DilatedSubmatrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense DilatedSubmatrix. Note that
// you can NOT assume that all matrix elements lie adjacent to each other! The dense DilatedSubmatrix
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline typename DilatedSubmatrix<MT,false,true,CSAs...>::Pointer
   DilatedSubmatrix<MT,false,true,CSAs...>::data() noexcept
{
   return matrix_.data() + row()*spacing() + column();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the DilatedSubmatrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense DilatedSubmatrix. Note that
// you can NOT assume that all matrix elements lie adjacent to each other! The dense DilatedSubmatrix
// may use techniques such as padding to improve the alignment of the data.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline typename DilatedSubmatrix<MT,false,true,CSAs...>::ConstPointer
   DilatedSubmatrix<MT,false,true,CSAs...>::data() const noexcept
{
   return matrix_.data() + row()*spacing() + column();
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
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline typename DilatedSubmatrix<MT,false,true,CSAs...>::Iterator
   DilatedSubmatrix<MT,false,true,CSAs...>::begin( size_t i )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense DilatedSubmatrix row access index" );
   return Iterator( matrix_.begin( row() + i ) + column(), rowdilation(), columndilation() );
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
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline typename DilatedSubmatrix<MT,false,true,CSAs...>::ConstIterator
   DilatedSubmatrix<MT,false,true,CSAs...>::begin( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense DilatedSubmatrix row access index" );
   return ConstIterator( matrix_.cbegin( row() + i ) + column(), rowdilation(), columndilation() );
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
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline typename DilatedSubmatrix<MT,false,true,CSAs...>::ConstIterator
   DilatedSubmatrix<MT,false,true,CSAs...>::cbegin( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense DilatedSubmatrix row access index" );
   return ConstIterator( matrix_.cbegin( row() + i ) + column(), rowdilation(), columndilation()  );
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
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline typename DilatedSubmatrix<MT,false,true,CSAs...>::Iterator
   DilatedSubmatrix<MT,false,true,CSAs...>::end( size_t i )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense DilatedSubmatrix row access index" );
   return Iterator( matrix_.begin( row() + i ) + column() + columns() * columndilation(),
      rowdilation(), columndilation()  );
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
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline typename DilatedSubmatrix<MT,false,true,CSAs...>::ConstIterator
   DilatedSubmatrix<MT,false,true,CSAs...>::end( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense DilatedSubmatrix row access index" );
   return ConstIterator( matrix_.cbegin( row() + i ) + column() + columns() * columndilation(),
      rowdilation(), columndilation()  );
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
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline typename DilatedSubmatrix<MT,false,true,CSAs...>::ConstIterator
   DilatedSubmatrix<MT,false,true,CSAs...>::cend( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid dense DilatedSubmatrix row access index" );
   return ConstIterator( matrix_.cbegin( row() + i ) + column() + columns() * columndilation(),
      rowdilation(), columndilation() );
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
/*!\brief Homogenous assignment to all DilatedSubmatrix elements.
//
// \param rhs Scalar value to be assigned to all DilatedSubmatrix elements.
// \return Reference to the assigned DilatedSubmatrix.
//
// This function homogeneously assigns the given value to all dense matrix elements. Note that in
// case the underlying dense matrix is a lower/upper matrix only lower/upper and diagonal elements
// of the underlying matrix are modified.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline DilatedSubmatrix<MT,false,true,CSAs...>&
   DilatedSubmatrix<MT,false,true,CSAs...>::operator=( const ElementType& rhs )
{
   const size_t iend( row( ) + rows( ) * rowdilation( ) );
   decltype(auto) left( derestrict( matrix_ ) );

   for( size_t i = row( ); i < iend; i += rowdilation( ) )
   {
      const size_t jbegin( ( IsUpper_v<MT> )
                           ?( ( IsUniUpper_v<MT> || IsStrictlyUpper_v<MT> )
                              ?( max( i+1UL, column() ) )
                              :( max( i, column() ) ) )
                           :( column() ) );
      const size_t jend  ( ( IsLower_v<MT> )
                           ?( ( IsUniLower_v<MT> || IsStrictlyLower_v<MT> )
                              ?( min( i, column()+columns()* columndilation() ) )
                              :( min( i+1UL, column()+columns()* columndilation() ) ) )
                           :( column()+columns() * columndilation() ) );

      for( size_t j = jbegin; j < jend; j += columndilation( ) )
      {
         if( !IsRestricted_v<MT> || IsTriangular_v<MT> || trySet( matrix_, i, j, rhs ) )
            left(i,j) = rhs;
      }
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all DilatedSubmatrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid initializer list dimension.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// This assignment operator offers the option to directly assign to all elements of the DilatedSubmatrix
// by means of an initializer list. The DilatedSubmatrix elements are assigned the values from the given
// initializer list. Missing values are initialized as default. Note that in case the size of the
// top-level initializer list does not match the number of rows of the DilatedSubmatrix or the size of
// any nested list exceeds the number of columns, a \a std::invalid_argument exception is thrown.
// Also, if the underlying matrix \a MT is restricted and the assignment would violate an
// invariant of the matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline DilatedSubmatrix<MT,false,true,CSAs...>&
   DilatedSubmatrix<MT,false,true,CSAs...>::operator=( initializer_list< initializer_list<ElementType> > list )
{
   if( list.size() != rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to DilatedSubmatrix" );
   }

   if( IsRestricted_v<MT> ) {
      const InitializerMatrix<ElementType> tmp( list, columns() );
      if( !tryAssign( matrix_, tmp, row(), column() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
      }
   }

   decltype(auto) left( derestrict( *this ) );
   size_t i( 0UL );

   for( const auto& rowList : list ) {
      std::fill( std::copy( rowList.begin(), rowList.end(), left.begin(i) ), left.end(i), ElementType() );
      ++i;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for DilatedSubmatrix.
//
// \param rhs Dense DilatedSubmatrix to be copied.
// \return Reference to the assigned DilatedSubmatrix.
// \exception std::invalid_argument DilatedSubmatrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The dense DilatedSubmatrix is initialized as a copy of the given dense DilatedSubmatrix. In case the current
// sizes of the two submatrices don't match, a \a std::invalid_argument exception is thrown. Also,
// if the underlying matrix \a MT is a lower triangular, upper triangular, or symmetric matrix
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline DilatedSubmatrix<MT,false,true,CSAs...>&
   DilatedSubmatrix<MT,false,true,CSAs...>::operator=( const DilatedSubmatrix& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( this == &rhs || ( &matrix_ == &rhs.matrix_ && row() == rhs.row() && column() == rhs.column() &&
      rowdilation() == rhs.rowdilation() && columndilation() == rhs.columndilation() ) )
      return *this;

   if( rows() != rhs.rows() || columns() != rhs.columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "DilatedSubmatrix sizes do not match" );
   }

   if( !tryAssign( matrix_, rhs, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( rhs.canAlias( &matrix_ ) ) {
      const ResultType tmp( rhs );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different matrices.
//
// \param rhs Matrix to be assigned.
// \return Reference to the assigned DilatedSubmatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The dense DilatedSubmatrix is initialized as a copy of the given matrix. In case the current sizes
// of the two matrices don't match, a \a std::invalid_argument exception is thrown. Also, if
// the underlying matrix \a MT is a lower triangular, upper triangular, or symmetric matrix
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO2 >        // Storage order of the right-hand side matrix
inline DilatedSubmatrix<MT,false,true,CSAs...>&
   DilatedSubmatrix<MT,false,true,CSAs...>::operator=( const Matrix<MT2,SO2>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<MT>, CompositeType_t<MT2>, const MT2& >;
   Right right( ~rhs );

   if( !tryAssign( matrix_, right, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &matrix_ ) ) {
      const ResultType_t<MT2> tmp( right );
      if( IsSparseMatrix_v<MT2> )
         reset();
      smpAssign( left, tmp );
   }
   else {
      if( IsSparseMatrix_v<MT2> )
         reset();
      smpAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the DilatedSubmatrix.
// \return Reference to the dense DilatedSubmatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO2 >        // Storage order of the right-hand side matrix
inline auto DilatedSubmatrix<MT,false,true,CSAs...>::operator+=( const Matrix<MT2,SO2>& rhs )
   -> DisableIf_t< EnforceEvaluation_v<MT,MT2>, DilatedSubmatrix& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using AddType = AddTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !tryAddAssign( matrix_, ~rhs, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( ( ( IsSymmetric_v<MT> || IsHermitian_v<MT> ) && hasOverlap() ) ||
       (~rhs).canAlias( &matrix_ ) ) {
      const AddType tmp( *this + (~rhs) );
      smpAssign( left, tmp );
   }
   else {
      smpAddAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the DilatedSubmatrix.
// \return Reference to the dense DilatedSubmatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO2 >        // Storage order of the right-hand side matrix
inline auto DilatedSubmatrix<MT,false,true,CSAs...>::operator+=( const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< EnforceEvaluation_v<MT,MT2>, DilatedSubmatrix& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using AddType = AddTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( AddType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const AddType tmp( *this + (~rhs) );

   if( !tryAssign( matrix_, tmp, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the DilatedSubmatrix.
// \return Reference to the dense DilatedSubmatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO2 >        // Storage order of the right-hand side matrix
inline auto DilatedSubmatrix<MT,false,true,CSAs...>::operator-=( const Matrix<MT2,SO2>& rhs )
   -> DisableIf_t< EnforceEvaluation_v<MT,MT2>, DilatedSubmatrix& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using SubType = SubTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !trySubAssign( matrix_, ~rhs, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( ( ( IsSymmetric_v<MT> || IsHermitian_v<MT> ) && hasOverlap() ) ||
       (~rhs).canAlias( &matrix_ ) ) {
      const SubType tmp( *this - (~rhs ) );
      smpAssign( left, tmp );
   }
   else {
      smpSubAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the DilatedSubmatrix.
// \return Reference to the dense DilatedSubmatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO2 >        // Storage order of the right-hand side matrix
inline auto DilatedSubmatrix<MT,false,true,CSAs...>::operator-=( const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< EnforceEvaluation_v<MT,MT2>, DilatedSubmatrix& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using SubType = SubTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( SubType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const SubType tmp( *this - (~rhs) );

   if( !tryAssign( matrix_, tmp, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Schur product assignment operator for the multiplication of a matrix (\f$ A=B \f$).
//
// \param rhs The right-hand side matrix for the Schur product.
// \return Reference to the dense DilatedSubmatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO2 >        // Storage order of the right-hand side matrix
inline auto DilatedSubmatrix<MT,false,true,CSAs...>::operator%=( const Matrix<MT2,SO2>& rhs )
   -> DisableIf_t< EnforceEvaluation_v<MT,MT2>, DilatedSubmatrix& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using SchurType = SchurTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SchurType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !trySchurAssign( matrix_, ~rhs, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( ( ( IsSymmetric_v<MT> || IsHermitian_v<MT> ) && hasOverlap() ) ||
       (~rhs).canAlias( &matrix_ ) ) {
      const SchurType tmp( *this % (~rhs) );
      if( IsSparseMatrix_v<SchurType> )
         reset();
      smpAssign( left, tmp );
   }
   else {
      smpSchurAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Schur product assignment operator for the multiplication of a matrix (\f$ A=B \f$).
//
// \param rhs The right-hand side matrix for the Schur product.
// \return Reference to the dense DilatedSubmatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO2 >        // Storage order of the right-hand side matrix
inline auto DilatedSubmatrix<MT,false,true,CSAs...>::operator%=( const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< EnforceEvaluation_v<MT,MT2>, DilatedSubmatrix& >
{
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using SchurType = SchurTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SchurType );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const SchurType tmp( *this % (~rhs) );

   if( !tryAssign( matrix_, tmp, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsSparseMatrix_v<SchurType> ) {
      reset();
   }

   smpAssign( left, tmp );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

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
/*!\brief Returns the matrix containing the DilatedSubmatrix.
//
// \return The matrix containing the DilatedSubmatrix.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline MT& DilatedSubmatrix<MT,false,true,CSAs...>::operand() noexcept
{
   return matrix_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the matrix containing the DilatedSubmatrix.
//
// \return The matrix containing the DilatedSubmatrix.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline const MT& DilatedSubmatrix<MT,false,true,CSAs...>::operand() const noexcept
{
   return matrix_;
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
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline size_t DilatedSubmatrix<MT,false,true,CSAs...>::spacing() const noexcept
{
   return matrix_.spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense DilatedSubmatrix.
//
// \return The capacity of the dense DilatedSubmatrix.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline size_t DilatedSubmatrix<MT,false,true,CSAs...>::capacity() const noexcept
{
   return rows() * columns();
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
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline size_t DilatedSubmatrix<MT,false,true,CSAs...>::capacity( size_t i ) const noexcept
{
   MAYBE_UNUSED( i );

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   return columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the dense DilatedSubmatrix
//
// \return The number of non-zero elements in the dense DilatedSubmatrix.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline size_t DilatedSubmatrix<MT,false,true,CSAs...>::nonZeros() const
{
   const size_t iend( row() + rows() * rowdilation() );
   const size_t jend( column() + columns() * columndilation() );
   size_t nonzeros( 0UL );

   for( size_t i=row(); i<iend; i+=rowdilation() )
      for( size_t j=column(); j<jend; j+=columndilation )
         if( !isDefault( matrix_(i,j) ) )
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
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline size_t DilatedSubmatrix<MT,false,true,CSAs...>::nonZeros( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   const size_t jend( column() + columns() * columndilation() );
   size_t nonzeros( 0UL );

   for( size_t j=column(); j<jend; j+=columndilation() )
      if( !isDefault( matrix_(row()+i,j) ) )
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
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline void DilatedSubmatrix<MT,false,true,CSAs...>::reset()
{
   using blaze::clear;

   for( size_t i=row(); i<row()+rows()*rowdilation(); i+=rowdilation() )
   {
      const size_t jbegin( ( IsUpper_v<MT> )
                           ?( ( IsUniUpper_v<MT> || IsStrictlyUpper_v<MT> )
                              ?( max( i+1UL, column() ) )
                              :( max( i, column() ) ) )
                           :( column() ) );
      const size_t jend  ( ( IsLower_v<MT> )
                           ?( ( IsUniLower_v<MT> || IsStrictlyLower_v<MT> )
                              ?( min( i, column()+columns()*columndilation() ) )
                              :( min( i+1UL, column()+columns()*columndilation() ) ) )
                           :( column()+columns()*columndilation() ) );

      for( size_t j=jbegin; j<jend; j+=columndilation() )
         clear( matrix_(i,j) );
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
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline void DilatedSubmatrix<MT,false,true,CSAs...>::reset( size_t i )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   const size_t jbegin( ( IsUpper_v<MT> )
                        ?( ( IsUniUpper_v<MT> || IsStrictlyUpper_v<MT> )
                           ?( max( i+1UL, column() ) )
                           :( max( i, column() ) ) )
                        :( column() ) );
   const size_t jend  ( ( IsLower_v<MT> )
                        ?( ( IsUniLower_v<MT> || IsStrictlyLower_v<MT> )
                           ?( min( i, column()+columns()*columndilation() ) )
                           :( min( i+1UL, column()+columns()*columndilation() ) ) )
                        :( column()+columns()*columndilation() ) );

   for( size_t j=jbegin; j<jend; j+=columndilation()  )
      clear( matrix_(row()+i,j) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checking whether there exists an overlap in the context of a symmetric matrix.
//
// \return \a true in case an overlap exists, \a false if not.
//
// This function checks if in the context of a symmetric matrix the DilatedSubmatrix has an overlap with
// its counterpart. In case an overlap exists, the function return \a true, otherwise it returns
// \a false.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline bool DilatedSubmatrix<MT,false,true,CSAs...>::hasOverlap() const noexcept
{
   BLAZE_INTERNAL_ASSERT( IsSymmetric_v<MT> || IsHermitian_v<MT>, "Invalid matrix detected" );

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
/*!\brief In-place transpose of the DilatedSubmatrix.
//
// \return Reference to the transposed DilatedSubmatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic DilatedSubmatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense DilatedSubmatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the DilatedSubmatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the DilatedSubmatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline DilatedSubmatrix<MT,false,true,CSAs...>&
   DilatedSubmatrix<MT,false,true,CSAs...>::transpose()
{
   if( rows() != columns() ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic DilatedSubmatrix" );
   }

   if( !tryAssign( matrix_, trans( *this ), row(), column() ) ) {
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
/*!\brief In-place conjugate transpose of the DilatedSubmatrix.
//
// \return Reference to the transposed DilatedSubmatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic DilatedSubmatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the dense DilatedSubmatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the DilatedSubmatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the DilatedSubmatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline DilatedSubmatrix<MT,false,true,CSAs...>&
   DilatedSubmatrix<MT,false,true,CSAs...>::ctranspose()
{
   if( rows() != columns() ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic DilatedSubmatrix" );
   }

   if( !tryAssign( matrix_, ctrans( *this ), row(), column() ) ) {
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
/*!\brief Scaling of the dense DilatedSubmatrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the DilatedSubmatrix scaling.
// \return Reference to the dense DilatedSubmatrix.
//
// This function scales the DilatedSubmatrix by applying the given scalar value \a scalar to each
// element of the DilatedSubmatrix. For built-in and \c complex data types it has the same effect
// as using the multiplication assignment operator. Note that the function cannot be used
// to scale a DilatedSubmatrix on a lower or upper unitriangular matrix. The attempt to scale
// such a DilatedSubmatrix results in a compile time error!
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename Other >  // Data type of the scalar value
inline DilatedSubmatrix<MT,false,true,CSAs...>&
   DilatedSubmatrix<MT,false,true,CSAs...>::scale( const Other& scalar )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   const size_t iend( row() + rows()*rowdilation() );

   for( size_t i=row(); i<iend; i+=rowdilation() )
   {
      const size_t jbegin( ( IsUpper_v<MT> )
                           ?( ( IsStrictlyUpper_v<MT> )
                              ?( max( i+1UL, column() ) )
                              :( max( i, column() ) ) )
                           :( column() ) );
      const size_t jend  ( ( IsLower_v<MT> )
                           ?( ( IsStrictlyLower_v<MT> )
                              ?( min( i, column()+columns()*columndilation() ) )
                              :( min( i+1UL, column()+columns()*columndilation() ) ) )
                           :( column()+columns()*columndilation() ) );

      for( size_t j=jbegin; j<jend; j+=columndilation()  )
         matrix_(i,j) *= scalar;
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
/*!\brief Returns whether the DilatedSubmatrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this DilatedSubmatrix, \a false if not.
//
// This function returns whether the given address can alias with the DilatedSubmatrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename Other >  // Data type of the foreign expression
inline bool DilatedSubmatrix<MT,false,true,CSAs...>::canAlias( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the DilatedSubmatrix can alias with the given dense DilatedSubmatrix \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this DilatedSubmatrix, \a false if not.
//
// This function returns whether the given address can alias with the DilatedSubmatrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT        // Type of the dense matrix
        , size_t... CSAs >   // Compile time DilatedSubmatrix arguments
template< typename MT2       // Data type of the foreign dense DilatedSubmatrix
        , bool SO2           // Storage order of the foreign dense DilatedSubmatrix
        , size_t... CSAs2 >  // Compile time DilatedSubmatrix arguments of the foreign dense DilatedSubmatrix
inline bool
   DilatedSubmatrix<MT,false,true,CSAs...>::canAlias( const DilatedSubmatrix<MT2,SO2,true,CSAs2...>* alias ) const noexcept
{
   return ( matrix_.isAliased( &alias->matrix_ ) &&
            ( row() + rows() > alias->row() ) &&
            ( row() < alias->row() + alias->rows() ) &&
            ( column() + columns() > alias->column() ) &&
            ( column() < alias->column() + alias->columns() ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the DilatedSubmatrix is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this DilatedSubmatrix, \a false if not.
//
// This function returns whether the given address is aliased with the DilatedSubmatrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename Other >  // Data type of the foreign expression
inline bool DilatedSubmatrix<MT,false,true,CSAs...>::isAliased( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the DilatedSubmatrix is aliased with the given dense DilatedSubmatrix \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this DilatedSubmatrix, \a false if not.
//
// This function returns whether the given address is aliased with the DilatedSubmatrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT        // Type of the dense matrix
        , size_t... CSAs >   // Compile time DilatedSubmatrix arguments
template< typename MT2       // Data type of the foreign dense DilatedSubmatrix
        , bool SO2           // Storage order of the foreign dense DilatedSubmatrix
        , size_t... CSAs2 >  // Compile time DilatedSubmatrix arguments of the foreign dense DilatedSubmatrix
inline bool
   DilatedSubmatrix<MT,false,true,CSAs...>::isAliased( const DilatedSubmatrix<MT2,SO2,true,CSAs2...>* alias ) const noexcept
{
   return ( matrix_.isAliased( &alias->matrix_ ) &&
            ( row() + rows()*rowdilation() > alias->row() ) &&
            ( row() < alias->row() + (alias->rows()-1)*rowdilation()+1 ) &&
            ( column() + columns()*columndilation() > alias->column() ) &&
            ( column() < alias->column() + (alias->columns()-1)*columndilation()+1 ) );
}
/*! \endcond */
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the DilatedSubmatrix can be used in SMP assignments.
//
// \return \a true in case the DilatedSubmatrix can be used in SMP assignments, \a false if not.
//
// This function returns whether the DilatedSubmatrix can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the DilatedSubmatrix).
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
inline bool DilatedSubmatrix<MT,false,true,CSAs...>::canSMPAssign() const noexcept
{
   return ( rows() * columns() >= SMP_DMATASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline void DilatedSubmatrix<MT,false,true,CSAs...>::assign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<rows(); ++i ) {
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         (*this)(i,j    ) = (~rhs)(i,j    );
         (*this)(i,j+1UL) = (~rhs)(i,j+1UL);
      }
      if( jpos < columns() ) {
         (*this)(i,jpos) = (~rhs)(i,jpos);
      }
   }
}
/*! \endcond */
//*************************************************************************************************

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline void DilatedSubmatrix<MT,false,true,CSAs...>::assign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   constexpr size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<rows(); ii+=block ) {
      const size_t iend( ( rows()<(ii+block) )?( rows() ):( ii+block ) );
      for( size_t jj=0UL; jj<columns(); jj+=block ) {
         const size_t jend( ( columns()<(jj+block) )?( columns() ):( jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               (*this)(i,j) = (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline void DilatedSubmatrix<MT,false,true,CSAs...>::addAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<rows(); ++i )
   {
      if( IsDiagonal_v<MT2> ) {
         (*this)(i,i) += (~rhs)(i,i);
      }
      else {
         for( size_t j=0UL; j<jpos; j+=2UL ) {
            (*this)(i,j    ) += (~rhs)(i,j    );
            (*this)(i,j+1UL) += (~rhs)(i,j+1UL);
         }
         if( jpos < columns() ) {
            (*this)(i,jpos) += (~rhs)(i,jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline void DilatedSubmatrix<MT,false,true,CSAs...>::addAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   constexpr size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<rows(); ii+=block ) {
      const size_t iend( ( rows()<(ii+block) )?( rows() ):( ii+block ) );
      for( size_t jj=0UL; jj<columns(); jj+=block ) {
         const size_t jend( ( columns()<(jj+block) )?( columns() ):( jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               (*this)(i,j) += (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline void DilatedSubmatrix<MT,false,true,CSAs...>::subAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<rows(); ++i )
   {
      if( IsDiagonal_v<MT2> ) {
         (*this)(i,i) -= (~rhs)(i,i);
      }
      else {
         for( size_t j=0UL; j<jpos; j+=2UL ) {
            (*this)(i,j    ) -= (~rhs)(i,j    );
            (*this)(i,j+1UL) -= (~rhs)(i,j+1UL);
         }
         if( jpos < columns() ) {
            (*this)(i,jpos) -= (~rhs)(i,jpos);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline void DilatedSubmatrix<MT,false,true,CSAs...>::subAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   constexpr size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<rows(); ii+=block ) {
      const size_t iend( ( rows()<(ii+block) )?( rows() ):( ii+block ) );
      for( size_t jj=0UL; jj<columns(); jj+=block ) {
         const size_t jend( ( columns()<(jj+block) )?( columns() ):( jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               (*this)(i,j) -= (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the Schur product assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline void DilatedSubmatrix<MT,false,true,CSAs...>::schurAssign( const DenseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<rows(); ++i ) {
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         (*this)(i,j    ) *= (~rhs)(i,j    );
         (*this)(i,j+1UL) *= (~rhs)(i,j+1UL);
      }
      if( jpos < columns() ) {
         (*this)(i,jpos) *= (~rhs)(i,jpos);
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the Schur product assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense matrix
        , size_t... CSAs >  // Compile time DilatedSubmatrix arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline void DilatedSubmatrix<MT,false,true,CSAs...>::schurAssign( const DenseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   constexpr size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<rows(); ii+=block ) {
      const size_t iend( ( rows()<(ii+block) )?( rows() ):( ii+block ) );
      for( size_t jj=0UL; jj<columns(); jj+=block ) {
         const size_t jend( ( columns()<(jj+block) )?( columns() ):( jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               (*this)(i,j) *= (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
