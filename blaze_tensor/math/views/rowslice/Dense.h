//=================================================================================================
/*!
//  \file blaze_tensor/math/views/rowslice/Dense.h
//  \brief RowSlice specialization for dense tensors
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_ROWSLICE_DENSE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_ROWSLICE_DENSE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/views/row/Dense.h>

#include <blaze_tensor/math/InitializerList.h>
#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/Subtensor.h>
#include <blaze_tensor/math/traits/RowSliceTrait.h>
#include <blaze_tensor/math/views/rowslice/BaseTemplate.h>
#include <blaze_tensor/math/views/rowslice/RowSliceData.h>

namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR DENSE TENSORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of RowSlice for rowslices on dense tensors.
// \ingroup rowslice
//
// This specialization of RowSlice adapts the class template to the requirements of
// dense tensors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
class RowSlice
   : public View< DenseMatrix< RowSlice<MT,CRAs...>, false > >
   , private RowSliceData<CRAs...>
{
 private:
   //**Type definitions****************************************************************************
   using DataType = RowSliceData<CRAs...>;                     //!< The type of the RowSliceData base class.
   using Operand  = If_t< IsExpression_v<MT>, MT, MT& >;  //!< Composite data type of the dense tensor expression.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this RowSlice instance.
   using This = RowSlice<MT,CRAs...>;

   using BaseType      = DenseMatrix<This,false>;      //!< Base type of this RowSlice instance.
   using ViewedType    = MT;                           //!< The type viewed by this RowSlice instance.
   using ResultType    = RowSliceTrait_t<MT,CRAs...>;      //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;    //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;  //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<MT>;            //!< Type of the rowslice elements.
   using SIMDType      = SIMDTrait_t<ElementType>;     //!< SIMD type of the rowslice elements.
   using ReturnType    = ReturnType_t<MT>;             //!< Return type for expression template evaluations
   using CompositeType = const RowSlice&;                  //!< Data type for composite expression templates.

   //! Reference to a constant rowslice value.
   using ConstReference = ConstReference_t<MT>;

   //! Reference to a non-constant rowslice value.
   using Reference = If_t< IsConst_v<MT>, ConstReference, Reference_t<MT> >;

   //! Pointer to a constant rowslice value.
   using ConstPointer = ConstPointer_t<MT>;

   //! Pointer to a non-constant rowslice value.
   using Pointer = If_t< IsConst_v<MT> || !HasMutableDataAccess_v<MT>, ConstPointer, Pointer_t<MT> >;

   //! Iterator over constant elements.
   using ConstIterator = ConstIterator_t<MT>;

   //! Iterator over non-constant elements.
   using Iterator = If_t< IsConst_v<MT>, ConstIterator, Iterator_t<MT> >;
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
   template< typename... RRAs >
   explicit inline RowSlice( MT& tensor, RRAs... args );

   RowSlice( const RowSlice& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~RowSlice() = default;
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator()( size_t j, size_t k );
   inline ConstReference operator()( size_t j, size_t k ) const;
   inline Reference      at( size_t j, size_t k );
   inline ConstReference at( size_t j, size_t k ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Pointer        data  ( size_t i ) noexcept;
   inline ConstPointer   data  ( size_t i ) const noexcept;
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
   inline RowSlice& operator=( const ElementType& rhs );
   inline RowSlice& operator=( initializer_list<initializer_list<ElementType> > list );
   inline RowSlice& operator=( const RowSlice& rhs );

   template< typename VT > inline RowSlice& operator= ( const Matrix<VT,false>& rhs );
   template< typename VT > inline RowSlice& operator+=( const Matrix<VT,false>& rhs );
   template< typename VT > inline RowSlice& operator-=( const Matrix<VT,false>& rhs );
   template< typename VT > inline RowSlice& operator%=( const Matrix<VT,false>& rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   using DataType::row;

   inline MT&       operand() noexcept;
   inline const MT& operand() const noexcept;

   inline size_t  rows() const noexcept;
   inline size_t  columns() const noexcept;
   inline size_t  spacing() const noexcept;
   inline size_t  capacity() const noexcept;
   inline size_t  capacity( size_t i ) const noexcept;
   inline size_t  nonZeros() const;
   inline size_t  nonZeros( size_t i ) const;
   inline void    reset();
   inline void    reset( size_t i );
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   template< typename Other > inline RowSlice& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename VT >
   static constexpr bool VectorizedAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && VT::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<VT> > );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename VT >
   static constexpr bool VectorizedAddAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && VT::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<VT> > &&
        HasSIMDAdd_v< ElementType, ElementType_t<VT> > );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename VT >
   static constexpr bool VectorizedSubAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && VT::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<VT> > &&
        HasSIMDSub_v< ElementType, ElementType_t<VT> > );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename VT >
   static constexpr bool VectorizedSchurAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && VT::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<VT> > &&
        HasSIMDMult_v< ElementType, ElementType_t<VT> > );
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

   template< typename MT2, size_t... CRAs2 >
   inline bool canAlias( const RowSlice<MT2,CRAs2...>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename MT2, size_t... CRAs2 >
   inline bool isAliased( const RowSlice<MT2,CRAs2...>* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t j, size_t k ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t j, size_t k ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t j, size_t k ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t j, size_t k, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t j, size_t k, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t j, size_t k, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t j, size_t k, const SIMDType& value ) noexcept;

   template< typename VT >
   inline auto assign( const DenseMatrix<VT,false>& rhs ) -> DisableIf_t< VectorizedAssign_v<VT> >;

   template< typename VT >
   inline auto assign( const DenseMatrix<VT,false>& rhs ) -> EnableIf_t< VectorizedAssign_v<VT> >;

   template< typename VT >
   inline auto addAssign( const DenseMatrix<VT,false>& rhs ) -> DisableIf_t< VectorizedAddAssign_v<VT> >;

   template< typename VT >
   inline auto addAssign( const DenseMatrix<VT,false>& rhs ) -> EnableIf_t< VectorizedAddAssign_v<VT> >;

   template< typename VT >
   inline auto subAssign( const DenseMatrix<VT,false>& rhs ) -> DisableIf_t< VectorizedSubAssign_v<VT> >;

   template< typename VT >
   inline auto subAssign( const DenseMatrix<VT,false>& rhs ) -> EnableIf_t< VectorizedSubAssign_v<VT> >;

   template< typename VT >
   inline auto schurAssign( const DenseMatrix<VT,false>& rhs ) -> DisableIf_t< VectorizedSchurAssign_v<VT> >;

   template< typename VT >
   inline auto schurAssign( const DenseMatrix<VT,false>& rhs ) -> EnableIf_t< VectorizedSchurAssign_v<VT> >;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand tensor_;  //!< The tensor containing the rowslice.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename MT2, size_t... CRAs2 > friend class RowSlice;
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
/*!\brief Constructor for rowslices on dense tensors.
//
// \param tensor The tensor containing the rowslice.
// \param args The runtime rowslice arguments.
// \exception std::invalid_argument Invalid rowslice access index.
//
// By default, the provided rowslice arguments are checked at runtime. In case the rowslice is not properly
// specified (i.e. if the specified index is greater than the number of pages of the given tensor)
// a \a std::invalid_argument exception is thrown. The checks can be skipped by providing the
// optional \a blaze::unchecked argument.
*/
template< typename MT         // Type of the dense tensor
        , size_t... CRAs >    // Compile time rowslice arguments
template< typename... RRAs >  // Runtime rowslice arguments
inline RowSlice<MT,CRAs...>::RowSlice( MT& tensor, RRAs... args )
   : DataType( args... )  // Base class initialization
   , tensor_ ( tensor  )  // The tensor containing the rowslice
{
   if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
      if( tensor_.rows() <= row() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid rowslice access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( row() < tensor_.rows(), "Invalid rowslice access index" );
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
/*!\brief Subscript operator for the direct access to the rowslice elements.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::Reference
   RowSlice<MT,CRAs...>::operator()( size_t j, size_t k )
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid row access index" );
   BLAZE_USER_ASSERT( k < rows()  , "Invalid columns access index" );
   return tensor_(row(), j, k);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subscript operator for the direct access to the rowslice elements.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::ConstReference
   RowSlice<MT,CRAs...>::operator()( size_t j, size_t k ) const
{
   return const_cast<const MT&>( tensor_ )(row(), j, k);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the rowslice elements.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid rowslice access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::Reference
   RowSlice<MT,CRAs...>::at( size_t j, size_t k )
{
   if( j >= columns() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   if( k >= rows() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
   }
   return (*this)(j, k);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the rowslice elements.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid rowslice access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::ConstReference
   RowSlice<MT,CRAs...>::at( size_t j, size_t k ) const
{
   if( j >= columns() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   if( k >= rows() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
   }
   return (*this)(j, k);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the rowslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense rowslice. Note that in case
// of a column-major tensor you can NOT assume that the rowslice elements lie adjacent to each other!
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::Pointer
   RowSlice<MT,CRAs...>::data() noexcept
{
   return tensor_.data( row(), 0UL );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the rowslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense rowslice. Note that in case
// of a column-major tensor you can NOT assume that the rowslice elements lie adjacent to each other!
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::ConstPointer
   RowSlice<MT,CRAs...>::data() const noexcept
{
   return tensor_.data( row(), 0UL );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the rowslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense rowslice. Note that in case
// of a column-major tensor you can NOT assume that the rowslice elements lie adjacent to each other!
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::Pointer
   RowSlice<MT,CRAs...>::data( size_t k ) noexcept
{
   return tensor_.data( row(), k );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the rowslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense rowslice. Note that in case
// of a column-major tensor you can NOT assume that the rowslice elements lie adjacent to each other!
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::ConstPointer
   RowSlice<MT,CRAs...>::data( size_t k ) const noexcept
{
   return tensor_.data( row(), k );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the rowslice.
//
// \param i The row/column index.
// \return Iterator to the first element of the given row on this rowslice.
//
// This function returns an iterator to the first element of the rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::Iterator
   RowSlice<MT,CRAs...>::begin( size_t i )
{
   return tensor_.begin( row(), i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the rowslice.
//
// \param i The row/column index.
// \return Iterator to the first element of the rowslice.
//
// This function returns an iterator to the first element of the rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::ConstIterator
   RowSlice<MT,CRAs...>::begin( size_t i ) const
{
   return tensor_.cbegin( row(), i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the rowslice.
//
// \param i The row/column index.
// \return Iterator to the first element of the rowslice.
//
// This function returns an iterator to the first element of the rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::ConstIterator
   RowSlice<MT,CRAs...>::cbegin( size_t i ) const
{
   return tensor_.cbegin( row(), i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the rowslice.
//
// \param i The row/column index.
// \return Iterator just past the last element of the rowslice.
//
// This function returns an iterator just past the last element of the rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::Iterator
   RowSlice<MT,CRAs...>::end( size_t i )
{
   return tensor_.end( row(), i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the rowslice.
//
// \param i The row/column index.
// \return Iterator just past the last element of the rowslice.
//
// This function returns an iterator just past the last element of the rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::ConstIterator
   RowSlice<MT,CRAs...>::end( size_t i ) const
{
   return tensor_.cend( row(), i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the rowslice.
//
// \param i The row/column index.
// \return Iterator just past the last element of the rowslice.
//
// This function returns an iterator just past the last element of the rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline typename RowSlice<MT,CRAs...>::ConstIterator
   RowSlice<MT,CRAs...>::cend( size_t i ) const
{
   return tensor_.cend( row(), i );
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
/*!\brief Homogeneous assignment to all rowslice elements.
//
// \param rhs Scalar value to be assigned to all rowslice elements.
// \return Reference to the assigned rowslice.
//
// This function homogeneously assigns the given value to all elements of the rowslice. Note that in
// case the underlying dense tensor is a lower/upper tensor only lower/upper and diagonal elements
// of the underlying tensor are modified.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline RowSlice<MT,CRAs...>&
   RowSlice<MT,CRAs...>::operator=( const ElementType& rhs )
{
   decltype(auto) left( derestrict( tensor_ ) );

   for (size_t k=0UL; k<rows(); ++k)
   {
      for (size_t j=0UL; j<columns(); ++j)
      {
         if (!IsRestricted_v<MT> || trySet(*this, k, j, rhs))
         {
            left(row(), j, k) = rhs;
         }
      }
   }
   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all rowslice elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to rowslice.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// This assignment operator offers the option to directly assign to all elements of the dense
// rowslice by means of an initializer list. The rowslice elements are assigned the values from the given
// initializer list. Missing values are reset to their default state. Note that in case the size
// of the initializer list exceeds the size of the rowslice, a \a std::invalid_argument exception is
// thrown. Also, if the underlying tensor \a MT is restricted and the assignment would violate
// an invariant of the tensor, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline RowSlice<MT,CRAs...>&
   RowSlice<MT,CRAs...>::operator=(initializer_list<initializer_list<ElementType> > list)
{
   if (list.size() > rows() || determineColumns(list) > columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to rowslice" );
   }

   if( IsRestricted_v<MT> ) {
      const InitializerMatrix<ElementType> tmp( list );
      if( !tryAssign( tensor_, tmp, row(), 0UL, 0UL ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
      }
   }

   decltype(auto) left( derestrict( *this ) );

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      std::fill( std::copy( rowList.begin(), rowList.end(), left.begin( i ) ), left.end( i ), ElementType() );
      ++i;
   }

   BLAZE_INTERNAL_ASSERT( isIntact( tensor_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for RowSlice.
//
// \param rhs Dense rowslice to be copied.
// \return Reference to the assigned rowslice.
// \exception std::invalid_argument RowSlice sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two rowslices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower or upper triangular tensor and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline RowSlice<MT,CRAs...>&
   RowSlice<MT,CRAs...>::operator=( const RowSlice& rhs )
{
   if( &rhs == this ) return *this;

   if( rows() != rhs.rows() || columns() != rhs.columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "RowSlice sizes do not match" );
   }

   if( !tryAssign( tensor_, rhs, row(), 0UL, 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsExpression_v<MT> && rhs.canAlias( &tensor_ ) ) {
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
// \param rhs Matrix to be assigned.
// \return Reference to the assigned rowslice.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower or upper triangular tensor and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename VT >     // Type of the right-hand side matrix
inline RowSlice<MT,CRAs...>&
   RowSlice<MT,CRAs...>::operator=( const Matrix<VT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<VT> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<VT> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<MT>, CompositeType_t<VT>, const VT& >;
   Right right( ~rhs );

   if( !tryAssign( tensor_, right, row(), 0UL, 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &tensor_ ) ) {
      const ResultType_t<VT> tmp( right );
      smpAssign( left, tmp );
   }
   else {
      if( IsSparseMatrix_v<VT> )
         reset();
      smpAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( tensor_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ \vec{a}+=\vec{b} \f$).
//
// \param rhs The right-hand side matrix to be added to the dense rowslice.
// \return Reference to the assigned rowslice.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower or upper triangular tensor and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename VT >     // Type of the right-hand side matrix
inline RowSlice<MT,CRAs...>&
   RowSlice<MT,CRAs...>::operator+=( const Matrix<VT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<VT> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<VT> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<MT>, CompositeType_t<VT>, const VT& >;
   Right right( ~rhs );

   if( !tryAddAssign( tensor_, right, row(), 0UL, 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &tensor_ ) ) {
      const ResultType_t<VT> tmp( right );
      smpAddAssign( left, tmp );
   }
   else {
      smpAddAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( tensor_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ \vec{a}-=\vec{b} \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the dense rowslice.
// \return Reference to the assigned rowslice.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower or upper triangular tensor and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename VT >     // Type of the right-hand side matrix
inline RowSlice<MT,CRAs...>&
   RowSlice<MT,CRAs...>::operator-=( const Matrix<VT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<VT> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<VT> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<MT>, CompositeType_t<VT>, const VT& >;
   Right right( ~rhs );

   if( !trySubAssign( tensor_, right, row(), 0UL, 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &tensor_ ) ) {
      const ResultType_t<VT> tmp( right );
      smpSubAssign( left, tmp );
   }
   else {
      smpSubAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( tensor_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Schur product assignment operator for the multiplication of a matrix
//        (\f$ \vec{a}\times=\vec{b} \f$).
//
// \param rhs The right-hand side matrix for the cross product.
// \return Reference to the assigned rowslice.
// \exception std::invalid_argument Invalid matrix size for cross product.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current size of any of the two matrices is not equal to 3, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename VT >     // Type of the right-hand side matrix
inline RowSlice<MT,CRAs...>&
   RowSlice<MT,CRAs...>::operator%=( const Matrix<VT,false>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<VT> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<VT> );

   using SchurType = SchurTrait_t< ResultType, ResultType_t<VT> >;

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !trySchurAssign( tensor_, (~rhs), row(), 0UL, 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted tensor" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<MT> && (~rhs).canAlias( &tensor_ ) ) {
      const SchurType tmp( *this % (~rhs) );
      smpSchurAssign( left, tmp );
   }
   else {
      smpSchurAssign( left, ~rhs );
   }

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
/*!\brief Returns the tensor containing the rowslice.
//
// \return The tensor containing the rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline MT& RowSlice<MT,CRAs...>::operand() noexcept
{
   return tensor_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the tensor containing the rowslice.
//
// \return The tensor containing the rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline const MT& RowSlice<MT,CRAs...>::operand() const noexcept
{
   return tensor_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current size/dimension of the rowslice.
//
// \return The size of the rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline size_t RowSlice<MT,CRAs...>::rows() const noexcept
{
   return tensor_.pages();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current size/dimension of the rowslice.
//
// \return The size of the rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline size_t RowSlice<MT,CRAs...>::columns() const noexcept
{
   return tensor_.columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the minimum capacity of the rowslice.
//
// \return The minimum capacity of the rowslice.
//
// This function returns the minimum capacity of the rowslice, which corresponds to the current size
// plus padding.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline size_t RowSlice<MT,CRAs...>::spacing() const noexcept
{
   return tensor_.spacing() * tensor_.rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense rowslice.
//
// \return The maximum capacity of the dense rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline size_t RowSlice<MT,CRAs...>::capacity() const noexcept
{
   return tensor_.capacity( row(), 0UL ) * tensor_.pages();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense rowslice.
//
// \return The maximum capacity of the dense rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline size_t RowSlice<MT,CRAs...>::capacity( size_t i ) const noexcept
{
   return tensor_.capacity( row(), i ) * tensor_.pages();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the rowslice.
//
// \return The number of non-zero elements in the rowslice.
//
// Note that the number of non-zero elements is always less than or equal to the current number
// of columns of the tensor containing the rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline size_t RowSlice<MT,CRAs...>::nonZeros() const
{
   size_t count ( 0 );
   for ( size_t i = 0; i < rows(); ++i ) {
      count += tensor_.nonZeros( row(), i );
   }
   return count;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the rowslice.
//
// \return The number of non-zero elements in the rowslice.
//
// Note that the number of non-zero elements is always less than or equal to the current number
// of columns of the tensor containing the rowslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline size_t RowSlice<MT,CRAs...>::nonZeros( size_t i ) const
{
   return tensor_.nonZeros( row(), i );
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
        , size_t... CRAs >  // Compile time rowslice arguments
inline void RowSlice<MT,CRAs...>::reset()
{
   for ( size_t i = 0; i < rows(); ++i ) {
      tensor_.reset( row(), i );
   }
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
        , size_t... CRAs >  // Compile time rowslice arguments
inline void RowSlice<MT,CRAs...>::reset( size_t k )
{
   tensor_.reset( row(), k );
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
/*!\brief Scaling of the rowslice by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the rowslice scaling.
// \return Reference to the dense rowslice.
//
// This function scales the rowslice by applying the given scalar value \a scalar to each element
// of the rowslice. For built-in and \c complex data types it has the same effect as using the
// multiplication assignment operator. Note that the function cannot be used to scale a rowslice
// on a lower or upper unitriangular tensor. The attempt to scale such a rowslice results in a
// compile time error!
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename Other >  // Data type of the scalar value
inline RowSlice<MT,CRAs...>&
   RowSlice<MT,CRAs...>::scale( const Other& scalar )
{
   for ( size_t k = 0; k < rows(); ++k ) {
      for ( size_t j = 0; j < columns(); ++j ) {
         tensor_(row(), j, k) *= scalar;
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
/*!\brief Returns whether the dense rowslice can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense rowslice, \a false if not.
//
// This function returns whether the given address can alias with the dense rowslice. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename Other >  // Data type of the foreign expression
inline bool RowSlice<MT,CRAs...>::canAlias( const Other* alias ) const noexcept
{
   return tensor_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense rowslice can alias with the given dense rowslice \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense rowslice, \a false if not.
//
// This function returns whether the given address can alias with the dense rowslice. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT        // Type of the dense tensor
        , size_t... CRAs >   // Compile time rowslice arguments
template< typename MT2       // Data type of the foreign dense rowslice
        , size_t... CRAs2 >  // Compile time rowslice arguments of the foreign dense rowslice
inline bool
   RowSlice<MT,CRAs...>::canAlias( const RowSlice<MT2,CRAs2...>* alias ) const noexcept
{
   return tensor_.isAliased( &alias->tensor_ ) && ( row() == alias->row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense rowslice is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense rowslice, \a false if not.
//
// This function returns whether the given address is aliased with the dense rowslice. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename Other >  // Data type of the foreign expression
inline bool RowSlice<MT,CRAs...>::isAliased( const Other* alias ) const noexcept
{
   return tensor_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense rowslice is aliased with the given dense rowslice \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense rowslice, \a false if not.
//
// This function returns whether the given address is aliased with the dense rowslice. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT        // Type of the dense tensor
        , size_t... CRAs >   // Compile time rowslice arguments
template< typename MT2       // Data type of the foreign dense rowslice
        , size_t... CRAs2 >  // Compile time rowslice arguments of the foreign dense rowslice
inline bool
   RowSlice<MT,CRAs...>::isAliased( const RowSlice<MT2,CRAs2...>* alias ) const noexcept
{
   return tensor_.isAliased( &alias->tensor_ ) && ( row() == alias->row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense rowslice is properly aligned in memory.
//
// \return \a true in case the dense rowslice is aligned, \a false if not.
//
// This function returns whether the dense rowslice is guaranteed to be properly aligned in memory,
// i.e. whether the beginning and the end of the dense rowslice are guaranteed to conform to the
// alignment restrictions of the element type \a Type.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline bool RowSlice<MT,CRAs...>::isAligned() const noexcept
{
   return tensor_.isAligned();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense rowslice can be used in SMP assignments.
//
// \return \a true in case the dense rowslice can be used in SMP assignments, \a false if not.
//
// This function returns whether the dense rowslice can be used in SMP assignments. In contrast to
// the \a smpAssignable member enumeration, which is based solely on compile time information,
// this function additionally provides runtime information (as for instance the current size
// of the dense rowslice).
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
inline bool RowSlice<MT,CRAs...>::canSMPAssign() const noexcept
{
   return ( rows() * columns() > SMP_DMATASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Load of a SIMD element of the dense rowslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense rowslice. The index
// must be smaller than the number of tensor columns. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
BLAZE_ALWAYS_INLINE typename RowSlice<MT,CRAs...>::SIMDType
   RowSlice<MT,CRAs...>::load( size_t j, size_t k ) const noexcept
{
   return tensor_.load( row(), j, k );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the dense rowslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense rowslice.
// The index must be smaller than the number of tensor columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
BLAZE_ALWAYS_INLINE typename RowSlice<MT,CRAs...>::SIMDType
   RowSlice<MT,CRAs...>::loada( size_t j, size_t k ) const noexcept
{
   return tensor_.loada( row(), j, k );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the dense rowslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense rowslice.
// The index must be smaller than the number of tensor columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
BLAZE_ALWAYS_INLINE typename RowSlice<MT,CRAs...>::SIMDType
   RowSlice<MT,CRAs...>::loadu( size_t j, size_t k ) const noexcept
{
   return tensor_.loadu( row(), j, k );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Store of a SIMD element of the dense rowslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store a specific SIMD element of the dense rowslice. The index
// must be smaller than the number of tensor columns. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
BLAZE_ALWAYS_INLINE void
   RowSlice<MT,CRAs...>::store( size_t j, size_t k, const SIMDType& value ) noexcept
{
   tensor_.store( row(), j, k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the dense rowslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store a specific SIMD element of the dense rowslice. The
// index must be smaller than the number of tensor columns. This function must \b NOT be
// called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
BLAZE_ALWAYS_INLINE void
   RowSlice<MT,CRAs...>::storea( size_t j, size_t k, const SIMDType& value ) noexcept
{
   tensor_.storea( row(), j, k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unligned store of a SIMD element of the dense rowslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store a specific SIMD element of the dense rowslice.
// The index must be smaller than the number of tensor columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
BLAZE_ALWAYS_INLINE void
   RowSlice<MT,CRAs...>::storeu( size_t j, size_t k, const SIMDType& value ) noexcept
{
   tensor_.storeu( row(), j, k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned, non-temporal store of a SIMD element of the dense rowslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store a specific SIMD element of the dense
// rowslice. The index must be smaller than the number of tensor columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
BLAZE_ALWAYS_INLINE void
   RowSlice<MT,CRAs...>::stream( size_t j, size_t k, const SIMDType& value ) noexcept
{
   tensor_.stream( row(), j, k, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename VT >     // Type of the right-hand side dense matrix
inline auto RowSlice<MT,CRAs...>::assign( const DenseMatrix<VT,false>& rhs )
   -> DisableIf_t< VectorizedAssign_v<VT> >
{
   BLAZE_INTERNAL_ASSERT( rows() == (~rhs).rows(), "Invalid matrix sizes" );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid matrix sizes" );

   for (size_t i = 0UL; i < (~rhs).rows(); ++i ) {
      const size_t jpos( (~rhs).columns() & size_t(-2) );
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         tensor_(row(),j,i) = (~rhs)(i,j);
         tensor_(row(),j+1UL,i) = (~rhs)(i,j+1UL);
      }
      if( jpos < (~rhs).columns() )
         tensor_(row(),jpos,i) = (~rhs)(i,jpos);
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the assignment of a dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename VT >     // Type of the right-hand side dense matrix
inline auto RowSlice<MT,CRAs...>::assign( const DenseMatrix<VT,false>& rhs )
   -> EnableIf_t< VectorizedAssign_v<VT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   constexpr bool remainder( !IsPadded_v<MT> || !IsPadded_v<VT> );

   const size_t cols( columns() );

   for (size_t i = 0; i < (~rhs).rows(); ++i) {
      const size_t jpos( ( remainder )?( cols & size_t(-SIMDSIZE) ):( cols ) );
      BLAZE_INTERNAL_ASSERT( !remainder || ( cols - ( cols % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );
      Iterator left( begin(i) );
      ConstIterator_t<VT> right( (~rhs).begin(i) );

      if( useStreaming && cols > ( cacheSize/( sizeof(ElementType) * 3UL ) ) && !(~rhs).isAliased( &tensor_ ) )
      {
         for( ; j<jpos; j+=SIMDSIZE ) {
            left.stream( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; remainder && j<cols; ++j ) {
            *left = *right; ++left; ++right;
         }
      }
      else
      {
         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; j<jpos; j+=SIMDSIZE ) {
            left.store( right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         }
         for( ; remainder && j<cols; ++j ) {
            *left = *right; ++left; ++right;
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename VT >     // Type of the right-hand side dense matrix
inline auto RowSlice<MT,CRAs...>::addAssign( const DenseMatrix<VT,false>& rhs )
   -> DisableIf_t< VectorizedAddAssign_v<VT> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   for (size_t i = 0UL; i < (~rhs).rows(); ++i ) {
      const size_t jpos( (~rhs).columns() & size_t(-2) );
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         tensor_(row(),j,i    ) += (~rhs)(i,j);
         tensor_(row(),j+1UL,i) += (~rhs)(i,j+1UL);
      }
      if( jpos < (~rhs).columns() )
         tensor_(row(),jpos,i) += (~rhs)(i,jpos);
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the addition assignment of a dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename VT >     // Type of the right-hand side dense matrix
inline auto RowSlice<MT,CRAs...>::addAssign( const DenseMatrix<VT,false>& rhs )
   -> EnableIf_t< VectorizedAddAssign_v<VT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   constexpr bool remainder( !IsPadded_v<MT> || !IsPadded_v<VT> );

   const size_t cols( columns() );

   for (size_t i = 0; i < (~rhs).rows(); ++i) {
      const size_t jpos( ( remainder )?( cols & size_t(-SIMDSIZE) ):( cols ) );
      BLAZE_INTERNAL_ASSERT( !remainder || ( cols - ( cols % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );
      Iterator left( begin(i) );
      ConstIterator_t<VT> right( (~rhs).begin(i) );

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; j<jpos; j+=SIMDSIZE ) {
         left.store( left.load() + right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; remainder && j<cols; ++j ) {
         *left += *right; ++left; ++right;
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename VT >     // Type of the right-hand side dense matrix
inline auto RowSlice<MT,CRAs...>::subAssign( const DenseMatrix<VT,false>& rhs )
   -> DisableIf_t< VectorizedSubAssign_v<VT> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   for (size_t i = 0UL; i < (~rhs).rows(); ++i ) {
      const size_t jpos( (~rhs).columns() & size_t(-2) );
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         tensor_(row(),j,i    ) -= (~rhs)(i,j);
         tensor_(row(),j+1UL,i) -= (~rhs)(i,j+1UL);
      }
      if( jpos < (~rhs).columns() )
         tensor_(row(),jpos,i) -= (~rhs)(i,jpos);
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the subtraction assignment of a dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time rowslice arguments
template< typename VT >     // Type of the right-hand side dense matrix
inline auto RowSlice<MT,CRAs...>::subAssign( const DenseMatrix<VT,false>& rhs )
   -> EnableIf_t< VectorizedSubAssign_v<VT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   constexpr bool remainder( !IsPadded_v<MT> || !IsPadded_v<VT> );

   const size_t cols( columns() );

   for (size_t i = 0; i < (~rhs).rows(); ++i) {
      const size_t jpos( ( remainder )?( cols & size_t(-SIMDSIZE) ):( cols ) );
      BLAZE_INTERNAL_ASSERT( !remainder || ( cols - ( cols % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );
      Iterator left( begin(i) );
      ConstIterator_t<VT> right( (~rhs).begin(i) );

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; j<jpos; j+=SIMDSIZE ) {
         left.store( left.load() - right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; remainder && j<cols; ++j ) {
         *left -= *right; ++left; ++right;
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
template< typename MT       // Type of the tensor
        , size_t... CSAs >  // Compile time rowslice arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline auto RowSlice<MT,CSAs...>::schurAssign( const DenseMatrix<MT2,false>& rhs )
   -> DisableIf_t< VectorizedSchurAssign_v<MT2> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<rows(); ++i ) {
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         tensor_(row(),j,i    ) *= (~rhs)(i,j    );
         tensor_(row(),j+1UL,i) *= (~rhs)(i,j+1UL);
      }
      if( jpos < columns() ) {
         tensor_(row(),jpos,i) *= (~rhs)(i,jpos);
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief SIMD optimized implementation of the Schur product assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the tensor
        , size_t... CSAs >  // Compile time rowslice arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline auto RowSlice<MT,CSAs...>::schurAssign( const DenseMatrix<MT2,false>& rhs )
   -> EnableIf_t< VectorizedSchurAssign_v<MT2> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   constexpr bool remainder( !IsPadded_v<MT> || !IsPadded_v<MT2> );

   const size_t cols( columns() );

   for( size_t i=0UL; i<rows(); ++i )
   {
      const size_t jpos( ( remainder )?( cols & size_t(-SIMDSIZE) ):( cols ) );
      BLAZE_INTERNAL_ASSERT( !remainder || ( cols - ( cols % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );
      Iterator left( begin(i) );
      ConstIterator_t<MT2> right( (~rhs).begin(i) );

      for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL ) {
         left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
         left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; j<jpos; j+=SIMDSIZE ) {
         left.store( left.load() * right.load() ); left += SIMDSIZE; right += SIMDSIZE;
      }
      for( ; remainder && j<cols; ++j ) {
         *left *= *right; ++left; ++right;
      }
   }
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
