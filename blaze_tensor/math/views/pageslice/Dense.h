//=================================================================================================
/*!
//  \file blaze_tensor/math/views/pageslice/Dense.h
//  \brief PageSlice specialization for dense tensors
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_PAGESLICE_DENSE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_PAGESLICE_DENSE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/views/row/Dense.h>

#include <blaze_tensor/math/InitializerList.h>
#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/Subtensor.h>
#include <blaze_tensor/math/traits/PageSliceTrait.h>
#include <blaze_tensor/math/views/pageslice/BaseTemplate.h>
#include <blaze_tensor/math/views/pageslice/PageSliceData.h>

namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR DENSE TENSORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of PageSlice for pageslices on pageslice-major dense tensors.
// \ingroup pageslice
//
// This specialization of PageSlice adapts the class template to the requirements of pageslice-major
// dense tensors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
class PageSlice
   : public View< DenseMatrix< PageSlice<MT,CRAs...>, false > >
   , private PageSliceData<CRAs...>
{
 private:
   //**Type definitions****************************************************************************
   using DataType = PageSliceData<CRAs...>;                     //!< The type of the PageSliceData base class.
   using Operand  = If_t< IsExpression_v<MT>, MT, MT& >;  //!< Composite data type of the dense tensor expression.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this PageSlice instance.
   using This = PageSlice<MT,CRAs...>;

   using BaseType      = DenseMatrix<This,false>;      //!< Base type of this PageSlice instance.
   using ViewedType    = MT;                           //!< The type viewed by this PageSlice instance.
   using ResultType    = PageSliceTrait_t<MT,CRAs...>;      //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;    //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;  //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<MT>;            //!< Type of the pageslice elements.
   using SIMDType      = SIMDTrait_t<ElementType>;     //!< SIMD type of the pageslice elements.
   using ReturnType    = ReturnType_t<MT>;             //!< Return type for expression template evaluations
   using CompositeType = const PageSlice&;                  //!< Data type for composite expression templates.

   //! Reference to a constant pageslice value.
   using ConstReference = ConstReference_t<MT>;

   //! Reference to a non-constant pageslice value.
   using Reference = If_t< IsConst_v<MT>, ConstReference, Reference_t<MT> >;

   //! Pointer to a constant pageslice value.
   using ConstPointer = ConstPointer_t<MT>;

   //! Pointer to a non-constant pageslice value.
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
   explicit inline PageSlice( MT& tensor, RRAs... args );

   PageSlice( const PageSlice& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~PageSlice() = default;
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
   inline PageSlice& operator=( const ElementType& rhs );
   inline PageSlice& operator=( initializer_list<initializer_list<ElementType> > list );
   inline PageSlice& operator=( const PageSlice& rhs );

   template< typename VT > inline PageSlice& operator= ( const Matrix<VT,false>& rhs );
   template< typename VT > inline PageSlice& operator+=( const Matrix<VT,false>& rhs );
   template< typename VT > inline PageSlice& operator-=( const Matrix<VT,false>& rhs );
   template< typename VT > inline PageSlice& operator%=( const Matrix<VT,false>& rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   using DataType::page;

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
   template< typename Other > inline PageSlice& scale( const Other& scalar );
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
   inline bool canAlias( const PageSlice<MT2,CRAs2...>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename MT2, size_t... CRAs2 >
   inline bool isAliased( const PageSlice<MT2,CRAs2...>* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t i, size_t j ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t i, size_t j, const SIMDType& value ) noexcept;

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
   Operand tensor_;  //!< The tensor containing the pageslice.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename MT2, size_t... CRAs2 > friend class PageSlice;
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE   ( MT );
//    BLAZE_CONSTRAINT_MUST_NOT_BE_SUBTENSOR_TYPE   ( MT );
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
/*!\brief Constructor for pageslices on pageslice-major dense tensors.
//
// \param tensor The tensor containing the pageslice.
// \param args The runtime pageslice arguments.
// \exception std::invalid_argument Invalid pageslice access index.
//
// By default, the provided pageslice arguments are checked at runtime. In case the pageslice is not properly
// specified (i.e. if the specified index is greater than the number of pages of the given tensor)
// a \a std::invalid_argument exception is thrown. The checks can be skipped by providing the
// optional \a blaze::unchecked argument.
*/
template< typename MT         // Type of the dense tensor
        , size_t... CRAs >    // Compile time pageslice arguments
template< typename... RRAs >  // Runtime pageslice arguments
inline PageSlice<MT,CRAs...>::PageSlice( MT& tensor, RRAs... args )
   : DataType( args... )  // Base class initialization
   , tensor_ ( tensor  )  // The tensor containing the pageslice
{
   if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
      if( tensor_.pages() <= page() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid pageslice access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( page() < tensor_.pages(), "Invalid pageslice access index" );
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
/*!\brief Subscript operator for the direct access to the pageslice elements.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::Reference
   PageSlice<MT,CRAs...>::operator()( size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i < rows(),    "Invalid row access index" );
   BLAZE_USER_ASSERT( j < columns(), "Invalid columns access index" );
   return tensor_(page(), i, j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subscript operator for the direct access to the pageslice elements.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::ConstReference
   PageSlice<MT,CRAs...>::operator()( size_t i, size_t j ) const
{
   return const_cast<const MT&>( tensor_ )(page(), i, j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the pageslice elements.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid pageslice access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::Reference
   PageSlice<MT,CRAs...>::at( size_t i, size_t j )
{
   if( i >= rows() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= columns() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i, j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the pageslice elements.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid pageslice access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::ConstReference
   PageSlice<MT,CRAs...>::at( size_t i, size_t j ) const
{
   if( i >= rows() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= columns() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i, j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the pageslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense pageslice. Note that in case
// of a column-major tensor you can NOT assume that the pageslice elements lie adjacent to each other!
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::Pointer
   PageSlice<MT,CRAs...>::data() noexcept
{
   return tensor_.data( 0, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the pageslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense pageslice. Note that in case
// of a column-major tensor you can NOT assume that the pageslice elements lie adjacent to each other!
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::ConstPointer
   PageSlice<MT,CRAs...>::data() const noexcept
{
   return tensor_.data( 0, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the pageslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense pageslice. Note that in case
// of a column-major tensor you can NOT assume that the pageslice elements lie adjacent to each other!
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::Pointer
   PageSlice<MT,CRAs...>::data( size_t i ) noexcept
{
   return tensor_.data( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the pageslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense pageslice. Note that in case
// of a column-major tensor you can NOT assume that the pageslice elements lie adjacent to each other!
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::ConstPointer
   PageSlice<MT,CRAs...>::data( size_t i ) const noexcept
{
   return tensor_.data( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the pageslice.
//
// \param i The row/column index.
// \return Iterator to the first element of the given row on this pageslice.
//
// This function returns an iterator to the first element of the pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::Iterator
   PageSlice<MT,CRAs...>::begin( size_t i )
{
   return tensor_.begin( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the pageslice.
//
// \param i The row/column index.
// \return Iterator to the first element of the pageslice.
//
// This function returns an iterator to the first element of the pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::ConstIterator
   PageSlice<MT,CRAs...>::begin( size_t i ) const
{
   return tensor_.cbegin( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the pageslice.
//
// \param i The row/column index.
// \return Iterator to the first element of the pageslice.
//
// This function returns an iterator to the first element of the pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::ConstIterator
   PageSlice<MT,CRAs...>::cbegin( size_t i ) const
{
   return tensor_.cbegin( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the pageslice.
//
// \param i The row/column index.
// \return Iterator just past the last element of the pageslice.
//
// This function returns an iterator just past the last element of the pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::Iterator
   PageSlice<MT,CRAs...>::end( size_t i )
{
   return tensor_.end( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the pageslice.
//
// \param i The row/column index.
// \return Iterator just past the last element of the pageslice.
//
// This function returns an iterator just past the last element of the pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::ConstIterator
   PageSlice<MT,CRAs...>::end( size_t i ) const
{
   return tensor_.cend( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the pageslice.
//
// \param i The row/column index.
// \return Iterator just past the last element of the pageslice.
//
// This function returns an iterator just past the last element of the pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline typename PageSlice<MT,CRAs...>::ConstIterator
   PageSlice<MT,CRAs...>::cend( size_t i ) const
{
   return tensor_.cend( i, page() );
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
/*!\brief Homogeneous assignment to all pageslice elements.
//
// \param rhs Scalar value to be assigned to all pageslice elements.
// \return Reference to the assigned pageslice.
//
// This function homogeneously assigns the given value to all elements of the pageslice. Note that in
// case the underlying dense tensor is a lower/upper tensor only lower/upper and diagonal elements
// of the underlying tensor are modified.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline PageSlice<MT,CRAs...>&
   PageSlice<MT,CRAs...>::operator=( const ElementType& rhs )
{
   decltype(auto) left( derestrict( tensor_ ) );

   for (size_t i=0UL; i<rows(); ++i)
   {
      for (size_t j=0UL; j<columns(); ++j)
      {
         if (!IsRestricted_v<MT> || trySet(*this, i, j, rhs))
         {
            left(page(), i, j) = rhs;
         }
      }
   }
   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all pageslice elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to pageslice.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// This assignment operator offers the option to directly assign to all elements of the dense
// pageslice by means of an initializer list. The pageslice elements are assigned the values from the given
// initializer list. Missing values are reset to their default state. Note that in case the size
// of the initializer list exceeds the size of the pageslice, a \a std::invalid_argument exception is
// thrown. Also, if the underlying tensor \a MT is restricted and the assignment would violate
// an invariant of the tensor, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline PageSlice<MT,CRAs...>&
   PageSlice<MT,CRAs...>::operator=(initializer_list<initializer_list<ElementType> > list)
{
   if (list.size() > rows() || determineColumns(list) > columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to pageslice" );
   }

   if( IsRestricted_v<MT> ) {
      const InitializerMatrix<ElementType> tmp( list );
      if( !tryAssign( tensor_, tmp, 0UL, 0UL, page() ) ) {
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
/*!\brief Copy assignment operator for PageSlice.
//
// \param rhs Dense pageslice to be copied.
// \return Reference to the assigned pageslice.
// \exception std::invalid_argument PageSlice sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two pageslices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower or upper triangular tensor and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline PageSlice<MT,CRAs...>&
   PageSlice<MT,CRAs...>::operator=( const PageSlice& rhs )
{
   if( &rhs == this ) return *this;

   if( rows() != rhs.rows() || columns() != rhs.columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "PageSlice sizes do not match" );
   }

   if( !tryAssign( tensor_, rhs, 0UL, 0UL, page() ) ) {
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
// \return Reference to the assigned pageslice.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower or upper triangular tensor and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename VT >     // Type of the right-hand side matrix
inline PageSlice<MT,CRAs...>&
   PageSlice<MT,CRAs...>::operator=( const Matrix<VT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<VT> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<VT> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<MT>, CompositeType_t<VT>, const VT& >;
   Right right( ~rhs );

   if( !tryAssign( tensor_, right, 0UL, 0UL, page() ) ) {
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
// \param rhs The right-hand side matrix to be added to the dense pageslice.
// \return Reference to the assigned pageslice.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower or upper triangular tensor and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename VT >     // Type of the right-hand side matrix
inline PageSlice<MT,CRAs...>&
   PageSlice<MT,CRAs...>::operator+=( const Matrix<VT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<VT> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<VT> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<MT>, CompositeType_t<VT>, const VT& >;
   Right right( ~rhs );

   if( !tryAddAssign( tensor_, right, 0UL, 0UL, page() ) ) {
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
// \param rhs The right-hand side matrix to be subtracted from the dense pageslice.
// \return Reference to the assigned pageslice.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying tensor \a MT is a lower or upper triangular tensor and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename VT >     // Type of the right-hand side matrix
inline PageSlice<MT,CRAs...>&
   PageSlice<MT,CRAs...>::operator-=( const Matrix<VT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<VT> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<VT> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<MT>, CompositeType_t<VT>, const VT& >;
   Right right( ~rhs );

   if( !trySubAssign( tensor_, right, 0UL, 0UL, page() ) ) {
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
// \return Reference to the assigned pageslice.
// \exception std::invalid_argument Invalid matrix size for cross product.
// \exception std::invalid_argument Invalid assignment to restricted tensor.
//
// In case the current size of any of the two matrices is not equal to 3, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename VT >     // Type of the right-hand side matrix
inline PageSlice<MT,CRAs...>&
   PageSlice<MT,CRAs...>::operator%=( const Matrix<VT,false>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<VT> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<VT> );

   using SchurType = SchurTrait_t< ResultType, ResultType_t<VT> >;

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( !trySchurAssign( tensor_, (~rhs), 0UL, 0UL, page() ) ) {
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
/*!\brief Returns the tensor containing the pageslice.
//
// \return The tensor containing the pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline MT& PageSlice<MT,CRAs...>::operand() noexcept
{
   return tensor_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the tensor containing the pageslice.
//
// \return The tensor containing the pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline const MT& PageSlice<MT,CRAs...>::operand() const noexcept
{
   return tensor_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current size/dimension of the pageslice.
//
// \return The size of the pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline size_t PageSlice<MT,CRAs...>::rows() const noexcept
{
   return tensor_.rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current size/dimension of the pageslice.
//
// \return The size of the pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline size_t PageSlice<MT,CRAs...>::columns() const noexcept
{
   return tensor_.columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the minimum capacity of the pageslice.
//
// \return The minimum capacity of the pageslice.
//
// This function returns the minimum capacity of the pageslice, which corresponds to the current size
// plus padding.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline size_t PageSlice<MT,CRAs...>::spacing() const noexcept
{
   return tensor_.spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense pageslice.
//
// \return The maximum capacity of the dense pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline size_t PageSlice<MT,CRAs...>::capacity() const noexcept
{
   return tensor_.capacity( 0UL, page() ) * tensor_.rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense pageslice.
//
// \return The maximum capacity of the dense pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline size_t PageSlice<MT,CRAs...>::capacity( size_t i ) const noexcept
{
   return tensor_.capacity( i, page() ) * tensor_.rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the pageslice.
//
// \return The number of non-zero elements in the pageslice.
//
// Note that the number of non-zero elements is always less than or equal to the current number
// of columns of the tensor containing the pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline size_t PageSlice<MT,CRAs...>::nonZeros() const
{
   size_t count ( 0 );
   for ( size_t i = 0; i < rows(); ++i ) {
      count += tensor_.nonZeros( i, page() );
   }
   return count;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the pageslice.
//
// \return The number of non-zero elements in the pageslice.
//
// Note that the number of non-zero elements is always less than or equal to the current number
// of columns of the tensor containing the pageslice.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline size_t PageSlice<MT,CRAs...>::nonZeros( size_t i ) const
{
   return tensor_.nonZeros( i, page() );
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
        , size_t... CRAs >  // Compile time pageslice arguments
inline void PageSlice<MT,CRAs...>::reset()
{
   for ( size_t i = 0; i < rows(); ++i ) {
      tensor_.reset( i, page() );
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
        , size_t... CRAs >  // Compile time pageslice arguments
inline void PageSlice<MT,CRAs...>::reset( size_t i )
{
   tensor_.reset( i, page() );
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
/*!\brief Scaling of the pageslice by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the pageslice scaling.
// \return Reference to the dense pageslice.
//
// This function scales the pageslice by applying the given scalar value \a scalar to each element
// of the pageslice. For built-in and \c complex data types it has the same effect as using the
// multiplication assignment operator. Note that the function cannot be used to scale a pageslice
// on a lower or upper unitriangular tensor. The attempt to scale such a pageslice results in a
// compile time error!
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename Other >  // Data type of the scalar value
inline PageSlice<MT,CRAs...>&
   PageSlice<MT,CRAs...>::scale( const Other& scalar )
{
   for ( size_t i = 0; i < rows(); ++i ) {
      for ( size_t j = 0; j < columns(); ++j ) {
         tensor_(page(), i, j) *= scalar;
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
/*!\brief Returns whether the dense pageslice can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense pageslice, \a false if not.
//
// This function returns whether the given address can alias with the dense pageslice. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename Other >  // Data type of the foreign expression
inline bool PageSlice<MT,CRAs...>::canAlias( const Other* alias ) const noexcept
{
   return tensor_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense pageslice can alias with the given dense pageslice \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense pageslice, \a false if not.
//
// This function returns whether the given address can alias with the dense pageslice. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT        // Type of the dense tensor
        , size_t... CRAs >   // Compile time pageslice arguments
template< typename MT2       // Data type of the foreign dense pageslice
        , size_t... CRAs2 >  // Compile time pageslice arguments of the foreign dense pageslice
inline bool
   PageSlice<MT,CRAs...>::canAlias( const PageSlice<MT2,CRAs2...>* alias ) const noexcept
{
   return tensor_.isAliased( &alias->tensor_ ) && ( page() == alias->page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense pageslice is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense pageslice, \a false if not.
//
// This function returns whether the given address is aliased with the dense pageslice. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename Other >  // Data type of the foreign expression
inline bool PageSlice<MT,CRAs...>::isAliased( const Other* alias ) const noexcept
{
   return tensor_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense pageslice is aliased with the given dense pageslice \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense pageslice, \a false if not.
//
// This function returns whether the given address is aliased with the dense pageslice. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT        // Type of the dense tensor
        , size_t... CRAs >   // Compile time pageslice arguments
template< typename MT2       // Data type of the foreign dense pageslice
        , size_t... CRAs2 >  // Compile time pageslice arguments of the foreign dense pageslice
inline bool
   PageSlice<MT,CRAs...>::isAliased( const PageSlice<MT2,CRAs2...>* alias ) const noexcept
{
   return tensor_.isAliased( &alias->tensor_ ) && ( page() == alias->page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense pageslice is properly aligned in memory.
//
// \return \a true in case the dense pageslice is aligned, \a false if not.
//
// This function returns whether the dense pageslice is guaranteed to be properly aligned in memory,
// i.e. whether the beginning and the end of the dense pageslice are guaranteed to conform to the
// alignment restrictions of the element type \a Type.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline bool PageSlice<MT,CRAs...>::isAligned() const noexcept
{
   return tensor_.isAligned();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense pageslice can be used in SMP assignments.
//
// \return \a true in case the dense pageslice can be used in SMP assignments, \a false if not.
//
// This function returns whether the dense pageslice can be used in SMP assignments. In contrast to
// the \a smpAssignable member enumeration, which is based solely on compile time information,
// this function additionally provides runtime information (as for instance the current size
// of the dense pageslice).
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
inline bool PageSlice<MT,CRAs...>::canSMPAssign() const noexcept
{
   return ( rows() * columns() > SMP_DMATASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Load of a SIMD element of the dense pageslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense pageslice. The index
// must be smaller than the number of tensor columns. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
BLAZE_ALWAYS_INLINE typename PageSlice<MT,CRAs...>::SIMDType
   PageSlice<MT,CRAs...>::load( size_t i, size_t j ) const noexcept
{
   return tensor_.load( page(), i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the dense pageslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense pageslice.
// The index must be smaller than the number of tensor columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
BLAZE_ALWAYS_INLINE typename PageSlice<MT,CRAs...>::SIMDType
   PageSlice<MT,CRAs...>::loada( size_t i, size_t j ) const noexcept
{
   return tensor_.loada( page(), i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the dense pageslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense pageslice.
// The index must be smaller than the number of tensor columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
BLAZE_ALWAYS_INLINE typename PageSlice<MT,CRAs...>::SIMDType
   PageSlice<MT,CRAs...>::loadu( size_t i, size_t j ) const noexcept
{
   return tensor_.loadu( page(), i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Store of a SIMD element of the dense pageslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store a specific SIMD element of the dense pageslice. The index
// must be smaller than the number of tensor columns. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
BLAZE_ALWAYS_INLINE void
   PageSlice<MT,CRAs...>::store( size_t i, size_t j, const SIMDType& value ) noexcept
{
   tensor_.store( page(), i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the dense pageslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store a specific SIMD element of the dense pageslice. The
// index must be smaller than the number of tensor columns. This function must \b NOT be
// called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
BLAZE_ALWAYS_INLINE void
   PageSlice<MT,CRAs...>::storea( size_t i, size_t j, const SIMDType& value ) noexcept
{
   tensor_.storea( page(), i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unligned store of a SIMD element of the dense pageslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store a specific SIMD element of the dense pageslice.
// The index must be smaller than the number of tensor columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
BLAZE_ALWAYS_INLINE void
   PageSlice<MT,CRAs...>::storeu( size_t i, size_t j, const SIMDType& value ) noexcept
{
   tensor_.storeu( page(), i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned, non-temporal store of a SIMD element of the dense pageslice.
//
// \param index Access index. The index must be smaller than the number of tensor columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store a specific SIMD element of the dense
// pageslice. The index must be smaller than the number of tensor columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename MT       // Type of the dense tensor
        , size_t... CRAs >  // Compile time pageslice arguments
BLAZE_ALWAYS_INLINE void
   PageSlice<MT,CRAs...>::stream( size_t i, size_t j, const SIMDType& value ) noexcept
{
   tensor_.stream( page(), i, j, value );
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
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename VT >     // Type of the right-hand side dense matrix
inline auto PageSlice<MT,CRAs...>::assign( const DenseMatrix<VT,false>& rhs )
   -> DisableIf_t< VectorizedAssign_v<VT> >
{
   BLAZE_INTERNAL_ASSERT( rows() == (~rhs).rows(), "Invalid matrix sizes" );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid matrix sizes" );

   for (size_t i = 0UL; i < (~rhs).rows(); ++i ) {
      const size_t jpos( (~rhs).columns() & size_t(-2) );
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         tensor_(page(),i,j) = (~rhs)(i,j);
         tensor_(page(),i,j+1UL) = (~rhs)(i,j+1UL);
      }
      if( jpos < (~rhs).columns() )
         tensor_(page(),i,jpos) = (~rhs)(i,jpos);
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
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename VT >     // Type of the right-hand side dense matrix
inline auto PageSlice<MT,CRAs...>::assign( const DenseMatrix<VT,false>& rhs )
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
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename VT >     // Type of the right-hand side dense matrix
inline auto PageSlice<MT,CRAs...>::addAssign( const DenseMatrix<VT,false>& rhs )
   -> DisableIf_t< VectorizedAddAssign_v<VT> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   for (size_t i = 0UL; i < (~rhs).rows(); ++i ) {
      const size_t jpos( (~rhs).columns() & size_t(-2) );
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         tensor_(page(),i,j    ) += (~rhs)(i,j);
         tensor_(page(),i,j+1UL) += (~rhs)(i,j+1UL);
      }
      if( jpos < (~rhs).columns() )
         tensor_(page(),i,jpos) += (~rhs)(i,jpos);
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
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename VT >     // Type of the right-hand side dense matrix
inline auto PageSlice<MT,CRAs...>::addAssign( const DenseMatrix<VT,false>& rhs )
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
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename VT >     // Type of the right-hand side dense matrix
inline auto PageSlice<MT,CRAs...>::subAssign( const DenseMatrix<VT,false>& rhs )
   -> DisableIf_t< VectorizedSubAssign_v<VT> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   for (size_t i = 0UL; i < (~rhs).rows(); ++i ) {
      const size_t jpos( (~rhs).columns() & size_t(-2) );
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         tensor_(page(),i,j    ) -= (~rhs)(i,j);
         tensor_(page(),i,j+1UL) -= (~rhs)(i,j+1UL);
      }
      if( jpos < (~rhs).columns() )
         tensor_(page(),i,jpos) -= (~rhs)(i,jpos);
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
        , size_t... CRAs >  // Compile time pageslice arguments
template< typename VT >     // Type of the right-hand side dense matrix
inline auto PageSlice<MT,CRAs...>::subAssign( const DenseMatrix<VT,false>& rhs )
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
        , size_t... CSAs >  // Compile time pageslice arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline auto PageSlice<MT,CSAs...>::schurAssign( const DenseMatrix<MT2,false>& rhs )
   -> DisableIf_t< VectorizedSchurAssign_v<MT2> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<rows(); ++i ) {
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         tensor_(page(),i,j    ) *= (~rhs)(i,j    );
         tensor_(page(),i,j+1UL) *= (~rhs)(i,j+1UL);
      }
      if( jpos < columns() ) {
         tensor_(page(),i,jpos) *= (~rhs)(i,jpos);
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
        , size_t... CSAs >  // Compile time pageslice arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline auto PageSlice<MT,CSAs...>::schurAssign( const DenseMatrix<MT2,false>& rhs )
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
