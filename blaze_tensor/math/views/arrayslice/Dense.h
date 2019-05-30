//=================================================================================================
/*!
//  \file blaze_array/math/views/arrayslice/Dense.h
//  \brief ArraySlice specialization for dense arrays
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_ARRAYSLICE_DENSE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_ARRAYSLICE_DENSE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <array>
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
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDDiv.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/HasSIMDSub.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsRestricted.h>
#include <blaze/math/typetraits/IsSIMDCombinable.h>
#include <blaze/math/typetraits/IsSparseVector.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsTriangular.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/IsUniUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/views/Check.h>
#include <blaze/system/CacheSize.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/TypeList.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsReference.h>
#include <blaze/util/typetraits/RemoveReference.h>

#include <blaze_tensor/math/InitializerList.h>
#include <blaze_tensor/math/constraints/DenseArray.h>
// #include <blaze_tensor/math/constraints/Subarray.h>
#include <blaze_tensor/math/traits/ArraySliceTrait.h>
#include <blaze_tensor/math/views/arrayslice/BaseTemplate.h>
#include <blaze_tensor/math/views/arrayslice/ArraySliceData.h>


namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR DENSE TENSORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of ArraySlice for arrayslices on arrayslice-major dense arrays.
// \ingroup arrayslice
//
// This specialization of ArraySlice adapts the class template to the requirements of arrayslice-major
// dense arrays.
*/
template< size_t M          // Dimension of the ArraysSlice
        , typename MT       // Type of the dense array
        , size_t... CRAs >  // Compile time arrayslice arguments
class ArraySlice
   : public View< DenseArray< ArraySlice<M,MT,CRAs...> > >
   , private ArraySliceData<CRAs...>
{
 private:
   //**Type definitions****************************************************************************
   using DataType = ArraySliceData<CRAs...>;                     //!< The type of the ArraySliceData base class.
   using Operand  = If_t< IsExpression_v<MT>, MT, MT& >;  //!< Composite data type of the dense array expression.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this ArraySlice instance.
   using This          = ArraySlice<M,MT,CRAs...>;

   using BaseType      = DenseArray<This>;             //!< Base type of this ArraySlice instance.
   using ViewedType    = MT;                           //!< The type viewed by this ArraySlice instance.
   using ResultType    = ArraySliceTrait_t<M,MT,CRAs...>; //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;   //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;  //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<MT>;            //!< Type of the arrayslice elements.
   using SIMDType      = SIMDTrait_t<ElementType>;     //!< SIMD type of the arrayslice elements.
   using ReturnType    = ReturnType_t<MT>;             //!< Return type for expression template evaluations
   using CompositeType = const ArraySlice&;             //!< Data type for composite expression templates.

   //! Reference to a constant arrayslice value.
   using ConstReference = ConstReference_t<MT>;

   //! Reference to a non-constant arrayslice value.
   using Reference = If_t< IsConst_v<MT>, ConstReference, Reference_t<MT> >;

   //! Pointer to a constant arrayslice value.
   using ConstPointer = ConstPointer_t<MT>;

   //! Pointer to a non-constant arrayslice value.
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

   //! Compile time dimensions of this array slice.
   static constexpr size_t N = MT::num_dimensions() - 1;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RRAs >
   explicit inline ArraySlice( MT& array, RRAs... args );

   ArraySlice( const ArraySlice& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~ArraySlice() = default;
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   template< typename... Dims >
   inline Reference      operator()( Dims... dims );
   template< typename... Dims >
   inline ConstReference operator()( Dims... dims ) const;
   inline Reference      operator()( std::array< size_t, N > const& indices );
   inline ConstReference operator()( std::array< size_t, N > const& indices ) const;
   template< typename... Dims >
   inline Reference      at( Dims... dims );
   template< typename... Dims >
   inline ConstReference at( Dims... dims ) const;
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
   inline ArraySlice& operator=( const ElementType& rhs );
   inline ArraySlice& operator=( nested_initializer_list<N, ElementType> list );
   inline ArraySlice& operator=( const ArraySlice& rhs );

   template< typename MT2 > inline ArraySlice& operator= ( const Array<MT2>& rhs );
   template< typename MT2 > inline ArraySlice& operator+=( const Array<MT2>& rhs );
   template< typename MT2 > inline ArraySlice& operator-=( const Array<MT2>& rhs );
   template< typename MT2 > inline ArraySlice& operator%=( const Array<MT2>& rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   using DataType::index;

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
   template< typename Other > inline ArraySlice& scale( const Other& scalar );
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
        HasSIMDAdd_v< ElementType, ElementType_t<MT2> > );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT2 >
   static constexpr bool VectorizedSubAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && MT2::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<MT2> > &&
        HasSIMDSub_v< ElementType, ElementType_t<MT2> > );
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

   template< size_t M2, typename MT2, size_t... CRAs2 >
   inline bool canAlias( const ArraySlice<M2,MT2,CRAs2...>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< size_t M2, typename MT2, size_t... CRAs2 >
   inline bool isAliased( const ArraySlice<M2,MT2,CRAs2...>* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   template< typename... Dims >
   BLAZE_ALWAYS_INLINE SIMDType load ( Dims... dims ) const noexcept;
   template< typename... Dims >
   BLAZE_ALWAYS_INLINE SIMDType loada( Dims... dims ) const noexcept;
   template< typename... Dims >
   BLAZE_ALWAYS_INLINE SIMDType loadu( Dims... dims ) const noexcept;

   template< typename... Dims >
   BLAZE_ALWAYS_INLINE void store ( const SIMDType& value, Dims... dims ) noexcept;
   template< typename... Dims >
   BLAZE_ALWAYS_INLINE void storea( const SIMDType& value, Dims... dims ) noexcept;
   template< typename... Dims >
   BLAZE_ALWAYS_INLINE void storeu( const SIMDType& value, Dims... dims ) noexcept;
   template< typename... Dims >
   BLAZE_ALWAYS_INLINE void stream( const SIMDType& value, Dims... dims ) noexcept;

   template< typename MT2 >
   inline auto assign( const DenseArray<MT2>& rhs ) -> DisableIf_t< VectorizedAssign_v<MT2> >;

   template< typename MT2 >
   inline auto assign( const DenseArray<MT2>& rhs ) -> EnableIf_t< VectorizedAssign_v<MT2> >;

   template< typename MT2 >
   inline auto addAssign( const DenseArray<MT2>& rhs ) -> DisableIf_t< VectorizedAddAssign_v<MT2> >;

   template< typename MT2 >
   inline auto addAssign( const DenseArray<MT2>& rhs ) -> EnableIf_t< VectorizedAddAssign_v<MT2> >;

   template< typename MT2 >
   inline auto subAssign( const DenseArray<MT2>& rhs ) -> DisableIf_t< VectorizedSubAssign_v<MT2> >;

   template< typename MT2 >
   inline auto subAssign( const DenseArray<MT2>& rhs ) -> EnableIf_t< VectorizedSubAssign_v<MT2> >;

   template< typename MT2 >
   inline auto schurAssign( const DenseArray<MT2>& rhs ) -> DisableIf_t< VectorizedSchurAssign_v<MT2> >;

   template< typename MT2 >
   inline auto schurAssign( const DenseArray<MT2>& rhs ) -> EnableIf_t< VectorizedSchurAssign_v<MT2> >;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand array_;  //!< The array containing the arrayslice.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< size_t M2, typename MT2, size_t... CRAs2 > friend class ArraySlice;
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE   ( MT );
//    BLAZE_CONSTRAINT_MUST_NOT_BE_SUBARRAY_TYPE   ( MT );
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
/*!\brief Constructor for arrayslices on arrayslice-major dense arrays.
//
// \param array The array containing the arrayslice.
// \param args The runtime arrayslice arguments.
// \exception std::invalid_argument Invalid arrayslice access index.
//
// By default, the provided arrayslice arguments are checked at runtime. In case the arrayslice is not properly
// specified (i.e. if the specified index is greater than the number of pages of the given array)
// a \a std::invalid_argument exception is thrown. The checks can be skipped by providing the
// optional \a blaze::unchecked argument.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename... RRAs >  // Runtime arrayslice arguments
inline ArraySlice<M,MT,CRAs...>::ArraySlice( MT& array, RRAs... args )
   : DataType( args... )  // Base class initialization
   , array_ ( array  )  // The array containing the arrayslice
{
   if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
      if( array_.template dimension<M>() <= index() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid arrayslice access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( index() < array_..template dimension<M>(), "Invalid arrayslice access index" );
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
/*!\brief Subscript operator for the direct access to the arrayslice elements.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename... Dims >
inline typename ArraySlice<M,MT,CRAs...>::Reference
   ArraySlice<M,MT,CRAs...>::operator()( Dims... dims )
{
   BLAZE_STATIC_ASSERT( M < sizeof...(Dims) );

   return array_( fused_indices< M >( index(), dims... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subscript operator for the direct access to the arrayslice elements.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename... Dims >
inline typename ArraySlice<M,MT,CRAs...>::ConstReference
   ArraySlice<M,MT,CRAs...>::operator()( Dims... dims ) const
{
   BLAZE_STATIC_ASSERT( M < sizeof...(Dims) );

   return const_cast< const MT& >( array_ )( fused_indices< M >( index(), dims... ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subscript operator for the direct access to the arrayslice elements.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline typename ArraySlice<M,MT,CRAs...>::Reference
   ArraySlice<M,MT,CRAs...>::operator()( std::array< size_t, N > const& indices )
{
   BLAZE_STATIC_ASSERT( M < N );

   return array_( fused_indices< M >( index(), indices ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subscript operator for the direct access to the arrayslice elements.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline typename ArraySlice<M,MT,CRAs...>::ConstReference
   ArraySlice<M,MT,CRAs...>::operator()( std::array< size_t, N > const& indices ) const
{
   BLAZE_STATIC_ASSERT( M < N );

   return const_cast< const MT& >( array_ )( fused_indices< M >( index(), indices ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the arrayslice elements.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid arrayslice access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename... Dims >
inline typename ArraySlice<M,MT,CRAs...>::Reference
   ArraySlice<M,MT,CRAs...>::at( Dims... dims )
{
   size_t indices[] = { size_t(dims)... };
   ArrayDimForEach( dimensions(), [&]( size_t i ) {
      if( indices[N - i - 1] >= dimensions()[i] ) {
         BLAZE_THROW_OUT_OF_RANGE("Invalid array access index");
      }
   } );

   return ( *this )( dims... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the arrayslice elements.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid arrayslice access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename... Dims >
inline typename ArraySlice<M,MT,CRAs...>::ConstReference
   ArraySlice<M,MT,CRAs...>::at( Dims... dims) const
{
   size_t indices[] = { size_t(dims)... };
   ArrayDimForEach( dimensions(), [&]( size_t i ) {
      if( indices[N - i - 1] >= dimensions()[i] ) {
         BLAZE_THROW_OUT_OF_RANGE("Invalid array access index");
      }
   } );

   return ( *this )( dims... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the arrayslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense arrayslice. Note that in case
// of a column-major array you can NOT assume that the arrayslice elements lie adjacent to each other!
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline typename ArraySlice<M,MT,CRAs...>::Pointer
   ArraySlice<M,MT,CRAs...>::data() noexcept
{
   return nullptr; //array_.data( fused_indices<M>( index(), 0, 0 ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the arrayslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense arrayslice. Note that in case
// of a column-major array you can NOT assume that the arrayslice elements lie adjacent to each other!
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline typename ArraySlice<M,MT,CRAs...>::ConstPointer
   ArraySlice<M,MT,CRAs...>::data() const noexcept
{
   return nullptr; //array_.data( 0, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the arrayslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense arrayslice. Note that in case
// of a column-major array you can NOT assume that the arrayslice elements lie adjacent to each other!
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline typename ArraySlice<M,MT,CRAs...>::Pointer
   ArraySlice<M,MT,CRAs...>::data( size_t i ) noexcept
{
   return nullptr; // array_.data( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the arrayslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense arrayslice. Note that in case
// of a column-major array you can NOT assume that the arrayslice elements lie adjacent to each other!
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline typename ArraySlice<M,MT,CRAs...>::ConstPointer
   ArraySlice<M,MT,CRAs...>::data( size_t i ) const noexcept
{
   return nullptr; // array_.data( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the arrayslice.
//
// \param i The row/column index.
// \return Iterator to the first element of the given row on this arrayslice.
//
// This function returns an iterator to the first element of the arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline typename ArraySlice<M,MT,CRAs...>::Iterator
   ArraySlice<M,MT,CRAs...>::begin( size_t i )
{
   return array_.begin( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the arrayslice.
//
// \param i The row/column index.
// \return Iterator to the first element of the arrayslice.
//
// This function returns an iterator to the first element of the arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline typename ArraySlice<M,MT,CRAs...>::ConstIterator
   ArraySlice<M,MT,CRAs...>::begin( size_t i ) const
{
   return array_.cbegin( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the arrayslice.
//
// \param i The row/column index.
// \return Iterator to the first element of the arrayslice.
//
// This function returns an iterator to the first element of the arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline typename ArraySlice<M,MT,CRAs...>::ConstIterator
   ArraySlice<M,MT,CRAs...>::cbegin( size_t i ) const
{
   return array_.cbegin( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the arrayslice.
//
// \param i The row/column index.
// \return Iterator just past the last element of the arrayslice.
//
// This function returns an iterator just past the last element of the arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline typename ArraySlice<M,MT,CRAs...>::Iterator
   ArraySlice<M,MT,CRAs...>::end( size_t i )
{
   return array_.end( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the arrayslice.
//
// \param i The row/column index.
// \return Iterator just past the last element of the arrayslice.
//
// This function returns an iterator just past the last element of the arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline typename ArraySlice<M,MT,CRAs...>::ConstIterator
   ArraySlice<M,MT,CRAs...>::end( size_t i ) const
{
   return array_.cend( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the arrayslice.
//
// \param i The row/column index.
// \return Iterator just past the last element of the arrayslice.
//
// This function returns an iterator just past the last element of the arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline typename ArraySlice<M,MT,CRAs...>::ConstIterator
   ArraySlice<M,MT,CRAs...>::cend( size_t i ) const
{
   return array_.cend( i, page() );
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
/*!\brief Homogeneous assignment to all arrayslice elements.
//
// \param rhs Scalar value to be assigned to all arrayslice elements.
// \return Reference to the assigned arrayslice.
//
// This function homogeneously assigns the given value to all elements of the arrayslice. Note that in
// case the underlying dense array is a lower/upper array only lower/upper and diagonal elements
// of the underlying array are modified.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline ArraySlice<M,MT,CRAs...>&
   ArraySlice<M,MT,CRAs...>::operator=( const ElementType& rhs )
{
   decltype(auto) left( derestrict( array_ ) );

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
/*!\brief List assignment to all arrayslice elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to arrayslice.
// \exception std::invalid_argument Invalid assignment to restricted array.
//
// This assignment operator offers the option to directly assign to all elements of the dense
// arrayslice by means of an initializer list. The arrayslice elements are assigned the values from the given
// initializer list. Missing values are reset to their default state. Note that in case the size
// of the initializer list exceeds the size of the arrayslice, a \a std::invalid_argument exception is
// thrown. Also, if the underlying array \a MT is restricted and the assignment would violate
// an invariant of the array, a \a std::invalid_argument exception is thrown.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline ArraySlice<M,MT,CRAs...>&
   ArraySlice<M,MT,CRAs...>::operator=(nested_initializer_list<N, ElementType> list)
{
   if (list.size() > rows() || determineColumns(list) > columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to arrayslice" );
   }

   if( IsRestricted_v<MT> ) {
      const InitializerArray<ElementType> tmp( list );
      if( !tryAssign( array_, tmp, 0UL, 0UL, page() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted array" );
      }
   }

   decltype(auto) left( derestrict( *this ) );

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      std::fill( std::copy( rowList.begin(), rowList.end(), left.begin( i ) ), left.end( i ), ElementType() );
      ++i;
   }

   BLAZE_INTERNAL_ASSERT( isIntact( array_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for ArraySlice.
//
// \param rhs Dense arrayslice to be copied.
// \return Reference to the assigned arrayslice.
// \exception std::invalid_argument ArraySlice sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted array.
//
// In case the current sizes of the two arrayslices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying array \a MT is a lower or upper triangular array and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline ArraySlice<M,MT,CRAs...>&
   ArraySlice<M,MT,CRAs...>::operator=( const ArraySlice& rhs )
{
   if( &rhs == this ) return *this;

   if( rows() != rhs.rows() || columns() != rhs.columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "ArraySlice sizes do not match" );
   }

   if( !tryAssign( array_, rhs, 0UL, 0UL, page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted array" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsExpression_v<MT> && rhs.canAlias( &array_ ) ) {
      const ResultType tmp( rhs );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( array_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different matrices.
//
// \param rhs Array to be assigned.
// \return Reference to the assigned arrayslice.
// \exception std::invalid_argument Array sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted array.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying array \a MT is a lower or upper triangular array and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename MT2 >    // Type of the right-hand side matrix
inline ArraySlice<M,MT,CRAs...>&
   ArraySlice<M,MT,CRAs...>::operator=( const Array<MT2>& rhs )
{
   //BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<MT2> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<MT2> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<MT>, CompositeType_t<MT2>, const MT2& >;
   Right right( ~rhs );

   if( !tryAssign( array_, right, 0UL, 0UL, page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted array" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &array_ ) ) {
      const ResultType_t<MT2> tmp( right );
      smpAssign( left, tmp );
   }
   else {
      if( IsSparseArray_v<MT2> )
         reset();
      smpAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( array_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ \vec{a}+=\vec{b} \f$).
//
// \param rhs The right-hand side matrix to be added to the dense arrayslice.
// \return Reference to the assigned arrayslice.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted array.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying array \a MT is a lower or upper triangular array and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename MT2 >     // Type of the right-hand side matrix
inline ArraySlice<M,MT,CRAs...>&
   ArraySlice<M,MT,CRAs...>::operator+=( const Array<MT2>& rhs )
{
   //BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<MT2> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<MT2> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<MT>, CompositeType_t<MT2>, const MT2& >;
   Right right( ~rhs );

   if( !tryAddAssign( array_, right, 0UL, 0UL, page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted array" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &array_ ) ) {
      const ResultType_t<MT2> tmp( right );
      smpAddAssign( left, tmp );
   }
   else {
      smpAddAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( array_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ \vec{a}-=\vec{b} \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the dense arrayslice.
// \return Reference to the assigned arrayslice.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted array.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying array \a MT is a lower or upper triangular array and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename MT2 >     // Type of the right-hand side matrix
inline ArraySlice<M,MT,CRAs...>&
   ArraySlice<M,MT,CRAs...>::operator-=( const Array<MT2>& rhs )
{
   //BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<MT2> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<MT2> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<MT>, CompositeType_t<MT2>, const MT2& >;
   Right right( ~rhs );

   if( !trySubAssign( array_, right, 0UL, 0UL, page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted array" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &array_ ) ) {
      const ResultType_t<MT2> tmp( right );
      smpSubAssign( left, tmp );
   }
   else {
      smpSubAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( array_ ), "Invariant violation detected" );

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
// \return Reference to the assigned arrayslice.
// \exception std::invalid_argument Invalid matrix size for cross product.
// \exception std::invalid_argument Invalid assignment to restricted array.
//
// In case the current size of any of the two matrices is not equal to 3, a \a std::invalid_argument
// exception is thrown.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename MT2 >     // Type of the right-hand side matrix
inline ArraySlice<M,MT,CRAs...>&
   ArraySlice<M,MT,CRAs...>::operator%=( const Array<MT2>& rhs )
{
   using blaze::assign;

   //BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<MT2> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<MT2> );

   using SchurType = SchurTrait_t< ResultType, ResultType_t<MT2> >;

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array sizes do not match" );
   }

   if( !trySchurAssign( array_, (~rhs), 0UL, 0UL, page() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted array" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<MT> && (~rhs).canAlias( &array_ ) ) {
      const SchurType tmp( *this % (~rhs) );
      smpSchurAssign( left, tmp );
   }
   else {
      smpSchurAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( array_ ), "Invariant violation detected" );

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
/*!\brief Returns the array containing the arrayslice.
//
// \return The array containing the arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline MT& ArraySlice<M,MT,CRAs...>::operand() noexcept
{
   return array_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the array containing the arrayslice.
//
// \return The array containing the arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline const MT& ArraySlice<M,MT,CRAs...>::operand() const noexcept
{
   return array_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current size/dimension of the arrayslice.
//
// \return The size of the arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline size_t ArraySlice<M,MT,CRAs...>::rows() const noexcept
{
   return array_.rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current size/dimension of the arrayslice.
//
// \return The size of the arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline size_t ArraySlice<M,MT,CRAs...>::columns() const noexcept
{
   return array_.columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the minimum capacity of the arrayslice.
//
// \return The minimum capacity of the arrayslice.
//
// This function returns the minimum capacity of the arrayslice, which corresponds to the current size
// plus padding.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline size_t ArraySlice<M,MT,CRAs...>::spacing() const noexcept
{
   return array_.spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense arrayslice.
//
// \return The maximum capacity of the dense arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline size_t ArraySlice<M,MT,CRAs...>::capacity() const noexcept
{
   return array_.capacity( 0UL, page() ) * array_.rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense arrayslice.
//
// \return The maximum capacity of the dense arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline size_t ArraySlice<M,MT,CRAs...>::capacity( size_t i ) const noexcept
{
   return array_.capacity( i, page() ) * array_.rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the arrayslice.
//
// \return The number of non-zero elements in the arrayslice.
//
// Note that the number of non-zero elements is always less than or equal to the current number
// of columns of the array containing the arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline size_t ArraySlice<M,MT,CRAs...>::nonZeros() const
{
   size_t count ( 0 );
   for ( size_t i = 0; i < rows(); ++i ) {
      count += array_.nonZeros( i, page() );
   }
   return count;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the arrayslice.
//
// \return The number of non-zero elements in the arrayslice.
//
// Note that the number of non-zero elements is always less than or equal to the current number
// of columns of the array containing the arrayslice.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline size_t ArraySlice<M,MT,CRAs...>::nonZeros( size_t i ) const
{
   return array_.nonZeros( i, page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline void ArraySlice<M,MT,CRAs...>::reset()
{
   for ( size_t i = 0; i < rows(); ++i ) {
      array_.reset( i, page() );
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
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline void ArraySlice<M,MT,CRAs...>::reset( size_t i )
{
   array_.reset( i, page() );
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
/*!\brief Scaling of the arrayslice by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the arrayslice scaling.
// \return Reference to the dense arrayslice.
//
// This function scales the arrayslice by applying the given scalar value \a scalar to each element
// of the arrayslice. For built-in and \c complex data types it has the same effect as using the
// multiplication assignment operator. Note that the function cannot be used to scale a arrayslice
// on a lower or upper unitriangular array. The attempt to scale such a arrayslice results in a
// compile time error!
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename Other >  // Data type of the scalar value
inline ArraySlice<M,MT,CRAs...>&
   ArraySlice<M,MT,CRAs...>::scale( const Other& scalar )
{
   for ( size_t i = 0; i < rows(); ++i ) {
      for ( size_t j = 0; j < columns(); ++j ) {
         array_(page(), i, j) *= scalar;
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
/*!\brief Returns whether the dense arrayslice can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense arrayslice, \a false if not.
//
// This function returns whether the given address can alias with the dense arrayslice. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename Other >  // Data type of the foreign expression
inline bool ArraySlice<M,MT,CRAs...>::canAlias( const Other* alias ) const noexcept
{
   return array_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense arrayslice can alias with the given dense arrayslice \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense arrayslice, \a false if not.
//
// This function returns whether the given address can alias with the dense arrayslice. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< size_t M2           // Dimension of the ArraysSlice
        , typename MT2        // Data type of the foreign dense arrayslice
        , size_t... CRAs2 >   // Compile time arrayslice arguments of the foreign dense arrayslice
inline bool
   ArraySlice<M,MT,CRAs...>::canAlias( const ArraySlice<M2,MT2,CRAs2...>* alias ) const noexcept
{
   return array_.isAliased( &alias->array_ ) && ( page() == alias->page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense arrayslice is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense arrayslice, \a false if not.
//
// This function returns whether the given address is aliased with the dense arrayslice. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename Other >  // Data type of the foreign expression
inline bool ArraySlice<M,MT,CRAs...>::isAliased( const Other* alias ) const noexcept
{
   return array_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense arrayslice is aliased with the given dense arrayslice \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense arrayslice, \a false if not.
//
// This function returns whether the given address is aliased with the dense arrayslice. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< size_t M2           // Dimension of the ArraysSlice
        , typename MT2        // Data type of the foreign dense arrayslice
        , size_t... CRAs2 >   // Compile time arrayslice arguments of the foreign dense arrayslice
inline bool
   ArraySlice<M,MT,CRAs...>::isAliased( const ArraySlice<M2,MT2,CRAs2...>* alias ) const noexcept
{
   return array_.isAliased( &alias->array_ ) && ( page() == alias->page() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense arrayslice is properly aligned in memory.
//
// \return \a true in case the dense arrayslice is aligned, \a false if not.
//
// This function returns whether the dense arrayslice is guaranteed to be properly aligned in memory,
// i.e. whether the beginning and the end of the dense arrayslice are guaranteed to conform to the
// alignment restrictions of the element type \a Type.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline bool ArraySlice<M,MT,CRAs...>::isAligned() const noexcept
{
   return array_.isAligned();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense arrayslice can be used in SMP assignments.
//
// \return \a true in case the dense arrayslice can be used in SMP assignments, \a false if not.
//
// This function returns whether the dense arrayslice can be used in SMP assignments. In contrast to
// the \a smpAssignable member enumeration, which is based solely on compile time information,
// this function additionally provides runtime information (as for instance the current size
// of the dense arrayslice).
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
inline bool ArraySlice<M,MT,CRAs...>::canSMPAssign() const noexcept
{
   return ( rows() * columns() > SMP_DMATASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Load of a SIMD element of the dense arrayslice.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense arrayslice. The index
// must be smaller than the number of array columns. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename... Dims >
BLAZE_ALWAYS_INLINE typename ArraySlice<M,MT,CRAs...>::SIMDType
   ArraySlice<M,MT,CRAs...>::load( Dims... dims ) const noexcept
{
   return array_.load( page(), i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the dense arrayslice.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense arrayslice.
// The index must be smaller than the number of array columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename... Dims >
BLAZE_ALWAYS_INLINE typename ArraySlice<M,MT,CRAs...>::SIMDType
   ArraySlice<M,MT,CRAs...>::loada( Dims... dims ) const noexcept
{
   return array_.loada( page(), i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the dense arrayslice.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense arrayslice.
// The index must be smaller than the number of array columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename... Dims >
BLAZE_ALWAYS_INLINE typename ArraySlice<M,MT,CRAs...>::SIMDType
   ArraySlice<M,MT,CRAs...>::loadu( Dims... dims ) const noexcept
{
   return array_.loadu( page(), i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Store of a SIMD element of the dense arrayslice.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store a specific SIMD element of the dense arrayslice. The index
// must be smaller than the number of array columns. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename... Dims >
BLAZE_ALWAYS_INLINE void
   ArraySlice<M,MT,CRAs...>::store( const SIMDType& value, Dims... dims ) noexcept
{
   array_.store( page(), i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the dense arrayslice.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store a specific SIMD element of the dense arrayslice. The
// index must be smaller than the number of array columns. This function must \b NOT be
// called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename... Dims >
BLAZE_ALWAYS_INLINE void
   ArraySlice<M,MT,CRAs...>::storea( const SIMDType& value, Dims... dims ) noexcept
{
   array_.storea( page(), i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unligned store of a SIMD element of the dense arrayslice.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store a specific SIMD element of the dense arrayslice.
// The index must be smaller than the number of array columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename... Dims >
BLAZE_ALWAYS_INLINE void
   ArraySlice<M,MT,CRAs...>::storeu( const SIMDType& value, Dims... dims ) noexcept
{
   array_.storeu( page(), i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned, non-temporal store of a SIMD element of the dense arrayslice.
//
// \param index Access index. The index must be smaller than the number of array columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store a specific SIMD element of the dense
// arrayslice. The index must be smaller than the number of array columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename... Dims >
BLAZE_ALWAYS_INLINE void
   ArraySlice<M,MT,CRAs...>::stream( const SIMDType& value, Dims... dims ) noexcept
{
   array_.stream( page(), i, j, value );
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
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename MT2 >     // Type of the right-hand side dense matrix
inline auto ArraySlice<M,MT,CRAs...>::assign( const DenseArray<MT2>& rhs )
   -> DisableIf_t< VectorizedAssign_v<MT2> >
{
   BLAZE_INTERNAL_ASSERT( rows() == (~rhs).rows(), "Invalid matrix sizes" );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid matrix sizes" );

   for (size_t i = 0UL; i < (~rhs).rows(); ++i ) {
      const size_t jpos( (~rhs).columns() & size_t(-2) );
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         array_(page(),i,j) = (~rhs)(i,j);
         array_(page(),i,j+1UL) = (~rhs)(i,j+1UL);
      }
      if( jpos < (~rhs).columns() )
         array_(page(),i,jpos) = (~rhs)(i,jpos);
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
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename MT2 >     // Type of the right-hand side dense matrix
inline auto ArraySlice<M,MT,CRAs...>::assign( const DenseArray<MT2>& rhs )
   -> EnableIf_t< VectorizedAssign_v<MT2> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   constexpr bool remainder( !IsPadded_v<MT> || !IsPadded_v<MT2> );

   const size_t cols( columns() );

   for (size_t i = 0; i < (~rhs).rows(); ++i) {
      const size_t jpos( ( remainder )?( cols & size_t(-SIMDSIZE) ):( cols ) );
      BLAZE_INTERNAL_ASSERT( !remainder || ( cols - ( cols % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );
      Iterator left( begin(i) );
      ConstIterator_t<MT2> right( (~rhs).begin(i) );

      if( useStreaming && cols > ( cacheSize/( sizeof(ElementType) * 3UL ) ) && !(~rhs).isAliased( &array_ ) )
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
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename MT2 >     // Type of the right-hand side dense matrix
inline auto ArraySlice<M,MT,CRAs...>::addAssign( const DenseArray<MT2>& rhs )
   -> DisableIf_t< VectorizedAddAssign_v<MT2> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   for (size_t i = 0UL; i < (~rhs).rows(); ++i ) {
      const size_t jpos( (~rhs).columns() & size_t(-2) );
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         array_(page(),i,j    ) += (~rhs)(i,j);
         array_(page(),i,j+1UL) += (~rhs)(i,j+1UL);
      }
      if( jpos < (~rhs).columns() )
         array_(page(),i,jpos) += (~rhs)(i,jpos);
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
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename MT2 >     // Type of the right-hand side dense matrix
inline auto ArraySlice<M,MT,CRAs...>::addAssign( const DenseArray<MT2>& rhs )
   -> EnableIf_t< VectorizedAddAssign_v<MT2> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   constexpr bool remainder( !IsPadded_v<MT> || !IsPadded_v<MT2> );

   const size_t cols( columns() );

   for (size_t i = 0; i < (~rhs).rows(); ++i) {
      const size_t jpos( ( remainder )?( cols & size_t(-SIMDSIZE) ):( cols ) );
      BLAZE_INTERNAL_ASSERT( !remainder || ( cols - ( cols % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );
      Iterator left( begin(i) );
      ConstIterator_t<MT2> right( (~rhs).begin(i) );

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
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename MT2 >     // Type of the right-hand side dense matrix
inline auto ArraySlice<M,MT,CRAs...>::subAssign( const DenseArray<MT2>& rhs )
   -> DisableIf_t< VectorizedSubAssign_v<MT2> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   for (size_t i = 0UL; i < (~rhs).rows(); ++i ) {
      const size_t jpos( (~rhs).columns() & size_t(-2) );
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         array_(page(),i,j    ) -= (~rhs)(i,j);
         array_(page(),i,j+1UL) -= (~rhs)(i,j+1UL);
      }
      if( jpos < (~rhs).columns() )
         array_(page(),i,jpos) -= (~rhs)(i,jpos);
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
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CRAs >    // Compile time arrayslice arguments
template< typename MT2 >     // Type of the right-hand side dense matrix
inline auto ArraySlice<M,MT,CRAs...>::subAssign( const DenseArray<MT2>& rhs )
   -> EnableIf_t< VectorizedSubAssign_v<MT2> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   constexpr bool remainder( !IsPadded_v<MT> || !IsPadded_v<MT2> );

   const size_t cols( columns() );

   for (size_t i = 0; i < (~rhs).rows(); ++i) {
      const size_t jpos( ( remainder )?( cols & size_t(-SIMDSIZE) ):( cols ) );
      BLAZE_INTERNAL_ASSERT( !remainder || ( cols - ( cols % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );
      Iterator left( begin(i) );
      ConstIterator_t<MT2> right( (~rhs).begin(i) );

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
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CSAs >    // Compile time arrayslice arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline auto ArraySlice<M,MT,CSAs...>::schurAssign( const DenseArray<MT2>& rhs )
   -> DisableIf_t< VectorizedSchurAssign_v<MT2> >
{
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( columns() & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( columns() - ( columns() % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<rows(); ++i ) {
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         array_(page(),i,j    ) *= (~rhs)(i,j    );
         array_(page(),i,j+1UL) *= (~rhs)(i,j+1UL);
      }
      if( jpos < columns() ) {
         array_(page(),i,jpos) *= (~rhs)(i,jpos);
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
template< size_t M            // Dimension of the ArraysSlice
        , typename MT         // Type of the dense array
        , size_t... CSAs >    // Compile time arrayslice arguments
template< typename MT2 >    // Type of the right-hand side dense matrix
inline auto ArraySlice<M,MT,CSAs...>::schurAssign( const DenseArray<MT2>& rhs )
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