//=================================================================================================
/*!
//  \file blaze_quaternion/math/views/quatslice/Dense.h
//  \brief QuatSlice specialization for dense quaternions
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_QUATSLICE_DENSE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_QUATSLICE_DENSE_H_


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
#include <blaze_tensor/math/expressions/Array.h>
#include <blaze_tensor/math/traits/QuatSliceTrait.h>
#include <blaze_tensor/math/views/quatslice/BaseTemplate.h>
#include <blaze_tensor/math/views/quatslice/QuatSliceData.h>


namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR DENSE TENSORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of QuatSlice for quatslices on quatslice-major dense quaternions.
// \ingroup quatslice
//
// This specialization of QuatSlice adapts the class template to the requirements of quatslice-major
// dense quaternions.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
class QuatSlice
   : public View< DenseTensor< QuatSlice<AT,CRAs...> > >
   , private QuatSliceData<CRAs...>
{
 private:
   //**Type definitions****************************************************************************
   using DataType = QuatSliceData<CRAs...>;                     //!< The type of the QuatSliceData base class.
   using Operand  = If_t< IsExpression_v<AT>, AT, AT& >;  //!< Composite data type of the dense quaternion expression.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this QuatSlice instance.
   using This = QuatSlice<AT,CRAs...>;

   using BaseType      = DenseTensor<This>;            //!< Base type of this QuatSlice instance.
   using ViewedType    = AT;                           //!< The type viewed by this QuatSlice instance.
   using ResultType    = QuatSliceTrait_t<AT,CRAs...>; //!< Result type for expression template evaluations.
   //using OppositeType  = OppositeType_t<ResultType>;   //!< Result type with opposite storage order for expression template evaluations.
   //using TransposeType = TransposeType_t<ResultType>;  //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<AT>;            //!< Type of the quatslice elements.
   using SIMDType      = SIMDTrait_t<ElementType>;     //!< SIMD type of the quatslice elements.
   using ReturnType    = ReturnType_t<AT>;             //!< Return type for expression template evaluations
   using CompositeType = const QuatSlice&;             //!< Data type for composite expression templates.

   //! Reference to a constant quatslice value.
   using ConstReference = ConstReference_t<AT>;

   //! Reference to a non-constant quatslice value.
   using Reference = If_t< IsConst_v<AT>, ConstReference, Reference_t<AT> >;

   //! Pointer to a constant quatslice value.
   using ConstPointer = ConstPointer_t<AT>;

   //! Pointer to a non-constant quatslice value.
   using Pointer = If_t< IsConst_v<AT> || !HasMutableDataAccess_v<AT>, ConstPointer, Pointer_t<AT> >;

   //! Iterator over constant elements.
   using ConstIterator = ConstIterator_t<AT>;

   //! Iterator over non-constant elements.
   using Iterator = If_t< IsConst_v<AT>, ConstIterator, Iterator_t<AT> >;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = AT::simdEnabled;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = AT::smpAssignable;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RRAs >
   explicit inline QuatSlice( AT& quaternion, RRAs... args );

   QuatSlice( const QuatSlice& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~QuatSlice() = default;
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
   inline QuatSlice& operator=( const ElementType& rhs );
   inline QuatSlice& operator=( initializer_list< initializer_list< initializer_list<ElementType> > > list );
   inline QuatSlice& operator=( const QuatSlice& rhs );

   template< typename AT2 > inline QuatSlice& operator= ( const Tensor<AT2>& rhs );
   template< typename AT2 > inline QuatSlice& operator+=( const Tensor<AT2>& rhs );
   template< typename AT2 > inline QuatSlice& operator-=( const Tensor<AT2>& rhs );
   template< typename AT2 > inline QuatSlice& operator%=( const Tensor<AT2>& rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   using DataType::quat;

   inline AT&       operand() noexcept;
   inline const AT& operand() const noexcept;

   inline size_t  pages() const noexcept;
   inline size_t  rows() const noexcept;
   inline size_t  columns() const noexcept;
   inline size_t  spacing() const noexcept;
   inline size_t  capacity() const noexcept;
   inline size_t  capacity( size_t i, size_t k ) const noexcept;
   inline size_t  nonZeros() const;
   inline size_t  nonZeros( size_t i, size_t k ) const;
   inline void    reset();
   inline void    reset( size_t i, size_t k );
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   template< typename Other > inline QuatSlice& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename AT2 >
   static constexpr bool VectorizedAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && AT2::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<AT2> > );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename AT2 >
   static constexpr bool VectorizedAddAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && AT2::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<AT2> > &&
        HasSIMDAdd_v< ElementType, ElementType_t<AT2> > );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename AT2 >
   static constexpr bool VectorizedSubAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && AT2::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<AT2> > &&
        HasSIMDSub_v< ElementType, ElementType_t<AT2> > );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename AT2 >
   static constexpr bool VectorizedSchurAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && AT2::simdEnabled &&
        IsSIMDCombinable_v< ElementType, ElementType_t<AT2> > &&
        HasSIMDMult_v< ElementType, ElementType_t<AT2> > );
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

   template< typename AT2, size_t... CRAs2 >
   inline bool canAlias( const QuatSlice<AT2,CRAs2...>* alias ) const noexcept;

   template< typename Other >
   inline bool isAliased( const Other* alias ) const noexcept;

   template< typename AT2, size_t... CRAs2 >
   inline bool isAliased( const QuatSlice<AT2,CRAs2...>* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t k, size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t k, size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t k, size_t i, size_t j ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept;

   template< typename AT2 >
   inline auto assign( const DenseTensor<AT2>& rhs ) -> DisableIf_t< VectorizedAssign_v<AT2> >;

   template< typename AT2 >
   inline auto assign( const DenseTensor<AT2>& rhs ) -> EnableIf_t< VectorizedAssign_v<AT2> >;

   template< typename AT2 >
   inline auto addAssign( const DenseTensor<AT2>& rhs ) -> DisableIf_t< VectorizedAddAssign_v<AT2> >;

   template< typename AT2 >
   inline auto addAssign( const DenseTensor<AT2>& rhs ) -> EnableIf_t< VectorizedAddAssign_v<AT2> >;

   template< typename AT2 >
   inline auto subAssign( const DenseTensor<AT2>& rhs ) -> DisableIf_t< VectorizedSubAssign_v<AT2> >;

   template< typename AT2 >
   inline auto subAssign( const DenseTensor<AT2>& rhs ) -> EnableIf_t< VectorizedSubAssign_v<AT2> >;

   template< typename AT2 >
   inline auto schurAssign( const DenseTensor<AT2>& rhs ) -> DisableIf_t< VectorizedSchurAssign_v<AT2> >;

   template< typename AT2 >
   inline auto schurAssign( const DenseTensor<AT2>& rhs ) -> EnableIf_t< VectorizedSchurAssign_v<AT2> >;

   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand quaternion_;  //!< The quaternion containing the quatslice.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename AT2, size_t... CRAs2 > friend class QuatSlice;
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE    ( AT );
   //BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE ( AT );
   //BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE   ( AT );
//    BLAZE_CONSTRAINT_MUST_NOT_BE_SUBTENSOR_TYPE   ( AT );
   //BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE     ( AT );
   //BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE   ( AT );
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
/*!\brief Constructor for quatslices on quatslice-major dense quaternions.
//
// \param quaternion The quaternion containing the quatslice.
// \param args The runtime quatslice arguments.
// \exception std::invalid_argument Invalid quatslice access index.
//
// By default, the provided quatslice arguments are checked at runtime. In case the quatslice is not properly
// specified (i.e. if the specified index is greater than the number of quats of the given quaternion)
// a \a std::invalid_argument exception is thrown. The checks can be skipped by providing the
// optional \a blaze::unchecked argument.
*/
template< typename AT         // Type of the dense quaternion
        , size_t... CRAs >    // Compile time quatslice arguments
template< typename... RRAs >  // Runtime quatslice arguments
inline QuatSlice<AT,CRAs...>::QuatSlice( AT& quaternion, RRAs... args )
   : DataType( args... )  // Base class initialization
   , quaternion_ ( quaternion  )  // The quaternion containing the quatslice
{
   if( !Contains_v< TypeList<RRAs...>, Unchecked > ) {
      if( quaternion_.quats() <= quat() ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid quatslice access index" );
      }
   }
   else {
      BLAZE_USER_ASSERT( quat() < quaternion_.quats(), "Invalid quatslice access index" );
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
/*!\brief Subscript operator for the direct access to the quatslice elements.
//
// \param index Access index. The index must be smaller than the number of quaternion columns.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::Reference
   QuatSlice<AT,CRAs...>::operator()( size_t k, size_t i, size_t j )
{
   BLAZE_USER_ASSERT( k < pages()  , "Invalid page access index" );
   BLAZE_USER_ASSERT( i < rows(),    "Invalid row access index" );
   BLAZE_USER_ASSERT( j < columns(), "Invalid columns access index" );

   return quaternion_(quat(), k, i, j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subscript operator for the direct access to the quatslice elements.
//
// \param index Access index. The index must be smaller than the number of quaternion columns.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::ConstReference
   QuatSlice<AT,CRAs...>::operator()( size_t k, size_t i, size_t j ) const
{
   BLAZE_USER_ASSERT( k < pages()  , "Invalid page access index" );
   BLAZE_USER_ASSERT( i < rows(),    "Invalid row access index" );
   BLAZE_USER_ASSERT( j < columns(), "Invalid columns access index" );

   return const_cast<const AT&>( quaternion_ )(quat(), k, i, j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the quatslice elements.
//
// \param index Access index. The index must be smaller than the number of quaternion columns.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid quatslice access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::Reference
   QuatSlice<AT,CRAs...>::at( size_t k, size_t i, size_t j )
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
   return (*this)(k, i, j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the quatslice elements.
//
// \param index Access index. The index must be smaller than the number of quaternion columns.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid quatslice access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::ConstReference
   QuatSlice<AT,CRAs...>::at( size_t k, size_t i, size_t j ) const
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
   return (*this)(k, i, j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the quatslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense quatslice. Note that in case
// of a column-major quaternion you can NOT assume that the quatslice elements lie adjacent to each other!
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::Pointer
   QuatSlice<AT,CRAs...>::data() noexcept
{
   return quaternion_.data( 0, 0, quat() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the quatslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense quatslice. Note that in case
// of a column-major quaternion you can NOT assume that the quatslice elements lie adjacent to each other!
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::ConstPointer
   QuatSlice<AT,CRAs...>::data() const noexcept
{
   return quaternion_.data( 0, 0, quat() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the quatslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense quatslice. Note that in case
// of a column-major quaternion you can NOT assume that the quatslice elements lie adjacent to each other!
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::Pointer
   QuatSlice<AT,CRAs...>::data( size_t i, size_t k  ) noexcept
{
   return quaternion_.data( k, i, quat() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the quatslice elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dense quatslice. Note that in case
// of a column-major quaternion you can NOT assume that the quatslice elements lie adjacent to each other!
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::ConstPointer
   QuatSlice<AT,CRAs...>::data( size_t k, size_t i ) const noexcept
{
   return quaternion_.data( k, i, quat() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the quatslice.
//
// \param i The row/column index.
// \return Iterator to the first element of the given row on this quatslice.
//
// This function returns an iterator to the first element of the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::Iterator
   QuatSlice<AT,CRAs...>::begin( size_t k, size_t i )
{
   return quaternion_.begin( i, quat(), k );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the quatslice.
//
// \param i The row/column index.
// \return Iterator to the first element of the quatslice.
//
// This function returns an iterator to the first element of the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::ConstIterator
   QuatSlice<AT,CRAs...>::begin( size_t k, size_t i ) const
{
   return quaternion_.cbegin( i, quat(), k );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of the quatslice.
//
// \param i The row/column index.
// \return Iterator to the first element of the quatslice.
//
// This function returns an iterator to the first element of the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::ConstIterator
   QuatSlice<AT,CRAs...>::cbegin( size_t k, size_t i ) const
{
   return quaternion_.cbegin( i, quat(), k );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the quatslice.
//
// \param i The row/column index.
// \return Iterator just past the last element of the quatslice.
//
// This function returns an iterator just past the last element of the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::Iterator
   QuatSlice<AT,CRAs...>::end( size_t k, size_t i )
{
   return quaternion_.end( i, quat(), k );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the quatslice.
//
// \param i The row/column index.
// \return Iterator just past the last element of the quatslice.
//
// This function returns an iterator just past the last element of the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::ConstIterator
   QuatSlice<AT,CRAs...>::end( size_t k, size_t i ) const
{
   return quaternion_.cend( i, quat(), k );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of the quatslice.
//
// \param i The row/column index.
// \return Iterator just past the last element of the quatslice.
//
// This function returns an iterator just past the last element of the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline typename QuatSlice<AT,CRAs...>::ConstIterator
   QuatSlice<AT,CRAs...>::cend( size_t k, size_t i ) const
{
   return quaternion_.cend( i, quat(), k );
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
/*!\brief Homogeneous assignment to all quatslice elements.
//
// \param rhs Scalar value to be assigned to all quatslice elements.
// \return Reference to the assigned quatslice.
//
// This function homogeneously assigns the given value to all elements of the quatslice. Note that in
// case the underlying dense quaternion is a lower/upper quaternion only lower/upper and diagonal elements
// of the underlying quaternion are modified.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline QuatSlice<AT,CRAs...>&
   QuatSlice<AT,CRAs...>::operator=( const ElementType& rhs )
{
   decltype(auto) left( derestrict( quaternion_ ) );

   for( size_t k=0; k<pages(); ++k )
   {
       for( size_t i=0; i<rows(); ++i )
       {
          for (size_t j = 0; j < columns(); ++j) {
             if (!IsRestricted_v<AT> || IsTriangular_v<AT> || trySet(quaternion_, std::array<size_t, 3>{ i, j, k }, rhs))
                left(quat(),k,i,j) = rhs;
          }
       }
   }
   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all quatslice elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to quatslice.
// \exception std::invalid_argument Invalid assignment to restricted quaternion.
//
// This assignment operator offers the option to directly assign to all elements of the dense
// quatslice by means of an initializer list. The quatslice elements are assigned the values from the given
// initializer list. Missing values are reset to their default state. Note that in case the size
// of the initializer list exceeds the size of the quatslice, a \a std::invalid_argument exception is
// thrown. Also, if the underlying quaternion \a AT is restricted and the assignment would violate
// an invariant of the quaternion, a \a std::invalid_argument exception is thrown.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline QuatSlice<AT,CRAs...>&
   QuatSlice<AT,CRAs...>::operator=( initializer_list< initializer_list< initializer_list<ElementType> > > list )
{
   if (list.size() != pages() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to quatslice" );
   }

   //if( IsRestricted_v<AT> ) {
   //   const InitializerTensor<ElementType> tmp( list, rows(), columns()  );
   //   if (!tryAssign(quaternion_, tmp, std::array<size_t, 4>{quat(),0,0,0})) {
   //      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted quaternion" );
   //   }
   //}


   decltype(auto) left( derestrict( *this ) );

   size_t k( 0UL );
   for( const auto& colList : list ) {
      size_t i( 0UL );
      for( const auto& rowList : colList ) {
         std::fill( std::copy( rowList.begin(), rowList.end(), left.begin( k,i) ), left.end(k,i), ElementType() );
         ++i;
      }
      ++k;
   }

   BLAZE_INTERNAL_ASSERT( isIntact( quaternion_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for QuatSlice.
//
// \param rhs Dense quatslice to be copied.
// \return Reference to the assigned quatslice.
// \exception std::invalid_argument QuatSlice sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted quaternion.
//
// In case the current sizes of the two quatslices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying quaternion \a AT is a lower or upper triangular quaternion and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline QuatSlice<AT,CRAs...>&
   QuatSlice<AT,CRAs...>::operator=( const QuatSlice& rhs )
{
   if( &rhs == this ) return *this;

   if( rows() != rhs.rows() || columns() != rhs.columns() || pages() != rhs.pages() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "QuatSlice sizes do not match" );
   }

   //if (!tryAssign(quaternion_, rhs, std::array<size_t, 4>{quat(), 0UL, 0UL, 0UL})) {
   //   BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted quaternion" );
   //}

   decltype(auto) left( derestrict( *this ) );

   if( IsExpression_v<AT> && rhs.canAlias( &quaternion_ ) ) {
      const ResultType tmp( rhs );
      smpAssign( left, tmp );
   }
   else {
      smpAssign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( quaternion_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different matrices.
//
// \param rhs Tensor to be assigned.
// \return Reference to the assigned quatslice.
// \exception std::invalid_argument Tensor sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted quaternion.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying quaternion \a AT is a lower or upper triangular quaternion and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename AT2 >    // Type of the right-hand side matrix
inline QuatSlice<AT,CRAs...>&
   QuatSlice<AT,CRAs...>::operator=( const Tensor<AT2>& rhs )
{
   //BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<AT2> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<AT2> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() || pages() != (~rhs).pages() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<AT>, CompositeType_t<AT2>, const AT2& >;
   Right right( ~rhs );

   //if( !tryAssign( quaternion_, right, std::array<size_t, 4>{quat(), 0UL, 0UL, 0UL} ) ) {
   //   BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted quaternion" );
   //}

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &quaternion_ ) ) {
      const ResultType_t<AT2> tmp( right );
      smpAssign( left, tmp );
   }
   else {
      //if( IsSparseTensor_v<AT2> )
      //   reset();
      smpAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( quaternion_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ \vec{a}+=\vec{b} \f$).
//
// \param rhs The right-hand side matrix to be added to the dense quatslice.
// \return Reference to the assigned quatslice.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted quaternion.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying quaternion \a AT is a lower or upper triangular quaternion and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename AT2 >    // Type of the right-hand side matrix
inline QuatSlice<AT,CRAs...>&
   QuatSlice<AT,CRAs...>::operator+=( const Tensor<AT2>& rhs )
{
   //BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<AT2> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<AT2> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() || pages() != (~rhs).pages() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<AT>, CompositeType_t<AT2>, const AT2& >;
   Right right( ~rhs );

   //if( !tryAddAssign( quaternion_, right, quat(), 0UL, 0UL, 0UL ) ) {
   //   BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted quaternion" );
   //}

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &quaternion_ ) ) {
      const ResultType_t<AT2> tmp( right );
      smpAddAssign( left, tmp );
   }
   else {
      smpAddAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( quaternion_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ \vec{a}-=\vec{b} \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the dense quatslice.
// \return Reference to the assigned quatslice.
// \exception std::invalid_argument Vector sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted quaternion.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying quaternion \a AT is a lower or upper triangular quaternion and the
// assignment would violate its lower or upper property, respectively, a \a std::invalid_argument
// exception is thrown.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename AT2 >    // Type of the right-hand side matrix
inline QuatSlice<AT,CRAs...>&
   QuatSlice<AT,CRAs...>::operator-=( const Tensor<AT2>& rhs )
{
   //BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<AT2> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<AT2> );

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() || pages() != (~rhs).pages() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   using Right = If_t< IsRestricted_v<AT>, CompositeType_t<AT2>, const AT2& >;
   Right right( ~rhs );

   //if( !trySubAssign( quaternion_, right, quat(), 0UL, 0UL, 0UL ) ) {
   //   BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted quaternion" );
   //}

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( &quaternion_ ) ) {
      const ResultType_t<AT2> tmp( right );
      smpSubAssign( left, tmp );
   }
   else {
      smpSubAssign( left, right );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( quaternion_ ), "Invariant violation detected" );

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
// \return Reference to the assigned quatslice.
// \exception std::invalid_argument Invalid matrix size for cross product.
// \exception std::invalid_argument Invalid assignment to restricted quaternion.
//
// In case the current size of any of the two matrices is not equal to 3, a \a std::invalid_argument
// exception is thrown.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename AT2 >    // Type of the right-hand side matrix
inline QuatSlice<AT,CRAs...>&
   QuatSlice<AT,CRAs...>::operator%=( const Tensor<AT2>& rhs )
{
   using blaze::assign;

   //BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType_t<AT2> );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION  ( ResultType_t<AT2> );

   using SchurType = SchurTrait_t< ResultType, ResultType_t<AT2> >;

   if( rows() != (~rhs).rows() || columns() != (~rhs).columns() || pages() != (~rhs).pages()  ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   //if( !trySchurAssign( quaternion_, (~rhs), quat(), 0UL, 0UL, 0UL ) ) {
   //   BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted quaternion" );
   //}

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<AT> && (~rhs).canAlias( &quaternion_ ) ) {
      const SchurType tmp( *this % (~rhs) );
      smpSchurAssign( left, tmp );
   }
   else {
      smpSchurAssign( left, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( quaternion_ ), "Invariant violation detected" );

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
/*!\brief Returns the quaternion containing the quatslice.
//
// \return The quaternion containing the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline AT& QuatSlice<AT,CRAs...>::operand() noexcept
{
   return quaternion_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the quaternion containing the quatslice.
//
// \return The quaternion containing the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline const AT& QuatSlice<AT,CRAs...>::operand() const noexcept
{
   return quaternion_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current size/dimension of the quatslice.
//
// \return The size of the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline size_t QuatSlice<AT,CRAs...>::pages() const noexcept
{
   return quaternion_.pages();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current size/dimension of the quatslice.
//
// \return The size of the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline size_t QuatSlice<AT,CRAs...>::rows() const noexcept
{
   return quaternion_.rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current size/dimension of the quatslice.
//
// \return The size of the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline size_t QuatSlice<AT,CRAs...>::columns() const noexcept
{
   return quaternion_.columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the minimum capacity of the quatslice.
//
// \return The minimum capacity of the quatslice.
//
// This function returns the minimum capacity of the quatslice, which corresponds to the current size
// plus padding.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline size_t QuatSlice<AT,CRAs...>::spacing() const noexcept
{
   return quaternion_.spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense quatslice.
//
// \return The maximum capacity of the dense quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline size_t QuatSlice<AT,CRAs...>::capacity() const noexcept
{
   return quaternion_.capacity( quat(), 0UL, 0UL ) * quaternion_.pages() * quaternion_.rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the dense quatslice.
//
// \return The maximum capacity of the dense quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline size_t QuatSlice<AT,CRAs...>::capacity( size_t i, size_t k  ) const noexcept
{
   return quaternion_.capacity( quat(), k, i ) * quaternion_.pages() * quaternion_.rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the quatslice.
//
// \return The number of non-zero elements in the quatslice.
//
// Note that the number of non-zero elements is always less than or equal to the current number
// of columns of the quaternion containing the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline size_t QuatSlice<AT,CRAs...>::nonZeros() const
{
   size_t count ( 0 );
   for (size_t k = 0; k < pages(); ++k) {
      for (size_t i = 0; i < rows(); ++i) {
         count += quaternion_.nonZeros( i, quat(), k);
      }
   }
   return count;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the quatslice.
//
// \return The number of non-zero elements in the quatslice.
//
// Note that the number of non-zero elements is always less than or equal to the current number
// of columns of the quaternion containing the quatslice.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline size_t QuatSlice<AT,CRAs...>::nonZeros( size_t i, size_t k ) const
{
   return quaternion_.nonZeros( quat(), k, i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline void QuatSlice<AT,CRAs...>::reset()
{
   for (size_t k = 0; k < pages(); ++k) {
      for (size_t i = 0; i < rows(); ++i) {
         quaternion_.reset( quat(), k, i );
      }
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
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline void QuatSlice<AT,CRAs...>::reset( size_t i, size_t k )
{
   quaternion_.reset( quat(), k, i );
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
/*!\brief Scaling of the quatslice by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the quatslice scaling.
// \return Reference to the dense quatslice.
//
// This function scales the quatslice by applying the given scalar value \a scalar to each element
// of the quatslice. For built-in and \c complex data types it has the same effect as using the
// multiplication assignment operator. Note that the function cannot be used to scale a quatslice
// on a lower or upper unitriangular quaternion. The attempt to scale such a quatslice results in a
// compile time error!
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename Other >  // Data type of the scalar value
inline QuatSlice<AT,CRAs...>&
   QuatSlice<AT,CRAs...>::scale( const Other& scalar )
{
   for (size_t k = 0; k < pages(); ++k) {
      for (size_t i = 0; i < rows(); ++i) {
         for (size_t j = 0; j < columns(); ++j) {
            quaternion_(quat(), k, i, j) *= scalar;
         }
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
/*!\brief Returns whether the dense quatslice can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense quatslice, \a false if not.
//
// This function returns whether the given address can alias with the dense quatslice. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename Other >  // Data type of the foreign expression
inline bool QuatSlice<AT,CRAs...>::canAlias( const Other* alias ) const noexcept
{
   return quaternion_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense quatslice can alias with the given dense quatslice \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense quatslice, \a false if not.
//
// This function returns whether the given address can alias with the dense quatslice. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename AT        // Type of the dense quaternion
        , size_t... CRAs >   // Compile time quatslice arguments
template< typename AT2       // Data type of the foreign dense quatslice
        , size_t... CRAs2 >  // Compile time quatslice arguments of the foreign dense quatslice
inline bool
   QuatSlice<AT,CRAs...>::canAlias( const QuatSlice<AT2,CRAs2...>* alias ) const noexcept
{
   return quaternion_.isAliased( &alias->quaternion_ ) && ( quat() == alias->quat() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense quatslice is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense quatslice, \a false if not.
//
// This function returns whether the given address is aliased with the dense quatslice. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename Other >  // Data type of the foreign expression
inline bool QuatSlice<AT,CRAs...>::isAliased( const Other* alias ) const noexcept
{
   return quaternion_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense quatslice is aliased with the given dense quatslice \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this dense quatslice, \a false if not.
//
// This function returns whether the given address is aliased with the dense quatslice. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename AT        // Type of the dense quaternion
        , size_t... CRAs >   // Compile time quatslice arguments
template< typename AT2       // Data type of the foreign dense quatslice
        , size_t... CRAs2 >  // Compile time quatslice arguments of the foreign dense quatslice
inline bool
   QuatSlice<AT,CRAs...>::isAliased( const QuatSlice<AT2,CRAs2...>* alias ) const noexcept
{
   return quaternion_.isAliased( &alias->quaternion_ ) && ( quat() == alias->quat() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense quatslice is properly aligned in memory.
//
// \return \a true in case the dense quatslice is aligned, \a false if not.
//
// This function returns whether the dense quatslice is guaranteed to be properly aligned in memory,
// i.e. whether the beginning and the end of the dense quatslice are guaranteed to conform to the
// alignment restrictions of the element type \a Type.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline bool QuatSlice<AT,CRAs...>::isAligned() const noexcept
{
   return quaternion_.isAligned();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the dense quatslice can be used in SMP assignments.
//
// \return \a true in case the dense quatslice can be used in SMP assignments, \a false if not.
//
// This function returns whether the dense quatslice can be used in SMP assignments. In contrast to
// the \a smpAssignable member enumeration, which is based solely on compile time information,
// this function additionally provides runtime information (as for instance the current size
// of the dense quatslice).
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
inline bool QuatSlice<AT,CRAs...>::canSMPAssign() const noexcept
{
   return ( pages() * rows() * columns() > SMP_DMATASSIGN_THRESHOLD );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Load of a SIMD element of the dense quatslice.
//
// \param index Access index. The index must be smaller than the number of quaternion columns.
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense quatslice. The index
// must be smaller than the number of quaternion columns. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
BLAZE_ALWAYS_INLINE typename QuatSlice<AT,CRAs...>::SIMDType
   QuatSlice<AT,CRAs...>::load( size_t k, size_t i, size_t j ) const noexcept
{
   return quaternion_.load( quat(), k, i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the dense quatslice.
//
// \param index Access index. The index must be smaller than the number of quaternion columns.
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense quatslice.
// The index must be smaller than the number of quaternion columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
BLAZE_ALWAYS_INLINE typename QuatSlice<AT,CRAs...>::SIMDType
   QuatSlice<AT,CRAs...>::loada( size_t k, size_t i, size_t j ) const noexcept
{
   return quaternion_.loada( quat(), k, i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the dense quatslice.
//
// \param index Access index. The index must be smaller than the number of quaternion columns.
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense quatslice.
// The index must be smaller than the number of quaternion columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
BLAZE_ALWAYS_INLINE typename QuatSlice<AT,CRAs...>::SIMDType
   QuatSlice<AT,CRAs...>::loadu( size_t k, size_t i, size_t j ) const noexcept
{
   return quaternion_.loadu( quat(), k, i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Store of a SIMD element of the dense quatslice.
//
// \param index Access index. The index must be smaller than the number of quaternion columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store a specific SIMD element of the dense quatslice. The index
// must be smaller than the number of quaternion columns. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
BLAZE_ALWAYS_INLINE void
   QuatSlice<AT,CRAs...>::store( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   quaternion_.store( quat(), k, i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned store of a SIMD element of the dense quatslice.
//
// \param index Access index. The index must be smaller than the number of quaternion columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store a specific SIMD element of the dense quatslice. The
// index must be smaller than the number of quaternion columns. This function must \b NOT be
// called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
BLAZE_ALWAYS_INLINE void
   QuatSlice<AT,CRAs...>::storea( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   quaternion_.storea( quat(), k, i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unligned store of a SIMD element of the dense quatslice.
//
// \param index Access index. The index must be smaller than the number of quaternion columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store a specific SIMD element of the dense quatslice.
// The index must be smaller than the number of quaternion columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
BLAZE_ALWAYS_INLINE void
   QuatSlice<AT,CRAs...>::storeu( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   quaternion_.storeu( quat(), k, i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned, non-temporal store of a SIMD element of the dense quatslice.
//
// \param index Access index. The index must be smaller than the number of quaternion columns.
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store a specific SIMD element of the dense
// quatslice. The index must be smaller than the number of quaternion columns. This function must \b NOT
// be called explicitly! It is used internally for the performance optimized evaluation of
// expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
BLAZE_ALWAYS_INLINE void
   QuatSlice<AT,CRAs...>::stream( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   quaternion_.stream( quat(), k, i, j, value );
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
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename AT2 >     // Type of the right-hand side dense matrix
inline auto QuatSlice<AT,CRAs...>::assign( const DenseTensor<AT2>& rhs )
   -> DisableIf_t< VectorizedAssign_v<AT2> >
{
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages(),   "Invalid number of pages" );
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows(),    "Invalid number of rows" );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos((~rhs).columns() & size_t(-2));
   for (size_t k = 0UL; k < pages(); ++k)
   {
      for (size_t i = 0UL; i < (~rhs).rows(); ++i)
      {
         for (size_t j = 0UL; j < jpos; j += 2UL) {
            quaternion_(quat(), k, i, j) = (~rhs)(k, i, j);
            quaternion_(quat(), k, i, j + 1UL) = (~rhs)(k, i, j + 1UL);
         }
         if (jpos < (~rhs).columns())
            quaternion_(quat(), k, i, jpos) = (~rhs)(k, i, jpos);
      }
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
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename AT2 >     // Type of the right-hand side dense matrix
inline auto QuatSlice<AT,CRAs...>::assign( const DenseTensor<AT2>& rhs )
   -> EnableIf_t< VectorizedAssign_v<AT2> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages"  );
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"   );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns");

   constexpr bool remainder( !IsPadded_v<AT> || !IsPadded_v<AT2> );

   const size_t cols( columns() );
   const size_t jpos( ( remainder )?( cols & size_t(-SIMDSIZE) ):( cols ) );

   for( size_t k=0UL; k<pages(); ++k )
   {
      for (size_t i = 0UL; i < rows(); ++i)
      {
         size_t j(0UL);
         Iterator left(begin(i, k));
         ConstIterator_t<AT2> right((~rhs).begin(i, k));

         if (useStreaming && cols > (cacheSize / (sizeof(ElementType) * 3UL)) && !(~rhs).isAliased(&quaternion_))
         {
            for (; j < jpos; j += SIMDSIZE) {
               left.stream(right.load()); left += SIMDSIZE; right += SIMDSIZE;
            }
            for (; remainder && j < cols; ++j) {
               *left = *right; ++left; ++right;
            }
         }
         else
         {
            for (; (j + SIMDSIZE * 3UL) < jpos; j += SIMDSIZE * 4UL) {
               left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
               left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
               left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
               left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
            }
            for (; j < jpos; j += SIMDSIZE) {
               left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
            }
            for (; remainder && j < cols; ++j) {
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
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename AT2 >     // Type of the right-hand side dense matrix
inline auto QuatSlice<AT,CRAs...>::addAssign( const DenseTensor<AT2>& rhs )
   -> DisableIf_t< VectorizedAddAssign_v<AT2> >
{
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages" );
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   for (size_t k = 0UL; k < pages(); ++k) {
      for (size_t i = 0UL; i < (~rhs).rows(); ++i) {
         const size_t jpos((~rhs).columns() & size_t(-2));
         for (size_t j = 0UL; j < jpos; j += 2UL) {
            quaternion_(quat(), k, i, j)       += (~rhs)(k, i, j);
            quaternion_(quat(), k, i, j + 1UL) += (~rhs)(k, i, j + 1UL);
         }
         if (jpos < (~rhs).columns())
            quaternion_(quat(), k, i, jpos) += (~rhs)(k, i, jpos);
      }
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
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename AT2 >     // Type of the right-hand side dense matrix
inline auto QuatSlice<AT,CRAs...>::addAssign( const DenseTensor<AT2>& rhs )
   -> EnableIf_t< VectorizedAddAssign_v<AT2> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages"  );
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"   );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns");

   constexpr bool remainder( !IsPadded_v<AT> || !IsPadded_v<AT2> );

   const size_t cols( columns() );
   const size_t jpos( ( remainder )?( cols & size_t(-SIMDSIZE) ):( cols ) );

   for( size_t k=0UL; k<pages(); ++k )
   {
      for (size_t i = 0UL; i < rows(); ++i)
      {
         size_t j(0UL);
         Iterator left(begin(i, k));
         ConstIterator_t<AT2> right((~rhs).begin(i, k));

         for (; (j + SIMDSIZE * 3UL) < jpos; j += SIMDSIZE * 4UL) {
            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
         }
         for (; j < jpos; j += SIMDSIZE) {
            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
         }
         for (; remainder && j < cols; ++j) {
            *left += *right; ++left; ++right;
         }
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
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename AT2 >     // Type of the right-hand side dense matrix
inline auto QuatSlice<AT,CRAs...>::subAssign( const DenseTensor<AT2>& rhs )
   -> DisableIf_t< VectorizedSubAssign_v<AT2> >
{
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages" );
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   for (size_t k = 0UL; k < pages(); ++k) {
      for (size_t i = 0UL; i < (~rhs).rows(); ++i) {
         const size_t jpos((~rhs).columns() & size_t(-2));
         for (size_t j = 0UL; j < jpos; j += 2UL) {
            quaternion_(quat(), k, i, j)       -= (~rhs)(k, i, j);
            quaternion_(quat(), k, i, j + 1UL) -= (~rhs)(k, i, j + 1UL);
         }
         if (jpos < (~rhs).columns())
            quaternion_(quat(), k, i, jpos) -= (~rhs)(k, i, jpos);
      }
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
template< typename AT       // Type of the dense quaternion
        , size_t... CRAs >  // Compile time quatslice arguments
template< typename AT2 >     // Type of the right-hand side dense matrix
inline auto QuatSlice<AT,CRAs...>::subAssign( const DenseTensor<AT2>& rhs )
   -> EnableIf_t< VectorizedSubAssign_v<AT2> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages"  );
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"   );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns");

   constexpr bool remainder( !IsPadded_v<AT> || !IsPadded_v<AT2> );

   const size_t cols( columns() );
   const size_t jpos( ( remainder )?( cols & size_t(-SIMDSIZE) ):( cols ) );

   for( size_t k=0UL; k<pages(); ++k )
   {
      for (size_t i = 0UL; i < rows(); ++i)
      {
         size_t j(0UL);
         Iterator left(begin(i, k));
         ConstIterator_t<AT2> right((~rhs).begin(i, k));

         for (; (j + SIMDSIZE * 3UL) < jpos; j += SIMDSIZE * 4UL) {
            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
         }
         for (; j < jpos; j += SIMDSIZE) {
            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
         }
         for (; remainder && j < cols; ++j) {
            *left -= *right; ++left; ++right;
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
template< typename AT       // Type of the quaternion
        , size_t... CSAs >  // Compile time quatslice arguments
template< typename AT2 >    // Type of the right-hand side dense matrix
inline auto QuatSlice<AT,CSAs...>::schurAssign( const DenseTensor<AT2>& rhs )
   -> DisableIf_t< VectorizedSchurAssign_v<AT2> >
{
   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages" );
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns" );

   for (size_t k = 0UL; k < pages(); ++k) {
      for (size_t i = 0UL; i < (~rhs).rows(); ++i) {
         const size_t jpos((~rhs).columns() & size_t(-2));
         for (size_t j = 0UL; j < jpos; j += 2UL) {
            quaternion_(quat(), k, i, j)       *= (~rhs)(k, i, j);
            quaternion_(quat(), k, i, j + 1UL) *= (~rhs)(k, i, j + 1UL);
         }
         if (jpos < (~rhs).columns())
            quaternion_(quat(), k, i, jpos) *= (~rhs)(k, i, jpos);
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
template< typename AT       // Type of the quaternion
        , size_t... CSAs >  // Compile time quatslice arguments
template< typename AT2 >    // Type of the right-hand side dense matrix
inline auto QuatSlice<AT,CSAs...>::schurAssign( const DenseTensor<AT2>& rhs )
   -> EnableIf_t< VectorizedSchurAssign_v<AT2> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( ElementType );

   BLAZE_INTERNAL_ASSERT( pages()   == (~rhs).pages()  , "Invalid number of pages"  );
   BLAZE_INTERNAL_ASSERT( rows()    == (~rhs).rows()   , "Invalid number of rows"   );
   BLAZE_INTERNAL_ASSERT( columns() == (~rhs).columns(), "Invalid number of columns");

   constexpr bool remainder( !IsPadded_v<AT> || !IsPadded_v<AT2> );

   const size_t cols( columns() );
   const size_t jpos( ( remainder )?( cols & size_t(-SIMDSIZE) ):( cols ) );

   for( size_t k=0UL; k<pages(); ++k )
   {
      for (size_t i = 0UL; i < rows(); ++i)
      {
         size_t j(0UL);
         Iterator left(begin(i, k));
         ConstIterator_t<AT2> right((~rhs).begin(i, k));

         for (; (j + SIMDSIZE * 3UL) < jpos; j += SIMDSIZE * 4UL) {
            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
         }
         for (; j < jpos; j += SIMDSIZE) {
            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
         }
         for (; remainder && j < cols; ++j) {
            *left *= *right; ++left; ++right;
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


} // namespace blaze

#endif
