//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/DynamicArray.h
//  \brief Header file for the implementation of a dynamic LxOxMxN array
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_DYNAMIC_ARRAY_H_
#define _BLAZE_TENSOR_MATH_DENSE_DYNAMIC_ARRAY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <array>

#include <blaze/math/IntegerSequence.h>
#include <blaze/math/dense/DynamicMatrix.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/ExpandTrait.h>
#include <blaze/math/traits/MapTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsContiguous.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsShrinkable.h>
#include <blaze/system/Optimizations.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/StaticAssert.h>

#include <blaze_tensor/math/Array.h>
#include <blaze_tensor/math/CustomArray.h>
#include <blaze_tensor/math/Forward.h>
#include <blaze_tensor/math/InitFromValue.h>
#include <blaze_tensor/math/InitializerList.h>
#include <blaze_tensor/math/SMP.h>
#include <blaze_tensor/math/dense/DynamicTensor.h>
#include <blaze_tensor/math/dense/Transposition.h>
#include <blaze_tensor/math/expressions/DenseArray.h>
//#include <blaze_tensor/math/traits/ArraySliceTrait.h>
#include <blaze_tensor/math/traits/QuatSliceTrait.h>
#include <blaze_tensor/math/typetraits/IsNdArray.h>
#include <blaze_tensor/math/typetraits/IsDenseArray.h>
#include <blaze_tensor/math/typetraits/IsRowMajorArray.h>
#include <blaze_tensor/util/ArrayForEach.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup dynamic_array DynamicArray
// \ingroup dense_array
*/
/*!\brief Efficient implementation of a dynamic \f$ M \times N \f$ array.
// \ingroup dynamic_array
//
// The DynamicArray class template is the representation of an arbitrary sized array with
// \f$ M \times N \f$ dynamically allocated elements of arbitrary type. The type of the elements
// and the storage order of the array can be specified via the two template parameters:

   \code
   template< typename Type >
   class DynamicArray;
   \endcode

//  - Type: specifies the type of the array elements. DynamicArray can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//
// The use of DynamicArray is very natural and intuitive. All operations (addition, subtraction,
// multiplication, scaling, ...) can be performed on all possible combinations of matrices with
// fitting element types. The following example an impression of the use of DynamicArray:

   \code
   using blaze::DynamicArray;
   using blaze::CompressedArray;

   DynamicArray<4, double> A( 2, 3, 4, 5 );              // Default constructed, non-initialized, 2x3x4x5 array
   A(0,0,0,0) = 1.0; A(0,0,0,1) = 2.0; A(0,0,0,2) = 3.0; // Initialization of the first row
   A(0,0,1,0) = 4.0; A(0,0,1,1) = 5.0; A(0,0,1,2) = 6.0; // Initialization of the second row

   DynamicArray<2, float> B( 2, 3 );                   // Default constructed column-major single precision 2x3 array
   B(0,0) = 1.0; B(0,1) = 3.0; B(0,2) = 5.0;    // Initialization of the first row
   B(1,0) = 2.0; B(1,1) = 4.0; B(1,2) = 6.0;    // Initialization of the second row

   CompressedArray<float> C( 2, 3 );        // Empty sparse single precision array
   DynamicArray<2, float>    D( 3, 2, 4.0F );  // Directly, homogeneously initialized single precision 3x2 array

   DynamicArray<2, double>    E( A );          // Creation of a new array as a copy of A
   DynamicArray<2, double> F;                  // Creation of a default column-major array

   E = A + B;     // Array addition and assignment to a array
   F = A - C;     // Array subtraction and assignment to a column-major array
   F = A * D;     // Array multiplication between two matrices of different element types

   A *= 2.0;      // In-place scaling of array A
   E  = 2.0 * B;  // Scaling of array B
   F  = D * 2.0;  // Scaling of array D

   E += A - B;    // Addition assignment
   E -= A + C;    // Subtraction assignment
   F *= A * D;    // Multiplication assignment
   \endcode
*/
template< size_t N, typename Type >                   // Data type of the array
class DynamicArray
   : public DenseArray< DynamicArray<N, Type> >
{
 public:
   //**Type definitions****************************************************************************
   using This          = DynamicArray<N, Type>;     //!< Type of this DynamicArray instance.
   using BaseType      = DenseArray<This>;          //!< Base type of this DynamicArray instance.
   using ResultType    = This;                      //!< Result type for expression template evaluations.
   using OppositeType  = DynamicArray<N, Type>;     //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = DynamicArray<N, Type>;     //!< Transpose type for expression template evaluations.
   using ElementType   = Type;                      //!< Type of the array elements.
   using SIMDType      = SIMDTrait_t<ElementType>;  //!< SIMD type of the array elements.
   using ReturnType    = const Type&;               //!< Return type for expression template evaluations.
   using CompositeType = const This&;               //!< Data type for composite expression templates.

   using Reference      = Type&;        //!< Reference to a non-constant array value.
   using ConstReference = const Type&;  //!< Reference to a constant array value.
   using Pointer        = Type*;        //!< Pointer to a non-constant array value.
   using ConstPointer   = const Type*;  //!< Pointer to a constant array value.

   using Iterator      = DenseIterator<Type,usePadding>;        //!< Iterator over non-constant elements.
   using ConstIterator = DenseIterator<const Type,usePadding>;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a DynamicArray with different data/element type.
   */
   template< typename NewType >  // Data type of the other array
   struct Rebind {
      using Other = DynamicArray<N, NewType>;  //!< The type of the other DynamicArray.
   };
   //**********************************************************************************************

   //**Resize struct definition********************************************************************
   /*!\brief Resize mechanism to obtain a DynamicArray with different fixed dimensions.
   */
//    template< size_t... NewDims >  // Dimensions of the other array
//    struct Resize {
//       BLAZE_STATIC_ASSERT_MSG(N == sizeof...(NewDims), "incompatible dimensionality of other array");
//       using Other = DynamicArray<N, Type>;  //!< The type of the other DynamicArray.
//    };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SIMD optimization.
   /*! The \a simdEnabled compilation flag indicates whether expressions the array is involved
       in can be optimized via SIMD operations. In case the element type of the array is a
       vectorizable data type, the \a simdEnabled compilation flag is set to \a true, otherwise
       it is set to \a false. */
   static constexpr bool simdEnabled = IsVectorizable_v<Type>;

   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the array can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   static constexpr bool smpAssignable = !IsSMPAssignable_v<Type>;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline DynamicArray() noexcept;
   template< typename... Dims, typename = EnableIf_t< sizeof...(Dims) == N - 1 > >
   explicit inline DynamicArray( size_t dim0, Dims... dims );
   explicit inline DynamicArray( std::array< size_t, N> const& dims );
   template< typename... Dims >
   explicit inline DynamicArray( InitFromValue, const Type& init, Dims... dims );
   explicit inline DynamicArray( nested_initializer_list< N, Type > list );

   template< typename Other, typename... Dims, typename = EnableIf_t< sizeof...(Dims) == N > >
   explicit inline DynamicArray( const Other* array, Dims... dims );

                                     inline DynamicArray( const DynamicArray& m );
                                     inline DynamicArray( DynamicArray&& m ) noexcept;
   template< typename MT >           inline DynamicArray( const Array<MT>& m );

   template< typename MT, bool TF, size_t M = N, typename = EnableIf_t< M == 1 > >
   inline DynamicArray( const Vector<MT, TF>& m );
   template< typename MT, bool SO, size_t M = N, typename = EnableIf_t< M == 2 > >
   inline DynamicArray( const Matrix<MT, SO>& m );
   template< typename MT, size_t M = N, typename = EnableIf_t< M == 3 > >
   inline DynamicArray( const Tensor<MT>& m );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~DynamicArray();
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   template< typename... Dims >
   inline Reference      operator()( Dims... dims ) noexcept;
   template< typename... Dims >
   inline ConstReference operator()( Dims... dims ) const noexcept;
   inline Reference      operator()( std::array< size_t, N > const& indices ) noexcept;
   inline ConstReference operator()( std::array< size_t, N > const& indices ) const noexcept;
   template< typename... Dims >
   inline Reference      at( Dims... dims );
   template< typename... Dims >
   inline ConstReference at( Dims... dims ) const;
   inline Reference      at( std::array< size_t, N > const& indices );
   inline ConstReference at( std::array< size_t, N > const& indices ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   template< typename... Dims >
   inline Pointer        data  ( size_t i, Dims... subdims ) noexcept;
   template< typename... Dims >
   inline ConstPointer   data  ( size_t i, Dims... subdims ) const noexcept;
   template< typename... Dims >
   inline Iterator       begin ( size_t i, Dims... subdims ) noexcept;
   template< typename... Dims >
   inline ConstIterator  begin ( size_t i, Dims... subdims ) const noexcept;
   template< typename... Dims >
   inline ConstIterator  cbegin( size_t i, Dims... subdims ) const noexcept;
   template< typename... Dims >
   inline Iterator       end   ( size_t i, Dims... subdims ) noexcept;
   template< typename... Dims >
   inline ConstIterator  end   ( size_t i, Dims... subdims ) const noexcept;
   template< typename... Dims >
   inline ConstIterator  cend  ( size_t i, Dims... subdims ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline DynamicArray& operator=( const Type& rhs );
   inline DynamicArray& operator=( nested_initializer_list< N, Type > list );

   inline DynamicArray& operator=( const DynamicArray& rhs );
   inline DynamicArray& operator=( DynamicArray&& rhs ) noexcept;

   template< typename MT > inline DynamicArray& operator= ( const Array<MT>& rhs );
   template< typename MT > inline DynamicArray& operator+=( const Array<MT>& rhs );
   template< typename MT > inline DynamicArray& operator-=( const Array<MT>& rhs );
   template< typename MT > inline DynamicArray& operator%=( const Array<MT>& rhs );

   template< typename MT, bool TF, size_t M = N, typename = EnableIf_t< M == 1 > >
   inline DynamicArray& operator= ( const Vector<MT, TF>& m );
   template< typename MT, bool TF, size_t M = N, typename = EnableIf_t< M == 1 > >
   inline DynamicArray& operator+=( const Vector<MT, TF>& m );
   template< typename MT, bool TF, size_t M = N, typename = EnableIf_t< M == 1 > >
   inline DynamicArray& operator-=( const Vector<MT, TF>& m );
   template< typename MT, bool TF, size_t M = N, typename = EnableIf_t< M == 1 > >
   inline DynamicArray& operator%=( const Vector<MT, TF>& m );

   template< typename MT, bool SO, size_t M = N, typename = EnableIf_t< M == 2 > >
   inline DynamicArray& operator= ( const Matrix<MT, SO>& m );
   template< typename MT, bool SO, size_t M = N, typename = EnableIf_t< M == 2 > >
   inline DynamicArray& operator+=( const Matrix<MT, SO>& m );
   template< typename MT, bool SO, size_t M = N, typename = EnableIf_t< M == 2 > >
   inline DynamicArray& operator-=( const Matrix<MT, SO>& m );
   template< typename MT, bool SO, size_t M = N, typename = EnableIf_t< M == 2 > >
   inline DynamicArray& operator%=( const Matrix<MT, SO>& m );

   template< typename MT, size_t M = N, typename = EnableIf_t< M == 3 > >
   inline DynamicArray& operator= ( const Tensor<MT>& m );
   template< typename MT, size_t M = N, typename = EnableIf_t< M == 3 > >
   inline DynamicArray& operator+=( const Tensor<MT>& m );
   template< typename MT, size_t M = N, typename = EnableIf_t< M == 3 > >
   inline DynamicArray& operator-=( const Tensor<MT>& m );
   template< typename MT, size_t M = N, typename = EnableIf_t< M == 3 > >
   inline DynamicArray& operator%=( const Tensor<MT>& m );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline static constexpr size_t num_dimensions() noexcept { return N; }
   inline constexpr std::array< size_t, N > const& dimensions() const noexcept;
   inline size_t quats() const noexcept;
   inline size_t pages() const noexcept;
   inline size_t rows() const noexcept;
   inline size_t columns() const noexcept;
   template < size_t Dim >
   inline size_t dimension() const noexcept;
   inline size_t spacing() const noexcept;
   inline size_t capacity() const noexcept;
   template< typename... Dims >
   inline size_t capacity( size_t i, Dims... subdims ) const noexcept;
   inline size_t nonZeros() const;
   template< typename... Dims >
   inline size_t nonZeros( size_t i, Dims... subdims ) const;
   inline void   reset();
   template< typename... Dims >
   inline void   reset( size_t i, Dims... subdims );
   inline void   clear();
          void   resize( std::array< size_t, N > const& dims, bool preserve = true );
   inline void   extend( std::array< size_t, N > const& dims, bool preserve = true );
   inline void   reserve( size_t elements );
   inline void   shrinkToFit();
   inline void   swap( DynamicArray& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   inline DynamicArray& transpose();
   inline DynamicArray& ctranspose();
   template < typename T >
   inline DynamicArray& transpose( const T* indices, size_t n );
   template < typename T >
   inline DynamicArray& ctranspose( const T* indices, size_t n );

   template< typename Other > inline DynamicArray& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool VectorizedAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && MT::simdEnabled &&
        IsSIMDCombinable_v< Type, ElementType_t<MT> > &&
        IsRowMajorArray_v< MT >);
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool VectorizedAddAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && MT::simdEnabled &&
        IsSIMDCombinable_v< Type, ElementType_t<MT> > &&
        HasSIMDAdd_v< Type, ElementType_t<MT> > &&
        IsRowMajorArray_v< MT >);
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool VectorizedSubAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && MT::simdEnabled &&
        IsSIMDCombinable_v< Type, ElementType_t<MT> > &&
        HasSIMDSub_v< Type, ElementType_t<MT> > &&
        IsRowMajorArray_v< MT >);
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool VectorizedSchurAssign_v =
      ( useOptimizedKernels &&
        simdEnabled && MT::simdEnabled &&
        IsSIMDCombinable_v< Type, ElementType_t<MT> > &&
        HasSIMDMult_v< Type, ElementType_t<MT> > &&
        IsRowMajorArray_v< MT >);
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   //! The number of elements packed within a single SIMD element.
   static constexpr size_t SIMDSIZE = SIMDTrait<ElementType>::size;
   //**********************************************************************************************

 public:
   //**Debugging functions*************************************************************************
   /*!\name Debugging functions */
   //@{
   inline bool isIntact() const noexcept;
   //@}
   //**********************************************************************************************

   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other > inline bool canAlias ( const Other* alias ) const noexcept;
   template< typename Other > inline bool isAliased( const Other* alias ) const noexcept;

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

   template< typename MT >
   inline auto assign( const DenseArray<MT>& rhs ) /*-> DisableIf_t< VectorizedAssign_v<MT> >*/;

   //template< typename MT >
   //inline auto assign( const DenseArray<MT>& rhs ) -> EnableIf_t< VectorizedAssign_v<MT> >;

   template< typename MT >
   inline auto addAssign( const DenseArray<MT>& rhs ) /*-> DisableIf_t< VectorizedAddAssign_v<MT> >*/;

   //template< typename MT >
   //inline auto addAssign( const DenseArray<MT>& rhs ) -> EnableIf_t< VectorizedAddAssign_v<MT> >;

   template< typename MT >
   inline auto subAssign( const DenseArray<MT>& rhs ) /*-> DisableIf_t< VectorizedSubAssign_v<MT> >*/;

   //template< typename MT >
   //inline auto subAssign( const DenseArray<MT>& rhs ) -> EnableIf_t< VectorizedSubAssign_v<MT> >;

   template< typename MT >
   inline auto schurAssign( const DenseArray<MT>& rhs ) /*-> DisableIf_t< VectorizedSchurAssign_v<MT> >*/;

   //template< typename MT >
   //inline auto schurAssign( const DenseArray<MT>& rhs ) -> EnableIf_t< VectorizedSchurAssign_v<MT> >;
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   template< typename... Dims >
   inline static std::array< size_t, N > initDimensions( Dims... dims ) noexcept;
   inline static size_t addPadding( size_t value ) noexcept;
   inline size_t calcCapacity() const noexcept;
   template< typename... Dims >
   inline size_t index( Dims... dims ) const noexcept;
   inline size_t index( std::array< size_t, N > const& indices ) const noexcept;
   template< typename... Dims >
   inline size_t row_index( size_t i, Dims... subdims ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::array< size_t, N > dims_;  //!< The current dimensions of the array (dims[1]...dims_[N]) .
   size_t nn_;                     //!< The alignment adjusted number of columns.
   size_t capacity_;               //!< The maximum capacity of the array.
   Type* BLAZE_RESTRICT v_;        //!< The dynamically allocated array elements.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE  ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST         ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE      ( Type );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The default constructor for DynamicArray.
*/
template< size_t N         // The dimensionality of the array
        , typename Type > // Data type of the array
inline DynamicArray<N, Type>::DynamicArray() noexcept
   : dims_    (  )         // The current dimensions of the array
   , nn_      ( 0UL )      // The length of a padded row
   , capacity_( 0UL )      // The maximum capacity of the array
   , v_       ( nullptr )  // The array elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a array of size \f$ m \times n \f$. No element initialization is performed!
//
// \param m The number of rows of the array.
// \param n The number of columns of the array.
//
// \note This constructor is only responsible to allocate the required dynamic memory. No
// element initialization is performed!
*/
template< size_t N         // The dimensionality of the array
        , typename Type > // Data type of the array
template< typename... Dims, typename Enable >
inline DynamicArray< N, Type >::DynamicArray( size_t dim0, Dims... dims )
   : dims_    ( initDimensions( dim0, dims... ) )   // The current dimensions of the array
   , nn_      ( addPadding( dims_[0] ) )  // The length of a padded row
   , capacity_( calcCapacity() )          // The maximum capacity of the array
   , v_( allocate< Type >( capacity_ ) )  // The array elements
{
   BLAZE_STATIC_ASSERT( N - 1 == sizeof...( dims ) );

   if( IsVectorizable_v<Type> ) {
      ArrayForEachPadded( dims_, nn_, [&]( size_t i ) { v_[i] = Type(); } );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a homogeneous initialization of all \f$ m \times n \f$ array elements.
//
// \param m The number of rows of the array.
// \param n The number of columns of the array.
// \param init The initial value of the array elements.
//
// All array elements are initialized with the specified value.
*/
template< size_t N         // The dimensionality of the array
        , typename Type > // Data type of the array
template< typename... Dims >
inline DynamicArray<N, Type>::DynamicArray( InitFromValue, const Type& init, Dims...dims )
   : DynamicArray( dims... )
{
   BLAZE_STATIC_ASSERT( N == sizeof...( dims ) );

   ArrayForEach( dims_, nn_, [&]( size_t i ) { v_[i] = init; } );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List initialization of all array elements.
//
// \param list The initializer list.
//
// This constructor provides the option to explicitly initialize the elements of the array by
// means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::DynamicArray<3, int> A{ { { 1, 2, 3 },
                                  { 4, 5 },
                                  { 7, 8, 9 } },
                                { { 1, 2, 3 },
                                  { 4, 5 },
                                  { 7, 8, 9 } } };
   \endcode

// The array is sized according to the size of the initializer list and all its elements are
// initialized by the values of the given initializer list. Missing values are initialized as
// default (as e.g. the value 6 in the example).
*/
template< size_t N         // The dimensionality of the array
        , typename Type > // Data type of the array
inline DynamicArray<N, Type>::DynamicArray( nested_initializer_list< N, Type > list )
   : DynamicArray( list.dimensions() )
{
   list.transfer_data( *this );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all array elements.
//
// \param m The number of rows of the array.
// \param n The number of columns of the array.
// \param array Dynamic array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the array with
// a dynamic array:

   \code
   using blaze::rowMajor;

   int* array = new int[20];
   // ... Initialization of the dynamic array
   blaze::DynamicArray<3, int> v( array, 6UL, 4UL, 5UL );
   delete[] array;
   \endcode

// The array is sized according to the given size of the array and initialized with the values
// from the given array. Note that it is expected that the given \a array has at least \a m by
// \a n elements. Providing an array with less elements results in undefined behavior!
*/
template< size_t N         // The dimensionality of the array
        , typename Type > // Data type of the array
template< typename Other, typename... Dims, typename Enable >  // Data type of the initialization array
inline DynamicArray<N, Type>::DynamicArray( const Other* array, Dims... dims )
   : DynamicArray( dims... )
{
   BLAZE_STATIC_ASSERT( N == sizeof...( dims ) );

   if( IsNothrowMoveAssignable_v< Type > ) {
      ArrayForEach2( dims_, nn_, [&]( size_t i, size_t j ) {
         v_[j] = std::move( array[i] );
      } );
   }
   else {
      ArrayForEach2( dims_, nn_, [&]( size_t i, size_t j ) {
         v_[j] = array[i];
      } );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for DynamicArray.
//
// \param m Array to be copied.
//
// The copy constructor is explicitly defined due to the required dynamic memory management
// and in order to enable/facilitate NRV optimization.
*/
template< size_t N         // The dimensionality of the array
        , typename Type > // Data type of the array
inline DynamicArray<N, Type>::DynamicArray( const DynamicArray& m )
   : dims_    ( m.dims_ )                 // The current dimensions of the array
   , nn_      ( m.nn_ )                   // The length of a padded row
   , capacity_( m.capacity_ )             // The maximum capacity of the array
   , v_( allocate< Type >( capacity_ ) )  // The array elements
{
   smpAssign( *this, m );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The move constructor for DynamicArray.
//
// \param m The array to be move into this instance.
*/
template< size_t N         // The dimensionality of the array
        , typename Type > // Data type of the array
inline DynamicArray<N, Type>::DynamicArray( DynamicArray&& m ) noexcept
   : dims_    ( std::move(m.dims_) )   // The current dimensions of the array
   , nn_      ( m.nn_ )                // The length of a padded row
   , capacity_( m.capacity_ )          // The maximum capacity of the array
   , v_       ( m.v_        )          // The array elements
{
   m.nn_       = 0UL;
   m.capacity_ = 0UL;
   m.v_        = nullptr;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different arrays.
//
// \param m Array to be copied.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT >    // Type of the foreign array
inline DynamicArray<N, Type>::DynamicArray( const Array<MT>& rhs )
   : DynamicArray( (~rhs).dimensions() )
{
   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAssign( *this, tmp );
   }
   else {
      smpAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different arrays.
//
// \param m Array to be copied.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, bool TF, size_t M, typename Enable >
inline DynamicArray<N, Type>::DynamicArray( const Vector<MT, TF>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<1, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAssign( *this, custom_array( tmp.data(), tmp.size(), tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpAssign( *this, custom_array( tmp.data(), tmp.size(), tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Conversion constructor from different arrays.
//
// \param m Array to be copied.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, bool SO, size_t M, typename Enable >
inline DynamicArray<N, Type>::DynamicArray( const Matrix<MT,SO>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<2, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAssign( *this,
         custom_array( tmp.data(), tmp.rows(), tmp.columns(), tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpAssign( *this,
         custom_array( tmp.data(), tmp.rows(), tmp.columns(), tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Conversion constructor from different arrays.
//
// \param m Array to be copied.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, size_t M, typename Enable >
inline DynamicArray<N, Type>::DynamicArray( const Tensor<MT>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<2, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAssign( *this,
         custom_array( tmp.data(), tmp.pages(), tmp.rows(), tmp.columns(),
            tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpAssign( *this,
         custom_array( tmp.data(), tmp.pages(), tmp.rows(), tmp.columns(),
            tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************

//*************************************************************************************************
/*!\brief Constructor from array bounds.
//
// \param m Array to be copied.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline DynamicArray<N, Type>::DynamicArray( const std::array<size_t, N>& dims )
   : dims_    ( dims )         // The current dimensions of the array
   , nn_      ( addPadding( dims_[0] ) )     // The length of a padded row
   , capacity_( calcCapacity() )             // The maximum capacity of the array
   , v_( allocate< Type >( capacity_ ) )     // The array elements
{
   if( IsVectorizable_v<Type> ) {
      ArrayForEachPadded( dims_, nn_, [&]( size_t i ) { v_[i] = Type(); } );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************



//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The destructor for DynamicArray.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline DynamicArray<N, Type>::~DynamicArray()
{
   deallocate( v_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief ND-access to the array elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline typename DynamicArray<N, Type>::Reference
   DynamicArray<N, Type>::operator()( Dims... dims ) noexcept
{
   BLAZE_STATIC_ASSERT( N == sizeof...( dims ) );

#if defined(BLAZE_USER_ASSERTION)
   size_t indices[] = { size_t(dims)... };

   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_USER_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
   MAYBE_UNUSED( indices );
#endif

   return v_[index( dims... )];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief ND-access to the array elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline typename DynamicArray<N, Type>::ConstReference
   DynamicArray<N, Type>::operator()( Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N == sizeof...( dims ) );

#if defined(BLAZE_USER_ASSERTION)
   size_t indices[] = { size_t(dims)... };

   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_USER_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
   MAYBE_UNUSED( indices );
#endif

   return v_[index( dims... )];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief ND-access to the array elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline typename DynamicArray<N, Type>::Reference
   DynamicArray<N, Type>::operator()( std::array< size_t, N > const& indices ) noexcept
{
#if defined(BLAZE_USER_ASSERTION)
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_USER_ASSERT( indices[i] < dim, "Invalid array access index" );
   } );
#endif

   return v_[index( indices )];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief ND-access to the array elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline typename DynamicArray<N, Type>::ConstReference
   DynamicArray<N, Type>::operator()( std::array< size_t, N > const& indices ) const noexcept
{
#if defined(BLAZE_USER_ASSERTION)
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_USER_ASSERT( indices[i] < dim, "Invalid array access index" );
   } );
#endif

   return v_[index( indices )];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the array elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid array access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline typename DynamicArray<N, Type>::Reference
   DynamicArray<N, Type>::at( Dims... dims )
{
   BLAZE_STATIC_ASSERT( N == sizeof...( dims ) );

   size_t indices[] = { size_t(dims)... };

   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      if( indices[N - i - 1] >= dim ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid array access index" );
      }
   } );

   return ( *this )( dims... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the array elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid array access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline typename DynamicArray<N, Type>::ConstReference
   DynamicArray<N, Type>::at( Dims... dims ) const
{
   BLAZE_STATIC_ASSERT( N == sizeof...( dims ) );

   size_t indices[] = { size_t(dims)... };

   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      if( indices[N - i - 1] >= dim ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid array access index" );
      }
   } );

   return ( *this )( dims... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the array elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid array access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline typename DynamicArray<N, Type>::Reference
   DynamicArray<N, Type>::at( std::array< size_t, N > const& indices )
{
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      if( indices[i] >= dim ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid array access index" );
      }
   } );

   return ( *this )( indices );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the array elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid array access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline typename DynamicArray<N, Type>::ConstReference
   DynamicArray<N, Type>::at( std::array< size_t, N > const& indices ) const
{
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      if( indices[i] >= dim ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid array access index" );
      }
   } );

   return ( *this )( indices );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the array elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic array. Note that you
// can NOT assume that all array elements lie adjacent to each other! The dynamic array may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline typename DynamicArray<N, Type>::Pointer
   DynamicArray<N, Type>::data() noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the array elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic array. Note that you
// can NOT assume that all array elements lie adjacent to each other! The dynamic array may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline typename DynamicArray<N, Type>::ConstPointer
   DynamicArray<N, Type>::data() const noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the array elements of row/column \a i.
//
// \param i The row/column index.
// \param j The page index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline typename DynamicArray<N, Type>::Pointer
   DynamicArray<N, Type>::data( size_t i, Dims... dims ) noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return v_ + row_index( i, dims... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the array elements of row/column \a i.
//
// \param i The row/column index.
// \param j The page index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline typename DynamicArray<N, Type>::ConstPointer
   DynamicArray<N, Type>::data( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return v_ + row_index( i, dims... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \param j The page index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline typename DynamicArray<N, Type>::Iterator
   DynamicArray<N, Type>::begin( size_t i, Dims... dims ) noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return Iterator( v_ + row_index( i, dims... ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \param j The page index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline typename DynamicArray<N, Type>::ConstIterator
   DynamicArray<N, Type>::begin( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return ConstIterator( v_ + row_index( i, dims... ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \param j The page index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline typename DynamicArray<N, Type>::ConstIterator
   DynamicArray<N, Type>::cbegin( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return ConstIterator( v_ + row_index( i, dims... ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \param j The page index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline typename DynamicArray<N, Type>::Iterator
   DynamicArray<N, Type>::end( size_t i, Dims... dims ) noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return Iterator( v_ + row_index( i, dims... ) + dims_[0] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \param j The page index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline typename DynamicArray<N, Type>::ConstIterator
   DynamicArray<N, Type>::end( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return ConstIterator( v_ + row_index( i, dims... ) + dims_[0] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \param j The page index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline typename DynamicArray<N, Type>::ConstIterator
   DynamicArray<N, Type>::cend( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return ConstIterator( v_ + row_index( i, dims... ) + dims_[0] );
}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Homogeneous assignment to all array elements.
//
// \param rhs Scalar value to be assigned to all array elements.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator=(const Type& rhs)
{
   ArrayForEach( dims_, nn_, [&]( size_t i ) { v_[i] = rhs; } );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List assignment to all array elements.
//
// \param list The initializer list.
//
// This assignment operator offers the option to directly assign to all elements of the array
// by means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::DynamicArray<3, int> A;
   A = { { { 1, 2, 3 },
           { 4, 5 },
           { 7, 8, 9 } } };
   \endcode

// The array is resized according to the given initializer list and all its elements are
// assigned the values from the given initializer list. Missing values are initialized as
// default (as e.g. the value 6 in the example).
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline DynamicArray<N, Type>&
   DynamicArray<N, Type>::operator=( nested_initializer_list< N, Type > list )
{
   resize( list.dimensions(), false );

   list.transfer_data( *this );

   if( IsVectorizable_v<Type> ) {
      ArrayForEachPadded( dims_, nn_, [&]( size_t i ) { v_[i] = Type(); } );
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy assignment operator for DynamicArray.
//
// \param rhs Array to be copied.
// \return Reference to the assigned array.
//
// The array is resized according to the given \f$ M \times N \f$ array and initialized as a
// copy of this array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator=( const DynamicArray& rhs )
{
   if( &rhs == this ) return *this;

   resize( rhs.dimensions(), false );

   smpAssign( *this, ~rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Move assignment operator for DynamicArray.
//
// \param rhs The array to be moved into this instance.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator=( DynamicArray&& rhs ) noexcept
{
   deallocate( v_ );

   dims_     = std::move( rhs.dims_ );
   nn_       = rhs.nn_;
   capacity_ = rhs.capacity_;
   v_        = rhs.v_;

   rhs.nn_       = 0UL;
   rhs.capacity_ = 0UL;
   rhs.v_        = nullptr;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different matrices.
//
// \param rhs Array to be copied.
// \return Reference to the assigned array.
//
// The array is resized according to the given \f$ M \times N \f$ array and initialized as a
// copy of this array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT >  // Type of the right-hand side array
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator=( const Array<MT>& rhs )
{
   if( (~rhs).canAlias( this ) ) {
      DynamicArray tmp( ~rhs );
      swap( tmp );
   }
   else {
      resize( (~rhs).dimensions(), false );
      smpAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment operator for the addition of a array (\f$ A+=B \f$).
//
// \param rhs The right-hand side array to be added to the array.
// \return Reference to the array.
// \exception std::invalid_argument Array sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT >   // Type of the right-hand side array
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator+=( const Array<MT>& rhs )
{
   if( (~rhs).dimensions() != dims_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAddAssign( *this, tmp );
   }
   else {
      smpAddAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment operator for the subtraction of a array (\f$ A-=B \f$).
//
// \param rhs The right-hand side array to be subtracted from the array.
// \return Reference to the array.
// \exception std::invalid_argument Array sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT >   // Type of the right-hand side array
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator-=( const Array<MT>& rhs )
{
   if( (~rhs).dimensions() != dims_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpSubAssign( *this, tmp );
   }
   else {
      smpSubAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Schur product assignment operator for the multiplication of a array (\f$ A\circ=B \f$).
//
// \param rhs The right-hand side array for the Schur product.
// \return Reference to the array.
// \exception std::invalid_argument Array sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT >  // Type of the right-hand side array
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator%=( const Array<MT>& rhs )
{
   if( (~rhs).dimensions() != dims_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpSchurAssign( *this, tmp );
   }
   else {
      smpSchurAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Homogeneous assignment to all array elements.
//
// \param rhs Array to be copied.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, bool TF, size_t M, typename Enable >
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator=( const Vector<MT, TF>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<1, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAssign( *this, custom_array( tmp.data(), tmp.size(), tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpAssign( *this, custom_array( tmp.data(), tmp.size(), tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment operator for the addition of a array (\f$ A+=B \f$).
//
// \param rhs The right-hand side array to be added to the array.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, bool TF, size_t M, typename Enable >
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator+=( const Vector<MT, TF>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<1, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAddAssign( *this, custom_array( tmp.data(), tmp.size(), tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpAddAssign( *this, custom_array( tmp.data(), tmp.size(), tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment operator for the subtraction of a array (\f$ A-=B \f$).
//
// \param rhs The right-hand side array to be subtracted from the array.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, bool TF, size_t M, typename Enable >
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator-=( const Vector<MT, TF>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<1, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpSubAssign( *this, custom_array( tmp.data(), tmp.size(), tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpSubAssign( *this, custom_array( tmp.data(), tmp.size(), tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Schur product assignment operator for the multiplication of a array (\f$ A\circ=B \f$).
//
// \param rhs The right-hand side array for the Schur product.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, bool TF, size_t M, typename Enable >
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator%=( const Vector<MT, TF>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<1, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpSchurAssign( *this, custom_array( tmp.data(), tmp.size(), tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpSchurAssign( *this, custom_array( tmp.data(), tmp.size(), tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Homogeneous assignment to all array elements.
//
// \param rhs Array to be copied.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, bool SO, size_t M, typename Enable >
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator=( const Matrix<MT, SO>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<2, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAssign( *this,
         custom_array( tmp.data(), tmp.rows(), tmp.columns(), tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpAssign( *this,
         custom_array( tmp.data(), tmp.rows(), tmp.columns(), tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment operator for the addition of a array (\f$ A+=B \f$).
//
// \param rhs The right-hand side array to be added to the array.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, bool SO, size_t M, typename Enable >
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator+=( const Matrix<MT, SO>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<2, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAddAssign( *this,
         custom_array( tmp.data(), tmp.rows(), tmp.columns(), tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpAddAssign( *this,
         custom_array( tmp.data(), tmp.rows(), tmp.columns(), tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment operator for the subtraction of a array (\f$ A-=B \f$).
//
// \param rhs The right-hand side array to be subtracted from the array.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, bool SO, size_t M, typename Enable >
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator-=( const Matrix<MT, SO>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<2, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpSubAssign( *this,
         custom_array( tmp.data(), tmp.rows(), tmp.columns(), tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpSubAssign( *this,
         custom_array( tmp.data(), tmp.rows(), tmp.columns(), tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Schur product assignment operator for the multiplication of a array (\f$ A\circ=B \f$).
//
// \param rhs The right-hand side array for the Schur product.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, bool SO, size_t M, typename Enable >
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator%=( const Matrix<MT, SO>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<2, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpSchurAssign( *this,
         custom_array( tmp.data(), tmp.rows(), tmp.columns(), tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpSchurAssign( *this,
         custom_array( tmp.data(), tmp.rows(), tmp.columns(), tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Homogeneous assignment to all array elements.
//
// \param rhs Array to be copied.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, size_t M, typename Enable >
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator=( const Tensor<MT>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<3, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAssign( *this,
         custom_array( tmp.data(), tmp.pages(), tmp.rows(), tmp.columns(),
            tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpAssign( *this,
         custom_array( tmp.data(), tmp.pages(), tmp.rows(), tmp.columns(),
            tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment operator for the subtraction of a array (\f$ A+=B \f$).
//
// \param rhs The right-hand side array to be added to the array.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, size_t M, typename Enable >
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator+=( const Tensor<MT>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<3, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAddAssign( *this,
         custom_array( tmp.data(), tmp.pages(), tmp.rows(), tmp.columns(),
            tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpAddAssign( *this,
         custom_array( tmp.data(), tmp.pages(), tmp.rows(), tmp.columns(),
            tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment operator for the subtraction of a array (\f$ A-=B \f$).
//
// \param rhs The right-hand side array to be subtracted from the array.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, size_t M, typename Enable >
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator-=( const Tensor<MT>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<3, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpSubAssign( *this,
         custom_array( tmp.data(), tmp.pages(), tmp.rows(), tmp.columns(),
            tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpSubAssign( *this,
         custom_array( tmp.data(), tmp.pages(), tmp.rows(), tmp.columns(),
            tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Schur product assignment operator for the multiplication of a array (\f$ A\circ=B \f$).
//
// \param rhs The right-hand side array for the Schur product.
// \return Reference to the assigned array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT, size_t M, typename Enable >
inline DynamicArray<N, Type>& DynamicArray<N, Type>::operator%=( const Tensor<MT>& rhs )
{
   using ET = ElementType_t<MT>;
   using custom_array = CustomArray<3, ET, false, true>;

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpSchurAssign( *this,
         custom_array( tmp.data(), tmp.pages(), tmp.rows(), tmp.columns(),
            tmp.spacing() ) );
   }
   else {
      auto const& tmp = ~rhs;
      smpSchurAssign( *this,
         custom_array( tmp.data(), tmp.pages(), tmp.rows(), tmp.columns(),
            tmp.spacing() ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************






//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the current number of elements in the given dimension of the array.
//
// \return The number of cubes of the array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template < size_t Dim >
inline size_t DynamicArray<N, Type>::dimension() const noexcept
{
   BLAZE_STATIC_ASSERT( Dim < N );

   return dims_[Dim];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the spacing between the beginning of two rows/columns.
//
// \return The spacing between the beginning of two rows/columns.
//
// This function returns the spacing between the beginning of two rows/columns, i.e. the
// total number of elements of a row/column. In case the storage order is set to \a rowMajor
// the function returns the spacing between two rows, in case the storage flag is set to
// \a columnMajor the function returns the spacing between two columns.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline size_t DynamicArray<N, Type>::spacing() const noexcept
{
   return nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the array.
//
// \return The capacity of the array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline size_t DynamicArray<N, Type>::capacity() const noexcept
{
   return capacity_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current capacity of the specified row/column.
//
// \param i The index of the row/column.
// \param k The index of the page.
// \return The current capacity of row/column \a i.
//
// This function returns the current capacity of the specified row/column.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline size_t DynamicArray<N, Type>::capacity( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( Dims ) );

#if defined(BLAZE_USER_ASSERTION)
   size_t indices[] = { size_t(dims)..., i, 0 };
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_USER_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
   MAYBE_UNUSED( indices );
#endif

   MAYBE_UNUSED( dims... );

   BLAZE_USER_ASSERT( i < dimension<1>(), "Invalid row access index" );

   return nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of non-zero elements in the array
//
// \return The number of non-zero elements in the dense array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline size_t DynamicArray<N, Type>::nonZeros() const
{
   size_t nonzeros( 0UL );

   ArrayForEach( dims_, nn_, [&]( size_t i ) {
      if( !isDefault( v_[i] ) ) {
         ++nonzeros;
      }
   } );

   return nonzeros;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the specified row.
//
// \param i The index of the row.
// \return The number of non-zero elements of row \a i.
//
// This function returns the current number of non-zero elements in the specified row.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline size_t DynamicArray<N, Type>::nonZeros( size_t i, Dims... dims ) const
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( Dims ) );

#if defined(BLAZE_USER_ASSERTION)
   size_t indices[] = { size_t(dims)..., i, 0 };
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_USER_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
   MAYBE_UNUSED( indices );
#endif

   const size_t jstart = row_index( i, dims... );
   const size_t jend = jstart + dims_[0];
   size_t nonzeros( 0UL );

   for( size_t j = jstart; j < jend; ++j )
      if( !isDefault( v_[j] ) )
         ++nonzeros;

   return nonzeros;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void DynamicArray<N, Type>::reset()
{
   using blaze::clear;
   ArrayForEach( dims_, nn_, [&]( size_t i ) { clear( v_[i] ); } );
}
//*************************************************************************************************


//*************************************************************************************************
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
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline void DynamicArray<N, Type>::reset( size_t i, Dims... dims )
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   using blaze::clear;

   BLAZE_USER_ASSERT( i < dimension<1>(), "Invalid row access index" );

   size_t row_elements = row_index( i, dims... );

   for( size_t j = 0UL; j < dims_[0]; ++j )
      clear( v_[row_elements + j] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the \f$ M \times N \f$ array.
//
// \return void
//
// After the clear() function, the size of the array is 0.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void DynamicArray<N, Type>::clear()
{
   resize( std::array< size_t, N >{}, false );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the array.
//
// \param m The new number of rows of the array.
// \param n The new number of columns of the array.
// \param preserve \a true if the old values of the array should be preserved, \a false if not.
// \return void
//
// This function resizes the array using the given size to \f$ m \times n \f$. During this
// operation, new dynamic memory may be allocated in case the capacity of the array is too
// small. Note that this function may invalidate all existing views (submatrices, rows, columns,
// ...) on the array if it is used to shrink the array. Additionally, the resize operation
// potentially changes all array elements. In order to preserve the old array values, the
// \a preserve flag can be set to \a true. However, new array elements are not initialized!
//
// The following example illustrates the resize operation of a \f$ 2 \times 4 \f$ array to a
// \f$ 4 \times 2 \f$ array. The new, uninitialized elements are marked with \a x:

                              \f[
                              \left(\begin{array}{*{4}{c}}
                              1 & 2 & 3 & 4 \\
                              5 & 6 & 7 & 8 \\
                              \end{array}\right)

                              \Longrightarrow

                              \left(\begin{array}{*{2}{c}}
                              1 & 2 \\
                              5 & 6 \\
                              x & x \\
                              x & x \\
                              \end{array}\right)
                              \f]
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
void DynamicArray<N, Type>::resize( std::array< size_t, N > const& dims, bool preserve )
{
   BLAZE_USER_ASSERT( N == n, "invalid dimensionality specified" );

   using std::swap;
   using blaze::min;

   // return if no change is requested
   if( ArrayDimAllOf(
          dims_, [&]( size_t i, size_t dim ) { return dim == dims[i]; } ) ) {
      return;
   }

   const size_t nn( addPadding( dims[0] ) );

   size_t new_capacity = nn;
   for( size_t i = 1; i < N; ++i ) {
      new_capacity *= dims[i];
   }

   if( preserve )
   {
      Type* BLAZE_RESTRICT v = allocate<Type>( new_capacity );

//       const size_t min_m( min( m, m_ ) );
//       const size_t min_n( min( n, n_ ) );
//       const size_t min_o( min( o, o_ ) );
//       const size_t min_l( min( l, l_ ) );
//
//       for( size_t c = 0UL; c < min_l; ++c ) {
//          for( size_t k = 0UL; k < min_o; ++k ) {
//             for( size_t i = 0UL; i < min_m; ++i ) {
//                transfer( v_ + ( ( k + c * o_ ) * m_ + i ) * nn_,
//                   v_ + ( ( k + c * o_ ) * m_ + i ) * nn_ + min_n,
//                   v + ( ( k + c * o ) * m + i ) * nn );
//             }
//          }
//       }

      swap( v_, v );
      deallocate( v );
      capacity_ = new_capacity;
   }
   else if( new_capacity > capacity_ ) {
      Type* BLAZE_RESTRICT v = allocate<Type>( new_capacity );
      swap( v_, v );
      deallocate( v );
      capacity_ = new_capacity;
   }

   dims_ = dims;
   nn_ = nn;

   if( IsVectorizable_v< Type > ) {
      ArrayForEachPadded( dims_, nn_, [&]( size_t i ) { v_[i] = Type(); } );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Extending the size of the array.
//
// \param m Number of additional rows.
// \param n Number of additional columns.
// \param preserve \a true if the old values of the array should be preserved, \a false if not.
// \return void
//
// This function increases the array size by \a m rows and \a n columns. During this operation,
// new dynamic memory may be allocated in case the capacity of the array is too small. Therefore
// this function potentially changes all array elements. In order to preserve the old array
// values, the \a preserve flag can be set to \a true. However, new array elements are not
// initialized!
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void DynamicArray<N, Type>::extend( std::array< size_t, N > const& dims, bool preserve )
{
   BLAZE_USER_ASSERT( N == n, "invalid dimensionality specified" );

   std::array< size_t, N > newdims;
   ArrayDimForEach(
      dims_, [&]( size_t i, size_t dim ) { newdims[i] = dim + dims[i]; } );
   resize( newdims, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the minimum capacity of the array.
//
// \param elements The new minimum capacity of the dense array.
// \return void
//
// This function increases the capacity of the dense array to at least \a elements elements.
// The current values of the array elements are preserved.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void DynamicArray<N, Type>::reserve( size_t elements )
{
   using std::swap;

   if( elements > capacity_ )
   {
      // Allocating a new array
      Type* BLAZE_RESTRICT tmp = allocate<Type>( elements );

      // Initializing the new array
      transfer( v_, v_ + capacity_, tmp );

      if( IsVectorizable_v<Type> ) {
         for( size_t i = capacity_; i < elements; ++i )
            tmp[i] = Type();
      }

      // Replacing the old array
      swap( tmp, v_ );
      deallocate( tmp );
      capacity_ = elements;
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Requesting the removal of unused capacity.
//
// \return void
//
// This function minimizes the capacity of the array by removing unused capacity. Please note
// that due to padding the capacity might not be reduced exactly to rows() times columns().
// Please also note that in case a reallocation occurs, all iterators (including end() iterators),
// all pointers and references to elements of this array are invalidated.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void DynamicArray<N, Type>::shrinkToFit()
{
   if( calcCapacity() < capacity_ ) {
      DynamicArray( *this ).swap( *this );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two matrices.
//
// \param m The array to be swapped.
// \return void
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void DynamicArray<N, Type>::swap( DynamicArray& m ) noexcept
{
   using std::swap;

   swap( dims_ , m.dims_  );
   swap( capacity_, m.capacity_ );
   swap( v_ , m.v_  );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Add the necessary amount of padding to the given value.
//
// \param value The value to be padded.
// \return The padded value.
//
// This function increments the given \a value by the necessary amount of padding based on the
// vector's data type \a Type.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline size_t DynamicArray<N, Type>::addPadding( size_t value ) noexcept
{
   if( usePadding && IsVectorizable_v<Type> )
      return nextMultiple<size_t>( value, SIMDSIZE );
   return value;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initialize the dimensions array.
//
// \param value The dimensions for this DynamicArray.
// \return The dimensions array.
//
// This function initializes the internal dimensions array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline std::array< size_t, N > DynamicArray<N, Type>::initDimensions( Dims... dims ) noexcept
{
   BLAZE_STATIC_ASSERT( N == sizeof...( dims ) );

   // the last given dimension is always the lowest
   const size_t indices[] = { dims... };

   std::array< size_t, N > result;
   for( size_t i = 0; i != N; ++i ) {
      result[i] = indices[N - i - 1];
   }
   return result;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculate the capacity for the array.
//
// \return The array capacity.
//
// This function calculates the overall needed capacity for the array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline size_t DynamicArray<N, Type>::calcCapacity() const noexcept
{
   size_t capacity = nn_;
   for( size_t i = 1; i < N; ++i ) {
      capacity *= dims_[i];
   }
   return capacity;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculate index of first element in given row.
//
// \param value The index-array for the row access.
// \return The index of the first element in the given row.
//
// This function calculates the overall index of the first of the give row
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline size_t DynamicArray<N, Type>::row_index( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   size_t indices[] = { static_cast<size_t>(dims)..., i, 0UL };

   size_t idx = 0UL;
   for( size_t i = N - 1; i > 1; --i ) {
      BLAZE_USER_ASSERT(indices[N - i - 1] < dims_[i], "Invalid access index" );
      idx = (idx + indices[N - i - 1]) * dims_[i - 1];
   }

   BLAZE_USER_ASSERT(indices[N - 2] < dims_[1], "Invalid access index" );
   BLAZE_USER_ASSERT(indices[N - 1] < dims_[0], "Invalid access index" );

   return (idx + indices[N - 2]) * nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculate element index.
//
// \param value The index-array for the element access.
// \return The element index.
//
// This function calculates the overall element index into the underlying memory from the ND indices
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
inline size_t DynamicArray<N, Type>::index( Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N == sizeof...( dims ) );

   size_t indices[] = { size_t(dims)... };

   size_t idx = 0UL;
   for( size_t i = N - 1; i > 1; --i ) {
      BLAZE_USER_ASSERT(indices[N - i] < dims_[i], "Invalid access index" );
      idx = (idx + indices[N - i - 1]) * dims_[i - 1];
   }

   BLAZE_USER_ASSERT(indices[N - 2] < dims_[1], "Invalid access index" );
   BLAZE_USER_ASSERT(indices[N - 1] < dims_[0], "Invalid access index" );

   return (idx + indices[N - 2]) * nn_ + indices[N - 1];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculate element index.
//
// \param value The index-array for the element access.
// \return The element index.
//
// This function calculates the overall element index into the underlying memory from the ND indices
*/
//       3         3    2     2    1     0     0
// ( ( ( c + 0 ) * o_ + k ) * m_ + i ) * nn_ + j
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline size_t DynamicArray<N, Type>::index( std::array< size_t, N > const& indices ) const noexcept
{
   size_t idx = 0UL;
   for( size_t i = N - 1; i > 1; --i ) {
      BLAZE_USER_ASSERT( indices[i] < dims_[i], "Invalid access index" );
      idx = (idx + indices[i]) * dims_[i - 1];
   }

   BLAZE_USER_ASSERT(indices[1] < dims_[1], "Invalid access index" );
   BLAZE_USER_ASSERT(indices[0] < dims_[0], "Invalid access index" );

   return (idx + indices[1]) * nn_ + indices[0];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculate index of first element in given row.
//
// \param value The index-array for the row access.
// \return The index of the first element in the given row.
//
// This function calculates the overall index of the first of the give row
*/
template< size_t N   // The dimensionality of the array
        , typename Type >   // Data type of the array
inline constexpr std::array< size_t, N > const& DynamicArray< N, Type >::dimensions() const noexcept
{
   return dims_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculate index of first element in given row.
//
// \param value The index-array for the row access.
// \return The index of the first element in the given row.
//
// This function calculates the overall index of the first of the give row
*/
template< size_t N   // The dimensionality of the array
        , typename Type >   // Data type of the array
inline size_t DynamicArray< N, Type >::quats() const noexcept
{
   return dims_[3];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculate index of first element in given row.
//
// \param value The index-array for the row access.
// \return The index of the first element in the given row.
//
// This function calculates the overall index of the first of the give row
*/
template< size_t N   // The dimensionality of the array
        , typename Type >   // Data type of the array
inline size_t DynamicArray< N, Type >::pages() const noexcept
{
   return dims_[2];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculate index of first element in given row.
//
// \param value The index-array for the row access.
// \return The index of the first element in the given row.
//
// This function calculates the overall index of the first of the give row
*/
template< size_t N   // The dimensionality of the array
        , typename Type >   // Data type of the array
inline size_t DynamicArray< N, Type >::rows() const noexcept
{
   return dims_[1];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculate index of first element in given row.
//
// \param value The index-array for the row access.
// \return The index of the first element in the given row.
//
// This function calculates the overall index of the first of the give row
*/
template< size_t N   // The dimensionality of the array
        , typename Type >   // Data type of the array
inline size_t DynamicArray< N, Type >::columns() const noexcept
{
   return dims_[0];
}
//*************************************************************************************************


//=================================================================================================
//
//  NUMERIC FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief In-place transpose of the array.
//
// \return Reference to the transposed array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline DynamicArray<N, Type>& DynamicArray<N, Type>::transpose()
{
//    if( l_ == n_ && o_ == n_ && m_ == n_ )
//    {
//       transposeGeneral( *this );
//    }
//    else
//    {
//       DynamicArray tmp( trans( *this ) );
//       this->swap( tmp );
//    }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the array.
//
// \return Reference to the transposed array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename T >     // Type of the mapping indices
inline DynamicArray<N, Type>& DynamicArray<N, Type>::transpose( const T* indices, size_t n )
{
//    if( l_ == n_ && o_ == n_ && m_ == n_ )
//    {
//       transposeGeneral( *this, indices, n );
//    }
//    else
//    {
//       DynamicArray tmp( trans(*this, indices, n ) );
//       this->swap( tmp );
//    }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the array.
//
// \return Reference to the transposed array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline DynamicArray<N, Type>& DynamicArray<N, Type>::ctranspose()
{
//    constexpr size_t block( BLOCK_SIZE );
//
//    if( o_ == n_ && m_ == n_ )
//    {
//       for( size_t ii=0UL; ii<m_; ii+=block ) {
//          const size_t iend( min( ii+block, m_ ) );
//          for( size_t jj=0UL; jj<ii; jj+=block ) {
//             const size_t jend( min( jj+block, n_ ) );
//             for( size_t i=ii; i<iend; ++i ) {
//                for( size_t j=jj; j<jend; ++j ) {
//                   cswap( v_[i*nn_+j], v_[j*nn_+i] );
//                }
//             }
//          }
//          for( size_t i=ii; i<iend; ++i ) {
//             for( size_t j=ii; j<i; ++j ) {
//                cswap( v_[i*nn_+j], v_[j*nn_+i] );
//             }
//             conjugate( v_[i*nn_+i] );
//          }
//       }
//    }
//    else
//    {
//       DynamicArray tmp( ctrans( *this ) );
//       swap( tmp );
//    }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the array.
//
// \return Reference to the transposed array.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename T >     // Type of the mapping indices
inline DynamicArray<N, Type>& DynamicArray<N, Type>::ctranspose( const T* indices, size_t n )
{
//    constexpr size_t block( BLOCK_SIZE );
//
//    if( o_ == n_ && m_ == n_ )
//    {
//       for( size_t ii=0UL; ii<m_; ii+=block ) {
//          const size_t iend( min( ii+block, m_ ) );
//          for( size_t jj=0UL; jj<ii; jj+=block ) {
//             const size_t jend( min( jj+block, n_ ) );
//             for( size_t i=ii; i<iend; ++i ) {
//                for( size_t j=jj; j<jend; ++j ) {
//                   cswap( v_[i*nn_+j], v_[j*nn_+i] );
//                }
//             }
//          }
//          for( size_t i=ii; i<iend; ++i ) {
//             for( size_t j=ii; j<i; ++j ) {
//                cswap( v_[i*nn_+j], v_[j*nn_+i] );
//             }
//             conjugate( v_[i*nn_+i] );
//          }
//       }
//    }
//    else
//    {
//       DynamicArray tmp( ctrans(*this, indices, n ) );
//       swap( tmp );
//    }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the array by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the array scaling.
// \return Reference to the array.
//
// This function scales the array by applying the given scalar value \a scalar to each element
// of the array. For built-in and \c complex data types it has the same effect as using the
// multiplication assignment operator:

   \code
   blaze::DynamicArray<3, int> A;
   // ... Resizing and initialization
   A *= 4;        // Scaling of the array
   A.scale( 4 );  // Same effect as above
   \endcode
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename Other >  // Data type of the scalar value
inline DynamicArray<N, Type>& DynamicArray<N, Type>::scale( const Other& scalar )
{
   ArrayForEach( dims_, nn_, [&]( size_t i ) { v_[i] *= scalar; } );
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  DEBUGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the invariants of the dynamic array are intact.
//
// \return \a true in case the dynamic array's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the dynamic array are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline bool DynamicArray<N, Type>::isIntact() const noexcept
{
   if( calcCapacity() > capacity_ )
      return false;

   if (IsVectorizable_v<Type>) {
      bool is_intact = true;
      ArrayForEachPadded( dims_, nn_, [&]( size_t i ) {
         if( v_[i] != Type() )
            is_intact = false;
      } );
      return is_intact;
   }

   return true;
}
//*************************************************************************************************




//=================================================================================================
//
//  EXPRESSION TEMPLATE EVALUATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the array can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this array, \a false if not.
//
// This function returns whether the given address can alias with the array. In contrast
// to the isAliased() function this function is allowed to use compile time expressions
// to optimize the evaluation.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename Other >  // Data type of the foreign expression
inline bool DynamicArray<N, Type>::canAlias( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the array is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this array, \a false if not.
//
// This function returns whether the given address is aliased with the array. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions
// to optimize the evaluation.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename Other >  // Data type of the foreign expression
inline bool DynamicArray<N, Type>::isAliased( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the array is properly aligned in memory.
//
// \return \a true in case the array is aligned, \a false if not.
//
// This function returns whether the array is guaranteed to be properly aligned in memory, i.e.
// whether the beginning and the end of each row/column of the array are guaranteed to conform
// to the alignment restrictions of the element type \a Type.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline bool DynamicArray<N, Type>::isAligned() const noexcept
{
   return ( usePadding || dims_[0] % SIMDSIZE == 0UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the array can be used in SMP assignments.
//
// \return \a true in case the array can be used in SMP assignments, \a false if not.
//
// This function returns whether the array can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the array).
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline bool DynamicArray<N, Type>::canSMPAssign() const noexcept
{
   return ( capacity_ >= SMP_DMATASSIGN_THRESHOLD );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Load of a SIMD element of the array.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense array. The row index
// must be smaller than the number of rows and the column index must be smaller then the number
// of columns. Additionally, the column index (in case of a array) or the row index
// (in case of a column-major array) must be a multiple of the number of values inside the
// SIMD element. This function must \b NOT be called explicitly! It is used internally for the
// performance optimized evaluation of expression templates. Calling this function explicitly
// might result in erroneous results and/or in compilation errors.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
BLAZE_ALWAYS_INLINE typename DynamicArray<N, Type>::SIMDType
   DynamicArray<N, Type>::load( Dims... dims ) const noexcept
{
   if( usePadding )
      return loada( dims... );
   else
      return loadu( dims... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned load of a SIMD element of the array.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense array.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a array)
// or the row index (in case of a column-major array) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
BLAZE_ALWAYS_INLINE typename DynamicArray<N, Type>::SIMDType
   DynamicArray<N, Type>::loada( Dims... dims ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

#if defined(BLAZE_INTERNAL_ASSERTION)
   size_t indices[] = { size_t(dims)... };
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
   MAYBE_UNUSED( indices );
#endif
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= nn_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_ + index( dims ) ), "Invalid alignment detected" );

   return loada( v_ + index( dims... ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unaligned load of a SIMD element of the array.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense array.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a array)
// or the row index (in case of a column-major array) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
BLAZE_ALWAYS_INLINE typename DynamicArray<N, Type>::SIMDType
   DynamicArray<N, Type>::loadu( Dims... dims ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

#if defined(BLAZE_INTERNAL_ASSERTION)
   size_t indices[] = { size_t(dims)... };
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
   MAYBE_UNUSED( indices );
#endif
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= nn_, "Invalid column access index" );

   return loadu( v_ + index( dims... ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Store of a SIMD element of the array.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store of a specific SIMD element of the dense array. The row index
// must be smaller than the number of rows and the column index must be smaller than the number
// of columns. Additionally, the column index (in case of a array) or the row index
// (in case of a column-major array) must be a multiple of the number of values inside the
// SIMD element. This function must \b NOT be called explicitly! It is used internally for the
// performance optimized evaluation of expression templates. Calling this function explicitly
// might result in erroneous results and/or in compilation errors.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
BLAZE_ALWAYS_INLINE void
   DynamicArray<N, Type>::store( const SIMDType& value, Dims... dims ) noexcept
{
   if( usePadding )
      storea( value, dims... );
   else
      storeu( value, dims... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned store of a SIMD element of the array.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store of a specific SIMD element of the dense array.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a array)
// or the row index (in case of a column-major array) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
BLAZE_ALWAYS_INLINE void
   DynamicArray<N, Type>::storea( const SIMDType& value, Dims... dims ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

#if defined(BLAZE_INTERNAL_ASSERTION)
   size_t indices[] = { size_t(dims)... };
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
#endif
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= nn_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_ + index( dims... ) ), "Invalid alignment detected" );

   storea( v_ + index( dims... ), value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unaligned store of a SIMD element of the array.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store of a specific SIMD element of the dense array.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a array)
// or the row index (in case of a column-major array) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
BLAZE_ALWAYS_INLINE void
   DynamicArray<N, Type>::storeu( const SIMDType& value, Dims... dims ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

#if defined(BLAZE_INTERNAL_ASSERTION)
   size_t indices[] = { size_t(dims)... };
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
#endif
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= nn_, "Invalid column access index" );

   storeu( v_ + index( dims... ), value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned, non-temporal store of a SIMD element of the array.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store of a specific SIMD element of the
// dense array. The row index must be smaller than the number of rows and the column index
// must be smaller than the number of columns. Additionally, the column index (in case of a
// array) or the row index (in case of a column-major array) must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename... Dims >
BLAZE_ALWAYS_INLINE void
   DynamicArray<N, Type>::stream( const SIMDType& value, Dims... dims ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

#if defined(BLAZE_INTERNAL_ASSERTION)
   size_t indices[] = { size_t(dims)... };
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
#endif
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= nn_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_ + index( dims... ) ), "Invalid alignment detected" );

   stream( v_ + index( dims... ), value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a dense array.
//
// \param rhs The right-hand side dense array to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT >  // Type of the right-hand side dense array
inline auto DynamicArray<N, Type>::assign( const DenseArray<MT>& rhs )
   //-> DisableIf_t< VectorizedAssign_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( dims_ == (~rhs).dimensions()   , "Invalid array access index"    );

   const size_t jpos( dims_[0] & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( dims_[0] - ( dims_[0] % 2UL ) ) == jpos, "Invalid end calculation" );

   ArrayForEachGrouped(
      dims_, nn_, [&]( size_t i, std::array< size_t, N > const& dims ) {
         v_[i] = ( ~rhs )( dims );
      } );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SIMD optimized implementation of the assignment of a dense array.
//
// \param rhs The right-hand side dense array to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
//template< size_t N         // The dimensionality of the array
//        , typename Type >  // Data type of the array
//template< typename MT >  // Type of the right-hand side dense array
//inline auto DynamicArray<N, Type>::assign( const DenseArray<MT>& rhs )
//   -> EnableIf_t< VectorizedAssign_v<MT> >
//{
//   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
//
//   BLAZE_INTERNAL_ASSERT( dims_ == (~rhs).dimensions()   , "Invalid array access index"    );
//
//   constexpr bool remainder( !usePadding || !IsPadded_v<MT> );
//
//   const size_t jpos( ( remainder )?( dims_[0] & size_t(-SIMDSIZE) ):( dims_[0] ) );
//   BLAZE_INTERNAL_ASSERT( !remainder || ( dims_[0] - ( dims_[0] % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

//    if( usePadding && useStreaming &&
//        ( o_*m_*n_ > ( cacheSize / ( sizeof(Type) * 3UL ) ) ) && !(~rhs).isAliased( this ) )
//    {
//       for (size_t k=0UL; k<o_; ++k) {
//          for (size_t i=0UL; i<m_; ++i) {
//             size_t j(0UL);
//             Iterator left(begin(i, k));
//             ConstIterator_t<MT> right((~rhs).begin(i, k));
//
//             for (; j<jpos; j+=SIMDSIZE, left+=SIMDSIZE, right+=SIMDSIZE) {
//                left.stream(right.load());
//             }
//             for (; remainder && j<n_; ++j, ++left, ++right) {
//                *left = *right;
//             }
//          }
//       }
//    }
//    else
//    {
//       for (size_t k=0UL; k<o_; ++k) {
//          for (size_t i=0UL; i<m_; ++i) {
//             size_t j(0UL);
//             Iterator left(begin(i, k));
//             ConstIterator_t<MT> right((~rhs).begin(i, k));
//
//             for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
//                left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
//                left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
//                left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
//                left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
//             }
//             for (; j<jpos; j+=SIMDSIZE) {
//                left.store(right.load()); left+=SIMDSIZE, right+=SIMDSIZE;
//             }
//             for (; remainder && j<n_; ++j) {
//                *left = *right; ++left; ++right;
//             }
//          }
//       }
//    }
//}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a dense array.
//
// \param rhs The right-hand side dense array to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT >  // Type of the right-hand side dense array
inline auto DynamicArray<N, Type>::addAssign( const DenseArray<MT>& rhs )
   //-> DisableIf_t< VectorizedAddAssign_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( dims_ == (~rhs).dimensions()   , "Invalid array access index"    );

   const size_t jpos( dims_[0] & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( dims_[0] - ( dims_[0] % 2UL ) ) == jpos, "Invalid end calculation" );

   ArrayForEachGrouped(
      dims_, nn_, [&]( size_t i, std::array< size_t, N > const& dims ) {
         v_[i] += ( ~rhs )( dims );
      } );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SIMD optimized implementation of the addition assignment of a dense array.
//
// \param rhs The right-hand side dense array to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
//template< size_t N         // The dimensionality of the array
//        , typename Type >  // Data type of the array
//template< typename MT >  // Type of the right-hand side dense array
//inline auto DynamicArray<N, Type>::addAssign( const DenseArray<MT>& rhs )
//   -> EnableIf_t< VectorizedAddAssign_v<MT> >
//{
//   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
//
//   BLAZE_INTERNAL_ASSERT( dims_ == (~rhs).dimensions()   , "Invalid array access index"    );
//
//   constexpr bool remainder( !usePadding || !IsPadded_v<MT> );
//
//   for (size_t k=0UL; k<o_; ++k) {
//      for (size_t i=0UL; i<m_; ++i) {
//         const size_t jbegin(0UL);
//         const size_t jend  (n_);
//         BLAZE_INTERNAL_ASSERT(jbegin <= jend, "Invalid loop indices detected");
//
//         const size_t jpos((remainder)?(jend & size_t(-SIMDSIZE)):(jend));
//         BLAZE_INTERNAL_ASSERT(!remainder || (jend - (jend % (SIMDSIZE))) == jpos, "Invalid end calculation");
//
//         size_t j(jbegin);
//         Iterator left(begin(i, k) + jbegin);
//         ConstIterator_t<MT> right((~rhs).begin(i, k) + jbegin);
//
//         for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
//            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
//            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
//            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
//            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
//         }
//         for (; j<jpos; j+=SIMDSIZE) {
//            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
//         }
//         for (; remainder && j<jend; ++j) {
//            *left += *right; ++left; ++right;
//         }
//      }
//   }
//}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a dense array.
//
// \param rhs The right-hand side dense array to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT >  // Type of the right-hand side dense array
inline auto DynamicArray<N, Type>::subAssign( const DenseArray<MT>& rhs )
   //-> DisableIf_t< VectorizedSubAssign_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( dims_ == (~rhs).dimensions()   , "Invalid array access index"    );

   const size_t jpos( dims_[0] & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( dims_[0] - ( dims_[0] % 2UL ) ) == jpos, "Invalid end calculation" );

   ArrayForEachGrouped(
      dims_, nn_, [&]( size_t i, std::array< size_t, N > const& dims ) {
         v_[i] -= ( ~rhs )( dims );
      } );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SIMD optimized implementation of the subtraction assignment of a dense array.
//
// \param rhs The right-hand side dense array to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
//template< size_t N         // The dimensionality of the array
//        , typename Type >  // Data type of the array
//template< typename MT >  // Type of the right-hand side dense array
//inline auto DynamicArray<N, Type>::subAssign( const DenseArray<MT>& rhs )
//   -> EnableIf_t< VectorizedSubAssign_v<MT> >
//{
//   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
//
//   BLAZE_INTERNAL_ASSERT( dims_ == (~rhs).dimensions()   , "Invalid array access index"    );
//
//   constexpr bool remainder( !usePadding || !IsPadded_v<MT> );
//
//   for (size_t k=0UL; k<o_; ++k) {
//      for (size_t i=0UL; i<m_; ++i)
//      {
//         const size_t jbegin(0UL);
//         const size_t jend  (n_);
//         BLAZE_INTERNAL_ASSERT(jbegin <= jend, "Invalid loop indices detected");
//
//         const size_t jpos((remainder)?(jend & size_t(-SIMDSIZE)):(jend));
//         BLAZE_INTERNAL_ASSERT(!remainder || (jend - (jend % (SIMDSIZE))) == jpos, "Invalid end calculation");
//
//         size_t j(jbegin);
//         Iterator left(begin(i, k) + jbegin);
//         ConstIterator_t<MT> right((~rhs).begin(i, k) + jbegin);
//
//         for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
//            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
//            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
//            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
//            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
//         }
//         for (; j<jpos; j+=SIMDSIZE) {
//            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
//         }
//         for (; remainder && j<jend; ++j) {
//            *left -= *right; ++left; ++right;
//         }
//      }
//   }
//}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the Schur product assignment of a dense array.
//
// \param rhs The right-hand side dense array for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
template< typename MT >  // Type of the right-hand side dense array
inline auto DynamicArray<N, Type>::schurAssign( const DenseArray<MT>& rhs )
   //-> DisableIf_t< VectorizedSchurAssign_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( dims_ == (~rhs).dimensions()   , "Invalid array access index"    );

   const size_t jpos( dims_[0] & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( dims_[0] - ( dims_[0] % 2UL ) ) == jpos, "Invalid end calculation" );

   ArrayForEachGrouped(
      dims_, nn_, [&]( size_t i, std::array< size_t, N > const& dims ) {
         v_[i] *= ( ~rhs )( dims );
      } );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SIMD optimized implementation of the Schur product assignment of a dense array.
//
// \param rhs The right-hand side dense array for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
//template< size_t N         // The dimensionality of the array
//        , typename Type >  // Data type of the array
//template< typename MT >  // Type of the right-hand side dense array
//inline auto DynamicArray<N, Type>::schurAssign( const DenseArray<MT>& rhs )
//   -> EnableIf_t< VectorizedSchurAssign_v<MT> >
//{
//   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
//
//   BLAZE_INTERNAL_ASSERT( dims_ == (~rhs).dimensions()   , "Invalid array access index"    );
//
//   constexpr bool remainder( !usePadding || !IsPadded_v<MT> );
//
//   for (size_t k=0UL; k<o_; ++k) {
//      for (size_t i=0UL; i<m_; ++i)
//      {
//         const size_t jpos((remainder)?(n_ & size_t(-SIMDSIZE)):(n_));
//         BLAZE_INTERNAL_ASSERT(!remainder || (n_ - (n_ % (SIMDSIZE))) == jpos, "Invalid end calculation");
//
//         size_t j(0UL);
//         Iterator left(begin(i, k));
//         ConstIterator_t<MT> right((~rhs).begin(i, k));
//
//         for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
//            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
//            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
//            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
//            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
//         }
//         for (; j<jpos; j+=SIMDSIZE) {
//            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
//         }
//         for (; remainder && j<n_; ++j) {
//            *left *= *right; ++left; ++right;
//         }
//      }
//   }
//}
//*************************************************************************************************






//=================================================================================================
//
//  DynamicArray OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DynamicArray operators */
//@{
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void reset( DynamicArray<N, Type>& m );

template< size_t N            // The dimensionality of the array
        , typename Type       // Data type of the array
        , typename... Dims >  // list of row indices
inline void reset( DynamicArray<N, Type>& m, size_t i, Dims... dims );

template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void clear( DynamicArray<N, Type>& m );

template< bool RF
        , size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline bool isDefault( const DynamicArray<N, Type>& m );

template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline bool isIntact( const DynamicArray<N, Type>& m ) noexcept;

template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void swap( DynamicArray<N, Type>& a, DynamicArray<N, Type>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given dynamic array.
// \ingroup dynamic_array
//
// \param m The array to be resetted.
// \return void
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void reset( DynamicArray<N, Type>& m )
{
   m.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset the specified row/column of the given dynamic array.
// \ingroup dynamic_array
//
// \param m The array to reset.
// \param i The index of the row/column to reset.
// \param k The index of the page to reset.
// \return void
//
// This function resets the values in the specified row/column of the given dynamic array to
// their default value. In case the given array is a \a rowMajor array the function resets the
// values in row \a i, if it is a \a columnMajor array the function resets the values in column
// \a i. Note that the capacity of the row/column remains unchanged.
*/
template< size_t N            // The dimensionality of the array
        , typename Type       // Data type of the array
        , typename... Dims >  // list of row indices
inline void reset( DynamicArray<N, Type>& m, size_t i, Dims... dims )
{
   m.reset( i, dims... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given dynamic array.
// \ingroup dynamic_array
//
// \param m The array to be cleared.
// \return void
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void clear( DynamicArray<N, Type>& m )
{
   m.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given dynamic array is in default state.
// \ingroup dynamic_array
//
// \param m The array to be tested for its default state.
// \return \a true in case the given array's rows and columns are zero, \a false otherwise.
//
// This function checks whether the dynamic array is in default (constructed) state, i.e. if
// it's number of rows and columns is 0. In case it is in default state, the function returns
// \a true, else it will return \a false. The following example demonstrates the use of the
// \a isDefault() function:

   \code
   blaze::DynamicArray<3, int> A;
   // ... Resizing and initialization
   if( isDefault( A ) ) { ... }
   \endcode

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isDefault<relaxed>( A ) ) { ... }
   \endcode
*/
template< bool RF        // Relaxation flag
        , size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline bool isDefault( const DynamicArray<N, Type>& m )
{
   auto const& dims = m.dimensions();
   return ArrayDimAllOf( dims, [&]( size_t, size_t dim ) { return dim == 0; } );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given dynamic array are intact.
// \ingroup dynamic_array
//
// \param m The dynamic array to be tested.
// \return \a true in case the given array's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the dynamic array are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   blaze::DynamicArray<3, int> A;
   // ... Resizing and initialization
   if( isIntact( A ) ) { ... }
   \endcode
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline bool isIntact( const DynamicArray<N, Type>& m ) noexcept
{
   return m.isIntact();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two dynamic matrices.
// \ingroup dynamic_array
//
// \param a The first array to be swapped.
// \param b The second array to be swapped.
// \return void
*/
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
inline void swap( DynamicArray<N, Type>& a, DynamicArray<N, Type>& b ) noexcept
{
   a.swap( b );
}
//*************************************************************************************************




//=================================================================================================
//
//  HASCONSTDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
struct HasConstDataAccess< DynamicArray<N, Type> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HASMUTABLEDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
struct HasMutableDataAccess< DynamicArray<N, Type> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISALIGNED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
struct IsAligned< DynamicArray<N, Type> >
   : public BoolConstant<usePadding>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISCONTIGUOUS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
struct IsContiguous< DynamicArray<N, Type> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISPADDED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
struct IsPadded< DynamicArray<N, Type> >
   : public BoolConstant<usePadding>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISRESIZABLE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
struct IsResizable< DynamicArray<N, Type> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSHRINKABLE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t N         // The dimensionality of the array
        , typename Type >  // Data type of the array
struct IsShrinkable< DynamicArray<N, Type> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ADDTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
// template< typename T1, typename T2 >
// struct AddTraitEval2< T1, T2
//                     , EnableIf_t< IsArray_v<T1> &&
//                                   IsArray_v<T2> &&
//                                   ( IsDenseArray_v<T1> || IsDenseArray_v<T2> ) &&
//                                   ( Size_v<T1,0UL> == DefaultSize_v ) &&
//                                   ( Size_v<T2,0UL> == DefaultSize_v ) &&
//                                   ( Size_v<T1,1UL> == DefaultSize_v ) &&
//                                   ( Size_v<T2,1UL> == DefaultSize_v ) &&
//                                   ( Size_v<T1,2UL> == DefaultSize_v ) &&
//                                   ( Size_v<T2,2UL> == DefaultSize_v ) &&
//                                   ( MaxSize_v<T1,0UL> == DefaultSize_v ) &&
//                                   ( MaxSize_v<T2,0UL> == DefaultSize_v ) &&
//                                   ( MaxSize_v<T1,1UL> == DefaultSize_v ) &&
//                                   ( MaxSize_v<T2,1UL> == DefaultSize_v ) &&
//                                   ( MaxSize_v<T1,2UL> == DefaultSize_v ) &&
//                                   ( MaxSize_v<T2,2UL> == DefaultSize_v ) > >
// {
//    using ET1 = ElementType_t<T1>;
//    using ET2 = ElementType_t<T2>;
//
//    using Type = DynamicArray< AddTrait_t<ET1,ET2> >;
// };
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
// template< typename T1, typename T2 >
// struct SubTraitEval2< T1, T2
//                     , EnableIf_t< IsArray_v<T1> &&
//                                   IsArray_v<T2> &&
//                                   ( IsDenseArray_v<T1> || IsDenseArray_v<T2> ) &&
//                                   ( Size_v<T1,0UL> == DefaultSize_v ) &&
//                                   ( Size_v<T2,0UL> == DefaultSize_v ) &&
//                                   ( Size_v<T1,1UL> == DefaultSize_v ) &&
//                                   ( Size_v<T2,1UL> == DefaultSize_v ) &&
//                                   ( Size_v<T1,2UL> == DefaultSize_v ) &&
//                                   ( Size_v<T2,2UL> == DefaultSize_v ) &&
//                                   ( MaxSize_v<T1,0UL> == DefaultMaxSize_v ) &&
//                                   ( MaxSize_v<T2,0UL> == DefaultMaxSize_v ) &&
//                                   ( MaxSize_v<T1,1UL> == DefaultMaxSize_v ) &&
//                                   ( MaxSize_v<T2,1UL> == DefaultMaxSize_v ) &&
//                                   ( MaxSize_v<T1,2UL> == DefaultMaxSize_v ) &&
//                                   ( MaxSize_v<T2,2UL> == DefaultMaxSize_v ) > >
// {
//    using ET1 = ElementType_t<T1>;
//    using ET2 = ElementType_t<T2>;
//
//    using Type = DynamicArray< SubTrait_t<ET1,ET2> >;
// };
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SCHURTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
// template< typename T1, typename T2 >
// struct SchurTraitEval2< T1, T2
//                       , EnableIf_t< IsDenseArray_v<T1> &&
//                                     IsDenseArray_v<T2> &&
//                                     ( Size_v<T1,0UL> == DefaultSize_v ) &&
//                                     ( Size_v<T2,0UL> == DefaultSize_v ) &&
//                                     ( Size_v<T1,1UL> == DefaultSize_v ) &&
//                                     ( Size_v<T2,1UL> == DefaultSize_v ) &&
//                                     ( Size_v<T1,2UL> == DefaultSize_v ) &&
//                                     ( Size_v<T2,2UL> == DefaultSize_v ) &&
//                                     ( MaxSize_v<T1,0UL> == DefaultMaxSize_v ) &&
//                                     ( MaxSize_v<T2,0UL> == DefaultMaxSize_v ) &&
//                                     ( MaxSize_v<T1,1UL> == DefaultMaxSize_v ) &&
//                                     ( MaxSize_v<T2,1UL> == DefaultMaxSize_v ) &&
//                                     ( MaxSize_v<T1,2UL> == DefaultMaxSize_v ) &&
//                                     ( MaxSize_v<T2,2UL> == DefaultMaxSize_v ) > >
// {
//    using ET1 = ElementType_t<T1>;
//    using ET2 = ElementType_t<T2>;
//
//    using Type = DynamicArray< MultTrait_t<ET1,ET2> >;
// };
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MULTTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t N, typename ET1, typename T2 >
struct MultTraitEval2< DynamicArray<N, ET1>, T2
                     , EnableIf_t< IsNumeric_v<T2>  > >
{
   using Type = DynamicArray< N, MultTrait_t<ET1,T2> >;
};

template< typename T1, size_t N, typename ET2 >
struct MultTraitEval2< T1, DynamicArray<N, ET2>
                     , EnableIf_t< IsNumeric_v<T1> > >
{
   using Type = DynamicArray< N, MultTrait_t<T1,ET2> >;
};

// template< typename T1, typename T2 >
// struct MultTraitEval2< T1, T2
//                      , EnableIf_t< IsArray_v<T1> &&
//                                    IsArray_v<T2> &&
//                                    ( IsDenseArray_v<T1> || IsDenseArray_v<T2> ) &&
//                                    ( ( Size_v<T1,0UL> == DefaultSize_v &&
//                                        ( !IsSquare_v<T1> || Size_v<T2,0UL> == DefaultSize_v ) ) ||
//                                      ( Size_v<T2,1UL> == DefaultSize_v &&
//                                        ( !IsSquare_v<T2> || Size_v<T1,1UL> == DefaultSize_v ) ) ||
//                                      ( Size_v<T2,2UL> == DefaultSize_v &&
//                                        ( !IsSquare_v<T2> || Size_v<T1,2UL> == DefaultSize_v ) ) ) &&
//                                    ( ( MaxSize_v<T1,0UL> == DefaultMaxSize_v &&
//                                        ( !IsSquare_v<T1> || MaxSize_v<T2,0UL> == DefaultMaxSize_v ) ) ||
//                                      ( MaxSize_v<T2,1UL> == DefaultMaxSize_v &&
//                                        ( !IsSquare_v<T2> || MaxSize_v<T1,1UL> == DefaultMaxSize_v ) ) ||
//                                      ( MaxSize_v<T2,2UL> == DefaultMaxSize_v &&
//                                        ( !IsSquare_v<T2> || MaxSize_v<T1,2UL> == DefaultMaxSize_v ) ) ) > >
// {
//    using ET1 = ElementType_t<T1>;
//    using ET2 = ElementType_t<T2>;
//
//    using Type = DynamicArray< MultTrait_t<ET1,ET2> >;
// };
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DIVTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t N, typename ET1, typename T2 >
struct DivTraitEval2< DynamicArray<N, ET1>, T2
                    , EnableIf_t< IsNumeric_v<T2> > >
{
   using Type = DynamicArray< N, DivTrait_t<ET1,T2> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ARRAYSLICETRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//template< size_t M, size_t N, typename ET, size_t I >
//struct ArraySliceTraitEval2< M, DynamicArray<N, ET>, I >
//{
//   using Type = DynamicArray< N - 1, ET >;
//};
/*! \endcond */
//*************************************************************************************************



//=================================================================================================
//
//  MAPTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t N, typename ET, typename OP >
struct UnaryMapTraitEval2< DynamicArray<N, ET>, OP>
{
   using Type = DynamicArray< N, MapTrait_t<ET,OP> >;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t N, typename ET1, typename ET2, typename OP >
struct BinaryMapTraitEval2< DynamicArray< N, ET1 > , DynamicArray< N, ET2 >, OP >
{
   using Type = DynamicArray< N, MapTrait_t<ET1,ET2,OP> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  EXPANDTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
// template< typename T  // Type to be expanded
//         , size_t E >  // Compile time expansion
// struct ExpandTraitEval2< T, E
//                        , EnableIf_t< IsDenseMatrix_v<T> &&
//                                      ( ( E == inf ) ||
//                                        ( ( Size_v<T,0UL> == DefaultSize_v ) &&
//                                          ( MaxSize_v<T,0UL> == DefaultMaxSize_v ) &&
//                                          ( Size_v<T,1UL> == DefaultSize_v ) &&
//                                          ( MaxSize_v<T,1UL> == DefaultMaxSize_v ) ) ) > >
// {
//    using Type = DynamicArray< ElementType_t<T> >;
// };
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RAVELTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
// FIXME: this needs to go into math/dense/DynamicMatrix.h
// template< typename T > // Type to be expanded
// struct RavelTraitEval2< T
//                        , EnableIf_t< IsDenseArray_v<T> &&
//                                      ( ( ( Size_v<T,0UL> == DefaultSize_v ) &&
//                                          ( MaxSize_v<T,0UL> == DefaultMaxSize_v ) &&
//                                          ( Size_v<T,1UL> == DefaultSize_v ) &&
//                                          ( MaxSize_v<T,1UL> == DefaultMaxSize_v ) &&
//                                          ( Size_v<T,2UL> == DefaultSize_v ) &&
//                                          ( MaxSize_v<T,2UL> == DefaultMaxSize_v ) ) ) > >
// {
//    using Type = DynamicVector< ElementType_t<T>, rowVector >;
// };
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HIGHTYPE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
// template< typename T1, typename T2 >
// struct HighType< DynamicArray<T1>, DynamicArray<T2> >
// {
//    using Type = DynamicArray< typename HighType<T1,T2>::Type >;
// };
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LOWTYPE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
// template< typename T1, typename T2 >
// struct LowType< DynamicArray<T1>, DynamicArray<T2> >
// {
//    using Type = DynamicArray< typename LowType<T1,T2>::Type >;
// };
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
