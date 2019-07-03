//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/CustomArray.h
//  \brief Header file for the implementation of a customizable array
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_CUSTOMARRAY_H_
#define _BLAZE_TENSOR_MATH_DENSE_CUSTOMARRAY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <utility>
#include <blaze/math/dense/CustomMatrix.h>
#include <blaze/util/EnableIf.h>

#include <blaze_tensor/math/Forward.h>
#include <blaze_tensor/math/InitializerList.h>
#include <blaze_tensor/math/Array.h>
#include <blaze_tensor/math/expressions/DenseArray.h>
#include <blaze_tensor/math/SMP.h>
#include <blaze_tensor/math/typetraits/IsDenseArray.h>
#include <blaze_tensor/math/typetraits/IsNdArray.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup custom_array CustomArray
// \ingroup dense_array
*/
/*!\brief Efficient implementation of a customizable array.
// \ingroup custom_array
//
// The CustomArray class template provides the functionality to represent an external array of
// elements of arbitrary type and a fixed size as a native \b Blaze dense array data structure.
// Thus in contrast to all other dense array types a custom array does not perform any kind
// of memory allocation by itself, but it is provided with an existing array of element during
// construction. A custom array can therefore be considered an alias to the existing array.
//
// The type of the elements, the properties of the given array of elements and the storage order
// of the array can be specified via the following four template parameters:

   \code
   template< size_t N, typename T, bool AF, bool PF, typename RT >
   class CustomArray;
   \endcode

//  - N:    dimensionality of the dense array
//  - Type: specifies the type of the array elements. CustomArray can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//  - AF  : specifies whether the represented, external arrays are properly aligned with
//          respect to the available instruction set (SSE, AVX, ...) or not.
//  - PF  : specified whether the represented, external arrays are properly padded with
//          respect to the available instruction set (SSE, AVX, ...) or not.
//
// The following examples give an impression of several possible types of custom matrices:

   \code
   using blaze::CustomArray;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;

   // Definition of a 3-D custom array for unaligned, unpadded integer arrays
   using UnalignedUnpadded = CustomArray<3,int,unaligned,unpadded>;

   // Definition of a custom 4-D array for unaligned but padded 'float' arrays
   using UnalignedPadded = CustomArray<4,float,unaligned,padded>;

   // Definition of a custom 4-D array for aligned, unpadded 'double' arrays
   using AlignedUnpadded = CustomArray<4,double,aligned,unpadded>;

   // Definition of a custom 3-D array for aligned, padded 'complex<double>' arrays
   using AlignedPadded = CustomArray<3,complex<double>,aligned,padded>;
   \endcode

// \n \section customarray_special_properties Special Properties of Custom Matrices
//
// In comparison with the remaining \b Blaze dense array types CustomArray has several special
// characteristics. All of these result from the fact that a custom array is not performing any
// kind of memory allocation, but instead is given an existing array of elements. The following
// sections discuss all of these characteristics:
//
//  -# <b>\ref customarray_memory_management</b>
//  -# <b>\ref customarray_copy_operations</b>
//  -# <b>\ref customarray_alignment</b>
//  -# <b>\ref customarray_padding</b>
//
// \n \subsection customarray_memory_management Memory Management
//
// The CustomArray class template acts as an adaptor for an existing array of elements. As such
// it provides everything that is required to use the array just like a native \b Blaze dense
// array data structure. However, this flexibility comes with the price that the user of a custom
// array is responsible for the resource management.
//
// The following examples give an impression of several possible custom matrices:

   \code
   using blaze::CustomArray;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;

   // Definition of a 3x4 custom array with unaligned, unpadded and externally
   // managed integer array. Note that the std::vector must be guaranteed to outlive the
   // custom array!
   std::vector<int> vec( 12UL );
   CustomArray<2,int,unaligned,unpadded> A( &vec[0], 3UL, 4UL );

   // Definition of a custom 8x12 array for an aligned and padded integer array of
   // capacity 128 (including 8 padding elements per row). Note that the std::unique_ptr
   // must be guaranteed to outlive the custom array!
   std::unique_ptr<int[],Deallocate> memory( allocate<int>( 128UL ) );
   CustomArray<2,int,aligned,padded> B( memory.get(), 8UL, 12UL, 16UL );
   \endcode

// \n \subsection customarray_copy_operations Copy Operations
//
// As with all dense matrices it is possible to copy construct a custom array:

   \code
   using blaze::CustomArray;
   using blaze::unaligned;
   using blaze::unpadded;

   using CustomType = CustomArray<2,int,unaligned,unpadded>;

   std::vector<int> vec( 6UL, 10 );    // Vector of 6 integers of the value 10
   CustomType A( &vec[0], 2UL, 3UL );  // Represent the std::vector as Blaze dense array
   a[1] = 20;                          // Also modifies the std::vector

   CustomType B( a );  // Creating a copy of vector a
   b[2] = 20;          // Also affects array A and the std::vector
   \endcode

// It is important to note that a custom array acts as a reference to the specified array. Thus
// the result of the copy constructor is a new custom array that is referencing and representing
// the same array as the original custom array.
//
// In contrast to copy construction, just as with references, copy assignment does not change
// which array is referenced by the custom matrices, but modifies the values of the array:

   \code
   std::vector<int> vec2( 6UL, 4 );     // Vector of 6 integers of the value 4
   CustomType C( &vec2[0], 2UL, 3UL );  // Represent the std::vector as Blaze dense array

   A = C;  // Copy assignment: Set all values of array A and B to 4.
   \endcode

// \n \subsection customarray_alignment Alignment
//
// In case the custom array is specified as \a aligned the passed array must adhere to some
// alignment restrictions based on the alignment requirements of the used data type and the
// used instruction set (SSE, AVX, ...). The restriction applies to the first element of each
// row/column: In case of a array the first element of each row must be properly
// aligned, in case of a column-major array the first element of each column must be properly
// aligned. For instance, if a array is used and AVX is active the first element of
// each row must be 32-bit aligned:

   \code
   using blaze::CustomArray;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::padded;
   using blaze::rowMajor;

   // Allocation of 32-bit aligned memory
   std::unique_ptr<int[],Deallocate> memory( allocate<int>( 40UL ) );

   CustomArray<2,int,aligned,padded> A( memory.get(), 8UL, 5UL, 6UL );
   \endcode

// In the example, the array has six columns. However, since with AVX eight integer
// values are loaded together the array is padded with two additional elements. This guarantees
// that the first element of each row is 32-bit aligned. In case the alignment requirements are
// violated, a \a std::invalid_argument exception is thrown.
//
// \n \subsection customarray_padding Padding
//
// Adding padding elements to the end of an array can have a significant impact on performance.
// For instance, assuming that AVX is available, then two aligned, padded, 3x3 double precision
// matrices can be added via three SIMD addition operations:

   \code
   using blaze::CustomArray;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::padded;

   using CustomType = CustomArray<2,double,aligned,padded>;

   std::unique_ptr<int[],Deallocate> memory1( allocate<double>( 12UL ) );
   std::unique_ptr<int[],Deallocate> memory2( allocate<double>( 12UL ) );
   std::unique_ptr<int[],Deallocate> memory3( allocate<double>( 12UL ) );

   // Creating padded custom 3x3 array with an additional padding element in each row
   CustomType A( memory1.get(), 3UL, 3UL, 4UL );
   CustomType B( memory2.get(), 3UL, 3UL, 4UL );
   CustomType C( memory3.get(), 3UL, 3UL, 4UL );

   // ... Initialization

   C = A + B;  // AVX-based array addition
   \endcode

// In this example, maximum performance is possible. However, in case no padding elements are
// inserted a scalar addition has to be used:

   \code
   using blaze::CustomArray;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::unpadded;

   using CustomType = CustomArray<2,double,aligned,unpadded>;

   std::unique_ptr<int[],Deallocate> memory1( allocate<double>( 9UL ) );
   std::unique_ptr<int[],Deallocate> memory2( allocate<double>( 9UL ) );
   std::unique_ptr<int[],Deallocate> memory3( allocate<double>( 9UL ) );

   // Creating unpadded custom 3x3 array
   CustomType A( memory1.get(), 3UL, 3UL );
   CustomType B( memory2.get(), 3UL, 3UL );
   CustomType C( memory3.get(), 3UL, 3UL );

   // ... Initialization

   C = A + B;  // Scalar array addition
   \endcode

// Note that the construction of padded and unpadded aligned matrices looks identical. However,
// in case of padded matrices, \b Blaze will zero initialize the padding element and use them
// in all computations in order to achieve maximum performance. In case of an unpadded array
// \b Blaze will ignore the elements with the downside that it is not possible to load a complete
// row to an AVX register, which makes it necessary to fall back to a scalar addition.
//
// The number of padding elements is required to be sufficient with respect to the available
// instruction set: In case of an aligned padded custom array the added padding elements must
// guarantee that the total number of elements in each row/column is a multiple of the SIMD
// vector width. In case of an unaligned padded array the number of padding elements can be
// greater or equal the number of padding elements of an aligned padded custom array. In case
// the padding is insufficient with respect to the available instruction set, a
// \a std::invalid_argument exception is thrown.
//
//
// \n \section customarray_arithmetic_operations Arithmetic Operations
//
// The use of custom matrices in arithmetic operations is designed to be as natural and intuitive
// as possible. All operations (addition, subtraction, multiplication, scaling, ...) can be
// expressed similar to a text book representation. Also, custom matrices can be combined with all
// other dense and sparse vectors and matrices. The following example gives an impression of the
// use of CustomArray:

   \code
   using blaze::CustomArray;
   using blaze::CompressedArray;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Non-initialized custom 2x3 array. All given arrays are considered to be
   // unaligned and unpadded. The memory is managed via a 'std::vector'.
   std::vector<double> memory1( 6UL );
   CustomArray<2,double,unaligned,unpadded> A( memory1.data(), 2UL, 3UL );

   A(0,0) = 1.0; A(0,1) = 2.0; A(0,2) = 3.0;  // Initialization of the first row
   A(1,0) = 4.0; A(1,1) = 5.0; A(1,2) = 6.0;  // Initialization of the second row

   // Non-initialized custom 2x3 array with padding elements. All given arrays are
   // required to be properly aligned and padded. The memory is managed via a 'std::unique_ptr'.
   std::unique_ptr<double[],Deallocate> memory2( allocate<double>( 16UL ) );
   CustomArray<2,double,aligned,padded> B( memory2.get(), 8UL, 2UL, 3UL );

   B(0,0) = 1.0; B(0,1) = 3.0; B(0,2) = 5.0;    // Initialization of the first row
   B(1,0) = 2.0; B(1,1) = 4.0; B(1,2) = 6.0;    // Initialization of the second row

   CompressedArray<2,float> C( 2, 3 );        // Empty sparse single precision array
   DynamicArray<2,float>    D( init_from_value, 4.0F, 3, 2 );  // Directly, homogeneously initialized single precision 3x2 array

   DynamicArray<2,double>    E( A );  // Creation of a new array as a copy of A
   DynamicArray<2,double> F;       // Creation of a default column-major array

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
template< size_t N                                               // Dimensionality of the array
        , typename Type                                          // Data type of the array
        , bool AF                                                // Alignment flag
        , bool PF                                                // Padding flag
        , typename RT = DynamicArray<N, RemoveConst_t<Type>> >   // Result type
class CustomArray
   : public DenseArray< CustomArray<N,Type,AF,PF,RT> >
{
 public:
   //**Type definitions****************************************************************************
   using This     = CustomArray<N,Type,AF,PF,RT>;  //!< Type of this CustomArray instance.
   using BaseType = DenseArray<This>;              //!< Base type of this CustomArray instance.

   //! Result type for expression template evaluations.
   using ResultType = RT;

   //! Result type with opposite storage order for expression template evaluations.
   using OppositeType = OppositeType_t<RT>;

   //! Transpose type for expression template evaluations.
   using TransposeType = TransposeType_t<RT>;

   using ElementType   = Type;                      //!< Type of the array elements.
   using SIMDType      = SIMDTrait_t<ElementType>;  //!< SIMD type of the array elements.
   using ReturnType    = const Type&;               //!< Return type for expression template evaluations.
   using CompositeType = const This&;               //!< Data type for composite expression templates.

   using Reference      = Type&;        //!< Reference to a non-constant array value.
   using ConstReference = const Type&;  //!< Reference to a constant array value.
   using Pointer        = Type*;        //!< Pointer to a non-constant array value.
   using ConstPointer   = const Type*;  //!< Pointer to a constant array value.

   using Iterator      = DenseIterator<Type,AF>;        //!< Iterator over non-constant elements.
   using ConstIterator = DenseIterator<const Type,AF>;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a CustomArray with different data/element type.
   */
   template< typename NewType >  // Data type of the other array
   struct Rebind {
      using RRT   = Rebind_t< RT, RemoveConst_t<NewType> >;  //!< The rebound result type.
      using Other = CustomArray<N,NewType,AF,PF,RRT>;        //!< The type of the other CustomArray.
   };
   //**********************************************************************************************

   //**Resize struct definition********************************************************************
   /*!\brief Resize mechanism to obtain a CustomArray with different fixed dimensions.
   */
   template< size_t... New > // Dimensionalities of the other array
   struct Resize {
      using RRT   = Resize_t<RT,New...>;            //!< The resized result type.
      using Other = CustomArray<N,Type,AF,PF,RRT>;  //!< The type of the other CustomArray.
   };
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
   explicit inline CustomArray();
   template< typename... Dims >
   explicit inline CustomArray( Type const* ptr, Dims... dims );

   inline CustomArray( const CustomArray& m );
   inline CustomArray( CustomArray&& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~CustomArray() = default;
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
   inline CustomArray& operator=( const Type& set );
   inline CustomArray& operator=( nested_initializer_list< N, Type > list );

   inline CustomArray& operator=( const CustomArray& rhs );
   inline CustomArray& operator=( CustomArray&& rhs ) noexcept;

   template< typename MT > inline CustomArray& operator= ( const Array<MT>& rhs );
   template< typename MT > inline CustomArray& operator+=( const Array<MT>& rhs );
   template< typename MT > inline CustomArray& operator-=( const Array<MT>& rhs );
   template< typename MT > inline CustomArray& operator%=( const Array<MT>& rhs );

   //template< typename MT, bool TF, size_t M = N, typename = EnableIf_t< M == 1 > >
   //inline ArraySlice& operator= ( const Vector<MT, TF>& m );
   //template< typename MT, bool TF, size_t M = N, typename = EnableIf_t< M == 1 > >
   //inline ArraySlice& operator+=( const Vector<MT, TF>& m );
   //template< typename MT, bool TF, size_t M = N, typename = EnableIf_t< M == 1 > >
   //inline ArraySlice& operator-=( const Vector<MT, TF>& m );
   //template< typename MT, bool TF, size_t M = N, typename = EnableIf_t< M == 1 > >
   //inline ArraySlice& operator%=( const Vector<MT, TF>& m );

   //template< typename MT, bool SO, size_t M = N, typename = EnableIf_t< M == 2 > >
   //inline ArraySlice& operator= ( const Matrix<MT, SO>& m );
   //template< typename MT, bool SO, size_t M = N, typename = EnableIf_t< M == 2 > >
   //inline ArraySlice& operator+=( const Matrix<MT, SO>& m );
   //template< typename MT, bool SO, size_t M = N, typename = EnableIf_t< M == 2 > >
   //inline ArraySlice& operator-=( const Matrix<MT, SO>& m );
   //template< typename MT, bool SO, size_t M = N, typename = EnableIf_t< M == 2 > >
   //inline ArraySlice& operator%=( const Matrix<MT, SO>& m );

   //template< typename MT, size_t M = N, typename = EnableIf_t< M == 3 > >
   //inline ArraySlice& operator= ( const Tensor<MT>& m );
   //template< typename MT, size_t M = N, typename = EnableIf_t< M == 3 > >
   //inline ArraySlice& operator+=( const Tensor<MT>& m );
   //template< typename MT, size_t M = N, typename = EnableIf_t< M == 3 > >
   //inline ArraySlice& operator-=( const Tensor<MT>& m );
   //template< typename MT, size_t M = N, typename = EnableIf_t< M == 3 > >
   //inline ArraySlice& operator%=( const Tensor<MT>& m );
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
   inline void   swap( CustomArray& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   inline CustomArray& transpose();
   inline CustomArray& ctranspose();
   template< typename T >
   inline CustomArray& transpose( const T* indices, size_t n );
   template< typename T >
   inline CustomArray& ctranspose( const T* indices, size_t n );

   template< typename Other > inline CustomArray& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

   //**Resource management functions***************************************************************
   /*!\name Resource management functions */
   //@{
   template< typename... Dims >
   inline void reset( Type* ptr, Dims... dims );
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
        IsSIMDCombinable_v< Type, ElementType_t<MT> > );
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
        !IsDiagonal_v<MT> );
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
        !IsDiagonal_v<MT> );
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
        HasSIMDMult_v< Type, ElementType_t<MT> > );
   /*! \endcond */
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   static constexpr size_t SIMDSIZE = SIMDTrait<ElementType>::size;
   //**********************************************************************************************

 public:
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
   inline auto assign( const DenseArray<MT>& rhs ) -> DisableIf_t< VectorizedAssign_v<MT> >;

   template< typename MT >
   inline auto assign( const DenseArray<MT>& rhs ) -> EnableIf_t< VectorizedAssign_v<MT> >;

   template< typename MT >
   inline auto addAssign( const DenseArray<MT>& rhs ) -> DisableIf_t< VectorizedAddAssign_v<MT> >;

   template< typename MT >
   inline auto addAssign( const DenseArray<MT>& rhs ) -> EnableIf_t< VectorizedAddAssign_v<MT> >;

   template< typename MT >
   inline auto subAssign( const DenseArray<MT>& rhs ) -> DisableIf_t< VectorizedSubAssign_v<MT> >;

   template< typename MT >
   inline auto subAssign( const DenseArray<MT>& rhs ) -> EnableIf_t< VectorizedSubAssign_v<MT> >;

   template< typename MT >
   inline auto schurAssign( const DenseArray<MT>& rhs ) -> DisableIf_t< VectorizedSchurAssign_v<MT> >;

   template< typename MT >
   inline auto schurAssign( const DenseArray<MT>& rhs ) -> EnableIf_t< VectorizedSchurAssign_v<MT> >;
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   template< typename... Dims >
   inline static std::array< size_t, N > initDimensions( Dims... dims ) noexcept;
   template< typename... Dims >
   inline static size_t initSpacing( Dims... dims ) noexcept;
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
   size_t nn_;                     //!< The number of elements between two rows.
   Type* BLAZE_RESTRICT v_;       //!< The custom array of elements.
                /*!< Access to the array elements is gained via the function call
                     operator. */
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE  ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE( Type );
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
/*!\brief The default constructor for CustomArray.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomArray<N,Type,AF,PF,RT>::CustomArray()
   : dims_(  )       // The current dimensions of the array
   , nn_( 0UL )      // The number of elements between two rows
   , v_ ( nullptr )  // The custom array of elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a array of size \f$ m \times n \f$.
//
// \param ptr The array of elements to be used by the array.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param o The number of pages of the array of elements.
// \param nn The total number of elements between two rows/columns.
// \exception std::invalid_argument Invalid setup of custom array.
//
// This constructor creates a custom array of size \f$ m \times n \f$. The construction fails
// if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified spacing \a nn is insufficient for the given data type \a Type and the
//    available instruction set.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note The custom array does \b NOT take responsibility for the given array of elements!
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline CustomArray<N,Type,AF,PF,RT>::CustomArray( Type const* ptr, Dims... dims )
   : dims_ ( initDimensions( dims... ) )   // The current dimensions of the array
   , nn_( initSpacing( dims... ) )         // The number of elements between two rows
   , v_ ( const_cast< Type* BLAZE_RESTRICT >(ptr) )  // The custom array of elements
{
   using blaze::clear;

   using ClearFunctor = If_t< PF || !IsConst_v<Type>, Clear, Noop >;

   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && ( !checkAlignment( ptr ) || nn_ % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   if( sizeof...(dims) == N + 1 ) {
      if( PF && IsVectorizable_v<Type> && ( nn_ < nextMultiple<size_t>( dims_[0], SIMDSIZE ) ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Insufficient capacity for padded array" );
      }

      if( PF && IsVectorizable_v<Type> ) {
         ClearFunctor clear;
         ArrayForEachPadded( dims_, nn_, [&]( size_t i ) { clear( v_[i] ); } );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for CustomArray.
//
// \param m Array to be copied.
//
// The copy constructor initializes the custom array as an exact copy of the given custom array.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomArray<N,Type,AF,PF,RT>::CustomArray( const CustomArray& m )
   : dims_( m.dims_ )   // The current number of pages of the array
   , nn_( m.nn_ )       // The number of elements between two rows
   , v_ ( m.v_ )        // The custom array of elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The move constructor for CustomArray.
//
// \param m The array to be moved into this instance.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomArray<N,Type,AF,PF,RT>::CustomArray( CustomArray&& m ) noexcept
   : dims_( std::move( m.dims_ ) )   // The current number of pages of the array
   , nn_( m.nn_ )       // The number of elements between two rows
   , v_ ( m.v_ )        // The custom array of elements
{
   m.dims_ = std::array< size_t, N >{};
   m.nn_ = 0UL;
   m.v_  = nullptr;

   BLAZE_INTERNAL_ASSERT( m.data() == nullptr, "Invalid data reference detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initialize the dimensions array.
//
// \param value The dimensions for this CustomArray.
// \return The dimensions array.
//
// This function initializes the internal dimensions array.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline std::array< size_t, N > CustomArray<N,Type,AF,PF,RT>::initDimensions( Dims... dims ) noexcept
{
   BLAZE_STATIC_ASSERT( N == sizeof...( Dims ) || N + 1 == sizeof...( Dims ) );

   // the last dimension could be the spacing that needs to be ignored
   constexpr size_t padding_offset = sizeof...( Dims ) - N;

   // the last given dimension is always the lowest
   const size_t indices[] = { dims... };

   std::array< size_t, N > result;
   for( size_t i = 0; i != N; ++i ) {
      result[i] = indices[sizeof...( Dims ) - i - 1 - padding_offset];
   }
   return result;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initialize the spacing.
//
// \param value The dimensions for this CustomArray.
// \return The dimensions array.
//
// This function initializes the internal dimensions array.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline size_t CustomArray<N,Type,AF,PF,RT>::initSpacing( Dims... dims ) noexcept
{
   BLAZE_STATIC_ASSERT( N == sizeof...( Dims ) || N + 1 == sizeof...( Dims ) );

   // the last given dimension is the spacing
   const size_t indices[] = { dims... };
   return indices[sizeof...( Dims ) - 1];
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline size_t CustomArray<N,Type,AF,PF,RT>::row_index( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   size_t indices[] = { dims..., i, 0 };

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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline size_t CustomArray<N,Type,AF,PF,RT>::index( Dims... dims ) const noexcept
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomArray<N,Type,AF,PF,RT>::index( std::array< size_t, N > const& indices ) const noexcept
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline constexpr std::array< size_t, N > const& CustomArray<N,Type,AF,PF,RT>::dimensions() const noexcept
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
template< size_t N       // The dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomArray<N,Type,AF,PF,RT>::quats() const noexcept
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
template< size_t N       // The dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomArray<N,Type,AF,PF,RT>::pages() const noexcept
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
template< size_t N       // The dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomArray<N,Type,AF,PF,RT>::rows() const noexcept
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
template< size_t N       // The dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomArray<N,Type,AF,PF,RT>::columns() const noexcept
{
   return dims_[0];
}
//*************************************************************************************************


//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief 2D-access to the array elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline typename CustomArray<N,Type,AF,PF,RT>::Reference
   CustomArray<N,Type,AF,PF,RT>::operator()( Dims... dims ) noexcept
{
   BLAZE_STATIC_ASSERT( N == sizeof...( Dims ) );

#if defined(BLAZE_USER_ASSERTION)
   size_t indices[] = { size_t(dims)... };

   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_USER_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
#endif

   return v_[index( dims... )];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief 2D-access to the array elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline typename CustomArray<N,Type,AF,PF,RT>::ConstReference
   CustomArray<N,Type,AF,PF,RT>::operator()( Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N == sizeof...( Dims ) );

#if defined(BLAZE_USER_ASSERTION)
   size_t indices[] = { size_t(dims)... };

   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_USER_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomArray<N,Type,AF,PF,RT>::Reference
   CustomArray<N,Type,AF,PF,RT>::operator()( std::array< size_t, N > const& indices ) noexcept
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomArray<N,Type,AF,PF,RT>::ConstReference
   CustomArray<N,Type,AF,PF,RT>::operator()( std::array< size_t, N > const& indices ) const noexcept
{
#if defined(BLAZE_USER_ASSERTION)
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_USER_ASSERT( indices[i] < dim, "Invalid array access index" );
   } );
#endif

   size_t idx = index( indices );

   return v_[index( indices )];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the array elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid array access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline typename CustomArray<N,Type,AF,PF,RT>::Reference
   CustomArray<N,Type,AF,PF,RT>::at( Dims... dims )
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
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid array access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline typename CustomArray<N,Type,AF,PF,RT>::ConstReference
   CustomArray<N,Type,AF,PF,RT>::at( Dims... dims ) const
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomArray<N,Type,AF,PF,RT>::Reference
   CustomArray<N,Type,AF,PF,RT>::at( std::array< size_t, N > const& indices )
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomArray<N,Type,AF,PF,RT>::ConstReference
   CustomArray<N,Type,AF,PF,RT>::at( std::array< size_t, N > const& indices ) const
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomArray<N,Type,AF,PF,RT>::Pointer
   CustomArray<N,Type,AF,PF,RT>::data() noexcept
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomArray<N,Type,AF,PF,RT>::ConstPointer
   CustomArray<N,Type,AF,PF,RT>::data() const noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the array elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline typename CustomArray<N,Type,AF,PF,RT>::Pointer
   CustomArray<N,Type,AF,PF,RT>::data( size_t i, Dims... dims ) noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return v_ + row_index( i, dims... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the array elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline typename CustomArray<N,Type,AF,PF,RT>::ConstPointer
   CustomArray<N,Type,AF,PF,RT>::data( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return v_ + row_index( i, dims... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline typename CustomArray<N,Type,AF,PF,RT>::Iterator
   CustomArray<N,Type,AF,PF,RT>::begin( size_t i, Dims... dims ) noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return Iterator( v_ + row_index( i, dims... ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline typename CustomArray<N,Type,AF,PF,RT>::ConstIterator
   CustomArray<N,Type,AF,PF,RT>::begin( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return ConstIterator( v_ + row_index( i, dims... ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline typename CustomArray<N,Type,AF,PF,RT>::ConstIterator
   CustomArray<N,Type,AF,PF,RT>::cbegin( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return ConstIterator( v_ + row_index( i, dims... ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline typename CustomArray<N,Type,AF,PF,RT>::Iterator
   CustomArray<N,Type,AF,PF,RT>::end( size_t i, Dims... dims ) noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return Iterator( v_ + row_index( i, dims... ) + dims_[0] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline typename CustomArray<N,Type,AF,PF,RT>::ConstIterator
   CustomArray<N,Type,AF,PF,RT>::end( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( dims ) );

   return ConstIterator( v_ + row_index( i, dims... ) + dims_[0] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline typename CustomArray<N,Type,AF,PF,RT>::ConstIterator
   CustomArray<N,Type,AF,PF,RT>::cend( size_t i, Dims... dims ) const noexcept
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomArray<N,Type,AF,PF,RT>&
   CustomArray<N,Type,AF,PF,RT>::operator=( const Type& rhs )
{
   ArrayForEach( dims_, nn_, [&]( size_t i ) { v_[i] = rhs; } );
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List assignment to all array elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to static array.
//
// This assignment operator offers the option to directly assign to all elements of the array
// by means of an initializer list:

   \code
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;

   const int array[9] = { 0, 0, 0,
                          0, 0, 0,
                          0, 0, 0 };
   blaze::CustomArray<int,unaligned,unpadded> A( array, 3UL, 3UL );
   A = { { 1, 2, 3 },
         { 4, 5 },
         { 7, 8, 9 } };
   \endcode

// The array elements are assigned the values from the given initializer list. Missing values
// are initialized as default (as e.g. the value 6 in the example). Note that in case the size
// of the top-level initializer list exceeds the number of rows or the size of any nested list
// exceeds the number of columns, a \a std::invalid_argument exception is thrown.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomArray<N,Type,AF,PF,RT>&
   CustomArray<N,Type,AF,PF,RT>::operator=( nested_initializer_list< N, Type > list )
{
   auto list_dims = list.dimensions();
   if( ArrayDimAnyOf( dims_,
          [&]( size_t i, size_t dim ) { return list_dims[i] != dim; } ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to custom array" );
   }

   list.transfer_data( *this );

   if( IsVectorizable_v<Type> ) {
      ArrayForEachPadded( dims_, nn_, [&]( size_t i ) { v_[i] = Type(); } );
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy assignment operator for CustomArray.
//
// \param rhs Array to be copied.
// \return Reference to the assigned array.
// \exception std::invalid_argument Array sizes do not match.
//
// The array is initialized as a copy of the given array. In case the current sizes of the two
// matrices don't match, a \a std::invalid_argument exception is thrown.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomArray<N,Type,AF,PF,RT>&
   CustomArray<N,Type,AF,PF,RT>::operator=( const CustomArray& rhs )
{
   auto rhsdims = rhs.dimensions();
   if( ArrayDimAnyOf( dims_,
          [&]( size_t i, size_t dim ) { return rhsdims[i] != dim; } ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array sizes do not match" );
   }

   smpAssign( *this, ~rhs );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Move assignment operator for CustomArray.
//
// \param rhs Array to be copied.
// \return Reference to the assigned array.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomArray<N,Type,AF,PF,RT>&
   CustomArray<N,Type,AF,PF,RT>::operator=( CustomArray&& rhs ) noexcept
{
   dims_ = std::move( rhs.dims_ );   // The current dimensions of the array
   nn_ = rhs.nn_;   // The number of elements between two rows
   v_ = rhs.v_;     // The custom array of elements

   rhs.dims_ = std::array< size_t, N >{};
   rhs.nn_ = 0UL;
   rhs.v_  = nullptr;

   BLAZE_INTERNAL_ASSERT( rhs.data() == nullptr, "Invalid data reference detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different matrices.
//
// \param rhs Array to be copied.
// \return Reference to the assigned array.
// \exception std::invalid_argument Array sizes do not match.
//
// The array is initialized as a copy of the given array. In case the current sizes of the two
// matrices don't match, a \a std::invalid_argument exception is thrown.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side array
inline CustomArray<N,Type,AF,PF,RT>&
   CustomArray<N,Type,AF,PF,RT>::operator=( const Array<MT>& rhs )
{
   if( dims_ != (~rhs).dimensions() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAssign( *this, tmp );
   }
   else {
      smpAssign( *this, ~rhs );
   }

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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side array
inline CustomArray<N,Type,AF,PF,RT>&
   CustomArray<N,Type,AF,PF,RT>::operator+=( const Array<MT>& rhs )
{
   if( dims_ != (~rhs).dimensions() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpAddAssign( *this, tmp );
   }
   else {
      smpAddAssign( *this, ~rhs );
   }

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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side array
inline CustomArray<N,Type,AF,PF,RT>&
   CustomArray<N,Type,AF,PF,RT>::operator-=( const Array<MT>& rhs )
{
   if( dims_ != (~rhs).dimensions() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpSubAssign( *this, tmp );
   }
   else {
      smpSubAssign( *this, ~rhs );
   }

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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side array
inline CustomArray<N,Type,AF,PF,RT>&
   CustomArray<N,Type,AF,PF,RT>::operator%=( const Array<MT>& rhs )
{
   if( dims_ != (~rhs).dimensions() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Array sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      smpSchurAssign( *this, tmp );
   }
   else {
      smpSchurAssign( *this, ~rhs );
   }

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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template < size_t Dim >
inline size_t CustomArray<N,Type,AF,PF,RT>::dimension() const noexcept
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomArray<N,Type,AF,PF,RT>::spacing() const noexcept
{
   return nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the array.
//
// \return The capacity of the array.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomArray<N,Type,AF,PF,RT>::capacity() const noexcept
{
   size_t capacity = nn_;
   for( size_t i = 1; i < N; ++i ) {
      capacity *= dims_[i];
   }
   return capacity;
}
//*************************************************************************************************


//*************************************************************************************************
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline size_t CustomArray<N,Type,AF,PF,RT>::capacity( size_t i, Dims... dims ) const noexcept
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( Dims ) );

   MAYBE_UNUSED( dims... );

#if defined(BLAZE_USER_ASSERTION)
   size_t indices[] = { size_t(dims)..., i, 0 };

   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_USER_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
#endif

   return nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of non-zero elements in the array
//
// \return The number of non-zero elements in the dense array.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomArray<N,Type,AF,PF,RT>::nonZeros() const
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline size_t CustomArray<N,Type,AF,PF,RT>::nonZeros( size_t i, Dims... dims ) const
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( Dims ) );

#if defined(BLAZE_USER_ASSERTION)
   size_t indices[] = { size_t(dims)..., i, 0 };

   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_USER_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void CustomArray<N,Type,AF,PF,RT>::reset()
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline void CustomArray<N,Type,AF,PF,RT>::reset( size_t i, Dims... dims )
{
   BLAZE_STATIC_ASSERT( N - 2 == sizeof...( Dims ) );

#if defined(BLAZE_USER_ASSERTION)
   size_t indices[] = { size_t(dims)..., i, 0 };

   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_USER_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
#endif

   using blaze::clear;

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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void CustomArray<N,Type,AF,PF,RT>::clear()
{
   dims_ = std::array< size_t, N >{};
   nn_ = 0UL;
   v_  = nullptr;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two matrices.
//
// \param m The array to be swapped.
// \return void
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void CustomArray<N,Type,AF,PF,RT>::swap( CustomArray& m ) noexcept
{
   using std::swap;

   swap( dims_ , m.dims_  );
   swap( nn_, m.nn_ );
   swap( v_ , m.v_  );
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
// \exception std::logic_error Impossible transpose operation.
//
// In case the array is not a square array, a \a std::logic_error exception is thrown.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomArray<N,Type,AF,PF,RT>& CustomArray<N,Type,AF,PF,RT>::transpose()
{
//    using std::swap;
//
//    if( o_ != m_ || m_ != n_ ) {
//       BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
//    }
//
//    for( size_t i=1UL; i<m_; ++i )
//       for( size_t j=0UL; j<i; ++j )
//          swap( v_[i*nn_+j], v_[j*nn_+i] );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the array.
//
// \return Reference to the transposed array.
// \exception std::logic_error Impossible transpose operation.
//
// In case the array is not a square array, a \a std::logic_error exception is thrown.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename T >   // Type of the mapping indices
inline CustomArray<N,Type,AF,PF,RT>& CustomArray<N,Type,AF,PF,RT>::transpose( const T* indices, size_t n )
{
//    using std::swap;
//
//    if( o_ != m_ || m_ != n_ ) {
//       BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
//    }
//
//    for( size_t i=1UL; i<m_; ++i )
//       for( size_t j=0UL; j<i; ++j )
//          swap( v_[i*nn_+j], v_[j*nn_+i] );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the array.
//
// \return Reference to the transposed array.
// \exception std::logic_error Impossible transpose operation.
//
// In case the array is not a square array, a \a std::logic_error exception is thrown.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomArray<N,Type,AF,PF,RT>& CustomArray<N,Type,AF,PF,RT>::ctranspose()
{
//    if( o_ != m_ || m_ != n_ ) {
//       BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
//    }
//
//    for( size_t i=0UL; i<m_; ++i ) {
//       for( size_t j=0UL; j<i; ++j ) {
//          cswap( v_[i*nn_+j], v_[j*nn_+i] );
//       }
//       conjugate( v_[i*nn_+i] );
//    }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the array.
//
// \return Reference to the transposed array.
// \exception std::logic_error Impossible transpose operation.
//
// In case the array is not a square array, a \a std::logic_error exception is thrown.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename T >   // Type of the mapping indices
inline CustomArray<N,Type,AF,PF,RT>& CustomArray<N,Type,AF,PF,RT>::ctranspose( const T* indices, size_t n )
{
//    if( o_ != m_ || m_ != n_ ) {
//       BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
//    }
//
//    for( size_t i=0UL; i<m_; ++i ) {
//       for( size_t j=0UL; j<i; ++j ) {
//          cswap( v_[i*nn_+j], v_[j*nn_+i] );
//       }
//       conjugate( v_[i*nn_+i] );
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
   using blaze::CustomVector;
   using blaze::unaliged;
   using blaze::unpadded;

   CustomArray<int,unaligned,unpadded> A( ... );

   A *= 4;        // Scaling of the array
   A.scale( 4 );  // Same effect as above
   \endcode
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename Other >  // Data type of the scalar value
inline CustomArray<N,Type,AF,PF,RT>& CustomArray<N,Type,AF,PF,RT>::scale( const Other& scalar )
{
   ArrayForEach( dims_, nn_, [&]( size_t i ) { v_[i] *= scalar; } );
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  RESOURCE MANAGEMENT FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Resets the custom array and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the array.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \return void
// \exception std::invalid_argument Invalid setup of custom array.
//
// This function resets the custom array to the given array of elements of size \f$ m \times n \f$.
// The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This function is \b NOT available for padded custom matrices!
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom array referencing the array goes out of scope.
// \note The custom array does NOT take responsibility for the new array of elements!
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
inline void CustomArray<N,Type,AF,PF,RT>::reset( Type* ptr, Dims... dims )
{
   BLAZE_STATIC_ASSERT( N == sizeof...( Dims ) || N + 1 == sizeof...( Dims ) );
   BLAZE_STATIC_ASSERT( PF == unpadded || N + 1 == sizeof...( Dims ) );

   CustomArray tmp( ptr, dims... );
   swap( tmp );
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename Other >  // Data type of the foreign expression
inline bool CustomArray<N,Type,AF,PF,RT>::canAlias( const Other* alias ) const noexcept
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename Other >  // Data type of the foreign expression
inline bool CustomArray<N,Type,AF,PF,RT>::isAliased( const Other* alias ) const noexcept
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline bool CustomArray<N,Type,AF,PF,RT>::isAligned() const noexcept
{
   return ( AF || ( checkAlignment( v_ ) && dimension<0>() % SIMDSIZE == 0UL ) );
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline bool CustomArray<N,Type,AF,PF,RT>::canSMPAssign() const noexcept
{
   return ( capacity() >= SMP_DMATASSIGN_THRESHOLD );
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
BLAZE_ALWAYS_INLINE typename CustomArray<N,Type,AF,PF,RT>::SIMDType
   CustomArray<N,Type,AF,PF,RT>::load( Dims... dims ) const noexcept
{
   if( AF && PF )
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
BLAZE_ALWAYS_INLINE typename CustomArray<N,Type,AF,PF,RT>::SIMDType
   CustomArray<N,Type,AF,PF,RT>::loada( Dims... dims ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

#if defined(BLAZE_INTERNAL_ASSERTION)
   size_t indices[] = { size_t(dims)... };
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
#endif

   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !PF || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_ + index( dims... ) ), "Invalid alignment detected" );

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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
BLAZE_ALWAYS_INLINE typename CustomArray<N,Type,AF,PF,RT>::SIMDType
   CustomArray<N,Type,AF,PF,RT>::loadu( Dims... dims ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

#if defined(BLAZE_INTERNAL_ASSERTION)
   size_t indices[] = { size_t(dims)... };
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
#endif
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );

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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
BLAZE_ALWAYS_INLINE void
   CustomArray<N,Type,AF,PF,RT>::store( const SIMDType& value, Dims... dims ) noexcept
{
   if( AF && PF )
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
BLAZE_ALWAYS_INLINE void
   CustomArray<N,Type,AF,PF,RT>::storea( const SIMDType& value, Dims... dims ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

#if defined(BLAZE_INTERNAL_ASSERTION)
   size_t indices[] = { size_t(dims)... };
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
#endif
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !PF || j % SIMDSIZE == 0UL, "Invalid column access index" );
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
BLAZE_ALWAYS_INLINE void
   CustomArray<N,Type,AF,PF,RT>::storeu( const SIMDType& value, Dims... dims ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

#if defined(BLAZE_INTERNAL_ASSERTION)
   size_t indices[] = { size_t(dims)... };
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
#endif
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );

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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename... Dims >
BLAZE_ALWAYS_INLINE void
   CustomArray<N,Type,AF,PF,RT>::stream( const SIMDType& value, Dims... dims ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

#if defined(BLAZE_INTERNAL_ASSERTION)
   size_t indices[] = { size_t(dims)... };
   ArrayDimForEach( dims_, [&]( size_t i, size_t dim ) {
      BLAZE_INTERNAL_ASSERT( indices[N - i - 1] < dim, "Invalid array access index" );
   } );
#endif
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !PF || j % SIMDSIZE == 0UL, "Invalid column access index" );
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense array
inline auto CustomArray<N,Type,AF,PF,RT>::assign( const DenseArray<MT>& rhs )
   -> DisableIf_t< VectorizedAssign_v<MT> >
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense array
inline auto CustomArray<N,Type,AF,PF,RT>::assign( const DenseArray<MT>& rhs )
   -> EnableIf_t< VectorizedAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( dims_ == (~rhs).dimensions()   , "Invalid array access index"    );

//    constexpr bool remainder( !PF || !IsPadded_v<MT> );
//
//    const size_t jpos( ( remainder )?( n_ & size_t(-SIMDSIZE) ):( n_ ) );
//    BLAZE_INTERNAL_ASSERT( !remainder || ( n_ - ( n_ % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );
//
//    if( AF && PF && useStreaming &&
//        ( m_*n_*o_ > ( cacheSize / ( sizeof(Type) * 3UL ) ) ) && !(~rhs).isAliased( this ) )
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
//          for (size_t i=0UL; i<m_; ++i)
//          {
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
}
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense array
inline auto CustomArray<N,Type,AF,PF,RT>::addAssign( const DenseArray<MT>& rhs )
   -> DisableIf_t< VectorizedAddAssign_v<MT> >
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense array
inline auto CustomArray<N,Type,AF,PF,RT>::addAssign( const DenseArray<MT>& rhs )
   -> EnableIf_t< VectorizedAddAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( dims_ == (~rhs).dimensions()   , "Invalid array access index"    );

//    constexpr bool remainder( !PF || !IsPadded_v<MT> );
//
//    for (size_t k=0UL; k<o_; ++k) {
//       for (size_t i=0UL; i<m_; ++i)
//       {
//          const size_t jbegin(0UL);
//          const size_t jend  (n_);
//          BLAZE_INTERNAL_ASSERT(jbegin <= jend, "Invalid loop indices detected");
//
//          const size_t jpos((remainder)?(jend & size_t(-SIMDSIZE)):(jend));
//          BLAZE_INTERNAL_ASSERT(!remainder || (jend - (jend % (SIMDSIZE))) == jpos, "Invalid end calculation");
//
//          size_t j(jbegin);
//          Iterator left(begin(i, k) + jbegin);
//          ConstIterator_t<MT> right((~rhs).begin(i, k) + jbegin);
//
//          for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
//             left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
//             left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
//             left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
//             left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
//          }
//          for (; j<jpos; j+=SIMDSIZE) {
//             left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
//          }
//          for (; remainder && j<jend; ++j) {
//             *left += *right; ++left; ++right;
//          }
//       }
//    }
}
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense array
inline auto CustomArray<N,Type,AF,PF,RT>::subAssign( const DenseArray<MT>& rhs )
   -> DisableIf_t< VectorizedSubAssign_v<MT> >
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense array
inline auto CustomArray<N,Type,AF,PF,RT>::subAssign( const DenseArray<MT>& rhs )
   -> EnableIf_t< VectorizedSubAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( dims_ == (~rhs).dimensions()   , "Invalid array access index"    );

//    constexpr bool remainder( !PF || !IsPadded_v<MT> );
//
//    for (size_t k=0UL; k<o_; ++k) {
//       for (size_t i=0UL; i<m_; ++i)
//       {
//          const size_t jbegin(0UL);
//          const size_t jend  (n_);
//          BLAZE_INTERNAL_ASSERT(jbegin <= jend, "Invalid loop indices detected");
//
//          const size_t jpos((remainder)?(jend & size_t(-SIMDSIZE)):(jend));
//          BLAZE_INTERNAL_ASSERT(!remainder || (jend - (jend % (SIMDSIZE))) == jpos, "Invalid end calculation");
//
//          size_t j(jbegin);
//          Iterator left(begin(i, k) + jbegin);
//          ConstIterator_t<MT> right((~rhs).begin(i, k) + jbegin);
//
//          for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
//             left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
//             left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
//             left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
//             left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
//          }
//          for (; j<jpos; j+=SIMDSIZE) {
//             left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
//          }
//          for (; remainder && j<jend; ++j) {
//             *left -= *right; ++left; ++right;
//          }
//       }
//    }
}
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense array
inline auto CustomArray<N,Type,AF,PF,RT>::schurAssign( const DenseArray<MT>& rhs )
   -> DisableIf_t< VectorizedSchurAssign_v<MT> >
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
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense array
inline auto CustomArray<N,Type,AF,PF,RT>::schurAssign( const DenseArray<MT>& rhs )
   -> EnableIf_t< VectorizedSchurAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( dims_ == (~rhs).dimensions()   , "Invalid array access index"    );

//    constexpr bool remainder( !PF || !IsPadded_v<MT> );
//
//    for (size_t k=0UL; k<o_; ++k) {
//       for (size_t i=0UL; i<m_; ++i)
//       {
//          const size_t jpos((remainder)?(n_ & size_t(-SIMDSIZE)):(n_));
//          BLAZE_INTERNAL_ASSERT(!remainder || (n_ - (n_ % (SIMDSIZE))) == jpos, "Invalid end calculation");
//
//          size_t j(0UL);
//          Iterator left(begin(i, k));
//          ConstIterator_t<MT> right((~rhs).begin(i, k));
//
//          for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
//             left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
//             left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
//             left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
//             left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
//          }
//          for (; j<jpos; j+=SIMDSIZE) {
//             left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
//          }
//          for (; remainder && j<n_; ++j) {
//             *left *= *right; ++left; ++right;
//          }
//       }
//    }
}
//*************************************************************************************************







//=================================================================================================
//
//  CUSTOMTENSOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name CustomArray operators */
//@{
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void reset( CustomArray<N,Type,AF,PF,RT>& m );

template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT    // Result type
        , typename... Dims >   // indices of row
inline void reset( CustomArray<N,Type,AF,PF,RT>& m, size_t i, Dims... dims );

template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void clear( CustomArray<N,Type,AF,PF,RT>& m );

template< bool RF        // Relaxation flag
        , size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline bool isDefault( const CustomArray<N,Type,AF,PF,RT>& m );

template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline bool isIntact( const CustomArray<N,Type,AF,PF,RT>& m );

template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void swap( CustomArray<N,Type,AF,PF,RT>& a, CustomArray<N,Type,AF,PF,RT>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given custom array.
// \ingroup custom_array
//
// \param m The array to reset.
// \return void
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void reset( CustomArray<N,Type,AF,PF,RT>& m )
{
   m.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset the specified row/column of the given custom array.
// \ingroup custom_array
//
// \param m The array to reset.
// \param i The index of the row/column to reset.
// \param k The index of the page to reset.
// \return void
//
// This function resets the values in the specified row/column of the given custom array to
// their default value. In case the given array is a \a rowMajor array the function resets the
// values in row \a i, if it is a \a columnMajor array the function resets the values in column
// \a i. Note that the capacity of the row/column remains unchanged.
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT    // Result type
        , typename... Dims >   // indices of row
inline void reset( CustomArray<N,Type,AF,PF,RT>& m, size_t i, Dims... dims )
{
   m.reset( i, dims... );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given custom array.
// \ingroup custom_array
//
// \param m The array to be cleared.
// \return void
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void clear( CustomArray<N,Type,AF,PF,RT>& m )
{
   m.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given custom array is in default state.
// \ingroup custom_array
//
// \param m The array to be tested for its default state.
// \return \a true in case the given array's rows and columns are zero, \a false otherwise.
//
// This function checks whether the custom array is in default (constructed) state, i.e. if
// it's number of rows and columns is 0. In case it is in default state, the function returns
// \a true, else it will return \a false. The following example demonstrates the use of the
// \a isDefault() function:

   \code
   using blaze::aligned;
   using blaze::padded;

   blaze::CustomArray<int,aligned,padded> A( ... );
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
        , size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline bool isDefault( const CustomArray<N,Type,AF,PF,RT>& m )
{
   return ArrayDimAllOf(
      m.dimensions(), []( size_t, size_t dim ) { return dim == 0UL; } );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given custom array are intact.
// \ingroup custom_array
//
// \param m The custom array to be tested.
// \return \a true in case the given array's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the custom array are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   using blaze::aligned;
   using blaze::padded;

   blaze::CustomArray<int,aligned,padded> A( ... );
   // ... Resizing and initialization
   if( isIntact( A ) ) { ... }
   \endcode
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline bool isIntact( const CustomArray<N,Type,AF,PF,RT>& m )
{
   size_t capacity = m.spacing();
   for( size_t i = 1; i < N; ++i ) {
      capacity *= m.dimensions()[i];
   }
   return ( capacity <= m.capacity() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two custom matrices.
// \ingroup custom_array
//
// \param a The first array to be swapped.
// \param b The second array to be swapped.
// \return void
*/
template< size_t N       // Dimensionality of the array
        , typename Type  // Data type of the array
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void swap( CustomArray<N,Type,AF,PF,RT>& a, CustomArray<N,Type,AF,PF,RT>& b ) noexcept
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
template< size_t N, typename T, bool AF, bool PF, typename RT >
struct HasConstDataAccess< CustomArray<N,T,AF,PF,RT> >
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
template< size_t N, typename T, bool AF, bool PF, typename RT >
struct HasMutableDataAccess< CustomArray<N,T,AF,PF,RT> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISCUSTOM SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< size_t N, typename T, bool AF, bool PF, typename RT >
struct IsCustom< CustomArray<N,T,AF,PF,RT> >
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
template< size_t N, typename T, bool PF, typename RT >
struct IsAligned< CustomArray<N,T,aligned,PF,RT> >
   : public TrueType
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
template< size_t N, typename T, bool AF, bool PF, typename RT >
struct IsContiguous< CustomArray<N,T,AF,PF,RT> >
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
template< size_t N, typename T, bool AF, typename RT >
struct IsPadded< CustomArray<N,T,AF,padded,RT> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
