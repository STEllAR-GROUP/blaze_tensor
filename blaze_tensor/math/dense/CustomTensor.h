//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/CustomTensor.h
//  \brief Header file for the implementation of a customizable tensor
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_CUSTOMTENSOR_H_
#define _BLAZE_TENSOR_MATH_DENSE_CUSTOMTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <utility>
#include <blaze/math/dense/CustomMatrix.h>

#include <blaze_tensor/math/Forward.h>
#include <blaze_tensor/math/InitializerList.h>
#include <blaze_tensor/math/Tensor.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>
#include <blaze_tensor/math/SMP.h>
#include <blaze_tensor/math/typetraits/IsDenseTensor.h>
#include <blaze_tensor/math/typetraits/IsTensor.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup custom_tensor CustomTensor
// \ingroup dense_tensor
*/
/*!\brief Efficient implementation of a customizable tensor.
// \ingroup custom_tensor
//
// The CustomTensor class template provides the functionality to represent an external array of
// elements of arbitrary type and a fixed size as a native \b Blaze dense tensor data structure.
// Thus in contrast to all other dense tensor types a custom tensor does not perform any kind
// of memory allocation by itself, but it is provided with an existing array of element during
// construction. A custom tensor can therefore be considered an alias to the existing array.
//
// The type of the elements, the properties of the given array of elements and the storage order
// of the tensor can be specified via the following four template parameters:

   \code
   template< typename Type, bool AF, bool PF >
   class CustomTensor;
   \endcode

//  - Type: specifies the type of the tensor elements. CustomTensor can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//  - AF  : specifies whether the represented, external arrays are properly aligned with
//          respect to the available instruction set (SSE, AVX, ...) or not.
//  - PF  : specified whether the represented, external arrays are properly padded with
//          respect to the available instruction set (SSE, AVX, ...) or not.
//
// The following examples give an impression of several possible types of custom matrices:

   \code
   using blaze::CustomTensor;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;

   // Definition of a custom tensor for unaligned, unpadded integer arrays
   using UnalignedUnpadded = CustomTensor<int,unaligned,unpadded>;

   // Definition of a custom column-major tensor for unaligned but padded 'float' arrays
   using UnalignedPadded = CustomTensor<float,unaligned,padded>;

   // Definition of a custom tensor for aligned, unpadded 'double' arrays
   using AlignedUnpadded = CustomTensor<double,aligned,unpadded>;

   // Definition of a custom column-major tensor for aligned, padded 'complex<double>' arrays
   using AlignedPadded = CustomTensor<complex<double>,aligned,padded>;
   \endcode

// \n \section customtensor_special_properties Special Properties of Custom Matrices
//
// In comparison with the remaining \b Blaze dense tensor types CustomTensor has several special
// characteristics. All of these result from the fact that a custom tensor is not performing any
// kind of memory allocation, but instead is given an existing array of elements. The following
// sections discuss all of these characteristics:
//
//  -# <b>\ref customtensor_memory_management</b>
//  -# <b>\ref customtensor_copy_operations</b>
//  -# <b>\ref customtensor_alignment</b>
//  -# <b>\ref customtensor_padding</b>
//
// \n \subsection customtensor_memory_management Memory Management
//
// The CustomTensor class template acts as an adaptor for an existing array of elements. As such
// it provides everything that is required to use the array just like a native \b Blaze dense
// tensor data structure. However, this flexibility comes with the price that the user of a custom
// tensor is responsible for the resource management.
//
// The following examples give an impression of several possible custom matrices:

   \code
   using blaze::CustomTensor;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;

   // Definition of a 3x4 custom tensor with unaligned, unpadded and externally
   // managed integer array. Note that the std::vector must be guaranteed to outlive the
   // custom tensor!
   std::vector<int> vec( 12UL );
   CustomTensor<int,unaligned,unpadded> A( &vec[0], 3UL, 4UL );

   // Definition of a custom 8x12 tensor for an aligned and padded integer array of
   // capacity 128 (including 8 padding elements per row). Note that the std::unique_ptr
   // must be guaranteed to outlive the custom tensor!
   std::unique_ptr<int[],Deallocate> memory( allocate<int>( 128UL ) );
   CustomTensor<int,aligned,padded> B( memory.get(), 8UL, 12UL, 16UL );
   \endcode

// \n \subsection customtensor_copy_operations Copy Operations
//
// As with all dense matrices it is possible to copy construct a custom tensor:

   \code
   using blaze::CustomTensor;
   using blaze::unaligned;
   using blaze::unpadded;

   using CustomType = CustomTensor<int,unaligned,unpadded>;

   std::vector<int> vec( 6UL, 10 );    // Vector of 6 integers of the value 10
   CustomType A( &vec[0], 2UL, 3UL );  // Represent the std::vector as Blaze dense tensor
   a[1] = 20;                          // Also modifies the std::vector

   CustomType B( a );  // Creating a copy of vector a
   b[2] = 20;          // Also affects tensor A and the std::vector
   \endcode

// It is important to note that a custom tensor acts as a reference to the specified array. Thus
// the result of the copy constructor is a new custom tensor that is referencing and representing
// the same array as the original custom tensor.
//
// In contrast to copy construction, just as with references, copy assignment does not change
// which array is referenced by the custom matrices, but modifies the values of the array:

   \code
   std::vector<int> vec2( 6UL, 4 );     // Vector of 6 integers of the value 4
   CustomType C( &vec2[0], 2UL, 3UL );  // Represent the std::vector as Blaze dense tensor

   A = C;  // Copy assignment: Set all values of tensor A and B to 4.
   \endcode

// \n \subsection customtensor_alignment Alignment
//
// In case the custom tensor is specified as \a aligned the passed array must adhere to some
// alignment restrictions based on the alignment requirements of the used data type and the
// used instruction set (SSE, AVX, ...). The restriction applies to the first element of each
// row/column: In case of a tensor the first element of each row must be properly
// aligned, in case of a column-major tensor the first element of each column must be properly
// aligned. For instance, if a tensor is used and AVX is active the first element of
// each row must be 32-bit aligned:

   \code
   using blaze::CustomTensor;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::padded;
   using blaze::rowMajor;

   // Allocation of 32-bit aligned memory
   std::unique_ptr<int[],Deallocate> memory( allocate<int>( 40UL ) );

   CustomTensor<int,aligned,padded> A( memory.get(), 5UL, 6UL, 8UL );
   \endcode

// In the example, the tensor has six columns. However, since with AVX eight integer
// values are loaded together the tensor is padded with two additional elements. This guarantees
// that the first element of each row is 32-bit aligned. In case the alignment requirements are
// violated, a \a std::invalid_argument exception is thrown.
//
// \n \subsection customtensor_padding Padding
//
// Adding padding elements to the end of an array can have a significant impact on performance.
// For instance, assuming that AVX is available, then two aligned, padded, 3x3 double precision
// matrices can be added via three SIMD addition operations:

   \code
   using blaze::CustomTensor;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::padded;

   using CustomType = CustomTensor<double,aligned,padded>;

   std::unique_ptr<int[],Deallocate> memory1( allocate<double>( 12UL ) );
   std::unique_ptr<int[],Deallocate> memory2( allocate<double>( 12UL ) );
   std::unique_ptr<int[],Deallocate> memory3( allocate<double>( 12UL ) );

   // Creating padded custom 3x3 tensor with an additional padding element in each row
   CustomType A( memory1.get(), 3UL, 3UL, 4UL );
   CustomType B( memory2.get(), 3UL, 3UL, 4UL );
   CustomType C( memory3.get(), 3UL, 3UL, 4UL );

   // ... Initialization

   C = A + B;  // AVX-based tensor addition
   \endcode

// In this example, maximum performance is possible. However, in case no padding elements are
// inserted a scalar addition has to be used:

   \code
   using blaze::CustomTensor;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::unpadded;

   using CustomType = CustomTensor<double,aligned,unpadded>;

   std::unique_ptr<int[],Deallocate> memory1( allocate<double>( 9UL ) );
   std::unique_ptr<int[],Deallocate> memory2( allocate<double>( 9UL ) );
   std::unique_ptr<int[],Deallocate> memory3( allocate<double>( 9UL ) );

   // Creating unpadded custom 3x3 tensor
   CustomType A( memory1.get(), 3UL, 3UL );
   CustomType B( memory2.get(), 3UL, 3UL );
   CustomType C( memory3.get(), 3UL, 3UL );

   // ... Initialization

   C = A + B;  // Scalar tensor addition
   \endcode

// Note that the construction of padded and unpadded aligned matrices looks identical. However,
// in case of padded matrices, \b Blaze will zero initialize the padding element and use them
// in all computations in order to achieve maximum performance. In case of an unpadded tensor
// \b Blaze will ignore the elements with the downside that it is not possible to load a complete
// row to an AVX register, which makes it necessary to fall back to a scalar addition.
//
// The number of padding elements is required to be sufficient with respect to the available
// instruction set: In case of an aligned padded custom tensor the added padding elements must
// guarantee that the total number of elements in each row/column is a multiple of the SIMD
// vector width. In case of an unaligned padded tensor the number of padding elements can be
// greater or equal the number of padding elements of an aligned padded custom tensor. In case
// the padding is insufficient with respect to the available instruction set, a
// \a std::invalid_argument exception is thrown.
//
//
// \n \section customtensor_arithmetic_operations Arithmetic Operations
//
// The use of custom matrices in arithmetic operations is designed to be as natural and intuitive
// as possible. All operations (addition, subtraction, multiplication, scaling, ...) can be
// expressed similar to a text book representation. Also, custom matrices can be combined with all
// other dense and sparse vectors and matrices. The following example gives an impression of the
// use of CustomTensor:

   \code
   using blaze::CustomTensor;
   using blaze::CompressedTensor;
   using blaze::Deallocate;
   using blaze::allocate;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;
   using blaze::rowMajor;
   using blaze::columnMajor;

   // Non-initialized custom 2x3 tensor. All given arrays are considered to be
   // unaligned and unpadded. The memory is managed via a 'std::vector'.
   std::vector<double> memory1( 6UL );
   CustomTensor<double,unaligned,unpadded> A( memory1.data(), 2UL, 3UL );

   A(0,0) = 1.0; A(0,1) = 2.0; A(0,2) = 3.0;  // Initialization of the first row
   A(1,0) = 4.0; A(1,1) = 5.0; A(1,2) = 6.0;  // Initialization of the second row

   // Non-initialized custom 2x3 tensor with padding elements. All given arrays are
   // required to be properly aligned and padded. The memory is managed via a 'std::unique_ptr'.
   std::unique_ptr<double[],Deallocate> memory2( allocate<double>( 16UL ) );
   CustomTensor<double,aligned,padded> B( memory2.get(), 2UL, 3UL, 8UL );

   B(0,0) = 1.0; B(0,1) = 3.0; B(0,2) = 5.0;    // Initialization of the first row
   B(1,0) = 2.0; B(1,1) = 4.0; B(1,2) = 6.0;    // Initialization of the second row

   CompressedTensor<float> C( 2, 3 );        // Empty sparse single precision tensor
   DynamicTensor<float>    D( 3, 2, 4.0F );  // Directly, homogeneously initialized single precision 3x2 tensor

   DynamicTensor<double>    E( A );  // Creation of a new tensor as a copy of A
   DynamicTensor<double> F;       // Creation of a default column-major tensor

   E = A + B;     // Tensor addition and assignment to a tensor
   F = A - C;     // Tensor subtraction and assignment to a column-major tensor
   F = A * D;     // Tensor multiplication between two matrices of different element types

   A *= 2.0;      // In-place scaling of tensor A
   E  = 2.0 * B;  // Scaling of tensor B
   F  = D * 2.0;  // Scaling of tensor D

   E += A - B;    // Addition assignment
   E -= A + C;    // Subtraction assignment
   F *= A * D;    // Multiplication assignment
   \endcode
*/
template< typename Type                                          // Data type of the tensor
        , bool AF                                                // Alignment flag
        , bool PF                                                // Padding flag
        , typename RT = DynamicTensor<RemoveConst_t<Type>> >     // Result type
class CustomTensor
   : public DenseTensor< CustomTensor<Type,AF,PF,RT> >
{
 public:
   //**Type definitions****************************************************************************
   using This     = CustomTensor<Type,AF,PF,RT>;  //!< Type of this CustomTensor instance.
   using BaseType = DenseTensor<This>;            //!< Base type of this CustomTensor instance.

   //! Result type for expression template evaluations.
   using ResultType = RT;

   //! Result type with opposite storage order for expression template evaluations.
   using OppositeType = OppositeType_t<RT>;

   //! Transpose type for expression template evaluations.
   using TransposeType = TransposeType_t<RT>;

   using ElementType   = Type;                      //!< Type of the tensor elements.
   using SIMDType      = SIMDTrait_t<ElementType>;  //!< SIMD type of the tensor elements.
   using ReturnType    = const Type&;               //!< Return type for expression template evaluations.
   using CompositeType = const This&;               //!< Data type for composite expression templates.

   using Reference      = Type&;        //!< Reference to a non-constant tensor value.
   using ConstReference = const Type&;  //!< Reference to a constant tensor value.
   using Pointer        = Type*;        //!< Pointer to a non-constant tensor value.
   using ConstPointer   = const Type*;  //!< Pointer to a constant tensor value.

   using Iterator      = DenseIterator<Type,AF>;        //!< Iterator over non-constant elements.
   using ConstIterator = DenseIterator<const Type,AF>;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a CustomTensor with different data/element type.
   */
   template< typename NewType >  // Data type of the other tensor
   struct Rebind {
      using RRT   = Rebind_t< RT, RemoveConst_t<NewType> >;  //!< The rebound result type.
      using Other = CustomTensor<NewType,AF,PF,RRT>;      //!< The type of the other CustomTensor.
   };
   //**********************************************************************************************

   //**Resize struct definition********************************************************************
   /*!\brief Resize mechanism to obtain a CustomTensor with different fixed dimensions.
   */
   template< size_t NewM    // Number of rows of the other tensor
           , size_t NewN    // Number of columns of the other tensor
           , size_t NewO >  // Number of columns of the other tensor
   struct Resize {
      using RRT   = Resize_t<RT,NewM,NewN,NewO>;   //!< The resized result type.
      using Other = CustomTensor<Type,AF,PF,RRT>;  //!< The type of the other CustomTensor.
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SIMD optimization.
   /*! The \a simdEnabled compilation flag indicates whether expressions the tensor is involved
       in can be optimized via SIMD operations. In case the element type of the tensor is a
       vectorizable data type, the \a simdEnabled compilation flag is set to \a true, otherwise
       it is set to \a false. */
   static constexpr bool simdEnabled = IsVectorizable_v<Type>;

   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the tensor can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   static constexpr bool smpAssignable = !IsSMPAssignable_v<Type>;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline CustomTensor();
   explicit inline CustomTensor( Type* ptr, size_t m, size_t n, size_t o );
   explicit inline CustomTensor( Type* ptr, size_t m, size_t n, size_t o, size_t nn );

   inline CustomTensor( const CustomTensor& m );
   inline CustomTensor( CustomTensor&& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~CustomTensor() = default;
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator()( size_t i, size_t j, size_t k ) noexcept;
   inline ConstReference operator()( size_t i, size_t j, size_t k ) const noexcept;
   inline Reference      at( size_t i, size_t j, size_t k );
   inline ConstReference at( size_t i, size_t j, size_t k ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Pointer        data  ( size_t i, size_t k ) noexcept;
   inline ConstPointer   data  ( size_t i, size_t k ) const noexcept;
   inline Iterator       begin ( size_t i, size_t k ) noexcept;
   inline ConstIterator  begin ( size_t i, size_t k ) const noexcept;
   inline ConstIterator  cbegin( size_t i, size_t k ) const noexcept;
   inline Iterator       end   ( size_t i, size_t k ) noexcept;
   inline ConstIterator  end   ( size_t i, size_t k ) const noexcept;
   inline ConstIterator  cend  ( size_t i, size_t k ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline CustomTensor& operator=( const Type& set );
   inline CustomTensor& operator=( initializer_list< initializer_list< initializer_list<Type> > > list );

   template< typename Other, size_t M, size_t N, size_t O >
   inline CustomTensor& operator=( const Other (&array)[O][M][N] );

   inline CustomTensor& operator=( const CustomTensor& rhs );
   inline CustomTensor& operator=( CustomTensor&& rhs ) noexcept;

   template< typename MT > inline CustomTensor& operator= ( const Tensor<MT>& rhs );
   template< typename MT > inline CustomTensor& operator+=( const Tensor<MT>& rhs );
   template< typename MT > inline CustomTensor& operator-=( const Tensor<MT>& rhs );
   template< typename MT > inline CustomTensor& operator%=( const Tensor<MT>& rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t rows() const noexcept;
   inline size_t columns() const noexcept;
   inline size_t pages() const noexcept;
   inline size_t spacing() const noexcept;
   inline size_t capacity() const noexcept;
   inline size_t capacity( size_t i, size_t k ) const noexcept;
   inline size_t nonZeros() const;
   inline size_t nonZeros( size_t i, size_t k ) const;
   inline void   reset();
   inline void   reset( size_t i, size_t k );
   inline void   clear();
   inline void   swap( CustomTensor& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
//    inline CustomTensor& transpose();
//    inline CustomTensor& ctranspose();

   template< typename Other > inline CustomTensor& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

   //**Resource management functions***************************************************************
   /*!\name Resource management functions */
   //@{
   inline void reset( Type* ptr, size_t m, size_t n, size_t k );
   inline void reset( Type* ptr, size_t m, size_t n, size_t k, size_t nn );
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

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t i, size_t j, size_t k ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t i, size_t j, size_t k ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t i, size_t j, size_t k ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t i, size_t j, size_t k, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t i, size_t j, size_t k, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t i, size_t j, size_t k, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t i, size_t j, size_t k, const SIMDType& value ) noexcept;

   template< typename MT >
   inline auto assign( const DenseTensor<MT>& rhs ) -> DisableIf_t< VectorizedAssign_v<MT> >;

   template< typename MT >
   inline auto assign( const DenseTensor<MT>& rhs ) -> EnableIf_t< VectorizedAssign_v<MT> >;

   template< typename MT >
   inline auto addAssign( const DenseTensor<MT>& rhs ) -> DisableIf_t< VectorizedAddAssign_v<MT> >;

   template< typename MT >
   inline auto addAssign( const DenseTensor<MT>& rhs ) -> EnableIf_t< VectorizedAddAssign_v<MT> >;

   template< typename MT >
   inline auto subAssign( const DenseTensor<MT>& rhs ) -> DisableIf_t< VectorizedSubAssign_v<MT> >;

   template< typename MT >
   inline auto subAssign( const DenseTensor<MT>& rhs ) -> EnableIf_t< VectorizedSubAssign_v<MT> >;

   template< typename MT >
   inline auto schurAssign( const DenseTensor<MT>& rhs ) -> DisableIf_t< VectorizedSchurAssign_v<MT> >;

   template< typename MT >
   inline auto schurAssign( const DenseTensor<MT>& rhs ) -> EnableIf_t< VectorizedSchurAssign_v<MT> >;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t m_;   //!< The current number of rows of the tensor.
   size_t n_;   //!< The current number of columns of the tensor.
   size_t o_;   //!< The current number of pages of the tensor.
   size_t nn_;  //!< The number of elements between two rows.
   Type* v_;    //!< The custom array of elements.
                /*!< Access to the tensor elements is gained via the function call
                     operator. In case of order the memory layout of the
                     elements is
                     \f[\left(\begin{array}{*{5}{c}}
                     0            & 1             & 2             & \cdots & N-1         \\
                     N            & N+1           & N+2           & \cdots & 2 \cdot N-1 \\
                     \vdots       & \vdots        & \vdots        & \ddots & \vdots      \\
                     M \cdot N-N  & M \cdot N-N+1 & M \cdot N-N+2 & \cdots & M \cdot N-1 \\
                     \end{array}\right)\f]. */
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
/*!\brief The default constructor for CustomTensor.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomTensor<Type,AF,PF,RT>::CustomTensor()
   : m_ ( 0UL )      // The current number of rows of the tensor
   , n_ ( 0UL )      // The current number of columns of the tensor
   , o_ ( 0UL )      // The current number of pages of the tensor
   , nn_( 0UL )      // The number of elements between two rows
   , v_ ( nullptr )  // The custom array of elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a tensor of size \f$ m \times n \f$.
//
// \param ptr The array of elements to be used by the tensor.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param o The number of pages of the array of elements.
// \exception std::invalid_argument Invalid setup of custom tensor.
//
// This constructor creates an unpadded custom tensor of size \f$ m \times n \f$. The construction
// fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This constructor is \b NOT available for padded custom matrices!
// \note The custom tensor does \b NOT take responsibility for the given array of elements!
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomTensor<Type,AF,PF,RT>::CustomTensor( Type* ptr, size_t m, size_t n, size_t o )
   : m_ ( m )    // The current number of rows of the tensor
   , n_ ( n )    // The current number of columns of the tensor
   , o_ ( o )    // The current number of pages of the tensor
   , nn_( n )    // The number of elements between two rows
   , v_ ( ptr )  // The custom array of elements
{
   BLAZE_STATIC_ASSERT( PF == unpadded );

   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && ( !checkAlignment( ptr ) || nn_ % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a tensor of size \f$ m \times n \f$.
//
// \param ptr The array of elements to be used by the tensor.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param o The number of pages of the array of elements.
// \param nn The total number of elements between two rows/columns.
// \exception std::invalid_argument Invalid setup of custom tensor.
//
// This constructor creates a custom tensor of size \f$ m \times n \f$. The construction fails
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
// \note The custom tensor does \b NOT take responsibility for the given array of elements!
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomTensor<Type,AF,PF,RT>::CustomTensor( Type* ptr, size_t m, size_t n, size_t o, size_t nn )
   : m_ ( m )    // The current number of rows of the tensor
   , n_ ( n )    // The current number of columns of the tensor
   , o_ ( o )    // The current number of pages of the tensor
   , nn_( nn )   // The number of elements between two rows
   , v_ ( ptr )  // The custom array of elements
{
   using blaze::clear;

   using ClearFunctor = If_t< PF || !IsConst_v<Type>, Clear, Noop >;

   if( ptr == nullptr ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array of elements" );
   }

   if( AF && ( !checkAlignment( ptr ) || nn_ % SIMDSIZE != 0UL ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid alignment detected" );
   }

   if( PF && IsVectorizable_v<Type> && ( nn_ < nextMultiple<size_t>( n_, SIMDSIZE ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Insufficient capacity for padded tensor" );
   }

   if( PF && IsVectorizable_v<Type> ) {
      ClearFunctor clear;
      for( size_t i=0UL; i<m_; ++i ) {
         for( size_t j=n_; j<nn_; ++j )
            clear( v_[i*nn_+j] );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for CustomTensor.
//
// \param m Tensor to be copied.
//
// The copy constructor initializes the custom tensor as an exact copy of the given custom tensor.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomTensor<Type,AF,PF,RT>::CustomTensor( const CustomTensor& m )
   : m_ ( m.m_ )   // The current number of rows of the tensor
   , n_ ( m.n_ )   // The current number of columns of the tensor
   , o_ ( m.o_ )   // The current number of pages of the tensor
   , nn_( m.nn_ )  // The number of elements between two rows
   , v_ ( m.v_ )   // The custom array of elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The move constructor for CustomTensor.
//
// \param m The tensor to be moved into this instance.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomTensor<Type,AF,PF,RT>::CustomTensor( CustomTensor&& m ) noexcept
   : m_ ( m.m_ )   // The current number of rows of the tensor
   , n_ ( m.n_ )   // The current number of columns of the tensor
   , o_ ( m.o_ )   // The current number of pages of the tensor
   , nn_( m.nn_ )  // The number of elements between two rows
   , v_ ( m.v_ )   // The custom array of elements
{
   m.m_  = 0UL;
   m.n_  = 0UL;
   m.o_  = 0UL;
   m.nn_ = 0UL;
   m.v_  = nullptr;

   BLAZE_INTERNAL_ASSERT( m.data() == nullptr, "Invalid data reference detected" );
}
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief 2D-access to the tensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::Reference
   CustomTensor<Type,AF,PF,RT>::operator()( size_t i, size_t j, size_t k ) noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   BLAZE_USER_ASSERT( k<o_, "Invalid page access index"   );
   return v_[(k*m_+i)*nn_+j];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief 2D-access to the tensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::ConstReference
   CustomTensor<Type,AF,PF,RT>::operator()( size_t i, size_t j, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   BLAZE_USER_ASSERT( k<o_, "Invalid page access index"   );
   return v_[(k*m_+i)*nn_+j];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the tensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid tensor access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::Reference
   CustomTensor<Type,AF,PF,RT>::at( size_t i, size_t j, size_t k )
{
   if( i >= m_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= n_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   if( k >= o_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
   }
   return (*this)(i,j,k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the tensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid tensor access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::ConstReference
   CustomTensor<Type,AF,PF,RT>::at( size_t i, size_t j, size_t k ) const
{
   if( i >= m_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= n_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   if( k >= o_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
   }
   return (*this)(i,j,k);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the tensor elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic tensor. Note that you
// can NOT assume that all tensor elements lie adjacent to each other! The dynamic tensor may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::Pointer
   CustomTensor<Type,AF,PF,RT>::data() noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the tensor elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic tensor. Note that you
// can NOT assume that all tensor elements lie adjacent to each other! The dynamic tensor may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::ConstPointer
   CustomTensor<Type,AF,PF,RT>::data() const noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the tensor elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::Pointer
   CustomTensor<Type,AF,PF,RT>::data( size_t i, size_t k ) noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return v_+(k*m_+i)*nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the tensor elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::ConstPointer
   CustomTensor<Type,AF,PF,RT>::data( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return v_+(k*m_+i)*nn_;
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
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::Iterator
   CustomTensor<Type,AF,PF,RT>::begin( size_t i, size_t k ) noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return Iterator( v_+(k*m_+i)*nn_ );
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
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::ConstIterator
   CustomTensor<Type,AF,PF,RT>::begin( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return ConstIterator( v_+(k*m_+i)*nn_ );
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
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::ConstIterator
   CustomTensor<Type,AF,PF,RT>::cbegin( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return ConstIterator( v_+(k*m_+i)*nn_ );
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
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::Iterator
   CustomTensor<Type,AF,PF,RT>::end( size_t i, size_t k ) noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return Iterator( v_+(k*m_+i)*nn_+n_ );
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
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::ConstIterator
   CustomTensor<Type,AF,PF,RT>::end( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return ConstIterator( v_+(k*m_+i)*nn_+n_ );
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
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline typename CustomTensor<Type,AF,PF,RT>::ConstIterator
   CustomTensor<Type,AF,PF,RT>::cend( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return ConstIterator( v_+(k*m_+i)*nn_+n_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Homogeneous assignment to all tensor elements.
//
// \param rhs Scalar value to be assigned to all tensor elements.
// \return Reference to the assigned tensor.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomTensor<Type,AF,PF,RT>&
   CustomTensor<Type,AF,PF,RT>::operator=( const Type& rhs )
{
   for (size_t k=0UL; k<o_; ++k) {
      for (size_t i=0UL; i<m_; ++i) {
         size_t row_elements = (k*m_+i)*nn_;
         for (size_t j=0UL; j<n_; ++j) {
            v_[row_elements+j] = rhs;
         }
      }
   }
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List assignment to all tensor elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to static tensor.
//
// This assignment operator offers the option to directly assign to all elements of the tensor
// by means of an initializer list:

   \code
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;

   const int array[9] = { 0, 0, 0,
                          0, 0, 0,
                          0, 0, 0 };
   blaze::CustomTensor<int,unaligned,unpadded> A( array, 3UL, 3UL );
   A = { { 1, 2, 3 },
         { 4, 5 },
         { 7, 8, 9 } };
   \endcode

// The tensor elements are assigned the values from the given initializer list. Missing values
// are initialized as default (as e.g. the value 6 in the example). Note that in case the size
// of the top-level initializer list exceeds the number of rows or the size of any nested list
// exceeds the number of columns, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomTensor<Type,AF,PF,RT>&
   CustomTensor<Type,AF,PF,RT>::operator=( initializer_list< initializer_list< initializer_list<Type> > > list )
{
   if( list.size() != o_ || determineColumns( list ) > n_ || determineRows( list ) > m_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to custom tensor" );
   }

   size_t k( 0UL );

   for (const auto& page : list) {
      size_t i( 0UL );
      for (const auto& rowList : page) {
         std::fill(std::copy(rowList.begin(), rowList.end(), v_+(k*m_+i)*nn_), v_+(k*m_+i)*nn_+( PF ? nn_ : n_ ), Type());
         ++i;
      }
      ++k;
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array assignment to all tensor elements.
//
// \param array \f$ M \times N \f$ dimensional array for the assignment.
// \return Reference to the assigned tensor.
// \exception std::invalid_argument Invalid array size.
//
// This assignment operator offers the option to directly set all elements of the tensor:

   \code
   using blaze::unaligned;
   using blaze::unpadded;
   using blaze::rowMajor;

   const int array[9] = { 0, 0, 0,
                          0, 0, 0,
                          0, 0, 0 };
   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::CustomTensor<int,unaligned,unpadded> A( array, 3UL, 3UL );
   A = init;
   \endcode

// The tensor is assigned the values from the given array. Missing values are initialized with
// default values (as e.g. the value 6 in the example). Note that the size of the array must
// match the size of the custom tensor. Otherwise a \a std::invalid_argument exception is thrown.
// Also note that after the assignment \a array will have the same entries as \a init.
*/
template< typename Type   // Data type of the tensor
        , bool AF         // Alignment flag
        , bool PF         // Padding flag
        , typename RT >   // Result type
template< typename Other  // Data type of the initialization array
        , size_t M        // Number of rows of the initialization array
        , size_t N        // Number of columns of the initialization array
        , size_t O >      // Number of pages of the initialization array
inline CustomTensor<Type,AF,PF,RT>&
   CustomTensor<Type,AF,PF,RT>::operator=( const Other (&array)[O][M][N] )
{
   if( m_ != M || n_ != N || o_ != O ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid array size" );
   }

   for (size_t k=0UL; k<O; ++k) {
      for (size_t i=0UL; i<M; ++i) {
         for (size_t j=0UL; j<N; ++j) {
            v_[(k*m_+i)*nn_+j] = array[k][i][j];
         }
      }
   }
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy assignment operator for CustomTensor.
//
// \param rhs Tensor to be copied.
// \return Reference to the assigned tensor.
// \exception std::invalid_argument Tensor sizes do not match.
//
// The tensor is initialized as a copy of the given tensor. In case the current sizes of the two
// matrices don't match, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomTensor<Type,AF,PF,RT>&
   CustomTensor<Type,AF,PF,RT>::operator=( const CustomTensor& rhs )
{
   if( rhs.rows() != m_ || rhs.columns() != n_ || rhs.pages() != o_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   smpAssign( *this, ~rhs );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Move assignment operator for CustomTensor.
//
// \param rhs Tensor to be copied.
// \return Reference to the assigned tensor.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline CustomTensor<Type,AF,PF,RT>&
   CustomTensor<Type,AF,PF,RT>::operator=( CustomTensor&& rhs ) noexcept
{
   m_  = rhs.m_;
   n_  = rhs.n_;
   o_  = rhs.o_;
   nn_ = rhs.nn_;
   v_  = rhs.v_;

   rhs.m_  = 0UL;
   rhs.n_  = 0UL;
   rhs.o_  = 0UL;
   rhs.nn_ = 0UL;
   rhs.v_  = nullptr;

   BLAZE_INTERNAL_ASSERT( rhs.data() == nullptr, "Invalid data reference detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different matrices.
//
// \param rhs Tensor to be copied.
// \return Reference to the assigned tensor.
// \exception std::invalid_argument Tensor sizes do not match.
//
// The tensor is initialized as a copy of the given tensor. In case the current sizes of the two
// matrices don't match, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side tensor
inline CustomTensor<Type,AF,PF,RT>&
   CustomTensor<Type,AF,PF,RT>::operator=( const Tensor<MT>& rhs )
{
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
/*!\brief Addition assignment operator for the addition of a tensor (\f$ A+=B \f$).
//
// \param rhs The right-hand side tensor to be added to the tensor.
// \return Reference to the tensor.
// \exception std::invalid_argument Tensor sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side tensor
inline CustomTensor<Type,AF,PF,RT>&
   CustomTensor<Type,AF,PF,RT>::operator+=( const Tensor<MT>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ || (~rhs).pages() != o_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
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
/*!\brief Subtraction assignment operator for the subtraction of a tensor (\f$ A-=B \f$).
//
// \param rhs The right-hand side tensor to be subtracted from the tensor.
// \return Reference to the tensor.
// \exception std::invalid_argument Tensor sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side tensor
inline CustomTensor<Type,AF,PF,RT>&
   CustomTensor<Type,AF,PF,RT>::operator-=( const Tensor<MT>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ || (~rhs).pages() != o_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
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
/*!\brief Schur product assignment operator for the multiplication of a tensor (\f$ A\circ=B \f$).
//
// \param rhs The right-hand side tensor for the Schur product.
// \return Reference to the tensor.
// \exception std::invalid_argument Tensor sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side tensor
inline CustomTensor<Type,AF,PF,RT>&
   CustomTensor<Type,AF,PF,RT>::operator%=( const Tensor<MT>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ || (~rhs).pages() != o_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
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
/*!\brief Returns the current number of rows of the tensor.
//
// \return The number of rows of the tensor.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomTensor<Type,AF,PF,RT>::rows() const noexcept
{
   return m_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of columns of the tensor.
//
// \return The number of columns of the tensor.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomTensor<Type,AF,PF,RT>::columns() const noexcept
{
   return n_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of columns of the tensor.
//
// \return The number of columns of the tensor.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomTensor<Type,AF,PF,RT>::pages() const noexcept
{
   return o_;
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
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomTensor<Type,AF,PF,RT>::spacing() const noexcept
{
   return nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the tensor.
//
// \return The capacity of the tensor.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomTensor<Type,AF,PF,RT>::capacity() const noexcept
{
   return m_ * nn_ * o_;
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
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomTensor<Type,AF,PF,RT>::capacity( size_t i, size_t k ) const noexcept
{
   UNUSED_PARAMETER( i );
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index" );
   return nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of non-zero elements in the tensor
//
// \return The number of non-zero elements in the dense tensor.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomTensor<Type,AF,PF,RT>::nonZeros() const
{
   size_t nonzeros( 0UL );

   for (size_t k=0UL; k<o_; ++k) {
      for (size_t i=0UL; i<m_; ++i) {
         size_t row_elements = (k*m_+i)*nn_;
         for (size_t j=0UL; j<n_; ++j) {
            if (!isDefault(v_[row_elements+j]))
               ++nonzeros;
         }
      }
   }
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
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline size_t CustomTensor<Type,AF,PF,RT>::nonZeros( size_t i, size_t k ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index" );

   const size_t jend( (k*m_+i)*nn_ + n_ );
   size_t nonzeros( 0UL );

   for( size_t j=(k*m_+i)*nn_; j<jend; ++j )
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
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void CustomTensor<Type,AF,PF,RT>::reset()
{
   using blaze::clear;

   for (size_t k=0UL; k<o_; ++k) {
      for (size_t i=0UL; i<m_; ++i) {
         size_t row_elements = (k*m_+i)*nn_;
         for (size_t j=0UL; j<n_; ++j) {
            clear(v_[row_elements+j]);
         }
      }
   }
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
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void CustomTensor<Type,AF,PF,RT>::reset( size_t i, size_t k )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index" );

   size_t row_elements = (k*m_+i)*nn_;

   for( size_t j=0UL; j<n_; ++j )
      clear( v_[row_elements+j] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the \f$ M \times N \f$ tensor.
//
// \return void
//
// After the clear() function, the size of the tensor is 0.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void CustomTensor<Type,AF,PF,RT>::clear()
{
   m_  = 0UL;
   n_  = 0UL;
   o_  = 0UL;
   nn_ = 0UL;
   v_  = nullptr;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two matrices.
//
// \param m The tensor to be swapped.
// \return void
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void CustomTensor<Type,AF,PF,RT>::swap( CustomTensor& m ) noexcept
{
   using std::swap;

   swap( m_ , m.m_  );
   swap( n_ , m.n_  );
   swap( o_ , m.o_  );
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
/*!\brief In-place transpose of the tensor.
//
// \return Reference to the transposed tensor.
// \exception std::logic_error Impossible transpose operation.
//
// In case the tensor is not a square tensor, a \a std::logic_error exception is thrown.
*/
// template< typename Type  // Data type of the tensor
//         , bool AF        // Alignment flag
//         , bool PF        // Padding flag
//         , typename RT >  // Result type
// inline CustomTensor<Type,AF,PF,RT>& CustomTensor<Type,AF,PF,RT>::transpose()
// {
//    using std::swap;
//
//    if( m_ != n_ ) {
//       BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
//    }
//
//    for( size_t i=1UL; i<m_; ++i )
//       for( size_t j=0UL; j<i; ++j )
//          swap( v_[i*nn_+j], v_[j*nn_+i] );
//
//    return *this;
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the tensor.
//
// \return Reference to the transposed tensor.
// \exception std::logic_error Impossible transpose operation.
//
// In case the tensor is not a square tensor, a \a std::logic_error exception is thrown.
*/
// template< typename Type  // Data type of the tensor
//         , bool AF        // Alignment flag
//         , bool PF        // Padding flag
//         , typename RT >  // Result type
// inline CustomTensor<Type,AF,PF,RT>& CustomTensor<Type,AF,PF,RT>::ctranspose()
// {
//    if( m_ != n_ ) {
//       BLAZE_THROW_LOGIC_ERROR( "Impossible transpose operation" );
//    }
//
//    for( size_t i=0UL; i<m_; ++i ) {
//       for( size_t j=0UL; j<i; ++j ) {
//          cswap( v_[i*nn_+j], v_[j*nn_+i] );
//       }
//       conjugate( v_[i*nn_+i] );
//    }
//
//    return *this;
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the tensor by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the tensor scaling.
// \return Reference to the tensor.
//
// This function scales the tensor by applying the given scalar value \a scalar to each element
// of the tensor. For built-in and \c complex data types it has the same effect as using the
// multiplication assignment operator:

   \code
   using blaze::CustomVector;
   using blaze::unaliged;
   using blaze::unpadded;

   CustomTensor<int,unaligned,unpadded> A( ... );

   A *= 4;        // Scaling of the tensor
   A.scale( 4 );  // Same effect as above
   \endcode
*/
template< typename Type     // Data type of the tensor
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , typename RT >     // Result type
template< typename Other >  // Data type of the scalar value
inline CustomTensor<Type,AF,PF,RT>& CustomTensor<Type,AF,PF,RT>::scale( const Other& scalar )
{
   for (size_t k=0UL; k<o_; ++k) {
      for (size_t i=0UL; i<m_; ++i) {
         size_t row_elements = (k*m_+i)*nn_;
         for (size_t j=0UL; j<n_; ++j) {
            v_[row_elements+j] *= scalar;
         }
      }
   }
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  RESOURCE MANAGEMENT FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Resets the custom tensor and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the tensor.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \return void
// \exception std::invalid_argument Invalid setup of custom tensor.
//
// This function resets the custom tensor to the given array of elements of size \f$ m \times n \f$.
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
//       when the last custom tensor referencing the array goes out of scope.
// \note The custom tensor does NOT take responsibility for the new array of elements!
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void CustomTensor<Type,AF,PF,RT>::reset( Type* ptr, size_t m, size_t n, size_t o )
{
   BLAZE_STATIC_ASSERT( PF == unpadded );

   CustomTensor tmp( ptr, m, n, o );
   swap( tmp );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resets the custom tensor and replaces the array of elements with the given array.
//
// \param ptr The array of elements to be used by the tensor.
// \param m The number of rows of the array of elements.
// \param n The number of columns of the array of elements.
// \param nn The total number of elements between two rows/columns.
// \return void
// \exception std::invalid_argument Invalid setup of custom tensor.
//
// This function resets the custom tensor to the given array of elements of size \f$ m \times n \f$.
// The function fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...).
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note In case a deleter was specified, the previously referenced array will only be destroyed
//       when the last custom tensor referencing the array goes out of scope.
// \note The custom tensor does NOT take responsibility for the new array of elements!
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void CustomTensor<Type,AF,PF,RT>::reset( Type* ptr, size_t m, size_t n, size_t o, size_t nn )
{
   CustomTensor tmp( ptr, m, n, o, nn );
   swap( tmp );
}
//*************************************************************************************************




//=================================================================================================
//
//  EXPRESSION TEMPLATE EVALUATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the tensor can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this tensor, \a false if not.
//
// This function returns whether the given address can alias with the tensor. In contrast
// to the isAliased() function this function is allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type     // Data type of the tensor
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , typename RT >     // Result type
template< typename Other >  // Data type of the foreign expression
inline bool CustomTensor<Type,AF,PF,RT>::canAlias( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the tensor is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this tensor, \a false if not.
//
// This function returns whether the given address is aliased with the tensor. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type     // Data type of the tensor
        , bool AF           // Alignment flag
        , bool PF           // Padding flag
        , typename RT >     // Result type
template< typename Other >  // Data type of the foreign expression
inline bool CustomTensor<Type,AF,PF,RT>::isAliased( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the tensor is properly aligned in memory.
//
// \return \a true in case the tensor is aligned, \a false if not.
//
// This function returns whether the tensor is guaranteed to be properly aligned in memory, i.e.
// whether the beginning and the end of each row/column of the tensor are guaranteed to conform
// to the alignment restrictions of the element type \a Type.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline bool CustomTensor<Type,AF,PF,RT>::isAligned() const noexcept
{
   return ( AF || ( checkAlignment( v_ ) && columns() % SIMDSIZE == 0UL ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the tensor can be used in SMP assignments.
//
// \return \a true in case the tensor can be used in SMP assignments, \a false if not.
//
// This function returns whether the tensor can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the tensor).
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline bool CustomTensor<Type,AF,PF,RT>::canSMPAssign() const noexcept
{
   return ( rows() * columns() * pages() >= SMP_DMATASSIGN_THRESHOLD );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Load of a SIMD element of the tensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the dense tensor. The row index
// must be smaller than the number of rows and the column index must be smaller then the number
// of columns. Additionally, the column index (in case of a tensor) or the row index
// (in case of a column-major tensor) must be a multiple of the number of values inside the
// SIMD element. This function must \b NOT be called explicitly! It is used internally for the
// performance optimized evaluation of expression templates. Calling this function explicitly
// might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
BLAZE_ALWAYS_INLINE typename CustomTensor<Type,AF,PF,RT>::SIMDType
   CustomTensor<Type,AF,PF,RT>::load( size_t i, size_t j, size_t k ) const noexcept
{
   if( AF && PF )
      return loada( i, j, k );
   else
      return loadu( i, j, k );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned load of a SIMD element of the tensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the dense tensor.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
BLAZE_ALWAYS_INLINE typename CustomTensor<Type,AF,PF,RT>::SIMDType
   CustomTensor<Type,AF,PF,RT>::loada( size_t i, size_t j, size_t k ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < o_, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !PF || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_+(k*m_+i)*nn_+j ), "Invalid alignment detected" );

   return loada( v_+(k*m_+i)*nn_+j );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unaligned load of a SIMD element of the tensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the dense tensor.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
BLAZE_ALWAYS_INLINE typename CustomTensor<Type,AF,PF,RT>::SIMDType
   CustomTensor<Type,AF,PF,RT>::loadu( size_t i, size_t j, size_t k ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < o_, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );

   return loadu( v_+(k*m_+i)*nn_+j );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Store of a SIMD element of the tensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store of a specific SIMD element of the dense tensor. The row index
// must be smaller than the number of rows and the column index must be smaller than the number
// of columns. Additionally, the column index (in case of a tensor) or the row index
// (in case of a column-major tensor) must be a multiple of the number of values inside the
// SIMD element. This function must \b NOT be called explicitly! It is used internally for the
// performance optimized evaluation of expression templates. Calling this function explicitly
// might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
BLAZE_ALWAYS_INLINE void
   CustomTensor<Type,AF,PF,RT>::store( size_t i, size_t j, size_t k, const SIMDType& value ) noexcept
{
   if( AF && PF )
      storea( i, j, k, value );
   else
      storeu( i, j, k, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned store of a SIMD element of the tensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store of a specific SIMD element of the dense tensor.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
BLAZE_ALWAYS_INLINE void
   CustomTensor<Type,AF,PF,RT>::storea( size_t i, size_t j, size_t k, const SIMDType& value ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < o_, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !PF || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_+(k*m_+i)*nn_+j ), "Invalid alignment detected" );

   storea( v_+(k*m_+i)*nn_+j, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unaligned store of a SIMD element of the tensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store of a specific SIMD element of the dense tensor.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
BLAZE_ALWAYS_INLINE void
   CustomTensor<Type,AF,PF,RT>::storeu( size_t i, size_t j, size_t k, const SIMDType& value ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < o_, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );

   storeu( v_+(k*m_+i)*nn_+j, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned, non-temporal store of a SIMD element of the tensor.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store of a specific SIMD element of the
// dense tensor. The row index must be smaller than the number of rows and the column index
// must be smaller than the number of columns. Additionally, the column index (in case of a
// tensor) or the row index (in case of a column-major tensor) must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
BLAZE_ALWAYS_INLINE void
   CustomTensor<Type,AF,PF,RT>::stream( size_t i, size_t j, size_t k, const SIMDType& value ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < o_, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= ( PF ? nn_ : n_ ), "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !PF || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( v_+(k*m_+i)*nn_+j ), "Invalid alignment detected" );

   stream( v_+(k*m_+i)*nn_+j, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a dense tensor.
//
// \param rhs The right-hand side dense tensor to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense tensor
inline auto CustomTensor<Type,AF,PF,RT>::assign( const DenseTensor<MT>& rhs )
   -> DisableIf_t< VectorizedAssign_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( o_ == (~rhs).pages(),   "Invalid number of pages" );

   const size_t jpos( n_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % 2UL ) ) == jpos, "Invalid end calculation" );

   for (size_t k=0UL; k<o_; ++k) {
      for (size_t i=0UL; i<m_; ++i) {
         size_t row_elements = (k*m_+i)*nn_;
         for (size_t j=0UL; j<jpos; j+=2UL) {
            v_[row_elements+j] = (~rhs)(i, j, k);
            v_[row_elements+j+1UL] = (~rhs)(i, j+1UL, k);
         }
         if (jpos < n_) {
            v_[row_elements+jpos] = (~rhs)(i, jpos, k);
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SIMD optimized implementation of the assignment of a dense tensor.
//
// \param rhs The right-hand side dense tensor to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense tensor
inline auto CustomTensor<Type,AF,PF,RT>::assign( const DenseTensor<MT>& rhs )
   -> EnableIf_t< VectorizedAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( o_ == (~rhs).pages(),   "Invalid number of pages" );

   constexpr bool remainder( !PF || !IsPadded_v<MT> );

   const size_t jpos( ( remainder )?( n_ & size_t(-SIMDSIZE) ):( n_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( n_ - ( n_ % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

   if( AF && PF && useStreaming &&
       ( m_*n_*o_ > ( cacheSize / ( sizeof(Type) * 3UL ) ) ) && !(~rhs).isAliased( this ) )
   {
      for (size_t k=0UL; k<o_; ++k) {
         for (size_t i=0UL; i<m_; ++i) {
            size_t j(0UL);
            Iterator left(begin(i, k));
            ConstIterator_t<MT> right((~rhs).begin(i, k));

            for (; j<jpos; j+=SIMDSIZE, left+=SIMDSIZE, right+=SIMDSIZE) {
               left.stream(right.load());
            }
            for (; remainder && j<n_; ++j, ++left, ++right) {
               *left = *right;
            }
         }
      }
   }
   else
   {
      for (size_t k=0UL; k<o_; ++k) {
         for (size_t i=0UL; i<m_; ++i)
         {
            size_t j(0UL);
            Iterator left(begin(i, k));
            ConstIterator_t<MT> right((~rhs).begin(i, k));

            for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
               left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
               left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
               left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
               left.store(right.load()); left += SIMDSIZE; right += SIMDSIZE;
            }
            for (; j<jpos; j+=SIMDSIZE) {
               left.store(right.load()); left+=SIMDSIZE, right+=SIMDSIZE;
            }
            for (; remainder && j<n_; ++j) {
               *left = *right; ++left; ++right;
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a dense tensor.
//
// \param rhs The right-hand side dense tensor to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense tensor
inline auto CustomTensor<Type,AF,PF,RT>::addAssign( const DenseTensor<MT>& rhs )
   -> DisableIf_t< VectorizedAddAssign_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( o_ == (~rhs).pages(),   "Invalid number of pages" );

   for (size_t k=0UL; k<o_; ++k) {
      for (size_t i=0UL; i<m_; ++i) {
         size_t row_elements = (k*m_+i)*nn_;
         const size_t jbegin(0UL);
         const size_t jend  (n_);
         BLAZE_INTERNAL_ASSERT(jbegin <= jend, "Invalid loop indices detected");

         size_t j(jbegin);

         for (; (j+2UL) <= jend; j+=2UL) {
            v_[row_elements+j] += (~rhs)(i, j, k);
            v_[row_elements+j+1UL] += (~rhs)(i, j+1UL, k);
         }
         if (j < jend) {
            v_[row_elements+j] += (~rhs)(i, j, k);
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SIMD optimized implementation of the addition assignment of a dense tensor.
//
// \param rhs The right-hand side dense tensor to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense tensor
inline auto CustomTensor<Type,AF,PF,RT>::addAssign( const DenseTensor<MT>& rhs )
   -> EnableIf_t< VectorizedAddAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( o_ == (~rhs).pages(),   "Invalid number of pages" );

   constexpr bool remainder( !PF || !IsPadded_v<MT> );

   for (size_t k=0UL; k<o_; ++k) {
      for (size_t i=0UL; i<m_; ++i)
      {
         const size_t jbegin(0UL);
         const size_t jend  (n_);
         BLAZE_INTERNAL_ASSERT(jbegin <= jend, "Invalid loop indices detected");

         const size_t jpos((remainder)?(jend & size_t(-SIMDSIZE)):(jend));
         BLAZE_INTERNAL_ASSERT(!remainder || (jend - (jend % (SIMDSIZE))) == jpos, "Invalid end calculation");

         size_t j(jbegin);
         Iterator left(begin(i, k) + jbegin);
         ConstIterator_t<MT> right((~rhs).begin(i, k) + jbegin);

         for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
         }
         for (; j<jpos; j+=SIMDSIZE) {
            left.store(left.load() + right.load()); left += SIMDSIZE; right += SIMDSIZE;
         }
         for (; remainder && j<jend; ++j) {
            *left += *right; ++left; ++right;
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a dense tensor.
//
// \param rhs The right-hand side dense tensor to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense tensor
inline auto CustomTensor<Type,AF,PF,RT>::subAssign( const DenseTensor<MT>& rhs )
   -> DisableIf_t< VectorizedSubAssign_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( o_ == (~rhs).pages(),   "Invalid number of pages" );

   for (size_t k=0UL; k<o_; ++k) {
      for (size_t i=0UL; i<m_; ++i) {
         size_t row_elements = (k*m_+i)*nn_;
         const size_t jbegin(0UL);
         const size_t jend  (n_);
         BLAZE_INTERNAL_ASSERT(jbegin <= jend, "Invalid loop indices detected");

         size_t j(jbegin);

         for (; (j+2UL) <= jend; j+=2UL) {
            v_[row_elements+j] -= (~rhs)(i, j, k);
            v_[row_elements+j+1UL] -= (~rhs)(i, j+1UL, k);
         }
         if (j < jend) {
            v_[row_elements+j] -= (~rhs)(i, j, k);
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SIMD optimized implementation of the subtraction assignment of a dense tensor.
//
// \param rhs The right-hand side dense tensor to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense tensor
inline auto CustomTensor<Type,AF,PF,RT>::subAssign( const DenseTensor<MT>& rhs )
   -> EnableIf_t< VectorizedSubAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( o_ == (~rhs).pages(),   "Invalid number of pages" );

   constexpr bool remainder( !PF || !IsPadded_v<MT> );

   for (size_t k=0UL; k<o_; ++k) {
      for (size_t i=0UL; i<m_; ++i)
      {
         const size_t jbegin(0UL);
         const size_t jend  (n_);
         BLAZE_INTERNAL_ASSERT(jbegin <= jend, "Invalid loop indices detected");

         const size_t jpos((remainder)?(jend & size_t(-SIMDSIZE)):(jend));
         BLAZE_INTERNAL_ASSERT(!remainder || (jend - (jend % (SIMDSIZE))) == jpos, "Invalid end calculation");

         size_t j(jbegin);
         Iterator left(begin(i, k) + jbegin);
         ConstIterator_t<MT> right((~rhs).begin(i, k) + jbegin);

         for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
         }
         for (; j<jpos; j+=SIMDSIZE) {
            left.store(left.load() - right.load()); left += SIMDSIZE; right += SIMDSIZE;
         }
         for (; remainder && j<jend; ++j) {
            *left -= *right; ++left; ++right;
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the Schur product assignment of a dense tensor.
//
// \param rhs The right-hand side dense tensor for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense tensor
inline auto CustomTensor<Type,AF,PF,RT>::schurAssign( const DenseTensor<MT>& rhs )
   -> DisableIf_t< VectorizedSchurAssign_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( o_ == (~rhs).pages(),   "Invalid number of pages" );

   const size_t jpos( n_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % 2UL ) ) == jpos, "Invalid end calculation" );

   for (size_t k=0UL; k<o_; ++k) {
      for (size_t i=0UL; i<m_; ++i) {
         size_t row_elements = (k*m_+i)*nn_;
         for (size_t j=0UL; j<jpos; j+=2UL) {
            v_[row_elements+j] *= (~rhs)(i, j, k);
            v_[row_elements+j+1UL] *= (~rhs)(i, j+1UL, k);
         }
         if (jpos < n_) {
            v_[row_elements+jpos] *= (~rhs)(i, jpos, k);
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SIMD optimized implementation of the Schur product assignment of a dense tensor.
//
// \param rhs The right-hand side dense tensor for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
template< typename MT >  // Type of the right-hand side dense tensor
inline auto CustomTensor<Type,AF,PF,RT>::schurAssign( const DenseTensor<MT>& rhs )
   -> EnableIf_t< VectorizedSchurAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( o_ == (~rhs).pages(),   "Invalid number of pages" );

   constexpr bool remainder( !PF || !IsPadded_v<MT> );

   for (size_t k=0UL; k<o_; ++k) {
      for (size_t i=0UL; i<m_; ++i)
      {
         const size_t jpos((remainder)?(n_ & size_t(-SIMDSIZE)):(n_));
         BLAZE_INTERNAL_ASSERT(!remainder || (n_ - (n_ % (SIMDSIZE))) == jpos, "Invalid end calculation");

         size_t j(0UL);
         Iterator left(begin(i, k));
         ConstIterator_t<MT> right((~rhs).begin(i, k));

         for (; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL) {
            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
         }
         for (; j<jpos; j+=SIMDSIZE) {
            left.store(left.load() * right.load()); left += SIMDSIZE; right += SIMDSIZE;
         }
         for (; remainder && j<n_; ++j) {
            *left *= *right; ++left; ++right;
         }
      }
   }
}
//*************************************************************************************************







//=================================================================================================
//
//  CUSTOMTENSOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name CustomTensor operators */
//@{
template< typename Type, bool AF, bool PF, typename RT >
inline void reset( CustomTensor<Type,AF,PF,RT>& m );

template< typename Type, bool AF, bool PF, typename RT >
inline void reset( CustomTensor<Type,AF,PF,RT>& m, size_t i, size_t k );

template< typename Type, bool AF, bool PF, typename RT >
inline void clear( CustomTensor<Type,AF,PF,RT>& m );

template< bool RF, typename Type, bool AF, bool PF, typename RT >
inline bool isDefault( const CustomTensor<Type,AF,PF,RT>& m );

template< typename Type, bool AF, bool PF, typename RT >
inline bool isIntact( const CustomTensor<Type,AF,PF,RT>& m );

template< typename Type, bool AF, bool PF, typename RT >
inline void swap( CustomTensor<Type,AF,PF,RT>& a, CustomTensor<Type,AF,PF,RT>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given custom tensor.
// \ingroup custom_tensor
//
// \param m The tensor to reset.
// \return void
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void reset( CustomTensor<Type,AF,PF,RT>& m )
{
   m.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset the specified row/column of the given custom tensor.
// \ingroup custom_tensor
//
// \param m The tensor to reset.
// \param i The index of the row/column to reset.
// \param k The index of the page to reset.
// \return void
//
// This function resets the values in the specified row/column of the given custom tensor to
// their default value. In case the given tensor is a \a rowMajor tensor the function resets the
// values in row \a i, if it is a \a columnMajor tensor the function resets the values in column
// \a i. Note that the capacity of the row/column remains unchanged.
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void reset( CustomTensor<Type,AF,PF,RT>& m, size_t i, size_t k )
{
   m.reset( i, k );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given custom tensor.
// \ingroup custom_tensor
//
// \param m The tensor to be cleared.
// \return void
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void clear( CustomTensor<Type,AF,PF,RT>& m )
{
   m.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given custom tensor is in default state.
// \ingroup custom_tensor
//
// \param m The tensor to be tested for its default state.
// \return \a true in case the given tensor's rows and columns are zero, \a false otherwise.
//
// This function checks whether the custom tensor is in default (constructed) state, i.e. if
// it's number of rows and columns is 0. In case it is in default state, the function returns
// \a true, else it will return \a false. The following example demonstrates the use of the
// \a isDefault() function:

   \code
   using blaze::aligned;
   using blaze::padded;

   blaze::CustomTensor<int,aligned,padded> A( ... );
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
        , typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline bool isDefault( const CustomTensor<Type,AF,PF,RT>& m )
{
   return ( m.rows() == 0UL && m.columns() == 0UL && m.pages() == 0UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given custom tensor are intact.
// \ingroup custom_tensor
//
// \param m The custom tensor to be tested.
// \return \a true in case the given tensor's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the custom tensor are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   using blaze::aligned;
   using blaze::padded;

   blaze::CustomTensor<int,aligned,padded> A( ... );
   // ... Resizing and initialization
   if( isIntact( A ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline bool isIntact( const CustomTensor<Type,AF,PF,RT>& m )
{
   return ( m.rows() * m.columns() * m.pages() <= m.capacity() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two custom matrices.
// \ingroup custom_tensor
//
// \param a The first tensor to be swapped.
// \param b The second tensor to be swapped.
// \return void
*/
template< typename Type  // Data type of the tensor
        , bool AF        // Alignment flag
        , bool PF        // Padding flag
        , typename RT >  // Result type
inline void swap( CustomTensor<Type,AF,PF,RT>& a, CustomTensor<Type,AF,PF,RT>& b ) noexcept
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
template< typename T, bool AF, bool PF, typename RT >
struct HasConstDataAccess< CustomTensor<T,AF,PF,RT> >
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
template< typename T, bool AF, bool PF, typename RT >
struct HasMutableDataAccess< CustomTensor<T,AF,PF,RT> >
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
template< typename T, bool AF, bool PF, typename RT >
struct IsCustom< CustomTensor<T,AF,PF,RT> >
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
template< typename T, bool PF, typename RT >
struct IsAligned< CustomTensor<T,aligned,PF,RT> >
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
template< typename T, bool AF, bool PF, typename RT >
struct IsContiguous< CustomTensor<T,AF,PF,RT> >
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
template< typename T, bool AF, typename RT >
struct IsPadded< CustomTensor<T,AF,padded,RT> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
