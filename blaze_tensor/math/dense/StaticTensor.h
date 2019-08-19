//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/StaticTensor.h
//  \brief Header file for the implementation of a fixed-size tensor
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_STATICTENSOR_H_
#define _BLAZE_TENSOR_MATH_DENSE_STATICTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <utility>

#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>
#include <blaze/math/Forward.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/StaticMatrix.h>
#include <blaze/math/constraints/Diagonal.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/dense/DenseIterator.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Conjugate.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/NextMultiple.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/ExpandTrait.h>
#include <blaze/math/traits/MapTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/SchurTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/HasSIMDSub.h>
#include <blaze/math/typetraits/HighType.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsContiguous.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsSIMDCombinable.h>
#include <blaze/math/typetraits/IsSquare.h>
#include <blaze/math/typetraits/IsStatic.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/LowType.h>
#include <blaze/math/typetraits/MaxSize.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/StorageOrder.h>
#include <blaze/system/TransposeFlag.h>
#include <blaze/util/AlignedArray.h>
#include <blaze/util/AlignmentCheck.h>
#include <blaze/util/Assert.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/Memory.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>
#include <blaze/util/MaybeUnused.h>
#include <blaze/util/algorithms/Max.h>
#include <blaze/util/algorithms/Min.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/FloatingPoint.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsSame.h>
#include <blaze/util/typetraits/IsVectorizable.h>
#include <blaze/util/typetraits/RemoveConst.h>

// #include <blaze_tensor/math/traits/ColumnSlicesTrait.h>
// #include <blaze_tensor/math/traits/RowSlicesTrait.h>
// #include <blaze_tensor/math/typetraits/IsSparseTensor.h>
#include <blaze_tensor/math/InitializerList.h>
#include <blaze_tensor/math/dense/Forward.h>
#include <blaze_tensor/math/dense/StaticMatrix.h>
#include <blaze_tensor/math/dense/Transposition.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>
#include <blaze_tensor/math/traits/ColumnSliceTrait.h>
#include <blaze_tensor/math/traits/DilatedSubtensorTrait.h>
#include <blaze_tensor/math/traits/PageSliceTrait.h>
#include <blaze_tensor/math/traits/RavelTrait.h>
#include <blaze_tensor/math/traits/RowSliceTrait.h>
#include <blaze_tensor/math/traits/SubtensorTrait.h>
#include <blaze_tensor/math/typetraits/IsColumnMajorTensor.h>
#include <blaze_tensor/math/typetraits/IsDenseTensor.h>
#include <blaze_tensor/math/typetraits/IsRowMajorTensor.h>
#include <blaze_tensor/math/typetraits/IsTensor.h>
#include <blaze_tensor/math/typetraits/StorageOrder.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup static_tensor StaticTensor
// \ingroup dense_tensor
*/
/*!\brief Efficient implementation of a fixed-sized tensor.
// \ingroup static_tensor
//
// The StaticTensor class template is the representation of a fixed-size tensor with statically
// allocated elements of arbitrary type. The type of the elements, the number of rows and columns
// and the storage order of the tensor can be specified via the four template parameters:

   \code
   template< typename Type, size_t M, size_t N >
   class StaticTensor;
   \endcode

//  - Type: specifies the type of the tensor elements. StaticTensor can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//  - M   : specifies the total number of rows of the tensor.
//  - N   : specifies the total number of columns of the tensor. Note that it is expected
//          that StaticTensor is only used for tiny and small matrices.
//  - SO  : specifies the storage order (blaze::rowMajor, blaze::columnMajor) of the tensor.
//          The default value is blaze::rowMajor.
//
// Depending on the storage order, the tensor elements are either stored in a row-wise fashion
// or in a column-wise fashion. Given the 2x3 tensor

                          \f[\left(\begin{array}{*{3}{c}}
                          1 & 2 & 3 \\
                          4 & 5 & 6 \\
                          \end{array}\right)\f]\n

// in case of row-major order the elements are stored in the order

                          \f[\left(\begin{array}{*{6}{c}}
                          1 & 2 & 3 & 4 & 5 & 6. \\
                          \end{array}\right)\f]

// In case of column-major order the elements are stored in the order

                          \f[\left(\begin{array}{*{6}{c}}
                          1 & 4 & 2 & 5 & 3 & 6. \\
                          \end{array}\right)\f]

// The use of StaticTensor is very natural and intuitive. All operations (addition, subtraction,
// multiplication, scaling, ...) can be performed on all possible combinations of row-major and
// column-major dense and sparse matrices with fitting element types. The following example gives
// an impression of the use of StaticTensor:

   \code
   using blaze::StaticTensor;
   using blaze::CompressedTensor;
   using blaze::rowMajor;
   using blaze::columnMajor;

   StaticTensor<double,2UL,3UL,rowMajor> A;   // Default constructed, non-initialized, row-major 2x3 tensor
   A(0,0) = 1.0; A(0,1) = 2.0; A(0,2) = 3.0;  // Initialization of the first row
   A(1,0) = 4.0; A(1,1) = 5.0; A(1,2) = 6.0;  // Initialization of the second row

   DynamicTensor<float,2UL,3UL,columnMajor> B;  // Default constructed column-major single precision 2x3 tensor
   B(0,0) = 1.0; B(0,1) = 3.0; B(0,2) = 5.0;    // Initialization of the first row
   B(1,0) = 2.0; B(1,1) = 4.0; B(1,2) = 6.0;    // Initialization of the second row

   CompressedTensor<float>     C( 2, 3 );  // Empty row-major sparse single precision tensor
   StaticTensor<float,3UL,2UL> D( 4.0F );  // Directly, homogeneously initialized single precision 3x2 tensor

   StaticTensor<double,2UL,3UL,rowMajor>    E( A );  // Creation of a new row-major tensor as a copy of A
   StaticTensor<double,2UL,2UL,columnMajor> F;       // Creation of a default column-major tensor

   E = A + B;     // Tensor addition and assignment to a row-major tensor
   F = A - C;     // Tensor subtraction and assignment to a column-major tensor
   F = A * D;     // Tensor multiplication between two matrices of different element types

   A *= 2.0;      // In-place scaling of tensor A
   E  = 2.0 * B;  // Scaling of tensor B
   E  = B * 2.0;  // Scaling of tensor B

   E += A - B;    // Addition assignment
   E -= A + C;    // Subtraction assignment
   F *= A * D;    // Multiplication assignment
   \endcode
*/
template< typename Type                    // Data type of the tensor
        , size_t O                         // Number of pages
        , size_t M                         // Number of rows
        , size_t N >                       // Number of columns
class StaticTensor
   : public DenseTensor< StaticTensor<Type,O,M,N> >
{
 private:
   //**********************************************************************************************
   //! The number of elements packed within a single SIMD vector.
   static constexpr size_t SIMDSIZE = SIMDTrait<Type>::size;

   //! Alignment adjustment.
   static constexpr size_t NN = ( usePadding ? nextMultiple( N, SIMDSIZE ) : N );

   //! Compilation switch for the choice of alignment.
   static constexpr bool align = ( usePadding || NN % SIMDSIZE == 0UL );
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This          = StaticTensor<Type,O,M,N>;    //!< Type of this StaticTensor instance.
   using BaseType      = DenseTensor<This>;           //!< Base type of this StaticTensor instance.
   using ResultType    = This;                        //!< Result type for expression template evaluations.
   using OppositeType  = StaticTensor<Type,O,M,N>;    //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = StaticTensor<Type,O,N,M>;    //!< Transpose type for expression template evaluations.
   using ElementType   = Type;                        //!< Type of the tensor elements.
   using SIMDType      = SIMDTrait_t<ElementType>;    //!< SIMD type of the tensor elements.
   using ReturnType    = const Type&;                 //!< Return type for expression template evaluations.
   using CompositeType = const This&;                 //!< Data type for composite expression templates.

   using Reference      = Type&;        //!< Reference to a non-constant tensor value.
   using ConstReference = const Type&;  //!< Reference to a constant tensor value.
   using Pointer        = Type*;        //!< Pointer to a non-constant tensor value.
   using ConstPointer   = const Type*;  //!< Pointer to a constant tensor value.

   using Iterator      = DenseIterator<Type,align>;        //!< Iterator over non-constant elements.
   using ConstIterator = DenseIterator<const Type,align>;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a StaticTensor with different data/element type.
   */
   template< typename NewType >  // Data type of the other tensor
   struct Rebind {
      using Other = StaticTensor<NewType,O,M,N>;  //!< The type of the other StaticTensor.
   };
   //**********************************************************************************************

   //**Resize struct definition********************************************************************
   /*!\brief Resize mechanism to obtain a StaticTensor with different fixed dimensions.
   */
   template< size_t NewO    // Number of pages of the other tensor
           , size_t NewM    // Number of rows of the other tensor
           , size_t NewN >  // Number of columns of the other tensor
   struct Resize {
      using Other = StaticTensor<Type,NewO,NewM,NewN>;  //!< The type of the other StaticTensor.
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
   static constexpr bool smpAssignable = false;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline           StaticTensor();
   explicit inline           StaticTensor( const Type& init );
   explicit inline constexpr StaticTensor( initializer_list< initializer_list< initializer_list<Type> > > list );

   template< typename Other >
   explicit inline constexpr StaticTensor( size_t o, size_t m, size_t n, const Other* array );

   template< typename Other, size_t Pages, size_t Rows, size_t Cols >
   explicit inline constexpr StaticTensor( const Other (&array)[Pages][Rows][Cols] );

                                        inline StaticTensor( const StaticTensor& m );
   template< typename Other > inline StaticTensor( const StaticTensor<Other,O,M,N>& m );
   template< typename MT    > inline StaticTensor( const Tensor<MT>& m );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~StaticTensor() = default;
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline constexpr Reference      operator()( size_t k, size_t i, size_t j ) noexcept;
   inline constexpr ConstReference operator()( size_t k, size_t i, size_t j ) const noexcept;
   inline           Reference      at( size_t k, size_t i, size_t j );
   inline           ConstReference at( size_t k, size_t i, size_t j ) const;
   inline constexpr Pointer        data  () noexcept;
   inline constexpr ConstPointer   data  () const noexcept;
   inline constexpr Pointer        data  ( size_t i, size_t k ) noexcept;
   inline constexpr ConstPointer   data  ( size_t i, size_t k ) const noexcept;
   inline constexpr Iterator       begin ( size_t i, size_t k ) noexcept;
   inline constexpr ConstIterator  begin ( size_t i, size_t k ) const noexcept;
   inline constexpr ConstIterator  cbegin( size_t i, size_t k ) const noexcept;
   inline constexpr Iterator       end   ( size_t i, size_t k ) noexcept;
   inline constexpr ConstIterator  end   ( size_t i, size_t k ) const noexcept;
   inline constexpr ConstIterator  cend  ( size_t i, size_t k ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline constexpr StaticTensor& operator=( const Type& set );
   inline constexpr StaticTensor& operator=( initializer_list< initializer_list< initializer_list<Type> > > list );

   template< typename Other, size_t Pages, size_t Rows, size_t Cols >
   inline constexpr StaticTensor& operator=( const Other (&array)[Pages][Rows][Cols] );

                              inline StaticTensor& operator= ( const StaticTensor& rhs );
   template< typename Other > inline StaticTensor& operator= ( const StaticTensor<Other,O,M,N>& rhs );
   template< typename MT    > inline StaticTensor& operator= ( const Tensor<MT>& rhs );
   template< typename MT    > inline StaticTensor& operator+=( const Tensor<MT>& rhs );
   template< typename MT    > inline StaticTensor& operator-=( const Tensor<MT>& rhs );
   template< typename MT    > inline StaticTensor& operator%=( const Tensor<MT>& rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   static inline constexpr size_t rows() noexcept;
   static inline constexpr size_t columns() noexcept;
   static inline constexpr size_t pages() noexcept;
   static inline constexpr size_t spacing() noexcept;
   static inline constexpr size_t capacity() noexcept;
          inline           size_t capacity( size_t i, size_t k ) const noexcept;
          inline           size_t nonZeros() const;
          inline           size_t nonZeros( size_t i, size_t k ) const;
          inline constexpr void   reset();
          inline           void   reset( size_t i, size_t k );
          inline           void   swap( StaticTensor& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   inline StaticTensor& transpose();
   inline StaticTensor& ctranspose();
   template< typename T >
   inline StaticTensor& transpose( const T* indices, size_t n );
   template< typename T >
   inline StaticTensor& ctranspose( const T* indices, size_t n );

   template< typename Other > inline StaticTensor& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

   //**Memory functions****************************************************************************
   /*!\name Memory functions */
   //@{
   static inline void* operator new  ( std::size_t size );
   static inline void* operator new[]( std::size_t size );
   static inline void* operator new  ( std::size_t size, const std::nothrow_t& );
   static inline void* operator new[]( std::size_t size, const std::nothrow_t& );

   static inline void operator delete  ( void* ptr );
   static inline void operator delete[]( void* ptr );
   static inline void operator delete  ( void* ptr, const std::nothrow_t& );
   static inline void operator delete[]( void* ptr, const std::nothrow_t& );
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
        IsRowMajorTensor_v< MT >);
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
        !IsDiagonal_v<MT> &&
        IsRowMajorTensor_v< MT >);
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
        !IsDiagonal_v<MT> &&
        IsRowMajorTensor_v< MT >);
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
        IsRowMajorTensor_v< MT >);
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Debugging functions*************************************************************************
   /*!\name Debugging functions */
   //@{
   inline constexpr bool isIntact() const noexcept;
   //@}
   //**********************************************************************************************

   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other > inline bool canAlias ( const Other* alias ) const noexcept;
   template< typename Other > inline bool isAliased( const Other* alias ) const noexcept;

   static inline constexpr bool isAligned() noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t k, size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t k, size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t k, size_t i, size_t j ) const noexcept;

   BLAZE_ALWAYS_INLINE void store ( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storea( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void storeu( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept;
   BLAZE_ALWAYS_INLINE void stream( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept;

   template< typename MT >
   inline auto assign( const DenseTensor<MT>& rhs ) -> DisableIf_t< VectorizedAssign_v<MT> >;

   template< typename MT >
   inline auto assign( const DenseTensor<MT>& rhs ) -> EnableIf_t< VectorizedAssign_v<MT> >;

//    template< typename MT > inline void assign( const SparseTensor<MT>&  rhs );
//    template< typename MT > inline void assign( const SparseTensor<MT,!SO>& rhs );

   template< typename MT >
   inline auto addAssign( const DenseTensor<MT>& rhs ) -> DisableIf_t< VectorizedAddAssign_v<MT> >;

   template< typename MT >
   inline auto addAssign( const DenseTensor<MT>& rhs ) -> EnableIf_t< VectorizedAddAssign_v<MT> >;

//    template< typename MT > inline void addAssign( const SparseTensor<MT>&  rhs );
//    template< typename MT > inline void addAssign( const SparseTensor<MT,!SO>& rhs );

   template< typename MT >
   inline auto subAssign( const DenseTensor<MT>& rhs ) -> DisableIf_t< VectorizedSubAssign_v<MT> >;

   template< typename MT >
   inline auto subAssign( const DenseTensor<MT>& rhs ) -> EnableIf_t< VectorizedSubAssign_v<MT> >;

//    template< typename MT > inline void subAssign( const SparseTensor<MT>&  rhs );
//    template< typename MT > inline void subAssign( const SparseTensor<MT,!SO>& rhs );

   template< typename MT >
   inline auto schurAssign( const DenseTensor<MT>& rhs ) -> DisableIf_t< VectorizedSchurAssign_v<MT> >;

   template< typename MT >
   inline auto schurAssign( const DenseTensor<MT>& rhs ) -> EnableIf_t< VectorizedSchurAssign_v<MT> >;

//    template< typename MT > inline void schurAssign( const SparseTensor<MT>&  rhs );
//    template< typename MT > inline void schurAssign( const SparseTensor<MT,!SO>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*! \cond BLAZE_INTERNAL */
//    inline void transpose ( TrueType  );
//    inline void transpose ( FalseType );
//    inline void ctranspose( TrueType  );
//    inline void ctranspose( FalseType );
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   //! Alignment of the data elements.
   static constexpr size_t Alignment =
      ( align ? AlignmentOf_v<Type> : std::alignment_of<Type>::value );

   //! Type of the aligned storage.
   using AlignedStorage = AlignedArray<Type,O*M*NN,Alignment>;
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   AlignedStorage v_;  //!< The statically allocated tensor elements.
                       /*!< Access to the tensor elements is gained via the function call
                            operator. In case of row-major order the memory layout of the
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
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST         ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE      ( Type );
   BLAZE_STATIC_ASSERT( !usePadding || NN % SIMDSIZE == 0UL );
   BLAZE_STATIC_ASSERT( NN >= N );
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
/*!\brief The default constructor for StaticTensor.
//
// All tensor elements are initialized to the default value (i.e. 0 for integral data types).
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline StaticTensor<Type,O,M,N>::StaticTensor()
   : v_()  // The statically allocated tensor elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable_v<Type> || NN == N );

   if( IsNumeric_v<Type> ) {
      for( size_t i=0UL; i<O*M*NN; ++i )
         v_[i] = Type();
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a homogeneous initialization of all elements.
//
// \param init Initial value for all tensor elements.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline StaticTensor<Type,O,M,N>::StaticTensor( const Type& init )
   : v_()  // The statically allocated tensor elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable_v<Type> || NN == N );

   for( size_t k=0UL; k<O; ++k ) {
      for( size_t i=0UL; i<M; ++i ) {
         for( size_t j=0UL; j<N; ++j )
            v_[(k*M+i)*NN+j] = init;

         for( size_t j=N; j<NN; ++j )
            v_[(k*M+i)*NN+j] = Type();
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List initialization of all tensor elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid setup of static tensor.
//
// This constructor provides the option to explicitly initialize the elements of the tensor by
// means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::StaticTensor<int,3,3,rowMajor> A{ { 1, 2, 3 },
                                            { 4, 5 },
                                            { 7, 8, 9 } };
   \endcode

// The tensor elements are initialized by the values of the given initializer list. Missing values
// are initialized as default (as e.g. the value 6 in the example). Note that in case the size of
// the top-level initializer list does not match the number of rows of the tensor or the size of
// any nested list exceeds the number of columns, a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr StaticTensor<Type,O,M,N>::StaticTensor( initializer_list< initializer_list< initializer_list<Type> > > list )
   : v_()  // The statically allocated tensor elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable_v<Type> || NN == N );

   if( list.size() != O || determineRows( list ) > M || determineColumns( list ) > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of static tensor" );
   }

   size_t k( 0UL );

   for (const auto& page : list) {
      size_t i( 0UL );
      for (const auto& rowList : page) {
         std::fill(std::copy(rowList.begin(), rowList.end(), v_+(k*M+i)*NN), v_+(k*M+i+1UL)*NN, Type());
         ++i;
      }
      ++k;
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all tensor elements.
//
// \param m The number of rows of the tensor.
// \param n The number of columns of the tensor.
// \param array Dynamic array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the tensor with
// a dynamic array:

   \code
   using blaze::rowMajor;

   int* array = new int[6];
   // ... Initialization of the dynamic array
   blaze::StaticTensor<int,3,4,rowMajor> v( array, 2UL, 3UL );
   delete[] array;
   \endcode

// The tensor is initialized with the values from the given array. Missing values are initialized
// with default values. In case the specified number of rows and/or columns exceeds the maximum
// number of rows/column of the static tensor (i.e. \a m is larger than M or \a n is larger than
// N), a \a std::invalid_argument exception is thrown.\n
// Note that it is expected that the given \a array has at least \a m by \a n elements. Providing
// an array with less elements results in undefined behavior!
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename Other >  // Data type of the initialization array
inline constexpr StaticTensor<Type,O,M,N>::StaticTensor( size_t o, size_t m, size_t n, const Other* array )
   : v_()  // The statically allocated tensor elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable_v<Type> || NN == N );

   if( o > O || m > M || n > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of static tensor" );
   }

   for (size_t k=0UL; k<o; ++k) {
      for( size_t i=0UL; i<m; ++i ) {
         for( size_t j=0UL; j<n; ++j )
            v_[(k*M+i)*NN+j] = array[(k*m+i)*n+j];

         if( IsNumeric_v<Type> ) {
            for( size_t j=n; j<NN; ++j )
               v_[(k*M+i)*NN+j] = Type();
         }
      }
   }

   if( IsNumeric_v<Type> ) {
      for (size_t k=0UL; k<O; ++k) {
         if( k < o )
         {
            for( size_t i=m; i<M; ++i ) {
               for( size_t j=0UL; j<NN; ++j )
                  v_[(k*M+i)*NN+j] = Type();
            }
         }
         else
         {
            for( size_t i=0UL; i<M; ++i ) {
               for( size_t j=0UL; j<NN; ++j )
                  v_[(k*M+i)*NN+j] = Type();
            }
         }
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all tensor elements.
//
// \param array \f$ M \times N \f$ dimensional array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the tensor with
// a static array:

   \code
   using blaze::rowMajor;

   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::StaticTensor<int,3,3,rowMajor> A( init );
   \endcode

// The tensor is initialized with the values from the given array. Missing values are initialized
// with default values (as e.g. the value 6 in the example).
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename Other  // Data type of the initialization array
        , size_t Pages    // Number of pages of the initialization array
        , size_t Rows     // Number of rows of the initialization array
        , size_t Cols >   // Number of columns of the initialization array
inline constexpr StaticTensor<Type,O,M,N>::StaticTensor( const Other (&array)[Pages][Rows][Cols] )
   : v_()  // The statically allocated tensor elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable_v<Type> || NN == N );
   BLAZE_STATIC_ASSERT( Pages == O && Rows == M && Cols == N );

   for (size_t k=0UL; k<O; ++k) {
      for( size_t i=0UL; i<M; ++i ) {
         for( size_t j=0UL; j<N; ++j )
            v_[(k*M+i)*NN+j] = array[k][i][j];

         for( size_t j=N; j<NN; ++j )
               v_[(k*M+i)*NN+j] = Type();
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for StaticTensor.
//
// \param m Tensor to be copied.
//
// The copy constructor is explicitly defined in order to enable/facilitate NRV optimization.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline StaticTensor<Type,O,M,N>::StaticTensor( const StaticTensor& m )
   : v_()  // The statically allocated tensor elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable_v<Type> || NN == N );

   for( size_t i=0UL; i<O*M*NN; ++i )
      v_[i] = m.v_[i];

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different StaticTensor instances.
//
// \param m Tensor to be copied.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename Other > // Data type of the foreign tensor
inline StaticTensor<Type,O,M,N>::StaticTensor( const StaticTensor<Other,O,M,N>& m )
   : v_()  // The statically allocated tensor elements
{
   BLAZE_STATIC_ASSERT( IsVectorizable_v<Type> || NN == N );

   for (size_t k=0UL; k<O; ++k) {
      for( size_t i=0UL; i<M; ++i ) {
         for( size_t j=0UL; j<N; ++j )
            v_[(k*M+i)*NN+j] = m(k,i,j);

         for( size_t j=N; j<NN; ++j )
            v_[(k*M+i)*NN+j] = Type();
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different matrices.
//
// \param m Tensor to be copied.
// \exception std::invalid_argument Invalid setup of static tensor.
//
// This constructor initializes the static tensor from the given tensor. In case the size of
// the given tensor does not match the size of the static tensor (i.e. the number of rows is
// not M or the number of columns is not N), a \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the foreign tensor
inline StaticTensor<Type,O,M,N>::StaticTensor( const Tensor<MT>& m )
   : v_()  // The statically allocated tensor elements
{
   using blaze::assign;

   BLAZE_STATIC_ASSERT( IsVectorizable_v<Type> || NN == N );

   if( (~m).pages() != O || (~m).rows() != M || (~m).columns() != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of static tensor" );
   }

   for (size_t k=0UL; k<O; ++k) {
      for( size_t i=0UL; i<M; ++i ) {
         for( size_t j=( /*IsSparseTensor_v<MT> ? 0UL : */N ); j<NN; ++j ) {
            v_[(k*M+i)*NN+j] = Type();
         }
      }
   }

   assign( *this, ~m );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
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
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr typename StaticTensor<Type,O,M,N>::Reference
   StaticTensor<Type,O,M,N>::operator()( size_t k, size_t i, size_t j ) noexcept
{
   BLAZE_USER_ASSERT( k<O, "Invalid page access index"   );
   BLAZE_USER_ASSERT( i<M, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<N, "Invalid column access index" );
   return v_[(k*M+i)*NN+j];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief 2D-access to the tensor elements.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return Reference-to-const to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr typename StaticTensor<Type,O,M,N>::ConstReference
   StaticTensor<Type,O,M,N>::operator()( size_t k, size_t i, size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( k<O, "Invalid page access index"   );
   BLAZE_USER_ASSERT( i<M, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<N, "Invalid column access index" );
   return v_[(k*M+i)*NN+j];
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename StaticTensor<Type,O,M,N>::Reference
   StaticTensor<Type,O,M,N>::at( size_t k, size_t i, size_t j )
{
   if( k >= O ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
   }
   if( i >= M ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= N ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(k,i,j);
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline typename StaticTensor<Type,O,M,N>::ConstReference
   StaticTensor<Type,O,M,N>::at( size_t k, size_t i, size_t j ) const
{
   if( k >= O ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
   }
   if( i >= M ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= N ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(k,i,j);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the tensor elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the static tensor. Note that you
// can NOT assume that all tensor elements lie adjacent to each other! The static tensor may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr typename StaticTensor<Type,O,M,N>::Pointer
   StaticTensor<Type,O,M,N>::data() noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the tensor elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the static tensor. Note that you
// can NOT assume that all tensor elements lie adjacent to each other! The static tensor may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr typename StaticTensor<Type,O,M,N>::ConstPointer
   StaticTensor<Type,O,M,N>::data() const noexcept
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr typename StaticTensor<Type,O,M,N>::Pointer
   StaticTensor<Type,O,M,N>::data( size_t i, size_t k ) noexcept
{
   BLAZE_USER_ASSERT( k < O, "Invalid page access index"   );
   BLAZE_USER_ASSERT( i < M, "Invalid dense tensor row access index" );
   return v_ + (k*M+i)*NN;
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr typename StaticTensor<Type,O,M,N>::ConstPointer
   StaticTensor<Type,O,M,N>::data( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( k < O, "Invalid page access index"   );
   BLAZE_USER_ASSERT( i < M, "Invalid dense tensor row access index" );
   return v_ + (k*M+i)*NN;
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr typename StaticTensor<Type,O,M,N>::Iterator
   StaticTensor<Type,O,M,N>::begin( size_t i, size_t k ) noexcept
{
   BLAZE_USER_ASSERT( k < O, "Invalid page access index"   );
   BLAZE_USER_ASSERT( i < M, "Invalid dense tensor row access index" );
   return Iterator( v_ + (k*M+i)*NN );
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr typename StaticTensor<Type,O,M,N>::ConstIterator
   StaticTensor<Type,O,M,N>::begin( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( k < O, "Invalid page access index"   );
   BLAZE_USER_ASSERT( i < M, "Invalid dense tensor row access index" );
   return ConstIterator( v_ + (k*M+i)*NN );
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr typename StaticTensor<Type,O,M,N>::ConstIterator
   StaticTensor<Type,O,M,N>::cbegin( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( k < O, "Invalid page access index"   );
   BLAZE_USER_ASSERT( i < M, "Invalid dense tensor row access index" );
   return ConstIterator( v_ + (k*M+i)*NN );
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr typename StaticTensor<Type,O,M,N>::Iterator
   StaticTensor<Type,O,M,N>::end( size_t i, size_t k ) noexcept
{
   BLAZE_USER_ASSERT( k < O, "Invalid page access index"   );
   BLAZE_USER_ASSERT( i < M, "Invalid dense tensor row access index" );
   return Iterator( v_ + (k*M+i)*NN + N );
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr typename StaticTensor<Type,O,M,N>::ConstIterator
   StaticTensor<Type,O,M,N>::end( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( k < O, "Invalid page access index"   );
   BLAZE_USER_ASSERT( i < M, "Invalid dense tensor row access index" );
   return ConstIterator( v_ + (k*M+i)*NN + N );
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr typename StaticTensor<Type,O,M,N>::ConstIterator
   StaticTensor<Type,O,M,N>::cend( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( k < O, "Invalid page access index"   );
   BLAZE_USER_ASSERT( i < M, "Invalid dense tensor row access index" );
   return ConstIterator( v_ + (k*M+i)*NN + N );
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
// \param set Scalar value to be assigned to all tensor elements.
// \return Reference to the assigned tensor.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr StaticTensor<Type,O,M,N>& StaticTensor<Type,O,M,N>::operator=( const Type& set )
{
   for (size_t k=0UL; k<O; ++k)
      for( size_t i=0UL; i<M; ++i )
         for( size_t j=0UL; j<N; ++j )
            v_[(k*M+i)*NN+j] = set;

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
   using blaze::rowMajor;

   blaze::StaticTensor<int,3,3,rowMajor> A;
   A = { { 1, 2, 3 },
         { 4, 5 },
         { 7, 8, 9 } };
   \endcode

// The tensor elements are assigned the values from the given initializer list. Missing values
// are initialized as default (as e.g. the value 6 in the example). Note that in case the size
// of the top-level initializer list does not match the number of rows of the tensor or the size
// of any nested list exceeds the number of columns, a \a std::invalid_argument exception is
// thrown.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr StaticTensor<Type,O,M,N>&
   StaticTensor<Type,O,M,N>::operator=( initializer_list< initializer_list< initializer_list<Type> > > list )
{
   if( list.size() != O || determineRows( list ) > M || determineColumns( list ) > N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to static tensor" );
   }

   size_t k( 0UL );

   for (const auto& page : list) {
      size_t i( 0UL );
      for (const auto& rowList : page) {
         std::fill(std::copy(rowList.begin(), rowList.end(), v_+(k*M+i)*NN), v_+(k*M+i+1UL)*NN, Type());
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
//
// This assignment operator offers the option to directly set all elements of the tensor:

   \code
   using blaze::rowMajor;

   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::StaticTensor<int,3UL,3UL,rowMajor> A;
   A = init;
   \endcode

// The tensor is assigned the values from the given array. Missing values are initialized with
// default values (as e.g. the value 6 in the example).
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename Other  // Data type of the initialization array
        , size_t Pages    // Number of pages of the initialization array
        , size_t Rows     // Number of rows of the initialization array
        , size_t Cols >   // Number of columns of the initialization array
inline constexpr StaticTensor<Type,O,M,N>& StaticTensor<Type,O,M,N>::operator=( const Other (&array)[Pages][Rows][Cols] )
{
   BLAZE_STATIC_ASSERT( Pages == O && Rows == M && Cols == N );

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i )
         for( size_t j=0UL; j<N; ++j )
            v_[(k*M+i)*NN+j] = array[k][i][j];

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy assignment operator for StaticTensor.
//
// \param rhs Tensor to be copied.
// \return Reference to the assigned tensor.
//
// Explicit definition of a copy assignment operator for performance reasons.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline StaticTensor<Type,O,M,N>& StaticTensor<Type,O,M,N>::operator=( const StaticTensor& rhs )
{
   using blaze::assign;

   assign( *this, ~rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different StaticTensor instances.
//
// \param rhs Tensor to be copied.
// \return Reference to the assigned tensor.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename Other > // Data type of the foreign tensor
inline StaticTensor<Type,O,M,N>&
   StaticTensor<Type,O,M,N>::operator=( const StaticTensor<Other,O,M,N>& rhs )
{
   using blaze::assign;

   assign( *this, ~rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different matrices.
//
// \param rhs Tensor to be copied.
// \return Reference to the assigned tensor.
// \exception std::invalid_argument Invalid assignment to static tensor.
//
// This constructor initializes the tensor as a copy of the given tensor. In case the
// number of rows of the given tensor is not M or the number of columns is not N, a
// \a std::invalid_argument exception is thrown.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side tensor
inline StaticTensor<Type,O,M,N>& StaticTensor<Type,O,M,N>::operator=( const Tensor<MT>& rhs )
{
   using blaze::assign;

//    using TT = decltype( trans( *this ) );
//    using CT = decltype( ctrans( *this ) );
//    using IT = decltype( inv( *this ) );

   if( (~rhs).pages() != O || (~rhs).rows() != M || (~rhs).columns() != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to static tensor" );
   }

//    if( IsSame_v<MT,TT> && (~rhs).isAliased( this ) ) {
//       transpose( typename IsSquare<This>::Type() );
//    }
//    else if( IsSame_v<MT,CT> && (~rhs).isAliased( this ) ) {
//       ctranspose( typename IsSquare<This>::Type() );
//    }
//   else
   if( /*!IsSame_v<MT,IT> &&*/ (~rhs).canAlias( this ) ) {
      StaticTensor tmp( ~rhs );
      assign( *this, tmp );
   }
   else {
//       if( IsSparseTensor_v<MT> )
//          reset();
      assign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side tensor
inline StaticTensor<Type,O,M,N>& StaticTensor<Type,O,M,N>::operator+=( const Tensor<MT>& rhs )
{
   using blaze::addAssign;

   if( (~rhs).pages() != O || (~rhs).rows() != M || (~rhs).columns() != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      addAssign( *this, tmp );
   }
   else {
      addAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side tensor
inline StaticTensor<Type,O,M,N>& StaticTensor<Type,O,M,N>::operator-=( const Tensor<MT>& rhs )
{
   using blaze::subAssign;

   if( (~rhs).pages() != O || (~rhs).rows() != M || (~rhs).columns() != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      subAssign( *this, tmp );
   }
   else {
      subAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side tensor
inline StaticTensor<Type,O,M,N>& StaticTensor<Type,O,M,N>::operator%=( const Tensor<MT>& rhs )
{
   using blaze::schurAssign;

   if( (~rhs).pages() != O || (~rhs).rows() != M || (~rhs).columns() != N ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      schurAssign( *this, tmp );
   }
   else {
      schurAssign( *this, ~rhs );
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
/*!\brief Returns the current number of rows of the tensor.
//
// \return The number of rows of the tensor.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr size_t StaticTensor<Type,O,M,N>::rows() noexcept
{
   return M;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of columns of the tensor.
//
// \return The number of columns of the tensor.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr size_t StaticTensor<Type,O,M,N>::columns() noexcept
{
   return N;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of pages of the tensor.
//
// \return The number of pages of the tensor.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr size_t StaticTensor<Type,O,M,N>::pages() noexcept
{
   return O;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the spacing between the beginning of two rows.
//
// \return The spacing between the beginning of two rows.
//
// This function returns the spacing between the beginning of two rows, i.e. the total number
// of elements of a row.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr size_t StaticTensor<Type,O,M,N>::spacing() noexcept
{
   return NN;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the tensor.
//
// \return The capacity of the tensor.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr size_t StaticTensor<Type,O,M,N>::capacity() noexcept
{
   return O*M*NN;
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline size_t StaticTensor<Type,O,M,N>::capacity( size_t i, size_t k ) const noexcept
{
   MAYBE_UNUSED( i );

   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index" );
   BLAZE_USER_ASSERT( i < rows() , "Invalid row access index" );

   return NN;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of non-zero elements in the tensor
//
// \return The number of non-zero elements in the tensor.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline size_t StaticTensor<Type,O,M,N>::nonZeros() const
{
   size_t nonzeros( 0UL );

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i )
         for( size_t j=0UL; j<N; ++j )
            if( !isDefault( v_[(k*M+i)*NN+j] ) )
               ++nonzeros;

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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline size_t StaticTensor<Type,O,M,N>::nonZeros( size_t i, size_t k ) const
{
   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index" );
   BLAZE_USER_ASSERT( i < rows() , "Invalid row access index" );

   const size_t jend( (k*M+i)*NN + N );
   size_t nonzeros( 0UL );

   for( size_t j=(k*M+i)*NN; j<jend; ++j )
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr void StaticTensor<Type,O,M,N>::reset()
{
   using blaze::clear;

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i )
         for( size_t j=0UL; j<N; ++j )
            clear( v_[(k*M+i)*NN+j] );
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void StaticTensor<Type,O,M,N>::reset( size_t i, size_t k )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index" );
   BLAZE_USER_ASSERT( i < rows() , "Invalid row access index" );
   for( size_t j=0UL; j<N; ++j )
      clear( v_[(k*M+i)*NN+j] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two static matrices.
//
// \param m The tensor to be swapped.
// \return void
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void StaticTensor<Type,O,M,N>::swap( StaticTensor& m ) noexcept
{
   using std::swap;

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i ) {
         for( size_t j=0UL; j<N; ++j ) {
            swap( v_[(k*M+i)*NN+j], m(k,i,j) );
         }
      }
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
//
// This function transposes the static tensor in-place. Note that this function can only be used
// for square static matrices, i.e. if \a M is equal to N.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline StaticTensor<Type,O,M,N>& StaticTensor<Type,O,M,N>::transpose()
{
   BLAZE_STATIC_ASSERT( O == M && M == N );

   transposeGeneral( *this );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the tensor.
//
// \return Reference to the transposed tensor.
//
// This function transposes the static tensor in-place. Note that this function can only be used
// for square static matrices, i.e. if \a M is equal to N.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename T >   // Type of the mapping indices
inline StaticTensor<Type,O,M,N>& StaticTensor<Type,O,M,N>::transpose( const T* indices, size_t n )
{
   BLAZE_STATIC_ASSERT( O == M && M == N );

   transposeGeneral( *this, indices, n );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Helper function for self-transpose via the trans() function.
//
// \return void
//
// This function assists in the evaluation of self-transpose via the trans() function:

   \code
   blaze::StaticTensor<int,3UL,3UL,blaze::rowMajor> A;

   A = trans( A );
   \endcode
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// inline void StaticTensor<Type,O,M,N>::transpose( TrueType )
// {
//    transpose();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Helper function for self-transpose via the trans() function.
//
// \return void
//
// This function assists in the evaluation of self-transpose via the trans() function:

   \code
   blaze::StaticTensor<int,3UL,3UL,blaze::rowMajor> A;

   A = trans( A );
   \endcode
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// inline void StaticTensor<Type,O,M,N>::transpose( FalseType )
// {}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the tensor.
//
// \return Reference to the transposed tensor.
//
// This function transposes the static tensor in-place. Note that this function can only be used
// for square static matrices, i.e. if \a M is equal to N.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline StaticTensor<Type,O,M,N>& StaticTensor<Type,O,M,N>::ctranspose()
{
   BLAZE_STATIC_ASSERT( M == N );

//    for( size_t i=0UL; i<M; ++i ) {
//       for( size_t j=0UL; j<i; ++j ) {
//          cswap( v_[i*NN+j], v_[j*NN+i] );
//       }
//       conjugate( v_[i*NN+i] );
//    }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the tensor.
//
// \return Reference to the transposed tensor.
//
// This function transposes the static tensor in-place. Note that this function can only be used
// for square static matrices, i.e. if \a M is equal to N.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename T >   // Type of the mapping indices
inline StaticTensor<Type,O,M,N>& StaticTensor<Type,O,M,N>::ctranspose( const T* indices, size_t n )
{
   BLAZE_STATIC_ASSERT( M == N );

//    for( size_t i=0UL; i<M; ++i ) {
//       for( size_t j=0UL; j<i; ++j ) {
//          cswap( v_[i*NN+j], v_[j*NN+i] );
//       }
//       conjugate( v_[i*NN+i] );
//    }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Helper function for self-transpose via the ctrans() function.
//
// \return void
//
// This function assists in the evaluation of self-transpose via the ctrans() function:

   \code
   blaze::StaticTensor<int,3UL,3UL,blaze::rowMajor> A;

   A = ctrans( A );
   \endcode
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// inline void StaticTensor<Type,O,M,N>::ctranspose( TrueType )
// {
//    ctranspose();
// }
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Helper function for self-transpose via the ctrans() function.
//
// \return void
//
// This function assists in the evaluation of self-transpose via the ctrans() function:

   \code
   blaze::StaticTensor<int,3UL,3UL,blaze::rowMajor> A;

   A = ctrans( A );
   \endcode
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// inline void StaticTensor<Type,O,M,N>::ctranspose( FalseType )
// {}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the tensor by the scalar value \a scalar (\f$ A*=s \f$).
//
// \param scalar The scalar value for the tensor scaling.
// \return Reference to the tensor.
//
// This function scales the tensor by applying the given scalar value \a scalar to each element
// of the tensor. For built-in and \c complex data types it has the same effect as using the
// multiplication assignment operator:

   \code
   blaze::StaticTensor<int,2,3> A;
   // ... Initialization
   A *= 4;        // Scaling of the tensor
   A.scale( 4 );  // Same effect as above
   \endcode
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename Other >  // Data type of the scalar value
inline StaticTensor<Type,O,M,N>& StaticTensor<Type,O,M,N>::scale( const Other& scalar )
{
   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i )
         for( size_t j=0UL; j<N; ++j )
            v_[(k*M+i)*NN+j] *= scalar;

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  MEMORY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Class specific implementation of operator new.
//
// \param size The total number of bytes to be allocated.
// \return Pointer to the newly allocated memory.
// \exception std::bad_alloc Allocation failed.
//
// This class-specific implementation of operator new provides the functionality to allocate
// dynamic memory based on the alignment restrictions of the StaticTensor class template.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void* StaticTensor<Type,O,M,N>::operator new( std::size_t size )
{
   MAYBE_UNUSED( size );

   BLAZE_INTERNAL_ASSERT( size == sizeof( StaticTensor ), "Invalid number of bytes detected" );

   return allocate<StaticTensor>( 1UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of operator new[].
//
// \param size The total number of bytes to be allocated.
// \return Pointer to the newly allocated memory.
// \exception std::bad_alloc Allocation failed.
//
// This class-specific implementation of operator new provides the functionality to allocate
// dynamic memory based on the alignment restrictions of the StaticTensor class template.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void* StaticTensor<Type,O,M,N>::operator new[]( std::size_t size )
{
   BLAZE_INTERNAL_ASSERT( size >= sizeof( StaticTensor )       , "Invalid number of bytes detected" );
   BLAZE_INTERNAL_ASSERT( size %  sizeof( StaticTensor ) == 0UL, "Invalid number of bytes detected" );

   return allocate<StaticTensor>( size/sizeof(StaticTensor) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of the no-throw operator new.
//
// \param size The total number of bytes to be allocated.
// \return Pointer to the newly allocated memory.
// \exception std::bad_alloc Allocation failed.
//
// This class-specific implementation of operator new provides the functionality to allocate
// dynamic memory based on the alignment restrictions of the StaticTensor class template.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void* StaticTensor<Type,O,M,N>::operator new( std::size_t size, const std::nothrow_t& )
{
   MAYBE_UNUSED( size );

   BLAZE_INTERNAL_ASSERT( size == sizeof( StaticTensor ), "Invalid number of bytes detected" );

   return allocate<StaticTensor>( 1UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of the no-throw operator new[].
//
// \param size The total number of bytes to be allocated.
// \return Pointer to the newly allocated memory.
// \exception std::bad_alloc Allocation failed.
//
// This class-specific implementation of operator new provides the functionality to allocate
// dynamic memory based on the alignment restrictions of the StaticTensor class template.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void* StaticTensor<Type,O,M,N>::operator new[]( std::size_t size, const std::nothrow_t& )
{
   BLAZE_INTERNAL_ASSERT( size >= sizeof( StaticTensor )       , "Invalid number of bytes detected" );
   BLAZE_INTERNAL_ASSERT( size %  sizeof( StaticTensor ) == 0UL, "Invalid number of bytes detected" );

   return allocate<StaticTensor>( size/sizeof(StaticTensor) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of operator delete.
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void StaticTensor<Type,O,M,N>::operator delete( void* ptr )
{
   deallocate( static_cast<StaticTensor*>( ptr ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of operator delete[].
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void StaticTensor<Type,O,M,N>::operator delete[]( void* ptr )
{
   deallocate( static_cast<StaticTensor*>( ptr ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of no-throw operator delete.
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void StaticTensor<Type,O,M,N>::operator delete( void* ptr, const std::nothrow_t& )
{
   deallocate( static_cast<StaticTensor*>( ptr ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Class specific implementation of no-throw operator delete[].
//
// \param ptr The memory to be deallocated.
// \return void
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline void StaticTensor<Type,O,M,N>::operator delete[]( void* ptr, const std::nothrow_t& )
{
   deallocate( static_cast<StaticTensor*>( ptr ) );
}
//*************************************************************************************************




//=================================================================================================
//
//  DEBUGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the invariants of the static tensor are intact.
//
// \return \a true in case the static tensor's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the static tensor are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr bool StaticTensor<Type,O,M,N>::isIntact() const noexcept
{
   if( IsNumeric_v<Type> ) {
      for( size_t k=0UL; k<O; ++k ) {
         for( size_t i=0UL; i<M; ++i ) {
            for( size_t j=N; j<NN; ++j ) {
               if( v_[(k*M+i)*NN+j] != Type() )
                  return false;
            }
         }
      }
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
/*!\brief Returns whether the tensor can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this tensor, \a false if not.
//
// This function returns whether the given address can alias with the tensor. In contrast
// to the isAliased() function this function is allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename Other >  // Data type of the foreign expression
inline bool StaticTensor<Type,O,M,N>::canAlias( const Other* alias ) const noexcept
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
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename Other >  // Data type of the foreign expression
inline bool StaticTensor<Type,O,M,N>::isAliased( const Other* alias ) const noexcept
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
inline constexpr bool StaticTensor<Type,O,M,N>::isAligned() noexcept
{
   return align;
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
// of columns. Additionally, the column index (in case of a row-major tensor) or the row index
// (in case of a column-major tensor) must be a multiple of the number of values inside the
// SIMD element. This function must \b NOT be called explicitly! It is used internally for the
// performance optimized evaluation of expression templates. Calling this function explicitly
// might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE typename StaticTensor<Type,O,M,N>::SIMDType
   StaticTensor<Type,O,M,N>::load( size_t k, size_t i, size_t j ) const noexcept
{
   if( align )
      return loada( k, i, j );
   else
      return loadu( k, i, j );
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
// than the number of columns. Additionally, the column index (in case of a row-major tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE typename StaticTensor<Type,O,M,N>::SIMDType
   StaticTensor<Type,O,M,N>::loada( size_t k, size_t i, size_t j ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( k < O, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( i < M, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < N, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= NN, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( &v_[(k*M+i)*NN+j] ), "Invalid alignment detected" );

   return loada( &v_[(k*M+i)*NN+j] );
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
// than the number of columns. Additionally, the column index (in case of a row-major tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE typename StaticTensor<Type,O,M,N>::SIMDType
   StaticTensor<Type,O,M,N>::loadu( size_t k, size_t i, size_t j ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( k < O, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( i < M, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < N, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= NN, "Invalid column access index" );

   return loadu( &v_[(k*M+i)*NN+j] );
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
// of columns. Additionally, the column index (in case of a row-major tensor) or the row index
// (in case of a column-major tensor) must be a multiple of the number of values inside the
// SIMD element. This function must \b NOT be called explicitly! It is used internally for the
// performance optimized evaluation of expression templates. Calling this function explicitly
// might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE void
   StaticTensor<Type,O,M,N>::store( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   if( align )
      storea( k, i, j, value );
   else
      storeu( k, i, j, value );
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
// than the number of columns. Additionally, the column index (in case of a row-major tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE void
   StaticTensor<Type,O,M,N>::storea( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( k < O, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( i < M, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < N, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= NN, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( &v_[(k*M+i)*NN+j] ), "Invalid alignment detected" );

   storea( &v_[(k*M+i)*NN+j], value );
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
// than the number of columns. Additionally, the column index (in case of a row-major tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE void
   StaticTensor<Type,O,M,N>::storeu( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( k < O, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( i < M, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < N, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= NN, "Invalid column access index" );

   storeu( &v_[(k*M+i)*NN+j], value );
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
// row-major tensor) or the row index (in case of a column-major tensor) must be a multiple
// of the number of values inside the SIMD element. This function must \b NOT be called
// explicitly! It is used internally for the performance optimized evaluation of expression
// templates. Calling this function explicitly might result in erroneous results and/or in
// compilation errors.
*/
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
BLAZE_ALWAYS_INLINE void
   StaticTensor<Type,O,M,N>::stream( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( k < O, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( i < M, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < N, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= NN, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || j % SIMDSIZE == 0UL, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( checkAlignment( &v_[(k*M+i)*NN+j] ), "Invalid alignment detected" );

   stream( &v_[(k*M+i)*NN+j], value );
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side dense tensor
inline auto StaticTensor<Type,O,M,N>::assign( const DenseTensor<MT>& rhs )
   -> DisableIf_t< VectorizedAssign_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( (~rhs).pages() == O && (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i ) {
         for( size_t j=0UL; j<N; ++j ) {
            v_[(k*M+i)*NN+j] = (~rhs)(k,i,j);
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side dense tensor
inline auto StaticTensor<Type,O,M,N>::assign( const DenseTensor<MT>& rhs )
   -> EnableIf_t< VectorizedAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( (~rhs).pages() == O && (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );

   constexpr bool remainder( !usePadding || !IsPadded_v<MT> );

   const size_t jpos( ( remainder )?( N & size_t(-SIMDSIZE) ):( N ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i )
      {
         size_t j( 0UL );

         for( ; j<jpos; j+=SIMDSIZE ) {
            store( k, i, j, (~rhs).load(k,i,j) );
         }
         for( ; remainder && j<N; ++j ) {
            v_[(k*M+i)*NN+j] = (~rhs)(k,i,j);
         }
      }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a row-major sparse tensor.
//
// \param rhs The right-hand side sparse tensor to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// template< typename MT >  // Type of the right-hand side sparse tensor
// inline void StaticTensor<Type,O,M,N>::assign( const SparseTensor<MT>& rhs )
// {
//    BLAZE_INTERNAL_ASSERT( (~rhs).pages() == O && (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );
//
//    for( size_t k=0UL; k<O; ++k )
//       for( size_t i=0UL; i<M; ++i )
//          for( ConstIterator_t<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
//             v_[(k*M+i)*NN+element->index()] = element->value();
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a column-major sparse tensor.
//
// \param rhs The right-hand side sparse tensor to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// template< typename MT >  // Type of the right-hand side sparse tensor
// inline void StaticTensor<Type,O,M,N>::assign( const SparseTensor<MT,!SO>& rhs )
// {
//    BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_TENSOR_TYPE( MT );
//
//    BLAZE_INTERNAL_ASSERT( (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );
//
//    for( size_t j=0UL; j<N; ++j )
//       for( ConstIterator_t<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
//          v_[element->index()*NN+j] = element->value();
// }
//*************************************************************************************************


//*************************************************************************************************
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
template< typename Type  // Data type of the tensor
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side dense tensor
inline auto StaticTensor<Type,O,M,N>::addAssign( const DenseTensor<MT>& rhs )
   -> DisableIf_t< VectorizedAddAssign_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( (~rhs).pages() == O && (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i )
      {
//          if( IsDiagonal_v<MT> )
//          {
//             v_[(k*M+i)*NN+i] += (~rhs)(i,i,i);
//          }
//          else
         {
            const size_t jbegin( 0UL );
            const size_t jend  ( N );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            for( size_t j=jbegin; j<jend; ++j ) {
               v_[(k*M+i)*NN+j] += (~rhs)(k,i,j);
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side dense tensor
inline auto StaticTensor<Type,O,M,N>::addAssign( const DenseTensor<MT>& rhs )
   -> EnableIf_t< VectorizedAddAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
//    BLAZE_CONSTRAINT_MUST_NOT_BE_DIAGONAL_TENSOR_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( (~rhs).pages() == O && (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );

   constexpr bool remainder( !usePadding || !IsPadded_v<MT> );

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( 0UL );
         const size_t jend  ( N );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( ( remainder )?( jend & size_t(-SIMDSIZE) ):( jend ) );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            store( k, i, j, load(k,i,j) + (~rhs).load(k,i,j) );
         }
         for( ; remainder && j<jend; ++j ) {
            v_[(k*M+i)*NN+j] += (~rhs)(k,i,j);
         }
      }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a row-major sparse tensor.
//
// \param rhs The right-hand side sparse tensor to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// template< typename MT >  // Type of the right-hand side sparse tensor
// inline void StaticTensor<Type,O,M,N>::addAssign( const SparseTensor<MT>& rhs )
// {
//    BLAZE_INTERNAL_ASSERT( (~rhs).pages() == O && (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );
//
//    for( size_t k=0UL; k<O; ++k )
//       for( size_t i=0UL; i<M; ++i )
//          for( ConstIterator_t<MT> element=(~rhs).begin(i, k); element!=(~rhs).end(i, k); ++element )
//             v_[(k*M+i)*NN+element->index()] += element->value();
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a column-major sparse tensor.
//
// \param rhs The right-hand side sparse tensor to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// template< typename MT >  // Type of the right-hand side sparse tensor
// inline void StaticTensor<Type,O,M,N>::addAssign( const SparseTensor<MT,!SO>& rhs )
// {
//    BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_TENSOR_TYPE( MT );
//
//    BLAZE_INTERNAL_ASSERT( (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );
//
//    for( size_t j=0UL; j<N; ++j )
//       for( ConstIterator_t<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
//          v_[element->index()*NN+j] += element->value();
// }
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side dense tensor
inline auto StaticTensor<Type,O,M,N>::subAssign( const DenseTensor<MT>& rhs )
   -> DisableIf_t< VectorizedSubAssign_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( (~rhs).pages() == O && (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i )
      {
//          if( IsDiagonal_v<MT> )
//          {
//             v_[i*NN+i] -= (~rhs)(i,i);
//          }
//          else
         {
            const size_t jbegin( 0UL );
            const size_t jend  ( N );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            for( size_t j=jbegin; j<jend; ++j ) {
               v_[(k*M+i)*NN+j] -= (~rhs)(k,i,j);
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side dense tensor
inline auto StaticTensor<Type,O,M,N>::subAssign( const DenseTensor<MT>& rhs )
   -> EnableIf_t< VectorizedSubAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );
//    BLAZE_CONSTRAINT_MUST_NOT_BE_DIAGONAL_TENSOR_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );

   constexpr bool remainder( !usePadding || !IsPadded_v<MT> );

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( 0UL );
         const size_t jend  ( N );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jpos( ( remainder )?( jend & size_t(-SIMDSIZE) ):( jend ) );
         BLAZE_INTERNAL_ASSERT( !remainder || ( jend - ( jend % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( jbegin );

         for( ; j<jpos; j+=SIMDSIZE ) {
            store( k, i, j, load(k,i,j) - (~rhs).load(k,i,j) );
         }
         for( ; remainder && j<jend; ++j ) {
            v_[(k*M+i)*NN+j] -= (~rhs)(k,i,j);
         }
      }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a row-major sparse tensor.
//
// \param rhs The right-hand side sparse tensor to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// template< typename MT >  // Type of the right-hand side sparse tensor
// inline void StaticTensor<Type,O,M,N>::subAssign( const SparseTensor<MT>& rhs )
// {
//    BLAZE_INTERNAL_ASSERT( (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );
//
//    for( size_t i=0UL; i<M; ++i )
//       for( ConstIterator_t<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
//          v_[i*NN+element->index()] -= element->value();
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a column-major sparse tensor.
//
// \param rhs The right-hand side sparse tensor to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// template< typename MT >  // Type of the right-hand side sparse tensor
// inline void StaticTensor<Type,O,M,N>::subAssign( const SparseTensor<MT,!SO>& rhs )
// {
//    BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_TENSOR_TYPE( MT );
//
//    BLAZE_INTERNAL_ASSERT( (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );
//
//    for( size_t j=0UL; j<N; ++j )
//       for( ConstIterator_t<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
//          v_[element->index()*NN+j] -= element->value();
// }
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side dense tensor
inline auto StaticTensor<Type,O,M,N>::schurAssign( const DenseTensor<MT>& rhs )
   -> DisableIf_t< VectorizedSchurAssign_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( (~rhs).pages() == O && (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i ) {
         for( size_t j=0UL; j<N; ++j ) {
            v_[(k*M+i)*NN+j] *= (~rhs)(k,i,j);
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
        , size_t O       // Number of pages
        , size_t M       // Number of rows
        , size_t N >     // Number of columns
template< typename MT >  // Type of the right-hand side dense tensor
inline auto StaticTensor<Type,O,M,N>::schurAssign( const DenseTensor<MT>& rhs )
   -> EnableIf_t< VectorizedSchurAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );

   constexpr bool remainder( !usePadding || !IsPadded_v<MT> );

   for( size_t k=0UL; k<O; ++k )
      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jpos( ( remainder )?( N & size_t(-SIMDSIZE) ):( N ) );
         BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

         size_t j( 0UL );

         for( ; j<jpos; j+=SIMDSIZE ) {
            store( k, i, j, load(k,i,j) * (~rhs).load(k,i,j) );
         }
         for( ; remainder && j<N; ++j ) {
            v_[(k*M+i)*NN+j] *= (~rhs)(k,i,j);
         }
      }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the Schur product assignment of a row-major sparse tensor.
//
// \param rhs The right-hand side sparse tensor for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// template< typename MT >  // Type of the right-hand side sparse tensor
// inline void StaticTensor<Type,O,M,N>::schurAssign( const SparseTensor<MT>& rhs )
// {
//    BLAZE_INTERNAL_ASSERT( (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );
//
//    const StaticTensor tmp( serial( *this ) );
//
//    reset();
//
//    for( size_t i=0UL; i<M; ++i )
//       for( ConstIterator_t<MT> element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
//          v_[i*NN+element->index()] = tmp.v_[i*NN+element->index()] * element->value();
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the Schur product assignment of a column-major sparse tensor.
//
// \param rhs The right-hand side sparse tensor for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
// template< typename Type  // Data type of the tensor
//         , size_t O       // Number of pages
//         , size_t M       // Number of rows
//         , size_t N >     // Number of columns
// template< typename MT >  // Type of the right-hand side sparse tensor
// inline void StaticTensor<Type,O,M,N>::schurAssign( const SparseTensor<MT,!SO>& rhs )
// {
//    BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_TENSOR_TYPE( MT );
//
//    BLAZE_INTERNAL_ASSERT( (~rhs).rows() == M && (~rhs).columns() == N, "Invalid tensor size" );
//
//    const StaticTensor tmp( serial( *this ) );
//
//    reset();
//
//    for( size_t j=0UL; j<N; ++j )
//       for( ConstIterator_t<MT> element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
//          v_[element->index()*NN+j] = tmp.v_[element->index()*NN+j] * element->value();
// }
//*************************************************************************************************








//=================================================================================================
//
//  SIZE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, size_t O, size_t M, size_t N >
struct Size< StaticTensor<T,O,M,N>, 0UL >
   : public Ptrdiff_t<O>
{};

template< typename T, size_t O, size_t M, size_t N >
struct Size< StaticTensor<T,O,M,N>, 1UL >
   : public Ptrdiff_t<M>
{};

template< typename T, size_t O, size_t M, size_t N >
struct Size< StaticTensor<T,O,M,N>, 2UL >
   : public Ptrdiff_t<N>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MAXSIZE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, size_t O, size_t M, size_t N >
struct MaxSize< StaticTensor<T,O,M,N>, 0UL >
   : public Ptrdiff_t<O>
{};

template< typename T, size_t O, size_t M, size_t N >
struct MaxSize< StaticTensor<T,O,M,N>, 1UL >
   : public Ptrdiff_t<M>
{};

template< typename T, size_t O, size_t M, size_t N >
struct MaxSize< StaticTensor<T,O,M,N>, 2UL >
   : public Ptrdiff_t<N>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSQUARE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
// template< typename T, size_t N >
// struct IsSquare< StaticTensor<T,N,N> >
//    : public TrueType
// {};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HASCONSTDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, size_t O, size_t M, size_t N >
struct HasConstDataAccess< StaticTensor<T,O,M,N> >
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
template< typename T, size_t O, size_t M, size_t N >
struct HasMutableDataAccess< StaticTensor<T,O,M,N> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSTATIC SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, size_t O, size_t M, size_t N >
struct IsStatic< StaticTensor<T,O,M,N> >
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
template< typename T, size_t O, size_t M, size_t N >
struct IsAligned< StaticTensor<T,O,M,N> >
   : public BoolConstant< StaticTensor<T,O,M,N>::isAligned() >
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
template< typename T, size_t O, size_t M, size_t N >
struct IsContiguous< StaticTensor<T,O,M,N> >
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
template< typename T, size_t O, size_t M, size_t N >
struct IsPadded< StaticTensor<T,O,M,N> >
   : public BoolConstant<usePadding>
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
template< typename T1, typename T2 >
struct AddTraitEval2< T1, T2
                    , EnableIf_t< IsTensor_v<T1> &&
                                  IsTensor_v<T2> &&
                                  ( Size_v<T1,0UL> != DefaultSize_v ||
                                    Size_v<T2,0UL> != DefaultSize_v ) &&
                                  ( Size_v<T1,1UL> != DefaultSize_v ||
                                    Size_v<T2,1UL> != DefaultSize_v ) &&
                                  ( Size_v<T1,2UL> != DefaultSize_v ||
                                    Size_v<T2,2UL> != DefaultSize_v ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   static constexpr size_t O = max( Size_v<T1,0UL>, Size_v<T2,0UL> );
   static constexpr size_t M = max( Size_v<T1,1UL>, Size_v<T2,1UL> );
   static constexpr size_t N = max( Size_v<T1,2UL>, Size_v<T2,2UL> );

   using Type = StaticTensor< AddTrait_t<ET1,ET2>, O, M, N >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2 >
struct SubTraitEval2< T1, T2
                    , EnableIf_t< IsTensor_v<T1> &&
                                  IsTensor_v<T2> &&
                                  ( Size_v<T1,0UL> != DefaultSize_v ||
                                    Size_v<T2,0UL> != DefaultSize_v ) &&
                                  ( Size_v<T1,1UL> != DefaultSize_v ||
                                    Size_v<T2,1UL> != DefaultSize_v ) &&
                                  ( Size_v<T1,2UL> != DefaultSize_v ||
                                    Size_v<T2,2UL> != DefaultSize_v ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   static constexpr size_t O = max( Size_v<T1,0UL>, Size_v<T2,0UL> );
   static constexpr size_t M = max( Size_v<T1,1UL>, Size_v<T2,1UL> );
   static constexpr size_t N = max( Size_v<T1,2UL>, Size_v<T2,2UL> );

   using Type = StaticTensor< SubTrait_t<ET1,ET2>, O, M, N >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SCHURTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2 >
struct SchurTraitEval2< T1, T2
                      , EnableIf_t< IsDenseTensor_v<T1> &&
                                    IsDenseTensor_v<T2> &&
                                  ( Size_v<T1,0UL> != DefaultSize_v ||
                                    Size_v<T2,0UL> != DefaultSize_v ) &&
                                  ( Size_v<T1,1UL> != DefaultSize_v ||
                                    Size_v<T2,1UL> != DefaultSize_v ) &&
                                  ( Size_v<T1,2UL> != DefaultSize_v ||
                                    Size_v<T2,2UL> != DefaultSize_v ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   static constexpr size_t O = max( Size_v<T1,0UL>, Size_v<T2,0UL> );
   static constexpr size_t M = max( Size_v<T1,1UL>, Size_v<T2,1UL> );
   static constexpr size_t N = max( Size_v<T1,2UL>, Size_v<T2,2UL> );

   using Type = StaticTensor< MultTrait_t<ET1,ET2>, O, M, N >;
};

template< typename T1, typename T2 >
struct SchurTraitEval2< T1, T2
                      , EnableIf_t< IsDenseTensor_v<T1> &&
                                    IsDenseMatrix_v<T2> &&
                                  ( Size_v<T1,0UL> != DefaultSize_v &&
                                    Size_v<T2,0UL> != DefaultSize_v ) &&
                                  ( Size_v<T1,1UL> != DefaultSize_v &&
                                    Size_v<T2,1UL> != DefaultSize_v ) &&
                                  ( Size_v<T1,2UL> != DefaultSize_v ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   static constexpr size_t O = Size_v<T1,0UL>;
   static constexpr size_t M = max( Size_v<T1,1UL>, Size_v<T2,0UL> );
   static constexpr size_t N = max( Size_v<T1,2UL>, Size_v<T2,1UL> );

   using Type = StaticTensor< MultTrait_t<ET1,ET2>, O, M, N >;
};
/*! \endcond */
//*************************************************************************************************


//=================================================================================================
//
//  MULTTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2 >
struct MultTraitEval2< T1, T2
                     , EnableIf_t< IsTensor_v<T1> &&
                                   IsNumeric_v<T2> &&
                                   ( Size_v<T1,0UL> != DefaultSize_v ) &&
                                   ( Size_v<T1,1UL> != DefaultSize_v ) &&
                                   ( Size_v<T1,2UL> != DefaultSize_v ) > >
{
   using ET1 = ElementType_t<T1>;

   static constexpr size_t O = Size_v<T1,0UL>;
   static constexpr size_t M = Size_v<T1,1UL>;
   static constexpr size_t N = Size_v<T1,2UL>;

   using Type = StaticTensor< MultTrait_t<ET1,T2>, O, M, N >;
};

template< typename T1, typename T2 >
struct MultTraitEval2< T1, T2
                     , EnableIf_t< IsNumeric_v<T1> &&
                                   IsTensor_v<T2> &&
                                   ( Size_v<T2,0UL> != DefaultSize_v ) &&
                                   ( Size_v<T2,1UL> != DefaultSize_v ) &&
                                   ( Size_v<T2,2UL> != DefaultSize_v ) > >
{
   using ET2 = ElementType_t<T2>;

   static constexpr size_t O = Size_v<T2,0UL>;
   static constexpr size_t M = Size_v<T2,0UL>;
   static constexpr size_t N = Size_v<T2,1UL>;

   using Type = StaticTensor< MultTrait_t<T1,ET2>, O, M, N >;
};

template< typename T1, typename T2 >
struct MultTraitEval2< T1, T2
                     , EnableIf_t< IsTensor_v<T1> &&
                                   IsTensor_v<T2> &&
                                   ( Size_v<T1,0UL> != DefaultSize_v ||
                                     ( IsSquare_v<T1> && Size_v<T2,1UL> != DefaultSize_v ) ) &&
                                   ( Size_v<T2,1UL> != DefaultSize_v ||
                                     ( IsSquare_v<T2> && Size_v<T1,2UL> != DefaultSize_v ) ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   static constexpr size_t M = ( Size_v<T1,0UL> != DefaultSize_v ? Size_v<T1,1UL> : Size_v<T2,1UL> );
   static constexpr size_t N = ( Size_v<T2,1UL> != DefaultSize_v ? Size_v<T2,2UL> : Size_v<T1,2UL> );

   using Type = StaticTensor< MultTrait_t<ET1,ET2>, Size_v<T1,0UL>, M, N >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DIVTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2 >
struct DivTraitEval2< T1, T2
                    , EnableIf_t< IsTensor_v<T1> &&
                                  IsNumeric_v<T2> &&
                                  ( Size_v<T1,0UL> != DefaultSize_v ) &&
                                  ( Size_v<T1,1UL> != DefaultSize_v ) &&
                                  ( Size_v<T1,2UL> != DefaultSize_v ) > >
{
   using ET1 = ElementType_t<T1>;

   static constexpr size_t O = Size_v<T1,0UL>;
   static constexpr size_t M = Size_v<T1,1UL>;
   static constexpr size_t N = Size_v<T1,2UL>;

   using Type = StaticTensor< DivTrait_t<ET1,T2>, O, M, N >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DILATEDSUBTENSORTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename TT, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N,
                     size_t PageDilation, size_t RowDilation, size_t ColumnDilation >
struct DilatedSubtensorTraitEval2< TT, K, I, J, O, M, N, PageDilation, RowDilation, ColumnDilation
                          , EnableIf_t< K != inf && I != inf && J != inf &&
                                        O != inf && M != inf && N != inf &&
                                        PageDilation != inf && RowDilation != inf && ColumnDilation != inf &&
                                        IsDenseTensor_v<TT> > >
{
   using Type = StaticTensor< RemoveConst_t< ElementType_t<TT> >, O, M, N>;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MAPTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, typename OP >
struct UnaryMapTraitEval2< T, OP
                         , EnableIf_t< IsTensor_v<T> &&
                                  ( Size_v<T,0UL> != DefaultSize_v ) &&
                                  ( Size_v<T,1UL> != DefaultSize_v ) &&
                                  ( Size_v<T,2UL> != DefaultSize_v ) > >
{
   using ET = ElementType_t<T>;

   using Type = StaticTensor< MapTrait_t<ET,OP>, Size_v<T,0UL>, Size_v<T,1UL>, Size_v<T,2UL> >;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2, typename OP >
struct BinaryMapTraitEval2< T1, T2, OP
                          , EnableIf_t< IsTensor_v<T1> &&
                                        IsTensor_v<T2> &&
                                        ( Size_v<T1,0UL> != DefaultSize_v ||
                                          Size_v<T2,0UL> != DefaultSize_v ) &&
                                        ( Size_v<T1,1UL> != DefaultSize_v ||
                                          Size_v<T2,1UL> != DefaultSize_v ) &&
                                        ( Size_v<T1,2UL> != DefaultSize_v ||
                                          Size_v<T2,2UL> != DefaultSize_v ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   static constexpr size_t O = max( Size_v<T1,0UL>, Size_v<T2,0UL> );
   static constexpr size_t M = max( Size_v<T1,1UL>, Size_v<T2,1UL> );
   static constexpr size_t N = max( Size_v<T1,2UL>, Size_v<T2,2UL> );

   using Type = StaticTensor< MapTrait_t<ET1,ET2,OP>, O, M, N >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  RAVELTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T > // Type to be expanded
struct RavelTraitEval2< T
                       , EnableIf_t< IsDenseTensor_v<T> &&
                                     ( Size_v<T,0UL> != DefaultSize_v ) &&
                                     ( MaxSize_v<T,0UL> != DefaultMaxSize_v ) &&
                                     ( Size_v<T,1UL> != DefaultSize_v ) &&
                                     ( MaxSize_v<T,1UL> != DefaultMaxSize_v ) &&
                                     ( Size_v<T,2UL> != DefaultSize_v ) &&
                                     ( MaxSize_v<T,2UL> != DefaultMaxSize_v ) > >
{
   using Type = StaticVector< ElementType_t<T>, Size_v<T,0UL> * Size_v<T,1UL> * Size_v<T,2UL>, rowVector >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HIGHTYPE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, size_t O, size_t M, size_t N, typename T2 >
struct HighType< StaticTensor<T1,O,M,N>, StaticTensor<T2,O,M,N> >
{
   using Type = StaticTensor< typename HighType<T1,T2>::Type, O, M, N >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LOWTYPE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, size_t O, size_t M, size_t N, typename T2 >
struct LowType< StaticTensor<T1,O,M,N>, StaticTensor<T2,O,M,N> >
{
   using Type = StaticTensor< typename LowType<T1,T2>::Type, O, M, N >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COLUMNSLICETRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template <typename MT, size_t M>
struct ColumnSliceTraitEval2<
   MT, M,
   EnableIf_t< IsDenseTensor_v<MT> &&
               M != 0UL && Size_v< MT,0UL > != DefaultSize_v &&
                           Size_v< MT,1UL > != DefaultSize_v > >
{
   using Type = StaticMatrix< RemoveConst_t< ElementType_t<MT> >, Size_v< MT,0UL >, Size_v< MT,1UL >, rowMajor >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  PAGESLICETRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template <typename MT, size_t M>
struct PageSliceTraitEval2<
   MT, M,
   EnableIf_t< IsDenseTensor_v<MT> &&
               M != 0UL && Size_v< MT,1UL > != DefaultSize_v &&
                           Size_v< MT,2UL > != DefaultSize_v > >
{
   using Type = StaticMatrix< RemoveConst_t< ElementType_t<MT> >, Size_v< MT,1UL >, Size_v< MT,2UL >, rowMajor >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ROWSLICETRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template <typename MT, size_t M>
struct RowSliceTraitEval2<
   MT, M,
   EnableIf_t< IsDenseTensor_v<MT> &&
               M != 0UL && Size_v< MT,0UL > != DefaultSize_v &&
                           Size_v< MT,2UL > != DefaultSize_v > >
{
   using Type = StaticMatrix< RemoveConst_t< ElementType_t<MT> >, Size_v< MT,2UL >, Size_v< MT,0UL >, columnMajor >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBTENSORTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, size_t K, size_t I, size_t J, size_t O, size_t M, size_t N >
struct SubtensorTraitEval2< MT, K, I, J, O, M, N
                          , EnableIf_t< K != inf && I != inf && J != inf && O != inf && M != inf && N != inf &&
                                        IsDenseTensor_v<MT> > >
{
   using Type = StaticTensor< RemoveConst_t< ElementType_t<MT> >, O, M, N >;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT >
struct SubtensorTraitEval2< MT, inf, inf, inf, inf, inf, inf,
   EnableIf_t< IsDenseTensor_v<MT> && (
               ( Size_v< MT,0UL > != DefaultSize_v &&
                 Size_v< MT,1UL > != DefaultSize_v &&
                 Size_v< MT,2UL > != DefaultSize_v ) ||
               ( MaxSize_v< MT,0UL > != DefaultMaxSize_v &&
                 MaxSize_v< MT,1UL > != DefaultMaxSize_v &&
                 MaxSize_v< MT,2UL > != DefaultMaxSize_v ) ) > >
{
//    static constexpr size_t O = max( Size_v<MT,0UL>, MaxSize_v<MT,0UL> );
//    static constexpr size_t M = max( Size_v<MT,1UL>, MaxSize_v<MT,1UL> );
//    static constexpr size_t N = max( Size_v<MT,2UL>, MaxSize_v<MT,2UL> );

   // FIXME: change this to HybridTensor, once available
   using Type = DynamicTensor< RemoveConst_t< ElementType_t<MT> > >;
};
/*! \endcond */
//*************************************************************************************************



//=================================================================================================
//
//  ROWSTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
// template< typename MT, size_t M >
// struct RowSlicesTraitEval2< MT, M
//                      , EnableIf_t< M != 0UL &&
//                                    IsDenseTensor_v<MT> &&
//                                    Size_v<MT,1UL> != DefaultSize_v > >
// {
//    using Type = StaticTensor< ElementType_t<MT>, M, Size_v<MT,1UL>, false >;
// };
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COLUMNSTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
// template< typename MT, size_t N >
// struct ColumnSlicesTraitEval2< MT, N
//                         , EnableIf_t< N != 0UL &&
//                                       IsDenseTensor_v<MT> &&
//                                       Size_v<MT,0UL> != DefaultSize_v > >
// {
//    using Type = StaticTensor< ElementType_t<MT>, Size_v<MT,0UL>, N, true >;
// };
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
