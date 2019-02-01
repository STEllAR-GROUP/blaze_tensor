//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/DynamicTensor.h
//  \brief Header file for the implementation of a dynamic MxN tensor
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_DYNAMICTENSOR_H_
#define _BLAZE_TENSOR_MATH_DENSE_DYNAMICTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/dense/DynamicMatrix.h>

#include <blaze_tensor/math/Forward.h>
#include <blaze_tensor/math/InitializerList.h>
#include <blaze_tensor/math/SMP.h>
#include <blaze_tensor/math/Tensor.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>
#include <blaze_tensor/math/traits/ColumnSliceTrait.h>
#include <blaze_tensor/math/traits/PageSliceTrait.h>
#include <blaze_tensor/math/traits/RowSliceTrait.h>
#include <blaze_tensor/math/traits/SubtensorTrait.h>
#include <blaze_tensor/math/typetraits/IsDenseTensor.h>
#include <blaze_tensor/math/typetraits/IsRowMajorTensor.h>
#include <blaze_tensor/math/typetraits/IsTensor.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup dynamic_tensor DynamicTensor
// \ingroup dense_tensor
*/
/*!\brief Efficient implementation of a dynamic \f$ M \times N \f$ tensor.
// \ingroup dynamic_tensor
//
// The DynamicTensor class template is the representation of an arbitrary sized tensor with
// \f$ M \times N \f$ dynamically allocated elements of arbitrary type. The type of the elements
// and the storage order of the tensor can be specified via the two template parameters:

   \code
   template< typename Type >
   class DynamicTensor;
   \endcode

//  - Type: specifies the type of the tensor elements. DynamicTensor can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//
// The use of DynamicTensor is very natural and intuitive. All operations (addition, subtraction,
// multiplication, scaling, ...) can be performed on all possible combinations of matrices with
// fitting element types. The following example an impression of the use of DynamicTensor:

   \code
   using blaze::DynamicTensor;
   using blaze::CompressedTensor;

   DynamicTensor<double> A( 2, 3, 4 );              // Default constructed, non-initialized, 2x3 tensor
   A(0,0,0) = 1.0; A(0,0,1) = 2.0; A(0,0,2) = 3.0;  // Initialization of the first row
   A(0,1,0) = 4.0; A(0,1,1) = 5.0; A(0,1,2) = 6.0;  // Initialization of the second row

   DynamicTensor<float> B( 2, 3 );              // Default constructed column-major single precision 2x3 tensor
   B(0,0) = 1.0; B(0,1) = 3.0; B(0,2) = 5.0;    // Initialization of the first row
   B(1,0) = 2.0; B(1,1) = 4.0; B(1,2) = 6.0;    // Initialization of the second row

   CompressedTensor<float> C( 2, 3 );        // Empty sparse single precision tensor
   DynamicTensor<float>    D( 3, 2, 4.0F );  // Directly, homogeneously initialized single precision 3x2 tensor

   DynamicTensor<double>    E( A );          // Creation of a new tensor as a copy of A
   DynamicTensor<double> F;                  // Creation of a default column-major tensor

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
template< typename Type >                   // Data type of the tensor
class DynamicTensor
   : public DenseTensor< DynamicTensor<Type> >
{
 public:
   //**Type definitions****************************************************************************
   using This          = DynamicTensor<Type>;       //!< Type of this DynamicTensor instance.
   using BaseType      = DenseTensor<This>;         //!< Base type of this DynamicTensor instance.
   using ResultType    = This;                      //!< Result type for expression template evaluations.
   using OppositeType  = DynamicTensor<Type>;       //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = DynamicTensor<Type>;       //!< Transpose type for expression template evaluations.
   using ElementType   = Type;                      //!< Type of the tensor elements.
   using SIMDType      = SIMDTrait_t<ElementType>;  //!< SIMD type of the tensor elements.
   using ReturnType    = const Type&;               //!< Return type for expression template evaluations.
   using CompositeType = const This&;               //!< Data type for composite expression templates.

   using Reference      = Type&;        //!< Reference to a non-constant tensor value.
   using ConstReference = const Type&;  //!< Reference to a constant tensor value.
   using Pointer        = Type*;        //!< Pointer to a non-constant tensor value.
   using ConstPointer   = const Type*;  //!< Pointer to a constant tensor value.

   using Iterator      = DenseIterator<Type,usePadding>;        //!< Iterator over non-constant elements.
   using ConstIterator = DenseIterator<const Type,usePadding>;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a DynamicTensor with different data/element type.
   */
   template< typename NewType >  // Data type of the other tensor
   struct Rebind {
      using Other = DynamicTensor<NewType>;  //!< The type of the other DynamicTensor.
   };
   //**********************************************************************************************

   //**Resize struct definition********************************************************************
   /*!\brief Resize mechanism to obtain a DynamicTensor with different fixed dimensions.
   */
   template< size_t NewO    // Number of pages of the other tensor
           , size_t NewM    // Number of rows of the other tensor
           , size_t NewN >  // Number of columns of the other tensor
   struct Resize {
      using Other = DynamicTensor<Type>;  //!< The type of the other DynamicTensor.
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
   explicit inline DynamicTensor() noexcept;
   explicit inline DynamicTensor( size_t o, size_t m, size_t n );
   explicit inline DynamicTensor( size_t o, size_t m, size_t n, const Type& init );
   explicit inline DynamicTensor( initializer_list< initializer_list< initializer_list<Type> > > list );

   template< typename Other >
   explicit inline DynamicTensor( size_t o, size_t m, size_t n, const Other* array );

   template< typename Other, size_t Rows, size_t Cols, size_t Pages >
   explicit inline DynamicTensor( const Other (&array)[Pages][Rows][Cols] );

                                     inline DynamicTensor( const DynamicTensor& m );
                                     inline DynamicTensor( DynamicTensor&& m ) noexcept;
   template< typename MT>            inline DynamicTensor( const Tensor<MT>& m );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~DynamicTensor();
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator()( size_t k, size_t i, size_t j ) noexcept;
   inline ConstReference operator()( size_t k, size_t i, size_t j ) const noexcept;
   inline Reference      at( size_t k, size_t i, size_t j );
   inline ConstReference at( size_t k, size_t i, size_t j ) const;
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
   inline DynamicTensor& operator=( const Type& rhs );
   inline DynamicTensor& operator=( initializer_list< initializer_list< initializer_list<Type> > > list );

   template< typename Other, size_t Rows, size_t Cols, size_t Pages >
   inline DynamicTensor& operator=( const Other (&array)[Pages][Rows][Cols] );

   inline DynamicTensor& operator=( const DynamicTensor& rhs );
   inline DynamicTensor& operator=( DynamicTensor&& rhs ) noexcept;

   template< typename MT > inline DynamicTensor& operator= ( const Tensor<MT>& rhs );
   template< typename MT > inline DynamicTensor& operator+=( const Tensor<MT>& rhs );
   template< typename MT > inline DynamicTensor& operator-=( const Tensor<MT>& rhs );
   template< typename MT > inline DynamicTensor& operator%=( const Tensor<MT>& rhs );
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
          void   resize ( size_t o, size_t m, size_t n, bool preserve=true );
   inline void   extend ( size_t o, size_t m, size_t n, bool preserve=true );
   inline void   reserve( size_t elements );
   inline void   shrinkToFit();
   inline void   swap( DynamicTensor& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   inline DynamicTensor& transpose();
   inline DynamicTensor& ctranspose();
   template < typename T >
   inline DynamicTensor& transpose( const T* indices, size_t n );
   template < typename T >
   inline DynamicTensor& ctranspose( const T* indices, size_t n );

   template< typename Other > inline DynamicTensor& scale( const Other& scalar );
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
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t addPadding( size_t value ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t o_;                //!< The current number of pages of the tensor.
   size_t m_;                //!< The current number of rows of the tensor.
   size_t n_;                //!< The current number of columns of the tensor.
   size_t nn_;               //!< The alignment adjusted number of columns.
   size_t capacity_;         //!< The maximum capacity of the tensor.
   Type* BLAZE_RESTRICT v_;  //!< The dynamically allocated tensor elements.
                             /*!< Access to the tensor elements is gained via the function call
                                  operator. The memory layout of the elements is
                                  \f[\left(\begin{array}{*{5}{c}}
                                  0            & 1             & 2             & \cdots & N-1         \\
                                  N            & N+1           & N+2           & \cdots & 2 \cdot N-1 \\
                                  \vdots       & \vdots        & \vdots        & \ddots & \vdots      \\
                                  M \cdot N-N  & M \cdot N-N+1 & M \cdot N-N+2 & \cdots & M \cdot N-1 \\
                                  \cdot repeated O times
                                  \end{array}\right)\f]. */
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
/*!\brief The default constructor for DynamicTensor.
*/
template< typename Type > // Data type of the tensor
inline DynamicTensor<Type>::DynamicTensor() noexcept
   : o_       ( 0UL )      // The current number of pages of the tensor
   , m_       ( 0UL )      // The current number of rows of the tensor
   , n_       ( 0UL )      // The current number of columns of the tensor
   , nn_      ( 0UL )      // The alignment adjusted number of columns
   , capacity_( 0UL )      // The maximum capacity of the tensor
   , v_       ( nullptr )  // The tensor elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a tensor of size \f$ m \times n \f$. No element initialization is performed!
//
// \param m The number of rows of the tensor.
// \param n The number of columns of the tensor.
//
// \note This constructor is only responsible to allocate the required dynamic memory. No
// element initialization is performed!
*/
template< typename Type > // Data type of the tensor
inline DynamicTensor<Type>::DynamicTensor( size_t o, size_t m, size_t n )
   : o_       ( o )                            // The current number of pages of the tensor
   , m_       ( m )                            // The current number of rows of the tensor
   , n_       ( n )                            // The current number of columns of the tensor
   , nn_      ( addPadding( n ) )              // The alignment adjusted number of columns
   , capacity_( m_*nn_*o_ )                    // The maximum capacity of the tensor
   , v_       ( allocate<Type>( capacity_ ) )  // The tensor elements
{
   if( IsVectorizable_v<Type> ) {
      for (size_t k=0UL; k<o_; ++k) {
         for (size_t i=0UL; i<m_; ++i) {
            size_t row_elements = (k*m_+i)*nn_;
            for (size_t j=n_; j<nn_; ++j) {
               v_[row_elements+j] = Type();
            }
         }
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a homogeneous initialization of all \f$ m \times n \f$ tensor elements.
//
// \param m The number of rows of the tensor.
// \param n The number of columns of the tensor.
// \param init The initial value of the tensor elements.
//
// All tensor elements are initialized with the specified value.
*/
template< typename Type > // Data type of the tensor
inline DynamicTensor<Type>::DynamicTensor( size_t o, size_t m, size_t n, const Type& init )
   : DynamicTensor( o, m, n )
{
   for (size_t k=0UL; k<o; ++k) {
      for (size_t i=0UL; i<m; ++i) {
         size_t row_elements = (k*m+i)*nn_;
         for (size_t j=0UL; j<n_; ++j) {
            v_[row_elements+j] = init;
         }
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List initialization of all tensor elements.
//
// \param list The initializer list.
//
// This constructor provides the option to explicitly initialize the elements of the tensor by
// means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::DynamicTensor<int> A{ { { 1, 2, 3 },
                                  { 4, 5 },
                                  { 7, 8, 9 } },
                                { { 1, 2, 3 },
                                  { 4, 5 },
                                  { 7, 8, 9 } } };
   \endcode

// The tensor is sized according to the size of the initializer list and all its elements are
// initialized by the values of the given initializer list. Missing values are initialized as
// default (as e.g. the value 6 in the example).
*/
template< typename Type > // Data type of the tensor
inline DynamicTensor<Type>::DynamicTensor( initializer_list< initializer_list< initializer_list<Type> > > list )
   : DynamicTensor( list.size(), determineRows( list ), determineColumns( list ) )
{
   size_t k( 0UL );

   for (const auto& page : list) {
      size_t i( 0UL );
      for (const auto& rowList : page) {
         std::fill(std::copy(rowList.begin(), rowList.end(), begin(i, k)), end(i, k), Type());
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

   int* array = new int[20];
   // ... Initialization of the dynamic array
   blaze::DynamicTensor<int> v( 6UL, 4UL, 5UL, array );
   delete[] array;
   \endcode

// The tensor is sized according to the given size of the array and initialized with the values
// from the given array. Note that it is expected that the given \a array has at least \a m by
// \a n elements. Providing an array with less elements results in undefined behavior!
*/
template< typename Type > // Data type of the tensor
template< typename Other >  // Data type of the initialization array
inline DynamicTensor<Type>::DynamicTensor( size_t o, size_t m, size_t n, const Other* array )
   : DynamicTensor( o, m, n )
{
   for (size_t k=0UL; k<o; ++k) {
      for (size_t i=0UL; i<m; ++i) {
         size_t row_elements = k*m+i;
         for (size_t j=0UL; j<n; ++j) {
            v_[row_elements*nn_+j] = array[row_elements*n+j];
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

   const int init[3][3][3] = { { { 1, 2, 3 },
                                 { 4, 5, 6 },
                                 { 7, 8, 9 } },
                               { { 1, 2, 3 },
                                 { 4, 5, 6 },
                                 { 7, 8, 9 } },
                               { { 1, 2, 3 },
                                 { 4, 5, 6 },
                                 { 7, 8, 9 } } };
   blaze::DynamicTensor<int> A( init );
   \endcode

// The tensor is sized according to the size of the array and initialized with the values from
// the given array. Missing values are initialized with default values (as e.g. the value 6 in
// the example).
*/
template< typename Type > // Data type of the tensor
template< typename Other  // Data type of the initialization array
        , size_t Rows     // Number of rows of the initialization array
        , size_t Cols     // Number of columns of the initialization array
        , size_t Pages >  // Number of pages of the initialization array
inline DynamicTensor<Type>::DynamicTensor( const Other (&array)[Pages][Rows][Cols] )
   : DynamicTensor( Pages, Rows, Cols )
{
   for (size_t k=0UL; k<Pages; ++k) {
      for (size_t i=0UL; i<Rows; ++i) {
         size_t row_elements = (k*Rows+i)*nn_;
         for (size_t j=0UL; j<Cols; ++j) {
            v_[row_elements+j] = array[k][i][j];
         }
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for DynamicTensor.
//
// \param m Tensor to be copied.
//
// The copy constructor is explicitly defined due to the required dynamic memory management
// and in order to enable/facilitate NRV optimization.
*/
template< typename Type > // Data type of the tensor
inline DynamicTensor<Type>::DynamicTensor( const DynamicTensor& m )
   : DynamicTensor( m.o_, m.m_, m.n_ )
{
   BLAZE_INTERNAL_ASSERT( capacity_ <= m.capacity_, "Invalid capacity estimation" );

   smpAssign( *this, m );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The move constructor for DynamicTensor.
//
// \param m The tensor to be move into this instance.
*/
template< typename Type > // Data type of the tensor
inline DynamicTensor<Type>::DynamicTensor( DynamicTensor&& m ) noexcept
   : o_       ( m.o_        )  // The current number of pages of the tensor
   , m_       ( m.m_        )  // The current number of rows of the tensor
   , n_       ( m.n_        )  // The current number of columns of the tensor
   , nn_      ( m.nn_       )  // The alignment adjusted number of columns
   , capacity_( m.capacity_ )  // The maximum capacity of the tensor
   , v_       ( m.v_        )  // The tensor elements
{
   m.o_        = 0UL;
   m.m_        = 0UL;
   m.n_        = 0UL;
   m.nn_       = 0UL;
   m.capacity_ = 0UL;
   m.v_        = nullptr;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different matrices.
//
// \param m Tensor to be copied.
*/
template< typename Type > // Data type of the tensor
template< typename MT >   // Type of the foreign tensor
inline DynamicTensor<Type>::DynamicTensor( const Tensor<MT>& m )
   : DynamicTensor( (~m).pages(), (~m).rows(), (~m).columns() )
{
   smpAssign( *this, ~m );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The destructor for DynamicTensor.
*/
template< typename Type > // Data type of the tensor
inline DynamicTensor<Type>::~DynamicTensor()
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
/*!\brief 2D-access to the tensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::Reference
   DynamicTensor<Type>::operator()( size_t k, size_t i, size_t j ) noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   BLAZE_USER_ASSERT( k<o_, "Invalid page access index" );
   return v_[(k*m_+i)*nn_+j];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief 2D-access to the tensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::ConstReference
   DynamicTensor<Type>::operator()( size_t k, size_t i, size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   BLAZE_USER_ASSERT( k<o_, "Invalid page access index" );
   return v_[(k*m_+i)*nn_+j];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the tensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid tensor access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::Reference
   DynamicTensor<Type>::at( size_t k, size_t i, size_t j )
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
   return (*this)(k,i,j);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the tensor elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid tensor access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::ConstReference
   DynamicTensor<Type>::at( size_t k, size_t i, size_t j ) const
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
   return (*this)(k,i,j);
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
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::Pointer
   DynamicTensor<Type>::data() noexcept
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
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::ConstPointer
   DynamicTensor<Type>::data() const noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the tensor elements of row/column \a i.
//
// \param i The row/column index.
// \param j The page index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::Pointer
   DynamicTensor<Type>::data( size_t i, size_t k ) noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor row access index" );
   return v_ + (k*m_+i)*nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the tensor elements of row/column \a i.
//
// \param i The row/column index.
// \param j The page index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::ConstPointer
   DynamicTensor<Type>::data( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return v_ + (k*m_+i)*nn_;
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
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::Iterator
   DynamicTensor<Type>::begin( size_t i, size_t k ) noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return Iterator( v_ + (k*m_+i)*nn_ );
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
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::ConstIterator
   DynamicTensor<Type>::begin( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return ConstIterator( v_ + (k*m_+i)*nn_ );
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
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::ConstIterator
   DynamicTensor<Type>::cbegin( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return ConstIterator( v_ + (k*m_+i)*nn_ );
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
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::Iterator
   DynamicTensor<Type>::end( size_t i, size_t k ) noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return Iterator( v_ + (k*m_+i)*nn_ + n_ );
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
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::ConstIterator
   DynamicTensor<Type>::end( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return ConstIterator( v_ + (k*m_+i)*nn_ + n_ );
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
template< typename Type > // Data type of the tensor
inline typename DynamicTensor<Type>::ConstIterator
   DynamicTensor<Type>::cend( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return ConstIterator( v_ + (k*m_+i)*nn_ + n_ );
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
template< typename Type > // Data type of the tensor
inline DynamicTensor<Type>& DynamicTensor<Type>::operator=(const Type& rhs)
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
//
// This assignment operator offers the option to directly assign to all elements of the tensor
// by means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::DynamicTensor<int> A;
   A = { { { 1, 2, 3 },
           { 4, 5 },
           { 7, 8, 9 } } };
   \endcode

// The tensor is resized according to the given initializer list and all its elements are
// assigned the values from the given initializer list. Missing values are initialized as
// default (as e.g. the value 6 in the example).
*/
template< typename Type > // Data type of the tensor
inline DynamicTensor<Type>&
   DynamicTensor<Type>::operator=( initializer_list< initializer_list<initializer_list<Type> > > list )
{
   resize( list.size(), determineRows( list ), determineColumns( list ), false );

   size_t k( 0UL );

   for (const auto& page : list) {
      size_t i( 0UL );
      for (const auto& rowList : page) {
         std::fill(std::copy(rowList.begin(), rowList.end(), v_+(k*m_+i)*nn_), v_+(k*m_+i+1UL)*nn_, Type());
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

   const int init[3][3][3] = { { { 1, 2, 3 },
                                 { 4, 5, 6 },
                                 { 7, 8, 9 } },
                               { { 1, 2, 3 },
                                 { 4, 5, 6 },
                                 { 7, 8, 9 } },
                               { { 1, 2, 3 },
                                 { 4, 5, 6 },
                                 { 7, 8, 9 } } };
   blaze::DynamicTensor<int> A;
   A = init;
   \endcode

// The tensor is resized according to the size of the array and assigned the values of the given
// array. Missing values are initialized with default values (as e.g. the value 6 in the example).
*/
template< typename Type > // Data type of the tensor
template< typename Other  // Data type of the initialization array
        , size_t Rows     // Number of rows of the initialization array
        , size_t Cols     // Number of columns of the initialization array
        , size_t Pages >  // Number of pages of the initialization array
inline DynamicTensor<Type>& DynamicTensor<Type>::operator=( const Other (&array)[Pages][Rows][Cols] )
{
   resize( Pages, Rows, Cols, false );

   for (size_t k=0UL; k<Pages; ++k) {
      for (size_t i=0UL; i<Rows; ++i) {
         for (size_t j=0UL; j<Cols; ++j) {
            v_[(k*m_+i)*nn_+j] = array[k][i][j];
         }
      }
   }
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy assignment operator for DynamicTensor.
//
// \param rhs Tensor to be copied.
// \return Reference to the assigned tensor.
//
// The tensor is resized according to the given \f$ M \times N \f$ tensor and initialized as a
// copy of this tensor.
*/
template< typename Type > // Data type of the tensor
inline DynamicTensor<Type>& DynamicTensor<Type>::operator=( const DynamicTensor& rhs )
{
   if( &rhs == this ) return *this;

   resize( rhs.o_, rhs.m_, rhs.n_, false );
   smpAssign( *this, ~rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Move assignment operator for DynamicTensor.
//
// \param rhs The tensor to be moved into this instance.
// \return Reference to the assigned tensor.
*/
template< typename Type > // Data type of the tensor
inline DynamicTensor<Type>& DynamicTensor<Type>::operator=( DynamicTensor&& rhs ) noexcept
{
   deallocate( v_ );

   o_        = rhs.o_;
   m_        = rhs.m_;
   n_        = rhs.n_;
   nn_       = rhs.nn_;
   capacity_ = rhs.capacity_;
   v_        = rhs.v_;

   rhs.o_        = 0UL;
   rhs.m_        = 0UL;
   rhs.n_        = 0UL;
   rhs.nn_       = 0UL;
   rhs.capacity_ = 0UL;
   rhs.v_        = nullptr;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different matrices.
//
// \param rhs Tensor to be copied.
// \return Reference to the assigned tensor.
//
// The tensor is resized according to the given \f$ M \times N \f$ tensor and initialized as a
// copy of this tensor.
*/
template< typename Type > // Data type of the tensor
template< typename MT >  // Type of the right-hand side tensor
inline DynamicTensor<Type>& DynamicTensor<Type>::operator=( const Tensor<MT>& rhs )
{
   if( (~rhs).canAlias( this ) ) {
      DynamicTensor tmp( ~rhs );
      swap( tmp );
   }
   else {
      resize( (~rhs).pages(), (~rhs).rows(), (~rhs).columns(), false );
      smpAssign( *this, ~rhs );
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
template< typename Type > // Data type of the tensor
template< typename MT >   // Type of the right-hand side tensor
inline DynamicTensor<Type>& DynamicTensor<Type>::operator+=( const Tensor<MT>& rhs )
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
template< typename Type > // Data type of the tensor
template< typename MT >   // Type of the right-hand side tensor
inline DynamicTensor<Type>& DynamicTensor<Type>::operator-=( const Tensor<MT>& rhs )
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
template< typename Type > // Data type of the tensor
template< typename MT >  // Type of the right-hand side tensor
inline DynamicTensor<Type>& DynamicTensor<Type>::operator%=( const Tensor<MT>& rhs )
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
template< typename Type > // Data type of the tensor
inline size_t DynamicTensor<Type>::rows() const noexcept
{
   return m_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of columns of the tensor.
//
// \return The number of columns of the tensor.
*/
template< typename Type > // Data type of the tensor
inline size_t DynamicTensor<Type>::columns() const noexcept
{
   return n_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of pages of the tensor.
//
// \return The number of pages of the tensor.
*/
template< typename Type > // Data type of the tensor
inline size_t DynamicTensor<Type>::pages() const noexcept
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
template< typename Type > // Data type of the tensor
inline size_t DynamicTensor<Type>::spacing() const noexcept
{
   return nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the tensor.
//
// \return The capacity of the tensor.
*/
template< typename Type > // Data type of the tensor
inline size_t DynamicTensor<Type>::capacity() const noexcept
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
template< typename Type > // Data type of the tensor
inline size_t DynamicTensor<Type>::capacity( size_t i, size_t k ) const noexcept
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
template< typename Type > // Data type of the tensor
inline size_t DynamicTensor<Type>::nonZeros() const
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
template< typename Type > // Data type of the tensor
inline size_t DynamicTensor<Type>::nonZeros( size_t i, size_t k ) const
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
template< typename Type > // Data type of the tensor
inline void DynamicTensor<Type>::reset()
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
template< typename Type > // Data type of the tensor
inline void DynamicTensor<Type>::reset( size_t i, size_t k )
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
template< typename Type > // Data type of the tensor
inline void DynamicTensor<Type>::clear()
{
   resize( 0UL, 0UL, 0UL, false );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the tensor.
//
// \param m The new number of rows of the tensor.
// \param n The new number of columns of the tensor.
// \param preserve \a true if the old values of the tensor should be preserved, \a false if not.
// \return void
//
// This function resizes the tensor using the given size to \f$ m \times n \f$. During this
// operation, new dynamic memory may be allocated in case the capacity of the tensor is too
// small. Note that this function may invalidate all existing views (submatrices, rows, columns,
// ...) on the tensor if it is used to shrink the tensor. Additionally, the resize operation
// potentially changes all tensor elements. In order to preserve the old tensor values, the
// \a preserve flag can be set to \a true. However, new tensor elements are not initialized!
//
// The following example illustrates the resize operation of a \f$ 2 \times 4 \f$ tensor to a
// \f$ 4 \times 2 \f$ tensor. The new, uninitialized elements are marked with \a x:

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
template< typename Type > // Data type of the tensor
void DynamicTensor<Type>::resize( size_t o, size_t m, size_t n, bool preserve )
{
   using std::swap;
   using blaze::min;

   if( m == m_ && n == n_ && o == o_ ) return;

   const size_t nn( addPadding( n ) );

   if( preserve )
   {
      Type* BLAZE_RESTRICT v = allocate<Type>( o*m*nn );
      const size_t min_m( min( m, m_ ) );
      const size_t min_n( min( n, n_ ) );
      const size_t min_o( min( o, o_ ) );

      for (size_t k=0UL; k<min_o; ++k) {
         for (size_t i=0UL; i<min_m; ++i) {
            transfer(v_+(k*o_+i)*nn_, v_+(k*o_+i)*nn_+min_n, v+(k*o+i)*nn);
         }
      }
      swap( v_, v );
      deallocate( v );
      capacity_ = o*m*nn;
   }
   else if( o*m*nn > capacity_ ) {
      Type* BLAZE_RESTRICT v = allocate<Type>( o*m*nn );
      swap( v_, v );
      deallocate( v );
      capacity_ = o*m*nn;
   }

   if( IsVectorizable_v<Type> ) {
      for (size_t k=0UL; k<o; ++k) {
         for (size_t i=0UL; i<m; ++i) {
            size_t row_elements = (k*m+i)*nn;
            for (size_t j=n; j<nn; ++j) {
               v_[row_elements+j] = Type();
            }
         }
      }
   }

   m_  = m;
   n_  = n;
   o_  = o;
   nn_ = nn;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Extending the size of the tensor.
//
// \param m Number of additional rows.
// \param n Number of additional columns.
// \param preserve \a true if the old values of the tensor should be preserved, \a false if not.
// \return void
//
// This function increases the tensor size by \a m rows and \a n columns. During this operation,
// new dynamic memory may be allocated in case the capacity of the tensor is too small. Therefore
// this function potentially changes all tensor elements. In order to preserve the old tensor
// values, the \a preserve flag can be set to \a true. However, new tensor elements are not
// initialized!
*/
template< typename Type > // Data type of the tensor
inline void DynamicTensor<Type>::extend( size_t o, size_t m, size_t n, bool preserve )
{
   resize( o_+o, m_+m, n_+n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the minimum capacity of the tensor.
//
// \param elements The new minimum capacity of the dense tensor.
// \return void
//
// This function increases the capacity of the dense tensor to at least \a elements elements.
// The current values of the tensor elements are preserved.
*/
template< typename Type > // Data type of the tensor
inline void DynamicTensor<Type>::reserve( size_t elements )
{
   using std::swap;

   if( elements > capacity_ )
   {
      // Allocating a new array
      Type* BLAZE_RESTRICT tmp = allocate<Type>( elements );

      // Initializing the new array
      transfer( v_, v_+capacity_, tmp );

      if( IsVectorizable_v<Type> ) {
         for( size_t i=capacity_; i<elements; ++i )
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
// This function minimizes the capacity of the tensor by removing unused capacity. Please note
// that due to padding the capacity might not be reduced exactly to rows() times columns().
// Please also note that in case a reallocation occurs, all iterators (including end() iterators),
// all pointers and references to elements of this tensor are invalidated.
*/
template< typename Type > // Data type of the tensor
inline void DynamicTensor<Type>::shrinkToFit()
{
   if( ( o_ * m_ * nn_ ) < capacity_ ) {
      DynamicTensor( *this ).swap( *this );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two matrices.
//
// \param m The tensor to be swapped.
// \return void
*/
template< typename Type > // Data type of the tensor
inline void DynamicTensor<Type>::swap( DynamicTensor& m ) noexcept
{
   using std::swap;

   swap( o_ , m.o_  );
   swap( m_ , m.m_  );
   swap( n_ , m.n_  );
   swap( nn_, m.nn_ );
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
template< typename Type > // Data type of the tensor
inline size_t DynamicTensor<Type>::addPadding( size_t value ) const noexcept
{
   if( usePadding && IsVectorizable_v<Type> )
      return nextMultiple<size_t>( value, SIMDSIZE );
   else return value;
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
*/
template< typename Type >  // Data type of the tensor
inline DynamicTensor<Type>& DynamicTensor<Type>::transpose()
{
//    using std::swap;
//
//    constexpr size_t block( BLOCK_SIZE );
//
//    if( o_ == n_ && m_ == n_ )
//    {
//       for( size_t ii=0UL; ii<m_; ii+=block ) {
//          const size_t iend( min( ii+block, m_ ) );
//          for( size_t jj=0UL; jj<=ii; jj+=block ) {
//             for( size_t i=ii; i<iend; ++i ) {
//                const size_t jend( min( jj+block, n_, i ) );
//                for( size_t j=jj; j<jend; ++j ) {
//                   swap( v_[i*nn_+j], v_[j*nn_+i] );
//                }
//             }
//          }
//       }
//    }
//    else
//    {
      DynamicTensor tmp( trans( *this ) );
      this->swap( tmp );
//    }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the tensor.
//
// \return Reference to the transposed tensor.
*/
template< typename Type >  // Data type of the tensor
template< typename T >     // Type of the mapping indices
inline DynamicTensor<Type>& DynamicTensor<Type>::transpose( const T* indices, size_t n )
{
//    using std::swap;
//
//    constexpr size_t block( BLOCK_SIZE );
//
//    if( o_ == n_ && m_ == n_ )
//    {
//       for( size_t ii=0UL; ii<m_; ii+=block ) {
//          const size_t iend( min( ii+block, m_ ) );
//          for( size_t jj=0UL; jj<=ii; jj+=block ) {
//             for( size_t i=ii; i<iend; ++i ) {
//                const size_t jend( min( jj+block, n_, i ) );
//                for( size_t j=jj; j<jend; ++j ) {
//                   swap( v_[i*nn_+j], v_[j*nn_+i] );
//                }
//             }
//          }
//       }
//    }
//    else
//    {
      DynamicTensor tmp( trans(*this, indices, n ) );
      this->swap( tmp );
//    }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the tensor.
//
// \return Reference to the transposed tensor.
*/
template< typename Type >  // Data type of the tensor
inline DynamicTensor<Type>& DynamicTensor<Type>::ctranspose()
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
      DynamicTensor tmp( ctrans( *this ) );
      swap( tmp );
//    }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the tensor.
//
// \return Reference to the transposed tensor.
*/
template< typename Type >  // Data type of the tensor
template< typename T >     // Type of the mapping indices
inline DynamicTensor<Type>& DynamicTensor<Type>::ctranspose( const T* indices, size_t n )
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
      DynamicTensor tmp( ctrans(*this, indices, n ) );
      swap( tmp );
//    }

   return *this;
}
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
   blaze::DynamicTensor<int> A;
   // ... Resizing and initialization
   A *= 4;        // Scaling of the tensor
   A.scale( 4 );  // Same effect as above
   \endcode
*/
template< typename Type > // Data type of the tensor
template< typename Other >  // Data type of the scalar value
inline DynamicTensor<Type>& DynamicTensor<Type>::scale( const Other& scalar )
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
//  DEBUGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the invariants of the dynamic tensor are intact.
//
// \return \a true in case the dynamic tensor's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the dynamic tensor are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< typename Type > // Data type of the tensor
inline bool DynamicTensor<Type>::isIntact() const noexcept
{
   if( o_ * m_ * n_ > capacity_ )
      return false;

   if (IsVectorizable_v<Type>) {
      for (size_t k=0UL; k<o_; ++k) {
         for (size_t i=0UL; i<m_; ++i) {
            size_t row_elements = (k*m_+i)*nn_;
            for (size_t j=n_; j<nn_; ++j) {
               if (v_[row_elements+j] != Type())
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
template< typename Type > // Data type of the tensor
template< typename Other >  // Data type of the foreign expression
inline bool DynamicTensor<Type>::canAlias( const Other* alias ) const noexcept
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
template< typename Type > // Data type of the tensor
template< typename Other >  // Data type of the foreign expression
inline bool DynamicTensor<Type>::isAliased( const Other* alias ) const noexcept
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
template< typename Type > // Data type of the tensor
inline bool DynamicTensor<Type>::isAligned() const noexcept
{
   return ( usePadding || columns() % SIMDSIZE == 0UL );
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
template< typename Type > // Data type of the tensor
inline bool DynamicTensor<Type>::canSMPAssign() const noexcept
{
   return ( rows() * columns() >= SMP_DMATASSIGN_THRESHOLD );
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
template< typename Type > // Data type of the tensor
BLAZE_ALWAYS_INLINE typename DynamicTensor<Type>::SIMDType
   DynamicTensor<Type>::load( size_t k, size_t i, size_t j ) const noexcept
{
   if( usePadding )
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
// than the number of columns. Additionally, the column index (in case of a tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type > // Data type of the tensor
BLAZE_ALWAYS_INLINE typename DynamicTensor<Type>::SIMDType
   DynamicTensor<Type>::loada( size_t k, size_t i, size_t j ) const noexcept
{
   using blaze::loada;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < o_, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= nn_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || j % SIMDSIZE == 0UL, "Invalid column access index" );
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
template< typename Type > // Data type of the tensor
BLAZE_ALWAYS_INLINE typename DynamicTensor<Type>::SIMDType
   DynamicTensor<Type>::loadu( size_t k, size_t i, size_t j ) const noexcept
{
   using blaze::loadu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < o_, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= nn_, "Invalid column access index" );

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
template< typename Type > // Data type of the tensor
BLAZE_ALWAYS_INLINE void
   DynamicTensor<Type>::store( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   if( usePadding )
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
// than the number of columns. Additionally, the column index (in case of a tensor)
// or the row index (in case of a column-major tensor) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename Type > // Data type of the tensor
BLAZE_ALWAYS_INLINE void
   DynamicTensor<Type>::storea( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::storea;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < o_, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= nn_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || j % SIMDSIZE == 0UL, "Invalid column access index" );
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
template< typename Type > // Data type of the tensor
BLAZE_ALWAYS_INLINE void
   DynamicTensor<Type>::storeu( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::storeu;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < o_, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= nn_, "Invalid column access index" );

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
template< typename Type > // Data type of the tensor
BLAZE_ALWAYS_INLINE void
   DynamicTensor<Type>::stream( size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
{
   using blaze::stream;

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= nn_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( k < o_, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( !usePadding || j % SIMDSIZE == 0UL, "Invalid column access index" );
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
template< typename Type > // Data type of the tensor
template< typename MT >  // Type of the right-hand side dense tensor
inline auto DynamicTensor<Type>::assign( const DenseTensor<MT>& rhs )
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
            v_[row_elements+j] = (~rhs)(k, i, j);
            v_[row_elements+j+1UL] = (~rhs)(k, i, j+1UL);
         }
         if (jpos < n_) {
            v_[row_elements+jpos] = (~rhs)(k, i, jpos);
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
template< typename Type > // Data type of the tensor
template< typename MT >  // Type of the right-hand side dense tensor
inline auto DynamicTensor<Type>::assign( const DenseTensor<MT>& rhs )
   -> EnableIf_t< VectorizedAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( o_ == (~rhs).pages(),   "Invalid number of pages" );

   constexpr bool remainder( !usePadding || !IsPadded_v<MT> );

   const size_t jpos( ( remainder )?( n_ & size_t(-SIMDSIZE) ):( n_ ) );
   BLAZE_INTERNAL_ASSERT( !remainder || ( n_ - ( n_ % (SIMDSIZE) ) ) == jpos, "Invalid end calculation" );

   if( usePadding && useStreaming &&
       ( o_*m_*n_ > ( cacheSize / ( sizeof(Type) * 3UL ) ) ) && !(~rhs).isAliased( this ) )
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
         for (size_t i=0UL; i<m_; ++i) {
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
template< typename Type > // Data type of the tensor
template< typename MT >  // Type of the right-hand side dense tensor
inline auto DynamicTensor<Type>::addAssign( const DenseTensor<MT>& rhs )
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
            v_[row_elements+j] += (~rhs)(k, i, j);
            v_[row_elements+j+1UL] += (~rhs)(k, i, j+1UL);
         }
         if (j < jend) {
            v_[row_elements+j] += (~rhs)(k, i, j);
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
template< typename Type > // Data type of the tensor
template< typename MT >  // Type of the right-hand side dense tensor
inline auto DynamicTensor<Type>::addAssign( const DenseTensor<MT>& rhs )
   -> EnableIf_t< VectorizedAddAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( o_ == (~rhs).pages(),   "Invalid number of pages" );

   constexpr bool remainder( !usePadding || !IsPadded_v<MT> );

   for (size_t k=0UL; k<o_; ++k) {
      for (size_t i=0UL; i<m_; ++i) {
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
template< typename Type > // Data type of the tensor
template< typename MT >  // Type of the right-hand side dense tensor
inline auto DynamicTensor<Type>::subAssign( const DenseTensor<MT>& rhs )
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
            v_[row_elements+j] -= (~rhs)(k, i, j);
            v_[row_elements+j+1UL] -= (~rhs)(k, i, j+1UL);
         }
         if (j < jend) {
            v_[row_elements+j] -= (~rhs)(k, i, j);
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
template< typename Type > // Data type of the tensor
template< typename MT >  // Type of the right-hand side dense tensor
inline auto DynamicTensor<Type>::subAssign( const DenseTensor<MT>& rhs )
   -> EnableIf_t< VectorizedSubAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( o_ == (~rhs).pages(),   "Invalid number of pages" );

   constexpr bool remainder( !usePadding || !IsPadded_v<MT> );

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
template< typename Type > // Data type of the tensor
template< typename MT >  // Type of the right-hand side dense tensor
inline auto DynamicTensor<Type>::schurAssign( const DenseTensor<MT>& rhs )
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
            v_[row_elements+j] *= (~rhs)(k, i, j);
            v_[row_elements+j+1UL] *= (~rhs)(k, i, j+1UL);
         }
         if (jpos < n_) {
            v_[row_elements+jpos] *= (~rhs)(k, i, jpos);
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
template< typename Type > // Data type of the tensor
template< typename MT >  // Type of the right-hand side dense tensor
inline auto DynamicTensor<Type>::schurAssign( const DenseTensor<MT>& rhs )
   -> EnableIf_t< VectorizedSchurAssign_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( o_ == (~rhs).pages(),   "Invalid number of pages" );

   constexpr bool remainder( !usePadding || !IsPadded_v<MT> );

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
//  DYNAMICTENSOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DynamicTensor operators */
//@{
template< typename Type >
inline void reset( DynamicTensor<Type>& m );

template< typename Type >
inline void reset( DynamicTensor<Type>& m, size_t i, size_t k );

template< typename Type >
inline void clear( DynamicTensor<Type>& m );

template< bool RF, typename Type >
inline bool isDefault( const DynamicTensor<Type>& m );

template< typename Type >
inline bool isIntact( const DynamicTensor<Type>& m ) noexcept;

template< typename Type >
inline void swap( DynamicTensor<Type>& a, DynamicTensor<Type>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given dynamic tensor.
// \ingroup dynamic_tensor
//
// \param m The tensor to be resetted.
// \return void
*/
template< typename Type > // Data type of the tensor
inline void reset( DynamicTensor<Type>& m )
{
   m.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset the specified row/column of the given dynamic tensor.
// \ingroup dynamic_tensor
//
// \param m The tensor to reset.
// \param i The index of the row/column to reset.
// \param k The index of the page to reset.
// \return void
//
// This function resets the values in the specified row/column of the given dynamic tensor to
// their default value. In case the given tensor is a \a rowMajor tensor the function resets the
// values in row \a i, if it is a \a columnMajor tensor the function resets the values in column
// \a i. Note that the capacity of the row/column remains unchanged.
*/
template< typename Type > // Data type of the tensor
inline void reset( DynamicTensor<Type>& m, size_t i, size_t k )
{
   m.reset( i, k );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given dynamic tensor.
// \ingroup dynamic_tensor
//
// \param m The tensor to be cleared.
// \return void
*/
template< typename Type > // Data type of the tensor
inline void clear( DynamicTensor<Type>& m )
{
   m.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given dynamic tensor is in default state.
// \ingroup dynamic_tensor
//
// \param m The tensor to be tested for its default state.
// \return \a true in case the given tensor's rows and columns are zero, \a false otherwise.
//
// This function checks whether the dynamic tensor is in default (constructed) state, i.e. if
// it's number of rows and columns is 0. In case it is in default state, the function returns
// \a true, else it will return \a false. The following example demonstrates the use of the
// \a isDefault() function:

   \code
   blaze::DynamicTensor<int> A;
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
        , typename Type > // Data type of the tensor
inline bool isDefault( const DynamicTensor<Type>& m )
{
   return ( m.rows() == 0UL && m.columns() == 0UL && m.pages() == 0UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given dynamic tensor are intact.
// \ingroup dynamic_tensor
//
// \param m The dynamic tensor to be tested.
// \return \a true in case the given tensor's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the dynamic tensor are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   blaze::DynamicTensor<int> A;
   // ... Resizing and initialization
   if( isIntact( A ) ) { ... }
   \endcode
*/
template< typename Type > // Data type of the tensor
inline bool isIntact( const DynamicTensor<Type>& m ) noexcept
{
   return m.isIntact();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two dynamic matrices.
// \ingroup dynamic_tensor
//
// \param a The first tensor to be swapped.
// \param b The second tensor to be swapped.
// \return void
*/
template< typename Type > // Data type of the tensor
inline void swap( DynamicTensor<Type>& a, DynamicTensor<Type>& b ) noexcept
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
template< typename T > // Data type of the tensor
struct HasConstDataAccess< DynamicTensor<T> >
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
template< typename T > // Data type of the tensor
struct HasMutableDataAccess< DynamicTensor<T> >
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
template< typename T > // Data type of the tensor
struct IsAligned< DynamicTensor<T> >
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
template< typename T > // Data type of the tensor
struct IsContiguous< DynamicTensor<T> >
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
template< typename T > // Data type of the tensor
struct IsPadded< DynamicTensor<T> >
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
template< typename T > // Data type of the tensor
struct IsResizable< DynamicTensor<T> >
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
template< typename T > // Data type of the tensor
struct IsShrinkable< DynamicTensor<T> >
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
template< typename T1, typename T2 >
struct AddTraitEval2< T1, T2
                    , EnableIf_t< IsTensor_v<T1> &&
                                  IsTensor_v<T2> &&
                                  ( IsDenseTensor_v<T1> || IsDenseTensor_v<T2> ) &&
                                  ( Size_v<T1,0UL> == DefaultSize_v ) &&
                                  ( Size_v<T2,0UL> == DefaultSize_v ) &&
                                  ( Size_v<T1,1UL> == DefaultSize_v ) &&
                                  ( Size_v<T2,1UL> == DefaultSize_v ) &&
                                  ( Size_v<T1,2UL> == DefaultSize_v ) &&
                                  ( Size_v<T2,2UL> == DefaultSize_v ) &&
                                  ( MaxSize_v<T1,0UL> == DefaultSize_v ) &&
                                  ( MaxSize_v<T2,0UL> == DefaultSize_v ) &&
                                  ( MaxSize_v<T1,1UL> == DefaultSize_v ) &&
                                  ( MaxSize_v<T2,1UL> == DefaultSize_v ) &&
                                  ( MaxSize_v<T1,2UL> == DefaultSize_v ) &&
                                  ( MaxSize_v<T2,2UL> == DefaultSize_v ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = DynamicTensor< AddTrait_t<ET1,ET2> >;
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
                                  ( IsDenseTensor_v<T1> || IsDenseTensor_v<T2> ) &&
                                  ( Size_v<T1,0UL> == DefaultSize_v ) &&
                                  ( Size_v<T2,0UL> == DefaultSize_v ) &&
                                  ( Size_v<T1,1UL> == DefaultSize_v ) &&
                                  ( Size_v<T2,1UL> == DefaultSize_v ) &&
                                  ( Size_v<T1,2UL> == DefaultSize_v ) &&
                                  ( Size_v<T2,2UL> == DefaultSize_v ) &&
                                  ( MaxSize_v<T1,0UL> == DefaultMaxSize_v ) &&
                                  ( MaxSize_v<T2,0UL> == DefaultMaxSize_v ) &&
                                  ( MaxSize_v<T1,1UL> == DefaultMaxSize_v ) &&
                                  ( MaxSize_v<T2,1UL> == DefaultMaxSize_v ) &&
                                  ( MaxSize_v<T1,2UL> == DefaultMaxSize_v ) &&
                                  ( MaxSize_v<T2,2UL> == DefaultMaxSize_v ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = DynamicTensor< SubTrait_t<ET1,ET2> >;
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
                                    ( Size_v<T1,0UL> == DefaultSize_v ) &&
                                    ( Size_v<T2,0UL> == DefaultSize_v ) &&
                                    ( Size_v<T1,1UL> == DefaultSize_v ) &&
                                    ( Size_v<T2,1UL> == DefaultSize_v ) &&
                                    ( Size_v<T1,2UL> == DefaultSize_v ) &&
                                    ( Size_v<T2,2UL> == DefaultSize_v ) &&
                                    ( MaxSize_v<T1,0UL> == DefaultMaxSize_v ) &&
                                    ( MaxSize_v<T2,0UL> == DefaultMaxSize_v ) &&
                                    ( MaxSize_v<T1,1UL> == DefaultMaxSize_v ) &&
                                    ( MaxSize_v<T2,1UL> == DefaultMaxSize_v ) &&
                                    ( MaxSize_v<T1,2UL> == DefaultMaxSize_v ) &&
                                    ( MaxSize_v<T2,2UL> == DefaultMaxSize_v ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = DynamicTensor< MultTrait_t<ET1,ET2> >;
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
                     , EnableIf_t< IsDenseTensor_v<T1> &&
                                   IsNumeric_v<T2> &&
                                   ( Size_v<T1,0UL> == DefaultSize_v ) &&
                                   ( Size_v<T1,1UL> == DefaultSize_v ) &&
                                   ( Size_v<T1,2UL> == DefaultSize_v ) &&
                                   ( MaxSize_v<T1,0UL> == DefaultMaxSize_v ) &&
                                   ( MaxSize_v<T1,1UL> == DefaultMaxSize_v ) &&
                                   ( MaxSize_v<T1,2UL> == DefaultMaxSize_v ) > >
{
   using ET1 = ElementType_t<T1>;

   using Type = DynamicTensor< MultTrait_t<ET1,T2> >;
};

template< typename T1, typename T2 >
struct MultTraitEval2< T1, T2
                     , EnableIf_t< IsNumeric_v<T1> &&
                                   IsDenseTensor_v<T2> &&
                                   ( Size_v<T2,0UL> == DefaultSize_v ) &&
                                   ( Size_v<T2,1UL> == DefaultSize_v ) &&
                                   ( Size_v<T2,2UL> == DefaultSize_v ) &&
                                   ( MaxSize_v<T2,0UL> == DefaultMaxSize_v ) &&
                                   ( MaxSize_v<T2,1UL> == DefaultMaxSize_v ) &&
                                   ( MaxSize_v<T2,2UL> == DefaultMaxSize_v ) > >
{
   using ET2 = ElementType_t<T2>;

   using Type = DynamicTensor< MultTrait_t<T1,ET2> >;
};

template< typename T1, typename T2 >
struct MultTraitEval2< T1, T2
                     , EnableIf_t< IsTensor_v<T1> &&
                                   IsTensor_v<T2> &&
                                   ( IsDenseTensor_v<T1> || IsDenseTensor_v<T2> ) &&
                                   ( ( Size_v<T1,0UL> == DefaultSize_v &&
                                       ( !IsSquare_v<T1> || Size_v<T2,0UL> == DefaultSize_v ) ) ||
                                     ( Size_v<T2,1UL> == DefaultSize_v &&
                                       ( !IsSquare_v<T2> || Size_v<T1,1UL> == DefaultSize_v ) ) ||
                                     ( Size_v<T2,2UL> == DefaultSize_v &&
                                       ( !IsSquare_v<T2> || Size_v<T1,2UL> == DefaultSize_v ) ) ) &&
                                   ( ( MaxSize_v<T1,0UL> == DefaultMaxSize_v &&
                                       ( !IsSquare_v<T1> || MaxSize_v<T2,0UL> == DefaultMaxSize_v ) ) ||
                                     ( MaxSize_v<T2,1UL> == DefaultMaxSize_v &&
                                       ( !IsSquare_v<T2> || MaxSize_v<T1,1UL> == DefaultMaxSize_v ) ) ||
                                     ( MaxSize_v<T2,2UL> == DefaultMaxSize_v &&
                                       ( !IsSquare_v<T2> || MaxSize_v<T1,2UL> == DefaultMaxSize_v ) ) ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = DynamicTensor< MultTrait_t<ET1,ET2> >;
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
                    , EnableIf_t< IsDenseTensor_v<T1> &&
                                  IsNumeric_v<T2> &&
                                  ( Size_v<T1,0UL> == DefaultSize_v ) &&
                                  ( Size_v<T1,1UL> == DefaultSize_v ) &&
                                  ( Size_v<T1,2UL> == DefaultSize_v ) &&
                                  ( MaxSize_v<T1,0UL> == DefaultMaxSize_v ) &&
                                  ( MaxSize_v<T1,1UL> == DefaultMaxSize_v ) &&
                                  ( MaxSize_v<T1,2UL> == DefaultMaxSize_v ) > >
{
   using ET1 = ElementType_t<T1>;

   using Type = DynamicTensor< DivTrait_t<ET1,T2> >;
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
                         , EnableIf_t< IsDenseTensor_v<T> &&
                                       ( Size_v<T,0UL> == DefaultSize_v ||
                                         Size_v<T,1UL> == DefaultSize_v ||
                                         Size_v<T,2UL> == DefaultSize_v ) &&
                                       ( MaxSize_v<T,0UL> == DefaultMaxSize_v ||
                                         MaxSize_v<T,1UL> == DefaultMaxSize_v ||
                                         MaxSize_v<T,2UL> == DefaultMaxSize_v ) > >
{
   using ET = ElementType_t<T>;

   using Type = DynamicTensor< MapTrait_t<ET,OP> >;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2, typename OP >
struct BinaryMapTraitEval2< T1, T2, OP
                          , EnableIf_t< IsTensor_v<T1> &&
                                        IsTensor_v<T2> &&
                                        ( Size_v<T1,0UL> == DefaultSize_v ) &&
                                        ( Size_v<T2,0UL> == DefaultSize_v ) &&
                                        ( Size_v<T1,1UL> == DefaultSize_v ) &&
                                        ( Size_v<T2,1UL> == DefaultSize_v ) &&
                                        ( Size_v<T1,2UL> == DefaultSize_v ) &&
                                        ( Size_v<T2,2UL> == DefaultSize_v ) &&
                                        ( MaxSize_v<T1,0UL> == DefaultSize_v ) &&
                                        ( MaxSize_v<T2,0UL> == DefaultSize_v ) &&
                                        ( MaxSize_v<T1,1UL> == DefaultSize_v ) &&
                                        ( MaxSize_v<T2,1UL> == DefaultSize_v ) &&
                                        ( MaxSize_v<T1,2UL> == DefaultSize_v ) &&
                                        ( MaxSize_v<T2,2UL> == DefaultSize_v ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = DynamicTensor< MapTrait_t<ET1,ET2,OP> >;
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
template< typename T  // Type to be expanded
        , size_t E >  // Compile time expansion
struct ExpandTraitEval2< T, E
                       , EnableIf_t< IsDenseMatrix_v<T> &&
                                     ( ( E == inf ) ||
                                       ( ( Size_v<T,0UL> == DefaultSize_v ) &&
                                         ( MaxSize_v<T,0UL> == DefaultMaxSize_v ) &&
                                         ( Size_v<T,1UL> == DefaultSize_v ) &&
                                         ( MaxSize_v<T,1UL> == DefaultMaxSize_v ) ) ) > >
{
   using Type = DynamicTensor< ElementType_t<T> >;
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
template< typename T1, typename T2 >
struct HighType< DynamicTensor<T1>, DynamicTensor<T2> >
{
   using Type = DynamicTensor< typename HighType<T1,T2>::Type >;
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
template< typename T1, typename T2 >
struct LowType< DynamicTensor<T1>, DynamicTensor<T2> >
{
   using Type = DynamicTensor< typename LowType<T1,T2>::Type >;
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
   EnableIf_t< IsDenseTensor_v<MT> && !IsUniform_v<MT> &&
               ( M == 0UL || Size_v< MT,0UL > == DefaultSize_v ||
                             Size_v< MT,1UL > == DefaultSize_v ) &&
               ( M == 0UL || MaxSize_v< MT,0UL > == DefaultMaxSize_v ||
                             MaxSize_v< MT,1UL > == DefaultMaxSize_v ) > >
{
   using Type = DynamicMatrix< RemoveConst_t< ElementType_t<MT> >, rowMajor >;
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
   EnableIf_t< IsDenseTensor_v<MT> && !IsUniform_v<MT> &&
               ( M == 0UL || Size_v< MT,1UL > == DefaultSize_v ||
                             Size_v< MT,2UL > == DefaultSize_v ) &&
               ( M == 0UL || MaxSize_v< MT,1UL > == DefaultMaxSize_v ||
                             MaxSize_v< MT,2UL > == DefaultMaxSize_v ) > >
{
   using Type = DynamicMatrix< RemoveConst_t< ElementType_t<MT> >, rowMajor >;
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
   EnableIf_t< IsDenseTensor_v<MT> && !IsUniform_v<MT> &&
               ( M == 0UL || Size_v< MT,0UL > == DefaultSize_v ||
                             Size_v< MT,2UL > == DefaultSize_v ) &&
               ( M == 0UL || MaxSize_v< MT,0UL > == DefaultMaxSize_v ||
                             MaxSize_v< MT,2UL > == DefaultMaxSize_v ) > >
{
   using Type = DynamicMatrix< RemoveConst_t< ElementType_t<MT> >, columnMajor >;
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
template< typename MT >
struct SubtensorTraitEval2< MT, inf, inf, inf, inf, inf, inf,
   EnableIf_t< IsDenseTensor_v<MT> &&
               ( Size_v< MT,0UL > == DefaultSize_v ||
                 Size_v< MT,1UL > == DefaultSize_v ||
                 Size_v< MT,2UL > == DefaultSize_v ) &&
               ( MaxSize_v< MT,0UL > == DefaultMaxSize_v ||
                 MaxSize_v< MT,1UL > == DefaultMaxSize_v ||
                 MaxSize_v< MT,2UL > == DefaultMaxSize_v ) > >
{
   using Type = DynamicTensor< RemoveConst_t< ElementType_t<MT> > >;
};
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
// struct ColumnsTraitEval2< MT, N
//                         , EnableIf_t< IsDenseTensor_v<MT> &&
//                                       ( N == 0UL || Size_v<MT,0UL> == DefaultSize_v ) &&
//                                       ( N == 0UL || MaxSize_v<MT,0UL> == DefaultMaxSize_v ) > >
// {
//    using Type = DynamicMatrix< ElementType_t<MT> >;
// };
/*! \endcond */
//*************************************************************************************************


} // namespace blaze

#endif
