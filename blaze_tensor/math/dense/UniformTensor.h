//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/UniformTensor.h
//  \brief Header file for the implementation of a uniform tensor
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_UNIFORMTENSOR_H_
#define _BLAZE_TENSOR_MATH_DENSE_UNIFORMTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/Exception.h>
#include <blaze/math/Forward.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/UniformMatrix.h>
#include <blaze/math/dense/UniformIterator.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Conjugate.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/ColumnsTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/ExpandTrait.h>
#include <blaze/math/traits/MapTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/RowsTrait.h>
#include <blaze/math/traits/SchurTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/typetraits/HighType.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsColumnVector.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsRowVector.h>
#include <blaze/math/typetraits/IsSMPAssignable.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsUniform.h>
#include <blaze/math/typetraits/IsVector.h>
#include <blaze/math/typetraits/IsZero.h>
#include <blaze/math/typetraits/LowType.h>
#include <blaze/math/typetraits/StorageOrder.h>
#include <blaze/math/typetraits/YieldsUniform.h>
#include <blaze/math/typetraits/YieldsZero.h>
#include <blaze/system/Inline.h>
#include <blaze/system/StorageOrder.h>
#include <blaze/system/Thresholds.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/Types.h>
#include <blaze/util/MaybeUnused.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsVectorizable.h>
#include <blaze/util/typetraits/RemoveConst.h>

#include <blaze_tensor/math/dense/UniformMatrix.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>
#include <blaze_tensor/math/traits/ColumnSliceTrait.h>
#include <blaze_tensor/math/traits/PageSliceTrait.h>
#include <blaze_tensor/math/traits/RowSliceTrait.h>
#include <blaze_tensor/math/traits/SubtensorTrait.h>
#include <blaze_tensor/math/typetraits/IsDenseTensor.h>
#include <blaze_tensor/math/typetraits/IsTensor.h>

#include <utility>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup uniform_tensor UniformTensor
// \ingroup dense_tensor
*/
/*!\brief Efficient implementation of a uniform tensor.
// \ingroup uniform_tensor
//
// The UniformTensor class template is the representation of an arbitrary sized uniform tensor
// with elements of arbitrary type. The type of the elements and the storage order of the tensor
// can be specified via the two template parameters:

   \code
   template< typename Type, bool SO >
   class UniformTensor;
   \endcode

//  - Type: specifies the type of the tensor elements. UniformTensor can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
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

// The use of UniformTensor is very natural and intuitive. All operations (addition, subtraction,
// multiplication, scaling, ...) can be performed on all possible combinations of row-major and
// column-major dense and sparse matrices with fitting element types. The following example gives
// an impression of the use of UniformTensor:

   \code
   using blaze::UniformTensor;
   using blaze::CompressedTensor;
   using blaze::rowMajor;
   using blaze::columnMajor;

   UniformTensor<double,rowMajor> A( 2, 3 );  // Default initialized, row-major 2x3 uniform tensor
   A = 1.0;                                   // Assignment to all elements of the uniform tensor

   UniformTensor<float,columnMajor> B( 2, 3, 2.0F );  // Directly, uniformly initialized 2x3 tensor
   CompressedTensor<float> C( 2, 3 );                 // Empty row-major sparse single precision tensor

   UniformTensor<double,rowMajor> D;  // Creation of a new row-major tensor as a copy of A

   D = A + B;     // Tensor addition and assignment to a row-major tensor
   D = A - C;     // Tensor subtraction and assignment to a column-major tensor
   D = A * B;     // Tensor multiplication between two matrices of different element types

   A *= 2.0;      // In-place scaling of tensor
   D  = 2.0 * A;  // Scaling of tensor A
   D  = A * 2.0;  // Scaling of tensor A

   D += A - B;    // Addition assignment
   D -= A + C;    // Subtraction assignment
   D *= A * B;    // Multiplication assignment
   \endcode
*/
template< typename Type >                   // Data type of the tensor
class UniformTensor
   : public DenseTensor< UniformTensor<Type> >
{
 public:
   //**Type definitions****************************************************************************
   using This          = UniformTensor<Type>;    //!< Type of this UniformTensor instance.
   using BaseType      = DenseTensor<This>;      //!< Base type of this UniformTensor instance.
   using ResultType    = This;                      //!< Result type for expression template evaluations.
   using OppositeType  = UniformTensor<Type>;   //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = UniformTensor<Type>;   //!< Transpose type for expression template evaluations.
   using ElementType   = Type;                      //!< Type of the tensor elements.
   using SIMDType      = SIMDTrait_t<ElementType>;  //!< SIMD type of the tensor elements.
   using ReturnType    = const Type&;               //!< Return type for expression template evaluations.
   using CompositeType = const This&;               //!< Data type for composite expression templates.

   using Reference      = const Type&;  //!< Reference to a non-constant tensor value.
   using ConstReference = const Type&;  //!< Reference to a constant tensor value.
   using Pointer        = const Type*;  //!< Pointer to a non-constant tensor value.
   using ConstPointer   = const Type*;  //!< Pointer to a constant tensor value.

   using Iterator      = UniformIterator<const Type,aligned>;  //!< Iterator over non-constant elements.
   using ConstIterator = UniformIterator<const Type,aligned>;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a UniformTensor with different data/element type.
   */
   template< typename NewType >  // Data type of the other tensor
   struct Rebind {
      using Other = UniformTensor<NewType>;  //!< The type of the other UniformTensor.
   };
   //**********************************************************************************************

   //**Resize struct definition********************************************************************
   /*!\brief Resize mechanism to obtain a UniformTensor with different fixed dimensions.
   */
   template< size_t NewO    // Number of pages of the other tensor
           , size_t NewM    // Number of rows of the other tensor
           , size_t NewN >  // Number of columns of the other tensor
   struct Resize {
      using Other = UniformTensor<Type>;  //!< The type of the other UniformTensor.
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
   explicit inline constexpr UniformTensor() noexcept;
   explicit inline constexpr UniformTensor( size_t o, size_t m, size_t n );
   explicit inline constexpr UniformTensor( size_t o, size_t m, size_t n, const Type& init );

   template< typename MT >
   inline UniformTensor( const Tensor<MT>& m );

   inline UniformTensor( const UniformTensor& m ) = default;
   inline UniformTensor( UniformTensor&& m ) = default;

   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~UniformTensor() = default;
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline constexpr ConstReference operator()( size_t k, size_t i, size_t j ) const noexcept;
   inline           ConstReference at( size_t k, size_t i, size_t j ) const;
   inline constexpr ConstPointer   data  () const noexcept;
   inline constexpr ConstPointer   data  ( size_t i, size_t k ) const noexcept;
   inline constexpr ConstIterator  begin ( size_t i, size_t k ) const noexcept;
   inline constexpr ConstIterator  cbegin( size_t i, size_t k ) const noexcept;
   inline constexpr ConstIterator  end   ( size_t i, size_t k ) const noexcept;
   inline constexpr ConstIterator  cend  ( size_t i, size_t k ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline constexpr UniformTensor& operator=( const Type& rhs );

   UniformTensor& operator=( const UniformTensor& ) = default;
   UniformTensor& operator=( UniformTensor&& ) = default;

   template< typename MT > inline UniformTensor& operator= ( const Tensor<MT>& rhs );
   template< typename MT > inline UniformTensor& operator+=( const Tensor<MT>& rhs );
   template< typename MT > inline UniformTensor& operator-=( const Tensor<MT>& rhs );
   template< typename MT > inline UniformTensor& operator%=( const Tensor<MT>& rhs );
//    template< typename MT > inline UniformTensor& operator*=( const Tensor<MT>& rhs );

   template< typename ST >
   inline auto operator*=( ST rhs ) -> EnableIf_t< IsNumeric_v<ST>, UniformTensor& >;

   template< typename ST >
   inline auto operator/=( ST rhs ) -> EnableIf_t< IsNumeric_v<ST>, UniformTensor& >;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline constexpr size_t rows() const noexcept;
   inline constexpr size_t columns() const noexcept;
   inline constexpr size_t pages() const noexcept;
   inline constexpr size_t spacing() const noexcept;
   inline constexpr size_t capacity() const noexcept;
   inline constexpr size_t capacity( size_t i, size_t k ) const noexcept;
   inline           size_t nonZeros() const;
   inline           size_t nonZeros( size_t i, size_t k ) const;
   inline constexpr void   reset();
   inline constexpr void   clear();
          constexpr void   resize ( size_t o, size_t m, size_t n, bool preserve=true );
   inline constexpr void   extend ( size_t o, size_t m, size_t n, bool preserve=true );
   inline constexpr void   swap( UniformTensor& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   inline constexpr UniformTensor& transpose();
   inline constexpr UniformTensor& ctranspose();
   template< typename T >
   inline constexpr UniformTensor& transpose( const T* indices, size_t n );
   template< typename T >
   inline constexpr UniformTensor& ctranspose( const T* indices, size_t n );

   template< typename Other > inline UniformTensor& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
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

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t k, size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t k, size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t k, size_t i, size_t j ) const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t o_;      //!< The current number of pages of the tensor.
   size_t m_;      //!< The current number of rows of the tensor.
   size_t n_;      //!< The current number of columns of the tensor.
   Type   value_;  //!< The value of all elements of the uniform tensor.
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
/*!\brief The default constructor for UniformTensor.
*/
template< typename Type > // Data type of the tensor
inline constexpr UniformTensor<Type>::UniformTensor() noexcept
   : o_    ( 0UL )  // The current number of pages of the tensor
   , m_    ( 0UL )  // The current number of rows of the tensor
   , n_    ( 0UL )  // The current number of columns of the tensor
   , value_()       // The value of all elements of the uniform tensor
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a tensor of size \f$ m \times n \f$.
//
// \param m The number of rows of the tensor.
// \param n The number of columns of the tensor.
*/
template< typename Type > // Data type of the tensor
inline constexpr UniformTensor<Type>::UniformTensor( size_t o, size_t m, size_t n )
   : o_    ( o )  // The current number of pages of the tensor
   , m_    ( m )  // The current number of rows of the tensor
   , n_    ( n )  // The current number of columns of the tensor
   , value_()     // The value of all elements of the uniform tensor
{}
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
inline constexpr UniformTensor<Type>::UniformTensor( size_t o, size_t m, size_t n, const Type& init )
   : o_    ( o )     // The current number of pages of the tensor
   , m_    ( m )     // The current number of rows of the tensor
   , n_    ( n )     // The current number of columns of the tensor
   , value_( init )  // The value of all elements of the uniform tensor
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different matrices.
//
// \param m Tensor to be copied.
// \exception std::invalid_argument Invalid setup of uniform vector.
//
// The tensor is sized according to the given uniform tensor and initialized as a copy of this
// tensor.
*/
template< typename Type > // Data type of the tensor
template< typename MT    // Type of the foreign tensor
         >     // Storage order of the foreign tensor
inline UniformTensor<Type>::UniformTensor( const Tensor<MT>& m )
   : o_    ( (~m).pages()   )  // The current number of pages of the tensor
   , m_    ( (~m).rows()    )  // The current number of rows of the tensor
   , n_    ( (~m).columns() )  // The current number of columns of the tensor
   , value_()                  // The value of all elements of the uniform vector
{
   if( !IsUniform_v<MT> && !isUniform( ~m ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of uniform tensor" );
   }

   if( o_ > 0UL && m_ > 0UL && n_ > 0UL ) {
      value_ = (~m)(0UL,0UL,0UL);
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief 3D-access to the tensor elements.
//
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type > // Data type of the tensor
inline constexpr typename UniformTensor<Type>::ConstReference
   UniformTensor<Type>::operator()( size_t k, size_t i, size_t j ) const noexcept
{
   MAYBE_UNUSED( k, i, j );

   BLAZE_USER_ASSERT( k < o_, "Invalid page access index"    );
   BLAZE_USER_ASSERT( i < m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < n_, "Invalid column access index" );

   return value_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the tensor elements.
//
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid tensor access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type > // Data type of the tensor
inline typename UniformTensor<Type>::ConstReference
   UniformTensor<Type>::at( size_t k, size_t i, size_t j ) const
{
   if( k >= o_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
   }
   if( i >= m_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= n_ ) {
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
// This function returns a pointer to the internal storage of the uniform tensor. Note that you
// can NOT assume that all tensor elements lie adjacent to each other! The uniform tensor may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< typename Type > // Data type of the tensor
inline constexpr typename UniformTensor<Type>::ConstPointer
   UniformTensor<Type>::data() const noexcept
{
   return &value_;
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
template< typename Type > // Data type of the tensor
inline constexpr typename UniformTensor<Type>::ConstPointer
   UniformTensor<Type>::data( size_t i, size_t k ) const noexcept
{
   MAYBE_UNUSED( i, k );

   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );

   return &value_;
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
template< typename Type > // Data type of the tensor
inline constexpr typename UniformTensor<Type>::ConstIterator
   UniformTensor<Type>::begin( size_t i, size_t k ) const noexcept
{
   MAYBE_UNUSED( i, k );

   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );

   return ConstIterator( &value_, 0UL );
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
template< typename Type > // Data type of the tensor
inline constexpr typename UniformTensor<Type>::ConstIterator
   UniformTensor<Type>::cbegin( size_t i, size_t k ) const noexcept
{
   MAYBE_UNUSED( i, k );

   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );

   return ConstIterator( &value_, 0UL );
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
template< typename Type > // Data type of the tensor
inline constexpr typename UniformTensor<Type>::ConstIterator
   UniformTensor<Type>::end( size_t i, size_t k ) const noexcept
{
   MAYBE_UNUSED( i, k );

   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );

   return ConstIterator( &value_, n_ );
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
template< typename Type > // Data type of the tensor
inline constexpr typename UniformTensor<Type>::ConstIterator
   UniformTensor<Type>::cend( size_t i, size_t k ) const noexcept
{
   MAYBE_UNUSED( i );

   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );

   return ConstIterator( &value_, n_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Homogenous assignment to all tensor elements.
//
// \param rhs Scalar value to be assigned to all tensor elements.
// \return Reference to the assigned tensor.
*/
template< typename Type > // Data type of the tensor
inline constexpr UniformTensor<Type>& UniformTensor<Type>::operator=( const Type& rhs )
{
   value_ = rhs;

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
template< typename MT >   // Type of the right-hand side tensor
inline UniformTensor<Type>& UniformTensor<Type>::operator=( const Tensor<MT>& rhs )
{
   if( !IsUniform_v<MT> && !isUniform( ~rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment of uniform tensor" );
   }

   if( (~rhs).canAlias( this ) ) {
      UniformTensor tmp( ~rhs );
      swap( tmp );
   }
   else {
      o_ = (~rhs).pages();
      m_ = (~rhs).rows();
      n_ = (~rhs).columns();

      if( o_ > 0UL && m_ > 0UL && n_ > 0UL ) {
         value_ = (~rhs)(0UL,0UL,0UL);
      }
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
template< typename Type > // Data type of the tensor
template< typename MT >   // Type of the right-hand side tensor
inline UniformTensor<Type>& UniformTensor<Type>::operator+=( const Tensor<MT>& rhs )
{
   if( (~rhs).pages() != o_ || (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   if( !IsUniform_v<MT> && !isUniform( ~rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid addition assignment to uniform tensor" );
   }

   if( o_ > 0UL && m_ > 0UL && n_ > 0UL ) {
      value_ += (~rhs)(0UL,0UL,0UL);
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
template< typename Type > // Data type of the tensor
template< typename MT >   // Type of the right-hand side tensor
inline UniformTensor<Type>& UniformTensor<Type>::operator-=( const Tensor<MT>& rhs )
{
   if( (~rhs).pages() != o_ || (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   if( !IsUniform_v<MT> && !isUniform( ~rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid subtraction assignment to uniform tensor" );
   }

   if( o_ > 0UL && m_ > 0UL && n_ > 0UL ) {
      value_ -= (~rhs)(0UL,0UL,0UL);
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
template< typename Type > // Data type of the tensor
template< typename MT >   // Type of the right-hand side tensor
inline UniformTensor<Type>& UniformTensor<Type>::operator%=( const Tensor<MT>& rhs )
{
   if( (~rhs).pages() != o_ || (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
   }

   if( !IsUniform_v<MT> && !isUniform( ~rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid Schur product assignment to uniform tensor" );
   }

   if( o_ > 0UL && m_ > 0UL && n_ > 0UL ) {
      value_ *= (~rhs)(0UL,0UL,0UL);
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication of a tensor (\f$ A*=B \f$).
//
// \param rhs The right-hand side tensor for the multiplication.
// \return Reference to the tensor.
// \exception std::invalid_argument Tensor sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
// template< typename Type > // Data type of the tensor
// template< typename MT >   // Type of the right-hand side tensor
// inline UniformTensor<Type>& UniformTensor<Type>::operator*=( const Tensor<MT>& rhs )
// {
//    if( (~rhs).rows() != n_ ) {
//       BLAZE_THROW_INVALID_ARGUMENT( "Tensor sizes do not match" );
//    }
//
//    if( !IsUniform_v<MT> && !isUniform( ~rhs ) ) {
//       BLAZE_THROW_INVALID_ARGUMENT( "Invalid multiplication assignment to uniform tensor" );
//    }
//
//    n_ = (~rhs).columns();
//
//    if( m_ > 0UL && n_ > 0UL ) {
//       value_ = ( value_ * (~rhs)(0UL,0UL) ) * Type( (~rhs).rows() );
//    }
//
//    return *this;
// }
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication between a tensor and
//        a scalar value (\f$ A*=s \f$).
//
// \param scalar The right-hand side scalar value for the multiplication.
// \return Reference to the tensor.
*/
template< typename Type > // Data type of the tensor
template< typename ST >  // Data type of the right-hand side scalar
inline auto UniformTensor<Type>::operator*=( ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, UniformTensor& >
{
   if( pages() > 0UL && rows() > 0UL && columns() > 0UL ) {
      value_ *= scalar;
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment operator for the division between a tensor and a scalar value
//        (\f$ A/=s \f$).
//
// \param scalar The right-hand side scalar value for the division.
// \return Reference to the tensor.
*/
template< typename Type > // Data type of the tensor
template< typename ST >  // Data type of the right-hand side scalar
inline auto UniformTensor<Type>::operator/=( ST scalar )
   -> EnableIf_t< IsNumeric_v<ST>, UniformTensor& >
{
   if( pages() > 0UL && rows() > 0UL && columns() > 0UL ) {
      value_ /= scalar;
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
/*!\brief Returns the current number of pages of the tensor.
//
// \return The number of pages of the tensor.
*/
template< typename Type > // Data type of the tensor
inline constexpr size_t UniformTensor<Type>::pages() const noexcept
{
   return o_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of rows of the tensor.
//
// \return The number of rows of the tensor.
*/
template< typename Type > // Data type of the tensor
inline constexpr size_t UniformTensor<Type>::rows() const noexcept
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
inline constexpr size_t UniformTensor<Type>::columns() const noexcept
{
   return n_;
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
inline constexpr size_t UniformTensor<Type>::spacing() const noexcept
{
   return n_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the tensor.
//
// \return The capacity of the tensor.
*/
template< typename Type > // Data type of the tensor
inline constexpr size_t UniformTensor<Type>::capacity() const noexcept
{
   return o_ * m_ * n_;
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
template< typename Type > // Data type of the tensor
inline constexpr size_t UniformTensor<Type>::capacity( size_t i, size_t k ) const noexcept
{
   MAYBE_UNUSED( i, k );
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );
   return n_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of non-zero elements in the tensor
//
// \return The number of non-zero elements in the dense tensor.
*/
template< typename Type > // Data type of the tensor
inline size_t UniformTensor<Type>::nonZeros() const
{
   if( o_ == 0UL || m_ == 0UL || n_ == 0UL || isDefault( value_ ) )
      return 0UL;
   else
      return o_ * m_ * n_;
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
inline size_t UniformTensor<Type>::nonZeros( size_t i, size_t k ) const
{
   MAYBE_UNUSED( i );

   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor page access index" );

   if( n_ == 0UL || isDefault( value_ ) )
      return 0UL;
   else
      return n_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename Type > // Data type of the tensor
inline constexpr void UniformTensor<Type>::reset()
{
   using blaze::clear;

   clear( value_ );
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
inline constexpr void UniformTensor<Type>::clear()
{
   o_ = 0UL;
   m_ = 0UL;
   n_ = 0UL;
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
// This function resizes the tensor using the given size to \f$ m \times n \f$. Note that this
// function may invalidate all existing views (submatrices, rows, columns, ...) on the tensor
// if it is used to shrink the tensor. Additionally, the resize operation potentially changes
// all tensor elements. In order to preserve the old tensor values, the \a preserve flag can
// be set to \a true.
*/
template< typename Type > // Data type of the tensor
void constexpr UniformTensor<Type>::resize( size_t k, size_t m, size_t n, bool preserve )
{
   MAYBE_UNUSED( preserve );

   o_  = k;
   m_  = m;
   n_  = n;
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
// This function increases the tensor size by \a m rows and \a n columns. Note that this function
// potentially changes all tensor elements. In order to preserve the old tensor values, the
// \a preserve flag can be set to \a true.
*/
template< typename Type > // Data type of the tensor
inline constexpr void UniformTensor<Type>::extend( size_t o, size_t m, size_t n, bool preserve )
{
   resize( o_+o, m_+m, n_+n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two matrices.
//
// \param m The tensor to be swapped.
// \return void
*/
template< typename Type > // Data type of the tensor
inline constexpr void UniformTensor<Type>::swap( UniformTensor& m ) noexcept
{
   using std::swap;

   swap( o_, m.o_ );
   swap( m_, m.m_ );
   swap( n_, m.n_ );
   swap( value_, m.value_ );
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
template< typename Type > // Data type of the tensor
inline constexpr UniformTensor<Type>& UniformTensor<Type>::transpose()
{
   using std::swap;

   swap( o_, n_ );      // {2, 1, 0}

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place transpose of the tensor.
//
// \return Reference to the transposed tensor.
*/
template< typename Type > // Data type of the tensor
template< typename T >    // Type of the mapping indices
inline constexpr UniformTensor<Type>& UniformTensor<Type>::transpose( const T* indices, size_t n )
{
   using std::swap;

   if ( indices[0] == 0 )
   {
      if ( indices[1] == 2 )
      {
         swap( m_, n_ );      // {0, 2, 1}
      }
   }
   else if ( indices[0] == 1 )
   {
      if ( indices[1] == 2 )
      {
         auto t = o_;         // {1, 2, 0}
         o_ = m_;
         m_ = n_;
         n_ = t;
      }
      else
      {
         swap ( o_, m_ );     // {1, 0, 2}
      }
   }
   else
   {
      // indices[0] == 2
      if ( indices[1] == 1 )
      {
         swap( o_, n_ );      // {2, 1, 0}
      }
      else
      {
         auto t = o_;         // {2, 0, 1}
         o_ = n_;
         n_ = m_;
         m_ = t;
      }
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the tensor.
//
// \return Reference to the transposed tensor.
*/
template< typename Type >  // Data type of the tensor
inline constexpr UniformTensor<Type>& UniformTensor<Type>::ctranspose()
{
   using std::swap;

   transpose();
   conjugate( value_ );

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
inline constexpr UniformTensor<Type>& UniformTensor<Type>::ctranspose( const T* indices, size_t n )
{
   using std::swap;

   transpose( indices, n );
   conjugate( value_ );

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
   blaze::UniformTensor<int> A;
   // ... Resizing and initialization
   A *= 4;        // Scaling of the tensor
   A.scale( 4 );  // Same effect as above
   \endcode
*/
template< typename Type >    // Data type of the tensor
template< typename Other >  // Data type of the scalar value
inline UniformTensor<Type>& UniformTensor<Type>::scale( const Other& scalar )
{
   if( o_ > 0UL && m_ > 0UL && n_ > 0UL ) {
      value_ *= scalar;
   }

   return *this;
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
template< typename Type >    // Data type of the tensor
template< typename Other >  // Data type of the foreign expression
inline bool UniformTensor<Type>::canAlias( const Other* alias ) const noexcept
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
template< typename Type >    // Data type of the tensor
template< typename Other >  // Data type of the foreign expression
inline bool UniformTensor<Type>::isAliased( const Other* alias ) const noexcept
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
inline bool UniformTensor<Type>::isAligned() const noexcept
{
   return true;
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
inline bool UniformTensor<Type>::canSMPAssign() const noexcept
{
   return ( pages() * rows() * columns() >= SMP_DMATASSIGN_THRESHOLD );
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
template< typename Type > // Data type of the tensor
BLAZE_ALWAYS_INLINE typename UniformTensor<Type>::SIMDType
   UniformTensor<Type>::load( size_t k, size_t i, size_t j ) const noexcept
{
   return loada( k, i, j );
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
template< typename Type > // Data type of the tensor
BLAZE_ALWAYS_INLINE typename UniformTensor<Type>::SIMDType
   UniformTensor<Type>::loada( size_t k, size_t i, size_t j ) const noexcept
{
   MAYBE_UNUSED( k, i, j );

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( k < o_, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= n_, "Invalid column access index" );

   return set( value_ );
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
template< typename Type > // Data type of the tensor
BLAZE_ALWAYS_INLINE typename UniformTensor<Type>::SIMDType
   UniformTensor<Type>::loadu( size_t k, size_t i, size_t j ) const noexcept
{
   MAYBE_UNUSED( k, i, j );

   BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE( Type );

   BLAZE_INTERNAL_ASSERT( k < o_, "Invalid page access index" );
   BLAZE_INTERNAL_ASSERT( i < m_, "Invalid row access index" );
   BLAZE_INTERNAL_ASSERT( j < n_, "Invalid column access index" );
   BLAZE_INTERNAL_ASSERT( j + SIMDSIZE <= n_, "Invalid column access index" );

   return set( value_ );
}
//*************************************************************************************************








//=================================================================================================
//
//  UNIFORMTENSOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name UniformTensor operators */
//@{
template< typename Type, bool SO >
constexpr void reset( UniformTensor<Type>& m );

template< typename Type, bool SO >
constexpr void clear( UniformTensor<Type>& m );

template< bool RF, typename Type, bool SO >
constexpr bool isDefault( const UniformTensor<Type>& m );

template< typename Type, bool SO >
constexpr bool isIntact( const UniformTensor<Type>& m ) noexcept;

template< typename Type, bool SO >
constexpr void swap( UniformTensor<Type>& a, UniformTensor<Type>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given uniform tensor.
// \ingroup uniform_tensor
//
// \param m The tensor to be resetted.
// \return void
*/
template< typename Type > // Data type of the tensor
inline constexpr void reset( UniformTensor<Type>& m )
{
   m.reset();
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
inline constexpr void clear( UniformTensor<Type>& m )
{
   m.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given uniform tensor is in default state.
// \ingroup uniform_tensor
//
// \param m The tensor to be tested for its default state.
// \return \a true in case the given tensor's rows and columns are zero, \a false otherwise.
//
// This function checks whether the uniform tensor is in default (constructed) state, i.e. if
// it's number of rows and columns is 0. In case it is in default state, the function returns
// \a true, else it will return \a false. The following example demonstrates the use of the
// \a isDefault() function:

   \code
   blaze::UniformTensor<int> A;
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
inline constexpr bool isDefault( const UniformTensor<Type>& m )
{
   return ( m.pages() == 0UL && m.rows() == 0UL && m.columns() == 0UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given uniform tensor are intact.
// \ingroup uniform_tensor
//
// \param m The uniform tensor to be tested.
// \return \a true in case the given tensor's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the uniform tensor are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   blaze::UniformTensor<int> A;
   // ... Resizing and initialization
   if( isIntact( A ) ) { ... }
   \endcode
*/
template< typename Type > // Data type of the tensor
inline constexpr bool isIntact( const UniformTensor<Type>& m ) noexcept
{
   MAYBE_UNUSED( m );

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two uniform matrices.
// \ingroup uniform_tensor
//
// \param a The first tensor to be swapped.
// \param b The second tensor to be swapped.
// \return void
*/
template< typename Type > // Data type of the tensor
inline constexpr void swap( UniformTensor<Type>& a, UniformTensor<Type>& b ) noexcept
{
   a.swap( b );
}
//*************************************************************************************************




//=================================================================================================
//
//  ISUNIFORM SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename Type >
struct IsUniform< UniformTensor<Type> >
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
template< typename Type >
struct IsAligned< UniformTensor<Type> >
   : public TrueType
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
template< typename Type >
struct IsResizable< UniformTensor<Type> >
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
struct AddTraitEval1< T1, T2
                    , EnableIf_t< IsTensor_v<T1> &&
                                  IsTensor_v<T2> &&
                                  ( IsUniform_v<T1> && IsUniform_v<T2> ) &&
                                  !( IsZero_v<T1> || IsZero_v<T2> ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = UniformTensor< AddTrait_t<ET1,ET2> >;
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
struct SubTraitEval1< T1, T2
                    , EnableIf_t< IsTensor_v<T1> &&
                                  IsTensor_v<T2> &&
                                  ( IsUniform_v<T1> && IsUniform_v<T2> ) &&
                                  !( IsZero_v<T1> || IsZero_v<T2> ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = UniformTensor< SubTrait_t<ET1,ET2> >;
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
struct SchurTraitEval1< T1, T2
                      , EnableIf_t< IsTensor_v<T1> &&
                                    IsTensor_v<T2> &&
                                    ( IsUniform_v<T1> && IsUniform_v<T2> ) &&
                                    !( IsZero_v<T1> || IsZero_v<T2> ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = UniformTensor< MultTrait_t<ET1,ET2> >;
};

template< typename T1, typename T2 >
struct SchurTraitEval1< T1, T2
                      , EnableIf_t< IsTensor_v<T1> &&
                                    IsMatrix_v<T2> &&
                                    ( IsUniform_v<T1> && IsUniform_v<T2> ) &&
                                    !( IsZero_v<T1> || IsZero_v<T2> ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = UniformTensor< MultTrait_t<ET1,ET2> >;
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
struct MultTraitEval1< T1, T2
                     , EnableIf_t< IsTensor_v<T1> &&
                                   IsUniform_v<T1> &&
                                   !IsZero_v<T1> &&
                                   IsNumeric_v<T2> > >
{
   using ET1 = ElementType_t<T1>;

   using Type = UniformTensor< MultTrait_t<ET1,T2> >;
};

template< typename T1, typename T2 >
struct MultTraitEval1< T1, T2
                     , EnableIf_t< IsNumeric_v<T1> &&
                                   IsTensor_v<T2> &&
                                   IsUniform_v<T2> &&
                                   !IsZero_v<T2> > >
{
   using ET2 = ElementType_t<T2>;

   using Type = UniformTensor< MultTrait_t<T1,ET2> >;
};

// template< typename T1, typename T2 >
// struct MultTraitEval1< T1, T2
//                      , EnableIf_t< IsColumnVector_v<T1> &&
//                                    IsRowVector_v<T2> &&
//                                    ( IsUniform_v<T1> && IsUniform_v<T2> ) &&
//                                    !( IsZero_v<T1> || IsZero_v<T2> ) > >
// {
//    using ET1 = ElementType_t<T1>;
//    using ET2 = ElementType_t<T2>;
//
//    using Type = UniformTensor< MultTrait_t<ET1,ET2> >;
// };

template< typename T1, typename T2 >
struct MultTraitEval1< T1, T2
                     , EnableIf_t< IsTensor_v<T1> &&
                                   IsTensor_v<T2> &&
                                   ( IsUniform_v<T1> && IsUniform_v<T2> ) &&
                                   !( IsZero_v<T1> || IsZero_v<T2> ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = UniformTensor< MultTrait_t<ET1,ET2> >;
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
struct DivTraitEval1< T1, T2
                    , EnableIf_t< IsTensor_v<T1> &&
                                  IsNumeric_v<T2> &&
                                  IsUniform_v<T1> && !IsZero_v<T1> > >
{
   using ET1 = ElementType_t<T1>;

   using Type = UniformTensor< DivTrait_t<ET1,T2> >;
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
struct UnaryMapTraitEval1< T, OP
                         , EnableIf_t< IsTensor_v<T> &&
                                       YieldsUniform_v<OP,T> &&
                                       !YieldsZero_v<OP,T> > >
{
   using ET = ElementType_t<T>;

   using Type = UniformTensor< MapTrait_t<ET,OP> >;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2, typename OP >
struct BinaryMapTraitEval1< T1, T2, OP
                          , EnableIf_t< IsTensor_v<T1> &&
                                        IsTensor_v<T2> &&
                                        YieldsUniform_v<OP,T1,T2> &&
                                        !YieldsZero_v<OP,T1,T2> > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = UniformTensor< MapTrait_t<ET1,ET2,OP> >;
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
struct HighType< UniformTensor<T1>, UniformTensor<T2> >
{
   using Type = UniformTensor< typename HighType<T1,T2>::Type >;
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
struct LowType< UniformTensor<T1>, UniformTensor<T2> >
{
   using Type = UniformTensor< typename LowType<T1,T2>::Type >;
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
struct SubtensorTraitEval1< MT, K, I, J, O, M, N
                          , EnableIf_t< IsUniform_v<MT> && !IsZero_v<MT> > >
{
   using Type = UniformTensor< RemoveConst_t< ElementType_t<MT> > >;
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
struct ColumnSliceTraitEval2< MT, M
                            , EnableIf_t< IsUniform_v<MT> && !IsZero_v<MT> > >
{
   using Type = UniformMatrix< RemoveConst_t< ElementType_t<MT> >, rowMajor >;
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
struct PageSliceTraitEval2< MT, M
                          , EnableIf_t< IsUniform_v<MT> && !IsZero_v<MT> > >
{
   using Type = UniformMatrix< RemoveConst_t< ElementType_t<MT> >, rowMajor >;
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
struct RowSliceTraitEval2< MT, M
                         , EnableIf_t< IsUniform_v<MT> && !IsZero_v<MT> > >
{
   using Type = UniformMatrix< RemoveConst_t< ElementType_t<MT> >, columnMajor >;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
