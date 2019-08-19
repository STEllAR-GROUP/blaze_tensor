//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/InitializerTensor.h
//  \brief Header file for the implementation of a tensor representation of an initializer list
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_INITIALIZERTENSOR_H_
#define _BLAZE_TENSOR_MATH_DENSE_INITIALIZERTENSOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <iterator>
#include <blaze/math/dense/InitializerIterator.h>
#include <blaze/math/Exception.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HighType.h>
#include <blaze/math/typetraits/IsInitializer.h>
#include <blaze/math/typetraits/LowType.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/Types.h>
#include <blaze/util/MaybeUnused.h>

#include <blaze_tensor/math/Forward.h>
#include <blaze_tensor/math/InitializerList.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup initializer_tensor InitializerTensor
// \ingroup dense_tensor
*/
/*!\brief Dense tensor representation of an initializer list.
// \ingroup initializer_tensor
//
// The InitializerTensor class template is a dense tensor representation of an (extended)
// initializer list of arbitrary type. The type of the elements of the tensor can be specified
// via the single template parameters:

   \code
   template< typename Type >
   class InitializerTensor;
   \endcode

// \a Type specifies the type of the tensor elements. InitializerTensor can be used with any
// non-cv-qualified, non-reference, non-pointer element type.
//
// On construction, an InitializerTensor is immediately bound to an initializer list:

   \code
   const blaze::initializer_list< initializer_list<int> > list = { { 2, 6, -1 },
                                                                   { 3, 5 } };

   blaze::InitializerTensor<int> A( list );  // Representation of the initializer list as dense tensor
   \endcode

// It is possible to only represent an extended initializer list by explicitly specifying the
// number of columns:

   \code
   const initializer_list< initializer_list<int> > list = { { 2, 6, -1 },
                                                            { 3, 5 } };

   blaze::InitializerVector<int> B( list, 3UL );  // Representation of the original initializer list
   blaze::InitializerVector<int> C( list, 4UL );  // Representing the initializer list { { 2, 6, -1, 0 }, { 3, 5, 0, 0 } }
   \endcode

// Since an InitializerTensor represents a specific initializer list, its lifetime is bound to
// the lifetime of the according initializer list. When the initializer list goes out of scope
// access to the initializer list via an InitializerTensor results in undefined behavior:

   \code
   blaze::InitializerTensor<int> D{ { 1, 2, 3 }, { 4, 5, 6 } };         // Undefined behavior!
   blaze::InitializerTensor<int> E( { { 0, 3, 2 }, { -1, 1 } }, 3UL );  // Undefined behavior!
   \endcode

// Also, an InitializerTensor can only be used on the right of an assignment as its elements
// are considered to be immutable. The following example gives an impression on the usage of an
// InitializerTensor:

   \code
   const blaze::initializer_list< initializer_list<int> > list = { { 2, 6, -1 },
                                                                   { 3, 5 } };

   blaze::InitializerTensor<int> F( list );  // Representation of the initializer list as dense tensor
   blaze::DynamicTensor<int> G;

   G = F;  // Initialize vector G via vector F
   F = G;  // Compilation error! Cannot assign to an initializer tensor
   \endcode

// An initializer tensor can be used as operand in arithmetic operations. All operations (addition,
// subtraction, multiplication, scaling, ...) can be performed on all possible combinations of
// row-major and column-major dense and sparse tensors with fitting element types. The following
// example gives an impression of the use of InitializerTensor:

   \code
   using blaze::initializer_list;
   using blaze::InitializerTensor;
   using blaze::DynamicTensor;
   using blaze::CompressedTensor;
   using blaze::rowMajor;
   using blaze::columnMajor;

   const blaze::initializer_list< initializer_list<double> > list = { { 1.0, 2.0, 3.0 },
                                                                      { 4.0, 5.0, 6.0 } };

   InitializerTensor<double> A( list );

   DynamicTensor<float,columnMajor> B( 2, 3 );  // Default constructed column-major single precision 2x3 tensor
   B(0,0) = 1.0; B(0,1) = 3.0; B(0,2) = 5.0;    // Initialization of the first row
   B(1,0) = 2.0; B(1,1) = 4.0; B(1,2) = 6.0;    // Initialization of the second row

   CompressedTensor<float> C( 2, 3 );        // Empty row-major sparse single precision tensor
   DynamicTensor<float>    D( 3, 2, 4.0F );  // Directly, homogeneously initialized single precision 3x2 tensor

   DynamicTensor<double,rowMajor>    E( A );  // Creation of a new row-major tensor as a copy of A
   DynamicTensor<double,columnMajor> F;       // Creation of a default column-major tensor

   E = A + B;    // Tensor addition and assignment to a row-major tensor
   F = A - C;    // Tensor subtraction and assignment to a column-major tensor
   F = A * D;    // Tensor multiplication between two tensors of different element types

   E = 2.0 * B;  // Scaling of tensor B
   F = D * 2.0;  // Scaling of tensor D

   E += A - B;   // Addition assignment
   E -= A + C;   // Subtraction assignment
   F *= A * D;   // Multiplication assignment
   \endcode
*/
template< typename Type >  // Data type of the tensor
class InitializerTensor
   : public DenseTensor< InitializerTensor<Type> >
{
 public:
   //**Type definitions****************************************************************************
   using This          = InitializerTensor<Type>;    //!< Type of this InitializerTensor instance.
   using BaseType      = DenseTensor<This>;          //!< Base type of this InitializerTensor instance.
   using ResultType    = DynamicTensor<Type>;        //!< Result type for expression template evaluations.
   using OppositeType  = DynamicTensor<Type>;        //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = DynamicTensor<Type>;        //!< Transpose type for expression template evaluations.
   using ElementType   = Type;                       //!< Type of the tensor elements.
   using ReturnType    = const Type&;                //!< Return type for expression template evaluations.
   using CompositeType = const This&;                //!< Data type for composite expression templates.

   using Reference      = const Type&;  //!< Reference to a non-constant tensor value.
   using ConstReference = const Type&;  //!< Reference to a constant tensor value.
   using Pointer        = const Type*;  //!< Pointer to a non-constant tensor value.
   using ConstPointer   = const Type*;  //!< Pointer to a constant tensor value.

   using Iterator      = InitializerIterator<Type>;  //!< Iterator over non-constant elements.
   using ConstIterator = InitializerIterator<Type>;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a InitializerTensor with different data/element type.
   */
   template< typename NewType >  // Data type of the other tensor
   struct Rebind {
      using Other = InitializerTensor<NewType>;  //!< The type of the other InitializerTensor.
   };
   //**********************************************************************************************

   //**Resize struct definition********************************************************************
   /*!\brief Resize mechanism to obtain a InitializerTensor with different fixed dimensions.
   */
   template< size_t NewM    // Number of rows of the other tensor
           , size_t NewN    // Number of columns of the other tensor
           , size_t NewO >  // Number of pages of the other tensor
   struct Resize {
      using Other = InitializerTensor<Type>;  //!< The type of the other InitializerTensor.
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SIMD optimization.
   /*! The \a simdEnabled compilation flag indicates whether expressions the tensor is involved
       in can be optimized via SIMD operations. In case the element type of the tensor is a
       vectorizable data type, the \a simdEnabled compilation flag is set to \a true, otherwise
       it is set to \a false. */
   static constexpr bool simdEnabled = false;

   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the tensor can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   static constexpr bool smpAssignable = false;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline InitializerTensor( initializer_list< initializer_list< initializer_list<Type> > > list ) noexcept;
   explicit inline InitializerTensor( initializer_list< initializer_list< initializer_list<Type> > > list, size_t m, size_t n );

   InitializerTensor( const InitializerTensor& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~InitializerTensor() = default;
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline ConstReference operator()( size_t k, size_t i, size_t j ) const noexcept;
   inline ConstReference at( size_t k, size_t i, size_t j ) const;
   inline ConstPointer   data  () const noexcept;
   inline ConstPointer   data  ( size_t i, size_t k ) const noexcept;
   inline ConstIterator  begin ( size_t i, size_t k ) const noexcept;
   inline ConstIterator  cbegin( size_t i, size_t k ) const noexcept;
   inline ConstIterator  end   ( size_t i, size_t k ) const noexcept;
   inline ConstIterator  cend  ( size_t i, size_t k ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   InitializerTensor& operator=( const InitializerTensor& ) = delete;
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
   inline void   swap( InitializerTensor& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other > inline bool canAlias ( const Other* alias ) const noexcept;
   template< typename Other > inline bool isAliased( const Other* alias ) const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Type definitions****************************************************************************
   using ListType = initializer_list< initializer_list< initializer_list<Type> > >;  //!< Type of the represented initializer list.
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t m_;       //!< The current number of rows of the tensor.
   size_t n_;       //!< The current number of columns of the tensor.
   size_t o_;       //!< The current number of pages of the tensor.
   ListType list_;  //!< The initializer list represented by the tensor.
                    /*!< Access to the tensor elements is gained via the function call
                         operator. The memory layout of the elements is
                         \f[\left(\begin{array}{*{5}{c}}
                         0            & 1             & 2             & \cdots & N-1         \\
                         N            & N+1           & N+2           & \cdots & 2 \cdot N-1 \\
                         \vdots       & \vdots        & \vdots        & \ddots & \vdots      \\
                         M \cdot N-N  & M \cdot N-N+1 & M \cdot N-N+2 & \cdots & M \cdot N-1 \\
                         \end{array}\right)\f]. */

   static const Type zero_;  //!< Neutral element for accesses to zero elements.
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
//  DEFINITION AND INITIALIZATION OF THE STATIC MEMBER VARIABLES
//
//=================================================================================================

template< typename Type >  // Data type of the tensor
const Type InitializerTensor<Type>::zero_{};




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for InitializerTensor.
//
// \param list The initializer list represented by the tensor.
*/
template< typename Type >  // Data type of the tensor
inline InitializerTensor<Type>::InitializerTensor( initializer_list< initializer_list< initializer_list<Type> > > list ) noexcept
   : o_   ( list.size() )               // The current number of pages of the tensor
   , m_   ( determineRows( list ) )     // The current number of rows of the tensor
   , n_   ( determineColumns( list ) )  // The current number of columns of the tensor
   , list_( list )                      // The initializer list represented by the tensor
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for InitializerTensor.
//
// \param list The initializer list represented by the tensor.
// \param n The number of columns of the tensor.
// \exception std::invalid_argument Invalid initializer list dimension.
*/
template< typename Type >  // Data type of the tensor
inline InitializerTensor<Type>::InitializerTensor( initializer_list< initializer_list< initializer_list<Type> > > list, size_t m, size_t n )
   : o_   ( list.size() )  // The current number of pages of the tensor
   , m_   ( m )            // The current number of rows of the tensor
   , n_   ( n    )         // The current number of columns of the tensor
   , list_( list )         // The initializer list represented by the tensor
{
   if( m < determineRows( list ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid initializer list dimension" );
   }
   if( n < determineColumns( list ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid initializer list dimension" );
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
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \param k Access index for the page. The index has to be in the range \f$[0..O-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type >  // Data type of the tensor
inline typename InitializerTensor<Type>::ConstReference
   InitializerTensor<Type>::operator()( size_t k, size_t i, size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   BLAZE_USER_ASSERT( k<o_, "Invalid page access index" );

   if( k >= list_.size() )
      return zero_;

   const initializer_list< initializer_list<Type> >& rowlist( list_.begin()[k] );
   if( i >= rowlist.size() )
      return zero_;

   const initializer_list<Type>& list( rowlist.begin()[i] );
   if( j < list.size() )
      return list.begin()[j];
   else
      return zero_;
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
template< typename Type >  // Data type of the tensor
inline typename InitializerTensor<Type>::ConstReference
   InitializerTensor<Type>::at( size_t k, size_t i, size_t j ) const
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
template< typename Type >  // Data type of the tensor
inline typename InitializerTensor<Type>::ConstPointer
   InitializerTensor<Type>::data() const noexcept
{
   return list_.begin()->begin()->begin();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the tensor elements of row \a i.
//
// \param i The row index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row \a i.
*/
template< typename Type >  // Data type of the tensor
inline typename InitializerTensor<Type>::ConstPointer
   InitializerTensor<Type>::data( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   BLAZE_USER_ASSERT( k < o_, "Invalid dense tensor row access index" );
   return list_.begin()[k].begin()[i];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row \a i.
//
// \param i The row index.
// \return Iterator to the first element of row \a i.
//
// This function returns a row iterator to the first element of row \a i.
*/
template< typename Type >  // Data type of the tensor
inline typename InitializerTensor<Type>::ConstIterator
   InitializerTensor<Type>::begin( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   return ConstIterator( 0UL, list_.begin()[k].begin()[i] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row \a i.
//
// \param i The row index.
// \return Iterator to the first element of row \a i.
//
// This function returns a row iterator to the first element of row \a i
*/
template< typename Type >  // Data type of the tensor
inline typename InitializerTensor<Type>::ConstIterator
   InitializerTensor<Type>::cbegin( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   return ConstIterator( 0UL, list_.begin()[k].begin()[i] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row \a i.
//
// \param i The row index.
// \return Iterator just past the last element of row \a i.
//
// This function returns an row iterator just past the last element of row \a i.
*/
template< typename Type >  // Data type of the tensor
inline typename InitializerTensor<Type>::ConstIterator
   InitializerTensor<Type>::end( size_t i, size_t k    ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   return ConstIterator( n_, list_.begin()[k].begin()[i] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row \a i.
//
// \param i The row index.
// \return Iterator just past the last element of row \a i.
//
// This function returns an row iterator just past the last element of row \a i.
*/
template< typename Type >  // Data type of the tensor
inline typename InitializerTensor<Type>::ConstIterator
   InitializerTensor<Type>::cend( size_t i, size_t k ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense tensor row access index" );
   return ConstIterator( n_, list_.begin()[k].begin()[i] );
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
template< typename Type >  // Data type of the tensor
inline size_t InitializerTensor<Type>::rows() const noexcept
{
   return m_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of columns of the tensor.
//
// \return The number of columns of the tensor.
*/
template< typename Type >  // Data type of the tensor
inline size_t InitializerTensor<Type>::columns() const noexcept
{
   return n_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of pages of the tensor.
//
// \return The number of pages of the tensor.
*/
template< typename Type >  // Data type of the tensor
inline size_t InitializerTensor<Type>::pages() const noexcept
{
   return o_;
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
template< typename Type >  // Data type of the tensor
inline size_t InitializerTensor<Type>::spacing() const noexcept
{
   return m_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the tensor.
//
// \return The capacity of the tensor.
*/
template< typename Type >  // Data type of the tensor
inline size_t InitializerTensor<Type>::capacity() const noexcept
{
   return m_ * n_ * o_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current capacity of the specified row.
//
// \param i The index of the row.
// \return The current capacity of row \a i.
//
// This function returns the current capacity of the specified row.
*/
template< typename Type >  // Data type of the tensor
inline size_t InitializerTensor<Type>::capacity( size_t i, size_t k ) const noexcept
{
   MAYBE_UNUSED( i, k );
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid row access index" );
   return n_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of non-zero elements in the tensor
//
// \return The number of non-zero elements in the dense tensor.
*/
template< typename Type >  // Data type of the tensor
inline size_t InitializerTensor<Type>::nonZeros() const
{
   size_t nonzeros( 0 );

   for( const auto& pageList : list_ ) {
      for( const auto& rowList : pageList ) {
         for( size_t i=0UL; i<rowList.size(); ++i ) {
            if( !isDefault( rowList.begin()[i] ) )
               ++nonzeros;
         }
      }
   }
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
template< typename Type >  // Data type of the tensor
inline size_t InitializerTensor<Type>::nonZeros( size_t i, size_t k ) const
{
   using blaze::nonZeros;

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   BLAZE_USER_ASSERT( k < pages(), "Invalid page access index" );

   if( k >= list_.size() ) return 0UL;

   auto& rowList = list_.begin()[k];

   if( i >= rowList.size() ) return 0UL;

   return nonZeros( rowList.begin()[i] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two tensors.
//
// \param m The tensor to be swapped.
// \return void
*/
template< typename Type >  // Data type of the tensor
inline void InitializerTensor<Type>::swap( InitializerTensor& m ) noexcept
{
   using std::swap;

   swap( o_   , m.o_    );
   swap( m_   , m.m_    );
   swap( n_   , m.n_    );
   swap( list_, m.list_ );
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
template< typename Type >   // Data type of the tensor
template< typename Other >  // Data type of the foreign expression
inline bool InitializerTensor<Type>::canAlias( const Other* alias ) const noexcept
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
template< typename Type >   // Data type of the tensor
template< typename Other >  // Data type of the foreign expression
inline bool InitializerTensor<Type>::isAliased( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************




//=================================================================================================
//
//  INITIALIZERTENSOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name InitializerTensor operators */
//@{
template< typename Type >
inline bool isIntact( const InitializerTensor<Type>& m ) noexcept;

template< typename Type >
inline void swap( InitializerTensor<Type>& a, InitializerTensor<Type>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given initializer tensor are intact.
// \ingroup initializer_tensor
//
// \param m The initializer tensor to be tested.
// \return \a true in case the given tensor's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the initializer tensor are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it will
// return \a false. The following example demonstrates the use of the \a isIntact() function:

   \code
   blaze::InitializerTensor<int> A;
   // ... Resizing and initialization
   if( isIntact( A ) ) { ... }
   \endcode
*/
template< typename Type >
inline bool isIntact( const InitializerTensor<Type>& m ) noexcept
{
   MAYBE_UNUSED( m );

   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two initializer tensors.
// \ingroup initializer_tensor
//
// \param a The first tensor to be swapped.
// \param b The second tensor to be swapped.
// \return void
*/
template< typename Type >
inline void swap( InitializerTensor<Type>& a, InitializerTensor<Type>& b ) noexcept
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
template< typename T >
struct HasConstDataAccess< InitializerTensor<T> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISINITIALIZER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T >
struct IsInitializer< InitializerTensor<T> >
   : public TrueType
{};
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
struct HighType< InitializerTensor<T1>, InitializerTensor<T2> >
{
   using Type = InitializerTensor< typename HighType<T1,T2>::Type >;
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
struct LowType< InitializerTensor<T1>, InitializerTensor<T2> >
{
   using Type = InitializerTensor< typename LowType<T1,T2>::Type >;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
