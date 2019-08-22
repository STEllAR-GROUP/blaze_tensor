//=================================================================================================
/*!
//  \file blaze_tensor/math/expressions/DQuatTransposer.h
//  \brief Header file for the dense tensor transposer
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DQUATTRANSPOSER_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DQUATTRANSPOSER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/simd/SIMDTrait.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/MaxSize.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Types.h>

#include <blaze_tensor/math/constraints/DenseArray.h>
#include <blaze_tensor/math/expressions/DenseArray.h>
#include <blaze_tensor/math/expressions/DQuatTransExprData.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DQUATTRANSPOSER
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for the transposition of a dense tensor.
// \ingroup dense_tensor_expression
//
// The DQuatTransposer class is a wrapper object for the temporary transposition of a dense tensor.
*/
template< typename MT,        // Type of the dense tensor
          size_t... CTAs >    // Compile time arguments
class DQuatTransposer
   : public DenseArray< DQuatTransposer<MT> >
   , public DQuatTransExprData<CTAs...>
{
 public:
   //**Type definitions****************************************************************************
   using DataType       = DQuatTransExprData<CTAs...>;  //!< The type of the DQuatTransExprData base class.
   using This           = DQuatTransposer<MT>;          //!< Type of this DQuatTransposer instance.
   using ResultType     = TransposeType_t<MT>;          //!< Result type for expression template evaluations.
   using OppositeType   = OppositeType_t<MT>;           //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType  = ResultType_t<MT>;             //!< Transpose type for expression template evaluations.
   using ElementType    = ElementType_t<MT>;            //!< Type of the tensor elements.
   using SIMDType       = SIMDTrait_t<ElementType>;     //!< SIMD type of the tensor elements.
   using ReturnType     = ReturnType_t<MT>;             //!< Return type for expression template evaluations.
   using CompositeType  = const This&;                  //!< Data type for composite expression templates.
   using Reference      = Reference_t<MT>;              //!< Reference to a non-constant tensor value.
   using ConstReference = ConstReference_t<MT>;         //!< Reference to a constant tensor value.
   using Pointer        = Pointer_t<MT>;                //!< Pointer to a non-constant tensor value.
   using ConstPointer   = ConstPointer_t<MT>;           //!< Pointer to a constant tensor value.
   using Iterator       = Iterator_t<MT>;               //!< Iterator over non-constant elements.
   using ConstIterator  = ConstIterator_t<MT>;          //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SIMD optimization.
   /*! The \a simdEnabled compilation flag indicates whether expressions the tensor is involved
       in can be optimized via SIMD operations. In case the dense tensor operand is vectorizable,
       the \a simdEnabled compilation flag is set to \a true, otherwise it is set to \a false. */
   static constexpr bool simdEnabled = MT::simdEnabled;

   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the tensor can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   static constexpr bool smpAssignable = MT::smpAssignable;
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   using DataType::idces;
   using DataType::quat;
   using DataType::page;
   using DataType::row;
   using DataType::column;
   using DataType::reverse_quat;
   using DataType::reverse_page;
   using DataType::reverse_row;
   using DataType::reverse_column;
   //@}
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DQuatTransposer class.
   //
   // \param dm The dense tensor operand.
   */
   template< typename... RTAs >
   explicit inline DQuatTransposer( MT& dm, RTAs... args ) noexcept
      : DataType( args... )   // Base class initialization
      , dm_( dm )             // Dense tensor of the transposition expression
   {}
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 2D-access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return Reference to the accessed value.
   */
   inline Reference operator()( size_t l, size_t k, size_t i, size_t j ) {
      BLAZE_INTERNAL_ASSERT( l < quats()  , "Invalid quats access index"    );
      BLAZE_INTERNAL_ASSERT( k < pages()  , "Invalid pages access index"    );
      BLAZE_INTERNAL_ASSERT( i < columns(), "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < rows()   , "Invalid column access index" );
      return dm_( reverse_quat(l, k, i, j), reverse_page(l, k, i, j), reverse_row(l, k, i, j), reverse_column(l, k, i, j) );
   }
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 2D-access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return Reference to the accessed value.
   */
   inline ConstReference operator()( size_t l, size_t k, size_t i, size_t j ) const {
      BLAZE_INTERNAL_ASSERT( l < quats()  , "Invalid quats access index"    );
      BLAZE_INTERNAL_ASSERT( k < pages()  , "Invalid pages access index"    );
      BLAZE_INTERNAL_ASSERT( i < rows()   , "Invalid row access index"      );
      BLAZE_INTERNAL_ASSERT( j < columns(), "Invalid column access index"   );
      return const_cast<const MT&>( dm_ )( reverse_quat(l, k, i, j), reverse_page(l, k, i, j), reverse_row(l, k, i, j), reverse_column(l, k, i, j) );
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid tensor access index.
   */
   inline Reference at( size_t l, size_t k, size_t i, size_t j ) {
      if( l >= quats() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid quat access index" );
      }
      if( k >= pages() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
      }
      if( i >= rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      return (*this)(l,k,i,j);
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid tensor access index.
   */
   inline ConstReference at( size_t l, size_t k, size_t i, size_t j ) const {
      if( l >= quats() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid quat access index" );
      }
      if( k >= pages() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
      }
      if( i >= dm_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= dm_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      return (*this)(l,k,i,j);
   }
   //**********************************************************************************************

   //**Low-level data access***********************************************************************
   /*!\brief Low-level data access to the tensor elements.
   //
   // \return Pointer to the internal element storage.
   */
   inline Pointer data() noexcept {
      return dm_.data();
   }
   //**********************************************************************************************

   //**Low-level data access***********************************************************************
   /*!\brief Low-level data access to the tensor elements.
   //
   // \return Pointer to the internal element storage.
   */
   inline ConstPointer data() const noexcept {
      return dm_.data();
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator to the first non-zero element of row/column \a i.
   //
   // This function returns a row/column iterator to the first non-zero element of row/column \a i.
   // In case the storage order is set to \a rowMajor the function returns an iterator to the first
   // non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
   // returns an iterator to the first non-zero element of column \a i.
   */
   inline Iterator begin( size_t i, size_t l, size_t k ) {
      return dm_.begin( reverse_row(l, k, i, 0), reverse_quat(l, k, i, 0), reverse_page(l, k, i, 0) );
   }
   //**********************************************************************************************

   //**Begin function******************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator to the first non-zero element of row/column \a i.
   //
   // This function returns a row/column iterator to the first non-zero element of row/column \a i.
   // In case the storage order is set to \a rowMajor the function returns an iterator to the first
   // non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
   // returns an iterator to the first non-zero element of column \a i.
   */
   inline ConstIterator begin( size_t i, size_t l, size_t k ) const {
      return dm_.cbegin( reverse_row(l, k, i, 0), reverse_quat(l, k, i, 0), reverse_page(l, k, i, 0) );
   }
   //**********************************************************************************************

   //**Cbegin function*****************************************************************************
   /*!\brief Returns an iterator to the first non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator to the first non-zero element of row/column \a i.
   //
   // This function returns a row/column iterator to the first non-zero element of row/column \a i.
   // In case the storage order is set to \a rowMajor the function returns an iterator to the first
   // non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
   // returns an iterator to the first non-zero element of column \a i.
   */
   inline ConstIterator cbegin( size_t i, size_t l, size_t k ) const {
      return dm_.cbegin( reverse_row(l, k, i, 0), reverse_quat(l, k, i, 0), reverse_page(l, k, i, 0) );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator just past the last non-zero element of row/column \a i.
   //
   // This function returns an row/column iterator just past the last non-zero element of row/column
   // \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
   // past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
   // the function returns an iterator just past the last non-zero element of column \a i.
   */
   inline Iterator end( size_t i, size_t l, size_t k ) {
      return dm_.end( reverse_row(l, k, i, 0), reverse_quat(l, k, i, 0), reverse_page(l, k, i, 0) );
   }
   //**********************************************************************************************

   //**End function********************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator just past the last non-zero element of row/column \a i.
   //
   // This function returns an row/column iterator just past the last non-zero element of row/column
   // \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
   // past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
   // the function returns an iterator just past the last non-zero element of column \a i.
   */
   inline ConstIterator end( size_t i, size_t l, size_t k ) const {
      return dm_.cend( reverse_row(l, k, i, 0), reverse_quat(l, k, i, 0), reverse_page(l, k, i, 0) );
   }
   //**********************************************************************************************

   //**Cend function*******************************************************************************
   /*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
   //
   // \param i The row/column index.
   // \return Iterator just past the last non-zero element of row/column \a i.
   //
   // This function returns an row/column iterator just past the last non-zero element of row/column
   // \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
   // past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
   // the function returns an iterator just past the last non-zero element of column \a i.
   */
   inline ConstIterator cend( size_t i, size_t l, size_t k ) const {
      return dm_.cend( reverse_row(l, k, i, 0), reverse_quat(l, k, i, 0), reverse_page(l, k, i, 0) );
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the tensor.
   //
   // \return The number of rows of the tensor.
   */
   inline size_t rows() const noexcept {
      return row( dm_.quats(), dm_.pages(), dm_.rows(), dm_.columns() );
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the tensor.
   //
   // \return The number of columns of the tensor.
   */
   inline size_t columns() const noexcept {
      return column( dm_.quats(), dm_.pages(), dm_.rows(), dm_.columns() );
   }
   //**********************************************************************************************

   //**Pages function****************************************************************************
   /*!\brief Returns the current number of pages of the tensor.
   //
   // \return The number of pages of the tensor.
   */
   inline size_t pages() const noexcept {
      return page( dm_.quats(), dm_.pages(), dm_.rows(), dm_.columns() );
   }
   //**********************************************************************************************

   //**Quats function****************************************************************************
   /*!\brief Returns the current number of quats of the quaternion.
   //
   // \return The number of quats of the tensor.
   */
   inline size_t quats() const noexcept {
      return quat( dm_.quats(), dm_.pages(), dm_.rows(), dm_.columns() );
   }
   //**********************************************************************************************

   //**Spacing function****************************************************************************
   /*!\brief Returns the spacing between the beginning of two rows.
   //
   // \return The spacing between the beginning of two rows.
   */
   inline size_t spacing() const noexcept {
      return dm_.spacing();
   }
   //**********************************************************************************************

   //**Reset function******************************************************************************
   /*!\brief Resets the tensor elements.
   //
   // \return void
   */
   inline void reset() {
      return dm_.reset();
   }
   //**********************************************************************************************

   //**IsIntact function***************************************************************************
   /*!\brief Returns whether the invariants of the tensor are intact.
   //
   // \return \a true in case the tensor's invariants are intact, \a false otherwise.
   */
   inline bool isIntact() const noexcept {
      using blaze::isIntact;
      return isIntact( dm_ );
   }
   //**********************************************************************************************

   //**CanAliased function*************************************************************************
   /*!\brief Returns whether the tensor can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the alias corresponds to this tensor, \a false if not.
   */
   template< typename Other >  // Data type of the foreign expression
   inline bool canAlias( const Other* alias ) const noexcept
   {
      return dm_.canAlias( alias );
   }
   //**********************************************************************************************

   //**IsAliased function**************************************************************************
   /*!\brief Returns whether the tensor is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the alias corresponds to this tensor, \a false if not.
   */
   template< typename Other >  // Data type of the foreign expression
   inline bool isAliased( const Other* alias ) const noexcept
   {
      return dm_.isAliased( alias );
   }
   //**********************************************************************************************

   //**IsAligned function**************************************************************************
   /*!\brief Returns whether the tensor is properly aligned in memory.
   //
   // \return \a true in case the tensor is aligned, \a false if not.
   */
   inline bool isAligned() const noexcept
   {
      return dm_.isAligned();
   }
   //**********************************************************************************************

   //**CanSMPAssign function***********************************************************************
   /*!\brief Returns whether the tensor can be used in SMP assignments.
   //
   // \return \a true in case the tensor can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept
   {
      return dm_.canSMPAssign();
   }
   //**********************************************************************************************

   //**Load function*******************************************************************************
   /*!\brief Load of a SIMD element of the tensor.
   //
   // \param i Access index for the row. The index has to be in the range [0..M-1].
   // \param j Access index for the column. The index has to be in the range [0..N-1].
   // \return The loaded SIMD element.
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors.
   */
   BLAZE_ALWAYS_INLINE SIMDType load( size_t l, size_t k, size_t i, size_t j ) const noexcept
   {
      return dm_.load( reverse_quat(l, k, i, j), reverse_page(l, k, i, j), reverse_row(l, k, i, j), reverse_column(l, k, i, j) );
   }
   //**********************************************************************************************

   //**Loada function******************************************************************************
   /*!\brief Aligned load of a SIMD element of the tensor.
   //
   // \param i Access index for the row. The index has to be in the range [0..M-1].
   // \param j Access index for the column. The index has to be in the range [0..N-1].
   // \return The loaded SIMD element.
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors.
   */
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t l, size_t k, size_t i, size_t j ) const noexcept
   {
      return dm_.loada( reverse_quat(l, k, i, j), reverse_page(l, k, i, j), reverse_row(l, k, i, j), reverse_column(l, k, i, j) );
   }
   //**********************************************************************************************

   //**Loadu function******************************************************************************
   /*!\brief Unaligned load of a SIMD element of the tensor.
   //
   // \param i Access index for the row. The index has to be in the range [0..M-1].
   // \param j Access index for the column. The index has to be in the range [0..N-1].
   // \return The loaded SIMD element.
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors.
   */
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t l, size_t k, size_t i, size_t j ) const noexcept
   {
      return dm_.loadu( reverse_quat(l, k, i, j), reverse_page(l, k, i, j), reverse_row(l, k, i, j), reverse_column(l, k, i, j) );
   }
   //**********************************************************************************************

   //**Store function******************************************************************************
   /*!\brief Store of a SIMD element of the tensor.
   //
   // \param i Access index for the row. The index has to be in the range [0..M-1].
   // \param j Access index for the column. The index has to be in the range [0..N-1].
   // \param value The SIMD element to be stored.
   // \return void
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors.
   */
   BLAZE_ALWAYS_INLINE void store( size_t l, size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
   {
      dm_.store( reverse_quat(l, k, i, j), reverse_page(l, k, i, j), reverse_row(l, k, i, j), reverse_column(l, k, i, j), value );
   }
   //**********************************************************************************************

   //**Storea function******************************************************************************
   /*!\brief Aligned store of a SIMD element of the tensor.
   //
   // \param i Access index for the row. The index has to be in the range [0..M-1].
   // \param j Access index for the column. The index has to be in the range [0..N-1].
   // \param value The SIMD element to be stored.
   // \return void
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors.
   */
   BLAZE_ALWAYS_INLINE void storea( size_t l, size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
   {
      dm_.storea( reverse_quat(l, k, i, j), reverse_page(l, k, i, j), reverse_row(l, k, i, j), reverse_column(l, k, i, j), value );
   }
   //**********************************************************************************************

   //**Storeu function*****************************************************************************
   /*!\brief Unaligned store of a SIMD element of the tensor.
   //
   // \param i Access index for the row. The index has to be in the range [0..M-1].
   // \param j Access index for the column. The index has to be in the range [0..N-1].
   // \param value The SIMD element to be stored.
   // \return void
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors.
   */
   BLAZE_ALWAYS_INLINE void storeu( size_t l, size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
   {
      dm_.storeu( reverse_quat(l, k, i, j), reverse_page(l, k, i, j), reverse_row(l, k, i, j), reverse_column(l, k, i, j), value );
   }
   //**********************************************************************************************

   //**Stream function*****************************************************************************
   /*!\brief Aligned, non-temporal store of a SIMD element of the tensor.
   //
   // \param i Access index for the row. The index has to be in the range [0..M-1].
   // \param j Access index for the column. The index has to be in the range [0..N-1].
   // \param value The SIMD element to be stored.
   // \return void
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors.
   */
   BLAZE_ALWAYS_INLINE void stream( size_t l, size_t k, size_t i, size_t j, const SIMDType& value ) noexcept
   {
      dm_.stream( reverse_quat(l, k, i, j), reverse_page(l, k, i, j), reverse_row(l, k, i, j), reverse_column(l, k, i, j), value );
   }
   //**********************************************************************************************

   //**Transpose assignment of matrices************************************************************
   /*!\brief Implementation of the transpose assignment of a tensor.
   //
   // \param rhs The right-hand side tensor to be assigned.
   // \return void
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors. Instead of using this function use the
   // assignment operator.
   */
   template< typename MT2         // Type of the right-hand side tensor
            , typename... RTAs >  // Runtime arguments
   inline void assign( const Array<MT2>& rhs, RTAs... args )
   {
      dm_.assign( trans( ~rhs, args... ) );
   }
   //**********************************************************************************************

   //**Transpose addition assignment of matrices***************************************************
   /*!\brief Implementation of the transpose addition assignment of a tensor.
   //
   // \param rhs The right-hand side tensor to be added.
   // \return void
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors. Instead of using this function use the
   // assignment operator.
   */
   template< typename MT2         // Type of the right-hand side tensor
            , typename... RTAs >  // Runtime arguments
   inline void addAssign( const Array<MT2>& rhs, RTAs... args )
   {
      dm_.addAssign( trans( ~rhs, args... ) );
   }
   //**********************************************************************************************

   //**Transpose subtraction assignment of matrices************************************************
   /*!\brief Implementation of the transpose subtraction assignment of a tensor.
   //
   // \param rhs The right-hand side tensor to be subtracted.
   // \return void
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors. Instead of using this function use the
   // assignment operator.
   */
   template< typename MT2         // Type of the right-hand side tensor
            , typename... RTAs >  // Runtime arguments
   inline void subAssign( const Array<MT2>& rhs, RTAs... args )
   {
      dm_.subAssign( trans( ~rhs, args... ) );
   }
   //**********************************************************************************************

   //**Transpose Schur product assignment of matrices**********************************************
   /*!\brief Implementation of the transpose Schur product assignment of a tensor.
   //
   // \param rhs The right-hand side tensor for the Schur product.
   // \return void
   //
   // This function must \b NOT be called explicitly! It is used internally for the performance
   // optimized evaluation of expression templates. Calling this function explicitly might result
   // in erroneous results and/or in compilation errors. Instead of using this function use the
   // assignment operator.
   */
   template< typename MT2         // Type of the right-hand side tensor
            , typename... RTAs >  // Runtime arguments
   inline void schurAssign( const Array<MT2>& rhs, RTAs... args )
   {
      dm_.schurAssign( trans( ~rhs, args... ) );
   }
   //**********************************************************************************************

   //**Transpose multiplication assignment of matrices*********************************************
   // No special implementation for the transpose multiplication assignment of matrices.
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   MT& dm_;  //!< The dense tensor operand.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_ARRAY_TYPE   ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Resetting the dense tensor contained in a DQuatTransposer.
// \ingroup dense_tensor_expression
//
// \param m The dense tensor to be resetted.
// \return void
*/
template< size_t ... CTAs    // Compile time arguments
        , typename MT >      // Type of the dense tensor
inline void reset( DQuatTransposer<MT, CTAs...>& m )
{
   m.reset();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the given DQuatTransposer are intact.
// \ingroup dense_tensor_expression
//
// \param m The dense tensor to be tested.
// \return \a true in caes the given tensor's invariants are intact, \a false otherwise.
*/
template< size_t ... CTAs    // Compile time arguments
        , typename MT >      // Type of the dense tensor
inline bool isIntact( const DQuatTransposer<MT, CTAs...>& m ) noexcept
{
   return m.isIntact();
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SIZE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, size_t ... CTAs >
struct Size< DQuatTransposer<MT, CTAs...>, 0UL >
   : public Size<MT,0UL>
{};

template< typename MT, size_t ... CTAs >
struct Size< DQuatTransposer<MT, CTAs...>, 1UL >
   : public Size<MT,1UL>
{};

template< typename MT, size_t ... CTAs >
struct Size< DQuatTransposer<MT, CTAs...>, 2UL >
   : public Size<MT,2UL>
{};

template< typename MT, size_t ... CTAs >
struct Size< DQuatTransposer<MT, CTAs...>, 3UL >
   : public Size<MT,3UL>
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
template< typename MT, size_t ... CTAs >
struct MaxSize< DQuatTransposer<MT, CTAs...>, 0UL >
   : public MaxSize<MT,0UL>
{};

template< typename MT, size_t ... CTAs >
struct MaxSize< DQuatTransposer<MT, CTAs...>, 1UL >
   : public MaxSize<MT,1UL>
{};

template< typename MT, size_t ... CTAs >
struct MaxSize< DQuatTransposer<MT, CTAs...>, 2UL >
   : public MaxSize<MT,2UL>
{};

template< typename MT, size_t ... CTAs >
struct MaxSize< DQuatTransposer<MT, CTAs...>, 3UL >
   : public MaxSize<MT,3UL>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HASCONSTDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, size_t ... CTAs >
struct HasConstDataAccess< DQuatTransposer<MT, CTAs...> >
   : public HasConstDataAccess<MT>
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
template< typename MT, size_t ... CTAs >
struct HasMutableDataAccess< DQuatTransposer<MT, CTAs...> >
   : public HasMutableDataAccess<MT>
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
template< typename MT, size_t ... CTAs >
struct IsAligned< DQuatTransposer<MT, CTAs...> >
   : public IsAligned<MT>
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
template< typename MT, size_t ... CTAs >
struct IsPadded< DQuatTransposer<MT, CTAs...> >
   : public IsPadded<MT>
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
