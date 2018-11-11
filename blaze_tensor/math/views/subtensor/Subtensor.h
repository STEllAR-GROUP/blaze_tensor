//=================================================================================================
/*!
//  \file blaze_tensor/math/views/subtensor/Subtensor.h
//  \brief Subtensor documentation
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_SUBTENSOR_SUBTENSOR_H_
#define _BLAZE_TENSOR_MATH_VIEWS_SUBTENSOR_SUBTENSOR_H_


//=================================================================================================
//
//  DOXYGEN DOCUMENTATION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup subtensor Subtensor
// \ingroup views
//
// Subtensors provide views on a specific part of a dense or sparse tensors just as subvectors
// provide views on specific parts of vectors. As such, subtensors act as a reference to a
// specific block within a tensor. This reference is valid and can be used in every way any
// other dense or sparse tensor can be used as long as the tensor containing the subtensor is
// not resized or entirely destroyed. The subtensor also acts as an alias to the tensor elements
// in the specified block: Changes made to the elements (e.g. modifying values, inserting or
// erasing elements) are immediately visible in the tensor and changes made via the tensor are
// immediately visible in the subtensor.
//
//
// \n \section subtensor_setup Setup of Subtensors
//
// A view on a dense or sparse subtensor can be created very conveniently via the \c subtensor()
// function. It can be included via the header file

   \code
   #include <blaze_tensor/math/Subtensor.h>
   \endcode

// The first and second parameter specify the row and column of the first element of the subtensor.
// The third and fourth parameter specify the number of rows and columns, respectively. The four
// parameters can be specified either at compile time or at runtime:

   \code
   blaze::DynamicTensor<double> A;
   // ... Resizing and initialization

   // Creating a dense subtensor of size 4x812, starting in row 3, column 0, and page 2 (compile time arguments)
   auto sm1 = subtensor<3UL,0UL,1UL,4UL,8UL,2UL>( A );

   // Creating a dense subtensor of size 8x16x1, starting in row 0, column 4, and page 3 (runtime arguments)
   auto sm2 = subtensor( A, 0UL, 4UL, 3UL, 8UL, 16UL, 1UL );
   \endcode

// The \c subtensor() function returns an expression representing the subtensor view. The type of
// this expression depends on the given subtensor arguments, primarily the type of the tensor and
// the compile time arguments. If the type is required, it can be determined via \c decltype
// specifier:

   \code
   using TensorType = blaze::DynamicTensor<int>;
   using SubtensorType = decltype( blaze::subtensor<3UL,0UL,1UL,4UL,8UL,2UL>( std::declval<TensorType>() ) );
   \endcode

// The resulting view can be treated as any other dense or sparse tensor, i.e. it can be assigned
// to, it can be copied from, and it can be used in arithmetic operations. A subtensor created from
// a row-major tensor will itself be a row-major tensor, a subtensor created from a column-major
// tensor will be a column-major tensor. The view can also be used on both sides of an assignment:
// The subtensor can either be used as an alias to grant write access to a specific subtensor
// of a tensor primitive on the left-hand side of an assignment or to grant read-access to
// a specific subtensor of a tensor primitive or expression on the right-hand side of an
// assignment. The following example demonstrates this in detail:

   \code
   blaze::DynamicTensor<double,blaze::columnMajor> A, B;
   blaze::CompressedMatrix<double,blaze::rowMajor> C;
   // ... Resizing and initialization

   // Creating a dense subtensor of size 8x4, starting in row 0 and column 2
   auto sm = subtensor( A, 0UL, 2UL, 8UL, 4UL );

   // Setting the subtensor of A to a 8x4 subtensor of B
   sm = subtensor( B, 0UL, 0UL, 8UL, 4UL );

   // Copying the sparse tensor C into another 8x4 subtensor of A
   subtensor( A, 8UL, 2UL, 8UL, 4UL ) = C;

   // Assigning part of the result of a tensor addition to the first subtensor
   sm = subtensor( B + C, 0UL, 0UL, 8UL, 4UL );
   \endcode

// \n \section subtensor_element_access Element access
//
// The elements of a subtensor can be directly accessed with the function call operator:

   \code
   blaze::DynamicTensor<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   // Creating a 8x8 subtensor, starting from position (4,4)
   auto sm = subtensor( A, 4UL, 4UL, 8UL, 8UL );

   // Setting the element (0,0) of the subtensor, which corresponds to
   // the element at position (4,4) in tensor A
   sm(0,0) = 2.0;
   \endcode

// Alternatively, the elements of a subtensor can be traversed via (const) iterators. Just as
// with matrices, in case of non-const subtensors, \c begin() and \c end() return an iterator,
// which allows to manipulate the elements, in case of constant subtensors an iterator to
// immutable elements is returned:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A( 256UL, 512UL );
   // ... Resizing and initialization

   // Creating a reference to a specific subtensor of tensor A
   auto sm = subtensor( A, 16UL, 16UL, 64UL, 128UL );

   // Traversing the elements of the 0th row via iterators to non-const elements
   for( auto it=sm.begin(0); it!=sm.end(0); ++it ) {
      *it = ...;  // OK: Write access to the dense subtensor value.
      ... = *it;  // OK: Read access to the dense subtensor value.
   }

   // Traversing the elements of the 1st row via iterators to const elements
   for( auto it=sm.cbegin(1); it!=sm.cend(1); ++it ) {
      *it = ...;  // Compilation error: Assignment to the value via iterator-to-const is invalid.
      ... = *it;  // OK: Read access to the dense subtensor value.
   }
   \endcode

   \code
   blaze::CompressedMatrix<int,blaze::rowMajor> A( 256UL, 512UL );
   // ... Resizing and initialization

   // Creating a reference to a specific subtensor of tensor A
   auto sm = subtensor( A, 16UL, 16UL, 64UL, 128UL );

   // Traversing the elements of the 0th row via iterators to non-const elements
   for( auto it=sm.begin(0); it!=sm.end(0); ++it ) {
      it->value() = ...;  // OK: Write access to the value of the non-zero element.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }

   // Traversing the elements of the 1st row via iterators to const elements
   for( auto it=sm.cbegin(1); it!=sm.cend(1); ++it ) {
      it->value() = ...;  // Compilation error: Assignment to the value via iterator-to-const is invalid.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }
   \endcode

// \n \section subtensor_element_insertion Element Insertion
//
// Inserting/accessing elements in a sparse subtensor can be done by several alternative functions.
// The following example demonstrates all options:

   \code
   blaze::CompressedMatrix<double,blaze::rowMajor> A( 256UL, 512UL );  // Non-initialized tensor of size 256x512

   auto sm = subtensor( A, 10UL, 10UL, 16UL, 16UL );  // View on a 16x16 subtensor of A

   // The function call operator provides access to all possible elements of the sparse subtensor,
   // including the zero elements. In case the function call operator is used to access an element
   // that is currently not stored in the sparse subtensor, the element is inserted into the
   // subtensor.
   sm(2,4) = 2.0;

   // The second operation for inserting elements is the set() function. In case the element is
   // not contained in the subtensor it is inserted into the subtensor, if it is already contained
   // in the subtensor its value is modified.
   sm.set( 2UL, 5UL, -1.2 );

   // An alternative for inserting elements into the subtensor is the \c insert() function. However,
   // it inserts the element only in case the element is not already contained in the subtensor.
   sm.insert( 2UL, 6UL, 3.7 );

   // Just as in the case of sparse matrices, elements can also be inserted via the \c append()
   // function. In case of subtensors, \c append() also requires that the appended element's
   // index is strictly larger than the currently largest non-zero index in the according row
   // or column of the subtensor and that the according row's or column's capacity is large enough
   // to hold the new element. Note however that due to the nature of a subtensor, which may be an
   // alias to the middle of a sparse tensor, the \c append() function does not work as efficiently
   // for a subtensor as it does for a tensor.
   sm.reserve( 2UL, 10UL );
   sm.append( 2UL, 10UL, -2.1 );
   \endcode

// \n \section subtensor_common_operations Common Operations
//
// A subtensor view can be used like any other dense or sparse tensor. For instance, the current
// size of the tensor, i.e. the number of rows or columns can be obtained via the \c rows() and
// \c columns() functions, the current total capacity via the \c capacity() function, and the
// number of non-zero elements via the \c nonZeros() function. However, since subtensors are
// views on a specific subtensor of a tensor, several operations are not possible, such as
// resizing and swapping:

   \code
   blaze::DynamicTensor<int,blaze::rowMajor> A( 42UL, 42UL );
   // ... Resizing and initialization

   // Creating a view on the a 8x12 subtensor of tensor A
   auto sm = subtensor( A, 0UL, 0UL, 8UL, 12UL );

   sm.rows();      // Returns the number of rows of the subtensor
   sm.columns();   // Returns the number of columns of the subtensor
   sm.capacity();  // Returns the capacity of the subtensor
   sm.nonZeros();  // Returns the number of non-zero elements contained in the subtensor

   sm.resize( 10UL, 8UL );  // Compilation error: Cannot resize a subtensor of a tensor

   auto sm2 = subtensor( A, 8UL, 0UL, 12UL, 8UL );
   swap( sm, sm2 );  // Compilation error: Swap operation not allowed
   \endcode

// \n \section subtensor_arithmetic_operations Arithmetic Operations
//
// Both dense and sparse subtensors can be used in all arithmetic operations that any other dense
// or sparse tensor can be used in. The following example gives an impression of the use of dense
// subtensors within arithmetic operations. All operations (addition, subtraction, multiplication,
// scaling, ...) can be performed on all possible combinations of dense and sparse matrices with
// fitting element types:

   \code
   blaze::DynamicTensor<double,blaze::rowMajor> D1, D2, D3;
   blaze::CompressedMatrix<double,blaze::rowMajor> S1, S2;

   blaze::CompressedVector<double,blaze::columnVector> a, b;

   // ... Resizing and initialization

   auto sm = subtensor( D1, 0UL, 0UL, 8UL, 8UL );  // View on the 8x8 subtensor of tensor D1
                                                   // starting from row 0 and column 0

   subtensor( D1, 0UL, 8UL, 8UL, 8UL ) = D2;  // Dense tensor initialization of the 8x8 subtensor
                                              // starting in row 0 and column 8
   sm = S1;                                   // Sparse tensor initialization of the second 8x8 subtensor

   D3 = sm + D2;                                   // Dense tensor/dense tensor addition
   S2 = S1 - subtensor( D1, 8UL, 0UL, 8UL, 8UL );  // Sparse tensor/dense tensor subtraction
   D2 = sm * subtensor( D1, 8UL, 8UL, 8UL, 8UL );  // Dense tensor/dense tensor multiplication

   subtensor( D1, 8UL, 0UL, 8UL, 8UL ) *= 2.0;      // In-place scaling of a subtensor of D1
   D2 = subtensor( D1, 8UL, 8UL, 8UL, 8UL ) * 2.0;  // Scaling of the a subtensor of D1
   D2 = 2.0 * sm;                                   // Scaling of the a subtensor of D1

   subtensor( D1, 0UL, 8UL, 8UL, 8UL ) += D2;  // Addition assignment
   subtensor( D1, 8UL, 0UL, 8UL, 8UL ) -= S1;  // Subtraction assignment
   subtensor( D1, 8UL, 8UL, 8UL, 8UL ) *= sm;  // Multiplication assignment

   a = subtensor( D1, 4UL, 4UL, 8UL, 8UL ) * b;  // Dense tensor/sparse vector multiplication
   \endcode

// \n \section subtensor_aligned_subtensor Aligned Subtensors
//
// Usually subtensors can be defined anywhere within a tensor. They may start at any position and
// may have an arbitrary extension (only restricted by the extension of the underlying tensor).
// However, in contrast to matrices themselves, which are always properly aligned in memory and
// therefore can provide maximum performance, this means that subtensors in general have to be
// considered to be unaligned. This can be made explicit by the blaze::unaligned flag:

   \code
   using blaze::unaligned;

   blaze::DynamicTensor<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   // Identical creations of an unaligned subtensor of size 8x8, starting in row 0 and column 0
   auto sm1 = subtensor           ( A, 0UL, 0UL, 8UL, 8UL );
   auto sm2 = subtensor<unaligned>( A, 0UL, 0UL, 8UL, 8UL );
   auto sm3 = subtensor<0UL,0UL,8UL,8UL>          ( A );
   auto sm4 = subtensor<unaligned,0UL,0UL,8UL,8UL>( A );
   \endcode

// All of these calls to the \c subtensor() function are identical. Whether the alignment flag is
// explicitly specified or not, it always returns an unaligned subtensor. Whereas this may provide
// full flexibility in the creation of subtensors, this might result in performance disadvantages
// in comparison to tensor primitives (even in case the specified subtensor could be aligned).
// Whereas tensor primitives are guaranteed to be properly aligned and therefore provide maximum
// performance in all operations, a general view on a tensor might not be properly aligned. This
// may cause a performance penalty on some platforms and/or for some operations.
//
// However, it is also possible to create aligned subtensors. Aligned subtensors are identical to
// unaligned subtensors in all aspects, except that they may pose additional alignment restrictions
// and therefore have less flexibility during creation, but don't suffer from performance penalties
// and provide the same performance as the underlying tensor. Aligned subtensors are created by
// explicitly specifying the blaze::aligned flag:

   \code
   using blaze::aligned;

   // Creating an aligned subtensor of size 8x8, starting in row 0 and column 0
   auto sv1 = subtensor<aligned>( A, 0UL, 0UL, 8UL, 8UL );
   auto sv2 = subtensor<aligned,0UL,0UL,8UL,8UL>( A );
   \endcode

// The alignment restrictions refer to system dependent address restrictions for the used element
// type and the available vectorization mode (SSE, AVX, ...). In order to be properly aligned the
// first element of each row/column of the subtensor must be aligned. The following source code
// gives some examples for a double precision row-major dynamic tensor, assuming that padding is
// enabled and that AVX is available, which packs 4 \c double values into a SIMD vector:

   \code
   using blaze::aligned;

   blaze::DynamicTensor<double,blaze::rowMajor> D( 13UL, 17UL );
   // ... Resizing and initialization

   // OK: Starts at position (0,0), i.e. the first element of each row is aligned (due to padding)
   auto dsm1 = subtensor<aligned>( D, 0UL, 0UL, 7UL, 11UL );

   // OK: First column is a multiple of 4, i.e. the first element of each row is aligned (due to padding)
   auto dsm2 = subtensor<aligned>( D, 3UL, 12UL, 8UL, 16UL );

   // OK: First column is a multiple of 4 and the subtensor includes the last row and column
   auto dsm3 = subtensor<aligned>( D, 4UL, 0UL, 9UL, 17UL );

   // Error: First column is not a multiple of 4, i.e. the first element is not aligned
   auto dsm4 = subtensor<aligned>( D, 2UL, 3UL, 12UL, 12UL );
   \endcode

// Note that the discussed alignment restrictions are only valid for aligned dense subtensors.
// In contrast, aligned sparse subtensors at this time don't pose any additional restrictions.
// Therefore aligned and unaligned sparse subtensors are truly fully identical. Still, in case
// the blaze::aligned flag is specified during setup, an aligned subtensor is created:

   \code
   using blaze::aligned;

   blaze::CompressedMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   // Creating an aligned subtensor of size 8x8, starting in row 0 and column 0
   auto sv = subtensor<aligned>( A, 0UL, 0UL, 8UL, 8UL );
   \endcode

// \n \section subtensor_on_symmetric_matrices Subtensor on Symmetric Matrices
//
// Subtensors can also be created on symmetric matrices (see the \c SymmetricMatrix class template):

   \code
   using blaze::DynamicTensor;
   using blaze::SymmetricMatrix;

   // Setup of a 16x16 symmetric tensor
   SymmetricMatrix< DynamicTensor<int> > A( 16UL );

   // Creating a dense subtensor of size 8x12, starting in row 2 and column 4
   auto sm = subtensor( A, 2UL, 4UL, 8UL, 12UL );
   \endcode

// It is important to note, however, that (compound) assignments to such subtensors have a
// special restriction: The symmetry of the underlying symmetric tensor must not be broken!
// Since the modification of element \f$ a_{ij} \f$ of a symmetric tensor also modifies the
// element \f$ a_{ji} \f$, the tensor to be assigned must be structured such that the symmetry
// of the symmetric tensor is preserved. Otherwise a \a std::invalid_argument exception is
// thrown:

   \code
   using blaze::DynamicTensor;
   using blaze::SymmetricMatrix;

   // Setup of two default 4x4 symmetric matrices
   SymmetricMatrix< DynamicTensor<int> > A1( 4 ), A2( 4 );

   // Setup of the 3x2 dynamic tensor
   //
   //       ( 1 2 )
   //   B = ( 3 4 )
   //       ( 5 6 )
   //
   DynamicTensor<int> B{ { 1, 2 }, { 3, 4 }, { 5, 6 } };

   // OK: Assigning B to a subtensor of A1 such that the symmetry can be preserved
   //
   //        ( 0 0 1 2 )
   //   A1 = ( 0 0 3 4 )
   //        ( 1 3 5 6 )
   //        ( 2 4 6 0 )
   //
   subtensor( A1, 0UL, 2UL, 3UL, 2UL ) = B;  // OK

   // Error: Assigning B to a subtensor of A2 such that the symmetry cannot be preserved!
   //   The elements marked with X cannot be assigned unambiguously!
   //
   //        ( 0 1 2 0 )
   //   A2 = ( 1 3 X 0 )
   //        ( 2 X 6 0 )
   //        ( 0 0 0 0 )
   //
   subtensor( A2, 0UL, 1UL, 3UL, 2UL ) = B;  // Assignment throws an exception!
   \endcode
*/
//*************************************************************************************************

} // namespace blaze

#endif
