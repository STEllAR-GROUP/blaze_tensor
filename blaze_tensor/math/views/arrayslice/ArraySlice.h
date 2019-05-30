//=================================================================================================
/*!
//  \file blaze_tensor/math/views/pageslice/PageSlice.h
//  \brief PageSlice documentation
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_PAGESLICE_PAGESLICE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_PAGESLICE_PAGESLICE_H_


//=================================================================================================
//
//  DOXYGEN DOCUMENTATION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup pageslice PageSlice
// \ingroup views
//
// PageSlices provide views on a specific pageslice of a dense or sparse tensor. As such, pageslices act as a
// reference to a specific pageslice. This reference is valid and can be used in every way any other
// pageslice matrix can be used as long as the tensor containing the pageslice is not resized or entirely
// destroyed. The pageslice also acts as an alias to the pageslice elements: Changes made to the elements
// (e.g. modifying values, inserting or erasing elements) are immediately visible in the tensor
// and changes made via the tensor are immediately visible in the pageslice.
//
//
// \n \section pageslice_setup Setup of PageSlices
//
// \image html pageslice.png
// \image latex pageslice.eps "PageSlice view" width=250pt
//
// A reference to a dense or sparse pageslice can be created very conveniently via the \c pageslice() function.
// It can be included via the header file

   \code
   #include <blaze_tensor/math/PageSlice.h>
   \endcode

// The pageslice index must be in the range from \f$[0..M-1]\f$, where \c M is the total number of pageslices
// of the tensor, and can be specified both at compile time or at runtime:

   \code
   blaze::DynamicTensor<double> A;
   // ... Resizing and initialization

   // Creating a reference to the 1st pageslice of tensor A (compile time index)
   auto pageslice1 = pageslice<1UL>( A );

   // Creating a reference to the 2nd pageslice of tensor A (runtime index)
   auto pageslice2 = pageslice( A, 2UL );
   \endcode

// The \c pageslice() function returns an expression representing the pageslice view. The type of this
// expression depends on the given pageslice arguments, primarily the type of the tensor and the compile
// time arguments. If the type is required, it can be determined via \c decltype specifier:

   \code
   using TensorType = blaze::DynamicTensor<int>;
   using PageSliceType = decltype( blaze::pageslice<1UL>( std::declval<TensorType>() ) );
   \endcode

// The resulting view can be treated as any other pageslice matrix, i.e. it can be assigned to, it can
// be copied from, and it can be used in arithmetic operations. The reference can also be used on
// both sides of an assignment: The pageslice can either be used as an alias to grant write access to a
// specific pageslice of a tensor primitive on the left-hand side of an assignment or to grant read-access
// to a specific pageslice of a tensor primitive or expression on the right-hand side of an assignment.
// The following example demonstrates this in detail:

   \code
   blaze::DynamicMatrix<double> x;
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization

   // Setting the 2nd pageslice of tensor A to x
   auto pageslice2 = pageslice( A, 2UL );
   pageslice2 = x;

   // Setting the 3rd pageslice of tensor B to x
   pageslice( B, 3UL ) = x;

   // Setting x to the 4th pageslice of the result of the tensor multiplication
   x = pageslice( A * B, 4UL );
   \endcode

// \n \section pageslice_element_access Element access
//
// The elements of a pageslice can be directly accessed with the subscript operator:

   \code
   blaze::DynamicTensor<double> A;
   // ... Resizing and initialization

   // Creating a view on the 4th pageslice of tensor A
   auto pageslice4 = pageslice( A, 4UL );

   // Setting the 1st element of the dense pageslice, which corresponds
   // to the 1st element in the 4th pageslice of tensor A
   pageslice4(0, 0) = 2.0;
   \endcode

// The numbering of the pageslice elements is

                             \f[\left(\begin{array}{*{5}{c}}
                             0 & 1 & 2 & \cdots & N-1 \\
                             \end{array}\right),\f]

// where N is the number of columns of the referenced tensor. Alternatively, the elements of a
// pageslice can be traversed via iterators. Just as with vectors, in case of non-const pageslices, \c begin()
// and \c end() return an iterator, which allows to manipulate the elements, in case of constant
// pageslices an iterator to immutable elements is returned:

   \code
   blaze::DynamicTensor<int> A( 128UL, 256UL );
   // ... Resizing and initialization

   // Creating a reference to the 31st pageslice of tensor A
   auto pageslice31 = pageslice( A, 31UL );

   // Traversing the elements via iterators to non-const elements
   for( auto it=pageslice31.begin(); it!=pageslice31.end(); ++it ) {
      *it = ...;  // OK; Write access to the dense pageslice value
      ... = *it;  // OK: Read access to the dense pageslice value.
   }

   // Traversing the elements via iterators to const elements
   for( auto it=pageslice31.cbegin(); it!=pageslice31.cend(); ++it ) {
      *it = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = *it;  // OK: Read access to the dense pageslice value.
   }
   \endcode

   \code
   blaze::CompressedMatrix<int> A( 128UL, 256UL );
   // ... Resizing and initialization

   // Creating a reference to the 31st pageslice of tensor A
   auto pageslice31 = pageslice( A, 31UL );

   // Traversing the elements via iterators to non-const elements
   for( auto it=pageslice31.begin(); it!=pageslice31.end(); ++it ) {
      it->value() = ...;  // OK: Write access to the value of the non-zero element.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }

   // Traversing the elements via iterators to const elements
   for( auto it=pageslice31.cbegin(); it!=pageslice31.cend(); ++it ) {
      it->value() = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }
   \endcode

// \n \section sparse_pageslice_element_insertion Element Insertion
//
// Inserting/accessing elements in a sparse pageslice can be done by several alternative functions.
// The following example demonstrates all options:

   \code
   blaze::CompressedMatrix<double> A( 10UL, 100UL );  // Non-initialized 10x100 tensor

   auto pageslice0( pageslice( A, 0UL ) );  // Reference to the 0th pageslice of A

   // The subscript operator provides access to all possible elements of the sparse pageslice,
   // including the zero elements. In case the subscript operator is used to access an element
   // that is currently not stored in the sparse pageslice, the element is inserted into the pageslice.
   pageslice0[42] = 2.0;

   // The second operation for inserting elements is the set() function. In case the element
   // is not contained in the pageslice it is inserted into the pageslice, if it is already contained in
   // the pageslice its value is modified.
   pageslice0.set( 45UL, -1.2 );

   // An alternative for inserting elements into the pageslice is the insert() function. However,
   // it inserts the element only in case the element is not already contained in the pageslice.
   pageslice0.insert( 50UL, 3.7 );

   // A very efficient way to add new elements to a sparse pageslice is the append() function.
   // Note that append() requires that the appended element's index is strictly larger than
   // the currently largest non-zero index of the pageslice and that the pageslice's capacity is large
   // enough to hold the new element.
   pageslice0.reserve( 10UL );
   pageslice0.append( 51UL, -2.1 );
   \endcode

// \n \section pageslice_common_operations Common Operations
//
// A pageslice view can be used like any other pageslice vector. For instance, the current number of pageslice
// elements can be obtained via the \c size() function, the current capacity via the \c capacity()
// function, and the number of non-zero elements via the \c nonZeros() function. However, since
// pageslices are references to specific pageslices of a tensor, several operations are not possible, such as
// resizing and swapping. The following example shows this by means of a dense pageslice view:

   \code
   blaze::DynamicTensor<int> A( 42UL, 42UL );
   // ... Resizing and initialization

   // Creating a reference to the 2nd pageslice of tensor A
   auto pageslice2 = pageslice( A, 2UL );

   pageslice2.size();          // Returns the number of elements in the pageslice
   pageslice2.capacity();      // Returns the capacity of the pageslice
   pageslice2.nonZeros();      // Returns the number of non-zero elements contained in the pageslice

   pageslice2.resize( 84UL );  // Compilation error: Cannot resize a single pageslice of a tensor

   auto pageslice3 = pageslice( A, 3UL );
   swap( pageslice2, pageslice3 );   // Compilation error: Swap operation not allowed
   \endcode

// \n \section pageslice_arithmetic_operations Arithmetic Operations
//
// Both dense and sparse pageslices can be used in all arithmetic operations that any other dense or
// sparse pageslice vector can be used in. The following example gives an impression of the use of
// dense pageslices within arithmetic operations. All operations (addition, subtraction, multiplication,
// scaling, ...) can be performed on all possible combinations of dense and sparse pageslices with
// fitting element types:

   \code
   blaze::DynamicVector<double> a( 2UL, 2.0 ), b;
   blaze::CompressedVector<double> c( 2UL );
   c[1] = 3.0;

   blaze::DynamicTensor<double> A( 4UL, 2UL );  // Non-initialized 4x2 tensor

   auto pageslice0( pageslice( A, 0UL ) );  // Reference to the 0th pageslice of A

   pageslice0[0] = 0.0;        // Manual initialization of the 0th pageslice of A
   pageslice0[1] = 0.0;
   pageslice( A, 1UL ) = 1.0;  // Homogeneous initialization of the 1st pageslice of A
   pageslice( A, 2UL ) = a;    // Dense vector initialization of the 2nd pageslice of A
   pageslice( A, 3UL ) = c;    // Sparse vector initialization of the 3rd pageslice of A

   b = pageslice0 + a;              // Dense vector/dense vector addition
   b = c + pageslice( A, 1UL );     // Sparse vector/dense vector addition
   b = pageslice0 * pageslice( A, 2UL );  // Component-wise vector multiplication

   pageslice( A, 1UL ) *= 2.0;     // In-place scaling of the 1st pageslice
   b = pageslice( A, 1UL ) * 2.0;  // Scaling of the 1st pageslice
   b = 2.0 * pageslice( A, 1UL );  // Scaling of the 1st pageslice

   pageslice( A, 2UL ) += a;              // Addition assignment
   pageslice( A, 2UL ) -= c;              // Subtraction assignment
   pageslice( A, 2UL ) *= pageslice( A, 0UL );  // Multiplication assignment

   double scalar = pageslice( A, 1UL ) * trans( c );  // Scalar/dot/inner product between two vectors

   A = trans( c ) * pageslice( A, 1UL );  // Outer product between two vectors
   \endcode

// \n \section pageslice_on_column_major_tensor PageSlices on Column-Major Matrices
//
// Especially noteworthy is that pageslice views can be created for both pageslice-major and column-major
// matrices. Whereas the interface of a pageslice-major tensor only allows to traverse a pageslice directly
// and the interface of a column-major tensor only allows to traverse a column, via views it is
// possible to traverse a pageslice of a column-major tensor or a column of a pageslice-major tensor. For
// instance:

   \code
   blaze::DynamicTensor<int> A( 64UL, 32UL );
   // ... Resizing and initialization

   // Creating a reference to the 1st pageslice of a column-major tensor A
   auto pageslice1 = pageslice( A, 1UL );

   for( auto it=pageslice1.begin(); it!=pageslice1.end(); ++it ) {
      // ...
   }
   \endcode

// However, please note that creating a pageslice view on a tensor stored in a column-major fashion
// can result in a considerable performance decrease in comparison to a pageslice view on a tensor
// with pageslice-major storage format. This is due to the non-contiguous storage of the tensor
// elements. Therefore care has to be taken in the choice of the most suitable storage order:

   \code
   // Setup of two column-major matrices
   blaze::DynamicTensor<double> A( 128UL, 128UL );
   blaze::DynamicTensor<double> B( 128UL, 128UL );
   // ... Resizing and initialization

   // The computation of the 15th pageslice of the multiplication between A and B ...
   blaze::DynamicVector<double> x = pageslice( A * B, 15UL );

   // ... is essentially the same as the following computation, which multiplies
   // the 15th pageslice of the column-major tensor A with B.
   blaze::DynamicVector<double> x = pageslice( A, 15UL ) * B;
   \endcode

// Although Blaze performs the resulting vector/tensor multiplication as efficiently as possible
// using a pageslice-major storage order for tensor \c A would result in a more efficient evaluation.
*/
//*************************************************************************************************

#endif
