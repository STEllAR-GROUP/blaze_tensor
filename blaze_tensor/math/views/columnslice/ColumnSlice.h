//=================================================================================================
/*!
//  \file blaze_tensor/math/views/columnslice/ColumnSlice.h
//  \brief ColumnSlice documentation
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_COLUMNSLICE_COLUMNSLICE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_COLUMNSLICE_COLUMNSLICE_H_


//=================================================================================================
//
//  DOXYGEN DOCUMENTATION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup columnslice ColumnSlice
// \ingroup views
//
// ColumnSlices provide views on a specific columnslice of a dense or sparse tensor. As such, columnslices act as a
// reference to a specific columnslice. This reference is valid and can be used in every way any other
// columnslice matrix can be used as long as the tensor containing the columnslice is not resized or entirely
// destroyed. The columnslice also acts as an alias to the columnslice elements: Changes made to the elements
// (e.g. modifying values, inserting or erasing elements) are immediately visible in the tensor
// and changes made via the tensor are immediately visible in the columnslice.
//
//
// \n \section columnslice_setup Setup of ColumnSlices
//
// \image html columnslice.png
// \image latex columnslice.eps "ColumnSlice view" width=250pt
//
// A reference to a dense or sparse columnslice can be created very conveniently via the \c columnslice() function.
// It can be included via the header file

   \code
   #include <blaze_tensor/math/ColumnSlice.h>
   \endcode

// The columnslice index must be in the range from \f$[0..M-1]\f$, where \c M is the total number of columnslices
// of the tensor, and can be specified both at compile time or at runtime:

   \code
   blaze::DynamicTensor<double> A;
   // ... Resizing and initialization

   // Creating a reference to the 1st columnslice of tensor A (compile time index)
   auto columnslice1 = columnslice<1UL>( A );

   // Creating a reference to the 2nd columnslice of tensor A (runtime index)
   auto columnslice2 = columnslice( A, 2UL );
   \endcode

// The \c columnslice() function returns an expression representing the columnslice view. The type of this
// expression depends on the given columnslice arguments, primarily the type of the tensor and the compile
// time arguments. If the type is required, it can be determined via \c decltype specifier:

   \code
   using TensorType = blaze::DynamicTensor<int>;
   using ColumnSliceType = decltype( blaze::columnslice<1UL>( std::declval<TensorType>() ) );
   \endcode

// The resulting view can be treated as any other columnslice matrix, i.e. it can be assigned to, it can
// be copied from, and it can be used in arithmetic operations. The reference can also be used on
// both sides of an assignment: The columnslice can either be used as an alias to grant write access to a
// specific columnslice of a tensor primitive on the left-hand side of an assignment or to grant read-access
// to a specific columnslice of a tensor primitive or expression on the right-hand side of an assignment.
// The following example demonstrates this in detail:

   \code
   blaze::DynamicMatrix<double> x;
   blaze::DynamicTensor<double> A, B;
   // ... Resizing and initialization

   // Setting the 2nd columnslice of tensor A to x
   auto columnslice2 = columnslice( A, 2UL );
   columnslice2 = x;

   // Setting the 3rd columnslice of tensor B to x
   columnslice( B, 3UL ) = x;

   // Setting x to the 4th columnslice of the result of the tensor multiplication
   x = columnslice( A * B, 4UL );
   \endcode

// \n \section columnslice_element_access Element access
//
// The elements of a columnslice can be directly accessed with the subscript operator:

   \code
   blaze::DynamicTensor<double> A;
   // ... Resizing and initialization

   // Creating a view on the 4th columnslice of tensor A
   auto columnslice4 = columnslice( A, 4UL );

   // Setting the 1st element of the dense columnslice, which corresponds
   // to the 1st element in the 4th columnslice of tensor A
   columnslice4(0, 0) = 2.0;
   \endcode

// The numbering of the columnslice elements is

                             \f[\left(\begin{array}{*{5}{c}}
                             0 & 1 & 2 & \cdots & N-1 \\
                             \end{array}\right),\f]

// where N is the number of columns of the referenced tensor. Alternatively, the elements of a
// columnslice can be traversed via iterators. Just as with vectors, in case of non-const columnslices, \c begin()
// and \c end() return an iterator, which allows to manipulate the elements, in case of constant
// columnslices an iterator to immutable elements is returned:

   \code
   blaze::DynamicTensor<int> A( 128UL, 256UL );
   // ... Resizing and initialization

   // Creating a reference to the 31st columnslice of tensor A
   auto columnslice31 = columnslice( A, 31UL );

   // Traversing the elements via iterators to non-const elements
   for( auto it=columnslice31.begin(); it!=columnslice31.end(); ++it ) {
      *it = ...;  // OK; Write access to the dense columnslice value
      ... = *it;  // OK: Read access to the dense columnslice value.
   }

   // Traversing the elements via iterators to const elements
   for( auto it=columnslice31.cbegin(); it!=columnslice31.cend(); ++it ) {
      *it = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = *it;  // OK: Read access to the dense columnslice value.
   }
   \endcode

   \code
   blaze::CompressedMatrix<int> A( 128UL, 256UL );
   // ... Resizing and initialization

   // Creating a reference to the 31st columnslice of tensor A
   auto columnslice31 = columnslice( A, 31UL );

   // Traversing the elements via iterators to non-const elements
   for( auto it=columnslice31.begin(); it!=columnslice31.end(); ++it ) {
      it->value() = ...;  // OK: Write access to the value of the non-zero element.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }

   // Traversing the elements via iterators to const elements
   for( auto it=columnslice31.cbegin(); it!=columnslice31.cend(); ++it ) {
      it->value() = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }
   \endcode

// \n \section sparse_columnslice_element_insertion Element Insertion
//
// Inserting/accessing elements in a sparse columnslice can be done by several alternative functions.
// The following example demonstrates all options:

   \code
   blaze::CompressedMatrix<double> A( 10UL, 100UL );  // Non-initialized 10x100 tensor

   auto columnslice0( columnslice( A, 0UL ) );  // Reference to the 0th columnslice of A

   // The subscript operator provides access to all possible elements of the sparse columnslice,
   // including the zero elements. In case the subscript operator is used to access an element
   // that is currently not stored in the sparse columnslice, the element is inserted into the columnslice.
   columnslice0[42] = 2.0;

   // The second operation for inserting elements is the set() function. In case the element
   // is not contained in the columnslice it is inserted into the columnslice, if it is already contained in
   // the columnslice its value is modified.
   columnslice0.set( 45UL, -1.2 );

   // An alternative for inserting elements into the columnslice is the insert() function. However,
   // it inserts the element only in case the element is not already contained in the columnslice.
   columnslice0.insert( 50UL, 3.7 );

   // A very efficient way to add new elements to a sparse columnslice is the append() function.
   // Note that append() requires that the appended element's index is strictly larger than
   // the currently largest non-zero index of the columnslice and that the columnslice's capacity is large
   // enough to hold the new element.
   columnslice0.reserve( 10UL );
   columnslice0.append( 51UL, -2.1 );
   \endcode

// \n \section columnslice_common_operations Common Operations
//
// A columnslice view can be used like any other columnslice vector. For instance, the current number of columnslice
// elements can be obtained via the \c size() function, the current capacity via the \c capacity()
// function, and the number of non-zero elements via the \c nonZeros() function. However, since
// columnslices are references to specific columnslices of a tensor, several operations are not possible, such as
// resizing and swapping. The following example shows this by means of a dense columnslice view:

   \code
   blaze::DynamicTensor<int> A( 42UL, 42UL );
   // ... Resizing and initialization

   // Creating a reference to the 2nd columnslice of tensor A
   auto columnslice2 = columnslice( A, 2UL );

   columnslice2.size();          // Returns the number of elements in the columnslice
   columnslice2.capacity();      // Returns the capacity of the columnslice
   columnslice2.nonZeros();      // Returns the number of non-zero elements contained in the columnslice

   columnslice2.resize( 84UL );  // Compilation error: Cannot resize a single columnslice of a tensor

   auto columnslice3 = columnslice( A, 3UL );
   swap( columnslice2, columnslice3 );   // Compilation error: Swap operation not allowed
   \endcode

// \n \section columnslice_arithmetic_operations Arithmetic Operations
//
// Both dense and sparse columnslices can be used in all arithmetic operations that any other dense or
// sparse columnslice vector can be used in. The following example gives an impression of the use of
// dense columnslices within arithmetic operations. All operations (addition, subtraction, multiplication,
// scaling, ...) can be performed on all possible combinations of dense and sparse columnslices with
// fitting element types:

   \code
   blaze::DynamicVector<double> a( 2UL, 2.0 ), b;
   blaze::CompressedVector<double> c( 2UL );
   c[1] = 3.0;

   blaze::DynamicTensor<double> A( 4UL, 2UL );  // Non-initialized 4x2 tensor

   auto columnslice0( columnslice( A, 0UL ) );  // Reference to the 0th columnslice of A

   columnslice0[0] = 0.0;        // Manual initialization of the 0th columnslice of A
   columnslice0[1] = 0.0;
   columnslice( A, 1UL ) = 1.0;  // Homogeneous initialization of the 1st columnslice of A
   columnslice( A, 2UL ) = a;    // Dense vector initialization of the 2nd columnslice of A
   columnslice( A, 3UL ) = c;    // Sparse vector initialization of the 3rd columnslice of A

   b = columnslice0 + a;              // Dense vector/dense vector addition
   b = c + columnslice( A, 1UL );     // Sparse vector/dense vector addition
   b = columnslice0 * columnslice( A, 2UL );  // Component-wise vector multiplication

   columnslice( A, 1UL ) *= 2.0;     // In-place scaling of the 1st columnslice
   b = columnslice( A, 1UL ) * 2.0;  // Scaling of the 1st columnslice
   b = 2.0 * columnslice( A, 1UL );  // Scaling of the 1st columnslice

   columnslice( A, 2UL ) += a;              // Addition assignment
   columnslice( A, 2UL ) -= c;              // Subtraction assignment
   columnslice( A, 2UL ) *= columnslice( A, 0UL );  // Multiplication assignment

   double scalar = columnslice( A, 1UL ) * trans( c );  // Scalar/dot/inner product between two vectors

   A = trans( c ) * columnslice( A, 1UL );  // Outer product between two vectors
   \endcode

// \n \section columnslice_on_column_major_tensor ColumnSlices on Column-Major Matrices
//
// Especially noteworthy is that columnslice views can be created for both columnslice-major and column-major
// matrices. Whereas the interface of a columnslice-major tensor only allows to traverse a columnslice directly
// and the interface of a column-major tensor only allows to traverse a column, via views it is
// possible to traverse a columnslice of a column-major tensor or a column of a columnslice-major tensor. For
// instance:

   \code
   blaze::DynamicTensor<int> A( 64UL, 32UL );
   // ... Resizing and initialization

   // Creating a reference to the 1st columnslice of a column-major tensor A
   auto columnslice1 = columnslice( A, 1UL );

   for( auto it=columnslice1.begin(); it!=columnslice1.end(); ++it ) {
      // ...
   }
   \endcode

// However, please note that creating a columnslice view on a tensor stored in a column-major fashion
// can result in a considerable performance decrease in comparison to a columnslice view on a tensor
// with columnslice-major storage format. This is due to the non-contiguous storage of the tensor
// elements. Therefore care has to be taken in the choice of the most suitable storage order:

   \code
   // Setup of two column-major matrices
   blaze::DynamicTensor<double> A( 128UL, 128UL );
   blaze::DynamicTensor<double> B( 128UL, 128UL );
   // ... Resizing and initialization

   // The computation of the 15th columnslice of the multiplication between A and B ...
   blaze::DynamicVector<double> x = columnslice( A * B, 15UL );

   // ... is essentially the same as the following computation, which multiplies
   // the 15th columnslice of the column-major tensor A with B.
   blaze::DynamicVector<double> x = columnslice( A, 15UL ) * B;
   \endcode

// Although Blaze performs the resulting vector/tensor multiplication as efficiently as possible
// using a columnslice-major storage order for tensor \c A would result in a more efficient evaluation.
*/
//*************************************************************************************************

#endif
