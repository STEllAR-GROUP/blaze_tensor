//=================================================================================================
/*!
//  \file blaze_tensor/math/views/quatslice/QuatSlice.h
//  \brief QuatSlice documentation
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_QUATSLICE_QUATSLICE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_QUATSLICE_QUATSLICE_H_


//=================================================================================================
//
//  DOXYGEN DOCUMENTATION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup quatslice QuatSlice
// \ingroup views
//
// QuatSlices provide views on a specific quatslice of a dense or sparse tensor. As such, quatslices act as a
// reference to a specific quatslice. This reference is valid and can be used in every way any other
// quatslice matrix can be used as long as the tensor containing the quatslice is not resized or entirely
// destroyed. The quatslice also acts as an alias to the quatslice elements: Changes made to the elements
// (e.g. modifying values, inserting or erasing elements) are immediately visible in the tensor
// and changes made via the tensor are immediately visible in the quatslice.
//
//
// \n \section quatslice_setup Setup of QuatSlices
//
// \image html quatslice.png
// \image latex quatslice.eps "QuatSlice view" width=250pt
//
// A reference to a dense or sparse quatslice can be created very conveniently via the \c quatslice() function.
// It can be included via the header file

   \code
   #include <blaze_tensor/math/QuatSlice.h>
   \endcode

// The quatslice index must be in the range from \f$[0..M-1]\f$, where \c M is the total number of quatslices
// of the tensor, and can be specified both at compile time or at runtime:

   \code
   blaze::DynamicArray<double> A;
   // ... Resizing and initialization

   // Creating a reference to the 1st quatslice of tensor A (compile time index)
   auto quatslice1 = quatslice<1UL>( A );

   // Creating a reference to the 2nd quatslice of tensor A (runtime index)
   auto quatslice2 = quatslice( A, 2UL );
   \endcode

// The \c quatslice() function returns an expression representing the quatslice view. The type of this
// expression depends on the given quatslice arguments, primarily the type of the tensor and the compile
// time arguments. If the type is required, it can be determined via \c decltype specifier:

   \code
   using ArrayType = blaze::DynamicArray<int>;
   using QuatSliceType = decltype( blaze::quatslice<1UL>( std::declval<ArrayType>() ) );
   \endcode

// The resulting view can be treated as any other quatslice matrix, i.e. it can be assigned to, it can
// be copied from, and it can be used in arithmetic operations. The reference can also be used on
// both sides of an assignment: The quatslice can either be used as an alias to grant write access to a
// specific quatslice of a tensor primitive on the left-hand side of an assignment or to grant read-access
// to a specific quatslice of a tensor primitive or expression on the right-hand side of an assignment.
// The following example demonstrates this in detail:

   \code
   blaze::DynamicMatrix<double> x;
   blaze::DynamicArray<double> A, B;
   // ... Resizing and initialization

   // Setting the 2nd quatslice of tensor A to x
   auto quatslice2 = quatslice( A, 2UL );
   quatslice2 = x;

   // Setting the 3rd quatslice of tensor B to x
   quatslice( B, 3UL ) = x;

   // Setting x to the 4th quatslice of the result of the tensor multiplication
   x = quatslice( A * B, 4UL );
   \endcode

// \n \section quatslice_element_access Element access
//
// The elements of a quatslice can be directly accessed with the subscript operator:

   \code
   blaze::DynamicArray<double> A;
   // ... Resizing and initialization

   // Creating a view on the 4th quatslice of tensor A
   auto quatslice4 = quatslice( A, 4UL );

   // Setting the 1st element of the dense quatslice, which corresponds
   // to the 1st element in the 4th quatslice of tensor A
   quatslice4(0, 0) = 2.0;
   \endcode

// The numbering of the quatslice elements is

                             \f[\left(\begin{array}{*{5}{c}}
                             0 & 1 & 2 & \cdots & N-1 \\
                             \end{array}\right),\f]

// where N is the number of columns of the referenced tensor. Alternatively, the elements of a
// quatslice can be traversed via iterators. Just as with vectors, in case of non-const quatslices, \c begin()
// and \c end() return an iterator, which allows to manipulate the elements, in case of constant
// quatslices an iterator to immutable elements is returned:

   \code
   blaze::DynamicArray<int> A( 128UL, 256UL );
   // ... Resizing and initialization

   // Creating a reference to the 31st quatslice of tensor A
   auto quatslice31 = quatslice( A, 31UL );

   // Traversing the elements via iterators to non-const elements
   for( auto it=quatslice31.begin(); it!=quatslice31.end(); ++it ) {
      *it = ...;  // OK; Write access to the dense quatslice value
      ... = *it;  // OK: Read access to the dense quatslice value.
   }

   // Traversing the elements via iterators to const elements
   for( auto it=quatslice31.cbegin(); it!=quatslice31.cend(); ++it ) {
      *it = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = *it;  // OK: Read access to the dense quatslice value.
   }
   \endcode

   \code
   blaze::CompressedMatrix<int> A( 128UL, 256UL );
   // ... Resizing and initialization

   // Creating a reference to the 31st quatslice of tensor A
   auto quatslice31 = quatslice( A, 31UL );

   // Traversing the elements via iterators to non-const elements
   for( auto it=quatslice31.begin(); it!=quatslice31.end(); ++it ) {
      it->value() = ...;  // OK: Write access to the value of the non-zero element.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }

   // Traversing the elements via iterators to const elements
   for( auto it=quatslice31.cbegin(); it!=quatslice31.cend(); ++it ) {
      it->value() = ...;  // Compilation error: Assignment to the value via a ConstIterator is invalid.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }
   \endcode

// \n \section sparse_quatslice_element_insertion Element Insertion
//
// Inserting/accessing elements in a sparse quatslice can be done by several alternative functions.
// The following example demonstrates all options:

   \code
   blaze::CompressedMatrix<double> A( 10UL, 100UL );  // Non-initialized 10x100 tensor

   auto quatslice0( quatslice( A, 0UL ) );  // Reference to the 0th quatslice of A

   // The subscript operator provides access to all possible elements of the sparse quatslice,
   // including the zero elements. In case the subscript operator is used to access an element
   // that is currently not stored in the sparse quatslice, the element is inserted into the quatslice.
   quatslice0[42] = 2.0;

   // The second operation for inserting elements is the set() function. In case the element
   // is not contained in the quatslice it is inserted into the quatslice, if it is already contained in
   // the quatslice its value is modified.
   quatslice0.set( 45UL, -1.2 );

   // An alternative for inserting elements into the quatslice is the insert() function. However,
   // it inserts the element only in case the element is not already contained in the quatslice.
   quatslice0.insert( 50UL, 3.7 );

   // A very efficient way to add new elements to a sparse quatslice is the append() function.
   // Note that append() requires that the appended element's index is strictly larger than
   // the currently largest non-zero index of the quatslice and that the quatslice's capacity is large
   // enough to hold the new element.
   quatslice0.reserve( 10UL );
   quatslice0.append( 51UL, -2.1 );
   \endcode

// \n \section quatslice_common_operations Common Operations
//
// A quatslice view can be used like any other quatslice vector. For instance, the current number of quatslice
// elements can be obtained via the \c size() function, the current capacity via the \c capacity()
// function, and the number of non-zero elements via the \c nonZeros() function. However, since
// quatslices are references to specific quatslices of a tensor, several operations are not possible, such as
// resizing and swapping. The following example shows this by means of a dense quatslice view:

   \code
   blaze::DynamicArray<int> A( 42UL, 42UL );
   // ... Resizing and initialization

   // Creating a reference to the 2nd quatslice of tensor A
   auto quatslice2 = quatslice( A, 2UL );

   quatslice2.size();          // Returns the number of elements in the quatslice
   quatslice2.capacity();      // Returns the capacity of the quatslice
   quatslice2.nonZeros();      // Returns the number of non-zero elements contained in the quatslice

   quatslice2.resize( 84UL );  // Compilation error: Cannot resize a single quatslice of a tensor

   auto quatslice3 = quatslice( A, 3UL );
   swap( quatslice2, quatslice3 );   // Compilation error: Swap operation not allowed
   \endcode

// \n \section quatslice_arithmetic_operations Arithmetic Operations
//
// Both dense and sparse quatslices can be used in all arithmetic operations that any other dense or
// sparse quatslice vector can be used in. The following example gives an impression of the use of
// dense quatslices within arithmetic operations. All operations (addition, subtraction, multiplication,
// scaling, ...) can be performed on all possible combinations of dense and sparse quatslices with
// fitting element types:

   \code
   blaze::DynamicVector<double> a( 2UL, 2.0 ), b;
   blaze::CompressedVector<double> c( 2UL );
   c[1] = 3.0;

   blaze::DynamicArray<double> A( 4UL, 2UL );  // Non-initialized 4x2 tensor

   auto quatslice0( quatslice( A, 0UL ) );  // Reference to the 0th quatslice of A

   quatslice0[0] = 0.0;        // Manual initialization of the 0th quatslice of A
   quatslice0[1] = 0.0;
   quatslice( A, 1UL ) = 1.0;  // Homogeneous initialization of the 1st quatslice of A
   quatslice( A, 2UL ) = a;    // Dense vector initialization of the 2nd quatslice of A
   quatslice( A, 3UL ) = c;    // Sparse vector initialization of the 3rd quatslice of A

   b = quatslice0 + a;              // Dense vector/dense vector addition
   b = c + quatslice( A, 1UL );     // Sparse vector/dense vector addition
   b = quatslice0 * quatslice( A, 2UL );  // Component-wise vector multiplication

   quatslice( A, 1UL ) *= 2.0;     // In-place scaling of the 1st quatslice
   b = quatslice( A, 1UL ) * 2.0;  // Scaling of the 1st quatslice
   b = 2.0 * quatslice( A, 1UL );  // Scaling of the 1st quatslice

   quatslice( A, 2UL ) += a;              // Addition assignment
   quatslice( A, 2UL ) -= c;              // Subtraction assignment
   quatslice( A, 2UL ) *= quatslice( A, 0UL );  // Multiplication assignment

   double scalar = quatslice( A, 1UL ) * trans( c );  // Scalar/dot/inner product between two vectors

   A = trans( c ) * quatslice( A, 1UL );  // Outer product between two vectors
   \endcode

// \n \section quatslice_on_column_major_tensor QuatSlices on Column-Major Matrices
//
// Especially noteworthy is that quatslice views can be created for both quatslice-major and column-major
// matrices. Whereas the interface of a quatslice-major tensor only allows to traverse a quatslice directly
// and the interface of a column-major tensor only allows to traverse a column, via views it is
// possible to traverse a quatslice of a column-major tensor or a column of a quatslice-major tensor. For
// instance:

   \code
   blaze::DynamicArray<int> A( 64UL, 32UL );
   // ... Resizing and initialization

   // Creating a reference to the 1st quatslice of a column-major tensor A
   auto quatslice1 = quatslice( A, 1UL );

   for( auto it=quatslice1.begin(); it!=quatslice1.end(); ++it ) {
      // ...
   }
   \endcode

// However, please note that creating a quatslice view on a tensor stored in a column-major fashion
// can result in a considerable performance decrease in comparison to a quatslice view on a tensor
// with quatslice-major storage format. This is due to the non-contiguous storage of the tensor
// elements. Therefore care has to be taken in the choice of the most suitable storage order:

   \code
   // Setup of two column-major matrices
   blaze::DynamicArray<double> A( 128UL, 128UL );
   blaze::DynamicArray<double> B( 128UL, 128UL );
   // ... Resizing and initialization

   // The computation of the 15th quatslice of the multiplication between A and B ...
   blaze::DynamicVector<double> x = quatslice( A * B, 15UL );

   // ... is essentially the same as the following computation, which multiplies
   // the 15th quatslice of the column-major tensor A with B.
   blaze::DynamicVector<double> x = quatslice( A, 15UL ) * B;
   \endcode

// Although Blaze performs the resulting vector/tensor multiplication as efficiently as possible
// using a quatslice-major storage order for tensor \c A would result in a more efficient evaluation.
*/
//*************************************************************************************************

#endif
