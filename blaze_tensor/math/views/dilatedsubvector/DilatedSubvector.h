//=================================================================================================
/*!
//  \file blaze/math/views/dilatedsubvector/DilatedSubvector.h
//  \brief DilatedSubvector documentation
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018-2019 Hartmut Kaiser - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBVECTOR_DILATEDSUBVECTOR_H_
#define _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBVECTOR_DILATEDSUBVECTOR_H_


//=================================================================================================
//
//  DOXYGEN DOCUMENTATION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup dilatedsubvector DilatedSubvector
// \ingroup views
//
// DilatedSubvectors provide views on a specific part of a dense or sparse vector. As such, dilatedsubvectors
// act as a reference to a specific range within a vector. This reference is valid and can be
// used in every way any other dense or sparse vector can be used as long as the vector containing
// the dilatedsubvector is not resized or entirely destroyed. The dilatedsubvector also acts as an alias to the
// vector elements in the specified range: Changes made to the elements (e.g. modifying values,
// inserting or erasing elements) are immediately visible in the vector and changes made via the
// vector are immediately visible in the dilatedsubvector.
//
//
// \n \section dilatedsubvector_setup Setup of DilatedSubvectors
//
// A view on a dense or sparse dilatedsubvector can be created very conveniently via the \c dilatedsubvector()
// function. It can be included via the header file

   \code
   #include <blaze/math/DilatedSubvector.h>
   \endcode

// The first parameter specifies the offset of the dilatedsubvector within the underlying dense or sparse
// vector, the second parameter specifies the size of the dilatedsubvector. The two parameters can be
// specified either at compile time or at runtime:

   \code
   blaze::DynamicVector<double,blaze::rowVector> x;
   // ... Resizing and initialization

   // Create a dilatedsubvector from index 4 with a size of 12 (i.e. in the range [4..15]) and with step-size 2 (compile time arguments)
   auto sv1 = dilatedsubvector<4UL,12UL,2UL>( x );

   // Create a dilatedsubvector from index 8 with a size of 16 (i.e. in the range [8..23]) and with step-size 3 (runtime arguments)
   auto sv2 = dilatedsubvector( x, 8UL, 16UL, 3UL );
   \endcode

// The \c dilatedsubvector() function returns an expression representing the dilatedsubvector view. The type of
// this expression depends on the given dilatedsubvector arguments, primarily the type of the vector and
// the compile time arguments. If the type is required, it can be determined via \c decltype
// specifier:

   \code
   using VectorType = blaze::DynamicVector<int>;
   using DilatedSubvectorType = decltype( blaze::dilatedsubvector<4UL,12UL,2UL>( std::declval<VectorType>() ) );
   \endcode

// The resulting view can be treated as any other dense or sparse vector, i.e. it can be assigned
// to, it can be copied from, and it can be used in arithmetic operations. A dilatedsubvector created
// from a row vector can be used as any other row vector, a dilatedsubvector created from a column vector
// can be used as any other column vector. The view can also be used on both sides of an assignment:
// The dilatedsubvector can either be used as an alias to grant write access to a specific dilatedsubvector of a
// vector primitive on the left-hand side of an assignment or to grant read-access to a specific
// dilatedsubvector of a vector primitive or expression on the right-hand side of an assignment. The
// following example demonstrates this in detail:

   \code
   blaze::DynamicVector<double,blaze::rowVector> x;
   blaze::CompressedVector<double,blaze::rowVector> y;
   blaze::DynamicMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   // Create a dilatedsubvector from index 0 with a size of 10 (i.e. in the range [0..9]) and with step-size 3
   auto sv = dilatedsubvector( x, 0UL, 10UL, 3UL );

   // Setting the first ten elements of x to the 2nd row of matrix A
   sv = row( A, 2UL );

   // Setting every second element of the second twenty elements of x to y
   dilatedsubvector( x, 10UL, 10UL, 2UL ) = y;

   // Setting the 3rd row of A to a dilatedsubvector of x
   row( A, 3UL ) = dilatedsubvector( x, 3UL, 10UL, 4UL );

   // Setting x to a dilatedsubvector of the result of the addition between y and the 1st row of A
   x = dilatedsubvector( y + row( A, 1UL ), 2UL, 5UL, 2UL )
   \endcode

// \n \section dilatedsubvector_element_access Element access
//
// The elements of a dilatedsubvector can be directly accessed via the subscript operator:

   \code
   blaze::DynamicVector<double,blaze::rowVector> v;
   // ... Resizing and initialization

   // Creating an 8-dimensional dilatedsubvector, starting from index 4, step-size 3
   auto sv = dilatedsubvector( v, 4UL, 8UL, 3UL );

   // Setting the 1st element of the dilatedsubvector, which corresponds to
   // the element at index 5 in vector v
   sv[1] = 2.0;
   \endcode

// The numbering of the dilatedsubvector elements is

                             \f[\left(\begin{array}{*{5}{c}}
                             0 & 3 & 6 & \cdots & (N-1) * 3 \\
                             \end{array}\right),\f]

// where N is the specified size of the dilatedsubvector. Alternatively, the elements of a dilatedsubvector can
// be traversed via iterators. Just as with vectors, in case of non-const dilatedsubvectors, \c begin()
// and \c end() return an iterator, which allows to manipulate the elements, in case of constant
// dilatedsubvectors an iterator to immutable elements is returned:

   \code
   blaze::DynamicVector<int,blaze::rowVector> v( 256UL );
   // ... Resizing and initialization

   // Creating a reference to a specific dilatedsubvector of vector v
   auto sv = dilatedsubvector( v, 16UL, 64UL, 2UL );

   // Traversing the elements via iterators to non-const elements
   for( auto it=sv.begin(); it!=sv.end(); ++it ) {
      *it = ...;  // OK: Write access to the dense dilatedsubvector value.
      ... = *it;  // OK: Read access to the dense dilatedsubvector value.
   }

   // Traversing the elements via iterators to const elements
   for( auto it=sv.cbegin(); it!=sv.cend(); ++it ) {
      *it = ...;  // Compilation error: Assignment to the value via iterator-to-const is invalid.
      ... = *it;  // OK: Read access to the dense dilatedsubvector value.
   }
   \endcode

   \code
   blaze::CompressedVector<int,blaze::rowVector> v( 256UL );
   // ... Resizing and initialization

   // Creating a reference to a specific dilatedsubvector of vector v
   auto sv = dilatedsubvector( v, 16UL, 64UL, 3UL );

   // Traversing the elements via iterators to non-const elements
   for( auto it=sv.begin(); it!=sv.end(); ++it ) {
      it->value() = ...;  // OK: Write access to the value of the non-zero element.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }

   // Traversing the elements via iterators to const elements
   for( auto it=sv.cbegin(); it!=sv.cend(); ++it ) {
      it->value() = ...;  // Compilation error: Assignment to the value via iterator-to-const is invalid.
      ... = it->value();  // OK: Read access to the value of the non-zero element.
      it->index() = ...;  // Compilation error: The index of a non-zero element cannot be changed.
      ... = it->index();  // OK: Read access to the index of the sparse element.
   }
   \endcode

// \n \section dilatedsubvector_element_insertion Element Insertion
//
// Inserting/accessing elements in a sparse dilatedsubvector can be done by several alternative functions.
// The following example demonstrates all options:

   \code
   blaze::CompressedVector<double,blaze::rowVector> v( 256UL );  // Non-initialized vector of size 256

   auto sv = dilatedsubvector( v, 10UL, 60UL, 2UL );  // View on the range [10..129] of v with step-size 2

   // The subscript operator provides access to all possible elements of the sparse dilatedsubvector,
   // including the zero elements. In case the subscript operator is used to access an element
   // that is currently not stored in the sparse dilatedsubvector, the element is inserted into the
   // dilatedsubvector.
   sv[42] = 2.0;

   // The second operation for inserting elements is the set() function. In case the element is
   // not contained in the dilatedsubvector it is inserted into the dilatedsubvector, if it is already contained
   // in the dilatedsubvector its value is modified.
   sv.set( 45UL, -1.2 );

   // An alternative for inserting elements into the dilatedsubvector is the insert() function. However,
   // it inserts the element only in case the element is not already contained in the dilatedsubvector.
   sv.insert( 50UL, 3.7 );

   // Just as in case of vectors, elements can also be inserted via the append() function. In
   // case of dilatedsubvectors, append() also requires that the appended element's index is strictly
   // larger than the currently largest non-zero index of the dilatedsubvector and that the dilatedsubvector's
   // capacity is large enough to hold the new element. Note however that due to the nature of
   // a dilatedsubvector, which may be an alias to the middle of a sparse vector, the append() function
   // does not work as efficiently for a dilatedsubvector as it does for a vector.
   sv.reserve( 10UL );
   sv.append( 51UL, -2.1 );
   \endcode

// \n \section dilatedsubvector_common_operations Common Operations
//
// A dilatedsubvector view can be used like any other dense or sparse vector. For instance, the current
// number of elements can be obtained via the \c size() function, the current capacity via the
// \c capacity() function, and the number of non-zero elements via the \c nonZeros() function.
// However, since dilatedsubvectors are references to a specific range of a vector, several operations
// are not possible, such as resizing and swapping. The following example shows this by means of
// a dense dilatedsubvector view:

   \code
   blaze::DynamicVector<int,blaze::rowVector> v( 42UL );
   // ... Resizing and initialization

   // Creating a view on the range [5..35] with step-size 2 of vector v
   auto sv = dilatedsubvector( v, 5UL, 10UL, 2UL );

   sv.size();          // Returns the number of elements in the dilatedsubvector
   sv.capacity();      // Returns the capacity of the dilatedsubvector
   sv.nonZeros();      // Returns the number of non-zero elements contained in the dilatedsubvector

   sv.resize( 84UL );  // Compilation error: Cannot resize a dilatedsubvector of a vector

   auto sv2 = dilatedsubvector( v, 15UL, 10UL, 2UL );
   swap( sv, sv2 );   // Compilation error: Swap operation not allowed
   \endcode

// \n \section dilatedsubvector_arithmetic_operations Arithmetic Operations
//
// Both dense and sparse dilatedsubvectors can be used in all arithmetic operations that any other dense
// or sparse vector can be used in. The following example gives an impression of the use of dense
// dilatedsubvectors within arithmetic operations. All operations (addition, subtraction, multiplication,
// scaling, ...) can be performed on all possible combinations of dense and sparse dilatedsubvectors with
// fitting element types:

   \code
   blaze::DynamicVector<double,blaze::rowVector> d1, d2, d3;
   blaze::CompressedVector<double,blaze::rowVector> s1, s2;

   // ... Resizing and initialization

   blaze::DynamicMatrix<double,blaze::rowMajor> A;

   auto sv( dilatedsubvector( d1, 0UL, 10UL, 2UL ) );  // View on the range [0..19] with step-size 2 of vector d1

   sv = d2;                                       // Dense vector initialization of the range [0..19]
   dilatedsubvector( d1, 10UL, 10UL, 2UL ) = s1;  // Sparse vector initialization of the range [10..29]

   d3 = sv + d2;                           // Dense vector/dense vector addition
   s2 = s1 + dilatedsubvector( d1, 10UL, 10UL, 2UL );  // Sparse vector/dense vector addition
   d2 = sv * dilatedsubvector( d1, 20UL, 10UL, 2UL );  // Component-wise vector multiplication

   dilatedsubvector( d1, 3UL, 4UL, 2UL ) *= 2.0; // In-place scaling of every second element in the range [3..10]
   d2 = dilatedsubvector( d1, 7UL, 3UL, 3UL ) * 2.0;  // Scaling of every third element in the range [7..15]
   d2 = 2.0 * dilatedsubvector( d1, 7UL, 3UL, 3UL );  // Scaling of every third element in the range [7..15]

   dilatedsubvector( d1, 0UL , 10UL, 2UL ) += d2;  // Addition assignment
   dilatedsubvector( d1, 10UL, 10UL, 2UL ) -= s2;  // Subtraction assignment
   dilatedsubvector( d1, 20UL, 10UL, 2UL ) *= sv;  // Multiplication assignment

   double scalar = dilatedsubvector( d1, 5UL, 10UL, 3UL ) * trans( s1 );  // Scalar/dot/inner product between two vectors

   A = trans( s1 ) * dilatedsubvector( d1, 4UL, 16UL, 4UL );  // Outer product between two vectors
   \endcode

// \n \section dilatedsubvector_aligned_dilatedsubvector Aligned DilatedSubvectors
//
// Usually dilatedsubvectors can be defined anywhere within a vector. They may start at any position and
// may have an arbitrary size (only restricted by the size of the underlying vector). However, in
// contrast to vectors themselves, which are always properly aligned in memory and therefore can
// provide maximum performance, this means that dilatedsubvectors in general have to be considered to be
// unaligned. This can be made explicit by the blaze::unaligned flag:

   \code
   using blaze::unaligned;

   blaze::DynamicVector<double,blaze::rowVector> x;
   // ... Resizing and initialization

   // Identical creations of an (unaligned) dilatedsubvector in the range [8..39]
   auto sv1 = dilatedsubvector( x, 8UL, 16UL, 2UL );
   auto sv2 = dilatedsubvector<8UL,16UL,2UL>( x );
   \endcode

// All of these calls to the \c dilatedsubvector() function are identical. It always returns an
// unaligned dilatedsubvector. Whereas this may provide
// full flexibility in the creation of dilatedsubvectors, this might result in performance disadvantages
// in comparison to vector primitives (even in case the specified dilatedsubvector could be aligned).
// Whereas vector primitives are guaranteed to be properly aligned and therefore provide maximum
// performance in all operations, a general view on a vector might not be properly aligned. This
// may cause a performance penalty on some platforms and/or for some operations.
*/
//*************************************************************************************************

#endif
