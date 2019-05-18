//=================================================================================================
/*!
//  \file blaze_tensor/math/views/dilatedsubmatrix/DilatedSubmatrixData.h
//  \brief Header file for the implementation of the DilatedSubmatrixData class template
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBMATRIX_DILATEDSUBMATRIXDATA_H_
#define _BLAZE_TENSOR_MATH_VIEWS_DILATEDSUBMATRIX_DILATEDSUBMATRIXDATA_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/MaybeUnused.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class template for the data members of the DilatedSubmatrix class.
// \ingroup DilatedSubmatrix
//
// The auxiliary DilatedSubmatrixData class template represents an abstraction of the data members of
// the DilatedSubmatrix class template. The necessary set of data members is selected depending on the
// number of compile time DilatedSubmatrix arguments.
*/
template< size_t... CSAs >  // Compile time DilatedSubmatrix arguments
struct DilatedSubmatrixData
{};
//*************************************************************************************************




//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ZERO COMPILE TIME ARGUMENTS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the DilatedSubmatrixData class template for zero compile time DilatedSubmatrix
//        arguments.
// \ingroup DilatedSubmatrix
//
// This specialization of DilatedSubmatrixData adapts the class template to the requirements of zero
// compile time DilatedSubmatrix arguments.
*/
template<>
struct DilatedSubmatrixData<>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RSAs >
   explicit inline DilatedSubmatrixData( size_t rindex, size_t cindex, size_t m, size_t n,
      size_t rowdilation, size_t columndilation, RSAs... args );

   DilatedSubmatrixData( const DilatedSubmatrixData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DilatedSubmatrixData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DilatedSubmatrixData& operator=( const DilatedSubmatrixData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t row    () const noexcept;
   inline size_t column () const noexcept;
   inline size_t rows   () const noexcept;
   inline size_t columns() const noexcept;
   inline size_t rowdilation   () const noexcept;
   inline size_t columndilation() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   const size_t row_;             //!< The first row of the DilatedSubmatrix.
   const size_t column_;          //!< The first column of the DilatedSubmatrix.
   const size_t m_;               //!< The number of rows of the DilatedSubmatrix.
   const size_t n_;               //!< The number of columns of the DilatedSubmatrix.
   const size_t rowdilation_;     //!< The row step-size of the dilatedsubmatrix
   const size_t columndilation_;  //!< The column step-size of the dilatedsubmatrix
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for DilatedSubmatrixData.
//
// \param rindex The index of the first row of the DilatedSubmatrix in the given matrix.
// \param cindex The index of the first column of the DilatedSubmatrix in the given matrix.
// \param m The number of rows of the DilatedSubmatrix.
// \param n The number of columns of the DilatedSubmatrix.
// \param args The optional DilatedSubmatrix arguments.
*/
template< typename... RSAs >          // Optional DilatedSubmatrix arguments
inline DilatedSubmatrixData<>::DilatedSubmatrixData( size_t rindex, size_t cindex, size_t m, size_t n,
   size_t rowdilation, size_t columndilation, RSAs... args )
   : row_   ( rindex )                 // The first row of the DilatedSubmatrix
   , column_( cindex )                 // The first column of the DilatedSubmatrix
   , m_     ( m      )                 // The number of rows of the DilatedSubmatrix
   , n_     ( n      )                 // The number of columns of the DilatedSubmatrix
   , rowdilation_   ( rowdilation )    // The row step-size of the dilatedsubmatrix
   , columndilation_( columndilation ) // The column step-size of the dilatedsubmatrix
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first row of the DilatedSubmatrix in the underlying matrix.
//
// \return The index of the first row.
*/
inline size_t DilatedSubmatrixData<>::row() const noexcept
{
   return row_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first column of the DilatedSubmatrix in the underlying matrix.
//
// \return The index of the first column.
*/
inline size_t DilatedSubmatrixData<>::column() const noexcept
{
   return column_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of rows of the DilatedSubmatrix.
//
// \return The number of rows of the DilatedSubmatrix.
*/
inline size_t DilatedSubmatrixData<>::rows() const noexcept
{
   return m_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of columns of the DilatedSubmatrix.
//
// \return The number of columns of the DilatedSubmatrix.
*/
inline size_t DilatedSubmatrixData<>::columns() const noexcept
{
   return n_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first row of the DilatedSubmatrix in the underlying matrix.
//
// \return The index of the first row.
*/
inline size_t DilatedSubmatrixData<>::rowdilation() const noexcept
{
   return rowdilation_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first column of the DilatedSubmatrix in the underlying matrix.
//
// \return The index of the first column.
*/
inline size_t DilatedSubmatrixData<>::columndilation() const noexcept
{
   return columndilation_;
}
/*! \endcond */
//*************************************************************************************************



//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR FOUR COMPILE TIME ARGUMENTS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the DilatedSubmatrixData class template for four compile time DilatedSubmatrix
//        arguments.
// \ingroup DilatedSubmatrix
//
// This specialization of DilatedSubmatrixData adapts the class template to the requirements of two
// compile time arguments.
*/
template< size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t RowDilation     // The row step-size of the dilatedsubmatrix
        , size_t ColumnDilation >// The column step-size of the dilatedsubmatrix
struct DilatedSubmatrixData<I,J,M,N,RowDilation,ColumnDilation>
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RSAs >
   explicit inline DilatedSubmatrixData( RSAs... args );

   DilatedSubmatrixData( const DilatedSubmatrixData& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~DilatedSubmatrixData() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   DilatedSubmatrixData& operator=( const DilatedSubmatrixData& ) = delete;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   static inline constexpr size_t row    () noexcept;
   static inline constexpr size_t column () noexcept;
   static inline constexpr size_t rows   () noexcept;
   static inline constexpr size_t columns() noexcept;
   static inline constexpr size_t rowdilation   () noexcept;
   static inline constexpr size_t columndilation() noexcept;
   //@}
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The constructor for DilatedSubmatrixData.
//
// \param args The optional DilatedSubmatrix arguments.
*/
template< size_t I                  // Index of the first row
        , size_t J                  // Index of the first column
        , size_t M                  // Number of rows
        , size_t N                  // Number of columns
        , size_t RowDilation        // The row step-size of the dilatedsubmatrix
        , size_t ColumnDilation >   // The column step-size of the dilatedsubmatrix
template< typename... RSAs >        // Optional DilatedSubmatrix arguments
inline DilatedSubmatrixData<I,J,M,N,RowDilation,ColumnDilation>::DilatedSubmatrixData( RSAs... args )
{
   MAYBE_UNUSED( args... );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first row of the DilatedSubmatrix in the underlying matrix.
//
// \return The index of the first row.
*/
template< size_t I                // Index of the first row
        , size_t J                // Index of the first column
        , size_t M                // Number of rows
        , size_t N                // Number of columns
        , size_t RowDilation      // The row step-size of the dilatedsubmatrix
        , size_t ColumnDilation > // The column step-size of the dilatedsubmatrix
inline constexpr size_t DilatedSubmatrixData<I,J,M,N,RowDilation,ColumnDilation>::row() noexcept
{
   return I;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first column of the DilatedSubmatrix in the underlying matrix.
//
// \return The index of the first column.
*/
template< size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t RowDilation     // The row step-size of the dilatedsubmatrix
        , size_t ColumnDilation >// The column step-size of the dilatedsubmatrix
inline constexpr size_t DilatedSubmatrixData<I,J,M,N,RowDilation,ColumnDilation>::column() noexcept
{
   return J;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of rows of the DilatedSubmatrix.
//
// \return The number of rows of the DilatedSubmatrix.
*/
template< size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t RowDilation     // The row step-size of the dilatedsubmatrix
        , size_t ColumnDilation >// The column step-size of the dilatedsubmatrix
inline constexpr size_t DilatedSubmatrixData<I,J,M,N,RowDilation,ColumnDilation>::rows() noexcept
{
   return M;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of columns of the DilatedSubmatrix.
//
// \return The number of columns of the DilatedSubmatrix.
*/
template< size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t RowDilation     // The row step-size of the dilatedsubmatrix
        , size_t ColumnDilation >// The column step-size of the dilatedsubmatrix
inline constexpr size_t DilatedSubmatrixData<I,J,M,N,RowDilation,ColumnDilation>::columns() noexcept
{
   return N;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first row of the DilatedSubmatrix in the underlying matrix.
//
// \return The index of the first row.
*/
template< size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t RowDilation     // The row step-size of the dilatedsubmatrix
        , size_t ColumnDilation >// The column step-size of the dilatedsubmatrix
inline constexpr size_t DilatedSubmatrixData<I,J,M,N,RowDilation,ColumnDilation>::rowdilation() noexcept
{
   return RowDilation;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the index of the first column of the DilatedSubmatrix in the underlying matrix.
//
// \return The index of the first column.
*/
template< size_t I               // Index of the first row
        , size_t J               // Index of the first column
        , size_t M               // Number of rows
        , size_t N               // Number of columns
        , size_t RowDilation     // The row step-size of the dilatedsubmatrix
        , size_t ColumnDilation >// The column step-size of the dilatedsubmatrix
inline constexpr size_t DilatedSubmatrixData<I,J,M,N,RowDilation,ColumnDilation>::columndilation() noexcept
{
   return ColumnDilation;
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
