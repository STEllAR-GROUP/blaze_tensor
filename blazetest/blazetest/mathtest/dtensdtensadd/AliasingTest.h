//=================================================================================================
/*!
//  \file blazetest/mathtest/dtensdtensadd/AliasingTest.h
//  \brief Header file for the dense tensor/dense tensor addition aliasing test
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

#ifndef _BLAZETEST_MATHTEST_DMATDMATADD_ALIASINGTEST_H_
#define _BLAZETEST_MATHTEST_DMATDMATADD_ALIASINGTEST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <sstream>
#include <stdexcept>
#include <string>

#include <blaze_tensor/math/StaticTensor.h>
#include <blaze_tensor/math/DynamicTensor.h>

namespace blazetest {

namespace mathtest {

namespace dtensdtensadd {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class template for the dense tensor/dense tensor addition aliasing test.
//
// This class represents a test suite for all dense tensor/dense tensor addition aliasing
// tests. It performs a series of runtime tests to assure that all mathematical operations
// work correctly even in the presence of aliasing.
*/
class AliasingTest
{
 private:
   //**Type definitions****************************************************************************
   using DMat  = blaze::DynamicTensor<int>;         //!< Row-major dense tensor type.
//    using TDMat = blaze::DynamicTensor<int,blaze::columnMajor>;      //!< Column-major dense tensor type.
   using RMat  = blaze::StaticTensor<int,2UL,3UL,3UL>;  //!< Result row-major tensor type.
   //**********************************************************************************************

 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit AliasingTest();
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

 private:
   //**Test functions******************************************************************************
   /*!\name Test functions */
   //@{
   void testDTensDTensAdd ();
   void testDTensTDTensAdd();

   template< typename T1, typename T2 >
   void checkResult( const T1& computedResult, const T2& expectedResult );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   void initialize();
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   DMat dA2x3x4_;    //!< The first row-major dense tensor.
                   /*!< The \f$ 3 \times 4 \f$ tensor is initialized as
                        \f[\left(\begin{array}{*{4}{c}}
                        -1 & 0 & -2 & 0 \\
                         0 & 2 & -3 & 1 \\
                         0 & 1 &  2 & 2 \\
                        \end{array}\right)\f]. */
   DMat dB2x4x3_;    //!< The second row-major dense tensor.
                   /*!< The \f$ 4 \times 3 \f$ tensor is initialized as
                        \f[\left(\begin{array}{*{3}{c}}
                        1 &  0 & -3 \\
                        0 & -1 &  0 \\
                        0 &  2 &  1 \\
                        2 &  1 & -2 \\
                        \end{array}\right)\f]. */
   DMat dC2x3x3_;    //!< The third row-major dense tensor.
                   /*!< The \f$ 3 \times 3 \f$ tensor is initialized as
                        \f[\left(\begin{array}{*{3}{c}}
                         1 & 0 &  2 \\
                         0 & 3 & -1 \\
                        -1 & 0 &  2 \\
                        \end{array}\right)\f]. */
   DMat dD2x3x3_;    //!< The fourth row-major dense tensor.
                   /*!< The \f$ 3 \times 3 \f$ tensor is initialized as
                        \f[\left(\begin{array}{*{3}{c}}
                        0 & -1 &  0 \\
                        1 & -2 &  2 \\
                        0 &  0 & -3 \\
                        \end{array}\right)\f]. */
   DMat dE2x3x3_;    //!< The fifth row-major dense tensor.
                   /*!< The \f$ 3 \times 3 \f$ tensor is initialized as
                        \f[\left(\begin{array}{*{3}{c}}
                        2 & 0 &  0 \\
                        0 & 1 & -2 \\
                        1 & 0 &  0 \\
                        \end{array}\right)\f]. */
//    TDMat tdA3x4_;  //!< The first column-major dense tensor.
//                    /*!< The \f$ 3 \times 4 \f$ tensor is initialized as
//                         \f[\left(\begin{array}{*{4}{c}}
//                         -1 & 0 & -2 & 0 \\
//                          0 & 2 & -3 & 1 \\
//                          0 & 1 &  2 & 2 \\
//                         \end{array}\right)\f]. */
//    TDMat tdB4x3_;  //!< The second column-major dense tensor.
//                    /*!< The \f$ 4 \times 3 \f$ tensor is initialized as
//                         \f[\left(\begin{array}{*{3}{c}}
//                         1 &  0 & -3 \\
//                         0 & -1 &  0 \\
//                         0 &  2 &  1 \\
//                         2 &  1 & -2 \\
//                         \end{array}\right)\f]. */
//    TDMat tdC3x3_;  //!< The third column-major dense tensor.
//                    /*!< The \f$ 3 \times 3 \f$ tensor is initialized as
//                         \f[\left(\begin{array}{*{3}{c}}
//                          1 & 0 &  2 \\
//                          0 & 3 & -1 \\
//                         -1 & 0 &  2 \\
//                         \end{array}\right)\f]. */
//    TDMat tdD3x3_;  //!< The fourth column-major dense tensor.
//                    /*!< The \f$ 3 \times 3 \f$ tensor is initialized as
//                         \f[\left(\begin{array}{*{3}{c}}
//                         0 & -1 &  0 \\
//                         1 & -2 &  2 \\
//                         0 &  0 & -3 \\
//                         \end{array}\right)\f]. */
//    TDMat tdE3x3_;  //!< The fifth column-major dense tensor.
//                    /*!< The \f$ 3 \times 3 \f$ tensor is initialized as
//                         \f[\left(\begin{array}{*{3}{c}}
//                         2 & 0 &  0 \\
//                         0 & 1 & -2 \\
//                         1 & 0 &  0 \\
//                         \end{array}\right)\f]. */
   RMat result_;   //!< The dense tensor for the reference result.

   std::string test_;  //!< Label of the currently performed test.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Checking and comparing the computed result.
//
// \param computedResult The computed result.
// \param expectedResult The expected result.
// \return void
// \exception std::runtime_error Incorrect result detected.
//
// This function is called after each test case to check and compare the computed result.
// In case the computed and the expected result differ in any way, a \a std::runtime_error
// exception is thrown.
*/
template< typename T1    // Tensor type of the computed result
        , typename T2 >  // Tensor type of the expected result
void AliasingTest::checkResult( const T1& computedResult, const T2& expectedResult )
{
   if( computedResult != expectedResult ) {
      std::ostringstream oss;
      oss.precision( 20 );
      oss << " Test : " << test_ << "\n"
          << " Error: Incorrect result detected\n"
          << " Details:\n"
          << "   Computed result:\n" << computedResult << "\n"
          << "   Expected result:\n" << expectedResult << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Testing the dense tensor/dense tensor addition in the presence of aliasing.
//
// \return void
*/
void runTest()
{
   AliasingTest();
}
//*************************************************************************************************




//=================================================================================================
//
//  MACRO DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of the dense tensor/dense tensor addition aliasing test.
*/
#define RUN_DTENSDTENSADD_ALIASING_TEST \
   blazetest::mathtest::dtensdtensadd::runTest()
/*! \endcond */
//*************************************************************************************************

} // namespace dtensdtensadd

} // namespace mathtest

} // namespace blazetest

#endif
