//=================================================================================================
/*!
//  \file src/mathtest/dtensdtensadd/TUbMDb.cpp
//  \brief Source file for the TUbMDb dense tensor/dense tensor addition math test
//
//  Copyright (C) 2012-2018 Klaus Iglberger - All Rights Reserved
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


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cstdlib>
#include <iostream>
#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/UniformTensor.h>
#include <blazetest/mathtest/creator/DynamicTensor.h>
#include <blazetest/mathtest/creator/UniformTensor.h>
#include <blazetest/mathtest/dtensdtensadd/OperationTest.h>
#include <blazetest/system/MathTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


//=================================================================================================
//
//  MAIN FUNCTION
//
//=================================================================================================

//*************************************************************************************************
int main()
{
   std::cout << "   Running 'TUbMDb'..." << std::endl;

   using blazetest::mathtest::TypeB;

   try
   {
      // Tensor type definitions
      using TUb = blaze::UniformTensor<TypeB>;
      using MDb = blaze::DynamicTensor<TypeB>;

      // Creator type definitions
      using CTUb = blazetest::Creator<TUb>;
      using CMDb = blazetest::Creator<MDb>;

      // Running tests with small matrices
      for( size_t k=0UL; k<=5UL; ++k ) {
         for( size_t i=0UL; i<=5UL; ++i ) {
            for( size_t j=0UL; j<=5UL; ++j ) {
               RUN_DTENSDTENSADD_OPERATION_TEST( CTUb( k, i, j ), CMDb( k, i, j ) );
            }
         }
      }

      // Running tests with large matrices
      RUN_DTENSDTENSADD_OPERATION_TEST( CTUb( 3UL,  67UL,  67UL ), CMDb( 3UL,  67UL,  67UL ) );
      RUN_DTENSDTENSADD_OPERATION_TEST( CTUb( 3UL,  67UL, 127UL ), CMDb( 3UL,  67UL, 127UL ) );
      RUN_DTENSDTENSADD_OPERATION_TEST( CTUb( 8UL, 128UL,  64UL ), CMDb( 8UL, 128UL,  64UL ) );
      RUN_DTENSDTENSADD_OPERATION_TEST( CTUb( 8UL, 128UL, 128UL ), CMDb( 8UL, 128UL, 128UL ) );
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during dense tensor/dense tensor addition:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
