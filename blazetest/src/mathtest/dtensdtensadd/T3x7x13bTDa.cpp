//=================================================================================================
/*!
//  \file src/mathtest/dtensdtensadd/T3x7x13bTDa.cpp
//  \brief Source file for the T3x7x13bTDa dense tensor/dense tensor addition math test
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


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cstdlib>
#include <iostream>
#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/StaticTensor.h>
#include <blazetest/mathtest/creator/DynamicTensor.h>
#include <blazetest/mathtest/creator/StaticTensor.h>
#include <blazetest/mathtest/dtensdtensadd/OperationTest.h>
#include <blazetest/system/MathTest.h>


//=================================================================================================
//
//  MAIN FUNCTION
//
//=================================================================================================

#if defined(BLAZE_USE_HPX_THREADS)
#include <hpx/hpx_main.hpp>
#endif

//*************************************************************************************************
int main()

{
   std::cout << "   Running 'T3x7x13bTDa'..." << std::endl;

   using blazetest::mathtest::TypeA;
   using blazetest::mathtest::TypeB;

   try
   {
      // Tensor type definitions
      using T3x7x13b = blaze::StaticTensor<TypeB,3UL,7UL,13UL>;
      using TDa = blaze::DynamicTensor<TypeA>;

      // Creator type definitions
      using CT3x7x13b = blazetest::Creator<T3x7x13b>;
      using CTDa = blazetest::Creator<TDa>;

      // Running the tests
      RUN_DTENSDTENSADD_OPERATION_TEST( CT3x7x13b(), CTDa( 3UL, 7UL, 13UL ) );
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during dense tensor/dense tensor addition:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
