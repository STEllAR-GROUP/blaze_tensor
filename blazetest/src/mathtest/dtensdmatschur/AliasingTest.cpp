//=================================================================================================
/*!
//  \file src/mathtest/dtensdmatschur/AliasingTest.cpp
//  \brief Source file for the dense tensor/dense tensor addition aliasing test
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


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cstdlib>
#include <iostream>

#include <blazetest/mathtest/dtensdmatschur/AliasingTest.h>


namespace blazetest {

namespace mathtest {

namespace dtensdmatschur {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the aliasing test class.
//
// \exception std::runtime_error Operation error detected.
*/
AliasingTest::AliasingTest()
   : dA2x3x4_ ( 2UL, 3UL, 4UL )
   , dC2x3x3_ ( 2UL, 3UL, 3UL )
   , dD2x3x3_ ( 2UL, 3UL, 3UL )
   , dA3x3_   ( 3UL, 3UL )
   , dB3x4_   ( 3UL, 4UL )

{
   testDTensDMatSchur ();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the dense tensor/dense tensor addition.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs aliasing tests for the dense tensor/dense tensor addition.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void AliasingTest::testDTensDMatSchur()
{
   //=====================================================================================
   // Schur
   //=====================================================================================

   // Assignment to left-hand side operand
   {
      test_ = "DTensDMatSchur - Assignment to left-hand side operand";

      initialize();

      result_   = dC2x3x3_ % dA3x3_;
      dC2x3x3_  = dC2x3x3_ % dA3x3_;

      checkResult( dC2x3x3_, result_ );
   }

   // Assignment to first operand of left-hand side compound
    {
       test_ = "DTensDMatSchur - Assignment to first operand of left-hand side compound";

       initialize();

       res_      = ( dA2x3x4_ % dB3x4_ ) % dA2x3x4_;
       dA2x3x4_  = ( dA2x3x4_ % dB3x4_ ) % dA2x3x4_;

       checkResult( dA2x3x4_, res_ );
    }

   // Assignment to right-hand side operand
   {
      test_ = "DTensDMatSchur - Assignment to right-hand side operand";

      initialize();

      result_  = eval( dC2x3x3_  % dA3x3_);
      dC2x3x3_ = eval( dC2x3x3_  % dA3x3_);

      checkResult( dC2x3x3_, result_ );
   }

   // Complex operation: A = ( 2*A ) % ( B % C *3 )
    {
       test_ = "DTensDMatSchur - Complex operation: A = ( 2*A ) % ( B % C *3 )";

       initialize();

       result_   = ( 2*dC2x3x3_ ) % ( dD2x3x3_ % dA3x3_ * 3);
       dC2x3x3_  = ( 2*dC2x3x3_ ) % ( dD2x3x3_ % dA3x3_ * 3);

       checkResult( dC2x3x3_, result_ );
    }


   //=====================================================================================
   // Schur with addition assignment
   //=====================================================================================

   // Addition Assignment to left-hand side operand
   {
      test_ = "DTensDMatSchur - Addition Assignment to left-hand side operand";

      initialize();

      result_ = dC2x3x3_;
      result_  += dC2x3x3_ % dA3x3_;
      dC2x3x3_ += dC2x3x3_ % dA3x3_;

      checkResult( dC2x3x3_, result_ );
   }

   // Assignment to first operand of left-hand side compound
    {
       test_ = "DTensDMatSchur - Addition Assignment to first operand of left-hand side compound";

       initialize();

       res_ = dA2x3x4_;
       res_     += ( dA2x3x4_ % dB3x4_ ) % dA2x3x4_;
       dA2x3x4_ += ( dA2x3x4_ % dB3x4_ ) % dA2x3x4_;

       checkResult( dA2x3x4_, res_ );
    }

   // Assignment to right-hand side operand
   {
      test_ = "DTensDMatSchur - Addition Assignment to right-hand side operand";

      initialize();

      result_ = dC2x3x3_;
      result_  += eval( dC2x3x3_  % dA3x3_);
      dC2x3x3_ += eval( dC2x3x3_  % dA3x3_);

      checkResult( dC2x3x3_, result_ );
   }

   // Complex operation: A = ( 2*A ) % ( B % C *3 )
    {
       test_ = "DTensDMatSchur - Complex operation: A += ( 2*A ) % ( B % C *3 )";

       initialize();

       result_ = dC2x3x3_;
       result_  += ( 2*dC2x3x3_ ) % ( dD2x3x3_ % dA3x3_ * 3);
       dC2x3x3_ += ( 2*dC2x3x3_ ) % ( dD2x3x3_ % dA3x3_ * 3);

       checkResult( dC2x3x3_, result_ );
    }

   //=====================================================================================
   // Schur with subtraction assignment
   //=====================================================================================

   // Subtraction Assignment to left-hand side operand
   {
      test_ = "DTensDMatSchur - Subtraction Assignment to left-hand side operand";

      initialize();

      result_ = dC2x3x3_;
      result_  -= dC2x3x3_ % dA3x3_;
      dC2x3x3_ -= dC2x3x3_ % dA3x3_;

      checkResult( dC2x3x3_, result_ );
   }

   // Subtraction to first operand of left-hand side compound
    {
       test_ = "DTensDMatSchur - Subtraction Assignment to first operand of left-hand side compound";

       initialize();

       res_ = dA2x3x4_;
       res_     -= ( dA2x3x4_ % dB3x4_ ) % dA2x3x4_;
       dA2x3x4_ -= ( dA2x3x4_ % dB3x4_ ) % dA2x3x4_;

       checkResult( dA2x3x4_, res_ );
    }

   // Subtraction Assignment to right-hand side operand
   {
      test_ = "DTensDMatSchur - Subtraction Assignment to right-hand side operand";

      initialize();

      result_ = dC2x3x3_;
      result_  -= eval( dC2x3x3_  % dA3x3_);
      dC2x3x3_ -= eval( dC2x3x3_  % dA3x3_);

      checkResult( dC2x3x3_, result_ );
   }

   // Complex operation: A -= ( 2*A ) % ( B % C *3 )
    {
       test_ = "DTensDMatSchur - Complex operation: A -= ( 2*A ) % ( B % C *3 )";

       initialize();

       result_ = dC2x3x3_;
       result_  -= ( 2*dC2x3x3_ ) % ( dD2x3x3_ % dA3x3_ * 3);
       dC2x3x3_ -= ( 2*dC2x3x3_ ) % ( dD2x3x3_ % dA3x3_ * 3);

       checkResult( dC2x3x3_, result_ );
    }


   //=====================================================================================
   // Schur product with Schur product assignment
   //=====================================================================================

   // Schur Assignment to left-hand side operand
   {
      test_ = "DTensDMatSchur - Schur Assignment to left-hand side operand";

      initialize();

      result_ = dC2x3x3_;
      result_  %= dC2x3x3_ % dA3x3_;
      dC2x3x3_ %= dC2x3x3_ % dA3x3_;

      checkResult( dC2x3x3_, result_ );
   }

   // Schur Assignment to first operand of left-hand side compound
    {
       test_ = "DTensDMatSchur - Schur Assignment to first operand of left-hand side compound";

       initialize();

       res_ = dA2x3x4_;
       res_     %= ( dA2x3x4_ % dB3x4_ ) % dA2x3x4_;
       dA2x3x4_ %= ( dA2x3x4_ % dB3x4_ ) % dA2x3x4_;

       checkResult( dA2x3x4_, res_ );
    }

   // Schur Assignment to right-hand side operand
   {
      test_ = "DTensDMatSchur - Schur Assignment to right-hand side operand";

      initialize();

      result_ = dC2x3x3_;
      result_  %= eval( dC2x3x3_  % dA3x3_);
      dC2x3x3_ %= eval( dC2x3x3_  % dA3x3_);

      checkResult( dC2x3x3_, result_ );
   }

   // Complex operation: A %= ( 2*A ) % ( B % C *3 )
    {
       test_ = "DTensDMatSchur - Complex operation: A %= ( 2*A ) % ( B % C *3 )";

       initialize();

       result_ = dC2x3x3_;
       result_  %= ( 2*dC2x3x3_ ) % ( dD2x3x3_ % dA3x3_ * 3);
       dC2x3x3_ %= ( 2*dC2x3x3_ ) % ( dD2x3x3_ % dA3x3_ * 3);

       checkResult( dC2x3x3_, result_ );
    }

}
//*************************************************************************************************



//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Initialization of all member vectors and matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function initializes all member vectors and matrices to specific predetermined values.
*/
void AliasingTest::initialize()
{
   // Initializing the first row-major dense tensor
   dA2x3x4_.resize( 2UL, 3UL, 4UL, false );
   dA2x3x4_(0,0,0) = -1;
   dA2x3x4_(0,0,1) =  0;
   dA2x3x4_(0,0,2) = -2;
   dA2x3x4_(0,0,3) =  0;
   dA2x3x4_(0,1,0) =  0;
   dA2x3x4_(0,1,1) =  2;
   dA2x3x4_(0,1,2) = -3;
   dA2x3x4_(0,1,3) =  1;
   dA2x3x4_(0,2,0) =  0;
   dA2x3x4_(0,2,1) =  1;
   dA2x3x4_(0,2,2) =  2;
   dA2x3x4_(0,2,3) =  2;
   dA2x3x4_(1,0,0) = -1;
   dA2x3x4_(1,0,1) =  0;
   dA2x3x4_(1,0,2) = -2;
   dA2x3x4_(1,0,3) =  0;
   dA2x3x4_(1,1,0) =  0;
   dA2x3x4_(1,1,1) =  2;
   dA2x3x4_(1,1,2) = -3;
   dA2x3x4_(1,1,3) =  1;
   dA2x3x4_(1,2,0) =  0;
   dA2x3x4_(1,2,1) =  1;
   dA2x3x4_(1,2,2) =  2;
   dA2x3x4_(1,2,3) =  2;

   // Initializing the third row-major dense tensor
   dC2x3x3_.resize( 2UL, 3UL, 3UL, false );
   dC2x3x3_(0,0,0) =  1;
   dC2x3x3_(0,0,1) =  0;
   dC2x3x3_(0,0,2) =  2;
   dC2x3x3_(0,1,0) =  0;
   dC2x3x3_(0,1,1) =  3;
   dC2x3x3_(0,1,2) = -1;
   dC2x3x3_(0,2,0) = -1;
   dC2x3x3_(0,2,1) =  0;
   dC2x3x3_(0,2,2) =  2;
   dC2x3x3_(1,0,0) =  1;
   dC2x3x3_(1,0,1) =  0;
   dC2x3x3_(1,0,2) =  2;
   dC2x3x3_(1,1,0) =  0;
   dC2x3x3_(1,1,1) =  3;
   dC2x3x3_(1,1,2) = -1;
   dC2x3x3_(1,2,0) = -1;
   dC2x3x3_(1,2,1) =  0;
   dC2x3x3_(1,2,2) =  2;

   // Initializing the fourth row-major dense tensor
   dD2x3x3_.resize( 2UL, 3UL, 3UL, false );
   dD2x3x3_(0,0,0) =  0;
   dD2x3x3_(0,0,1) = -1;
   dD2x3x3_(0,0,2) =  0;
   dD2x3x3_(0,1,0) =  1;
   dD2x3x3_(0,1,1) = -2;
   dD2x3x3_(0,1,2) =  2;
   dD2x3x3_(0,2,0) =  0;
   dD2x3x3_(0,2,1) =  0;
   dD2x3x3_(0,2,2) = -3;
   dD2x3x3_(1,0,0) =  0;
   dD2x3x3_(1,0,1) = -1;
   dD2x3x3_(1,0,2) =  0;
   dD2x3x3_(1,1,0) =  1;
   dD2x3x3_(1,1,1) = -2;
   dD2x3x3_(1,1,2) =  2;
   dD2x3x3_(1,2,0) =  0;
   dD2x3x3_(1,2,1) =  0;
   dD2x3x3_(1,2,2) = -3;

   // Initializing the first row-major dense matrix
   dA3x3_.resize( 3UL, 3UL, false );
   dA3x3_(0,0) =  1;
   dA3x3_(0,1) =  0;
   dA3x3_(0,2) =  2;
   dA3x3_(1,0) =  0;
   dA3x3_(1,1) =  3;
   dA3x3_(1,2) = -1;
   dA3x3_(2,0) = -1;
   dA3x3_(2,1) =  0;
   dA3x3_(2,2) =  2;

   // Initializing the first row-major dense tensor
   dB3x4_.resize( 3UL, 4UL, false );
   dB3x4_(0,0) =  1;
   dB3x4_(0,1) =  0;
   dB3x4_(0,2) =  2;
   dB3x4_(0,3) =  4;
   dB3x4_(1,0) =  0;
   dB3x4_(1,1) =  3;
   dB3x4_(1,2) = -1;
   dB3x4_(1,3) =  4;
   dB3x4_(2,0) = -1;
   dB3x4_(2,1) =  0;
   dB3x4_(2,2) =  2;
   dB3x4_(2,3) = -2;

}
//*************************************************************************************************

} // namespace dtensdmatschur

} // namespace mathtest

} // namespace blazetest




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
   std::cout << "   Running aliasing test..." << std::endl;

   try
   {
      RUN_DTENSDMATSCHUR_ALIASING_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during aliasing test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
