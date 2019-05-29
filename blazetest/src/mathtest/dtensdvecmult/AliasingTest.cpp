//=================================================================================================
/*!
//  \file src/mathtest/dtensdvecmult/AliasingTest.cpp
//  \brief Source file for the dense tensrix/dense vector multiplication aliasing test
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
//     of conditions and the following disclaimer in the documentation and/or other tenserials
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
#include <blazetest/mathtest/dtensdvecmult/AliasingTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace dtensdvecmult {

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
   : dB3x3x3_ ( 3UL, 3UL, 3UL )
   , dc3_   ( 3UL )
   , dd3_   ( 3UL )
   , res_()
{
   testDTensDVecMult ();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the dense tensrix/dense vector multiplication.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs aliasing tests for the dense tensrix/dense vector multiplication.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void AliasingTest::testDTensDVecMult()
{
   //=====================================================================================
   // Multiplication
   //=====================================================================================

   // Assignment to left-hand side operand
   {
      test_ = "DTensDVecMult - Assignment to right-hand side vector operand";

      initialize();

      res_ = (dB3x3x3_ * dc3_) * dc3_;
      dc3_ = (dB3x3x3_ * dc3_) * dc3_;

      checkResult( dc3_, res_ );
   }

   // Assignment to first operand of right-hand side compound
   {
      test_ = "DTensDVecMult - Assignment to first operand of right-hand side compound";

      initialize();

      res_ = dB3x3x3_ * ( dc3_ + dd3_ ) * dc3_;
      dc3_ = dB3x3x3_ * ( dc3_ + dd3_ ) * dc3_;

      checkResult( dc3_, res_ );
   }

   //=====================================================================================
   // Multiplication with addition assignment
   //=====================================================================================

   // Addition assignment to left-hand side operand
   {
      test_ = "DTensDVecMult - Addition assignment to right-hand side vector operand";

      initialize();

      res_ =  dc3_;
      res_ += (dB3x3x3_ * dc3_) * dc3_;
      dc3_ += (dB3x3x3_ * dc3_) * dc3_;

      checkResult( dc3_, res_ );
   }

   // Addition assignment to first operand of right-hand side compound
   {
      test_ = "DTensDVecMult - Addition assignment to first operand of left-hand side compound";

      initialize();

      res_ =  dc3_;
      res_ += dB3x3x3_ * ( dc3_ + dd3_ ) * dc3_;
      dc3_ += dB3x3x3_ * ( dc3_ + dd3_ ) * dc3_;

      checkResult( dc3_, res_ );
   }

   //=====================================================================================
   // Multiplication with subtraction assignment
   //=====================================================================================

   // subtraction assignment to left-hand side operand
   {
      test_ = "DTensDVecMult - subtraction assignment to right-hand side vector operand";

      initialize();

      res_ =  dc3_;
      res_ -= (dB3x3x3_ * dc3_) * dc3_;
      dc3_ -= (dB3x3x3_ * dc3_) * dc3_;

      checkResult( dc3_, res_ );
   }

   // subtraction assignment to first operand of right-hand side compound
   {
      test_ = "DTensDVecMult - subtraction assignment to first operand of left-hand side compound";

      initialize();

      res_ =  dc3_;
      res_ -= dB3x3x3_ * ( dc3_ + dd3_ ) * dc3_;
      dc3_ -= dB3x3x3_ * ( dc3_ + dd3_ ) * dc3_;

      checkResult( dc3_, res_ );
   }

   //=====================================================================================
   // Multiplication with schur assignment
   //=====================================================================================

   // schur assignment to left-hand side operand
   {
      test_ = "DTensDVecMult - schur assignment to right-hand side vector operand";

      initialize();

      res_ =  dc3_;
      res_ %= (dB3x3x3_ * dc3_) * dc3_;
      dc3_ %= (dB3x3x3_ * dc3_) * dc3_;

      checkResult( dc3_, res_ );
   }

   // schur assignment to first operand of right-hand side compound
   {
      test_ = "DTensDVecMult - schur assignment to first operand of left-hand side compound";

      initialize();

      res_ =  dc3_;
      res_ %= dB3x3x3_ * ( dc3_ + dd3_ ) * dc3_;
      dc3_ %= dB3x3x3_ * ( dc3_ + dd3_ ) * dc3_;

      checkResult( dc3_, res_ );
   }

}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Initialization of all member vectors and tensors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function initializes all member vectors and tensors to specific predetermined values.
*/
void AliasingTest::initialize()
{
   //=====================================================================================
   // Initialization of the dense tensors
   //=====================================================================================

   // Initializing the fisrt row-major dense tensor
   dB3x3x3_.resize( 3UL, 3UL, 3UL, false );
   dB3x3x3_(0,0,0) =  1;
   dB3x3x3_(0,0,1) =  0;
   dB3x3x3_(0,0,2) = -3;
   dB3x3x3_(0,1,0) =  0;
   dB3x3x3_(0,1,1) = -1;
   dB3x3x3_(0,1,2) =  0;
   dB3x3x3_(0,2,0) =  0;
   dB3x3x3_(0,2,1) =  2;
   dB3x3x3_(0,2,2) =  1;
   dB3x3x3_(1,0,0) =  1;
   dB3x3x3_(1,0,1) =  0;
   dB3x3x3_(1,0,2) = -3;
   dB3x3x3_(1,1,0) =  0;
   dB3x3x3_(1,1,1) = -1;
   dB3x3x3_(1,1,2) =  0;
   dB3x3x3_(1,2,0) =  0;
   dB3x3x3_(1,2,1) =  2;
   dB3x3x3_(1,2,2) =  1;
   dB3x3x3_(2,0,0) = -1;
   dB3x3x3_(2,0,1) = -2;
   dB3x3x3_(2,0,2) = -3;
   dB3x3x3_(2,1,0) =  0;
   dB3x3x3_(2,1,1) = -1;
   dB3x3x3_(2,1,2) =  4;
   dB3x3x3_(2,2,0) =  0;
   dB3x3x3_(2,2,1) =  2;
   dB3x3x3_(2,2,2) =  2;


   //=====================================================================================
   // Initialization of the dense vectors
   //=====================================================================================

   // Initializing the third dense column vector
   dc3_.resize( 3UL, false );
   dc3_[0] = 1;
   dc3_[1] = 2;
   dc3_[2] = 3;

   // Initializing the fourth dense column vector
   dd3_.resize( 3UL, false );
   dd3_[0] = 0;
   dd3_[1] = 2;
   dd3_[2] = 1;


}
//*************************************************************************************************

} // namespace dtensdvecmult

} // namespace mathtest

} // namespace blazetest




//=================================================================================================
//
//  MAIN FUNCTION
//
//=================================================================================================

//*************************************************************************************************
int main()
{
   std::cout << "   Running aliasing test..." << std::endl;

   try
   {
      RUN_DTENSDVECMULT_ALIASING_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during aliasing test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
