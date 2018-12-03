//=================================================================================================
/*!
//  \file blazetest/config/MathTest.h
//  \brief General configuration file for the math tests of the blaze test suite
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

#include <blazetest/config/MathTest.h>

//*************************************************************************************************
/*!\brief Compilation switch for the subtensor tests.
//
// This compilation switch triggers the subtensor test scenarios. In case the subtensor tests are
// activated, subtensor tests in combination with all activated operations are included in the
// tests. In case the subtensor tests are disabled, all kinds of subtensor tests are excluded.
//
// The following settings are possible:
//
//   - 0: The subtensor tests are not included in the compilation process and not executed
//   - 1: The subtensor tests are included in the compilation process, but not executed
//   - 2: The subtensor tests are included in the compilation process and executed
*/
#define BLAZETEST_MATHTEST_TEST_SUBTENSOR_OPERATION 2
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the rowslice tests.
//
// This compilation switch triggers the rowslice test scenarios. In case the rowslice tests are
// activated, rowslice tests in combination with all activated operations are included in the
// tests. In case the rowslice tests are disabled, all kinds of rowslice tests are excluded.
//
// The following settings are possible:
//
//   - 0: The rowslice tests are not included in the compilation process and not executed
//   - 1: The rowslice tests are included in the compilation process, but not executed
//   - 2: The rowslice tests are included in the compilation process and executed
*/
#define BLAZETEST_MATHTEST_TEST_ROWSLICE_OPERATION 2
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the columnslice tests.
//
// This compilation switch triggers the columnslice test scenarios. In case the columnslice tests are
// activated, columnslice tests in combination with all activated operations are included in the
// tests. In case the columnslice tests are disabled, all kinds of columnslice tests are excluded.
//
// The following settings are possible:
//
//   - 0: The columnslice tests are not included in the compilation process and not executed
//   - 1: The columnslice tests are included in the compilation process, but not executed
//   - 2: The columnslice tests are included in the compilation process and executed
*/
#define BLAZETEST_MATHTEST_TEST_COLUMNSLICE_OPERATION 2
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the pageslice tests.
//
// This compilation switch triggers the pageslice test scenarios. In case the pageslice tests are
// activated, pageslice tests in combination with all activated operations are included in the
// tests. In case the pageslice tests are disabled, all kinds of pageslice tests are excluded.
//
// The following settings are possible:
//
//   - 0: The pageslice tests are not included in the compilation process and not executed
//   - 1: The pageslice tests are included in the compilation process, but not executed
//   - 2: The pageslice tests are included in the compilation process and executed
*/
#define BLAZETEST_MATHTEST_TEST_PAGESLICE_OPERATION 2
//*************************************************************************************************

