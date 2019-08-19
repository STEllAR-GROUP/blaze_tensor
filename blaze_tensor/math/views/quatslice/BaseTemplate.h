//=================================================================================================
/*!
//  \file blaze_tensor/math/views/quatslice/BaseTemplate.h
//  \brief Header file for the implementation of the QuatSlice base template
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

#ifndef _BLAZE_TENSOR_MATH_VIEWS_QUATSLICE_BASETEMPLATE_H_
#define _BLAZE_TENSOR_MATH_VIEWS_QUATSLICE_BASETEMPLATE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/DeclExpr.h>
#include <blaze/math/expressions/SchurExpr.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsContiguous.h>
#include <blaze/math/typetraits/IsOpposedView.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/MaxSize.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/math/views/Check.h>
#include <blaze/util/Assert.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/TypeList.h>
#include <blaze/util/Types.h>
#include <blaze/util/MaybeUnused.h>

// #include <blaze_tensor/math/expressions/ArrEvalExpr.h>
// #include <blaze_tensor/math/expressions/ArrArrAddExpr.h>
#include <blaze_tensor/math/expressions/ArrArrMapExpr.h>
// #include <blaze_tensor/math/expressions/ArrArrMultExpr.h>
// #include <blaze_tensor/math/expressions/ArrTensSubExpr.h>
#include <blaze_tensor/math/expressions/ArrScalarDivExpr.h>
#include <blaze_tensor/math/expressions/ArrScalarMultExpr.h>
// #include <blaze_tensor/math/expressions/ArrSerialExpr.h>
// #include <blaze_tensor/math/expressions/ArrTransExpr.h>
#include <blaze_tensor/math/typetraits/IsDenseTensor.h>
#include <blaze_tensor/math/typetraits/IsRowMajorTensor.h>


namespace blaze {

//=================================================================================================
//
//  ::blaze NAMESPACE FORWARD DECLARATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Base template of the QuatSlice class template.
// \ingroup quatslice
*/
template< typename AT                       // Type of the 4d array
        , size_t... CRAs >                  // Compile time quatslice arguments
class QuatSlice;
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ALIAS DECLARATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary alias declaration for the QuatSlice class template.
// \ingroup quatslice
//
// The QuatSlice_ alias declaration represents a convenient shortcut for the specification of the
// non-derived template arguments of the QuatSlice class template.
*/
template< typename AT       // Type of the 4d array
        , size_t... CRAs >  // Compile time quatslice arguments
using QuatSlice_ = QuatSlice< AT
                , CRAs... >;
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
