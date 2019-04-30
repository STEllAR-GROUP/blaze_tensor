//=================================================================================================
/*!
//  \file blaze_tensor/math/dense/HybridVector.h
//  \brief Header file for the HybridVector class template
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2018-2019 Hartmut Kaiser - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//
//  * The names of its contributors may not be used to endorse or promote products derived
//    from this software without specific prior written permission.
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

#ifndef _BLAZE_TENSOR_MATH_DENSE_HYBRIDVECTOR_H_
#define _BLAZE_TENSOR_MATH_DENSE_HYBRIDVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/dense/HybridVector.h>
#include <blaze_tensor/math/traits/DilatedSubvectorTrait.h>


namespace blaze {

//=================================================================================================
//
//  SUBVECTORTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT >
struct DilatedSubvectorTraitEval2< VT, inf, inf, inf
                          , EnableIf_t< IsDenseVector_v<VT> &&
                                        ( Size_v<VT,0UL> != DefaultSize_v ||
                                          MaxSize_v<VT,0UL> != DefaultMaxSize_v ) > >
{
   static constexpr size_t N = max( Size_v<VT,0UL>, MaxSize_v<VT,0UL> );

   using Type = HybridVector< RemoveConst_t< ElementType_t<VT> >, N, TransposeFlag_v<VT> >;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
