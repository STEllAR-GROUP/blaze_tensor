//=================================================================================================
/*!
//  \file blaze_tensor/math/expressions/DTensDTensMultExpr.h
//  \brief Header file for the dense tensor/dense tensor multiplication expression
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

#ifndef _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSDTENSMULTEXPR_H_
#define _BLAZE_TENSOR_MATH_EXPRESSIONS_DTENSDTENSMULTEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/DMatDMatMultExpr.h>

#include <blaze_tensor/math/constraints/DenseTensor.h>
#include <blaze_tensor/math/constraints/TensTensMultExpr.h>
#include <blaze_tensor/math/expressions/DTensScalarMultExpr.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>
#include <blaze_tensor/math/expressions/Forward.h>
#include <blaze_tensor/math/expressions/TensScalarMultExpr.h>
#include <blaze_tensor/math/expressions/TensTensMultExpr.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DTENSDTENSMULTEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for dense tensor-dense tensor multiplications.
// \ingroup dense_tensor_expression
//
// The DTensDTensMultExpr class represents the compile time expression for multiplications between
// row-major dense tensors.
*/
template< typename MT1  // Type of the left-hand side dense tensor
        , typename MT2 > // Type of the right-hand side dense tensor
class DTensDTensMultExpr
   : public TensTensMultExpr< DenseTensor< DTensDTensMultExpr<MT1,MT2> > >
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   using RT1 = ResultType_t<MT1>;     //!< Result type of the left-hand side dense tensor expression.
   using RT2 = ResultType_t<MT2>;     //!< Result type of the right-hand side dense tensor expression.
   using ET1 = ElementType_t<RT1>;    //!< Element type of the left-hand side dense tensor expression.
   using ET2 = ElementType_t<RT2>;    //!< Element type of the right-hand side dense tensor expression.
   using CT1 = CompositeType_t<MT1>;  //!< Composite type of the left-hand side dense tensor expression.
   using CT2 = CompositeType_t<MT2>;  //!< Composite type of the right-hand side dense tensor expression.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the left-hand side dense tensor expression.
   static constexpr bool evaluateLeft = ( IsComputation_v<MT1> || RequiresEvaluation_v<MT1> );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the right-hand side dense tensor expression.
   static constexpr bool evaluateRight = ( IsComputation_v<MT2> || RequiresEvaluation_v<MT2> );
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! This variable template is a helper for the selection of the optimal evaluation strategy. In
       case the target tensor is column-major and either of the two tensor operands is symmetric,
       the variable is set to 1 and an optimized evaluation strategy is selected. Otherwise the
       variable is set to 0 and the default strategy is chosen. */
   template< typename T1, typename T2, typename T3 >
   static constexpr bool CanExploitSymmetry_v =
      ( IsColumnMajorMatrix_v<T1> && ( IsSymmetric_v<T2> || IsSymmetric_v<T3> ) );
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! This variable template is a helper for the selection of the parallel evaluation strategy.
       In case either of the two tensor operands requires an intermediate evaluation, the variable
       will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   static constexpr bool IsEvaluationRequired_v =
      ( ( evaluateLeft || evaluateRight ) && !CanExploitSymmetry_v<T1,T2,T3> );
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! In case the types of all three involved tensors are suited for a BLAS kernel, the variable
       will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   static constexpr bool UseBlasKernel_v =
      ( BLAZE_BLAS_MODE && BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION &&
        IsContiguous_v<T1> && HasMutableDataAccess_v<T1> &&
        IsContiguous_v<T2> && HasConstDataAccess_v<T2> &&
        IsContiguous_v<T3> && HasConstDataAccess_v<T3> &&
        !IsDiagonal_v<T2> && !IsDiagonal_v<T3> &&
        T1::simdEnabled && T2::simdEnabled && T3::simdEnabled &&
        IsBLASCompatible_v< ElementType_t<T1> > &&
        IsBLASCompatible_v< ElementType_t<T2> > &&
        IsBLASCompatible_v< ElementType_t<T3> > &&
        IsSame_v< ElementType_t<T1>, ElementType_t<T2> > &&
        IsSame_v< ElementType_t<T1>, ElementType_t<T3> > );
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! In case all three involved data types are suited for a vectorized computation of the
       tensor multiplication, the variable will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   static constexpr bool UseVectorizedDefaultKernel_v =
      ( useOptimizedKernels &&
        !IsDiagonal_v<T2> && !IsDiagonal_v<T3> &&
        T1::simdEnabled && T2::simdEnabled && T3::simdEnabled &&
        IsSIMDCombinable_v< ElementType_t<T1>
                          , ElementType_t<T2>
                          , ElementType_t<T3> > &&
        HasSIMDAdd_v< ElementType_t<T2>, ElementType_t<T3> > &&
        HasSIMDMult_v< ElementType_t<T2>, ElementType_t<T3> > );
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Type of the functor for forwarding an expression to another assign kernel.
   /*! In case a temporary tensor needs to be created, this functor is used to forward the
       resulting expression to another assign kernel. */
   using ForwardFunctor = Noop;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this DTensDTensMultExpr instance.
   using This = DTensDTensMultExpr<MT1,MT2>;

   using ResultType    = MultTrait_t<RT1,RT2>;         //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;   //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;  //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<ResultType>;    //!< Resulting element type.
   using SIMDType      = SIMDTrait_t<ElementType>;     //!< Resulting SIMD element type.
   using ReturnType    = const ElementType;            //!< Return type for expression template evaluations.
   using CompositeType = const ResultType;             //!< Data type for composite expression templates.

   //! Composite type of the left-hand side dense tensor expression.
   using LeftOperand = If_t< IsExpression_v<MT1>, const MT1, const MT1& >;

   //! Composite type of the right-hand side dense tensor expression.
   using RightOperand = If_t< IsExpression_v<MT2>, const MT2, const MT2& >;

   //! Type for the assignment of the left-hand side dense tensor operand.
   using LT = If_t< evaluateLeft, const RT1, CT1 >;

   //! Type for the assignment of the right-hand side dense tensor operand.
   using RT = If_t< evaluateRight, const RT2, CT2 >;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled =
      ( !IsDiagonal_v<MT2> &&
        MT1::simdEnabled && MT2::simdEnabled &&
        HasSIMDAdd_v<ET1,ET2> &&
        HasSIMDMult_v<ET1,ET2> );

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable =
      ( !evaluateLeft  && MT1::smpAssignable && !evaluateRight && MT2::smpAssignable );
   //**********************************************************************************************

   //**********************************************************************************************
   static constexpr bool SYM  = false;  //!< Flag for symmetric tensors.
   static constexpr bool HERM = false;  //!< Flag for Hermitian tensors.
   static constexpr bool LOW  = false;  //!< Flag for lower tensors.
   static constexpr bool UPP  = false;  //!< Flag for upper tensors.
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   static constexpr size_t SIMDSIZE = SIMDTrait<ElementType>::size;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DTensDTensMultExpr class.
   //
   // \param lhs The left-hand side operand of the multiplication expression.
   // \param rhs The right-hand side operand of the multiplication expression.
   */
   explicit inline DTensDTensMultExpr( const MT1& lhs, const MT2& rhs ) noexcept
      : lhs_( lhs )  // Left-hand side dense tensor of the multiplication expression
      , rhs_( rhs )  // Right-hand side dense tensor of the multiplication expression
   {
      BLAZE_INTERNAL_ASSERT( lhs.columns() == rhs.rows(), "Invalid tensor sizes" );
   }
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 2D-access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator()( size_t k, size_t i, size_t j ) const {
      BLAZE_INTERNAL_ASSERT( i < lhs_.rows()   , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( k < lhs_.pages(),   "Invalid page access index" );
      BLAZE_INTERNAL_ASSERT( j < rhs_.columns(), "Invalid column access index" );
      BLAZE_INTERNAL_ASSERT( k < rhs_.pages(),   "Invalid page access index" );

      return row( lhs_, i, k, unchecked ) * column( rhs_, j, k, unchecked );
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid tensor access index.
   */
   inline ReturnType at( size_t k, size_t i, size_t j ) const {
      if( i >= lhs_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( k >= lhs_.pages() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
      }
      if( j >= rhs_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      if( k >= rhs_.pages() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
      }
      return (*this)(k,i,j);
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the tensor.
   //
   // \return The number of rows of the tensor.
   */
   inline size_t rows() const noexcept {
      return lhs_.rows();
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the tensor.
   //
   // \return The number of columns of the tensor.
   */
   inline size_t columns() const noexcept {
      return rhs_.columns();
   }
   //**********************************************************************************************

   //**Pages function****************************************************************************
   /*!\brief Returns the current number of pages of the tensor.
   //
   // \return The number of pages of the tensor.
   */
   inline size_t pages() const noexcept {
      return rhs_.pages();
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side dense tensor operand.
   //
   // \return The left-hand side dense tensor operand.
   */
   inline LeftOperand leftOperand() const noexcept {
      return lhs_;
   }
   //**********************************************************************************************

   //**Right operand access************************************************************************
   /*!\brief Returns the right-hand side dense tensor operand.
   //
   // \return The right-hand side dense tensor operand.
   */
   inline RightOperand rightOperand() const noexcept {
      return rhs_;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the expression can alias, \a false otherwise.
   */
   template< typename T >
   inline bool canAlias( const T* alias ) const noexcept {
      return ( lhs_.canAlias( alias ) || rhs_.canAlias( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case an alias effect is detected, \a false otherwise.
   */
   template< typename T >
   inline bool isAliased( const T* alias ) const noexcept {
      return ( lhs_.isAliased( alias ) || rhs_.isAliased( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const noexcept {
      return lhs_.isAligned() && rhs_.isAligned();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return ( !BLAZE_BLAS_MODE ||
               !BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION ||
               !BLAZE_BLAS_IS_PARALLEL ||
               ( rows() * columns() < DMATDMATMULT_THRESHOLD ) ) &&
             ( rows() * columns() >= SMP_DMATDMATMULT_THRESHOLD ) &&
             !IsDiagonal_v<MT1> && !IsDiagonal_v<MT2>;
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  lhs_;  //!< Left-hand side dense tensor of the multiplication expression.
   RightOperand rhs_;  //!< Right-hand side dense tensor of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense tensors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a dense tensor-dense tensor multiplication to a dense tensor
   //        (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a dense tensor-dense
   // tensor multiplication expression to a dense tensor.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline void
      assign( DenseTensor<MT>& lhs, const DTensDTensMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).pages() == rhs.pages(),     "Invalid number of pages" );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || (~lhs).pages() == 0UL ) {
         return;
      }
      else if( rhs.lhs_.columns() == 0UL || (~rhs).pages() == 0UL ) {
         reset( ~lhs );
         return;
      }

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense tensor operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.pages() == rhs.lhs_.pages(),     "Invalid number of pages" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages() == rhs.rhs_.pages(),     "Invalid number of pages" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.pages() == (~lhs).pages()  ,     "Invalid number of pages" );

      DTensDTensMultExpr::selectAssignKernel( ~lhs, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to dense tensors (kernel selection)*********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for an assignment of a dense tensor-dense tensor
   //        multiplication to a dense tensor (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline void selectAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      if( ( IsDiagonal_v<MT5> ) ||
          ( !BLAZE_DEBUG_MODE && B.columns() <= SIMDSIZE*10UL ) ||
          ( C.rows() * C.columns() < DMATDMATMULT_THRESHOLD ) )
         selectSmallAssignKernel( C, A, B );
      else
         selectBlasAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense tensors (general/general)**************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a general dense tensor-general dense tensor multiplication
   //        (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default assignment of a general dense tensor-general dense
   // tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline void
      selectDefaultAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      BLAZE_INTERNAL_ASSERT( !( SYM || HERM || LOW || UPP ) || ( M == N ), "Broken invariant detected" );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t kbegin( ( IsUpper_v<MT4> )
                              ?( IsStrictlyUpper_v<MT4> ? i+1UL : i )
                              :( 0UL ) );
         const size_t kend( ( IsLower_v<MT4> )
                            ?( IsStrictlyLower_v<MT4> ? i : i+1UL )
                            :( K ) );
         BLAZE_INTERNAL_ASSERT( kbegin <= kend, "Invalid loop indices detected" );

         if( IsStrictlyTriangular_v<MT4> && kbegin == kend ) {
            for( size_t j=0UL; j<N; ++j ) {
               reset( C(i,j) );
            }
            continue;
         }

         {
            const size_t jbegin( ( IsUpper_v<MT5> )
                                 ?( ( IsStrictlyUpper_v<MT5> )
                                    ?( UPP ? max(i,kbegin+1UL) : kbegin+1UL )
                                    :( UPP ? max(i,kbegin) : kbegin ) )
                                 :( UPP ? i : 0UL ) );
            const size_t jend( ( IsLower_v<MT5> )
                               ?( ( IsStrictlyLower_v<MT5> )
                                  ?( LOW ? min(i+1UL,kbegin) : kbegin )
                                  :( LOW ? min(i,kbegin)+1UL : kbegin+1UL ) )
                               :( LOW ? i+1UL : N ) );

            if( ( IsUpper_v<MT4> && IsUpper_v<MT5> ) || UPP ) {
               for( size_t j=0UL; j<jbegin; ++j ) {
                  reset( C(i,j) );
               }
            }
            else if( IsStrictlyUpper_v<MT5> ) {
               reset( C(i,0UL) );
            }
            for( size_t j=jbegin; j<jend; ++j ) {
               C(i,j) = A(i,kbegin) * B(kbegin,j);
            }
            if( ( IsLower_v<MT4> && IsLower_v<MT5> ) || LOW ) {
               for( size_t j=jend; j<N; ++j ) {
                  reset( C(i,j) );
               }
            }
            else if( IsStrictlyLower_v<MT5> ) {
               reset( C(i,N-1UL) );
            }
         }

         for( size_t k=kbegin+1UL; k<kend; ++k )
         {
            const size_t jbegin( ( IsUpper_v<MT5> )
                                 ?( ( IsStrictlyUpper_v<MT5> )
                                    ?( SYM || HERM || UPP ? max( i, k+1UL ) : k+1UL )
                                    :( SYM || HERM || UPP ? max( i, k ) : k ) )
                                 :( SYM || HERM || UPP ? i : 0UL ) );
            const size_t jend( ( IsLower_v<MT5> )
                               ?( ( IsStrictlyLower_v<MT5> )
                                  ?( LOW ? min(i+1UL,k-1UL) : k-1UL )
                                  :( LOW ? min(i+1UL,k) : k ) )
                               :( LOW ? i+1UL : N ) );

            if( ( SYM || HERM || LOW || UPP ) && ( jbegin > jend ) ) continue;
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            for( size_t j=jbegin; j<jend; ++j ) {
               C(i,j) += A(i,k) * B(k,j);
            }
            if( IsLower_v<MT5> ) {
               C(i,jend) = A(i,k) * B(k,jend);
            }
         }
      }

      if( SYM || HERM ) {
         for( size_t i=1UL; i<M; ++i ) {
            for( size_t j=0UL; j<i; ++j ) {
               C(i,j) = HERM ? conj( C(j,i) ) : C(j,i);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense tensors (general/diagonal)*************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a general dense tensor-diagonal dense tensor multiplication
   //        (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default assignment of a general dense tensor-diagonal dense
   // tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< !IsDiagonal_v<MT4> && IsDiagonal_v<MT5> >
      selectDefaultAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper_v<MT4> )
                              ?( IsStrictlyUpper_v<MT4> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower_v<MT4> )
                            ?( IsStrictlyLower_v<MT4> ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         if( IsUpper_v<MT4> ) {
            for( size_t j=0UL; j<jbegin; ++j ) {
               reset( C(i,j) );
            }
         }
         for( size_t j=jbegin; j<jend; ++j ) {
            C(i,j) = A(i,j) * B(j,j);
         }
         if( IsLower_v<MT4> ) {
            for( size_t j=jend; j<N; ++j ) {
               reset( C(i,j) );
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense tensors (diagonal/general)*************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a diagonal dense tensor-general dense tensor multiplication
   //        (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default assignment of a diagonal dense tensor-general dense
   // tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< IsDiagonal_v<MT4> && !IsDiagonal_v<MT5> >
      selectDefaultAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper_v<MT5> )
                              ?( IsStrictlyUpper_v<MT5> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower_v<MT5> )
                            ?( IsStrictlyLower_v<MT5> ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         if( IsUpper_v<MT5> ) {
            for( size_t j=0UL; j<jbegin; ++j ) {
               reset( C(i,j) );
            }
         }
         for( size_t j=jbegin; j<jend; ++j ) {
            C(i,j) = A(i,i) * B(i,j);
         }
         if( IsLower_v<MT5> ) {
            for( size_t j=jend; j<N; ++j ) {
               reset( C(i,j) );
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense tensors (diagonal/diagonal)************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a diagonal dense tensor-diagonal dense tensor multiplication
   //        (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default assignment of a diagonal dense tensor-diagonal dense
   // tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< IsDiagonal_v<MT4> && IsDiagonal_v<MT5> >
      selectDefaultAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      reset( C );

      for( size_t i=0UL; i<A.rows(); ++i ) {
         C(i,i) = A(i,i) * B(i,i);
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense tensors (small tensors)***************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a small dense tensor-dense tensor multiplication (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a dense tensor-
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline DisableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5> >
      selectSmallAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectDefaultAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default assignment to row-major dense tensors (small tensors)******************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default assignment of a small dense tensor-dense tensor multiplication
   //        (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default assignment of a dense tensor-dense tensor
   // multiplication expression to a row-major dense tensor. This kernel is optimized for small
   // tensors.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5> >
      selectSmallAssignKernel( DenseTensor<MT3>& C, const MT4& A, const MT5& B )
   {
      constexpr bool remainder( !IsPadded_v<MT3> || !IsPadded_v<MT5> );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      BLAZE_INTERNAL_ASSERT( ( M == N ), "Broken invariant detected" );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      if( N > SIMDSIZE*3UL ) {
         reset( ~C );
      }

      {
         size_t j( 0UL );

         if( IsIntegral_v<ElementType> )
         {
            for( ; (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL ) {
               for( size_t i=0UL; i<M; ++i )
               {
                  const size_t kbegin( 0UL );
                  const size_t kend( K );

                  SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                  for( size_t k=kbegin; k<kend; ++k ) {
                     const SIMDType a1( set( A(i,k) ) );
                     xmm1 += a1 * B.load(k,j             );
                     xmm2 += a1 * B.load(k,j+SIMDSIZE    );
                     xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
                     xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
                     xmm5 += a1 * B.load(k,j+SIMDSIZE*4UL);
                     xmm6 += a1 * B.load(k,j+SIMDSIZE*5UL);
                     xmm7 += a1 * B.load(k,j+SIMDSIZE*6UL);
                     xmm8 += a1 * B.load(k,j+SIMDSIZE*7UL);
                  }

                  (~C).store( i, j             , xmm1 );
                  (~C).store( i, j+SIMDSIZE    , xmm2 );
                  (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
                  (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
                  (~C).store( i, j+SIMDSIZE*4UL, xmm5 );
                  (~C).store( i, j+SIMDSIZE*5UL, xmm6 );
                  (~C).store( i, j+SIMDSIZE*6UL, xmm7 );
                  (~C).store( i, j+SIMDSIZE*7UL, xmm8 );
               }
            }
         }

         for( ; (j+SIMDSIZE*4UL) < jpos; j+=SIMDSIZE*5UL )
         {
            size_t i( 0UL );

            for( ; (i+2UL) <= M; i+=2UL )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i    ,k) ) );
                  const SIMDType a2( set( A(i+1UL,k) ) );
                  const SIMDType b1( B.load(k,j             ) );
                  const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
                  const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
                  const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
                  const SIMDType b5( B.load(k,j+SIMDSIZE*4UL) );
                  xmm1  += a1 * b1;
                  xmm2  += a1 * b2;
                  xmm3  += a1 * b3;
                  xmm4  += a1 * b4;
                  xmm5  += a1 * b5;
                  xmm6  += a2 * b1;
                  xmm7  += a2 * b2;
                  xmm8  += a2 * b3;
                  xmm9  += a2 * b4;
                  xmm10 += a2 * b5;
               }

               (~C).store( i    , j             , xmm1  );
               (~C).store( i    , j+SIMDSIZE    , xmm2  );
               (~C).store( i    , j+SIMDSIZE*2UL, xmm3  );
               (~C).store( i    , j+SIMDSIZE*3UL, xmm4  );
               (~C).store( i    , j+SIMDSIZE*4UL, xmm5  );
               (~C).store( i+1UL, j             , xmm6  );
               (~C).store( i+1UL, j+SIMDSIZE    , xmm7  );
               (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm8  );
               (~C).store( i+1UL, j+SIMDSIZE*3UL, xmm9  );
               (~C).store( i+1UL, j+SIMDSIZE*4UL, xmm10 );
            }

            if( i < M )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i,k) ) );
                  xmm1 += a1 * B.load(k,j             );
                  xmm2 += a1 * B.load(k,j+SIMDSIZE    );
                  xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
                  xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
                  xmm5 += a1 * B.load(k,j+SIMDSIZE*4UL);
               }

               (~C).store( i, j             , xmm1 );
               (~C).store( i, j+SIMDSIZE    , xmm2 );
               (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
               (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
               (~C).store( i, j+SIMDSIZE*4UL, xmm5 );
            }
         }

         for( ; (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
         {
            const size_t iend( M );
            size_t i( 0UL );

            for( ; (i+2UL) <= iend; i+=2UL )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i    ,k) ) );
                  const SIMDType a2( set( A(i+1UL,k) ) );
                  const SIMDType b1( B.load(k,j             ) );
                  const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
                  const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
                  const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
                  xmm1 += a1 * b1;
                  xmm2 += a1 * b2;
                  xmm3 += a1 * b3;
                  xmm4 += a1 * b4;
                  xmm5 += a2 * b1;
                  xmm6 += a2 * b2;
                  xmm7 += a2 * b3;
                  xmm8 += a2 * b4;
               }

               (~C).store( i    , j             , xmm1 );
               (~C).store( i    , j+SIMDSIZE    , xmm2 );
               (~C).store( i    , j+SIMDSIZE*2UL, xmm3 );
               (~C).store( i    , j+SIMDSIZE*3UL, xmm4 );
               (~C).store( i+1UL, j             , xmm5 );
               (~C).store( i+1UL, j+SIMDSIZE    , xmm6 );
               (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm7 );
               (~C).store( i+1UL, j+SIMDSIZE*3UL, xmm8 );
            }

            if( i < iend )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3, xmm4;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i,k) ) );
                  xmm1 += a1 * B.load(k,j             );
                  xmm2 += a1 * B.load(k,j+SIMDSIZE    );
                  xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
                  xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
               }

               (~C).store( i, j             , xmm1 );
               (~C).store( i, j+SIMDSIZE    , xmm2 );
               (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
               (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
            }
         }

         for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
         {
            const size_t iend( M );
            size_t i( 0UL );

            for( ; (i+2UL) <= iend; i+=2UL )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i    ,k) ) );
                  const SIMDType a2( set( A(i+1UL,k) ) );
                  const SIMDType b1( B.load(k,j             ) );
                  const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
                  const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
                  xmm1 += a1 * b1;
                  xmm2 += a1 * b2;
                  xmm3 += a1 * b3;
                  xmm4 += a2 * b1;
                  xmm5 += a2 * b2;
                  xmm6 += a2 * b3;
               }

               (~C).store( i    , j             , xmm1 );
               (~C).store( i    , j+SIMDSIZE    , xmm2 );
               (~C).store( i    , j+SIMDSIZE*2UL, xmm3 );
               (~C).store( i+1UL, j             , xmm4 );
               (~C).store( i+1UL, j+SIMDSIZE    , xmm5 );
               (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm6 );
            }

            if( i < iend )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i,k) ) );
                  xmm1 += a1 * B.load(k,j             );
                  xmm2 += a1 * B.load(k,j+SIMDSIZE    );
                  xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
               }

               (~C).store( i, j             , xmm1 );
               (~C).store( i, j+SIMDSIZE    , xmm2 );
               (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
            }
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
         {
            const size_t iend( M );
            size_t i( 0UL );

            for( ; (i+4UL) <= iend; i+=4UL )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i    ,k) ) );
                  const SIMDType a2( set( A(i+1UL,k) ) );
                  const SIMDType a3( set( A(i+2UL,k) ) );
                  const SIMDType a4( set( A(i+3UL,k) ) );
                  const SIMDType b1( B.load(k,j         ) );
                  const SIMDType b2( B.load(k,j+SIMDSIZE) );
                  xmm1 += a1 * b1;
                  xmm2 += a1 * b2;
                  xmm3 += a2 * b1;
                  xmm4 += a2 * b2;
                  xmm5 += a3 * b1;
                  xmm6 += a3 * b2;
                  xmm7 += a4 * b1;
                  xmm8 += a4 * b2;
               }

               (~C).store( i    , j         , xmm1 );
               (~C).store( i    , j+SIMDSIZE, xmm2 );
               (~C).store( i+1UL, j         , xmm3 );
               (~C).store( i+1UL, j+SIMDSIZE, xmm4 );
               (~C).store( i+2UL, j         , xmm5 );
               (~C).store( i+2UL, j+SIMDSIZE, xmm6 );
               (~C).store( i+3UL, j         , xmm7 );
               (~C).store( i+3UL, j+SIMDSIZE, xmm8 );
            }

            for( ; (i+3UL) <= iend; i+=3UL )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i    ,k) ) );
                  const SIMDType a2( set( A(i+1UL,k) ) );
                  const SIMDType a3( set( A(i+2UL,k) ) );
                  const SIMDType b1( B.load(k,j         ) );
                  const SIMDType b2( B.load(k,j+SIMDSIZE) );
                  xmm1 += a1 * b1;
                  xmm2 += a1 * b2;
                  xmm3 += a2 * b1;
                  xmm4 += a2 * b2;
                  xmm5 += a3 * b1;
                  xmm6 += a3 * b2;
               }

               (~C).store( i    , j         , xmm1 );
               (~C).store( i    , j+SIMDSIZE, xmm2 );
               (~C).store( i+1UL, j         , xmm3 );
               (~C).store( i+1UL, j+SIMDSIZE, xmm4 );
               (~C).store( i+2UL, j         , xmm5 );
               (~C).store( i+2UL, j+SIMDSIZE, xmm6 );
            }

            for( ; (i+2UL) <= iend; i+=2UL )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
               size_t k( kbegin );

               for( ; (k+2UL) <= kend; k+=2UL ) {
                  const SIMDType a1( set( A(i    ,k    ) ) );
                  const SIMDType a2( set( A(i+1UL,k    ) ) );
                  const SIMDType a3( set( A(i    ,k+1UL) ) );
                  const SIMDType a4( set( A(i+1UL,k+1UL) ) );
                  const SIMDType b1( B.load(k    ,j         ) );
                  const SIMDType b2( B.load(k    ,j+SIMDSIZE) );
                  const SIMDType b3( B.load(k+1UL,j         ) );
                  const SIMDType b4( B.load(k+1UL,j+SIMDSIZE) );
                  xmm1 += a1 * b1;
                  xmm2 += a1 * b2;
                  xmm3 += a2 * b1;
                  xmm4 += a2 * b2;
                  xmm5 += a3 * b3;
                  xmm6 += a3 * b4;
                  xmm7 += a4 * b3;
                  xmm8 += a4 * b4;
               }

               for( ; k<kend; ++k ) {
                  const SIMDType a1( set( A(i    ,k) ) );
                  const SIMDType a2( set( A(i+1UL,k) ) );
                  const SIMDType b1( B.load(k,j         ) );
                  const SIMDType b2( B.load(k,j+SIMDSIZE) );
                  xmm1 += a1 * b1;
                  xmm2 += a1 * b2;
                  xmm3 += a2 * b1;
                  xmm4 += a2 * b2;
               }

               (~C).store( i    , j         , xmm1+xmm5 );
               (~C).store( i    , j+SIMDSIZE, xmm2+xmm6 );
               (~C).store( i+1UL, j         , xmm3+xmm7 );
               (~C).store( i+1UL, j+SIMDSIZE, xmm4+xmm8 );
            }

            if( i < iend )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3, xmm4;
               size_t k( kbegin );

               for( ; (k+2UL) <= kend; k+=2UL ) {
                  const SIMDType a1( set( A(i,k    ) ) );
                  const SIMDType a2( set( A(i,k+1UL) ) );
                  xmm1 += a1 * B.load(k    ,j         );
                  xmm2 += a1 * B.load(k    ,j+SIMDSIZE);
                  xmm3 += a2 * B.load(k+1UL,j         );
                  xmm4 += a2 * B.load(k+1UL,j+SIMDSIZE);
               }

               for( ; k<kend; ++k ) {
                  const SIMDType a1( set( A(i,k) ) );
                  xmm1 += a1 * B.load(k,j         );
                  xmm2 += a1 * B.load(k,j+SIMDSIZE);
               }

               (~C).store( i, j         , xmm1+xmm3 );
               (~C).store( i, j+SIMDSIZE, xmm2+xmm4 );
            }
         }

         for( ; j<jpos; j+=SIMDSIZE )
         {
            const size_t iend( M );
            size_t i( 0UL );

            for( ; (i+4UL) <= iend; i+=4UL )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
               size_t k( kbegin );

               for( ; (k+2UL) <= kend; k+=2UL ) {
                  const SIMDType b1( B.load(k    ,j) );
                  const SIMDType b2( B.load(k+1UL,j) );
                  xmm1 += set( A(i    ,k    ) ) * b1;
                  xmm2 += set( A(i+1UL,k    ) ) * b1;
                  xmm3 += set( A(i+2UL,k    ) ) * b1;
                  xmm4 += set( A(i+3UL,k    ) ) * b1;
                  xmm5 += set( A(i    ,k+1UL) ) * b2;
                  xmm6 += set( A(i+1UL,k+1UL) ) * b2;
                  xmm7 += set( A(i+2UL,k+1UL) ) * b2;
                  xmm8 += set( A(i+3UL,k+1UL) ) * b2;
               }

               for( ; k<kend; ++k ) {
                  const SIMDType b1( B.load(k,j) );
                  xmm1 += set( A(i    ,k) ) * b1;
                  xmm2 += set( A(i+1UL,k) ) * b1;
                  xmm3 += set( A(i+2UL,k) ) * b1;
                  xmm4 += set( A(i+3UL,k) ) * b1;
               }

               (~C).store( i    , j, xmm1+xmm5 );
               (~C).store( i+1UL, j, xmm2+xmm6 );
               (~C).store( i+2UL, j, xmm3+xmm7 );
               (~C).store( i+3UL, j, xmm4+xmm8 );
            }

            for( ; (i+3UL) <= iend; i+=3UL )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;
               size_t k( kbegin );

               for( ; (k+2UL) <= kend; k+=2UL ) {
                  const SIMDType b1( B.load(k    ,j) );
                  const SIMDType b2( B.load(k+1UL,j) );
                  xmm1 += set( A(i    ,k    ) ) * b1;
                  xmm2 += set( A(i+1UL,k    ) ) * b1;
                  xmm3 += set( A(i+2UL,k    ) ) * b1;
                  xmm4 += set( A(i    ,k+1UL) ) * b2;
                  xmm5 += set( A(i+1UL,k+1UL) ) * b2;
                  xmm6 += set( A(i+2UL,k+1UL) ) * b2;
               }

               for( ; k<kend; ++k ) {
                  const SIMDType b1( B.load(k,j) );
                  xmm1 += set( A(i    ,k) ) * b1;
                  xmm2 += set( A(i+1UL,k) ) * b1;
                  xmm3 += set( A(i+2UL,k) ) * b1;
               }

               (~C).store( i    , j, xmm1+xmm4 );
               (~C).store( i+1UL, j, xmm2+xmm5 );
               (~C).store( i+2UL, j, xmm3+xmm6 );
            }

            for( ; (i+2UL) <= iend; i+=2UL )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               SIMDType xmm1, xmm2, xmm3, xmm4;
               size_t k( kbegin );

               for( ; (k+2UL) <= kend; k+=2UL ) {
                  const SIMDType b1( B.load(k    ,j) );
                  const SIMDType b2( B.load(k+1UL,j) );
                  xmm1 += set( A(i    ,k    ) ) * b1;
                  xmm2 += set( A(i+1UL,k    ) ) * b1;
                  xmm3 += set( A(i    ,k+1UL) ) * b2;
                  xmm4 += set( A(i+1UL,k+1UL) ) * b2;
               }

               for( ; k<kend; ++k ) {
                  const SIMDType b1( B.load(k,j) );
                  xmm1 += set( A(i    ,k) ) * b1;
                  xmm2 += set( A(i+1UL,k) ) * b1;
               }

               (~C).store( i    , j, xmm1+xmm3 );
               (~C).store( i+1UL, j, xmm2+xmm4 );
            }

            if( i < iend )
            {
               const size_t kbegin( 0UL );

               SIMDType xmm1, xmm2;
               size_t k( kbegin );

               for( ; (k+2UL) <= K; k+=2UL ) {
                  xmm1 += set( A(i,k    ) ) * B.load(k    ,j);
                  xmm2 += set( A(i,k+1UL) ) * B.load(k+1UL,j);
               }

               for( ; k<K; ++k ) {
                  xmm1 += set( A(i,k) ) * B.load(k,j);
               }

               (~C).store( i, j, xmm1+xmm2 );
            }
         }

         for( ; remainder && j<N; ++j )
         {
            size_t i( LOW && UPP ? j : 0UL );

            for( ; (i+2UL) <= M; i+=2UL )
            {
               const size_t kbegin( 0UL );
               const size_t kend( K );

               ElementType value1{};
               ElementType value2{};

               for( size_t k=kbegin; k<kend; ++k ) {
                  value1 += A(i    ,k) * B(k,j);
                  value2 += A(i+1UL,k) * B(k,j);
               }

               (~C)(i    ,j) = value1;
               (~C)(i+1UL,j) = value2;
            }

            if( i < M )
            {
               const size_t kbegin( 0UL );

               ElementType value{};

               for( size_t k=kbegin; k<K; ++k ) {
                  value += A(i,k) * B(k,j);
               }

               (~C)(i,j) = value;
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default assignment to dense tensors (large tensors)***************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a large dense tensor-dense tensor multiplication (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a dense tensor-
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline DisableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5> >
      selectLargeAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectDefaultAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default assignment to dense tensors (large tensors)****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default assignment of a large dense tensor-dense tensor multiplication
   //        (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default assignment of a dense tensor-dense tensor
   // multiplication expression to a dense tensor. This kernel is optimized for large tensors.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5> >
      selectLargeAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      if( SYM )
         smmm( C, A, B, ElementType(1) );
      else if( HERM )
         hmmm( C, A, B, ElementType(1) );
      else if( LOW )
         lmmm( C, A, B, ElementType(1), ElementType(0) );
      else if( UPP )
         ummm( C, A, B, ElementType(1), ElementType(0) );
      else
         mmm( C, A, B, ElementType(1), ElementType(0) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based assignment to dense tensors (default)*******************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a dense tensor-dense tensor multiplication (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a large dense
   // tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline DisableIf_t< UseBlasKernel_v<MT3,MT4,MT5> >
      selectBlasAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectLargeAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based assignment to dense tensors*****************************************************
#if BLAZE_BLAS_MODE && BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION
   /*! \cond BLAZE_INTERNAL */
   /*!\brief BLAS-based assignment of a dense tensor-dense tensor multiplication (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function performs the dense tensor-dense tensor multiplication based on the according
   // BLAS functionality.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< UseBlasKernel_v<MT3,MT4,MT5> >
      selectBlasAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      using ET = ElementType_t<MT3>;

      gemm( C, A, B, ET(1), ET(0) );
   }
   /*! \endcond */
#endif
   //**********************************************************************************************

   //**Addition assignment to dense tensors*******************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a dense tensor-dense tensor multiplication to a dense tensor
   //        (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a dense tensor-
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline void
      addAssign( DenseTensor<MT>& lhs, const DTensDTensMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || rhs.lhs_.columns() == 0UL ) {
         return;
      }

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense tensor operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      DTensDTensMultExpr::selectAddAssignKernel( ~lhs, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense tensors (kernel selection)************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for an addition assignment of a dense tensor-dense tensor
   //        multiplication to a dense tensor (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline void selectAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      if( ( IsDiagonal_v<MT5> ) ||
          ( !BLAZE_DEBUG_MODE && B.columns() <= SIMDSIZE*10UL ) ||
          ( C.rows() * C.columns() < DMATDMATMULT_THRESHOLD ) )
         selectSmallAddAssignKernel( C, A, B );
      else
         selectBlasAddAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense tensors (general/general)*****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a general dense tensor-general dense tensor
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default addition assignment of a general dense tensor-general
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< !IsDiagonal_v<MT4> && !IsDiagonal_v<MT5> >
      selectDefaultAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      BLAZE_INTERNAL_ASSERT( !( LOW || UPP ) || ( M == N ), "Broken invariant detected" );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t kbegin( ( IsUpper_v<MT4> )
                              ?( IsStrictlyUpper_v<MT4> ? i+1UL : i )
                              :( 0UL ) );
         const size_t kend( ( IsLower_v<MT4> )
                            ?( IsStrictlyLower_v<MT4> ? i : i+1UL )
                            :( K ) );
         BLAZE_INTERNAL_ASSERT( kbegin <= kend, "Invalid loop indices detected" );

         for( size_t k=kbegin; k<kend; ++k )
         {
            const size_t jbegin( ( IsUpper_v<MT5> )
                                 ?( ( IsStrictlyUpper_v<MT5> )
                                    ?( UPP ? max(i,k+1UL) : k+1UL )
                                    :( UPP ? max(i,k) : k ) )
                                 :( UPP ? i : 0UL ) );
            const size_t jend( ( IsLower_v<MT5> )
                               ?( ( IsStrictlyLower_v<MT5> )
                                  ?( LOW ? min(i+1UL,k) : k )
                                  :( LOW ? min(i,k)+1UL : k+1UL ) )
                               :( LOW ? i+1UL : N ) );

            if( ( LOW || UPP ) && ( jbegin >= jend ) ) continue;
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jnum( jend - jbegin );
            const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

            for( size_t j=jbegin; j<jpos; j+=2UL ) {
               C(i,j    ) += A(i,k) * B(k,j    );
               C(i,j+1UL) += A(i,k) * B(k,j+1UL);
            }
            if( jpos < jend ) {
               C(i,jpos) += A(i,k) * B(k,jpos);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense tensors (general/diagonal)****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a general dense tensor-diagonal dense tensor
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default addition assignment of a general dense tensor-diagonal
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< !IsDiagonal_v<MT4> && IsDiagonal_v<MT5> >
      selectDefaultAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper_v<MT4> )
                              ?( IsStrictlyUpper_v<MT4> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower_v<MT4> )
                            ?( IsStrictlyLower_v<MT4> ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jnum( jend - jbegin );
         const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

         for( size_t j=jbegin; j<jpos; j+=2UL ) {
            C(i,j    ) += A(i,j    ) * B(j    ,j    );
            C(i,j+1UL) += A(i,j+1UL) * B(j+1UL,j+1UL);
         }
         if( jpos < jend ) {
            C(i,jpos) += A(i,jpos) * B(jpos,jpos);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense tensors (diagonal/general)****************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a diagonal dense tensor-general dense tensor
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default addition assignment of a diagonal dense tensor-general
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< IsDiagonal_v<MT4> && !IsDiagonal_v<MT5> >
      selectDefaultAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper_v<MT5> )
                              ?( IsStrictlyUpper_v<MT5> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower_v<MT5> )
                            ?( IsStrictlyLower_v<MT5> ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jnum( jend - jbegin );
         const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

         for( size_t j=jbegin; j<jpos; j+=2UL ) {
            C(i,j    ) += A(i,i) * B(i,j    );
            C(i,j+1UL) += A(i,i) * B(i,j+1UL);
         }
         if( jpos < jend ) {
            C(i,jpos) += A(i,i) * B(i,jpos);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense tensors (diagonal/diagonal)***************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a diagonal dense tensor-diagonal dense tensor
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default addition assignment of a diagonal dense tensor-diagonal
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< IsDiagonal_v<MT4> && IsDiagonal_v<MT5> >
      selectDefaultAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      for( size_t i=0UL; i<A.rows(); ++i ) {
         C(i,i) += A(i,i) * B(i,i);
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense tensors (small tensors)******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a small dense tensor-dense tensor multiplication
   //        (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a dense
   // tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline DisableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5> >
      selectSmallAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectDefaultAddAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default addition assignment to row-major dense tensors (small tensors)*********
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default addition assignment of a small dense tensor-dense tensor
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default addition assignment of a dense tensor-dense
   // tensor multiplication expression to a row-major dense tensor. This kernel is optimized for
   // small tensors.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5> >
      selectSmallAddAssignKernel( DenseTensor<MT3>& C, const MT4& A, const MT5& B )
   {
      constexpr bool remainder( !IsPadded_v<MT3> || !IsPadded_v<MT5> );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      BLAZE_INTERNAL_ASSERT( !( LOW || UPP ) || ( M == N ), "Broken invariant detected" );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );

      if( IsIntegral_v<ElementType> )
      {
         for( ; !LOW && !UPP && (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL ) {
            for( size_t i=0UL; i<M; ++i )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( ( IsUpper_v<MT5> )
                                     ?( min( ( IsStrictlyLower_v<MT4> ? i : i+1UL ), j+SIMDSIZE*8UL, K ) )
                                     :( IsStrictlyLower_v<MT4> ? i : i+1UL ) )
                                  :( IsUpper_v<MT5> ? min( j+SIMDSIZE*8UL, K ) : K ) );

               SIMDType xmm1( (~C).load(i,j             ) );
               SIMDType xmm2( (~C).load(i,j+SIMDSIZE    ) );
               SIMDType xmm3( (~C).load(i,j+SIMDSIZE*2UL) );
               SIMDType xmm4( (~C).load(i,j+SIMDSIZE*3UL) );
               SIMDType xmm5( (~C).load(i,j+SIMDSIZE*4UL) );
               SIMDType xmm6( (~C).load(i,j+SIMDSIZE*5UL) );
               SIMDType xmm7( (~C).load(i,j+SIMDSIZE*6UL) );
               SIMDType xmm8( (~C).load(i,j+SIMDSIZE*7UL) );

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i,k) ) );
                  xmm1 += a1 * B.load(k,j             );
                  xmm2 += a1 * B.load(k,j+SIMDSIZE    );
                  xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
                  xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
                  xmm5 += a1 * B.load(k,j+SIMDSIZE*4UL);
                  xmm6 += a1 * B.load(k,j+SIMDSIZE*5UL);
                  xmm7 += a1 * B.load(k,j+SIMDSIZE*6UL);
                  xmm8 += a1 * B.load(k,j+SIMDSIZE*7UL);
               }

               (~C).store( i, j             , xmm1 );
               (~C).store( i, j+SIMDSIZE    , xmm2 );
               (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
               (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
               (~C).store( i, j+SIMDSIZE*4UL, xmm5 );
               (~C).store( i, j+SIMDSIZE*5UL, xmm6 );
               (~C).store( i, j+SIMDSIZE*6UL, xmm7 );
               (~C).store( i, j+SIMDSIZE*7UL, xmm8 );
            }
         }
      }

      for( ; !LOW && !UPP && (j+SIMDSIZE*4UL) < jpos; j+=SIMDSIZE*5UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*5UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*5UL, K ) : K ) );

            SIMDType xmm1 ( (~C).load(i    ,j             ) );
            SIMDType xmm2 ( (~C).load(i    ,j+SIMDSIZE    ) );
            SIMDType xmm3 ( (~C).load(i    ,j+SIMDSIZE*2UL) );
            SIMDType xmm4 ( (~C).load(i    ,j+SIMDSIZE*3UL) );
            SIMDType xmm5 ( (~C).load(i    ,j+SIMDSIZE*4UL) );
            SIMDType xmm6 ( (~C).load(i+1UL,j             ) );
            SIMDType xmm7 ( (~C).load(i+1UL,j+SIMDSIZE    ) );
            SIMDType xmm8 ( (~C).load(i+1UL,j+SIMDSIZE*2UL) );
            SIMDType xmm9 ( (~C).load(i+1UL,j+SIMDSIZE*3UL) );
            SIMDType xmm10( (~C).load(i+1UL,j+SIMDSIZE*4UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               const SIMDType b5( B.load(k,j+SIMDSIZE*4UL) );
               xmm1  += a1 * b1;
               xmm2  += a1 * b2;
               xmm3  += a1 * b3;
               xmm4  += a1 * b4;
               xmm5  += a1 * b5;
               xmm6  += a2 * b1;
               xmm7  += a2 * b2;
               xmm8  += a2 * b3;
               xmm9  += a2 * b4;
               xmm10 += a2 * b5;
            }

            (~C).store( i    , j             , xmm1  );
            (~C).store( i    , j+SIMDSIZE    , xmm2  );
            (~C).store( i    , j+SIMDSIZE*2UL, xmm3  );
            (~C).store( i    , j+SIMDSIZE*3UL, xmm4  );
            (~C).store( i    , j+SIMDSIZE*4UL, xmm5  );
            (~C).store( i+1UL, j             , xmm6  );
            (~C).store( i+1UL, j+SIMDSIZE    , xmm7  );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm8  );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, xmm9  );
            (~C).store( i+1UL, j+SIMDSIZE*4UL, xmm10 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*5UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i,j             ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i,j+SIMDSIZE*3UL) );
            SIMDType xmm5( (~C).load(i,j+SIMDSIZE*4UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 += a1 * B.load(k,j             );
               xmm2 += a1 * B.load(k,j+SIMDSIZE    );
               xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
               xmm5 += a1 * B.load(k,j+SIMDSIZE*4UL);
            }

            (~C).store( i, j             , xmm1 );
            (~C).store( i, j+SIMDSIZE    , xmm2 );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
            (~C).store( i, j+SIMDSIZE*4UL, xmm5 );
         }
      }

      for( ; !LOW && !UPP && (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*4UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j             ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i    ,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i    ,j+SIMDSIZE*3UL) );
            SIMDType xmm5( (~C).load(i+1UL,j             ) );
            SIMDType xmm6( (~C).load(i+1UL,j+SIMDSIZE    ) );
            SIMDType xmm7( (~C).load(i+1UL,j+SIMDSIZE*2UL) );
            SIMDType xmm8( (~C).load(i+1UL,j+SIMDSIZE*3UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a1 * b3;
               xmm4 += a1 * b4;
               xmm5 += a2 * b1;
               xmm6 += a2 * b2;
               xmm7 += a2 * b3;
               xmm8 += a2 * b4;
            }

            (~C).store( i    , j             , xmm1 );
            (~C).store( i    , j+SIMDSIZE    , xmm2 );
            (~C).store( i    , j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i    , j+SIMDSIZE*3UL, xmm4 );
            (~C).store( i+1UL, j             , xmm5 );
            (~C).store( i+1UL, j+SIMDSIZE    , xmm6 );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm7 );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, xmm8 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i,j             ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i,j+SIMDSIZE*3UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 += a1 * B.load(k,j             );
               xmm2 += a1 * B.load(k,j+SIMDSIZE    );
               xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
            }

            (~C).store( i, j             , xmm1 );
            (~C).store( i, j+SIMDSIZE    , xmm2 );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
         }
      }

      for( ; !LOW && !UPP && (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*3UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*3UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j             ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i    ,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i+1UL,j             ) );
            SIMDType xmm5( (~C).load(i+1UL,j+SIMDSIZE    ) );
            SIMDType xmm6( (~C).load(i+1UL,j+SIMDSIZE*2UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a1 * b3;
               xmm4 += a2 * b1;
               xmm5 += a2 * b2;
               xmm6 += a2 * b3;
            }

            (~C).store( i    , j             , xmm1 );
            (~C).store( i    , j+SIMDSIZE    , xmm2 );
            (~C).store( i    , j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i+1UL, j             , xmm4 );
            (~C).store( i+1UL, j+SIMDSIZE    , xmm5 );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm6 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*3UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i,j             ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i,j+SIMDSIZE*2UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 += a1 * B.load(k,j             );
               xmm2 += a1 * B.load(k,j+SIMDSIZE    );
               xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
            }

            (~C).store( i, j             , xmm1 );
            (~C).store( i, j+SIMDSIZE    , xmm2 );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
         }
      }

      for( ; !( LOW && UPP ) && (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         const size_t iend( UPP ? min(j+SIMDSIZE*2UL,M) : M );
         size_t i( LOW ? j : 0UL );

         for( ; (i+4UL) <= iend; i+=4UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j         ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE) );
            SIMDType xmm3( (~C).load(i+1UL,j         ) );
            SIMDType xmm4( (~C).load(i+1UL,j+SIMDSIZE) );
            SIMDType xmm5( (~C).load(i+2UL,j         ) );
            SIMDType xmm6( (~C).load(i+2UL,j+SIMDSIZE) );
            SIMDType xmm7( (~C).load(i+3UL,j         ) );
            SIMDType xmm8( (~C).load(i+3UL,j+SIMDSIZE) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType a3( set( A(i+2UL,k) ) );
               const SIMDType a4( set( A(i+3UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a2 * b1;
               xmm4 += a2 * b2;
               xmm5 += a3 * b1;
               xmm6 += a3 * b2;
               xmm7 += a4 * b1;
               xmm8 += a4 * b2;
            }

            (~C).store( i    , j         , xmm1 );
            (~C).store( i    , j+SIMDSIZE, xmm2 );
            (~C).store( i+1UL, j         , xmm3 );
            (~C).store( i+1UL, j+SIMDSIZE, xmm4 );
            (~C).store( i+2UL, j         , xmm5 );
            (~C).store( i+2UL, j+SIMDSIZE, xmm6 );
            (~C).store( i+3UL, j         , xmm7 );
            (~C).store( i+3UL, j+SIMDSIZE, xmm8 );
         }

         for( ; (i+3UL) <= iend; i+=3UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j         ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE) );
            SIMDType xmm3( (~C).load(i+1UL,j         ) );
            SIMDType xmm4( (~C).load(i+1UL,j+SIMDSIZE) );
            SIMDType xmm5( (~C).load(i+2UL,j         ) );
            SIMDType xmm6( (~C).load(i+2UL,j+SIMDSIZE) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType a3( set( A(i+2UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a2 * b1;
               xmm4 += a2 * b2;
               xmm5 += a3 * b1;
               xmm6 += a3 * b2;
            }

            (~C).store( i    , j         , xmm1 );
            (~C).store( i    , j+SIMDSIZE, xmm2 );
            (~C).store( i+1UL, j         , xmm3 );
            (~C).store( i+1UL, j+SIMDSIZE, xmm4 );
            (~C).store( i+2UL, j         , xmm5 );
            (~C).store( i+2UL, j+SIMDSIZE, xmm6 );
         }

         for( ; (i+2UL) <= iend; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j         ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE) );
            SIMDType xmm3( (~C).load(i+1UL,j         ) );
            SIMDType xmm4( (~C).load(i+1UL,j+SIMDSIZE) );
            SIMDType xmm5, xmm6, xmm7, xmm8;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType a1( set( A(i    ,k    ) ) );
               const SIMDType a2( set( A(i+1UL,k    ) ) );
               const SIMDType a3( set( A(i    ,k+1UL) ) );
               const SIMDType a4( set( A(i+1UL,k+1UL) ) );
               const SIMDType b1( B.load(k    ,j         ) );
               const SIMDType b2( B.load(k    ,j+SIMDSIZE) );
               const SIMDType b3( B.load(k+1UL,j         ) );
               const SIMDType b4( B.load(k+1UL,j+SIMDSIZE) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a2 * b1;
               xmm4 += a2 * b2;
               xmm5 += a3 * b3;
               xmm6 += a3 * b4;
               xmm7 += a4 * b3;
               xmm8 += a4 * b4;
            }

            for( ; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a2 * b1;
               xmm4 += a2 * b2;
            }

            (~C).store( i    , j         , xmm1+xmm5 );
            (~C).store( i    , j+SIMDSIZE, xmm2+xmm6 );
            (~C).store( i+1UL, j         , xmm3+xmm7 );
            (~C).store( i+1UL, j+SIMDSIZE, xmm4+xmm8 );
         }

         if( i < iend )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i,j         ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE) );
            SIMDType xmm3, xmm4;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType a1( set( A(i,k    ) ) );
               const SIMDType a2( set( A(i,k+1UL) ) );
               xmm1 += a1 * B.load(k    ,j         );
               xmm2 += a1 * B.load(k    ,j+SIMDSIZE);
               xmm3 += a2 * B.load(k+1UL,j         );
               xmm4 += a2 * B.load(k+1UL,j+SIMDSIZE);
            }

            for( ; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 += a1 * B.load(k,j         );
               xmm2 += a1 * B.load(k,j+SIMDSIZE);
            }

            (~C).store( i, j         , xmm1+xmm3 );
            (~C).store( i, j+SIMDSIZE, xmm2+xmm4 );
         }
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         const size_t iend( LOW && UPP ? min(j+SIMDSIZE,M) : M );
         size_t i( LOW ? j : 0UL );

         for( ; (i+4UL) <= iend; i+=4UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL )
                               :( K ) );

            SIMDType xmm1( (~C).load(i    ,j) );
            SIMDType xmm2( (~C).load(i+1UL,j) );
            SIMDType xmm3( (~C).load(i+2UL,j) );
            SIMDType xmm4( (~C).load(i+3UL,j) );
            SIMDType xmm5, xmm6, xmm7, xmm8;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType b1( B.load(k    ,j) );
               const SIMDType b2( B.load(k+1UL,j) );
               xmm1 += set( A(i    ,k    ) ) * b1;
               xmm2 += set( A(i+1UL,k    ) ) * b1;
               xmm3 += set( A(i+2UL,k    ) ) * b1;
               xmm4 += set( A(i+3UL,k    ) ) * b1;
               xmm5 += set( A(i    ,k+1UL) ) * b2;
               xmm6 += set( A(i+1UL,k+1UL) ) * b2;
               xmm7 += set( A(i+2UL,k+1UL) ) * b2;
               xmm8 += set( A(i+3UL,k+1UL) ) * b2;
            }

            for( ; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 += set( A(i    ,k) ) * b1;
               xmm2 += set( A(i+1UL,k) ) * b1;
               xmm3 += set( A(i+2UL,k) ) * b1;
               xmm4 += set( A(i+3UL,k) ) * b1;
            }

            (~C).store( i    , j, xmm1+xmm5 );
            (~C).store( i+1UL, j, xmm2+xmm6 );
            (~C).store( i+2UL, j, xmm3+xmm7 );
            (~C).store( i+3UL, j, xmm4+xmm8 );
         }

         for( ; (i+3UL) <= iend; i+=3UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL )
                               :( K ) );

            SIMDType xmm1( (~C).load(i    ,j) );
            SIMDType xmm2( (~C).load(i+1UL,j) );
            SIMDType xmm3( (~C).load(i+2UL,j) );
            SIMDType xmm4, xmm5, xmm6;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType b1( B.load(k    ,j) );
               const SIMDType b2( B.load(k+1UL,j) );
               xmm1 += set( A(i    ,k    ) ) * b1;
               xmm2 += set( A(i+1UL,k    ) ) * b1;
               xmm3 += set( A(i+2UL,k    ) ) * b1;
               xmm4 += set( A(i    ,k+1UL) ) * b2;
               xmm5 += set( A(i+1UL,k+1UL) ) * b2;
               xmm6 += set( A(i+2UL,k+1UL) ) * b2;
            }

            for( ; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 += set( A(i    ,k) ) * b1;
               xmm2 += set( A(i+1UL,k) ) * b1;
               xmm3 += set( A(i+2UL,k) ) * b1;
            }

            (~C).store( i    , j, xmm1+xmm4 );
            (~C).store( i+1UL, j, xmm2+xmm5 );
            (~C).store( i+2UL, j, xmm3+xmm6 );
         }

         for( ; (i+2UL) <= iend; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL )
                               :( K ) );

            SIMDType xmm1( (~C).load(i    ,j) );
            SIMDType xmm2( (~C).load(i+1UL,j) );
            SIMDType xmm3, xmm4;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType b1( B.load(k    ,j) );
               const SIMDType b2( B.load(k+1UL,j) );
               xmm1 += set( A(i    ,k    ) ) * b1;
               xmm2 += set( A(i+1UL,k    ) ) * b1;
               xmm3 += set( A(i    ,k+1UL) ) * b2;
               xmm4 += set( A(i+1UL,k+1UL) ) * b2;
            }

            for( ; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 += set( A(i    ,k) ) * b1;
               xmm2 += set( A(i+1UL,k) ) * b1;
            }

            (~C).store( i    , j, xmm1+xmm3 );
            (~C).store( i+1UL, j, xmm2+xmm4 );
         }

         if( i < iend )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );

            SIMDType xmm1( (~C).load(i,j) );
            SIMDType xmm2;
            size_t k( kbegin );

            for( ; (k+2UL) <= K; k+=2UL ) {
               xmm1 += set( A(i,k    ) ) * B.load(k    ,j);
               xmm2 += set( A(i,k+1UL) ) * B.load(k+1UL,j);
            }

            for( ; k<K; ++k ) {
               xmm1 += set( A(i,k) ) * B.load(k,j);
            }

            (~C).store( i, j, xmm1+xmm2 );
         }
      }

      for( ; remainder && j<N; ++j )
      {
         const size_t iend( UPP ? j+1UL : M );
         size_t i( LOW ? j : 0UL );

         for( ; (i+2UL) <= iend; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL )
                               :( K ) );

            ElementType value1( (~C)(i    ,j) );
            ElementType value2( (~C)(i+1UL,j) );;

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i    ,k) * B(k,j);
               value2 += A(i+1UL,k) * B(k,j);
            }

            (~C)(i    ,j) = value1;
            (~C)(i+1UL,j) = value2;
         }

         if( i < iend )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );

            ElementType value( (~C)(i,j) );

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) = value;
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default addition assignment to dense tensors (large tensors)******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a large dense tensor-dense tensor multiplication
   //        (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a dense
   // tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline DisableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5> >
      selectLargeAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectDefaultAddAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default addition assignment to dense tensors (large tensors)*******************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default addition assignment of a large dense tensor-dense tensor
   //        multiplication (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default addition assignment of a dense tensor-
   // dense tensor multiplication expression to a dense tensor. This kernel is optimized for
   // large tensors.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5> >
      selectLargeAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      if( LOW )
         lmmm( C, A, B, ElementType(1), ElementType(1) );
      else if( UPP )
         ummm( C, A, B, ElementType(1), ElementType(1) );
      else
         mmm( C, A, B, ElementType(1), ElementType(1) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense tensors (default)**********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default addition assignment of a dense tensor-dense tensor multiplication
   //        (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a large
   // dense tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline DisableIf_t< UseBlasKernel_v<MT3,MT4,MT5> >
      selectBlasAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectLargeAddAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense tensors********************************************
#if BLAZE_BLAS_MODE && BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION
   /*! \cond BLAZE_INTERNAL */
   /*!\brief BLAS-based addition assignment of a dense tensor-dense tensor multiplication
   //        (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function performs the dense tensor-dense tensor multiplication based on the according
   // BLAS functionality.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< UseBlasKernel_v<MT3,MT4,MT5> >
      selectBlasAddAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      using ET = ElementType_t<MT3>;

      if( IsTriangular_v<MT4> ) {
         ResultType_t<MT3> tmp( serial( B ) );
         trmm( tmp, A, CblasLeft, ( IsLower_v<MT4> )?( CblasLower ):( CblasUpper ), ET(1) );
         addAssign( C, tmp );
      }
      else if( IsTriangular_v<MT5> ) {
         ResultType_t<MT3> tmp( serial( A ) );
         trmm( tmp, B, CblasRight, ( IsLower_v<MT5> )?( CblasLower ):( CblasUpper ), ET(1) );
         addAssign( C, tmp );
      }
      else {
         gemm( C, A, B, ET(1), ET(1) );
      }
   }
   /*! \endcond */
#endif
   //**********************************************************************************************

   //**Subtraction assignment to dense tensors****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a dense tensor-dense tensor multiplication to a
   //        dense tensor (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a dense tensor-
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline void
      subAssign( DenseTensor<MT>& lhs, const DTensDTensMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || rhs.lhs_.columns() == 0UL ) {
         return;
      }

      LT A( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side dense tensor operand
      RT B( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      DTensDTensMultExpr::selectSubAssignKernel( ~lhs, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to dense tensors (kernel selection)*********************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Selection of the kernel for a subtraction assignment of a dense tensor-dense tensor
   //        multiplication to a dense tensor (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline void selectSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      if( ( IsDiagonal_v<MT5> ) ||
          ( !BLAZE_DEBUG_MODE && B.columns() <= SIMDSIZE*10UL ) ||
          ( C.rows() * C.columns() < DMATDMATMULT_THRESHOLD ) )
         selectSmallSubAssignKernel( C, A, B );
      else
         selectBlasSubAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense tensors (general/general)**************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a general dense tensor-general dense tensor
   //        multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default subtraction assignment of a general dense tensor-
   // general dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< !IsDiagonal_v<MT4> && !IsDiagonal_v<MT5> >
      selectDefaultSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      BLAZE_INTERNAL_ASSERT( !( LOW || UPP ) || ( M == N ), "Broken invariant detected" );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t kbegin( ( IsUpper_v<MT4> )
                              ?( IsStrictlyUpper_v<MT4> ? i+1UL : i )
                              :( 0UL ) );
         const size_t kend( ( IsLower_v<MT4> )
                            ?( IsStrictlyLower_v<MT4> ? i : i+1UL )
                            :( K ) );
         BLAZE_INTERNAL_ASSERT( kbegin <= kend, "Invalid loop indices detected" );

         for( size_t k=kbegin; k<kend; ++k )
         {
            const size_t jbegin( ( IsUpper_v<MT5> )
                                 ?( ( IsStrictlyUpper_v<MT5> )
                                    ?( UPP ? max(i,k+1UL) : k+1UL )
                                    :( UPP ? max(i,k) : k ) )
                                 :( UPP ? i : 0UL ) );
            const size_t jend( ( IsLower_v<MT5> )
                               ?( ( IsStrictlyLower_v<MT5> )
                                  ?( LOW ? min(i+1UL,k) : k )
                                  :( LOW ? min(i,k)+1UL : k+1UL ) )
                               :( LOW ? i+1UL : N ) );

            if( ( LOW || UPP ) && ( jbegin >= jend ) ) continue;
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            const size_t jnum( jend - jbegin );
            const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

            for( size_t j=jbegin; j<jpos; j+=2UL ) {
               C(i,j    ) -= A(i,k) * B(k,j    );
               C(i,j+1UL) -= A(i,k) * B(k,j+1UL);
            }
            if( jpos < jend ) {
               C(i,jpos) -= A(i,k) * B(k,jpos);
            }
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense tensors (general/diagonal)*************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a general dense tensor-diagonal dense tensor
   //        multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default subtraction assignment of a general dense tensor-
   // diagonal dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< !IsDiagonal_v<MT4> && IsDiagonal_v<MT5> >
      selectDefaultSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper_v<MT4> )
                              ?( IsStrictlyUpper_v<MT4> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower_v<MT4> )
                            ?( IsStrictlyLower_v<MT4> ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jnum( jend - jbegin );
         const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

         for( size_t j=jbegin; j<jpos; j+=2UL ) {
            C(i,j    ) -= A(i,j    ) * B(j    ,j    );
            C(i,j+1UL) -= A(i,j+1UL) * B(j+1UL,j+1UL);
         }
         if( jpos < jend ) {
            C(i,jpos) -= A(i,jpos) * B(jpos,jpos);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense tensors (diagonal/general)*************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a diagonal dense tensor-general dense tensor
   //        multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default subtraction assignment of a diagonal dense tensor-
   // general dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< IsDiagonal_v<MT4> && !IsDiagonal_v<MT5> >
      selectDefaultSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper_v<MT5> )
                              ?( IsStrictlyUpper_v<MT5> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower_v<MT5> )
                            ?( IsStrictlyLower_v<MT5> ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jnum( jend - jbegin );
         const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

         for( size_t j=jbegin; j<jpos; j+=2UL ) {
            C(i,j    ) -= A(i,i) * B(i,j    );
            C(i,j+1UL) -= A(i,i) * B(i,j+1UL);
         }
         if( jpos < jend ) {
            C(i,jpos) -= A(i,i) * B(i,jpos);
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense tensors (diagonal/diagonal)************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a diagonal dense tensor-diagonal dense tensor
   //        multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the default subtraction assignment of a diagonal dense tensor-
   // diagonal dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< IsDiagonal_v<MT4> && IsDiagonal_v<MT5> >
      selectDefaultSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      for( size_t i=0UL; i<A.rows(); ++i ) {
         C(i,i) -= A(i,i) * B(i,i);
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense tensors (small tensors)***************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a small dense tensor-dense tensor multiplication
   //        (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a dense
   // tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline DisableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5> >
      selectSmallSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectDefaultSubAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to row-major dense tensors (small tensors)******
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default subtraction assignment of a small dense tensor-dense tensor
   //        multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment of a dense tensor-
   // dense tensor multiplication expression to a row-major dense tensor. This kernel is optimized
   // for small tensors.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5> >
      selectSmallSubAssignKernel( DenseTensor<MT3>& C, const MT4& A, const MT5& B )
   {
      constexpr bool remainder( !IsPadded_v<MT3> || !IsPadded_v<MT5> );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      BLAZE_INTERNAL_ASSERT( !( LOW || UPP ) || ( M == N ), "Broken invariant detected" );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      size_t j( 0UL );

      if( IsIntegral_v<ElementType> )
      {
         for( ; !LOW && !UPP && (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL ) {
            for( size_t i=0UL; i<M; ++i )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( ( IsUpper_v<MT5> )
                                     ?( min( ( IsStrictlyLower_v<MT4> ? i : i+1UL ), j+SIMDSIZE*8UL, K ) )
                                     :( IsStrictlyLower_v<MT4> ? i : i+1UL ) )
                                  :( IsUpper_v<MT5> ? min( j+SIMDSIZE*8UL, K ) : K ) );

               SIMDType xmm1( (~C).load(i,j             ) );
               SIMDType xmm2( (~C).load(i,j+SIMDSIZE    ) );
               SIMDType xmm3( (~C).load(i,j+SIMDSIZE*2UL) );
               SIMDType xmm4( (~C).load(i,j+SIMDSIZE*3UL) );
               SIMDType xmm5( (~C).load(i,j+SIMDSIZE*4UL) );
               SIMDType xmm6( (~C).load(i,j+SIMDSIZE*5UL) );
               SIMDType xmm7( (~C).load(i,j+SIMDSIZE*6UL) );
               SIMDType xmm8( (~C).load(i,j+SIMDSIZE*7UL) );

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i,k) ) );
                  xmm1 -= a1 * B.load(k,j             );
                  xmm2 -= a1 * B.load(k,j+SIMDSIZE    );
                  xmm3 -= a1 * B.load(k,j+SIMDSIZE*2UL);
                  xmm4 -= a1 * B.load(k,j+SIMDSIZE*3UL);
                  xmm5 -= a1 * B.load(k,j+SIMDSIZE*4UL);
                  xmm6 -= a1 * B.load(k,j+SIMDSIZE*5UL);
                  xmm7 -= a1 * B.load(k,j+SIMDSIZE*6UL);
                  xmm8 -= a1 * B.load(k,j+SIMDSIZE*7UL);
               }

               (~C).store( i, j             , xmm1 );
               (~C).store( i, j+SIMDSIZE    , xmm2 );
               (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
               (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
               (~C).store( i, j+SIMDSIZE*4UL, xmm5 );
               (~C).store( i, j+SIMDSIZE*5UL, xmm6 );
               (~C).store( i, j+SIMDSIZE*6UL, xmm7 );
               (~C).store( i, j+SIMDSIZE*7UL, xmm8 );
            }
         }
      }

      for( ; !LOW && !UPP && (j+SIMDSIZE*4UL) < jpos; j+=SIMDSIZE*5UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*5UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*5UL, K ) : K ) );

            SIMDType xmm1 ( (~C).load(i    ,j             ) );
            SIMDType xmm2 ( (~C).load(i    ,j+SIMDSIZE    ) );
            SIMDType xmm3 ( (~C).load(i    ,j+SIMDSIZE*2UL) );
            SIMDType xmm4 ( (~C).load(i    ,j+SIMDSIZE*3UL) );
            SIMDType xmm5 ( (~C).load(i    ,j+SIMDSIZE*4UL) );
            SIMDType xmm6 ( (~C).load(i+1UL,j             ) );
            SIMDType xmm7 ( (~C).load(i+1UL,j+SIMDSIZE    ) );
            SIMDType xmm8 ( (~C).load(i+1UL,j+SIMDSIZE*2UL) );
            SIMDType xmm9 ( (~C).load(i+1UL,j+SIMDSIZE*3UL) );
            SIMDType xmm10( (~C).load(i+1UL,j+SIMDSIZE*4UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               const SIMDType b5( B.load(k,j+SIMDSIZE*4UL) );
               xmm1  -= a1 * b1;
               xmm2  -= a1 * b2;
               xmm3  -= a1 * b3;
               xmm4  -= a1 * b4;
               xmm5  -= a1 * b5;
               xmm6  -= a2 * b1;
               xmm7  -= a2 * b2;
               xmm8  -= a2 * b3;
               xmm9  -= a2 * b4;
               xmm10 -= a2 * b5;
            }

            (~C).store( i    , j             , xmm1  );
            (~C).store( i    , j+SIMDSIZE    , xmm2  );
            (~C).store( i    , j+SIMDSIZE*2UL, xmm3  );
            (~C).store( i    , j+SIMDSIZE*3UL, xmm4  );
            (~C).store( i    , j+SIMDSIZE*4UL, xmm5  );
            (~C).store( i+1UL, j             , xmm6  );
            (~C).store( i+1UL, j+SIMDSIZE    , xmm7  );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm8  );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, xmm9  );
            (~C).store( i+1UL, j+SIMDSIZE*4UL, xmm10 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*5UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i,j             ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i,j+SIMDSIZE*3UL) );
            SIMDType xmm5( (~C).load(i,j+SIMDSIZE*4UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 -= a1 * B.load(k,j             );
               xmm2 -= a1 * B.load(k,j+SIMDSIZE    );
               xmm3 -= a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 -= a1 * B.load(k,j+SIMDSIZE*3UL);
               xmm5 -= a1 * B.load(k,j+SIMDSIZE*4UL);
            }

            (~C).store( i, j             , xmm1 );
            (~C).store( i, j+SIMDSIZE    , xmm2 );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
            (~C).store( i, j+SIMDSIZE*4UL, xmm5 );
         }
      }

      for( ; !LOW && !UPP && (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*4UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j             ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i    ,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i    ,j+SIMDSIZE*3UL) );
            SIMDType xmm5( (~C).load(i+1UL,j             ) );
            SIMDType xmm6( (~C).load(i+1UL,j+SIMDSIZE    ) );
            SIMDType xmm7( (~C).load(i+1UL,j+SIMDSIZE*2UL) );
            SIMDType xmm8( (~C).load(i+1UL,j+SIMDSIZE*3UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               xmm1 -= a1 * b1;
               xmm2 -= a1 * b2;
               xmm3 -= a1 * b3;
               xmm4 -= a1 * b4;
               xmm5 -= a2 * b1;
               xmm6 -= a2 * b2;
               xmm7 -= a2 * b3;
               xmm8 -= a2 * b4;
            }

            (~C).store( i    , j             , xmm1 );
            (~C).store( i    , j+SIMDSIZE    , xmm2 );
            (~C).store( i    , j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i    , j+SIMDSIZE*3UL, xmm4 );
            (~C).store( i+1UL, j             , xmm5 );
            (~C).store( i+1UL, j+SIMDSIZE    , xmm6 );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm7 );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, xmm8 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i,j             ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i,j+SIMDSIZE*3UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 -= a1 * B.load(k,j             );
               xmm2 -= a1 * B.load(k,j+SIMDSIZE    );
               xmm3 -= a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 -= a1 * B.load(k,j+SIMDSIZE*3UL);
            }

            (~C).store( i, j             , xmm1 );
            (~C).store( i, j+SIMDSIZE    , xmm2 );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i, j+SIMDSIZE*3UL, xmm4 );
         }
      }

      for( ; !LOW && !UPP && (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*3UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*3UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j             ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i    ,j+SIMDSIZE*2UL) );
            SIMDType xmm4( (~C).load(i+1UL,j             ) );
            SIMDType xmm5( (~C).load(i+1UL,j+SIMDSIZE    ) );
            SIMDType xmm6( (~C).load(i+1UL,j+SIMDSIZE*2UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               xmm1 -= a1 * b1;
               xmm2 -= a1 * b2;
               xmm3 -= a1 * b3;
               xmm4 -= a2 * b1;
               xmm5 -= a2 * b2;
               xmm6 -= a2 * b3;
            }

            (~C).store( i    , j             , xmm1 );
            (~C).store( i    , j+SIMDSIZE    , xmm2 );
            (~C).store( i    , j+SIMDSIZE*2UL, xmm3 );
            (~C).store( i+1UL, j             , xmm4 );
            (~C).store( i+1UL, j+SIMDSIZE    , xmm5 );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm6 );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*3UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i,j             ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE    ) );
            SIMDType xmm3( (~C).load(i,j+SIMDSIZE*2UL) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 -= a1 * B.load(k,j             );
               xmm2 -= a1 * B.load(k,j+SIMDSIZE    );
               xmm3 -= a1 * B.load(k,j+SIMDSIZE*2UL);
            }

            (~C).store( i, j             , xmm1 );
            (~C).store( i, j+SIMDSIZE    , xmm2 );
            (~C).store( i, j+SIMDSIZE*2UL, xmm3 );
         }
      }

      for( ; !( LOW && UPP ) && (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         const size_t iend( UPP ? min(j+SIMDSIZE*2UL,M) : M );
         size_t i( LOW ? j : 0UL );

         for( ; (i+4UL) <= iend; i+=4UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j         ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE) );
            SIMDType xmm3( (~C).load(i+1UL,j         ) );
            SIMDType xmm4( (~C).load(i+1UL,j+SIMDSIZE) );
            SIMDType xmm5( (~C).load(i+2UL,j         ) );
            SIMDType xmm6( (~C).load(i+2UL,j+SIMDSIZE) );
            SIMDType xmm7( (~C).load(i+3UL,j         ) );
            SIMDType xmm8( (~C).load(i+3UL,j+SIMDSIZE) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType a3( set( A(i+2UL,k) ) );
               const SIMDType a4( set( A(i+3UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 -= a1 * b1;
               xmm2 -= a1 * b2;
               xmm3 -= a2 * b1;
               xmm4 -= a2 * b2;
               xmm5 -= a3 * b1;
               xmm6 -= a3 * b2;
               xmm7 -= a4 * b1;
               xmm8 -= a4 * b2;
            }

            (~C).store( i    , j         , xmm1 );
            (~C).store( i    , j+SIMDSIZE, xmm2 );
            (~C).store( i+1UL, j         , xmm3 );
            (~C).store( i+1UL, j+SIMDSIZE, xmm4 );
            (~C).store( i+2UL, j         , xmm5 );
            (~C).store( i+2UL, j+SIMDSIZE, xmm6 );
            (~C).store( i+3UL, j         , xmm7 );
            (~C).store( i+3UL, j+SIMDSIZE, xmm8 );
         }

         for( ; (i+3UL) <= iend; i+=3UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j         ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE) );
            SIMDType xmm3( (~C).load(i+1UL,j         ) );
            SIMDType xmm4( (~C).load(i+1UL,j+SIMDSIZE) );
            SIMDType xmm5( (~C).load(i+2UL,j         ) );
            SIMDType xmm6( (~C).load(i+2UL,j+SIMDSIZE) );

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType a3( set( A(i+2UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 -= a1 * b1;
               xmm2 -= a1 * b2;
               xmm3 -= a2 * b1;
               xmm4 -= a2 * b2;
               xmm5 -= a3 * b1;
               xmm6 -= a3 * b2;
            }

            (~C).store( i    , j         , xmm1 );
            (~C).store( i    , j+SIMDSIZE, xmm2 );
            (~C).store( i+1UL, j         , xmm3 );
            (~C).store( i+1UL, j+SIMDSIZE, xmm4 );
            (~C).store( i+2UL, j         , xmm5 );
            (~C).store( i+2UL, j+SIMDSIZE, xmm6 );
         }

         for( ; (i+2UL) <= iend; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1( (~C).load(i    ,j         ) );
            SIMDType xmm2( (~C).load(i    ,j+SIMDSIZE) );
            SIMDType xmm3( (~C).load(i+1UL,j         ) );
            SIMDType xmm4( (~C).load(i+1UL,j+SIMDSIZE) );
            SIMDType xmm5, xmm6, xmm7, xmm8;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType a1( set( A(i    ,k    ) ) );
               const SIMDType a2( set( A(i+1UL,k    ) ) );
               const SIMDType a3( set( A(i    ,k+1UL) ) );
               const SIMDType a4( set( A(i+1UL,k+1UL) ) );
               const SIMDType b1( B.load(k    ,j         ) );
               const SIMDType b2( B.load(k    ,j+SIMDSIZE) );
               const SIMDType b3( B.load(k+1UL,j         ) );
               const SIMDType b4( B.load(k+1UL,j+SIMDSIZE) );
               xmm1 -= a1 * b1;
               xmm2 -= a1 * b2;
               xmm3 -= a2 * b1;
               xmm4 -= a2 * b2;
               xmm5 -= a3 * b3;
               xmm6 -= a3 * b4;
               xmm7 -= a4 * b3;
               xmm8 -= a4 * b4;
            }

            for( ; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 -= a1 * b1;
               xmm2 -= a1 * b2;
               xmm3 -= a2 * b1;
               xmm4 -= a2 * b2;
            }

            (~C).store( i    , j         , xmm1+xmm5 );
            (~C).store( i    , j+SIMDSIZE, xmm2+xmm6 );
            (~C).store( i+1UL, j         , xmm3+xmm7 );
            (~C).store( i+1UL, j+SIMDSIZE, xmm4+xmm8 );
         }

         if( i < iend )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1( (~C).load(i,j         ) );
            SIMDType xmm2( (~C).load(i,j+SIMDSIZE) );
            SIMDType xmm3, xmm4;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType a1( set( A(i,k    ) ) );
               const SIMDType a2( set( A(i,k+1UL) ) );
               xmm1 -= a1 * B.load(k    ,j         );
               xmm2 -= a1 * B.load(k    ,j+SIMDSIZE);
               xmm3 -= a2 * B.load(k+1UL,j         );
               xmm4 -= a2 * B.load(k+1UL,j+SIMDSIZE);
            }

            for( ; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 -= a1 * B.load(k,j         );
               xmm2 -= a1 * B.load(k,j+SIMDSIZE);
            }

            (~C).store( i, j         , xmm1+xmm3 );
            (~C).store( i, j+SIMDSIZE, xmm2+xmm4 );
         }
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         const size_t iend( LOW && UPP ? min(j+SIMDSIZE,M) : M );
         size_t i( LOW ? j : 0UL );

         for( ; (i+4UL) <= iend; i+=4UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL )
                               :( K ) );

            SIMDType xmm1( (~C).load(i    ,j) );
            SIMDType xmm2( (~C).load(i+1UL,j) );
            SIMDType xmm3( (~C).load(i+2UL,j) );
            SIMDType xmm4( (~C).load(i+3UL,j) );
            SIMDType xmm5, xmm6, xmm7, xmm8;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType b1( B.load(k    ,j) );
               const SIMDType b2( B.load(k+1UL,j) );
               xmm1 -= set( A(i    ,k    ) ) * b1;
               xmm2 -= set( A(i+1UL,k    ) ) * b1;
               xmm3 -= set( A(i+2UL,k    ) ) * b1;
               xmm4 -= set( A(i+3UL,k    ) ) * b1;
               xmm5 -= set( A(i    ,k+1UL) ) * b2;
               xmm6 -= set( A(i+1UL,k+1UL) ) * b2;
               xmm7 -= set( A(i+2UL,k+1UL) ) * b2;
               xmm8 -= set( A(i+3UL,k+1UL) ) * b2;
            }

            for( ; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 -= set( A(i    ,k) ) * b1;
               xmm2 -= set( A(i+1UL,k) ) * b1;
               xmm3 -= set( A(i+2UL,k) ) * b1;
               xmm4 -= set( A(i+3UL,k) ) * b1;
            }

            (~C).store( i    , j, xmm1+xmm5 );
            (~C).store( i+1UL, j, xmm2+xmm6 );
            (~C).store( i+2UL, j, xmm3+xmm7 );
            (~C).store( i+3UL, j, xmm4+xmm8 );
         }

         for( ; (i+3UL) <= iend; i+=3UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL )
                               :( K ) );

            SIMDType xmm1( (~C).load(i    ,j) );
            SIMDType xmm2( (~C).load(i+1UL,j) );
            SIMDType xmm3( (~C).load(i+2UL,j) );
            SIMDType xmm4, xmm5, xmm6;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType b1( B.load(k    ,j) );
               const SIMDType b2( B.load(k+1UL,j) );
               xmm1 -= set( A(i    ,k    ) ) * b1;
               xmm2 -= set( A(i+1UL,k    ) ) * b1;
               xmm3 -= set( A(i+2UL,k    ) ) * b1;
               xmm4 -= set( A(i    ,k+1UL) ) * b2;
               xmm5 -= set( A(i+1UL,k+1UL) ) * b2;
               xmm6 -= set( A(i+2UL,k+1UL) ) * b2;
            }

            for( ; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 -= set( A(i    ,k) ) * b1;
               xmm2 -= set( A(i+1UL,k) ) * b1;
               xmm3 -= set( A(i+2UL,k) ) * b1;
            }

            (~C).store( i    , j, xmm1+xmm4 );
            (~C).store( i+1UL, j, xmm2+xmm5 );
            (~C).store( i+2UL, j, xmm3+xmm6 );
         }

         for( ; (i+2UL) <= iend; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL )
                               :( K ) );

            SIMDType xmm1( (~C).load(i    ,j) );
            SIMDType xmm2( (~C).load(i+1UL,j) );
            SIMDType xmm3, xmm4;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType b1( B.load(k    ,j) );
               const SIMDType b2( B.load(k+1UL,j) );
               xmm1 -= set( A(i    ,k    ) ) * b1;
               xmm2 -= set( A(i+1UL,k    ) ) * b1;
               xmm3 -= set( A(i    ,k+1UL) ) * b2;
               xmm4 -= set( A(i+1UL,k+1UL) ) * b2;
            }

            for( ; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 -= set( A(i    ,k) ) * b1;
               xmm2 -= set( A(i+1UL,k) ) * b1;
            }

            (~C).store( i    , j, xmm1+xmm3 );
            (~C).store( i+1UL, j, xmm2+xmm4 );
         }

         if( i < iend )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );

            SIMDType xmm1( (~C).load(i,j) );
            SIMDType xmm2;
            size_t k( kbegin );

            for( ; (k+2UL) <= K; k+=2UL ) {
               xmm1 -= set( A(i,k    ) ) * B.load(k    ,j);
               xmm2 -= set( A(i,k+1UL) ) * B.load(k+1UL,j);
            }

            for( ; k<K; ++k ) {
               xmm1 -= set( A(i,k) ) * B.load(k,j);
            }

            (~C).store( i, j, xmm1+xmm2 );
         }
      }

      for( ; remainder && j<N; ++j )
      {
         const size_t iend( UPP ? j+1UL : M );
         size_t i( LOW ? j : 0UL );

         for( ; (i+2UL) <= iend; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL )
                               :( K ) );

            ElementType value1( (~C)(i    ,j) );
            ElementType value2( (~C)(i+1UL,j) );

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 -= A(i    ,k) * B(k,j);
               value2 -= A(i+1UL,k) * B(k,j);
            }

            (~C)(i    ,j) = value1;
            (~C)(i+1UL,j) = value2;
         }

         if( i < iend )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );

            ElementType value( (~C)(i,j) );

            for( size_t k=kbegin; k<K; ++k ) {
               value -= A(i,k) * B(k,j);
            }

            (~C)(i,j) = value;
         }
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Default subtraction assignment to dense tensors (large tensors)***************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a large dense tensor-dense tensor multiplication
   //        (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a dense
   // tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline DisableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5> >
      selectLargeSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectDefaultSubAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to dense tensors (large tensors)****************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Vectorized default subtraction assignment of a large dense tensor-dense tensor
   //        multiplication (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment of a dense tensor-
   // dense tensor multiplication expression to a dense tensor. This kernel is optimized for
   // large tensors.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5> >
      selectLargeSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      if( LOW )
         lmmm( C, A, B, ElementType(-1), ElementType(1) );
      else if( UPP )
         ummm( C, A, B, ElementType(-1), ElementType(1) );
      else
         mmm( C, A, B, ElementType(-1), ElementType(1) );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense tensors (default)*******************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default subtraction assignment of a dense tensor-dense tensor multiplication
   //        (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a large
   // dense tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline DisableIf_t< UseBlasKernel_v<MT3,MT4,MT5> >
      selectBlasSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      selectLargeSubAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**BLAS-based subraction assignment to dense tensors******************************************
#if BLAZE_BLAS_MODE && BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION
   /*! \cond BLAZE_INTERNAL */
   /*!\brief BLAS-based subraction assignment of a dense tensor-dense tensor multiplication
   //        (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function performs the dense tensor-dense tensor multiplication based on the according
   // BLAS functionality.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5 >  // Type of the right-hand side tensor operand
   static inline EnableIf_t< UseBlasKernel_v<MT3,MT4,MT5> >
      selectBlasSubAssignKernel( MT3& C, const MT4& A, const MT5& B )
   {
      using ET = ElementType_t<MT3>;

      if( IsTriangular_v<MT4> ) {
         ResultType_t<MT3> tmp( serial( B ) );
         trmm( tmp, A, CblasLeft, ( IsLower_v<MT4> )?( CblasLower ):( CblasUpper ), ET(1) );
         subAssign( C, tmp );
      }
      else if( IsTriangular_v<MT5> ) {
         ResultType_t<MT3> tmp( serial( A ) );
         trmm( tmp, B, CblasRight, ( IsLower_v<MT5> )?( CblasLower ):( CblasUpper ), ET(1) );
         subAssign( C, tmp );
      }
      else {
         gemm( C, A, B, ET(-1), ET(1) );
      }
   }
   /*! \endcond */
#endif
   //**********************************************************************************************

   //**Schur product assignment to dense tensors**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Schur product assignment of a dense tensor-dense tensor multiplication to a dense
   //        tensor (\f$ C\circ=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side multiplication expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized Schur product assignment of a dense
   // tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline void schurAssign( DenseTensor<MT>& lhs, const DTensDTensMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const ResultType tmp( serial( rhs ) );
      schurAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP assignment to dense tensors************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP assignment of a dense tensor-dense tensor multiplication to a dense tensor
   //        (\f$ C=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized SMP assignment of a dense tensor-dense
   // tensor multiplication expression to a dense tensor. Due to the explicit application of the
   // SFINAE principle this function can only be selected by the compiler in case either of the
   // two tensor operands requires an intermediate evaluation and no symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline EnableIf_t< IsEvaluationRequired_v<MT,MT1,MT2> >
      smpAssign( DenseTensor<MT>& lhs, const DTensDTensMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL ) {
         return;
      }
      else if( rhs.lhs_.columns() == 0UL ) {
         reset( ~lhs );
         return;
      }

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side dense tensor operand
      RT B( rhs.rhs_ );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      smpAssign( ~lhs, A * B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP addition assignment to dense tensors***************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP addition assignment of a dense tensor-dense tensor multiplication to a dense
   //        tensor (\f$ C+=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized SMP addition assignment of a dense
   // tensor-dense tensor multiplication expression to a dense tensor. Due to the explicit
   // application of the SFINAE principle this function can only be selected by the compiler
   // in case either of the two tensor operands requires an intermediate evaluation and no
   // symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline EnableIf_t< IsEvaluationRequired_v<MT,MT1,MT2> >
      smpAddAssign( DenseTensor<MT>& lhs, const DTensDTensMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || rhs.lhs_.columns() == 0UL ) {
         return;
      }

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side dense tensor operand
      RT B( rhs.rhs_ );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      smpAddAssign( ~lhs, A * B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP subtraction assignment to dense tensors************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP subtraction assignment of a dense tensor-dense tensor multiplication to a
   //        dense tensor (\f$ C-=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a dense
   // tensor-dense tensor multiplication expression to a dense tensor. Due to the explicit
   // application of the SFINAE principle this function can only be selected by the compiler
   // in case either of the two tensor operands requires an intermediate evaluation and no
   // symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline EnableIf_t< IsEvaluationRequired_v<MT,MT1,MT2> >
      smpSubAssign( DenseTensor<MT>& lhs, const DTensDTensMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || rhs.lhs_.columns() == 0UL ) {
         return;
      }

      LT A( rhs.lhs_ );  // Evaluation of the left-hand side dense tensor operand
      RT B( rhs.rhs_ );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.lhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == rhs.lhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rhs_.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rhs_.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

      smpSubAssign( ~lhs, A * B );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**SMP Schur product assignment to dense tensors**********************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief SMP Schur product assignment of a dense tensor-dense tensor multiplication to a
   //        dense tensor (\f$ C\circ=A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side multiplication expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized SMP Schur product assignment of a
   // dense tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline void smpSchurAssign( DenseTensor<MT>& lhs, const DTensDTensMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const ResultType tmp( rhs );
      smpSchurAssign( ~lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_TENSOR_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_FORM_VALID_TENSTENSMULTEXPR( MT1, MT2 );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  DTENSSCALARMULTEXPR SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Expression object for scaled dense tensor-dense tensor multiplications.
// \ingroup dense_tensor_expression
//
// This specialization of the DTensScalarMultExpr class represents the compile time expression
// for scaled multiplications between row-major dense tensors.
*/
template< typename MT1   // Type of the left-hand side dense tensor
        , typename MT2   // Type of the right-hand side dense tensor
        , typename ST >  // Type of the right-hand side scalar value
class DTensScalarMultExpr< DTensDTensMultExpr<MT1,MT2>, ST >
   : public MatScalarMultExpr< DenseTensor< DTensScalarMultExpr< DTensDTensMultExpr<MT1,MT2>, ST > > >
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   //! Type of the dense tensor multiplication expression.
   using MMM = DTensDTensMultExpr<MT1,MT2>;

   using RES = ResultType_t<MMM>;     //!< Result type of the dense tensor multiplication expression.
   using RT1 = ResultType_t<MT1>;     //!< Result type of the left-hand side dense tensor expression.
   using RT2 = ResultType_t<MT2>;     //!< Result type of the right-hand side dense tensor expression.
   using ET1 = ElementType_t<RT1>;    //!< Element type of the left-hand side dense tensor expression.
   using ET2 = ElementType_t<RT2>;    //!< Element type of the right-hand side dense tensor expression.
   using CT1 = CompositeType_t<MT1>;  //!< Composite type of the left-hand side dense tensor expression.
   using CT2 = CompositeType_t<MT2>;  //!< Composite type of the right-hand side dense tensor expression.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the left-hand side dense tensor expression.
   static constexpr bool evaluateLeft = ( IsComputation_v<MT1> || RequiresEvaluation_v<MT1> );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Compilation switch for the composite type of the right-hand side dense tensor expression.
   static constexpr bool evaluateRight = ( IsComputation_v<MT2> || RequiresEvaluation_v<MT2> );
   //**********************************************************************************************

   //**********************************************************************************************
   static constexpr bool SYM  = false;  //!< Flag for symmetric tensors.
   static constexpr bool HERM = false;  //!< Flag for Hermitian tensors.
   static constexpr bool LOW  = false;  //!< Flag for lower tensors.
   static constexpr bool UPP  = false;  //!< Flag for upper tensors.
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! This variable template is a helper for the selection of the parallel evaluation strategy.
       In case either of the two tensor operands requires an intermediate evaluation, the variable
       will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3 >
   static constexpr bool IsEvaluationRequired_v = ( ( evaluateLeft || evaluateRight ) );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! In case the types of all three involved tensors and the scalar type are suited for a BLAS
       kernel, the variable will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3, typename T4 >
   static constexpr bool UseBlasKernel_v =
      ( BLAZE_BLAS_MODE && BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION &&
        IsContiguous_v<T1> && HasMutableDataAccess_v<T1> &&
        IsContiguous_v<T2> && HasConstDataAccess_v<T2> &&
        IsContiguous_v<T3> && HasConstDataAccess_v<T3> &&
        !IsDiagonal_v<T2> && !IsDiagonal_v<T3> &&
        T1::simdEnabled && T2::simdEnabled && T3::simdEnabled &&
        IsBLASCompatible_v< ElementType_t<T1> > &&
        IsBLASCompatible_v< ElementType_t<T2> > &&
        IsBLASCompatible_v< ElementType_t<T3> > &&
        IsSame_v< ElementType_t<T1>, ElementType_t<T2> > &&
        IsSame_v< ElementType_t<T1>, ElementType_t<T3> > &&
        !( IsBuiltin_v< ElementType_t<T1> > && IsComplex_v<T4> ) );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   /*! In case all four involved data types are suited for a vectorized computation of the
       tensor multiplication, the variable will be set to 1, otherwise it will be 0. */
   template< typename T1, typename T2, typename T3, typename T4 >
   static constexpr bool UseVectorizedDefaultKernel_v =
      ( useOptimizedKernels &&
        !IsDiagonal_v<T3> &&
        T1::simdEnabled && T2::simdEnabled && T3::simdEnabled &&
        IsSIMDCombinable_v< ElementType_t<T1>
                          , ElementType_t<T2>
                          , ElementType_t<T3>
                          , T4 > &&
        HasSIMDAdd_v< ElementType_t<T2>, ElementType_t<T3> > &&
        HasSIMDMult_v< ElementType_t<T2>, ElementType_t<T3> > );
   //**********************************************************************************************

   //**********************************************************************************************
   //! Type of the functor for forwarding an expression to another assign kernel.
   /*! In case a temporary tensor needs to be created, this functor is used to forward the
       resulting expression to another assign kernel. */
   using ForwardFunctor = Noop;
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this DTensScalarMultExpr instance.
   using This = DTensScalarMultExpr<MMM,ST>;

   using ResultType    = MultTrait_t<RES,ST>;          //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;   //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;  //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<ResultType>;    //!< Resulting element type.
   using SIMDType      = SIMDTrait_t<ElementType>;     //!< Resulting SIMD element type.
   using ReturnType    = const ElementType;            //!< Return type for expression template evaluations.
   using CompositeType = const ResultType;             //!< Data type for composite expression templates.

   //! Composite type of the left-hand side dense tensor expression.
   using LeftOperand = const DTensDTensMultExpr<MT1,MT2>;

   //! Composite type of the right-hand side scalar value.
   using RightOperand = ST;

   //! Type for the assignment of the left-hand side dense tensor operand.
   using LT = If_t< evaluateLeft, const RT1, CT1 >;

   //! Type for the assignment of the right-hand side dense tensor operand.
   using RT = If_t< evaluateRight, const RT2, CT2 >;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled =
      ( !IsDiagonal_v<MT2> &&
        MT1::simdEnabled && MT2::simdEnabled &&
        IsSIMDCombinable_v<ET1,ET2,ST> &&
        HasSIMDAdd_v<ET1,ET2> &&
        HasSIMDMult_v<ET1,ET2> );

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable =
      ( !evaluateLeft && MT1::smpAssignable && !evaluateRight && MT2::smpAssignable );
   //**********************************************************************************************

   //**SIMD properties*****************************************************************************
   //! The number of elements packed within a single SIMD element.
   static constexpr size_t SIMDSIZE = SIMDTrait<ElementType>::size;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the DTensScalarMultExpr class.
   //
   // \param tensor The left-hand side dense tensor of the multiplication expression.
   // \param scalar The right-hand side scalar of the multiplication expression.
   */
   explicit inline DTensScalarMultExpr( const MMM& tensor, ST scalar )
      : tensor_( tensor )  // Left-hand side dense tensor of the multiplication expression
      , scalar_( scalar )  // Right-hand side scalar of the multiplication expression
   {}
   //**********************************************************************************************

   //**Access operator*****************************************************************************
   /*!\brief 2D-access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator()( size_t k, size_t i, size_t j ) const {
      BLAZE_INTERNAL_ASSERT( i < tensor_.rows()   , "Invalid row access index"    );
      BLAZE_INTERNAL_ASSERT( j < tensor_.columns(), "Invalid column access index" );
      BLAZE_INTERNAL_ASSERT( k < tensor_.pages(),   "Invalid page access index" );
      return tensor_(k,i,j) * scalar_;
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the tensor elements.
   //
   // \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
   // \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid tensor access index.
   */
   inline ReturnType at( size_t k, size_t i, size_t j ) const {
      if( i >= tensor_.rows() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
      }
      if( j >= tensor_.columns() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
      }
      if( k >= tensor_.pages() ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid page access index" );
      }
      return (*this)(k,i,j);
   }
   //**********************************************************************************************

   //**Rows function*******************************************************************************
   /*!\brief Returns the current number of rows of the tensor.
   //
   // \return The number of rows of the tensor.
   */
   inline size_t rows() const {
      return tensor_.rows();
   }
   //**********************************************************************************************

   //**Columns function****************************************************************************
   /*!\brief Returns the current number of columns of the tensor.
   //
   // \return The number of columns of the tensor.
   */
   inline size_t columns() const {
      return tensor_.columns();
   }
   //**********************************************************************************************

   //**Pages function****************************************************************************
   /*!\brief Returns the current number of pages of the tensor.
   //
   // \return The number of pages of the tensor.
   */
   inline size_t pages() const noexcept {
      return tensor_.pages();
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side dense tensor operand.
   //
   // \return The left-hand side dense tensor operand.
   */
   inline LeftOperand leftOperand() const {
      return tensor_;
   }
   //**********************************************************************************************

   //**Right operand access************************************************************************
   /*!\brief Returns the right-hand side scalar operand.
   //
   // \return The right-hand side scalar operand.
   */
   inline RightOperand rightOperand() const {
      return scalar_;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the expression can alias, \a false otherwise.
   */
   template< typename T >
   inline bool canAlias( const T* alias ) const {
      return tensor_.canAlias( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case an alias effect is detected, \a false otherwise.
   */
   template< typename T >
   inline bool isAliased( const T* alias ) const {
      return tensor_.isAliased( alias );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operands of the expression are properly aligned in memory.
   //
   // \return \a true in case the operands are aligned, \a false if not.
   */
   inline bool isAligned() const {
      return tensor_.isAligned();
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can be used in SMP assignments.
   //
   // \return \a true in case the expression can be used in SMP assignments, \a false if not.
   */
   inline bool canSMPAssign() const noexcept {
      return ( !BLAZE_BLAS_MODE ||
               !BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION ||
               !BLAZE_BLAS_IS_PARALLEL ||
               ( rows() * columns() < DMATDMATMULT_THRESHOLD ) ) &&
             ( rows() * columns() >= SMP_DMATDMATMULT_THRESHOLD );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  tensor_;  //!< Left-hand side dense tensor of the multiplication expression.
   RightOperand scalar_;  //!< Right-hand side scalar of the multiplication expression.
   //**********************************************************************************************

   //**Assignment to dense tensors****************************************************************
   /*!\brief Assignment of a scaled dense tensor-dense tensor multiplication to a dense tensor
   //        (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a scaled dense tensor-
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT > // Type of the target dense tensor
   friend inline void
      assign( DenseTensor<MT>& lhs, const DTensScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( (~lhs).pages() == rhs.pages(),     "Invalid number of pages" );

      LeftOperand_t<MMM>  left ( rhs.tensor_.leftOperand()  );
      RightOperand_t<MMM> right( rhs.tensor_.rightOperand() );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || (~lhs).pages() == 0UL ) {
         return;
      }
      else if( left.columns() == 0UL || left.pages() == 0UL ) {
         reset( ~lhs );
         return;
      }

      LT A( serial( left  ) );  // Evaluation of the left-hand side dense tensor operand
      RT B( serial( right ) );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == right.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == right.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns(), "Invalid number of columns" );

      DTensScalarMultExpr::selectAssignKernel( ~lhs, A, B, rhs.scalar_ );
   }
   //**********************************************************************************************

   //**Assignment to dense tensors (kernel selection)*********************************************
   /*!\brief Selection of the kernel for an assignment of a scaled dense tensor-dense tensor
   //        multiplication to a dense tensor (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      if( ( IsDiagonal_v<MT5> ) ||
          ( !BLAZE_DEBUG_MODE && B.columns() <= SIMDSIZE*10UL ) ||
          ( C.rows() * C.columns() < DMATDMATMULT_THRESHOLD ) )
         selectSmallAssignKernel( C, A, B, scalar );
      else
         selectBlasAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Default assignment to dense tensors (general/general)**************************************
   /*!\brief Default assignment of a scaled general dense tensor-general dense tensor
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment of a scaled general dense tensor-general
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< !IsDiagonal_v<MT4> && !IsDiagonal_v<MT5> >
      selectDefaultAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      BLAZE_INTERNAL_ASSERT( !( SYM || HERM || LOW || UPP ) || ( M == N ), "Broken invariant detected" );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t kbegin( ( IsUpper_v<MT4> )
                              ?( IsStrictlyUpper_v<MT4> ? i+1UL : i )
                              :( 0UL ) );
         const size_t kend( ( IsLower_v<MT4> )
                            ?( IsStrictlyLower_v<MT4> ? i : i+1UL )
                            :( K ) );
         BLAZE_INTERNAL_ASSERT( kbegin <= kend, "Invalid loop indices detected" );

         if( IsStrictlyTriangular_v<MT4> && kbegin == kend ) {
            for( size_t j=0UL; j<N; ++j ) {
               reset( C(i,j) );
            }
            continue;
         }

         {
            const size_t jbegin( ( IsUpper_v<MT5> )
                                 ?( ( IsStrictlyUpper_v<MT5> )
                                    ?( UPP ? max(i,kbegin+1UL) : kbegin+1UL )
                                    :( UPP ? max(i,kbegin) : kbegin ) )
                                 :( UPP ? i : 0UL ) );
            const size_t jend( ( IsLower_v<MT5> )
                               ?( ( IsStrictlyLower_v<MT5> )
                                  ?( LOW ? min(i+1UL,kbegin) : kbegin )
                                  :( LOW ? min(i,kbegin)+1UL : kbegin+1UL ) )
                               :( LOW ? i+1UL : N ) );

            if( ( IsUpper_v<MT4> && IsUpper_v<MT5> ) || UPP ) {
               for( size_t j=0UL; j<jbegin; ++j ) {
                  reset( C(i,j) );
               }
            }
            else if( IsStrictlyUpper_v<MT5> ) {
               reset( C(i,0UL) );
            }
            for( size_t j=jbegin; j<jend; ++j ) {
               C(i,j) = A(i,kbegin) * B(kbegin,j);
            }
            if( ( IsLower_v<MT4> && IsLower_v<MT5> ) || LOW ) {
               for( size_t j=jend; j<N; ++j ) {
                  reset( C(i,j) );
               }
            }
            else if( IsStrictlyLower_v<MT5> ) {
               reset( C(i,N-1UL) );
            }
         }

         for( size_t k=kbegin+1UL; k<kend; ++k )
         {
            const size_t jbegin( ( IsUpper_v<MT5> )
                                 ?( ( IsStrictlyUpper_v<MT5> )
                                    ?( SYM || HERM || UPP ? max( i, k+1UL ) : k+1UL )
                                    :( SYM || HERM || UPP ? max( i, k ) : k ) )
                                 :( SYM || HERM || UPP ? i : 0UL ) );
            const size_t jend( ( IsLower_v<MT5> )
                               ?( ( IsStrictlyLower_v<MT5> )
                                  ?( LOW ? min(i+1UL,k-1UL) : k-1UL )
                                  :( LOW ? min(i+1UL,k) : k ) )
                               :( LOW ? i+1UL : N ) );

            if( ( SYM || HERM || LOW || UPP ) && ( jbegin > jend ) ) continue;
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            for( size_t j=jbegin; j<jend; ++j ) {
               C(i,j) += A(i,k) * B(k,j);
            }
            if( IsLower_v<MT5> ) {
               C(i,jend) = A(i,k) * B(k,jend);
            }
         }

         {
            const size_t jbegin( ( IsUpper_v<MT4> && IsUpper_v<MT5> )
                                 ?( IsStrictlyUpper_v<MT4> || IsStrictlyUpper_v<MT5> ? i+1UL : i )
                                 :( SYM || HERM || UPP ? i : 0UL ) );
            const size_t jend( ( IsLower_v<MT4> && IsLower_v<MT5> )
                               ?( IsStrictlyLower_v<MT4> || IsStrictlyLower_v<MT5> ? i : i+1UL )
                               :( LOW ? i+1UL : N ) );

            if( ( SYM || HERM || LOW || UPP ) && ( jbegin > jend ) ) continue;
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            for( size_t j=jbegin; j<jend; ++j ) {
               C(i,j) *= scalar;
            }
         }
      }

      if( SYM || HERM ) {
         for( size_t i=1UL; i<M; ++i ) {
            for( size_t j=0UL; j<i; ++j ) {
               C(i,j) = HERM ? conj( C(j,i) ) : C(j,i);
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to dense tensors (general/diagonal)*************************************
   /*!\brief Default assignment of a scaled general dense tensor-diagonal dense tensor
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment of a scaled general dense tensor-diagonal
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< !IsDiagonal_v<MT4> && IsDiagonal_v<MT5> >
      selectDefaultAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper_v<MT4> )
                              ?( IsStrictlyUpper_v<MT4> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower_v<MT4> )
                            ?( IsStrictlyLower_v<MT4> ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         if( IsUpper_v<MT4> ) {
            for( size_t j=0UL; j<jbegin; ++j ) {
               reset( C(i,j) );
            }
         }
         for( size_t j=jbegin; j<jend; ++j ) {
            C(i,j) = A(i,j) * B(j,j) * scalar;
         }
         if( IsLower_v<MT4> ) {
            for( size_t j=jend; j<N; ++j ) {
               reset( C(i,j) );
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to dense tensors (diagonal/general)*************************************
   /*!\brief Default assignment of a scaled diagonal dense tensor-general dense tensor
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment of a scaled diagonal dense tensor-general
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< IsDiagonal_v<MT4> && !IsDiagonal_v<MT5> >
      selectDefaultAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper_v<MT5> )
                              ?( IsStrictlyUpper_v<MT5> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower_v<MT5> )
                            ?( IsStrictlyLower_v<MT5> ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         if( IsUpper_v<MT5> ) {
            for( size_t j=0UL; j<jbegin; ++j ) {
               reset( C(i,j) );
            }
         }
         for( size_t j=jbegin; j<jend; ++j ) {
            C(i,j) = A(i,i) * B(i,j) * scalar;
         }
         if( IsLower_v<MT5> ) {
            for( size_t j=jend; j<N; ++j ) {
               reset( C(i,j) );
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to dense tensors (diagonal/diagonal)************************************
   /*!\brief Default assignment of a scaled diagonal dense tensor-diagonal dense tensor
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default assignment of a scaled diagonal dense tensor-diagional
   // dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< IsDiagonal_v<MT4> && IsDiagonal_v<MT5> >
      selectDefaultAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      reset( C );

      for( size_t i=0UL; i<A.rows(); ++i ) {
         C(i,i) = A(i,i) * B(i,i) * scalar;
      }
   }
   //**********************************************************************************************

   //**Default assignment to dense tensors (small tensors)***************************************
   /*!\brief Default assignment of a small scaled dense tensor-dense tensor multiplication
   //        (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a scaled dense
   // tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5,ST2> >
      selectSmallAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectDefaultAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default assignment to row-major dense tensors (small tensors)******************
   /*!\brief Vectorized default assignment of a small scaled dense tensor-dense tensor
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default assignment of a scaled dense tensor-dense
   // tensor multiplication expression to a row-major dense tensor. This kernel is optimized for
   // small tensors.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5,ST2> >
      selectSmallAssignKernel( DenseTensor<MT3>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      constexpr bool remainder( !IsPadded_v<MT3> || !IsPadded_v<MT5> );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      BLAZE_INTERNAL_ASSERT( !( SYM || HERM || LOW || UPP ) || ( M == N ), "Broken invariant detected" );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      const SIMDType factor( set( scalar ) );

      if( LOW && UPP && N > SIMDSIZE*3UL ) {
         reset( ~C );
      }

      {
         size_t j( 0UL );

         if( IsIntegral_v<ElementType> )
         {
            for( ; !SYM && !HERM && !LOW && !UPP && (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL ) {
               for( size_t i=0UL; i<M; ++i )
               {
                  const size_t kbegin( ( IsUpper_v<MT4> )
                                       ?( ( IsLower_v<MT5> )
                                          ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                          :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                       :( IsLower_v<MT5> ? j : 0UL ) );
                  const size_t kend( ( IsLower_v<MT4> )
                                     ?( ( IsUpper_v<MT5> )
                                        ?( min( ( IsStrictlyLower_v<MT4> ? i : i+1UL ), j+SIMDSIZE*8UL, K ) )
                                        :( IsStrictlyLower_v<MT4> ? i : i+1UL ) )
                                     :( IsUpper_v<MT5> ? min( j+SIMDSIZE*8UL, K ) : K ) );

                  SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

                  for( size_t k=kbegin; k<kend; ++k ) {
                     const SIMDType a1( set( A(i,k) ) );
                     xmm1 += a1 * B.load(k,j             );
                     xmm2 += a1 * B.load(k,j+SIMDSIZE    );
                     xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
                     xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
                     xmm5 += a1 * B.load(k,j+SIMDSIZE*4UL);
                     xmm6 += a1 * B.load(k,j+SIMDSIZE*5UL);
                     xmm7 += a1 * B.load(k,j+SIMDSIZE*6UL);
                     xmm8 += a1 * B.load(k,j+SIMDSIZE*7UL);
                  }

                  (~C).store( i, j             , xmm1 * factor );
                  (~C).store( i, j+SIMDSIZE    , xmm2 * factor );
                  (~C).store( i, j+SIMDSIZE*2UL, xmm3 * factor );
                  (~C).store( i, j+SIMDSIZE*3UL, xmm4 * factor );
                  (~C).store( i, j+SIMDSIZE*4UL, xmm5 * factor );
                  (~C).store( i, j+SIMDSIZE*5UL, xmm6 * factor );
                  (~C).store( i, j+SIMDSIZE*6UL, xmm7 * factor );
                  (~C).store( i, j+SIMDSIZE*7UL, xmm8 * factor );
               }
            }
         }

         for( ; !SYM && !HERM && !LOW && !UPP && (j+SIMDSIZE*4UL) < jpos; j+=SIMDSIZE*5UL )
         {
            size_t i( 0UL );

            for( ; (i+2UL) <= M; i+=2UL )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( ( IsUpper_v<MT5> )
                                     ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*5UL, K ) )
                                     :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                                  :( IsUpper_v<MT5> ? min( j+SIMDSIZE*5UL, K ) : K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i    ,k) ) );
                  const SIMDType a2( set( A(i+1UL,k) ) );
                  const SIMDType b1( B.load(k,j             ) );
                  const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
                  const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
                  const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
                  const SIMDType b5( B.load(k,j+SIMDSIZE*4UL) );
                  xmm1  += a1 * b1;
                  xmm2  += a1 * b2;
                  xmm3  += a1 * b3;
                  xmm4  += a1 * b4;
                  xmm5  += a1 * b5;
                  xmm6  += a2 * b1;
                  xmm7  += a2 * b2;
                  xmm8  += a2 * b3;
                  xmm9  += a2 * b4;
                  xmm10 += a2 * b5;
               }

               (~C).store( i    , j             , xmm1  * factor );
               (~C).store( i    , j+SIMDSIZE    , xmm2  * factor );
               (~C).store( i    , j+SIMDSIZE*2UL, xmm3  * factor );
               (~C).store( i    , j+SIMDSIZE*3UL, xmm4  * factor );
               (~C).store( i    , j+SIMDSIZE*4UL, xmm5  * factor );
               (~C).store( i+1UL, j             , xmm6  * factor );
               (~C).store( i+1UL, j+SIMDSIZE    , xmm7  * factor );
               (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm8  * factor );
               (~C).store( i+1UL, j+SIMDSIZE*3UL, xmm9  * factor );
               (~C).store( i+1UL, j+SIMDSIZE*4UL, xmm10 * factor );
            }

            if( i < M )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*5UL, K ) ):( K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i,k) ) );
                  xmm1 += a1 * B.load(k,j             );
                  xmm2 += a1 * B.load(k,j+SIMDSIZE    );
                  xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
                  xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
                  xmm5 += a1 * B.load(k,j+SIMDSIZE*4UL);
               }

               (~C).store( i, j             , xmm1 * factor );
               (~C).store( i, j+SIMDSIZE    , xmm2 * factor );
               (~C).store( i, j+SIMDSIZE*2UL, xmm3 * factor );
               (~C).store( i, j+SIMDSIZE*3UL, xmm4 * factor );
               (~C).store( i, j+SIMDSIZE*4UL, xmm5 * factor );
            }
         }

         for( ; !( LOW && UPP ) && (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
         {
            const size_t iend( SYM || HERM || UPP ? min(j+SIMDSIZE*4UL,M) : M );
            size_t i( LOW ? j : 0UL );

            for( ; (i+2UL) <= iend; i+=2UL )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( ( IsUpper_v<MT5> )
                                     ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*4UL, K ) )
                                     :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                                  :( IsUpper_v<MT5> ? min( j+SIMDSIZE*4UL, K ) : K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i    ,k) ) );
                  const SIMDType a2( set( A(i+1UL,k) ) );
                  const SIMDType b1( B.load(k,j             ) );
                  const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
                  const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
                  const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
                  xmm1 += a1 * b1;
                  xmm2 += a1 * b2;
                  xmm3 += a1 * b3;
                  xmm4 += a1 * b4;
                  xmm5 += a2 * b1;
                  xmm6 += a2 * b2;
                  xmm7 += a2 * b3;
                  xmm8 += a2 * b4;
               }

               (~C).store( i    , j             , xmm1 * factor );
               (~C).store( i    , j+SIMDSIZE    , xmm2 * factor );
               (~C).store( i    , j+SIMDSIZE*2UL, xmm3 * factor );
               (~C).store( i    , j+SIMDSIZE*3UL, xmm4 * factor );
               (~C).store( i+1UL, j             , xmm5 * factor );
               (~C).store( i+1UL, j+SIMDSIZE    , xmm6 * factor );
               (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm7 * factor );
               (~C).store( i+1UL, j+SIMDSIZE*3UL, xmm8 * factor );
            }

            if( i < iend )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*4UL, K ) ):( K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i,k) ) );
                  xmm1 += a1 * B.load(k,j             );
                  xmm2 += a1 * B.load(k,j+SIMDSIZE    );
                  xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
                  xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
               }

               (~C).store( i, j             , xmm1 * factor );
               (~C).store( i, j+SIMDSIZE    , xmm2 * factor );
               (~C).store( i, j+SIMDSIZE*2UL, xmm3 * factor );
               (~C).store( i, j+SIMDSIZE*3UL, xmm4 * factor );
            }
         }

         for( ; (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
         {
            const size_t iend( SYM || HERM || UPP ? min(j+SIMDSIZE*3UL,M) : M );
            size_t i( LOW ? j : 0UL );

            for( ; (i+2UL) <= iend; i+=2UL )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( ( IsUpper_v<MT5> )
                                     ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*3UL, K ) )
                                     :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                                  :( IsUpper_v<MT5> ? min( j+SIMDSIZE*3UL, K ) : K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i    ,k) ) );
                  const SIMDType a2( set( A(i+1UL,k) ) );
                  const SIMDType b1( B.load(k,j             ) );
                  const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
                  const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
                  xmm1 += a1 * b1;
                  xmm2 += a1 * b2;
                  xmm3 += a1 * b3;
                  xmm4 += a2 * b1;
                  xmm5 += a2 * b2;
                  xmm6 += a2 * b3;
               }

               (~C).store( i    , j             , xmm1 * factor );
               (~C).store( i    , j+SIMDSIZE    , xmm2 * factor );
               (~C).store( i    , j+SIMDSIZE*2UL, xmm3 * factor );
               (~C).store( i+1UL, j             , xmm4 * factor );
               (~C).store( i+1UL, j+SIMDSIZE    , xmm5 * factor );
               (~C).store( i+1UL, j+SIMDSIZE*2UL, xmm6 * factor );
            }

            if( i < iend )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*3UL, K ) ):( K ) );

               SIMDType xmm1, xmm2, xmm3;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i,k) ) );
                  xmm1 += a1 * B.load(k,j             );
                  xmm2 += a1 * B.load(k,j+SIMDSIZE    );
                  xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
               }

               (~C).store( i, j             , xmm1 * factor );
               (~C).store( i, j+SIMDSIZE    , xmm2 * factor );
               (~C).store( i, j+SIMDSIZE*2UL, xmm3 * factor );
            }
         }

         for( ; (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
         {
            const size_t iend( SYM || HERM || UPP ? min(j+SIMDSIZE*2UL,M) : M );
            size_t i( LOW ? j : 0UL );

            for( ; (i+4UL) <= iend; i+=4UL )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( ( IsUpper_v<MT5> )
                                     ?( min( ( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL ), j+SIMDSIZE*2UL, K ) )
                                     :( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL ) )
                                  :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i    ,k) ) );
                  const SIMDType a2( set( A(i+1UL,k) ) );
                  const SIMDType a3( set( A(i+2UL,k) ) );
                  const SIMDType a4( set( A(i+3UL,k) ) );
                  const SIMDType b1( B.load(k,j         ) );
                  const SIMDType b2( B.load(k,j+SIMDSIZE) );
                  xmm1 += a1 * b1;
                  xmm2 += a1 * b2;
                  xmm3 += a2 * b1;
                  xmm4 += a2 * b2;
                  xmm5 += a3 * b1;
                  xmm6 += a3 * b2;
                  xmm7 += a4 * b1;
                  xmm8 += a4 * b2;
               }

               (~C).store( i    , j         , xmm1 * factor );
               (~C).store( i    , j+SIMDSIZE, xmm2 * factor );
               (~C).store( i+1UL, j         , xmm3 * factor );
               (~C).store( i+1UL, j+SIMDSIZE, xmm4 * factor );
               (~C).store( i+2UL, j         , xmm5 * factor );
               (~C).store( i+2UL, j+SIMDSIZE, xmm6 * factor );
               (~C).store( i+3UL, j         , xmm7 * factor );
               (~C).store( i+3UL, j+SIMDSIZE, xmm8 * factor );
            }

            for( ; (i+3UL) <= iend; i+=3UL )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( ( IsUpper_v<MT5> )
                                     ?( min( ( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL ), j+SIMDSIZE*2UL, K ) )
                                     :( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL ) )
                                  :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i    ,k) ) );
                  const SIMDType a2( set( A(i+1UL,k) ) );
                  const SIMDType a3( set( A(i+2UL,k) ) );
                  const SIMDType b1( B.load(k,j         ) );
                  const SIMDType b2( B.load(k,j+SIMDSIZE) );
                  xmm1 += a1 * b1;
                  xmm2 += a1 * b2;
                  xmm3 += a2 * b1;
                  xmm4 += a2 * b2;
                  xmm5 += a3 * b1;
                  xmm6 += a3 * b2;
               }

               (~C).store( i    , j         , xmm1 * factor );
               (~C).store( i    , j+SIMDSIZE, xmm2 * factor );
               (~C).store( i+1UL, j         , xmm3 * factor );
               (~C).store( i+1UL, j+SIMDSIZE, xmm4 * factor );
               (~C).store( i+2UL, j         , xmm5 * factor );
               (~C).store( i+2UL, j+SIMDSIZE, xmm6 * factor );
            }

            for( ; (i+2UL) <= iend; i+=2UL )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( ( IsUpper_v<MT5> )
                                     ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*2UL, K ) )
                                     :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                                  :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
               size_t k( kbegin );

               for( ; (k+2UL) <= kend; k+=2UL ) {
                  const SIMDType a1( set( A(i    ,k    ) ) );
                  const SIMDType a2( set( A(i+1UL,k    ) ) );
                  const SIMDType a3( set( A(i    ,k+1UL) ) );
                  const SIMDType a4( set( A(i+1UL,k+1UL) ) );
                  const SIMDType b1( B.load(k    ,j         ) );
                  const SIMDType b2( B.load(k    ,j+SIMDSIZE) );
                  const SIMDType b3( B.load(k+1UL,j         ) );
                  const SIMDType b4( B.load(k+1UL,j+SIMDSIZE) );
                  xmm1 += a1 * b1;
                  xmm2 += a1 * b2;
                  xmm3 += a2 * b1;
                  xmm4 += a2 * b2;
                  xmm5 += a3 * b3;
                  xmm6 += a3 * b4;
                  xmm7 += a4 * b3;
                  xmm8 += a4 * b4;
               }

               for( ; k<kend; ++k ) {
                  const SIMDType a1( set( A(i    ,k) ) );
                  const SIMDType a2( set( A(i+1UL,k) ) );
                  const SIMDType b1( B.load(k,j         ) );
                  const SIMDType b2( B.load(k,j+SIMDSIZE) );
                  xmm1 += a1 * b1;
                  xmm2 += a1 * b2;
                  xmm3 += a2 * b1;
                  xmm4 += a2 * b2;
               }

               (~C).store( i    , j         , (xmm1+xmm5) * factor );
               (~C).store( i    , j+SIMDSIZE, (xmm2+xmm6) * factor );
               (~C).store( i+1UL, j         , (xmm3+xmm7) * factor );
               (~C).store( i+1UL, j+SIMDSIZE, (xmm4+xmm8) * factor );
            }

            if( i < iend )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*2UL, K ) ):( K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4;
               size_t k( kbegin );

               for( ; (k+2UL) <= kend; k+=2UL ) {
                  const SIMDType a1( set( A(i,k    ) ) );
                  const SIMDType a2( set( A(i,k+1UL) ) );
                  xmm1 += a1 * B.load(k    ,j         );
                  xmm2 += a1 * B.load(k    ,j+SIMDSIZE);
                  xmm3 += a2 * B.load(k+1UL,j         );
                  xmm4 += a2 * B.load(k+1UL,j+SIMDSIZE);
               }

               for( ; k<kend; ++k ) {
                  const SIMDType a1( set( A(i,k) ) );
                  xmm1 += a1 * B.load(k,j         );
                  xmm2 += a1 * B.load(k,j+SIMDSIZE);
               }

               (~C).store( i, j         , (xmm1+xmm3) * factor );
               (~C).store( i, j+SIMDSIZE, (xmm2+xmm4) * factor );
            }
         }

         for( ; j<jpos; j+=SIMDSIZE )
         {
            const size_t iend( SYM || HERM || UPP ? min(j+SIMDSIZE,M) : M );
            size_t i( LOW ? j : 0UL );

            for( ; (i+4UL) <= iend; i+=4UL )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL )
                                  :( K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
               size_t k( kbegin );

               for( ; (k+2UL) <= kend; k+=2UL ) {
                  const SIMDType b1( B.load(k    ,j) );
                  const SIMDType b2( B.load(k+1UL,j) );
                  xmm1 += set( A(i    ,k    ) ) * b1;
                  xmm2 += set( A(i+1UL,k    ) ) * b1;
                  xmm3 += set( A(i+2UL,k    ) ) * b1;
                  xmm4 += set( A(i+3UL,k    ) ) * b1;
                  xmm5 += set( A(i    ,k+1UL) ) * b2;
                  xmm6 += set( A(i+1UL,k+1UL) ) * b2;
                  xmm7 += set( A(i+2UL,k+1UL) ) * b2;
                  xmm8 += set( A(i+3UL,k+1UL) ) * b2;
               }

               for( ; k<kend; ++k ) {
                  const SIMDType b1( B.load(k,j) );
                  xmm1 += set( A(i    ,k) ) * b1;
                  xmm2 += set( A(i+1UL,k) ) * b1;
                  xmm3 += set( A(i+2UL,k) ) * b1;
                  xmm4 += set( A(i+3UL,k) ) * b1;
               }

               (~C).store( i    , j, (xmm1+xmm5) * factor );
               (~C).store( i+1UL, j, (xmm2+xmm6) * factor );
               (~C).store( i+2UL, j, (xmm3+xmm7) * factor );
               (~C).store( i+3UL, j, (xmm4+xmm8) * factor );
            }

            for( ; (i+3UL) <= iend; i+=3UL )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL )
                                  :( K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;
               size_t k( kbegin );

               for( ; (k+2UL) <= kend; k+=2UL ) {
                  const SIMDType b1( B.load(k    ,j) );
                  const SIMDType b2( B.load(k+1UL,j) );
                  xmm1 += set( A(i    ,k    ) ) * b1;
                  xmm2 += set( A(i+1UL,k    ) ) * b1;
                  xmm3 += set( A(i+2UL,k    ) ) * b1;
                  xmm4 += set( A(i    ,k+1UL) ) * b2;
                  xmm5 += set( A(i+1UL,k+1UL) ) * b2;
                  xmm6 += set( A(i+2UL,k+1UL) ) * b2;
               }

               for( ; k<kend; ++k ) {
                  const SIMDType b1( B.load(k,j) );
                  xmm1 += set( A(i    ,k) ) * b1;
                  xmm2 += set( A(i+1UL,k) ) * b1;
                  xmm3 += set( A(i+2UL,k) ) * b1;
               }

               (~C).store( i    , j, (xmm1+xmm4) * factor );
               (~C).store( i+1UL, j, (xmm2+xmm5) * factor );
               (~C).store( i+2UL, j, (xmm3+xmm6) * factor );
            }

            for( ; (i+2UL) <= iend; i+=2UL )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL )
                                  :( K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4;
               size_t k( kbegin );

               for( ; (k+2UL) <= kend; k+=2UL ) {
                  const SIMDType b1( B.load(k    ,j) );
                  const SIMDType b2( B.load(k+1UL,j) );
                  xmm1 += set( A(i    ,k    ) ) * b1;
                  xmm2 += set( A(i+1UL,k    ) ) * b1;
                  xmm3 += set( A(i    ,k+1UL) ) * b2;
                  xmm4 += set( A(i+1UL,k+1UL) ) * b2;
               }

               for( ; k<kend; ++k ) {
                  const SIMDType b1( B.load(k,j) );
                  xmm1 += set( A(i    ,k) ) * b1;
                  xmm2 += set( A(i+1UL,k) ) * b1;
               }

               (~C).store( i    , j, (xmm1+xmm3) * factor );
               (~C).store( i+1UL, j, (xmm2+xmm4) * factor );
            }

            if( i < iend )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );

               SIMDType xmm1, xmm2;
               size_t k( kbegin );

               for( ; (k+2UL) <= K; k+=2UL ) {
                  xmm1 += set( A(i,k    ) ) * B.load(k    ,j);
                  xmm2 += set( A(i,k+1UL) ) * B.load(k+1UL,j);
               }

               for( ; k<K; ++k ) {
                  xmm1 += set( A(i,k) ) * B.load(k,j);
               }

               (~C).store( i, j, (xmm1+xmm2) * factor );
            }
         }

         for( ; remainder && j<N; ++j )
         {
            size_t i( LOW && UPP ? j : 0UL );

            for( ; (i+2UL) <= M; i+=2UL )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL )
                                  :( K ) );

               ElementType value1{};
               ElementType value2{};

               for( size_t k=kbegin; k<kend; ++k ) {
                  value1 += A(i    ,k) * B(k,j);
                  value2 += A(i+1UL,k) * B(k,j);
               }

               (~C)(i    ,j) = value1 * scalar;
               (~C)(i+1UL,j) = value2 * scalar;
            }

            if( i < M )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );

               ElementType value{};

               for( size_t k=kbegin; k<K; ++k ) {
                  value += A(i,k) * B(k,j);
               }

               (~C)(i,j) = value * scalar;
            }
         }
      }

      if( ( SYM || HERM ) && ( N > SIMDSIZE*4UL ) ) {
         for( size_t i=SIMDSIZE*4UL; i<M; ++i ) {
            const size_t jend( ( SIMDSIZE*4UL ) * ( i / (SIMDSIZE*4UL) ) );
            for( size_t j=0UL; j<jend; ++j ) {
               (~C)(i,j) = HERM ? conj( (~C)(j,i) ) : (~C)(j,i);
            }
         }
      }
      else if( LOW && !UPP && N > SIMDSIZE*4UL ) {
         for( size_t j=SIMDSIZE*4UL; j<N; ++j ) {
            const size_t iend( ( SIMDSIZE*4UL ) * ( j / (SIMDSIZE*4UL) ) );
            for( size_t i=0UL; i<iend; ++i ) {
               reset( (~C)(i,j) );
            }
         }
      }
      else if( !LOW && UPP && N > SIMDSIZE*4UL ) {
         for( size_t i=SIMDSIZE*4UL; i<M; ++i ) {
            const size_t jend( ( SIMDSIZE*4UL ) * ( i / (SIMDSIZE*4UL) ) );
            for( size_t j=0UL; j<jend; ++j ) {
               reset( (~C)(i,j) );
            }
         }
      }
   }
   //**********************************************************************************************

   //**Default assignment to dense tensors (large tensors)***************************************
   /*!\brief Default assignment of a large scaled dense tensor-dense tensor multiplication
   //        (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a scaled dense
   // tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5,ST2> >
      selectLargeAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectDefaultAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default assignment to dense tensors (large tensors)****************************
   /*!\brief Vectorized default assignment of a large scaled dense tensor-dense tensor
   //        multiplication (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default assignment of a scaled dense tensor-dense
   // tensor multiplication expression to a dense tensor. This kernel is optimized for large
   // tensors.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5,ST2> >
      selectLargeAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      if( SYM )
         smmm( C, A, B, scalar );
      else if( HERM )
         hmmm( C, A, B, scalar );
      else if( LOW )
         lmmm( C, A, B, scalar, ST2(0) );
      else if( UPP )
         ummm( C, A, B, scalar, ST2(0) );
      else
         mmm( C, A, B, scalar, ST2(0) );
   }
   //**********************************************************************************************

   //**BLAS-based assignment to dense tensors (default)*******************************************
   /*!\brief Default assignment of a scaled dense tensor-dense tensor multiplication
   //        (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a large scaled
   // dense tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_t< UseBlasKernel_v<MT3,MT4,MT5,ST2> >
      selectBlasAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectLargeAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**BLAS-based assignment to dense tensors*****************************************************
#if BLAZE_BLAS_MODE && BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION
   /*!\brief BLAS-based assignment of a scaled dense tensor-dense tensor multiplication
   //        (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled dense tensor-dense tensor multiplication based on the
   // according BLAS functionality.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< UseBlasKernel_v<MT3,MT4,MT5,ST2> >
      selectBlasAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      using ET = ElementType_t<MT3>;

      if( IsTriangular_v<MT4> ) {
         assign( C, B );
         trmm( C, A, CblasLeft, ( IsLower_v<MT4> )?( CblasLower ):( CblasUpper ), ET(scalar) );
      }
      else if( IsTriangular_v<MT5> ) {
         assign( C, A );
         trmm( C, B, CblasRight, ( IsLower_v<MT5> )?( CblasLower ):( CblasUpper ), ET(scalar) );
      }
      else {
         gemm( C, A, B, ET(scalar), ET(0) );
      }
   }
#endif
   //**********************************************************************************************

   //**Addition assignment to dense tensors*******************************************************
   /*!\brief Addition assignment of a scaled dense tensor-dense tensor multiplication to a
   //        dense tensor (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side scaled multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a scaled dense
   // tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline void
      addAssign( DenseTensor<MT>& lhs, const DTensScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LeftOperand_t<MMM>  left ( rhs.tensor_.leftOperand()  );
      RightOperand_t<MMM> right( rhs.tensor_.rightOperand() );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || left.columns() == 0UL ) {
         return;
      }

      LT A( serial( left  ) );  // Evaluation of the left-hand side dense tensor operand
      RT B( serial( right ) );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == right.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == right.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns(), "Invalid number of columns" );

      DTensScalarMultExpr::selectAddAssignKernel( ~lhs, A, B, rhs.scalar_ );
   }
   //**********************************************************************************************

   //**Addition assignment to dense tensors (kernel selection)************************************
   /*!\brief Selection of the kernel for an addition assignment of a scaled dense tensor-dense
   //        tensor multiplication to a dense tensor (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      if( ( IsDiagonal_v<MT5> ) ||
          ( !BLAZE_DEBUG_MODE && B.columns() <= SIMDSIZE*10UL ) ||
          ( C.rows() * C.columns() < DMATDMATMULT_THRESHOLD ) )
         selectSmallAddAssignKernel( C, A, B, scalar );
      else
         selectBlasAddAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Default addition assignment to dense tensors (general/general)*****************************
   /*!\brief Default addition assignment of a scaled general dense tensor-general dense tensor
   //        multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment of a scaled dense tensor-dense
   // tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< !IsDiagonal_v<MT4> && !IsDiagonal_v<MT5> >
      selectDefaultAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const ResultType tmp( serial( A * B * scalar ) );
      addAssign( C, tmp );
   }
   //**********************************************************************************************

   //**Default addition assignment to dense tensors (general/diagonal)****************************
   /*!\brief Default addition assignment of a scaled general dense tensor-diagonal dense tensor
   //        multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment of a scaled general dense tensor-
   // diagonal dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< !IsDiagonal_v<MT4> && IsDiagonal_v<MT5> >
      selectDefaultAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper_v<MT4> )
                              ?( IsStrictlyUpper_v<MT4> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower_v<MT4> )
                            ?( IsStrictlyLower_v<MT4> ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jnum( jend - jbegin );
         const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

         for( size_t j=jbegin; j<jpos; j+=2UL ) {
            C(i,j    ) += A(i,j    ) * B(j    ,j    ) * scalar;
            C(i,j+1UL) += A(i,j+1UL) * B(j+1UL,j+1UL) * scalar;
         }
         if( jpos < jend ) {
            C(i,jpos) += A(i,jpos) * B(jpos,jpos) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to dense tensors (diagonal/general)****************************
   /*!\brief Default addition assignment of a scaled diagonal dense tensor-general dense tensor
   //        multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment of a scaled diagonal dense tensor-
   // general dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< IsDiagonal_v<MT4> && !IsDiagonal_v<MT5> >
      selectDefaultAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper_v<MT5> )
                              ?( IsStrictlyUpper_v<MT5> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower_v<MT5> )
                            ?( IsStrictlyLower_v<MT5> ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jnum( jend - jbegin );
         const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

         for( size_t j=jbegin; j<jpos; j+=2UL ) {
            C(i,j    ) += A(i,i) * B(i,j    ) * scalar;
            C(i,j+1UL) += A(i,i) * B(i,j+1UL) * scalar;
         }
         if( jpos < jend ) {
            C(i,jpos) += A(i,i) * B(i,jpos) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to dense tensors (diagonal/diagonal)***************************
   /*!\brief Default addition assignment of a scaled diagonal dense tensor-diagonal dense tensor
   //        multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default addition assignment of a scaled diagonal dense tensor-
   // diagonal dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< IsDiagonal_v<MT4> && IsDiagonal_v<MT5> >
      selectDefaultAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      for( size_t i=0UL; i<A.rows(); ++i ) {
         C(i,i) += A(i,i) * B(i,i) * scalar;
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to dense tensors (small tensors)******************************
   /*!\brief Default addition assignment of a small scaled dense tensor-dense tensor multiplication
   //        (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a scaled
   // dense tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5,ST2> >
      selectSmallAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectDefaultAddAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default addition assignment to row-major dense tensors (small tensors)*********
   /*!\brief Vectorized default addition assignment of a small scaled dense tensor-dense tensor
   //        multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default addition assignment of a scaled dense
   // tensor-dense tensor multiplication expression to a row-major dense tensor. This kernel
   // is optimized for small tensors.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5,ST2> >
      selectSmallAddAssignKernel( DenseTensor<MT3>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      constexpr bool remainder( !IsPadded_v<MT3> || !IsPadded_v<MT5> );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      BLAZE_INTERNAL_ASSERT( !( LOW || UPP ) || ( M == N ), "Broken invariant detected" );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      const SIMDType factor( set( scalar ) );

      size_t j( 0UL );

      if( IsIntegral_v<ElementType> )
      {
         for( ; !LOW && !UPP && (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL ) {
            for( size_t i=0UL; i<M; ++i )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( ( IsUpper_v<MT5> )
                                     ?( min( ( IsStrictlyLower_v<MT4> ? i : i+1UL ), j+SIMDSIZE*8UL, K ) )
                                     :( IsStrictlyLower_v<MT4> ? i : i+1UL ) )
                                  :( IsUpper_v<MT5> ? min( j+SIMDSIZE*8UL, K ) : K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i,k) ) );
                  xmm1 += a1 * B.load(k,j             );
                  xmm2 += a1 * B.load(k,j+SIMDSIZE    );
                  xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
                  xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
                  xmm5 += a1 * B.load(k,j+SIMDSIZE*4UL);
                  xmm6 += a1 * B.load(k,j+SIMDSIZE*5UL);
                  xmm7 += a1 * B.load(k,j+SIMDSIZE*6UL);
                  xmm8 += a1 * B.load(k,j+SIMDSIZE*7UL);
               }

               (~C).store( i, j             , (~C).load(i,j             ) + xmm1 * factor );
               (~C).store( i, j+SIMDSIZE    , (~C).load(i,j+SIMDSIZE    ) + xmm2 * factor );
               (~C).store( i, j+SIMDSIZE*2UL, (~C).load(i,j+SIMDSIZE*2UL) + xmm3 * factor );
               (~C).store( i, j+SIMDSIZE*3UL, (~C).load(i,j+SIMDSIZE*3UL) + xmm4 * factor );
               (~C).store( i, j+SIMDSIZE*4UL, (~C).load(i,j+SIMDSIZE*4UL) + xmm5 * factor );
               (~C).store( i, j+SIMDSIZE*5UL, (~C).load(i,j+SIMDSIZE*5UL) + xmm6 * factor );
               (~C).store( i, j+SIMDSIZE*6UL, (~C).load(i,j+SIMDSIZE*6UL) + xmm7 * factor );
               (~C).store( i, j+SIMDSIZE*7UL, (~C).load(i,j+SIMDSIZE*7UL) + xmm8 * factor );
            }
         }
      }

      for( ; !LOW && !UPP && (j+SIMDSIZE*4UL) < jpos; j+=SIMDSIZE*5UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*5UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*5UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               const SIMDType b5( B.load(k,j+SIMDSIZE*4UL) );
               xmm1  += a1 * b1;
               xmm2  += a1 * b2;
               xmm3  += a1 * b3;
               xmm4  += a1 * b4;
               xmm5  += a1 * b5;
               xmm6  += a2 * b1;
               xmm7  += a2 * b2;
               xmm8  += a2 * b3;
               xmm9  += a2 * b4;
               xmm10 += a2 * b5;
            }

            (~C).store( i    , j             , (~C).load(i    ,j             ) + xmm1  * factor );
            (~C).store( i    , j+SIMDSIZE    , (~C).load(i    ,j+SIMDSIZE    ) + xmm2  * factor );
            (~C).store( i    , j+SIMDSIZE*2UL, (~C).load(i    ,j+SIMDSIZE*2UL) + xmm3  * factor );
            (~C).store( i    , j+SIMDSIZE*3UL, (~C).load(i    ,j+SIMDSIZE*3UL) + xmm4  * factor );
            (~C).store( i    , j+SIMDSIZE*4UL, (~C).load(i    ,j+SIMDSIZE*4UL) + xmm5  * factor );
            (~C).store( i+1UL, j             , (~C).load(i+1UL,j             ) + xmm6  * factor );
            (~C).store( i+1UL, j+SIMDSIZE    , (~C).load(i+1UL,j+SIMDSIZE    ) + xmm7  * factor );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, (~C).load(i+1UL,j+SIMDSIZE*2UL) + xmm8  * factor );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, (~C).load(i+1UL,j+SIMDSIZE*3UL) + xmm9  * factor );
            (~C).store( i+1UL, j+SIMDSIZE*4UL, (~C).load(i+1UL,j+SIMDSIZE*4UL) + xmm10 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*5UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 += a1 * B.load(k,j             );
               xmm2 += a1 * B.load(k,j+SIMDSIZE    );
               xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
               xmm5 += a1 * B.load(k,j+SIMDSIZE*4UL);
            }

            (~C).store( i, j             , (~C).load(i,j             ) + xmm1 * factor );
            (~C).store( i, j+SIMDSIZE    , (~C).load(i,j+SIMDSIZE    ) + xmm2 * factor );
            (~C).store( i, j+SIMDSIZE*2UL, (~C).load(i,j+SIMDSIZE*2UL) + xmm3 * factor );
            (~C).store( i, j+SIMDSIZE*3UL, (~C).load(i,j+SIMDSIZE*3UL) + xmm4 * factor );
            (~C).store( i, j+SIMDSIZE*4UL, (~C).load(i,j+SIMDSIZE*4UL) + xmm5 * factor );
         }
      }

      for( ; !LOW && !UPP && (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*4UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a1 * b3;
               xmm4 += a1 * b4;
               xmm5 += a2 * b1;
               xmm6 += a2 * b2;
               xmm7 += a2 * b3;
               xmm8 += a2 * b4;
            }

            (~C).store( i    , j             , (~C).load(i    ,j             ) + xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE    , (~C).load(i    ,j+SIMDSIZE    ) + xmm2 * factor );
            (~C).store( i    , j+SIMDSIZE*2UL, (~C).load(i    ,j+SIMDSIZE*2UL) + xmm3 * factor );
            (~C).store( i    , j+SIMDSIZE*3UL, (~C).load(i    ,j+SIMDSIZE*3UL) + xmm4 * factor );
            (~C).store( i+1UL, j             , (~C).load(i+1UL,j             ) + xmm5 * factor );
            (~C).store( i+1UL, j+SIMDSIZE    , (~C).load(i+1UL,j+SIMDSIZE    ) + xmm6 * factor );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, (~C).load(i+1UL,j+SIMDSIZE*2UL) + xmm7 * factor );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, (~C).load(i+1UL,j+SIMDSIZE*3UL) + xmm8 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 += a1 * B.load(k,j             );
               xmm2 += a1 * B.load(k,j+SIMDSIZE    );
               xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
            }

            (~C).store( i, j             , (~C).load(i,j             ) + xmm1 * factor );
            (~C).store( i, j+SIMDSIZE    , (~C).load(i,j+SIMDSIZE    ) + xmm2 * factor );
            (~C).store( i, j+SIMDSIZE*2UL, (~C).load(i,j+SIMDSIZE*2UL) + xmm3 * factor );
            (~C).store( i, j+SIMDSIZE*3UL, (~C).load(i,j+SIMDSIZE*3UL) + xmm4 * factor );
         }
      }

      for( ; !LOW && !UPP && (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*3UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*3UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a1 * b3;
               xmm4 += a2 * b1;
               xmm5 += a2 * b2;
               xmm6 += a2 * b3;
            }

            (~C).store( i    , j             , (~C).load(i    ,j             ) + xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE    , (~C).load(i    ,j+SIMDSIZE    ) + xmm2 * factor );
            (~C).store( i    , j+SIMDSIZE*2UL, (~C).load(i    ,j+SIMDSIZE*2UL) + xmm3 * factor );
            (~C).store( i+1UL, j             , (~C).load(i+1UL,j             ) + xmm4 * factor );
            (~C).store( i+1UL, j+SIMDSIZE    , (~C).load(i+1UL,j+SIMDSIZE    ) + xmm5 * factor );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, (~C).load(i+1UL,j+SIMDSIZE*2UL) + xmm6 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*3UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 += a1 * B.load(k,j             );
               xmm2 += a1 * B.load(k,j+SIMDSIZE    );
               xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
            }

            (~C).store( i, j             , (~C).load(i,j             ) + xmm1 * factor );
            (~C).store( i, j+SIMDSIZE    , (~C).load(i,j+SIMDSIZE    ) + xmm2 * factor );
            (~C).store( i, j+SIMDSIZE*2UL, (~C).load(i,j+SIMDSIZE*2UL) + xmm3 * factor );
         }
      }

      for( ; !( LOW && UPP ) && (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         const size_t iend( UPP ? min(j+SIMDSIZE*2UL,M) : M );
         size_t i( LOW ? j : 0UL );

         for( ; (i+4UL) <= iend; i+=4UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType a3( set( A(i+2UL,k) ) );
               const SIMDType a4( set( A(i+3UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a2 * b1;
               xmm4 += a2 * b2;
               xmm5 += a3 * b1;
               xmm6 += a3 * b2;
               xmm7 += a4 * b1;
               xmm8 += a4 * b2;
            }

            (~C).store( i    , j         , (~C).load(i    ,j         ) + xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE, (~C).load(i    ,j+SIMDSIZE) + xmm2 * factor );
            (~C).store( i+1UL, j         , (~C).load(i+1UL,j         ) + xmm3 * factor );
            (~C).store( i+1UL, j+SIMDSIZE, (~C).load(i+1UL,j+SIMDSIZE) + xmm4 * factor );
            (~C).store( i+2UL, j         , (~C).load(i+2UL,j         ) + xmm5 * factor );
            (~C).store( i+2UL, j+SIMDSIZE, (~C).load(i+2UL,j+SIMDSIZE) + xmm6 * factor );
            (~C).store( i+3UL, j         , (~C).load(i+3UL,j         ) + xmm7 * factor );
            (~C).store( i+3UL, j+SIMDSIZE, (~C).load(i+3UL,j+SIMDSIZE) + xmm8 * factor );
         }

         for( ; (i+3UL) <= iend; i+=3UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType a3( set( A(i+2UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a2 * b1;
               xmm4 += a2 * b2;
               xmm5 += a3 * b1;
               xmm6 += a3 * b2;
            }

            (~C).store( i    , j         , (~C).load(i    ,j         ) + xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE, (~C).load(i    ,j+SIMDSIZE) + xmm2 * factor );
            (~C).store( i+1UL, j         , (~C).load(i+1UL,j         ) + xmm3 * factor );
            (~C).store( i+1UL, j+SIMDSIZE, (~C).load(i+1UL,j+SIMDSIZE) + xmm4 * factor );
            (~C).store( i+2UL, j         , (~C).load(i+2UL,j         ) + xmm5 * factor );
            (~C).store( i+2UL, j+SIMDSIZE, (~C).load(i+2UL,j+SIMDSIZE) + xmm6 * factor );
         }

         for( ; (i+2UL) <= iend; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType a1( set( A(i    ,k    ) ) );
               const SIMDType a2( set( A(i+1UL,k    ) ) );
               const SIMDType a3( set( A(i    ,k+1UL) ) );
               const SIMDType a4( set( A(i+1UL,k+1UL) ) );
               const SIMDType b1( B.load(k    ,j         ) );
               const SIMDType b2( B.load(k    ,j+SIMDSIZE) );
               const SIMDType b3( B.load(k+1UL,j         ) );
               const SIMDType b4( B.load(k+1UL,j+SIMDSIZE) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a2 * b1;
               xmm4 += a2 * b2;
               xmm5 += a3 * b3;
               xmm6 += a3 * b4;
               xmm7 += a4 * b3;
               xmm8 += a4 * b4;
            }

            for( ; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a2 * b1;
               xmm4 += a2 * b2;
            }

            (~C).store( i    , j         , (~C).load(i    ,j         ) + (xmm1+xmm5) * factor );
            (~C).store( i    , j+SIMDSIZE, (~C).load(i    ,j+SIMDSIZE) + (xmm2+xmm6) * factor );
            (~C).store( i+1UL, j         , (~C).load(i+1UL,j         ) + (xmm3+xmm7) * factor );
            (~C).store( i+1UL, j+SIMDSIZE, (~C).load(i+1UL,j+SIMDSIZE) + (xmm4+xmm8) * factor );
         }

         if( i < iend )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType a1( set( A(i,k    ) ) );
               const SIMDType a2( set( A(i,k+1UL) ) );
               xmm1 += a1 * B.load(k    ,j         );
               xmm2 += a1 * B.load(k    ,j+SIMDSIZE);
               xmm3 += a2 * B.load(k+1UL,j         );
               xmm4 += a2 * B.load(k+1UL,j+SIMDSIZE);
            }

            for( ; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 += a1 * B.load(k,j         );
               xmm2 += a1 * B.load(k,j+SIMDSIZE);
            }

            (~C).store( i, j         , (~C).load(i,j         ) + (xmm1+xmm3) * factor );
            (~C).store( i, j+SIMDSIZE, (~C).load(i,j+SIMDSIZE) + (xmm2+xmm4) * factor );
         }
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         const size_t iend( LOW && UPP ? min(j+SIMDSIZE,M) : M );
         size_t i( LOW ? j : 0UL );

         for( ; (i+4UL) <= iend; i+=4UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL )
                               :( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType b1( B.load(k    ,j) );
               const SIMDType b2( B.load(k+1UL,j) );
               xmm1 += set( A(i    ,k    ) ) * b1;
               xmm2 += set( A(i+1UL,k    ) ) * b1;
               xmm3 += set( A(i+2UL,k    ) ) * b1;
               xmm4 += set( A(i+3UL,k    ) ) * b1;
               xmm5 += set( A(i    ,k+1UL) ) * b2;
               xmm6 += set( A(i+1UL,k+1UL) ) * b2;
               xmm7 += set( A(i+2UL,k+1UL) ) * b2;
               xmm8 += set( A(i+3UL,k+1UL) ) * b2;
            }

            for( ; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 += set( A(i    ,k) ) * b1;
               xmm2 += set( A(i+1UL,k) ) * b1;
               xmm3 += set( A(i+2UL,k) ) * b1;
               xmm4 += set( A(i+3UL,k) ) * b1;
            }

            (~C).store( i    , j, (~C).load(i    ,j) + (xmm1+xmm5) * factor );
            (~C).store( i+1UL, j, (~C).load(i+1UL,j) + (xmm2+xmm6) * factor );
            (~C).store( i+2UL, j, (~C).load(i+2UL,j) + (xmm3+xmm7) * factor );
            (~C).store( i+3UL, j, (~C).load(i+3UL,j) + (xmm4+xmm8) * factor );
         }

         for( ; (i+3UL) <= iend; i+=3UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL )
                               :( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType b1( B.load(k    ,j) );
               const SIMDType b2( B.load(k+1UL,j) );
               xmm1 += set( A(i    ,k    ) ) * b1;
               xmm2 += set( A(i+1UL,k    ) ) * b1;
               xmm3 += set( A(i+2UL,k    ) ) * b1;
               xmm4 += set( A(i    ,k+1UL) ) * b2;
               xmm5 += set( A(i+1UL,k+1UL) ) * b2;
               xmm6 += set( A(i+2UL,k+1UL) ) * b2;
            }

            for( ; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 += set( A(i    ,k) ) * b1;
               xmm2 += set( A(i+1UL,k) ) * b1;
               xmm3 += set( A(i+2UL,k) ) * b1;
            }

            (~C).store( i    , j, (~C).load(i    ,j) + (xmm1+xmm4) * factor );
            (~C).store( i+1UL, j, (~C).load(i+1UL,j) + (xmm2+xmm5) * factor );
            (~C).store( i+2UL, j, (~C).load(i+2UL,j) + (xmm3+xmm6) * factor );
         }

         for( ; (i+2UL) <= iend; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL )
                               :( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType b1( B.load(k    ,j) );
               const SIMDType b2( B.load(k+1UL,j) );
               xmm1 += set( A(i    ,k    ) ) * b1;
               xmm2 += set( A(i+1UL,k    ) ) * b1;
               xmm3 += set( A(i    ,k+1UL) ) * b2;
               xmm4 += set( A(i+1UL,k+1UL) ) * b2;
            }

            for( ; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 += set( A(i    ,k) ) * b1;
               xmm2 += set( A(i+1UL,k) ) * b1;
            }

            (~C).store( i    , j, (~C).load(i    ,j) + (xmm1+xmm3) * factor );
            (~C).store( i+1UL, j, (~C).load(i+1UL,j) + (xmm2+xmm4) * factor );
         }

         if( i < iend )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );

            SIMDType xmm1, xmm2;
            size_t k( kbegin );

            for( ; (k+2UL) <= K; k+=2UL ) {
               xmm1 += set( A(i,k    ) ) * B.load(k    ,j);
               xmm2 += set( A(i,k+1UL) ) * B.load(k+1UL,j);
            }

            for( ; k<K; ++k ) {
               xmm1 += set( A(i,k) ) * B.load(k,j);
            }

            (~C).store( i, j, (~C).load(i,j) + (xmm1+xmm2) * factor );
         }
      }

      for( ; remainder && j<N; ++j )
      {
         const size_t iend( UPP ? j+1UL : M );
         size_t i( LOW ? j : 0UL );

         for( ; (i+2UL) <= iend; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL )
                               :( K ) );

            ElementType value1{};
            ElementType value2{};

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i    ,k) * B(k,j);
               value2 += A(i+1UL,k) * B(k,j);
            }

            (~C)(i    ,j) += value1 * scalar;
            (~C)(i+1UL,j) += value2 * scalar;
         }

         if( i < iend )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );

            ElementType value{};

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) += value * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default addition assignment to dense tensors (large tensors)******************************
   /*!\brief Default addition assignment of a large scaled dense tensor-dense tensor multiplication
   //        (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a scaled
   // dense tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5,ST2> >
      selectLargeAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectDefaultAddAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default addition assignment to dense tensors (large tensors)*******************
   /*!\brief Vectorized default addition assignment of a large scaled dense tensor-dense tensor
   //        multiplication (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default addition assignment of a scaled dense
   // tensor-dense tensor multiplication expression to a dense tensor. This kernel is optimized
   // for large tensors.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5,ST2> >
      selectLargeAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      if( LOW )
         lmmm( C, A, B, scalar, ST2(1) );
      else if( UPP )
         ummm( C, A, B, scalar, ST2(1) );
      else
         mmm( C, A, B, scalar, ST2(1) );
   }
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense tensors (default)**********************************
   /*!\brief Default addition assignment of a scaled dense tensor-dense tensor multiplication
   //        (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the addition assignment of a large
   // scaled dense tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_t< UseBlasKernel_v<MT3,MT4,MT5,ST2> >
      selectBlasAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectLargeAddAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**BLAS-based addition assignment to dense tensors********************************************
#if BLAZE_BLAS_MODE && BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION
   /*!\brief BLAS-based addition assignment of a scaled dense tensor-dense tensor multiplication
   //        (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled dense tensor-dense tensor multiplication based on the
   // according BLAS functionality.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< UseBlasKernel_v<MT3,MT4,MT5,ST2> >
      selectBlasAddAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      using ET = ElementType_t<MT3>;

      if( IsTriangular_v<MT4> ) {
         ResultType_t<MT3> tmp( serial( B ) );
         trmm( tmp, A, CblasLeft, ( IsLower_v<MT4> )?( CblasLower ):( CblasUpper ), ET(scalar) );
         addAssign( C, tmp );
      }
      else if( IsTriangular_v<MT5> ) {
         ResultType_t<MT3> tmp( serial( A ) );
         trmm( tmp, B, CblasRight, ( IsLower_v<MT5> )?( CblasLower ):( CblasUpper ), ET(scalar) );
         addAssign( C, tmp );
      }
      else {
         gemm( C, A, B, ET(scalar), ET(1) );
      }
   }
#endif
   //**********************************************************************************************

   //**Subtraction assignment to dense tensors****************************************************
   /*!\brief Subtraction assignment of a scaled dense tensor-dense tensor multiplication to a
   //        dense tensor (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side scaled multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a scaled dense
   // tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline void
      subAssign( DenseTensor<MT>& lhs, const DTensScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LeftOperand_t<MMM>  left ( rhs.tensor_.leftOperand()  );
      RightOperand_t<MMM> right( rhs.tensor_.rightOperand() );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || left.columns() == 0UL ) {
         return;
      }

      LT A( serial( left  ) );  // Evaluation of the left-hand side dense tensor operand
      RT B( serial( right ) );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == right.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == right.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns(), "Invalid number of columns" );

      DTensScalarMultExpr::selectSubAssignKernel( ~lhs, A, B, rhs.scalar_ );
   }
   //**********************************************************************************************

   //**Subtraction assignment to dense tensors (kernel selection)*********************************
   /*!\brief Selection of the kernel for a subtraction assignment of a scaled dense tensor-dense
   //        tensor multiplication to a dense tensor (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline void selectSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      if( ( IsDiagonal_v<MT5> ) ||
          ( !BLAZE_DEBUG_MODE && B.columns() <= SIMDSIZE*10UL ) ||
          ( C.rows() * C.columns() < DMATDMATMULT_THRESHOLD ) )
         selectSmallSubAssignKernel( C, A, B, scalar );
      else
         selectBlasSubAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense tensors (general/general)**************************
   /*!\brief Default subtraction assignment of a scaled general dense tensor-general dense tensor
   //        multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment of a scaled general dense
   // tensor-general dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< !IsDiagonal_v<MT4> && !IsDiagonal_v<MT5> >
      selectDefaultSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      const ResultType tmp( serial( A * B * scalar ) );
      subAssign( C, tmp );
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense tensors (general/diagonal)*************************
   /*!\brief Default subtraction assignment of a scaled general dense tensor-diagonal dense tensor
   //        multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment of a scaled general dense
   // tensor-diagonal dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< !IsDiagonal_v<MT4> && IsDiagonal_v<MT5> >
      selectDefaultSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper_v<MT4> )
                              ?( IsStrictlyUpper_v<MT4> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower_v<MT4> )
                            ?( IsStrictlyLower_v<MT4> ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jnum( jend - jbegin );
         const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

         for( size_t j=jbegin; j<jpos; j+=2UL ) {
            C(i,j    ) -= A(i,j    ) * B(j    ,j    ) * scalar;
            C(i,j+1UL) -= A(i,j+1UL) * B(j+1UL,j+1UL) * scalar;
         }
         if( jpos < jend ) {
            C(i,jpos) -= A(i,jpos) * B(jpos,jpos) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense tensors (diagonal/general)*************************
   /*!\brief Default subtraction assignment of a scaled diagonal dense tensor-general dense tensor
   //        multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment of a scaled diagonal dense
   // tensor-general dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< IsDiagonal_v<MT4> && !IsDiagonal_v<MT5> >
      selectDefaultSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );

      for( size_t i=0UL; i<M; ++i )
      {
         const size_t jbegin( ( IsUpper_v<MT5> )
                              ?( IsStrictlyUpper_v<MT5> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend( ( IsLower_v<MT5> )
                            ?( IsStrictlyLower_v<MT5> ? i : i+1UL )
                            :( N ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         const size_t jnum( jend - jbegin );
         const size_t jpos( jbegin + ( jnum & size_t(-2) ) );

         for( size_t j=jbegin; j<jpos; j+=2UL ) {
            C(i,j    ) -= A(i,i) * B(i,j    ) * scalar;
            C(i,j+1UL) -= A(i,i) * B(i,j+1UL) * scalar;
         }
         if( jpos < jend ) {
            C(i,jpos) -= A(i,i) * B(i,jpos) * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense tensors (diagonal/diagonal)************************
   /*!\brief Default subtraction assignment of a scaled diagonal dense tensor-diagonal dense
   //        tensor multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the default subtraction assignment of a scaled diagonal dense
   // tensor-diagonal dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< IsDiagonal_v<MT4> && IsDiagonal_v<MT5> >
      selectDefaultSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT3 );

      for( size_t i=0UL; i<A.rows(); ++i ) {
         C(i,i) -= A(i,i) * B(i,i) * scalar;
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense tensors (small tensors)***************************
   /*!\brief Default subtraction assignment of a small scaled dense tensor-dense tensor
   //        multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a scaled
   // dense tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5,ST2> >
      selectSmallSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectDefaultSubAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to row-major dense tensors (small tensors)******
   /*!\brief Vectorized default subtraction assignment of a small scaled dense tensor-dense tensor
   //        multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment of a scaled dense
   // tensor-dense tensor multiplication expression to a row-major dense tensor. This kernel is
   // optimized for small tensors.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5,ST2> >
      selectSmallSubAssignKernel( DenseTensor<MT3>& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      constexpr bool remainder( !IsPadded_v<MT3> || !IsPadded_v<MT5> );

      const size_t M( A.rows()    );
      const size_t N( B.columns() );
      const size_t K( A.columns() );

      BLAZE_INTERNAL_ASSERT( !( LOW || UPP ) || ( M == N ), "Broken invariant detected" );

      const size_t jpos( remainder ? ( N & size_t(-SIMDSIZE) ) : N );
      BLAZE_INTERNAL_ASSERT( !remainder || ( N - ( N % SIMDSIZE ) ) == jpos, "Invalid end calculation" );

      const SIMDType factor( set( scalar ) );

      size_t j( 0UL );

      if( IsIntegral_v<ElementType> )
      {
         for( ; !LOW && !UPP && (j+SIMDSIZE*7UL) < jpos; j+=SIMDSIZE*8UL ) {
            for( size_t i=0UL; i<M; ++i )
            {
               const size_t kbegin( ( IsUpper_v<MT4> )
                                    ?( ( IsLower_v<MT5> )
                                       ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                       :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                    :( IsLower_v<MT5> ? j : 0UL ) );
               const size_t kend( ( IsLower_v<MT4> )
                                  ?( ( IsUpper_v<MT5> )
                                     ?( min( ( IsStrictlyLower_v<MT4> ? i : i+1UL ), j+SIMDSIZE*8UL, K ) )
                                     :( IsStrictlyLower_v<MT4> ? i : i+1UL ) )
                                  :( IsUpper_v<MT5> ? min( j+SIMDSIZE*8UL, K ) : K ) );

               SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

               for( size_t k=kbegin; k<kend; ++k ) {
                  const SIMDType a1( set( A(i,k) ) );
                  xmm1 += a1 * B.load(k,j             );
                  xmm2 += a1 * B.load(k,j+SIMDSIZE    );
                  xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
                  xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
                  xmm5 += a1 * B.load(k,j+SIMDSIZE*4UL);
                  xmm6 += a1 * B.load(k,j+SIMDSIZE*5UL);
                  xmm7 += a1 * B.load(k,j+SIMDSIZE*6UL);
                  xmm8 += a1 * B.load(k,j+SIMDSIZE*7UL);
               }

               (~C).store( i, j             , (~C).load(i,j             ) - xmm1 * factor );
               (~C).store( i, j+SIMDSIZE    , (~C).load(i,j+SIMDSIZE    ) - xmm2 * factor );
               (~C).store( i, j+SIMDSIZE*2UL, (~C).load(i,j+SIMDSIZE*2UL) - xmm3 * factor );
               (~C).store( i, j+SIMDSIZE*3UL, (~C).load(i,j+SIMDSIZE*3UL) - xmm4 * factor );
               (~C).store( i, j+SIMDSIZE*4UL, (~C).load(i,j+SIMDSIZE*4UL) - xmm5 * factor );
               (~C).store( i, j+SIMDSIZE*5UL, (~C).load(i,j+SIMDSIZE*5UL) - xmm6 * factor );
               (~C).store( i, j+SIMDSIZE*6UL, (~C).load(i,j+SIMDSIZE*6UL) - xmm7 * factor );
               (~C).store( i, j+SIMDSIZE*7UL, (~C).load(i,j+SIMDSIZE*7UL) - xmm8 * factor );
            }
         }
      }

      for( ; !LOW && !UPP && (j+SIMDSIZE*4UL) < jpos; j+=SIMDSIZE*5UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*5UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*5UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               const SIMDType b5( B.load(k,j+SIMDSIZE*4UL) );
               xmm1  += a1 * b1;
               xmm2  += a1 * b2;
               xmm3  += a1 * b3;
               xmm4  += a1 * b4;
               xmm5  += a1 * b5;
               xmm6  += a2 * b1;
               xmm7  += a2 * b2;
               xmm8  += a2 * b3;
               xmm9  += a2 * b4;
               xmm10 += a2 * b5;
            }

            (~C).store( i    , j             , (~C).load(i    ,j             ) - xmm1  * factor );
            (~C).store( i    , j+SIMDSIZE    , (~C).load(i    ,j+SIMDSIZE    ) - xmm2  * factor );
            (~C).store( i    , j+SIMDSIZE*2UL, (~C).load(i    ,j+SIMDSIZE*2UL) - xmm3  * factor );
            (~C).store( i    , j+SIMDSIZE*3UL, (~C).load(i    ,j+SIMDSIZE*3UL) - xmm4  * factor );
            (~C).store( i    , j+SIMDSIZE*4UL, (~C).load(i    ,j+SIMDSIZE*4UL) - xmm5  * factor );
            (~C).store( i+1UL, j             , (~C).load(i+1UL,j             ) - xmm6  * factor );
            (~C).store( i+1UL, j+SIMDSIZE    , (~C).load(i+1UL,j+SIMDSIZE    ) - xmm7  * factor );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, (~C).load(i+1UL,j+SIMDSIZE*2UL) - xmm8  * factor );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, (~C).load(i+1UL,j+SIMDSIZE*3UL) - xmm9  * factor );
            (~C).store( i+1UL, j+SIMDSIZE*4UL, (~C).load(i+1UL,j+SIMDSIZE*4UL) - xmm10 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*5UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 += a1 * B.load(k,j             );
               xmm2 += a1 * B.load(k,j+SIMDSIZE    );
               xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
               xmm5 += a1 * B.load(k,j+SIMDSIZE*4UL);
            }

            (~C).store( i, j             , (~C).load(i,j             ) - xmm1 * factor );
            (~C).store( i, j+SIMDSIZE    , (~C).load(i,j+SIMDSIZE    ) - xmm2 * factor );
            (~C).store( i, j+SIMDSIZE*2UL, (~C).load(i,j+SIMDSIZE*2UL) - xmm3 * factor );
            (~C).store( i, j+SIMDSIZE*3UL, (~C).load(i,j+SIMDSIZE*3UL) - xmm4 * factor );
            (~C).store( i, j+SIMDSIZE*4UL, (~C).load(i,j+SIMDSIZE*4UL) - xmm5 * factor );
         }
      }

      for( ; !LOW && !UPP && (j+SIMDSIZE*3UL) < jpos; j+=SIMDSIZE*4UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*4UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*4UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               const SIMDType b4( B.load(k,j+SIMDSIZE*3UL) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a1 * b3;
               xmm4 += a1 * b4;
               xmm5 += a2 * b1;
               xmm6 += a2 * b2;
               xmm7 += a2 * b3;
               xmm8 += a2 * b4;
            }

            (~C).store( i    , j             , (~C).load(i    ,j             ) - xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE    , (~C).load(i    ,j+SIMDSIZE    ) - xmm2 * factor );
            (~C).store( i    , j+SIMDSIZE*2UL, (~C).load(i    ,j+SIMDSIZE*2UL) - xmm3 * factor );
            (~C).store( i    , j+SIMDSIZE*3UL, (~C).load(i    ,j+SIMDSIZE*3UL) - xmm4 * factor );
            (~C).store( i+1UL, j             , (~C).load(i+1UL,j             ) - xmm5 * factor );
            (~C).store( i+1UL, j+SIMDSIZE    , (~C).load(i+1UL,j+SIMDSIZE    ) - xmm6 * factor );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, (~C).load(i+1UL,j+SIMDSIZE*2UL) - xmm7 * factor );
            (~C).store( i+1UL, j+SIMDSIZE*3UL, (~C).load(i+1UL,j+SIMDSIZE*3UL) - xmm8 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*4UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 += a1 * B.load(k,j             );
               xmm2 += a1 * B.load(k,j+SIMDSIZE    );
               xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
               xmm4 += a1 * B.load(k,j+SIMDSIZE*3UL);
            }

            (~C).store( i, j             , (~C).load(i,j             ) - xmm1 * factor );
            (~C).store( i, j+SIMDSIZE    , (~C).load(i,j+SIMDSIZE    ) - xmm2 * factor );
            (~C).store( i, j+SIMDSIZE*2UL, (~C).load(i,j+SIMDSIZE*2UL) - xmm3 * factor );
            (~C).store( i, j+SIMDSIZE*3UL, (~C).load(i,j+SIMDSIZE*3UL) - xmm4 * factor );
         }
      }

      for( ; !LOW && !UPP && (j+SIMDSIZE*2UL) < jpos; j+=SIMDSIZE*3UL )
      {
         size_t i( 0UL );

         for( ; (i+2UL) <= M; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*3UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*3UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j             ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE    ) );
               const SIMDType b3( B.load(k,j+SIMDSIZE*2UL) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a1 * b3;
               xmm4 += a2 * b1;
               xmm5 += a2 * b2;
               xmm6 += a2 * b3;
            }

            (~C).store( i    , j             , (~C).load(i    ,j             ) - xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE    , (~C).load(i    ,j+SIMDSIZE    ) - xmm2 * factor );
            (~C).store( i    , j+SIMDSIZE*2UL, (~C).load(i    ,j+SIMDSIZE*2UL) - xmm3 * factor );
            (~C).store( i+1UL, j             , (~C).load(i+1UL,j             ) - xmm4 * factor );
            (~C).store( i+1UL, j+SIMDSIZE    , (~C).load(i+1UL,j+SIMDSIZE    ) - xmm5 * factor );
            (~C).store( i+1UL, j+SIMDSIZE*2UL, (~C).load(i+1UL,j+SIMDSIZE*2UL) - xmm6 * factor );
         }

         if( i < M )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*3UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 += a1 * B.load(k,j             );
               xmm2 += a1 * B.load(k,j+SIMDSIZE    );
               xmm3 += a1 * B.load(k,j+SIMDSIZE*2UL);
            }

            (~C).store( i, j             , (~C).load(i,j             ) - xmm1 * factor );
            (~C).store( i, j+SIMDSIZE    , (~C).load(i,j+SIMDSIZE    ) - xmm2 * factor );
            (~C).store( i, j+SIMDSIZE*2UL, (~C).load(i,j+SIMDSIZE*2UL) - xmm3 * factor );
         }
      }

      for( ; !( LOW && UPP ) && (j+SIMDSIZE) < jpos; j+=SIMDSIZE*2UL )
      {
         const size_t iend( UPP ? min(j+SIMDSIZE*2UL,M) : M );
         size_t i( LOW ? j : 0UL );

         for( ; (i+4UL) <= iend; i+=4UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType a3( set( A(i+2UL,k) ) );
               const SIMDType a4( set( A(i+3UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a2 * b1;
               xmm4 += a2 * b2;
               xmm5 += a3 * b1;
               xmm6 += a3 * b2;
               xmm7 += a4 * b1;
               xmm8 += a4 * b2;
            }

            (~C).store( i    , j         , (~C).load(i    ,j         ) - xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE, (~C).load(i    ,j+SIMDSIZE) - xmm2 * factor );
            (~C).store( i+1UL, j         , (~C).load(i+1UL,j         ) - xmm3 * factor );
            (~C).store( i+1UL, j+SIMDSIZE, (~C).load(i+1UL,j+SIMDSIZE) - xmm4 * factor );
            (~C).store( i+2UL, j         , (~C).load(i+2UL,j         ) - xmm5 * factor );
            (~C).store( i+2UL, j+SIMDSIZE, (~C).load(i+2UL,j+SIMDSIZE) - xmm6 * factor );
            (~C).store( i+3UL, j         , (~C).load(i+3UL,j         ) - xmm7 * factor );
            (~C).store( i+3UL, j+SIMDSIZE, (~C).load(i+3UL,j+SIMDSIZE) - xmm8 * factor );
         }

         for( ; (i+3UL) <= iend; i+=3UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

            for( size_t k=kbegin; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType a3( set( A(i+2UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a2 * b1;
               xmm4 += a2 * b2;
               xmm5 += a3 * b1;
               xmm6 += a3 * b2;
            }

            (~C).store( i    , j         , (~C).load(i    ,j         ) - xmm1 * factor );
            (~C).store( i    , j+SIMDSIZE, (~C).load(i    ,j+SIMDSIZE) - xmm2 * factor );
            (~C).store( i+1UL, j         , (~C).load(i+1UL,j         ) - xmm3 * factor );
            (~C).store( i+1UL, j+SIMDSIZE, (~C).load(i+1UL,j+SIMDSIZE) - xmm4 * factor );
            (~C).store( i+2UL, j         , (~C).load(i+2UL,j         ) - xmm5 * factor );
            (~C).store( i+2UL, j+SIMDSIZE, (~C).load(i+2UL,j+SIMDSIZE) - xmm6 * factor );
         }

         for( ; (i+2UL) <= iend; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( ( IsUpper_v<MT5> )
                                  ?( min( ( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ), j+SIMDSIZE*2UL, K ) )
                                  :( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL ) )
                               :( IsUpper_v<MT5> ? min( j+SIMDSIZE*2UL, K ) : K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType a1( set( A(i    ,k    ) ) );
               const SIMDType a2( set( A(i+1UL,k    ) ) );
               const SIMDType a3( set( A(i    ,k+1UL) ) );
               const SIMDType a4( set( A(i+1UL,k+1UL) ) );
               const SIMDType b1( B.load(k    ,j         ) );
               const SIMDType b2( B.load(k    ,j+SIMDSIZE) );
               const SIMDType b3( B.load(k+1UL,j         ) );
               const SIMDType b4( B.load(k+1UL,j+SIMDSIZE) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a2 * b1;
               xmm4 += a2 * b2;
               xmm5 += a3 * b3;
               xmm6 += a3 * b4;
               xmm7 += a4 * b3;
               xmm8 += a4 * b4;
            }

            for( ; k<kend; ++k ) {
               const SIMDType a1( set( A(i    ,k) ) );
               const SIMDType a2( set( A(i+1UL,k) ) );
               const SIMDType b1( B.load(k,j         ) );
               const SIMDType b2( B.load(k,j+SIMDSIZE) );
               xmm1 += a1 * b1;
               xmm2 += a1 * b2;
               xmm3 += a2 * b1;
               xmm4 += a2 * b2;
            }

            (~C).store( i    , j         , (~C).load(i    ,j         ) - (xmm1+xmm5) * factor );
            (~C).store( i    , j+SIMDSIZE, (~C).load(i    ,j+SIMDSIZE) - (xmm2+xmm6) * factor );
            (~C).store( i+1UL, j         , (~C).load(i+1UL,j         ) - (xmm3+xmm7) * factor );
            (~C).store( i+1UL, j+SIMDSIZE, (~C).load(i+1UL,j+SIMDSIZE) - (xmm4+xmm8) * factor );
         }

         if( i < iend )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsUpper_v<MT5> )?( min( j+SIMDSIZE*2UL, K ) ):( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType a1( set( A(i,k    ) ) );
               const SIMDType a2( set( A(i,k+1UL) ) );
               xmm1 += a1 * B.load(k    ,j         );
               xmm2 += a1 * B.load(k    ,j+SIMDSIZE);
               xmm3 += a2 * B.load(k+1UL,j         );
               xmm4 += a2 * B.load(k+1UL,j+SIMDSIZE);
            }

            for( ; k<kend; ++k ) {
               const SIMDType a1( set( A(i,k) ) );
               xmm1 += a1 * B.load(k,j         );
               xmm2 += a1 * B.load(k,j+SIMDSIZE);
            }

            (~C).store( i, j         , (~C).load(i,j         ) - (xmm1+xmm3) * factor );
            (~C).store( i, j+SIMDSIZE, (~C).load(i,j+SIMDSIZE) - (xmm2+xmm4) * factor );
         }
      }

      for( ; j<jpos; j+=SIMDSIZE )
      {
         const size_t iend( LOW && UPP ? min(j+SIMDSIZE,M) : M );
         size_t i( LOW ? j : 0UL );

         for( ; (i+4UL) <= iend; i+=4UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+3UL : i+4UL )
                               :( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType b1( B.load(k    ,j) );
               const SIMDType b2( B.load(k+1UL,j) );
               xmm1 += set( A(i    ,k    ) ) * b1;
               xmm2 += set( A(i+1UL,k    ) ) * b1;
               xmm3 += set( A(i+2UL,k    ) ) * b1;
               xmm4 += set( A(i+3UL,k    ) ) * b1;
               xmm5 += set( A(i    ,k+1UL) ) * b2;
               xmm6 += set( A(i+1UL,k+1UL) ) * b2;
               xmm7 += set( A(i+2UL,k+1UL) ) * b2;
               xmm8 += set( A(i+3UL,k+1UL) ) * b2;
            }

            for( ; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 += set( A(i    ,k) ) * b1;
               xmm2 += set( A(i+1UL,k) ) * b1;
               xmm3 += set( A(i+2UL,k) ) * b1;
               xmm4 += set( A(i+3UL,k) ) * b1;
            }

            (~C).store( i    , j, (~C).load(i    ,j) - (xmm1+xmm5) * factor );
            (~C).store( i+1UL, j, (~C).load(i+1UL,j) - (xmm2+xmm6) * factor );
            (~C).store( i+2UL, j, (~C).load(i+2UL,j) - (xmm3+xmm7) * factor );
            (~C).store( i+3UL, j, (~C).load(i+3UL,j) - (xmm4+xmm8) * factor );
         }

         for( ; (i+3UL) <= iend; i+=3UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+2UL : i+3UL )
                               :( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType b1( B.load(k    ,j) );
               const SIMDType b2( B.load(k+1UL,j) );
               xmm1 += set( A(i    ,k    ) ) * b1;
               xmm2 += set( A(i+1UL,k    ) ) * b1;
               xmm3 += set( A(i+2UL,k    ) ) * b1;
               xmm4 += set( A(i    ,k+1UL) ) * b2;
               xmm5 += set( A(i+1UL,k+1UL) ) * b2;
               xmm6 += set( A(i+2UL,k+1UL) ) * b2;
            }

            for( ; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 += set( A(i    ,k) ) * b1;
               xmm2 += set( A(i+1UL,k) ) * b1;
               xmm3 += set( A(i+2UL,k) ) * b1;
            }

            (~C).store( i    , j, (~C).load(i    ,j) - (xmm1+xmm4) * factor );
            (~C).store( i+1UL, j, (~C).load(i+1UL,j) - (xmm2+xmm5) * factor );
            (~C).store( i+2UL, j, (~C).load(i+2UL,j) - (xmm3+xmm6) * factor );
         }

         for( ; (i+2UL) <= iend; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL )
                               :( K ) );

            SIMDType xmm1, xmm2, xmm3, xmm4;
            size_t k( kbegin );

            for( ; (k+2UL) <= kend; k+=2UL ) {
               const SIMDType b1( B.load(k    ,j) );
               const SIMDType b2( B.load(k+1UL,j) );
               xmm1 += set( A(i    ,k    ) ) * b1;
               xmm2 += set( A(i+1UL,k    ) ) * b1;
               xmm3 += set( A(i    ,k+1UL) ) * b2;
               xmm4 += set( A(i+1UL,k+1UL) ) * b2;
            }

            for( ; k<kend; ++k ) {
               const SIMDType b1( B.load(k,j) );
               xmm1 += set( A(i    ,k) ) * b1;
               xmm2 += set( A(i+1UL,k) ) * b1;
            }

            (~C).store( i    , j, (~C).load(i    ,j) - (xmm1+xmm3) * factor );
            (~C).store( i+1UL, j, (~C).load(i+1UL,j) - (xmm2+xmm4) * factor );
         }

         if( i < iend )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );

            SIMDType xmm1, xmm2;
            size_t k( kbegin );

            for( ; (k+2UL) <= K; k+=2UL ) {
               xmm1 += set( A(i,k    ) ) * B.load(k    ,j);
               xmm2 += set( A(i,k+1UL) ) * B.load(k+1UL,j);
            }

            for( ; k<K; ++k ) {
               xmm1 += set( A(i,k) ) * B.load(k,j);
            }

            (~C).store( i, j, (~C).load(i,j) - (xmm1+xmm2) * factor );
         }
      }

      for( ; remainder && j<N; ++j )
      {
         const size_t iend( UPP ? j+1UL : M );
         size_t i( LOW ? j : 0UL );

         for( ; (i+2UL) <= iend; i+=2UL )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );
            const size_t kend( ( IsLower_v<MT4> )
                               ?( IsStrictlyLower_v<MT4> ? i+1UL : i+2UL )
                               :( K ) );

            ElementType value1{};
            ElementType value2{};

            for( size_t k=kbegin; k<kend; ++k ) {
               value1 += A(i    ,k) * B(k,j);
               value2 += A(i+1UL,k) * B(k,j);
            }

            (~C)(i    ,j) -= value1 * scalar;
            (~C)(i+1UL,j) -= value2 * scalar;
         }

         if( i < iend )
         {
            const size_t kbegin( ( IsUpper_v<MT4> )
                                 ?( ( IsLower_v<MT5> )
                                    ?( max( ( IsStrictlyUpper_v<MT4> ? i+1UL : i ), j ) )
                                    :( IsStrictlyUpper_v<MT4> ? i+1UL : i ) )
                                 :( IsLower_v<MT5> ? j : 0UL ) );

            ElementType value{};

            for( size_t k=kbegin; k<K; ++k ) {
               value += A(i,k) * B(k,j);
            }

            (~C)(i,j) -= value * scalar;
         }
      }
   }
   //**********************************************************************************************

   //**Default subtraction assignment to dense tensors (large tensors)***************************
   /*!\brief Default subtraction assignment of a large scaled dense tensor-dense tensor
   //        multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a scaled
   // dense tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5,ST2> >
      selectLargeSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectDefaultSubAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**Vectorized default subtraction assignment to dense tensors (large tensors)****************
   /*!\brief Vectorized default subtraction assignment of a large scaled dense tensor-dense tensor
   //        multiplication (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function implements the vectorized default subtraction assignment of a scaled dense
   // tensor-dense tensor multiplication expression to a dense tensor. This kernel is optimized
   // for large tensors.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< UseVectorizedDefaultKernel_v<MT3,MT4,MT5,ST2> >
      selectLargeSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      if( LOW )
         lmmm( C, A, B, -scalar, ST2(1) );
      else if( UPP )
         ummm( C, A, B, -scalar, ST2(1) );
      else
         mmm( C, A, B, -scalar, ST2(1) );
   }
   //**********************************************************************************************

   //**BLAS-based subtraction assignment to dense tensors (default)*******************************
   /*!\brief Default subtraction assignment of a scaled dense tensor-dense tensor multiplication
   //        (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the subtraction assignment of a large
   // scaled dense tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline DisableIf_t< UseBlasKernel_v<MT3,MT4,MT5,ST2> >
      selectBlasSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      selectLargeSubAssignKernel( C, A, B, scalar );
   }
   //**********************************************************************************************

   //**BLAS-based subraction assignment to dense tensors******************************************
#if BLAZE_BLAS_MODE && BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION
   /*!\brief BLAS-based subraction assignment of a scaled dense tensor-dense tensor multiplication
   //        (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param C The target left-hand side dense tensor.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function performs the scaled dense tensor-dense tensor multiplication based on the
   // according BLAS functionality.
   */
   template< typename MT3    // Type of the left-hand side target tensor
           , typename MT4    // Type of the left-hand side tensor operand
           , typename MT5    // Type of the right-hand side tensor operand
           , typename ST2 >  // Type of the scalar value
   static inline EnableIf_t< UseBlasKernel_v<MT3,MT4,MT5,ST2> >
      selectBlasSubAssignKernel( MT3& C, const MT4& A, const MT5& B, ST2 scalar )
   {
      using ET = ElementType_t<MT3>;

      if( IsTriangular_v<MT4> ) {
         ResultType_t<MT3> tmp( serial( B ) );
         trmm( tmp, A, CblasLeft, ( IsLower_v<MT4> )?( CblasLower ):( CblasUpper ), ET(scalar) );
         subAssign( C, tmp );
      }
      else if( IsTriangular_v<MT5> ) {
         ResultType_t<MT3> tmp( serial( A ) );
         trmm( tmp, B, CblasRight, ( IsLower_v<MT5> )?( CblasLower ):( CblasUpper ), ET(scalar) );
         subAssign( C, tmp );
      }
      else {
         gemm( C, A, B, ET(-scalar), ET(1) );
      }
   }
#endif
   //**********************************************************************************************

   //**Schur product assignment to dense tensors**************************************************
   /*!\brief Schur product assignment of a scaled dense tensor-dense tensor multiplication to a
   //        dense tensor (\f$ C\circ=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side scaled multiplication expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized Schur product assignment of a scaled
   // dense tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline void schurAssign( DenseTensor<MT>& lhs, const DTensScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const ResultType tmp( serial( rhs ) );
      schurAssign( ~lhs, tmp );
   }
   //**********************************************************************************************

   //**Schur product assignment to sparse tensors*************************************************
   // No special implementation for the Schur product assignment to sparse tensors.
   //**********************************************************************************************

   //**Multiplication assignment to dense tensors*************************************************
   // No special implementation for the multiplication assignment to dense tensors.
   //**********************************************************************************************

   //**Multiplication assignment to sparse tensors************************************************
   // No special implementation for the multiplication assignment to sparse tensors.
   //**********************************************************************************************

   //**SMP assignment to dense tensors************************************************************
   /*!\brief SMP assignment of a scaled dense tensor-dense tensor multiplication to a dense tensor
   //        (\f$ C=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side scaled multiplication expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a scaled dense tensor-
   // dense tensor multiplication expression to a dense tensor. Due to the explicit application
   // of the SFINAE principle this function can only be selected by the compiler in case either
   // of the two tensor operands requires an intermediate evaluation and no symmetry can be
   // exploited.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline EnableIf_t< IsEvaluationRequired_v<MT,MT1,MT2> >
      smpAssign( DenseTensor<MT>& lhs, const DTensScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LeftOperand_t<MMM>  left ( rhs.tensor_.leftOperand()  );
      RightOperand_t<MMM> right( rhs.tensor_.rightOperand() );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL ) {
         return;
      }
      else if( left.columns() == 0UL ) {
         reset( ~lhs );
         return;
      }

      LT A( left  );  // Evaluation of the left-hand side dense tensor operand
      RT B( right );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == right.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == right.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns(), "Invalid number of columns" );

      smpAssign( ~lhs, A * B * rhs.scalar_ );
   }
   //**********************************************************************************************

   //**SMP addition assignment to dense tensors***************************************************
   /*!\brief SMP addition assignment of a scaled dense tensor-dense tensor multiplication to a
   //        dense tensor (\f$ C+=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side scaled multiplication expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a scaled dense
   // tensor-dense tensor multiplication expression to a dense tensor. Due to the explicit
   // application of the SFINAE principle this function can only be selected by the compiler
   // in case either of the two tensor operands requires an intermediate evaluation and no
   // symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline EnableIf_t< IsEvaluationRequired_v<MT,MT1,MT2> >
      smpAddAssign( DenseTensor<MT>& lhs, const DTensScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LeftOperand_t<MMM>  left ( rhs.tensor_.leftOperand()  );
      RightOperand_t<MMM> right( rhs.tensor_.rightOperand() );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || left.columns() == 0UL ) {
         return;
      }

      LT A( left  );  // Evaluation of the left-hand side dense tensor operand
      RT B( right );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == right.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == right.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns(), "Invalid number of columns" );

      smpAddAssign( ~lhs, A * B * rhs.scalar_ );
   }
   //**********************************************************************************************

   //**SMP addition assignment to sparse tensors**************************************************
   // No special implementation for the SMP addition assignment to sparse tensors.
   //**********************************************************************************************

   //**SMP subtraction assignment to dense tensors************************************************
   /*!\brief SMP subtraction assignment of a scaled dense tensor-dense tensor multiplication to a
   //        dense tensor (\f$ C-=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side scaled multiplication expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized SMP subtraction assignment of a scaled
   // dense tensor-dense tensor multiplication expression to a dense tensor. Due to the explicit
   // application of the SFINAE principle this function can only be selected by the compiler
   // in case either of the two tensor operands requires an intermediate evaluation and no
   // symmetry can be exploited.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline EnableIf_t< IsEvaluationRequired_v<MT,MT1,MT2> >
      smpSubAssign( DenseTensor<MT>& lhs, const DTensScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      LeftOperand_t<MMM>  left ( rhs.tensor_.leftOperand()  );
      RightOperand_t<MMM> right( rhs.tensor_.rightOperand() );

      if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL || left.columns() == 0UL ) {
         return;
      }

      LT A( left  );  // Evaluation of the left-hand side dense tensor operand
      RT B( right );  // Evaluation of the right-hand side dense tensor operand

      BLAZE_INTERNAL_ASSERT( A.rows()    == left.rows()     , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( A.columns() == left.columns()  , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( B.rows()    == right.rows()    , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == right.columns() , "Invalid number of columns" );
      BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns(), "Invalid number of columns" );

      smpSubAssign( ~lhs, A * B * rhs.scalar_ );
   }
   //**********************************************************************************************

   //**SMP subtraction assignment to sparse tensors***********************************************
   // No special implementation for the SMP subtraction assignment to sparse tensors.
   //**********************************************************************************************

   //**SMP Schur product assignment to dense tensors**********************************************
   /*!\brief SMP Schur product assignment of a scaled dense tensor-dense tensor multiplication to
   //        a dense tensor (\f$ C\circ=s*A*B \f$).
   // \ingroup dense_tensor
   //
   // \param lhs The target left-hand side dense tensor.
   // \param rhs The right-hand side scaled multiplication expression for the Schur product.
   // \return void
   //
   // This function implements the performance optimized Schur product assignment of a scaled
   // dense tensor-dense tensor multiplication expression to a dense tensor.
   */
   template< typename MT  // Type of the target dense tensor
           , bool SO >    // Storage order of the target dense tensor
   friend inline void smpSchurAssign( DenseTensor<MT>& lhs, const DTensScalarMultExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( ResultType );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
      BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

      const ResultType tmp( rhs );
      smpSchurAssign( ~lhs, tmp );
   }
   //**********************************************************************************************

   //**SMP Schur product assignment to sparse tensors*********************************************
   // No special implementation for the SMP Schur product assignment to sparse tensors.
   //**********************************************************************************************

   //**SMP multiplication assignment to dense tensors*********************************************
   // No special implementation for the SMP multiplication assignment to dense tensors.
   //**********************************************************************************************

   //**SMP multiplication assignment to sparse tensors********************************************
   // No special implementation for the SMP multiplication assignment to sparse tensors.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MMM );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MMM );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_BE_NUMERIC_TYPE( ST );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ST, RightOperand );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Multiplication operator for the multiplication of two row-major dense tensors
//        (\f$ A=B*C \f$).
// \ingroup dense_tensor
//
// \param lhs The left-hand side tensor for the multiplication.
// \param rhs The right-hand side tensor for the multiplication.
// \return The resulting tensor.
// \exception std::invalid_argument Matrix sizes do not match.
//
// This operator represents the multiplication of two row-major dense tensors:

   \code
   using blaze::rowMajor;

   blaze::DynamicMatrix<double,rowMajor> A, B, C;
   // ... Resizing and initialization
   C = A * B;
   \endcode

// The operator returns an expression representing a dense tensor of the higher-order element
// type of the two involved tensor element types \a MT1::ElementType and \a MT2::ElementType.
// Both tensor types \a MT1 and \a MT2 as well as the two element types \a MT1::ElementType
// and \a MT2::ElementType have to be supported by the MultTrait class template.\n
// In case the current number of columns of \a lhs and the current number of rows of \a rhs
// don't match, a \a std::invalid_argument is thrown.
*/
template< typename MT1    // Type of the left-hand side dense tensor
        , typename MT2 >  // Type of the right-hand side dense tensor
inline decltype(auto)
   operator*( const DenseTensor<MT1>& lhs, const DenseTensor<MT2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   if( (~lhs).columns() != (~rhs).rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   using ReturnType = const DTensDTensMultExpr<MT1,MT2>;
   return ReturnType( ~lhs, ~rhs );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================




//=================================================================================================
//
//  SIZE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct Size< DTensDTensMultExpr<MT1,MT2>, 0UL >
   : public Size<MT1,0UL>
{};

template< typename MT1, typename MT2 >
struct Size< DTensDTensMultExpr<MT1,MT2>, 1UL >
   : public Size<MT2,1UL>
{};

template< typename MT1, typename MT2 >
struct Size< DTensDTensMultExpr<MT1,MT2>, 2UL >
   : public Size<MT2,2UL>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISALIGNED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct IsAligned< DTensDTensMultExpr<MT1,MT2> >
   : public BoolConstant< IsAligned_v<MT1> && IsAligned_v<MT2> >
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSYMMETRIC SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct IsSymmetric< DTensDTensMultExpr<MT1,MT2> >
   : public FalseType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISHERMITIAN SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct IsHermitian< DTensDTensMultExpr<MT1,MT2> >
   : public FalseType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISLOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct IsLower< DTensDTensMultExpr<MT1,MT2> >
   : public FalseType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUNILOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct IsUniLower< DTensDTensMultExpr<MT1,MT2> >
   : public FalseType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSTRICTLYLOWER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct IsStrictlyLower< DTensDTensMultExpr<MT1,MT2> >
   : public FalseType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUPPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct IsUpper< DTensDTensMultExpr<MT1,MT2> >
   : public FalseType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISUNIUPPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct IsUniUpper< DTensDTensMultExpr<MT1,MT2> >
   : public FalseType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSTRICTLYUPPER SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT1, typename MT2 >
struct IsStrictlyUpper< DTensDTensMultExpr<MT1,MT2> >
   : public FalseType
{};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
