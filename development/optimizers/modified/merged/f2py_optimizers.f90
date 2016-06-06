!******************************************************************************
!******************************************************************************
!   
!   This module serves as the F2PY interface to the core functions. All the 
!   functions have counterparts as PYTHON implementations.
!
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_newuoa(fval, p_final, p_start, func_dim_int)


  USE newuoa_interface

  USE bfgs_function
  USE criterion_function


  INTEGER , INTENT(IN) :: func_dim_int

  DOUBLE PRECISION, INTENT(IN)  :: p_start(func_dim_int)

    DOUBLE PRECISION, INTENT(OUT)      :: fval

  DOUBLE PRECISION, INTENT(OUT)  :: p_final(func_dim_int)

  INTEGER:: NPT, IPRINT, MAXFUN

  DOUBLE PRECISION :: RHOEND, RHOBEG

   IPRINT=0
      MAXFUN=50000
      RHOBEG= MAXVAL(p_start)
      RHOEND=1e-6 *RHOBEG


    
!    p_final = p_start
  NPT=min(func_dim_int * 2, func_dim_int+2)

CALL NEWUOA (criterion_func, p_final, NPT, RHOBEG, RHOEND, IPRINT, MAXFUN)   

!fval = fret


END SUBROUTINE
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
SUBROUTINE f2py_bfgs(fval, p_final, p_start, func_dim_int)


  USE bfgs_function
  USE criterion_function

  INTEGER , INTENT(IN) :: func_dim_int

  DOUBLE PRECISION, INTENT(IN)  :: p_start(func_dim_int)

    DOUBLE PRECISION, INTENT(OUT)      :: fval

  DOUBLE PRECISION, INTENT(OUT)  :: p_final(func_dim_int)

    INTEGER :: iter
    DOUBLE PRECISION:: gtol = 1e-08
    DOUBLE PRECISION :: fret
INTEGER     :: maxiter = 200000000

    
    p_final = p_start
    CALL dfpmin(criterion_func, criterion_dfunc, p_final, gtol, iter, fret, maxiter)
    fval = fret


END SUBROUTINE