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

CALL NEWUOA (p_final, NPT, RHOBEG, RHOEND, IPRINT, MAXFUN, func_dim_int)   

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


    
    p_final = p_start
    CALL dfpmin(p_final, gtol, iter, fret, criterion_func, criterion_dfunc, func_dim_int)
    fval = fret


END SUBROUTINE