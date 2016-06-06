!******************************************************************************
!******************************************************************************
!   
!   This module serves as the F2PY interface to the core functions. All the 
!   functions have counterparts as PYTHON implementations.
!
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_newuoa(fval, p_final, p_start, func_dim_int)


  USE newuoa_module

  USE criterion_function


  INTEGER , INTENT(IN) :: func_dim_int

  DOUBLE PRECISION, INTENT(IN)  :: p_start(func_dim_int)

    DOUBLE PRECISION, INTENT(OUT)      :: fval

  DOUBLE PRECISION, INTENT(OUT)  :: p_final(func_dim_int)

  INTEGER:: NPT, IPRINT, MAXFUN

  DOUBLE PRECISION :: RHOEND, RHOBEG

  CHARACTER(150):: MESSAGE
  LOGICAL :: SUCCESS

   IPRINT=0
      MAXFUN=50000
      RHOBEG= MAXVAL(p_start)
      RHOEND=1e-6 *RHOBEG


    
    p_final = p_start
  NPT=min(func_dim_int * 2, func_dim_int+2)

  CALL NEWUOA (criterion_func, p_final, NPT, RHOBEG, RHOEND, IPRINT, MAXFUN, SUCCESS, MESSAGE)   

  fval = criterion_func(p_final)


END SUBROUTINE
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
SUBROUTINE f2py_bfgs(fval, p_final, p_start, func_dim_int)


  USE dfpmin_module
  USE criterion_function

  INTEGER , INTENT(IN) :: func_dim_int

  DOUBLE PRECISION, INTENT(IN)  :: p_start(func_dim_int)

    DOUBLE PRECISION, INTENT(OUT)      :: fval

  DOUBLE PRECISION, INTENT(OUT)  :: p_final(func_dim_int)

    INTEGER :: iter
    DOUBLE PRECISION:: gtol = 1e-08, stpmx = 100.0_our_dble
INTEGER     :: maxiter = 200


  CHARACTER(150):: message
  LOGICAL :: success
    
  

    p_final = p_start

    CALL dfpmin(criterion_func, criterion_dfunc, p_final, gtol, iter, maxiter, stpmx, success, message)

    fval = criterion_func(p_final)

END SUBROUTINE