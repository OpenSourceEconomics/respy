!******************************************************************************
!******************************************************************************
!   
!   This module serves as the F2PY interface to the core functions. All the 
!   functions have counterparts as PYTHON implementations.
!
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_newuoa(fval, p_final, p_start, maxfun, rhobeg, rhoend, npt, func_dim_int)


  USE newuoa_module

  USE criterion_function


  INTEGER , INTENT(IN) :: func_dim_int, maxfun, npt

  DOUBLE PRECISION, INTENT(IN)  :: p_start(func_dim_int), rhobeg, rhoend

    DOUBLE PRECISION, INTENT(OUT)      :: fval

  DOUBLE PRECISION, INTENT(OUT)  :: p_final(func_dim_int)


  CHARACTER(150):: message
  LOGICAL :: success
  INTEGER   :: iter
  
  


    
  p_final = p_start

  CALL NEWUOA (criterion_func, p_final, npt, rhobeg, rhoend, 0, maxfun, success, message, iter)   

  fval = criterion_func(p_final)


END SUBROUTINE
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
SUBROUTINE f2py_bfgs(fval, p_final, p_start, gtol, maxiter, stpmx, func_dim_int)

  USE dfpmin_module
  USE criterion_function

  INTEGER , INTENT(IN) :: func_dim_int, maxiter

  DOUBLE PRECISION, INTENT(IN)  :: p_start(func_dim_int), stpmx

  DOUBLE PRECISION, INTENT(IN)  :: gtol


    DOUBLE PRECISION, INTENT(OUT)      :: fval

  DOUBLE PRECISION, INTENT(OUT)  :: p_final(func_dim_int)

    INTEGER :: iter

  CHARACTER(150):: message
  LOGICAL :: success
    
    ! DEAL WITH ITER IN 
  

    p_final = p_start

    CALL dfpmin(criterion_func, criterion_dfunc, p_final, gtol, maxiter, stpmx, success, message, iter)

    fval = criterion_func(p_final)

END SUBROUTINE