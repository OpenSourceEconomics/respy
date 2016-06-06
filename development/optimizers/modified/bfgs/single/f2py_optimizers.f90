!******************************************************************************
!******************************************************************************
!   
!   This module serves as the F2PY interface to the core functions. All the 
!   functions have counterparts as PYTHON implementations.
!
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_bfgs(fval, p_final, p_start, func_dim_int)


  USE bfgs_library

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
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------