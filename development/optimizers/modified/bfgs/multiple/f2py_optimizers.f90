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

  INTEGER(I4B), INTENT(IN) :: func_dim_int

  REAL(SP), INTENT(IN)  :: p_start(func_dim_int)

    REAL(SP), INTENT(OUT)      :: fval

  REAL(SP), INTENT(OUT)  :: p_final(func_dim_int)

    INTEGER(I4B):: iter
    REAL(SP):: gtol = 1e-08
    REAL(SP) :: fret


      

     func_dim = SIZE(p_start) 

    p_final = p_start
    CALL dfpmin(p_final, gtol, iter, fret, criterion_func, criterion_dfunc)
    fval = fret


END SUBROUTINE
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------