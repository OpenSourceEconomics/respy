!******************************************************************************
!******************************************************************************
!   
!   This module serves as the F2PY interface to the core functions. All the 
!   functions have counterparts as PYTHON implementations.
!
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_powell(fval, p_final, p_start, func_dim_int)

    !/* external libraries      */

   ! USE powell_library
   USE nrtype
   USE powell_library
   USE shared_variables
    !/* setup                   */

    IMPLICIT NONE

    REAL(SP), INTENT(IN)      :: p_start(func_dim_int)
    REAL(SP), INTENT(OUT)      :: fval
    REAL(SP), INTENT(OUT)      :: p_final(func_dim_int)

    INTEGER(I4B), INTENT(IN)  :: func_dim_int

    !/* external objects        */

    REAL(SP), ALLOCATABLE    :: xi(:, :)

    REAL(SP)    :: ftol = 0.0000001, fret
    INTEGER(I4B)   :: iter, i

  !  INTEGER(I4B):: func_dim    
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    

    ! Must be at least two dimensions
    func_dim = func_dim_int


    ALLOCATE(xi(func_dim, func_dim))

    xi = 0.0
    DO i = 1, func_dim
        xi(i, i) = 1.0
    END DO

    p_final = p_start
  CALL powell(p_final,xi,ftol,iter,fret)
  
  fval = fret



END SUBROUTINE
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------