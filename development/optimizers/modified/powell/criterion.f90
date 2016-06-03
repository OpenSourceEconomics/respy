MODULE criterion_function
    !/* external modules        */

    USE shared_variables
    USE nrtype   

     !/* setup                   */

    IMPLICIT NONE

    PUBLIC

CONTAINS

!******************************************************************************
!******************************************************************************
FUNCTION func(x)
    
    USE nrtype

    !/* external objects    */

    REAL(SP)     :: func
    REAL(SP), INTENT(IN)      :: x(:)

    !/* internal objects    */

    INTEGER(I4B)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    func = 0.00


    DO i = 2, func_dim
        func = func + 100.00 * (x(i) - x(i - 1)**2)**2
        func = func + (1 - x(i - 1))**2
    END DO


END FUNCTION
END MODULE