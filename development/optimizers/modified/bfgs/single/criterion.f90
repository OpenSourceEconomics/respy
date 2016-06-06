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
FUNCTION criterion_func(x)
    
    USE nrtype

    !/* external objects    */

    REAL(SP)     :: criterion_func
    REAL(SP), INTENT(IN)      :: x(:)

    !/* internal objects    */

    INTEGER(I4B)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    criterion_func = 0.00


    DO i = 2, func_dim
        criterion_func = criterion_func + 100.00 * (x(i) - x(i - 1)**2)**2
        criterion_func = criterion_func + (1 - x(i - 1))**2
    END DO


END FUNCTION
!*******************************************************************************
!*******************************************************************************
FUNCTION criterion_dfunc(x)

    !/* external objects    *

    REAL(SP), INTENT(IN)      :: x(:)

    REAL(SP)                        :: criterion_dfunc(SIZE(x))

    !/* internals objects    */

    REAL(SP)                  :: xm(func_dim - 2)
    REAL(SP)                  :: xm_m1(func_dim - 2)
    REAL(SP)                  :: xm_p1(func_dim - 2)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Extract sets of evaluation points
    xm = x(2:(func_dim - 1))

    xm_m1 = x(:(func_dim - 2))

    xm_p1 = x(3:)

    ! Construct derivative information
    criterion_dfunc(1) = -400.00 * x(1) * (x(2) - x(1) ** 2) - 2 * (1 - x(1))

    criterion_dfunc(2:(func_dim - 1)) =  (200.00 * (xm - xm_m1 ** 2) - &
            400.00 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm))

    criterion_dfunc(func_dim) = 200.00 * (x(func_dim) - x(func_dim - 1) ** 2)

END FUNCTION
!*******************************************************************************
!*******************************************************************************
END MODULE