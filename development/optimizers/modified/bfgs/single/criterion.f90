MODULE criterion_function
    !/* external modules        */

    USE shared_constants   

     !/* setup                   */

    IMPLICIT NONE

    PUBLIC

CONTAINS

!******************************************************************************
!******************************************************************************
FUNCTION criterion_func(x)
    

    !/* external objects    */

    REAL(our_dble)     :: criterion_func
    REAL(our_dble), INTENT(IN)      :: x(:)

    !/* internal objects    */

    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    criterion_func = 0.00


    DO i = 2, SIZE(x)
        criterion_func = criterion_func + 100.00 * (x(i) - x(i - 1)**2)**2
        criterion_func = criterion_func + (1 - x(i - 1))**2
    END DO


END FUNCTION
!*******************************************************************************
!*******************************************************************************
FUNCTION criterion_dfunc(x)

    !/* external objects    *

    REAL(our_dble), INTENT(IN)      :: x(:)

    REAL(our_dble)                        :: criterion_dfunc(SIZE(x))

    !/* internals objects    */

    REAL(our_dble)                  :: xm(SIZE(x) - 2)
    REAL(our_dble)                  :: xm_m1(SIZE(x) - 2)
    REAL(our_dble)                  :: xm_p1(SIZE(x) - 2)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Extract sets of evaluation points
    xm = x(2:(SIZE(x) - 1))

    xm_m1 = x(:(SIZE(x) - 2))

    xm_p1 = x(3:)

    ! Construct derivative information
    criterion_dfunc(1) = -400.00 * x(1) * (x(2) - x(1) ** 2) - 2 * (1 - x(1))

    criterion_dfunc(2:(SIZE(x) - 1)) =  (200.00 * (xm - xm_m1 ** 2) - &
            400.00 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm))

    criterion_dfunc(SIZE(x)) = 200.00 * (x(SIZE(x)) - x(SIZE(x) - 1) ** 2)

END FUNCTION
!*******************************************************************************
!*******************************************************************************
END MODULE