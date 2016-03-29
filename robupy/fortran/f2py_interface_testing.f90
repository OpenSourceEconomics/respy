!*******************************************************************************
!*******************************************************************************
!
!   This subroutine is just a wrapper for selected functions that are used for
!   testing purposes in the development process.
!
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_slsqp_debug(x_internal, x_start, maxiter, ftol, num_dim)

    !/* external libraries    */

    USE robufort_testing

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: x_internal(num_dim)
    DOUBLE PRECISION, INTENT(IN)    :: x_start(num_dim)
    DOUBLE PRECISION, INTENT(IN)    :: ftol

    INTEGER, INTENT(IN)             :: num_dim
    INTEGER, INTENT(IN)             :: maxiter

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL slsqp_debug(x_internal, x_start, maxiter, ftol, num_dim)

END SUBROUTINE

!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_criterion_debug_function(rslt, x, n)

    !/* external libraries    */

    USE robufort_testing

    !/* external objects    */

    INTEGER, INTENT(IN)             :: n

    DOUBLE PRECISION, INTENT(OUT)   :: rslt
    DOUBLE PRECISION, INTENT(IN)    :: x(n)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

   CALL criterion_debug_function(rslt, x, n)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_criterion_debug_derivative(rslt, x, n)

    !/* external libraries    */

    USE robufort_testing

    !/* external objects    */

    INTEGER, INTENT(IN)             :: n

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(n + 1)
    DOUBLE PRECISION, INTENT(IN)    :: x(n)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL criterion_debug_derivative(rslt, x, n)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************