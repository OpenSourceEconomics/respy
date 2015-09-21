!*******************************************************************************
!*******************************************************************************
!
!   This subroutine is just a wrapper for selected functions that are used for
!   testing purposes in the development process.
!
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_slsqp_robufort(x_internal, x_start, maxiter, ftol, eps, &
            num_draws, eps_standard, period, k, payoffs_ex_ante, edu_max, &
            edu_start, mapping_state_idx, states_all, num_periods, &
            periods_emax, eps_cholesky, delta, debug, cov, level)

    !/* external libraries    */

    USE robufort_library 
    
    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: x_internal(2)
    DOUBLE PRECISION, INTENT(IN)    :: x_start(2)
    DOUBLE PRECISION, INTENT(IN)    :: cov(4,4)
    DOUBLE PRECISION, INTENT(IN)    :: level
    DOUBLE PRECISION, INTENT(IN)    :: ftol

    INTEGER, INTENT(IN)             :: maxiter

    DOUBLE PRECISION, INTENT(IN)    :: eps_cholesky(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: eps_standard(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: payoffs_ex_ante(:)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: delta
    DOUBLE PRECISION, INTENT(IN)    :: eps

    INTEGER, INTENT(IN)             :: mapping_state_idx(:,:,:,:,:)
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

    LOGICAL, INTENT(IN)             :: debug

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL slsqp_robufort(x_internal, x_start, maxiter, ftol, eps, num_draws, &
            eps_standard, period, k, payoffs_ex_ante, edu_max, edu_start, &
            mapping_state_idx, states_all, num_periods, periods_emax, &
            eps_cholesky, delta, debug, cov, level)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_slsqp_debug(x_internal, x_start, maxiter, &
                ftol, num_dim)

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
SUBROUTINE wrapper_debug_criterion_function(rslt, x, n)

    !/* external libraries    */

    USE robufort_testing

    !/* external objects    */

    INTEGER, INTENT(IN)             :: n

    DOUBLE PRECISION, INTENT(OUT)   :: rslt
    DOUBLE PRECISION, INTENT(IN)    :: x(n)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

   CALL debug_criterion_function(rslt, x, n)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_debug_criterion_derivative(rslt, x, n)

    !/* external libraries    */

    USE robufort_testing

    !/* external objects    */

    INTEGER, INTENT(IN)             :: n

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(n + 1)
    DOUBLE PRECISION, INTENT(IN)    :: x(n)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL debug_criterion_derivative(rslt, x, n)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************