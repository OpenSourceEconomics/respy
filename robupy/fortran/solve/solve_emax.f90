!*******************************************************************************
!*******************************************************************************
!
!   This delivers all functions and subroutines to the ROBUFORT library that 
!	are associated with the model under risk. 
!
!*******************************************************************************
!*******************************************************************************
MODULE robufort_emax

	!/*	external modules	    */

    USE robufort_constants

    USE shared_auxiliary

	!/*	setup                   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE simulate_emax(emax_simulated, num_periods, num_draws_emax, period, & 
                k, draws_emax, payoffs_systematic, edu_max, edu_start, & 
                periods_emax, states_all, mapping_state_idx, delta, & 
                shocks_cholesky, shocks_mean)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: emax_simulated

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: shocks_mean(:)

    REAL(our_dble), INTENT(IN)      :: delta

    !/* internals objects    */

    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: draws_emax_transformed(num_draws_emax, 4)
    REAL(our_dble)                  :: total_payoffs(4)
    REAL(our_dble)                  :: draws(4)
    REAL(our_dble)                  :: maximum

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Transform disturbances
    CALL transform_disturbances(draws_emax_transformed, draws_emax, &
            shocks_cholesky, shocks_mean, num_draws_emax)

    ! Iterate over Monte Carlo draws
    emax_simulated = zero_dble
    DO i = 1, num_draws_emax

        ! Select draws for this draw
        draws = draws_emax_transformed(i, :)

        ! Calculate total value
        CALL get_total_value(total_payoffs, &
                period, num_periods, delta, payoffs_systematic, draws, &
                edu_max, edu_start, mapping_state_idx, periods_emax, k, & 
                states_all)
   
        ! Determine optimal choice
        maximum = MAXVAL(total_payoffs)

        ! Recording expected future value
        emax_simulated = emax_simulated + maximum

    END DO

    ! Scaling
    emax_simulated = emax_simulated / num_draws_emax

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE transform_disturbances(draws_emax_transformed, draws_emax, &
                shocks_cholesky, shocks_mean, num_draws_emax)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: draws_emax_transformed(:, :)

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: shocks_mean(:)

    INTEGER, INTENT(IN)             :: num_draws_emax

    !/* internal objects        */

    INTEGER(our_int)                :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    DO i = 1, num_draws_emax
        draws_emax_transformed(i:i, :) = &
            TRANSPOSE(MATMUL(shocks_cholesky, TRANSPOSE(draws_emax(i:i, :))))
    END DO

    draws_emax_transformed(:, :2) = draws_emax_transformed(:, :2) + &
        SPREAD(shocks_mean, 1, num_draws_emax)

    DO i = 1, 2
        draws_emax_transformed(:, i) = EXP(draws_emax_transformed(:, i))
    END DO


END SUBROUTINE

END MODULE