!*******************************************************************************
!*******************************************************************************
MODULE simulate_fortran

	!/*	external modules	*/

    USE shared_constants

    USE shared_auxiliary

	!/*	setup	*/

    IMPLICIT NONE

    PUBLIC

 CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE fort_simulate(dataset, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, num_agents_sim, periods_draws_sims, shocks_cholesky)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: dataset(:, :)

    REAL(our_dble), INTENT(IN)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), INTENT(IN)      :: periods_draws_sims(:, :, :)
    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: num_agents_sim

    !/* internal objects        */

    REAL(our_dble)                  :: periods_draws_sims_transformed(num_periods, num_agents_sim, 4)

    INTEGER(our_int)                :: current_state(4)
    INTEGER(our_int)                :: edu_lagged
    INTEGER(our_int)                :: choice(1)
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: exp_a
    INTEGER(our_int)                :: exp_b
    INTEGER(our_int)                :: count
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: k

    REAL(our_dble)                  :: payoffs_systematic(4)
    REAL(our_dble)                  :: total_payoffs(4)
    REAL(our_dble)                  :: draws(4)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !Standard deviates transformed to the distributions relevant for
    ! the agents actual decision making as traversing the tree.
    DO period = 1, num_periods
        CALL transform_disturbances(periods_draws_sims_transformed(period, :, :), periods_draws_sims(period, :, :), shocks_cholesky, num_agents_sim)
    END DO

    ! Initialize containers
    dataset = MISSING_FLOAT

    ! Iterate over agents and periods
    count = 0

    DO i = 0, (num_agents_sim - 1)

        ! Baseline state
        current_state = states_all(1, 1, :)

        DO period = 0, (num_periods - 1)

            ! Distribute state space
            exp_a = current_state(1)
            exp_b = current_state(2)
            edu = current_state(3)
            edu_lagged = current_state(4)

            ! Getting state index
            k = mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu + 1, edu_lagged + 1)

            ! Write agent identifier and current period to data frame
            dataset(count + 1, 1) = DBLE(i)
            dataset(count + 1, 2) = DBLE(period)

            ! Calculate ex post payoffs
            payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)
            draws = periods_draws_sims_transformed(period + 1, i + 1, :)

            ! Calculate total utilities
            CALL get_total_value(total_payoffs, period, payoffs_systematic, draws, mapping_state_idx, periods_emax, k, states_all)

            ! Write relevant state space for period to data frame
            dataset(count + 1, 5:8) = current_state

            ! Special treatment for education
            dataset(count + 1, 7) = dataset(count + 1, 7) + edu_start

            ! Determine and record optimal choice
            choice = MAXLOC(total_payoffs)

            dataset(count + 1, 3) = DBLE(choice(1))

            !# Update work experiences and education
            IF (choice(1) .EQ. one_int) THEN
                current_state(1) = current_state(1) + 1
            END IF

            IF (choice(1) .EQ. two_int) THEN
                current_state(2) = current_state(2) + 1
            END IF

            IF (choice(1) .EQ. three_int) THEN
                current_state(3) = current_state(3) + 1
            END IF

            IF (choice(1) .EQ. three_int) THEN
                current_state(4) = one_int
            ELSE
                current_state(4) = zero_int
            END IF

            ! Record earnings
            IF (choice(1) .EQ. one_int) THEN
                dataset(count + 1, 4) = payoffs_systematic(1) * draws(1)
            END IF

            IF (choice(1) .EQ. two_int) THEN
                dataset(count + 1, 4) = payoffs_systematic(2) * draws(2)
            END IF

            ! Update row indicator
            count = count + 1

        END DO

    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE