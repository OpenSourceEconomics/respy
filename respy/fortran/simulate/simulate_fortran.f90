!******************************************************************************
!******************************************************************************
MODULE simulate_fortran

    !/*	external modules	*/

    USE recording_simulation

    USE simulate_auxiliary

    USE shared_interface

    !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

 CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_simulate(data_sim, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, num_agents_sim, periods_draws_sims, edu_start, edu_max, seed_sim, file_sim, optim_paras, num_types, type_spec, is_debug)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(OUT)    :: data_sim(:, :)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras
    TYPE(TYPE_DICT), INTENT(IN)         :: type_spec

    REAL(our_dble), INTENT(IN)      :: periods_rewards_systematic(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(IN)      :: periods_draws_sims(num_periods, num_agents_sim, 4)
    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2, num_types)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 5)
    INTEGER(our_int), INTENT(IN)    :: num_agents_sim
    INTEGER(our_int), INTENT(IN)    :: num_types
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: seed_sim
    INTEGER(our_int), INTENT(IN)    :: edu_max

    CHARACTER(225), INTENT(IN)      :: file_sim

    LOGICAL, INTENT(IN)             :: is_debug

    !/* internal objects        */

    REAL(our_dble)                  :: periods_draws_sims_transformed(num_periods, num_agents_sim, 4)
    REAL(our_dble)                  :: draws_sims_transformed(num_agents_sim, 4)
    REAL(our_dble)                  :: draws_sims(num_agents_sim, 4)
    REAL(our_dble)                  :: shocks_mean(4) = zero_dble

    INTEGER(our_int)                :: current_state(5)
    INTEGER(our_int)                :: edu_lagged
    INTEGER(our_int)                :: choice(1)
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: exp_a
    INTEGER(our_int)                :: exp_b
    INTEGER(our_int)                :: count
    INTEGER(our_int)                :: type_
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: types(num_agents_sim)

    REAL(our_dble)                  :: rewards_systematic(4)
    REAL(our_dble)                  :: total_values(4)
    REAL(our_dble)                  :: draws(4)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL record_simulation(num_agents_sim, seed_sim, file_sim)


    ALLOCATE(data_sim(num_periods * num_agents_sim, 22))

    !Standard deviates transformed to the distributions relevant for the agents actual decision making as traversing the tree.
    DO period = 1, num_periods
        draws_sims = periods_draws_sims(period, :, :)
        CALL transform_disturbances(draws_sims_transformed, draws_sims, shocks_mean, optim_paras%shocks_cholesky)
        periods_draws_sims_transformed(period, :, :) = draws_sims_transformed
    END DO

    ! Initialize containers
    data_sim = MISSING_FLOAT

    ! We also need to sample the set of initial conditions.
    types = get_random_types(num_types, type_spec, num_agents_sim, is_debug)

    ! Iterate over agents and periods
    count = 0

    DO i = 0, (num_agents_sim - 1)

        ! Baseline state
        current_state = states_all(1, 1, :)

        ! We need to modify the initial conditions.
        current_state(5) = types(i + 1)

        CALL record_simulation(i, file_sim)

        DO period = 0, (num_periods - 1)

            ! Distribute state space
            exp_a = current_state(1)
            exp_b = current_state(2)
            edu = current_state(3)
            edu_lagged = current_state(4)
            type_ = current_state(5)

            ! Getting state index
            k = mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu + 1, edu_lagged + 1, type_ + 1)

            ! Write agent identifier and current period to data frame
            data_sim(count + 1, 1) = DBLE(i)
            data_sim(count + 1, 2) = DBLE(period)

            ! Calculate ex post rewards
            rewards_systematic = periods_rewards_systematic(period + 1, k + 1, :)
            draws = periods_draws_sims_transformed(period + 1, i + 1, :)

            ! Calculate total utilities
            CALL get_total_values(total_values, period, num_periods, rewards_systematic, draws, mapping_state_idx, periods_emax, k, states_all, optim_paras, edu_start, edu_max)

            ! Write relevant state space for period to data frame
            data_sim(count + 1, 5:8) = current_state(:4)

            ! Special treatment for education
            data_sim(count + 1, 7) = data_sim(count + 1, 7) + edu_start

            ! As we are working with a simulated dataset, we can also output additional information that is not available in an observed dataset. The discount rate is included as this allows to construct the EMAX with the information provided in the simulation output.
            data_sim(count + 1,  9: 9) = type_
            data_sim(count + 1, 10:13) = total_values
            data_sim(count + 1, 14:17) = rewards_systematic
            data_sim(count + 1, 18:21) = draws
            data_sim(count + 1, 22:22) = optim_paras%delta

            ! Determine and record optimal choice
            choice = MAXLOC(total_values)

            data_sim(count + 1, 3) = DBLE(choice(1))

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

            ! Record wages
            IF (choice(1) .EQ. one_int) THEN
                data_sim(count + 1, 4) = rewards_systematic(1) * draws(1)
            END IF

            IF (choice(1) .EQ. two_int) THEN
                data_sim(count + 1, 4) = rewards_systematic(2) * draws(2)
            END IF

            ! Update row indicator
            count = count + 1

        END DO

    END DO

    CALL record_simulation(file_sim)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE
