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
SUBROUTINE fort_simulate(data_sim, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, num_agents_sim, periods_draws_sims, seed_sim, file_sim, edu_spec, optim_paras, num_types, is_debug)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(OUT)    :: data_sim(:, :)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras
    TYPE(EDU_DICT), INTENT(IN)          :: edu_spec

    REAL(our_dble), INTENT(IN)      :: periods_rewards_systematic(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(IN)      :: periods_draws_sims(num_periods, num_agents_sim, 4)
    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 4, num_types)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 5)
    INTEGER(our_int), INTENT(IN)    :: num_agents_sim
    INTEGER(our_int), INTENT(IN)    :: num_types
    INTEGER(our_int), INTENT(IN)    :: seed_sim

    CHARACTER(225), INTENT(IN)      :: file_sim

    LOGICAL, INTENT(IN)             :: is_debug

    !/* internal objects        */

    TYPE(COVARIATES_DICT)           :: covariates

    REAL(our_dble)                  :: periods_draws_sims_transformed(num_periods, num_agents_sim, 4)
    REAL(our_dble)                  :: draws_sims_transformed(num_agents_sim, 4)
    REAL(our_dble)                  :: draws_sims(num_agents_sim, 4)
    REAL(our_dble)                  :: shocks_mean(4) = zero_dble
    REAL(our_dble)                  :: rewards_systematic(4)
    REAL(our_dble)                  :: wages_systematic(2)
    REAL(our_dble)                  :: rewards_ex_post(4)
    REAL(our_dble)                  :: total_values(4)
    REAL(our_dble)                  :: draws(4)

    INTEGER(our_int)                :: edu_start(num_agents_sim, 2)
    INTEGER(our_int)                :: types(num_agents_sim)
    INTEGER(our_int)                :: current_state(5)
    INTEGER(our_int)                :: choice_lagged
    INTEGER(our_int)                :: choice
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: exp_a
    INTEGER(our_int)                :: exp_b
    INTEGER(our_int)                :: count
    INTEGER(our_int)                :: type_
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: k

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL record_simulation(num_agents_sim, seed_sim, file_sim)


    ALLOCATE(data_sim(num_periods * num_agents_sim, 29))

    !Standard deviates transformed to the distributions relevant for the agents actual decision making as traversing the tree.
    DO period = 1, num_periods
        draws_sims = periods_draws_sims(period, :, :)
        CALL transform_disturbances(draws_sims_transformed, draws_sims, shocks_mean, optim_paras%shocks_cholesky)
        periods_draws_sims_transformed(period, :, :) = draws_sims_transformed
    END DO

    ! Initialize containers
    data_sim = MISSING_FLOAT

    ! We also need to sample the set of initial conditions.
    edu_start = get_random_edu_start(edu_spec, is_debug)
    types = get_random_types(num_types, optim_paras, num_agents_sim, edu_start, is_debug)

    ! Iterate over agents and periods
    count = 0

    DO i = 0, (num_agents_sim - 1)

        ! Baseline state
        current_state = states_all(1, 1, :)

        ! We need to modify the initial conditions.
        current_state(3:4) = edu_start(i + 1, :)
        current_state(5) = types(i + 1)


        ! TODO: Remove
        !IF (edu_start(i + 1, 1) < 10) THEN
    !        current_state(4) = 4
!        ELSE
!        current_state(4) = 3
!        END IF

        CALL record_simulation(i, file_sim)

        DO period = 0, (num_periods - 1)

            ! Distribute state space
            exp_a = current_state(1)
            exp_b = current_state(2)
            edu = current_state(3)
            choice_lagged = current_state(4)
            type_ = current_state(5)

            ! Getting state index
            k = mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu + 1, choice_lagged, type_ + 1)

            ! Write agent identifier and current period to data frame
            data_sim(count + 1, 1) = DBLE(i)
            data_sim(count + 1, 2) = DBLE(period)

            ! Calculate ex post rewards
            rewards_systematic = periods_rewards_systematic(period + 1, k + 1, :)
            draws = periods_draws_sims_transformed(period + 1, i + 1, :)

            ! Calculate total utilities
            CALL get_total_values(total_values, rewards_ex_post, period, num_periods, rewards_systematic, draws, mapping_state_idx, periods_emax, k, states_all, optim_paras, edu_spec)

            ! We need to ensure that no individual chooses an inadmissible state. This cannot be done directly in the get_total_values function as the penalty otherwise dominates the interpolation equation. The parameter INADMISSIBILITY_PENALTY is a compromise. It is only relevant in very constructed cases.
            IF (edu >= edu_spec%max) total_values(3) = -HUGE_FLOAT

            ! Determine and record optimal choice
            choice = MAXLOC(total_values, DIM=one_int)

            data_sim(count + 1, 3) = DBLE(choice)

            ! Record wages
            IF ((choice .EQ. one_int) .OR. (choice .EQ. two_int)) THEN
                wages_systematic = back_out_systematic_wages(rewards_systematic, exp_a, exp_b, edu, choice_lagged, optim_paras)
                data_sim(count + 1, 4) = wages_systematic(choice) * draws(choice)
            END IF

            ! Write relevant state space for period to data frame
            data_sim(count + 1, 5:8) = current_state(:4)

            ! As we are working with a simulated dataset, we can also output additional information that is not available in an observed dataset. The discount rate is included as this allows to construct the EMAX with the information provided in the simulation output.
            data_sim(count + 1,  9: 9) = type_
            data_sim(count + 1, 10:13) = total_values
            data_sim(count + 1, 14:17) = rewards_systematic
            data_sim(count + 1, 18:21) = draws
            data_sim(count + 1, 22:22) = optim_paras%delta

            ! For testing purposes, we also explicitly include the general reward component and the common component.
            covariates = construct_covariates(exp_a, exp_b, edu, choice_lagged, type_, period)
            data_sim(count + 1, 23:24) = calculate_rewards_general(covariates, optim_paras)
            data_sim(count + 1, 25:25) = calculate_rewards_common(covariates, optim_paras)
            data_sim(count + 1, 26:29) = rewards_ex_post

            !# Update work experiences or education
            IF ((choice .EQ. one_int) .OR. (choice .EQ. two_int) .OR. (choice .EQ. three_int)) THEN
                current_state(choice) = current_state(choice) + 1
            END IF

            !# Update lagged activity variable.
            current_state(4) = choice

            ! Update row indicator
            count = count + 1

        END DO

    END DO

    CALL record_simulation(file_sim)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE
