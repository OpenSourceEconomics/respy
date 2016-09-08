!*****************************************************************************
!*****************************************************************************
PROGRAM resfort_scalar

    !/* external modules        */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: scaled_minimum
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: level(1)
    REAL(our_dble)                  :: crit_val

    REAL(our_dble), ALLOCATABLE     :: periods_draws_sims(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: data_sim(:, :)

    INTEGER(our_int)                :: num_slaves
    INTEGER(our_int)                :: num_procs
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: seed_sim

    LOGICAL                         :: is_scaled
    LOGICAL                         :: success

    CHARACTER(225)                  :: optimizer_used
    CHARACTER(225)                  :: exec_dir
    CHARACTER(150)                  :: message
    CHARACTER(10)                   :: request

    ! Temporary fix
    REAL(our_dble)                  :: x_tmp(27)
    LOGICAL, PARAMETER              :: all_free(27) = .False.

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL read_specification(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, edu_start, edu_max, delta, tau, seed_sim, seed_emax, seed_prob, num_procs, num_slaves, is_debug, is_interpolated, num_points_interp, is_myopic, request, exec_dir, maxfun, paras_fixed, num_free, is_scaled, scaled_minimum, measure, level, optimizer_used, dfunc_eps, optimizer_options)

    ! This is a temporary fix that aligns the numerical results between the parallel and scalar implementations of the model. Otherwise small numerical differences may arise (if ambiguity is present) as LOG and EXP operations are done in the parallel implementation before any solution or estimation efforts. Due to the two lines below, this is also the case in the scalar impelementation now.
    CALL get_free_optim_paras(x_tmp, level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, all_free)
    CALL dist_optim_paras(level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x_tmp)


    CALL create_draws(periods_draws_emax, num_draws_emax, seed_emax, is_debug)

    IF (request == 'estimate') THEN

        CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob, is_debug)

        CALL read_dataset(data_est, num_agents_est)

        CALL fort_estimate(crit_val, success, message, level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed, optimizer_used, maxfun, is_scaled, scaled_minimum, optimizer_options)

    ELSE IF (request == 'simulate') THEN

        CALL fort_solve(periods_rewards_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, is_interpolated, num_points_interp, num_draws_emax, num_periods, is_myopic, edu_start, is_debug, edu_max, min_idx, delta, periods_draws_emax, measure, level, optimizer_options)

        CALL create_draws(periods_draws_sims, num_agents_sim, seed_sim, is_debug)

        CALL fort_simulate(data_sim, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, num_agents_sim, periods_draws_sims, shocks_cholesky, delta, edu_start, edu_max, seed_sim)

    END IF


    CALL store_results(request, mapping_state_idx, states_all, periods_rewards_systematic, states_number_period, periods_emax, data_sim)

!******************************************************************************
!******************************************************************************
END PROGRAM
