!*****************************************************************************
!***************************************************************************** 
PROGRAM resfort_scalar

    !/* external modules        */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: x_start(26)
    REAL(our_dble)                  :: x_final(26)
    REAL(our_dble)                  :: crit_val

    REAL(our_dble), ALLOCATABLE     :: periods_draws_sims(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: data_sim(:, :)

    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: seed_sim

    INTEGER(our_int)                :: bfgs_maxiter
    REAL(our_dble)                  :: bfgs_stpmx
    REAL(our_dble)                  :: bfgs_gtol


    INTEGER(our_int)                :: newuoa_maxfun
    INTEGER(our_int)                :: newuoa_npt
    
    REAL(our_dble)                  :: newuoa_rhobeg
    REAL(our_dble)                  :: newuoa_rhoend    

    INTEGER(our_int)            :: num_procs
    INTEGER(our_int)                :: iter
    LOGICAL                         :: success


    CHARACTER(150)                  :: message

    CHARACTER(225)                  :: optimizer_used

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------


    CALL read_specification(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, delta, tau, seed_sim, seed_emax, seed_prob, num_procs, is_debug, is_interpolated, maxfun, optimizer_used, newuoa_npt, newuoa_maxfun, newuoa_rhobeg, newuoa_rhoend, bfgs_epsilon, bfgs_gtol, bfgs_stpmx, bfgs_maxiter)

    CALL create_draws(periods_draws_emax, num_draws_emax, seed_emax, is_debug)

    ALLOCATE(data_sim(num_periods * num_agents_sim, 8))


    IF (request == 'solve') THEN

        CALL fort_solve(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, periods_draws_emax, delta, is_debug, is_interpolated)

    ELSE IF (request == 'estimate') THEN

        CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob, is_debug)

        CALL read_dataset(data_est, num_agents_est)

        CALL fort_estimate(crit_val, success, message, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, optimizer_used, maxfun, newuoa_npt, newuoa_rhobeg, newuoa_rhoend, newuoa_maxfun, bfgs_gtol, bfgs_maxiter, bfgs_stpmx)

  ELSE IF (request == 'simulate') THEN

        CALL create_draws(periods_draws_sims, num_agents_sim, seed_sim, is_debug)

        CALL fort_solve(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, periods_draws_emax, delta, is_debug, is_interpolated)

        CALL fort_simulate(data_sim, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, num_agents_sim, periods_draws_sims, shocks_cholesky, delta)

    END IF


    CALL store_results(mapping_state_idx, states_all, periods_payoffs_systematic, states_number_period, periods_emax, crit_val, data_sim)

!******************************************************************************
!******************************************************************************
END PROGRAM