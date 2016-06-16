!******************************************************************************
!******************************************************************************
PROGRAM resfort_parallel

    !/* external modules        */

    USE parallelism_constants

    USE parallelism_auxiliary    

    USE resfort_library 
    
    USE mpi
    
    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */
    
    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: newuoa_rhobeg
    REAL(our_dble)                  :: newuoa_rhoend    
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: bfgs_stpmx
    REAL(our_dble)                  :: bfgs_gtol
    REAL(our_dble)                  :: crit_val

    REAL(our_dble), ALLOCATABLE     :: periods_draws_sims(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: data_sim(:, :)

    INTEGER(our_int)                :: newuoa_maxfun
    INTEGER(our_int)                :: bfgs_maxiter
    INTEGER(our_int)                :: newuoa_npt
    INTEGER(our_int)                :: num_procs
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: seed_sim
 
    LOGICAL                         :: success

    CHARACTER(225)                  :: optimizer_used
    CHARACTER(225)                  :: exec_dir
    CHARACTER(150)                  :: message
    CHARACTER(10)                   :: request

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL MPI_INIT(ierr)


    CALL read_specification(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, edu_start, edu_max, delta, tau, seed_sim, seed_emax, seed_prob, num_procs, is_debug, is_interpolated, is_myopic, request, exec_dir, maxfun, paras_fixed, optimizer_used, newuoa_npt, newuoa_maxfun, newuoa_rhobeg, newuoa_rhoend, bfgs_epsilon, bfgs_gtol, bfgs_stpmx, bfgs_maxiter)

    CALL create_draws(periods_draws_emax, num_draws_emax, seed_emax, is_debug)

    ALLOCATE(data_sim(num_periods * num_agents_sim, 8))

    
    CALL MPI_COMM_SPAWN(TRIM(exec_dir) // '/resfort_parallel_slave', MPI_ARGV_NULL, (num_procs - 1), MPI_INFO_NULL, 0, MPI_COMM_WORLD, SLAVECOMM, MPI_ERRCODES_IGNORE, ierr)

    IF (request == 'solve') THEN

        CALL fort_solve_parallel(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start, edu_max)

    ELSE IF (request == 'estimate') THEN 

        CALL fort_solve_parallel(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start, edu_max)

        CALL fort_evaluate_parallel(crit_val)

    ELSE IF (request == 'simulate') THEN

        CALL create_draws(periods_draws_sims, num_agents_sim, seed_sim, is_debug)

        CALL fort_solve_parallel(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start, edu_max)

        CALL fort_simulate(data_sim, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, num_agents_sim, periods_draws_sims, shocks_cholesky, delta, edu_start, edu_max)

    END IF


    CALL store_results(request, mapping_state_idx, states_all, periods_payoffs_systematic, states_number_period, periods_emax, data_sim)

    CALL MPI_Bcast(1, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
    CALL MPI_FINALIZE (ierr)
    

END PROGRAM
!******************************************************************************
!******************************************************************************
