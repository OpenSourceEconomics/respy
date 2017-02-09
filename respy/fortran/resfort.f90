!*****************************************************************************
!*****************************************************************************
PROGRAM resfort_scalar

    !/* external modules        */

    USE resfort_library

#if MPI_AVAILABLE

    USE parallelism_constants

    USE parallelism_auxiliary

    USE mpi

#endif

    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */

    REAL(our_dble)                  :: precond_minimum
    REAL(our_dble)                  :: crit_val

    REAL(our_dble), ALLOCATABLE     :: periods_draws_sims(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: data_sim(:, :)

    INTEGER(our_int)                :: num_procs
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: seed_sim

    LOGICAL                         :: success

    CHARACTER(225)                  :: optimizer_used
    CHARACTER(225)                  :: file_sim
    CHARACTER(225)                  :: exec_dir
    CHARACTER(150)                  :: message
    CHARACTER(10)                   :: request
    CHARACTER(50)                   :: precond_type

    ! Temporary fix
    REAL(our_dble)                  :: x_tmp(27)
    LOGICAL, PARAMETER              :: all_free(27) = .False.

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL read_specification(optim_paras, edu_start, edu_max, tau, seed_sim, seed_emax, seed_prob, num_procs, num_slaves, is_debug, is_interpolated, num_points_interp, is_myopic, request, exec_dir, maxfun, paras_fixed, num_free, precond_type, precond_minimum, measure, optimizer_used, optimizer_options, file_sim, num_obs)

    ! We now distinguish between the scalar and parallel execution.
    IF (num_procs == 1) THEN

        ! This is a temporary fix that aligns the numerical results between the parallel and scalar implementations of the model. Otherwise small numerical differences may arise (if ambiguity is present) as LOG and EXP operations are done in the parallel implementation before any solution or estimation efforts. Due to the two lines below, this is also the case in the scalar impelementation now.
        CALL get_free_optim_paras(x_tmp, optim_paras, all_free)
        CALL dist_optim_paras(optim_paras, x_tmp)

        CALL create_draws(periods_draws_emax, num_draws_emax, seed_emax, is_debug)

        IF (request == 'estimate') THEN

            CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob, is_debug)

            CALL read_dataset(data_est, num_obs)

            CALL fort_estimate(crit_val, success, message, optim_paras, paras_fixed, optimizer_used, maxfun, precond_type, precond_minimum, optimizer_options)

        ELSE IF (request == 'simulate') THEN

            CALL fort_solve(periods_rewards_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, optim_paras, is_interpolated, num_points_interp, num_draws_emax, num_periods, is_myopic, edu_start, is_debug, edu_max, min_idx, periods_draws_emax, measure, optimizer_options, file_sim)

            CALL create_draws(periods_draws_sims, num_agents_sim, seed_sim, is_debug)

            CALL fort_simulate(data_sim, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, num_agents_sim, periods_draws_sims, edu_start, edu_max, seed_sim, file_sim, optim_paras)

        END IF

    ELSE

#if MPI_AVAILABLE

        CALL MPI_INIT(ierr)

        CALL MPI_COMM_SPAWN(TRIM(exec_dir) // '/resfort_slave', MPI_ARGV_NULL, num_slaves, MPI_INFO_NULL, 0, MPI_COMM_WORLD, SLAVECOMM, MPI_ERRCODES_IGNORE, ierr)

        IF (request == 'estimate') THEN

            CALL fort_estimate_parallel(crit_val, success, message, optim_paras, paras_fixed, optimizer_used, maxfun, precond_type, precond_minimum, optimizer_options)

        ELSE IF (request == 'simulate') THEN

            CALL fort_solve_parallel(periods_rewards_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, optim_paras, edu_start, edu_max)

            CALL create_draws(periods_draws_sims, num_agents_sim, seed_sim, is_debug)

            CALL fort_simulate(data_sim, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, num_agents_sim, periods_draws_sims, edu_start, edu_max, seed_sim, file_sim, optim_paras)

        END IF

        CALL MPI_Bcast(1, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
        CALL MPI_FINALIZE(ierr)

#endif

    END IF

    CALL store_results(request, mapping_state_idx, states_all, periods_rewards_systematic, states_number_period, periods_emax, data_sim)

!******************************************************************************
!******************************************************************************
END PROGRAM
