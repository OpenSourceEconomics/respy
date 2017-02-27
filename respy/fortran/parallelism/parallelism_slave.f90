!******************************************************************************
!******************************************************************************
PROGRAM resfort_parallel_slave

    !/* external modules        */

    USE parallelism_constants

    USE parallelism_auxiliary

    USE recording_solution

    USE resfort_library

    USE mpi

    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */
    TYPE(PRECOND_DICT)              :: precond_spec

    INTEGER(our_int), ALLOCATABLE   :: opt_ambi_summary_slaves(:, :)
    INTEGER(our_int), ALLOCATABLE   :: num_states_slaves(:, :)
    INTEGER(our_int), ALLOCATABLE   :: num_obs_slaves(:)
    INTEGER(our_int), ALLOCATABLE   :: displs(:)

    INTEGER(our_int)                :: lower_bound_states
    INTEGER(our_int)                :: upper_bound_states
    INTEGER(our_int)                :: lower_bound_obs
    INTEGER(our_int)                :: upper_bound_obs
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: num_procs
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: seed_sim
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: task
    INTEGER(our_int)                :: k

    REAL(our_dble), ALLOCATABLE     :: opt_ambi_details(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: data_slave(:, :)
    REAL(our_dble), ALLOCATABLE     :: contribs(:)

    REAL(our_dble)                  :: x_optim_all_unscaled(28)
    REAL(our_dble)                  :: precond_minimum

    LOGICAL                         :: STAY_AVAILABLE = .TRUE.

    CHARACTER(225)                  :: optimizer_used
    CHARACTER(225)                  :: file_sim
    CHARACTER(225)                  :: exec_dir
    CHARACTER(10)                   :: request

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL MPI_INIT(ierr)

    CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    CALL MPI_COMM_GET_PARENT(PARENTCOMM, ierr)


    CALL read_specification(optim_paras, edu_start, edu_max, tau, seed_sim, seed_emax, seed_prob, num_procs, num_slaves, is_debug, is_interpolated, num_points_interp, is_myopic, request, exec_dir, maxfun, num_free, precond_spec, ambi_spec, optimizer_used, optimizer_options, file_sim, num_obs)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, num_periods, edu_start, edu_max, min_idx)

    CALL distribute_workload(num_states_slaves, num_obs_slaves)

    CALL create_draws(periods_draws_emax, num_draws_emax, seed_emax, is_debug)

    ALLOCATE(opt_ambi_summary_slaves(2, num_slaves)); opt_ambi_summary_slaves = MISSING_INT
    ALLOCATE(displs(num_slaves)); displs = MISSING_INT


    DO WHILE (STAY_AVAILABLE)

        CALL MPI_Bcast(task, 1, MPI_INT, 0, PARENTCOMM, ierr)

        IF (task == 1) THEN
            CALL MPI_FINALIZE(ierr)
            STAY_AVAILABLE = .FALSE.
            CYCLE
        END IF


        CALL MPI_Bcast(x_optim_all_unscaled, 28, MPI_DOUBLE, 0, PARENTCOMM, ierr)

        CALL dist_optim_paras(optim_paras, x_optim_all_unscaled)

        IF(task == 2) THEN

            ! This is required to keep the logging aligned between the scalar and the parallel implementations. We cannot have the master write the log for the state space creation as this interferes with other write requests for the slaves leading to an unreadable file.
            IF (rank == zero_int) THEN
                CALL record_solution(1, file_sim); CALL record_solution(-1, file_sim)
            END IF

            IF (rank == zero_int) CALL record_solution(2, file_sim)

            CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, edu_start, max_states_period, optim_paras)

            IF (rank == zero_int) CALL record_solution(-1, file_sim)

            IF (rank == zero_int) CALL record_solution(3, file_sim)

            CALL fort_backward_induction_slave(periods_emax, opt_ambi_details, num_periods, periods_draws_emax, states_number_period, periods_rewards_systematic, mapping_state_idx, states_all, is_debug, is_interpolated, num_points_interp, is_myopic, edu_start, edu_max, ambi_spec, optim_paras, optimizer_options, file_sim, num_states_slaves, .True.)

            DO period = 1, num_periods

                lower_bound_states = SUM(num_states_slaves(period, :rank)) + 1
                upper_bound_states = SUM(num_states_slaves(period, :rank + 1))

                DO k = 1, 8
                    CALL MPI_GATHERV(opt_ambi_details(period, lower_bound_states:upper_bound_states, k), num_states_slaves(period, rank + 1), MPI_DOUBLE, opt_ambi_details, 0, displs, MPI_DOUBLE, 0, PARENTCOMM, ierr)
                END DO

            END DO

            IF (rank == zero_int .AND. .NOT. is_myopic) THEN
                CALL record_solution(-1, file_sim)
            ELSEIF (rank == zero_int) THEN
                CALL record_solution(-2, file_sim)
            END IF

        ELSEIF (task == 3) THEN

            IF (.NOT. ALLOCATED(data_est)) THEN

                CALL read_dataset(data_est, num_obs)

                CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob, is_debug)

                ALLOCATE(contribs(num_obs))

                ALLOCATE(data_slave(num_obs_slaves(rank + 1), 8))

                lower_bound_obs = SUM(num_obs_slaves(:rank)) + 1
                upper_bound_obs = SUM(num_obs_slaves(:rank + 1))

                data_slave = data_est(lower_bound_obs:upper_bound_obs, :)

            END IF

            CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, edu_start, max_states_period, optim_paras)

            CALL fort_backward_induction_slave(periods_emax, opt_ambi_details, num_periods, periods_draws_emax, states_number_period, periods_rewards_systematic, mapping_state_idx, states_all, is_debug, is_interpolated, num_points_interp, is_myopic, edu_start, edu_max, ambi_spec, optim_paras, optimizer_options, file_sim, num_states_slaves, .False.)

            CALL fort_contributions(contribs(lower_bound_obs:upper_bound_obs), periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, data_slave, periods_draws_prob, tau, edu_start, edu_max, num_periods, num_draws_prob, optim_paras)

            CALL MPI_GATHERV(contribs(lower_bound_obs:upper_bound_obs), num_obs_slaves(rank + 1), MPI_DOUBLE, contribs, 0, displs, MPI_DOUBLE, 0, PARENTCOMM, ierr)

            ! We also need to monitor the quality of the worst-case determination. We do not send around detailed information to save on communication time. The details are provided for simulations only.
            CALL summarize_worst_case_success(opt_ambi_summary_slaves(:, rank + 1), opt_ambi_details)

            CALL MPI_GATHER(opt_ambi_summary_slaves(:, rank + 1), 2, MPI_INT, opt_ambi_summary_slaves, 0, MPI_INT, 0, PARENTCOMM, ierr)

        END IF

    END DO

END PROGRAM
!******************************************************************************
!******************************************************************************
