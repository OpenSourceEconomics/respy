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
    INTEGER(our_int), ALLOCATABLE   :: num_agents_slaves(:)
    INTEGER(our_int), ALLOCATABLE   :: displs(:)

    INTEGER(our_int)                :: lower_bound_states
    INTEGER(our_int)                :: upper_bound_states
    INTEGER(our_int)                :: lower_bound_obs
    INTEGER(our_int)                :: upper_bound_obs
    INTEGER(our_int)                :: start_agent
    INTEGER(our_int)                :: stop_agent
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: num_procs
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: seed_sim
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: task
    INTEGER(our_int)                :: k

    REAL(our_dble), ALLOCATABLE     :: opt_ambi_details(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: x_optim_all_unscaled(:)
    REAL(our_dble), ALLOCATABLE     :: data_slave(:, :)
    REAL(our_dble), ALLOCATABLE     :: contribs(:)

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


    CALL read_specification(optim_paras, tau, seed_sim, seed_emax, seed_prob, num_procs, num_slaves, is_debug, is_interpolated, num_points_interp, is_myopic, request, exec_dir, maxfun, num_free, edu_spec, precond_spec, ambi_spec, optimizer_used, optimizer_options, file_sim, num_rows, num_paras)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, num_periods, num_types, edu_spec)

    CALL distribute_workload(num_states_slaves, num_agents_slaves)


    CALL create_draws(periods_draws_emax, num_draws_emax, seed_emax, is_debug)

    ALLOCATE(opt_ambi_summary_slaves(2, num_slaves)); opt_ambi_summary_slaves = MISSING_INT
    ALLOCATE(displs(num_slaves)); displs = MISSING_INT
    ALLOCATE(x_optim_all_unscaled(num_paras))

    DO WHILE (STAY_AVAILABLE)

        CALL MPI_Bcast(task, 1, MPI_INT, 0, PARENTCOMM, ierr)

        IF (task == 1) THEN
            CALL MPI_FINALIZE(ierr)
            STAY_AVAILABLE = .FALSE.
            CYCLE
        END IF


        CALL MPI_Bcast(x_optim_all_unscaled, num_paras, MPI_DOUBLE, 0, PARENTCOMM, ierr)

        CALL dist_optim_paras(optim_paras, x_optim_all_unscaled)

        IF(task == 2) THEN

            ! This is required to keep the logging aligned between the scalar and the parallel implementations. We cannot have the master write the log for the state space creation as this interferes with other write requests for the slaves leading to an unreadable file.
            IF (rank == zero_int) THEN
                CALL record_solution(1, file_sim); CALL record_solution(-1, file_sim)
            END IF

            IF (rank == zero_int) CALL record_solution(2, file_sim)

            CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, max_states_period, optim_paras)

            IF (rank == zero_int) CALL record_solution(-1, file_sim)

            IF (rank == zero_int) CALL record_solution(3, file_sim)

            CALL fort_backward_induction_slave(periods_emax, opt_ambi_details, num_periods, periods_draws_emax, states_number_period, periods_rewards_systematic, mapping_state_idx, states_all, is_debug, is_interpolated, num_points_interp, is_myopic, edu_spec, ambi_spec, optim_paras, optimizer_options, file_sim, num_states_slaves, .True.)

            DO period = 1, num_periods

                lower_bound_states = SUM(num_states_slaves(period, :rank)) + 1
                upper_bound_states = SUM(num_states_slaves(period, :rank + 1))

                DO k = 1, 7
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


                CALL read_dataset(data_est, num_rows)

                CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob, is_debug)

                ! Now we need to determine the number precise bounds for the dataset.
                start_agent = SUM(num_agents_slaves(:rank)) + 1
                stop_agent = SUM(num_agents_slaves(:rank + 1))
                num_obs_agent = get_num_obs_agent(data_est)

                lower_bound_obs = SUM(num_obs_agent(:start_agent - 1)) + 1
                upper_bound_obs = SUM(num_obs_agent(:stop_agent))

                ALLOCATE(contribs(num_agents_est))
                ALLOCATE(data_slave(upper_bound_obs - lower_bound_obs + 1, 8))

                data_slave = data_est(lower_bound_obs:upper_bound_obs, :)

            END IF

            CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, max_states_period, optim_paras)

            CALL fort_backward_induction_slave(periods_emax, opt_ambi_details, num_periods, periods_draws_emax, states_number_period, periods_rewards_systematic, mapping_state_idx, states_all, is_debug, is_interpolated, num_points_interp, is_myopic, edu_spec, ambi_spec, optim_paras, optimizer_options, file_sim, num_states_slaves, .False.)

            CALL fort_contributions(contribs(start_agent:stop_agent), periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, data_slave, periods_draws_prob, tau, num_periods, num_draws_prob, num_agents_slaves(rank + 1), num_obs_agent(start_agent:stop_agent), num_types, edu_spec, optim_paras)

            CALL MPI_GATHERV(contribs(start_agent:stop_agent), num_agents_slaves(rank + 1), MPI_DOUBLE, contribs, 0, displs, MPI_DOUBLE, 0, PARENTCOMM, ierr)

            ! We also need to monitor the quality of the worst-case determination. We do not send around detailed information to save on communication time. The details are provided for simulations only.
            CALL summarize_worst_case_success(opt_ambi_summary_slaves(:, rank + 1), opt_ambi_details)

            CALL MPI_GATHER(opt_ambi_summary_slaves(:, rank + 1), 2, MPI_INT, opt_ambi_summary_slaves, 0, MPI_INT, 0, PARENTCOMM, ierr)

        END IF

    END DO

END PROGRAM
!******************************************************************************
!******************************************************************************
