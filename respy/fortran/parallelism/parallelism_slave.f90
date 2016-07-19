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

    INTEGER(our_int), ALLOCATABLE   :: num_emax_slaves(:, :)
    INTEGER(our_int), ALLOCATABLE   :: num_obs_slaves(:)

    INTEGER(our_int)                :: lower_bound
    INTEGER(our_int)                :: upper_bound
    INTEGER(our_int)                :: task

    REAL(our_dble), ALLOCATABLE     :: data_slave(:, :)
    REAL(our_dble), ALLOCATABLE     :: contribs(:)

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: scaled_minimum
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)

    LOGICAL                         :: STAY_AVAILABLE = .TRUE.
    LOGICAL                         :: is_scaled

    REAL(our_dble)                  :: newuoa_rhobeg
    REAL(our_dble)                  :: newuoa_rhoend
    REAL(our_dble)                  :: bfgs_stpmx
    REAL(our_dble)                  :: bfgs_gtol

    INTEGER(our_int)                :: newuoa_maxfun
    INTEGER(our_int)                :: bfgs_maxiter
    INTEGER(our_int)                :: newuoa_npt
    INTEGER(our_int)                :: num_procs
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: seed_sim

    CHARACTER(225)                  :: optimizer_used
    CHARACTER(225)                  :: exec_dir
    CHARACTER(10)                   :: request
    character(len=1024) :: filename
    CHARACTER(55)                   :: today_char
    CHARACTER(55)                   :: now_char
    CHARACTER(155)                  :: val_char
    CHARACTER(50)                   :: tmp_char
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL MPI_INIT(ierr)

    CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    CALL MPI_COMM_GET_PARENT(PARENTCOMM, ierr)

    CALL get_time(today_char, now_char)

    write (filename, "(A5,I2,A10)") 'core-', rank, '.respy.log'
    120 FORMAT(3x,A25,27X,A8)

    OPEN(UNIT=99, FILE=TRIM(filename)); CLOSE(99, STATUS='delete')

    OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')

        WRITE(12, 120) 'Ready to go', now_char
    CLOSE(12)


    CALL read_specification(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, edu_start, edu_max, delta, tau, seed_sim, seed_emax, seed_prob, num_procs, num_slaves, is_debug, is_interpolated, is_myopic, request, exec_dir, maxfun, paras_fixed, num_free, is_scaled, scaled_minimum, optimizer_used, dfunc_eps, newuoa_npt, newuoa_maxfun, newuoa_rhobeg, newuoa_rhoend, bfgs_gtol, bfgs_stpmx, bfgs_maxiter)

    OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')
        CALL get_time(today_char, now_char)
        WRITE(12, 120) 'Stat space', now_char
    CLOSE(12)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, edu_start, edu_max)



    CALL distribute_workload(num_emax_slaves, num_obs_slaves)

    OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')
        CALL get_time(today_char, now_char)
        WRITE(12, 120) 'draws', now_char
    CLOSE(12)

    CALL create_draws(periods_draws_emax, num_draws_emax, seed_emax, is_debug)

    OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')
        CALL get_time(today_char, now_char)
        WRITE(12, 120) 'entering loop', now_char
    CLOSE(12)


    DO WHILE (STAY_AVAILABLE)
      OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')
          CALL get_time(today_char, now_char)
          WRITE(12, 120) 'beginning lopp', now_char
      CLOSE(12)

        CALL MPI_Bcast(task, 1, MPI_INT, 0, PARENTCOMM, ierr)

        IF (task == 1) THEN
            CALL MPI_FINALIZE(ierr)
            STAY_AVAILABLE = .FALSE.
            CYCLE
        END IF


        CALL MPI_Bcast(x_all_current, 26, MPI_DOUBLE, 0, PARENTCOMM, ierr)

        CALL dist_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x_all_current)

        OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')
            CALL get_time(today_char, now_char)
            WRITE(12, 120) 'go for task', now_char
        CLOSE(12)


        IF(task == 2) THEN


            ! This is required to keep the logging aligned between the scalar and the parallel implementations. We cannot have the master write the log for the state space creation as this interferes with other write requests for the slaves leading to an unreadable file.
            IF (rank == zero_int) THEN
                CALL record_solution(1)
                CALL record_solution(-1)
            END IF

            IF (rank == zero_int) CALL record_solution(2)

            CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start)

            IF (rank == zero_int) CALL record_solution(-1)

            IF (rank == zero_int) CALL record_solution(3)

            CALL fort_backward_induction_slave(periods_emax, periods_draws_emax, states_number_period, periods_payoffs_systematic, mapping_state_idx, states_all, shocks_cholesky, delta, is_debug, is_interpolated, is_myopic, edu_start, edu_max, num_emax_slaves, .True.)

            IF (rank == zero_int .AND. .NOT. is_myopic) THEN
                CALL record_solution(-1)
            ELSEIF (rank == zero_int) THEN
                CALL record_solution(-2)
            END IF

        ELSEIF (task == 3) THEN

                  OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')
                  CALL get_time(today_char, now_char)
                      WRITE(12, 120) 'go for task 3', now_char
                  CLOSE(12)

            IF (.NOT. ALLOCATED(data_est)) THEN

              OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')
              CALL get_time(today_char, now_char)

                  WRITE(12, 120) 'Setup data', now_char
              CLOSE(12)

                CALL read_dataset(data_est, num_agents_est)

                CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob, is_debug)

                ALLOCATE(contribs(num_obs_slaves(rank + 1)))

                ALLOCATE(data_slave(num_obs_slaves(rank + 1), 8))

                lower_bound = SUM(num_obs_slaves(:rank)) + 1
                upper_bound = SUM(num_obs_slaves(:rank + 1))

                data_slave = data_est(lower_bound:upper_bound, :)

            END IF


            OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')
              CALL get_time(today_char, now_char)
              WRITE(12, 120) 'Start calc', now_char
            CLOSE(12)

            CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start)

            CALL fort_calculate_payoffs_systematic_slave(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start, num_emax_slaves)

            OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')
              CALL get_time(today_char, now_char)
              WRITE(12, 120) 'Start backw', now_char
            CLOSE(12)

            CALL fort_backward_induction_slave(periods_emax, periods_draws_emax, states_number_period, periods_payoffs_systematic, mapping_state_idx, states_all, shocks_cholesky, delta, is_debug, is_interpolated, is_myopic, edu_start, edu_max, num_emax_slaves, .False.)

            OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')
              CALL get_time(today_char, now_char)
              WRITE(12, 120) 'Start contrib', now_char
            CLOSE(12)

            CALL fort_contributions(contribs, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_slave, periods_draws_prob, delta, tau, edu_start, edu_max)
            OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')
              CALL get_time(today_char, now_char)
              WRITE(12, 120) 'done contrib', now_char
            CLOSE(12)


            CALL MPI_SEND(contribs, num_obs_slaves(rank + 1), MPI_DOUBLE, 0, rank, PARENTCOMM, ierr)

        END IF

        OPEN(UNIT=12, FILE=TRIM(filename), ACCESS='APPEND')

          WRITE(12, *)
          WRITE(12, *)
          WRITE(12, *)

        CLOSE(12)

    END DO

END PROGRAM
!******************************************************************************
!******************************************************************************
