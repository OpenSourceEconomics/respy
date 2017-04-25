!******************************************************************************
!******************************************************************************
MODULE parallelism_auxiliary

    !/* external modules        */

    USE recording_estimation

    USE recording_warning

    USE optimizers_interfaces

    USE shared_interface

    USE solve_ambiguity

    USE solve_fortran

#if MPI_AVAILABLE

    USE parallelism_constants

    USE mpi

#endif

    !/* setup                   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_solve_parallel(periods_rewards_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, edu_start, edu_max, optim_paras, num_paras, file_sim)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: mapping_state_idx(:, :, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_all(:, :, :)

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_rewards_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(:, :)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)               :: optim_paras

    INTEGER(our_int), INTENT(IN)                    :: num_paras
    INTEGER(our_int), INTENT(IN)                    :: edu_start
    INTEGER(our_int), INTENT(IN)                    :: edu_max

    CHARACTER(225), INTENT(IN)                      :: file_sim

    !/* internal objects        */

    REAL(our_dble), ALLOCATABLE                     :: opt_ambi_details(:, :, :)
    REAL(our_dble)                                  :: x_all_current(num_paras)

    INTEGER(our_int), ALLOCATABLE                   :: num_states_slaves(:, :)
    INTEGER(our_int), ALLOCATABLE                   :: num_obs_slaves(:)

    INTEGER(our_int)                                :: displs(num_slaves)
    INTEGER(our_int)                                :: num_states
    INTEGER(our_int)                                :: period
    INTEGER(our_int)                                :: i
    INTEGER(our_int)                                :: k

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

#if MPI_AVAILABLE

    CALL MPI_Bcast(2, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)


    CALL get_optim_paras(x_all_current, optim_paras, .True.)

    CALL MPI_Bcast(x_all_current, num_paras, MPI_DOUBLE, MPI_ROOT, SLAVECOMM, ierr)


    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, num_periods, edu_start, edu_max, min_idx, num_types)

    CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, edu_start, max_states_period, optim_paras)


    ALLOCATE(periods_emax(num_periods, max_states_period))
    periods_emax = MISSING_FLOAT

    DO period = (num_periods - 1), 0, -1

        num_states = states_number_period(period + 1)

        CALL MPI_RECV(periods_emax(period + 1, :num_states) , num_states, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, SLAVECOMM, status, ierr)

    END DO

    ! I also need the information about the performance of the worst-case optimization.
    IF (.NOT. ALLOCATED(num_states_slaves)) THEN
        CALL distribute_workload(num_states_slaves, num_obs_slaves)
    END IF

    ALLOCATE(opt_ambi_details(num_periods, max_states_period, 7))
    opt_ambi_details = MISSING_FLOAT
    DO period = 1, num_periods
        DO i = 1, num_slaves
            displs(i) = SUM(num_states_slaves(period, :i - 1))
        END DO
        DO k = 1, 7
            CALL MPI_GATHERV(opt_ambi_details(period, :, k), 0, MPI_DOUBLE, opt_ambi_details(period, :, k), num_states_slaves(period, :), displs, MPI_DOUBLE, MPI_ROOT, SLAVECOMM, ierr)
        END DO
    END DO

    IF (optim_paras%level(1) .GT. MIN_AMBIGUITY) CALL record_ambiguity(opt_ambi_details, states_number_period, file_sim, optim_paras)

#endif

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE distribute_workload(num_states_slaves, num_obs_slaves)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(OUT)   :: num_states_slaves(:, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(OUT)   :: num_obs_slaves(:)

    !/* internal objects        */

    INTEGER(our_int)                    :: period

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ALLOCATE(num_states_slaves(num_periods, num_slaves), num_obs_slaves(num_slaves))

    CALL determine_workload(num_obs_slaves, num_obs)

    DO period = 1, num_periods
        CALL determine_workload(num_states_slaves(period, :), states_number_period(period))
    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE determine_workload(jobs_slaves, jobs_total)

    !/* external objects        */

    INTEGER(our_int), INTENT(INOUT)     :: jobs_slaves(num_slaves)

    INTEGER(our_int), INTENT(IN)        :: jobs_total

    !/* internal objects        */

    INTEGER(our_int)                    :: j
    INTEGER(our_int)                    :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    jobs_slaves = zero_int

    j = 1

    DO i = 1, jobs_total

        IF (j .GT. num_slaves) j = 1

        jobs_slaves(j) = jobs_slaves(j) + 1

        j = j + 1

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE distribute_information_slaves(num_states_slaves, period, send_slave, recieve_slaves)

    ! DEVELOPMENT NOTES
    !
    ! The assumed-shape input arguments allow to use this subroutine repeatedly.

    !/* external objects        */

    REAL(our_dble), INTENT(INOUT)       :: recieve_slaves(:)

    REAL(our_dble), INTENT(IN)          :: send_slave(:)

    INTEGER(our_int), INTENT(IN)        :: num_states_slaves(num_periods, num_slaves)
    INTEGER(our_int), INTENT(IN)        :: period

    !/* internal objects        */

    INTEGER(our_int)                    :: rcounts(num_slaves)
    INTEGER(our_int)                    :: scounts(num_slaves)
    INTEGER(our_int)                    :: displs(num_slaves)
    INTEGER(our_int)                    :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
#if MPI_AVAILABLE

    ! Parameterize the communication.
    scounts = num_states_slaves(period + 1, :)
    rcounts = scounts
    DO i = 1, num_slaves
        displs(i) = SUM(scounts(:i - 1))
    END DO

    CALL MPI_ALLGATHERV(send_slave, scounts(rank + 1), MPI_DOUBLE, recieve_slaves, rcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD, ierr)

#endif

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_backward_induction_slave(periods_emax, opt_ambi_details, num_periods, periods_draws_emax, states_number_period, periods_rewards_systematic, mapping_state_idx, states_all, is_debug, is_interpolated, num_points_interp, is_myopic, edu_start, edu_max, ambi_spec, optim_paras, optimizer_options, file_sim, num_states_slaves, update_master)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: opt_ambi_details(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(:, :)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)       :: optim_paras
    TYPE(AMBI_DICT), INTENT(IN)             :: ambi_spec

    REAL(our_dble), INTENT(IN)          :: periods_rewards_systematic(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(IN)          :: periods_draws_emax(num_periods, num_draws_emax, 4)

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2, num_types)
    INTEGER(our_int), INTENT(IN)        :: states_all(num_periods, max_states_period, 5)
    INTEGER(our_int), INTENT(IN)        :: num_states_slaves(num_periods, num_slaves)
    INTEGER(our_int), INTENT(IN)        :: states_number_period(num_periods)
    INTEGER(our_int), INTENT(IN)        :: num_points_interp
    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max

    LOGICAL, INTENT(IN)                 :: is_interpolated
    LOGICAL, INTENT(IN)                 :: update_master
    LOGICAL, INTENT(IN)                 :: is_myopic
    LOGICAL, INTENT(IN)                 :: is_debug

    CHARACTER(225), INTENT(IN)          :: file_sim

    TYPE(optimizer_collection), INTENT(IN)  :: optimizer_options

    !/* internal objects        */

    INTEGER(our_int)                    :: seed_inflated(15)
    INTEGER(our_int)                    :: lower_bound
    INTEGER(our_int)                    :: upper_bound
    INTEGER(our_int)                    :: num_states
    INTEGER(our_int)                    :: seed_size
    INTEGER(our_int)                    :: period
    INTEGER(our_int)                    :: count
    INTEGER(our_int)                    :: info
    INTEGER(our_int)                    :: k
    INTEGER(our_int)                    :: i

    REAL(our_dble)                      :: rewards_systematic(4)
    REAL(our_dble)                      :: shocks_cov(4, 4)
    REAL(our_dble)                      :: shocks_mean(4)
    REAL(our_dble)                      :: shifts(4)
    REAL(our_dble)                      :: emax

    REAL(our_dble)                      :: draws_emax_ambiguity_standard(num_draws_emax, 4)
    REAL(our_dble)                      :: draws_emax_ambiguity_transformed(num_draws_emax, 4)
    REAL(our_dble)                      :: draws_emax_standard(num_draws_emax, 4)
    REAL(our_dble)                      :: draws_emax_risk(num_draws_emax, 4)

    LOGICAL, ALLOCATABLE                :: is_simulated(:)

    LOGICAL                             :: any_interpolated
    LOGICAL                             :: is_head
    LOGICAL                             :: is_write

    REAL(our_dble), ALLOCATABLE         :: periods_emax_slaves(:)
    REAL(our_dble), ALLOCATABLE         :: endogenous_slaves(:)
    REAL(our_dble), ALLOCATABLE         :: exogenous(:, :)
    REAL(our_dble), ALLOCATABLE         :: predictions(:)
    REAL(our_dble), ALLOCATABLE         :: endogenous(:)
    REAL(our_dble), ALLOCATABLE         :: maxe(:)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
#if MPI_AVAILABLE

    IF (.NOT. ALLOCATED(opt_ambi_details)) ALLOCATE(opt_ambi_details(num_periods, max_states_period, 7))
    IF (.NOT. ALLOCATED(periods_emax)) ALLOCATE(periods_emax(num_periods, max_states_period))

    opt_ambi_details = MISSING_FLOAT
    periods_emax = MISSING_FLOAT


    is_head = .False.
    IF(rank == zero_int) is_head = .True.

    is_write = (is_head .AND. update_master)

    IF (is_myopic) THEN
        DO period = (num_periods - 1), 0, -1
            num_states = states_number_period(period + 1)
            periods_emax(period + 1, :num_states) = zero_dble
            IF (is_write) CALL MPI_SEND(periods_emax(period + 1, :num_states), num_states, MPI_DOUBLE, 0, period, PARENTCOMM, ierr)
        END DO
        RETURN
    END IF

    ! Set random seed for interpolation grid.
    seed_inflated(:) = 123

    CALL RANDOM_SEED(size=seed_size)

    CALL RANDOM_SEED(put=seed_inflated)

    ! Construct auxiliary objects
    shocks_cov = MATMUL(optim_paras%shocks_cholesky, TRANSPOSE(optim_paras%shocks_cholesky))

    ! Shifts
    shifts = zero_dble
    CALL clip_value(shifts(1), EXP(shocks_cov(1, 1)/two_dble), zero_dble, HUGE_FLOAT, info)
    CALL clip_value(shifts(2), EXP(shocks_cov(2, 2)/two_dble), zero_dble, HUGE_FLOAT, info)

    ! Initialize containers for disturbances with empty values.
    draws_emax_ambiguity_transformed = MISSING_FLOAT
    draws_emax_ambiguity_standard = MISSING_FLOAT
    draws_emax_risk = MISSING_FLOAT

    shocks_mean = zero_int

    DO period = (num_periods - 1), 0, -1

        ! Extract draws and construct auxiliary objects
        draws_emax_standard = periods_draws_emax(period + 1, :, :)
        num_states = states_number_period(period + 1)

        ! Transform disturbances
        IF (optim_paras%level(1) .GT. MIN_AMBIGUITY) THEN
            IF (ambi_spec%mean) THEN
                DO i = 1, num_draws_emax
                    draws_emax_ambiguity_transformed(i:i, :) = TRANSPOSE(MATMUL(optim_paras%shocks_cholesky, TRANSPOSE(draws_emax_standard(i:i, :))))
                END DO
            ELSE
                draws_emax_ambiguity_standard = draws_emax_standard
            END IF
        ELSE
            CALL transform_disturbances(draws_emax_risk, draws_emax_standard, shocks_mean, optim_paras%shocks_cholesky)
        END IF

        ALLOCATE(periods_emax_slaves(num_states), endogenous_slaves(num_states))

        IF(is_write) CALL record_solution(4, file_sim, period, num_states)

        ! Distinguish case with and without interpolation
        any_interpolated = (num_points_interp .LE. num_states) .AND. is_interpolated

        ! Upper and lower bound of tasks
        lower_bound = SUM(num_states_slaves(period + 1, :rank))
        upper_bound = SUM(num_states_slaves(period + 1, :rank + 1))

        IF (any_interpolated) THEN

            ! Allocate period-specific containers
            ALLOCATE(is_simulated(num_states), endogenous(num_states), maxe(num_states), exogenous(num_states, 9), predictions(num_states))

            ! Constructing indicator for simulation points
            is_simulated = get_simulated_indicator(num_points_interp, num_states, period, is_debug)

            ! Constructing the dependent variable for all states, including the ones where simulation will take place. All information will be used in either the construction of the prediction model or the prediction step.
            CALL get_exogenous_variables(exogenous, maxe, period, num_states, periods_rewards_systematic, shifts, mapping_state_idx, periods_emax, states_all, optim_paras, edu_start, edu_max)

            ! Initialize missing values
            endogenous = MISSING_FLOAT
            endogenous_slaves = MISSING_FLOAT

            ! Construct dependent variables for the subset of interpolation points.
            count = 1
            DO k = lower_bound, upper_bound - 1

                ! Skip over points that will be predicted
                IF (.NOT. is_simulated(k + 1)) THEN
                    count = count + 1
                    CYCLE
                END IF

                ! Extract rewards
                rewards_systematic = periods_rewards_systematic(period + 1, k + 1, :)

                IF (optim_paras%level(1) .GT. MIN_AMBIGUITY) THEN
                    CALL construct_emax_ambiguity(emax, opt_ambi_details, num_periods, num_draws_emax, period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, ambi_spec, optim_paras, optimizer_options)
                ELSE
                    CALL construct_emax_risk(emax, period, k, draws_emax_risk, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, optim_paras)
                END IF

                ! Construct dependent variable
                endogenous_slaves(count) = emax - maxe(k + 1)
                count = count + 1

            END DO

            ! Distribute exogenous information
            CALL distribute_information_slaves(num_states_slaves, period, endogenous_slaves, endogenous)

            ! Create prediction model based on the random subset of points where the EMAX is actually simulated and thus endogenous and exogenous variables are available. For the interpolation  points, the actual values are used.
            CALL get_predictions(predictions, endogenous, exogenous, maxe, is_simulated, num_states, file_sim, is_write)

            ! Store results
            periods_emax(period + 1, :num_states) = predictions

            ! The leading slave updates the master period by period.
            IF (is_write) CALL MPI_SEND(periods_emax(period + 1, :num_states), num_states, MPI_DOUBLE, 0, period, PARENTCOMM, ierr)

            ! Deallocate containers
            DEALLOCATE(is_simulated, exogenous, maxe, endogenous, predictions)

        ELSE

            count =  1
            DO k = lower_bound, upper_bound - 1

                ! Extract rewards
                rewards_systematic = periods_rewards_systematic(period + 1, k + 1, :)

                IF (optim_paras%level(1) .GT. MIN_AMBIGUITY) THEN
                    CALL construct_emax_ambiguity(emax, opt_ambi_details, num_periods, num_draws_emax, period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, ambi_spec, optim_paras, optimizer_options)
                ELSE
                    CALL construct_emax_risk(emax, period, k, draws_emax_risk, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, optim_paras)
                END IF

                ! Collect information
                periods_emax_slaves(count) = emax

                count = count + 1

            END DO

            CALL distribute_information_slaves(num_states_slaves, period, periods_emax_slaves, periods_emax(period + 1, :))

            ! The leading slave updates the master period by period.
            IF (is_write) CALL MPI_SEND(periods_emax(period + 1, :num_states), num_states, MPI_DOUBLE, 0, period, PARENTCOMM, ierr)

        END IF

        DEALLOCATE(periods_emax_slaves, endogenous_slaves)

    END DO

#endif

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
