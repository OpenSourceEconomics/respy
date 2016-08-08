!******************************************************************************
!******************************************************************************
MODULE parallelism_auxiliary

    !/* external modules        */

    USE parallelism_constants

    USE resfort_library

    USE mpi

    !/* setup                   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE get_scales_parallel(auto_scales, x_free_start, scaled_minimum)

    !/* external objects    */

    REAL(our_dble), ALLOCATABLE, INTENT(OUT)     :: auto_scales(:, :)

    REAL(our_dble), INTENT(IN)                   :: x_free_start(:)
    REAL(our_dble), INTENT(IN)                   :: scaled_minimum

    !/* internal objects    */

    REAL(our_dble)                  :: grad(num_free)
    REAL(our_dble)                  :: val

    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    crit_estimation = .False.

    ALLOCATE(auto_scales(num_free, num_free))

    CALL record_estimation(auto_scales, x_free_start, .True.)

    grad = fort_dcriterion_parallel(x_free_start)

    auto_scales = zero_dble

    DO i = 1, num_free

        val = grad(i)

        IF (ABS(val) .LT. scaled_minimum) val = scaled_minimum

        auto_scales(i, i) = val

    END DO

    CALL record_estimation(auto_scales, x_free_start, .False.)

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

    ! Parameterize the communication.
    scounts = num_states_slaves(period + 1, :)
    rcounts = scounts
    DO i = 1, num_slaves
        displs(i) = SUM(scounts(:i - 1))
    END DO

    CALL MPI_ALLGATHERV(send_slave, scounts(rank + 1), MPI_DOUBLE, recieve_slaves, rcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD, ierr)

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

    CALL determine_workload(num_obs_slaves, (num_agents_est * num_periods))

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
SUBROUTINE fort_estimate_parallel(crit_val, success, message, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed, optimizer_used, maxfun, is_scaled, scaled_minimum, newuoa_npt, newuoa_rhobeg, newuoa_rhoend, newuoa_maxfun, bfgs_gtol, bfgs_maxiter, bfgs_stpmx)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: crit_val

    CHARACTER(150), INTENT(OUT)     :: message

    LOGICAL, INTENT(OUT)            :: success

    INTEGER(our_int), INTENT(IN)    :: newuoa_maxfun
    INTEGER(our_int), INTENT(IN)    :: newuoa_npt
    INTEGER(our_int), INTENT(IN)    :: maxfun
    INTEGER(our_int), INTENT(IN)    :: bfgs_maxiter

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)      :: scaled_minimum
    REAL(our_dble), INTENT(IN)      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)      :: newuoa_rhobeg
    REAL(our_dble), INTENT(IN)      :: newuoa_rhoend
    REAL(our_dble), INTENT(IN)      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)      :: coeffs_b(6)
    REAL(our_dble), INTENT(IN)      :: bfgs_stpmx
    REAL(our_dble), INTENT(IN)      :: bfgs_gtol

    CHARACTER(225), INTENT(IN)      :: optimizer_used

    LOGICAL, INTENT(IN)             :: paras_fixed(26)
    LOGICAL, INTENT(IN)             :: is_scaled

    !/* internal objects    */

    REAL(our_dble)                  :: x_free_start(COUNT(.not. paras_fixed))
    REAL(our_dble)                  :: x_free_final(COUNT(.not. paras_fixed))
    REAL(our_dble)                  :: x_all_final(26)

    INTEGER(our_int)                :: iter

    LOGICAL, PARAMETER              :: all_free(26) = .False.

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Some ingredients for the evaluation of the criterion function need to be created once and shared globally.
    CALL get_free_optim_paras(x_all_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, all_free)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, edu_start, edu_max)

    CALL get_free_optim_paras(x_free_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed)

    ! If a scaling of the criterion function is requested, then we determine the scaled and transform the starting values. Also, the boolean indicates that inside the criterion function the scaling is undone.
    IF (is_scaled .AND. (.NOT. maxfun == zero_int)) THEN

        CALL get_scales_parallel(auto_scales, x_free_start, scaled_minimum)

        x_free_start = apply_scaling(x_free_start, auto_scales, 'do')

        crit_scaled = .True.

    END IF


    crit_estimation = .True.

    IF (maxfun == zero_int) THEN

        success = .True.
        message = 'Single evaluation of criterion function at starting values.'

        x_free_final = x_free_start

    ELSEIF (optimizer_used == 'FORT-NEWUOA') THEN

        CALL newuoa(fort_criterion_parallel, x_free_start, newuoa_npt, newuoa_rhobeg, newuoa_rhoend, zero_int, MIN(maxfun, newuoa_maxfun), success, message, iter)

    ELSEIF (optimizer_used == 'FORT-BFGS') THEN

        CALL dfpmin(fort_criterion_parallel, fort_dcriterion_parallel, x_free_start, bfgs_gtol, bfgs_maxiter, bfgs_stpmx, maxfun, success, message, iter)

    END IF

    crit_estimation = .False.

    ! If scaling is requested, then we transform the resulting parameter vector and indicate that the critterion function is to be used with the actual parameters again.
    IF (is_scaled .AND. (.NOT. maxfun == zero_int)) THEN

        crit_scaled = .False.

        x_free_final = apply_scaling(x_free_start, auto_scales, 'undo')

    ELSE

        x_free_final = x_free_start

    END IF


    crit_val = fort_criterion_parallel(x_free_final)

    CALL construct_all_current_values(x_all_final, x_free_final, paras_fixed)

    CALL record_estimation(success, message, crit_val, x_all_final)

    CALL record_estimation()

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION fort_criterion_parallel(x)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)      :: x(:)
    REAL(our_dble)                  :: fort_criterion_parallel

    !/* internal objects    */

    REAL(our_dble)                  :: contribs(num_agents_est * num_periods)
    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: x_input(num_free)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)

    INTEGER(our_int), ALLOCATABLE   :: num_states_slaves(:, :)
    INTEGER(our_int), ALLOCATABLE   :: num_obs_slaves(:)

    INTEGER(our_int)                :: dist_optim_paras_info
    INTEGER(our_int)                :: displs(num_slaves)
    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Ensuring that the criterion function is not evaluated more than specified. However, there is the special request of MAXFUN equal to zero which needs to be allowed.
    IF ((num_eval == maxfun) .AND. crit_estimation .AND. (.NOT. maxfun == zero_int)) THEN
        fort_criterion_parallel = -HUGE_FLOAT
        RETURN
    END IF

    ! Undo the scaling (if required)
    IF (crit_scaled) THEN
        x_input = apply_scaling(x, auto_scales, 'undo')
    ELSE
        x_input = x
    END IF


    CALL construct_all_current_values(x_all_current, x_input, paras_fixed)

    CALL MPI_Bcast(3, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)

    CALL MPI_Bcast(x_all_current, 26, MPI_DOUBLE, MPI_ROOT, SLAVECOMM, ierr)

    ! This extra work is only required to align the logging across the scalar and parallel implementation. In the case of an otherwise zero variance, we stabilize the algorithm. However, we want this indicated as a warning in the log file.
    CALL dist_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x_all_current, dist_optim_paras_info)

    ! We need to know how the workload is distributed across the slaves.
    IF (.NOT. ALLOCATED(num_states_slaves)) THEN
        CALL distribute_workload(num_states_slaves, num_obs_slaves)

        DO i = 1, num_slaves
            displs(i) = SUM(num_obs_slaves(:i - 1))
        END DO

    END IF

    contribs = -HUGE_FLOAT

    CALL MPI_GATHERV(contribs, 0, MPI_DOUBLE, contribs, num_obs_slaves, displs, MPI_DOUBLE, MPI_ROOT, SLAVECOMM, ierr)

    fort_criterion_parallel = get_log_likl(contribs, num_agents_est, num_periods)

    IF (crit_estimation .OR. (maxfun == zero_int)) THEN

        num_eval = num_eval + 1

        CALL record_estimation(x_all_current, fort_criterion_parallel, num_eval)

        IF (dist_optim_paras_info .NE. zero_int) CALL record_warning(4)

    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION fort_dcriterion_parallel(x)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: x(:)
    REAL(our_dble)                  :: fort_dcriterion_parallel(SIZE(x))

    !/* internals objects       */

    REAL(our_dble)                  :: ei(COUNT(.NOT. paras_fixed))
    REAL(our_dble)                  :: d(COUNT(.NOT. paras_fixed))
    REAL(our_dble)                  :: f0
    REAL(our_dble)                  :: f1

    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    ei = zero_dble

    ! Evaluate baseline
    f0 = fort_criterion_parallel(x)

    DO j = 1, COUNT(.NOT. paras_fixed)

        ei(j) = one_dble

        d = dfunc_eps * ei

        f1 = fort_criterion_parallel(x + d)

        fort_dcriterion_parallel(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_solve_parallel(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, edu_start, edu_max)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_all(:, :, :)

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(:, :)

    REAL(our_dble), INTENT(IN)                      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)                      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)                      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)                      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)                      :: coeffs_b(6)

    INTEGER(our_int), INTENT(IN)                    :: edu_start
    INTEGER(our_int), INTENT(IN)                    :: edu_max

    !/* internal objects        */

    REAL(our_dble)                                  :: x_all_current(26)

    INTEGER(our_int)                                :: num_states
    INTEGER(our_int)                                :: period

    LOGICAL, PARAMETER                              :: all_free(26) = .False.

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL MPI_Bcast(2, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)


    CALL get_free_optim_paras(x_all_current, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, all_free)

    CALL MPI_Bcast(x_all_current, 26, MPI_DOUBLE, MPI_ROOT, SLAVECOMM, ierr)


    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, edu_start, edu_max)

    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start)


    ALLOCATE(periods_emax(num_periods, max_states_period))
    periods_emax = MISSING_FLOAT

    DO period = (num_periods - 1), 0, -1

        num_states = states_number_period(period + 1)

        CALL MPI_RECV(periods_emax(period + 1, :num_states) , num_states, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, SLAVECOMM, status, ierr)

    END DO


END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_backward_induction_slave(periods_emax, periods_draws_emax, states_number_period, periods_payoffs_systematic, mapping_state_idx, states_all, shocks_cholesky, delta, is_debug, is_interpolated, is_myopic, edu_start, edu_max, num_states_slaves, update_master)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)       :: periods_emax(:, :)

    REAL(our_dble), INTENT(IN)          :: periods_payoffs_systematic(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(IN)          :: periods_draws_emax(num_periods, num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)          :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)          :: delta

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)        :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)        :: num_states_slaves(num_periods, num_slaves)
    INTEGER(our_int), INTENT(IN)        :: states_number_period(num_periods)
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max

    LOGICAL, INTENT(IN)                 :: is_interpolated
    LOGICAL, INTENT(IN)                 :: update_master
    LOGICAL, INTENT(IN)                 :: is_myopic
    LOGICAL, INTENT(IN)                 :: is_debug

    !/* internal objects        */

    INTEGER(our_int)                :: seed_inflated(15)
    INTEGER(our_int)                :: lower_bound
    INTEGER(our_int)                :: upper_bound
    INTEGER(our_int)                :: num_states
    INTEGER(our_int)                :: seed_size
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: count
    INTEGER(our_int)                :: info
    INTEGER(our_int)                :: k

    REAL(our_dble)                  :: payoffs_systematic(4)
    REAL(our_dble)                  :: shocks_cov(4, 4)
    REAL(our_dble)                  :: emax_simulated
    REAL(our_dble)                  :: shifts(4)

    REAL(our_dble)                  :: draws_emax_transformed(num_draws_emax, 4)
    REAL(our_dble)                  :: draws_emax(num_draws_emax, 4)

    LOGICAL, ALLOCATABLE            :: is_simulated(:)

    LOGICAL                         :: any_interpolated
    LOGICAL                         :: is_head

    REAL(our_dble), ALLOCATABLE     :: periods_emax_slaves(:)
    REAL(our_dble), ALLOCATABLE     :: endogenous_slaves(:)
    REAL(our_dble), ALLOCATABLE     :: exogenous(:, :)
    REAL(our_dble), ALLOCATABLE     :: predictions(:)
    REAL(our_dble), ALLOCATABLE     :: endogenous(:)
    REAL(our_dble), ALLOCATABLE     :: maxe(:)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF (.NOT. ALLOCATED(periods_emax)) THEN
        ALLOCATE(periods_emax(num_periods, max_states_period))
    END IF

    periods_emax = MISSING_FLOAT


    is_head = .False.
    IF(rank == zero_int) is_head = .True.

    IF (is_myopic) THEN
        DO period = (num_periods - 1), 0, -1
            num_states = states_number_period(period + 1)
            periods_emax(period + 1, :num_states) = zero_dble
            IF (is_head .AND. update_master) CALL MPI_SEND(periods_emax(period + 1, :num_states), num_states, MPI_DOUBLE, 0, period, PARENTCOMM, ierr)
        END DO
        RETURN
    END IF

    ! Set random seed for interpolation grid.
    seed_inflated(:) = 123

    CALL RANDOM_SEED(size=seed_size)

    CALL RANDOM_SEED(put=seed_inflated)

    ! Construct auxiliary objects
    shocks_cov = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))

    ! Shifts
    shifts = zero_dble
    CALL clip_value(shifts(1), EXP(shocks_cov(1, 1)/two_dble), zero_dble, HUGE_FLOAT, info)
    CALL clip_value(shifts(2), EXP(shocks_cov(2, 2)/two_dble), zero_dble, HUGE_FLOAT, info)


    DO period = (num_periods - 1), 0, -1

        ! Extract draws and construct auxiliary objects
        draws_emax = periods_draws_emax(period + 1, :, :)
        num_states = states_number_period(period + 1)

        ! Transform disturbances
        CALL transform_disturbances(draws_emax_transformed, draws_emax, shocks_cholesky, num_draws_emax)

        ALLOCATE(periods_emax_slaves(num_states), endogenous_slaves(num_states))

        IF(is_head .AND. update_master) CALL record_solution(4, period, num_states)

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
            CALL get_exogenous_variables(exogenous, maxe, period, num_states, periods_payoffs_systematic, shifts, mapping_state_idx, periods_emax, states_all, delta, edu_start, edu_max)

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

                ! Extract payoffs
                payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

                ! Get payoffs
                CALL get_future_value(emax_simulated, draws_emax_transformed, period, k, payoffs_systematic, mapping_state_idx, states_all, periods_emax, delta, edu_start, edu_max)

                ! Construct dependent variable
                endogenous_slaves(count) = emax_simulated - maxe(k + 1)
                count = count + 1

            END DO

            ! Distribute exogenous information
            CALL distribute_information_slaves(num_states_slaves, period, endogenous_slaves, endogenous)

            ! Create prediction model based on the random subset of points where the EMAX is actually simulated and thus endogenous and exogenous variables are available. For the interpolation  points, the actual values are used.
            CALL get_predictions(predictions, endogenous, exogenous, maxe, is_simulated, num_states, is_head .AND. update_master)

            ! Store results
            periods_emax(period + 1, :num_states) = predictions

            ! The leading slave updates the master period by period.
            IF (is_head .AND. update_master) CALL MPI_SEND(periods_emax(period + 1, :num_states), num_states, MPI_DOUBLE, 0, period, PARENTCOMM, ierr)

            ! Deallocate containers
            DEALLOCATE(is_simulated, exogenous, maxe, endogenous, predictions)

        ELSE

            count =  1
            DO k = lower_bound, upper_bound - 1

                ! Extract payoffs
                payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

                CALL get_future_value(emax_simulated, draws_emax_transformed, period, k, payoffs_systematic, mapping_state_idx, states_all, periods_emax, delta, edu_start, edu_max)

                ! Collect information
                periods_emax_slaves(count) = emax_simulated

                count = count + 1

            END DO

            CALL distribute_information_slaves(num_states_slaves, period, periods_emax_slaves, periods_emax(period + 1, :))

            ! The leading slave updates the master period by period.
            IF (is_head .AND. update_master) CALL MPI_SEND(periods_emax(period + 1, :num_states), num_states, MPI_DOUBLE, 0, period, PARENTCOMM, ierr)

        END IF

        DEALLOCATE(periods_emax_slaves, endogenous_slaves)

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
