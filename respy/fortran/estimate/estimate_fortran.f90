!******************************************************************************
!******************************************************************************
MODULE estimate_fortran

    !/* external modules    */

    USE optimizers_interfaces

    USE recording_estimation

    USE recording_warning

    USE shared_interface

    USE evaluate_fortran

    USE solve_fortran

    USE solve_ambiguity

#if MPI_AVAILABLE

    USE parallelism_constants

    USE parallelism_auxiliary

    USE mpi

#endif

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_estimate(crit_val, success, message, optim_paras, optimizer_used, maxfun, num_procs, precond_spec, optimizer_options, num_types)

    !/* external objects    */

    TYPE(OPTIMIZER_COLLECTION), INTENT(INOUT) :: optimizer_options

    REAL(our_dble), INTENT(OUT)         :: crit_val

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    TYPE(PRECOND_DICT), INTENT(IN)      :: precond_spec

    INTEGER(our_int), INTENT(IN)        :: num_procs
    INTEGER(our_int), INTENT(IN)        :: num_types
    INTEGER(our_int), INTENT(IN)        :: maxfun

    CHARACTER(225), INTENT(IN)          :: optimizer_used
    CHARACTER(150), INTENT(OUT)         :: message

    LOGICAL, INTENT(OUT)                :: success

    !/* internal objects    */

    REAL(our_dble)                  :: x_optim_free_scaled_start(num_free)

    REAL(our_dble)                  :: x_optim_free_unscaled_start(num_free)

    INTEGER(our_int)                :: iter

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF (num_procs == 1) THEN
        criterion_function => fort_criterion_scalar
    ELSE
        criterion_function => fort_criterion_parallel
    END IF


    ! Some ingredients for the evaluation of the criterion function need to be created once and shared globally.
    CALL get_optim_paras(x_all_start, optim_paras, .True.)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, num_periods, edu_start, edu_max, min_idx, num_types)

    CALL get_optim_paras(x_optim_free_unscaled_start, optim_paras, .False.)

    CALL get_precondition_matrix(precond_matrix, precond_spec, maxfun, x_optim_free_unscaled_start)

    x_optim_free_scaled_start = apply_scaling(x_optim_free_unscaled_start, precond_matrix, 'do')
    x_optim_bounds_free_scaled(1, :) = apply_scaling(x_optim_bounds_free_unscaled(1, :), precond_matrix, 'do')
    x_optim_bounds_free_scaled(2, :) = apply_scaling(x_optim_bounds_free_unscaled(2, :), precond_matrix, 'do')

    CALL record_estimation(precond_matrix, x_optim_free_unscaled_start, optim_paras, .False.)

    CALL auto_adjustment_optimizers(optimizer_options, optimizer_used)


    crit_estimation = .True.

    IF (maxfun == zero_int) THEN

        success = .True.
        message = 'Single evaluation of criterion function at starting values.'

        CALL record_estimation('Start')
        crit_val = criterion_function(x_optim_free_scaled_start)
        CALL record_estimation('Finish')

    ELSEIF (optimizer_used == 'FORT-NEWUOA') THEN

        CALL newuoa(criterion_function, x_optim_free_scaled_start, optimizer_options%newuoa%npt, optimizer_options%newuoa%rhobeg, optimizer_options%newuoa%rhoend, zero_int, MIN(maxfun, optimizer_options%newuoa%maxfun), success, message)

    ELSEIF (optimizer_used == 'FORT-BOBYQA') THEN

        ! The BOBYQA algorithm might adjust the starting values. So we simply make sure that the very first evaluation of the criterion function is at the actual starting values.
        crit_val = criterion_function(x_optim_free_scaled_start)
        CALL bobyqa(criterion_function, x_optim_free_scaled_start, optimizer_options%bobyqa%npt, optimizer_options%bobyqa%rhobeg, optimizer_options%bobyqa%rhoend, zero_int, MIN(maxfun, optimizer_options%bobyqa%maxfun), success, message)

    ELSEIF (optimizer_used == 'FORT-BFGS') THEN
        dfunc_eps = optimizer_options%bfgs%eps
        CALL dfpmin(criterion_function, fort_dcriterion, x_optim_free_scaled_start, optimizer_options%bfgs%gtol, optimizer_options%bfgs%maxiter, optimizer_options%bfgs%stpmx, maxfun, success, message, iter)
        dfunc_eps = -HUGE_FLOAT
    END IF

    crit_estimation = .False.

    CALL record_estimation(success, message)

    CALL record_estimation()

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION fort_criterion_scalar(x_optim_free_scaled)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)      :: x_optim_free_scaled(:)
    REAL(our_dble)                  :: fort_criterion_scalar

    !/* internal objects    */

    REAL(our_dble), ALLOCATABLE     :: opt_ambi_details(:, :, :)

    REAL(our_dble)                  :: x_optim_free_unscaled(num_free)
    REAL(our_dble)                  :: x_optim_all_unscaled(num_paras)
    REAL(our_dble)                  :: contribs(num_obs)
    REAL(our_dble)                  :: start

    INTEGER(our_int)                :: dist_optim_paras_info
    INTEGER(our_int)                :: opt_ambi_summary(2)

    ! This mock object is required as we cannot simply pass in '' as it turns out.
    CHARACTER(225)                  :: file_sim_mock

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! We intent to monitor the execution time of every evaluation of the criterion function.
    CALL CPU_TIME(start)

    ! We allow for early termination due to maximum number of iterations or user request.
    IF (check_early_termination(maxfun, num_eval, crit_estimation)) THEN
        fort_criterion_scalar = HUGE_FLOAT
        RETURN
    END IF

    x_optim_free_unscaled = apply_scaling(x_optim_free_scaled, precond_matrix, 'undo')

    CALL construct_all_current_values(x_optim_all_unscaled, x_optim_free_unscaled, optim_paras, num_paras)

    CALL dist_optim_paras(optim_paras, x_optim_all_unscaled, dist_optim_paras_info)

    CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, edu_start, max_states_period, optim_paras)

    CALL fort_backward_induction(periods_emax, opt_ambi_details, num_periods, is_myopic, max_states_period, periods_draws_emax, num_draws_emax, states_number_period, periods_rewards_systematic, edu_max, edu_start, mapping_state_idx, states_all, is_debug, is_interpolated, num_points_interp, ambi_spec, optim_paras, optimizer_options, file_sim_mock, .False.)

    CALL fort_contributions(contribs, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, data_est, periods_draws_prob, tau, edu_start, edu_max, num_periods, num_draws_prob, optim_paras, num_types)


    fort_criterion_scalar = get_log_likl(contribs)

    IF (crit_estimation .OR. (maxfun == zero_int)) THEN

        num_eval = num_eval + 1

        CALL summarize_worst_case_success(opt_ambi_summary, opt_ambi_details)

        CALL record_estimation(x_optim_free_scaled, x_optim_all_unscaled, fort_criterion_scalar, num_eval, num_paras, num_types, optim_paras, start, opt_ambi_summary)

        IF (dist_optim_paras_info .NE. zero_int) CALL record_warning(4)

    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION fort_criterion_parallel(x)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)      :: x(:)
    REAL(our_dble)                  :: fort_criterion_parallel

    !/* internal objects    */

    REAL(our_dble)                  :: x_all_current(num_paras)
    REAL(our_dble)                  :: x_input(num_free)
    REAL(our_dble)                  :: contribs(num_obs)
    REAL(our_dble)                  :: start

    INTEGER(our_int), ALLOCATABLE   :: num_states_slaves(:, :)
    INTEGER(our_int), ALLOCATABLE   :: num_obs_slaves(:)

    INTEGER(our_int)                :: opt_ambi_summary_slaves(2, num_slaves)
    INTEGER(our_int)                :: dist_optim_paras_info
    INTEGER(our_int)                :: opt_ambi_summary(2)
    INTEGER(our_int)                :: displs(num_slaves)
    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

#if MPI_AVAILABLE

    ! We intent to monitor the execution time of every evaluation of the criterion function.
    CALL CPU_TIME(start)

    ! We allow for early termination due to maximum number of iterations or user request.
    IF (check_early_termination(maxfun, num_eval, crit_estimation)) THEN
        fort_criterion_parallel = HUGE_FLOAT
        RETURN
    END IF

    x_input = apply_scaling(x, precond_matrix, 'undo')

    CALL construct_all_current_values(x_all_current, x_input, optim_paras, num_paras)

    CALL MPI_Bcast(3, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)

    CALL MPI_Bcast(x_all_current, num_paras, MPI_DOUBLE, MPI_ROOT, SLAVECOMM, ierr)

    ! This extra work is only required to align the logging across the scalar and parallel implementation. In the case of an otherwise zero variance, we stabilize the algorithm. However, we want this indicated as a warning in the log file.
    CALL dist_optim_paras(optim_paras, x_all_current, dist_optim_paras_info)

    ! We need to know how the workload is distributed across the slaves.
    IF (.NOT. ALLOCATED(num_states_slaves)) THEN
        CALL distribute_workload(num_states_slaves, num_obs_slaves)

        DO i = 1, num_slaves
            displs(i) = SUM(num_obs_slaves(:i - 1))
        END DO

    END IF

    contribs = -HUGE_FLOAT

    CALL MPI_GATHERV(contribs, 0, MPI_DOUBLE, contribs, num_obs_slaves, displs, MPI_DOUBLE, MPI_ROOT, SLAVECOMM, ierr)

    fort_criterion_parallel = get_log_likl(contribs)

    ! We also need to aggregate the information about the worst-case determination.
    opt_ambi_summary = MISSING_INT
    CALL MPI_GATHER(opt_ambi_summary_slaves, 0, MPI_INT, opt_ambi_summary_slaves, 2, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
    opt_ambi_summary = SUM(opt_ambi_summary_slaves, 2)

    IF (crit_estimation .OR. (maxfun == zero_int)) THEN

        num_eval = num_eval + 1

        CALL record_estimation(x, x_all_current, fort_criterion_parallel, num_eval, num_paras, num_types, optim_paras, start, opt_ambi_summary)

        IF (dist_optim_paras_info .NE. zero_int) CALL record_warning(4)

    END IF

#endif

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION fort_dcriterion(x_optim_free_scaled)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: x_optim_free_scaled(:)
    REAL(our_dble)                  :: fort_dcriterion(SIZE(x_optim_free_scaled))

    !/* internals objects       */

    REAL(our_dble)                  :: ei(num_free)
    REAL(our_dble)                  :: d(num_free)
    REAL(our_dble)                  :: f0
    REAL(our_dble)                  :: f1

    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    ei = zero_dble

    ! Evaluate baseline
    f0 = criterion_function(x_optim_free_scaled)

    DO j = 1, num_free

        ei(j) = one_dble

        d = dfunc_eps * ei

        f1 = criterion_function(x_optim_free_scaled + d)

        fort_dcriterion(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE construct_all_current_values(x_optim_all_unscaled, x_optim_free_unscaled, optim_paras, num_paras)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: x_optim_all_unscaled(num_paras)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    REAL(our_dble), INTENT(IN)          :: x_optim_free_unscaled(COUNT(.not. optim_paras%paras_fixed))

    INTEGER(our_int), INTENT(IN)        :: num_paras

    !/* internal objects        */

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    j = 1

    DO i = 1, num_paras

        IF(optim_paras%paras_fixed(i)) THEN
            x_optim_all_unscaled(i) = x_all_start(i)
        ELSE
            x_optim_all_unscaled(i) = x_optim_free_unscaled(j)
            j = j + 1
        END IF

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_precondition_matrix(precond_matrix, precond_spec, maxfun, x_optim_free_unscaled_start)

    !/* external objects    */

    REAL(our_dble), ALLOCATABLE, INTENT(OUT)    :: precond_matrix(:, :)

    TYPE(PRECOND_DICT), INTENT(IN)  :: precond_spec

    INTEGER(our_int), INTENT(IN)    :: maxfun

    REAL(our_dble), INTENT(IN)      :: x_optim_free_unscaled_start(num_free)

    !/* internal objects    */

    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    crit_estimation = .False.

    ALLOCATE(precond_matrix(num_free, num_free))

    CALL record_estimation(precond_matrix, x_optim_free_unscaled_start, optim_paras, .True.)

    IF ((precond_spec%type == 'identity') .OR. (maxfun == zero_int)) THEN
        precond_matrix = create_identity(num_free)
    ELSEIF (precond_spec%type == 'magnitudes') THEN
        precond_matrix = get_scales_magnitudes(x_optim_free_unscaled_start)
    ELSEIF (precond_spec%type == 'gradient') THEN
        precond_matrix = get_scales_gradient(x_optim_free_unscaled_start, precond_spec)
    ELSE
        STOP ' Not implemented ...'
    END IF

    ! Write out scaling matrix to allow for restart.
    5000 FORMAT(100000(1x,f45.15))

    OPEN(UNIT=99, FILE='scaling.respy.out', ACTION='WRITE')

    DO i = 1, num_free
        WRITE(99, 5000) precond_matrix(i, :)
    END DO

    CLOSE(99)

    crit_estimation = .False.

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION get_scales_gradient(x_optim_free_start, precond_spec) RESULT(precond_matrix)

    !/* external objects    */

    REAL(our_dble)                              :: precond_matrix(num_free, num_free)

    REAL(our_dble), INTENT(IN)                  :: x_optim_free_start(num_free)

    TYPE(PRECOND_DICT), INTENT(IN)              :: precond_spec

    !/* internal objects    */

    REAL(our_dble)                              :: grad(num_free)
    REAL(our_dble)                              :: val

    INTEGER(our_int)                            :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    dfunc_eps = precond_spec%eps

    ! The precoditioning matrix needs to be initialized to approximate the gradient with a valid preconditioning matrix.
    precond_matrix = create_identity(num_free)
    grad = fort_dcriterion(x_optim_free_start)

    dfunc_eps = -HUGE_FLOAT

    precond_matrix = zero_dble

    DO i = 1, num_free

        val = ABS(grad(i))

        IF (val .LT. precond_spec%minimum) val = precond_spec%minimum

        precond_matrix(i, i) = val

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE auto_adjustment_optimizers(optimizer_options, optimizer_used)

    !/* external objects    */

    TYPE(OPTIMIZER_COLLECTION), INTENT(INOUT)   :: optimizer_options

    CHARACTER(225), INTENT(IN)                  :: optimizer_used

    !/* internal objects    */

    INTEGER(our_int)                            :: npt

    REAL(our_dble)                              :: tmp(num_free)
    REAL(our_dble)                              :: rhobeg

    LOGICAL                                     :: is_misspecified

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF (optimizer_used == 'FORT-NEWUOA') THEN

        npt = optimizer_options%newuoa%npt
        is_misspecified = (npt .LT. num_free + 2 .OR. npt .GT. ((num_free + 2)* num_free) / 2)
        IF (is_misspecified) THEN
            optimizer_options%newuoa%npt = (2 * num_free) + 1
            CALL record_estimation('NEWUOA', optimizer_options%newuoa%npt)
        END IF

    END IF

    IF (optimizer_used == 'FORT-BOBYQA') THEN

        npt = optimizer_options%bobyqa%npt
        is_misspecified = (npt .LT. num_free + 2 .OR. npt .GT. ((num_free + 2)* num_free) / 2)
        IF (is_misspecified) THEN
            optimizer_options%bobyqa%npt =  (2 * num_free) + 1
            CALL record_estimation('BOBYQA', optimizer_options%bobyqa%npt)
        END IF

        rhobeg = optimizer_options%bobyqa%rhobeg
        tmp = x_optim_bounds_free_scaled(2, :) - x_optim_bounds_free_scaled(1, :)

        rhobeg = optimizer_options%bobyqa%rhobeg
        is_misspecified = ANY(tmp .LT. rhobeg + rhobeg)
        IF (is_misspecified) THEN
            optimizer_options%bobyqa%rhobeg = MINval(tmp) * 0.5_our_dble
            optimizer_options%bobyqa%rhoend = optimizer_options%bobyqa%rhobeg * 1e-6
            CALL record_estimation('BOBYQA', optimizer_options%bobyqa%rhobeg, optimizer_options%bobyqa%rhoend)
        END IF

    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
