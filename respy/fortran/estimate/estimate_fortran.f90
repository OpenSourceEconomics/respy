!******************************************************************************
!******************************************************************************
MODULE estimate_fortran

    !/* external modules    */

    USE optimizers_interfaces

    USE recording_estimation

    USE shared_containers

    USE shared_utilities

    USE shared_auxiliary

    USE evaluate_fortran

    USE shared_constants

    USE solve_fortran

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_estimate(crit_val, success, message, optim_paras, paras_fixed, optimizer_used, maxfun, precond_type, precond_minimum, optimizer_options)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: crit_val

    TYPE(OPTIMIZATION_PARAMETERS), INTENT(IN) :: optim_paras

    REAL(our_dble), INTENT(IN)      :: precond_minimum

    INTEGER(our_int), INTENT(IN)    :: maxfun

    CHARACTER(225), INTENT(IN)      :: optimizer_used
    CHARACTER(150), INTENT(OUT)     :: message
    CHARACTER(50), INTENT(OUT)      :: precond_type

    LOGICAL, INTENT(IN)             :: paras_fixed(28)
    LOGICAL, INTENT(OUT)            :: success

    TYPE(OPTIMIZER_COLLECTION), INTENT(INOUT) :: optimizer_options

    !/* internal objects    */

    REAL(our_dble)                  :: x_optim_free_scaled_start(num_free)

    REAL(our_dble)                  :: x_optim_free_unscaled_start(num_free)

    INTEGER(our_int)                :: iter

    LOGICAL, PARAMETER              :: all_free(28) = .False.

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Some ingredients for the evaluation of the criterion function need to be created once and shared globally.
    CALL get_free_optim_paras(x_all_start, optim_paras, all_free)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, num_periods, edu_start, edu_max, min_idx)

    CALL get_free_optim_paras(x_optim_free_unscaled_start, optim_paras, paras_fixed)


    CALL get_precondition_matrix_scalar(precond_type, precond_minimum, maxfun, x_optim_free_unscaled_start)


    x_optim_free_scaled_start = apply_scaling(x_optim_free_unscaled_start, precond_matrix, 'do')
    x_optim_bounds_free_scaled(1, :) = apply_scaling(x_optim_bounds_free_unscaled(1, :), precond_matrix, 'do')
    x_optim_bounds_free_scaled(2, :) = apply_scaling(x_optim_bounds_free_unscaled(2, :), precond_matrix, 'do')

    CALL record_estimation(precond_matrix, x_optim_free_scaled_start, paras_fixed, .False.)


    CALL auto_adjustment_optimizers(optimizer_options, optimizer_used)


    crit_estimation = .True.

    IF (maxfun == zero_int) THEN

        success = .True.
        message = 'Single evaluation of criterion function at starting values.'

        CALL record_estimation('Start')
        crit_val = fort_criterion(x_optim_free_scaled_start)
        CALL record_estimation('Finish')

    ELSEIF (optimizer_used == 'FORT-NEWUOA') THEN

        CALL newuoa(fort_criterion, x_optim_free_scaled_start, optimizer_options%newuoa%npt, optimizer_options%newuoa%rhobeg, optimizer_options%newuoa%rhoend, zero_int, MIN(maxfun, optimizer_options%newuoa%maxfun), success, message)

    ELSEIF (optimizer_used == 'FORT-BOBYQA') THEN

        ! The BOBYQA algorithm might adjust the starting values. So we simply make sure that the very first evaluation of the criterion function is at the actual starting values.
        crit_val = fort_criterion(x_optim_free_scaled_start)
        CALL bobyqa(fort_criterion, x_optim_free_scaled_start, optimizer_options%bobyqa%npt, optimizer_options%bobyqa%rhobeg, optimizer_options%bobyqa%rhoend, zero_int, MIN(maxfun, optimizer_options%bobyqa%maxfun), success, message)

    ELSEIF (optimizer_used == 'FORT-BFGS') THEN
        dfunc_eps = optimizer_options%bfgs%eps
        CALL dfpmin(fort_criterion, fort_dcriterion, x_optim_free_scaled_start, optimizer_options%bfgs%gtol, optimizer_options%bfgs%maxiter, optimizer_options%bfgs%stpmx, maxfun, success, message, iter)
        dfunc_eps = -HUGE_FLOAT
    END IF

    crit_estimation = .False.

    CALL record_estimation(success, message)

    CALL record_estimation()

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION fort_criterion(x_optim_free_scaled)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)      :: x_optim_free_scaled(:)
    REAL(our_dble)                  :: fort_criterion

    !/* internal objects    */

    REAL(our_dble)                  :: x_optim_free_unscaled(num_free)
    REAL(our_dble)                  :: x_optim_all_unscaled(28)
    REAL(our_dble)                  :: contribs(num_obs)
    REAL(our_dble)                  :: start

    INTEGER(our_int)                :: dist_optim_paras_info

    ! This mock object is required as we cannot simply pass in '' as it turns out.
    CHARACTER(225)                  :: file_sim_mock

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! We intent to monitor the execution time of every evaluation of the criterion function.
    CALL CPU_TIME(start)

    ! Ensuring that the criterion function is not evaluated more than specified. However, there is the special request of MAXFUN equal to zero which needs to be allowed.
    IF ((num_eval == maxfun) .AND. crit_estimation .AND. (.NOT. maxfun == zero_int)) THEN
        fort_criterion = HUGE_FLOAT
        RETURN
    END IF

    x_optim_free_unscaled = apply_scaling(x_optim_free_scaled, precond_matrix, 'undo')

    CALL construct_all_current_values(x_optim_all_unscaled, x_optim_free_unscaled, paras_fixed)

    CALL dist_optim_paras(optim_paras, x_optim_all_unscaled, dist_optim_paras_info)

    CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, edu_start, max_states_period, optim_paras)

    CALL fort_backward_induction(periods_emax, num_periods, is_myopic, max_states_period, periods_draws_emax, num_draws_emax, states_number_period, periods_rewards_systematic, edu_max, edu_start, mapping_state_idx, states_all, is_debug, is_interpolated, num_points_interp, measure, optim_paras, optimizer_options, file_sim_mock, .False.)

    CALL fort_contributions(contribs, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, data_est, periods_draws_prob, tau, edu_start, edu_max, num_periods, num_draws_prob, optim_paras)


    fort_criterion = get_log_likl(contribs)

    IF (crit_estimation .OR. (maxfun == zero_int)) THEN

        num_eval = num_eval + 1

        CALL record_estimation(x_optim_free_scaled, x_optim_all_unscaled, fort_criterion, num_eval, paras_fixed, start)

        IF (dist_optim_paras_info .NE. zero_int) CALL record_warning(4)

    END IF

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
    f0 = fort_criterion(x_optim_free_scaled)

    DO j = 1, num_free

        ei(j) = one_dble

        d = dfunc_eps * ei

        f1 = fort_criterion(x_optim_free_scaled + d)

        fort_dcriterion(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE construct_all_current_values(x_optim_all_unscaled, x_optim_free_unscaled, paras_fixed)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: x_optim_all_unscaled(28)

    LOGICAL, INTENT(IN)             :: paras_fixed(28)

    REAL(our_dble), INTENT(IN)      :: x_optim_free_unscaled(COUNT(.not. paras_fixed))


    !/* internal objects        */

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    j = 1

    DO i = 1, 28

        IF(paras_fixed(i)) THEN
            x_optim_all_unscaled(i) = x_all_start(i)
        ELSE
            x_optim_all_unscaled(i) = x_optim_free_unscaled(j)
            j = j + 1
        END IF

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_scales_scalar(precond_matrix, x_optim_free_start, precond_minimum)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)                  :: precond_matrix(:, :)

    REAL(our_dble), INTENT(IN)                   :: x_optim_free_start(num_free)
    REAL(our_dble), INTENT(IN)                   :: precond_minimum

    !/* internal objects    */

    REAL(our_dble)                  :: grad(num_free)
    REAL(our_dble)                  :: val

    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    crit_estimation = .False.

    dfunc_eps = precond_eps
    grad = fort_dcriterion(x_optim_free_start)
    dfunc_eps = -HUGE_FLOAT

    precond_matrix = zero_dble

    DO i = 1, num_free

        val = ABS(grad(i))

        IF (val .LT. precond_minimum) val = precond_minimum

        precond_matrix(i, i) = val

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_precondition_matrix_scalar(precond_type, precond_minimum, maxfun, x_optim_free_unscaled_start)

    !/* external objects    */

    CHARACTER(50), INTENT(IN)      :: precond_type

    INTEGER(our_int), INTENT(IN)    :: maxfun

    REAL(our_dble), INTENT(IN)      :: x_optim_free_unscaled_start(num_free)
    REAL(our_dble), INTENT(IN)      :: precond_minimum

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    crit_estimation = .False.

    ALLOCATE(precond_matrix(num_free, num_free))

    precond_matrix = create_identity(num_free)

    CALL record_estimation(precond_matrix, x_optim_free_unscaled_start, paras_fixed, .True.)

    IF ((precond_type == 'identity') .OR. (maxfun == zero_int)) THEN
        precond_matrix = create_identity(num_free)
    ELSE
        CALL get_scales_scalar(precond_matrix, x_optim_free_unscaled_start, precond_minimum)
    END IF

    crit_estimation = .False.

END SUBROUTINE
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
