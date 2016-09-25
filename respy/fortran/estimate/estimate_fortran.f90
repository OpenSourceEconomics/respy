!******************************************************************************
!******************************************************************************
MODULE estimate_fortran

    !/* external modules    */

    USE estimate_auxiliary

    USE recording_estimation

    USE shared_containers

    USE evaluate_fortran

    USE shared_constants

    USE dfpmin_module

    USE newuoa_module

    USE bobyqa_module

    USE solve_fortran

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_estimate(crit_val, success, message, level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed, optimizer_used, maxfun, is_scaled, scaled_minimum, optimizer_options)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: crit_val

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)      :: scaled_minimum
    REAL(our_dble), INTENT(IN)      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)      :: coeffs_b(6)
    REAL(our_dble), INTENT(IN)      :: level(1)

    INTEGER(our_int), INTENT(IN)    :: maxfun

    CHARACTER(225), INTENT(IN)      :: optimizer_used
    CHARACTER(150), INTENT(OUT)     :: message

    LOGICAL, INTENT(IN)             :: paras_fixed(27)
    LOGICAL, INTENT(OUT)            :: is_scaled
    LOGICAL, INTENT(OUT)            :: success

    TYPE(OPTIMIZER_COLLECTION), INTENT(IN) :: optimizer_options

    !/* internal objects    */

    REAL(our_dble)                  :: x_free_start(num_free)
    REAL(our_dble)                  :: x_free_final(num_free)

    REAL(our_dble)                  :: x_all_final(27)

    INTEGER(our_int)                :: iter

    LOGICAL, PARAMETER              :: all_free(27) = .False.
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Some ingredients for the evaluation of the criterion function need to be created once and shared globally.
    CALL get_free_optim_paras(x_all_start, level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, all_free)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, num_periods, edu_start, edu_max, min_idx)

    CALL get_free_optim_paras(x_free_start, level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed)


    ! If a scaling of the criterion function is requested, then we determine the scaled and transform the starting values. Also, the boolean indicates that inside the criterion function the scaling is undone.
    IF (is_scaled .AND. (.NOT. maxfun == zero_int)) THEN

        CALL get_scales_scalar(auto_scales, x_free_start, scaled_minimum)

        x_free_start = apply_scaling(x_free_start, auto_scales, 'do')

        crit_scaled = .True.

    END IF


    crit_estimation = .True.

    IF (maxfun == zero_int) THEN

        success = .True.
        message = 'Single evaluation of criterion function at starting values.'

        x_free_final = x_free_start

    ELSEIF (optimizer_used == 'FORT-NEWUOA') THEN

        !CALL newuoa(fort_criterion, x_free_start, optimizer_options%newuoa%npt, optimizer_options%newuoa%rhobeg, optimizer_options%newuoa%rhoend, zero_int, MIN(maxfun, optimizer_options%newuoa%maxfun), success, message, iter)
        CALL bobyqa(fort_criterion, x_free_start, optimizer_options%newuoa%npt, optimizer_options%newuoa%rhobeg, optimizer_options%newuoa%rhoend, zero_int, MIN(maxfun, optimizer_options%newuoa%maxfun))

        success = .True.
        message = 'Juphiee BobYQUA.'

    ELSEIF (optimizer_used == 'FORT-BFGS') THEN

        CALL dfpmin(fort_criterion, fort_dcriterion, x_free_start, optimizer_options%bfgs%gtol, optimizer_options%bfgs%maxiter, optimizer_options%bfgs%stpmx, maxfun, success, message, iter)

    END IF

    crit_estimation = .False.


    ! If scaling is requested, then we transform the resulting parameter vector and indicate that the critterion function is to be used with the actual parameters again.
    IF (is_scaled .AND. (.NOT. maxfun == zero_int)) THEN

        crit_scaled = .False.

        x_free_final = apply_scaling(x_free_start, auto_scales, 'undo')

    ELSE

        x_free_final = x_free_start

    END IF

    ! The following allows for scalability exercise.
    IF (maxfun == zero_int) CALL record_estimation('Start')

    crit_val = fort_criterion(x_free_final)

    IF (maxfun == zero_int) CALL record_estimation('Finish')


    CALL construct_all_current_values(x_all_final, x_free_final, paras_fixed)

    CALL record_estimation(success, message, crit_val, x_all_final)

    CALL record_estimation()

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION fort_criterion(x)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)      :: x(:)
    REAL(our_dble)                  :: fort_criterion

    !/* internal objects    */

    REAL(our_dble)                  :: contribs(num_agents_est * num_periods)
    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: x_input(num_free)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: level(1)

    INTEGER(our_int)                :: dist_optim_paras_info

    ! This mock object is required as we cannot simply pass in '' as it turns out.
    CHARACTER(225)                  :: file_sim_mock

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Ensuring that the criterion function is not evaluated more than specified. However, there is the special request of MAXFUN equal to zero which needs to be allowed.
    IF ((num_eval == maxfun) .AND. crit_estimation .AND. (.NOT. maxfun == zero_int)) THEN
        fort_criterion = -HUGE_FLOAT
        RETURN
    END IF

    ! Undo the scaling (if required)
    IF (crit_scaled) THEN
        x_input = apply_scaling(x, auto_scales, 'undo')
    ELSE
        x_input = x
    END IF


    CALL construct_all_current_values(x_all_current, x_input, paras_fixed)

    CALL dist_optim_paras(level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x_all_current, dist_optim_paras_info)

    CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, edu_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, max_states_period)

    CALL fort_backward_induction(periods_emax, num_periods, is_myopic, max_states_period, periods_draws_emax, num_draws_emax, states_number_period, periods_rewards_systematic, edu_max, edu_start, mapping_state_idx, states_all, delta, is_debug, is_interpolated, num_points_interp, shocks_cholesky, measure, level, optimizer_options, file_sim_mock, .False.)

    CALL fort_contributions(contribs, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_est, periods_draws_prob, delta, tau, edu_start, edu_max, num_periods, num_draws_prob)


    fort_criterion = get_log_likl(contribs)

    IF (crit_estimation .OR. (maxfun == zero_int)) THEN

        num_eval = num_eval + 1

        CALL record_estimation(x_all_current, fort_criterion, num_eval)

        IF (dist_optim_paras_info .NE. zero_int) CALL record_warning(4)

    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION fort_dcriterion(x)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: x(:)
    REAL(our_dble)                  :: fort_dcriterion(SIZE(x))

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
    f0 = fort_criterion(x)

    DO j = 1, num_free

        ei(j) = one_dble

        d = dfunc_eps * ei

        f1 = fort_criterion(x + d)

        fort_dcriterion(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE construct_all_current_values(x_all_current, x, paras_fixed)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: x_all_current(27)

    LOGICAL, INTENT(IN)             :: paras_fixed(27)

    REAL(our_dble), INTENT(IN)      :: x(COUNT(.not. paras_fixed))


    !/* internal objects        */

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    j = 1

    DO i = 1, 27

        IF(paras_fixed(i)) THEN
            x_all_current(i) = x_all_start(i)
        ELSE
            x_all_current(i) = x(j)
            j = j + 1
        END IF

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_scales_scalar(auto_scales, x_free_start, scaled_minimum)

    !/* external objects    */

    REAL(our_dble), ALLOCATABLE, INTENT(OUT)     :: auto_scales(:, :)

    REAL(our_dble), INTENT(IN)                   :: x_free_start(num_free)
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

    grad = fort_dcriterion(x_free_start)

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
END MODULE
