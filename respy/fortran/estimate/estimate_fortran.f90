!******************************************************************************
!******************************************************************************
MODULE estimate_fortran

    !/* external modules    */

    USE estimate_auxiliary

    USE shared_containers

    USE evaluate_fortran

    USE shared_constants

    USE dfpmin_module

    USE newuoa_module

    USE solve_fortran

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_estimate(crit_val, success, message, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed, optimizer_used, maxfun, newuoa_npt, newuoa_rhobeg, newuoa_rhoend, newuoa_maxfun, bfgs_gtol, bfgs_maxiter, bfgs_stpmx)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: crit_val

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)      :: coeffs_b(6)

    INTEGER(our_int), INTENT(IN)    :: newuoa_maxfun
    INTEGER(our_int), INTENT(IN)    :: bfgs_maxiter
    INTEGER(our_int), INTENT(IN)    :: newuoa_npt
    INTEGER(our_int), INTENT(IN)    :: maxfun

    REAL(our_dble), INTENT(IN)      :: newuoa_rhobeg
    REAL(our_dble), INTENT(IN)      :: newuoa_rhoend
    REAL(our_dble), INTENT(IN)      :: bfgs_stpmx
    REAL(our_dble), INTENT(IN)      :: bfgs_gtol

    CHARACTER(225), INTENT(IN)      :: optimizer_used
    CHARACTER(150), INTENT(OUT)     :: message

    LOGICAL, INTENT(IN)             :: paras_fixed(26) 
    LOGICAL, INTENT(OUT)            :: success

    LOGICAL, PARAMETER              :: all_free(26) = .False.

    !/* internal objects    */

    REAL(our_dble)                  :: x_free_scaled(num_free)
    REAL(our_dble)                  :: x_free_start(num_free)
    REAL(our_dble)                  :: x_free_final(num_free)
    
    INTEGER(our_int)                :: iter

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ! Some ingredients for the evaluation of the criterion function need to be created once and shared globally.
    CALL get_free_optim_paras(x_all_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, all_free)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, edu_start, edu_max)

    CALL get_free_optim_paras(x_free_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed)


    IF (maxfun == zero_int) THEN

        success = .True.
        message = 'Single evaluation of criterion function at starting values.'

        x_free_final = x_free_start

    ELSEIF (optimizer_used == 'FORT-NEWUOA') THEN

        CALL get_scales(auto_scales, x_free_start)

        x_free_scaled = apply_scaling(x_free_start, auto_scales, 'do')

        crit_scaled = .True.

        CALL newuoa(fort_criterion, x_free_scaled, newuoa_npt, newuoa_rhobeg, newuoa_rhoend, zero_int, MIN(maxfun, newuoa_maxfun) - 1, success, message, iter)

        crit_scaled = .False.

        x_free_final = apply_scaling(x_free_scaled, auto_scales, 'undo')
        
    ELSEIF (optimizer_used == 'FORT-BFGS') THEN

        CALL dfpmin(fort_criterion, fort_dcriterion, x_free_start, bfgs_gtol, bfgs_maxiter, bfgs_stpmx, maxfun, success, message, iter)

        x_free_final = x_free_start

    END IF   

    crit_val = fort_criterion(x_free_final)

    CALL logging_estimation_final(success, message, crit_val)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION apply_scaling(x_in, auto_scales, request)

    !/* external objects    */

    REAL(our_dble)                  :: apply_scaling(num_free)

    REAL(our_dble), INTENT(IN)      :: auto_scales(num_free, num_free)
    REAL(our_dble), INTENT(IN)      :: x_in(num_free)

    CHARACTER(*), INTENT(IN)        :: request

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF (request == 'do') THEN

        apply_scaling = MATMUL(auto_scales, x_in)

    ELSE

        apply_scaling = MATMUL(pinv(auto_scales, num_free), x_in)

    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE get_scales(auto_scales, x_free_start)

    !/* external objects    */

    REAL(our_dble), ALLOCATABLE, INTENT(OUT)     :: auto_scales(:, :)

    REAL(our_dble), INTENT(IN)                   :: x_free_start(:)

    !/* internal objects    */

    REAL(our_dble)                  :: grad(num_free)
    REAL(our_dble)                  :: val

    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    crit_logging = .False.

    crit_counter = .False.

    ALLOCATE(auto_scales(num_free, num_free))

    grad = fort_dcriterion(x_free_start)

    auto_scales = zero_dble

    DO i = 1, num_free

        val = grad(i)

        IF (ABS(val) .LT. SMALL_FLOAT) val = SMALL_FLOAT

        auto_scales(i, i) = val

    END DO

    crit_logging = .True.

    crit_counter = .True.

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION fort_criterion(x)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)      :: x(:)
    REAL(our_dble)                  :: fort_criterion

    !/* internal objects    */
    
    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)

    INTEGER(our_int), SAVE          :: num_step = - one_int

    REAL(our_dble), SAVE            :: value_step = HUGE_FLOAT

    LOGICAL                         :: is_start
    LOGICAL                         :: is_step
    
    REAL(our_dble)                  :: x_input(num_free)

    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Ensuring that the criterion function is not evaluated more than specified. However, there is the special request of MAXFUN equal to zero which needs to be allowed.
    IF ((num_eval == maxfun) .AND. (maxfun .GT. zero_int)) THEN
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

    CALL dist_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x_all_current)

    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start)

    CALL fort_backward_induction(periods_emax, periods_draws_emax, states_number_period, periods_payoffs_systematic, mapping_state_idx, states_all, shocks_cholesky, delta, is_debug, is_interpolated, is_myopic, edu_start, edu_max)

    CALL fort_evaluate(fort_criterion, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_est, periods_draws_prob, delta, tau, edu_start, edu_max)

    ! The counting is turned of during the determination of the auto scaling.
    IF (crit_counter) THEN
    
        num_eval = num_eval + 1

        is_start = (num_eval == 1)

        is_step = (value_step .GT. fort_criterion) 
     
        IF (is_step) THEN

            num_step = num_step + 1

            value_step = fort_criterion

        END IF

    END IF

    ! The logging can be turned during the determination of the auto scaling.
    IF (crit_logging) THEN
    
        CALL write_out_information(num_eval, fort_criterion, x_all_current, 'current')

        IF (is_start) THEN

            CALL write_out_information(zero_int, fort_criterion, x_all_current, 'start')

        END IF

        IF (is_step) THEN

            CALL write_out_information(num_step, fort_criterion, x_all_current, 'step')

            CALL logging_estimation_step(num_step, num_eval, fort_criterion)
            
        END IF

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

        d = bfgs_epsilon * ei

        f1 = fort_criterion(x + d)

        fort_dcriterion(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE construct_all_current_values(x_all_current, x, paras_fixed)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: x_all_current(26)

    LOGICAL, INTENT(IN)             :: paras_fixed(26)

    REAL(our_dble), INTENT(IN)      :: x(COUNT(.not. paras_fixed))


    !/* internal objects        */

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------    

    j = 1

    DO i = 1, 26

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
END MODULE