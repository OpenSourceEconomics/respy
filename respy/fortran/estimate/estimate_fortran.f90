!******************************************************************************
!******************************************************************************
MODULE estimate_fortran

    !/* external modules    */

    USE dfpmin_module

    USE newuoa_module


    USE estimate_auxiliary

    USE solve_fortran
    USE evaluate_fortran
    USE shared_constants

    USE shared_containers

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

    INTEGER(our_int), INTENT(IN)    :: maxfun
    INTEGER(our_int), INTENT(IN)    :: newuoa_maxfun
    INTEGER(our_int), INTENT(IN)    :: newuoa_npt
    

    INTEGER(our_int)                :: bfgs_maxiter
    REAL(our_dble)                  :: bfgs_stpmx
    REAL(our_dble)                  :: bfgs_gtol

    REAL(our_dble), INTENT(IN)      :: newuoa_rhobeg
    REAL(our_dble), INTENT(IN)      :: newuoa_rhoend

    CHARACTER(225), INTENT(IN)      :: optimizer_used

    LOGICAL, INTENT(IN)             :: paras_fixed(26) 
    !/* internal objects    */

    REAL(our_dble)                  :: x_free_start(COUNT(.not. paras_fixed))
    REAL(our_dble)                  :: x_free_final(COUNT(.not. paras_fixed))
    
    INTEGER(our_int)                :: iter
    INTEGER(our_int)                :: maxfun_int
    LOGICAL, INTENT(OUT)                         :: success
    CHARACTER(150), INTENT(OUT)                  :: message

    LOGICAL, PARAMETER :: all_free(26) = .False.
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    CALL get_free_optim_paras(x_free_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed)

    CALL get_free_optim_paras(x_all_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, all_free)

    x_free_final = x_free_start


    IF (maxfun == zero_int) THEN


    ELSEIF (optimizer_used == 'FORT-NEWUOA') THEN

        ! This is required to keep the original design of the algorithm intact
        maxfun_int = MIN(maxfun, newuoa_maxfun) - 1 

        CALL newuoa(fort_criterion, x_free_final, newuoa_npt, newuoa_rhobeg, newuoa_rhoend, zero_int, maxfun_int, success, message, iter)
        
    ELSEIF (optimizer_used == 'FORT-BFGS') THEN

        CALL dfpmin(fort_criterion, fort_dcriterion, x_free_final, bfgs_gtol, bfgs_maxiter, bfgs_stpmx, maxfun, success, message, iter)

    END IF
    
    crit_val = fort_criterion(x_free_final)

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

    INTEGER(our_int), SAVE          :: num_step = zero_int

    REAL(our_dble), SAVE            :: value_step = HUGE_FLOAT

    LOGICAL                         :: is_start
    LOGICAL                         :: is_step

    INTEGER(our_int)                :: i, j, k
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Ensuring that the criterion function is not evaluated more than specified. However, there is the special request of MAXFUN equal to zero which needs to be allowed.
    IF ((num_eval == maxfun) .AND. (maxfun .GT. zero_int)) THEN
        fort_criterion = -HUGE_FLOAT
        RETURN
    END IF


    ! GET the current parameters
    j = 1
    k = 1

    DO i =1, 26

        IF(paras_fixed(i)) THEN

            x_all_current(i) = x_all_start(i)

        ELSE
            x_all_current(i) = x(j)
            j = j + 1

        END IF


    END DO


    CALL dist_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x_all_current)

    CALL fort_solve(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, periods_draws_emax, delta, is_debug, is_interpolated, is_myopic, edu_start, edu_max)

    CALL fort_evaluate(fort_criterion, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_est, periods_draws_prob, delta, tau, edu_start, edu_max)


    num_eval = num_eval + 1

    is_start = (num_eval == 1)



    is_step = (value_step .GT. fort_criterion) 
 
    IF (is_step) THEN

        num_step = num_step + 1

        value_step = fort_criterion

    END IF

    
    CALL write_out_information(num_eval, fort_criterion, x_all_current, 'current')

    IF (is_start) THEN

        CALL write_out_information(zero_int, fort_criterion, x_all_current, 'start')

    END IF

    IF (is_step) THEN

        CALL write_out_information(num_step, fort_criterion, x_all_current, 'step')

    END IF

    
END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION fort_dcriterion(x)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: x(:)
    REAL(our_dble)                  :: fort_dcriterion(SIZE(x))

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
    f0 = fort_criterion(x)

    DO j = 1, COUNT(.NOT. paras_fixed)

        ei(j) = one_dble

        d = bfgs_epsilon * ei

        f1 = fort_criterion(x + d)

        fort_dcriterion(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
END MODULE