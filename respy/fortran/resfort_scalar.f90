!*****************************************************************************
!***************************************************************************** 
PROGRAM resfort_scalar

    !/* external modules        */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: crit_val

    REAL(our_dble)                  :: x_start(26), rhobeg, rhoend

    INTEGER(our_int)                :: i, npt, maxfun, iter

    LOGICAL :: success
    CHARACTER(150) :: message
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Read specification of model. This is the FORTRAN replacement for the RespyCls instance that carries the model parametrization for the PYTHON/F2PY implementations.
    CALL read_specification(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky)

    ! This part creates (or reads from disk) the draws for the Monte Carlo integration of the EMAX. For is_debugging purposes, these might also be read in from disk or set to zero/one.   
    CALL create_draws(periods_draws_emax, num_draws_emax, seed_emax)

    ! Execute on request.
    IF (request == 'solve') THEN

        ! Solve the model for a given parametrization.    
        CALL fort_solve(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, periods_draws_emax)

    ELSE IF (request == 'evaluate') THEN

        ! This part creates (or reads from disk) the draws for the Monte Carlo integration of the choice probabilities. For is_debugging purposes, these might also be read in from disk or set to zero/one.   
        CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob)

        ! Read observed dataset from disk.
        CALL read_dataset(data_array, num_agents_est)

        ! Prepare interface for evaluation of criterion function
        CALL get_optim_paras(x_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky)

        ! Evaluate criterion function.
        crit_val = fort_criterion(x_start)
        

!        DO i = 1, 26

!            PRINT *, x_start(i)!

 !       END DO
        
 !       PRINT *, ''
 !       PRINT *, ''
        
 !       PRINT *, crit_val
!
!        npt = min(26 * 2, 26+2)
!        rhobeg = maxval(x_start)
!        rhoend = 1e-6 * rhobeg
!        maxfun = 5000

!      CALL NEWUOA (fort_criterion, x_start, npt, rhobeg, rhoend, 0, maxfun, success, message, iter)   

  ELSE IF (request == 'estimate') THEN

        ! This part creates (or reads from disk) the draws for the Monte Carlo integration of the choice probabilities. For is_debugging purposes, these might also be read in from disk or set to zero/one.   
        CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob)

        ! Read observed dataset from disk.
        CALL read_dataset(data_array, num_agents_est)

        ! Prepare interface for evaluation of criterion function
        CALL get_optim_paras(x_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky)

        ! Evaluate criterion function.
        crit_val = fort_criterion(x_start)

    END IF

    ! Store results. These are read in by the PYTHON wrapper and added to the RespyCls instance.
    CALL store_results(mapping_state_idx, states_all, periods_payoffs_systematic, states_number_period, periods_emax, crit_val)


!******************************************************************************
!******************************************************************************
END PROGRAM