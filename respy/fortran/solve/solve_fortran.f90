!******************************************************************************
!******************************************************************************
MODULE solve_fortran

	!/*	external modules	*/

    USE shared_constants

    USE solve_auxiliary

	!/*	setup	*/

    IMPLICIT NONE

    PUBLIC

 CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_solve(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, periods_draws_emax, delta)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_all(:, :, :)

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(: ,:)

    REAL(our_dble), INTENT(IN)                      :: periods_draws_emax(num_periods, num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)                      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)                      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)                      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)                      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)                      :: coeffs_b(6)
    REAL(our_dble), INTENT(IN)                      :: delta
    
    !/* internal objects        */

    INTEGER(our_int)                                :: period

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL logging_solution(1)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, periods_emax, periods_payoffs_systematic)

    CALL logging_solution(-1)


    CALL logging_solution(2)

    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

    CALL logging_solution(-1)


    CALL logging_solution(3)

    IF (is_myopic) THEN

        DO period = 1,  num_periods
            periods_emax(period, :states_number_period(period)) = zero_dble
        END DO

        CALL logging_solution(-2)
    
    ELSE

        CALL fort_backward_induction(periods_emax, periods_draws_emax, states_number_period, periods_payoffs_systematic, mapping_state_idx, states_all, shocks_cholesky, delta)

        CALL logging_solution(-1)
        
    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE