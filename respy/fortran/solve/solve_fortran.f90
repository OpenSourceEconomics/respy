!******************************************************************************
!******************************************************************************
MODULE solve_fortran

	!/*	external modules	*/

    USE recording_solution

    USE shared_constants

    USE solve_auxiliary

	!/*	setup	*/

    IMPLICIT NONE

    PUBLIC

 CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_solve(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, periods_draws_emax, delta, is_debug, is_interpolated, is_myopic, edu_start, edu_max)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_all(:, :, :)

    INTEGER(our_int), INTENT(IN)                    :: edu_start
    INTEGER(our_int), INTENT(IN)                    :: edu_max

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(: ,:)

    REAL(our_dble), INTENT(IN)                      :: periods_draws_emax(num_periods, num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)                      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)                      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)                      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)                      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)                      :: coeffs_b(6)
    REAL(our_dble), INTENT(IN)                      :: delta

    LOGICAL, INTENT(IN)                             :: is_interpolated    
    LOGICAL, INTENT(IN)                             :: is_myopic
    LOGICAL, INTENT(IN)                             :: is_debug

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL record_solution(1)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, edu_start, edu_max)

    CALL record_solution(-1)


    CALL record_solution(2)

    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start)

    CALL record_solution(-1)


    CALL record_solution(3)
    
    CALL fort_backward_induction(periods_emax, periods_draws_emax, states_number_period, periods_payoffs_systematic, mapping_state_idx, states_all, shocks_cholesky, delta, is_debug, is_interpolated, is_myopic, edu_start, edu_max, .True.)

    IF (.NOT. is_myopic) THEN
        CALL record_solution(-1)
    ELSE
        CALL record_solution(-2)
    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE