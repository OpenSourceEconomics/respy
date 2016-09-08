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
SUBROUTINE fort_solve(periods_rewards_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, is_interpolated, num_points_interp, num_draws_emax, num_periods, is_myopic, edu_start, is_debug, edu_max, min_idx, delta, periods_draws_emax, measure, level, optimizer_options)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_all(:, :, :)

    INTEGER(our_int), INTENT(IN)                    :: num_points_interp
    INTEGER(our_int), INTENT(IN)                    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)                    :: num_periods
    INTEGER(our_int), INTENT(IN)                    :: edu_start
    INTEGER(our_int), INTENT(IN)                    :: edu_max
    INTEGER(our_int), INTENT(IN)                    :: min_idx

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_rewards_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(: ,:)

    REAL(our_dble), INTENT(IN)                      :: periods_draws_emax(num_periods, num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)                      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)                      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)                      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)                      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)                      :: coeffs_b(6)
    REAL(our_dble), INTENT(IN)                      :: level(1)
    REAL(our_dble), INTENT(IN)                      :: delta

    LOGICAL, INTENT(IN)                             :: is_interpolated
    LOGICAL, INTENT(IN)                             :: is_myopic
    LOGICAL, INTENT(IN)                             :: is_debug

    CHARACTER(10), INTENT(IN)                       :: measure

    TYPE(optimizer_collection), INTENT(IN)          :: optimizer_options

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL record_solution(1)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, num_periods, edu_start, edu_max, min_idx)

    CALL record_solution(-1)


    CALL record_solution(2)

    CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, edu_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, max_states_period)

    CALL record_solution(-1)


    CALL record_solution(3)

    CALL fort_backward_induction(periods_emax, num_periods, is_myopic, max_states_period, periods_draws_emax, num_draws_emax, states_number_period, periods_rewards_systematic, edu_max, edu_start, mapping_state_idx, states_all, delta, is_debug, is_interpolated, num_points_interp, shocks_cholesky, measure, level, optimizer_options, .True.)

    IF (.NOT. is_myopic) THEN
        CALL record_solution(-1)
    ELSE
        CALL record_solution(-2)
    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
