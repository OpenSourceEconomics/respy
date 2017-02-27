!******************************************************************************
!******************************************************************************
MODULE solve_fortran

    !/*	external modules	*/

    USE recording_solution

    USE shared_interface

    USE solve_auxiliary

    !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

 CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_solve(periods_rewards_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, is_interpolated, num_points_interp, num_draws_emax, num_periods, is_myopic, edu_start, is_debug, edu_max, min_idx, periods_draws_emax, ambi_spec, optim_paras, optimizer_options, file_sim)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_all(:, :, :)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)               :: optim_paras
    TYPE(AMBI_DICT), INTENT(IN)                     :: ambi_spec

    INTEGER(our_int), INTENT(IN)                    :: num_points_interp
    INTEGER(our_int), INTENT(IN)                    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)                    :: num_periods
    INTEGER(our_int), INTENT(IN)                    :: edu_start
    INTEGER(our_int), INTENT(IN)                    :: edu_max
    INTEGER(our_int), INTENT(IN)                    :: min_idx

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_rewards_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(: ,:)

    REAL(our_dble), INTENT(IN)                      :: periods_draws_emax(num_periods, num_draws_emax, 4)

    LOGICAL, INTENT(IN)                             :: is_interpolated
    LOGICAL, INTENT(IN)                             :: is_myopic
    LOGICAL, INTENT(IN)                             :: is_debug

    CHARACTER(225), INTENT(IN)                      :: file_sim

    TYPE(OPTIMIZER_COLLECTION), INTENT(IN)          :: optimizer_options

    !/* internal objects    */

    REAL(our_dble), ALLOCATABLE                     :: opt_ambi_details(:, :, :)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL record_solution(1, file_sim)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, num_periods, edu_start, edu_max, min_idx)

    CALL record_solution(-1, file_sim)


    CALL record_solution(2, file_sim)

    CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, edu_start, max_states_period, optim_paras)

    CALL record_solution(-1, file_sim)


    CALL record_solution(3, file_sim)

    CALL fort_backward_induction(periods_emax, opt_ambi_details, num_periods, is_myopic, max_states_period, periods_draws_emax, num_draws_emax, states_number_period, periods_rewards_systematic, edu_max, edu_start, mapping_state_idx, states_all, is_debug, is_interpolated, num_points_interp, ambi_spec, optim_paras, optimizer_options, file_sim, .True.)

    IF (optim_paras%level(1) .GT. MIN_AMBIGUITY) CALL record_ambiguity(opt_ambi_details, states_number_period, file_sim, optim_paras)

    IF (.NOT. is_myopic) THEN
        CALL record_solution(-1, file_sim)
    ELSE
        CALL record_solution(-2, file_sim)
    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
