!*******************************************************************************
!*******************************************************************************
!
!   Interface to ROBUPY library. This is the front-end to all functionality.
!   Subroutines and functions for the case of risk-only case are in the
!   evaluate_risk module. Building on the risk-only functionality, the module
!   robufort_ambugity provided the required subroutines and functions for the
!   case of ambiguity.
!
!*******************************************************************************
!*******************************************************************************
MODULE solve_fortran

	!/*	external modules	*/

    USE shared_constants

    USE solve_auxiliary

	!/*	setup	*/

    IMPLICIT NONE

    PUBLIC

 CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE fort_solve(periods_payoffs_systematic, states_number_period, &
                mapping_state_idx, periods_emax, states_all, coeffs_a, &
                coeffs_b, coeffs_edu, coeffs_home, shocks_cov, &
                is_deterministic, is_interpolated, num_draws_emax, &
                periods_draws_emax, is_ambiguous, num_periods, num_points, &
                edu_start, is_myopic, is_debug, edu_max, min_idx, &
                delta, level)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_all(:, :, :)

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(:, :)

    INTEGER(our_int), INTENT(IN)                    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)                    :: num_periods
    INTEGER(our_int), INTENT(IN)                    :: num_points
    INTEGER(our_int), INTENT(IN)                    :: edu_start
    INTEGER(our_int), INTENT(IN)                    :: edu_max
    INTEGER(our_int), INTENT(IN)                    :: min_idx

    REAL(our_dble), INTENT(IN)                      :: periods_draws_emax(:, :, :)
    REAL(our_dble), INTENT(IN)                      :: coeffs_home(:)
    REAL(our_dble), INTENT(IN)                      :: coeffs_edu(:)
    REAL(our_dble), INTENT(IN)                      :: shocks_cov(4, 4)
    REAL(our_dble), INTENT(IN)                      :: coeffs_a(:)
    REAL(our_dble), INTENT(IN)                      :: coeffs_b(:)
    REAL(our_dble), INTENT(IN)                      :: level
    REAL(our_dble), INTENT(IN)                      :: delta

    LOGICAL, INTENT(IN)                             :: is_deterministic
    LOGICAL, INTENT(IN)                             :: is_interpolated
    LOGICAL, INTENT(IN)                             :: is_ambiguous
    LOGICAL, INTENT(IN)                             :: is_myopic
    LOGICAL, INTENT(IN)                             :: is_debug

    !/* internal objects        */

    INTEGER(our_int), ALLOCATABLE                   :: states_all_tmp(:, :, :)

    INTEGER(our_int)                                :: max_states_period
    INTEGER(our_int)                                :: period

    REAL(our_dble)                                  :: shocks_cholesky(4, 4)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Construct Cholesky decomposition
    IF (is_deterministic) THEN
        shocks_cholesky = zero_dble
    ELSE
        CALL cholesky(shocks_cholesky, shocks_cov)
    END IF

    ! Allocate arrays
    ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
    ALLOCATE(states_all_tmp(num_periods, 100000, 4))
    ALLOCATE(states_number_period(num_periods))

    ! Create the state space of the model
    CALL fort_create_state_space(states_all_tmp, states_number_period, &
            mapping_state_idx, max_states_period, num_periods, edu_start, &
            edu_max)

    ! Cutting the states_all container to size. The required size is only known
    ! after the state space creation is completed.
    ALLOCATE(states_all(num_periods, max_states_period, 4))
    states_all = states_all_tmp(:, :max_states_period, :)
    DEALLOCATE(states_all_tmp)

    ! Allocate arrays
    ALLOCATE(periods_payoffs_systematic(num_periods, max_states_period, 4))
    ALLOCATE(periods_emax(num_periods, max_states_period))

    ! Calculate the systematic payoffs
    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, &
            num_periods, states_number_period, states_all, edu_start, &
            coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

    ! Initialize containers, which contain a lot of missing values as we
    ! capture the tree structure in arrays of fixed dimension.
    periods_emax = MISSING_FLOAT

    ! Perform backward induction procedure.
    IF (is_myopic) THEN

        ! All other objects remain set to MISSING_FLOAT. This align the
        ! treatment for the two special cases: (1) is_myopic and (2)
        ! is_interpolated.
        DO period = 1,  num_periods
            periods_emax(period, :states_number_period(period)) = zero_dble
        END DO

    ELSE

        CALL fort_backward_induction(periods_emax, num_periods, &
                periods_draws_emax, num_draws_emax, states_number_period, & 
                periods_payoffs_systematic, edu_max, edu_start, & 
                mapping_state_idx, states_all, delta, is_debug, &
                shocks_cov, level, is_ambiguous, is_interpolated, &
                num_points, is_deterministic, shocks_cholesky)

    END IF

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE