!*******************************************************************************
!*******************************************************************************
!   
!   This module serves as the F2PY interface to the core functions.
!
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_evaluate_criterion_function(rslt, mapping_state_idx, &
            periods_emax, periods_payoffs_systematic, states_all, shocks, &
            edu_max, delta, edu_start, num_periods, eps_cholesky, num_agents, & 
            num_sims, data_array, standard_deviates)

    !/* external libraries      */

    USE robufort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: rslt 

    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: num_agents
    INTEGER, INTENT(IN)             :: num_sims

    DOUBLE PRECISION, INTENT(IN)    :: standard_deviates(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: data_array(:, :)

    DOUBLE PRECISION, INTENT(IN)    :: eps_cholesky(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: delta 

    DOUBLE PRECISION, INTENT(IN)    :: periods_emax(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_payoffs_systematic(:, :, :)
    INTEGER, INTENT(IN)             :: mapping_state_idx(:, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_all(:, :, :)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
   
    CALL evaluate_criterion_function(rslt, mapping_state_idx, periods_emax, & 
            periods_payoffs_systematic, states_all, shocks, edu_max, delta, & 
            edu_start, num_periods, eps_cholesky, num_agents, num_sims, & 
            data_array, standard_deviates)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_simulate_sample(dataset, num_agents, states_all, num_periods, &
                mapping_state_idx, periods_payoffs_systematic, &
                periods_eps_relevant, edu_max, edu_start, periods_emax, delta)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: dataset(num_agents*num_periods, 8)

    DOUBLE PRECISION, INTENT(IN)    :: periods_emax(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_payoffs_systematic(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_eps_relevant(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: edu_start

    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: num_agents
    INTEGER, INTENT(IN)             :: mapping_state_idx(:, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_all(:, :, :)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL simulate_sample(dataset, num_agents, states_all, num_periods, &
                mapping_state_idx, periods_payoffs_systematic, &
                periods_eps_relevant, edu_max, edu_start, periods_emax, delta)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_backward_induction(periods_emax, periods_payoffs_ex_post, &
                periods_future_payoffs, num_periods, max_states_period, &
                periods_eps_relevant, num_draws, states_number_period, &
                periods_payoffs_systematic, edu_max, edu_start, & 
                mapping_state_idx, states_all, delta, is_debug, shocks, &
                level, is_ambiguous, measure, is_interpolated, num_points)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: periods_emax(num_periods, max_states_period)
    DOUBLE PRECISION, INTENT(OUT)   :: periods_payoffs_ex_post(num_periods, max_states_period, 4)
    DOUBLE PRECISION, INTENT(OUT)   :: periods_future_payoffs(num_periods, max_states_period, 4)

    DOUBLE PRECISION, INTENT(IN)    :: periods_eps_relevant(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_payoffs_systematic(:, :, :   )
    DOUBLE PRECISION, INTENT(IN)    :: shocks(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: delta
    DOUBLE PRECISION, INTENT(IN)    :: level

    INTEGER, INTENT(IN)             :: mapping_state_idx(:, :, :, :, :)    
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_points
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: states_number_period(:)
    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: max_states_period
    INTEGER, INTENT(IN)             :: states_all(:, :, :)

    LOGICAL, INTENT(IN)             :: is_interpolated
    LOGICAL, INTENT(IN)             :: is_ambiguous
    LOGICAL, INTENT(IN)             :: is_debug

    CHARACTER(10), INTENT(IN)       :: measure

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
        
    CALL backward_induction(periods_emax, periods_payoffs_ex_post, &
            periods_future_payoffs, num_periods, max_states_period, &
            periods_eps_relevant, num_draws, states_number_period, &
            periods_payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
            states_all, delta, is_debug, shocks, level, is_ambiguous, measure, &
            is_interpolated, num_points)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_create_state_space(states_all, states_number_period, &
                mapping_state_idx, num_periods, edu_start, edu_max, min_idx)
    
    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    INTEGER, INTENT(OUT)            :: states_all(num_periods, 100000, 4)
    INTEGER, INTENT(OUT)            :: states_number_period(num_periods)
    INTEGER, INTENT(OUT)            :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)

    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: min_idx

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL create_state_space(states_all, states_number_period, &
                mapping_state_idx, num_periods, edu_start, edu_max, min_idx)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_calculate_payoffs_systematic(periods_payoffs_systematic, num_periods, &
              states_number_period, states_all, edu_start, coeffs_a, & 
              coeffs_b, coeffs_edu, coeffs_home, max_states_period)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: periods_payoffs_systematic(num_periods, max_states_period, 4)

    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)

    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: states_number_period(:)
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: max_states_period

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL calculate_payoffs_systematic(periods_payoffs_systematic, num_periods, &
              states_number_period, states_all, edu_start, coeffs_a, & 
              coeffs_b, coeffs_edu, coeffs_home, max_states_period)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
