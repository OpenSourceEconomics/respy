!*******************************************************************************
!*******************************************************************************
!   
!   This module serves as the F2PY interface to the core functions. All the 
!   functions have counterparts as PYTHON implementations.
!
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_criterion(crit_val, x, is_deterministic, is_interpolated, & 
                num_draws_emax, is_ambiguous, num_periods, num_points, & 
                is_myopic, edu_start, is_debug, measure, edu_max, min_idx, & 
                delta, level, data_array, num_agents, num_draws_prob, & 
                tau, periods_draws_emax, periods_draws_prob)

    !/* external libraries      */

    USE estimate_auxiliary

    USE evaluate_fortran

    USE solve_fortran

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: crit_val

    DOUBLE PRECISION, INTENT(IN)   :: x(26)


    INTEGER, INTENT(IN)             :: num_agents
    INTEGER, INTENT(IN)             :: num_draws_emax, num_draws_prob
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_points
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: min_idx

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_prob(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: data_array(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: level
    DOUBLE PRECISION, INTENT(IN)    :: delta 
    DOUBLE PRECISION, INTENT(IN)    :: tau

    LOGICAL, INTENT(IN)             :: is_deterministic 
    LOGICAL, INTENT(IN)             :: is_interpolated
    LOGICAL, INTENT(IN)             :: is_ambiguous
    LOGICAL, INTENT(IN)             :: is_myopic
    LOGICAL, INTENT(IN)             :: is_debug

    CHARACTER(10), INTENT(IN)       :: measure

    !/* internal objects            */

    INTEGER, ALLOCATABLE            :: mapping_state_idx(:, :, :, :, :)
    INTEGER, ALLOCATABLE            :: states_number_period(:)
    INTEGER, ALLOCATABLE            :: states_all(:, :, :)

    DOUBLE PRECISION, ALLOCATABLE   :: periods_payoffs_systematic(:, :, :)
    DOUBLE PRECISION, ALLOCATABLE   :: periods_emax(:, :)

    DOUBLE PRECISION                :: shocks_cholesky(4, 4) 
    DOUBLE PRECISION                :: shocks_cov(4, 4)
    DOUBLE PRECISION                :: coeffs_home(1)
    DOUBLE PRECISION                :: coeffs_edu(3)
    DOUBLE PRECISION                :: coeffs_a(6)
    DOUBLE PRECISION                :: coeffs_b(6)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# Distribute model parameters
    CALL get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, &
                shocks_cov, shocks_cholesky, x)

    ! Solve requested model
    CALL fort_solve(periods_payoffs_systematic, states_number_period, & 
            mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, & 
            coeffs_edu, coeffs_home, shocks_cov, is_deterministic, & 
            is_interpolated, num_draws_emax, periods_draws_emax, & 
            is_ambiguous, num_periods, num_points, edu_start, is_myopic, & 
            is_debug, measure, edu_max, min_idx, delta, level)

    ! Evaluate criterion function for observed data
    CALL fort_evaluate(crit_val, periods_payoffs_systematic, & 
            mapping_state_idx, periods_emax, states_all, shocks_cov, & 
            is_deterministic, num_periods, edu_start, edu_max, delta, & 
            data_array, num_agents, num_draws_prob, periods_draws_prob, tau)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_solve(periods_payoffs_systematic, states_number_period, &
                mapping_state_idx, periods_emax, states_all, coeffs_a, &
                coeffs_b, coeffs_edu, coeffs_home, shocks_cov, &
                is_deterministic, is_interpolated, num_draws_emax, & 
                is_ambiguous, num_periods, num_points, is_myopic, edu_start, & 
                is_debug, measure, edu_max, min_idx, delta, level, & 
                periods_draws_emax, max_states_period)
    
    !
    ! The presence of max_states_period breaks the equality of interfaces. 
    ! However, this is required so that the size of the return arguments is
    ! known from the beginning.
    !

    !/* external libraries      */

    USE solve_fortran

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER, INTENT(OUT)            :: states_all(num_periods, max_states_period, 4)
    INTEGER, INTENT(OUT)            :: states_number_period(num_periods)

    DOUBLE PRECISION, INTENT(OUT)   :: periods_payoffs_systematic(num_periods, max_states_period, 4)
    DOUBLE PRECISION, INTENT(OUT)   :: periods_emax(num_periods, max_states_period)

    INTEGER, INTENT(IN)             :: max_states_period
    INTEGER, INTENT(IN)             :: num_draws_emax
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_points
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: min_idx

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cov(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(:)
    DOUBLE PRECISION, INTENT(IN)    :: level
    DOUBLE PRECISION, INTENT(IN)    :: delta 

    LOGICAL, INTENT(IN)             :: is_deterministic 
    LOGICAL, INTENT(IN)             :: is_interpolated
    LOGICAL, INTENT(IN)             :: is_ambiguous
    LOGICAL, INTENT(IN)             :: is_myopic
    LOGICAL, INTENT(IN)             :: is_debug

    CHARACTER(10), INTENT(IN)       :: measure

    !/* internal objects        */

        ! This container are required as output arguments cannot be of 
        ! assumed-shape type
    
    INTEGER, ALLOCATABLE            :: mapping_state_idx_int(:, :, :, :, :)
    INTEGER, ALLOCATABLE            :: states_number_period_int(:)
    INTEGER, ALLOCATABLE            :: states_all_int(:, :, :)

    DOUBLE PRECISION, ALLOCATABLE   :: periods_payoffs_systematic_int(:, :, :)
    DOUBLE PRECISION, ALLOCATABLE   :: periods_emax_int(:, :)


!-------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------- 

    ! Call FORTRAN solution
    CALL fort_solve(periods_payoffs_systematic_int, states_number_period_int, & 
            mapping_state_idx_int, periods_emax_int, states_all_int, & 
            coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, & 
            is_deterministic, is_interpolated, num_draws_emax, & 
            periods_draws_emax, is_ambiguous, num_periods, num_points, & 
            edu_start, is_myopic, is_debug, measure, edu_max, min_idx, & 
            delta, level)

    ! Assign to initial objects for return to PYTHON
    periods_payoffs_systematic = periods_payoffs_systematic_int   
    states_number_period = states_number_period_int
    mapping_state_idx = mapping_state_idx_int 
    periods_emax = periods_emax_int 
    states_all = states_all_int

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_evaluate(crit_val, coeffs_a, coeffs_b, coeffs_edu, & 
                coeffs_home, shocks_cov, is_deterministic, is_interpolated, & 
                num_draws_emax, is_ambiguous, num_periods, num_points, &
                is_myopic, edu_start, is_debug, measure, edu_max, min_idx, &
                delta, level, data_array, num_agents, num_draws_prob, tau, & 
                periods_draws_emax, periods_draws_prob)

    !/* external libraries      */

    USE evaluate_fortran

    USE solve_fortran

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: crit_val

    INTEGER, INTENT(IN)             :: num_draws_prob
    INTEGER, INTENT(IN)             :: num_draws_emax
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_points
    INTEGER, INTENT(IN)             :: num_agents
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: min_idx

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_prob(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cov(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: data_array(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)

    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(:)
    DOUBLE PRECISION, INTENT(IN)    :: delta 
    DOUBLE PRECISION, INTENT(IN)    :: level
    DOUBLE PRECISION, INTENT(IN)    :: tau

    LOGICAL, INTENT(IN)             :: is_deterministic 
    LOGICAL, INTENT(IN)             :: is_interpolated
    LOGICAL, INTENT(IN)             :: is_ambiguous
    LOGICAL, INTENT(IN)             :: is_myopic
    LOGICAL, INTENT(IN)             :: is_debug

    CHARACTER(10), INTENT(IN)       :: measure

    !/* internal */

    INTEGER, ALLOCATABLE            :: mapping_state_idx(:, :, :, :, :)
    INTEGER, ALLOCATABLE            :: states_number_period(:)
    INTEGER, ALLOCATABLE            :: states_all(:, :, :)

    DOUBLE PRECISION, ALLOCATABLE   :: periods_payoffs_systematic(:, :, :)
    DOUBLE PRECISION, ALLOCATABLE   :: periods_emax(:, :)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Solve them model for the given parametrization.
    CALL fort_solve(periods_payoffs_systematic, states_number_period, &
            mapping_state_idx, periods_emax, states_all, coeffs_a, & 
            coeffs_b, coeffs_edu, coeffs_home, shocks_cov, is_deterministic, & 
            is_interpolated, num_draws_emax, periods_draws_emax, & 
            is_ambiguous, num_periods, num_points, edu_start, is_myopic, & 
            is_debug, measure, edu_max, min_idx, delta, level)

    ! Evaluate the criterion function building on the solution.
    CALL fort_evaluate(crit_val, periods_payoffs_systematic, & 
            mapping_state_idx, periods_emax, states_all, shocks_cov, & 
            is_deterministic, num_periods, edu_start, edu_max, delta, &
            data_array, num_agents, num_draws_prob, periods_draws_prob, tau)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_simulate(dataset, periods_payoffs_systematic, & 
                mapping_state_idx, periods_emax, num_periods, states_all, & 
                num_agents, edu_start, edu_max, delta, periods_draws_sims, & 
                shocks_cholesky)

    !/* external libraries      */

    USE simulate_fortran

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: dataset(num_agents * num_periods, 8)

    DOUBLE PRECISION, INTENT(IN)    :: periods_payoffs_systematic(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_sims(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: edu_start

    INTEGER, INTENT(IN)             :: mapping_state_idx(:, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_all(:, :, :)
    INTEGER, INTENT(IN)             :: num_agents
    INTEGER, INTENT(IN)             :: edu_max

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL fort_simulate(dataset, periods_payoffs_systematic, & 
                mapping_state_idx, periods_emax, num_periods, states_all, & 
                num_agents, edu_start, edu_max, delta, periods_draws_sims, &
                shocks_cholesky)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_backward_induction(periods_emax, num_periods, & 
                max_states_period, periods_draws_emax, num_draws_emax, & 
                states_number_period, periods_payoffs_systematic, edu_max, & 
                edu_start, mapping_state_idx, states_all, delta, is_debug, & 
                shocks_cov, level, is_ambiguous, measure, is_interpolated, & 
                num_points, is_deterministic, shocks_cholesky)

    !/* external libraries      */

    USE solve_auxiliary

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: periods_emax(num_periods, max_states_period)

    DOUBLE PRECISION, INTENT(IN)    :: periods_payoffs_systematic(:, :, :   )
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cov(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: delta
    DOUBLE PRECISION, INTENT(IN)    :: level

    INTEGER, INTENT(IN)             :: mapping_state_idx(:, :, :, :, :)    
    INTEGER, INTENT(IN)             :: states_number_period(:)
    INTEGER, INTENT(IN)             :: states_all(:, :, :)
    INTEGER, INTENT(IN)             :: max_states_period
    INTEGER, INTENT(IN)             :: num_draws_emax
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_points
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max

    LOGICAL, INTENT(IN)             :: is_deterministic
    LOGICAL, INTENT(IN)             :: is_interpolated
    LOGICAL, INTENT(IN)             :: is_ambiguous
    LOGICAL, INTENT(IN)             :: is_debug

    CHARACTER(10), INTENT(IN)       :: measure

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! This assignment is required as the variables are initialized with zero 
    ! by the interface generator.
    periods_emax = MISSING_FLOAT

    ! Call actual function of interest
    CALL fort_backward_induction(periods_emax, num_periods, &
            periods_draws_emax, num_draws_emax, states_number_period, &
            periods_payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
            states_all, delta, is_debug, shocks_cov, level, is_ambiguous, & 
            measure, is_interpolated, num_points, is_deterministic, & 
            shocks_cholesky)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_create_state_space(states_all, states_number_period, &
                mapping_state_idx, max_states_period, num_periods, &
                edu_start, edu_max, min_idx)
    
    !/* external libraries      */

    USE solve_auxiliary

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER, INTENT(OUT)            :: states_all(num_periods, 100000, 4)
    INTEGER, INTENT(OUT)            :: states_number_period(num_periods)
    INTEGER, INTENT(OUT)            :: max_states_period

    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: min_idx

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL fort_create_state_space(states_all, states_number_period, &
            mapping_state_idx, max_states_period, num_periods, edu_start, & 
            edu_max)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_calculate_payoffs_systematic(periods_payoffs_systematic, & 
                num_periods, states_number_period, states_all, edu_start, & 
                coeffs_a, coeffs_b, coeffs_edu, coeffs_home, max_states_period)

    !/* external libraries      */

    USE solve_auxiliary

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: periods_payoffs_systematic(num_periods, max_states_period, 4)

    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(:)

    INTEGER, INTENT(IN)             :: states_number_period(:)
    INTEGER, INTENT(IN)             :: max_states_period
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: edu_start

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, &
            num_periods, states_number_period, states_all, edu_start, &
            coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************