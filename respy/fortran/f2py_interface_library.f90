!*******************************************************************************
!*******************************************************************************
!   
!   This module serves as the F2PY interface to the core functions. All the 
!   functions have counterparts as PYTHON implementations.
!
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_criterion(crit_val, x, is_interpolated_int, num_draws_emax_int, num_periods_int, num_points_interp_int, is_myopic_int, edu_start_int, is_debug_int, edu_max_int, min_idx_int, delta_int, data_array, num_agents_est_int, num_draws_prob_int, tau_int, periods_draws_emax, periods_draws_prob)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: crit_val

    DOUBLE PRECISION, INTENT(IN)    :: delta_int
    DOUBLE PRECISION, INTENT(IN)    :: x(26)

    INTEGER, INTENT(IN)             :: num_agents_est_int
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: num_draws_prob_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: min_idx_int

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_prob(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: data_array(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: tau_int

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int

    !/* internal objects            */

    INTEGER, ALLOCATABLE            :: mapping_state_idx(:, :, :, :, :)
    INTEGER, ALLOCATABLE            :: states_number_period(:)
    INTEGER, ALLOCATABLE            :: states_all(:, :, :)

    DOUBLE PRECISION, ALLOCATABLE   :: periods_payoffs_systematic(:, :, :)
    DOUBLE PRECISION, ALLOCATABLE   :: periods_emax(:, :)

    DOUBLE PRECISION                :: shocks_cholesky(4, 4)
    DOUBLE PRECISION                :: coeffs_home(1)
    DOUBLE PRECISION                :: coeffs_edu(3)
    DOUBLE PRECISION                :: coeffs_a(6)
    DOUBLE PRECISION                :: coeffs_b(6)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# Transfer auxiliary variable to global variable.
    num_points_interp = num_points_interp_int
    is_interpolated = is_interpolated_int
    num_agents_est = num_agents_est_int
    num_draws_emax = num_draws_emax_int
    num_draws_prob = num_draws_prob_int
    num_periods = num_periods_int
    is_myopic = is_myopic_int
    edu_start = edu_start_int
    is_debug = is_debug_int
    min_idx = min_idx_int
    edu_max = edu_max_int
    delta = delta_int
    tau = tau_int
    
    !# Distribute model parameters
    CALL dist_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x)

    ! Solve requested model
    CALL fort_solve(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, periods_draws_emax)

    ! Evaluate criterion function for observed data
    CALL fort_evaluate(crit_val, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_array, periods_draws_prob)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_solve(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, is_interpolated_int, num_draws_emax_int, num_periods_int, num_points_interp_int, is_myopic_int, edu_start_int, is_debug_int, edu_max_int, min_idx_int, delta_int, periods_draws_emax, max_states_period)
    
    !
    ! The presence of max_states_period breaks the equality of interfaces. 
    ! However, this is required so that the size of the return arguments is
    ! known from the beginning.
    !

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: mapping_state_idx(num_periods_int, num_periods_int, num_periods_int, min_idx_int, 2)
    INTEGER, INTENT(OUT)            :: states_all(num_periods_int, max_states_period, 4)
    INTEGER, INTENT(OUT)            :: states_number_period(num_periods_int)

    DOUBLE PRECISION, INTENT(OUT)   :: periods_payoffs_systematic(num_periods_int, max_states_period, 4)
    DOUBLE PRECISION, INTENT(OUT)   :: periods_emax(num_periods_int, max_states_period)
    DOUBLE PRECISION, INTENT(IN)    :: delta_int

    INTEGER, INTENT(IN)             :: max_states_period
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: edu_max_int 
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: min_idx_int

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(:)

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int

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

    !# Transfer auxiliary variable to global variable.
    num_points_interp = num_points_interp_int
    is_interpolated = is_interpolated_int
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int
    edu_start = edu_start_int
    is_myopic = is_myopic_int
    is_debug = is_debug_int
    min_idx = min_idx_int
    edu_max = edu_max_int
    delta = delta_int

    ! Call FORTRAN solution
    CALL fort_solve(periods_payoffs_systematic_int, states_number_period_int, mapping_state_idx_int, periods_emax_int, states_all_int, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, periods_draws_emax)

    ! Assign to initial objects for return to PYTHON
    periods_payoffs_systematic = periods_payoffs_systematic_int   
    states_number_period = states_number_period_int
    mapping_state_idx = mapping_state_idx_int 
    periods_emax = periods_emax_int 
    states_all = states_all_int

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_evaluate(crit_val, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, is_interpolated_int, num_draws_emax_int, num_periods_int, num_points_interp_int, is_myopic_int, edu_start_int, is_debug_int, edu_max_int, min_idx_int, delta_int, data_array, num_agents_est_int, num_draws_prob_int, tau_int, periods_draws_emax, periods_draws_prob)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: crit_val

    INTEGER, INTENT(IN)             :: num_draws_prob_int
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: num_agents_est_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: min_idx_int

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_prob(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: data_array(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)
    DOUBLE PRECISION, INTENT(IN)    :: delta_int

    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(:)
    DOUBLE PRECISION, INTENT(IN)    :: tau_int

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int

    !/* internal */

    INTEGER, ALLOCATABLE            :: mapping_state_idx(:, :, :, :, :)
    INTEGER, ALLOCATABLE            :: states_number_period(:)
    INTEGER, ALLOCATABLE            :: states_all(:, :, :)

    DOUBLE PRECISION, ALLOCATABLE   :: periods_payoffs_systematic(:, :, :)
    DOUBLE PRECISION, ALLOCATABLE   :: periods_emax(:, :)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# Transfer auxiliary variable to global variable.
    num_points_interp = num_points_interp_int
    is_interpolated = is_interpolated_int
    num_agents_est = num_agents_est_int
    num_draws_prob = num_draws_prob_int
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int
    edu_start = edu_start_int
    is_myopic = is_myopic_int
    is_debug = is_debug_int
    min_idx = min_idx_int
    edu_max = edu_max_int
    delta = delta_int
    tau = tau_int

    ! Solve them model for the given parametrization.
    CALL fort_solve(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, periods_draws_emax)

    ! Evaluate the criterion function building on the solution.
    CALL fort_evaluate(crit_val, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_array, periods_draws_prob)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_simulate(dataset, periods_payoffs_systematic, mapping_state_idx, periods_emax, num_periods_int, states_all, num_agents_sim, edu_start_int, edu_max_int, delta_int, periods_draws_sims, shocks_cholesky)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: dataset(num_agents_sim * num_periods_int, 8)

    DOUBLE PRECISION, INTENT(IN)    :: periods_payoffs_systematic(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_sims(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: delta_int

    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: edu_start_int

    INTEGER, INTENT(IN)             :: mapping_state_idx(:, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_all(:, :, :)
    INTEGER, INTENT(IN)             :: num_agents_sim

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# This assignment is required to allow for the explicit shape arguments.
    min_idx = SIZE(mapping_state_idx, 4)

    !# Transfer auxiliary variable to global variable.
    num_periods = num_periods_int
    edu_start = edu_start_int
    edu_max = edu_max_int
    delta = delta_int

    CALL fort_simulate(dataset, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, num_agents_sim, periods_draws_sims, shocks_cholesky)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_backward_induction(periods_emax, num_periods_int, max_states_period, periods_draws_emax, num_draws_emax_int, states_number_period, periods_payoffs_systematic, edu_max_int, edu_start_int, mapping_state_idx, states_all, delta_int, is_debug_int, is_interpolated_int, num_points_interp_int, shocks_cholesky)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: periods_emax(num_periods_int, max_states_period)

    DOUBLE PRECISION, INTENT(IN)    :: periods_payoffs_systematic(:, :, :   )
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: delta_int

    INTEGER, INTENT(IN)             :: mapping_state_idx(:, :, :, :, :)    
    INTEGER, INTENT(IN)             :: states_number_period(:)
    INTEGER, INTENT(IN)             :: states_all(:, :, :)
    INTEGER, INTENT(IN)             :: max_states_period
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: edu_start_int

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_debug_int

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    !# Transfer auxiliary variable to global variable.
    num_points_interp = num_points_interp_int
    is_interpolated = is_interpolated_int
    num_periods = num_periods_int
    edu_start = edu_start_int
    num_draws_emax = num_draws_emax_int
    is_debug = is_debug_int
    edu_max = edu_max_int
    delta = delta_int

    ! This assignment is required as the variables are initialized with zero 
    ! by the interface generator.
    periods_emax = MISSING_FLOAT

    ! Call actual function of interest
    CALL fort_backward_induction(periods_emax, periods_draws_emax, states_number_period, periods_payoffs_systematic, mapping_state_idx, states_all, shocks_cholesky)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_create_state_space(states_all, states_number_period, mapping_state_idx, max_states_period, num_periods_int, edu_start_int, edu_max_int, min_idx_int)
    
    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: mapping_state_idx(num_periods_int, num_periods_int, num_periods_int, min_idx_int, 2)
    INTEGER, INTENT(OUT)            :: states_all(num_periods_int, 100000, 4)
    INTEGER, INTENT(OUT)            :: states_number_period(num_periods_int)
    INTEGER, INTENT(OUT)            :: max_states_period

    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: min_idx_int

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# Transfer auxiliary variable to global variable.
    num_periods = num_periods_int
    edu_start = edu_start_int
    min_idx = min_idx_int
    edu_max = edu_max_int

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, max_states_period)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_calculate_payoffs_systematic(periods_payoffs_systematic, num_periods_int, states_number_period, states_all, edu_start_int, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, max_states_period)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: periods_payoffs_systematic(num_periods_int, max_states_period, 4)

    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(1)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(3)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(6)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(6)

    INTEGER, INTENT(IN)             :: states_number_period(:)
    INTEGER, INTENT(IN)             :: max_states_period
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    num_periods = num_periods_int
    edu_start = edu_start_int
    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************