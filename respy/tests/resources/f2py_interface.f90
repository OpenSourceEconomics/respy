!******************************************************************************
!******************************************************************************
!   
!   This module serves as the F2PY interface to the core functions. All the 
!   functions have counterparts as PYTHON implementations.
!
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_criterion(crit_val, x, is_interpolated_int, num_draws_emax_int, num_periods_int, num_points_interp_int, is_myopic_int, edu_start_int, is_debug_int, edu_max_int, min_idx_int, delta_int, data_est_int, num_agents_est_int, num_draws_prob_int, tau_int, periods_draws_emax_int, periods_draws_prob_int)

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

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_prob_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: data_est_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: tau_int

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int

    !/* internal objects            */

    DOUBLE PRECISION                :: shocks_cholesky(4, 4)
    DOUBLE PRECISION                :: coeffs_home(1)
    DOUBLE PRECISION                :: coeffs_edu(3)
    DOUBLE PRECISION                :: coeffs_a(6)
    DOUBLE PRECISION                :: coeffs_b(6)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Assign global RESFORT variables 
    max_states_period = SIZE(states_all, 2)

    ! Transfer global RESFORT variables
    num_points_interp = num_points_interp_int
    is_interpolated = is_interpolated_int
    num_agents_est = num_agents_est_int
    num_draws_emax = num_draws_emax_int
    num_draws_prob = num_draws_prob_int
    num_periods = num_periods_int
    data_est = data_est_int
    is_myopic = is_myopic_int
    edu_start = edu_start_int
    is_debug = is_debug_int
    min_idx = min_idx_int
    edu_max = edu_max_int
    delta = delta_int
    tau = tau_int
	
    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF (ALLOCATED(mapping_state_idx)) DEALLOCATE(mapping_state_idx)
    IF (ALLOCATED(periods_payoffs_systematic)) DEALLOCATE(periods_payoffs_systematic)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)
    IF (ALLOCATED(periods_emax)) DEALLOCATE(periods_emax)
    IF (ALLOCATED(states_number_period)) DEALLOCATE(states_number_period)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)

    !# Distribute model parameters
    CALL dist_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x)

    ! Solve requested model
    CALL fort_solve(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, periods_draws_emax_int, delta, is_debug, is_interpolated, is_myopic, edu_start, edu_max)

    ! Evaluate criterion function for observed data
    CALL fort_evaluate(crit_val, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_est, periods_draws_prob_int, delta, tau, edu_start, edu_max)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_solve(periods_payoffs_systematic_int, states_number_period_int, mapping_state_idx_int, periods_emax_int, states_all_int, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, is_interpolated_int, num_draws_emax_int, num_periods_int, num_points_interp_int, is_myopic_int, edu_start_int, is_debug_int, edu_max_int, min_idx_int, delta_int, periods_draws_emax_int, max_states_period_int)
    
    ! The presence of max_states_period breaks the equality of interfaces. However, this is required so that the size of the return arguments is known from the beginning.

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: mapping_state_idx_int(num_periods_int, num_periods_int, num_periods_int, min_idx_int, 2)
    INTEGER, INTENT(OUT)            :: states_all_int(num_periods_int, max_states_period_int, 4)
    INTEGER, INTENT(OUT)            :: states_number_period_int(num_periods_int)

    DOUBLE PRECISION, INTENT(OUT)   :: periods_payoffs_systematic_int(num_periods_int, max_states_period_int, 4)
    DOUBLE PRECISION, INTENT(OUT)   :: periods_emax_int(num_periods_int, max_states_period_int)
    DOUBLE PRECISION, INTENT(IN)    :: delta_int

    INTEGER, INTENT(IN)             :: max_states_period_int
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: edu_max_int 
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: min_idx_int

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(:)

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int

!-----------------------------------------------------------------------------
! Algorithm
!----------------------------------------------------------------------------- 

    !# Transfer global RESFORT variables
    max_states_period = max_states_period_int
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

    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF (ALLOCATED(mapping_state_idx)) DEALLOCATE(mapping_state_idx)
    IF (ALLOCATED(periods_payoffs_systematic)) DEALLOCATE(periods_payoffs_systematic)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)
    IF (ALLOCATED(periods_emax)) DEALLOCATE(periods_emax)
    IF (ALLOCATED(states_number_period)) DEALLOCATE(states_number_period)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)

    ! Call FORTRAN solution
    CALL fort_solve(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, periods_draws_emax_int, delta, is_debug, is_interpolated, is_myopic, edu_start, edu_max)

    ! Assign to initial objects for return to PYTHON
    periods_payoffs_systematic_int = periods_payoffs_systematic
    states_number_period_int = states_number_period
    mapping_state_idx_int = mapping_state_idx
    periods_emax_int = periods_emax
    states_all_int = states_all

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_evaluate(crit_val, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, is_interpolated_int, num_draws_emax_int, num_periods_int, num_points_interp_int, is_myopic_int, edu_start_int, is_debug_int, edu_max_int, min_idx_int, delta_int, data_est_int, num_agents_est_int, num_draws_prob_int, tau_int, periods_draws_emax_int, periods_draws_prob_int)

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

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_prob_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: data_est_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)
    DOUBLE PRECISION, INTENT(IN)    :: delta_int

    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(:)
    DOUBLE PRECISION, INTENT(IN)    :: tau_int

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Transfer global RESFORT variables
    num_points_interp = num_points_interp_int
    is_interpolated = is_interpolated_int
    num_agents_est = num_agents_est_int
    num_draws_prob = num_draws_prob_int
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int
    data_est = data_est_int
    edu_start = edu_start_int
    is_myopic = is_myopic_int
    is_debug = is_debug_int
    min_idx = min_idx_int
    edu_max = edu_max_int
    delta = delta_int
    tau = tau_int

    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF (ALLOCATED(mapping_state_idx)) DEALLOCATE(mapping_state_idx)
    IF (ALLOCATED(periods_payoffs_systematic)) DEALLOCATE(periods_payoffs_systematic)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)
    IF (ALLOCATED(periods_emax)) DEALLOCATE(periods_emax)
    IF (ALLOCATED(states_number_period)) DEALLOCATE(states_number_period)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)
    
    ! Solve requested model
    CALL fort_solve(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, periods_draws_emax_int, delta, is_debug, is_interpolated, is_myopic, edu_start, edu_max)

    ! Evaluate criterion function for observed data
    CALL fort_evaluate(crit_val, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_est, periods_draws_prob_int, delta, tau, edu_start, edu_max)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_simulate(data_sim_int, periods_payoffs_systematic_int, mapping_state_idx_int, periods_emax_int, states_all_int, shocks_cholesky, num_periods_int, edu_start_int, edu_max_int, delta_int, num_agents_sim_int, periods_draws_sims)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: data_sim_int(num_agents_sim_int * num_periods_int, 8)

    DOUBLE PRECISION, INTENT(IN)    :: periods_payoffs_systematic_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_sims(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: delta_int

    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: edu_start_int

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)             :: num_agents_sim_int

    DOUBLE PRECISION, ALLOCATABLE   :: data_sim(:, :)
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Assign global RESPFRT variables
    min_idx = SIZE(mapping_state_idx_int, 4)
    max_states_period = SIZE(states_all_int, 2)

    ! Transfer global RESFORT variables
    num_agents_sim = num_agents_sim_int
    num_periods = num_periods_int
    edu_start = edu_start_int
    edu_max = edu_max_int
    delta = delta_int

    ! Call function of interest
    CALL fort_simulate(data_sim, periods_payoffs_systematic_int, mapping_state_idx_int, periods_emax_int, states_all_int, num_agents_sim, periods_draws_sims, shocks_cholesky, delta, edu_start, edu_max)

    ! Assign to initial objects for return to PYTHON
    data_sim_int = data_sim

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_backward_induction(periods_emax_int, num_periods_int, max_states_period_int, periods_draws_emax_int, num_draws_emax_int, states_number_period_int, periods_payoffs_systematic_int, edu_max_int, edu_start_int, mapping_state_idx_int, states_all_int, delta_int, is_debug_int, is_interpolated_int, num_points_interp_int, shocks_cholesky)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: periods_emax_int(num_periods_int, max_states_period_int)

    DOUBLE PRECISION, INTENT(IN)    :: periods_payoffs_systematic_int(:, :, :   )
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: delta_int

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :)    
    INTEGER, INTENT(IN)             :: states_number_period_int(:)
    INTEGER, INTENT(IN)             :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)             :: max_states_period_int
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: edu_start_int

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_debug_int

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    !# Transfer auxiliary variable to global variable.
    max_states_period = max_states_period_int
    num_points_interp = num_points_interp_int
    is_interpolated = is_interpolated_int
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int
    edu_start = edu_start_int
    is_debug = is_debug_int
    edu_max = edu_max_int
    delta = delta_int

    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF(ALLOCATED(periods_emax)) DEALLOCATE(periods_emax)

    ! Call actual function of interest
    CALL fort_backward_induction(periods_emax, periods_draws_emax_int, states_number_period_int, periods_payoffs_systematic_int, mapping_state_idx_int, states_all_int, shocks_cholesky, delta, is_debug, is_interpolated, edu_start, edu_max)
    
    ! Allocate to intermidiaries
    periods_emax_int = periods_emax

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_create_state_space(states_all_int, states_number_period_int, mapping_state_idx_int, max_states_period_int, num_periods_int, edu_start_int, edu_max_int, min_idx_int)
    
    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: mapping_state_idx_int(num_periods_int, num_periods_int, num_periods_int, min_idx_int, 2)
    INTEGER, INTENT(OUT)            :: states_all_int(num_periods_int, 100000, 4)
    INTEGER, INTENT(OUT)            :: states_number_period_int(num_periods_int)
    INTEGER, INTENT(OUT)            :: max_states_period_int

    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: min_idx_int


!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    !# Transfer global RESFORT variables
    max_states_period = max_states_period_int
    num_periods = num_periods_int
    edu_start = edu_start_int
    min_idx = min_idx_int
    edu_max = edu_max_int
    
    states_all_int = MISSING_INT

    ! Ensure that there is no problem with the repeated allocation of the containers.    
    IF (ALLOCATED(mapping_state_idx)) DEALLOCATE(mapping_state_idx)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)
    IF (ALLOCATED(states_number_period)) DEALLOCATE(states_number_period)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, edu_start, edu_max)

    states_all_int(:, :max_states_period, :) = states_all
    states_number_period_int = states_number_period

    ! Updated global variables
    mapping_state_idx_int = mapping_state_idx
    max_states_period_int = max_states_period

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_calculate_payoffs_systematic(periods_payoffs_systematic_int, num_periods_int, states_number_period_int, states_all_int, edu_start_int, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, max_states_period_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: periods_payoffs_systematic_int(num_periods_int, max_states_period_int, 4)

    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(1)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(3)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(6)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(6)

    INTEGER, INTENT(IN)             :: states_number_period_int(:)
    INTEGER, INTENT(IN)             :: max_states_period_int
    INTEGER, INTENT(IN)             :: states_all_int(:,:,:)
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ! Transfer global RESOFORT variables
    max_states_period = max_states_period_int
    num_periods = num_periods_int
    edu_start = edu_start_int

    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF(ALLOCATED(periods_payoffs_systematic)) DEALLOCATE(periods_payoffs_systematic)

    ! Call function of interest
    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period_int, states_all_int, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start)

    periods_payoffs_systematic_int = periods_payoffs_systematic

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_normal_pdf(rslt, x, mean, sd)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)      :: rslt

    DOUBLE PRECISION, INTENT(IN)       :: x
    DOUBLE PRECISION, INTENT(IN)       :: mean
    DOUBLE PRECISION, INTENT(IN)       :: sd

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Call function of interest
    rslt = normal_pdf(x, mean, sd)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_pinv(rslt, A, m)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(m, m)

    DOUBLE PRECISION, INTENT(IN)    :: A(m, m)
    
    INTEGER, INTENT(IN)             :: m

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Call function of interest
    rslt = pinv(A, m)
    
END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_svd(U, S, VT, A, m)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: S(m) 
    DOUBLE PRECISION, INTENT(OUT)   :: U(m, m)
    DOUBLE PRECISION, INTENT(OUT)   :: VT(m, m)

    DOUBLE PRECISION, INTENT(IN)    :: A(m, m)
    
    INTEGER, INTENT(IN)             :: m

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ! Call function of interest
    CALL svd(U, S, VT, A, m)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_get_future_value(emax_simulated, num_periods_int, num_draws_emax_int, period, k, draws_emax_transformed, payoffs_systematic, edu_max_int, edu_start_int, periods_emax_int, states_all_int, mapping_state_idx_int, delta_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: emax_simulated

    DOUBLE PRECISION, INTENT(IN)    :: draws_emax_transformed(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: payoffs_systematic(:)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax_int(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: delta_int

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:,:,:,:,:)
    INTEGER, INTENT(IN)             :: states_all_int(:,:,:)
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ! Assign global RESFORT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)

    !# Transfer global RESFORT variables
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int
    edu_start = edu_start_int
    edu_max = edu_max_int
    delta = delta_int

    ! Call function of interest
    CALL get_future_value(emax_simulated, draws_emax_transformed, period, k, payoffs_systematic, mapping_state_idx_int, states_all_int, periods_emax_int, delta, edu_start, edu_max)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_standard_normal(draw, dim)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(IN)             :: dim
    
    DOUBLE PRECISION, INTENT(OUT)   :: draw(dim)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Call function of interest
    CALL standard_normal(draw)

END SUBROUTINE 
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_determinant(det, A)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: det

    DOUBLE PRECISION, INTENT(IN)    :: A(:, :)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Call function of interest
    det = determinant(A)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_inverse(inv, A, n)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: inv(n, n)

    DOUBLE PRECISION, INTENT(IN)    :: A(:, :)

    INTEGER, INTENT(IN)             :: n

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Call function of interest
    inv = inverse(A, n)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_trace(rslt, A)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT) :: rslt

    DOUBLE PRECISION, INTENT(IN)  :: A(:,:)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Call function of interest
    rslt = trace_fun(A)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_clip_value(clipped_value, value, lower_bound, upper_bound, num_values)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: clipped_value(num_values)

    DOUBLE PRECISION, INTENT(IN)        :: value(:)
    DOUBLE PRECISION, INTENT(IN)        :: lower_bound
    DOUBLE PRECISION, INTENT(IN)        :: upper_bound 

    INTEGER, INTENT(IN)                 :: num_values

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Call function of interest
    clipped_value = clip_value(value, lower_bound, upper_bound)


END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_get_pred_info(r_squared, bse, Y, P, X, num_states, num_covars)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: bse(num_covars)
    DOUBLE PRECISION, INTENT(OUT)   :: r_squared
    
    DOUBLE PRECISION, INTENT(IN)    :: X(num_states, num_covars)
    DOUBLE PRECISION, INTENT(IN)    :: Y(num_states)
    DOUBLE PRECISION, INTENT(IN)    :: P(num_states)
    
    INTEGER, INTENT(IN)             :: num_states
    INTEGER, INTENT(IN)             :: num_covars

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Call function of interest
    CALL get_pred_info(r_squared, bse, Y, P, X, num_states, num_covars)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_point_predictions(Y, X, coeffs, num_states)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: Y(num_states)

    DOUBLE PRECISION, INTENT(IN)        :: coeffs(:)
    DOUBLE PRECISION, INTENT(IN)        :: X(:,:)
    
    INTEGER, INTENT(IN)                 :: num_states

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Call function of interest
    CALL point_predictions(Y, X, coeffs, num_states)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_get_predictions(predictions, endogenous, exogenous, maxe, is_simulated, num_points_interp_int, num_states)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)               :: predictions(num_states)

    DOUBLE PRECISION, INTENT(IN)                :: exogenous(:, :)
    DOUBLE PRECISION, INTENT(IN)                :: endogenous(:)
    DOUBLE PRECISION, INTENT(IN)                :: maxe(:)

    INTEGER, INTENT(IN)                         :: num_states
    INTEGER, INTENT(IN)                         :: num_points_interp_int

    LOGICAL, INTENT(IN)                         :: is_simulated(:)

!------------------------------------------------------------------------------
! Algorithm

!------------------------------------------------------------------------------
    
    ! Transfer global RESFORT variables
    num_points_interp = num_points_interp_int

    ! Call function of interest
    CALL get_predictions(predictions, endogenous, exogenous, maxe, is_simulated, num_states)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_random_choice(sample, candidates, num_candidates, num_points)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: sample(num_points)

    INTEGER, INTENT(IN)             :: candidates(:)
    INTEGER, INTENT(IN)             :: num_candidates
    INTEGER, INTENT(IN)             :: num_points

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Call function of interest
     CALL random_choice(sample, candidates, num_candidates, num_points)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_get_coefficients(coeffs, Y, X, num_covars, num_states)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: coeffs(num_covars)

    DOUBLE PRECISION, INTENT(IN)    :: X(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: Y(:)
    
    INTEGER, INTENT(IN)             :: num_covars
    INTEGER, INTENT(IN)             :: num_states

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Call function of interest
    CALL get_coefficients(coeffs, Y, X, num_covars, num_states)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_get_endogenous_variable(exogenous_variable, period, num_periods_int, num_states, delta_int, periods_payoffs_systematic_int, edu_max_int, edu_start_int, mapping_state_idx_int, periods_emax_int, states_all_int, is_simulated, num_draws_emax_int, maxe, draws_emax_transformed)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: exogenous_variable(num_states)

    DOUBLE PRECISION, INTENT(IN)        :: periods_payoffs_systematic_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)        :: draws_emax_transformed(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: periods_emax_int(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: maxe(:)
    DOUBLE PRECISION, INTENT(IN)        :: delta_int

    INTEGER, INTENT(IN)                 :: mapping_state_idx_int(:, :, :, :, :)    
    INTEGER, INTENT(IN)                 :: states_all_int(:, :, :)    
    INTEGER, INTENT(IN)                 :: num_draws_emax_int
    INTEGER, INTENT(IN)                 :: edu_max_int    
    INTEGER, INTENT(IN)                 :: num_periods_int
    INTEGER, INTENT(IN)                 :: num_states
    INTEGER, INTENT(IN)                 :: edu_start_int
    INTEGER, INTENT(IN)                 :: period


    LOGICAL, INTENT(IN)                 :: is_simulated(:)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ! Transfer global RESFORT variables
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int
    edu_start = edu_start_int
    edu_max = edu_max_int
    delta = delta_int

    ! Call function of interest
    CALL get_endogenous_variable(exogenous_variable, period, num_states, periods_payoffs_systematic_int, mapping_state_idx_int, periods_emax_int, states_all_int, is_simulated, maxe, draws_emax_transformed, delta, edu_start, edu_max)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_get_exogenous_variables(independent_variables, maxe, period, num_periods_int, num_states, delta_int, periods_payoffs_systematic_int, shifts, edu_max_int, edu_start_int, mapping_state_idx_int, periods_emax_int, states_all_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)        :: independent_variables(num_states, 9)
    DOUBLE PRECISION, INTENT(OUT)        :: maxe(num_states)


    DOUBLE PRECISION, INTENT(IN)        :: periods_payoffs_systematic_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)        :: periods_emax_int(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: shifts(:)
    DOUBLE PRECISION, INTENT(IN)        :: delta_int

    INTEGER, INTENT(IN)                 :: mapping_state_idx_int(:, :, :, :, :)    
    INTEGER, INTENT(IN)                 :: states_all_int(:, :, :)    
    INTEGER, INTENT(IN)                 :: edu_max_int
    INTEGER, INTENT(IN)                 :: num_periods_int
    INTEGER, INTENT(IN)                 :: num_states
    INTEGER, INTENT(IN)                 :: edu_start_int
    INTEGER, INTENT(IN)                 :: period

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    !# Assign global RESFORT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)

    !# Transfer global RESFORT variables
    num_periods = num_periods_int
    edu_start = edu_start_int
    edu_max = edu_max_int
    delta = delta_int

    ! Call function of interest
    CALL get_exogenous_variables(independent_variables, maxe, period, num_states, periods_payoffs_systematic_int, shifts, mapping_state_idx_int, periods_emax_int, states_all_int, delta, edu_start, edu_max)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_get_simulated_indicator(is_simulated, num_points, num_states, period, is_debug_int, num_periods_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    LOGICAL, INTENT(OUT)            :: is_simulated(num_states)

    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: num_states
    INTEGER, INTENT(IN)             :: num_points
    INTEGER, INTENT(IN)             :: period

    LOGICAL, INTENT(IN)             :: is_debug_int

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    !# Transfer global RESFORT variables
    num_periods = num_periods_int
    is_debug = is_debug_int

    ! Call function of interest
    is_simulated = get_simulated_indicator(num_points, num_states, period, .True.)

END SUBROUTINE
!******************************************************************************
!******************************************************************************