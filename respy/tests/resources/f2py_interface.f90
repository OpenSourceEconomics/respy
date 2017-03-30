!******************************************************************************
!******************************************************************************
!
!   This module serves as the F2PY interface to the core functions. All the
!   functions have counterparts as PYTHON implementations.
!
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_criterion(crit_val, x, is_interpolated_int, num_draws_emax_int, num_periods_int, num_points_interp_int, is_myopic_int, edu_start_int, is_debug_int, edu_max_int, data_est_int, num_draws_prob_int, tau_int, periods_draws_emax_int, periods_draws_prob_int, states_all_int, states_number_period_int, mapping_state_idx_int, max_states_period_int, measure, mean, fort_slsqp_maxiter, fort_slsqp_ftol, fort_slsqp_eps)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objeFcts        */

    DOUBLE PRECISION, INTENT(OUT)   :: crit_val

    DOUBLE PRECISION, INTENT(IN)    :: x(32)

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_number_period_int(:)
    INTEGER, INTENT(IN)             :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: max_states_period_int
    INTEGER, INTENT(IN)             :: num_draws_prob_int
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: fort_slsqp_maxiter
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: edu_max_int

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_prob_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: data_est_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: fort_slsqp_ftol
    DOUBLE PRECISION, INTENT(IN)    :: fort_slsqp_eps
    DOUBLE PRECISION, INTENT(IN)    :: tau_int

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int
    LOGICAL, INTENT(IN)             :: mean

    CHARACTER(10), INTENT(IN)       :: measure

    !/* internal objects            */

    DOUBLE PRECISION, ALLOCATABLE   :: opt_ambi_details(:, :, :)

    DOUBLE PRECISION                :: contribs(SIZE(data_est_int, 1))

    INTEGER                         :: dist_optim_paras_info

    CHARACTER(225)                  :: file_sim_mock

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Assign global RESPFRT variables
    max_states_period = max_states_period_int
    min_idx = SIZE(mapping_state_idx_int, 4)
    num_obs = SIZE(data_est_int, 1)

    ! Transfer global RESFORT variables
    num_points_interp = num_points_interp_int
    num_agents_est = SIZE(data_est_int, 1) / INT(num_periods_int)
    num_draws_emax = num_draws_emax_int
    num_draws_prob = num_draws_prob_int
    num_periods = num_periods_int

    ! Construct derived types
    optimizer_options%slsqp%maxiter = fort_slsqp_maxiter
    optimizer_options%slsqp%ftol = fort_slsqp_ftol
    optimizer_options%slsqp%eps = fort_slsqp_eps

    ambi_spec%measure = measure
    ambi_spec%mean = mean

    !# Distribute model parameters
    CALL dist_optim_paras(optim_paras, x, dist_optim_paras_info)

    CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period_int, states_all_int, edu_start_int, max_states_period_int, optim_paras)

    CALL fort_backward_induction(periods_emax, opt_ambi_details, num_periods_int, is_myopic_int, max_states_period_int, periods_draws_emax_int, num_draws_emax_int, states_number_period_int, periods_rewards_systematic, edu_max_int, edu_start_int, mapping_state_idx_int, states_all_int, is_debug_int, is_interpolated_int, num_points_interp_int, ambi_spec, optim_paras, optimizer_options, file_sim_mock, .False.)

    CALL fort_contributions(contribs, periods_rewards_systematic, mapping_state_idx_int, periods_emax, states_all_int, data_est_int, periods_draws_prob_int, tau_int, edu_start_int, edu_max_int, num_periods_int, num_draws_prob_int, optim_paras)

    crit_val = get_log_likl(contribs)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_contributions(contribs, periods_rewards_systematic_int, mapping_state_idx_int, periods_emax_int, states_all_int, data_est_int, periods_draws_prob_int, tau_int, edu_start_int, edu_max_int, num_periods_int, num_draws_prob_int, shocks_cholesky, delta, num_agents_est_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: contribs(num_agents_est_int * num_periods_int)

    DOUBLE PRECISION, INTENT(IN)    :: periods_rewards_systematic_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_prob_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: data_est_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: tau_int
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)             :: num_agents_est_int
    INTEGER, INTENT(IN)             :: num_draws_prob_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: edu_max_int

    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(:, :)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Assign global RESPFRT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)

    ! Transfer global RESFORT variables
    periods_rewards_systematic = periods_rewards_systematic_int
    num_agents_est = num_agents_est_int
    num_draws_prob = num_draws_prob_int
    num_periods = num_periods_int
    tau = tau_int

    ! Construct derived types
    optim_paras%shocks_cholesky = shocks_cholesky
    optim_paras%delta = delta

    CALL fort_contributions(contribs, periods_rewards_systematic, mapping_state_idx_int, periods_emax_int, states_all_int, data_est_int, periods_draws_prob_int, tau_int, edu_start_int, edu_max_int, num_periods_int, num_draws_prob_int, optim_paras)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_solve(periods_rewards_systematic_int, states_number_period_int, mapping_state_idx_int, periods_emax_int, states_all_int, is_interpolated_int, num_points_interp_int, num_draws_emax_int, num_periods_int, is_myopic_int, edu_start_int, is_debug_int, edu_max_int, min_idx_int, periods_draws_emax_int, measure, mean, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, level, delta, file_sim, fort_slsqp_maxiter, fort_slsqp_ftol, fort_slsqp_eps, max_states_period_int)

    ! The presence of max_states_period breaks the quality of interfaces. However, this is required so that the size of the return arguments is known from the beginning.

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: mapping_state_idx_int(num_periods_int, num_periods_int, num_periods_int, min_idx_int, 2)
    INTEGER, INTENT(OUT)            :: states_all_int(num_periods_int, max_states_period_int, 4)
    INTEGER, INTENT(OUT)            :: states_number_period_int(num_periods_int)

    DOUBLE PRECISION, INTENT(OUT)   :: periods_rewards_systematic_int(num_periods_int, max_states_period_int, 4)
    DOUBLE PRECISION, INTENT(OUT)   :: periods_emax_int(num_periods_int, max_states_period_int)

    INTEGER, INTENT(IN)             :: max_states_period_int
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: fort_slsqp_maxiter
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: min_idx_int

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: fort_slsqp_ftol
    DOUBLE PRECISION, INTENT(IN)    :: fort_slsqp_eps
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(:)
    DOUBLE PRECISION, INTENT(IN)    :: level(1)
    DOUBLE PRECISION, INTENT(IN)    :: delta(1)

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int
    LOGICAL, INTENT(IN)             :: mean

    CHARACTER(225), INTENT(IN)      :: file_sim
    CHARACTER(10), INTENT(IN)       :: measure

!-----------------------------------------------------------------------------
! Algorithm
!-----------------------------------------------------------------------------

    !# Transfer global RESFORT variables
    num_points_interp = num_points_interp_int
    max_states_period = max_states_period_int
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int
    min_idx = min_idx_int

    IF (mean) THEN
        num_free_ambi = 2
    ELSE
        num_free_ambi = 4
    END IF

    ! Construct derived types
    optimizer_options%slsqp%maxiter = fort_slsqp_maxiter
    optimizer_options%slsqp%ftol = fort_slsqp_ftol
    optimizer_options%slsqp%eps = fort_slsqp_eps

    optim_paras%shocks_cholesky = shocks_cholesky
    optim_paras%coeffs_home = coeffs_home
    optim_paras%coeffs_edu = coeffs_edu
    optim_paras%coeffs_a = coeffs_a
    optim_paras%coeffs_b = coeffs_b
    optim_paras%level = level
    optim_paras%delta = delta

    ambi_spec%measure = measure
    ambi_spec%mean = mean

    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF (ALLOCATED(periods_rewards_systematic)) DEALLOCATE(periods_rewards_systematic)
    IF (ALLOCATED(states_number_period)) DEALLOCATE(states_number_period)
    IF (ALLOCATED(mapping_state_idx)) DEALLOCATE(mapping_state_idx)
    IF (ALLOCATED(periods_emax)) DEALLOCATE(periods_emax)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)

    ! Call FORTRAN solution
    CALL fort_solve(periods_rewards_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, is_interpolated_int, num_points_interp_int, num_draws_emax, num_periods, is_myopic_int, edu_start_int, is_debug_int, edu_max_int, min_idx, periods_draws_emax_int, ambi_spec, optim_paras, optimizer_options, file_sim)

    ! Assign to initial objects for return to PYTHON
    periods_rewards_systematic_int = periods_rewards_systematic
    states_number_period_int = states_number_period
    mapping_state_idx_int = mapping_state_idx
    periods_emax_int = periods_emax
    states_all_int = states_all

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_simulate(data_sim_int, periods_rewards_systematic_int, mapping_state_idx_int, periods_emax_int, states_all_int, num_periods_int, edu_start_int, edu_max_int, num_agents_sim_int, periods_draws_sims, seed_sim, file_sim, shocks_cholesky, delta)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: data_sim_int(num_agents_sim_int * num_periods_int, 21)

    DOUBLE PRECISION, INTENT(IN)    :: periods_rewards_systematic_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_sims(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: seed_sim

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)             :: num_agents_sim_int

    CHARACTER(225), INTENT(IN)      :: file_sim

    !/* internal objects        */

    DOUBLE PRECISION, ALLOCATABLE   :: data_sim(:, :)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Assign global RESPFRT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)

    ! Transfer global RESFORT variables
    num_agents_sim = num_agents_sim_int
    num_periods = num_periods_int

    ! Construct derived types
    optim_paras%shocks_cholesky = shocks_cholesky
    optim_paras%delta = delta

    ! Call function of interest
    CALL fort_simulate(data_sim, periods_rewards_systematic_int, mapping_state_idx_int, periods_emax_int, states_all_int, num_agents_sim, periods_draws_sims, edu_start_int, edu_max_int, seed_sim, file_sim, optim_paras)

    ! Assign to initial objects for return to PYTHON
    data_sim_int = data_sim

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_backward_induction(periods_emax_int, num_periods_int, is_myopic_int, max_states_period_int, periods_draws_emax_int, num_draws_emax_int, states_number_period_int, periods_rewards_systematic_int, edu_max_int, edu_start_int, mapping_state_idx_int, states_all_int,  is_debug_int, is_interpolated_int, num_points_interp_int, measure, mean, shocks_cholesky, level, delta, fort_slsqp_maxiter, fort_slsqp_ftol, fort_slsqp_eps, file_sim, is_write)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: periods_emax_int(num_periods_int, max_states_period_int)

    DOUBLE PRECISION, INTENT(IN)    :: periods_rewards_systematic_int(:, :, :   )
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: fort_slsqp_ftol
    DOUBLE PRECISION, INTENT(IN)    :: fort_slsqp_eps
    DOUBLE PRECISION, INTENT(IN)    :: level(1)
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_number_period_int(:)
    INTEGER, INTENT(IN)             :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)             :: max_states_period_int
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: fort_slsqp_maxiter
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: edu_max_int

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int
    LOGICAL, INTENT(IN)             :: is_write
    LOGICAL, INTENT(IN)             :: mean

    CHARACTER(225), INTENT(IN)      :: file_sim
    CHARACTER(10), INTENT(IN)       :: measure

    !/* internal objects*/

    DOUBLE PRECISION, ALLOCATABLE   :: opt_ambi_details(:, :, :)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    !# Transfer auxiliary variable to global variable.
    num_points_interp = num_points_interp_int
    max_states_period = max_states_period_int
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int

    IF (mean) THEN
        num_free_ambi = 2
    ELSE
        num_free_ambi = 4
    END IF

    ! Construct derived types
    optimizer_options%slsqp%maxiter = fort_slsqp_maxiter
    optimizer_options%slsqp%ftol = fort_slsqp_ftol
    optimizer_options%slsqp%eps = fort_slsqp_eps

    optim_paras%shocks_cholesky = shocks_cholesky
    optim_paras%level = level
    optim_paras%delta = delta

    ambi_spec%measure = measure
    ambi_spec%mean = mean

    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF(ALLOCATED(periods_emax)) DEALLOCATE(periods_emax)

    ! Call actual function of interest
    CALL fort_backward_induction(periods_emax, opt_ambi_details, num_periods_int, is_myopic_int, max_states_period_int, periods_draws_emax_int, num_draws_emax_int, states_number_period_int, periods_rewards_systematic, edu_max_int, edu_start_int, mapping_state_idx_int, states_all_int, is_debug_int, is_interpolated_int, num_points_interp_int, ambi_spec, optim_paras, optimizer_options, file_sim, is_write)

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
    min_idx = min_idx_int

    states_all_int = MISSING_INT

    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF (ALLOCATED(states_number_period)) DEALLOCATE(states_number_period)
    IF (ALLOCATED(mapping_state_idx)) DEALLOCATE(mapping_state_idx)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, num_periods_int, edu_start_int, edu_max_int, min_idx_int)

    states_all_int(:, :max_states_period, :) = states_all
    states_number_period_int = states_number_period

    ! Updated global variables
    mapping_state_idx_int = mapping_state_idx
    max_states_period_int = max_states_period

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE f2py_calculate_rewards_systematic(periods_rewards_systematic_int, num_periods_int, states_number_period_int, states_all_int, edu_start_int, max_states_period_int, coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: periods_rewards_systematic_int(num_periods_int, max_states_period_int, 4)

    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(1)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(3)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(8)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(8)

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

    ! Construct derived types
    optim_paras%coeffs_home = coeffs_home
    optim_paras%coeffs_edu = coeffs_edu
    optim_paras%coeffs_a = coeffs_a
    optim_paras%coeffs_b = coeffs_b

    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF(ALLOCATED(periods_rewards_systematic)) DEALLOCATE(periods_rewards_systematic)

    ! Call function of interest
    CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period_int, states_all_int, edu_start_int, max_states_period_int, optim_paras)

    periods_rewards_systematic_int = periods_rewards_systematic

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
SUBROUTINE wrapper_criterion_ambiguity(emax, x, num_periods_int, num_draws_emax_int, period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, rewards_systematic, edu_max_int, edu_start_int, periods_emax_int, states_all_int, mapping_state_idx_int, delta, shocks_cov, mean)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: emax

    DOUBLE PRECISION, INTENT(IN)    :: draws_emax_ambiguity_transformed(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: draws_emax_ambiguity_standard(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax_int(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: rewards_systematic(:)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cov(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: delta
    DOUBLE PRECISION, INTENT(IN)    :: x(:)

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:,:,:,:,:)
    INTEGER, INTENT(IN)             :: states_all_int(:,:,:)
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

    LOGICAL, INTENT(IN)             :: mean

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Assign global RESFORT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)
    num_free_ambi = SIZE(x, 1)

    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int

    ! Construct derived types
    optim_paras%delta = delta

    ambi_spec%mean = mean

    emax = criterion_ambiguity(x, num_periods_int, num_draws_emax_int, period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, rewards_systematic, edu_max_int, edu_start_int, periods_emax_int, states_all_int, mapping_state_idx_int, optim_paras, shocks_cov, ambi_spec)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_criterion_ambiguity_derivative(grad, x, num_periods_int, num_draws_emax_int, period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, rewards_systematic, edu_max_int, edu_start_int, periods_emax_int, states_all_int, mapping_state_idx_int, delta, shocks_cov, mean, fort_slsqp_eps, num_free_ambi_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: grad(num_free_ambi_int)

    DOUBLE PRECISION, INTENT(IN)    :: draws_emax_ambiguity_transformed(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: draws_emax_ambiguity_standard(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax_int(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: rewards_systematic(:)
    DOUBLE PRECISION, INTENT(IN)    :: x(num_free_ambi_int)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cov(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: fort_slsqp_eps
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:,:,:,:,:)
    INTEGER, INTENT(IN)             :: states_all_int(:,:,:)
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: num_free_ambi_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: edu_max_int

    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

    LOGICAL, INTENT(IN)             :: mean

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Assign global RESFORT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)
    num_draws_emax = num_draws_emax_int
    num_free_ambi = num_free_ambi_int
    num_periods = num_periods_int

    ! Construct derived types
    optim_paras%delta = delta

    ambi_spec%mean = mean

    grad = criterion_ambiguity_derivative(x, num_periods_int, num_draws_emax_int, period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, rewards_systematic, edu_max_int, edu_start_int, periods_emax_int, states_all_int, mapping_state_idx_int, optim_paras, shocks_cov, ambi_spec, fort_slsqp_eps)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_construct_emax_risk(emax, num_periods_int, num_draws_emax_int, period, k, draws_emax_risk, rewards_systematic, edu_max_int, edu_start_int, periods_emax_int, states_all_int, mapping_state_idx_int, delta)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: emax

    DOUBLE PRECISION, INTENT(IN)    :: draws_emax_risk(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: rewards_systematic(:)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax_int(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: delta

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

    ! Construct derived types
    optim_paras%delta = delta

    ! Call function of interest
    CALL construct_emax_risk(emax, period, k, draws_emax_risk, rewards_systematic, edu_max_int, edu_start_int, periods_emax_int, states_all_int, mapping_state_idx_int, optim_paras)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_construct_emax_ambiguity(emax, num_periods_int, num_draws_emax_int, period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, rewards_systematic, edu_max_int, edu_start_int, periods_emax_int, states_all_int, mapping_state_idx_int, shocks_cov, measure, mean, level, delta, fort_slsqp_maxiter, fort_slsqp_ftol, fort_slsqp_eps, file_sim, is_write)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: emax

    DOUBLE PRECISION, INTENT(IN)    :: draws_emax_ambiguity_transformed(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: draws_emax_ambiguity_standard(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: rewards_systematic(:)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax_int(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cov(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: fort_slsqp_ftol
    DOUBLE PRECISION, INTENT(IN)    :: fort_slsqp_eps
    DOUBLE PRECISION, INTENT(IN)    :: level
    DOUBLE PRECISION, INTENT(IN)    :: delta

    CHARACTER(225), INTENT(IN)      :: file_sim
    CHARACTER(10), INTENT(IN)       :: measure

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:,:,:,:,:)
    INTEGER, INTENT(IN)             :: states_all_int(:,:,:)
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: fort_slsqp_maxiter
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

    LOGICAL, INTENT(IN)             :: is_write
    LOGICAL, INTENT(IN)             :: mean

    !/* internal */

    DOUBLE PRECISION, ALLOCATABLE   :: opt_ambi_details(:, :, :)

    INTEGER                         :: info

    DOUBLE PRECISION                :: shocks_cholesky(4, 4)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Assign global RESFORT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)

    IF (mean) THEN
        num_free_ambi = 2
    ELSE
        num_free_ambi = 4
    END IF

    ! Allocate containers
    ALLOCATE(opt_ambi_details(num_periods_int, max_states_period, 7))

    ! We need to determine the Cholesky decomposition
    CALL get_cholesky_decomposition(shocks_cholesky, info, shocks_cov)
    IF (info .NE. zero_dble) THEN
        STOP 'Problem in the Cholesky decomposition'
    END IF

    ! Construct derived types
    optimizer_options%slsqp%maxiter = fort_slsqp_maxiter
    optimizer_options%slsqp%ftol = fort_slsqp_ftol
    optimizer_options%slsqp%eps = fort_slsqp_eps

    optim_paras%shocks_cholesky = shocks_cholesky
    optim_paras%level = level
    optim_paras%delta = delta

    ambi_spec%measure = measure
    ambi_spec%mean = mean

    !# Transfer global RESFORT variables
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int

    ! Call function of interest
    CALL construct_emax_ambiguity(emax, opt_ambi_details, num_periods_int, num_draws_emax_int, period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, rewards_systematic, edu_max_int, edu_start_int, periods_emax_int, states_all_int, mapping_state_idx_int, ambi_spec, optim_paras, optimizer_options)

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
    rslt = trace(A)

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

    !/* internal objects        */

    INTEGER, ALLOCATABLE                :: infos(:)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Call function of interest
    CALL clip_value(clipped_value, value, lower_bound, upper_bound, infos)


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
SUBROUTINE wrapper_get_predictions(predictions, endogenous, exogenous, maxe, is_simulated, num_points_interp_int, num_states, file_sim, is_write)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)               :: predictions(num_states)

    DOUBLE PRECISION, INTENT(IN)                :: exogenous(:, :)
    DOUBLE PRECISION, INTENT(IN)                :: endogenous(:)
    DOUBLE PRECISION, INTENT(IN)                :: maxe(:)

    INTEGER, INTENT(IN)                         :: num_points_interp_int
    INTEGER, INTENT(IN)                         :: num_states

    LOGICAL, INTENT(IN)                         :: is_simulated(:)
    LOGICAL, INTENT(IN)                         :: is_write

    CHARACTER(225), INTENT(IN)                  :: file_sim

!------------------------------------------------------------------------------
! Algorithm

!------------------------------------------------------------------------------

    ! Transfer global RESFORT variables
    num_points_interp = num_points_interp_int

    ! Call function of interest
    CALL get_predictions(predictions, endogenous, exogenous, maxe, is_simulated, num_states, file_sim, is_write)

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

    INTEGER, INTENT(IN)             :: num_candidates
    INTEGER, INTENT(IN)             :: candidates(:)
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
SUBROUTINE wrapper_get_endogenous_variable(exogenous_variable, period, num_periods_int, num_states, periods_rewards_systematic_int, edu_max_int, edu_start_int, mapping_state_idx_int, periods_emax_int, states_all_int, is_simulated, num_draws_emax_int, maxe, draws_emax_risk, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, shocks_cov, measure, mean, level, delta, fort_slsqp_maxiter, fort_slsqp_ftol, fort_slsqp_eps)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: exogenous_variable(num_states)

    DOUBLE PRECISION, INTENT(IN)        :: periods_rewards_systematic_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)        :: draws_emax_ambiguity_transformed(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: draws_emax_ambiguity_standard(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: draws_emax_risk(:, :)

    DOUBLE PRECISION, INTENT(IN)        :: periods_emax_int(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: shocks_cov(4, 4)
    DOUBLE PRECISION, INTENT(IN)        :: fort_slsqp_ftol
    DOUBLE PRECISION, INTENT(IN)        :: fort_slsqp_eps
    DOUBLE PRECISION, INTENT(IN)        :: level(1)
    DOUBLE PRECISION, INTENT(IN)        :: maxe(:)
    DOUBLE PRECISION, INTENT(IN)        :: delta

    INTEGER, INTENT(IN)                 :: mapping_state_idx_int(:, :, :, :, :)
    INTEGER, INTENT(IN)                 :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)                 :: fort_slsqp_maxiter
    INTEGER, INTENT(IN)                 :: num_draws_emax_int
    INTEGER, INTENT(IN)                 :: num_periods_int
    INTEGER, INTENT(IN)                 :: edu_start_int
    INTEGER, INTENT(IN)                 :: edu_max_int
    INTEGER, INTENT(IN)                 :: num_states
    INTEGER, INTENT(IN)                 :: period

    LOGICAL, INTENT(IN)                 :: is_simulated(:)
    LOGICAL, INTENT(IN)                 :: mean

    CHARACTER(10), INTENT(IN)           :: measure

    !/* internal arguments*/
    INTEGER                             :: info

    DOUBLE PRECISION, ALLOCATABLE       :: opt_ambi_details(:, :, :)
    DOUBLE PRECISION                    :: shocks_cholesky(4, 4)
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Construct auxiliary objects
    max_states_period = SIZE(states_all_int, 2)
    num_periods = SIZE(states_all_int, 1)

    ! Allocate containers
    ALLOCATE(opt_ambi_details(num_periods, max_states_period, 7))

    ! Transfer global RESFORT variables
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int

    IF (mean) THEN
        num_free_ambi = 2
    ELSE
        num_free_ambi = 4
    END IF

    CALL get_cholesky_decomposition(shocks_cholesky, info, shocks_cov)
    IF (info .NE. zero_dble) THEN
        STOP 'Problem in the Cholesky decomposition'
    END IF

    ! Construct derived types
    optimizer_options%slsqp%maxiter = fort_slsqp_maxiter
    optimizer_options%slsqp%ftol = fort_slsqp_ftol
    optimizer_options%slsqp%eps = fort_slsqp_eps

    optim_paras%shocks_cholesky = shocks_cholesky
    optim_paras%level = level
    optim_paras%delta = delta

    ambi_spec%measure = measure
    ambi_spec%mean = mean

    ! Call function of interest
    CALL get_endogenous_variable(exogenous_variable, opt_ambi_details, period, num_states, periods_rewards_systematic_int, mapping_state_idx_int, periods_emax_int, states_all_int, is_simulated, maxe, draws_emax_risk, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, edu_start_int, edu_max_int, ambi_spec, optim_paras, optimizer_options)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_get_exogenous_variables(independent_variables, maxe, period, num_periods_int, num_states, periods_rewards_systematic_int, shifts, edu_max_int, edu_start_int, mapping_state_idx_int, periods_emax_int, states_all_int, delta)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: independent_variables(num_states, 9)
    DOUBLE PRECISION, INTENT(OUT)       :: maxe(num_states)


    DOUBLE PRECISION, INTENT(IN)        :: periods_rewards_systematic_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)        :: periods_emax_int(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: shifts(:)
    DOUBLE PRECISION, INTENT(IN)        :: delta

    INTEGER, INTENT(IN)                 :: mapping_state_idx_int(:, :, :, :, :)
    INTEGER, INTENT(IN)                 :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)                 :: num_periods_int
    INTEGER, INTENT(IN)                 :: edu_start_int
    INTEGER, INTENT(IN)                 :: edu_max_int
    INTEGER, INTENT(IN)                 :: num_states
    INTEGER, INTENT(IN)                 :: period

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    !# Assign global RESFORT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)

    !# Transfer global RESFORT variables
    num_periods = num_periods_int

    !# Derived types
    optim_paras%delta = delta

    ! Call function of interest
    CALL get_exogenous_variables(independent_variables, maxe, period, num_states, periods_rewards_systematic_int, shifts, mapping_state_idx_int, periods_emax_int, states_all_int, optim_paras, edu_start_int, edu_max_int)

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

    ! Call function of interest
    is_simulated = get_simulated_indicator(num_points, num_states, period, is_debug_int)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_extract_cholesky(shocks_cholesky, info, x)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: shocks_cholesky(4, 4)

    DOUBLE PRECISION, INTENT(IN)    :: x(32)

    INTEGER, INTENT(OUT)            :: info

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL extract_cholesky(shocks_cholesky, x, info)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_kl_divergence(rslt, mean_old, cov_old, mean_new, cov_new)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: rslt

    DOUBLE PRECISION, INTENT(IN)    :: cov_old(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: cov_new(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: mean_old(:)
    DOUBLE PRECISION, INTENT(IN)    :: mean_new(:)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    rslt = kl_divergence(mean_old, cov_old, mean_new, cov_new)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_slsqp_debug(x_internal, x_start, maxiter, ftol, num_dim)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: x_internal(num_dim)
    DOUBLE PRECISION, INTENT(IN)        :: x_start(num_dim)
    DOUBLE PRECISION, INTENT(IN)        :: ftol

    INTEGER, INTENT(IN)                 :: num_dim
    INTEGER, INTENT(IN)                 :: maxiter

    !/* internal objects        */

    INTEGER                             :: m
    INTEGER                             :: meq
    INTEGER                             :: la
    INTEGER                             :: n
    INTEGER                             :: len_w
    INTEGER                             :: len_jw
    INTEGER                             :: mode
    INTEGER                             :: iter
    INTEGER                             :: n1
    INTEGER                             :: mieq
    INTEGER                             :: mineq
    INTEGER                             :: l_jw
    INTEGER                             :: l_w

    INTEGER, ALLOCATABLE                :: jw(:)

    DOUBLE PRECISION, ALLOCATABLE       :: a(:,:)
    DOUBLE PRECISION, ALLOCATABLE       :: xl(:)
    DOUBLE PRECISION, ALLOCATABLE       :: xu(:)
    DOUBLE PRECISION, ALLOCATABLE       :: c(:)
    DOUBLE PRECISION, ALLOCATABLE       :: g(:)
    DOUBLE PRECISION, ALLOCATABLE       :: w(:)

    DOUBLE PRECISION                    :: acc
    DOUBLE PRECISION                    :: f

    LOGICAL                             :: is_finished

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    meq = 1         ! Number of equality constraints
    mieq = 0        ! Number of inequality constraints

    ! Initialize starting values
    x_internal = x_start

    ! Derived attributes
    m = meq + mieq
    la = MAX(1, m)
    n = SIZE(x_internal)
    n1 = n + 1
    mineq = m - meq + n1 + n1

    len_w =  (3 * n1 + m) * (n1 + 1) + (n1 - meq + 1) * (mineq + 2) + 2 * mineq + (n1 + mineq) * (n1 - meq) + 2 * meq + n1 + ((n + 1) * n) / two_dble + 2 * m + 3 * n + 3 *  n1 + 1

    len_jw = mineq

    ! Allocate and initialize containers
    ALLOCATE(w(len_w)); w = zero_dble
    ALLOCATE(jw(len_jw)); jw = zero_int
    ALLOCATE(a(la, n + 1)); a = zero_dble

    ALLOCATE(g(n + 1)); g = zero_dble
    ALLOCATE(c(la)); c = zero_dble

    ! Decompose upper and lower bounds
    ALLOCATE(xl(n)); ALLOCATE(xu(n))
    xl = - HUGE_FLOAT; xu = HUGE_FLOAT

    ! Initialize the iteration counter and mode value
    acc = ftol
    iter = maxiter

    ! Transformations to match interface, deleted later
    l_jw = len_jw
    l_w = len_w

    ! Initialization of SLSQP
    mode = zero_int

    is_finished = .False.

    CALL debug_criterion_function(f, x_internal, n)
    CALL debug_criterion_derivative(g, x_internal, n)

    CALL debug_constraint_function(c, x_internal, n, la)
    CALL debug_constraint_derivative(a, x_internal, n, la)

    ! Iterate until completion
    DO WHILE (.NOT. is_finished)

        ! Evaluate criterion function and constraints
        IF (mode == one_int) THEN
            CALL debug_criterion_function(f, x_internal, n)
            CALL debug_constraint_function(c, x_internal, n, la)
        ! Evaluate gradient of criterion function and constraints
        ELSEIF (mode == - one_int) THEN
            CALL debug_criterion_derivative(g, x_internal, n)
            CALL debug_constraint_derivative(a, x_internal, n, la)
        END IF

        !SLSQP Interface
        CALL slsqp(m, meq, la, n, x_internal, xl, xu, f, c, g, a, acc, iter, mode, w, l_w, jw, l_jw)

        ! Check if SLSQP has completed
        IF (.NOT. ABS(mode) == one_int) THEN
            is_finished = .True.
        END IF

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE debug_criterion_function  (rslt, x, n)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE


    !/* external objects        */

    INTEGER, INTENT(IN)                 :: n

    DOUBLE PRECISION, INTENT(OUT)       :: rslt
    DOUBLE PRECISION, INTENT(IN)        :: x(n)

    !/* internal objects    */

    INTEGER                             :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    rslt = zero_dble

    DO i = 2, n
        rslt = rslt + 100_our_dble * (x(i) - x(i - 1) ** 2) ** 2
        rslt = rslt + (one_dble - x(i - 1)) ** 2
    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE debug_criterion_derivative(rslt, x, n)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(IN)                 :: n

    DOUBLE PRECISION, INTENT(OUT)       :: rslt(n + 1)
    DOUBLE PRECISION, INTENT(IN)        :: x(n)

    !/* internals objects       */

    DOUBLE PRECISION                    :: xm_m1(n - 2)
    DOUBLE PRECISION                    :: xm_p1(n - 2)
    DOUBLE PRECISION                    :: xm(n - 2)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Extract sets of evaluation points
    xm = x(2:(n - 1))

    xm_m1 = x(:(n - 2))

    xm_p1 = x(3:)

    ! Construct derivative information
    rslt(1) = -400_our_dble * x(1) * (x(2) - x(1) ** 2) - 2 * (1 - x(1))

    rslt(2:(n - 1)) = (200_our_dble * (xm - xm_m1 ** 2) - 400_our_dble * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm))

    rslt(n) = 200_our_dble * (x(n) - x(n - 1) ** 2)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE debug_constraint_function(rslt, x, n, la)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: rslt(la)

    DOUBLE PRECISION, INTENT(IN)        :: x(n)

    INTEGER, INTENT(IN)                 :: la
    INTEGER, INTENT(IN)                 :: n

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    rslt(:) = SUM(x) - 10_our_dble

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE debug_constraint_derivative(rslt, x, n, la)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: rslt(n + 1)

    DOUBLE PRECISION, INTENT(IN)        :: x(n)

    INTEGER, INTENT(IN)                 :: la
    INTEGER, INTENT(IN)                 :: n

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    rslt = one_dble

    rslt(n + 1) = zero_dble

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_constraint_ambiguity(rslt, x, shocks_cov, level, num_free_ambi_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: rslt

    DOUBLE PRECISION, INTENT(IN)        :: x(num_free_ambi_int)
    DOUBLE PRECISION, INTENT(IN)        :: shocks_cov(4, 4)
    DOUBLE PRECISION, INTENT(IN)        :: level(1)

    INTEGER, INTENT(IN)                 :: num_free_ambi_int

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Assign global RESFORT variables
    num_free_ambi = num_free_ambi_int

    ! Construct derived types
    optim_paras%level = level

    rslt = constraint_ambiguity(x, shocks_cov, optim_paras)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_get_worst_case(x_shift, is_success, mode, num_periods_int, num_draws_emax_int, period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, rewards_systematic_int, edu_max_int, edu_start_int, periods_emax_int, states_all_int, mapping_state_idx_int, shocks_cov, level, delta, fort_slsqp_maxiter, fort_slsqp_ftol, fort_slsqp_eps, mean, num_free_ambi_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: x_shift(num_free_ambi_int)
    DOUBLE PRECISION, INTENT(OUT)   :: is_success

    INTEGER, INTENT(OUT)            :: mode

    DOUBLE PRECISION, INTENT(IN)    :: draws_emax_ambiguity_transformed(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: rewards_systematic_int(:)
    DOUBLE PRECISION, INTENT(IN)    :: draws_emax_ambiguity_standard(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax_int(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: fort_slsqp_ftol
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cov(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: fort_slsqp_eps
    DOUBLE PRECISION, INTENT(IN)    :: level(1)
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: fort_slsqp_maxiter
    INTEGER, INTENT(IN)             :: num_free_ambi_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

    LOGICAL, INTENT(IN)             :: mean

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Assign global RESFORT variables
    IF (mean) THEN
        num_free_ambi = 2
    ELSE
        num_free_ambi = 4
    END IF

    ! Assign global RESPFRT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int

    ! Construct derived types
    optimizer_options%slsqp%maxiter = fort_slsqp_maxiter
    optimizer_options%slsqp%ftol = fort_slsqp_ftol
    optimizer_options%slsqp%eps = fort_slsqp_eps

    optim_paras%level = level
    optim_paras%delta = delta

    ambi_spec%mean = mean

    CALL get_worst_case(x_shift, is_success, mode, num_periods_int, num_draws_emax_int, period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, rewards_systematic_int, edu_max_int, edu_start_int, periods_emax_int, states_all_int, mapping_state_idx_int, shocks_cov, optim_paras, optimizer_options, ambi_spec)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_constraint_ambiguity_derivative(rslt, x, shocks_cov, level, eps_der_approx, num_free_ambi_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: rslt(num_free_ambi_int)

    DOUBLE PRECISION, INTENT(IN)        :: x(num_free_ambi_int)
    DOUBLE PRECISION, INTENT(IN)        :: shocks_cov(4, 4)
    DOUBLE PRECISION, INTENT(IN)        :: eps_der_approx
    DOUBLE PRECISION, INTENT(IN)        :: level(1)

    INTEGER, INTENT(IN)                 :: num_free_ambi_int

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Assign global RESFORT variables
    num_free_ambi = num_free_ambi_int

    ! Construct derived types
    optim_paras%level = level

    rslt = constraint_ambiguity_derivative(x, shocks_cov, optim_paras, eps_der_approx)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_spectral_condition_number(rslt, A)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: rslt

    DOUBLE PRECISION, INTENT(IN)        :: A(:, :)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL spectral_condition_number(rslt, A)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_get_cholesky_decomposition(cholesky, matrix, nrows)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: cholesky(nrows, nrows)

    DOUBLE PRECISION, INTENT(IN)        :: matrix(nrows, nrows)

    INTEGER, INTENT(IN)                 :: nrows

    !/* internal objects        */

    INTEGER                             :: info

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL get_cholesky_decomposition(cholesky, info, matrix)

    IF (info .NE. zero_dble) THEN
        STOP 'Problem in the Cholesky decomposition'
    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_correlation_to_covariance(cov, corr, sd, nrows)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: cov(nrows, nrows)

    DOUBLE PRECISION, INTENT(IN)        :: corr(nrows, nrows)

    DOUBLE PRECISION, INTENT(IN)        :: sd(nrows)

    INTEGER, INTENT(IN)                 :: nrows

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL correlation_to_covariance(cov, corr, sd)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_covariance_to_correlation(corr, cov, nrows)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: corr(nrows, nrows)

    DOUBLE PRECISION, INTENT(IN)        :: cov(nrows, nrows)

    INTEGER, INTENT(IN)                 :: nrows

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL covariance_to_correlation(corr, cov)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_get_relevant_dependence(shocks_cov_cand, shocks_cholesky_cand, shocks_cov, x)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: shocks_cov_cand(4, 4)
    DOUBLE PRECISION, INTENT(OUT)       :: shocks_cholesky_cand(4, 4)

    DOUBLE PRECISION, INTENT(IN)        :: shocks_cov(4, 4)
    DOUBLE PRECISION, INTENT(IN)        :: x(:)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL get_relevant_dependence(shocks_cov_cand, shocks_cholesky_cand, shocks_cov, x)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_get_scales_magnitude(precond_matrix_int, values, num_free_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: precond_matrix_int(num_free_int, num_free_int)

    DOUBLE PRECISION, INTENT(IN)        :: values(num_free_int)

    INTEGER, INTENT(IN)                 :: num_free_int

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Assign global RESFORT variables
    num_free = num_free_int

    precond_matrix_int = get_scales_magnitudes(values)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
