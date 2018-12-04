!*******************************************************************************
!*******************************************************************************
!
!   This module serves as the F2PY interface to the core functions. All the functions have counterparts as PYTHON implementations.
!
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_criterion(crit_val, x, is_interpolated_int, num_draws_emax_int, num_periods_int, num_points_interp_int, is_myopic_int, is_debug_int, data_est_int, num_draws_prob_int, tau_int, periods_draws_emax_int, periods_draws_prob_int, states_all_int, states_number_period_int, mapping_state_idx_int, max_states_period_int, num_agents_est_int, num_obs_agent_int, num_types_int, edu_start, edu_max, edu_share, type_spec_shares, type_spec_shifts, num_paras_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objeFcts        */

    DOUBLE PRECISION, INTENT(OUT)   :: crit_val

    DOUBLE PRECISION, INTENT(IN)    :: x(:)

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_number_period_int(:)
    INTEGER, INTENT(IN)             :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: max_states_period_int
    INTEGER, INTENT(IN)             :: num_obs_agent_int(:)
    INTEGER, INTENT(IN)             :: num_draws_prob_int
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: num_agents_est_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: num_paras_int
    INTEGER, INTENT(IN)             :: num_types_int
    INTEGER, INTENT(IN)             :: edu_start(:)
    INTEGER, INTENT(IN)             :: edu_max

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_prob_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: type_spec_shifts(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: data_est_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: type_spec_shares(:)
    DOUBLE PRECISION, INTENT(IN)    :: edu_share(:)
    DOUBLE PRECISION, INTENT(IN)    :: tau_int

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int


    !/* internal objects            */

    DOUBLE PRECISION                :: contribs(num_agents_est_int)

    INTEGER                         :: dist_optim_paras_info

    CHARACTER(225)                  :: file_sim_mock

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Assign global RESPFRT variables
    max_states_period = max_states_period_int
    min_idx = SIZE(mapping_state_idx_int, 4)
    num_rows = SIZE(data_est_int, 1)

    ! Transfer global RESFORT variables
    num_points_interp = num_points_interp_int
    num_agents_est = SIZE(data_est_int, 1) / INT(num_periods_int)
    num_draws_emax = num_draws_emax_int
    num_draws_prob = num_draws_prob_int
    num_agents_est = num_agents_est_int
    num_obs_agent = num_obs_agent_int

    num_periods = num_periods_int
    num_types = num_types_int
    num_paras = num_paras_int

    ! Ensure that not already allocated
    IF(ALLOCATED(edu_spec%start)) DEALLOCATE(edu_spec%start)
    IF(ALLOCATED(edu_spec%share)) DEALLOCATE(edu_spec%share)

    optim_paras%type_shares = type_spec_shares
    optim_paras%type_shifts = type_spec_shifts

    edu_spec%share = edu_share
    edu_spec%start = edu_start
    edu_spec%max = edu_max

    CALL extract_parsing_info(num_paras, num_types, pinfo)

    CALL dist_optim_paras(optim_paras, x, dist_optim_paras_info)

    CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period_int, states_all_int, max_states_period_int, optim_paras)

    CALL fort_backward_induction(periods_emax, num_periods_int, is_myopic_int, max_states_period_int, periods_draws_emax_int, num_draws_emax_int, states_number_period_int, periods_rewards_systematic, mapping_state_idx_int, states_all_int, is_debug_int, is_interpolated_int, num_points_interp_int, edu_spec, optim_paras, file_sim_mock, .False.)

    CALL fort_contributions(contribs, periods_rewards_systematic, mapping_state_idx_int, periods_emax, states_all_int, data_est_int, periods_draws_prob_int, tau_int, num_periods_int, num_draws_prob_int, num_agents_est, num_obs_agent, num_types, edu_spec, optim_paras)

    crit_val = get_log_likl(contribs)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_contributions(contribs, periods_rewards_systematic_int, mapping_state_idx_int, periods_emax_int, states_all_int, data_est_int, periods_draws_prob_int, tau_int, num_periods_int, num_draws_prob_int, num_agents_est_int, num_obs_agent_int, num_types_int, edu_start, edu_max, shocks_cholesky, delta, type_spec_shares, type_spec_shifts)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: contribs(num_agents_est_int)

    DOUBLE PRECISION, INTENT(IN)    :: periods_rewards_systematic_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_prob_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: type_spec_shifts(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: type_spec_shares(:)
    DOUBLE PRECISION, INTENT(IN)    :: data_est_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: tau_int
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)             :: num_obs_agent_int(:)
    INTEGER, INTENT(IN)             :: num_agents_est_int
    INTEGER, INTENT(IN)             :: num_draws_prob_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: num_types_int
    INTEGER, INTENT(IN)             :: edu_start(:)
    INTEGER, INTENT(IN)             :: edu_max

    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(:, :)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Assign global RESPFRT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)

    ! Transfer global RESFORT variables
    periods_rewards_systematic = periods_rewards_systematic_int
    num_agents_est = num_agents_est_int
    num_draws_prob = num_draws_prob_int
    num_obs_agent = num_obs_agent_int
    num_periods = num_periods_int
    num_types = num_types_int
    tau = tau_int

    ! Esnure that not already allocated
    IF(ALLOCATED(edu_spec%start)) DEALLOCATE(edu_spec%start)

    ! Construct derived types
    optim_paras%shocks_cholesky = shocks_cholesky
    optim_paras%delta = delta

    optim_paras%type_shares = type_spec_shares
    optim_paras%type_shifts = type_spec_shifts

    edu_spec%start = edu_start
    edu_spec%max = edu_max

    CALL fort_contributions(contribs, periods_rewards_systematic, mapping_state_idx_int, periods_emax_int, states_all_int, data_est_int, periods_draws_prob_int, tau_int, num_periods_int, num_draws_prob_int, num_agents_est, num_obs_agent, num_types, edu_spec, optim_paras)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_solve(periods_rewards_systematic_int, states_number_period_int, mapping_state_idx_int, periods_emax_int, states_all_int, is_interpolated_int, num_points_interp_int, num_draws_emax_int, num_periods_int, is_myopic_int, is_debug_int, periods_draws_emax_int, min_idx_int, edu_start, edu_max, coeffs_common, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, delta, file_sim, max_states_period_int, num_types_int, type_spec_shares, type_spec_shifts)

    ! The presence of max_states_period breaks the quality of interfaces. However, this is required so that the size of the return arguments is known from the beginning.

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: mapping_state_idx_int(num_periods_int, num_periods_int, num_periods_int, min_idx_int, 4, num_types_int)
    INTEGER, INTENT(OUT)            :: states_all_int(num_periods_int, max_states_period_int, 5)
    INTEGER, INTENT(OUT)            :: states_number_period_int(num_periods_int)

    DOUBLE PRECISION, INTENT(OUT)   :: periods_rewards_systematic_int(num_periods_int, max_states_period_int, 4)
    DOUBLE PRECISION, INTENT(OUT)   :: periods_emax_int(num_periods_int, max_states_period_int)

    INTEGER, INTENT(IN)             :: max_states_period_int
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: num_types_int
    INTEGER, INTENT(IN)             :: edu_start(:)
    INTEGER, INTENT(IN)             :: min_idx_int
    INTEGER, INTENT(IN)             :: edu_max

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: type_spec_shifts(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: type_spec_shares(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_common(2)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(3)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(7)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(15)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(15)
    DOUBLE PRECISION, INTENT(IN)    :: delta(1)

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int

    CHARACTER(225), INTENT(IN)      :: file_sim

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# Transfer global RESFORT variables
    num_points_interp = num_points_interp_int
    max_states_period = max_states_period_int
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int
    num_types = num_types_int
    min_idx = edu_max + 1


    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF (ALLOCATED(periods_rewards_systematic)) DEALLOCATE(periods_rewards_systematic)
    IF (ALLOCATED(states_number_period)) DEALLOCATE(states_number_period)
    IF (ALLOCATED(mapping_state_idx)) DEALLOCATE(mapping_state_idx)
    IF (ALLOCATED(edu_spec%start)) DEALLOCATE(edu_spec%start)
    IF (ALLOCATED(periods_emax)) DEALLOCATE(periods_emax)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)

    optim_paras%shocks_cholesky = shocks_cholesky
    optim_paras%coeffs_common = coeffs_common
    optim_paras%coeffs_home = coeffs_home
    optim_paras%coeffs_edu = coeffs_edu
    optim_paras%coeffs_a = coeffs_a
    optim_paras%coeffs_b = coeffs_b
    optim_paras%delta = delta

    optim_paras%type_shares = type_spec_shares
    optim_paras%type_shifts = type_spec_shifts

    edu_spec%start = edu_start
    edu_spec%max = edu_max

    ! Call FORTRAN solution
    CALL fort_solve(periods_rewards_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, is_interpolated_int, num_points_interp_int, num_draws_emax, num_periods, is_myopic_int, is_debug_int, periods_draws_emax_int, edu_spec, optim_paras, file_sim)

    ! Assign to initial objects for return to PYTHON
    periods_rewards_systematic_int = periods_rewards_systematic
    states_number_period_int = states_number_period
    mapping_state_idx_int = mapping_state_idx
    periods_emax_int = periods_emax
    states_all_int = states_all

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_simulate(data_sim_int, periods_rewards_systematic_int, mapping_state_idx_int, periods_emax_int, states_all_int, num_periods_int, num_agents_sim_int, periods_draws_sims, seed_sim, file_sim, edu_start, edu_max, edu_share, edu_lagged, coeffs_common, coeffs_a, coeffs_b, shocks_cholesky, delta, num_types_int, type_spec_shares, type_spec_shifts, is_debug_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: data_sim_int(num_agents_sim_int * num_periods_int, 29)

    DOUBLE PRECISION, INTENT(IN)    :: periods_rewards_systematic_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_sims(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax_int(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: type_spec_shifts(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: type_spec_shares(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_common(2)
    DOUBLE PRECISION, INTENT(IN)    :: edu_lagged(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(15)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(15)
    DOUBLE PRECISION, INTENT(IN)    :: edu_share(:)
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: num_types_int
    INTEGER, INTENT(IN)             :: edu_start(:)
    INTEGER, INTENT(IN)             :: seed_sim
    INTEGER, INTENT(IN)             :: edu_max

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)             :: num_agents_sim_int

    CHARACTER(225), INTENT(IN)      :: file_sim

    LOGICAL, INTENT(IN)             :: is_debug_int

    !/* internal objects        */

    DOUBLE PRECISION, ALLOCATABLE   :: data_sim(:, :)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Assign global RESPFRT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)

    ! Transfer global RESFORT variables
    num_agents_sim = num_agents_sim_int
    num_periods = num_periods_int
    num_types = num_types_int
    is_debug = is_debug_int

    ! Ensure that not already allocated
    IF(ALLOCATED(edu_spec%lagged)) DEALLOCATE(edu_spec%lagged)
    IF(ALLOCATED(edu_spec%start)) DEALLOCATE(edu_spec%start)
    IF(ALLOCATED(edu_spec%share)) DEALLOCATE(edu_spec%share)

    ! Construct derived types
    optim_paras%shocks_cholesky = shocks_cholesky
    optim_paras%coeffs_common = coeffs_common
    optim_paras%coeffs_a = coeffs_a
    optim_paras%coeffs_b = coeffs_b
    optim_paras%delta = delta

    optim_paras%type_shares = type_spec_shares
    optim_paras%type_shifts = type_spec_shifts

    edu_spec%lagged = edu_lagged
    edu_spec%start = edu_start
    edu_spec%share = edu_share
    edu_spec%max = edu_max

    ! Call function of interest
    CALL fort_simulate(data_sim, periods_rewards_systematic_int, mapping_state_idx_int, periods_emax_int, states_all_int, num_agents_sim, periods_draws_sims, seed_sim, file_sim, edu_spec, optim_paras, num_types, is_debug)

    ! Assign to initial objects for return to PYTHON
    data_sim_int = data_sim

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_backward_induction(periods_emax_int, num_periods_int, is_myopic_int, max_states_period_int, periods_draws_emax_int, num_draws_emax_int, states_number_period_int, periods_rewards_systematic_int, mapping_state_idx_int, states_all_int,  is_debug_int, is_interpolated_int, num_points_interp_int, edu_start, edu_max, shocks_cholesky, delta, coeffs_common, coeffs_a, coeffs_b, file_sim, is_write)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: periods_emax_int(num_periods_int, max_states_period_int)

    DOUBLE PRECISION, INTENT(IN)    :: periods_rewards_systematic_int(:, :, :   )
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_common(2)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(15)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(15)
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_number_period_int(:)
    INTEGER, INTENT(IN)             :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)             :: max_states_period_int
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start(:)
    INTEGER, INTENT(IN)             :: edu_max

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int
    LOGICAL, INTENT(IN)             :: is_write

    CHARACTER(225), INTENT(IN)      :: file_sim

    !/* internal objects*/

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# Transfer auxiliary variable to global variable.
    num_points_interp = num_points_interp_int
    max_states_period = max_states_period_int
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int
    min_idx = edu_max + 1

    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF(ALLOCATED(edu_spec%start)) DEALLOCATE(edu_spec%start)
    IF(ALLOCATED(periods_emax)) DEALLOCATE(periods_emax)

    optim_paras%shocks_cholesky = shocks_cholesky
    optim_paras%coeffs_common = coeffs_common
    optim_paras%coeffs_a = coeffs_a
    optim_paras%coeffs_b = coeffs_b
    optim_paras%delta = delta

    edu_spec%start = edu_start
    edu_spec%max = edu_max

    ! Call actual function of interest
    CALL fort_backward_induction(periods_emax, num_periods_int, is_myopic_int, max_states_period_int, periods_draws_emax_int, num_draws_emax_int, states_number_period_int, periods_rewards_systematic, mapping_state_idx_int, states_all_int, is_debug_int, is_interpolated_int, num_points_interp_int, edu_spec, optim_paras, file_sim, is_write)

    ! Allocate to intermidiaries
    periods_emax_int = periods_emax

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_create_state_space(states_all_int, states_number_period_int, mapping_state_idx_int, max_states_period_int, num_periods_int, num_types_int, edu_start, edu_max, min_idx_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: mapping_state_idx_int(num_periods_int, num_periods_int, num_periods_int, min_idx_int, 4, num_types_int)
    INTEGER, INTENT(OUT)            :: states_all_int(num_periods_int, 150000, 5)
    INTEGER, INTENT(OUT)            :: states_number_period_int(num_periods_int)
    INTEGER, INTENT(OUT)            :: max_states_period_int

    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: num_types_int
    INTEGER, INTENT(IN)             :: edu_start(:)
    INTEGER, INTENT(IN)             :: min_idx_int
    INTEGER, INTENT(IN)             :: edu_max

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# Transfer global RESFORT variables
    num_periods = num_periods_int
    num_types = num_types_int
    min_idx = min_idx_int

    states_all_int = MISSING_INT

    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF (ALLOCATED(states_number_period)) DEALLOCATE(states_number_period)
    IF (ALLOCATED(mapping_state_idx)) DEALLOCATE(mapping_state_idx)
    IF (ALLOCATED(edu_spec%start)) DEALLOCATE(edu_spec%start)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)

    ! Construct derived types
    edu_spec%start = edu_start
    edu_spec%max = edu_max

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, num_periods_int, num_types, edu_spec)

    states_all_int(:, :max_states_period, :) = states_all
    states_number_period_int = states_number_period

    ! Updated global variables
    mapping_state_idx_int = mapping_state_idx
    max_states_period_int = max_states_period

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_calculate_rewards_systematic(periods_rewards_systematic_int, num_periods_int, states_number_period_int, states_all_int, max_states_period_int, coeffs_common, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, type_spec_shares, type_spec_shifts)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: periods_rewards_systematic_int(num_periods_int, max_states_period_int, 4)

    DOUBLE PRECISION, INTENT(IN)    :: type_spec_shifts(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: type_spec_shares(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_common(2)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(3)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(7)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(15)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(15)

    INTEGER, INTENT(IN)             :: states_number_period_int(:)
    INTEGER, INTENT(IN)             :: max_states_period_int
    INTEGER, INTENT(IN)             :: states_all_int(:,:,:)
    INTEGER, INTENT(IN)             :: num_periods_int

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Transfer global RESOFORT variables
    max_states_period = max_states_period_int
    num_periods = num_periods_int

    ! Construct derived types
    optim_paras%coeffs_common = coeffs_common
    optim_paras%coeffs_home = coeffs_home
    optim_paras%coeffs_edu = coeffs_edu
    optim_paras%coeffs_a = coeffs_a
    optim_paras%coeffs_b = coeffs_b

    optim_paras%type_shares = type_spec_shares
    optim_paras%type_shifts = type_spec_shifts

    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF(ALLOCATED(periods_rewards_systematic)) DEALLOCATE(periods_rewards_systematic)

    ! Call function of interest
    CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period_int, states_all_int, max_states_period_int, optim_paras)

    periods_rewards_systematic_int = periods_rewards_systematic

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Call function of interest
    rslt = normal_pdf(x, mean, sd)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_pinv(rslt, A, m)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(m, m)

    DOUBLE PRECISION, INTENT(IN)    :: A(m, m)

    INTEGER, INTENT(IN)             :: m

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Call function of interest
    rslt = pinv(A, m)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Call function of interest
    CALL svd(U, S, VT, A, m)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************

!*******************************************************************************
!*******************************************************************************

!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_construct_emax_risk(emax, num_periods_int, num_draws_emax_int, period, k, draws_emax_risk, rewards_systematic, periods_emax_int, states_all_int, mapping_state_idx_int, edu_start, edu_max, delta, coeffs_common, coeffs_a, coeffs_b, num_types_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: emax

    DOUBLE PRECISION, INTENT(IN)    :: draws_emax_risk(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: rewards_systematic(:)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax_int(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_common(2)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(15)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(15)
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_all_int(:,:,:)
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: edu_start(:)
    INTEGER, INTENT(IN)             :: num_types_int
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Assign global RESFORT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)

    !# Transfer global RESFORT variables
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int
    num_types = num_types_int
    min_idx = edu_max + 1

    ! Ensure array not already allocated
    IF(ALLOCATED(edu_spec%start)) DEALLOCATE(edu_spec%start)

    ! Construct derived types
    optim_paras%coeffs_common = coeffs_common
    optim_paras%coeffs_a = coeffs_a
    optim_paras%coeffs_b = coeffs_b
    optim_paras%delta = delta

    edu_spec%start = edu_start
    edu_spec%max = edu_max

    ! Call function of interest
    CALL construct_emax_risk(emax, period, k, draws_emax_risk, rewards_systematic, periods_emax_int, states_all_int, mapping_state_idx_int, edu_spec, optim_paras)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************

!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_standard_normal(draw, dim)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(IN)             :: dim

    DOUBLE PRECISION, INTENT(OUT)   :: draw(dim)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Call function of interest
    CALL standard_normal(draw)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_determinant(det, A)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: det

    DOUBLE PRECISION, INTENT(IN)    :: A(:, :)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Call function of interest
    det = determinant(A)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_inverse(inv, A, n)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: inv(n, n)

    DOUBLE PRECISION, INTENT(IN)    :: A(:, :)

    INTEGER, INTENT(IN)             :: n

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Call function of interest
    inv = inverse(A, n)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_trace(rslt, A)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT) :: rslt

    DOUBLE PRECISION, INTENT(IN)  :: A(:,:)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Call function of interest
    rslt = trace(A)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Call function of interest
    CALL clip_value(clipped_value, value, lower_bound, upper_bound, infos)


END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Call function of interest
    CALL get_pred_info(r_squared, bse, Y, P, X, num_states, num_covars)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Call function of interest
    CALL point_predictions(Y, X, coeffs, num_states)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm

!-------------------------------------------------------------------------------

    ! Transfer global RESFORT variables
    num_points_interp = num_points_interp_int

    ! Call function of interest
    CALL get_predictions(predictions, endogenous, exogenous, maxe, is_simulated, num_states, file_sim, is_write)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Call function of interest
     CALL random_choice(sample, candidates, num_candidates, num_points)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Call function of interest
    CALL get_coefficients(coeffs, Y, X, num_covars, num_states)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_endogenous_variable(exogenous_variable, period, num_periods_int, num_states, periods_rewards_systematic_int, mapping_state_idx_int, periods_emax_int, states_all_int, is_simulated, num_draws_emax_int, maxe, draws_emax_risk, edu_start, edu_max, shocks_cov, delta, coeffs_common, coeffs_a, coeffs_b)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: exogenous_variable(num_states)

    DOUBLE PRECISION, INTENT(IN)        :: periods_rewards_systematic_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)        :: draws_emax_risk(:, :)

    DOUBLE PRECISION, INTENT(IN)        :: periods_emax_int(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: shocks_cov(4, 4)
    DOUBLE PRECISION, INTENT(IN)        :: coeffs_common(2)
    DOUBLE PRECISION, INTENT(IN)        :: coeffs_a(15)
    DOUBLE PRECISION, INTENT(IN)        :: coeffs_b(15)
    DOUBLE PRECISION, INTENT(IN)        :: maxe(:)
    DOUBLE PRECISION, INTENT(IN)        :: delta

    INTEGER, INTENT(IN)                 :: mapping_state_idx_int(:, :, :, :, :, :)
    INTEGER, INTENT(IN)                 :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)                 :: num_draws_emax_int
    INTEGER, INTENT(IN)                 :: num_periods_int
    INTEGER, INTENT(IN)                 :: edu_start(:)
    INTEGER, INTENT(IN)                 :: num_states
    INTEGER, INTENT(IN)                 :: edu_max
    INTEGER, INTENT(IN)                 :: period

    LOGICAL, INTENT(IN)                 :: is_simulated(:)


    !/* internal arguments*/
    INTEGER                             :: info

    DOUBLE PRECISION                    :: shocks_cholesky(4, 4)
!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Construct auxiliary objects
    max_states_period = SIZE(states_all_int, 2)
    num_periods = SIZE(states_all_int, 1)


    ! Transfer global RESFORT variables
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int
    min_idx = edu_max + 1


    CALL get_cholesky_decomposition(shocks_cholesky, info, shocks_cov)
    IF (info .NE. zero_dble) THEN
        STOP 'Problem in the Cholesky decomposition'
    END IF

    ! Ensure that array not already allocated
    IF(ALLOCATED(edu_spec%start)) DEALLOCATE(edu_spec%start)

    optim_paras%shocks_cholesky = shocks_cholesky
    optim_paras%coeffs_common = coeffs_common
    optim_paras%coeffs_a = coeffs_a
    optim_paras%coeffs_b = coeffs_b
    optim_paras%delta = delta


    edu_spec%start = edu_start
    edu_spec%max = edu_max

    ! Call function of interest
    CALL get_endogenous_variable(exogenous_variable, period, num_states, periods_rewards_systematic_int, mapping_state_idx_int, periods_emax_int, states_all_int, is_simulated, maxe, draws_emax_risk, edu_spec, optim_paras)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_exogenous_variables(independent_variables, maxe, period, num_periods_int, num_states, periods_rewards_systematic_int, shifts, mapping_state_idx_int, periods_emax_int, states_all_int, edu_start, edu_max, delta, coeffs_common, coeffs_a, coeffs_b, num_types_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: independent_variables(num_states, 9)
    DOUBLE PRECISION, INTENT(OUT)       :: maxe(num_states)

    DOUBLE PRECISION, INTENT(IN)        :: periods_rewards_systematic_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)        :: periods_emax_int(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: coeffs_common(2)
    DOUBLE PRECISION, INTENT(IN)        :: coeffs_a(15)
    DOUBLE PRECISION, INTENT(IN)        :: coeffs_b(15)
    DOUBLE PRECISION, INTENT(IN)        :: shifts(:)
    DOUBLE PRECISION, INTENT(IN)        :: delta

    INTEGER, INTENT(IN)                 :: mapping_state_idx_int(:, :, :, :, :, :)
    INTEGER, INTENT(IN)                 :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)                 :: num_periods_int
    INTEGER, INTENT(IN)                 :: num_types_int
    INTEGER, INTENT(IN)                 :: edu_start(:)
    INTEGER, INTENT(IN)                 :: num_states
    INTEGER, INTENT(IN)                 :: edu_max
    INTEGER, INTENT(IN)                 :: period

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# Assign global RESFORT variables
    max_states_period = SIZE(states_all_int, 2)
    min_idx = SIZE(mapping_state_idx_int, 4)

    !# Transfer global RESFORT variables
    num_periods = num_periods_int
    num_types = num_types_int

    ! Esnure that array not already allocated
    IF(ALLOCATED(edu_spec%start)) DEALLOCATE(edu_spec%start)

    !# Derived types
    optim_paras%coeffs_common = coeffs_common
    optim_paras%coeffs_a = coeffs_a
    optim_paras%coeffs_b = coeffs_b
    optim_paras%delta = delta

    edu_spec%start = edu_start
    edu_spec%max = edu_max

    ! Call function of interest
    CALL get_exogenous_variables(independent_variables, maxe, period, num_states, periods_rewards_systematic_int, shifts, mapping_state_idx_int, periods_emax_int, states_all_int, edu_spec, optim_paras)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# Transfer global RESFORT variables
    num_periods = num_periods_int

    ! Call function of interest
    is_simulated = get_simulated_indicator(num_points, num_states, period, is_debug_int)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_extract_cholesky(shocks_cholesky, info, x)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: shocks_cholesky(4, 4)

    DOUBLE PRECISION, INTENT(IN)    :: x(54)

    INTEGER, INTENT(OUT)            :: info

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    CALL extract_parsing_info(num_paras, num_types, pinfo)
    CALL extract_cholesky(shocks_cholesky, x, info)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************

!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Initialize containers
    rslt = zero_dble

    DO i = 2, n
        rslt = rslt + 100_our_dble * (x(i) - x(i - 1) ** 2) ** 2
        rslt = rslt + (one_dble - x(i - 1)) ** 2
    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Extract sets of evaluation points
    xm = x(2:(n - 1))

    xm_m1 = x(:(n - 2))

    xm_p1 = x(3:)

    ! Construct derivative information
    rslt(1) = -400_our_dble * x(1) * (x(2) - x(1) ** 2) - 2 * (1 - x(1))

    rslt(2:(n - 1)) = (200_our_dble * (xm - xm_m1 ** 2) - 400_our_dble * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm))

    rslt(n) = 200_our_dble * (x(n) - x(n - 1) ** 2)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt(:) = SUM(x) - 10_our_dble

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt = one_dble

    rslt(n + 1) = zero_dble

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************

!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_spectral_condition_number(rslt, A)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: rslt

    DOUBLE PRECISION, INTENT(IN)        :: A(:, :)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL spectral_condition_number(rslt, A)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL get_cholesky_decomposition(cholesky, info, matrix)

    IF (info .NE. zero_dble) THEN
        STOP 'Problem in the Cholesky decomposition'
    END IF

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_num_obs_agent(num_rows_agent, data_array, num_agents_est_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: num_rows_agent(num_agents_est_int)

    DOUBLE PRECISION, INTENT(IN)        :: data_array(:, :)

    INTEGER, INTENT(IN)                 :: num_agents_est_int

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Assign global RESFORT variables
    num_agents_est = num_agents_est_int

    num_rows_agent = get_num_obs_agent(data_array)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_conditional_probabilities(probs, type_shares, edu_start, num_types_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: probs(num_types_int)

    DOUBLE PRECISION, INTENT(IN)        :: type_shares(num_types_int * 2)

    INTEGER, INTENT(IN)                 :: num_types_int
    INTEGER, INTENT(IN)                 :: edu_start

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Assign global resfort variables
    num_types = num_types_int

    probs = get_conditional_probabilities(type_shares, edu_start)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_scales_magnitude(precond_matrix_int, values, num_free_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: precond_matrix_int(num_free_int, num_free_int)

    DOUBLE PRECISION, INTENT(IN)        :: values(num_free_int)

    INTEGER, INTENT(IN)                 :: num_free_int

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Assign global RESFORT variables
    num_free = num_free_int

    precond_matrix_int = get_scales_magnitudes(values)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_back_out_systematic_wages(wages_systematic, rewards_systematic, exp_a, exp_b, edu, choice_lagged, coeffs_a, coeffs_b)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: wages_systematic(2)

    DOUBLE PRECISION, INTENT(IN)        :: rewards_systematic(4)
    DOUBLE PRECISION, INTENT(IN)        :: coeffs_a(15)
    DOUBLE PRECISION, INTENT(IN)        :: coeffs_b(15)

    INTEGER, INTENT(IN)                 :: choice_lagged
    INTEGER, INTENT(IN)                 :: exp_a
    INTEGER, INTENT(IN)                 :: exp_b
    INTEGER, INTENT(IN)                 :: edu

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Construct derived types
    optim_paras%coeffs_a = coeffs_a
    optim_paras%coeffs_b = coeffs_b

    ! Call interface
    wages_systematic = back_out_systematic_wages(rewards_systematic, exp_a, exp_b, edu, choice_lagged, optim_paras)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_sorted(sorted_array, input_array, num_elements)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: sorted_array(num_elements)

    DOUBLE PRECISION, INTENT(IN)        :: input_array(num_elements)

    INTEGER, INTENT(IN)                 :: num_elements

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    sorted_array = sorted(input_array, num_elements)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_sort_edu_spec(edu_start_sorted, edu_share_sorted, edu_max_sorted, edu_start, edu_share, edu_max, num_elements)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: edu_start_sorted(num_elements)
    INTEGER, INTENT(OUT)            :: edu_max_sorted

    DOUBLE PRECISION, INTENT(OUT)   :: edu_share_sorted(num_elements)

    INTEGER, INTENT(IN)             :: edu_start(num_elements)
    INTEGER, INTENT(IN)             :: num_elements
    INTEGER, INTENT(IN)             :: edu_max

    DOUBLE PRECISION, INTENT(IN)    :: edu_share(num_elements)

    !/* internal objects        */

    TYPE(EDU_DICT)                  :: edu_spec_sorted

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Construct derived types
    edu_spec%share = edu_share
    edu_spec%start = edu_start
    edu_spec%max = edu_max

    edu_spec_sorted = sort_edu_spec(edu_spec)

    edu_start_sorted = edu_spec_sorted%start
    edu_share_sorted = edu_spec_sorted%share
    edu_max_sorted = edu_spec_sorted%max

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_sort_type_info(type_info_order, type_info_shares, type_shares, num_types_int)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: type_info_shares(num_types_int * 2)

    INTEGER, INTENT(OUT)       		:: type_info_order(num_types_int)

    INTEGER, INTENT(IN)                 :: num_types_int

    DOUBLE PRECISION, INTENT(IN)        :: type_shares(num_types_int * 2)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Assing global RESPFRT variables
    num_types = num_types_int

    CALL sort_type_info(type_info_order, type_info_shares, type_shares)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
