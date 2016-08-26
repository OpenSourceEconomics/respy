!******************************************************************************
!******************************************************************************
MODULE solve_ambiguity

    !/*	external modules	*/

    USE shared_constants

    USE shared_auxiliary

    USE solve_risk

    !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
FUNCTION kl_divergence(mean_old, cov_old, mean_new, cov_new)

    !/* external objects        */

    REAL(our_dble)                      :: kl_divergence

    REAL(our_dble), INTENT(IN)          :: cov_old(:, :)
    REAL(our_dble), INTENT(IN)          :: cov_new(:, :)
    REAL(our_dble), INTENT(IN)          :: mean_old(:)
    REAL(our_dble), INTENT(IN)          :: mean_new(:)

    !/* internal objects        */

    INTEGER(our_int)                    :: num_dims

    REAL(our_dble), ALLOCATABLE         :: cov_old_inv(:, :)
    REAL(our_dble), ALLOCATABLE         :: mean_diff(:, :)

    REAL(our_dble)                      :: comp_b(1, 1)
    REAL(our_dble)                      :: comp_a
    REAL(our_dble)                      :: comp_c

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    num_dims = SIZE(mean_old)

    ALLOCATE(cov_old_inv(num_dims, num_dims))
    ALLOCATE(mean_diff(num_dims, 1))

    mean_diff = RESHAPE(mean_old, (/num_dims, 1/)) - RESHAPE(mean_new, (/num_dims, 1/))
    cov_old_inv = inverse(cov_old, num_dims)

    comp_a = trace(MATMUL(cov_old_inv, cov_new))
    comp_b = MATMUL(MATMUL(TRANSPOSE(mean_diff), cov_old_inv), mean_diff)
    comp_c = LOG(determinant(cov_old) / determinant(cov_new))

    kl_divergence = half_dble * (comp_a + comp_b(1, 1) - num_dims + comp_c)

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE construct_emax_ambiguity(emax, draws_emax_transformed, period, k, rewards_systematic, mapping_state_idx, states_all, periods_emax, delta, edu_start, edu_max, level)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: emax

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: draws_emax_transformed(num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)      :: rewards_systematic(4)
    REAL(our_dble), INTENT(IN)      :: level
    REAL(our_dble), INTENT(IN)      :: delta

    !/* internals objects    */

    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: total_values(4)
    REAL(our_dble)                  :: draws(4)
    REAL(our_dble)                  :: maximum

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL get_worst_case(emax, draws_emax_transformed, period, k, rewards_systematic, mapping_state_idx, states_all, periods_emax, delta, edu_start, edu_max, level)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_worst_case(emax, draws_emax_transformed, period, k, rewards_systematic, mapping_state_idx, states_all, periods_emax, delta, edu_start, edu_max, level)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: emax

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: draws_emax_transformed(num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)      :: rewards_systematic(4)
    REAL(our_dble), INTENT(IN)      :: level
    REAL(our_dble), INTENT(IN)      :: delta

    !/* internals objects    */

    REAL(our_dble)                  :: x(2)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    x = (/-level, -level/)

    CALL criterion_ambiguity(emax, x, draws_emax_transformed, period, k, rewards_systematic, mapping_state_idx, states_all, periods_emax, delta, edu_start, edu_max)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE criterion_ambiguity(emax, x, draws_emax_transformed, period, k, rewards_systematic, mapping_state_idx, states_all, periods_emax, delta, edu_start, edu_max)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: emax

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: draws_emax_transformed(num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)      :: rewards_systematic(4)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: x(2)

    !/* internals objects    */

    REAL(our_dble)                  :: draws_relevant(num_draws_emax, 4)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    draws_relevant = draws_emax_transformed
    draws_relevant(:, 1) = draws_relevant(:, 1) + x(1)
    draws_relevant(:, 2) = draws_relevant(:, 2) + x(2)

    CALL construct_emax_risk(emax, draws_relevant, period, k, rewards_systematic, mapping_state_idx, states_all, periods_emax, delta, edu_start, edu_max)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
