!******************************************************************************
!******************************************************************************
MODULE solve_risk

    !/*	external modules	*/

    USE recording_solution

    USE shared_interface

    !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE construct_emax_risk(emax, period, k, draws_emax_risk, rewards_systematic, periods_emax, states_all, mapping_state_idx, edu_spec, optim_paras)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)         :: emax

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras
    TYPE(EDU_DICT), INTENT(IN)          :: edu_spec

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 4, num_types)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 5)
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: draws_emax_risk(num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)      :: rewards_systematic(4)

    !/* internals objects    */

    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: total_values(4)
    REAL(our_dble)                  :: draws(4)
    REAL(our_dble)                  :: maximum

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Iterate over Monte Carlo draws
    emax = zero_dble
    DO i = 1, num_draws_emax

        ! Select draws for this draw
        draws = draws_emax_risk(i, :)

        ! Calculate total value
        CALL get_total_values(total_values, period, num_periods, rewards_systematic, draws, mapping_state_idx, periods_emax, k, states_all, optim_paras, edu_spec)

        ! Determine optimal choice
        maximum = MAXVAL(total_values)

        ! Recording expected future value
        emax = emax + maximum

    END DO

    ! Scaling
    emax = emax / num_draws_emax

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
