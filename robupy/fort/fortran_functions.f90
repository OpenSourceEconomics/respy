!******************************************************************************
!******************************************************************************
SUBROUTINE simulate_emax(emax_simulated, payoffs_ex_post, future_payoffs, num_periods, &
                 max_states_period, num_draws, period, k, eps_relevant, payoffs_ex_ante, &
                 edu_max, edu_start, emax, states_all, mapping_state_idx, delta)


    !/* external libraries    */

    USE robupy_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: emax_simulated
    DOUBLE PRECISION, INTENT(OUT)   :: payoffs_ex_post(4)
    DOUBLE PRECISION, INTENT(OUT)   :: future_payoffs(4)

    DOUBLE PRECISION, INTENT(IN)    :: eps_relevant(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: payoffs_ex_ante(:)
    DOUBLE PRECISION, INTENT(IN)    :: emax(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: mapping_state_idx(:,:,:,:,:)

    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: max_states_period
    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: k
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: edu_start

    REAL(our_dble)                  :: total_payoffs(4)
    REAL(our_dble)                  :: maximum

    !/* internals objects    */

    INTEGER(our_int)    :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
future_payoffs = 0.00
! Initialize containers
emax_simulated = zero_dble

DO i = 1, num_draws

    ! Calculate ex post payoffs
    payoffs_ex_post(1) = payoffs_ex_ante(1) * eps_relevant(i, 1)
    payoffs_ex_post(2) = payoffs_ex_ante(2) * eps_relevant(i, 2)
    payoffs_ex_post(3) = payoffs_ex_ante(3) + eps_relevant(i, 3)
    payoffs_ex_post(4) = payoffs_ex_ante(4) + eps_relevant(i, 4)

    ! Check applicability
    IF (period .EQ. (num_periods - 1)) THEN

        future_payoffs = - HUGE(future_payoffs)
        CONTINUE

    ELSE

        ! Get future values
        CALL get_future_payoffs_lib(future_payoffs, edu_max, edu_start, mapping_state_idx, period,  &
                                emax, k, states_all)


        ! Calculate total utilities
        total_payoffs = payoffs_ex_post + delta * future_payoffs

        ! Determine optimal choice
        maximum = MAXVAL(total_payoffs)

        ! Recording expected future value
        emax_simulated = emax_simulated + maximum

        ! Scaling
        emax_simulated = emax_simulated / num_draws

    END IF

END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE calculate_payoffs_ex_ante(period_payoffs_ex_ante, num_periods, &
              states_number_period, states_all, edu_start, &
              coeffs_A, coeffs_B, coeffs_edu, coeffs_home, max_states_period)


    ! Development Notes:
    !
    !     Future releases of the gfortran compiler will allow to assign NAN to the array
    !     using the IEEE_ARITHMETIC module.
    !

    !/* external libraries    */

    USE robupy_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: period_payoffs_ex_ante(num_periods, &
                                            max_states_period, 4)

    DOUBLE PRECISION, INTENT(IN)    :: coeffs_A(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_B(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)


    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: states_number_period(:)
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: max_states_period

    !/* internals objects    */

    INTEGER(our_int)    :: period
    INTEGER(our_int)    :: k
    INTEGER(our_int)    :: exp_A
    INTEGER(our_int)    :: exp_B
    INTEGER(our_int)    :: edu
    INTEGER(our_int)    :: edu_lagged

    REAL(our_dble)      :: covars(6)
    REAL(our_dble)      :: payoff

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

! Calculate systematic instantaneous payoffs
DO period = num_periods, 1, -1

    ! Loop over all possible states
    DO k = 1, states_number_period(period)

        ! Distribute state space
        exp_A = states_all(period, k, 1)
        exp_B = states_all(period, k, 2)
        edu = states_all(period, k, 3)
        edu_lagged = states_all(period, k, 4)

        ! Auxiliary objects
        covars(1) = one_dble
        covars(2) = edu + edu_start
        covars(3) = exp_A
        covars(4) = exp_A ** 2
        covars(5) = exp_B
        covars(6) = exp_B ** 2

        ! Calculate systematic part of payoff in occupation A
        period_payoffs_ex_ante(period, k, 1) = EXP(DOT_PRODUCT(covars, coeffs_A))

        ! Calculate systematic part of payoff in occupation B
        period_payoffs_ex_ante(period, k, 2) = EXP(DOT_PRODUCT(covars, coeffs_B))

        ! Calculate systematic part of schooling utility
        payoff = coeffs_edu(1)

        ! Tuition cost for higher education if agents move
        ! beyond high school.
        IF(edu + edu_start > 12) THEN
            payoff = payoff + coeffs_edu(2)
        END IF

        ! Psychic cost of going back to school
        IF(edu_lagged == 0) THEN
            payoff = payoff + coeffs_edu(3)
        END IF
        period_payoffs_ex_ante(period, k, 3) = payoff

        ! Calculate systematic part of payoff in home production
        period_payoffs_ex_ante(period, k, 4) = coeffs_home(1)

    END DO

END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_future_payoffs(future_payoffs, edu_max, edu_start, mapping_state_idx, period, emax, k, states_all)

    !/* external libraries    */

    USE robupy_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: future_payoffs(4)

    DOUBLE PRECISION, INTENT(IN)    :: emax(:,:)

    INTEGER, INTENT(IN)             :: k
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: mapping_state_idx(:,:,:,:,:)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

        CALL get_future_payoffs_lib(future_payoffs, edu_max, edu_start, mapping_state_idx, period,  &
                                emax, k, states_all)
END SUBROUTINE

