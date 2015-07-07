SUBROUTINE wrapper_calculate_payoffs_ex_ante(period_payoffs_ex_ante, num_periods, &
              states_number_period, states_all, edu_start, &
              coeffs_A, coeffs_B, coeffs_edu, coeffs_home, max_states_period)

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


    !/* external objects    */

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
