!******************************************************************************
!******************************************************************************
MODULE evaluate_fortran

	!/*	external modules	*/

    USE evaluate_auxiliary
    
    USE recording_warning
    
    USE shared_auxiliary

    USE shared_constants

	!/*	setup	*/

    IMPLICIT NONE

    PUBLIC

 CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_evaluate(rslt, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_evaluate, periods_draws_prob, delta, tau, edu_start, edu_max)

    !   DEVELOPMENT NOTES
    !
    !   This routine accepts assumed shape arrays to allow its use during parallel execution as well. This is also the reasoning behind relaxing the alignment with the PYTHON counterpart.

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: rslt

    REAL(our_dble), INTENT(IN)      :: periods_payoffs_systematic(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(IN)      :: periods_draws_prob(num_periods, num_draws_prob, 4)
    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)      :: data_evaluate(:, :)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: tau
        
    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max

    !/* internal objects        */

    REAL(our_dble), ALLOCATABLE     :: crit_val(:)

    INTEGER(our_int), ALLOCATABLE   :: infos(:)

    REAL(our_dble)                  :: draws_prob_raw(num_draws_prob, 4)
    REAL(our_dble)                  :: payoffs_systematic(4)
    REAL(our_dble)                  :: crit_val_contrib
    REAL(our_dble)                  :: shocks_cov(4, 4)
    REAL(our_dble)                  :: total_payoffs(4)
    REAL(our_dble)                  :: draws_cond(4)
    REAL(our_dble)                  :: draws_stan(4)
    REAL(our_dble)                  :: prob_choice
    REAL(our_dble)                  :: prob_wage
    REAL(our_dble)                  :: prob_obs
    REAL(our_dble)                  :: draws(4)
    REAL(our_dble)                  :: dist_1
    REAL(our_dble)                  :: dist_2
    REAL(our_dble)                  :: dist

    INTEGER(our_int)                :: edu_lagged
    INTEGER(our_int)                :: counts(4)
    INTEGER(our_int)                :: choice
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: num_obs
    INTEGER(our_int)                :: exp_a
    INTEGER(our_int)                :: exp_b
    INTEGER(our_int)                :: info
    INTEGER(our_int)                :: idx
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: s
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: j

    LOGICAL                         :: is_deterministic
    LOGICAL                         :: is_working

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Construct auxiliary objects
    num_obs = SIZE(data_evaluate, 1)

    shocks_cov = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))
    is_deterministic = ALL(shocks_cov .EQ. zero_dble)

    ! Initialize container for likelihood contributions
    ALLOCATE(crit_val(num_obs))
    crit_val = zero_dble

    
    DO j = 1, num_obs

            period = INT(data_evaluate(j, 2)) 

            ! Extract observable components of state space as well as agent decision.
            exp_a = INT(data_evaluate(j, 5))
            exp_b = INT(data_evaluate(j, 6))
            edu = INT(data_evaluate(j, 7))
            edu_lagged = INT(data_evaluate(j, 8))

            choice = INT(data_evaluate(j, 3))
            is_working = (choice == 1) .OR. (choice == 2)

            ! Transform total years of education to additional years of education and create an index from the choice.
            edu = edu - edu_start

            ! This is only done for alignment
            idx = choice

            ! Get state indicator to obtain the systematic component of the agents payoffs. These feed into the simulation of choice probabilities.
            k = mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu + 1, edu_lagged + 1)
            payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

            ! Extract relevant deviates from standard normal distribution.
            draws_prob_raw = periods_draws_prob(period + 1, :, :)

            ! Prepare to calculate product of likelihood contributions.
            crit_val_contrib = 1.0

            ! If an agent is observed working, then the the labor market shocks are observed and the conditional distribution is used to determine the choice probabilities.
            dist = zero_dble
            IF (is_working) THEN

                ! Calculate the disturbance, which follows a normal distribution.
                CALL clip_value(dist_1, LOG(data_evaluate(j, 4)), -HUGE_FLOAT, HUGE_FLOAT, info)
                CALL clip_value(dist_2, LOG(payoffs_systematic(idx)), -HUGE_FLOAT, HUGE_FLOAT, info) 
                dist = dist_1 - dist_2 

                ! If there is no random variation in payoffs, then the observed wages need to be identical their systematic components. The discrepancy between the observed wages and their systematic components might be small due to the reading in of the dataset.
                IF (is_deterministic) THEN
                    IF (dist .GT. SMALL_FLOAT) THEN
                        rslt = zero_dble
                        RETURN
                    END IF
                END IF


            END IF

            ! Simulate the conditional distribution of alternative-specific value functions and determine the choice probabilities.
            counts = zero_int
            prob_obs = zero_dble

            DO s = 1, num_draws_prob
                
                ! Extract deviates from (un-)conditional normal distributions.
                draws_stan = draws_prob_raw(s, :)

                ! Construct independent normal draws implied by the agents state experience. This is need to maintain the correlation structure of the disturbances. Special care is needed in case of a deterministic model, as otherwise a zero division error occurs. 
                IF (is_working) THEN 
                    
                    IF (is_deterministic) THEN

                        prob_wage = HUGE_FLOAT

                    ELSE

                        IF (choice == 1) THEN
                            draws_stan(idx) = dist / shocks_cholesky(idx, idx)
                        ELSE
                            draws_stan(idx) = (dist - shocks_cholesky(idx, 1) * draws_stan(1)) / shocks_cholesky(idx, idx)
                        END IF

                        prob_wage = normal_pdf(draws_stan(idx), zero_dble, one_dble) / SQRT(shocks_cov(idx, idx))
                    
                    END IF

                ELSE 
                    prob_wage = one_dble
                END IF

                ! As deviates are aligned with the state experiences, create the conditional draws. Note, that the realization of the random component of wages align withe their observed counterpart in the data.
                draws_cond = MATMUL(draws_stan, TRANSPOSE(shocks_cholesky))

                ! Extract deviates from (un-)conditional normal distributions and transform labor market shocks.
                draws = draws_cond
                CALL clip_value(draws(1), EXP(draws(1)), zero_dble, HUGE_FLOAT, info)
                CALL clip_value(draws(2), EXP(draws(2)), zero_dble, HUGE_FLOAT, info)

                ! Calculate total payoff.
                CALL get_total_value(total_payoffs, period, payoffs_systematic, draws, mapping_state_idx, periods_emax, k, states_all, delta, edu_start, edu_max)

                ! Record optimal choices
                counts(MAXLOC(total_payoffs)) = counts(MAXLOC(total_payoffs)) + 1

                ! Get the smoothed choice probability
                prob_choice = get_smoothed_probability(total_payoffs, idx, tau)
                prob_obs = prob_obs + prob_choice * prob_wage

            END DO

            ! Determine relative shares
            prob_obs = prob_obs / num_draws_prob

            ! If there is no random variation in payoffs, then this implies a unique optimal choice.
            IF (is_deterministic) THEN
                IF  ((counts(idx) .EQ. num_draws_prob) .EQV. .FALSE.) THEN
                    rslt = zero_dble
                    RETURN
                END IF
            END IF

            ! Adjust  and record likelihood contribution
            crit_val(j) = prob_obs

        END DO

    ! Scaling
    CALL clip_value(crit_val, LOG(crit_val), -HUGE_FLOAT, HUGE_FLOAT, infos)
    rslt = -SUM(crit_val) / (DBLE(num_periods) * DBLE(num_agents_est))

    IF (SUM(infos) > zero_int) CALL record_warning(5)

    ! If there is no random variation in payoffs and no agent violated the implications of observed wages and choices, then the evaluation return a value of one.
    IF (is_deterministic) THEN
        rslt = 1.0
    END IF
    
END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE