!******************************************************************************
!******************************************************************************
MODULE evaluate_fortran

    !/*	external modules	*/

    USE evaluate_auxiliary

    USE recording_warning

    USE shared_interface

    !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

 CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_contributions(contribs, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, data_evaluate, periods_draws_prob, tau, num_periods, num_draws_prob, num_agents_contrib, num_obs_agent, num_types, edu_spec, optim_paras)

    !   DEVELOPMENT NOTES
    !
    !   This routine accepts assumed shape arrays to allow its use during parallel execution as well. This is also the reasoning behind relaxing the alignment with the PYTHON counterpart.
    !
    ! 	There is the special case of deterministic shocks, which is maintained for debugging purposes. So, the sample likelihood (using get_log_likl) is zero if any agents violates the implication of the model and minus one otherwise.
    !

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: data_evaluate(:, :)

    REAL(our_dble), INTENT(OUT)     :: contribs(num_agents_contrib)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras
    TYPE(EDU_DICT), INTENT(IN)          :: edu_spec

    REAL(our_dble), INTENT(IN)      :: periods_rewards_systematic(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(IN)      :: periods_draws_prob(num_periods, num_draws_prob, 4)
    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: tau

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 4, num_types)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 5)
    INTEGER(our_int), INTENT(IN)    :: num_obs_agent(num_agents_contrib)
    INTEGER(our_int), INTENT(IN)    :: num_agents_contrib
    INTEGER(our_int), INTENT(IN)    :: num_draws_prob
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: num_types

    !/* internal objects        */

    REAL(our_dble)                  :: draws_prob_raw(num_draws_prob, 4)
    REAL(our_dble)                  :: type_shares(num_types)
    REAL(our_dble)                  :: rewards_systematic(4)
    REAL(our_dble)                  :: prob_obs(num_periods)
    REAL(our_dble)                  :: prob_type(num_types)
    REAL(our_dble)                  :: wages_systematic(2)
    REAL(our_dble)                  :: rewards_ex_post(4)
    REAL(our_dble)                  :: shocks_cov(4, 4)
    REAL(our_dble)                  :: total_values(4)
    REAL(our_dble)                  :: draws_cond(4)
    REAL(our_dble)                  :: draws_stan(4)
    REAL(our_dble)                  :: wage_observed
    REAL(our_dble)                  :: prob_choice
    REAL(our_dble)                  :: prob_wage
    REAL(our_dble)                  :: draws(4)
    REAL(our_dble)                  :: dist_1
    REAL(our_dble)                  :: dist_2
    REAL(our_dble)                  :: dist
    REAL(our_dble)                  :: mean
    REAL(our_dble)                  :: sd

    INTEGER(our_int)                :: choice_lagged
    INTEGER(our_int)                :: counts(4)
    INTEGER(our_int)                :: row_start
    INTEGER(our_int)                :: edu_start
    INTEGER(our_int)                :: num_obs
    INTEGER(our_int)                :: choice
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: exp_a
    INTEGER(our_int)                :: exp_b
    INTEGER(our_int)                :: type_
    INTEGER(our_int)                :: info
    INTEGER(our_int)                :: idx
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: s
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: p

    LOGICAL                         :: is_deterministic
    LOGICAL                         :: is_wage_missing
    LOGICAL                         :: is_working

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    shocks_cov = MATMUL(optim_paras%shocks_cholesky, TRANSPOSE(optim_paras%shocks_cholesky))
    is_deterministic = ALL(shocks_cov .EQ. zero_dble)

    ! Initialize container for likelihood contributions
    contribs = -HUGE_FLOAT

    DO j = 1, num_agents_contrib

        row_start = SUM(num_obs_agent(:(j -1))) + 1
        num_obs = num_obs_agent(j)

        edu_start = INT(data_evaluate(row_start, 7))
        type_shares = get_conditional_probabilities(optim_paras%type_shares, edu_start)

        prob_type = one_dble
        DO type_ = 0, num_types - 1


            prob_obs = zero_dble
            DO p = 0, num_obs - 1

                period = INT(data_evaluate(row_start + p, 2))

                ! Extract observable components of state space as well as agent decision.
                exp_a = INT(data_evaluate(row_start + p, 5))
                exp_b = INT(data_evaluate(row_start + p, 6))
                edu = INT(data_evaluate(row_start + p, 7))
                choice_lagged = INT(data_evaluate(row_start + p, 8))
                choice = INT(data_evaluate(row_start + p, 3))
                wage_observed = data_evaluate(row_start + p, 4)

                ! We now determine whether we also have information about the agent's wage.
                is_wage_missing = (wage_observed == HUGE_FLOAT)
                is_working = (choice == 1) .OR. (choice == 2)

                ! Create an index for the choice.
                idx = choice

                ! Extract relevant deviates from standard normal distribution.
                draws_prob_raw = periods_draws_prob(period + 1, :, :)

                ! Get state indicator to obtain the systematic component of the agents rewards. These feed into the simulation of choice probabilities.
                k = mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu + 1, choice_lagged, type_ + 1)
                rewards_systematic = periods_rewards_systematic(period + 1, k + 1, :)

                ! If an agent is observed working, then the the labor market shocks are observed and the conditional distribution is used to determine the choice probabilities.
                dist = zero_dble
                IF (is_working .AND. (.NOT. is_wage_missing)) THEN

                    wages_systematic = back_out_systematic_wages(rewards_systematic, exp_a, exp_b, edu, choice_lagged, optim_paras)

                    ! Calculate the disturbance, which follows a normal distribution.
                    CALL clip_value(dist_1, LOG(wage_observed), -HUGE_FLOAT, HUGE_FLOAT, info)
                    CALL clip_value(dist_2, LOG(wages_systematic(idx)), -HUGE_FLOAT, HUGE_FLOAT, info)
                    dist = dist_1 - dist_2

                    ! If there is no random variation in rewards, then the observed wages need to be identical their systematic components. The discrepancy between the observed wages and their systematic components might be small due to the reading in of the dataset.
                    IF (is_deterministic) THEN
                        IF (dist .GT. SMALL_FLOAT) THEN
                            contribs = one_dble
                            RETURN
                        END IF
                    END IF

                END IF

                ! Simulate the conditional distribution of alternative-specific value functions and determine the choice probabilities.
                counts = zero_int

                DO s = 1, num_draws_prob

                    ! Extract deviates from (un-)conditional normal distributions.
                    draws_stan = draws_prob_raw(s, :)

                    ! Construct independent normal draws implied by the agents state experience. This is need to maintain the correlation structure of the disturbances. Special care is needed in case of a deterministic model, as otherwise a zero division error occurs.
                    IF (is_working .AND. (.NOT. is_wage_missing)) THEN

                        IF (is_deterministic) THEN

                            prob_wage = HUGE_FLOAT

                        ELSE

                            IF (choice == 1) THEN
                                draws_stan(idx) = dist / optim_paras%shocks_cholesky(idx, idx)
                                mean = zero_dble
                                sd = ABS(optim_paras%shocks_cholesky(idx, idx))

                            ELSE
                                draws_stan(idx) = (dist - optim_paras%shocks_cholesky(idx, 1) * draws_stan(1)) / optim_paras%shocks_cholesky(idx, idx)
                                mean = optim_paras%shocks_cholesky(idx, 1) * draws_stan(1)
                                sd = ABS(optim_paras%shocks_cholesky(idx, idx))

                            END IF

                            prob_wage = normal_pdf(dist, mean, sd)

                        END IF

                    ELSE
                        prob_wage = one_dble
                    END IF

                    ! As deviates are aligned with the state experiences, create the conditional draws. Note, that the realization of the random component of wages align withe their observed counterpart in the data.
                    draws_cond = MATMUL(draws_stan, TRANSPOSE(optim_paras%shocks_cholesky))

                    ! Extract deviates from (un-)conditional normal distributions and transform labor market shocks.
                    draws = draws_cond
                    CALL clip_value(draws(1), EXP(draws(1)), zero_dble, HUGE_FLOAT, info)
                    CALL clip_value(draws(2), EXP(draws(2)), zero_dble, HUGE_FLOAT, info)

                    ! Calculate total values.

                    CALL get_total_values(total_values, rewards_ex_post, period, num_periods, rewards_systematic, draws, mapping_state_idx, periods_emax, k, states_all, optim_paras, edu_spec)

                    ! Record optimal choices
                    counts(MAXLOC(total_values)) = counts(MAXLOC(total_values)) + 1

                    ! Get the smoothed choice probability
                    prob_choice = get_smoothed_probability(total_values, idx, tau)
                    prob_obs(p + 1) = prob_obs(p + 1) + prob_choice * prob_wage

                END DO

                ! Determine relative shares
                prob_obs(p + 1) = prob_obs(p + 1) / num_draws_prob

                ! If there is no random variation in rewards, then this implies a unique optimal choice.
                IF (is_deterministic) THEN
                    IF  ((counts(idx) .EQ. num_draws_prob) .EQV. .FALSE.) THEN
                        contribs = one_dble
                        RETURN
                    END IF
                END IF

            END DO
            prob_type(type_ + 1) = PRODUCT(prob_obs(:num_obs))

        END DO

        ! Adjust and record likelihood contribution
        contribs(j) = SUM(prob_type * type_shares)

    END DO

    ! If there is no random variation in rewards and no agent violated the implications of observed wages and choices, then the evaluation return a value of one.
    IF (is_deterministic) THEN
        contribs = EXP(one_dble)
    END IF


END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
