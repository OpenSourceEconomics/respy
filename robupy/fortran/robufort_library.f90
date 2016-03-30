!*******************************************************************************
!*******************************************************************************
!
!   Interface to ROBUPY library. This is the front-end to all functionality. 
!   Subroutines and functions for the case of risk-only case are in the 
!   robufort_risk module. Building on the risk-only functionality, the module
!   robufort_ambugity provided the required subroutines and functions for the 
!   case of ambiguity.
!
!*******************************************************************************
!*******************************************************************************
MODULE robufort_library

	!/*	external modules	*/

    USE robufort_constants

    USE robufort_auxiliary

    USE robufort_ambiguity

    USE solve_auxiliary

    USE robufort_emax

    USE robufort_risk

	!/*	setup	*/

	IMPLICIT NONE

    PUBLIC
    
 CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE fort_evaluate(rslt, periods_payoffs_systematic, mapping_state_idx, & 
                periods_emax, states_all, shocks_cov, is_deterministic, & 
                num_periods, edu_start, edu_max, delta, data_array, & 
                num_agents, num_draws_prob, periods_draws_prob)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: rslt 


    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: num_draws_prob
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: num_agents
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max

    REAL(our_dble), INTENT(IN)      :: periods_draws_prob(:, :, :)
    REAL(our_dble), INTENT(IN)      :: data_array(:, :)
    REAL(our_dble), INTENT(IN)      :: shocks_cov(:, :)
    REAL(our_dble), INTENT(IN)      :: delta 

    !/* internal objects        */

    REAL(our_dble), ALLOCATABLE     :: crit_val(:)

    REAL(our_dble), INTENT(IN)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)

    INTEGER(our_int)                :: edu_lagged
    INTEGER(our_int)                :: counts(4)
    INTEGER(our_int)                :: choice
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: exp_a
    INTEGER(our_int)                :: exp_b
    INTEGER(our_int)                :: idx
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: s
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: j
    
    REAL(our_dble)                  :: conditional_draws(num_draws_prob, 4)
    REAL(our_dble)                  :: draws_prob(num_draws_prob, 4)
    REAL(our_dble)                  :: choice_probabilities(4)
    REAL(our_dble)                  :: payoffs_systematic(4)
    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: crit_val_contrib
    REAL(our_dble)                  :: total_payoffs(4)
    REAL(our_dble)                  :: draws(4)
    REAL(our_dble)                  :: dist

    LOGICAL                         :: is_deterministic
    LOGICAL                         :: is_working

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Construct Cholesky decomposition
    IF (is_deterministic) THEN
        shocks_cholesky = zero_dble
    ELSE
        CALL cholesky(shocks_cholesky, shocks_cov)
    END IF

    ! Initialize container for likelihood contributions
    ALLOCATE(crit_val(num_agents * num_periods)); crit_val = zero_dble

    j = 1   

    DO i = 0, num_agents - 1

        DO period = 0, num_periods -1

            ! Extract observable components of state space as well as agent
            ! decision.
            exp_a = INT(data_array(j, 5))
            exp_b = INT(data_array(j, 6))
            edu = INT(data_array(j, 7))
            edu_lagged = INT(data_array(j, 8))

            choice = INT(data_array(j, 3))
            is_working = (choice == 1) .OR. (choice == 2)

            ! Transform total years of education to additional years of
            ! education and create an index from the choice.
            edu = edu - edu_start

            ! This is only done for alignment
            idx = choice

            ! Get state indicator to obtain the systematic component of the
            ! agents payoffs. These feed into the simulation of choice
            ! probabilities.
            k = mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu + 1, edu_lagged + 1)
            payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

            ! Extract relevant deviates from standard normal distribution.
            draws_prob = periods_draws_prob(period + 1, :, :)

            ! Prepare to calculate product of likelihood contributions.
            crit_val_contrib = 1.0

            ! If an agent is observed working, then the the labor market shocks
            ! are observed and the conditional distribution is used to determine
            ! the choice probabilities.
            IF (is_working) THEN

                ! Calculate the disturbance, which follows a normal
                ! distribution.
                dist = LOG(data_array(j, 4)) - LOG(payoffs_systematic(idx))
                
                ! Construct independent normal draws implied by the observed
                ! wages.
                IF (choice == 1) THEN
                    draws_prob(:, idx) = dist / sqrt(shocks_cov(idx, idx))
                ELSE
                    draws_prob(:, idx) = (dist - shocks_cholesky(idx, 1) * draws_prob(:, 1)) / shocks_cholesky(idx, idx)
                END IF
                
                ! Record contribution of wage observation. REPLACE 0.0
                crit_val_contrib =  crit_val_contrib * normal_pdf(dist, DBLE(0.0), sqrt(shocks_cov(idx, idx)))

                ! If there is no random variation in payoffs, then the
                ! observed wages need to be identical their systematic
                ! components. The discrepancy between the observed wages and 
                ! their systematic components might be small due to the 
                ! reading in of the dataset.
                IF (is_deterministic) THEN
                    IF (dist .GT. SMALL_FLOAT) THEN
                        rslt = zero_dble
                        RETURN
                    END IF
                END IF


            END IF

            ! Determine conditional deviates. These correspond to the
            ! unconditional draws if the agent did not work in the labor market.
            DO s = 1, num_draws_prob
                conditional_draws(s, :) = & 
                    MATMUL(draws_prob(s, :), TRANSPOSE(shocks_cholesky))
            END DO

            counts = 0

            DO s = 1, num_draws_prob
                ! Extract deviates from (un-)conditional normal distributions.
                draws = conditional_draws(s, :)

                draws(1) = EXP(draws(1))
                draws(2) = EXP(draws(2))

                ! Calculate total payoff.
                CALL get_total_value(total_payoffs, period, num_periods, & 
                        delta, payoffs_systematic, draws, edu_max, edu_start, & 
                        mapping_state_idx, periods_emax, k, states_all)
                
                ! Record optimal choices
                counts(MAXLOC(total_payoffs)) = counts(MAXLOC(total_payoffs)) + 1

            END DO

            ! Determine relative shares. Special care required due to integer 
            ! arithmetic, transformed to mixed mode arithmetic.
            choice_probabilities = counts / DBLE(num_draws_prob)

            ! If there is no random variation in payoffs, then this implies a
            ! unique optimal choice.
            IF (is_deterministic) THEN
                IF  ((MAXVAL(counts) .EQ. num_draws_prob) .EQV. .FALSE.) THEN
                    rslt = zero_dble
                    RETURN
                END IF
            END IF

            ! Adjust  and record likelihood contribution
            crit_val_contrib = crit_val_contrib * choice_probabilities(idx)
            crit_val(j) = crit_val_contrib
            
            j = j + 1

        END DO

    END DO 

    ! Scaling
    DO i = 1, num_agents * num_periods
        crit_val(i) = clip_value(crit_val(i), TINY_FLOAT, HUGE_FLOAT)
    END DO

    rslt = -SUM(LOG(crit_val)) / (num_agents * num_periods)

    ! If there is no random variation in payoffs and no agent violated the
    ! implications of observed wages and choices, then the evaluation return
    ! a value of one.
    IF (is_deterministic) THEN
        rslt = 1.0
    END IF

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE fort_solve(periods_payoffs_systematic, states_number_period, & 
                mapping_state_idx, periods_emax, states_all, coeffs_a, & 
                coeffs_b, coeffs_edu, coeffs_home, shocks_cov, & 
                is_deterministic, is_interpolated, num_draws_emax, & 
                periods_draws_emax, is_ambiguous, num_periods, num_points, & 
                edu_start, is_myopic, is_debug, measure, edu_max, min_idx, & 
                delta, level)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_all(:, :, :)

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(:, :)

    INTEGER(our_int), INTENT(IN)                    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)                    :: num_periods
    INTEGER(our_int), INTENT(IN)                    :: num_points
    INTEGER(our_int), INTENT(IN)                    :: edu_start
    INTEGER(our_int), INTENT(IN)                    :: edu_max
    INTEGER(our_int), INTENT(IN)                    :: min_idx

    REAL(our_dble), INTENT(IN)                      :: periods_draws_emax(:, :, :)
    REAL(our_dble), INTENT(IN)                      :: coeffs_home(:)
    REAL(our_dble), INTENT(IN)                      :: coeffs_edu(:)
    REAL(our_dble), INTENT(IN)                      :: shocks_cov(4, 4)
    REAL(our_dble), INTENT(IN)                      :: coeffs_a(:)
    REAL(our_dble), INTENT(IN)                      :: coeffs_b(:)
    REAL(our_dble), INTENT(IN)                      :: level
    REAL(our_dble), INTENT(IN)                      :: delta

    LOGICAL, INTENT(IN)                             :: is_deterministic
    LOGICAL, INTENT(IN)                             :: is_interpolated
    LOGICAL, INTENT(IN)                             :: is_ambiguous
    LOGICAL, INTENT(IN)                             :: is_myopic
    LOGICAL, INTENT(IN)                             :: is_debug

    CHARACTER(10), INTENT(IN)                       :: measure

    !/* internal objects        */

    INTEGER(our_int), ALLOCATABLE                   :: states_all_tmp(:, :, :)

    INTEGER(our_int)                                :: max_states_period
    INTEGER(our_int)                                :: period
    
    REAL(our_dble)                                  :: shocks_cholesky(4, 4)
    
!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
   
    ! Construct Cholesky decomposition
    IF (is_deterministic) THEN
        shocks_cholesky = zero_dble
    ELSE
        CALL cholesky(shocks_cholesky, shocks_cov)
    END IF

    ! Allocate arrays
    ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
    ALLOCATE(states_all_tmp(num_periods, 100000, 4))
    ALLOCATE(states_number_period(num_periods))

    ! Create the state space of the model
    CALL fort_create_state_space(states_all_tmp, states_number_period, &
            mapping_state_idx, max_states_period, num_periods, edu_start, & 
            edu_max, min_idx)

    ! Cutting the states_all container to size. The required size is only known 
    ! after the state space creation is completed.
    ALLOCATE(states_all(num_periods, max_states_period, 4))
    states_all = states_all_tmp(:, :max_states_period, :)
    DEALLOCATE(states_all_tmp)

    ! Allocate arrays
    ALLOCATE(periods_payoffs_systematic(num_periods, max_states_period, 4))
    ALLOCATE(periods_emax(num_periods, max_states_period))

    ! Calculate the systematic payoffs
    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, &
            num_periods, states_number_period, states_all, edu_start, &
            coeffs_a, coeffs_b, coeffs_edu, coeffs_home, max_states_period)

    ! Initialize containers, which contain a lot of missing values as we
    ! capture the tree structure in arrays of fixed dimension.
    periods_emax = MISSING_FLOAT

    ! Perform backward induction procedure.
    IF (is_myopic) THEN

        ! All other objects remain set to MISSING_FLOAT. This align the
        ! treatment for the two special cases: (1) is_myopic and (2)
        ! is_interpolated.
        DO period = 1,  num_periods
            periods_emax(period, :states_number_period(period)) = zero_dble
        END DO

    ELSE

        CALL fort_backward_induction(periods_emax, num_periods, &
                max_states_period, periods_draws_emax, num_draws_emax, &
                states_number_period, periods_payoffs_systematic, edu_max, &
                edu_start, mapping_state_idx, states_all, delta, is_debug, &
                shocks_cov, level, is_ambiguous, measure, is_interpolated, &
                num_points, is_deterministic, shocks_cholesky)
        
    END IF

END SUBROUTINE   
!*******************************************************************************
!*******************************************************************************
SUBROUTINE fort_simulate(dataset, num_agents, states_all, num_periods, &
                mapping_state_idx, periods_payoffs_systematic, &
                periods_draws_sims, edu_max, edu_start, periods_emax, delta)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: dataset(:, :)

    REAL(our_dble), INTENT(IN)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), INTENT(IN)      :: periods_draws_sims(:, :, :)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: delta

    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: num_agents
    INTEGER(our_int), INTENT(IN)    :: edu_max

    !/* internal objects        */

    INTEGER(our_int)                :: current_state(4)
    INTEGER(our_int)                :: edu_lagged
    INTEGER(our_int)                :: choice(1)
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: exp_a
    INTEGER(our_int)                :: exp_b
    INTEGER(our_int)                :: count
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: i   
    INTEGER(our_int)                :: k

    REAL(our_dble)                  :: payoffs_systematic(4)
    REAL(our_dble)                  :: total_payoffs(4)
    REAL(our_dble)                  :: draws(4)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    ! Initialize containers
    dataset = MISSING_FLOAT

    ! Iterate over agents and periods
    count = 0

    DO i = 0, (num_agents - 1)

        ! Baseline state
        current_state = states_all(1, 1, :)
        
        DO period = 0, (num_periods - 1)
            
            ! Distribute state space
            exp_a = current_state(1)
            exp_b = current_state(2)
            edu = current_state(3)
            edu_lagged = current_state(4)
            
            ! Getting state index
            k = mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu + 1, edu_lagged + 1)

            ! Write agent identifier and current period to data frame
            dataset(count + 1, 1) = DBLE(i)
            dataset(count + 1, 2) = DBLE(period)

            ! Calculate ex post payoffs
            payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)
            draws = periods_draws_sims(period + 1, i + 1, :)

            ! Calculate total utilities
            CALL get_total_value(total_payoffs, period, num_periods, delta, &
                    payoffs_systematic, draws, edu_max, edu_start, & 
                    mapping_state_idx, periods_emax, k, states_all)

            ! Write relevant state space for period to data frame
            dataset(count + 1, 5:8) = current_state

            ! Special treatment for education
            dataset(count + 1, 7) = dataset(count + 1, 7) + edu_start

            ! Determine and record optimal choice
            choice = MAXLOC(total_payoffs) 

            dataset(count + 1, 3) = DBLE(choice(1)) 

            !# Update work experiences and education
            IF (choice(1) .EQ. one_int) THEN 
                current_state(1) = current_state(1) + 1
            END IF

            IF (choice(1) .EQ. two_int) THEN 
                current_state(2) = current_state(2) + 1
            END IF

            IF (choice(1) .EQ. three_int) THEN 
                current_state(3) = current_state(3) + 1
            END IF
            
            IF (choice(1) .EQ. three_int) THEN 
                current_state(4) = one_int
            ELSE
                current_state(4) = zero_int
            END IF

            ! Record earnings
            IF (choice(1) .EQ. one_int) THEN
                dataset(count + 1, 4) = payoffs_systematic(1) * draws(1)
            END IF

            IF (choice(1) .EQ. two_int) THEN
                dataset(count + 1, 4) = payoffs_systematic(2) * draws(2)
            END IF

            ! Update row indicator
            count = count + 1

        END DO

    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE  