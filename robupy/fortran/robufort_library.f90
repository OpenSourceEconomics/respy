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

    USE robufort_emax

    USE robufort_risk

	!/*	setup	*/

	IMPLICIT NONE

    PUBLIC
    
 CONTAINS
 !*******************************************************************************
!*******************************************************************************
SUBROUTINE fort_evaluate(rslt, periods_payoffs_systematic, mapping_state_idx, & 
                periods_emax, states_all, shocks_cov, shocks_cholesky, & 
                is_deterministic, num_periods, edu_start, edu_max, delta, &
                data_array, num_agents, num_draws_prob, periods_draws_prob)

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
    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
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
    REAL(our_dble)                  :: payoffs_ex_post(4)
    REAL(our_dble)                  :: total_payoffs(4)
    REAL(our_dble)                  :: draws(4)
    REAL(our_dble)                  :: crit_val_contrib
    REAL(our_dble)                  :: dist

    LOGICAL                         :: is_deterministic
    LOGICAL                         :: is_working

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

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
                CALL get_total_value(total_payoffs, payoffs_ex_post, & 
                        period, num_periods, delta, &
                        payoffs_systematic, draws, edu_max, edu_start, & 
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
SUBROUTINE fort_solve(periods_payoffs_systematic, &
                states_number_period, mapping_state_idx, periods_emax, &
                states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, & 
                shocks_cov, shocks_cholesky, is_deterministic, & 
                is_interpolated, num_draws_emax, periods_draws_emax, & 
                is_ambiguous, num_periods, num_points, edu_start, is_myopic, & 
                is_debug, measure, edu_max, min_idx, delta, level)

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
    REAL(our_dble), INTENT(IN)                      :: shocks_cholesky(:, :)
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
    
!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Allocate arrays
    ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
    ALLOCATE(states_all_tmp(num_periods, 100000, 4))
    ALLOCATE(states_number_period(num_periods))

    ! Create the state space of the model
    CALL create_state_space(states_all_tmp, states_number_period, & 
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
    CALL calculate_payoffs_systematic(periods_payoffs_systematic, num_periods, &
            states_number_period, states_all, edu_start, coeffs_a, coeffs_b, & 
            coeffs_edu, coeffs_home, max_states_period)

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

        CALL backward_induction(periods_emax, &
                num_periods, max_states_period, &
                periods_draws_emax, num_draws_emax, states_number_period, &
                periods_payoffs_systematic, edu_max, edu_start, & 
                mapping_state_idx, states_all, delta, is_debug, shocks_cov, & 
                level, is_ambiguous, measure, is_interpolated, num_points, & 
                is_deterministic, shocks_cholesky)
        
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
    REAL(our_dble)                  :: payoffs_ex_post(4)
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
            CALL get_total_value(total_payoffs, payoffs_ex_post, & 
                    period, num_periods, delta, &
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
                dataset(count + 1, 4) = payoffs_ex_post(1)
            END IF

            IF (choice(1) .EQ. two_int) THEN
                dataset(count + 1, 4) = payoffs_ex_post(2)
            END IF

            ! Update row indicator
            count = count + 1

        END DO

    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE calculate_payoffs_systematic(periods_payoffs_systematic, & 
                num_periods, states_number_period, states_all, edu_start, & 
                coeffs_a, coeffs_b, coeffs_edu, coeffs_home, max_states_period)

    !/* external objects        */

    REAL(our_dble), INTENT(INOUT)       :: periods_payoffs_systematic(:, :, :)

    REAL(our_dble), INTENT(IN)          :: coeffs_home(:)
    REAL(our_dble), INTENT(IN)          :: coeffs_edu(:)
    REAL(our_dble), INTENT(IN)          :: coeffs_a(:)
    REAL(our_dble), INTENT(IN)          :: coeffs_b(:)

    INTEGER(our_int), INTENT(IN)        :: states_number_period(:)
    INTEGER(our_int), INTENT(IN)        :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)        :: max_states_period
    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: edu_start

    !/* internals objects       */

    INTEGER(our_int)                    :: edu_lagged
    INTEGER(our_int)                    :: covars(6)
    INTEGER(our_int)                    :: period
    INTEGER(our_int)                    :: exp_a
    INTEGER(our_int)                    :: exp_b
    INTEGER(our_int)                    :: edu
    INTEGER(our_int)                    :: k

    REAL(our_dble)                      :: payoff

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Logging
    CALL logging_solution(2)

    ! Initialize missing value
    periods_payoffs_systematic = MISSING_FLOAT

    ! Calculate systematic instantaneous payoffs
    DO period = num_periods, 1, -1

        ! Loop over all possible states
        DO k = 1, states_number_period(period)

            ! Distribute state space
            exp_a = states_all(period, k, 1)
            exp_b = states_all(period, k, 2)
            edu = states_all(period, k, 3)
            edu_lagged = states_all(period, k, 4)

            ! Auxiliary objects
            covars(1) = one_int
            covars(2) = edu + edu_start
            covars(3) = exp_a
            covars(4) = exp_a ** 2
            covars(5) = exp_b
            covars(6) = exp_b ** 2

            ! Calculate systematic part of payoff in occupation A
            periods_payoffs_systematic(period, k, 1) =  &
                EXP(DOT_PRODUCT(covars, coeffs_a))

            ! Calculate systematic part of payoff in occupation B
            periods_payoffs_systematic(period, k, 2) = &
                EXP(DOT_PRODUCT(covars, coeffs_b))

            ! Calculate systematic part of schooling utility
            payoff = coeffs_edu(1)

            ! Tuition cost for higher education if agents move
            ! beyond high school.
            IF(edu + edu_start >= 12) THEN

                payoff = payoff + coeffs_edu(2)
            
            END IF

            ! Psychic cost of going back to school
            IF(edu_lagged == 0) THEN
            
                payoff = payoff + coeffs_edu(3)
            
            END IF
            periods_payoffs_systematic(period, k, 3) = payoff

            ! Calculate systematic part of payoff in home production
            periods_payoffs_systematic(period, k, 4) = coeffs_home(1)

        END DO

    END DO

    ! Logging
    CALL logging_solution(-1)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE backward_induction(periods_emax, &
                num_periods, max_states_period, &
                periods_draws_emax, num_draws_emax, states_number_period, & 
                periods_payoffs_systematic, edu_max, edu_start, & 
                mapping_state_idx, states_all, delta, is_debug, shocks_cov, & 
                level, is_ambiguous, measure, is_interpolated, num_points, & 
                is_deterministic, shocks_cholesky)

    !
    ! Development Notes
    ! -----------------
    !
    !   The input argument MEASURE is only present to align the interface 
    !   between the FORTRAN and PYTHON implementations.
    !

    !/* external objects        */

    REAL(our_dble), INTENT(INOUT)       :: periods_emax(:, :)

    REAL(our_dble), INTENT(IN)          :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), INTENT(IN)          :: periods_draws_emax(:, :, :)
    REAL(our_dble), INTENT(IN)          :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)          :: shocks_cov(:, :)
    REAL(our_dble), INTENT(IN)          :: delta
    REAL(our_dble), INTENT(IN)          :: level

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(:, :, :, :, :)    
    INTEGER(our_int), INTENT(IN)        :: states_number_period(:)
    INTEGER(our_int), INTENT(IN)        :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)        :: max_states_period
    INTEGER(our_int), INTENT(IN)        :: num_draws_emax
    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: num_points
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max

    LOGICAL, INTENT(IN)                 :: is_deterministic
    LOGICAL, INTENT(IN)                 :: is_interpolated
    LOGICAL, INTENT(IN)                 :: is_ambiguous
    LOGICAL, INTENT(IN)                 :: is_debug

    CHARACTER(10), INTENT(IN)           :: measure

    !/* internals objects       */

    INTEGER(our_int)                    :: num_states
    INTEGER(our_int)                    :: period
    INTEGER(our_int)                    :: k

    REAL(our_dble)                      :: draws_emax(num_draws_emax, 4)
    REAL(our_dble)                      :: payoffs_systematic(4)
    REAL(our_dble)                      :: payoffs_ex_post(4)
    REAL(our_dble)                      :: emax_simulated
    REAL(our_dble)                      :: shifts(4)

    REAL(our_dble), ALLOCATABLE         :: exogenous(:, :)
    REAL(our_dble), ALLOCATABLE         :: predictions(:)
    REAL(our_dble), ALLOCATABLE         :: endogenous(:)
    REAL(our_dble), ALLOCATABLE         :: maxe(:)

    LOGICAL                             :: any_interpolated

    LOGICAL, ALLOCATABLE                :: is_simulated(:)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Shifts
    shifts = zero_dble
    shifts(:2) = (/ EXP(shocks_cov(1, 1)/two_dble), EXP(shocks_cov(2, 2)/two_dble) /)

    ! Logging
    CALL logging_solution(3)

    ! Backward induction
    DO period = (num_periods - 1), 0, -1

        ! Extract draws and construct auxiliary objects
        draws_emax = periods_draws_emax(period + 1, :, :)
        num_states = states_number_period(period + 1)

        ! Logging
        CALL logging_solution(4, period, num_states)

        ! Distinguish case with and without interpolation
        any_interpolated = (num_points .LE. num_states) .AND. is_interpolated

        IF (any_interpolated) THEN

            ! Allocate period-specific containers
            ALLOCATE(is_simulated(num_states)); ALLOCATE(endogenous(num_states))
            ALLOCATE(maxe(num_states)); ALLOCATE(exogenous(num_states, 9))
            ALLOCATE(predictions(num_states))

            ! Constructing indicator for simulation points
            is_simulated = get_simulated_indicator(num_points, num_states, & 
                                period, num_periods, is_debug)

            ! Constructing the dependent variable for all states, including the
            ! ones where simulation will take place. All information will be
            ! used in either the construction of the prediction model or the
            ! prediction step.
            CALL get_exogenous_variables(exogenous, maxe, period, num_periods, &
                    num_states, delta, periods_payoffs_systematic, shifts, &
                    edu_max, edu_start, mapping_state_idx, periods_emax, &
                    states_all)
            
            ! Construct endogenous variables for the subset of simulation points.
            ! The rest is set to missing value.
            CALL get_endogenous_variable(endogenous, period, num_periods, &
                    num_states, delta, periods_payoffs_systematic, edu_max, & 
                    edu_start, mapping_state_idx, periods_emax, states_all, & 
                    is_simulated, num_draws_emax, shocks_cov, level, & 
                    is_ambiguous, is_debug, measure, maxe, draws_emax, & 
                    is_deterministic, shocks_cholesky)

            ! Create prediction model based on the random subset of points where
            ! the EMAX is actually simulated and thus endogenous and
            ! exogenous variables are available. For the interpolation 
            ! points, the actual values are used.
            CALL get_predictions(predictions, endogenous, exogenous, maxe, & 
                    is_simulated, num_points, num_states)
            
            ! Store results
            periods_emax(period + 1, :num_states) = predictions

            ! Deallocate containers
            DEALLOCATE(is_simulated); DEALLOCATE(exogenous); DEALLOCATE(maxe); 
            DEALLOCATE(endogenous); DEALLOCATE(predictions)

        ELSE

            ! Loop over all possible states
            DO k = 0, (states_number_period(period + 1) - 1)

                ! Extract payoffs
                payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

                CALL get_payoffs(emax_simulated, payoffs_ex_post, &
                        num_draws_emax, draws_emax, period, & 
                        k, payoffs_systematic, edu_max, edu_start, & 
                        mapping_state_idx, states_all, num_periods, & 
                        periods_emax, delta, is_debug, shocks_cov, level, & 
                        is_ambiguous, measure, is_deterministic, & 
                        shocks_cholesky)

                ! Collect information
                periods_emax(period + 1, k + 1) = emax_simulated

                ! This information is only available if no interpolation is 
                ! used. Otherwise all remain set to missing values (see above). 

            END DO

        END IF

    END DO

    ! Logging
    CALL logging_solution(-1)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE create_state_space(states_all, states_number_period, &
                mapping_state_idx, max_states_period, num_periods, edu_start, & 
                edu_max, min_idx)

    !/* external objects        */

    INTEGER(our_int), INTENT(INOUT)     :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(INOUT)     :: states_number_period(:)
    INTEGER(our_int), INTENT(INOUT)     :: states_all(:, :, :)
    INTEGER(our_int), INTENT(INOUT)     :: max_states_period

    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max
    INTEGER(our_int), INTENT(IN)        :: min_idx

    !/* internals objects       */

    INTEGER(our_int)                    :: edu_lagged
    INTEGER(our_int)                    :: period
    INTEGER(our_int)                    :: total
    INTEGER(our_int)                    :: exp_a
    INTEGER(our_int)                    :: exp_b
    INTEGER(our_int)                    :: edu
    INTEGER(our_int)                    :: k
 
!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Initialize output 
    states_number_period = MISSING_INT
    mapping_state_idx    = MISSING_INT
    states_all           = MISSING_INT

    ! Logging
    CALL logging_solution(1)

    ! Construct state space by periods
    DO period = 0, (num_periods - 1)

        ! Count admissible realizations of state space by period
        k = 0

        ! Loop over all admissible work experiences for occupation A
        DO exp_a = 0, num_periods

            ! Loop over all admissible work experience for occupation B
            DO exp_b = 0, num_periods
                
                ! Loop over all admissible additional education levels
                DO edu = 0, num_periods

                    ! Agent cannot attain more additional education
                    ! than (EDU_MAX - EDU_START).
                    IF (edu .GT. edu_max - edu_start) THEN
                        CYCLE
                    END IF

                    ! Loop over all admissible values for leisure. Note that
                    ! the leisure variable takes only zero/value. The time path
                    ! does not matter.
                    DO edu_lagged = 0, 1

                        ! Check if lagged education admissible. (1) In the
                        ! first period all agents have lagged schooling equal
                        ! to one.
                        IF (edu_lagged .EQ. zero_int) THEN
                            IF (period .EQ. zero_int) THEN
                                CYCLE
                            END IF
                        END IF
                        
                        ! (2) Whenever an agent has not acquired any additional
                        ! education and we are not in the first period,
                        ! then this cannot be the case.
                        IF (edu_lagged .EQ. one_int) THEN
                            IF (edu .EQ. zero_int) THEN
                                IF (period .GT. zero_int) THEN
                                    CYCLE
                                END IF
                            END IF
                        END IF

                        ! (3) Whenever an agent has only acquired additional
                        ! education, then edu_lagged cannot be zero.
                        IF (edu_lagged .EQ. zero_int) THEN
                            IF (edu .EQ. period) THEN
                                CYCLE
                            END IF
                        END IF

                        ! Check if admissible for time constraints
                        total = edu + exp_a + exp_b

                        ! Note that the total number of activities does not
                        ! have is less or equal to the total possible number of
                        ! activities as the rest is implicitly filled with
                        ! leisure.
                        IF (total .GT. period) THEN
                            CYCLE
                        END IF
                        
                        ! Collect all possible realizations of state space
                        states_all(period + 1, k + 1, 1) = exp_a
                        states_all(period + 1, k + 1, 2) = exp_b
                        states_all(period + 1, k + 1, 3) = edu
                        states_all(period + 1, k + 1, 4) = edu_lagged

                        ! Collect mapping of state space to array index.
                        mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, & 
                            edu + 1 , edu_lagged + 1) = k

                        ! Update count
                        k = k + 1

                     END DO

                 END DO

             END DO

         END DO
        
        ! Record maximum number of state space realizations by time period
        states_number_period(period + 1) = k

      END DO

      ! Logging
      CALL logging_solution(-1)

      ! Auxiliary object
      max_states_period = MAXVAL(states_number_period)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_payoffs(emax_simulated, payoffs_ex_post, &
                num_draws_emax, draws_emax, period, k, payoffs_systematic, & 
                edu_max, edu_start, mapping_state_idx, states_all, & 
                num_periods, periods_emax, delta, is_debug, shocks_cov, & 
                level, is_ambiguous, measure, is_deterministic, shocks_cholesky)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: payoffs_ex_post(:)
    REAL(our_dble), INTENT(OUT)         :: emax_simulated

    REAL(our_dble), INTENT(IN)          :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)          :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)          :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)          :: shocks_cov(:, :)
    REAL(our_dble), INTENT(IN)          :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)          :: delta
    REAL(our_dble), INTENT(IN)          :: level

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)        :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)        :: num_draws_emax
    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max
    INTEGER(our_int), INTENT(IN)        :: period
    INTEGER(our_int), INTENT(IN)        :: k 


    LOGICAL, INTENT(IN)                 :: is_deterministic    
    LOGICAL, INTENT(IN)                 :: is_ambiguous
    LOGICAL, INTENT(IN)                 :: is_debug

    CHARACTER(10), INTENT(IN)           :: measure

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Payoffs require different machinery depending on whether there is
    ! ambiguity or not.
    IF (is_ambiguous) THEN

        CALL get_payoffs_ambiguity(emax_simulated, payoffs_ex_post, &
                num_draws_emax, draws_emax, period, k, & 
                payoffs_systematic, edu_max, edu_start, mapping_state_idx, & 
                states_all, num_periods, periods_emax, delta, is_debug, & 
                shocks_cov, level, measure, is_deterministic, shocks_cholesky)

    ELSE 

        CALL get_payoffs_risk(emax_simulated, payoffs_ex_post, & 
                num_draws_emax, draws_emax, period, k, payoffs_systematic, & 
                edu_max, edu_start, mapping_state_idx, states_all, & 
                num_periods, periods_emax, delta, shocks_cholesky)

    END IF
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_endogenous_variable(endogenous, period, num_periods, &
                num_states, delta, periods_payoffs_systematic, edu_max, & 
                edu_start, mapping_state_idx, periods_emax, states_all, & 
                is_simulated, num_draws_emax, shocks_cov, level, is_ambiguous, &
                is_debug, measure, maxe, draws_emax, & 
                is_deterministic, shocks_cholesky)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: endogenous(:)

    REAL(our_dble), INTENT(IN)          :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), INTENT(IN)          :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)          :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)          :: shocks_cov(:, :)    
    REAL(our_dble), INTENT(IN)          :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)          :: maxe(:)
    REAL(our_dble), INTENT(IN)          :: level
    REAL(our_dble), INTENT(IN)          :: delta
 
    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(:, :, :, :, :)    
    INTEGER(our_int), INTENT(IN)        :: states_all(:, :, :)    
    INTEGER(our_int), INTENT(IN)        :: num_draws_emax
    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: num_states
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max
    INTEGER(our_int), INTENT(IN)        :: period


    LOGICAL, INTENT(IN)                 :: is_deterministic
    LOGICAL, INTENT(IN)                 :: is_simulated(:)
    LOGICAL, INTENT(IN)                 :: is_ambiguous
    LOGICAL, INTENT(IN)                 :: is_debug

    CHARACTER(10), INTENT(IN)           :: measure

    !/* internal objects        */

    REAL(our_dble)                      :: payoffs_systematic(4)
    REAL(our_dble)                      :: payoffs_ex_post(4)
    REAL(our_dble)                      :: emax_simulated

    INTEGER(our_int)                    :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Initialize missing values
    endogenous = MISSING_FLOAT

    ! Construct dependent variables for the subset of interpolation 
    ! points.
    DO k = 0, (num_states - 1)

        ! Skip over points that will be predicted
        IF (.NOT. is_simulated(k + 1)) THEN
            CYCLE
        END IF

        ! Extract payoffs 
        payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

        ! Get payoffs
        CALL get_payoffs(emax_simulated, payoffs_ex_post, &
                num_draws_emax, draws_emax, period, k, payoffs_systematic, & 
                edu_max, edu_start, mapping_state_idx, states_all, & 
                num_periods, periods_emax, delta, is_debug, shocks_cov, & 
                level, is_ambiguous, measure, is_deterministic, shocks_cholesky)

        ! Construct dependent variable
        endogenous(k + 1) = emax_simulated - maxe(k + 1)

    END DO
            
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE  