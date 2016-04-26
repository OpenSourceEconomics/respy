!*******************************************************************************
!*******************************************************************************
MODULE solve_auxiliary

	!/*	external modules	*/

    USE shared_auxiliary

    USE shared_constants

	!/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE fort_create_state_space(states_all, states_number_period, &
                mapping_state_idx, max_states_period, num_periods, edu_start, &
                edu_max)

    !/* external objects        */

    INTEGER(our_int), INTENT(INOUT)     :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(INOUT)     :: states_number_period(:)
    INTEGER(our_int), INTENT(INOUT)     :: states_all(:, :, :)
    INTEGER(our_int), INTENT(INOUT)     :: max_states_period

    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max

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
SUBROUTINE fort_calculate_payoffs_systematic(periods_payoffs_systematic, &
                num_periods, states_number_period, states_all, edu_start, &
                coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

    !/* external objects        */

    REAL(our_dble), INTENT(INOUT)       :: periods_payoffs_systematic(:, :, :)

    REAL(our_dble), INTENT(IN)          :: coeffs_home(:)
    REAL(our_dble), INTENT(IN)          :: coeffs_edu(:)
    REAL(our_dble), INTENT(IN)          :: coeffs_a(:)
    REAL(our_dble), INTENT(IN)          :: coeffs_b(:)

    INTEGER(our_int), INTENT(IN)        :: states_number_period(:)
    INTEGER(our_int), INTENT(IN)        :: states_all(:, :, :)
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
                clip_value(EXP(DOT_PRODUCT(covars, coeffs_a)), zero_dble, HUGE_FLOAT)
                

            ! Calculate systematic part of payoff in occupation B
            periods_payoffs_systematic(period, k, 2) = &
                clip_value(EXP(DOT_PRODUCT(covars, coeffs_b)), zero_dble, HUGE_FLOAT)

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
SUBROUTINE fort_backward_induction(periods_emax, num_periods, &
                periods_draws_emax, num_draws_emax, states_number_period, & 
                periods_payoffs_systematic, edu_max, edu_start, & 
                mapping_state_idx, states_all, delta, is_debug, shocks_cov, & 
                is_interpolated, num_points, &
                shocks_cholesky)

    !/* external objects        */

    REAL(our_dble), INTENT(INOUT)       :: periods_emax(:, :)

    REAL(our_dble), INTENT(IN)          :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), INTENT(IN)          :: periods_draws_emax(:, :, :)
    REAL(our_dble), INTENT(IN)          :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)          :: shocks_cov(:, :)
    REAL(our_dble), INTENT(IN)          :: delta

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)        :: states_number_period(:)
    INTEGER(our_int), INTENT(IN)        :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)        :: num_draws_emax
    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: num_points
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max

    LOGICAL, INTENT(IN)                 :: is_interpolated
    LOGICAL, INTENT(IN)                 :: is_debug

    !/* internals objects       */

    INTEGER(our_int)                    :: num_states
    INTEGER(our_int)                    :: period
    INTEGER(our_int)                    :: k

    REAL(our_dble)                      :: draws_emax(num_draws_emax, 4)
    REAL(our_dble)                      :: payoffs_systematic(4)
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
    shifts(1) = clip_value(EXP(shocks_cov(1, 1)/two_dble), zero_dble, HUGE_FLOAT)
    shifts(2) = clip_value(EXP(shocks_cov(2, 2)/two_dble), zero_dble, HUGE_FLOAT)

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
                                period, is_debug, num_periods)

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
                    is_simulated, num_draws_emax, &
                    maxe, draws_emax, shocks_cholesky)

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

                CALL get_payoffs(emax_simulated, num_draws_emax, draws_emax, &
                        period, k, payoffs_systematic, edu_max, edu_start, &
                        mapping_state_idx, states_all, num_periods, &
                        periods_emax, delta, shocks_cholesky)

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
SUBROUTINE get_payoffs(emax_simulated, num_draws_emax, draws_emax, period, &
                k, payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
                states_all, num_periods, periods_emax, delta, shocks_cholesky)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: emax_simulated

    REAL(our_dble), INTENT(IN)          :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)          :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)          :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)          :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)          :: delta

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)        :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)        :: num_draws_emax
    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max
    INTEGER(our_int), INTENT(IN)        :: period
    INTEGER(our_int), INTENT(IN)        :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Simulated expected future value
    CALL simulate_emax(emax_simulated, num_periods, num_draws_emax, & 
            period, k, draws_emax, payoffs_systematic, edu_max, edu_start, & 
            periods_emax, states_all, mapping_state_idx, delta, & 
            shocks_cholesky)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
FUNCTION get_simulated_indicator(num_points, num_states, period, is_debug, &
            num_periods)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)      :: num_periods
    INTEGER(our_int), INTENT(IN)      :: num_states
    INTEGER(our_int), INTENT(IN)      :: num_points
    INTEGER(our_int), INTENT(IN)      :: period

    LOGICAL, INTENT(IN)               :: is_debug

    !/* internal objects        */

    INTEGER(our_int)                  :: candidates(num_states)
    INTEGER(our_int)                  :: sample(num_points)
    INTEGER(our_int)                  :: i

    LOGICAL                           :: is_simulated_container(num_states, num_periods)
    LOGICAL                           :: get_simulated_indicator(num_states)
    LOGICAL                           :: is_simulated(num_states)
    LOGICAL                           :: READ_IN

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Initialize set of candidate values
    candidates = (/ (i, i = 1, num_states) /)

    ! Check if standardization requested
    IF (is_debug) THEN

        INQUIRE(FILE='interpolation.txt', EXIST=READ_IN)

        IF (READ_IN) THEN

            OPEN(12, file='interpolation.txt')

               DO i = 1, num_states

                    READ(12, *)  is_simulated_container(i, :)

                END DO

            CLOSE(12)

            get_simulated_indicator = is_simulated_container(:, period + 1)

            RETURN

        END IF

    END IF

    ! Handle special cases
    IF (num_points .EQ. num_states) THEN

        get_simulated_indicator = .TRUE.

        RETURN

    ELSEIF (num_points .GT. num_states) THEN

        PRINT *, ' Bad Request in interpolation code'

        STOP

    END IF

    ! Draw a random sample
    CALL random_choice(sample, candidates, num_states, num_points)

    ! Update information indicator
    is_simulated = .False.

    DO i = 1, num_points

        is_simulated(sample(i)) = .True.

    END DO

    ! Finish
    get_simulated_indicator = is_simulated

END FUNCTION
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_exogenous_variables(independent_variables, maxe, period, &
                num_periods, num_states, delta, periods_payoffs_systematic, &
                shifts, edu_max, edu_start, mapping_state_idx, periods_emax, &
                states_all)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: independent_variables(:, :)
    REAL(our_dble), INTENT(OUT)         :: maxe(:)

    REAL(our_dble), INTENT(IN)          :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), INTENT(IN)          :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)          :: shifts(:)
    REAL(our_dble), INTENT(IN)          :: delta

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)        :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: num_states
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max
    INTEGER(our_int), INTENT(IN)        :: period


    !/* internal objects        */

    REAL(our_dble)                      :: payoffs_systematic(4)
    REAL(our_dble)                      :: total_payoffs(4)
    REAL(our_dble)                      :: diff(4)

    INTEGER(our_int)                    :: k

    LOGICAL                             :: is_inadmissible

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Construct exogenous variable for all states
    DO k = 0, (num_states - 1)

        payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

        CALL get_total_value(total_payoffs, &
                period, num_periods, delta, payoffs_systematic, shifts, &
                edu_max, edu_start, mapping_state_idx, periods_emax, k, &
                states_all)

        ! Treatment of inadmissible states, which will show up in the regression
        ! in some way
        is_inadmissible = (total_payoffs(3) == -HUGE_FLOAT)

        IF (is_inadmissible) THEN

            total_payoffs(3) = interpolation_inadmissible_states

        END IF

        ! Implement level shifts
        maxe(k + 1) = MAXVAL(total_payoffs)

        diff = maxe(k + 1) - total_payoffs

        ! Construct regressors
        independent_variables(k + 1, 1:4) = diff
        independent_variables(k + 1, 5:8) = DSQRT(diff)

    END DO

    ! Add intercept to set of independent variables
    independent_variables(:, 9) = one_dble

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_endogenous_variable(endogenous, period, num_periods, &
                num_states, delta, periods_payoffs_systematic, edu_max, &
                edu_start, mapping_state_idx, periods_emax, states_all, &
                is_simulated, num_draws_emax, maxe, draws_emax, &
                shocks_cholesky)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: endogenous(:)

    REAL(our_dble), INTENT(IN)          :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), INTENT(IN)          :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)          :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)          :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)          :: maxe(:)
    REAL(our_dble), INTENT(IN)          :: delta

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)        :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)        :: num_draws_emax
    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: num_states
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max
    INTEGER(our_int), INTENT(IN)        :: period

    LOGICAL, INTENT(IN)                 :: is_simulated(:)

    !/* internal objects        */

    REAL(our_dble)                      :: payoffs_systematic(4)
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
        CALL get_payoffs(emax_simulated, num_draws_emax, draws_emax, period, &
                k, payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
                states_all, num_periods, periods_emax, delta, shocks_cholesky)

        ! Construct dependent variable
        endogenous(k + 1) = emax_simulated - maxe(k + 1)

    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_predictions(predictions, endogenous, exogenous, maxe, &
              is_simulated, num_points, num_states)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)       :: predictions(:)

    REAL(our_dble), INTENT(IN)        :: exogenous(:, :)
    REAL(our_dble), INTENT(IN)        :: endogenous(:)
    REAL(our_dble), INTENT(IN)        :: maxe(:)

    INTEGER, INTENT(IN)               :: num_states
    INTEGER, INTENT(IN)               :: num_points

    LOGICAL, INTENT(IN)               :: is_simulated(:)

    !/* internal objects        */

    REAL(our_dble)                    :: endogenous_predicted_available(num_points)
    REAL(our_dble)                    :: exogenous_is_available(num_points, 9)
    REAL(our_dble)                    :: endogenous_is_available(num_points)
    REAL(our_dble)                    :: endogenous_predicted(num_states)
    REAL(our_dble)                    :: coeffs(9)
    REAL(our_dble)                    :: r_squared

    INTEGER(our_int)                  :: i
    INTEGER(our_int)                  :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Select pairs where both endogenous and exogenous information is available.
    i = 1

    DO k = 0, (num_states - 1)

        IF (is_simulated(k + 1)) THEN

            endogenous_is_available(i) = endogenous(k + 1)
            exogenous_is_available(i, :) = exogenous(k + 1, :)

            i = i + 1

        END IF

    END DO

    ! Fit the prediction model on interpolation points.
    CALL get_coefficients(coeffs, endogenous_is_available, &
            exogenous_is_available, 9, num_points)

    ! Use the model to predict EMAX for all states and subsequently
    ! replace the values where actual values are available. As in
    ! Keane & Wolpin (1994), negative predictions are truncated to zero.
    CALL point_predictions(endogenous_predicted, exogenous, coeffs, num_states)

    ! Construct coefficient of determination for the subset of interpolation
    ! points.
    i = 1

    DO k = 0, (num_states - 1)

        IF (is_simulated(k + 1)) THEN

            endogenous_predicted_available(i) = endogenous_predicted(k + 1)

            i = i + 1

        END IF

    END DO

    CALL get_r_squared(r_squared, endogenous_is_available, &
            endogenous_predicted_available, num_points)

    endogenous_predicted = clip_value(endogenous_predicted, &
            zero_dble, HUGE_FLOAT)

    ! Construct predicted EMAX for all states and the replace
    ! interpolation points with simulated values.
    predictions = endogenous_predicted + maxe

    DO k = 0, (num_states - 1)

        IF (is_simulated(k + 1)) THEN

            predictions(k + 1) = endogenous(k + 1) + maxe(k + 1)

        END IF

    END DO

    ! Perform some basic logging to spot problems early.
    CALL logging_prediction_model(coeffs, r_squared)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE logging_prediction_model(coeffs, r_squared)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: coeffs(:)
    REAL(our_dble), INTENT(IN)      :: r_squared

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Define format for coefficients and R-squared
    1900 FORMAT(2x,A18,3x,9(1x,f10.4))

    2900 FORMAT(2x,A18,4x,f10.4)

    ! Write to file
    OPEN(UNIT=99, FILE='logging.respy.sol.log', ACCESS='APPEND')

        WRITE(99, *) "     Information about Prediction Model"

        WRITE(99, *) " "

        WRITE(99, 1900) 'Coefficients', coeffs

        WRITE(99, *) " "

        WRITE(99, 2900) 'R-squared', r_squared

        WRITE(99, *) " "

        WRITE(99, *) " "

    CLOSE(99)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE logging_solution(indicator, period, num_states)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)            :: indicator

    INTEGER(our_int), INTENT(IN), OPTIONAL  :: num_states
    INTEGER(our_int), INTENT(IN), OPTIONAL  :: period

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    OPEN(UNIT=99, FILE='logging.respy.sol.log', ACCESS='APPEND')

    ! State space creation
    IF (indicator == 1) THEN

        ! Delete  any previous versions
        CLOSE(99, STATUS ='DELETE')
        OPEN(UNIT=99, FILE='logging.respy.sol.log', ACCESS='APPEND')

        WRITE(99, *) " Starting state space creation   "
        WRITE(99, *) ""

    ! Ex Ante Payoffs
    ELSEIF (indicator == 2) THEN

      WRITE(99, *) " Starting calculation of systematic payoffs  "
      WRITE(99, *) ""

    ! Backward induction procedure
    ELSEIF (indicator == 3) THEN

      WRITE(99, *) " Starting backward induction procedure "
      WRITE(99, *) ""

    ELSEIF (indicator == 4) THEN

        1900 FORMAT(2x,A18,1x,i2,1x,A4,1x,i5,1x,A6)

        WRITE(99, 1900) "... solving period", period, 'with', num_states, 'states'
        WRITE(99, *) ""

    ! Finishing
    ELSEIF (indicator == -1) THEN

        WRITE(99, *) " ... finished "
        WRITE(99, *) ""
        WRITE(99, *) ""

    END IF

  CLOSE(99)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE random_choice(sample, candidates, num_candidates, num_points)

  !
  ! Source
  ! ------
  !
  !   KNUTH, D. The art of computer programming. Vol II.
  !     Seminumerical Algorithms. Reading, Mass: AddisonWesley, 1969
  !

    !/* external objects    */

    INTEGER, INTENT(OUT)          :: sample(:)

    INTEGER, INTENT(IN)           :: num_candidates
    INTEGER, INTENT(IN)           :: candidates(:)
    INTEGER, INTENT(IN)           :: num_points

    !/* internal objects    */

    INTEGER(our_int)              :: m
    INTEGER(our_int)              :: j
    INTEGER(our_int)              :: l

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Initialize auxiliary objects
    m = 0

    ! Draw random points from candidates
    DO j = 1, num_candidates

        l = INT(DBLE(num_candidates - j + 1) * RAN(0)) + 1

        IF (l .GT. (num_points - m)) THEN

            CONTINUE

        ELSE

            m = m + 1

            sample(m) = candidates(j)

            IF (m .GE. num_points) THEN

                RETURN

            END IF

        END IF

    END DO


END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE svd(U, S, VT, A, m)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: VT(:, :)
    REAL(our_dble), INTENT(OUT)     :: U(:, :)
    REAL(our_dble), INTENT(OUT)     :: S(:)

    REAL(our_dble), INTENT(IN)      :: A(:, :)

    INTEGER(our_int), INTENT(IN)    :: m

    !/* internal objects        */

    INTEGER(our_int)                :: LWORK
    INTEGER(our_int)                :: INFO

    REAL(our_dble), ALLOCATABLE     :: IWORK(:)
    REAL(our_dble), ALLOCATABLE     :: WORK(:)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Auxiliary objects
    LWORK =  M * (7 + 4 * M)

    ! Allocate containers
    ALLOCATE(WORK(LWORK)); ALLOCATE(IWORK(8 * M))

    ! Call LAPACK routine
    CALL DGESDD( 'A', m, m, A, m, S, U, m, VT, m, WORK, LWORK, IWORK, INFO)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
FUNCTION pinv(A, m)

    !/* external objects        */

    REAL(our_dble)                  :: pinv(m, m)

    REAL(our_dble), INTENT(IN)      :: A(:, :)

    INTEGER(our_int), INTENT(IN)    :: m


    !/* internal objects        */

    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: VT(m, m)
    REAL(our_dble)                  :: UT(m, m)
    REAL(our_dble)                  :: U(m, m)
    REAL(our_dble)                  :: cutoff
    REAL(our_dble)                  :: S(m)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL svd(U, S, VT, A, m)

    cutoff = 1e-15_our_dble * MAXVAL(S)

    DO i = 1, M

        IF (S(i) .GT. cutoff) THEN

            S(i) = one_dble / S(i)

        ELSE

            S(i) = zero_dble

        END IF

    END DO

    UT = TRANSPOSE(U)

    DO i = 1, M

        pinv(i, :) = S(i) * UT(i,:)

    END DO

    pinv = MATMUL(TRANSPOSE(VT), pinv)

END FUNCTION
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_coefficients(coeffs, Y, X, num_covars, num_states)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: coeffs(:)

    INTEGER, INTENT(IN)             :: num_covars
    INTEGER, INTENT(IN)             :: num_states

    REAL(our_dble), INTENT(IN)      :: X(:, :)
    REAL(our_dble), INTENT(IN)      :: Y(:)

    !/* internal objects        */

    REAL(our_dble)                  :: A(num_covars, num_covars)
    REAL(our_dble)                  :: C(num_covars, num_covars)
    REAL(our_dble)                  :: D(num_covars, num_states)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

   A = MATMUL(TRANSPOSE(X), X)

   C =  pinv(A, num_covars)

   D = MATMUL(C, TRANSPOSE(X))

   coeffs = MATMUL(D, Y)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE point_predictions(Y, X, coeffs, num_states)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: Y(:)

    REAL(our_dble), INTENT(IN)      :: coeffs(:)
    REAL(our_dble), INTENT(IN)      :: X(:, :)

    INTEGER(our_int), INTENT(IN)    :: num_states

    !/* internal objects        */

    INTEGER(our_int)                 :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    DO i = 1, num_states

        Y(i) = DOT_PRODUCT(coeffs, X(i, :))

    END DO

END SUBROUTINE

!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_r_squared(r_squared, observed, predicted, num_states)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: r_squared

    REAL(our_dble), INTENT(IN)      :: predicted(:)
    REAL(our_dble), INTENT(IN)      :: observed(:)

    INTEGER(our_int), INTENT(IN)    :: num_states

    !/* internal objects        */

    REAL(our_dble)                  :: mean_observed
    REAL(our_dble)                  :: ss_residuals
    REAL(our_dble)                  :: ss_total

    INTEGER(our_int)                :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Calculate mean of observed data
    mean_observed = SUM(observed) / DBLE(num_states)

    ! Sum of squared residuals
    ss_residuals = zero_dble

    DO i = 1, num_states

        ss_residuals = ss_residuals + (observed(i) - predicted(i))**2

    END DO

    ! Sum of squared residuals
    ss_total = zero_dble

    DO i = 1, num_states

        ss_total = ss_total + (observed(i) - mean_observed)**2

    END DO

    ! Turning off all randomness during testing requires special case to avoid 
    ! an error due to the division by zero. 
    IF (ss_residuals .EQ. zero_dble) THEN
        r_squared = one_dble
    ELSE
        r_squared = one_dble - ss_residuals / ss_total
    END IF
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE simulate_emax(emax_simulated, num_periods, num_draws_emax, period, & 
                k, draws_emax, payoffs_systematic, edu_max, edu_start, & 
                periods_emax, states_all, mapping_state_idx, delta, & 
                shocks_cholesky)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: emax_simulated

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: draws_emax(:, :)

    REAL(our_dble), INTENT(IN)      :: delta

    !/* internals objects    */

    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: draws_emax_transformed(num_draws_emax, 4)

    REAL(our_dble)                  :: total_payoffs(4)
    REAL(our_dble)                  :: draws(4)
    REAL(our_dble)                  :: maximum

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Transform disturbances
    CALL transform_disturbances(draws_emax_transformed, draws_emax, &
            shocks_cholesky, num_draws_emax)

    ! Iterate over Monte Carlo draws
    emax_simulated = zero_dble
    DO i = 1, num_draws_emax

        ! Select draws for this draw
        draws = draws_emax_transformed(i, :)

        ! Calculate total value
        CALL get_total_value(total_payoffs, period, num_periods, delta, &
                payoffs_systematic, draws, edu_max, edu_start, & 
                mapping_state_idx, periods_emax, k, states_all)
   
        ! Determine optimal choice
        maximum = MAXVAL(total_payoffs)

        ! Recording expected future value
        emax_simulated = emax_simulated + maximum

    END DO

    ! Scaling
    emax_simulated = emax_simulated / num_draws_emax

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE