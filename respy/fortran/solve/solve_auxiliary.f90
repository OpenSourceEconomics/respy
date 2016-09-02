!******************************************************************************
!******************************************************************************
MODULE solve_auxiliary

    !/*	external modules	*/

    USE recording_solution

    USE shared_auxiliary

    USE shared_constants

    USE shared_utilities

    USE solve_ambiguity

    USE solve_risk

    !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_create_state_space(states_all, states_number_period, mapping_state_idx, num_periods, edu_start, edu_max, min_idx)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)     :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)     :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)     :: states_all(:, :, :)

    INTEGER(our_int), INTENT(IN)                    :: num_periods
    INTEGER(our_int), INTENT(IN)                    :: edu_start
    INTEGER(our_int), INTENT(IN)                    :: edu_max
    INTEGER(our_int), INTENT(IN)                    :: min_idx

    !/* internals objects       */

    INTEGER(our_int)                    :: states_all_tmp(num_periods, 100000, 4)
    INTEGER(our_int)                    :: edu_lagged
    INTEGER(our_int)                    :: period
    INTEGER(our_int)                    :: total
    INTEGER(our_int)                    :: exp_a
    INTEGER(our_int)                    :: exp_b
    INTEGER(our_int)                    :: edu
    INTEGER(our_int)                    :: k

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Allocate containers that contain information about the model structure
    ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
    ALLOCATE(states_number_period(num_periods))

    ! Initialize output
    states_number_period = MISSING_INT
    mapping_state_idx    = MISSING_INT
    states_all_tmp       = MISSING_INT

    ! Construct state space by periods
    DO period = 0, (num_periods - 1)

        ! Count admissible realizations of state space by period
        k = 0

        ! Loop over all admissible work experiences for Occupation A
        DO exp_a = 0, num_periods

            ! Loop over all admissible work experience for Occupation B
            DO exp_b = 0, num_periods

                ! Loop over all admissible additional education levels
                DO edu = 0, num_periods

                    ! Agent cannot attain more additional education than (EDU_MAX - EDU_START).
                    IF (edu .GT. edu_max - edu_start) THEN
                        CYCLE
                    END IF

                    ! Loop over all admissible values for leisure. Note that the leisure variable takes only zero/value. The time path does not matter.
                    DO edu_lagged = 0, 1

                        ! Check if lagged education admissible. (1) In the first period all agents have lagged schooling equal to one.
                        IF (edu_lagged .EQ. zero_int) THEN
                            IF (period .EQ. zero_int) THEN
                                CYCLE
                            END IF
                        END IF

                        ! (2) Whenever an agent has not acquired any additional education and we are not in the first period, then this cannot be the case.
                        IF (edu_lagged .EQ. one_int) THEN
                            IF (edu .EQ. zero_int) THEN
                                IF (period .GT. zero_int) THEN
                                    CYCLE
                                END IF
                            END IF
                        END IF

                        ! (3) Whenever an agent has only acquired additional education, then edu_lagged cannot be zero.
                        IF (edu_lagged .EQ. zero_int) THEN
                            IF (edu .EQ. period) THEN
                                CYCLE
                            END IF
                        END IF

                        ! Check if admissible for time constraints
                        total = edu + exp_a + exp_b

                        ! Note that the total number of activities does not have is less or equal to the total possible number of activities as the rest is implicitly filled with leisure.
                        IF (total .GT. period) THEN
                            CYCLE
                        END IF

                        ! Collect all possible realizations of state space
                        states_all_tmp(period + 1, k + 1, 1) = exp_a
                        states_all_tmp(period + 1, k + 1, 2) = exp_b
                        states_all_tmp(period + 1, k + 1, 3) = edu
                        states_all_tmp(period + 1, k + 1, 4) = edu_lagged

                        ! Collect mapping of state space to array index.
                        mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu + 1 , edu_lagged + 1) = k

                        ! Update count
                        k = k + 1

                     END DO

                 END DO

             END DO

         END DO

        ! Record maximum number of state space realizations by time period
        states_number_period(period + 1) = k

    END DO

    ! Auxiliary object
    max_states_period = MAXVAL(states_number_period)

    ! Initialize a host of containers, whose dimensions are not clear.
    ALLOCATE(states_all(num_periods, max_states_period, 4))
    states_all = states_all_tmp(:, :max_states_period, :)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, edu_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, max_states_period)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)       :: periods_rewards_systematic(: ,:, :)

    REAL(our_dble), INTENT(IN)          :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)          :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)          :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)          :: coeffs_b(6)

    INTEGER(our_int), INTENT(IN)        :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)        :: states_number_period(num_periods)
    INTEGER(our_int), INTENT(IN)        :: max_states_period
    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: edu_start

    !/* internals objects       */

    INTEGER(our_int)                    :: edu_lagged
    INTEGER(our_int)                    :: covars(6)
    INTEGER(our_int)                    :: period
    INTEGER(our_int)                    :: exp_a
    INTEGER(our_int)                    :: exp_b
    INTEGER(our_int)                    :: info
    INTEGER(our_int)                    :: edu
    INTEGER(our_int)                    :: k

    REAL(our_dble)                      :: reward

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! ALlocate container (if required) and initilaize missing values.
    IF (.NOT. ALLOCATED(periods_rewards_systematic)) THEN
        ALLOCATE(periods_rewards_systematic(num_periods, max_states_period, 4))
    END IF
    periods_rewards_systematic = MISSING_FLOAT

    ! Calculate systematic instantaneous rewards
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

            ! Calculate systematic part of reward in Occupation A
            CALL clip_value(periods_rewards_systematic(period, k, 1), EXP(DOT_PRODUCT(covars, coeffs_a)), zero_dble, HUGE_FLOAT, info)

            ! Calculate systematic part of reward in Occupation B
            CALL clip_value(periods_rewards_systematic(period, k, 2), EXP(DOT_PRODUCT(covars, coeffs_b)), zero_dble, HUGE_FLOAT, info)

            ! Calculate systematic part of schooling utility
            reward = coeffs_edu(1)

            ! Tuition cost for higher education if agents move beyond high school.
            IF(edu + edu_start >= 12) reward = reward + coeffs_edu(2)

            ! Psychic cost of going back to school
            IF(edu_lagged == 0) reward = reward + coeffs_edu(3)

            periods_rewards_systematic(period, k, 3) = reward

            ! Calculate systematic part of reward in home production
            periods_rewards_systematic(period, k, 4) = coeffs_home(1)

        END DO

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_backward_induction(periods_emax, num_periods, is_myopic, max_states_period, periods_draws_emax, num_draws_emax, states_number_period, periods_rewards_systematic, edu_max, edu_start, mapping_state_idx, states_all, delta, is_debug, is_interpolated, num_points_interp, shocks_cholesky, is_ambiguity, measure, level, optimizer_options, is_write)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)       :: periods_emax(:, :)

    REAL(our_dble), INTENT(IN)          :: periods_rewards_systematic(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(IN)          :: periods_draws_emax(num_periods, num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)          :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)          :: delta
    REAL(our_dble), INTENT(IN)          :: level

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)        :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)        :: states_number_period(num_periods)
    INTEGER(our_int), INTENT(IN)        :: max_states_period
    INTEGER(our_int), INTENT(IN)        :: num_points_interp
    INTEGER(our_int), INTENT(IN)        :: num_draws_emax
    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max

    LOGICAL, INTENT(IN)                 :: is_interpolated
    LOGICAL, INTENT(IN)                 :: is_ambiguity
    LOGICAL, INTENT(IN)                 :: is_myopic
    LOGICAL, INTENT(IN)                 :: is_debug
    LOGICAL, INTENT(IN)                 :: is_write

    CHARACTER(10), INTENT(IN)           :: measure

    TYPE(optimizer_collection), INTENT(IN)  :: optimizer_options

    !/* internals objects       */

    INTEGER(our_int)                    :: num_states
    INTEGER(our_int)                    :: period
    INTEGER(our_int)                    :: info
    INTEGER(our_int)                    :: k

    REAL(our_dble)                      :: draws_emax_transformed(num_draws_emax, 4)
    REAL(our_dble)                      :: draws_emax(num_draws_emax, 4)
    REAL(our_dble)                      :: rewards_systematic(4)
    REAL(our_dble)                      :: shocks_cov(4, 4)
    REAL(our_dble)                      :: shifts(4)
    REAL(our_dble)                      :: emax

    REAL(our_dble), ALLOCATABLE         :: exogenous(:, :)
    REAL(our_dble), ALLOCATABLE         :: predictions(:)
    REAL(our_dble), ALLOCATABLE         :: endogenous(:)
    REAL(our_dble), ALLOCATABLE         :: maxe(:)

    LOGICAL                             :: any_interpolated

    LOGICAL, ALLOCATABLE                :: is_simulated(:)
    INTEGER(our_int)                    :: seed_inflated(15)
    INTEGER(our_int)                    :: seed_size

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! ALlocate container (if required) and initialize missing values.
    IF (.NOT. ALLOCATED(periods_emax)) ALLOCATE(periods_emax(num_periods, max_states_period))

    periods_emax = MISSING_FLOAT

    IF (is_myopic) THEN
        DO period = (num_periods - 1), 0, -1
            num_states = states_number_period(period + 1)
            periods_emax(period + 1, :num_states) = zero_dble
        END DO
        RETURN
    END IF

    seed_inflated(:) = 123

    CALL RANDOM_SEED(size=seed_size)

    CALL RANDOM_SEED(put=seed_inflated)

    shocks_cov = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))

    shifts = zero_dble
    CALL clip_value(shifts(1), EXP(shocks_cov(1, 1)/two_dble), zero_dble, HUGE_FLOAT, info)
    CALL clip_value(shifts(2), EXP(shocks_cov(2, 2)/two_dble), zero_dble, HUGE_FLOAT, info)

    DO period = (num_periods - 1), 0, -1

        draws_emax = periods_draws_emax(period + 1, :, :)
        num_states = states_number_period(period + 1)

        ! Transform disturbances
        CALL transform_disturbances(draws_emax_transformed, draws_emax, shocks_cholesky, num_draws_emax)


        IF (is_write) CALL record_solution(4, period, num_states)

        any_interpolated = (num_points_interp .LE. num_states) .AND. is_interpolated

        IF (any_interpolated) THEN

            ALLOCATE(is_simulated(num_states), endogenous(num_states), maxe(num_states), exogenous(num_states, 9), predictions(num_states))

            is_simulated = get_simulated_indicator(num_points_interp, num_states, period, is_debug)

            CALL get_exogenous_variables(exogenous, maxe, period, num_states, periods_rewards_systematic, shifts, mapping_state_idx, periods_emax, states_all, delta, edu_start, edu_max)

            CALL get_endogenous_variable(endogenous, period, num_states, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, is_simulated, maxe, draws_emax_transformed, delta, edu_start, edu_max, shocks_cov, is_ambiguity, measure, level, optimizer_options, is_write)

            CALL get_predictions(predictions, endogenous, exogenous, maxe, is_simulated, num_states, is_write)

            periods_emax(period + 1, :num_states) = predictions

            DEALLOCATE(is_simulated, exogenous, maxe, endogenous, predictions)

        ELSE

            DO k = 0, (states_number_period(period + 1) - 1)

                rewards_systematic = periods_rewards_systematic(period + 1, k + 1, :)

                IF (is_ambiguity) THEN
                    CALL construct_emax_ambiguity(emax, num_periods, num_draws_emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta, shocks_cov, measure, level, optimizer_options, is_write)
                ELSE
                    CALL construct_emax_risk(emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta)
                END IF

                periods_emax(period + 1, k + 1) = emax

            END DO

        END IF

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION get_simulated_indicator(num_points, num_states, period, is_debug)

    !/* external objects        */

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

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize set of candidate values
    candidates = (/ (i, i = 1, num_states) /)

    ! Check if standardization requested
    IF (is_debug) THEN

        INQUIRE(FILE='interpolation.txt', EXIST=READ_IN)

        IF (READ_IN) THEN

            OPEN(UNIT=99, FILE='interpolation.txt', ACTION='READ')

               DO i = 1, num_states

                    READ(99, *)  is_simulated_container(i, :)

                END DO

            CLOSE(99)

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
!******************************************************************************
!******************************************************************************
SUBROUTINE get_exogenous_variables(independent_variables, maxe, period, num_states, periods_rewards_systematic, shifts, mapping_state_idx, periods_emax, states_all, delta, edu_start, edu_max)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: independent_variables(num_states, 9)
    REAL(our_dble), INTENT(OUT)         :: maxe(num_states)

    REAL(our_dble), INTENT(IN)          :: periods_rewards_systematic(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(IN)          :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)          :: shifts(4)
    REAL(our_dble), INTENT(IN)          :: delta

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)        :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)        :: num_states
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max
    INTEGER(our_int), INTENT(IN)        :: period

    !/* internal objects        */

    REAL(our_dble)                      :: rewards_systematic(4)
    REAL(our_dble)                      :: total_values(4)
    REAL(our_dble)                      :: diff(4)

    INTEGER(our_int)                    :: k

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Construct exogenous variable for all states
    DO k = 0, (num_states - 1)

        rewards_systematic = periods_rewards_systematic(period + 1, k + 1, :)

        CALL get_total_values(total_values, period, num_periods, rewards_systematic, shifts, mapping_state_idx, periods_emax, k, states_all, delta, edu_start, edu_max)

        ! Implement level shifts
        maxe(k + 1) = MAXVAL(total_values)

        diff = maxe(k + 1) - total_values

        ! Construct regressors
        independent_variables(k + 1, 1:4) = diff
        independent_variables(k + 1, 5:8) = DSQRT(diff)

    END DO

    ! Add intercept to set of independent variables
    independent_variables(:, 9) = one_dble

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_endogenous_variable(endogenous, period, num_states, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, is_simulated, maxe, draws_emax_transformed, delta, edu_start, edu_max, shocks_cov, is_ambiguity, measure, level, optimizer_options, is_write)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: endogenous(num_states)

    REAL(our_dble), INTENT(IN)          :: periods_rewards_systematic(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(IN)          :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)          :: draws_emax_transformed(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)          :: maxe(num_states)
    REAL(our_dble), INTENT(IN)          :: shocks_cov(4, 4)
    REAL(our_dble), INTENT(IN)          :: delta
    REAL(our_dble), INTENT(IN)          :: level

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)        :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)        :: num_states
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max
    INTEGER(our_int), INTENT(IN)        :: period

    LOGICAL, INTENT(IN)                 :: is_simulated(num_states)
    LOGICAL, INTENT(IN)                 :: is_ambiguity
    LOGICAL, INTENT(IN)                 :: is_write

    CHARACTER(10), INTENT(IN)           :: measure

    TYPE(optimizer_collection), INTENT(IN)  :: optimizer_options

    !/* internal objects        */

    REAL(our_dble)                      :: rewards_systematic(4)
    REAL(our_dble)                      :: emax

    INTEGER(our_int)                    :: k

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize missing values
    endogenous = MISSING_FLOAT

    ! Construct dependent variables for the subset of interpolation
    ! points.
    DO k = 0, (num_states - 1)

        ! Skip over points that will be predicted
        IF (.NOT. is_simulated(k + 1)) THEN
            CYCLE
        END IF

        rewards_systematic = periods_rewards_systematic(period + 1, k + 1, :)

        IF (is_ambiguity) THEN
            CALL construct_emax_ambiguity(emax, num_periods, num_draws_emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta, shocks_cov, measure, level, optimizer_options, is_write)
        ELSE
            CALL construct_emax_risk(emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta)
        END IF

        ! Construct dependent variable
        endogenous(k + 1) = emax - maxe(k + 1)

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_predictions(predictions, endogenous, exogenous, maxe, is_simulated, num_states, is_write)

    ! DEVELOPMENT NOTE
    !
    !   The exogenous array remains assumed shape for now due to the
    !   modification for the STRUCT_RECOMPUTATION project.
    !

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: predictions(num_states)

    REAL(our_dble), INTENT(IN)          :: exogenous(:, :)
    REAL(our_dble), INTENT(IN)          :: endogenous(num_states)
    REAL(our_dble), INTENT(IN)          :: maxe(num_states)

    INTEGER, INTENT(IN)                 :: num_states

    LOGICAL, INTENT(IN)                 :: is_simulated(num_states)
    LOGICAL, INTENT(IN)                 :: is_write


    !/* internal objects        */

    INTEGER(our_int), ALLOCATABLE       :: infos(:)

    REAL(our_dble)                      :: endogenous_predicted_available(num_points_interp)
    REAL(our_dble)                      :: exogenous_is_available(num_points_interp, 9)
    REAL(our_dble)                      :: endogenous_is_available(num_points_interp)
    REAL(our_dble)                      :: endogenous_predicted(num_states)
    REAL(our_dble)                      :: endogenous_predicted_clipped(num_states)
    REAL(our_dble)                      :: coeffs(9)
    REAL(our_dble)                      :: r_squared
    REAL(our_dble)                      :: bse(9)

    INTEGER(our_int)                    :: i
    INTEGER(our_int)                    :: k

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

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
    CALL get_coefficients(coeffs, endogenous_is_available, exogenous_is_available, 9, num_points_interp)

    ! Use the model to predict EMAX for all states and subsequently replace the values where actual values are available. As in Keane & Wolpin (1994), negative predictions are truncated to zero.
    CALL point_predictions(endogenous_predicted, exogenous, coeffs, num_states)

    ! Construct coefficient of determination for the subset of interpolation points.
    i = 1

    DO k = 0, (num_states - 1)

        IF (is_simulated(k + 1)) THEN

            endogenous_predicted_available(i) = endogenous_predicted(k + 1)

            i = i + 1

        END IF

    END DO

    CALL get_pred_info(r_squared, bse, endogenous_is_available, endogenous_predicted_available, exogenous_is_available, num_points_interp, 9)

    CALL clip_value(endogenous_predicted_clipped, endogenous_predicted, zero_dble, HUGE_FLOAT, infos)

    ! Construct predicted EMAX for all states and the replace interpolation points with simulated values.
    predictions = endogenous_predicted_clipped + maxe

    DO k = 0, (num_states - 1)

        IF (is_simulated(k + 1)) THEN

            predictions(k + 1) = endogenous(k + 1) + maxe(k + 1)

        END IF

    END DO

    ! Perform some basic logging to spot problems early.
    IF(is_write) THEN
        CALL record_solution(coeffs, r_squared, bse)
    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
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


    ! Temporary clutter for the structRecomputation project.
    REAL(our_dble)                  :: A_sub(8, 8)
    REAL(our_dble)                  :: C_sub(8, 8)
    REAL(our_dble)                  :: D_sub(8, num_states)

    REAL(our_dble)                  :: X_sub(num_states, 8)
    REAL(our_dble)                  :: coeffs_sub(8)

    LOGICAL                         :: IS_TEMPORARY

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------


    ! This temporary modification allows to run the more restricted interpolation model for the structRecomputation project.
    INQUIRE(FILE='.structRecomputation.tmp', EXIST=IS_TEMPORARY)

    IF(IS_TEMPORARY) THEN

        X_sub(:, :2) = X(:, :2)
        X_sub(:, 3:) = X(:, 4:)

        A_sub = MATMUL(TRANSPOSE(X_sub), X_sub)

        C_sub =  pinv(A_sub, num_covars - 1)

        D_sub = MATMUL(C_sub, TRANSPOSE(X_sub))

        coeffs_sub = MATMUL(D_sub, Y)

        coeffs(:2) = coeffs_sub(:2)
        coeffs(3) = zero_dble
        coeffs(4:) = coeffs_sub(3:)

    ELSE

        A = MATMUL(TRANSPOSE(X), X)

        C =  pinv(A, num_covars)

        D = MATMUL(C, TRANSPOSE(X))

        coeffs = MATMUL(D, Y)

    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE point_predictions(Y, X, coeffs, num_states)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: Y(:)

    REAL(our_dble), INTENT(IN)      :: coeffs(:)
    REAL(our_dble), INTENT(IN)      :: X(:, :)

    INTEGER(our_int), INTENT(IN)    :: num_states

    !/* internal objects        */

    INTEGER(our_int)                 :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    DO i = 1, num_states

        Y(i) = DOT_PRODUCT(coeffs, X(i, :))

    END DO

END SUBROUTINE

!******************************************************************************
!******************************************************************************
SUBROUTINE get_pred_info(r_squared, bse, observed, predicted, exogenous, num_states, num_covars)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: r_squared
    REAL(our_dble), INTENT(OUT)     :: bse(num_covars)

    REAL(our_dble), INTENT(IN)      :: predicted(:)
    REAL(our_dble), INTENT(IN)      :: observed(:)
    REAL(our_dble), INTENT(IN)      :: exogenous(:, :)

    INTEGER(our_int), INTENT(IN)    :: num_states
    INTEGER(our_int), INTENT(IN)    :: num_covars

    !/* internal objects        */

    REAL(our_dble)                  :: residuals(num_states)
    REAL(our_dble)                  :: mean_observed
    REAL(our_dble)                  :: ss_residuals
    REAL(our_dble)                  :: cova(num_covars, num_covars)
    REAL(our_dble)                  :: ss_total
    REAL(our_dble)                  :: sigma

    INTEGER(our_int)                :: info
    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Calculate mean of observed data
    mean_observed = SUM(observed) / DBLE(num_states)

    ! Get array with residuals
    DO i = 1, num_states
        residuals(i) = observed(i) - predicted(i)
    END DO

    ! Construct standard errors
    sigma = (one_dble / (num_states - num_covars)) * DOT_PRODUCT(residuals, residuals)
    cova = sigma * pinv(MATMUL(TRANSPOSE(exogenous), exogenous), num_covars)
    DO i = 1, num_covars
        CALL clip_value(bse(i), cova(i, i), TINY_FLOAT, HUGE_FLOAT, info)
        bse(i) = SQRT(bse(i))
    END DO

    ! Sum of squared residuals
    ss_residuals = SUM(residuals ** 2)

    ! Total sum of squares
    ss_total = zero_dble

    DO i = 1, num_states
        ss_total = ss_total + (observed(i) - mean_observed)**2
    END DO

    ! Turning off all randomness during testing requires special case to avoid an error due to the division by zero.
    IF (ss_residuals .EQ. zero_dble) THEN
        r_squared = one_dble
    ELSE
        r_squared = one_dble - ss_residuals / ss_total
    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
