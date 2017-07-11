!******************************************************************************
!******************************************************************************
MODULE solve_auxiliary

    !/*	external modules	*/

    USE recording_solution

    USE shared_interface

    USE solve_ambiguity

    USE solve_risk

    !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_create_state_space(states_all, states_number_period, mapping_state_idx, num_periods, num_types, edu_spec)

    !/* external objects        */

    TYPE(EDU_DICT), INTENT(IN)                       :: edu_spec

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)     :: mapping_state_idx(:, :, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)     :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)     :: states_all(:, :, :)

    INTEGER(our_int), INTENT(IN)                    :: num_periods
    INTEGER(our_int), INTENT(IN)                    :: num_types

    !/* internals objects       */

    INTEGER(our_int)                    :: states_all_tmp(num_periods, 1000000, 5)
    INTEGER(our_int)                    :: activity_lagged
    INTEGER(our_int)                    :: num_edu_start
    INTEGER(our_int)                    :: edu_start
    INTEGER(our_int)                    :: edu_add
    INTEGER(our_int)                    :: min_idx
    INTEGER(our_int)                    :: period
    INTEGER(our_int)                    :: type_
    INTEGER(our_int)                    :: exp_a
    INTEGER(our_int)                    :: exp_b
    INTEGER(our_int)                    :: k
    INTEGER(our_int)                    :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Auxiliary variables
    num_edu_start = SIZE(edu_spec%start)
    min_idx = edu_spec%max + 1

    ! Allocate containers that contain information about the model structure
    ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 4    , num_types))
    ALLOCATE(states_number_period(num_periods))

    ! Initialize output
    states_number_period = MISSING_INT
    mapping_state_idx    = MISSING_INT
    states_all_tmp       = MISSING_INT

    ! Construct state space by periods
    DO period = 0, (num_periods - 1)

        ! Count admissible realizations of state space by period
        k = 0

        ! Loop over all types.
        DO type_ = 0, num_types - 1

            ! Loop over all initial level of schooling
            DO j = 1, num_edu_start

                edu_start = edu_spec%start(j)

                ! Loop over all admissible work experiences for Occupation A
                DO exp_a = 0, num_periods

                    ! Loop over all admissible work experience for Occupation B
                    DO exp_b = 0, num_periods

                        ! Loop over all admissible additional education levels
                        DO edu_add = 0, num_periods

                            ! Note that the total number of activities does not have is less or equal to the total possible number of activities as the rest is implicitly filled with leisure.
                            IF (edu_add + exp_a + exp_b .GT. period) CYCLE

                            ! Agent cannot attain more additional education than (EDU_MAX - EDU_START).
                            IF (edu_add .GT. (edu_spec%max - edu_start)) CYCLE

                            ! Loop over all admissible values for the lagged activity: (0) Home, (1) Education, (2) Occupation A, and (3) Occupation B.
                            DO activity_lagged = 0, 3

                                ! (0, 1) Whenever an agent has only acquired additional education, then activity_lagged cannot be zero.
                                IF ((activity_lagged .EQ. zero_int) .AND. (edu_add .EQ. period)) CYCLE

                                ! (0, 2) Whenever an agent has only worked in Occupation A, then activity_lagged cannot be zero.
                                IF ((activity_lagged .EQ. zero_int) .AND. (exp_a .EQ. period)) CYCLE

                                ! (0, 3) Whenever an agent has only worked in Occupation B, then activity_lagged cannot be zero.
                                IF ((activity_lagged .EQ. zero_int) .AND. (exp_b .EQ. period)) CYCLE

                                ! (1, 1) In the first period all agents have lagged schooling equal to one.
                                IF ((activity_lagged .NE. one_int) .AND. (period .EQ. zero_int)) CYCLE

                                ! (1, 2) Whenever an agent has not acquired any additional education and we are not in the first period, then lagged education cannot take a value of one.
                                IF ((activity_lagged .EQ. one_int) .AND. (edu_add .EQ. zero_int) .AND. (period .GT. zero_int)) CYCLE

                                ! (2, 1) An individual that has never worked in Occupation A cannot have a that lagged activity.
                                IF ((activity_lagged .EQ. two_int) .AND. (exp_a .EQ. zero_int)) CYCLE

                                ! (3, 1) An individual that has never worked in Occupation B cannot have a that lagged activity.
                                IF ((activity_lagged .EQ. three_int) .AND. (exp_b .EQ. zero_int)) CYCLE

                                ! If we have multiple initial conditions it might well be the case that we have a duplicate state, i.e. the same state is possible with other initial condition that period.
                                IF (mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu_start + edu_add + 1 , activity_lagged + 1, type_ + 1) .NE. MISSING_INT) CYCLE

                                ! Collect mapping of state space to array index.
                                mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu_start + edu_add + 1 , activity_lagged + 1, type_ + 1) = k

                                ! Collect all possible realizations of state space
                                states_all_tmp(period + 1, k + 1, :) = (/ exp_a, exp_b, edu_start + edu_add, activity_lagged, type_ /)

                                ! Update count
                                k = k + 1

                             END DO

                         END DO

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
    ALLOCATE(states_all(num_periods, max_states_period, 5))
    states_all = states_all_tmp(:, :max_states_period, :)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, max_states_period, optim_paras)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_rewards_systematic(: ,:, :)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)               :: optim_paras

    INTEGER(our_int), INTENT(IN)        :: states_all(num_periods, max_states_period, 5)
    INTEGER(our_int), INTENT(IN)        :: states_number_period(num_periods)
    INTEGER(our_int), INTENT(IN)        :: max_states_period
    INTEGER(our_int), INTENT(IN)        :: num_periods

    !/* internals objects       */

    TYPE(COVARIATES_DICT)               :: covariates

    INTEGER(our_int)                    :: activity_lagged

    INTEGER(our_int)                    :: covars_home(3)
    INTEGER(our_int)                    :: covars_edu(5)
    INTEGER(our_int)                    :: period
    INTEGER(our_int)                    :: type_
    INTEGER(our_int)                    :: exp_a
    INTEGER(our_int)                    :: exp_b
    INTEGER(our_int)                    :: edu
    INTEGER(our_int)                    :: k
    INTEGER(our_int)                    :: i

    REAL(our_dble)                      :: rewards_general(2)
    REAL(our_dble)                      :: rewards_common
    REAL(our_dble)                      :: rewards(4)
    REAL(our_dble)                      :: wages(2)

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
            activity_lagged = states_all(period, k, 4)
            type_ = states_all(period, k, 5)

            ! Construct auxiliary information
            covariates = construct_covariates(exp_a, exp_b, edu, activity_lagged, type_, period)

            ! Calculate common and general rewards component.
            rewards_general = calculate_rewards_general(covariates, optim_paras)
            rewards_common = calculate_rewards_common(covariates, optim_paras)

            ! Calculate the systematic part of OCCUPATION A and OCCUPATION B rewards. these are defined in a general sense, where not only wages matter.
            wages = calculate_wages_systematic(covariates, optim_paras)

            DO i = 1, 2
                rewards(i) = wages(i) + rewards_general(i)
            END DO

            ! Calculate systematic part of schooling utility
            covars_edu(1) = one_int
            covars_edu(2) = covariates%is_return_not_high_school
            covars_edu(3) = covariates%is_return_high_school
            covars_edu(4) = covariates%period - one_dble
            covars_edu(5) = covariates%is_minor

            rewards(3) = DOT_PRODUCT(covars_edu, optim_paras%coeffs_edu)

            ! Calculate systematic part of HOME
            covars_home(1) = one_int
            covars_home(2) = covariates%is_young_adult
            covars_home(3) = covariates%is_adult

            rewards(4) = DOT_PRODUCT(covars_home, optim_paras%coeffs_home)

            ! Now we add the type-specific deviation.
            DO i = 3, 4
                rewards(i) = rewards(i) + optim_paras%type_shifts(type_ + 1, i)
            END DO

            ! We can now also added the common component of rewards.
            DO i = 1, 4
                rewards(i) = rewards(i) + rewards_common
            END DO

            periods_rewards_systematic(period, k, :) = rewards

        END DO

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_backward_induction(periods_emax, opt_ambi_details, num_periods, is_myopic, max_states_period, periods_draws_emax, num_draws_emax, states_number_period, periods_rewards_systematic, mapping_state_idx, states_all, is_debug, is_interpolated, num_points_interp, edu_spec, ambi_spec, optim_paras, optimizer_options, file_sim, is_write)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)          :: opt_ambi_details(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)          :: periods_emax(:, :)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)           :: optim_paras
    TYPE(AMBI_DICT), INTENT(IN)                 :: ambi_spec
    TYPE(EDU_DICT), INTENT(IN)                  :: edu_spec

    REAL(our_dble), INTENT(IN)          :: periods_rewards_systematic(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(IN)          :: periods_draws_emax(num_periods, num_draws_emax, 4)

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 4, num_types)
    INTEGER(our_int), INTENT(IN)        :: states_all(num_periods, max_states_period, 5)
    INTEGER(our_int), INTENT(IN)        :: states_number_period(num_periods)
    INTEGER(our_int), INTENT(IN)        :: max_states_period
    INTEGER(our_int), INTENT(IN)        :: num_points_interp
    INTEGER(our_int), INTENT(IN)        :: num_draws_emax
    INTEGER(our_int), INTENT(IN)        :: num_periods

    LOGICAL, INTENT(IN)                 :: is_interpolated
    LOGICAL, INTENT(IN)                 :: is_myopic
    LOGICAL, INTENT(IN)                 :: is_debug
    LOGICAL, INTENT(IN)                 :: is_write

    CHARACTER(225), INTENT(IN)          :: file_sim

    TYPE(OPTIMIZER_COLLECTION), INTENT(IN)  :: optimizer_options

    !/* internals objects       */

    INTEGER(our_int)                    :: num_states
    INTEGER(our_int)                    :: seed_inflated(15)
    INTEGER(our_int)                    :: seed_size
    INTEGER(our_int)                    :: period
    INTEGER(our_int)                    :: info
    INTEGER(our_int)                    :: k
    INTEGER(our_int)                    :: i

    REAL(our_dble)                      :: draws_emax_ambiguity_transformed(num_draws_emax, 4)
    REAL(our_dble)                      :: draws_emax_ambiguity_standard(num_draws_emax, 4)
    REAL(our_dble)                      :: draws_emax_standard(num_draws_emax, 4)
    REAL(our_dble)                      :: draws_emax_risk(num_draws_emax, 4)
    REAL(our_dble)                      :: shocks_mean(4) = zero_dble
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

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! ALlocate container (if required) and initialize missing values.
    IF (.NOT. ALLOCATED(opt_ambi_details)) ALLOCATE(opt_ambi_details(num_periods, max_states_period, 7))
    IF (.NOT. ALLOCATED(periods_emax)) ALLOCATE(periods_emax(num_periods, max_states_period))

    opt_ambi_details = MISSING_FLOAT
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

    shocks_cov = MATMUL(optim_paras%shocks_cholesky, TRANSPOSE(optim_paras%shocks_cholesky))

    shifts = zero_dble
    CALL clip_value(shifts(1), EXP(shocks_cov(1, 1)/two_dble), zero_dble, HUGE_FLOAT, info)
    CALL clip_value(shifts(2), EXP(shocks_cov(2, 2)/two_dble), zero_dble, HUGE_FLOAT, info)

    ! Initialize containers for disturbances with empty values.
    draws_emax_ambiguity_transformed = MISSING_FLOAT
    draws_emax_ambiguity_standard = MISSING_FLOAT
    draws_emax_risk = MISSING_FLOAT


    DO period = (num_periods - 1), 0, -1

        draws_emax_standard = periods_draws_emax(period + 1, :, :)
        num_states = states_number_period(period + 1)

        ! Transform disturbances
        IF (optim_paras%level(1) .GT. MIN_AMBIGUITY) THEN
            IF (ambi_spec%mean) THEN
                DO i = 1, num_draws_emax
                    draws_emax_ambiguity_transformed(i:i, :) = TRANSPOSE(MATMUL(optim_paras%shocks_cholesky, TRANSPOSE(draws_emax_standard(i:i, :))))
                END DO
            ELSE
                draws_emax_ambiguity_standard = draws_emax_standard
            END IF
        ELSE
            CALL transform_disturbances(draws_emax_risk, draws_emax_standard, shocks_mean, optim_paras%shocks_cholesky)
        END IF

        IF (is_write) CALL record_solution(4, file_sim, period, num_states)

        any_interpolated = (num_points_interp .LE. num_states) .AND. is_interpolated

        IF (any_interpolated) THEN

            ALLOCATE(is_simulated(num_states), endogenous(num_states), maxe(num_states), exogenous(num_states, 9), predictions(num_states))

            is_simulated = get_simulated_indicator(num_points_interp, num_states, period, is_debug)

            CALL get_exogenous_variables(exogenous, maxe, period, num_states, periods_rewards_systematic, shifts, mapping_state_idx, periods_emax, states_all, edu_spec, optim_paras)

            CALL get_endogenous_variable(endogenous, opt_ambi_details, period, num_states, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, is_simulated, maxe, draws_emax_risk, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, edu_spec, ambi_spec, optim_paras, optimizer_options)

            CALL get_predictions(predictions, endogenous, exogenous, maxe, is_simulated, num_states, file_sim, is_write)

            periods_emax(period + 1, :num_states) = predictions

            DEALLOCATE(is_simulated, exogenous, maxe, endogenous, predictions)

        ELSE

            DO k = 0, (states_number_period(period + 1) - 1)

                rewards_systematic = periods_rewards_systematic(period + 1, k + 1, :)

                IF (optim_paras%level(1) .GT. MIN_AMBIGUITY) THEN
                    CALL construct_emax_ambiguity(emax, opt_ambi_details, num_periods, num_draws_emax, period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, rewards_systematic, periods_emax, states_all, mapping_state_idx, edu_spec, ambi_spec, optim_paras, optimizer_options)
                ELSE
                    CALL construct_emax_risk(emax, period, k, draws_emax_risk, rewards_systematic, periods_emax, states_all, mapping_state_idx, edu_spec, optim_paras)
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

        INQUIRE(FILE='.interpolation.respy.test', EXIST=READ_IN)

        IF (READ_IN) THEN

            OPEN(UNIT=99, FILE='.interpolation.respy.test', ACTION='READ')

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
SUBROUTINE get_exogenous_variables(independent_variables, maxe, period, num_states, periods_rewards_systematic, shifts, mapping_state_idx, periods_emax, states_all, edu_spec, optim_paras)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: independent_variables(num_states, 9)
    REAL(our_dble), INTENT(OUT)         :: maxe(num_states)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras
    TYPE(EDU_DICT), INTENT(IN)          :: edu_spec

    REAL(our_dble), INTENT(IN)          :: periods_rewards_systematic(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(IN)          :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)          :: shifts(4)

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 4, num_types)
    INTEGER(our_int), INTENT(IN)        :: states_all(num_periods, max_states_period, 5)
    INTEGER(our_int), INTENT(IN)        :: num_states
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

        CALL get_total_values(total_values, period, num_periods, rewards_systematic, shifts, mapping_state_idx, periods_emax, k, states_all, optim_paras, edu_spec)

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
SUBROUTINE get_endogenous_variable(endogenous, opt_ambi_details, period, num_states, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, is_simulated, maxe, draws_emax_risk, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, edu_spec, ambi_spec, optim_paras, optimizer_options)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: opt_ambi_details(num_periods, max_states_period, 7)
    REAL(our_dble), INTENT(OUT)         :: endogenous(num_states)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras
    TYPE(AMBI_DICT), INTENT(IN)         :: ambi_spec
    TYPE(EDU_DICT), INTENT(IN)          :: edu_spec

    REAL(our_dble), INTENT(IN)          :: periods_rewards_systematic(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(IN)          :: draws_emax_ambiguity_transformed(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)          :: draws_emax_ambiguity_standard(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)          :: draws_emax_risk(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)          :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)          :: maxe(num_states)

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 4, num_types)
    INTEGER(our_int), INTENT(IN)        :: states_all(num_periods, max_states_period, 5)
    INTEGER(our_int), INTENT(IN)        :: num_states
    INTEGER(our_int), INTENT(IN)        :: period

    LOGICAL, INTENT(IN)                 :: is_simulated(num_states)

    TYPE(OPTIMIZER_COLLECTION), INTENT(IN)  :: optimizer_options

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

        IF (optim_paras%level(1) .GT. MIN_AMBIGUITY) THEN
            CALL construct_emax_ambiguity(emax, opt_ambi_details, num_periods, num_draws_emax, period, k, draws_emax_ambiguity_standard, draws_emax_ambiguity_transformed, rewards_systematic, periods_emax, states_all, mapping_state_idx, edu_spec, ambi_spec, optim_paras, optimizer_options)
        ELSE
            CALL construct_emax_risk(emax, period, k, draws_emax_risk, rewards_systematic, periods_emax, states_all, mapping_state_idx, edu_spec, optim_paras)
        END IF

        ! Construct dependent variable
        endogenous(k + 1) = emax - maxe(k + 1)

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_predictions(predictions, endogenous, exogenous, maxe, is_simulated, num_states, file_sim, is_write)

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

    CHARACTER(225), INTENT(IN)          :: file_sim

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
        CALL record_solution(coeffs, r_squared, bse, file_sim)
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
FUNCTION calculate_wages_systematic(covariates, optim_paras) RESULT(wages)

    !/* external objects        */

    REAL(our_dble)                      :: wages(2)

    TYPE(COVARIATES_DICT), INTENT(IN)   :: covariates
    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    !/* internal objects        */

    INTEGER(our_int)                    :: info
    INTEGER(our_int)                    :: i

    REAL(our_dble)                      :: covars_wages(10)

    LOGICAL                             :: IS_RESTUD

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Auxiliary objects
    covars_wages(1) = one_dble
    covars_wages(2) = covariates%edu
    covars_wages(3) = covariates%exp_a
    covars_wages(4) = (covariates%exp_a ** 2) / one_hundred_dble
    covars_wages(5) = covariates%exp_b
    covars_wages(6) = (covariates%exp_b ** 2) / one_hundred_dble
    covars_wages(7) = covariates%hs_graduate
    covars_wages(8) = covariates%co_graduate
    covars_wages(9) = covariates%period - one_dble
    covars_wages(10) = covariates%is_minor

    ! This used for testing purposes, where we compare the results from the RESPY package to the original RESTUD program.
    INQUIRE(FILE='.restud.respy.scratch', EXIST=IS_RESTUD)
    IF (IS_RESTUD) THEN
        covars_wages(4) = covars_wages(4) * one_hundred_dble
        covars_wages(6) = covars_wages(6) * one_hundred_dble
    END IF

    ! Calculate systematic part of reward in OCCUPAION A and OCCUPATION B
    CALL clip_value(wages(1), EXP(DOT_PRODUCT(covars_wages, optim_paras%coeffs_a(:10))), zero_dble, HUGE_FLOAT, info)

    ! Calculate systematic part of reward in Occupation B
    CALL clip_value(wages(2), EXP(DOT_PRODUCT(covars_wages, optim_paras%coeffs_b(:10))), zero_dble, HUGE_FLOAT, info)

    DO i = 1, 2
        wages(i) = wages(i) * EXP(optim_paras%type_shifts(covariates%type + 1, i))
    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
END MODULE
