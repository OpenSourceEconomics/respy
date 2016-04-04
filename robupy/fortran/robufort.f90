!*******************************************************************************
!******************************************************************************* 
PROGRAM robufort

    !/* external modules        */

    USE evaluate_fortran

    USE shared_auxiliary

    USE solve_fortran

    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */

    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE   :: states_all(:, :, :)

    INTEGER(our_int)                :: max_states_period
    INTEGER(our_int)                :: num_draws_emax
    INTEGER(our_int)                :: num_draws_prob
    INTEGER(our_int)                :: num_periods
    INTEGER(our_int)                :: num_agents
    INTEGER(our_int)                :: num_points
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: edu_start
    INTEGER(our_int)                :: edu_max
    INTEGER(our_int)                :: min_idx

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_emax(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_prob(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax(:, :)
    REAL(our_dble), ALLOCATABLE     :: data_array(:, :)

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: shocks_cov(4, 4)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: crit_val
    REAL(our_dble)                  :: delta
    REAL(our_dble)                  :: level

    LOGICAL                         :: is_deterministic
    LOGICAL                         :: is_interpolated
    LOGICAL                         :: is_ambiguous
    LOGICAL                         :: is_myopic
    LOGICAL                         :: is_debug

    CHARACTER(10)                   :: measure 
    CHARACTER(10)                   :: request

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Read specification of model. This is the FORTRAN replacement for the 
    ! clsRobupy instance that carries the model parametrization for the 
    ! PYTHON/F2PY implementations.
    CALL read_specification(num_periods, delta, level, coeffs_a, coeffs_b, &
            coeffs_edu, edu_start, edu_max, coeffs_home, shocks_cov, &
            shocks_cholesky, num_draws_emax, seed_emax, seed_prob, &
            num_agents, is_debug, is_deterministic, is_interpolated, & 
            num_points, min_idx, is_ambiguous, measure, request, & 
            num_draws_prob, is_myopic)

    ! This part creates (or reads from disk) the draws for the Monte 
    ! Carlo integration of the EMAX. For is_debugging purposes, these might 
    ! also be read in from disk or set to zero/one.   
    CALL create_draws(periods_draws_emax, num_periods, num_draws_emax, &
            seed_emax, is_debug)

    ! Execute on request.
    IF (request == 'solve') THEN

        ! Solve the model for a given parametrization.    
        CALL fort_solve(periods_payoffs_systematic, states_number_period, & 
                mapping_state_idx, periods_emax, states_all, coeffs_a, & 
                coeffs_b, coeffs_edu, coeffs_home, shocks_cov, & 
                is_deterministic, is_interpolated, num_draws_emax, & 
                periods_draws_emax, is_ambiguous, num_periods, num_points, & 
                edu_start, is_myopic, is_debug, measure, edu_max, min_idx, & 
                delta, level)

    ELSE IF (request == 'evaluate') THEN

        ! This part creates (or reads from disk) the draws for the Monte 
        ! Carlo integration of the choice probabilities. For is_debugging 
        ! purposes, these might also be read in from disk or set to zero/one.   
        CALL create_draws(periods_draws_prob, num_periods, num_draws_prob, &
                seed_prob, is_debug)

        ! Read observed dataset from disk
        CALL read_dataset(data_array, num_periods, num_agents)

        ! Solve the model for a given parametrization.    
        CALL fort_solve(periods_payoffs_systematic, states_number_period, & 
                mapping_state_idx, periods_emax, states_all, coeffs_a, & 
                coeffs_b, coeffs_edu, coeffs_home, shocks_cov, & 
                is_deterministic, is_interpolated, num_draws_emax, & 
                periods_draws_emax, is_ambiguous, num_periods, num_points, & 
                edu_start, is_myopic, is_debug, measure, edu_max, min_idx, & 
                delta, level)

        CALL fort_evaluate(crit_val, periods_payoffs_systematic, & 
                mapping_state_idx, periods_emax, states_all, shocks_cov, & 
                is_deterministic, num_periods, edu_start, edu_max, delta, & 
                data_array, num_agents, num_draws_prob, periods_draws_prob)

    END IF

    ! Store results. These are read in by the PYTHON wrapper and added to the 
    ! clsRobupy instance.
    CALL store_results(mapping_state_idx, states_all, &
            periods_payoffs_systematic, states_number_period, periods_emax, &
            num_periods, min_idx, crit_val, request)

!*******************************************************************************
!*******************************************************************************
END PROGRAM