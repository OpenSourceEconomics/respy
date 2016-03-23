!*******************************************************************************
!*******************************************************************************
MODULE robufort_extension

    !/* external modules        */

    USE robufort_constants

    USE robufort_auxiliary

    !/* setup                   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE store_results(mapping_state_idx, states_all, & 
                periods_payoffs_ex_post, periods_payoffs_systematic, & 
                states_number_period, periods_emax, periods_payoffs_future, & 
                num_periods, min_idx, crit_val, request) 

    !/* external objects        */


    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_number_period(:)
    INTEGER(our_int), INTENT(IN)    :: states_all(:,:,:)
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: min_idx 

    REAL(our_dble), INTENT(IN)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), INTENT(IN)      :: periods_payoffs_ex_post(:, :, :)    
    REAL(our_dble), INTENT(IN)      :: periods_payoffs_future(:, :, :)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: crit_val

    CHARACTER(10), INTENT(IN)       :: request

    !/* internal objects        */

    INTEGER(our_int)                :: max_states_period
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! This is a break in design as otherwise I need to carry the integer up 
    ! from the solution level.
    max_states_period = MAXVAL(states_number_period)

    ! Write out results for the store results.
    1800 FORMAT(5(1x,i5))

    OPEN(UNIT=1, FILE='.mapping_state_idx.robufort.dat')

    DO period = 1, num_periods
        DO i = 1, num_periods
            DO j = 1, num_periods
                DO k = 1, min_idx
                    WRITE(1, 1800) mapping_state_idx(period, i, j, k, :)
                END DO
            END DO
        END DO
    END DO

    CLOSE(1)


    2000 FORMAT(4(1x,i5))

    OPEN(UNIT=1, FILE='.states_all.robufort.dat')

    DO period = 1, num_periods
        DO i = 1, max_states_period
            WRITE(1, 2000) states_all(period, i, :)
        END DO
    END DO

    CLOSE(1)


    1900 FORMAT(4(1x,f25.15))

    OPEN(UNIT=1, FILE='.periods_payoffs_systematic.robufort.dat')

    DO period = 1, num_periods
        DO i = 1, max_states_period
            WRITE(1, 1900) periods_payoffs_systematic(period, i, :)
        END DO
    END DO

    CLOSE(1)


    3100 FORMAT(4(1x,f25.15))

    OPEN(UNIT=1, FILE='.periods_payoffs_future.robufort.dat')

    DO period = 1, num_periods
        DO i = 1, max_states_period
            WRITE(1, 3100) periods_payoffs_future(period, i, :)
        END DO
    END DO

    CLOSE(1)


    OPEN(UNIT=1, FILE='.periods_payoffs_ex_post.robufort.dat')

    DO period = 1, num_periods
        DO i = 1, max_states_period
            WRITE(1, 1900) periods_payoffs_ex_post(period, i, :)
        END DO
    END DO

    CLOSE(1)


    2100 FORMAT(i5)

    OPEN(UNIT=1, FILE='.states_number_period.robufort.dat')

    DO period = 1, num_periods
        WRITE(1, 2100) states_number_period(period)
    END DO

    CLOSE(1)


    2200 FORMAT(i5)

    OPEN(UNIT=1, FILE='.max_states_period.robufort.dat')

    WRITE(1, 2200) max_states_period

    CLOSE(1)


    2400 FORMAT(100000(1x,f25.15))

    OPEN(UNIT=1, FILE='.periods_emax.robufort.dat')

    DO period = 1, num_periods
        WRITE(1, 2400) periods_emax(period, :)
    END DO

    CLOSE(1)

    ! Write out value of criterion function if evaluated.
    IF (request == 'evaluate') THEN

        2500 FORMAT(1x,f25.15)

        OPEN(UNIT=1, FILE='.eval.robufort.dat')

        WRITE(1, 2500) crit_val

        CLOSE(1)

    END IF

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE read_specification(num_periods, delta, level, coeffs_a, coeffs_b, &
                coeffs_edu, edu_start, edu_max, coeffs_home, shocks_cov, & 
                shocks_cholesky, num_draws_emax, seed_emax, seed_prob, &
                num_agents, seed_data, is_debug, is_deterministic, &
                is_interpolated, num_points, min_idx, is_ambiguous, measure, & 
                request, num_draws_prob, is_myopic)

    !
    !   This function serves as the replacement for the clsRobupy and reads in 
    !   all required information about the model parameterization. It just 
    !   reads in all required information.  
    !

    !/* external objects        */

    INTEGER(our_int), INTENT(OUT)   :: num_draws_emax
    INTEGER(our_int), INTENT(OUT)   :: num_draws_prob
    INTEGER(our_int), INTENT(OUT)   :: num_periods
    INTEGER(our_int), INTENT(OUT)   :: num_agents
    INTEGER(our_int), INTENT(OUT)   :: num_points
    INTEGER(our_int), INTENT(OUT)   :: seed_data
    INTEGER(our_int), INTENT(OUT)   :: seed_prob
    INTEGER(our_int), INTENT(OUT)   :: seed_emax
    INTEGER(our_int), INTENT(OUT)   :: edu_start
    INTEGER(our_int), INTENT(OUT)   :: edu_max
    INTEGER(our_int), INTENT(OUT)   :: min_idx

    REAL(our_dble), INTENT(OUT)     :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(OUT)     :: coeffs_home(1)
    REAL(our_dble), INTENT(OUT)     :: coeffs_edu(3)
    REAL(our_dble), INTENT(OUT)     :: shocks_cov(4, 4)
    REAL(our_dble), INTENT(OUT)     :: coeffs_a(6)
    REAL(our_dble), INTENT(OUT)     :: coeffs_b(6)
    REAL(our_dble), INTENT(OUT)     :: delta
    REAL(our_dble), INTENT(OUT)     :: level

    LOGICAL, INTENT(OUT)            :: is_interpolated
    LOGICAL, INTENT(OUT)            :: is_deterministic
    LOGICAL, INTENT(OUT)            :: is_ambiguous    
    LOGICAL, INTENT(OUT)            :: is_myopic
    LOGICAL, INTENT(OUT)            :: is_debug

    CHARACTER(10), INTENT(OUT)      :: measure 
    CHARACTER(10), INTENT(OUT)      :: request

    !/* internal objects        */

    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Fix formatting
    1500 FORMAT(6(1x,f15.10))
    1510 FORMAT(f15.10)

    1505 FORMAT(i10)
    1515 FORMAT(i10,1x,i10)

    ! Read model specification
    OPEN(UNIT=1, FILE='.model.robufort.ini')

        ! BASICS
        READ(1, 1505) num_periods
        READ(1, 1510) delta

        ! AMBIGUITY
        READ(1, 1510) level
        READ(1, *) measure

        ! WORK
        READ(1, 1500) coeffs_a
        READ(1, 1500) coeffs_b

        ! EDUCATION
        READ(1, 1500) coeffs_edu
        READ(1, 1515) edu_start, edu_max

        ! HOME
        READ(1, 1500) coeffs_home

        ! SHOCKS
        DO j = 1, 4
            READ(1, 1500) (shocks_cov(j, k), k=1, 4)
        END DO

        ! SOLUTION
        READ(1, 1505) num_draws_emax
        READ(1, 1505) seed_emax

        ! SIMULATION
        READ(1, 1505) num_agents
        READ(1, 1505) seed_data

        ! PROGRAM
        READ(1, *) is_debug

        ! INTERPOLATION
        READ(1, *) is_interpolated
        READ(1, 1505) num_points

        ! ESTIMATION
        READ(1, 1505) num_draws_prob
        READ(1, 1505) seed_prob

        ! AUXILIARY
        READ(1, 1505) min_idx
        READ(1, *) is_ambiguous
        READ(1, *) is_deterministic
        READ(1, *) is_myopic

        ! REQUUEST
        READ(1, *) request

    CLOSE(1, STATUS='delete')

    ! Construct auxiliary objects. The case distinction align the reasoning
    ! between the PYTHON/F2PY implementations.
    IF (is_deterministic) THEN
        shocks_cholesky = zero_dble
    ELSE
        CALL cholesky(shocks_cholesky, shocks_cov)
    END IF

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE create_draws(draws, num_periods, num_draws_emax, seed, is_debug, & 
                which, shocks_cholesky, is_ambiguous)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)  :: draws(:, :, :)

    INTEGER(our_int), INTENT(IN)                :: num_draws_emax
    INTEGER(our_int), INTENT(IN)                :: num_periods
    INTEGER(our_int), INTENT(IN)                :: seed 

    REAL(our_dble), INTENT(IN)                  :: shocks_cholesky(4, 4)

    LOGICAL, INTENT(IN)                         :: is_ambiguous
    LOGICAL, INTENT(IN)                         :: is_debug

    CHARACTER(4)                                :: which

    !/* internal objects        */

    INTEGER(our_int)                            :: seed_inflated(15)
    INTEGER(our_int)                            :: seed_size
    INTEGER(our_int)                            :: period
    INTEGER(our_int)                            :: j
    INTEGER(our_int)                            :: i
    
    LOGICAL                                     :: READ_IN

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

    ! Allocate containers
    ALLOCATE(draws(num_periods, num_draws_emax, 4))

    ! Set random seed
    seed_inflated(:) = seed
    
    CALL RANDOM_SEED(size=seed_size)

    CALL RANDOM_SEED(put=seed_inflated)

    ! Draw random deviates from a standard normal distribution or read it in
    ! from disk. The latter is available to allow for testing across
    ! implementations.
    INQUIRE(FILE='draws.txt', EXIST=READ_IN)

    IF ((READ_IN .EQV. .True.)  .AND. (is_debug .EQV. .True.)) THEN

        OPEN(12, file='draws.txt')

        DO period = 1, num_periods

            DO j = 1, num_draws_emax
        
                2000 FORMAT(4(1x,f15.10))
                READ(12,2000) draws(period, j, :)
        
            END DO
      
        END DO

        CLOSE(12)

    ELSE

        DO period = 1, num_periods

            CALL multivariate_normal(draws(period, :, :))
        
        END DO

    END IF
    ! Standard normal deviates used for the Monte Carlo integration of the
    ! expected future values in the solution step. Also, standard normal
    ! deviates for the Monte Carlo integration of the choice probabilities in
    ! the evaluation step.
    IF ((which == 'emax') .OR. (which == 'prob')) THEN

        draws = draws
    
    ! Deviates for the simulation of a synthetic agent population.
    ELSE IF (which == 'sims') THEN
        ! Standard deviates transformed to the distributions relevant for
        ! the agents actual decision making as traversing the tree.

        ! Transformations
        DO period = 1, num_periods
            
            ! Apply variance change
            DO i = 1, num_draws_emax
                draws(period, i:i, :) = &
                    TRANSPOSE(MATMUL(shocks_cholesky, TRANSPOSE(draws(period, i:i, :))))
            END DO

            DO j = 1, 2
                draws(period, :, j) =  EXP(draws(period, :, j))
            END DO
                
        END DO      

    END IF

END SUBROUTINE
!******************************************************************************* 
!******************************************************************************* 
SUBROUTINE read_dataset(data_array, num_periods, num_agents)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)  :: data_array(:, :)

    INTEGER(our_int), INTENT(IN)                :: num_periods
    INTEGER(our_int), INTENT(IN)                :: num_agents

    !/* internal objects        */

    INTEGER(our_int)                            :: j
    INTEGER(our_int)                            :: k
    
!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 
    
    ! Allocate data container
    ALLOCATE(data_array(num_periods * num_agents, 8))
    
    ! Read observed data to double precision array
    OPEN(UNIT=1, FILE='.data.robufort.dat')

        DO j = 1, num_periods * num_agents
            READ(1, *) (data_array(j, k), k = 1, 8)
        END DO
    
    CLOSE(1, STATUS='delete')

END SUBROUTINE
!******************************************************************************* 
!******************************************************************************* 
END MODULE 
!******************************************************************************* 
!******************************************************************************* 
PROGRAM robufort

    !/* external modules        */

    USE robufort_extension

    USE robufort_library

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
    INTEGER(our_int)                :: seed_data
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: edu_start
    INTEGER(our_int)                :: edu_max
    INTEGER(our_int)                :: min_idx

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_ex_post(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_future(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_emax(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_prob(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax(:, :)
    REAL(our_dble), ALLOCATABLE     :: data_array(:, :)

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: shocks_cov(4, 4)
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
            num_agents, seed_data, is_debug, is_deterministic, & 
            is_interpolated, num_points, min_idx, is_ambiguous, measure, & 
            request, num_draws_prob, is_myopic)

    ! This part creates (or reads from disk) the draws for the Monte 
    ! Carlo integration of the EMAX. For is_debugging purposes, these might 
    ! also be read in from disk or set to zero/one.   
    CALL create_draws(periods_draws_emax, num_periods, num_draws_emax, &
            seed_emax, is_debug, 'emax', shocks_cholesky, is_ambiguous)

    ! Execute on request.
    IF (request == 'solve') THEN

        ! Solve the model for a given parametrization.    
        CALL solve_fortran_bare(periods_payoffs_systematic, & 
                periods_payoffs_ex_post, periods_payoffs_future, & 
                states_number_period, mapping_state_idx, periods_emax, & 
                states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, & 
                shocks_cov, shocks_cholesky, is_deterministic, & 
                is_interpolated, num_draws_emax, periods_draws_emax, & 
                is_ambiguous, num_periods, num_points, edu_start, is_myopic, & 
                is_debug, measure, edu_max, min_idx, delta, level)

    ELSE IF (request == 'evaluate') THEN

        ! This part creates (or reads from disk) the draws for the Monte 
        ! Carlo integration of the choice probabilities. For is_debugging 
        ! purposes, these might also be read in from disk or set to zero/one.   
        CALL create_draws(periods_draws_prob, num_periods, num_draws_prob, &
                seed_prob, is_debug, 'prob', shocks_cholesky, is_ambiguous)

        ! Read observed dataset from disk
        CALL read_dataset(data_array, num_periods, num_agents)

        ! Solve the model for a given parametrization.    
        CALL solve_fortran_bare(periods_payoffs_systematic, & 
                periods_payoffs_ex_post, periods_payoffs_future, & 
                states_number_period, mapping_state_idx, periods_emax, & 
                states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, & 
                shocks_cov, shocks_cholesky, is_deterministic, & 
                is_interpolated, num_draws_emax, periods_draws_emax, & 
                is_ambiguous, num_periods, num_points, edu_start, is_myopic, & 
                is_debug, measure, edu_max, min_idx, delta, level)

        CALL evaluate_criterion_function(crit_val, mapping_state_idx, &
                periods_emax, periods_payoffs_systematic, states_all, & 
                shocks_cov, edu_max, delta, edu_start, num_periods, & 
                shocks_cholesky, num_agents, num_draws_prob, data_array, & 
                periods_draws_prob, is_deterministic)

    END IF

    ! Store results. These are read in by the PYTHON wrapper and added to the 
    ! clsRobupy instance.
    CALL store_results(mapping_state_idx, states_all, periods_payoffs_ex_post, & 
            periods_payoffs_systematic, states_number_period, periods_emax, &
            periods_payoffs_future, num_periods, min_idx, crit_val, request) 

!*******************************************************************************
!*******************************************************************************
END PROGRAM