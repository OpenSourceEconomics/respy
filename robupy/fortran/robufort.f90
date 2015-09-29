!******************************************************************************
!******************************************************************************
MODULE robufort_extension

    !/* external modules    */

    USE robufort_program_constants
    USE robufort_auxiliary

    !/* setup   */

    IMPLICIT NONE

    !/* core functions */

    PUBLIC :: read_specification
    PUBLIC :: get_disturbances
    PUBLIC :: store_results

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE store_results(mapping_state_idx, states_all, &
                periods_payoffs_ex_ante, states_number_period, periods_emax, &
                num_periods, min_idx, max_states_period) 

    !/* external objects    */

    INTEGER(our_int), INTENT(IN)    :: max_states_period
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: min_idx 
    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:,:,:)
    INTEGER(our_int), INTENT(IN)    :: states_number_period(:)
    
    REAL(our_dble), INTENT(IN)      :: periods_payoffs_ex_ante(:, :, :)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)

    !/* internal objects    */

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: period

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    

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

    OPEN(UNIT=1, FILE='.periods_payoffs_ex_ante.robufort.dat')

    DO period = 1, num_periods
        DO i = 1, max_states_period
            WRITE(1, 1900) periods_payoffs_ex_ante(period, i, :)
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


END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE read_specification(num_periods, delta, level, coeffs_A, &
                coeffs_B, coeffs_edu, edu_start, edu_max, coeffs_home, &
                shocks, num_draws, seed_solution, num_agents, &
                seed_simulation, is_debug, is_zero) 

    !/* external objects    */

    INTEGER(our_int), INTENT(OUT)   :: seed_simulation 
    INTEGER(our_int), INTENT(OUT)   :: seed_solution 
    INTEGER(our_int), INTENT(OUT)   :: num_periods
    INTEGER(our_int), INTENT(OUT)   :: num_agents
    INTEGER(our_int), INTENT(OUT)   :: num_draws
    INTEGER(our_int), INTENT(OUT)   :: edu_start
    INTEGER(our_int), INTENT(OUT)   :: edu_max

    REAL(our_dble), INTENT(OUT)     :: coeffs_home(1)
    REAL(our_dble), INTENT(OUT)     :: coeffs_edu(3)
    REAL(our_dble), INTENT(OUT)     :: shocks(4, 4)
    REAL(our_dble), INTENT(OUT)     :: coeffs_A(6)
    REAL(our_dble), INTENT(OUT)     :: coeffs_B(6)
    REAL(our_dble), INTENT(OUT)     :: delta
    REAL(our_dble), INTENT(OUT)     :: level

    LOGICAL, INTENT(OUT)            :: is_debug
    LOGICAL, INTENT(OUT)            :: is_zero

    !/* internal objects    */

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

    !1520 FORMAT(15)

    ! Read model specification
    OPEN(UNIT=1, FILE='.model.robufort.ini')

        ! BASICS
        READ(1, 1505) num_periods
        READ(1, 1510) delta

        ! AMBIGUITY
        READ(1, 1510) level

        ! WORK
        READ(1, 1500) coeffs_A
        READ(1, 1500) coeffs_B

        ! EDUCATION
        READ(1, 1500) coeffs_edu
        READ(1, 1515) edu_start, edu_max

        ! HOME
        READ(1, 1500) coeffs_home

        ! SHOCKS
        DO j = 1, 4
            READ(1, 1500) (shocks(j, k), k=1, 4)
        END DO

        ! SOLUTION
        READ(1, 1505) num_draws
        READ(1, 1505) seed_solution

        ! SIMULATION
        READ(1, 1505) num_agents
        READ(1, 1505) seed_simulation

        ! PROGRAM
        READ(1, *) is_debug

        ! AUXILIARY
        READ(1, *) is_zero

    CLOSE(1, STATUS='delete')

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_disturbances(periods_eps_relevant, level, shocks, seed, &
                is_debug, is_zero) 

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: periods_eps_relevant(:, :, :)

    REAL(our_dble), INTENT(IN)      :: shocks(4, 4)
    REAL(our_dble), INTENT(IN)      :: level

    INTEGER(our_int),INTENT(IN)     :: seed 

    LOGICAL, INTENT(IN)             :: is_debug
    LOGICAL, INTENT(IN)             :: is_zero

    !/* internal objects    */

    REAL(our_dble)                  :: eps_cholesky(4, 4)

    INTEGER(our_int)                :: seed_inflated(15)
    INTEGER(our_int)                :: num_periods
    INTEGER(our_int)                :: seed_size
    INTEGER(our_int)                :: num_draws
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: i
    
    LOGICAL                         :: READ_IN

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

    ! Auxiliary objects
    num_periods = SIZE(periods_eps_relevant, 1)

    num_draws = SIZE(periods_eps_relevant, 2)

    CALL cholesky(eps_cholesky, shocks)

    ! Set random seed
    seed_inflated(:) = seed
    
    CALL RANDOM_SEED(size=seed_size)

    CALL RANDOM_SEED(put=seed_inflated)

    ! Create standard deviates
    INQUIRE(FILE='disturbances.txt', EXIST=READ_IN)

    IF ((READ_IN .EQV. .True.)  .AND. (is_debug .EQV. .True.)) THEN

        OPEN(12, file='disturbances.txt')

        DO period = 1, num_periods

            DO j = 1, num_draws
        
                2000 FORMAT(4(1x,f15.10))
                READ(12,2000) periods_eps_relevant(period, j, :)
        
            END DO
      
        END DO

        CLOSE(12)

    ELSE

        DO period = 1, num_periods

            CALL multivariate_normal(periods_eps_relevant(period, :, :))
        
        END DO

    END IF

    ! Transformations
    DO period = 1, num_periods
        
        ! Apply variance change
        DO i = 1, num_draws
        
            periods_eps_relevant(period, i:i, :) = &
                TRANSPOSE(MATMUL(eps_cholesky, TRANSPOSE(periods_eps_relevant(period, i:i, :))))
        
        END DO

    END DO

    ! Transformation in case of risk-only. In the case of ambiguity, this 
    ! transformation is later as it needs adjustment for the switched means.
    IF (level .EQ. zero_dble) THEN
        
        ! Transform disturbance for occupations
        DO period = 1, num_periods

            DO j = 1, 2
            
                periods_eps_relevant(period, :, j) = &
                        EXP(periods_eps_relevant(period, :, j))
            
            END DO

        END DO
        
    END IF

    ! Special case of absence randomness. Note that the disturbances for the 
    ! two occupations are set to one instead of zero.
    IF (is_zero) THEN

        periods_eps_relevant = zero_dble

        DO period = 1, num_periods

            DO j = 1, 2

                periods_eps_relevant(period, :, j) = one_dble

            END DO

        END DO

    END IF

END SUBROUTINE
!******************************************************************************* 
!******************************************************************************* 
END MODULE 

!******************************************************************************* 
!******************************************************************************* 
PROGRAM robufort

    !/* external modules    */

    USE robufort_extension

    USE robufort_program_constants

    ! TODO: Remove
    USE robufort_library

    USE robufort_ambiguity

    !/* setup   */

    IMPLICIT NONE

    !/* objects */

    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE   :: states_all(:, :, :)

    INTEGER(our_int)                :: max_states_period
    INTEGER(our_int)                :: seed_simulation 
    INTEGER(our_int)                :: seed_solution 
    INTEGER(our_int)                :: num_periods
    INTEGER(our_int)                :: num_agents
    INTEGER(our_int)                :: edu_lagged
    INTEGER(our_int)                :: future_idx
    INTEGER(our_int)                :: edu_start
    INTEGER(our_int)                :: num_draws
    INTEGER(our_int)                :: covars(6)
    INTEGER(our_int)                :: maxiter
    INTEGER(our_int)                :: edu_max
    INTEGER(our_int)                :: min_idx
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: total
    INTEGER(our_int)                :: exp_A
    INTEGER(our_int)                :: exp_B
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: j

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_ex_ante(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_ex_post(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_future_payoffs(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_eps_relevant(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: eps_relevant(:, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax(:, :)
    REAL(our_dble), ALLOCATABLE     :: eps_input(:, :)
    
    REAL(our_dble)                  :: payoffs_ex_post(4)
    REAL(our_dble)                  :: payoffs_ex_ante(4)
    REAL(our_dble)                  :: future_payoffs(4)
    REAL(our_dble)                  :: total_payoffs(4)
    REAL(our_dble)                  :: disturbances(4)
    REAL(our_dble)                  :: emax_simulated
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: x_internal(2)    
    REAL(our_dble)                  :: shocks(4, 4)
    REAL(our_dble)                  :: coeffs_A(6)
    REAL(our_dble)                  :: coeffs_B(6)
    REAL(our_dble)                  :: x_start(2)
    REAL(our_dble)                  :: maximum    
    REAL(our_dble)                  :: payoff
    REAL(our_dble)                  :: delta
    REAL(our_dble)                  :: level
    REAL(our_dble)                  :: ftol
    REAL(our_dble)                  :: eps

    LOGICAL                         :: is_myopic
    LOGICAL                         :: is_debug
    LOGICAL                         :: is_huge
    LOGICAL                         :: is_zero
    LOGICAL                         :: is_ambiguous


    INTEGER(our_int)                :: m
    INTEGER(our_int)                :: meq
    INTEGER(our_int)                :: n
    INTEGER(our_int)                :: mode
    INTEGER(our_int)                :: iter
    INTEGER(our_int)                :: mieq
    INTEGER(our_int)                :: mineq
    INTEGER(our_int)                :: jw(7)
    INTEGER(our_int)                :: l_jw
    INTEGER(our_int)                :: l_w
    INTEGER(our_int)                :: la

    REAL(our_dble)                  :: a(1, 3)
    REAL(our_dble)                  :: xl(2)
    REAL(our_dble)                  :: xu(2)
    REAL(our_dble)                  :: c(1)
    REAL(our_dble)                  :: g(3)
    REAL(our_dble)                  :: w(144)
    REAL(our_dble)                  :: f


    LOGICAL                         :: is_finished

    REAL(our_dble)                  :: alt_mean(4, 1) = zero_dble
    REAL(our_dble)                  :: old_mean(4, 1) = zero_dble
    REAL(our_dble)                  :: alt_cov(4,4)
    REAL(our_dble)                  :: old_cov(4,4)
    REAL(our_dble)                  :: inv_old_cov(4,4)
    REAL(our_dble)                  :: comp_a
    REAL(our_dble)                  :: comp_b(1, 1)
    REAL(our_dble)                  :: comp_c
    REAL(our_dble)                  :: rslt

    REAL(our_dble)                  :: f0 

    REAL(our_dble)                  :: ei(2)
    REAL(our_dble)                  :: d(2)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Read specification of model
    CALL read_specification(num_periods, delta, level, coeffs_A, coeffs_B, & 
            coeffs_edu, edu_start, edu_max, coeffs_home, shocks, num_draws, &
            seed_solution, num_agents, seed_simulation, &
            is_debug, is_zero) 

    ! Auxiliary objects
    min_idx = MIN(num_periods, (edu_max - edu_start + 1))

    ! Allocate arrays
    ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
    ALLOCATE(states_all(num_periods, 100000, 4))
    ALLOCATE(states_number_period(num_periods))

    ! Create the state space of the model
    CALL create_state_space(states_all, states_number_period, & 
            mapping_state_idx, num_periods, edu_start, edu_max, min_idx)

    ! Auxiliary objects
    max_states_period = MAXVAL(states_number_period)
    
    ! Allocate arrays
    ALLOCATE(periods_payoffs_ex_ante(num_periods, max_states_period, 4))

    ! Calculate the ex ante payoffs
    CALL calculate_payoffs_ex_ante(periods_payoffs_ex_ante, num_periods, &
            states_number_period, states_all, edu_start, coeffs_A, coeffs_B, & 
            coeffs_edu, coeffs_home, max_states_period)

    ! Allocate additional containers
    ALLOCATE(periods_payoffs_ex_post(num_periods, max_states_period, 4))
    ALLOCATE(periods_future_payoffs(num_periods, max_states_period, 4))
    ALLOCATE(periods_eps_relevant(num_periods, num_draws, 4))
    ALLOCATE(periods_emax(num_periods, max_states_period))
    ALLOCATE(eps_relevant(num_draws, 4))
    ALLOCATE(eps_input(num_draws, 4))

    ! Draw random disturbances. For is_debugging purposes, these might also be 
    ! read in from disk or set to zero/one.
    CALL get_disturbances(periods_eps_relevant, level, shocks, seed_solution, &
            is_debug, is_zero)

    ! Perform backward induction procedure.
    CALL backward_induction(periods_emax, periods_payoffs_ex_post, &
            periods_future_payoffs, num_periods, max_states_period, &
            periods_eps_relevant, num_draws, states_number_period, & 
            periods_payoffs_ex_ante, edu_max, edu_start, &
            mapping_state_idx, states_all, delta, is_debug, shocks, &
            level)
   
    ! Store results. These are read in by the PYTHON wrapper and added 
    ! to the clsRobupy instance.
    CALL store_results(mapping_state_idx, states_all, periods_payoffs_ex_ante, &
            states_number_period, periods_emax, num_periods, min_idx, &
            max_states_period) 

!*******************************************************************************
!*******************************************************************************
END PROGRAM