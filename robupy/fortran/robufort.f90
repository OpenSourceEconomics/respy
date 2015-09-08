!******************************************************************************
!******************************************************************************
MODULE robufort_library

    !/* external modules    */

    USE robupy_program_constants
    USE robupy_auxiliary

    !/* setup   */

    IMPLICIT NONE

    !/* core functions */

    PUBLIC :: read_specification
    PUBLIC :: get_disturbances

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE read_specification(num_periods, delta, coeffs_A, coeffs_B, & 
                coeffs_edu, edu_start, edu_max, coeffs_home, shocks, & 
                num_draws, seed_solution, num_agents, seed_simulation) 

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

    ! Read model specification
    OPEN(UNIT=1, FILE='.model.robufort.ini')

        ! BASICS
        READ(1, 1505) num_periods
        READ(1, 1510) delta

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

    CLOSE(1)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_disturbances(periods_eps_relevant, shocks) 

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: periods_eps_relevant(:, :, :)

    REAL(our_dble), INTENT(IN)      :: shocks(4, 4)

    !/* internal objects    */

    INTEGER(our_int)                :: num_periods
    INTEGER(our_int)                :: period

    REAL(our_dble)                  :: mean(4)

!------------------------------------------------------------------------------ 
! Algorithm
!------------------------------------------------------------------------------ 
    ! Auxiliary objects
    num_periods = SIZE(periods_eps_relevant, 1)

    ! Initialize mean 
    mean = zero_dble

    DO period = 1, num_periods
    
        CALL multivariate_normal(periods_eps_relevant(period, :, :), mean, & 
                shocks)
    
    END DO

    PRINT *, ' Temporary Debug', num_periods, SIZE(periods_eps_relevant, 2)
        periods_eps_relevant = 0

    DO period = 1, num_periods

        periods_eps_relevant(period, :,:2) = one_dble

END DO

END SUBROUTINE
!****************************************************************************** 
!****************************************************************************** 
SUBROUTINE multivariate_normal_deviates(dim, draw)

    !/* external objects    */

    INTEGER, INTENT(IN)             :: dim

    REAL, INTENT(OUT)               :: draw(dim)
    
    !/* internal objects    */

    INTEGER                         :: g 

    REAL, ALLOCATABLE               :: u(:)
    REAL, ALLOCATABLE               :: r(:)

    !--------------------------------------------------------------------------- 
    ! Algorithm
    !--------------------------------------------------------------------------- 

    ! Allocate containers
    ALLOCATE(u(2*dim)); ALLOCATE(r(2*dim))

    ! Call uniform deviates
    CALL random_number(u)

    ! Apply Box-Muller transform
    DO g = 1, 2*dim, 2

       r(g)   = SQRT(-2*LOG(u(g)))*COS(2*pi*u(g+1)) 
       r(g+1) = SQRT(-2*LOG(u(g)))*SIN(2*pi*u(g+1)) 

    END DO

    ! Extract relevant floats
    DO g = 1, dim 

       draw(g) = r(g)     

    END DO

  END SUBROUTINE 
!****************************************************************************** 
!****************************************************************************** 
END MODULE 

!****************************************************************************** 
!****************************************************************************** 
PROGRAM robufort


    !/* external modules    */

    USE robufort_library

    USE robupy_program_constants

    !/* setup   */

    IMPLICIT NONE

    !/* objects */

    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE   :: states_all(:, :, :)

    INTEGER(our_int)                :: max_states_period
    INTEGER(our_int)                :: current_state(4)
    INTEGER(our_int)                :: seed_simulation
    INTEGER(our_int)                :: seed_solution
    INTEGER(our_int)                :: num_periods
    INTEGER(our_int)                :: num_agents
    INTEGER(our_int)                :: edu_lagged
    INTEGER(our_int)                :: future_idx
    INTEGER(our_int)                :: edu_start
    INTEGER(our_int)                :: num_draws
    INTEGER(our_int)                :: covars(6)
    INTEGER(our_int)                :: choice(1)
    INTEGER(our_int)                :: edu_max
    INTEGER(our_int)                :: min_idx
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: total
    INTEGER(our_int)                :: exp_A
    INTEGER(our_int)                :: count
    INTEGER(our_int)                :: exp_B
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_ex_ante(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_ex_post(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_future_payoffs(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_eps_relevant(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: eps_relevant(:, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax(:, :)
    REAL(our_dble), ALLOCATABLE     :: dataset(:, :)
    
    REAL(our_dble)                  :: payoffs_ex_post(4)
    REAL(our_dble)                  :: payoffs_ex_ante(4)
    REAL(our_dble)                  :: future_payoffs(4)
    REAL(our_dble)                  :: total_payoffs(4)
    REAL(our_dble)                  :: disturbances(4)
    REAL(our_dble)                  :: emax_simulated
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: shocks(4, 4)
    REAL(our_dble)                  :: coeffs_A(6)
    REAL(our_dble)                  :: coeffs_B(6)
    REAL(our_dble)                  :: maximum
    REAL(our_dble)                  :: payoff
    REAL(our_dble)                  :: delta
    
    LOGICAL                         :: is_myopic
    LOGICAL                         :: is_huge
    
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Read specification of model
    CALL read_specification(num_periods, delta, coeffs_A, coeffs_B, & 
            coeffs_edu, edu_start, edu_max, coeffs_home, shocks, num_draws, & 
            seed_solution, num_agents, seed_simulation) 

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

    ! Draw random disturbances. For debugging purposes, these might also be 
    ! read in from disk.
    CALL get_disturbances(periods_eps_relevant, shocks)

    ! Perform backward induction
    CALL backward_induction(periods_emax, periods_payoffs_ex_post, &
            periods_future_payoffs, num_periods, max_states_period, &
            periods_eps_relevant, num_draws, states_number_period, & 
            periods_payoffs_ex_ante, edu_max, edu_start, &
            mapping_state_idx, states_all, delta)

    ! Allocate additional containers
    ALLOCATE(dataset(num_agents * num_periods, 8))

    ! Simulate sample
    CALL simulate_sample(dataset, num_agents, states_all, num_periods, &
                mapping_state_idx, periods_payoffs_ex_ante, &
                periods_eps_relevant, edu_max, edu_start, periods_emax, delta)

    DO i = 1, 3
        DO j = 1, 10
            PRINT *, periods_eps_relevant(i, j, :)
        END DO
        PRINT *, ''
    END DO

    ! Read model specification
    OPEN(UNIT=1, FILE='data.robupy.dat')

    1600 FORMAT(3(1x,i5), 1x, A4, 4(1x,i5))

    DO i = 1, (num_agents * num_periods)

        WRITE(1, 1600) INT(dataset(i, 1)), INT(dataset(i, 2)), INT(dataset(i, 3)), &
            '.', INT(dataset(i, 5)), INT(dataset(i, 6)), INT(dataset(i, 7)), INT(dataset(i, 8))

    END DO







!******************************************************************************
!******************************************************************************
END PROGRAM