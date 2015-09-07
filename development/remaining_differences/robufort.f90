
!*******************************************************************************
!*******************************************************************************
PROGRAM robufort

USE robupy_program_constants
IMPLICIT NONE

INTEGER :: j, k, num_periods, edu_start, edu_max, num_draws, num_agents, & 
seed_simulation, seed_solution, period

1500 FORMAT(6(1x,f15.10))
1510 FORMAT(f15.10)

1505 FORMAT(i10)
1515 FORMAT(i10,1x,i10)

DOUBLE PRECISION :: coeffs_A(6), coeffs_B(6), delta, coeffs_edu(3), coeffs_home(1), shocks(4, 4)

INTEGER :: min_idx, max_states_period

    INTEGER                :: i
DOUBLE PRECISION                  :: total_payoffs(4), emax, payoff
   DOUBLE PRECISION                   :: disturbances(4)
   DOUBLE PRECISION                  :: maximum, emax_simulated
   DOUBLE PRECISION                  :: payoffs_ex_post(4), payoffs_ex_ante(4), future_payoffs(4)
    INTEGER                :: exp_A
    INTEGER               :: exp_B
    INTEGER                :: edu, covars(6)
    INTEGER       :: edu_lagged
    INTEGER               :: future_idx, total
LOGICAL                         :: is_huge
! Auxiliary objects

INTEGER, ALLOCATABLE  :: states_all(:, :, :)
INTEGER, ALLOCATABLE  :: states_number_period(:)
INTEGER, ALLOCATABLE  :: mapping_state_idx(:, :, :, :, :)
DOUBLE PRECISION, ALLOCATABLE  :: periods_payoffs_ex_ante(:, :, :)
DOUBLE PRECISION, ALLOCATABLE  :: periods_emax(:, :), periods_payoffs_ex_post(:, :, :)
DOUBLE PRECISION, ALLOCATABLE :: periods_future_payoffs(:, :, :)
DOUBLE PRECISION, ALLOCATABLE :: eps_relevant_periods(:, :, :), eps_relevant(:, :)

LOGICAL :: is_myopic


OPEN(UNIT=1, FILE='model.robufort.ini')

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

! Auxiliary
min_idx = MIN(num_periods, (edu_max - edu_start + 1))


! Allocate arrays
ALLOCATE(states_all(num_periods, 100000, 4))
ALLOCATE(states_number_period(num_periods))
ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))

! Create the state space of the model
CALL create_state_space(states_all, states_number_period, mapping_state_idx, & 
        num_periods, edu_start, edu_max, min_idx)

! Allocate additional containers
max_states_period = MAXVAL(states_number_period)
ALLOCATE(periods_payoffs_ex_ante(num_periods, max_states_period, 4))

! Calculate the ex ante payoffs
CALL calculate_payoffs_ex_ante(periods_payoffs_ex_ante, num_periods, &
        states_number_period, states_all, edu_start, coeffs_A, coeffs_B, & 
        coeffs_edu, coeffs_home, max_states_period)

! Allocate additional containers
ALLOCATE(periods_emax(num_periods, max_states_period))
ALLOCATE(periods_payoffs_ex_post(num_periods, max_states_period, 4))
ALLOCATE(periods_future_payoffs(num_periods, max_states_period, 4))
ALLOCATE(eps_relevant_periods(num_periods, num_draws, 4))

ALLOCATE(eps_relevant(num_draws, 4))

! Draw random disturbances
eps_relevant_periods = 1.00

CALL backward_induction(periods_emax, periods_payoffs_ex_post, &
                periods_future_payoffs, num_periods, max_states_period, &
                eps_relevant_periods, num_draws, states_number_period, & 
                periods_payoffs_ex_ante, edu_max, edu_start, &
                mapping_state_idx, states_all, delta)

!******************************************************************************
!******************************************************************************
END PROGRAM