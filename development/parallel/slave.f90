!*******************************************************************************
!******************************************************************************* 
PROGRAM slave

    !/* external modules        */

    USE shared_constants
    
    USE shared_auxiliary

    USE solve_auxiliary

    USE solve_fortran

    USE mpi
    
    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */

    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE   :: states_all(:, :, :)
    INTEGER(our_int), ALLOCATABLE   :: num_emax_slave(:, :)

    INTEGER(our_int)                :: num_draws_emax
    INTEGER(our_int)                :: num_draws_prob
    INTEGER(our_int)                :: num_agents_est
    INTEGER(our_int)                :: num_periods
    INTEGER(our_int)                :: num_points
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: edu_start
    INTEGER(our_int)                :: edu_max
    INTEGER(our_int)                :: min_idx, i, j, k

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_emax(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_prob(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax(:, :)
    REAL(our_dble), ALLOCATABLE     :: data_array(:, :)

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: crit_val
    REAL(our_dble)                  :: delta
    REAL(our_dble)                  :: tau

    LOGICAL                         :: is_interpolated
    LOGICAL                         :: is_myopic
    LOGICAL                         :: is_debug

    CHARACTER(10)                   :: request


INTEGER :: ierr, myrank, num_slaves, task, root = 0, parentcomm, nprocs
LOGICAL :: STAY_AVAILABLE = .TRUE.


    INTEGER(our_int), ALLOCATABLE                   :: states_all_tmp(:, :, :)

    INTEGER(our_int)                                :: max_states_period
    INTEGER(our_int)                                :: period
    INTEGER(our_int)                ::  test_gather_part
    INTEGER(our_int), ALLOCATABLE :: test_gather_all(:), rcounts(:), scounts(:), disps(:)

    INTEGER(our_int)                    :: seed_inflated(15)
    INTEGER(our_int)                    :: num_states
    INTEGER(our_int)                    :: seed_size, lower_bound, upper_bound


    REAL(our_dble)                      :: payoffs_systematic(4)
    REAL(our_dble)                      :: shocks_cov(4, 4)
    REAL(our_dble)                      :: emax_simulated
    REAL(our_dble)                      :: shifts(4)

    REAL(our_dble), ALLOCATABLE         :: exogenous(:, :)
    REAL(our_dble), ALLOCATABLE         :: predictions(:)
    REAL(our_dble), ALLOCATABLE         :: endogenous(:)
    REAL(our_dble), ALLOCATABLE         :: maxe(:), periods_emax_slaves(:)

    LOGICAL                             :: any_interpolated

    LOGICAL, ALLOCATABLE                :: is_simulated(:)

    REAL(our_dble), ALLOCATABLE  :: draws_emax(:, :)


CALL MPI_INIT(ierr)
CALL MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
CALL MPI_COMM_SIZE(MPI_COMM_WORLD, num_slaves, ierr)
CALL MPI_COMM_GET_PARENT(parentcomm, ierr)

ALLOCATE(test_gather_all(num_slaves))
ALLOCATE(rcounts(num_slaves), scounts(num_slaves), disps(num_slaves))

CALL read_specification(num_periods, delta, coeffs_a, coeffs_b, coeffs_edu, &
    edu_start, edu_max, coeffs_home, shocks_cholesky, num_draws_emax, & 
    seed_emax, seed_prob, num_agents_est, is_debug, is_interpolated, & 
    num_points, min_idx, request, num_draws_prob, is_myopic, tau)

ALLOCATE(draws_emax(num_draws_emax, 4))

! Allocate arrays
ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
ALLOCATE(states_all_tmp(num_periods, 100000, 4))
ALLOCATE(states_number_period(num_periods))

CALL fort_create_state_space(states_all_tmp, states_number_period, &
            mapping_state_idx, max_states_period, num_periods, edu_start, &
            edu_max)

ALLOCATE(periods_emax(num_periods, max_states_period))

ALLOCATE(periods_emax_slaves(max_states_period))


states_all = states_all_tmp(:, :max_states_period, :)
DEALLOCATE(states_all_tmp)

ALLOCATE(periods_payoffs_systematic(num_periods, max_states_period, 4))

! Calculate the systematic payoffs
CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, &
        num_periods, states_number_period, states_all, edu_start, &
        coeffs_a, coeffs_b, coeffs_edu, coeffs_home)


! This part creates (or reads from disk) the draws for the Monte 
! Carlo integration of the EMAX. For is_debugging purposes, these might 
! also be read in from disk or set to zero/one.   
CALL create_draws(periods_draws_emax, num_periods, num_draws_emax, seed_emax, is_debug)


CALL determine_workload(num_emax_slave, num_periods, num_slaves, states_number_period)









task = -99

        ! Allocate arrays
        !ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
        !ALLOCATE(states_all_tmp(num_periods, 100000, 4))
        !ALLOCATE(states_number_period(num_periods))

    periods_emax = MISSING_FLOAT

DO WHILE (STAY_AVAILABLE)  
    
    CALL MPI_Bcast(task, 1, MPI_INT, 0, parentcomm, ierr)


    IF(task == 1) THEN


        !PRINT *, 'shutting down'!, test_gather_all, test_gather_part, myrank

        CALL MPI_FINALIZE(ierr)
        STAY_AVAILABLE = .FALSE.

    ELSEIF(task == 2) THEN

        ! Construct auxiliary objects
        shocks_cov = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))

        DO period = (num_periods - 1), 0, -1

            ! Extract draws and construct auxiliary objects
            draws_emax = periods_draws_emax(period + 1, :, :)
            num_states = states_number_period(period + 1)

            ! Upper and lower bound of tasks
            lower_bound = SUM(num_emax_slave(period + 1, :myrank))
            upper_bound = SUM(num_emax_slave(period + 1, :myrank + 1))
 
            DO k = lower_bound, upper_bound - 1

                ! Extract payoffs
                payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

                CALL get_payoffs(emax_simulated, num_draws_emax, draws_emax, &
                        period, k, payoffs_systematic, edu_max, edu_start, &
                        mapping_state_idx, states_all, num_periods, &
                        periods_emax, delta, shocks_cholesky)

                ! Collect information
                periods_emax_slaves(k + 1) = emax_simulated

            END DO

            scounts(:) = num_emax_slave(period + 1, :)
            rcounts(:) = scounts
            DO j = 1, num_slaves
                disps(j) = SUM(scounts(:j - 1)) 
            END DO
            
    CALL MPI_ALLGATHERV(periods_emax_slaves(lower_bound + 1:upper_bound), & 
        scounts(myrank + 1), MPI_DOUBLE, periods_emax(period + 1, :), & 
        rcounts, disps, MPI_DOUBLE, MPI_COMM_WORLD, ierr)

    
            ! The leading slave updates the master period by period.
            IF (myrank == 0) THEN
                CALL MPI_SEND(periods_emax(period + 1, :num_states) , num_states, MPI_DOUBLE, 0, period, parentcomm, ierr)            
            END IF

            END DO
      !----------------------------------------------------------------------
      !----------------------------------------------------------------------
    END IF    

END DO





END PROGRAM
!******************************************************************************
!****************************************************************************** 
