!*******************************************************************************
!******************************************************************************* 
PROGRAM slave

USE mpi

USE shared_constants
USE shared_auxiliary
USE solve_fortran
USE solve_auxiliary


IMPLICIT NONE
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

call MPI_Init(ierr)
call MPI_Comm_Rank(MPI_COMM_WORLD, myrank, ierr)
call MPI_Comm_Size(MPI_COMM_WORLD, num_slaves, ierr)
CALL MPI_COMM_GET_PARENT(parentcomm, ierr)

PRINT *, 'How many of us are there? ', num_slaves
ALLOCATE(test_gather_all(num_slaves))
ALLOCATE(rcounts(num_slaves), scounts(num_slaves), disps(num_slaves))

CALL read_specification(num_periods, delta, coeffs_a, coeffs_b, coeffs_edu, &
    edu_start, edu_max, coeffs_home, shocks_cholesky, num_draws_emax, & 
    seed_emax, seed_prob, num_agents_est, is_debug, is_interpolated, & 
    num_points, min_idx, request, num_draws_prob, is_myopic, tau)


! Allocate arrays
ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
ALLOCATE(states_all_tmp(num_periods, 100000, 4))
ALLOCATE(states_number_period(num_periods))

CALL fort_create_state_space(states_all_tmp, states_number_period, &
            mapping_state_idx, max_states_period, num_periods, edu_start, &
            edu_max)

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









PRINT *, 'Greetings from slave ...', myrank
task = -99

        ! Allocate arrays
        !ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
        !ALLOCATE(states_all_tmp(num_periods, 100000, 4))
        !ALLOCATE(states_number_period(num_periods))


DO WHILE (STAY_AVAILABLE)  
    
    CALL MPI_Bcast(task, 1, MPI_INT, 0, parentcomm, ierr)


    IF(task == 1) THEN


        PRINT *, 'shutting down', test_gather_all, test_gather_part, myrank

        CALL MPI_FINALIZE(ierr)
        STAY_AVAILABLE = .FALSE.

    ELSEIF(task == 2) THEN

        IF (myrank == 1) PRINT *, ' Calculate EMAX ...'


        DO period = (num_periods - 1), 0, -1

            IF (myrank == 1) THEN
                PRINT *, 'Period', i
                PRINT *, ''
            END IF
    
            test_gather_part = myrank + 99

            scounts(:) = 1
            rcounts(:) = scounts(:) 


            DO j = 1, num_slaves
                disps(j) = SUM(scounts(:j))
            END DO

            CALL MPI_ALLGATHERV(test_gather_part, scounts(myrank + 1), MPI_INT, &
                test_gather_all, rcounts, disps - 1, MPI_INT, MPI_COMM_WORLD, ierr)


            ! One slave has to keep the the master updated.
            IF (myrank == 0) THEN

                ! TODO: This will not work with variable length?
                CALL MPI_SEND(test_gather_all, num_slaves, MPI_INT, 0, period, parentcomm, ierr)

            END IF

        END DO

        ! Cutting the states_all container to size. The required size is only known
        ! after the state space creation is completed.
        !ALLOCATE(states_all(num_periods, max_states_period, 4))
        !states_all = states_all_tmp(:, :max_states_period, :)
        !DEALLOCATE(states_all_tmp)

        ! Calculate the systematic payoffs
        !CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, &
        !        num_periods, states_number_period, states_all, edu_start, &
        !        coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

        ! Initialize containers, which contain a lot of missing values as we
        ! capture the tree structure in arrays of fixed dimension.
        !periods_emax = MISSING_FLOAT

    END IF    

END DO





CALL SLEEP(2)


END PROGRAM
!*******************************************************************************
!******************************************************************************* 
