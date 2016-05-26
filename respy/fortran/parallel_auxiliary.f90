!*******************************************************************************
!******************************************************************************* 
MODULE parallel_auxiliary

    !/* external modules    */

    USE resfort_library

    USE mpi

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE fort_solve_parallel(periods_payoffs_systematic, states_number_period, &
                mapping_state_idx, periods_emax, states_all, coeffs_a, &
                coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, &
                num_draws_emax, & 
                num_periods, num_points, edu_start, is_myopic, is_debug, & 
                min_idx, num_procs, SLAVECOMM, exec_dir)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_all(:, :, :)

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(:, :)

    INTEGER(our_int), INTENT(IN)                    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)                    :: num_periods
    INTEGER(our_int), INTENT(IN)                    :: num_points
    INTEGER(our_int), INTENT(IN)                    :: edu_start
    INTEGER(our_int), INTENT(IN)                    :: min_idx

    REAL(our_dble), INTENT(IN)                      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)                      :: coeffs_home(:)
    REAL(our_dble), INTENT(IN)                      :: coeffs_edu(:)
    REAL(our_dble), INTENT(IN)                      :: coeffs_a(:)
    REAL(our_dble), INTENT(IN)                      :: coeffs_b(:)

    LOGICAL, INTENT(IN)                             :: is_myopic
    LOGICAL, INTENT(IN)                             :: is_debug

    !NEw external
    INTEGER(our_int), INTENT(IN)                    :: num_procs, SLAVECOMM


    !/* internal objects        */

    INTEGER(our_int), ALLOCATABLE                   :: states_all_tmp(:, :, :)

    INTEGER(our_int)                                :: max_states_period
    INTEGER(our_int)                                :: period


    !NEW internal
    INTEGER(our_int)                                :: ierr
    INTEGER(our_int)                                :: num_states, status, task

  CHARACTER(len=225), INTENT(IN) :: exec_dir

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! If agents are not myopic, then we start a number of slaves and request to 
    ! help in the calculation of the EMAX.
    IF (.NOT. is_myopic) THEN
     
        CALL MPI_COMM_SPAWN(TRIM(exec_dir) // '/resfort_parallel_slave', & 
                MPI_ARGV_NULL, num_procs - 1, MPI_INFO_NULL, 0, & 
                MPI_COMM_WORLD, SLAVECOMM, MPI_ERRCODES_IGNORE, ierr)

        task = 2
        CALL MPI_Bcast(task, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
     
    END IF
    
    ! While we are waiting for the slaves to work on the EMAX calculation, the
    ! master can get some work done.

    ! Allocate arrays
    ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
    ALLOCATE(states_all_tmp(num_periods, 100000, 4))
    ALLOCATE(states_number_period(num_periods))

    IF(is_myopic) CALL logging_solution(1)

    CALL fort_create_state_space(states_all_tmp, states_number_period, &
            mapping_state_idx, max_states_period, num_periods, edu_start)

    IF(is_myopic) CALL logging_solution(-1)

    ALLOCATE(periods_emax(num_periods, max_states_period))

    ! Calculate the systematic payoffs
    ALLOCATE(states_all(num_periods, max_states_period, 4))
    states_all = states_all_tmp(:, :max_states_period, :)
    DEALLOCATE(states_all_tmp)

    ALLOCATE(periods_payoffs_systematic(num_periods, max_states_period, 4))

    IF(is_myopic) CALL logging_solution(2)
    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, &
            num_periods, states_number_period, states_all, edu_start, &
            coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

    IF(is_myopic) CALL logging_solution(-1)

    periods_emax = MISSING_FLOAT

    ! Perform backward induction procedure.
    

    ! The leading slave is kind enough to let the parent process know about the 
    ! intermediate outcomes.
    IF (is_myopic) THEN
        CALL logging_solution(3)
        
        CALL logging_solution(-2)

        ! All other objects remain set to MISSING_FLOAT. This align the
        ! treatment for the two special cases: (1) is_myopic and (2)
        ! is_interpolated.
        DO period = 1,  num_periods
            periods_emax(period, :states_number_period(period)) = zero_dble
        END DO

    ELSE

        DO period = (num_periods - 1), 0, -1

            num_states = states_number_period(period + 1)

            
                

            CALL MPI_RECV(periods_emax(period + 1, :num_states), num_states, & 
                    MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, SLAVECOMM, status, & 
                    ierr)

        END DO

        CALL logging_solution(-1)

        ! Shut down orderly
        task = 1
        CALL MPI_Bcast(task, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
        CALL MPI_FINALIZE (ierr)

    END IF


END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE