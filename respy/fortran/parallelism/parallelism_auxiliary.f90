!******************************************************************************
!******************************************************************************
MODULE parallel_auxiliary

    !/* external modules    */

    USE parallel_constants

    USE resfort_library

    USE mpi

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE distribute_information(num_emax_slaves, period, send_slave, recieve_slaves)
    
    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: num_emax_slaves(num_periods, num_slaves)
    INTEGER(our_int), INTENT(IN)    :: period
 
    REAL(our_dble), INTENT(IN)      :: send_slave(:)
    REAL(our_dble), INTENT(INOUT)   :: recieve_slaves(:)

    !/* internal objects        */

    INTEGER(our_int), ALLOCATABLE   :: rcounts(:)
    INTEGER(our_int), ALLOCATABLE   :: scounts(:)
    INTEGER(our_int), ALLOCATABLE   :: disps(:)

    INTEGER(our_int)                :: i


!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Parameterize the communication.
    ALLOCATE(rcounts(num_slaves), scounts(num_slaves), disps(num_slaves))
    scounts(:) = num_emax_slaves(period + 1, :)
    rcounts(:) = scounts
    DO i = 1, num_slaves
        disps(i) = SUM(scounts(:i - 1)) 
    END DO
    
    ! Aggregate the EMAX contributions across the slaves.    
    CALL MPI_ALLGATHERV(send_slave, scounts(rank + 1), MPI_DOUBLE, recieve_slaves, rcounts, disps, MPI_DOUBLE, MPI_COMM_WORLD, ierr)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE determine_workload(num_emax_slaves, states_number_period)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(OUT)   :: num_emax_slaves(:, :)
    
    INTEGER(our_int), INTENT(IN)    :: states_number_period(num_periods)

    !/* internal objects        */

    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ALLOCATE(num_emax_slaves(num_periods, num_slaves))

    num_emax_slaves = zero_int

    DO period = 1, num_periods

        j = 1

        DO i = 1, states_number_period(period)
            
            IF (j .GT. num_slaves) THEN
            
                j = 1

            END IF

            num_emax_slaves(period, j) = num_emax_slaves(period, j) + 1

            j = j + 1

        END DO
    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_solve_parallel(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_all(:, :, :)

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(:, :)

    REAL(our_dble), INTENT(IN)                      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)                      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)                      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)                      :: coeffs_b(6)

    !/* internal objects        */

    INTEGER(our_int)                                :: num_states
    INTEGER(our_int)                                :: period
    INTEGER(our_int)            :: status

    REAL(our_dble), ALLOCATABLE :: temporary_subset(:)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! If agents are not myopic, then we start a number of slaves and request to help in the calculation of the EMAX.
    IF (.NOT. is_myopic) THEN
        CALL MPI_COMM_SPAWN(TRIM(exec_dir) // '/resfort_parallel_slave', MPI_ARGV_NULL, (num_procs - 1), MPI_INFO_NULL, 0, MPI_COMM_WORLD, SLAVECOMM, MPI_ERRCODES_IGNORE, ierr)
        CALL MPI_Bcast(2, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
    END IF
  

    IF(is_myopic) CALL logging_solution(1)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, periods_emax, periods_payoffs_systematic)

    IF(is_myopic) CALL logging_solution(-1)



    IF(is_myopic) CALL logging_solution(2)

    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

    IF(is_myopic) CALL logging_solution(-1)



    ! The leading slave is kind enough to let the parent process know about the  intermediate outcomes.
    IF (is_myopic) THEN

        CALL logging_solution(3)
   
        ! All other objects remain set to MISSING_FLOAT. This align the treatment for the two special cases: (1) is_myopic and (2) is_interpolated.
        DO period = 1,  num_periods
            periods_emax(period, :states_number_period(period)) = zero_dble
        END DO
     
        CALL logging_solution(-2)

    ELSE

        DO period = (num_periods - 1), 0, -1

            num_states = states_number_period(period + 1)

            ALLOCATE(temporary_subset(num_states))
            CALL MPI_RECV(temporary_subset, num_states, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, SLAVECOMM, status, ierr)

            periods_emax(period + 1, :num_states) = temporary_subset

            DEALLOCATE(temporary_subset)
        END DO

        CALL logging_solution(-1)

        ! Shut down orderly
        CALL MPI_Bcast(1, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
        CALL MPI_FINALIZE (ierr)

    END IF


END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE