!******************************************************************************
!******************************************************************************
MODULE parallel_auxiliary

    !/* external modules        */

    USE parallel_constants

    USE resfort_library

    USE mpi

    !/* setup                   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE distribute_information(num_emax_slaves, period, send_slave, recieve_slaves)
    
    ! DEVELOPMENT NOTES
    !
    ! The assumed-shape input arguments allow to use this subroutine repeatedly.

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)        :: num_emax_slaves(num_periods, num_slaves)
    INTEGER(our_int), INTENT(IN)        :: period
 
    REAL(our_dble), INTENT(IN)          :: send_slave(:)
    REAL(our_dble), INTENT(INOUT)       :: recieve_slaves(:)

    !/* internal objects        */

    INTEGER(our_int), ALLOCATABLE       :: rcounts(:)
    INTEGER(our_int), ALLOCATABLE       :: scounts(:)
    INTEGER(our_int), ALLOCATABLE       :: disps(:)

    INTEGER(our_int)                    :: i


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
    
    CALL MPI_ALLGATHERV(send_slave, scounts(rank + 1), MPI_DOUBLE, recieve_slaves, rcounts, disps, MPI_DOUBLE, MPI_COMM_WORLD, ierr)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE determine_workload(jobs_slaves, jobs_total)

    !/* external objects        */

    INTEGER(our_int), INTENT(INOUT)     :: jobs_slaves(num_slaves)
    
    INTEGER(our_int), INTENT(IN)        :: jobs_total 

    !/* internal objects        */

    INTEGER(our_int)                    :: j
    INTEGER(our_int)                    :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    jobs_slaves = zero_int

    j = 1

    DO i = 1, jobs_total
            
        IF (j .GT. num_slaves) j = 1

        jobs_slaves(j) = jobs_slaves(j) + 1

        j = j + 1

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_evaluate_parallel(crit_val)

    !/* external objects        */

    REAL(our_dble), INTENT(INOUT)       :: crit_val

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Instruct slaves to assist in the calculation of the EMAX
    CALL MPI_Bcast(3, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)

    CALL MPI_RECV(crit_val, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, SLAVECOMM, status, ierr)

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

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
      
    CALL MPI_Bcast(2, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
    
    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, periods_emax, periods_payoffs_systematic)

    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

    DO period = (num_periods - 1), 0, -1

        num_states = states_number_period(period + 1)
        
        CALL MPI_RECV(periods_emax(period + 1, :num_states) , num_states, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, SLAVECOMM, status, ierr)
        
    END DO

    CALL logging_solution(-1)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE