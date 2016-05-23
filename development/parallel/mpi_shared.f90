!*******************************************************************************
!*******************************************************************************

! This module contains functionality that is shared between the slave and masters. 
!
!
!
!
!
MODULE solve_auxiliary

    !/* external modules    */

    USE shared_auxiliary

    USE shared_constants

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE determine_workload(num_tasks_slaves, num_periods, num_slaves, states_number_period)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(OUT)   :: num_tasks_slaves(:, :)
    INTEGER(our_int), INTENT(IN)   :: num_periods
    INTEGER(our_int), INTENT(IN)   :: num_slaves
    INTEGER(our_int), INTENT(IN)   :: states_number_period(:)

    INTEGER(our_int)    :: j, i, period

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
ALLOCATE(num_tasks_slaves(num_periods, num_slaves))
num_tasks_slaves = zero_int


DO period = 1, num_periods
    j = 1
    DO i = 1, states_number_period(period)
        
        IF (j .GT. num_slaves) THEN
        
            j = 1

        END IF

        num_tasks_slaves(period, j) = num_tasks_slaves(period, j) + 1

        j = j + 1

    END DO
END DO


END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE