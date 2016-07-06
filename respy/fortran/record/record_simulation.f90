!******************************************************************************
!******************************************************************************
MODULE recording_simulation

	!/*	external modules	*/

    USE shared_constants

	!/*	setup	*/

    IMPLICIT NONE

    PRIVATE

    PUBLIC :: record_simulation

    !/* explicit interface   */

    INTERFACE record_simulation

        MODULE PROCEDURE record_simulation_progress

    END INTERFACE

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE record_simulation_progress(i)

    !/* external objects        */
    
    INTEGER(our_int), INTENT(IN)    :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF ((i .NE. zero_int) .AND. (MOD(i, 100) == zero_dble)) THEN

        100 FORMAT(A16,i10,A7)

        OPEN(UNIT=99, FILE='sim.respy.log', ACCESS='APPEND')

            WRITE(99, 100) ' ... simulated ', i, ' agents'
            WRITE(99, *) 

        CLOSE(99)

    ELSE

        RETURN

    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE