!******************************************************************************
!******************************************************************************
MODULE recording_warning

	!/*	external modules	*/

    USE shared_containers 

    USE shared_constants

    USE shared_auxiliary 

    USE shared_utilities

	!/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE record_warning_crit_val(count)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: count

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND')

        IF (count == 1) WRITE(99, *) '  Warning: Starting value of criterion function too large to write to file, internals unaffected.'

        IF (count == 2) WRITE(99, *) '  Warning: Step value of criterion function too large to write to file, internals unaffected.'

        IF (count == 3) WRITE(99, *) '  Warning: Current value of criterion function too large to write to file, internals unaffected.'

        WRITE(99, *)

    CLOSE(1)


END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE