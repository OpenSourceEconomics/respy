!******************************************************************************
!******************************************************************************
MODULE logging_warning

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
SUBROUTINE log_warning_crit_val()

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND')

        WRITE(99, *) '  Warning: Value of criterion function too large to write to file, internals unaffected.'

    CLOSE(1)


END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE