!******************************************************************************
!******************************************************************************
MODULE recording_warning

  !/*	external modules	*/

    USE shared_constants

  !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE record_warning(count)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: count

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND', ACTION='WRITE')

        IF (count == 1) WRITE(99, *) '  Warning: Starting value of criterion function too large to write to file, internals unaffected.'

        IF (count == 2) WRITE(99, *) '  Warning: Step value of criterion function too large to write to file, internals unaffected.'

        IF (count == 3) WRITE(99, *) '  Warning: Current value of criterion function too large to write to file, internals unaffected.'

        IF (count == 4) WRITE(99, *) '  Warning: Stabilization of otherwise zero element on diagonal of Cholesky decomposition.'

        IF (count == 5) WRITE(99, *) '  Warning: Some agents have a numerically zero probability, stabilization of logarithm required.'

        WRITE(99, *)

    CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
