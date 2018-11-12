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

    !/* internal objects        */

    INTEGER(our_int)                :: u
    
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    OPEN(NEWUNIT=u, FILE='est.respy.log', POSITION='APPEND', ACTION='WRITE')

        IF (count == 1) WRITE(u, *) '  Warning: Starting value of criterion function too large to write to file, internals unaffected.'

        IF (count == 2) WRITE(u, *) '  Warning: Step value of criterion function too large to write to file, internals unaffected.'

        IF (count == 3) WRITE(u, *) '  Warning: Current value of criterion function too large to write to file, internals unaffected.'

        IF (count == 4) WRITE(u, *) '  Warning: Stabilization of otherwise zero element on diagonal of Cholesky decomposition.'

        IF (count == 5) WRITE(u, *) '  Warning: Some agents have a numerically zero probability, stabilization of logarithm required.'

        WRITE(u, *)

    CLOSE(u)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
