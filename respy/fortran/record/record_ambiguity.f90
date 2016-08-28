!******************************************************************************
!******************************************************************************
MODULE recording_ambiguity

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
SUBROUTINE record_ambiguity(period, k, x_shift, div, is_success, message)

    !/* external objects    */

    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: k

    REAL(our_dble)                  :: x_shift(2)
    REAL(our_dble)                  :: div

    CHARACTER(100)                  :: message

    LOGICAL                         :: is_success

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    100 FORMAT(1x,A6,i7,2x,A5,i7)
    110 FORMAT(3x,A12,f10.5)
    120 FORMAT(3x,A12,f10.5)
    130 FORMAT(3x,A7,8x,A5,20x)
    140 FORMAT(3x,A7,8x,A100)

    OPEN(UNIT=99, FILE='amb.respy.log', ACCESS='APPEND', ACTION='WRITE')

        WRITE(99, 100) 'PERIOD', period, 'STATE', k

        WRITE(99, *)
        WRITE(99, 110) 'Occupation A', x_shift(1)
        WRITE(99, 110) 'Occupation B', x_shift(2)

        WRITE(99, *)
        WRITE(99, 120) 'Divergence  ', div

        WRITE(99, *)

        IF(is_success) THEN
            WRITE(99, 130) 'Success', 'True '
        ELSE
            WRITE(99, 130) 'Success', 'False'
        END IF

        WRITE(99, 140) 'Message', ADJUSTL(message)
        WRITE(99, *)
        WRITE(99, *)

    CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
