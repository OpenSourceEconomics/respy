!******************************************************************************
! This module provides additional functions that allow to test this program
! against the RESPY package.
!******************************************************************************
MODULE PEI_ADDITIONS

  IMPLICIT NONE

  PUBLIC

  CONTAINS

!******************************************************************************
!******************************************************************************
SUBROUTINE READ_IN_DISTURBANCES(RNN, RNNL, NPER, DRAW, DRAW1)

  !/* external objects    */

  REAL, INTENT(INOUT)       :: RNN(:, :, :)
  REAL, INTENT(INOUT)       :: RNNL(:, :, :)

  INTEGER, INTENT(IN)       :: NPER

  REAL, INTENT(IN)          :: DRAW
  REAL, INTENT(IN)          :: DRAW1

  !/* internal objects    */

  LOGICAL                   :: READ_IN

  INTEGER                   :: T
  INTEGER                   :: J

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

  2001 FORMAT(4(1x,f15.10))

  ! Check applicability
  INQUIRE(FILE='draws.respy.test', EXIST=READ_IN)

  IF (READ_IN) THEN

      OPEN(UNIT=99, FILE='draws.respy.test', ACTION='READ')

          REWIND(UNIT=99)
          DO T = 1, NPER
              DO J = 1, DRAW
                  READ(99, 2001) RNN(J, T, :)
              END DO
          END DO

          REWIND(UNIT=99)
          DO T = 1, NPER
              DO J = 1, DRAW1
                  READ(99, 2001) RNNL(J, T, :)
              END DO
          END DO

      CLOSE(99)

  END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
