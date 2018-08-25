!******************************************************************************
! This module provides additional functions that allow to test this program
! against the RESPY package.
!******************************************************************************
MODULE TEST_ADDITIONS

  IMPLICIT NONE

  PUBLIC

  CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE READ_IN_DISTURBANCES_ESTIMATION(RNN, RNNL, NPER, DRAW, DRAW1)

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
    INTEGER                   :: U

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    2001 FORMAT(4(1x,f15.10))

    ! Check applicability
    INQUIRE(FILE='.draws.respy.test', EXIST=READ_IN)

    IF (READ_IN) THEN

        OPEN(NEWUNIT=U, FILE='.draws.respy.test', ACTION='READ')

            REWIND(UNIT=U)
            DO T = 1, NPER
                DO J = 1, INT(DRAW)
                    READ(U, 2001) RNN(J, T, :)
                END DO
            END DO

            REWIND(UNIT=U)
            DO T = 1, NPER
                DO J = 1, INT(DRAW1)
                    READ(U, 2001) RNNL(J, T, :)
                END DO
            END DO

        CLOSE(U)

    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE READ_IN_DISTURBANCES(EU1, EU2, C, B)

  !/* external objects    */

  REAL, INTENT(INOUT)       :: EU1(:, :)
  REAL, INTENT(INOUT)       :: EU2(:, :)
  REAL, INTENT(INOUT)       :: C(:, :)
  REAL, INTENT(INOUT)       :: B(:, :)

  !/* internal objects    */

  LOGICAL                   :: READ_IN

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
  ! Check applicability
  INQUIRE(FILE='.restud.testing.scratch', EXIST=READ_IN)

  IF (READ_IN) THEN

    OPEN(12, file='.restud.testing.scratch')

    CLOSE(12, STATUS='delete')

    EU1 = 1.0

    EU2 = 1.0

    C   = 0.0

    B   = 0.0

  END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
