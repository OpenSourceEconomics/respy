!*******************************************************************************
! This module provides additional functions that allow to test this program 
! against the ROBUPY package.
!*******************************************************************************
MODULE PEI_ADDITIONS
 

  IMPLICIT NONE

  PUBLIC

  CONTAINS

!*******************************************************************************
!*******************************************************************************
SUBROUTINE READ_IN_DISTURBANCES(EU1, EU2, C, B)

  !/* external objects    */

  REAL, INTENT(INOUT)           :: EU1(:, :)
  REAL, INTENT(INOUT)           :: EU2(:, :)
  REAL, INTENT(INOUT)           :: C(:, :)
  REAL, INTENT(INOUT)           :: B(:, :)

  !/* internal objects    */

  LOGICAL                       :: READ_IN

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 
   
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
!*******************************************************************************
!*******************************************************************************
END MODULE
