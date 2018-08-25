!******************************************************************************
!******************************************************************************
MODULE recording_solution

    !/*	external modules	*/

    USE shared_interface

  !/*	setup	*/

    IMPLICIT NONE

    PRIVATE

    PUBLIC :: record_solution

    !/* explicit interface   */

    INTERFACE record_solution

        MODULE PROCEDURE record_solution_progress, record_solution_prediction

    END INTERFACE

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE record_solution_progress(indicator, file_sim, period, num_states)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)            :: indicator

    INTEGER(our_int), INTENT(IN), OPTIONAL  :: num_states
    INTEGER(our_int), INTENT(IN), OPTIONAL  :: period

    CHARACTER(225), INTENT(IN)              :: file_sim

    !/* internal objects        */

    INTEGER(our_int)                        :: u

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    OPEN(NEWUNIT=u, FILE=TRIM(file_sim)//'.respy.sol', POSITION='APPEND', ACTION='WRITE')

    IF (indicator == 1) THEN

        CLOSE(u, STATUS ='DELETE')
        OPEN(NEWUNIT=u, FILE=TRIM(file_sim)//'.respy.sol', ACTION='WRITE')

        WRITE(u, *) ' Starting state space creation'
        WRITE(u, *)

    ELSEIF (indicator == 2) THEN

        WRITE(u, *) ' Starting calculation of systematic rewards'
        WRITE(u, *)

    ELSEIF (indicator == 3) THEN

        WRITE(u, *) ' Starting backward induction procedure'
        WRITE(u, *)

    ELSEIF (indicator == 4) THEN

        1900 FORMAT(2x,A18,1x,i2,1x,A4,1x,i7,1x,A7)

        WRITE(u, 1900) '... solving period', period, 'with', num_states, 'states'
        WRITE(u, *)

    ELSEIF (indicator == -1) THEN

        WRITE(u, *) ' ... finished'
        WRITE(u, *)
        WRITE(u, *)

    ELSEIF (indicator == -2) THEN

        WRITE(u, *) ' ... not required due to myopic agents'
        WRITE(u, *)

    END IF

  CLOSE(u)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_solution_prediction(coeffs, r_squared, bse, file_sim)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: coeffs(9)
    REAL(our_dble), INTENT(IN)      :: r_squared
    REAL(our_dble), INTENT(IN)      :: bse(9)

    CHARACTER(225), INTENT(IN)      :: file_sim

    !/* internal objects        */

    INTEGER(our_int)                :: u

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    100 FORMAT(8x,A12,7x,9(f15.4))
    110 FORMAT(8x,A15,4x,9(f15.4))
    120 FORMAT(8x,A9,10x,f15.4)

    OPEN(NEWUNIT=u, FILE=TRIM(file_sim)//'.respy.sol', POSITION='APPEND', ACTION='WRITE')

        WRITE(u, *) '     Information about Prediction Model '
        WRITE(u, *)

        WRITE(u, 100) 'Coefficients', coeffs
        WRITE(u, *)

        WRITE(u, 110) 'Standard Errors', bse
        WRITE(u, *)

        WRITE(u, 120) 'R-squared', r_squared
        WRITE(u, *) ''
        WRITE(u, *)

    CLOSE(u)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
