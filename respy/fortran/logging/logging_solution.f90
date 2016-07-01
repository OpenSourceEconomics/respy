!******************************************************************************
!******************************************************************************
MODULE logging_solution

	!/*	external modules	*/

    USE shared_containers 

    USE shared_constants

    USE shared_auxiliary 

    USE shared_utilities

	!/*	setup	*/

    IMPLICIT NONE

    PRIVATE

    PUBLIC :: log_solution

    !/* explicit interface   */

    INTERFACE log_solution

        MODULE PROCEDURE log_solution_progress, log_prediction_model

    END INTERFACE

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE log_solution_progress(indicator, period, num_states)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)            :: indicator

    INTEGER(our_int), INTENT(IN), OPTIONAL  :: num_states
    INTEGER(our_int), INTENT(IN), OPTIONAL  :: period

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    OPEN(UNIT=99, FILE='sol.respy.log', ACCESS='APPEND')

    IF (indicator == 1) THEN

        CLOSE(99, STATUS ='DELETE')
        OPEN(UNIT=99, FILE='sol.respy.log', ACCESS='APPEND')

        WRITE(99, *) ' Starting state space creation '
        WRITE(99, *)

    ELSEIF (indicator == 2) THEN

        WRITE(99, *) ' Starting calculation of systematic payoffs '
        WRITE(99, *)

    ELSEIF (indicator == 3) THEN

        WRITE(99, *) ' Starting backward induction procedure '
        WRITE(99, *)

    ELSEIF (indicator == 4) THEN

        1900 FORMAT(2x,A18,1x,i2,1x,A4,1x,i5,1x,A7)

        WRITE(99, 1900) '... solving period', period, 'with', num_states, 'states '
        WRITE(99, *)

    ELSEIF (indicator == -1) THEN

        WRITE(99, *) ' ... finished '
        WRITE(99, *) ''
        WRITE(99, *)

    ELSEIF (indicator == -2) THEN

        WRITE(99, *) ' ... not required due to myopic agents '
        WRITE(99, *) ''
        WRITE(99, *)

    END IF

  CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE log_prediction_model(coeffs, r_squared, bse)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: coeffs(9)
    REAL(our_dble), INTENT(IN)      :: r_squared
    REAL(our_dble), INTENT(IN)      :: bse(9)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    100 FORMAT(8x,A12,7x,9(f15.4))
    110 FORMAT(8x,A15,4x,9(f15.4))
    120 FORMAT(8x,A9,10x,f15.4)

    OPEN(UNIT=99, FILE='sol.respy.log', ACCESS='APPEND')

        WRITE(99, *) '     Information about Prediction Model '
        WRITE(99, *) 

        WRITE(99, 100) 'Coefficients', coeffs
        WRITE(99, *) 

        WRITE(99, 110) 'Standard Errors', bse
        WRITE(99, *) 

        WRITE(99, 120) 'R-squared', r_squared
        WRITE(99, *) ''
        WRITE(99, *) 

    CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE