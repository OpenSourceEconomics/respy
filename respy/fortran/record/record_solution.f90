!******************************************************************************
!******************************************************************************
MODULE recording_solution

  !/*	external modules	*/

    USE shared_containers

    USE shared_constants

    USE shared_auxiliary

    USE shared_utilities

  !/*	setup	*/

    IMPLICIT NONE

    PRIVATE

    PUBLIC :: record_solution, record_solution_time

    !/* explicit interface   */

    INTERFACE record_solution

        MODULE PROCEDURE record_solution_progress, record_prediction_model

    END INTERFACE

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE record_solution_progress(indicator, period, num_states)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)            :: indicator

    INTEGER(our_int), INTENT(IN), OPTIONAL  :: num_states
    INTEGER(our_int), INTENT(IN), OPTIONAL  :: period

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    OPEN(UNIT=99, FILE='sol.respy.log', ACCESS='APPEND', ACTION='WRITE')

    IF (indicator == 1) THEN

        CLOSE(99, STATUS ='DELETE')
        OPEN(UNIT=99, FILE='sol.respy.log', ACTION='WRITE')

        WRITE(99, *) ' Starting state space creation'
        WRITE(99, *)

    ELSEIF (indicator == 2) THEN

        WRITE(99, *) ' Starting calculation of systematic payoffs'
        WRITE(99, *)

    ELSEIF (indicator == 3) THEN

        WRITE(99, *) ' Starting backward induction procedure'
        WRITE(99, *)

    ELSEIF (indicator == 4) THEN

        1900 FORMAT(2x,A18,1x,i2,1x,A4,1x,i5,1x,A7)

        WRITE(99, 1900) '... solving period', period, 'with', num_states, 'states'
        WRITE(99, *)

    ELSEIF (indicator == -1) THEN

        WRITE(99, *) ' ... finished'
        WRITE(99, *)
        WRITE(99, *)

    ELSEIF (indicator == -2) THEN

        WRITE(99, *) ' ... not required due to myopic agents'
        WRITE(99, *)
        WRITE(99, *)

    END IF

  CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_prediction_model(coeffs, r_squared, bse)

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

    OPEN(UNIT=99, FILE='sol.respy.log', ACCESS='APPEND', ACTION='WRITE')

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
! The following two functions are only added to address the referees requests for the structRecomputation resbumission.
!******************************************************************************
!******************************************************************************
SUBROUTINE record_solution_time(which)

    !/* external objects        */

    CHARACTER(*), INTENT(IN)   :: which

    !/* internal objects        */

    CHARACTER(55)               :: today
    CHARACTER(55)               :: now

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

  115 FORMAT(3x,A5,6X,A10,5X,A8)
  125 FORMAT(3x,A4,7X,A10,5X,A8)

  CALL get_time_structRecomputation(today, now)

  IF (which == 'Start') THEN
    OPEN(UNIT=99, FILE='.structRecomputation.performance', ACTION='WRITE')
        WRITE(99, 115) which, today, now
  ELSE
    OPEN(UNIT=99, FILE='.structRecomputation.performance', ACCESS='APPEND', ACTION='WRITE')
        WRITE(99, 125) which, today, now
  END IF

  CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_time_structRecomputation(today_char, now_char)

    !/* external objects        */

    CHARACTER(*), INTENT(OUT)       :: today_char
    CHARACTER(*), INTENT(OUT)       :: now_char

    !/* internal objects        */

    INTEGER(our_int)                :: today(3)
    INTEGER(our_int)                :: now(3)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL IDATE(today)
    CALL ITIME(now)

    5503 FORMAT(i0.2,'/',i0.2,'/',i0.4)
    5504 FORMAT(i0.2,':',i0.2,':',i0.2)

    WRITE(today_char, 5503) today
    WRITE(now_char, 5504) now

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
