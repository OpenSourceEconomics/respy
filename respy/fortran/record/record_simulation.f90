!******************************************************************************
!******************************************************************************
MODULE recording_simulation

    !/*	external modules	   */

    USE shared_interface

    !/*	setup	              */

    IMPLICIT NONE

    PRIVATE

    PUBLIC :: record_simulation

    !/* explicit interface   */

    INTERFACE record_simulation

        MODULE PROCEDURE record_simulation_progress, record_simulation_start, record_simulation_stop

    END INTERFACE

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE record_simulation_start(num_agents_sim, seed_sim, file_sim)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: num_agents_sim
    INTEGER(our_int), INTENT(IN)    :: seed_sim

    CHARACTER(225), INTENT(IN)      :: file_sim

    !/* internal objects        */

    INTEGER(our_int)                :: u

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    100 FORMAT(2x,A32,1x,i8,1x,A16,1x,i8)

    OPEN(NEWUNIT=u, FILE=TRIM(file_sim)//'.respy.sim', ACTION='WRITE')

        WRITE(u, 100) 'Starting simulation of model for', num_agents_sim, 'agents with seed', seed_sim
        WRITE(u, *)

    CLOSE(u)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_simulation_stop(file_sim)

    !/* external objects        */

    CHARACTER(225), INTENT(IN)      :: file_sim

    !/* internal objects        */

    INTEGER(our_int)                :: u

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    OPEN(NEWUNIT=u, FILE=TRIM(file_sim)//'.respy.sim', POSITION='APPEND', ACTION='WRITE')

        WRITE(u, *) ' ... finished'
        WRITE(u, *)

    CLOSE(u)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_simulation_progress(i, file_sim)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: i

    CHARACTER(225), INTENT(IN)      :: file_sim

    !/* internal objects        */

    INTEGER(our_int)                :: u

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF ((i .NE. zero_int) .AND. (MOD(i, 100) == zero_dble)) THEN

        100 FORMAT(A16,i10,A7)

        OPEN(NEWUNIT=u, FILE=TRIM(file_sim)//'.respy.sim', POSITION='APPEND', ACTION='WRITE')

            WRITE(u, 100) ' ... simulated ', i, ' agents'
            WRITE(u, *)

        CLOSE(u)

    ELSE

        RETURN

    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
