!******************************************************************************
!******************************************************************************
MODULE recording_ambiguity

    !/*	external modules	*/

    USE shared_constants

    !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE record_ambiguity(opt_ambi_details, states_number_period, file_sim)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)      :: opt_ambi_details(num_periods, max_states_period, 5)

    INTEGER(our_int), INTENT(IN)    :: states_number_period(:)

    CHARACTER(225), INTENT(IN)      :: file_sim

    !/* internal objects    */

    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: mode
    INTEGER(our_int)                :: k

    REAL(our_dble)                  :: x_shift(2)
    REAL(our_dble)                  :: div(1)

    LOGICAL                         :: is_success

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    100 FORMAT(1x,A6,i7,2x,A5,i7)
    110 FORMAT(3x,A12,f10.5)
    120 FORMAT(3x,A12,f10.5)
    130 FORMAT(3x,A7,8x,A5,20x)
    140 FORMAT(3x,A7,8x,A100)

    OPEN(UNIT=99, FILE=TRIM(file_sim)//'.respy.amb', ACCESS='APPEND', ACTION='WRITE')

    DO period = num_periods - 1, 0, -1

        DO k = 0, (states_number_period(period + 1) - 1)

            WRITE(99, 100) 'PERIOD', period, 'STATE', k

            x_shift = opt_ambi_details(period + 1, k + 1, :2)
            div = opt_ambi_details(period + 1, k + 1, 3)
            is_success = (opt_ambi_details(period + 1, k + 1 , 4) == one_int)
            mode = DINT(opt_ambi_details(period + 1, k + 1, 5))

            ! We need to skip states that where not analyzed during an interpolation.
            IF (mode == MISSING_FLOAT) CYCLE

            WRITE(99, *)
            WRITE(99, 110) 'Occupation A', x_shift(1)
            WRITE(99, 110) 'Occupation B', x_shift(2)

            WRITE(99, *)
            WRITE(99, 120) 'Divergence  ', div(1)

            WRITE(99, *)

            IF(is_success) THEN
                WRITE(99, 130) 'Success', 'True '
            ELSE
                WRITE(99, 130) 'Success', 'False'
            END IF

                WRITE(99, 140) 'Message', ADJUSTL(get_message(mode))
                WRITE(99, *)
                WRITE(99, *)

        END DO

    END DO

    CLOSE(99)

    CALL record_ambiguity_summary(opt_ambi_details, states_number_period, file_sim)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_ambiguity_summary(opt_ambi_details, states_number_period, file_sim)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)          :: opt_ambi_details(num_periods, max_states_period, 5)

    INTEGER(our_int), INTENT(IN)        :: states_number_period(num_periods)

    CHARACTER(225), INTENT(IN)          :: file_sim

    !/* internal objects    */

    INTEGER(our_int)                    :: total
    INTEGER(our_int)                    :: period

    REAL(our_dble)                      :: success
    REAL(our_dble)                      :: failure

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    100 FORMAT(i10,1x,i10,1x,f10.2,1x,f10.2)

    OPEN(UNIT=99, FILE=TRIM(file_sim)//'.respy.amb', ACTION='WRITE', STATUS='OLD', ACCESS='APPEND')

        WRITE(99, *) 'SUMMARY'
        WRITE(99, *)
        WRITE(99, *) '   Period      Total    Success    Failure'
        WRITE(99, *)

        DO period = num_periods - 1, 0, -1
            total = states_number_period(period + 1)
            success = COUNT(opt_ambi_details(period + 1,:total, 4) == one_int) / DBLE(total)
            failure = COUNT(opt_ambi_details(period + 1,:total, 4) == zero_int) / DBLE(total)
            WRITE(99, 100) period, total, success, failure
        END DO

        WRITE(99, *)

    CLOSE(99)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
FUNCTION get_message(mode)

    !/* external objects        */

    CHARACTER(100)                  :: get_message

    INTEGER(our_int), INTENT(IN)    :: mode

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Optimizer get_message
    IF (mode == -1) THEN
        get_message = 'Gradient evaluation required (g & a)'
    ELSEIF (mode == 0) THEN
        get_message = 'Optimization terminated successfully'
    ELSEIF (mode == 1) THEN
        get_message = 'Function evaluation required (f & c)'
    ELSEIF (mode == 2) THEN
        get_message = 'More equality constraints than independent variables'
    ELSEIF (mode == 3) THEN
        get_message = 'More than 3*n iterations in LSQ subproblem'
    ELSEIF (mode == 4) THEN
        get_message = 'Inequality constraints incompatible'
    ELSEIF (mode == 5) THEN
        get_message = 'Singular matrix E in LSQ subproblem'
    ELSEIF (mode == 6) THEN
        get_message = 'Singular matrix C in LSQ subproblem'
    ELSEIF (mode == 7) THEN
        get_message = 'Rank-deficient equality constraint subproblem HFTI'
    ELSEIF (mode == 8) THEN
        get_message = 'Positive directional derivative for linesearch'
    ELSEIF (mode == 9) THEN
        get_message = 'Iteration limit exceeded'

    ! The following are project-specific return codes.
    ELSEIF (mode == 15) THEN
        get_message = 'No random variation in shocks'
    ELSEIF (mode == 16) THEN
        get_message = 'Optimization terminated successfully'
    ELSE
        STOP 'Misspecified mode'
    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
END MODULE
