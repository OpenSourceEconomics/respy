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

    IF(period == zero_int) CALL record_ambiguity_summary()

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_ambiguity_summary()

    !/* internal objects    */

    INTEGER(our_int), ALLOCATABLE   :: success_count(:, :)

    INTEGER(our_int)            :: total_count
    INTEGER(our_int)            :: iostat
    INTEGER(our_int)            :: period
    INTEGER(our_int)            :: i

    CHARACTER(200)              :: line

    LOGICAL                     :: success_info

    REAL(our_dble)               :: share_success
    REAL(our_dble)               :: share_failure

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    OPEN(UNIT=99, FILE='amb.respy.log', ACTION='READ', STATUS='OLD', ACCESS='sequential')

        DO

            READ(99, '(A)', IOSTAT=iostat, ADVANCE='NO')  line
            IF(iostat > 0) EXIT

            IF((line(:7) == ' PERIOD')) READ(line(13:15),*) period

            IF(.NOT. ALLOCATED(success_count)) THEN
                ALLOCATE(success_count(2, period + 1))
                success_count = zero_int
            END IF

            success_info = (line(4:11) == 'Success')
            IF(success_info .AND. (line(19:23) == 'True')) THEN
                success_count(1, period + 1) = success_count(1, period + 1) + 1
            ELSE IF(success_info) THEN
                success_count(2, period + 1) = success_count(2, period + 1) + 1
            END IF

        END DO

    CLOSE(99)


    100 FORMAT(i10,1x,i10,1x,f10.2,1x,f10.2)

    OPEN(UNIT=99, FILE='amb.respy.log', ACTION='WRITE', STATUS='OLD', ACCESS='APPEND')

        WRITE(99, *) 'SUMMARY'
        WRITE(99, *)
        WRITE(99, *) '   Period      Total    Success    Failure'
        WRITE(99, *)

        DO i = 1, SIZE(success_count, 2)
            total_count = SUM(success_count(:, i))
            share_success = success_count(1, i) / total_count
            share_failure = one_dble - share_success
            WRITE(99, 100) i - 1, total_count, share_success, share_failure
        END DO

        WRITE(99, *)

    CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
