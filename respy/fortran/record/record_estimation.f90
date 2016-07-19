!******************************************************************************
!******************************************************************************
MODULE recording_estimation

  !/*	external modules	*/

    USE recording_warning

    USE shared_containers

    USE shared_constants

    USE shared_auxiliary

    USE shared_utilities

  !/*	setup	*/

    IMPLICIT NONE

    PRIVATE

    PUBLIC :: record_estimation

    !/* explicit interface   */

    INTERFACE record_estimation

        MODULE PROCEDURE record_estimation_eval, record_estimation_final, record_scaling, record_estimation_stop

    END INTERFACE

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE record_estimation_stop()

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    OPEN(UNIT=99, FILE='est.respy.info', ACCESS='APPEND', ACTION='WRITE')

        WRITE(99, *)
        WRITE(99, *) 'TERMINATED'

    CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_estimation_eval(x_all_current, val_current, num_eval)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: x_all_current(26)
    REAL(our_dble), INTENT(IN)      :: val_current

    INTEGER(our_int), INTENT(IN)    :: num_eval

    !/* internal objects        */

    INTEGER(our_int), SAVE          :: num_step = - one_int

    REAL(our_dble), SAVE            :: x_container(26, 3)


    REAL(our_dble), SAVE            :: crit_vals(3)

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: shocks_cov(4, 4)

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

    LOGICAL                         :: is_large(3) = .False.
    LOGICAL                         :: is_start
    LOGICAL                         :: is_step

    CHARACTER(55)                   :: today_char
    CHARACTER(55)                   :: now_char
    CHARACTER(155)                  :: val_char
    CHARACTER(50)                   :: tmp_char

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    crit_vals(3) = val_current

    ! Determine events
    is_start = (num_eval == 1)

    IF (is_start) THEN
        crit_vals(1) = val_current
        crit_vals(2) = HUGE_FLOAT
    END IF

    is_step = (crit_vals(2) .GT. val_current)

    ! Update counters
    IF (is_step) THEN

        num_step = num_step + 1

        crit_vals(2) = val_current

    END IF

    ! Sometimes on the path of the optimizer, the value of the criterion
    ! function is just too large for pretty printing.
    DO i = 1, 3
        is_large(i) = (ABS(crit_vals(i)) > LARGE_FLOAT)
    END DO

    ! Update container
    IF (is_start) x_container(:, 1) = x_all_current

    IF (is_step) x_container(:, 2) = x_all_current

    x_container(:, 3) = x_all_current


    CALL get_time(today_char, now_char)


    100 FORMAT(1x,A4,i13,10x,A4,i10)
    110 FORMAT(3x,A4,25X,A10)
    120 FORMAT(3x,A4,27X,A8)
    130 FORMAT(3x,A9,5X,f25.15)
    135 FORMAT(3x,A9,5X,A25)
    140 FORMAT(3x,A10,3(4x,A25))
    150 FORMAT(3x,i10,3(4x,f25.15))

    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND', ACTION='WRITE')

        WRITE(99, 100) 'EVAL', num_eval, 'STEP', num_step
        WRITE(99, *)
        WRITE(99, 110) 'Date', today_char
        WRITE(99, 120) 'Time', now_char

        IF (.NOT. is_large(3)) THEN
            WRITE(99, 130) 'Criterion', crit_vals(3)
        ELSE
            WRITE(99, 135) 'Criterion', '---'

        END IF


        WRITE(99, *)
        WRITE(99, 140) 'Identifier', 'Start', 'Step', 'Current'
        WRITE(99, *)

        DO i = 1, 26
            WRITE(99, 150) (i - 1), x_container(i, :)
        END DO

        WRITE(99, *)

    CLOSE(99)


    200 FORMAT(A15,3(4x,A15))
    220 FORMAT(A15,3(4x,A15))
    210 FORMAT(A15,A57)
    230 FORMAT(i15,3(4x,f15.4))
    240 FORMAT(A15)
    250 FORMAT(f15.4,3(4x,f15.4))
    260 FORMAT(1x,A15,9x,i15)
    270 FORMAT(1x,A21,3x,i15)

    val_char = ''
    DO i = 1, 3
        IF (is_large(i)) THEN
            WRITE(tmp_char, '(4x,A15)') '---'
        ELSE
            WRITE(tmp_char, '(4x,f15.4)') crit_vals(i)
        END IF

        val_char = TRIM(val_char) // TRIM(tmp_char)
    END DO

    OPEN(UNIT=99, FILE='est.respy.info', ACTION='WRITE')

        WRITE(99, *)
        WRITE(99, *) 'Criterion Function'
        WRITE(99, *)
        WRITE(99, 200) '', 'Start', 'Step', 'Current'
        WRITE(99, *)
        WRITE(99, 210)  '', val_char
        WRITE(99, *)
        WRITE(99, *)
        WRITE(99,*) 'Optimization Parameters'
        WRITE(99, *)
        WRITE(99, 220) 'Identifier', 'Start', 'Step', 'Current'
        WRITE(99, *)

        DO i = 1, 26
            WRITE(99, 230) (i - 1), x_container(i, :)
        END DO

        WRITE(99, *)
        WRITE(99, *)
        WRITE(99, *) 'Covariance Matrix'
        WRITE(99, *)

        DO i = 1, 3

            IF (i == 1) WRITE(99, 240) 'Start'
            IF (i == 2) WRITE(99, 240) 'Step'
            IF (i == 3) WRITE(99, 240) 'Current'

            CALL get_cholesky(shocks_cholesky, x_container(:, 1))
            shocks_cov = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))

            WRITE(99, *)

            DO j = 1, 4
                WRITE(99, 250) shocks_cov(j, :)
            END DO

            WRITE(99, *)

        END DO

        WRITE(99, *)
        WRITE(99, 260) 'Number of Steps', num_step
        WRITE(99, *)
        WRITE(99, 270) 'Number of Evaluations', num_eval

    CLOSE(99)

    DO i = 1, 3
        IF (is_large(i)) CALL record_warning(i)
    END do

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_estimation_final(success, message, crit_val, x_all_final)

    !/* external objects        */

    LOGICAL, INTENT(IN)             :: success

    REAL(our_dble), INTENT(IN)      :: x_all_final(26)
    REAL(our_dble), INTENT(IN)      :: crit_val

    CHARACTER(*), INTENT(IN)        :: message

    !/* internal objects        */

    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    100 FORMAT(3x,A9,5X,f45.15)
    110 FORMAT(3x,A10,4x,A25)
    120 FORMAT(3x,i10,4x,f25.15)

    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND', ACTION='WRITE')

        WRITE(99, *) 'ESTIMATION REPORT'
        WRITE(99, *)

        IF (success) THEN
            WRITE(99, *) '  Success True'
        ELSE
            WRITE(99, *) '  Success False'
        END IF

        WRITE(99, *) '  Message ', TRIM(message)
        WRITE(99, *)
        WRITE(99, 100) 'Criterion', crit_val

        WRITE(99, *)
        WRITE(99, *)
        WRITE(99, 110), 'Identifier', 'Final'

        WRITE(99, *)

        DO i = 1, 26
            WRITE(99, 120) (i - 1), x_all_final(i)
        END DO

        WRITE(99, *)

    CLOSE(99)


END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_scaling(auto_scales, x_free_start, is_setup)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)      :: auto_scales(num_free, num_free)
    REAL(our_dble), INTENT(IN)      :: x_free_start(num_free)

    LOGICAL, INTENT(IN)             :: is_setup

    !/* internal objects    */

    REAL(our_dble)                  :: x_free_scaled(num_free)

    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    x_free_scaled = apply_scaling(x_free_start, auto_scales, 'do')


    120 FORMAT(3x,A10,3(4x,A25))
    130 FORMAT(3x,i10,3(4x,f25.15))

    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND', ACTION='WRITE')

        ! The initial setup serves to remind users that scaling is going on
        ! in the background. Otherwise, they remain puzzled as there is no
        ! output for quite some time if the gradient evaluations are
        ! time consuming.
        IF (is_setup) THEN

            WRITE(99, *) 'SCALING'
            WRITE(99, *)
            WRITE(99, 120) 'Identifier', 'Original', 'Scale', 'Transformed Value'
            WRITE(99, *)

        ELSE

            DO i = 1, num_free
                WRITE(99, 130) i, x_free_start(i), auto_scales(i, i), x_free_scaled(i)
            END DO

            WRITE(99, *)

        END IF

    CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_time(today_char, now_char)

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
