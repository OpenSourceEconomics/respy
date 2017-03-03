!******************************************************************************
!******************************************************************************
MODULE recording_estimation

  !/*	external modules	*/

    USE recording_warning

    USE shared_interface

  !/*	setup	*/

    IMPLICIT NONE

    PRIVATE

    PUBLIC :: record_estimation

    !/* explicit interface   */

    INTERFACE record_estimation

        MODULE PROCEDURE record_estimation_eval, record_estimation_final, record_scaling, record_estimation_stop, record_estimation_scalability, record_estimation_auto_npt, record_estimation_auto_rhobeg

    END INTERFACE

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE record_estimation_auto_rhobeg(algorithm, rhobeg, rhoend)

    !/* external objects        */

    CHARACTER(*), INTENT(IN)        :: algorithm

    REAL(our_dble), INTENT(IN)      :: rhobeg
    REAL(our_dble), INTENT(IN)      :: rhoend

    !/* internal objects        */

    CHARACTER(25)                   :: rho_char(2)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    WRITE(rho_char(1), '(f25.15)') rhobeg
    WRITE(rho_char(2), '(f25.15)') rhoend

    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND', ACTION='WRITE')
        WRITE(99, *) 'Warning: Automatic adjustment of rhobeg/rhoend for ' // algorithm // ' required. Both are set to their recommended values of (rhobeg/rhoend): (' // TRIM(ADJUSTL(rho_char(1))) // ',' // TRIM(ADJUSTL(rho_char(2))) // ')'
        WRITE(99, *)
    CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_estimation_auto_npt(algorithm, npt)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: npt

    CHARACTER(*), INTENT(IN)        :: algorithm

    !/* internal objects        */

    CHARACTER(10)                   :: npt_char
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    WRITE(npt_char, '(i10)') npt

    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND', ACTION='WRITE')
        WRITE(99, *) 'Warning: Automatic adjustment of NPT for ' // algorithm // ' required. NPT set to its recommended value of ' // TRIM(ADJUSTL(npt_char)) // '.'
        WRITE(99, *)
    CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_estimation_scalability(which)

    !/* external objects        */

    CHARACTER(*), INTENT(IN)   :: which

    !/* internal objects        */

    CHARACTER(55)               :: today
    CHARACTER(55)               :: now

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

  115 FORMAT(3x,A5,6X,A10,5X,A8)
  125 FORMAT(3x,A6,5X,A10,5X,A8)

  CALL get_time(today, now)

  IF (which == 'Start') THEN
    OPEN(UNIT=99, FILE='.scalability.respy.log', ACTION='WRITE')
        WRITE(99, 115) which, today, now
  ELSE
    OPEN(UNIT=99, FILE='.scalability.respy.log', ACCESS='APPEND', ACTION='WRITE')
        WRITE(99, 125) which, today, now
  END IF

  CLOSE(99)

END SUBROUTINE
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
SUBROUTINE record_estimation_eval(x_optim_free_scaled, x_optim_all_unscaled, val_current, num_eval, optim_paras, start, opt_ambi_summary)

    ! We record all things related to the optimization in est.respy.log. That is why we print the values actually relevant for the optimization, i.e. free and scaled. In est.respy.info we switch to the users perspective, all parameter are printed with their economic interpreation intact.

    !/* external objects        */

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    REAL(our_dble), INTENT(IN)      :: x_optim_free_scaled(num_free)
    REAL(our_dble), INTENT(IN)      :: x_optim_all_unscaled(28)
    REAL(our_dble), INTENT(IN)      :: val_current
    REAL(our_dble), INTENT(IN)      :: start

    INTEGER(our_int), INTENT(IN)    :: opt_ambi_summary(2)
    INTEGER(our_int), INTENT(IN)    :: num_eval

    !/* internal objects        */

    INTEGER(our_int), SAVE          :: num_step = - one_int

    REAL(our_dble), SAVE            :: x_optim_container(28, 3) = -HUGE_FLOAT
    REAL(our_dble), SAVE            :: x_econ_container(28, 3) = -HUGE_FLOAT

    REAL(our_dble), SAVE            :: crit_vals(3)

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: shocks_cov(3, 4, 4)
    REAL(our_dble)                  :: flattened_cov(3, 10)
    REAL(our_dble)                  :: cond(3)
    REAL(our_dble)                  :: finish

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: l

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

    ! Create the container for the *.log file. The subsetting is required as an automatic object cannot be saved.
    If(is_start) x_optim_container(:num_free, 1) = x_optim_free_scaled

    If(is_step) x_optim_container(:num_free, 2) = x_optim_free_scaled

    x_optim_container(:num_free, 3) = x_optim_free_scaled

    ! Create the container for the *.info file.
    DO i = 1, 3
        CALL extract_cholesky(shocks_cholesky, x_optim_all_unscaled)
        shocks_cov(i, :, :) = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))
        CALL spectral_condition_number(cond(i), shocks_cov(i, :, :))

        k = 1
        DO j = 1, 4
            DO l = j, 4
                flattened_cov(i, k) = shocks_cov(i, j, l)
                IF (j == l) flattened_cov(i, k) = SQRT(flattened_cov(i, k))

                k = k + 1
            END DO
        END DO
    END DO

    IF (is_start) THEN
        x_econ_container(:18, 1) = x_optim_all_unscaled(:18)
        x_econ_container(19:, 1) = flattened_cov(1, :)
    END IF

    IF (is_step) THEN
        x_econ_container(:18, 2) = x_optim_all_unscaled(:18)
        x_econ_container(19:, 2) = flattened_cov(2, :)
    END IF

    x_econ_container(:18, 3) = x_optim_all_unscaled(:18)
    x_econ_container(19:, 3) = flattened_cov(3, :)


    CALL get_time(today_char, now_char)
    CALL CPU_TIME(finish)

    100 FORMAT(1x,A4,i13,10x,A4,i10)
    110 FORMAT(3x,A4,25X,A10)
    120 FORMAT(3x,A4,27X,A8)
    125 FORMAT(3x,A8,23X,i8)
    130 FORMAT(3x,A9,5X,A25)
    135 FORMAT(3x,A9,5X,A25)
    140 FORMAT(3x,A10,3(4x,A25))
    150 FORMAT(3x,i10,3(4x,A25))
    155 FORMAT(3x,A9,1x,3(4x,f25.15))
    157 FORMAT(3x,A9,1x,4x,f24.2,A1)

    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND', ACTION='WRITE')

        WRITE(99, 100) 'EVAL', num_eval, 'STEP', num_step
        WRITE(99, *)
        WRITE(99, 110) 'Date', today_char
        WRITE(99, 120) 'Time', now_char
        WRITE(99, 125) 'Duration', INT(finish - start)
        WRITE(99, *)

        WRITE(99, 130) 'Criterion', char_floats(crit_vals(3:3))

        IF (.NOT. opt_ambi_summary(1) == zero_int) THEN
            WRITE(99, 157) 'Ambiguity', (DBLE(opt_ambi_summary(2)) / DBLE(opt_ambi_summary(1)) * 100), '%'
        ELSE
            WRITE(99, 135) 'Ambiguity', '---'
        END IF


        WRITE(99, *)
        WRITE(99, 140) 'Identifier', 'Start', 'Step', 'Current'
        WRITE(99, *)

        j = 1
        DO i = 1, 28
            IF(optim_paras%paras_fixed(i)) CYCLE
            WRITE(99, 150) i - 1, char_floats(x_optim_container(j, :))
            j = j + 1
        END DO

        WRITE(99, *)

        WRITE(99, 155) 'Condition', LOG(cond)

        WRITE(99, *)

    CLOSE(99)


    200 FORMAT(A25,3(4x,A25))
    210 FORMAT(A25,A87)
    220 FORMAT(A25,3(4x,A25))
    230 FORMAT(i25,3(4x,A25))

    250 FORMAT(A25)
    270 FORMAT(1x,A15,13x,i25)
    280 FORMAT(1x,A21,7x,i25)

    val_char = ''
    DO i = 1, 3
        IF (is_large(i)) THEN
            WRITE(tmp_char, '(4x,A25)') '---'
        ELSE
            WRITE(tmp_char, '(4x,f25.15)') crit_vals(i)
        END IF

        val_char = TRIM(val_char) // TRIM(tmp_char)
    END DO

    OPEN(UNIT=99, FILE='est.respy.info', ACTION='WRITE')

        WRITE(99, *)
        WRITE(99, 250) 'Criterion Function'
        WRITE(99, *)
        WRITE(99, 200) '', 'Start', 'Step', 'Current'
        WRITE(99, *)
        WRITE(99, 210)  '', val_char
        WRITE(99, *)
        WRITE(99, *)
        WRITE(99, 250) 'Economic Parameters'
        WRITE(99, *)
        WRITE(99, 220) 'Identifier', 'Start', 'Step', 'Current'
        WRITE(99, *)

        DO i = 1, 28
            WRITE(99, 230) (i - 1), char_floats(x_econ_container(i, :))
        END DO

        WRITE(99, *)
        WRITE(99, 270) 'Number of Steps', num_step
        WRITE(99, *)
        WRITE(99, 280) 'Number of Evaluations', num_eval

    CLOSE(99)

    DO i = 1, 3
        IF (is_large(i)) CALL record_warning(i)
    END do

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_estimation_final(success, message)

    !/* external objects        */

    LOGICAL, INTENT(IN)             :: success
    CHARACTER(*), INTENT(IN)        :: message

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND', ACTION='WRITE')

        WRITE(99, *) 'ESTIMATION REPORT'
        WRITE(99, *)

        IF (success) THEN
            WRITE(99, *) '  Success True'
        ELSE
            WRITE(99, *) '  Success False'
        END IF

        WRITE(99, *) '  Message ', TRIM(message)

    CLOSE(99)


END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE record_scaling(precond_matrix, x_free_start, optim_paras, is_setup)

    !/* external objects    */

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    REAL(our_dble), INTENT(IN)      :: precond_matrix(num_free, num_free)
    REAL(our_dble), INTENT(IN)      :: x_free_start(num_free)

    LOGICAL, INTENT(IN)             :: is_setup

    !/* internal objects    */

    REAL(our_dble)                  :: x_free_scaled(num_free)
    REAL(our_dble)                  :: floats(3)

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k

    CHARACTER(155)                  :: val_char
    CHARACTER(50)                   :: tmp_char
    LOGICAL                         :: no_bounds

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    x_free_scaled = apply_scaling(x_free_start, precond_matrix, 'do')

    120 FORMAT(3x,A10,5(4x,A25))
    135 FORMAT(3x,i10,3(4x,A25),A58)

    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND', ACTION='WRITE')

        ! The initial setup serves to remind users that scaling is going on
        ! in the background. Otherwise, they remain puzzled as there is no
        ! output for quite some time if the gradient evaluations are
        ! time consuming.
        IF (is_setup) THEN

            WRITE(99, *) 'PRECONDITIONING'
            WRITE(99, *)
            WRITE(99, 120) 'Identifier', 'Original', 'Scale', 'Transformed Value', 'Transformed Lower', 'Transformed Upper'
            WRITE(99, *)

        ELSE

            ! Sometimes on the bounds are just too large for pretty printing
            j = 1
            DO i = 1, 28
                IF(optim_paras%paras_fixed(i)) CYCLE

                ! We need to do some pre-processing for the transformed bounds.
                val_char = ''
                DO k = 1, 2
                    no_bounds = (ABS(x_optim_bounds_free_scaled(k, j)) > Large_FLOAT)

                    IF(no_bounds) THEN
                        WRITE(tmp_char, '(4x,A25)') '---'
                    ELSE
                        WRITE(tmp_char, '(4x,f25.15)') x_optim_bounds_free_scaled(k, j)
                    END IF
                    val_char = TRIM(val_char) // TRIM(tmp_char)

                END DO

                floats = (/ x_free_start(j), precond_matrix(j, j), x_free_scaled(j) /)
                WRITE(99, 135) i - 1, char_floats(floats) , val_char

                j = j + 1

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
FUNCTION char_floats(floats)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: floats(:)

    CHARACTER(50)                   :: char_floats(SIZE(floats))

    !/* internal objects        */

    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    910 FORMAT(f25.15)
    900 FORMAT(A25)

    DO i = 1, SIZE(floats)

        IF (ABS(floats(i)) > LARGE_FLOAT) THEN
            WRITE(char_floats(i), 900) '---'
        ELSE
            WRITE(char_floats(i), 910) floats(i)
        END IF

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
END MODULE
