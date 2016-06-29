!******************************************************************************
!******************************************************************************
MODULE estimate_auxiliary

	!/*	external modules	*/

    USE shared_containers 

    USE shared_constants

	!/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE logging_estimation_final(success, message, crit_val)

    !/* external objects        */

    LOGICAL, INTENT(IN)             :: success
    
    CHARACTER(*), INTENT(IN)        :: message

    REAL(our_dble), INTENT(IN)      :: crit_val

    !/* internal objects        */

    INTEGER(our_int)                :: today(3)
    INTEGER(our_int)                :: now(3)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ! Formatting
    5503 FORMAT(1x, A4,21X,i2,'/',i2,'/',i4)
    5504 FORMAT(1x, A4,23X,i2,':',i2,':',i2)
    5506 FORMAT(1x, A9 ,1X, f25.15)
 
    ! Obtain information about system time
    CALL IDATE(today)
    CALL ITIME(now)

    ! Write to file
    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND')
        WRITE(99, *) 
        WRITE(99, *) 'ESTIMATION REPORT'
        WRITE(99, *) 

        IF (success) THEN
            WRITE(99, *) 'Success True'
        ELSE
            WRITE(99, *) 'Success False'
        END IF

        WRITE(99, *) 'Message ', message
        WRITE(99, *) 
        WRITE(99, 5506) 'Criterion', crit_val
        WRITE(99, 5504) 'Time', now
        WRITE(99, 5503) 'Date', today(2), today(1), today(3) 

    CLOSE(99)

END SUBROUTINE    
!******************************************************************************
!******************************************************************************
SUBROUTINE logging_estimation_step(num_step, num_eval, crit_val)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: num_step
    INTEGER(our_int), INTENT(IN)    :: num_eval

    REAL(our_dble), INTENT(IN)      :: crit_val

    !/* internal objects        */

    INTEGER(our_int)                :: today(3)
    INTEGER(our_int)                :: now(3)

    LOGICAL, SAVE                   :: is_first = .True.

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ! Formatting
    5503 FORMAT(1x,A4,21X,i2,'/',i2,'/',i4)
    5504 FORMAT(1x,A4,23X,i2,':',i2,':',i2)
    5505 FORMAT(1x,A4, 25X, i6)
    5506 FORMAT(1x,A9 ,1X, f25.15)
 
    ! Obtain information about system time
    CALL IDATE(today)
    CALL ITIME(now)

    ! Write out header
    IF (is_first) THEN 
        OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND')
            WRITE(99, *) 'ESTIMATION PROGRESS' 
            WRITE(99, *) 
        CLOSE(99)
        is_first = .False.
    END IF

    ! Write to file
    OPEN(UNIT=99, FILE='est.respy.log', ACCESS='APPEND')
        WRITE(99, 5505) 'Step', num_step
        WRITE(99, 5505) 'Eval', num_eval
        WRITE(99, 5506) 'Criterion', crit_val
        WRITE(99, 5504) 'Time', now
        WRITE(99, 5503) 'Date', today(2), today(1), today(3) 
        WRITE(99, *) 
    CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE write_out_information(x_all_current, val_current, num_eval)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: x_all_current(26)
    REAL(our_dble), INTENT(IN)      :: val_current

    INTEGER(our_int), INTENT(IN)    :: num_eval


    !/* internal objects        */
    LOGICAL            :: is_step
    LOGICAL             :: is_start


    REAL(our_dble), SAVE            :: x_container(26, 3)


    INTEGER(our_int), SAVE          :: num_step = - one_int

    REAL(our_dble), SAVE            :: val_step = HUGE_FLOAT

    REAL(our_dble), SAVE            :: val_start = HUGE_FLOAT
    
    INTEGER(our_int)                 :: i
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ! Determine events
    is_start = (num_eval == 1)

    is_step = (val_step .GT. val_current) 
     
    ! Update counters
    IF (is_step) THEN

        num_step = num_step + 1

        val_step = val_current

    END IF

    IF (is_start) val_start = val_current

    ! Update conter
    IF (is_start) x_container(:, 1) = x_all_current

    IF (is_step) x_container(:, 2) = x_all_current

    x_container(:, 3) = x_all_current


    ! Write out to file
    20 FORMAT(3(1x,i45))
    30 FORMAT(1x,f45.15,1x,f45.25,1x,f45.15)

    OPEN(UNIT=1, FILE='est.respy.paras')

        WRITE(1, 20) zero_int, num_step, num_eval
        WRITE(1, 30) val_start, val_step, val_current

        DO i = 1, 26
            WRITE(1, 30) x_container(i, :)
        END DO

    CLOSE(1)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE dist_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(OUT)     :: coeffs_home(1)
    REAL(our_dble), INTENT(OUT)     :: coeffs_edu(3)
    REAL(our_dble), INTENT(OUT)     :: coeffs_a(6)
    REAL(our_dble), INTENT(OUT)     :: coeffs_b(6)

    REAL(our_dble), INTENT(IN)      :: x(26)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Extract model ingredients
    coeffs_a = x(1:6)

    coeffs_b = x(7:12)

    coeffs_edu = x(13:15)

    coeffs_home = x(16:16)


    shocks_cholesky = 0.0

    shocks_cholesky(1, :1) = x(17:17)

    shocks_cholesky(2, :2) = x(18:19)

    shocks_cholesky(3, :3) = x(20:22)

    shocks_cholesky(4, :4) = x(23:26)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_free_optim_paras(x, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)      :: coeffs_b(6)

    LOGICAL, INTENT(IN)             :: paras_fixed(26)

    REAL(our_dble), INTENT(OUT)     :: x(COUNT(.not. paras_fixed))

    !/* internal objects        */    

    REAL(our_dble)                  :: x_internal(26)

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    x_internal(1:6) = coeffs_a(:)
        
    x_internal(7:12) = coeffs_b(:)
        
    x_internal(13:15) = coeffs_edu(:)
        
    x_internal(16:16) = coeffs_home(:)

    x_internal(17:17) = shocks_cholesky(1, :1)
        
    x_internal(18:19) = shocks_cholesky(2, :2)
        
    x_internal(20:22) = shocks_cholesky(3, :3)
        
    x_internal(23:26) = shocks_cholesky(4, :4)


    j = 1

    DO i = 1, 26

        IF(paras_fixed(i)) CYCLE

        x(j) = x_internal(i)
        
        j = j + 1

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE