!******************************************************************************
!******************************************************************************
MODULE recording_ambiguity

    !/*	external modules	*/

    USE shared_interface

    !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE record_ambiguity(opt_ambi_details, states_number_period, file_sim, optim_paras)

    !/* external objects    */

    TYPE(OPTIMPARAS_DICT), INTENT(IN) :: optim_paras

    REAL(our_dble), INTENT(IN)      :: opt_ambi_details(num_periods, max_states_period, 7)

    INTEGER(our_int), INTENT(IN)    :: states_number_period(:)

    CHARACTER(225), INTENT(IN)      :: file_sim

    !/* internal objects    */

    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: mode
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: ambi_rslt_mean_subset(2)
    REAL(our_dble)                  :: ambi_rslt_chol_flat(3)
    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: ambi_rslt_chol(4, 4)
    REAL(our_dble)                  :: ambi_rslt_cov(4, 4)
    REAL(our_dble)                  :: shocks_cov(4, 4), shocks_corr_base(4, 4)
    REAL(our_dble)                  :: div(1), rslt_mean(4),rslt_sd(4), rslt_cov(4, 4)

    LOGICAL                         :: is_deterministic
    LOGICAL                         :: is_success

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    shocks_cholesky = optim_paras%shocks_cholesky
    shocks_cov = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))
    is_deterministic = ALL(shocks_cov .EQ. zero_dble)

    CALL covariance_to_correlation(shocks_corr_base, shocks_cov)

    100 FORMAT(1x,A6,i7,2x,A5,i7)
    110 FORMAT(3x,A12,f10.5)
    120 FORMAT(3x,A7,8x,A5,20x)
    130 FORMAT(3x,A7,8x,A100)
    140 FORMAT(3x,A4,11x,A10)
    150 FORMAT(3x,f10.5,2x,f10.5,f10.5,f10.5,f10.5)

    OPEN(UNIT=99, FILE=TRIM(file_sim)//'.respy.amb', ACCESS='APPEND', ACTION='WRITE')

    DO period = num_periods - 1, 0, -1

        DO k = 0, (states_number_period(period + 1) - 1)

            WRITE(99, 100) 'PERIOD', period, 'STATE', k

            rslt_mean = zero_dble
            rslt_mean(:2) = opt_ambi_details(period + 1, k + 1, 1:2)

            rslt_sd(:2) = opt_ambi_details(period + 1, k + 1, 3:4)
            rslt_sd(3:) = (/DSQRT(shocks_cov(3, 3)), DSQRT(shocks_cov(3, 3))/)

            !ambi_rslt_chol_flat = opt_ambi_details(period + 1, k + 1, 3:5)
            div = opt_ambi_details(period + 1, k + 1, 5)
            is_success = (opt_ambi_details(period + 1, k + 1 , 6) == one_dble)
            mode = opt_ambi_details(period + 1, k + 1, 7)


            IF (is_deterministic) THEN
                rslt_cov = zero_dble
            ELSE
                CALL correlation_to_covariance(rslt_cov, shocks_corr_base, rslt_sd)
            END IF



            !IF (.NOT. is_deterministic) THEN
        !        !CALL construct_full_covariances(ambi_rslt_cov, ambi_rslt_chol, ambi_rslt_chol_flat, shocks_cov)
        !    ELSE
        !        ambi_rslt_cov = zero_dble
        !    END IF
            ! We need to skip states that where not analyzed during an interpolation.
            IF (mode == MISSING_FLOAT) CYCLE

            WRITE(99, *)
            WRITE(99, 110) 'Divergence  ', div(1)

            WRITE(99, *)

            IF(is_success) THEN
                WRITE(99, 120) 'Success', 'True '
            ELSE
                WRITE(99, 120) 'Success', 'False'
            END IF

            WRITE(99, 130) 'Message', ADJUSTL(get_message(mode))
            WRITE(99, *)
            WRITE(99, 140) 'Mean', 'Covariance'
            WRITE(99, *)
            DO i = 1, 4
                WRITE(99, 150) rslt_mean(i), rslt_cov(i, :)
            END DO

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

    REAL(our_dble), INTENT(IN)          :: opt_ambi_details(num_periods, max_states_period, 8)

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
            success = COUNT(opt_ambi_details(period + 1,:total, 7) == one_int) / DBLE(total)
            failure = COUNT(opt_ambi_details(period + 1,:total, 7) == zero_int) / DBLE(total)
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
