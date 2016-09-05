!******************************************************************************
!******************************************************************************
MODULE estimate_auxiliary

    !/*	external modules	*/

    USE recording_estimation

    USE shared_containers

    USE shared_constants

    USE shared_utilities

    USE shared_auxiliary

    !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE dist_optim_paras(level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x, info)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(OUT)     :: coeffs_home(1)
    REAL(our_dble), INTENT(OUT)     :: coeffs_edu(3)
    REAL(our_dble), INTENT(OUT)     :: coeffs_a(6)
    REAL(our_dble), INTENT(OUT)     :: coeffs_b(6)
    REAL(our_dble), INTENT(OUT)     :: level(1)

    REAL(our_dble), INTENT(IN)      :: x(27)

    INTEGER(our_int), OPTIONAL, INTENT(OUT)   :: info

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Extract model ingredients
    level = EXP(x(1:1))

    coeffs_a = x(2:7)

    coeffs_b = x(8:13)

    coeffs_edu = x(14:16)

    coeffs_home = x(17:17)

    ! The information pertains to the stabilization of an otherwise zero variance.
    IF (PRESENT(info)) THEN
        CALL get_cholesky(shocks_cholesky, x, info)
    ELSE
        CALL get_cholesky(shocks_cholesky, x)
    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_free_optim_paras(x, level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)      :: coeffs_b(6)
    REAL(our_dble), INTENT(IN)      :: level(1)

    LOGICAL, INTENT(IN)             :: paras_fixed(27)

    REAL(our_dble), INTENT(OUT)     :: x(COUNT(.not. paras_fixed))

    !/* internal objects        */

    REAL(our_dble)                  :: x_internal(27)
    REAL(our_dble)                  :: level_clipped(1)

    INTEGER(our_int), ALLOCATABLE   :: infos(:)
    
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL clip_value(level_clipped, level, SMALL_FLOAT, HUGE_FLOAT, infos)

    x_internal(1:1) = LOG(level_clipped)

    x_internal(2:7) = coeffs_a(:)

    x_internal(8:13) = coeffs_b(:)

    x_internal(14:16) = coeffs_edu(:)

    x_internal(17:17) = coeffs_home(:)

    x_internal(18:18) = shocks_cholesky(1, :1)

    x_internal(19:20) = shocks_cholesky(2, :2)

    x_internal(21:23) = shocks_cholesky(3, :3)

    x_internal(24:27) = shocks_cholesky(4, :4)


    j = 1

    DO i = 1, 27

        IF(paras_fixed(i)) CYCLE

        x(j) = x_internal(i)

        j = j + 1

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
