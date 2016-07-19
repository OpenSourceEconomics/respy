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
SUBROUTINE dist_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x, info)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(OUT)     :: coeffs_home(1)
    REAL(our_dble), INTENT(OUT)     :: coeffs_edu(3)
    REAL(our_dble), INTENT(OUT)     :: coeffs_a(6)
    REAL(our_dble), INTENT(OUT)     :: coeffs_b(6)

    REAL(our_dble), INTENT(IN)      :: x(26)

    INTEGER(our_int), OPTIONAL, INTENT(OUT)   :: info

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Extract model ingredients
    coeffs_a = x(1:6)

    coeffs_b = x(7:12)

    coeffs_edu = x(13:15)

    coeffs_home = x(16:16)

    ! The information pertains to the stabilization of an otherwise zero variance.
    IF (PRESENT(info)) THEN
        CALL get_cholesky(shocks_cholesky, x, info)
    ELSE
        CALL get_cholesky(shocks_cholesky, x)
    END IF

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