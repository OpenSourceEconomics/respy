!*******************************************************************************
!*******************************************************************************
MODULE estimate_auxiliary

	!/*	external modules	*/

    USE shared_constants

	!/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS

!*******************************************************************************
!*******************************************************************************
SUBROUTINE dist_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, &
                shocks_cov, x)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: shocks_cov(:, :)
    REAL(our_dble), INTENT(OUT)     :: coeffs_home(:)
    REAL(our_dble), INTENT(OUT)     :: coeffs_edu(:)
    REAL(our_dble), INTENT(OUT)     :: coeffs_a(:)
    REAL(our_dble), INTENT(OUT)     :: coeffs_b(:)

    REAL(our_dble), INTENT(IN)      :: x(:)

    !/* internal objects        */

    REAL(our_dble)                  :: shocks_cholesky(4, 4)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Extract model ingredients
    coeffs_a = x(1:6)

    coeffs_b = x(7:12)

    coeffs_edu = x(13:15)

    coeffs_home = x(16:16)

    shocks_cholesky = 0.0

    shocks_cholesky(1, 1) = x(17)

    shocks_cholesky(2, 1:2) = x(18:19)

    shocks_cholesky(3, 1:3) = x(20:22)

    shocks_cholesky(4, 1:4) = x(23:26)

    ! Reconstruct the covariance matrix of reward shocks
    shocks_cov = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE