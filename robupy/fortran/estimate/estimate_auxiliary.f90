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
SUBROUTINE get_model_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, &
                shocks_cov, shocks_cholesky, x)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(OUT)     :: shocks_cov(:, :)
    REAL(our_dble), INTENT(OUT)     :: coeffs_home(:)
    REAL(our_dble), INTENT(OUT)     :: coeffs_edu(:)
    REAL(our_dble), INTENT(OUT)     :: coeffs_a(:)
    REAL(our_dble), INTENT(OUT)     :: coeffs_b(:)

    REAL(our_dble), INTENT(IN)      :: x(:)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Extract model ingredients
    coeffs_a = x(1:6)

    coeffs_b = x(7:12)

    coeffs_edu = x(13:15)

    coeffs_home = x(16:16)

    shocks_cholesky = 0.0

    shocks_cholesky(1:4, 1) = x(17:20)

    shocks_cholesky(2:4, 2) = x(21:23)

    shocks_cholesky(3:4, 3) = x(24:25)

    shocks_cholesky(4:4, 4) = x(26:26)

    ! Reconstruct the covariance matrix of reward shocks
    shocks_cov = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE