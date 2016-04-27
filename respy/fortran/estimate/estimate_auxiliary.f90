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

    ! Note that the Cholesky decomposition is the upper triangular in this 
    ! case. This is required to align the order of optimization parameters with
    ! the outline of the original authors. 
    shocks_cholesky = 0.0

    shocks_cholesky(1, 1:) = x(17:20)

    shocks_cholesky(2, 2:) = x(21:23)

    shocks_cholesky(3, 3:) = x(24:25)

    shocks_cholesky(4, 4:) = x(26:26)

    print *, x(22:)

    ! Reconstruct the covariance matrix of reward shocks
    shocks_cov = MATMUL(TRANSPOSE(shocks_cholesky), shocks_cholesky)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE