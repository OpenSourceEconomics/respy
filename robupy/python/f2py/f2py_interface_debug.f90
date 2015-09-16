!******************************************************************************
!******************************************************************************
!
!   This subroutine is just a wrapper for selected functions of the ROBUFORT 
!   library. Its sole purpose is to serve as a wrapper for debugging purposes.
!
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_criterion(emax_simulated, payoffs_ex_post, future_payoffs, &
                x, num_draws, eps_standard, period, k, payoffs_ex_ante, &
                edu_max, edu_start, mapping_state_idx, states_all, &
                num_periods, periods_emax, eps_cholesky, delta, debug)

    !/* external libraries    */

    USE robufort_development

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: payoffs_ex_post(4)
    DOUBLE PRECISION, INTENT(OUT)   :: future_payoffs(4)
    DOUBLE PRECISION, INTENT(OUT)   :: emax_simulated

    DOUBLE PRECISION, INTENT(IN)    :: eps_cholesky(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: eps_standard(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: payoffs_ex_ante(:)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: delta
    DOUBLE PRECISION, INTENT(IN)    :: x(:)

    INTEGER , INTENT(IN)            :: mapping_state_idx(:,:,:,:,:)
    INTEGER , INTENT(IN)            :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

    LOGICAL, INTENT(IN)             :: debug

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL criterion(emax_simulated, payoffs_ex_post, future_payoffs, &
                x, num_draws, eps_standard, period, k, payoffs_ex_ante, &
                edu_max, edu_start, mapping_state_idx, states_all, &
                num_periods, periods_emax, eps_cholesky, delta, debug)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_rosenbrock_derivative(rslt, x, dim)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(dim)
    DOUBLE PRECISION, INTENT(IN)    :: x(dim)

    INTEGER, INTENT(IN)             :: dim

!------------------------------------------------------------------------------ 
! Algorithm
!------------------------------------------------------------------------------ 
    
    ! Calculate derivative information
    CALL rosenbrock_derivative(rslt, x)

END SUBROUTINE 
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_rosenbrock(rslt, x, dim)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: rslt
    DOUBLE PRECISION, INTENT(IN)    :: x(dim)

    INTEGER, INTENT(IN)             :: dim

!------------------------------------------------------------------------------ 
! Algorithm
!------------------------------------------------------------------------------ 
    
    ! Evaluate Rosenbrock function
    CALL rosenbrock(rslt, x)

END SUBROUTINE 
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_divergence_approx_gradient(rslt, x, cov, level, eps)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)  :: rslt(2)
    DOUBLE PRECISION, INTENT(IN)   :: x(2)
    DOUBLE PRECISION, INTENT(IN)   :: eps
    DOUBLE PRECISION, INTENT(IN)   :: cov(4,4)
    DOUBLE PRECISION, INTENT(IN)   :: level

!------------------------------------------------------------------------------ 
! Algorithm
!------------------------------------------------------------------------------ 
    
    ! Approximate the gradient of the KL divergence
    CALL divergence_approx_gradient(rslt, x, cov, level, eps)

END SUBROUTINE 
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_multivariate_normal(draws, mean, covariance, num_draws, dim)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: dim

    DOUBLE PRECISION, INTENT(OUT)   :: draws(num_draws, dim)
    DOUBLE PRECISION, INTENT(IN)    :: mean(dim)
    DOUBLE PRECISION, INTENT(IN)    :: covariance(dim, dim)

!--------------------------------------------------------------------------- 
! Algorithm
!--------------------------------------------------------------------------- 
    
    ! Generate multivariate normal deviates    
    CALL multivariate_normal(draws, mean, covariance)
    
END SUBROUTINE 
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_standard_normal(draw, dim)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    INTEGER, INTENT(IN)             :: dim
    
    DOUBLE PRECISION, INTENT(OUT)   :: draw(dim)

!--------------------------------------------------------------------------- 
! Algorithm
!--------------------------------------------------------------------------- 
    
    ! Generate standard normal deviates
    CALL standard_normal(draw)

END SUBROUTINE 
!****************************************************************************** 
!****************************************************************************** 
SUBROUTINE wrapper_determinant(det, A)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: det

    DOUBLE PRECISION, INTENT(IN)    :: A(:, :)

!------------------------------------------------------------------------------ 
! Algorithm
!------------------------------------------------------------------------------ 
    
    ! Get determinant
    det = determinant(A)

END SUBROUTINE
!****************************************************************************** 
!****************************************************************************** 
SUBROUTINE wrapper_cholesky(factor, matrix, n)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: factor(n,n)

    DOUBLE PRECISION, INTENT(IN)    :: matrix(:,:)

    INTEGER, INTENT(IN)             :: n

!------------------------------------------------------------------------------ 
! Algorithm
!------------------------------------------------------------------------------ 

    ! Get Cholesky decomposition
    CALL cholesky(factor, matrix)

END SUBROUTINE
!****************************************************************************** 
!****************************************************************************** 
SUBROUTINE wrapper_inverse(inv, A, n)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: inv(n, n)

    DOUBLE PRECISION, INTENT(IN)    :: A(:, :)

    INTEGER, INTENT(IN)             :: n

!------------------------------------------------------------------------------ 
! Algorithm
!------------------------------------------------------------------------------ 

    ! Get inverse
    inv = inverse(A, n)

END SUBROUTINE
!****************************************************************************** 
!****************************************************************************** 
SUBROUTINE wrapper_trace(rslt, A)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT) :: rslt

    DOUBLE PRECISION, INTENT(IN)  :: A(:,:)

!------------------------------------------------------------------------------ 
! Algorithm
!------------------------------------------------------------------------------ 
    
    ! Get trace
    rslt = trace_fun(A)

END SUBROUTINE
!****************************************************************************** 
!****************************************************************************** 
SUBROUTINE wrapper_divergence(div, x, cov, level)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: div

    DOUBLE PRECISION, INTENT(IN)    :: x(2)
    DOUBLE PRECISION, INTENT(IN)    :: cov(4,4)
    DOUBLE PRECISION, INTENT(IN)    :: level

!------------------------------------------------------------------------------ 
! Algorithm
!------------------------------------------------------------------------------ 
    
    ! Calculate divergence
    CALL divergence(div, x, cov, level)

END SUBROUTINE
!****************************************************************************** 
!****************************************************************************** 
