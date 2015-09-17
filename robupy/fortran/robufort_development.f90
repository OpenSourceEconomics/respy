!*******************************************************************************
!*******************************************************************************
!
!   This module contains functions and subroutines that are only used in the
!   development process.
!
!*******************************************************************************
!*******************************************************************************
MODULE robufort_development

	!/*	external modules	*/

    USE robufort_program_constants
    USE robufort_library

 	!/*	setup	*/

  IMPLICIT NONE

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE criterion_approx_gradient(rslt, x, num_draws, eps_standard, &
                period, k, payoffs_ex_ante, edu_max, edu_start, &
                mapping_state_idx, states_all, num_periods, periods_emax, &
                eps_cholesky, delta, debug, eps)

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)     :: rslt(2)

    DOUBLE PRECISION, INTENT(IN)      :: eps_cholesky(:, :)
    DOUBLE PRECISION, INTENT(IN)      :: eps_standard(:, :)
    DOUBLE PRECISION, INTENT(IN)      :: payoffs_ex_ante(:)
    DOUBLE PRECISION, INTENT(IN)      :: periods_emax(:,:)
    DOUBLE PRECISION, INTENT(IN)      :: delta
    DOUBLE PRECISION, INTENT(IN)      :: x(:)
    DOUBLE PRECISION, INTENT(IN)      :: eps

    INTEGER , INTENT(IN)   :: mapping_state_idx(:,:,:,:,:)
    INTEGER  , INTENT(IN)   :: states_all(:,:,:)
    INTEGER , INTENT(IN)    :: num_periods
    INTEGER , INTENT(IN)    :: num_draws
    INTEGER , INTENT(IN)    :: edu_start
    INTEGER , INTENT(IN)    :: edu_max
    INTEGER , INTENT(IN)    :: period
    INTEGER , INTENT(IN)    :: k

    LOGICAL, INTENT(IN)             :: debug


    !/* internals objects    */


    DOUBLE PRECISION      :: payoffs_ex_post(4)
    DOUBLE PRECISION     :: future_payoffs(4)

    INTEGER              :: j

    DOUBLE PRECISION                  :: ei(2)
    DOUBLE PRECISION                  :: d(2)
    DOUBLE PRECISION                  :: f0
    DOUBLE PRECISION                 :: f1

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------


    ! Initialize containers
    ei = zero_dble

    ! Evaluate baseline
    CALL criterion(f0, payoffs_ex_post, future_payoffs, &
                x, num_draws, eps_standard, period, k, payoffs_ex_ante, &
                edu_max, edu_start, mapping_state_idx, states_all, &
                num_periods, periods_emax, eps_cholesky, delta, debug)

    ! Iterate over increments
    DO j = 1, 2

        ei(j) = one_dble

        d = eps * ei

        CALL criterion(f1, payoffs_ex_post, future_payoffs, &
                x + d, num_draws, eps_standard, period, k, payoffs_ex_ante, &
                edu_max, edu_start, mapping_state_idx, states_all, &
                num_periods, periods_emax, eps_cholesky, delta, debug)


        rslt(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE criterion(emax_simulated, payoffs_ex_post, future_payoffs, &
                x, num_draws, eps_standard, period, k, payoffs_ex_ante, &
                edu_max, edu_start, mapping_state_idx, states_all, &
                num_periods, periods_emax, eps_cholesky, delta, debug)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: payoffs_ex_post(4)
    REAL(our_dble), INTENT(OUT)     :: future_payoffs(4)
    REAL(our_dble), INTENT(OUT)     :: emax_simulated

    REAL(our_dble), INTENT(IN)      :: eps_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: eps_standard(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_ex_ante(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:,:)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: x(:)

    INTEGER(our_int) , INTENT(IN)   :: mapping_state_idx(:,:,:,:,:)
    INTEGER(our_int) , INTENT(IN)   :: states_all(:,:,:)
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: num_draws
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    LOGICAL, INTENT(IN)             :: debug

    !/* internal objects    */

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

    REAL(our_dble)                  ::eps_relevant(num_draws, 4)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Transform disturbances
    DO i = 1, num_draws
        eps_relevant(i:i, :) = MATMUL(eps_cholesky, TRANSPOSE(eps_standard(i:i,:)))
        eps_relevant(i, :2) = eps_relevant(i, :2) + x
    END DO

    ! Transform disturbance for occupations
    DO j = 1, 2
        eps_relevant(:, j) = EXP(eps_relevant(:, j))
    END DO

    ! Evaluate expected future value
    CALL simulate_emax(emax_simulated, payoffs_ex_post, future_payoffs, &
            num_periods, num_draws, period, k, eps_relevant, &
            payoffs_ex_ante, edu_max, edu_start, periods_emax, states_all, &
            mapping_state_idx, delta)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE