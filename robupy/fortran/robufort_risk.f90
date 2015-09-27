!*******************************************************************************
!*******************************************************************************
!
!   This delivers all functions and subroutines to the ROBUFORT library that 
!	are associated with the model under risk. 
!
!*******************************************************************************
!*******************************************************************************
MODULE robufort_risk

	!/*	external modules	*/

    USE robufort_program_constants

    USE robufort_auxiliary

    USE robufort_emax

	!/*	setup	*/

	IMPLICIT NONE

	PRIVATE

    !/* core functions */

    PUBLIC :: get_payoffs_risk 
 
CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_payoffs_risk(emax_simulated, payoffs_ex_post, future_payoffs, &
                num_draws, eps_relevant, period, k, payoffs_ex_ante, & 
                edu_max, edu_start, mapping_state_idx, states_all, num_periods, & 
                periods_emax, delta)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: emax_simulated
    REAL(our_dble), INTENT(OUT)     :: payoffs_ex_post(4)
    REAL(our_dble), INTENT(OUT)     :: future_payoffs(4)

    INTEGER(our_int), INTENT(IN)    :: num_draws
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k 
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)

    REAL(our_dble), INTENT(IN)      :: eps_relevant(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_ex_ante(:)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Simulated expected future value
    CALL simulate_emax(emax_simulated, payoffs_ex_post, future_payoffs, num_periods, & 
            num_draws, period, k, eps_relevant, payoffs_ex_ante, edu_max, & 
            edu_start, periods_emax, states_all, mapping_state_idx, delta)
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE



