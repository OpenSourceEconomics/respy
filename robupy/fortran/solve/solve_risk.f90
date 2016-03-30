!*******************************************************************************
!*******************************************************************************
!
!   This delivers all functions and subroutines to the ROBUFORT library that 
!	are associated with the model under risk. 
!
!*******************************************************************************
!*******************************************************************************
MODULE evaluate_risk

	!/*	external modules	    */

    USE shared_constants

    USE evaluate_emax

	!/*	setup	                */

	IMPLICIT NONE

	PUBLIC
 
CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_payoffs_risk(emax_simulated, num_draws_emax, draws_emax, & 
                period, k, payoffs_systematic, edu_max, edu_start, & 
                mapping_state_idx, states_all, num_periods, periods_emax, & 
                delta, shocks_cholesky)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: emax_simulated

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k 

    REAL(our_dble), INTENT(IN)      :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: delta

    !/* internal  objects       */

    REAL(our_dble)                  :: shocks_mean(2)  

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Auxiliary object
    shocks_mean = zero_dble
    
    ! Simulated expected future value
    CALL simulate_emax(emax_simulated, num_periods, num_draws_emax, & 
            period, k, draws_emax, payoffs_systematic, edu_max, edu_start, & 
            periods_emax, states_all, mapping_state_idx, delta, & 
            shocks_cholesky, shocks_mean)
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE



