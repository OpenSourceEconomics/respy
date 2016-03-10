!*******************************************************************************
!*******************************************************************************
!
!   This delivers all functions and subroutines to the ROBUFORT library that 
!	are associated with the model under risk. 
!
!*******************************************************************************
!*******************************************************************************
MODULE robufort_risk

	!/*	external modules	    */

    USE robufort_constants

    USE robufort_auxiliary

    USE robufort_emax

	!/*	setup	                */

	IMPLICIT NONE

	PUBLIC
 
CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_payoffs_risk(emax_simulated, payoffs_ex_post, payoffs_future, &
                num_draws_emax, eps_relevant, period, k, payoffs_systematic, &
                edu_max, edu_start, mapping_state_idx, states_all, & 
                num_periods, periods_emax, delta, is_debug, shocks, level, & 
                measure)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: payoffs_ex_post(4)
    REAL(our_dble), INTENT(OUT)     :: payoffs_future(4)
    REAL(our_dble), INTENT(OUT)     :: emax_simulated

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k 

    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: eps_relevant(:, :)
    REAL(our_dble), INTENT(IN)      :: shocks(:, :)
    REAL(our_dble), INTENT(IN)      :: level
    REAL(our_dble), INTENT(IN)      :: delta

    LOGICAL, INTENT(IN)             :: is_debug

    CHARACTER(10), INTENT(IN)       :: measure

    !/* internal  objects       */

    REAL(our_dble)                  :: eps_relevant_emax(num_draws_emax, 4)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ! Renaming for optimization step
    eps_relevant_emax = eps_relevant

    ! Simulated expected future value
    CALL simulate_emax(emax_simulated, payoffs_ex_post, payoffs_future, &
            num_periods, num_draws_emax, period, k, eps_relevant_emax, &
            payoffs_systematic, edu_max, edu_start, periods_emax, states_all, & 
            mapping_state_idx, delta)
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE



