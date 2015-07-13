!*************************************************************************
!**************************************************************************
!
!   Interface to ROBUPY library.
!
!**************************************************************************
!**************************************************************************
MODULE robupy_library

	!/*	external modules	*/

    USE robupy_program_constants

    USE robupy_auxiliary

	!/*	setup	*/

	IMPLICIT NONE

    PUBLIC :: get_future_payoffs_lib
    PUBLIC :: inverse_lib
    PUBLIC :: det_lib
    PUBLIC :: calculate_payoffs_ex_ante_lib

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE calculate_payoffs_ex_ante_lib(period_payoffs_ex_ante, num_periods, &
              states_number_period, states_all, edu_start, coeffs_A, coeffs_B, & 
              coeffs_edu, coeffs_home, max_states_period)

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: period_payoffs_ex_ante(num_periods, &
                                            max_states_period, 4)

    DOUBLE PRECISION, INTENT(IN)    :: coeffs_A(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_B(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)

    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: states_number_period(:)
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: max_states_period

    !/* internals objects    */

    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: exp_A
    INTEGER(our_int)                :: exp_B
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: edu_lagged

    REAL(our_dble)                  :: covars(6)
    REAL(our_dble)                  :: payoff

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Calculate systematic instantaneous payoffs
    DO period = num_periods, 1, -1

        ! Loop over all possible states
        DO k = 1, states_number_period(period)

            ! Distribute state space
            exp_A = states_all(period, k, 1)
            exp_B = states_all(period, k, 2)
            edu = states_all(period, k, 3)
            edu_lagged = states_all(period, k, 4)

            ! Auxiliary objects
            covars(1) = one_dble
            covars(2) = edu + edu_start
            covars(3) = exp_A
            covars(4) = exp_A ** 2
            covars(5) = exp_B
            covars(6) = exp_B ** 2

            ! Calculate systematic part of payoff in occupation A
            period_payoffs_ex_ante(period, k, 1) =  & 
                EXP(DOT_PRODUCT(covars, coeffs_A))

            ! Calculate systematic part of payoff in occupation B
            period_payoffs_ex_ante(period, k, 2) = & 
                EXP(DOT_PRODUCT(covars, coeffs_B))

            ! Calculate systematic part of schooling utility
            payoff = coeffs_edu(1)

            ! Tuition cost for higher education if agents move
            ! beyond high school.
            IF(edu + edu_start > 12) THEN

                payoff = payoff + coeffs_edu(2)
            
            END IF

            ! Psychic cost of going back to school
            IF(edu_lagged == 0) THEN
            
                payoff = payoff + coeffs_edu(3)
            
            END IF
            period_payoffs_ex_ante(period, k, 3) = payoff

            ! Calculate systematic part of payoff in home production
            period_payoffs_ex_ante(period, k, 4) = coeffs_home(1)

        END DO

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_future_payoffs_lib(future_payoffs, edu_max, edu_start, &
        mapping_state_idx, period, emax, k, states_all)

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: future_payoffs(4)

    DOUBLE PRECISION, INTENT(IN)    :: emax(:, :)

    INTEGER, INTENT(IN)             :: k
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: states_all(:, :, :)
    INTEGER, INTENT(IN)             :: mapping_state_idx(:, :, :, :, :)

    !/* internals objects    */

    INTEGER(our_int)    			:: exp_A
    INTEGER(our_int)    			:: exp_B
    INTEGER(our_int)    			:: edu
    INTEGER(our_int)    			:: edu_lagged
    INTEGER(our_int)    			:: future_idx

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

	! Distribute state space
	exp_A = states_all(period + 1, k + 1, 1)
	exp_B = states_all(period + 1, k + 1, 2)
	edu = states_all(period + 1, k + 1, 3)
	edu_lagged = states_all(period + 1, k + 1, 4)

	! Working in occupation A
	future_idx = mapping_state_idx(period + 1 + 1, exp_A + 1 + 1, & 
					exp_B + 1, edu + 1, 1)
	future_payoffs(1) = emax(period + 1 + 1, future_idx + 1)

	!Working in occupation B
	future_idx = mapping_state_idx(period + 1 + 1, exp_A + 1, & 
					exp_B + 1 + 1, edu + 1, 1)
	future_payoffs(2) = emax(period + 1 + 1, future_idx + 1)

	! Increasing schooling. Note that adding an additional year
	! of schooling is only possible for those that have strictly
	! less than the maximum level of additional education allowed.
	IF (edu < edu_max - edu_start) THEN
	    future_idx = mapping_state_idx(period + 1 + 1, exp_A + 1, &
	    				exp_B + 1, edu + 1 + 1, 2)
	    future_payoffs(3) = emax(period + 1 + 1, future_idx + 1)
	ELSE
	    future_payoffs(3) = -HUGE(future_payoffs)
	END IF

	! Staying at home
	future_idx = mapping_state_idx(period + 1 + 1, exp_A + 1, & 
					exp_B + 1, edu + 1, 1)
	future_payoffs(4) = emax(period + 1 + 1, future_idx + 1)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION inverse_lib(A, k)

    !/* external objects    */

	INTEGER(our_int), INTENT(IN)	:: k

	REAL(our_dble), INTENT(IN)		:: A(k, k)

    !/* internal objects    */
	
	REAL(our_dble), ALLOCATABLE		:: y(:, :)
	REAL(our_dble), ALLOCATABLE		:: B(:, :)
	REAL(our_dble)					:: d
	REAL(our_dble) 					:: inverse_lib(k, k)

	INTEGER(our_int), ALLOCATABLE	:: indx(:)	
	INTEGER(our_int)				:: n
	INTEGER(our_int)				:: i
	INTEGER(our_int)				:: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
	
	! Auxiliary objects
	n  = size(A, 1)

	! Allocate containers
	ALLOCATE(y(n, n))
	ALLOCATE(B(n, n))
	ALLOCATE(indx(n))

	! Initialize containers
	y = zero_dble
	B = A

	! Main
	DO i = 1, n
	
	   y(i, i) = 1
	
	END DO

	CALL ludcmp(B, d, indx)

	DO j = 1, n
	
	   CALL lubksb(B, y(:, j), indx)
	
	END DO
	
	! Collect result
	inverse_lib = y

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION det_lib(A)

    !/* external objects    */

	REAL(our_dble), INTENT(IN)		:: A(:, :)
	REAL(our_dble) 					:: det_lib

    !/* internal objects    */
	INTEGER(our_int), ALLOCATABLE	:: indx(:)
	INTEGER(our_int)				:: j
	INTEGER(our_int)				:: n

	REAL(our_dble), ALLOCATABLE		:: B(:, :)
	REAL(our_dble)					:: d

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

	! Auxiliary objects
	n  = size(A, 1)

	! Allocate containers
	ALLOCATE(B(n, n))
	ALLOCATE(indx(n))

	! Initialize containers
	B = A

	CALL ludcmp(B, d, indx)
	
	DO j = 1, n
	
	   d = d * B(j, j)
	
	END DO
	
	! Collect results
	det_lib = d

END FUNCTION
!******************************************************************************
!******************************************************************************
PURE FUNCTION trace_fun(A)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)     	:: A(:,:)
    REAL(our_dble) 					:: trace_fun

    !/* internals objects    */

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: n

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Get dimension
    n = SIZE(A, DIM = 1)

    ! Initialize results
    trace_fun = zero_dble

    ! Calculate trace
    DO i = 1, n

        trace_fun = trace_fun + A(i, i)

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
END MODULE  