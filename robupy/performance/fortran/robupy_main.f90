!******************************************************************************
!******************************************************************************
!
!   Interface to ROBUPY library.
!
!******************************************************************************
!******************************************************************************
MODULE robupy_library

	!/*	external modules	*/

    USE robupy_program_constants

    USE robupy_auxiliary

	!/*	setup	*/

	IMPLICIT NONE

    PUBLIC :: calculate_payoffs_ex_ante_lib
    PUBLIC :: get_future_payoffs_lib
    PUBLIC :: create_state_space_lib
    PUBLIC :: backward_induction_lib
    PUBLIC :: get_payoffs_risk_lib
    PUBLIC :: simulate_emax_lib
    PUBLIC :: divergence_lib
    PUBLIC :: inverse_lib
    PUBLIC :: det_lib

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE backward_induction_lib(periods_emax, periods_payoffs_ex_post, &
                periods_future_payoffs, num_periods, max_states_period, &
                eps_relevant_periods, num_draws, states_number_period, & 
                periods_payoffs_ex_ante, edu_max, edu_start, &
                mapping_state_idx, states_all, delta)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: periods_emax(num_periods, &
    max_states_period)
    REAL(our_dble), INTENT(OUT)     :: periods_payoffs_ex_post(num_periods, &
    max_states_period, 4)
    REAL(our_dble), INTENT(OUT)     :: periods_future_payoffs(num_periods, &
    max_states_period, 4)

    REAL(our_dble), INTENT(IN)      :: eps_relevant_periods(:, :, :)
    REAL(our_dble), INTENT(IN)      :: periods_payoffs_ex_ante(:, :, :   )
    REAL(our_dble), INTENT(IN)      :: delta

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)    
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: states_number_period(:)
    INTEGER(our_int), INTENT(IN)    :: num_draws
    INTEGER(our_int), INTENT(IN)    :: max_states_period
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)

    !/* internals objects    */

    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: k

    REAL(our_dble)                  :: eps_relevant(num_draws, 4)
    REAL(our_dble)                  :: payoffs_ex_ante(4)
    REAL(our_dble)                  :: payoffs_ex_post(4)
    REAL(our_dble)                  :: future_payoffs(4)
    REAL(our_dble)                  :: emax

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
        
    ! Set to missing value
    periods_emax = missing_dble
    periods_future_payoffs = missing_dble
    periods_payoffs_ex_post = missing_dble

    ! Backward induction
    DO period = (num_periods - 1), 0, -1

        ! Extract disturbances
        eps_relevant = eps_relevant_periods(period + 1, :, :)

        ! Loop over all possible states, CAN K BE SIMPLIFIED
        DO k = 0, (states_number_period(period + 1) - 1)

            ! Extract payoffs
            payoffs_ex_ante = periods_payoffs_ex_ante(period + 1, k + 1, :)

            CALL get_payoffs_risk_lib(emax, payoffs_ex_post, future_payoffs, &
                    num_draws, eps_relevant, period, k, payoffs_ex_ante, & 
                    edu_max, edu_start, mapping_state_idx, states_all, & 
                    num_periods, periods_emax, delta)

            ! Collect information            
            periods_payoffs_ex_post(period + 1, k + 1, :) = payoffs_ex_post
            periods_future_payoffs(period + 1, k + 1, :) = future_payoffs
            periods_emax(period + 1, k + 1) = emax

        END DO

    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_payoffs_risk_lib(emax, payoffs_ex_post, future_payoffs, &
                num_draws, eps_baseline, period, k, payoffs_ex_ante, & 
                edu_max, edu_start, mapping_state_idx, states_all, num_periods, & 
                periods_emax, delta)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: emax
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

    REAL(our_dble), INTENT(IN)      :: eps_baseline(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_ex_ante(:)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)

    !/* internals objects    */
    
    REAL(our_dble), ALLOCATABLE     :: eps_relevant(:, :)

    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Allocate
    ALLOCATE(eps_relevant(num_draws, 4))

    ! Transform disturbances for occupations
    eps_relevant = eps_baseline
    DO i = 1, 2
        eps_relevant(:, i) = EXP(eps_relevant(:, i))
    END DO

    ! Simulated expected future value
    CALL simulate_emax_lib(emax, payoffs_ex_post, future_payoffs, num_periods, & 
            num_draws, period, k, eps_relevant, payoffs_ex_ante, edu_max, & 
            edu_start, periods_emax, states_all, mapping_state_idx, delta)
 
END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE create_state_space_lib(states_all, states_number_period, &
                mapping_state_idx, num_periods, edu_start, edu_max, min_idx)

    !/* external objects    */

    INTEGER(our_int), INTENT(OUT)   :: states_all(num_periods, 100000, 4)
    INTEGER(our_int), INTENT(OUT)   :: states_number_period(num_periods)
    INTEGER(our_int), INTENT(OUT)   :: mapping_state_idx(num_periods, & 
                                        num_periods, num_periods, min_idx, 2)

    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: min_idx

    !/* internals objects    */

    INTEGER(our_int)                :: edu_lagged
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: total
    INTEGER(our_int)                :: exp_A
    INTEGER(our_int)                :: exp_B
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: k
 
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ! Initialize output 
    states_number_period = missing_int
    mapping_state_idx    = missing_int
    states_all           = missing_int

    ! Construct state space by periods
    DO period = 0, (num_periods - 1)

        ! Count admissible realizations of state space by period
        k = 0

        ! Loop over all admissible work experiences for occupation A
        DO exp_A = 0, num_periods

            ! Loop over all admissible work experience for occupation B
            DO exp_B = 0, num_periods
                
                ! Loop over all admissible additional education levels
                DO edu = 0, num_periods

                    ! Agent cannot attain more additional education
                    ! than (EDU_MAX - EDU_START).
                    IF (edu .GT. edu_max - edu_start) THEN
                        CYCLE
                    END IF

                    ! Loop over all admissible values for leisure. Note that
                    ! the leisure variable takes only zero/value. The time path
                    ! does not matter.
                    DO edu_lagged = 0, 1

                        ! Check if lagged education admissible. (1) In the
                        ! first period all agents have lagged schooling equal
                        ! to one.
                        IF (edu_lagged .EQ. zero_int) THEN
                            IF (period .EQ. zero_int) THEN
                                CYCLE
                            END IF
                        END IF
                        
                        ! (2) Whenever an agent has not acquired any additional
                        ! education and we are not in the first period,
                        ! then this cannot be the case.
                        IF (edu_lagged .EQ. one_int) THEN
                            IF (edu .EQ. zero_int) THEN
                                IF (period .GT. zero_int) THEN
                                    CYCLE
                                END IF
                            END IF
                        END IF

                        ! (3) Whenever an agent has only acquired additional
                        ! education, then edu_lagged cannot be zero.
                        IF (edu_lagged .EQ. zero_int) THEN
                            IF (edu .EQ. period) THEN
                                CYCLE
                            END IF
                        END IF

                        ! Check if admissible for time constraints
                        total = edu + exp_A + exp_B

                        ! Note that the total number of activities does not
                        ! have is less or equal to the total possible number of
                        ! activities as the rest is implicitly filled with
                        ! leisure.
                        IF (total .GT. period) THEN
                            CYCLE
                        END IF
                        
                        ! Collect all possible realizations of state space
                        states_all(period + 1, k + 1, 1) = exp_A
                        states_all(period + 1, k + 1, 2) = exp_B
                        states_all(period + 1, k + 1, 3) = edu
                        states_all(period + 1, k + 1, 4) = edu_lagged

                        ! Collect mapping of state space to array index.
                        mapping_state_idx(period + 1, exp_A + 1, exp_B + 1, & 
                            edu + 1 , edu_lagged + 1) = k

                        ! Update count
                        k = k + 1

                     END DO

                 END DO

             END DO

         END DO
        
        ! Record maximum number of state space realizations by time period
        states_number_period(period + 1) = k

      END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE divergence_lib(div, x, cov, level)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: div

    REAL(our_dble), INTENT(IN)      :: x(2)
    REAL(our_dble), INTENT(IN)      :: cov(4,4)
    REAL(our_dble), INTENT(IN)      :: level

    !/* internals objects    */

    REAL(our_dble)                  :: alt_mean(4, 1) = zero_dble
    REAL(our_dble)                  :: old_mean(4, 1) = zero_dble
    REAL(our_dble)                  :: alt_cov(4,4)
    REAL(our_dble)                  :: old_cov(4,4)
    REAL(our_dble)                  :: inv_old_cov(4,4)
    REAL(our_dble)                  :: comp_a
    REAL(our_dble)                  :: comp_b(1, 1)
    REAL(our_dble)                  :: comp_c
    REAL(our_dble)                  :: rslt

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Construct alternative distribution
    alt_mean(1,1) = x(1)
    alt_mean(2,1) = x(2)
    alt_cov = cov

    ! Construct baseline distribution
    old_cov = cov

    ! Construct auxiliary objects.
    inv_old_cov = inverse_lib(old_cov, 4)

    ! Calculate first component
    comp_a = trace_fun(MATMUL(inv_old_cov, alt_cov))

    ! Calculate second component
    comp_b = MATMUL(MATMUL(TRANSPOSE(old_mean - alt_mean), inv_old_cov), &
                old_mean - alt_mean)

    ! Calculate third component
    comp_c = LOG(det_lib(alt_cov) / det_lib(old_cov))

    ! Statistic
    rslt = half_dble * (comp_a + comp_b(1,1) - four_dble + comp_c)

    ! Divergence
    div = level - rslt

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE simulate_emax_lib(emax_simulated, payoffs_ex_post, future_payoffs, & 
                num_periods, num_draws, period, k, eps_relevant, & 
                payoffs_ex_ante, edu_max, edu_start, emax, states_all, & 
                mapping_state_idx, delta)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: payoffs_ex_post(4)
    REAL(our_dble), INTENT(OUT)     :: emax_simulated
    REAL(our_dble), INTENT(OUT)     :: future_payoffs(4)

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:,:,:,:,:)
    INTEGER(our_int), INTENT(IN)    :: states_all(:,:,:)
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: num_draws
    INTEGER(our_int), INTENT(IN)    :: k
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: edu_start

    REAL(our_dble), INTENT(IN)      :: payoffs_ex_ante(:)
    REAL(our_dble), INTENT(IN)      :: eps_relevant(:,:)
    REAL(our_dble), INTENT(IN)      :: emax(:,:)
    REAL(our_dble), INTENT(IN)      :: delta

    !/* internals objects    */

    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: total_payoffs(4)
    REAL(our_dble)                  :: maximum

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    payoffs_ex_post = zero_dble
    emax_simulated = zero_dble
    future_payoffs = zero_dble

    ! Iterate over Monte Carlo draws
    DO i = 1, num_draws 

        ! Calculate ex post payoffs
        payoffs_ex_post(1) = payoffs_ex_ante(1) * eps_relevant(i, 1)
        payoffs_ex_post(2) = payoffs_ex_ante(2) * eps_relevant(i, 2)
        payoffs_ex_post(3) = payoffs_ex_ante(3) + eps_relevant(i, 3)
        payoffs_ex_post(4) = payoffs_ex_ante(4) + eps_relevant(i, 4)

        ! Check applicability
        IF (period .NE. (num_periods - one_int)) THEN

            ! Get future values
            CALL get_future_payoffs_lib(future_payoffs, edu_max, edu_start, & 
                    mapping_state_idx, period,  emax, k, states_all)

        END IF

        ! Calculate total utilities
        total_payoffs = payoffs_ex_post + delta * future_payoffs

        ! Determine optimal choice
        maximum = MAXVAL(total_payoffs)

        ! Recording expected future value
        emax_simulated = emax_simulated + maximum

    END DO

    ! Scaling
    emax_simulated = emax_simulated / num_draws

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE calculate_payoffs_ex_ante_lib(periods_payoffs_ex_ante, num_periods, &
              states_number_period, states_all, edu_start, coeffs_A, coeffs_B, & 
              coeffs_edu, coeffs_home, max_states_period)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: periods_payoffs_ex_ante(num_periods, &
                                            max_states_period, 4)

    REAL(our_dble), INTENT(IN)      :: coeffs_A(:)
    REAL(our_dble), INTENT(IN)      :: coeffs_B(:)
    REAL(our_dble), INTENT(IN)      :: coeffs_edu(:)
    REAL(our_dble), INTENT(IN)      :: coeffs_home(:)

    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: states_number_period(:)
    INTEGER(our_int), INTENT(IN)    :: states_all(:,:,:)
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: max_states_period

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
            periods_payoffs_ex_ante(period, k, 1) =  &
                EXP(DOT_PRODUCT(covars, coeffs_A))

            ! Calculate systematic part of payoff in occupation B
            periods_payoffs_ex_ante(period, k, 2) = &
                EXP(DOT_PRODUCT(covars, coeffs_B))

            ! Calculate systematic part of schooling utility
            payoff = coeffs_edu(1)

            ! Tuition cost for higher education if agents move
            ! beyond high school.
            IF(edu + edu_start >= 12) THEN

                payoff = payoff + coeffs_edu(2)
            
            END IF

            ! Psychic cost of going back to school
            IF(edu_lagged == 0) THEN
            
                payoff = payoff + coeffs_edu(3)
            
            END IF
            periods_payoffs_ex_ante(period, k, 3) = payoff

            ! Calculate systematic part of payoff in home production
            periods_payoffs_ex_ante(period, k, 4) = coeffs_home(1)

        END DO

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_future_payoffs_lib(future_payoffs, edu_max, edu_start, &
        mapping_state_idx, period, emax, k, states_all)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: future_payoffs(4)

    INTEGER(our_int), INTENT(IN)    :: k
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)

    REAL(our_dble), INTENT(IN)      :: emax(:, :)

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