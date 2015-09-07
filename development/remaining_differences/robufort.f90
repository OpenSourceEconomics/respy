MODULE robupy_program_constants

    !/* setup   */

    IMPLICIT NONE

!------------------------------------------------------------------------------ 
!   Parameters and Types
!------------------------------------------------------------------------------ 

    INTEGER, PARAMETER :: our_int   = selected_int_kind(9)
    INTEGER, PARAMETER :: our_sgle  = selected_real_kind(6,37)
    INTEGER, PARAMETER :: our_dble  = selected_real_kind(15,307)

    INTEGER(our_int), PARAMETER :: missing_int  = -99_our_int
    INTEGER(our_int), PARAMETER :: zero_int     = 0_our_int
    INTEGER(our_int), PARAMETER :: one_int      = 1_our_int
    INTEGER(our_int), PARAMETER :: two_int      = 2_our_int
    INTEGER(our_int), PARAMETER :: three_int    = 3_our_int
    INTEGER(our_int), PARAMETER :: four_int     = 4_our_int

    REAL(our_dble), PARAMETER :: missing_dble = -99_our_dble
    REAL(our_dble), PARAMETER :: zero_dble    = 0.00_our_dble
    REAL(our_dble), PARAMETER :: quarter_dble = 0.25_our_dble
    REAL(our_dble), PARAMETER :: half_dble    = 0.50_our_dble
    REAL(our_dble), PARAMETER :: one_dble     = 1.00_our_dble
    REAL(our_dble), PARAMETER :: two_dble     = 2.00_our_dble
    REAL(our_dble), PARAMETER :: three_dble   = 3.00_our_dble
    REAL(our_dble), PARAMETER :: four_dble    = 4.00_our_dble
    REAL(our_dble), PARAMETER :: upperClip    = 0.999999999999_our_dble
    REAL(our_dble), PARAMETER :: lowerClip    = 0.000000000001_our_dble
    REAL(our_dble), PARAMETER :: tiny         = 1.0e-20_our_dble
    
!******************************************************************************
!******************************************************************************
END MODULE 
MODULE robupy_auxiliary

    !/* external modules    */

    USE robupy_program_constants

    !/* setup   */

    IMPLICIT NONE

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE divergence(div, x, cov, level)

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
    inv_old_cov = inverse(old_cov, 4)

    ! Calculate first component
    comp_a = trace_fun(MATMUL(inv_old_cov, alt_cov))

    ! Calculate second component
    comp_b = MATMUL(MATMUL(TRANSPOSE(old_mean - alt_mean), inv_old_cov), &
                old_mean - alt_mean)

    ! Calculate third component
    comp_c = LOG(determinant(alt_cov) / determinant(old_cov))

    ! Statistic
    rslt = half_dble * (comp_a + comp_b(1,1) - four_dble + comp_c)

    ! Divergence
    div = level - rslt

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE stabilize_myopic(total_payoffs, future_payoffs)


    !/* external objects    */

    REAL(our_dble), INTENT(INOUT)   :: total_payoffs(:)
    REAL(our_dble), INTENT(IN)      :: future_payoffs(:)

    !/* internals objects    */

    LOGICAL                         :: is_huge

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ! Determine NAN
    is_huge = (future_payoffs(3) == -HUGE(future_payoffs))

    IF (is_huge .EQV. .True.) THEN
        total_payoffs(3) = -HUGE(future_payoffs)
    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION inverse(A, k)

    !/* external objects    */

  INTEGER(our_int), INTENT(IN)  :: k

  REAL(our_dble), INTENT(IN)    :: A(k, k)

    !/* internal objects    */
  
  REAL(our_dble), ALLOCATABLE   :: y(:, :)
  REAL(our_dble), ALLOCATABLE   :: B(:, :)
  REAL(our_dble)          :: d
  REAL(our_dble)          :: inverse(k, k)

  INTEGER(our_int), ALLOCATABLE :: indx(:)  
  INTEGER(our_int)        :: n
  INTEGER(our_int)        :: i
  INTEGER(our_int)        :: j

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
  inverse = y

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION determinant(A)

    !/* external objects    */

  REAL(our_dble), INTENT(IN)    :: A(:, :)
  REAL(our_dble)          :: determinant

    !/* internal objects    */
  INTEGER(our_int), ALLOCATABLE :: indx(:)
  INTEGER(our_int)        :: j
  INTEGER(our_int)        :: n

  REAL(our_dble), ALLOCATABLE   :: B(:, :)
  REAL(our_dble)          :: d

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
  determinant = d

END FUNCTION
!******************************************************************************
!******************************************************************************
PURE FUNCTION trace_fun(A)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)      :: A(:,:)
    REAL(our_dble)          :: trace_fun

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
!******************************************************************************
!******************************************************************************
SUBROUTINE ludcmp(A, d, indx)

    !/* external objects    */
    
    INTEGER(our_int), INTENT(INOUT) :: indx(:)

    REAL(our_dble), INTENT(INOUT)       :: a(:,:)
    REAL(our_dble), INTENT(INOUT)       :: d

    !/* internal objects    */

    INTEGER(our_int)                      :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: imax
    INTEGER(our_int)                :: n

    REAL(our_dble), ALLOCATABLE     :: vv(:)
    REAL(our_dble)                        :: dum
    REAL(our_dble)                  :: sums
    REAL(our_dble)                  :: aamax

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Auxiliary objects
    n = SIZE(A, DIM = 1)

    ! Initialize containers
    ALLOCATE(vv(n))

    ! Allocate containers
    d = one_dble

    ! Main
    DO i = 1, n

       aamax = zero_dble

       DO j = 1, n

          IF(abs(a(i, j)) > aamax) aamax = abs(a(i, j))

       END DO

       vv(i) = one_dble / aamax

    END DO

    DO j = 1, n

       DO i = 1, (j - 1)
    
          sums = a(i, j)
    
          DO k = 1, (i - 1)
    
             sums = sums - a(i, k)*a(k, j)
    
          END DO
    
       a(i,j) = sums
    
       END DO
    
       aamax = zero_dble
    
       DO i = j, n

          sums = a(i, j)

          DO k = 1, (j - 1)

             sums = sums - a(i, k)*a(k, j)

          END DO

          a(i, j) = sums

          dum = vv(i) * abs(sums)

          IF(dum >= aamax) THEN

            imax  = i

            aamax = dum

          END IF

       END DO

       IF(j /= imax) THEN

         DO k = 1, n

            dum = a(imax, k)

            a(imax, k) = a(j, k)

            a(j, k) = dum

         END DO

         d = -d

         vv(imax) = vv(j)

       END IF

       indx(j) = imax
       
       IF(a(j, j) == zero_dble) a(j, j) = tiny
       
       IF(j /= n) THEN
       
         dum = one_dble / a(j, j)
       
         DO i = (j + 1), n
       
            a(i, j) = a(i, j) * dum
       
         END DO
       
       END IF
    
    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE lubksb(A, B, indx)

    !/* external objects    */

    INTEGER(our_int), INTENT(IN)    :: indx(:)

    REAL(our_dble), INTENT(INOUT)       :: A(:, :)
    REAL(our_dble), INTENT(INOUT)   :: B(:)

    !/* internal objects    */

    INTEGER(our_int)                    :: ii
    INTEGER(our_int)                :: n
    INTEGER(our_int)                :: ll
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: i

    REAL(our_dble)                      :: sums

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Auxiliary objects
    n = SIZE(A, DIM = 1)

    ! Allocate containers
    ii = zero_dble

    ! Main
    DO i = 1, n
    
       ll = indx(i)
    
       sums = B(ll)                  
    
       B(ll) = B(i)
    
       IF(ii /= zero_dble) THEN
    
         DO j = ii, (i - 1)
    
            sums = sums - a(i, j) * b(j)

         END DO
    
       ELSE IF(sums /= zero_dble) THEN
    
         ii = i
    
       END IF
    
       b(i) = sums
    
    END DO
    
    DO i = n, 1, -1
    
       sums = b(i)
    
       DO j = (i + 1), n
    
          sums = sums - a(i, j) * b(j)
    
       END DO
    
       b(i) = sums / a(i, i)
    
    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
!******************************************************************************
!******************************************************************************
!
!   Interface to ROBUPY library.
!
!******************************************************************************
!******************************************************************************
MODULE robupy_library

    !/* external modules    */

    USE robupy_program_constants

    USE robupy_auxiliary

    !/* setup   */

    IMPLICIT NONE

    !/* core functions */

    PUBLIC :: calculate_payoffs_ex_ante 
    PUBLIC :: backward_induction 
    PUBLIC :: create_state_space 
    PUBLIC :: simulate_sample 

    !/* auxiliary functions */

    PUBLIC :: get_future_payoffs 
    PUBLIC :: get_payoffs_risk 
    PUBLIC :: simulate_emax 

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE simulate_sample(dataset, num_agents, states_all, num_periods, &
                mapping_state_idx, periods_payoffs_ex_ante, &
                periods_eps_relevant, edu_max, edu_start, periods_emax, delta)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: dataset(num_agents*num_periods, 8)

    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: periods_payoffs_ex_ante(:, :, :)
    REAL(our_dble), INTENT(IN)      :: periods_eps_relevant(:, :, :)
    REAL(our_dble), INTENT(IN)      :: delta

    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start

    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: num_agents
    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)

    !/* internal objects    */

    INTEGER(our_int)                :: i   
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: exp_A
    INTEGER(our_int)                :: exp_B
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: edu_lagged
    INTEGER(our_int)                :: choice(1)
    INTEGER(our_int)                :: count
    INTEGER(our_int)                :: current_state(4)

    REAL(our_dble)                  :: payoffs_ex_post(4)
    REAL(our_dble)                  :: payoffs_ex_ante(4)
    REAL(our_dble)                  :: disturbances(4)
    REAL(our_dble)                  :: future_payoffs(4)
    REAL(our_dble)                  :: total_payoffs(4)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    ! Initialize containers
    dataset = missing_dble

    ! Iterate over agents and periods
    count = 0

    DO i = 0, (num_agents - 1)

        ! Baseline state
        current_state = states_all(1, 1, :)
        
        DO period = 0, (num_periods - 1)
            
            ! Distribute state space
            exp_A = current_state(1)
            exp_B = current_state(2)
            edu = current_state(3)
            edu_lagged = current_state(4)
            
            ! Getting state index
            k = mapping_state_idx(period + 1, exp_A + 1, exp_B + 1, edu + 1, edu_lagged + 1)

            ! Write agent identifier and current period to data frame
            dataset(count + 1, 1) = DBLE(i)
            dataset(count + 1, 2) = DBLE(period)

            ! Calculate ex post payoffs
            payoffs_ex_ante = periods_payoffs_ex_ante(period + 1, k + 1, :)
            disturbances = periods_eps_relevant(period + 1, i + 1, :)

            ! Calculate total utilities
            CALL get_total_value(total_payoffs, payoffs_ex_post, & 
                    future_payoffs, period, num_periods, delta, &
                    payoffs_ex_ante, disturbances, edu_max, edu_start, & 
                    mapping_state_idx, periods_emax, k, states_all)

            ! Write relevant state space for period to data frame
            dataset(count + 1, 5:8) = current_state

            ! Special treatment for education
            dataset(count + 1, 7) = dataset(count + 1, 7) + edu_start

            ! Determine and record optimal choice
            choice = MAXLOC(total_payoffs) 

            dataset(count + 1, 3) = DBLE(choice(1)) 

            !# Update work experiences and education
            IF (choice(1) .EQ. one_int) THEN 
                current_state(1) = current_state(1) + 1
            END IF

            IF (choice(1) .EQ. two_int) THEN 
                current_state(2) = current_state(2) + 1
            END IF

            IF (choice(1) .EQ. three_int) THEN 
                current_state(3) = current_state(3) + 1
            END IF
            
            IF (choice(1) .EQ. three_int) THEN 
                current_state(4) = one_int
            ELSE
                current_state(4) = zero_int
            END IF

            ! Record earnings
            IF (choice(1) .EQ. one_int) THEN
                dataset(count + 1, 4) = payoffs_ex_post(1)
            END IF

            IF (choice(1) .EQ. two_int) THEN
                dataset(count + 1, 4) = payoffs_ex_post(2)
            END IF

            ! Update row indicator
            count = count + 1

        END DO

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE calculate_payoffs_ex_ante(periods_payoffs_ex_ante, num_periods, &
              states_number_period, states_all, edu_start, coeffs_A, coeffs_B, & 
              coeffs_edu, coeffs_home, max_states_period)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: periods_payoffs_ex_ante(num_periods, max_states_period, 4)

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
!*******************************************************************************
!*******************************************************************************
SUBROUTINE backward_induction(periods_emax, periods_payoffs_ex_post, &
                periods_future_payoffs, num_periods, max_states_period, &
                eps_relevant_periods, num_draws, states_number_period, & 
                periods_payoffs_ex_ante, edu_max, edu_start, &
                mapping_state_idx, states_all, delta)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(OUT)     :: periods_payoffs_ex_post(num_periods, max_states_period, 4)
    REAL(our_dble), INTENT(OUT)     :: periods_future_payoffs(num_periods, max_states_period, 4)

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

            CALL get_payoffs_risk(emax, payoffs_ex_post, future_payoffs, &
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
!******************************************************************************
!******************************************************************************
SUBROUTINE create_state_space(states_all, states_number_period, &
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
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_payoffs_risk(emax, payoffs_ex_post, future_payoffs, &
                num_draws, eps_relevant, period, k, payoffs_ex_ante, & 
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

    REAL(our_dble), INTENT(IN)      :: eps_relevant(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_ex_ante(:)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Simulated expected future value
    CALL simulate_emax(emax, payoffs_ex_post, future_payoffs, num_periods, & 
            num_draws, period, k, eps_relevant, payoffs_ex_ante, edu_max, & 
            edu_start, periods_emax, states_all, mapping_state_idx, delta)
    
END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE simulate_emax(emax_simulated, payoffs_ex_post, future_payoffs, & 
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
    REAL(our_dble)                  :: disturbances(4)
    REAL(our_dble)                  :: maximum

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    payoffs_ex_post = zero_dble
    emax_simulated = zero_dble

    ! Iterate over Monte Carlo draws
    DO i = 1, num_draws 

        ! Select disturbances for this draw
        disturbances = eps_relevant(i, :)

        ! Calculate total value
        CALL get_total_value(total_payoffs, payoffs_ex_post, future_payoffs, &
                period, num_periods, delta, payoffs_ex_ante, disturbances, &
                edu_max, edu_start, mapping_state_idx, emax, k, states_all)
        
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
SUBROUTINE get_total_value(total_payoffs, payoffs_ex_post, future_payoffs, &
                period, num_periods, delta, payoffs_ex_ante, & 
                disturbances, edu_max, edu_start, mapping_state_idx, & 
                periods_emax, k, states_all)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: total_payoffs(4)
    REAL(our_dble), INTENT(OUT)     :: payoffs_ex_post(4)
    REAL(our_dble), INTENT(OUT)     :: future_payoffs(4)

    INTEGER(our_int), INTENT(IN)    :: k
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)

    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: payoffs_ex_ante(:)
    REAL(our_dble), INTENT(IN)      :: disturbances(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)

    !/* internals objects    */

    LOGICAL                         :: is_myopic
    
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    ! Initialize containers
    payoffs_ex_post = zero_dble

    ! Auxiliary objects
    is_myopic = (delta .EQ. zero_dble)

    ! Calculate ex post payoffs
    payoffs_ex_post(1) = payoffs_ex_ante(1) * disturbances(1)
    payoffs_ex_post(2) = payoffs_ex_ante(2) * disturbances(2)
    payoffs_ex_post(3) = payoffs_ex_ante(3) + disturbances(3)
    payoffs_ex_post(4) = payoffs_ex_ante(4) + disturbances(4)

    ! Get future values
    IF (period .NE. (num_periods - one_int)) THEN
        CALL get_future_payoffs(future_payoffs, edu_max, edu_start, & 
                mapping_state_idx, period,  periods_emax, k, states_all)
        ELSE
            future_payoffs = zero_dble
    END IF

    ! Calculate total utilities
    total_payoffs = payoffs_ex_post + delta * future_payoffs

    ! Stabilization in case of myopic agents
    IF (is_myopic .EQV. .True.) THEN
        CALL stabilize_myopic(total_payoffs, future_payoffs)
    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_future_payoffs(future_payoffs, edu_max, edu_start, &
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

    INTEGER(our_int)                :: exp_A
    INTEGER(our_int)                :: exp_B
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: edu_lagged
    INTEGER(our_int)                :: future_idx

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
END MODULE  
!*******************************************************************************
!*******************************************************************************
PROGRAM robufort

USE robupy_library
USE robupy_program_constants

INTEGER :: j, k, num_periods, edu_start, edu_max, num_draws, num_agents, & 
seed_simulation, seed_solution, period

1500 FORMAT(6(1x,f15.10))
1510 FORMAT(f15.10)

1505 FORMAT(i10)
1515 FORMAT(i10,1x,i10)

DOUBLE PRECISION :: coeffs_A(6), coeffs_B(6), delta, coeffs_edu(3), coeffs_home(1), shocks(4, 4)

INTEGER :: min_idx, max_states_period

    INTEGER                :: i
DOUBLE PRECISION                  :: total_payoffs(4), emax
   DOUBLE PRECISION                   :: disturbances(4)
   DOUBLE PRECISION                  :: maximum, emax_simulated
   DOUBLE PRECISION                  :: payoffs_ex_post(4), payoffs_ex_ante(4), future_payoffs(4)
    INTEGER                :: exp_A
    INTEGER               :: exp_B
    INTEGER                :: edu
    INTEGER       :: edu_lagged
    INTEGER               :: future_idx

! Auxiliary objects

INTEGER, ALLOCATABLE  :: states_all(:, :, :)
INTEGER, ALLOCATABLE  :: states_number_period(:)
INTEGER, ALLOCATABLE  :: mapping_state_idx(:, :, :, :, :)
DOUBLE PRECISION, ALLOCATABLE  :: periods_payoffs_ex_ante(:, :, :)
DOUBLE PRECISION, ALLOCATABLE  :: periods_emax(:, :), periods_payoffs_ex_post(:, :, :)
DOUBLE PRECISION, ALLOCATABLE :: periods_future_payoffs(:, :, :)
DOUBLE PRECISION, ALLOCATABLE :: eps_relevant_periods(:, :, :), eps_relevant(:, :)




OPEN(UNIT=1, FILE='model.robufort.ini')

    ! BASICS
    READ(1, 1505) num_periods
    READ(1, 1510) delta

    ! WORK
    READ(1, 1500) coeffs_A
    READ(1, 1500) coeffs_B

    ! EDUCATION
    READ(1, 1500) coeffs_edu
    READ(1, 1515) edu_start, edu_max

    ! HOME
    READ(1, 1500) coeffs_home

    ! SHOCKS
    DO j = 1, 4
        READ(1, 1500) (shocks(j, k), k=1, 4)
    END DO

    ! SOLUTION
    READ(1, 1505) num_draws
    READ(1, 1505) seed_solution

    ! SIMULATION
    READ(1, 1505) num_agents
    READ(1, 1505) seed_simulation


! Auxiliary
min_idx = MIN(num_periods, (edu_max - edu_start + 1))


! Allocate arrays
ALLOCATE(states_all(num_periods, 100000, 4))
ALLOCATE(states_number_period(num_periods))
ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))

! Create the state space of the model
CALL create_state_space(states_all, states_number_period, mapping_state_idx, & 
        num_periods, edu_start, edu_max, min_idx)

! Allocate additional containers
max_states_period = MAXVAL(states_number_period)
ALLOCATE(periods_payoffs_ex_ante(num_periods, max_states_period, 4))

! Calculate the ex ante payoffs
CALL calculate_payoffs_ex_ante(periods_payoffs_ex_ante, num_periods, &
        states_number_period, states_all, edu_start, coeffs_A, coeffs_B, & 
        coeffs_edu, coeffs_home, max_states_period)

! Allocate additional containers
ALLOCATE(periods_emax(num_periods, max_states_period))
ALLOCATE(periods_payoffs_ex_post(num_periods, max_states_period, 4))
ALLOCATE(periods_future_payoffs(num_periods, max_states_period, 4))
ALLOCATE(eps_relevant_periods(num_periods, num_draws, 4))

ALLOCATE(eps_relevant(num_draws, 4))

! Draw random disturbances
eps_relevant_periods = 1.00

IF (.True.) THEN

CALL backward_induction(periods_emax, periods_payoffs_ex_post, &
                periods_future_payoffs, num_periods, max_states_period, &
                eps_relevant_periods, num_draws, states_number_period, & 
                periods_payoffs_ex_ante, edu_max, edu_start, &
                mapping_state_idx, states_all, delta)
ELSE

    ! Set to missing value
    periods_emax = -99.0
    periods_future_payoffs = -99.0
    periods_payoffs_ex_post = -99.0

    ! Backward induction
    DO period = (num_periods - 1), 0, -1

    PRINT *, period
        ! Extract disturbances
        eps_relevant = eps_relevant_periods(period + 1, :, :)

        ! Loop over all possible states, CAN K BE SIMPLIFIED
        DO k = 0, (states_number_period(period + 1) - 1)

            ! Extract payoffs
            payoffs_ex_ante = periods_payoffs_ex_ante(period + 1, k + 1, :)

    ! Initialize containers
    payoffs_ex_post = 0.00
    emax_simulated = 0.00

    ! Iterate over Monte Carlo draws
    DO i = 1, num_draws 

        ! Select disturbances for this draw
        disturbances = eps_relevant(i, :)
    ! Initialize containers
    payoffs_ex_post = zero_dble

    ! Auxiliary objects
    is_myopic = (delta .EQ. zero_dble)

    ! Calculate ex post payoffs
    payoffs_ex_post(1) = payoffs_ex_ante(1) * disturbances(1)
    payoffs_ex_post(2) = payoffs_ex_ante(2) * disturbances(2)
    payoffs_ex_post(3) = payoffs_ex_ante(3) + disturbances(3)
    payoffs_ex_post(4) = payoffs_ex_ante(4) + disturbances(4)

    ! Get future values
    IF (period .NE. (num_periods - one_int)) THEN
        
        IF (.TRUE.) THEN
            !PRINT *, 'pei'
            CALL get_future_payoffs(future_payoffs, edu_max, edu_start, & 
                    mapping_state_idx, period,  periods_emax, k, states_all)

        ELSE IF (.TRUE.) THEN

            CALL get_future_payoffs(future_payoffs, edu_max, edu_start, & 
                    mapping_state_idx, period,  periods_emax, k, states_all)

         ELSE




                ! Distribute state space
                exp_A = states_all(period + 1, k + 1, 1)
                exp_B = states_all(period + 1, k + 1, 2)
                edu = states_all(period + 1, k + 1, 3)
                edu_lagged = states_all(period + 1, k + 1, 4)

                ! Working in occupation A
                future_idx = mapping_state_idx(period + 1 + 1, exp_A + 1 + 1, & 
                                exp_B + 1, edu + 1, 1)
                future_payoffs(1) = periods_emax(period + 1 + 1, future_idx + 1)

                !Working in occupation B
                future_idx = mapping_state_idx(period + 1 + 1, exp_A + 1, & 
                                exp_B + 1 + 1, edu + 1, 1)
                future_payoffs(2) = periods_emax(period + 1 + 1, future_idx + 1)

                ! Increasing schooling. Note that adding an additional year
                ! of schooling is only possible for those that have strictly
                ! less than the maximum level of additional education allowed.
                IF (edu < edu_max - edu_start) THEN
                    future_idx = mapping_state_idx(period + 1 + 1, exp_A + 1, &
                                    exp_B + 1, edu + 1 + 1, 2)
                    future_payoffs(3) = periods_emax(period + 1 + 1, future_idx + 1)
                ELSE
                    future_payoffs(3) = -HUGE(future_payoffs)
                END IF

                ! Staying at home
                future_idx = mapping_state_idx(period + 1 + 1, exp_A + 1, & 
                                exp_B + 1, edu + 1, 1)
                future_payoffs(4) = periods_emax(period + 1 + 1, future_idx + 1)


            END IF





        

        ELSE
            future_payoffs = zero_dble
    END IF
    ! Calculate total utilities
    total_payoffs = payoffs_ex_post + delta * future_payoffs

    ! Stabilization in case of myopic agents
    
        maximum = MAXVAL(total_payoffs)

        ! Recording expected future value
        emax_simulated = emax_simulated + maximum

    END DO

    ! Scaling
    emax_simulated = emax_simulated / num_draws

            ! Collect information            
            periods_payoffs_ex_post(period + 1, k + 1, :) = payoffs_ex_post
            periods_future_payoffs(period + 1, k + 1, :) = future_payoffs
            periods_emax(period + 1, k + 1) = emax

        END DO

    END DO

END IF

!******************************************************************************
!******************************************************************************
END PROGRAM