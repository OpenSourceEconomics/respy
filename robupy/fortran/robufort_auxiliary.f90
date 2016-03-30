MODULE robufort_auxiliary

    !/*	external modules	    */

    USE robufort_constants

	!/*	setup	                */

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
SUBROUTINE get_total_value(total_payoffs, period, num_periods, delta, &
                payoffs_systematic, draws, edu_max, edu_start, &
                mapping_state_idx, periods_emax, k, states_all)

    !   Development Note:
    !
    !       The VECTORIZATION supports the inlining and vectorization
    !       preparations in the build process.

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: total_payoffs(:)

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: draws(:)
    REAL(our_dble), INTENT(IN)      :: delta

    !/* internal objects        */

    REAL(our_dble)                  :: payoffs_future(4)
    REAL(our_dble)                  :: payoffs_ex_post(4)


!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Initialize containers
    payoffs_ex_post = zero_dble

    ! Calculate ex post payoffs
    payoffs_ex_post(1) = payoffs_systematic(1) * draws(1)
    payoffs_ex_post(2) = payoffs_systematic(2) * draws(2)
    payoffs_ex_post(3) = payoffs_systematic(3) + draws(3)
    payoffs_ex_post(4) = payoffs_systematic(4) + draws(4)

    ! Get future values
    IF (period .NE. (num_periods - one_int)) THEN
        CALL get_future_payoffs(payoffs_future, edu_max, edu_start, &
                mapping_state_idx, period,  periods_emax, k, states_all)
        ELSE
            payoffs_future = zero_dble
    END IF

    ! Calculate total utilities
    total_payoffs = payoffs_ex_post + delta * payoffs_future

    ! This is required to ensure that the agent does not choose any
    ! inadmissible states.
    IF (payoffs_future(3) == -HUGE_FLOAT) THEN
        total_payoffs(3) = -HUGE_FLOAT
    END IF

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_future_payoffs(payoffs_future, edu_max, edu_start, &
                mapping_state_idx, period, periods_emax, k, states_all)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: payoffs_future(:)

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)

    !/* internals objects       */

    INTEGER(our_int)                :: edu_lagged
    INTEGER(our_int)                :: future_idx
    INTEGER(our_int)    			:: exp_a
    INTEGER(our_int)    			:: exp_b
    INTEGER(our_int)    			:: edu

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Distribute state space
    exp_a = states_all(period + 1, k + 1, 1)
    exp_b = states_all(period + 1, k + 1, 2)
    edu = states_all(period + 1, k + 1, 3)
    edu_lagged = states_all(period + 1, k + 1, 4)

	! Working in occupation A
    future_idx = mapping_state_idx(period + 1 + 1, exp_a + 1 + 1, &
                    exp_b + 1, edu + 1, 1)
    payoffs_future(1) = periods_emax(period + 1 + 1, future_idx + 1)

	!Working in occupation B
    future_idx = mapping_state_idx(period + 1 + 1, exp_a + 1, &
                    exp_b + 1 + 1, edu + 1, 1)
    payoffs_future(2) = periods_emax(period + 1 + 1, future_idx + 1)

	! Increasing schooling. Note that adding an additional year
	! of schooling is only possible for those that have strictly
	! less than the maximum level of additional education allowed.
    IF (edu < edu_max - edu_start) THEN
        future_idx = mapping_state_idx(period + 1 + 1, exp_a + 1, &
                        exp_b + 1, edu + 1 + 1, 2)
        payoffs_future(3) = periods_emax(period + 1 + 1, future_idx + 1)
    ELSE
        payoffs_future(3) = -HUGE_FLOAT
    END IF

	! Staying at home
    future_idx = mapping_state_idx(period + 1 + 1, exp_a + 1, &
                    exp_b + 1, edu + 1, 1)
    payoffs_future(4) = periods_emax(period + 1 + 1, future_idx + 1)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_exogenous_variables(independent_variables, maxe, period, &
                num_periods, num_states, delta, periods_payoffs_systematic, &
                shifts, edu_max, edu_start, mapping_state_idx, periods_emax, &
                states_all)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: independent_variables(:, :)
    REAL(our_dble), INTENT(OUT)         :: maxe(:)

    REAL(our_dble), INTENT(IN)          :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), INTENT(IN)          :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)          :: shifts(:)
    REAL(our_dble), INTENT(IN)          :: delta

    INTEGER(our_int), INTENT(IN)        :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)        :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)        :: num_periods
    INTEGER(our_int), INTENT(IN)        :: num_states
    INTEGER(our_int), INTENT(IN)        :: edu_start
    INTEGER(our_int), INTENT(IN)        :: edu_max
    INTEGER(our_int), INTENT(IN)        :: period


    !/* internal objects        */

    REAL(our_dble)                      :: payoffs_systematic(4)
    REAL(our_dble)                      :: total_payoffs(4)
    REAL(our_dble)                      :: diff(4)

    INTEGER(our_int)                    :: k

    LOGICAL                             :: is_inadmissible

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Construct exogenous variable for all states
    DO k = 0, (num_states - 1)

        payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

        CALL get_total_value(total_payoffs, &
                period, num_periods, delta, payoffs_systematic, shifts, &
                edu_max, edu_start, mapping_state_idx, periods_emax, k, &
                states_all)

        ! Treatment of inadmissible states, which will show up in the regression
        ! in some way
        is_inadmissible = (total_payoffs(3) == -HUGE_FLOAT)

        IF (is_inadmissible) THEN

            total_payoffs(3) = interpolation_inadmissible_states

        END IF

        ! Implement level shifts
        maxe(k + 1) = MAXVAL(total_payoffs)

        diff = maxe(k + 1) - total_payoffs

        ! Construct regressors
        independent_variables(k + 1, 1:4) = diff
        independent_variables(k + 1, 5:8) = DSQRT(diff)

    END DO

    ! Add intercept to set of independent variables
    independent_variables(:, 9) = one_dble

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_predictions(predictions, endogenous, exogenous, maxe, &
              is_simulated, num_points, num_states)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)       :: predictions(:)

    REAL(our_dble), INTENT(IN)        :: exogenous(:, :)
    REAL(our_dble), INTENT(IN)        :: endogenous(:)
    REAL(our_dble), INTENT(IN)        :: maxe(:)

    INTEGER, INTENT(IN)               :: num_states
    INTEGER, INTENT(IN)               :: num_points

    LOGICAL, INTENT(IN)               :: is_simulated(:)

    !/* internal objects        */

    REAL(our_dble)                    :: endogenous_predicted_available(num_points)
    REAL(our_dble)                    :: exogenous_is_available(num_points, 9)
    REAL(our_dble)                    :: endogenous_is_available(num_points)
    REAL(our_dble)                    :: endogenous_predicted(num_states)
    REAL(our_dble)                    :: coeffs(9)
    REAL(our_dble)                    :: r_squared

    INTEGER(our_int)                  :: i
    INTEGER(our_int)                  :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Select pairs where both endogenous and exogenous information is available.
    i = 1
            
    DO k = 0, (num_states - 1) 

        IF (is_simulated(k + 1)) THEN

            endogenous_is_available(i) = endogenous(k + 1)
            exogenous_is_available(i, :) = exogenous(k + 1, :)

            i = i + 1

        END IF

    END DO

    ! Fit the prediction model on interpolation points.
    CALL get_coefficients(coeffs, endogenous_is_available, & 
            exogenous_is_available, 9, num_points)
 
    ! Use the model to predict EMAX for all states and subsequently
    ! replace the values where actual values are available. As in
    ! Keane & Wolpin (1994), negative predictions are truncated to zero.
    CALL point_predictions(endogenous_predicted, exogenous, coeffs, num_states)

    ! Construct coefficient of determination for the subset of interpolation 
    ! points.
    i = 1
            
    DO k = 0, (num_states - 1) 

        IF (is_simulated(k + 1)) THEN

            endogenous_predicted_available(i) = endogenous_predicted(k + 1)
           
            i = i + 1

        END IF

    END DO

    CALL get_r_squared(r_squared, endogenous_is_available, &
            endogenous_predicted_available, num_points) 
            
    CALL get_clipped_vector(endogenous_predicted, endogenous_predicted, & 
            zero_dble, HUGE_FLOAT, num_states)

    ! Construct predicted EMAX for all states and the replace
    ! interpolation points with simulated values.
    predictions = endogenous_predicted + maxe
            
    DO k = 0, (num_states - 1) 

        IF (is_simulated(k + 1)) THEN

            predictions(k + 1) = endogenous(k + 1) + maxe(k + 1)

        END IF

    END DO

    ! Perform some basic logging to spot problems early.
    CALL logging_prediction_model(coeffs, r_squared)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE random_choice(sample, candidates, num_candidates, num_points)

  !
  ! Source
  ! ------
  !
  !   KNUTH, D. The art of computer programming. Vol II.
  !     Seminumerical Algorithms. Reading, Mass: AddisonWesley, 1969
  !

    !/* external objects    */

    INTEGER, INTENT(OUT)          :: sample(:)

    INTEGER, INTENT(IN)           :: num_candidates
    INTEGER, INTENT(IN)           :: candidates(:)
    INTEGER, INTENT(IN)           :: num_points

    !/* internal objects    */

    INTEGER(our_int)              :: m
    INTEGER(our_int)              :: j
    INTEGER(our_int)              :: l

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
  
    ! Initialize auxiliary objects
    m = 0

    ! Draw random points from candidates
    DO j = 1, num_candidates

        l = INT(DBLE(num_candidates - j + 1) * RAN(0)) + 1

        IF (l .GT. (num_points - m)) THEN

            CONTINUE

        ELSE

            m = m + 1

            sample(m) = candidates(j)

            IF (m .GE. num_points) THEN

                RETURN

            END IF

        END IF

    END DO


END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE logging_prediction_model(coeffs, r_squared)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: coeffs(:)
    REAL(our_dble), INTENT(IN)      :: r_squared

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Define format for coefficients and R-squared
    1900 FORMAT(2x,A18,3x,9(1x,f10.4))

    2900 FORMAT(2x,A18,4x,f10.4)

    ! Write to file
    OPEN(UNIT=99, FILE='logging.robupy.sol.log', ACCESS='APPEND')

        WRITE(99, *) "     Information about Prediction Model"

        WRITE(99, *) " "

        WRITE(99, 1900) 'Coefficients', coeffs

        WRITE(99, *) " "

        WRITE(99, 2900) 'R-squared', r_squared    

        WRITE(99, *) " "

        WRITE(99, *) " "

    CLOSE(99)
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE logging_solution(indicator, period, num_states)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)            :: indicator

    INTEGER(our_int), INTENT(IN), OPTIONAL  :: num_states
    INTEGER(our_int), INTENT(IN), OPTIONAL  :: period

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    OPEN(UNIT=99, FILE='logging.robupy.sol.log', ACCESS='APPEND')
    
    ! State space creation
    IF (indicator == 1) THEN
    
        ! Delete  any previous versions
        CLOSE(99, STATUS ='DELETE')
        OPEN(UNIT=99, FILE='logging.robupy.sol.log', ACCESS='APPEND')

        WRITE(99, *) " Starting state space creation   "
        WRITE(99, *) ""

    ! Ex Ante Payoffs
    ELSEIF (indicator == 2) THEN

      WRITE(99, *) " Starting calculation of systematic payoffs  "
      WRITE(99, *) ""

    ! Backward induction procedure
    ELSEIF (indicator == 3) THEN

      WRITE(99, *) " Starting backward induction procedure "
      WRITE(99, *) ""
    
    ELSEIF (indicator == 4) THEN

        1900 FORMAT(2x,A18,1x,i2,1x,A4,1x,i5,1x,A6)

        WRITE(99, 1900) "... solving period", period, 'with', num_states, 'states'
        WRITE(99, *) ""
   
    ! Finishing
    ELSEIF (indicator == -1) THEN

        WRITE(99, *) " ... finished "
        WRITE(99, *) ""
        WRITE(99, *) ""
        
    END IF
  
  CLOSE(99)
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
FUNCTION get_simulated_indicator(num_points, num_states, period, num_periods, &
            is_debug)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)      :: num_periods
    INTEGER(our_int), INTENT(IN)      :: num_states
    INTEGER(our_int), INTENT(IN)      :: num_points
    INTEGER(our_int), INTENT(IN)      :: period

    LOGICAL, INTENT(IN)               :: is_debug

    !/* internal objects        */
  
    INTEGER(our_int)                  :: candidates(num_states)
    INTEGER(our_int)                  :: sample(num_points)
    INTEGER(our_int)                  :: i

    LOGICAL                           :: is_simulated_container(num_states, num_periods)
    LOGICAL                           :: get_simulated_indicator(num_states)
    LOGICAL                           :: is_simulated(num_states)
    LOGICAL                           :: READ_IN

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Initialize set of candidate values
    candidates = (/ (i, i = 1, num_states) /)

    ! Check if standardization requested
    IF (is_debug) THEN

        INQUIRE(FILE='interpolation.txt', EXIST=READ_IN)

        IF (READ_IN) THEN

            OPEN(12, file='interpolation.txt')

               DO i = 1, num_states
            
                    READ(12, *)  is_simulated_container(i, :)     
              
                END DO
        
            CLOSE(12)

            get_simulated_indicator = is_simulated_container(:, period + 1)

            RETURN

        END IF

    END IF

    ! Handle special cases
    IF (num_points .EQ. num_states) THEN

        get_simulated_indicator = .TRUE.

        RETURN

    ELSEIF (num_points .GT. num_states) THEN

        PRINT *, ' Bad Request in interpolation code'

        STOP

    END IF

    ! Draw a random sample
    CALL random_choice(sample, candidates, num_states, num_points)
    
    ! Update information indicator
    is_simulated = .False.

    DO i = 1, num_points

        is_simulated(sample(i)) = .True.

    END DO

    ! Finish
    get_simulated_indicator = is_simulated

END FUNCTION
!******************************************************************************
!******************************************************************************
PURE FUNCTION normal_pdf(x, mean, sd)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: mean
    REAL(our_dble), INTENT(IN)      :: sd
    REAL(our_dble), INTENT(IN)      :: x
    
    !/*  internal objects       */

    REAL(our_dble)                  :: normal_pdf
    REAL(our_dble)                  :: std

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    std = ((x - mean) / sd)

    normal_pdf = (one_dble / sd) * (one_dble / sqrt(two_dble * pi))

    normal_pdf = normal_pdf * exp( -(std * std) / two_dble)

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION clip_value(value, lower_bound, upper_bound)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)  :: lower_bound
    REAL(our_dble), INTENT(IN)  :: upper_bound
    REAL(our_dble), INTENT(IN)  :: value

    !/*  internal objects       */

    REAL(our_dble)              :: clip_value

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF(value < lower_bound) THEN

        clip_value = lower_bound

    ELSEIF(value > upper_bound) THEN

        clip_value = upper_bound

    ELSE

        clip_value = value

    END IF

END FUNCTION
!*******************************************************************************
!*******************************************************************************
FUNCTION inverse(A, k)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)  :: k

    REAL(our_dble), INTENT(IN)    :: A(:, :)

    !/* internal objects        */
  
    REAL(our_dble), ALLOCATABLE   :: y(:, :)
    REAL(our_dble), ALLOCATABLE   :: B(:, :)

    REAL(our_dble)                :: inverse(k, k)
    REAL(our_dble)                :: d

    INTEGER(our_int), ALLOCATABLE :: indx(:)  

    INTEGER(our_int)              :: n
    INTEGER(our_int)              :: i
    INTEGER(our_int)              :: j

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
  
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
!*******************************************************************************
!*******************************************************************************
FUNCTION determinant(A)

    !/* external objects        */

    REAL(our_dble)                :: determinant

    REAL(our_dble), INTENT(IN)    :: A(:, :)

    !/* internal objects        */

    INTEGER(our_int), ALLOCATABLE :: indx(:)

    INTEGER(our_int)              :: j
    INTEGER(our_int)              :: n

    REAL(our_dble), ALLOCATABLE   :: B(:, :)
    
    REAL(our_dble)                :: d

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

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
!*******************************************************************************
!*******************************************************************************
PURE FUNCTION trace_fun(A)

    !/* external objects        */

    REAL(our_dble)              :: trace_fun

    REAL(our_dble), INTENT(IN)  :: A(:,:)

    !/* internals objects       */

    INTEGER(our_int)            :: i
    INTEGER(our_int)            :: n

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Get dimension
    n = SIZE(A, DIM = 1)

    ! Initialize results
    trace_fun = zero_dble

    ! Calculate trace
    DO i = 1, n

        trace_fun = trace_fun + A(i, i)

    END DO

END FUNCTION
!*******************************************************************************
!*******************************************************************************
SUBROUTINE ludcmp(A, d, indx)

    !/* external objects        */
    
    INTEGER(our_int), INTENT(INOUT) :: indx(:)

    REAL(our_dble), INTENT(INOUT)   :: a(:,:)
    REAL(our_dble), INTENT(INOUT)   :: d

    !/* internal objects        */

    INTEGER(our_int)                :: imax
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: n

    REAL(our_dble), ALLOCATABLE     :: vv(:)


    REAL(our_dble)                  :: aamax
    REAL(our_dble)                  :: sums
    REAL(our_dble)                  :: dum

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Initialize containers
    imax = MISSING_INT 
    
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
       
       IF(a(j, j) == zero_dble) a(j, j) = TINY_FLOAT
       
       IF(j /= n) THEN
       
         dum = one_dble / a(j, j)
       
         DO i = (j + 1), n
       
            a(i, j) = a(i, j) * dum
       
         END DO
       
       END IF
    
    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE lubksb(A, B, indx)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: indx(:)

    REAL(our_dble), INTENT(INOUT)   :: A(:, :)
    REAL(our_dble), INTENT(INOUT)   :: B(:)

    !/* internal objects        */

    INTEGER(our_int)                :: ii
    INTEGER(our_int)                :: ll
    INTEGER(our_int)                :: n
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: sums

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Auxiliary objects
    n = SIZE(A, DIM = 1)

    ! Allocate containers
    ii = zero_int

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
!*******************************************************************************
!*******************************************************************************
SUBROUTINE svd(U, S, VT, A, m)
    
    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: VT(:, :)
    REAL(our_dble), INTENT(OUT)     :: U(:, :)
    REAL(our_dble), INTENT(OUT)     :: S(:) 

    REAL(our_dble), INTENT(IN)      :: A(:, :)
    
    INTEGER(our_int), INTENT(IN)    :: m

    !/* internal objects        */

    INTEGER(our_int)                :: LWORK
    INTEGER(our_int)                :: INFO

    REAL(our_dble), ALLOCATABLE     :: IWORK(:)
    REAL(our_dble), ALLOCATABLE     :: WORK(:)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
  
    ! Auxiliary objects
    LWORK =  M * (7 + 4 * M)

    ! Allocate containers
    ALLOCATE(WORK(LWORK)); ALLOCATE(IWORK(8 * M))

    ! Call LAPACK routine
    CALL DGESDD( 'A', m, m, A, m, S, U, m, VT, m, WORK, LWORK, IWORK, INFO)

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
FUNCTION pinv(A, m)

    !/* external objects        */

    REAL(our_dble)                  :: pinv(m, m)

    REAL(our_dble), INTENT(IN)      :: A(:, :)
    
    INTEGER(our_int), INTENT(IN)    :: m


    !/* internal objects        */

    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: VT(m, m)
    REAL(our_dble)                  :: UT(m, m) 
    REAL(our_dble)                  :: U(m, m)
    REAL(our_dble)                  :: cutoff
    REAL(our_dble)                  :: S(m) 
 
!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL svd(U, S, VT, A, m)

    cutoff = 1e-15_our_dble * MAXVAL(S)

    DO i = 1, M

        IF (S(i) .GT. cutoff) THEN

            S(i) = one_dble / S(i)

        ELSE 

            S(i) = zero_dble

        END IF

    END DO

    UT = TRANSPOSE(U)

    DO i = 1, M

        pinv(i, :) = S(i) * UT(i,:)

    END DO

    pinv = MATMUL(TRANSPOSE(VT), pinv)

END FUNCTION
!*******************************************************************************
!*******************************************************************************
SUBROUTINE cholesky(factor, matrix)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: factor(:,:)

    REAL(our_dble), INTENT(IN)      :: matrix(:, :)

    !/* internal objects        */

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: n
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: j

    REAL(our_dble), ALLOCATABLE     :: clon(:, :)

    REAL(our_dble)                  :: sums

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Initialize result
    factor = zero_dble

    ! Auxiliary objects
    n = size(matrix,1)
   
    ! Allocate containers
    ALLOCATE(clon(n,n))
    
    ! Apply Cholesky decomposition
    clon = matrix
    
    DO j = 1, n

      sums = 0.0
      
      DO k = 1, (j - 1)

        sums = sums + clon(j, k)**2

      END DO

      clon(j, j) = DSQRT(clon(j, j) - sums)
       
      DO i = (j + 1), n

        sums = zero_dble

        DO k = 1, (j - 1)

          sums = sums + clon(j, k) * clon(i, k)

        END DO

        clon(i, j) = (clon(i, j) - sums) / clon(j, j)

      END DO
    
    END DO
    
    ! Transfer information from matrix to factor
    DO i = 1, n
    
      DO j = 1, n  
    
        IF(i .LE. j) THEN
    
          factor(j, i) = clon(j, i) 
    
        END IF
    
      END DO
    
    END DO

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE standard_normal(draw)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: draw(:)

    !/* internal objects        */

    INTEGER(our_int)                :: dim
    INTEGER(our_int)                :: g
    
    REAL(our_dble), ALLOCATABLE     :: u(:)
    REAL(our_dble), ALLOCATABLE     :: r(:)

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

    ! Auxiliary objects
    dim = SIZE(draw)

    ! Allocate containers
    ALLOCATE(u(2 * dim)); ALLOCATE(r(2 * dim))

    ! Call uniform deviates
    CALL RANDOM_NUMBER(u)

    ! Apply Box-Muller transform
    DO g = 1, (2 * dim), 2

       r(g) = DSQRT(-two_dble * LOG(u(g)))*COS(two_dble *pi * u(g + one_int)) 
       r(g + 1) = DSQRT(-two_dble * LOG(u(g)))*SIN(two_dble *pi * u(g + one_int)) 

    END DO

    ! Extract relevant floats
    DO g = 1, dim 

       draw(g) = r(g)     

    END DO

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE multivariate_normal(draws, mean, covariance)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)           :: draws(:, :)

    REAL(our_dble), INTENT(IN), OPTIONAL  :: covariance(:, :)
    REAL(our_dble), INTENT(IN), OPTIONAL  :: mean(:)
    
    !/* internal objects        */
    
    INTEGER(our_int)                :: num_draws_emax
    INTEGER(our_int)                :: dim
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j  

    REAL(our_dble), ALLOCATABLE     :: covariance_internal(:, :)
    REAL(our_dble), ALLOCATABLE     :: mean_internal(:)
    REAL(our_dble), ALLOCATABLE     :: ch(:, :)
    REAL(our_dble), ALLOCATABLE     :: z(:, :)

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

    ! Auxiliary objects
    num_draws_emax = SIZE(draws, 1)

    dim       = SIZE(draws, 2)

    ! Handle optional arguments
    ALLOCATE(mean_internal(dim)); ALLOCATE(covariance_internal(dim, dim))

    IF (PRESENT(mean)) THEN

      mean_internal = mean

    ELSE

      mean_internal = zero_dble

    END IF

    IF (PRESENT(covariance)) THEN

      covariance_internal = covariance

    ELSE

      covariance_internal = zero_dble

      DO j = 1, dim

        covariance_internal(j, j) = one_dble

      END DO

    END IF

    ! Allocate containers
    ALLOCATE(z(dim, 1)); ALLOCATE(ch(dim, dim))

    ! Initialize containers
    ch = zero_dble

    ! Construct Cholesky decomposition
    CALL cholesky(ch, covariance_internal) 

    ! Draw deviates
    DO i = 1, num_draws_emax
       
       CALL standard_normal(z(:, 1))
       
       draws(i, :) = MATMUL(ch, z(:, 1)) + mean_internal(:)  
    
    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_clipped_vector(Y, X, lower_bound, upper_bound, num_values)

    !/* external objects        */

    REAL(our_dble), INTENT(INOUT)       :: Y(:)

    REAL(our_dble), INTENT(IN)          :: lower_bound
    REAL(our_dble), INTENT(IN)          :: upper_bound
    REAL(our_dble), INTENT(IN)          :: X(:)
    
    INTEGER(our_int), INTENT(IN)        :: num_values

    !/* internal objects        */
    
    INTEGER(our_int)                    :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    DO i = 1, num_values

        IF (X(i) .LT. lower_bound) THEN

            Y(i) = lower_bound

        ELSE IF (X(i) .GT. upper_bound) THEN

            Y(i) = upper_bound

        ELSE 

            Y(i) = X(i)

        END IF

    END DO


END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE point_predictions(Y, X, coeffs, num_agents)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: Y(:)

    REAL(our_dble), INTENT(IN)      :: coeffs(:)
    REAL(our_dble), INTENT(IN)      :: X(:, :)
    
    INTEGER(our_int), INTENT(IN)    :: num_agents

    !/* internal objects        */

    INTEGER(our_int)                 :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    DO i = 1, num_agents

        Y(i) = DOT_PRODUCT(coeffs, X(i, :))

    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_coefficients(coeffs, Y, X, num_covars, num_agents)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: coeffs(:)

    INTEGER, INTENT(IN)             :: num_covars
    INTEGER, INTENT(IN)             :: num_agents

    REAL(our_dble), INTENT(IN)      :: X(:, :)
    REAL(our_dble), INTENT(IN)      :: Y(:)
    
    !/* internal objects        */

    REAL(our_dble)                  :: A(num_covars, num_covars)
    REAL(our_dble)                  :: C(num_covars, num_covars)
    REAL(our_dble)                  :: D(num_covars, num_agents)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
   A = MATMUL(TRANSPOSE(X), X)

   C =  pinv(A, num_covars)

   D = MATMUL(C, TRANSPOSE(X))

   coeffs = MATMUL(D, Y)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_r_squared(r_squared, observed, predicted, num_agents)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: r_squared

    REAL(our_dble), INTENT(IN)      :: predicted(:)
    REAL(our_dble), INTENT(IN)      :: observed(:)
    
    INTEGER(our_int), INTENT(IN)    :: num_agents

    !/* internal objects        */

    REAL(our_dble)                  :: mean_observed
    REAL(our_dble)                  :: ss_residuals
    REAL(our_dble)                  :: ss_total
 
    INTEGER(our_int)                :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Calculate mean of observed data
    mean_observed = SUM(observed) / DBLE(num_agents)
    
    ! Sum of squared residuals
    ss_residuals = zero_dble

    DO i = 1, num_agents

        ss_residuals = ss_residuals + (observed(i) - predicted(i))**2

    END DO

    ! Sum of squared residuals
    ss_total = zero_dble

    DO i = 1, num_agents

        ss_total = ss_total + (observed(i) - mean_observed)**2

    END DO

    ! Construct result
    r_squared = one_dble - ss_residuals / ss_total

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE