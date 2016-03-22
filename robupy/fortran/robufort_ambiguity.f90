!*******************************************************************************
!*******************************************************************************
!
!   This delivers all functions and subroutines to the ROBUFORT library that 
!	are associated with the model under ambiguity. It uses functionality from 
!	the risk module.
!
!*******************************************************************************
!*******************************************************************************
MODULE robufort_ambiguity

	!/*	external modules	  */

    USE robufort_constants

    USE robufort_auxiliary

    USE robufort_emax

    USE robufort_risk

	!/*	setup	             */

	IMPLICIT NONE

    PUBLIC 
    
CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_payoffs_ambiguity(emax_simulated, payoffs_ex_post, &
                payoffs_future, num_draws_emax, draws_emax, period, & 
                k, payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
                states_all, num_periods, periods_emax, delta, is_debug, &
                shocks, level, measure, is_deterministic, shocks_cholesky)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: payoffs_ex_post(:)
    REAL(our_dble), INTENT(OUT)     :: payoffs_future(:)
    REAL(our_dble), INTENT(OUT)     :: emax_simulated

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:,:,:,:,:)
    INTEGER(our_int), INTENT(IN)    :: states_all(:,:,:)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:,:)
    REAL(our_dble), INTENT(IN)      :: shocks(:, :)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: level

    LOGICAL, INTENT(IN)             :: is_deterministic
    LOGICAL, INTENT(IN)             :: is_debug

    CHARACTER(10), INTENT(IN)       :: measure

    !/* internal objects        */

    INTEGER(our_int)                :: maxiter

    REAL(our_dble)                  :: x_internal(2)
    REAL(our_dble)                  :: x_start(2)
    REAL(our_dble)                  :: ftol
    REAL(our_dble)                  :: tiny

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Parameterizations for optimizations
    x_start = zero_dble
    maxiter = 100000000_our_int
    ftol = 1e-06_our_dble
    tiny = 1.4901161193847656e-08

    ! Determine the worst case, special attention to zero variability. The
    ! latter is included as a special case for debugging purposes. The worst
    ! case corresponds to zero.

    IF (is_deterministic) THEN

        CALL handle_shocks_zero(x_internal, is_debug, period, k)
    
    ELSE

        CALL slsqp_robufort(x_internal, x_start, maxiter, ftol, tiny, &
                num_draws_emax, draws_emax, period, k, & 
                payoffs_systematic, edu_max, edu_start, mapping_state_idx, & 
                states_all, num_periods, periods_emax, delta, is_debug, & 
                shocks, level, shocks_cholesky)

    END IF

    ! Evaluate expected future value for perturbed values
    CALL simulate_emax(emax_simulated, payoffs_ex_post, payoffs_future, &
            num_periods, num_draws_emax, period, k, draws_emax, & 
            payoffs_systematic, edu_max, edu_start, periods_emax, states_all, & 
            mapping_state_idx, delta, shocks_cholesky, x_internal)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE handle_shocks_zero(x_internal, is_debug, period, k)

    !/* external objects        */

    REAL(our_dble), INTENT(INOUT)       :: x_internal(2)

    INTEGER(our_int), INTENT(IN)        :: period
    INTEGER(our_int), INTENT(IN)        :: k

    LOGICAL, INTENT(IN)                 :: is_debug

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL logging_ambiguity(x_internal, zero_dble, 10, period, k, .FALSE.)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE slsqp_robufort(x_internal, x_start, maxiter, ftol, tiny, &
            num_draws_emax, draws_emax, period, k, & 
            payoffs_systematic, edu_max, edu_start, mapping_state_idx, & 
            states_all, num_periods, periods_emax, delta, is_debug, shocks, & 
            level, shocks_cholesky)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: x_internal(:)

    REAL(our_dble), INTENT(IN)      :: shocks(:,:)
    REAL(our_dble), INTENT(IN)      :: x_start(:)
    REAL(our_dble), INTENT(IN)      :: level
    REAL(our_dble), INTENT(IN)      :: ftol

    INTEGER(our_int), INTENT(IN)    :: maxiter

    REAL(our_dble), INTENT(IN)      :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:,:)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: tiny

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:,:,:,:,:)
    INTEGER(our_int), INTENT(IN)    :: states_all(:,:,:)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    LOGICAL, INTENT(IN)             :: is_debug

    !/* internal objects        */

    INTEGER(our_int)                :: mineq
    INTEGER(our_int)                :: mode
    INTEGER(our_int)                :: iter
    INTEGER(our_int)                :: mieq
    INTEGER(our_int)                :: l_jw
    INTEGER(our_int)                :: meq
    INTEGER(our_int)                :: l_w
    INTEGER(our_int)                :: la
    INTEGER(our_int)                :: m
    INTEGER(our_int)                :: n

    INTEGER(our_int)                :: jw(7)

    REAL(our_dble)                  :: a(1, 3)
    REAL(our_dble)                  :: w(144)
    REAL(our_dble)                  :: xl(2)
    REAL(our_dble)                  :: xu(2)
    REAL(our_dble)                  :: c(1)
    REAL(our_dble)                  :: g(3)
    REAL(our_dble)                  :: div
    REAL(our_dble)                  :: f

    LOGICAL                         :: is_finished
    LOGICAL                         :: is_success

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! This is hard-coded for the ROBUPY package requirements. What follows 
    ! below is based on this being 0, 1.
    !---------------------------------------------------------------------------
    meq = 1         ! Number of equality constraints
    mieq = 0        ! Number of inequality constraints
    !---------------------------------------------------------------------------
    !---------------------------------------------------------------------------

    ! Initialize starting values
    x_internal = x_start

    ! Derived attributes
    m = meq + mieq
    n = SIZE(x_internal)
    la = MAX(1, m)
    mineq = m - meq + (n + 1) + (n + 1)

    l_w =  (3 * (n + 1) + m) * ((n + 1) + 1) + ((n + 1) - meq + 1) * (mineq + 2) + &
           2 * mineq + ((n + 1) + mineq) * ((n + 1) - meq) + 2 * meq + (n + 1) + &
           ((n + 1) * n) / two_dble + 2 * m + 3 * n + 3 * (n + 1) + 1

    l_jw = mineq

    ! Decompose upper and lower bounds
    xl = - HUGE_FLOAT; xu = HUGE_FLOAT

    ! Initialize the iteration counter and mode value
    iter = maxiter
    mode = zero_int

    ! Initialization of SLSQP
    is_finished = .False.

    ! Initialize criterion function at starting values
    f = criterion(x_internal, num_draws_emax, draws_emax, period, & 
            k, payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
            states_all, num_periods, periods_emax, delta, shocks_cholesky)


    g(:2) = criterion_approx_gradient(x_internal, tiny, num_draws_emax, &
                draws_emax, period, k, payoffs_systematic, edu_max, & 
                edu_start, mapping_state_idx, states_all, num_periods, & 
                periods_emax, delta, shocks_cholesky)

    ! Initialize constraint at starting values
    c = divergence(x_internal, shocks, level)
    a(1,:2) = divergence_approx_gradient(x_internal, shocks, level, tiny)

    ! Iterate until completion
    DO WHILE (.NOT. is_finished)

        ! Evaluate criterion function and constraints
        IF (mode == one_int) THEN

            f = criterion(x_internal, num_draws_emax, draws_emax, & 
                    period, k, payoffs_systematic, edu_max, edu_start, &
                    mapping_state_idx, states_all, num_periods, periods_emax, & 
                    delta, shocks_cholesky)

            c = divergence(x_internal, shocks, level)

        ! Evaluate gradient of criterion function and constraints. Note that the
        ! a is of dimension (1, n + 1) and the last element needs to always
        ! be zero.
        ELSEIF (mode == - one_int) THEN

            g(:2) = criterion_approx_gradient(x_internal, tiny, & 
                    num_draws_emax, draws_emax, period, k, & 
                    payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
                    states_all, num_periods, periods_emax, delta, shocks_cholesky)

            a(1,:2) = divergence_approx_gradient(x_internal, shocks, level, tiny)

        END IF

        !Call to SLSQP code
        CALL slsqp(m, meq, la, n, x_internal, xl, xu, f, c, g, a, ftol, &
                iter, mode, w, l_w, jw, l_jw)

        ! Check if SLSQP has completed
        IF (.NOT. ABS(mode) == one_int) THEN
            is_finished = .True.
        END IF

    END DO
    
    ! Stabilization. If the optimization fails the starting values are
    ! used otherwise it happens that the constraint is not satisfied by far.
    is_success = (mode == zero_int)

    IF(.NOT. is_success) THEN
        x_internal = x_start
    END IF

    ! Logging.
    IF (is_debug) THEN

        ! Evaluate divergence at final value
        div = divergence(x_internal, shocks, level) - level
        ! Write to logging file
        CALL logging_ambiguity(x_internal, div, mode, period, k, is_success)

    END IF

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
FUNCTION criterion(x_internal, num_draws_emax, draws_emax, period, & 
             k, payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
            states_all, num_periods, periods_emax, delta, shocks_cholesky)

    !/* external objects        */

    REAL(our_dble)                  :: criterion

    REAL(our_dble), INTENT(IN)      :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:,:)
    REAL(our_dble), INTENT(IN)      :: x_internal(:)
    REAL(our_dble), INTENT(IN)      :: delta

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:,:,:,:,:)
    INTEGER(our_int), INTENT(IN)    :: states_all(:,:,:)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    !/* internal objects        */

    REAL(our_dble)                  :: payoffs_ex_post(4)
    REAL(our_dble)                  :: payoffs_future(4)
    REAL(our_dble)                  :: emax_simulated

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Evaluate expected future value
    CALL simulate_emax(emax_simulated, payoffs_ex_post, payoffs_future, &
            num_periods, num_draws_emax, period, k, draws_emax, & 
            payoffs_systematic, edu_max, edu_start, periods_emax, states_all, & 
            mapping_state_idx, delta, shocks_cholesky, x_internal)

    ! Finishing
    criterion = emax_simulated

END FUNCTION
!*******************************************************************************
!*******************************************************************************
FUNCTION criterion_approx_gradient(x_internal, tiny, num_draws_emax, &
            draws_emax, period, k, payoffs_systematic, edu_max, & 
            edu_start, mapping_state_idx, states_all, num_periods, & 
            periods_emax, delta, shocks_cholesky)

    !/* external objects        */

    REAL(our_dble)                  :: criterion_approx_gradient(2)

    REAL(our_dble), INTENT(IN)      :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:,:)
    REAL(our_dble), INTENT(IN)      :: x_internal(:)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: tiny

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:,:,:,:,:)
    INTEGER(our_int), INTENT(IN)    :: states_all(:,:,:)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    !/* internals objects       */

    REAL(our_dble)                  :: ei(2)
    REAL(our_dble)                  :: d(2)
    REAL(our_dble)                  :: f0
    REAL(our_dble)                  :: f1

    INTEGER(our_int)                :: j

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Initialize containers
    ei = zero_dble

    ! Evaluate baseline
    f0 = criterion(x_internal, num_draws_emax, draws_emax, period, & 
            k, payoffs_systematic, edu_max, edu_start, mapping_state_idx, & 
            states_all, num_periods, periods_emax, delta, shocks_cholesky)

    ! Iterate over increments
    DO j = 1, 2

        ei(j) = one_dble

        d = tiny * ei

        f1 = criterion(x_internal + d, num_draws_emax, draws_emax, & 
                period, k, payoffs_systematic, edu_max, edu_start, & 
                mapping_state_idx, states_all, num_periods, periods_emax, delta, & 
                shocks_cholesky)

        criterion_approx_gradient(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!*******************************************************************************
!*******************************************************************************
SUBROUTINE logging_ambiguity(x_internal, div, mode, period, k, is_success)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: x_internal(:)
    REAL(our_dble), INTENT(IN)      :: div

    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: mode
    INTEGER(our_int), INTENT(IN)    :: k

    LOGICAL, INTENT(IN)             :: is_success

    !/* internal objects        */

    CHARACTER(55)                   :: message_optimizer
    CHARACTER(5)                    :: message_success

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Success message
    IF(is_success) THEN
        message_success = 'True'
    ELSE
        message_success = 'False'
    END IF

    ! Optimizer message
    IF (mode == -1) THEN
        message_optimizer = 'Gradient evaluation required (g & a)'
    ELSEIF (mode == 0) THEN
        message_optimizer = 'Optimization terminated successfully.'
    ELSEIF (mode == 1) THEN
        message_optimizer = 'Function evaluation required (f & c)'
    ELSEIF (mode == 2) THEN
        message_optimizer = 'More equality constraints than independent variables'
    ELSEIF (mode == 3) THEN
        message_optimizer = 'More than 3*n iterations in LSQ subproblem'
    ELSEIF (mode == 4) THEN
        message_optimizer = 'Inequality constraints incompatible'
    ELSEIF (mode == 5) THEN
        message_optimizer = 'Singular matrix E in LSQ subproblem'
    ELSEIF (mode == 6) THEN
        message_optimizer = 'Singular matrix C in LSQ subproblem'
    ELSEIF (mode == 7) THEN
        message_optimizer = 'Rank-deficient equality constraint subproblem HFTI'
    ELSEIF (mode == 8) THEN
        message_optimizer = 'Positive directional derivative for linesearch'
    ELSEIF (mode == 9) THEN
        message_optimizer = 'Iteration limit exceeded'
    ELSEIF (mode == 10) THEN
        message_optimizer = 'No random variation in shocks.'
    END IF

    ! Write to file
    OPEN(UNIT=1, FILE='ambiguity.robupy.log', ACCESS='APPEND')

        1000 FORMAT(A,i7,A,i7)
        1010 FORMAT(A17,(2(1x,f10.4)))
        WRITE(1, 1000) " PERIOD", period, "  STATE", k
        WRITE(1, *)  
        WRITE(1, 1010) "    Result       ", x_internal
        WRITE(1, 1010) "    Divergence   ", div
        WRITE(1, *)  
        WRITE(1, *) "   Success ", message_success
        WRITE(1, *) "   Message ", message_optimizer
        WRITE(1, *) 

    CLOSE(1)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
FUNCTION divergence(x_internal, shocks, level)

    !/* external objects        */

    REAL(our_dble)                  :: divergence 

    REAL(our_dble), INTENT(IN)      :: x_internal(:)
    REAL(our_dble), INTENT(IN)      :: shocks(:,:)
    REAL(our_dble), INTENT(IN)      :: level

    !/* internals objects       */

    REAL(our_dble)                  :: alt_mean(4, 1) = zero_dble
    REAL(our_dble)                  :: old_mean(4, 1) = zero_dble
    REAL(our_dble)                  :: inv_old_cov(4,4)
    REAL(our_dble)                  :: alt_cov(4,4)
    REAL(our_dble)                  :: old_cov(4,4)
    REAL(our_dble)                  :: comp_b(1, 1)
    REAL(our_dble)                  :: comp_a
    REAL(our_dble)                  :: comp_c
    REAL(our_dble)                  :: rslt

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Construct alternative distribution
    alt_mean(1,1) = x_internal(1)
    alt_mean(2,1) = x_internal(2)
    alt_cov = shocks

    ! Construct baseline distribution
    old_cov = shocks

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
    divergence = level - rslt

END FUNCTION
!*******************************************************************************
!*******************************************************************************
FUNCTION divergence_approx_gradient(x, cov, level, tiny)

    !/* external objects        */

    REAL(our_dble)                  :: divergence_approx_gradient(2)

    REAL(our_dble), INTENT(IN)      :: cov(:,:)
    REAL(our_dble), INTENT(IN)      :: level
    REAL(our_dble), INTENT(IN)      :: x(:)
    REAL(our_dble), INTENT(IN)      :: tiny

    !/* internals objects       */

    INTEGER(our_int)                :: k

    REAL(our_dble)                  :: ei(2)
    REAL(our_dble)                  :: d(2)
    REAL(our_dble)                  :: f0 
    REAL(our_dble)                  :: f1 

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Initialize containers
    ei = zero_dble

    ! Evaluate baseline
    f0 = divergence(x, cov, level)

    ! Iterate over increments
    DO k = 1, 2

        ei(k) = one_dble

        d = tiny * ei

        f1 = divergence(x + d, cov, level)

        divergence_approx_gradient(k) = (f1 - f0) / d(k)

        ei(k) = zero_dble

    END DO

END FUNCTION
!*******************************************************************************
!*******************************************************************************
END MODULE