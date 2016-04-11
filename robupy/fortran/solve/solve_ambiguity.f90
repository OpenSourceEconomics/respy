!*******************************************************************************
!*******************************************************************************
!
!   This delivers all functions and subroutines to the ROBUFORT library that 
!	are associated with the model under ambiguity. It uses functionality from 
!	the risk module.
!
!*******************************************************************************
!*******************************************************************************
MODULE solve_ambiguity

	!/*	external modules	  */

    USE shared_constants

    USE evaluate_emax

    USE evaluate_risk

	!/*	setup	             */

    IMPLICIT NONE

    EXTERNAL SLSQP

    PUBLIC 
    
CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_payoffs_ambiguity(emax_simulated, num_draws_emax, draws_emax, & 
                period, k, payoffs_systematic, edu_max, edu_start, & 
                mapping_state_idx, states_all, num_periods, periods_emax, & 
                delta, is_debug, shocks_cov, level, measure, & 
                is_deterministic, shocks_cholesky)

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

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: shocks_cov(:, :)
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

        CALL get_worst_case(x_internal, x_start, maxiter, ftol, tiny, &
                num_draws_emax, draws_emax, period, k, payoffs_systematic, &
                edu_max, edu_start, mapping_state_idx, states_all, &
                num_periods, periods_emax, delta, is_debug, shocks_cov, &
                level, shocks_cholesky)

    END IF

    ! Evaluate expected future value for perturbed values
    CALL simulate_emax(emax_simulated, num_periods, num_draws_emax, period, & 
            k, draws_emax, payoffs_systematic, edu_max, edu_start, & 
            periods_emax, states_all, mapping_state_idx, delta, & 
            shocks_cholesky, x_internal)

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

    IF (is_debug) THEN
        CALL logging_ambiguity(x_internal, zero_dble, 10, period, k, .FALSE.)
    END IF

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_worst_case(x_internal, x_start, maxiter, ftol, tiny, &
                num_draws_emax, draws_emax, period, k, payoffs_systematic, &
                edu_max, edu_start, mapping_state_idx, states_all, & 
                num_periods, periods_emax, delta, is_debug, shocks_cov, & 
                level, shocks_cholesky)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: x_internal(:)

    REAL(our_dble), INTENT(IN)      :: shocks_cov(:, :)
    REAL(our_dble), INTENT(IN)      :: x_start(:)
    REAL(our_dble), INTENT(IN)      :: level
    REAL(our_dble), INTENT(IN)      :: ftol

    INTEGER(our_int), INTENT(IN)    :: maxiter

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:,:)
    REAL(our_dble), INTENT(IN)      :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: tiny

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    LOGICAL, INTENT(IN)             :: is_debug

    !/* internal objects        */



    INTEGER(our_int)                :: MINEQ
    INTEGER(our_int)                :: MODE
    INTEGER(our_int)                :: ITER
    INTEGER(our_int)                :: L_JW
    INTEGER(our_int)                :: MEQ
    INTEGER(our_int)                :: L_W
    INTEGER(our_int)                :: LA
    INTEGER(our_int)                :: M
    INTEGER(our_int)                :: N, N1

    INTEGER(our_int)                :: JW(7)

    REAL(our_dble)                  :: A(1, 3)
    REAL(our_dble)                  :: ACC
    REAL(our_dble)                  :: W(144)
    REAL(our_dble)                  :: XL(2)
    REAL(our_dble)                  :: XU(2)
    REAL(our_dble)                  :: C(1)
    REAL(our_dble)                  :: G(3)
    REAL(our_dble)                  :: div
    REAL(our_dble)                  :: F

    REAL(our_dble)                  :: X(2)

    LOGICAL                         :: is_finished
    LOGICAL                         :: is_success

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! This is hard-coded for the ROBUPY package requirements. What follows 
    ! below is based on this being 0, 1.
    !---------------------------------------------------------------------------
    M = 1
    MEQ = 1
    !---------------------------------------------------------------------------
    !---------------------------------------------------------------------------

    ! Initialize starting values
    ACC = ftol
    X = x_start

    ! Derived attributes
    N = SIZE(x_internal)
    LA = MAX(1, M)


N1 = N + 1
MINEQ = M - MEQ + N1 + N1
L_W = (3 * N1 + M) *( N1 + 1) + (N1 - MEQ + 1) * (MINEQ + 2) + 2 * MINEQ + & 
        (N1 + MINEQ) * (N1 - MEQ) + 2 * MEQ + N1 + (N + 1) * N / 2 + & 
         2 * M + 3 * N + 3 * N1 + 1
L_JW = MINEQ




    MINEQ = M - MEQ + (N + 1) + (N + 1)

    L_W =  (3 * (N + 1) + M) * ((N + 1) + 1) + ((N + 1) - MEQ + 1) * (MINEQ + 2) + &
           2 * MINEQ + ((N + 1) + MINEQ) * ((N + 1) - MEQ) + 2 * MEQ + (N + 1) + &
           ((N + 1) * N) / two_dble + 2 * M + 3 * N + 3 * (N + 1) + 1

    L_JW = MINEQ

    ! Decompose upper and lower bounds
    XL = - HUGE_FLOAT; XU = HUGE_FLOAT

    ! Initialize the iteration counter and MODE value
    ITER = maxiter
    MODE = zero_int

    ! Initialization of SLSQP
    is_finished = .False.

    ! Initialize criterion function at starting values
    F = criterion_ambiguity(X, num_draws_emax, draws_emax, period, &
            k, payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
            states_all, num_periods, periods_emax, delta, shocks_cholesky)


    G(:2) = criterion_ambiguity_derivative(X, tiny, num_draws_emax, &
                draws_emax, period, k, payoffs_systematic, edu_max, & 
                edu_start, mapping_state_idx, states_all, num_periods, & 
                periods_emax, delta, shocks_cholesky)

    ! Initialize constraint at starting values
    C = divergence(X, shocks_cov, level)
    A(1,:2) = divergence_derivative(X, shocks_cov, level, tiny)

    ! Iterate until completion
    DO WHILE (.NOT. is_finished)

        ! Evaluate criterion function and constraints
        IF (MODE == one_int) THEN

            F = criterion_ambiguity(X, num_draws_emax, draws_emax, &
                    period, k, payoffs_systematic, edu_max, edu_start, &
                    mapping_state_idx, states_all, num_periods, periods_emax, & 
                    delta, shocks_cholesky)

            C = divergence(X, shocks_cov, level)

        ! Evaluate gradient of criterion function and constraints. Note that the
        ! A is of dimension (1, N + 1) and the last element needs to always
        ! be zero.
        ELSEIF (MODE == - one_int) THEN

            G(:2) = criterion_ambiguity_derivative(X, tiny, &
                        num_draws_emax, draws_emax, period, k, &
                        payoffs_systematic, edu_max, edu_start, & 
                        mapping_state_idx, states_all, num_periods, & 
                        periods_emax, delta, shocks_cholesky)

            A(1,:2) = divergence_derivative(X, shocks_cov, &
                            level, tiny)

        END IF

        !Call to SLSQP code
        CALL slsqp(M, MEQ, LA, N, X, XL, XU, F, C, G, A, ACC, &
                ITER, MODE, W, L_W, JW, L_JW)

        ! Check if SLSQP has completed
        IF (.NOT. ABS(MODE) == one_int) THEN
            is_finished = .True.
        END IF

    END DO
    
    x_internal = X

    ! Stabilization. If the optimization fails the starting values are
    ! used otherwise it happens that the constraint is not satisfied by far.
    is_success = (MODE == zero_int)

    IF(.NOT. is_success) THEN
        x_internal = x_start
    END IF

    
    ! Logging.
    IF (is_debug) THEN
        ! Evaluate divergence at final value
        div = divergence(x_internal, shocks_cov, level) - level
        ! Write to logging file
        CALL logging_ambiguity(x_internal, div, MODE, period, k, is_success)
    END IF

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
FUNCTION criterion_ambiguity(x_internal, num_draws_emax, draws_emax, period, & 
            k, payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
            states_all, num_periods, periods_emax, delta, shocks_cholesky)

    !/* external objects        */

    REAL(our_dble)                  :: criterion_ambiguity

    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: x_internal(:)
    REAL(our_dble), INTENT(IN)      :: delta

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    !/* internal objects        */

    REAL(our_dble)                  :: emax_simulated

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Evaluate expected future value
    CALL simulate_emax(emax_simulated, num_periods, num_draws_emax, period, & 
            k, draws_emax, payoffs_systematic, edu_max, edu_start, & 
            periods_emax, states_all, mapping_state_idx, delta, & 
            shocks_cholesky, x_internal)

    ! Finishing
    criterion_ambiguity = emax_simulated

END FUNCTION
!*******************************************************************************
!*******************************************************************************
FUNCTION divergence(x_internal, shocks_cov, level)

    !/* external objects        */

    REAL(our_dble)                  :: divergence

    REAL(our_dble), INTENT(IN)      :: shocks_cov(:,:)
    REAL(our_dble), INTENT(IN)      :: x_internal(:)
    REAL(our_dble), INTENT(IN)      :: level

    !/* internals objects       */

    REAL(our_dble)                  :: alt_mean(4, 1) = zero_dble
    REAL(our_dble)                  :: old_mean(4, 1) = zero_dble
    REAL(our_dble)                  :: inv_old_cov(4, 4)
    REAL(our_dble)                  :: alt_cov(4, 4)
    REAL(our_dble)                  :: old_cov(4, 4)
    REAL(our_dble)                  :: comp_b(1, 1)
    REAL(our_dble)                  :: comp_a
    REAL(our_dble)                  :: comp_c
    REAL(our_dble)                  :: rslt

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Construct alternative distribution
    alt_mean(1, 1) = x_internal(1)
    alt_mean(2, 1) = x_internal(2)
    alt_cov = shocks_cov

    ! Construct baseline distribution
    old_cov = shocks_cov

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
FUNCTION criterion_ambiguity_derivative(x_internal, tiny, num_draws_emax, &
            draws_emax, period, k, payoffs_systematic, edu_max, edu_start, &
            mapping_state_idx, states_all, num_periods, periods_emax, delta, &
            shocks_cholesky)

    !/* external objects        */

    REAL(our_dble)                  :: criterion_ambiguity_derivative(2)

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:,:)
    REAL(our_dble), INTENT(IN)      :: draws_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: x_internal(:)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: tiny

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
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
    f0 = criterion_ambiguity(x_internal, num_draws_emax, draws_emax, period, &
            k, payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
            states_all, num_periods, periods_emax, delta, shocks_cholesky)

    ! Iterate over increments
    DO j = 1, 2

        ei(j) = one_dble

        d = tiny * ei

        f1 = criterion_ambiguity(x_internal + d, num_draws_emax, draws_emax, &
                period, k, payoffs_systematic, edu_max, edu_start, &
                mapping_state_idx, states_all, num_periods, periods_emax, &
                delta, shocks_cholesky)

        criterion_ambiguity_derivative(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!*******************************************************************************
!*******************************************************************************
FUNCTION divergence_derivative(x, cov, level, tiny)

    !/* external objects        */

    REAL(our_dble)                  :: divergence_derivative(2)

    REAL(our_dble), INTENT(IN)      :: cov(:, :)
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

        divergence_derivative(k) = (f1 - f0) / d(k)

        ei(k) = zero_dble

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
END MODULE