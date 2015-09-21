MODULE robufort_testing

	!/*	external modules	*/

    USE robufort_library

	!/*	setup	*/

    IMPLICIT NONE

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE slsqp_robufort(x_internal, x_start, maxiter, ftol, eps, num_draws, &
            eps_standard, period, k, payoffs_ex_ante, edu_max, edu_start, &
            mapping_state_idx, states_all, num_periods, periods_emax, &
            eps_cholesky, delta, debug, cov, level)

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: x_internal(2)
    REAL(our_dble), INTENT(IN)      :: x_start(2)
    REAL(our_dble), INTENT(IN)      :: cov(4,4)
    REAL(our_dble), INTENT(IN)      :: level
    REAL(our_dble), INTENT(IN)      :: ftol


    INTEGER(our_int), INTENT(IN)    :: maxiter

    REAL(our_dble), INTENT(IN)      :: eps_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: eps_standard(:, :)
    REAL(our_dble), INTENT(IN)      :: payoffs_ex_ante(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:,:)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: eps

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:,:,:,:,:)
    INTEGER(our_int), INTENT(IN)    :: states_all(:,:,:)
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: num_draws
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    LOGICAL, INTENT(IN)             :: debug

    !/* internal objects    */

    INTEGER(our_int)                :: m
    INTEGER(our_int)                :: meq
    INTEGER(our_int)                :: n
    INTEGER(our_int)                :: mode
    INTEGER(our_int)                :: iter
    INTEGER(our_int)                :: n1
    INTEGER(our_int)                :: mieq
    INTEGER(our_int)                :: mineq
    INTEGER(our_int)                :: l_w
    INTEGER(our_int)                :: l_jw
    INTEGER(our_int)                :: la

    INTEGER(our_int), ALLOCATABLE   :: jw(:)

    REAL(our_dble), ALLOCATABLE     :: a(:,:)
    REAL(our_dble), ALLOCATABLE     :: xl(:)
    REAL(our_dble), ALLOCATABLE     :: xu(:)
    REAL(our_dble), ALLOCATABLE     :: c(:)
    REAL(our_dble), ALLOCATABLE     :: g(:)
    REAL(our_dble), ALLOCATABLE     :: w(:)


    REAL(our_dble)                  :: payoffs_ex_post(4)
    REAL(our_dble)                  :: future_payoffs(4)
    REAL(our_dble)                  :: f

    LOGICAL                         :: is_finished

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! This is hard-coded for the ROBUPY package requirements. What follows below
    ! is based on this being 0, 1.
    !---------------------------------------------------------------------------
    meq = 0         ! Number of equality constraints
    mieq = 1        ! Number of inequality constraints
    !---------------------------------------------------------------------------
    !---------------------------------------------------------------------------

    ! Initialize starting values
    x_internal = x_start

    ! Derived attributes
    m = meq + mieq
    n = SIZE(x_internal)
    la = MAX(1, m)
    n1 = n + 1
    mineq = m - meq + n1 + n1

    l_w =  (3 * n1 + m) * (n1 + 1) + (n1 - meq + 1) * (mineq + 2) + &
           2 * mineq + (n1 + mineq) * (n1 - meq) + 2 * meq + n1 + &
           ((n + 1) * n) / two_dble + 2 * m + 3 * n + 3 * n1 + 1

    l_jw = mineq
    
    ! Allocate and initialize containers
    ALLOCATE(w(l_w)); ALLOCATE(jw(l_jw)); ALLOCATE(a(m, n + 1))
    ALLOCATE(g(n + 1)); ALLOCATE(c(m)); ALLOCATE(xl(n)); ALLOCATE(xu(n))

    ! Decompose upper and lower bounds
    xl = - huge_dble; xu = huge_dble

    ! Initialize the iteration counter and mode value
    iter = maxiter
    mode = zero_int

    ! Initialization of SLSQP
    is_finished = .False.

    ! Initialize criterion function at starting values
    CALL criterion(f, payoffs_ex_post, future_payoffs, x_internal, &
            num_draws, eps_standard, period, k, payoffs_ex_ante, edu_max, &
            edu_start, mapping_state_idx, states_all, num_periods, &
            periods_emax, eps_cholesky, delta, debug)

    CALL criterion_approx_gradient(g, x_internal, eps, num_draws, &
            eps_standard, period, k, payoffs_ex_ante, edu_max, edu_start, &
            mapping_state_idx, states_all, num_periods, periods_emax, &
            eps_cholesky, delta, debug)

    ! Initialize constraint at starting values
    CALL divergence(c, x_internal, cov, level)
    CALL divergence_approx_gradient(a, x_internal, cov, level, eps)

    ! Iterate until completion
    DO WHILE (.NOT. is_finished)

        ! Evaluate criterion function and constraints
        IF (mode == one_int) THEN

            CALL criterion(f, payoffs_ex_post, future_payoffs, &
                    x_internal, num_draws, eps_standard, period, k, &
                    payoffs_ex_ante, edu_max, edu_start, mapping_state_idx, &
                    states_all, num_periods, periods_emax, eps_cholesky, &
                    delta, debug)

            CALL divergence(c, x_internal, cov, level)

        ! Evaluate gradient of criterion function and constraints. Note that the
        ! a is of dimension (1, n + 1) and the last element needs to always
        ! be zero.
        ELSEIF (mode == - one_int) THEN

            CALL criterion_approx_gradient(g, x_internal, eps, num_draws, &
                    eps_standard, period, k, payoffs_ex_ante, edu_max, &
                    edu_start, mapping_state_idx, states_all, num_periods, &
                    periods_emax, eps_cholesky, delta, debug)

            CALL divergence_approx_gradient(a, x_internal, cov, level, eps)

        END IF

        !Call to SLSQP code
        CALL slsqp(m, meq, la, n, x_internal, xl, xu, f, c, g, a, ftol, &
                iter, mode, w, l_w, jw, l_jw)

        ! Check if SLSQP has completed
        IF (.NOT. ABS(mode) == one_int) THEN
            is_finished = .True.
        END IF

    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE slsqp_debug(x_internal, x_start, maxiter, ftol, &
                num_dim)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: x_internal(num_dim)
    REAL(our_dble), INTENT(IN)      :: x_start(num_dim)
    REAL(our_dble), INTENT(IN)      :: ftol

    INTEGER(our_int), INTENT(IN)    :: num_dim
    INTEGER(our_int), INTENT(IN)    :: maxiter

    !/* internal objects    */

    INTEGER(our_int)                :: m
    INTEGER(our_int)                :: meq
    INTEGER(our_int)                :: la
    INTEGER(our_int)                :: n
    INTEGER(our_int)                :: len_w
    INTEGER(our_int)                :: len_jw
    INTEGER(our_int)                :: mode
    INTEGER(our_int)                :: iter
    INTEGER(our_int)                :: n1
    INTEGER(our_int)                :: mieq
    INTEGER(our_int)                :: mineq
    INTEGER(our_int)                :: l_jw
    INTEGER(our_int)                :: l_w

    INTEGER(our_int), ALLOCATABLE   :: jw(:)

    REAL(our_dble), ALLOCATABLE     :: a(:,:)
    REAL(our_dble), ALLOCATABLE     :: xl(:)
    REAL(our_dble), ALLOCATABLE     :: xu(:)
    REAL(our_dble), ALLOCATABLE     :: c(:)
    REAL(our_dble), ALLOCATABLE     :: g(:)
    REAL(our_dble), ALLOCATABLE     :: w(:)

    REAL(our_dble)                  :: acc
    REAL(our_dble)                  :: f

    LOGICAL                         :: is_finished

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! This is hard-coded for the ROBUPY package requirements
    !---------------------------------------------------------------------------
    meq = 0         ! Number of equality constraints
    mieq = 1        ! Number of inequality constraints
    !---------------------------------------------------------------------------
    !---------------------------------------------------------------------------

    ! Initialize starting values
    x_internal = x_start

    ! Derived attributes
    m = meq + mieq
    la = MAX(1, m)
    n = SIZE(x_internal)
    n1 = n + 1
    mineq = m - meq + n1 + n1

    len_w =  (3 * n1 + m) * (n1 + 1) + (n1 - meq + 1) * (mineq + 2) + &
                2 * mineq + (n1 + mineq) * (n1 - meq) + 2 * meq + n1 + &
                ((n + 1) * n) / two_dble + 2 * m + 3 * n + 3 * n1 + 1

    len_jw = mineq

    ! Allocate and initialize containers
    ALLOCATE(w(len_w)); w = zero_dble
    ALLOCATE(jw(len_jw)); jw = zero_int
    ALLOCATE(a(la, n + 1)); a = zero_dble

    ALLOCATE(g(n + 1)); g = zero_dble
    ALLOCATE(c(la)); c = zero_dble

    ! Decompose upper and lower bounds
    ALLOCATE(xl(n)); ALLOCATE(xu(n))
    xl = - huge_dble; xu = huge_dble

    ! Initialize the iteration counter and mode value
    acc = ftol
    iter = maxiter

    ! Transformations to match interface, deleted later
    l_jw = len_jw
    l_w = len_w

    ! Initialization of SLSQP
    mode = zero_int

    is_finished = .False.

    CALL debug_criterion_function(f, x_internal, n)
    CALL debug_criterion_derivative(g, x_internal, n)

    CALL debug_constraint_function(c, x_internal, n, la)
    CALL debug_constraint_derivative(a, x_internal, n, la)

    ! Iterate until completion
    DO WHILE (.NOT. is_finished)

        ! Evaluate criterion function and constraints
        IF (mode == one_int) THEN
            CALL debug_criterion_function(f, x_internal, n)
            CALL debug_constraint_function(c, x_internal, n, la)
        ! Evaluate gradient of criterion function and constraints
        ELSEIF (mode == - one_int) THEN
            CALL debug_criterion_derivative(g, x_internal, n)
            CALL debug_constraint_derivative(a, x_internal, n, la)
        END IF

        !SLSQP Interface
        CALL slsqp(m, meq, la, n, x_internal, xl, xu, f, c, g, &
                    a, acc, iter, mode, w, l_w, jw, l_jw)
        
        ! Check if SLSQP has completed
        IF (.NOT. ABS(mode) == one_int) THEN
            is_finished = .True.
        END IF

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE debug_criterion_function(rslt, x, n)

    !/* external objects    */

    INTEGER(our_int), INTENT(IN)    :: n

    REAL(our_dble), INTENT(OUT)     :: rslt
    REAL(our_dble), INTENT(IN)      :: x(n)

    !/* internal objects    */

    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    rslt = zero_dble

    DO i = 2, n
        rslt = rslt + one_hundred_dble * (x(i) - x(i - 1)**2)**2
        rslt = rslt + (one_dble - x(i - 1))**2
    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE debug_criterion_derivative(rslt, x, n)

    !/* external objects    */

    INTEGER(our_int), INTENT(IN)    :: n

    REAL(our_dble), INTENT(OUT)     :: rslt(n + 1)
    REAL(our_dble), INTENT(IN)      :: x(n)

    !/* internals objects    */

    REAL(our_dble)                  :: xm(n - 2)
    REAL(our_dble)                  :: xm_m1(n - 2)
    REAL(our_dble)                  :: xm_p1(n - 2)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Extract sets of evaluation points
    xm = x(2:(n - 1))

    xm_m1 = x(:(n - 2))

    xm_p1 = x(3:)

    ! Construct derivative information
    rslt(1) = -four_hundred_dble * x(1) * (x(2) - x(1) ** 2) - 2 * (1 - x(1))

    rslt(2:(n - 1)) =  (two_hundred_dble * (xm - xm_m1 ** 2) - &
            four_hundred_dble * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm))

    rslt(n) = two_hundred_dble * (x(n) - x(n - 1) ** 2)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE debug_constraint_function(rslt, x, n, la)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: rslt(la)

    REAL(our_dble), INTENT(IN)      :: x(n)

    INTEGER(our_int), INTENT(IN)    :: n
    INTEGER(our_int), INTENT(IN)    :: la

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt(:) = SUM(x) - 10.0

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE debug_constraint_derivative(rslt, x, n, la)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: rslt(n + 1)

    REAL(our_dble), INTENT(IN)      :: x(n)

    INTEGER(our_int), INTENT(IN)    :: n
    INTEGER(our_int), INTENT(IN)    :: la

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt = 1

    rslt(n + 1) = 0.0

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE