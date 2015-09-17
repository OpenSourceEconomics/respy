!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_slsqp_robufort(x_internal, x_start, maxiter, ftol, eps, &
            num_draws, eps_standard, period, k, payoffs_ex_ante, edu_max, &
            edu_start, mapping_state_idx, states_all, num_periods, &
            periods_emax, eps_cholesky, delta, debug, cov, level, num_dim)

    !/* external libraries    */

    USE robufort_slsqp
    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: x_internal(num_dim)
    DOUBLE PRECISION, INTENT(IN)    :: x_start(num_dim)
    DOUBLE PRECISION, INTENT(IN)    :: cov(4,4)
    DOUBLE PRECISION, INTENT(IN)    :: level
    DOUBLE PRECISION, INTENT(IN)    :: ftol

    INTEGER, INTENT(IN)             :: num_dim
    INTEGER, INTENT(IN)             :: maxiter

    DOUBLE PRECISION, INTENT(IN)    :: eps_cholesky(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: eps_standard(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: payoffs_ex_ante(:)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: delta
    DOUBLE PRECISION, INTENT(IN)    :: eps

    INTEGER, INTENT(IN)             :: mapping_state_idx(:,:,:,:,:)
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

    LOGICAL, INTENT(IN)             :: debug

    !/* internal objects    */

    INTEGER                         :: m
    INTEGER                         :: meq
    INTEGER                         :: n
    INTEGER                         :: mode
    INTEGER                         :: iter
    INTEGER                         :: n1
    INTEGER                         :: mieq
    INTEGER                         :: mineq
    INTEGER                         :: l_w

    INTEGER, ALLOCATABLE            :: jw(:)

    DOUBLE PRECISION, ALLOCATABLE   :: xl(:)
    DOUBLE PRECISION, ALLOCATABLE   :: xu(:)
    DOUBLE PRECISION, ALLOCATABLE   :: c(:)
    DOUBLE PRECISION, ALLOCATABLE   :: g(:)
    DOUBLE PRECISION, ALLOCATABLE   :: a(:,:)
    DOUBLE PRECISION, ALLOCATABLE   :: w(:)

    DOUBLE PRECISION                :: payoffs_ex_post(4)
    DOUBLE PRECISION                :: future_payoffs(4)
    DOUBLE PRECISION                :: f

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
    n1 = n + 1
    mineq = m - meq + n1 + n1

    l_w =  (3 * n1 + m) * (n1 + 1) + (n1 - meq + 1) * (mineq + 2) + &
           2 * mineq + (n1 + mineq) * (n1 - meq) + 2 * meq + n1 + &
           ((n + 1) * n) / two_dble + 2 * m + 3 * n + 3 * n1 + 1

    ! Allocate and initialize containers
    ALLOCATE(w(l_w)); ALLOCATE(jw(mineq)); ALLOCATE(a(m, n + 1))
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
        CALL slsqp(m, meq, n, x_internal, xl, xu, f, c, g, a, ftol, iter, &
                mode, w, l_w)

        ! Check if SLSQP has completed
        IF (.NOT. ABS(mode) == one_int) THEN
            is_finished = .True.
        END IF

    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************