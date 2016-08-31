!******************************************************************************
!******************************************************************************
MODULE solve_ambiguity

    !/*	external modules	*/

    USE recording_ambiguity

    USE shared_constants

    USE shared_auxiliary

    USE solve_risk

    !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE construct_emax_ambiguity(emax, num_periods, num_draws_emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta, shocks_cov, measure, level, is_write)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: emax

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: draws_emax_transformed(num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)      :: rewards_systematic(4)
    REAL(our_dble), INTENT(IN)      :: shocks_cov(4, 4)
    REAL(our_dble), INTENT(IN)      :: level
    REAL(our_dble), INTENT(IN)      :: delta

    CHARACTER(10), INTENT(IN)       :: measure

    LOGICAL, INTENT(IN)             :: is_write

    !/* internals objects    */

    REAL(our_dble)                  :: x_shift(2), x_start(2) = zero_dble
    REAL(our_dble)                  :: div, ftol, tiny

    CHARACTER(100)                  :: message

    LOGICAL                         :: is_success

    INTEGER(our_int)                :: maxiter



!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF(TRIM(measure) == 'abs') THEN
        x_shift = (/-level, -level/)
        div = -level
        is_success = .True.
        message = 'Optimization terminated successfully.'

    ELSE

        ! Parameterizations for optimizations
        x_start = zero_dble
        maxiter = 100000000_our_int
        ftol = 1e-06_our_dble
        tiny = 1.4901161193847656e-08

        CALL get_worst_case(x_shift, x_start, maxiter, ftol, tiny, num_draws_emax, draws_emax_transformed, period, k, rewards_systematic, edu_max, edu_start, mapping_state_idx, states_all, num_periods, periods_emax, delta, is_debug, shocks_cov, level)

    END IF

    IF(is_write) CALL record_ambiguity(period, k, x_shift, div, is_success, message)

    emax = criterion_ambiguity(x_shift, num_periods, num_draws_emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_worst_case(x_internal, x_start, maxiter, ftol, tiny, num_draws_emax, draws_emax_transformed, period, k, rewards_systematic, edu_max, edu_start, mapping_state_idx, states_all, num_periods, periods_emax, delta, is_debug, shocks_cov, level)

    ! TODO: Fix array dimensions

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: x_internal(:)

    REAL(our_dble), INTENT(IN)      :: shocks_cov(:, :)
    REAL(our_dble), INTENT(IN)      :: x_start(:)
    REAL(our_dble), INTENT(IN)      :: level
    REAL(our_dble), INTENT(IN)      :: ftol

    INTEGER(our_int), INTENT(IN)    :: maxiter

    REAL(our_dble), INTENT(IN)      :: rewards_systematic(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:,:)
    REAL(our_dble), INTENT(IN)      :: draws_emax_transformed(:, :)
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

    REAL(our_dble)                  :: div

    LOGICAL                         :: is_finished
    LOGICAL                         :: is_success

    !/* SLSQP interface          */

    INTEGER(our_int)                :: M        ! Total number of constraints
    INTEGER(our_int)                :: MEQ      ! Total number of equality constraints
    INTEGER(our_int)                :: LA       ! MAX(M, 1)
    INTEGER(our_int)                :: N        ! Number of variables

    REAL(our_dble)                  :: X(2)     ! Current iterate
    REAL(our_dble)                  :: XL(2)    ! Lower bounds for x
    REAL(our_dble)                  :: XU(2)    ! Upper bounds for x
    REAL(our_dble)                  :: F        ! Value of objective function

    REAL(our_dble), ALLOCATABLE     :: C(:)     ! Stores the constraints
    REAL(our_dble), ALLOCATABLE     :: G(:)     ! Partials of objective function
    REAL(our_dble), ALLOCATABLE     :: A(:, :)  ! Normals of constraints

    REAL(our_dble)                  :: ACC      ! Final accuracy
    INTEGER(our_int)                :: ITER     ! Maximum number of iterations
    INTEGER(our_int)                :: MODE     ! Control for communication

    REAL(our_dble), ALLOCATABLE     :: W(:)     ! Working space
    INTEGER(our_int), ALLOCATABLE   :: JW(:)    ! Working space
    INTEGER(our_int)                :: L_JW     ! Working space
    INTEGER(our_int)                :: L_W      ! Working space

    INTEGER(our_int)                :: MINEQ    ! Locals
    INTEGER(our_int)                :: N1       ! Locals

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Preparing SLSQP interface
    ACC = ftol; X = x_start; M = 1; MEQ = 1
    N = SIZE(x_internal); LA = MAX(1, M)
    N1 = N + 1;  MINEQ = M - MEQ + N1 + N1
    L_W = (3 * N1 + M) *( N1 + 1) + (N1 - MEQ + 1) * (MINEQ + 2) + 2 * MINEQ + &
        (N1 + MINEQ) * (N1 - MEQ) + 2 * MEQ + N1 + (N + 1) * N / 2 + &
        2 * M + 3 * N + 3 * N1 + 1
    L_JW = MINEQ

    ALLOCATE(C(LA), G(N + 1)); ALLOCATE(A(LA, N + 1))
    ALLOCATE(W(L_W), JW(L_JW))

    ! Decompose upper and lower bounds
    XL = - HUGE_FLOAT; XU = HUGE_FLOAT

    ! Initialize the iteration counter and MODE value
    ITER = maxiter
    MODE = zero_int

    ! Initialization of SLSQP
    is_finished = .False.

    ! Initialize criterion function at starting values
    F = criterion_ambiguity(x, num_periods, num_draws_emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta)

    G(:2) = criterion_ambiguity_derivative(x, num_periods, num_draws_emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta)

    ! Initialize constraint at starting values
    C = constraint_ambiguity(x, shocks_cov, level)

    A(1,:2) = constraint_ambiguity_derivative(x, shocks_cov, level, tiny)


    ! Iterate until completion
    DO WHILE (.NOT. is_finished)

        ! Evaluate criterion function and constraints
        IF (MODE == one_int) THEN

                F = criterion_ambiguity(x, num_periods, num_draws_emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta)

             C = constraint_ambiguity(x, shocks_cov, level)

         ! Evaluate gradient of criterion function and constraints. Note that the
         ! A is of dimension (1, N + 1) and the last element needs to always
         ! be zero.
         ELSEIF (MODE == - one_int) THEN
             G(:2) = criterion_ambiguity_derivative(x, num_periods, num_draws_emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta)


        A(1,:2) = constraint_ambiguity_derivative(x, shocks_cov, level, tiny)

         END IF

         !Call to SLSQP code
         CALL SLSQP(M, MEQ, LA, N, X, XL, XU, F, C, G, A, ACC, &
                 ITER, MODE, W, L_W, JW, L_JW)

         ! Check if SLSQP has completed
         IF (.NOT. ABS(MODE) == one_int) THEN
             is_finished = .True.
         END IF

     END DO
    !
    x_internal = X
    !
    ! ! Stabilization. If the optimization fails the starting values are
    ! ! used otherwise it happens that the constraint is not satisfied by far.
    ! is_success = (MODE == zero_int)
    !
     IF(.NOT. is_success) THEN
         x_internal = x_start
     END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION criterion_ambiguity_derivative(x, num_periods, num_draws_emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta)

    !/* external objects        */

    REAL(our_dble)                  :: criterion_ambiguity_derivative(2)

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: draws_emax_transformed(num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)      :: rewards_systematic(4)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: x(2)

    !/* internals objects       */

    REAL(our_dble)                  :: ei(2)
    REAL(our_dble)                  :: d(2)
    REAL(our_dble)                  :: f0
    REAL(our_dble)                  :: f1

    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    ei = zero_dble

    ! Evaluate baseline
    f0 = criterion_ambiguity(x, num_periods, num_draws_emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta)

    DO j = 1, 2

        ei(j) = one_dble

        d = dfunc_eps * ei

        f1 = criterion_ambiguity(x + d, num_periods, num_draws_emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta)

        criterion_ambiguity_derivative(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION constraint_ambiguity_derivative(x, shocks_cov, level, dfunc_eps)

    !/* external objects        */

    REAL(our_dble)                      :: constraint_ambiguity_derivative(2)

    REAL(our_dble), INTENT(IN)          :: shocks_cov(4, 4)
    REAL(our_dble), INTENT(IN)          :: dfunc_eps
    REAL(our_dble), INTENT(IN)          :: level
    REAL(our_dble), INTENT(IN)          :: x(2)

    !/* internals objects       */

    REAL(our_dble)                      :: ei(2)
    REAL(our_dble)                      :: d(2)
    REAL(our_dble)                      :: f0
    REAL(our_dble)                      :: f1

    INTEGER(our_int)                    :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    ei = zero_dble

    ! Evaluate baseline
    f0 = constraint_ambiguity(x, shocks_cov, level )

    DO j = 1, 2

        ei(j) = one_dble

        d = dfunc_eps * ei

        f1 = constraint_ambiguity(x + d, shocks_cov, level)

        constraint_ambiguity_derivative(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION criterion_ambiguity(x, num_periods, num_draws_emax, period, k, draws_emax_transformed, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta)

    !/* external objects    */

    REAL(our_dble)                  :: criterion_ambiguity

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: draws_emax_transformed(num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)      :: rewards_systematic(4)
    REAL(our_dble), INTENT(IN)      :: delta
    REAL(our_dble), INTENT(IN)      :: x(2)

    !/* internals objects    */

    REAL(our_dble)                  :: draws_relevant(num_draws_emax, 4)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    draws_relevant = draws_emax_transformed
    draws_relevant(:, 1) = draws_relevant(:, 1) + x(1)
    draws_relevant(:, 2) = draws_relevant(:, 2) + x(2)

    CALL construct_emax_risk(criterion_ambiguity, period, k, draws_relevant, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, delta)

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION kl_divergence(mean_old, cov_old, mean_new, cov_new)

    !/* external objects        */

    REAL(our_dble)                      :: kl_divergence

    REAL(our_dble), INTENT(IN)          :: cov_old(:, :)
    REAL(our_dble), INTENT(IN)          :: cov_new(:, :)
    REAL(our_dble), INTENT(IN)          :: mean_old(:)
    REAL(our_dble), INTENT(IN)          :: mean_new(:)

    !/* internal objects        */

    INTEGER(our_int)                    :: num_dims

    REAL(our_dble), ALLOCATABLE         :: cov_old_inv(:, :)
    REAL(our_dble), ALLOCATABLE         :: mean_diff(:, :)

    REAL(our_dble)                      :: comp_b(1, 1)
    REAL(our_dble)                      :: comp_a
    REAL(our_dble)                      :: comp_c

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    num_dims = SIZE(mean_old)

    ALLOCATE(cov_old_inv(num_dims, num_dims))
    ALLOCATE(mean_diff(num_dims, 1))

    mean_diff = RESHAPE(mean_old, (/num_dims, 1/)) - RESHAPE(mean_new, (/num_dims, 1/))
    cov_old_inv = inverse(cov_old, num_dims)

    comp_a = trace(MATMUL(cov_old_inv, cov_new))
    comp_b = MATMUL(MATMUL(TRANSPOSE(mean_diff), cov_old_inv), mean_diff)
    comp_c = LOG(determinant(cov_old) / determinant(cov_new))

    kl_divergence = half_dble * (comp_a + comp_b(1, 1) - num_dims + comp_c)

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION constraint_ambiguity(x, shocks_cov, level)

    !/* external objects        */

    REAL(our_dble)                      :: constraint_ambiguity

    REAL(our_dble), INTENT(IN)          :: shocks_cov(4, 4)
    REAL(our_dble), INTENT(IN)          :: level
    REAL(our_dble), INTENT(IN)          :: x(2)

    !/* internal objects        */

    REAL(our_dble)                      :: cov_old(4, 4)
    REAL(our_dble)                      :: cov_new(4, 4)
    REAL(our_dble)                      :: mean_old(4)
    REAL(our_dble)                      :: mean_new(4)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    cov_old = shocks_cov; cov_new = shocks_cov

    mean_old = zero_dble; mean_new = zero_dble

    mean_new(:2) = x

    constraint_ambiguity = level - kl_divergence(mean_old, cov_old, mean_new, cov_new)

END FUNCTION
!******************************************************************************
!******************************************************************************
END MODULE
