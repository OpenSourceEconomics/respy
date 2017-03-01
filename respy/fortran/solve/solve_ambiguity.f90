!******************************************************************************
!******************************************************************************
MODULE solve_ambiguity

    !/*	external modules	*/

    USE recording_ambiguity

    USE shared_interface

    USE solve_risk

    !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE construct_emax_ambiguity(emax, opt_ambi_details, num_periods, num_draws_emax, period, k, draws_emax_standard, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, shocks_cov, ambi_spec, optim_paras, optimizer_options)

    ! TODO: Why do I pass in shocks cov and optim%shocks_cholesky ...

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)             :: opt_ambi_details(num_periods, max_states_period, 7)
    REAL(our_dble), INTENT(OUT)             :: emax

    TYPE(OPTIMIZER_COLLECTION), INTENT(IN)  :: optimizer_options
    TYPE(OPTIMPARAS_DICT), INTENT(IN)       :: optim_paras
    TYPE(AMBI_DICT), INTENT(IN)             :: ambi_spec

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: draws_emax_standard(num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)      :: rewards_systematic(4)
    REAL(our_dble), INTENT(IN)      :: shocks_cov(4, 4)

    !/* internals objects    */

    INTEGER(our_int)                :: mode

    REAL(our_dble)                  :: x_shift(num_free_ambi)
    REAL(our_dble)                  :: div(1)

    REAL(our_dble)                  :: ambi_rslt_mean_subset(2)
    REAL(our_dble)                  :: ambi_rslt_chol_subset(3)
    REAL(our_dble)                  :: ambi_rslt_all(5)
    REAL(our_dble)                  :: is_success

    LOGICAL                         :: is_deterministic
    REAL(our_dble)                  :: rslt_mean(2), rslt_sd(2), rslt_all(4)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------




    is_deterministic = ALL(shocks_cov .EQ. zero_dble)

    IF(is_deterministic) THEN
        rslt_mean = zero_dble; rslt_sd = zero_dble
        div = zero_dble; is_success = one_dble; mode = 15

    ELSE IF(TRIM(ambi_spec%measure) .EQ. 'abs') THEN
        rslt_mean = (/-optim_paras%level, -optim_paras%level/)
        rslt_sd = (/DSQRT(shocks_cov(1, 1)), DSQRT(shocks_cov(2, 2))/)
        div = optim_paras%level; is_success = one_dble; mode = 16

    ELSE

        rslt_mean = -MISSING_FLOAT
        rslt_sd = -MISSING_FLOAT
        !CALL get_worst_case(x_shift, is_success, mode, num_periods, num_draws_emax, period, k, draws_emax_standard, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, shocks_cov, optim_paras, optimizer_options)

        !ambi_rslt_mean_subset = x_shift(:2)
        !ambi_rslt_chol_subset = get_upper_cholesky(optim_paras)
        !IF (.NOT. ambi_spec%mean) ambi_rslt_chol_subset = x_shift(3:)

        !div = -(constraint_ambiguity(x_shift, shocks_cov, optim_paras) - optim_paras%level)
    END IF

    ! We collect the information from the optimization step for future recording.
    rslt_all = (/rslt_mean, rslt_sd/)

    opt_ambi_details(period + 1, k + 1, :) = (/rslt_all, div, is_success, DBLE(mode)/)

!    emax = criterion_ambiguity(ambi_rslt_all, num_periods, num_draws_emax, period, k, draws_emax_standard, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, optim_paras)
    emax = zero_dble

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_worst_case(x_shift, is_success, mode, num_periods, num_draws_emax, period, k, draws_emax_standard, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, shocks_cov, optim_paras, optimizer_options)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)             :: is_success

    INTEGER(our_int), INTENT(OUT)           :: mode

    REAL(our_dble), INTENT(INOUT)           :: x_shift(num_free_ambi)

    TYPE(OPTIMIZER_COLLECTION), INTENT(IN)  :: optimizer_options
    TYPE(OPTIMPARAS_DICT), INTENT(IN)       :: optim_paras

    REAL(our_dble), INTENT(IN)              :: draws_emax_standard(num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)              :: rewards_systematic(4)
    REAL(our_dble), INTENT(IN)              :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)              :: shocks_cov(4, 4)

    INTEGER(our_int), INTENT(IN)            :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)            :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)            :: num_draws_emax
    INTEGER(our_int), INTENT(IN)            :: num_periods
    INTEGER(our_int), INTENT(IN)            :: edu_start
    INTEGER(our_int), INTENT(IN)            :: edu_max
    INTEGER(our_int), INTENT(IN)            :: period
    INTEGER(our_int), INTENT(IN)            :: k

    !/* internal objects        */

    REAL(our_dble)                          :: eps_der_approx

    LOGICAL                                 :: is_finished

    !/* SLSQP interface          */

    INTEGER(our_int)                        :: ITER                     ! Maximum number of iterations
    INTEGER(our_int)                        :: MEQ                      ! Total number of equality constraints
    INTEGER(our_int)                        :: LA                       ! MAX(M, 1)
    INTEGER(our_int)                        :: M                        ! Total number of constraints
    INTEGER(our_int)                        :: N                        ! Number of variables

    REAL(our_dble)                          :: A(1, num_free_ambi + 1)  ! Normals of constraints
    REAL(our_dble)                          :: XL(num_free_ambi)        ! Lower bounds for x
    REAL(our_dble)                          :: XU(num_free_ambi)        ! Upper bounds for x
    REAL(our_dble)                          :: G(num_free_ambi  + 1)    ! Partials of objective function
    REAL(our_dble)                          :: X(num_free_ambi)         ! Current iterate
    REAL(our_dble)                          :: C(1)                     ! Stores the constraints
    REAL(our_dble)                          :: ACC                      ! Final accuracy
    REAL(our_dble)                          :: F                        ! Value of objective function

    REAL(our_dble)                          :: W(1200)  ! Working space
    INTEGER(our_int)                        :: JW(60)   ! Working space
    INTEGER(our_int)                        :: LEN_JW   ! Working space
    INTEGER(our_int)                        :: LEN_W    ! Working space

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Setting up the starting values
    IF (.NOT. ambi_spec%mean) x_shift(3:) = get_upper_cholesky(optim_paras)
    x_shift(:2) = zero_dble

    ! Setup
    eps_der_approx = optimizer_options%slsqp%eps

    ! Preparing SLSQP interface
    ITER = optimizer_options%slsqp%maxiter
    ACC = optimizer_options%slsqp%ftol
    mode = zero_int
    X = x_shift

    ! These settings are deduced from the documentation at the beginning of the source file slsq.f and hard-coded for the application at hand to save some cluttering code.
    M = 1; N = num_free_ambi; LA = 1; LEN_W = 1200; LEN_JW = 60; MEQ = 1
    XL = - HUGE_FLOAT; XU = HUGE_FLOAT

    ! Initialization of SLSQP
    is_finished = .False.

    ! Initialize criterion function at starting values
    F = criterion_ambiguity(x, num_periods, num_draws_emax, period, k, draws_emax_standard, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, optim_paras)

    G(:num_free_ambi) = criterion_ambiguity_derivative(x, num_periods, num_draws_emax, period, k, draws_emax_standard, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, optim_paras, eps_der_approx)

    ! Initialize constraint at starting values
    C = constraint_ambiguity(x, shocks_cov, optim_paras)

    A(1,:num_free_ambi) = constraint_ambiguity_derivative(x, shocks_cov, optim_paras, eps_der_approx)

    ! Iterate until completion
    DO WHILE (.NOT. is_finished)

        ! Evaluate criterion function and constraints
        IF (mode .EQ. one_int) THEN

            F = criterion_ambiguity(x, num_periods, num_draws_emax, period, k, draws_emax_standard, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, optim_paras)
            C = constraint_ambiguity(x, shocks_cov, optim_paras)

        ! Evaluate gradient of criterion function and constraints.
    ELSEIF (mode .EQ. - one_int) THEN
            G(:num_free_ambi) = criterion_ambiguity_derivative(x, num_periods, num_draws_emax, period, k, draws_emax_standard, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, optim_paras, eps_der_approx)
            A(1,:num_free_ambi) = constraint_ambiguity_derivative(x, shocks_cov, optim_paras, eps_der_approx)

        END IF

        !Call to SLSQP code
        CALL SLSQP(M, MEQ, LA, N, X, XL, XU, F, C, G, A, ACC, ITER, mode, W, LEN_W, JW, LEN_JW)

        ! Check if SLSQP has completed
        IF (.NOT. ABS(mode) .EQ. one_int) THEN
            is_finished = .True.
        END IF

    END DO

    ! Stabilization. If the optimization fails the starting values are
    ! used otherwise it happens that the constraint is not satisfied by far.
    is_success = zero_dble
    IF (mode .EQ. zero_int) is_success = one_dble

    IF(is_success .EQ. one_dble) THEN
        x_shift = X
    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION criterion_ambiguity_derivative(x, num_periods, num_draws_emax, period, k, draws_emax_standard, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, optim_paras, eps_der_approx)

    !/* external objects        */
    REAL(our_dble), INTENT(IN)      :: x(num_free_ambi)

    REAL(our_dble)                              :: criterion_ambiguity_derivative(num_free_ambi)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: draws_emax_standard(num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)      :: rewards_systematic(4)
    REAL(our_dble), INTENT(IN)      :: eps_der_approx

    !/* internals objects       */

    REAL(our_dble)                  :: ei(num_free_ambi)
    REAL(our_dble)                  :: d(num_free_ambi)
    REAL(our_dble)                  :: f0
    REAL(our_dble)                  :: f1

    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    ei = zero_dble

    ! Evaluate baseline
    f0 = criterion_ambiguity(x, num_periods, num_draws_emax, period, k, draws_emax_standard, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, optim_paras)

    DO j = 1, num_free_ambi

        ei(j) = one_dble

        d = eps_der_approx * ei

        f1 = criterion_ambiguity(x + d, num_periods, num_draws_emax, period, k, draws_emax_standard, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, optim_paras)

        criterion_ambiguity_derivative(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION constraint_ambiguity_derivative(x, shocks_cov, optim_paras, eps_der_approx)

    !/* external objects        */
    REAL(our_dble), INTENT(IN)          :: x(num_free_ambi)

    REAL(our_dble)                      :: constraint_ambiguity_derivative(num_free_ambi)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    REAL(our_dble), INTENT(IN)          :: shocks_cov(4, 4)
    REAL(our_dble), INTENT(IN)          :: eps_der_approx

    !/* internals objects       */

    REAL(our_dble)                      :: ei(num_free_ambi)
    REAL(our_dble)                      :: d(num_free_ambi)
    REAL(our_dble)                      :: f0
    REAL(our_dble)                      :: f1

    INTEGER(our_int)                    :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    ei = zero_dble

    ! Evaluate baseline
    f0 = constraint_ambiguity(x, shocks_cov, optim_paras)

    DO j = 1, num_free_ambi

        ei(j) = one_dble

        d = eps_der_approx * ei

        f1 = constraint_ambiguity(x + d, shocks_cov, optim_paras)

        constraint_ambiguity_derivative(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION criterion_ambiguity(x, num_periods, num_draws_emax, period, k, draws_emax_standard, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, optim_paras)

    !/* external objects    */

    REAL(our_dble)                      :: criterion_ambiguity

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 4)
    INTEGER(our_int), INTENT(IN)    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: edu_start
    INTEGER(our_int), INTENT(IN)    :: edu_max
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: draws_emax_standard(num_draws_emax, 4)
    REAL(our_dble), INTENT(IN)      :: rewards_systematic(4)
    REAL(our_dble), INTENT(IN)      :: x(num_free_ambi)

    !/* internals objects    */

    REAL(our_dble)                  :: draws_relevant(num_draws_emax, 4)
    REAL(our_dble)                  :: ambi_cand_mean_full(4), shocks_cov(4, 4)
    REAL(our_dble)                  :: ambi_cand_chol(4, 4), ambi_cand_cov(4, 4), ambi_cand_chol_flat(3)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    shocks_cov = MATMUL(optim_paras%shocks_cholesky, TRANSPOSE(optim_paras%shocks_cholesky))
    ambi_cand_mean_full(:2) = x(:2)
    ambi_cand_mean_full(3:) = (/zero_dble, zero_dble/)

    IF (num_free_ambi .EQ. 2) THEN
        ambi_cand_chol = optim_paras%shocks_cholesky
    ELSE
        ambi_cand_chol_flat = x(3:)
        !CALL construct_full_covariances(ambi_cand_cov, ambi_cand_chol, ambi_cand_chol_flat, shocks_cov)
    END IF


    CALL transform_disturbances(draws_relevant, draws_emax_standard, ambi_cand_mean_full, ambi_cand_chol)

    CALL construct_emax_risk(criterion_ambiguity, period, k, draws_relevant, rewards_systematic, edu_max, edu_start, periods_emax, states_all, mapping_state_idx, optim_paras)

END FUNCTION
!******************************************************************************
!******************************************************************************
PURE FUNCTION get_upper_cholesky(optim_paras)

    !/* external objects    */

    REAL(our_dble)                      :: get_upper_cholesky(3)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    !/* internal objects    */

    REAL(our_dble)                      :: shocks_cholesky(4, 4)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    shocks_cholesky = optim_paras%shocks_cholesky

    get_upper_cholesky(1) = shocks_cholesky(1, 1)
    get_upper_cholesky(2) = shocks_cholesky(2, 1)
    get_upper_cholesky(3) = shocks_cholesky(2, 2)

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
    cov_old_inv = pinv(cov_old, num_dims)

    comp_a = trace(MATMUL(cov_old_inv, cov_new))
    comp_b = MATMUL(MATMUL(TRANSPOSE(mean_diff), cov_old_inv), mean_diff)
    comp_c = LOG(determinant(cov_old) / determinant(cov_new))

    kl_divergence = half_dble * (comp_a + comp_b(1, 1) - num_dims + comp_c)

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION constraint_ambiguity(x, shocks_cov, optim_paras)

    !/* external objects        */

    REAL(our_dble)                              :: constraint_ambiguity

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    REAL(our_dble), INTENT(IN)          :: shocks_cov(4, 4)

    REAL(our_dble), INTENT(IN)          :: x(:)

    !/* internal objects        */

    REAL(our_dble)                      :: cov_old(4, 4)
    REAL(our_dble)                      :: cov_new(4, 4)
    REAL(our_dble)                      :: mean_old(4)
    REAL(our_dble)                      :: mean_new(4), mock_obj(4, 4)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    mean_old = zero_dble; mean_new = zero_dble
    mean_new(:2) = x(:2)

    cov_old = shocks_cov

    IF (num_free_ambi .EQ. 2) THEN
        cov_new = shocks_cov
    ELSE
        !CALL construct_full_covariances(cov_new, mock_obj, x(3:), shocks_cov)
    END IF


    constraint_ambiguity = optim_paras%level(1) - kl_divergence(mean_old, cov_old, mean_new, cov_new)

END FUNCTION
!******************************************************************************
!******************************************************************************
END MODULE
