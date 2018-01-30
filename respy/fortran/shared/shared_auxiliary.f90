!******************************************************************************
!******************************************************************************
MODULE shared_auxiliary

    !/* external modules    */

    USE shared_interfaces

    USE recording_warning

    USE shared_constants

    USE shared_utilities

    USE shared_types

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

    !/* explicit interfaces   */

    INTERFACE clip_value

        MODULE PROCEDURE clip_value_1, clip_value_2

    END INTERFACE

    INTERFACE to_boolean

        MODULE PROCEDURE float_to_boolean, integer_to_boolean

    END INTERFACE

CONTAINS
!******************************************************************************
!******************************************************************************
FUNCTION get_scales_magnitudes(x_optim_free_start) RESULT(precond_matrix)

    !/* external objects    */

    REAL(our_dble)                              :: precond_matrix(num_free, num_free)

    REAL(our_dble), INTENT(IN)                  :: x_optim_free_start(num_free)

    !/* internal objects    */

    INTEGER(our_int)                            :: magnitude
    INTEGER(our_int)                            :: i

    REAL(our_dble)                              :: scale
    REAL(our_dble)                              :: val

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    precond_matrix = zero_dble

    DO i = 1, num_free
        val = x_optim_free_start(i)

        IF (val .EQ. zero_dble) THEN
            scale = one_dble
        ELSE
            magnitude = FLOOR(LOG10(ABS(val)))
            IF (magnitude .EQ. zero_int) THEN
                scale = one_dble / ten_dble
            ELSE
                scale = (ten_dble ** magnitude) ** (-one_dble) / ten_dble
            END IF
        END IF

        precond_matrix(i, i) = scale

    END DO

END FUNCTION
!*******************************************************************************
!*******************************************************************************
FUNCTION get_conditional_probabilities(type_shares, edu_start) RESULT(probs)

    !/* external objects        */

    REAL(our_dble)                      :: probs(num_types)

    REAL(our_dble), INTENT(IN)          :: type_shares(num_types * 2)

    INTEGER(our_int), INTENT(IN)        :: edu_start

    !/* internal objects        */

    INTEGER(our_int)                    :: covariate
    INTEGER(our_int)                    :: lower
    INTEGER(our_int)                    :: upper
    INTEGER(our_int)                    :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    probs = -HUGE_FLOAT
    DO i = 1, num_types
        lower = (i - 1) * 2 + 1
        upper = i * 2

        covariate = TRANSFER(edu_start > 9, our_int)
        probs(i) = DEXP(SUM(type_shares(lower:upper) * (/one_int, covariate/)))
    END DO

    probs = probs / SUM(probs)

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION check_early_termination(maxfun, num_eval, crit_estimation) RESULT(is_termination)

    !/* external objects    */

    LOGICAL                         :: is_termination

    INTEGER, INTENT(IN)             :: num_eval
    INTEGER, INTENT(IN)             :: maxfun

    LOGICAL, INTENT(IN)             :: crit_estimation

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Ensuring that the criterion function is not evaluated more than specified. However, there is the special request of MAXFUN equal to zero which needs to be allowed.
    is_termination = (num_eval == maxfun) .AND. crit_estimation .AND. (.NOT. maxfun == zero_int)

    ! We also want to allow for a gentle termination by the user.
    IF (.NOT. is_termination) INQUIRE(FILE='.stop.respy.scratch', EXIST=is_termination)

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE correlation_to_covariance(cov, corr, sd)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)         :: cov(:, :)

    REAL(our_dble), INTENT(IN)          :: corr(:, :)
    REAL(our_dble), INTENT(IN)          :: sd(:)

    !/* internal objects        */

    INTEGER(our_int)                    :: nrows
    INTEGER(our_int)                    :: i
    INTEGER(our_int)                    :: j

    LOGICAL                             :: is_deterministic

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! This special case is maintained for testing purposes.
    is_deterministic = ALL(corr .EQ. zero_dble)
    IF (is_deterministic) THEN
        cov = zero_dble
        RETURN
    END IF

    ! Auxiliary objects
    nrows = SIZE(corr, 1)

    DO i = 1, nrows
        DO j  = 1, nrows
            cov(i, j) = corr(i, j) * sd(i) * sd(j)
        END DO
    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE covariance_to_correlation(corr, cov)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)         :: corr(:, :)

    REAL(our_dble), INTENT(IN)          :: cov(:, :)

    !/* internal objects        */

    INTEGER(our_int)                    :: nrows
    INTEGER(our_int)                    :: i
    INTEGER(our_int)                    :: j

    LOGICAL                             :: is_deterministic

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! This special case is maintained for testing purposes.
    is_deterministic = ALL(cov .EQ. zero_dble)
    IF (is_deterministic) THEN
        corr = zero_dble
        RETURN
    END IF

    ! Auxiliary objects
    nrows = SIZE(corr, 1)

    DO i = 1, nrows
        DO j = 1, nrows
            corr(i, j) = cov(i, j) / (DSQRT(cov(i, i)) * DSQRT(cov(j, j)))
        END DO
    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_cholesky_decomposition(C, info, A)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: C(:, :)

    INTEGER(our_int), INTENT(OUT)       :: info

    REAL(our_dble), INTENT(IN)          :: A(:, :)

    !/* internal objects        */

    REAL(our_dble)                      :: mock_obj(SIZE(A,1), SIZE(A,1))

    INTEGER(our_int)                    :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    mock_obj = A
    CALL DPOTRF('L', SIZE(A,1), mock_obj, SIZE(A,1), info)
    C = mock_obj

    DO i = 1, SIZE(A,1)
        C(i, (i + 1):) = zero_dble
    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION get_log_likl(contribs)

      !/* external objects    */

      REAL(our_dble)                  :: get_log_likl

      REAL(our_dble), INTENT(IN)      :: contribs(num_agents_est)

      !/* internal objects        */

      INTEGER(our_int), ALLOCATABLE   :: infos(:)

      REAL(our_dble)                  :: contribs_clipped(num_agents_est)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL clip_value_2(contribs_clipped, LOG(contribs), -HUGE_FLOAT, HUGE_FLOAT, infos)
    IF (SUM(infos) > zero_int) CALL record_warning(5)

    get_log_likl = -SUM(contribs_clipped) / SIZE(contribs, 1)

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION apply_scaling(x_in, precond_matrix, request)

    !/* external objects    */

    REAL(our_dble)                  :: apply_scaling(num_free)

    REAL(our_dble), INTENT(IN)      :: precond_matrix(num_free, num_free)
    REAL(our_dble), INTENT(IN)      :: x_in(num_free)

    CHARACTER(*), INTENT(IN)        :: request

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF (request == 'do') THEN
        apply_scaling = MATMUL(precond_matrix, x_in)
    ELSE
        apply_scaling = MATMUL(pinv(precond_matrix, num_free), x_in)
    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE summarize_worst_case_success(opt_ambi_summary, opt_ambi_details)

    !/* external objects        */

    INTEGER(our_int), INTENT(OUT)   :: opt_ambi_summary(2)

    REAL(our_dble), INTENT(IN)      :: opt_ambi_details(num_periods, max_states_period, 7)

    !/* internal objects        */

    INTEGER(our_int)                :: period

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    opt_ambi_summary = zero_int
    DO period = 1, num_periods
        opt_ambi_summary(1) = opt_ambi_summary(1) + COUNT(opt_ambi_details(period, : , 4) .GE. zero_int)
        opt_ambi_summary(2) = opt_ambi_summary(2) + COUNT(opt_ambi_details(period, : , 4) .EQ. one_int)
    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE extract_cholesky(shocks_cholesky, x, info)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: shocks_cholesky(4, 4)

    REAL(our_dble), INTENT(IN)      :: x(:)

    INTEGER(our_int), OPTIONAL, INTENT(OUT)    :: info

    !/* internal objects        */

    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: shocks_cov(4, 4)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    shocks_cholesky = zero_dble

    shocks_cholesky(1, :1) = x(45:45)

    shocks_cholesky(2, :2) = x(46:47)

    shocks_cholesky(3, :3) = x(48:50)

    shocks_cholesky(4, :4) = x(51:54)

    ! We need to ensure that the diagonal elements are larger than zero during an estimation. However, we want to allow for the special case of total absence of randomness for testing purposes of simulated datasets.
    IF (.NOT. ALL(shocks_cholesky .EQ. zero_dble)) THEN
        IF (PRESENT(info)) info = 0
        shocks_cov = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))
        DO i = 1, 4
            IF (ABS(shocks_cov(i, i)) .LT. TINY_FLOAT) THEN
                shocks_cholesky(i, i) = SQRT(TINY_FLOAT)
                IF (PRESENT(info)) info = 1
            END IF
        END DO

    END IF
END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE transform_disturbances(draws_transformed, draws, shocks_mean, shocks_cholesky)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: draws_transformed(:, :)

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)      :: shocks_mean(4)
    REAL(our_dble), INTENT(IN)      :: draws(:, :)

    !/* internal objects        */

    INTEGER(our_int), ALLOCATABLE   :: infos(:)

    INTEGER(our_int)                :: num_draws
    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Auxiliary objects
    num_draws = SIZE(draws, 1)

    DO i = 1, num_draws
        draws_transformed(i:i, :) = TRANSPOSE(MATMUL(shocks_cholesky, TRANSPOSE(draws(i:i, :))))
    END DO

    DO i = 1, 4
        draws_transformed(:, i) = draws_transformed(:, i) + shocks_mean(i)
    END DO

    DO i = 1, 2
        CALL clip_value_2(draws_transformed(:, i), EXP(draws_transformed(:, i)), zero_dble, HUGE_FLOAT, infos)
    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_total_values(total_values, rewards_ex_post, period, num_periods, rewards_systematic, draws, mapping_state_idx, periods_emax, k, states_all, optim_paras, edu_spec)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: rewards_ex_post(4)
    REAL(our_dble), INTENT(OUT)         :: total_values(4)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras
    TYPE(EDU_DICT), INTENT(IN)          :: edu_spec

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 4, num_types)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 5)
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)
    REAL(our_dble), INTENT(IN)      :: rewards_systematic(4)
    REAL(our_dble), INTENT(IN)      :: draws(4)

    !/* internal objects        */

    REAL(our_dble)                  :: wages_systematic(2)
    REAL(our_dble)                  :: total_increment(2)
    REAL(our_dble)                  :: emaxs(4)

    INTEGER(our_int)                :: activity_lagged
    INTEGER(our_int)                :: exp_a
    INTEGER(our_int)                :: exp_b
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! We need to back out the wages from the total systematic rewards to working in the labor market to add the shock properly.
    exp_a = states_all(period + 1, k + 1, 1)
    exp_b = states_all(period + 1, k + 1, 2)
    edu = states_all(period + 1, k + 1, 3)
    activity_lagged = states_all(period + 1, k + 1, 4)

    wages_systematic = back_out_systematic_wages(rewards_systematic, exp_a, exp_b, edu, activity_lagged, optim_paras)

    ! Initialize containers
    rewards_ex_post = zero_dble

    ! Calculate ex post rewards
    DO i = 1, 2
        total_increment(i) = rewards_systematic(i) - wages_systematic(i)
        rewards_ex_post(i) = wages_systematic(i) * draws(i) + total_increment(i)
    END DO

    Do i = 3, 4
        rewards_ex_post(i) = rewards_systematic(i) + draws(i)
    END DO

    ! Get future values
    IF (period .NE. (num_periods - one_int)) THEN
        CALL get_emaxs(emaxs, mapping_state_idx, period, periods_emax, k, states_all, edu_spec)
    ELSE
        emaxs = zero_dble
    END IF

    ! Calculate total utilities
    total_values = rewards_ex_post + optim_paras%delta(1) * emaxs

    ! This is required to ensure that the agent does not choose any inadmissible states. If the state is inadmissible emaxs takes value zero.
    IF (states_all(period + 1, k + 1, 3) >= edu_spec%max) THEN
        total_values(3) = total_values(3) + INADMISSIBILITY_PENALTY
    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_emaxs(emaxs, mapping_state_idx, period, periods_emax, k, states_all, edu_spec)

    !/* external objects        */

    TYPE(EDU_DICT), INTENT(IN)      :: edu_spec

    REAL(our_dble), INTENT(OUT)     :: emaxs(4)

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 4, num_types)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 5)
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(num_periods, max_states_period)

    !/* internals objects       */

    INTEGER(our_int)                :: future_idx
    INTEGER(our_int)                :: exp_a
    INTEGER(our_int)                :: exp_b
    INTEGER(our_int)                :: type_
    INTEGER(our_int)                :: edu

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Distribute state space
    exp_a = states_all(period + 1, k + 1, 1)
    exp_b = states_all(period + 1, k + 1, 2)
    edu = states_all(period + 1, k + 1, 3)
    type_ = states_all(period + 1, k + 1, 5)

    ! Working in Occupation A
    future_idx = mapping_state_idx(period + 1 + 1, exp_a + 1 + 1, exp_b + 1, edu + 1, 3, type_ + 1)
    emaxs(1) = periods_emax(period + 1 + 1, future_idx + 1)

    ! Working in Occupation B
    future_idx = mapping_state_idx(period + 1 + 1, exp_a + 1, exp_b + 1 + 1, edu + 1, 4, type_ + 1)
    emaxs(2) = periods_emax(period + 1 + 1, future_idx + 1)

    ! Increasing schooling. Note that adding an additional year of schooling is only possible for those that have strictly less than the maximum level of additional education allowed.
    IF(edu .GE. edu_spec%max) THEN
        emaxs(3) = zero_dble
    ELSE
        future_idx = mapping_state_idx(period + 1 + 1, exp_a + 1, exp_b + 1, edu + 1 + 1, 2, type_ + 1)
        emaxs(3) = periods_emax(period + 1 + 1, future_idx + 1)
    END IF

    ! Staying at home
    future_idx = mapping_state_idx(period + 1 + 1, exp_a + 1, exp_b + 1, edu + 1, 1, type_ + 1)
    emaxs(4) = periods_emax(period + 1 + 1, future_idx + 1)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE create_draws(draws, num_draws, seed, is_debug)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)  :: draws(:, :, :)

    INTEGER(our_int), INTENT(IN)                :: num_draws
    INTEGER(our_int), INTENT(IN)                :: seed

    LOGICAL, INTENT(IN)                         :: is_debug

    !/* internal objects        */

    INTEGER(our_int)                            :: seed_inflated(15)
    INTEGER(our_int)                            :: seed_size
    INTEGER(our_int)                            :: period
    INTEGER(our_int)                            :: j
    INTEGER(our_int)                            :: i

    REAL(our_dble)                              :: deviates(4)

    LOGICAL                                     :: READ_IN

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Allocate containers
    ALLOCATE(draws(num_periods, num_draws, 4))

    ! Set random seed
    seed_inflated(:) = seed

    CALL RANDOM_SEED(size=seed_size)

    CALL RANDOM_SEED(put=seed_inflated)

    ! Draw random deviates from a standard normal distribution or read it in
    ! from disk. The latter is available to allow for testing across
    ! implementations.
    INQUIRE(FILE='.draws.respy.test', EXIST=READ_IN)

    IF ((READ_IN .EQV. .True.)  .AND. (is_debug .EQV. .True.)) THEN

        OPEN(UNIT=99, FILE='.draws.respy.test', ACTION='READ')

        DO period = 1, num_periods

            DO j = 1, num_draws

                2000 FORMAT(4(1x,f15.10))
                READ(99,2000) draws(period, j, :)

            END DO

        END DO

        CLOSE(99)

    ELSE

        DO period = 1, num_periods

            DO i = 1, num_draws

               CALL standard_normal(deviates)

               draws(period, i, :) = deviates

            END DO

        END DO

    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE standard_normal(draw)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: draw(:)

    !/* internal objects        */

    INTEGER(our_int)                :: dim
    INTEGER(our_int)                :: g

    REAL(our_dble), ALLOCATABLE     :: u(:)
    REAL(our_dble), ALLOCATABLE     :: r(:)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

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
!******************************************************************************
!******************************************************************************
PURE FUNCTION trace(A)

    !/* external objects        */

    REAL(our_dble)              :: trace

    REAL(our_dble), INTENT(IN)  :: A(:,:)

    !/* internals objects       */

    INTEGER(our_int)            :: i
    INTEGER(our_int)            :: n

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Get dimension
    n = SIZE(A, DIM = 1)

    ! Initialize results
    trace = zero_dble

    ! Calculate trace
    DO i = 1, n

        trace = trace + A(i, i)

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION inverse(A, n)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: n

    REAL(our_dble), INTENT(IN)      :: A(:, :)

    !/* internal objects        */

    INTEGER(our_int)                :: ipiv(n)
    INTEGER(our_int)                :: info

    REAL(our_dble)                  :: inverse(n, n)
    REAL(our_dble)                  :: work(n)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize matrix for replacement
    inverse = A

    ! DGETRF computes an LU factorization of a general M-by-N matrix A
    ! using partial pivoting with row interchanges.
    CALL DGETRF(n, n, inverse, n, ipiv, info)

    IF (INFO .NE. zero_int) THEN
        STOP 'LU factorization failed'
    END IF

    ! DGETRI computes the inverse of a matrix using the LU factorization
    ! computed by DGETRF.
    CALL DGETRI(n, inverse, n, ipiv, work, n, info)

    IF (INFO .NE. zero_int) THEN
        STOP 'Matrix inversion failed'
    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION determinant(A)

    !/* external objects        */

    REAL(our_dble)                :: determinant

    REAL(our_dble), INTENT(IN)    :: A(:, :)

    !/* internal objects        */

    INTEGER(our_int), ALLOCATABLE :: IPIV(:)

    INTEGER(our_int)              :: INFO
    INTEGER(our_int)              :: N
    INTEGER(our_int)              :: i

    REAL(our_dble), ALLOCATABLE   :: B(:, :)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Auxiliary objects
    N = SIZE(A, 1)

    ! Allocate auxiliary containers
    ALLOCATE(B(N, N))
    ALLOCATE(IPIV(N))

    ! Initialize matrix for replacement
    B = A

    ! Compute an LU factorization of a general M-by-N matrix A using
    ! partial pivoting with row interchanges
    CALL DGETRF( N, N, B, N, IPIV, INFO )

    IF (INFO .NE. zero_int) THEN
        STOP 'LU factorization failed'
    END IF

    ! Compute the product of the diagonal elements, accounting for
    ! interchanges of rows.
    determinant = one_dble
    DO  i = 1, N
        IF(IPIV(i) .NE. i) THEN
            determinant = -determinant * B(i,i)
        ELSE
            determinant = determinant * B(i,i)
        END IF
    END DO


END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE store_results(request, mapping_state_idx, states_all, periods_rewards_systematic, states_number_period, periods_emax, data_sim)

    !/* external objects        */


    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 4, num_types)
    INTEGER(our_int), INTENT(IN)    :: states_all(num_periods, max_states_period, 5)
    INTEGER(our_int), INTENT(IN)    :: states_number_period(num_periods)

    REAL(our_dble), ALLOCATABLE, INTENT(IN)      :: periods_rewards_systematic(: ,:, :)
    REAL(our_dble), ALLOCATABLE, INTENT(IN)      :: periods_emax(: ,:)
    REAL(our_dble), ALLOCATABLE, INTENT(IN)      :: data_sim(:, :)

    CHARACTER(10), INTENT(IN)       :: request

    !/* internal objects        */

    INTEGER(our_int)                :: min_idx
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: l

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Auxiliary variables
    min_idx = SIZE(mapping_state_idx, 4)

    IF (request == 'simulate') THEN

        ! Write out results for the store results.
        1800 FORMAT(5(1x,i10))

        OPEN(UNIT=99, FILE='.mapping_state_idx.resfort.dat', ACTION='WRITE')

        DO period = 1, num_periods
            DO i = 1, num_periods
                DO j = 1, num_periods
                    DO k = 1, min_idx
                        DO l = 1, 4
                            WRITE(99, 1800) mapping_state_idx(period, i, j, k, l, :)
                        END DO
                    END DO
                END DO
            END DO
        END DO

        CLOSE(99)


        2000 FORMAT(5(1x,i5))

        OPEN(UNIT=99, FILE='.states_all.resfort.dat', ACTION='WRITE')

        DO period = 1, num_periods
            DO i = 1, max_states_period
                WRITE(99, 2000) states_all(period, i, :)
            END DO
        END DO

        CLOSE(9)


        1900 FORMAT(4(1x,f45.15))

        OPEN(UNIT=99, FILE='.periods_rewards_systematic.resfort.dat', ACTION='WRITE')

        DO period = 1, num_periods
            DO i = 1, max_states_period
                WRITE(99, 1900) periods_rewards_systematic(period, i, :)
            END DO
        END DO

        CLOSE(99)

        2100 FORMAT(i10)

        OPEN(UNIT=99, FILE='.states_number_period.resfort.dat', ACTION='WRITE')

        DO period = 1, num_periods
            WRITE(99, 2100) states_number_period(period)
        END DO

        CLOSE(99)


        2200 FORMAT(i10)

        OPEN(UNIT=99, FILE='.max_states_period.resfort.dat', ACTION='WRITE')

        WRITE(99, 2200) max_states_period

        CLOSE(99)


        2400 FORMAT(1000000(1x,f45.15))

        OPEN(UNIT=99, FILE='.periods_emax.resfort.dat', ACTION='WRITE')

        DO period = 1, num_periods
            WRITE(99, 2400) periods_emax(period, :)
        END DO

        CLOSE(99)

    END IF

    IF (request == 'simulate') THEN

        OPEN(UNIT=99, FILE='.simulated.resfort.dat', ACTION='WRITE')

        DO period = 1, num_periods * num_agents_sim
            WRITE(99, 2400) data_sim(period, :)
        END DO

        CLOSE(99)

    END IF

    ! Remove temporary files
    OPEN(UNIT=99, FILE='.model.resfort.ini'); CLOSE(99, STATUS='delete')
    OPEN(UNIT=99, FILE='.data.resfort.dat'); CLOSE(99, STATUS='delete')

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE read_specification(optim_paras, tau, seed_sim, seed_emax, seed_prob, num_procs, num_slaves, is_debug, is_interpolated, num_points_interp, is_myopic, request, exec_dir, maxfun, num_free, edu_spec, precond_spec, ambi_spec, optimizer_used, optimizer_options, file_sim, num_rows, num_paras)

    !
    !   This function serves as the replacement for the RespyCls and reads in all required information about the model parameterization. It just reads in all required information.
    !

    !/* external objects        */

    TYPE(PRECOND_DICT), INTENT(OUT)         :: precond_spec
    TYPE(OPTIMPARAS_DICT), INTENT(OUT)      :: optim_paras
    TYPE(AMBI_DICT), INTENT(OUT)            :: ambi_spec
    TYPE(EDU_DICT), INTENT(OUT)             :: edu_spec

    REAL(our_dble), INTENT(OUT)     :: tau

    INTEGER(our_int), INTENT(OUT)   :: num_points_interp
    INTEGER(our_int), INTENT(OUT)   :: num_slaves
    INTEGER(our_int), INTENT(OUT)   :: num_procs
    INTEGER(our_int), INTENT(OUT)   :: seed_prob
    INTEGER(our_int), INTENT(OUT)   :: seed_emax
    INTEGER(our_int), INTENT(OUT)   :: num_paras
    INTEGER(our_int), INTENT(OUT)   :: seed_sim
    INTEGER(our_int), INTENT(OUT)   :: num_free
    INTEGER(our_int), INTENT(OUT)   :: num_rows
    INTEGER(our_int), INTENT(OUT)   :: maxfun

    CHARACTER(225), INTENT(OUT)     :: optimizer_used
    CHARACTER(225), INTENT(OUT)     :: file_sim
    CHARACTER(225), INTENT(OUT)     :: exec_dir
    CHARACTER(10), INTENT(OUT)      :: request

    LOGICAL, INTENT(OUT)            :: is_interpolated
    LOGICAL, INTENT(OUT)            :: is_myopic
    LOGICAL, INTENT(OUT)            :: is_debug

    TYPE(OPTIMIZER_COLLECTION), INTENT(OUT)  :: optimizer_options

    !/* internal objects        */

    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Fix formatting
    1500 FORMAT(100(1x,i10))
    1510 FORMAT(100(1x,f25.15))

    ! Read model specification
    OPEN(UNIT=99, FILE='.model.resfort.ini', ACTION='READ')

        READ(99, 1500) num_paras
        READ(99, 1500) num_types
        READ(99, 1500) num_edu_start

        ALLOCATE(optim_paras%type_shifts(num_types, 4))
        ALLOCATE(optim_paras%type_shares(num_types * 2))
        ALLOCATE(optim_paras%paras_bounds(2, num_paras))
        ALLOCATE(optim_paras%paras_fixed(num_paras))

        ALLOCATE(edu_spec%start(num_edu_start))
        ALLOCATE(edu_spec%share(num_edu_start))

        ! BASICS
        READ(99, 1500) num_periods
        READ(99, 1510) optim_paras%delta

        ! COMMON
        READ(99, 1510) optim_paras%coeffs_common

        ! WORK
        READ(99, 1510) optim_paras%coeffs_a
        READ(99, 1510) optim_paras%coeffs_b

        ! EDUCATION
        READ(99, 1510) optim_paras%coeffs_edu
        READ(99, 1500) edu_spec%start
        READ(99, 1510) edu_spec%share
        READ(99, 1500) edu_spec%max

        ! HOME
        READ(99, 1510) optim_paras%coeffs_home

        ! SHOCKS
        DO j = 1, 4
            READ(99, 1510) (optim_paras%shocks_cholesky(j, k), k = 1, 4)
        END DO

        ! SOLUTION
        READ(99, 1500) num_draws_emax
        READ(99, 1500) seed_emax

        ! AMBIGUITY
        READ(99, *) ambi_spec%measure
        READ(99, *) ambi_spec%mean
        READ(99, 1510) optim_paras%level

        ! TYPES
        READ(99, 1510) optim_paras%type_shares
        DO j = 1, num_types
            READ(99, 1510) (optim_paras%type_shifts(j, k), k = 1, 4)
        END DO

        ! PROGRAM
        READ(99, *) is_debug
        READ(99, 1500) num_procs

        ! INTERPOLATION
        READ(99, *) is_interpolated
        READ(99, 1500) num_points_interp

        ! ESTIMATION
        READ(99, 1500) maxfun
        READ(99, 1500) num_agents_est
        READ(99, 1500) num_draws_prob
        READ(99, 1500) seed_prob
        READ(99, 1510) tau
        READ(99, 1500) num_rows

        ! SCALING
        READ(99, *) precond_spec%type
        READ(99, *) precond_spec%minimum
        READ(99, 1510) precond_spec%eps

        ! SIMULATION
        READ(99, 1500) num_agents_sim
        READ(99, 1500) seed_sim
        READ(99, *) file_sim

        ! AUXILIARY
        READ(99, *) is_myopic
        READ(99, *) optim_paras%paras_fixed

        ! REQUUEST
        READ(99, *) request

        ! EXECUTABLES
        READ(99, *) exec_dir

        ! OPTIMIZERS
        READ(99, *) optimizer_used

        READ(99, 1500) optimizer_options%newuoa%npt
        READ(99, 1500) optimizer_options%newuoa%maxfun
        READ(99, 1510) optimizer_options%newuoa%rhobeg
        READ(99, 1510) optimizer_options%newuoa%rhoend

        READ(99, 1500) optimizer_options%bobyqa%npt
        READ(99, 1500) optimizer_options%bobyqa%maxfun
        READ(99, 1510) optimizer_options%bobyqa%rhobeg
        READ(99, 1510) optimizer_options%bobyqa%rhoend

        READ(99, 1510) optimizer_options%bfgs%gtol
        READ(99, 1510) optimizer_options%bfgs%stpmx
        READ(99, 1500) optimizer_options%bfgs%maxiter
        READ(99, 1510) optimizer_options%bfgs%eps

        READ(99, 1510) optimizer_options%slsqp%ftol
        READ(99, 1500) optimizer_options%slsqp%maxiter
        READ(99, 1510) optimizer_options%slsqp%eps

        READ(99, 1510) optim_paras%paras_bounds(1, :)
        READ(99, 1510) optim_paras%paras_bounds(2, :)

    CLOSE(99)

    DO i = 1, num_paras
        IF(optim_paras%paras_bounds(1, i) == -MISSING_FLOAT) optim_paras%paras_bounds(1, i) = - HUGE_FLOAT
        IF(optim_paras%paras_bounds(2, i) == MISSING_FLOAT) optim_paras%paras_bounds(2, i) = HUGE_FLOAT
    END DO

    ALLOCATE(x_optim_bounds_free_scaled(2, COUNT(.NOT. optim_paras%paras_fixed)))
    ALLOCATE(x_optim_bounds_free_unscaled(2, COUNT(.NOT. optim_paras%paras_fixed)))

    k = 1
    DO i = 1, num_paras
        IF (.NOT. optim_paras%paras_fixed(i)) THEN
            DO j = 1, 2
                x_optim_bounds_free_unscaled(j, k) = optim_paras%paras_bounds(j, i)
            END DO
            k = k + 1
        END IF
    END DO

    ! Constructed attributes
    IF (ambi_spec%mean) THEN
        num_free_ambi = 2
    ELSE
        num_free_ambi = 4
    END IF

    num_free =  COUNT(.NOT. optim_paras%paras_fixed)
    num_slaves = num_procs - 1
    min_idx = edu_spec%max + 1

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE read_dataset(data_est, num_rows)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)  :: data_est(:, :)

    INTEGER(our_int), INTENT(IN)                :: num_rows

    !/* internal objects        */

    INTEGER(our_int)                            :: j
    INTEGER(our_int)                            :: k

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Allocate data container
    ALLOCATE(data_est(num_rows, 8))

    ! Read observed data to double precision array
    OPEN(UNIT=99, FILE='.data.resfort.dat', ACTION='READ')

        DO j = 1, num_rows
            READ(99, *) (data_est(j, k), k = 1, 8)
        END DO

    CLOSE(99)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE clip_value_1(clip_value, value, lower_bound, upper_bound, info)

    !/* external objects        */

    INTEGER(our_int), INTENT(OUT)   :: info

    REAL(our_dble), INTENT(OUT)     :: clip_value

    REAL(our_dble), INTENT(IN)      :: lower_bound
    REAL(our_dble), INTENT(IN)      :: upper_bound
    REAL(our_dble), INTENT(IN)      :: value

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    info = 0

    IF(value < lower_bound) THEN

        clip_value = lower_bound
        info = 1

    ELSEIF(value > upper_bound) THEN

        clip_value = upper_bound
        info = 2

    ELSE

        clip_value = value

    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE clip_value_2(clip_value, value, lower_bound, upper_bound, infos)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(OUT)   :: infos(:)

    REAL(our_dble), INTENT(OUT)     :: clip_value(:)

    REAL(our_dble), INTENT(IN)      :: lower_bound
    REAL(our_dble), INTENT(IN)      :: upper_bound
    REAL(our_dble), INTENT(IN)      :: value(:)

    !/*  internal objects       */

    INTEGER(our_int)                :: num_values
    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    num_values = SIZE(value)

    ! In this setup the same container can be used for multiple calls with
    ! different input arguments. Usually INFOS is immediately processed anyway
    ! and does not need to be available for multiple calls.
    IF (ALLOCATED(infos)) DEALLOCATE(infos)

    ALLOCATE(infos(num_values))


    infos = 0

    DO i = 1, num_values

        IF(value(i) < lower_bound) THEN

            clip_value(i) = lower_bound
            infos(i) = 1

        ELSEIF(value(i) > upper_bound) THEN
            clip_value(i) = upper_bound
            infos(i) = 2

        ELSE

            clip_value(i) = value(i)

        END IF

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE dist_optim_paras(optim_paras, x, info)

    !/* external objects        */

    TYPE(OPTIMPARAS_DICT), INTENT(INOUT)  :: optim_paras

    REAL(our_dble), INTENT(IN)      :: x(num_paras)

    INTEGER(our_int), OPTIONAL, INTENT(OUT)   :: info

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Extract model ingredients
    optim_paras%delta = MAX(x(1:1), zero_dble)

    optim_paras%level = MAX(x(2:2), zero_dble)

    optim_paras%coeffs_common = x(3:4)

    optim_paras%coeffs_a = x(5:19)

    optim_paras%coeffs_b = x(20:34)

    optim_paras%coeffs_edu = x(35:41)

    optim_paras%coeffs_home = x(42:44)

    ! The information pertains to the stabilization of an otherwise zero variance.
    IF (PRESENT(info)) THEN
        CALL extract_cholesky(optim_paras%shocks_cholesky, x, info)
    ELSE
        CALL extract_cholesky(optim_paras%shocks_cholesky, x)
    END IF

    optim_paras%type_shares = zero_dble
    optim_paras%type_shares(3:) = x(55:55 + (num_types - 1) * 2 - 1)

    optim_paras%type_shifts = zero_dble
    optim_paras%type_shifts(2:, :) =  TRANSPOSE(RESHAPE(x(55 + (num_types - 1) * 2:num_paras), (/4, num_types  - 1/)))

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION get_num_obs_agent(data_array) RESULT(num_obs_agent)

    !/* external objects        */

    INTEGER(our_int)                    :: num_obs_agent(num_agents_est)

    REAL(our_dble) , INTENT(IN)         :: data_array(:, :)

    !/* internal objects        */

    INTEGER(our_int)                    :: num_rows
    INTEGER(our_int)                    :: q
    INTEGER(our_int)                    :: i

    REAL(our_dble)                      :: agent_number

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    num_rows = SIZE(data_array, 1)
    num_obs_agent = zero_int
    q = 1

    agent_number = data_array(1, 1)

    DO i = 1, num_rows
        IF (data_array(i, 1) .NE. agent_number) THEN
            q = q + 1
            agent_number = data_array(i, 1)
        END IF

        num_obs_agent(q) = num_obs_agent(q) + 1

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE get_optim_paras(x, optim_paras, is_all)

    !/* external objects        */

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    REAL(our_dble), INTENT(OUT)     :: x(:)

    LOGICAL, INTENT(IN)             :: is_all

    !/* internal objects        */

    REAL(our_dble)                  :: x_internal(num_paras)
    REAL(our_dble)                  :: shifts(num_types * 4)

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    x_internal(1:1) = optim_paras%delta

    x_internal(2:2) = optim_paras%level

    x_internal(3:4) = optim_paras%coeffs_common(:)

    x_internal(5:19) = optim_paras%coeffs_a(:)

    x_internal(20:34) = optim_paras%coeffs_b(:)

    x_internal(35:41) = optim_paras%coeffs_edu(:)

    x_internal(42:44) = optim_paras%coeffs_home(:)

    x_internal(45:45) = optim_paras%shocks_cholesky(1, :1)

    x_internal(46:47) = optim_paras%shocks_cholesky(2, :2)

    x_internal(48:50) = optim_paras%shocks_cholesky(3, :3)

    x_internal(51:54) = optim_paras%shocks_cholesky(4, :4)

    x_internal(55:(55 + (num_types - 1) * 2) - 1) = optim_paras%type_shares(3:)

    shifts = PACK(TRANSPOSE(optim_paras%type_shifts), .TRUE.)
    x_internal(55 + (num_types - 1) * 2:num_paras) = shifts(5:)

    ! Sometimes it is useful to return all parameters instead of just those freed for the estimation.
    IF(is_all) THEN

        x = x_internal

    ELSE

        j = 1

        DO i = 1, num_paras

            IF(optim_paras%paras_fixed(i)) CYCLE

            x(j) = x_internal(i)

            j = j + 1

        END DO

    END IF

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION integer_to_boolean(input) RESULT(output)

    !/* external objects    */

    INTEGER(our_int), INTENT(IN)                :: input

    LOGICAL(our_dble)                           :: output

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF (input .EQ. one_int) THEN
        output = .TRUE.
    ELSEIF (input .EQ. zero_int) THEN
        output = .FALSE.
    ELSE
        STOP 'Misspecified request'
    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION float_to_boolean(input) RESULT(output)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)                  :: input

    LOGICAL(our_dble)                           :: output

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF (input .EQ. one_dble) THEN
        output = .TRUE.
    ELSEIF (input .EQ. zero_dble) THEN
        output = .FALSE.
    ELSE
        STOP 'Misspecified request'
    END IF

END FUNCTION
!*******************************************************************************
!*******************************************************************************
FUNCTION back_out_systematic_wages(rewards_systematic, exp_a, exp_b, edu, activity_lagged, optim_paras) RESULT(wages_systematic)

    !/* external objects        */

    REAL(our_dble)                              :: wages_systematic(2)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)           :: optim_paras

    INTEGER(our_int), INTENT(IN)                :: activity_lagged
    INTEGER(our_int), INTENT(IN)                :: exp_a
    INTEGER(our_int), INTENT(IN)                :: exp_b
    INTEGER(our_int), INTENT(IN)                :: edu

    REAL(our_dble), INTENT(IN)                  :: rewards_systematic(4)

    !/* internal objects        */

    TYPE(COVARIATES_DICT)                       :: covariates

    INTEGER(our_int)                            :: covars_general(3)
    INTEGER(our_int)                            :: covars_common(2)
    INTEGER(our_int)                            :: i

    REAL(our_dble)                              :: rewards_common
    REAL(our_dble)                              :: general(2)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    covariates = construct_covariates(exp_a, exp_b, edu, activity_lagged, MISSING_INT, MISSING_INT)

    covars_general = (/ one_int, covariates%not_exp_a_lagged, covariates%not_any_exp_a /)
    general(1) = DOT_PRODUCT(covars_general, optim_paras%coeffs_a(13:15))

    covars_general = (/ one_int, covariates%not_exp_b_lagged, covariates%not_any_exp_b /)
    general(2) = DOT_PRODUCT(covars_general, optim_paras%coeffs_b(13:15))

    ! Second we do the same with the common component.
    covars_common = (/ covariates%hs_graduate, covariates%co_graduate /)
    rewards_common = DOT_PRODUCT(optim_paras%coeffs_common, covars_common)

    DO i = 1, 2
        wages_systematic(i) = rewards_systematic(i) - general(i) - rewards_common
    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION construct_covariates(exp_a, exp_b, edu, activity_lagged, type_, period) RESULT(covariates)

    !/* external objects        */

    TYPE(COVARIATES_DICT)           :: covariates

    INTEGER(our_int), INTENT(IN)    :: activity_lagged
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: type_
    INTEGER(our_int), INTENT(IN)    :: exp_a
    INTEGER(our_int), INTENT(IN)    :: exp_b
    INTEGER(our_int), INTENT(IN)    :: edu

    !/* internal objects        */

    INTEGER(our_int)                :: hs_graduate
    INTEGER(our_int)                :: edu_lagged

    LOGICAL                         :: cond

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Auxiliary objects
    edu_lagged = TRANSFER(activity_lagged .EQ. one_int, our_int)
    hs_graduate = TRANSFER(edu .GE. 12, our_int)

    ! These are covariates that are supposed to capture the entry costs.
    cond = ((exp_a .GT. 0) .AND. activity_lagged .NE. two_int)
    covariates%not_exp_a_lagged = TRANSFER(cond, our_int)

    cond = ((exp_b .GT. 0) .AND. activity_lagged .NE. three_int)
    covariates%not_exp_b_lagged = TRANSFER(cond, our_int)

    covariates%work_a_lagged = TRANSFER(activity_lagged .EQ. two_int, our_int)
    covariates%work_b_lagged = TRANSFER(activity_lagged .EQ. three_int, our_int)
    covariates%edu_lagged = TRANSFER(activity_lagged .EQ. one_int, our_int)
    covariates%not_any_exp_a = TRANSFER(exp_a .EQ. 0, our_int)
    covariates%not_any_exp_b = TRANSFER(exp_b .EQ. 0, our_int)
    covariates%any_exp_a = TRANSFER(exp_a .GT. 0, our_int)
    covariates%any_exp_b = TRANSFER(exp_b .GT. 0, our_int)

    covariates%is_minor = TRANSFER(period .LT. 3, our_int)
    covariates%is_young_adult = TRANSFER(((period .GE. 3) .AND. (period .LT. 6)), our_int)
    covariates%is_adult = TRANSFER(period .GE. 6, our_int)

    covariates%co_graduate = TRANSFER(edu .GE. 16, our_int)
    covariates%hs_graduate = hs_graduate

    covariates%is_return_not_high_school = TRANSFER((.NOT. to_boolean(edu_lagged)) .AND. (.NOT. to_boolean(hs_graduate)), our_int)

    ! STRUCT_AMBIGUOUS: No distinction between reenrollment costs.
    covariates%is_return_high_school = covariates%is_return_not_high_school

    covariates%activity_lagged = activity_lagged
    covariates%period = period
    covariates%exp_a = exp_a
    covariates%exp_b = exp_b
    covariates%type = type_
    covariates%edu = edu

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION calculate_rewards_general(covariates, optim_paras) RESULT(rewards_general)

    !/* external objects        */

    REAL(our_dble)                      :: rewards_general(2)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras
    TYPE(COVARIATES_DICT), INTENT(IN)   :: covariates

    !/* internal objects        */

    REAL(our_dble)                      :: covars_general(3)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    covars_general = (/ one_int, covariates%not_exp_a_lagged, covariates%not_any_exp_a /)
    rewards_general(1) = DOT_PRODUCT(covars_general, optim_paras%coeffs_a(13:15))

    covars_general = (/ one_int, covariates%not_exp_b_lagged, covariates%not_any_exp_b /)
    rewards_general(2) = DOT_PRODUCT(covars_general, optim_paras%coeffs_b(13:15))

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION calculate_rewards_common(covariates, optim_paras) RESULT(rewards_common)

    !/* external objects        */

    REAL(our_dble)                      :: rewards_common

    TYPE(COVARIATES_DICT), INTENT(IN)   :: covariates
    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    !/* internal objects        */

    REAL(our_dble)                      :: covars_common(2)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    covars_common = (/ covariates%hs_graduate, covariates%co_graduate /)
    rewards_common = DOT_PRODUCT(optim_paras%coeffs_common, covars_common)

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION sort_edu_spec(edu_spec) RESULT(edu_spec_sorted)

    !/* external objects        */

    TYPE(EDU_DICT)                  :: edu_spec_sorted

    TYPE(EDU_DICT), INTENT(IN)      :: edu_spec


    !/* internal objects        */

    REAL(our_dble), ALLOCATABLE     :: edu_start_dble(:)

    INTEGER(our_int)                :: num_elements
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Auxiliary information
    num_elements = SIZE(edu_spec%start, 1)

    ! Allocate new dictionary with sorted elements and auxiliary object.
    ALLOCATE(edu_spec_sorted%start(num_elements))
    ALLOCATE(edu_spec_sorted%share(num_elements))
    ALLOCATE(edu_start_dble(num_elements))

    ! We need to put in some extra work to ensure the proper type conversions.
    edu_start_dble = DBLE(edu_spec%start)
    edu_start_dble = sorted(edu_start_dble, num_elements)
    edu_spec_sorted%start = INT(edu_start_dble)

    edu_spec_sorted%max = edu_spec%max
    edu_spec_sorted%share = -HUGE_FLOAT

    DO i = 1, num_elements

        DO j = 1, num_elements

            IF (edu_spec_sorted%start(i) .NE. edu_spec%start(j))  CYCLE

            edu_spec_sorted%share(i) = edu_spec%share(j)

        END DO

    END DO


END FUNCTION
!******************************************************************************
!******************************************************************************
END MODULE
