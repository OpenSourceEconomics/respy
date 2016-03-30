!*******************************************************************************
!*******************************************************************************
MODULE shared_auxiliary

	!/*	external modules	*/

    USE robufort_constants

	!/*	setup	*/

	IMPLICIT NONE

    PUBLIC

 CONTAINS
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
SUBROUTINE create_draws(draws, num_periods, num_draws_emax, seed, is_debug, &
                which, shocks_cholesky, is_ambiguous)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)  :: draws(:, :, :)

    INTEGER(our_int), INTENT(IN)                :: num_draws_emax
    INTEGER(our_int), INTENT(IN)                :: num_periods
    INTEGER(our_int), INTENT(IN)                :: seed

    REAL(our_dble), INTENT(IN)                  :: shocks_cholesky(4, 4)

    LOGICAL, INTENT(IN)                         :: is_ambiguous
    LOGICAL, INTENT(IN)                         :: is_debug

    CHARACTER(4)                                :: which

    !/* internal objects        */

    INTEGER(our_int)                            :: seed_inflated(15)
    INTEGER(our_int)                            :: seed_size
    INTEGER(our_int)                            :: period
    INTEGER(our_int)                            :: j
    INTEGER(our_int)                            :: i

    LOGICAL                                     :: READ_IN

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Allocate containers
    ALLOCATE(draws(num_periods, num_draws_emax, 4))

    ! Set random seed
    seed_inflated(:) = seed

    CALL RANDOM_SEED(size=seed_size)

    CALL RANDOM_SEED(put=seed_inflated)

    ! Draw random deviates from a standard normal distribution or read it in
    ! from disk. The latter is available to allow for testing across
    ! implementations.
    INQUIRE(FILE='draws.txt', EXIST=READ_IN)

    IF ((READ_IN .EQV. .True.)  .AND. (is_debug .EQV. .True.)) THEN

        OPEN(12, file='draws.txt')

        DO period = 1, num_periods

            DO j = 1, num_draws_emax

                2000 FORMAT(4(1x,f15.10))
                READ(12,2000) draws(period, j, :)

            END DO

        END DO

        CLOSE(12)

    ELSE

        DO period = 1, num_periods

            CALL multivariate_normal(draws(period, :, :))

        END DO

    END IF
    ! Standard normal deviates used for the Monte Carlo integration of the
    ! expected future values in the solution step. Also, standard normal
    ! deviates for the Monte Carlo integration of the choice probabilities in
    ! the evaluation step.
    IF ((which == 'emax') .OR. (which == 'prob')) THEN

        draws = draws

    ! Deviates for the simulation of a synthetic agent population.
    ELSE IF (which == 'sims') THEN
        ! Standard deviates transformed to the distributions relevant for
        ! the agents actual decision making as traversing the tree.

        ! Transformations
        DO period = 1, num_periods

            ! Apply variance change
            DO i = 1, num_draws_emax
                draws(period, i:i, :) = &
                    TRANSPOSE(MATMUL(shocks_cholesky, TRANSPOSE(draws(period, i:i, :))))
            END DO

            DO j = 1, 2
                draws(period, :, j) =  EXP(draws(period, :, j))
            END DO

        END DO

    END IF

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
END MODULE