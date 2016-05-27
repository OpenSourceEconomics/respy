!*******************************************************************************
!*******************************************************************************
MODULE shared_auxiliary

	!/*	external modules	*/

    USE shared_constants

	!/*	setup	*/

    IMPLICIT NONE

    PUBLIC

    !/* explicit interfaces   */

    INTERFACE clip_value

        MODULE PROCEDURE clip_value_1, clip_value_2

    END INTERFACE


    INTERFACE

        SUBROUTINE DPOTRF(UPLO, N, A, LDA, INFO)

            !/* external modules    */

            USE shared_constants

            !/* external objects    */

            REAL(our_dble), INTENT(INOUT)   :: A(N, N)
 
            INTEGER(our_int), INTENT(IN)    :: INFO
            INTEGER(our_int), INTENT(IN)    :: LDA
            INTEGER(our_int), INTENT(IN)    :: N

            CHARACTER(1), INTENT(IN)        :: UPLO

        END SUBROUTINE

    END INTERFACE


    INTERFACE

        SUBROUTINE DGETRF(M, N, A, LDA, IPIV, INFO)

            !/* external modules    */

            USE shared_constants

            !/* external objects    */

            INTEGER(our_int), INTENT(OUT)   :: IPIV(N)
            INTEGER(our_int), INTENT(OUT)   :: INFO

            REAL(our_dble), INTENT(INOUT)   :: A(N, N)

            INTEGER(our_int), INTENT(IN)    :: LDA
            INTEGER(our_int), INTENT(IN)    :: M
            INTEGER(our_int), INTENT(IN)    :: N

         END SUBROUTINE

    END INTERFACE


    INTERFACE

        SUBROUTINE DGETRI(N, A, LDA, IPIV, WORK, LWORK, INFO)
            
            !/* external modules    */

            USE shared_constants

            !/* external objects    */

            REAL(our_dble), INTENT(INOUT)   :: WORK(LWORK)
            REAL(our_dble), INTENT(INOUT)   :: A(N, N)

            INTEGER(our_int), INTENT(IN)    :: IPIV(N)
            INTEGER(our_int), INTENT(IN)    :: LWORK
            INTEGER(our_int), INTENT(IN)    :: INFO
            INTEGER(our_int), INTENT(IN)    :: LDA
            INTEGER(our_int), INTENT(IN)    :: N

         END SUBROUTINE

    END INTERFACE


 CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE transform_disturbances(draws_transformed, draws, shocks_cholesky, & 
                num_draws)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: draws_transformed(:, :)

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)      :: draws(:, :)

    INTEGER, INTENT(IN)             :: num_draws

    !/* internal objects        */

    INTEGER(our_int)                :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    DO i = 1, num_draws
        draws_transformed(i:i, :) = &
            TRANSPOSE(MATMUL(shocks_cholesky, TRANSPOSE(draws(i:i, :))))
    END DO

    DO i = 1, 2
        draws_transformed(:, i) = clip_value(EXP(draws_transformed(:, i)), zero_dble, HUGE_FLOAT)
    END DO


END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_total_value(total_payoffs, period, &
                payoffs_systematic, draws, &
                mapping_state_idx, periods_emax, k, states_all)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: total_payoffs(:)

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: payoffs_systematic(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: draws(:)

    !/* internal objects        */

    REAL(our_dble)                  :: payoffs_future(4)
    REAL(our_dble)                  :: payoffs_ex_post(4)

    LOGICAL                         :: is_inadmissible

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
        CALL get_future_payoffs(payoffs_future, is_inadmissible, & 
                mapping_state_idx, period, periods_emax, k, & 
                states_all)
    ELSE
        is_inadmissible = .False.
        payoffs_future = zero_dble
    END IF

    ! Calculate total utilities
    total_payoffs = payoffs_ex_post + delta * payoffs_future

    ! This is required to ensure that the agent does not choose any
    ! inadmissible states. If the state is inadmissible payoffs_future takes 
    ! value zero. This aligns the treatment of inadmissible values with the 
    ! original paper.
    IF (is_inadmissible) THEN
        total_payoffs(3) = total_payoffs(3) + INADMISSIBILITY_PENALTY
    END IF

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_future_payoffs(payoffs_future, is_inadmissible, &
                mapping_state_idx, period, periods_emax, k, states_all)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: payoffs_future(:)

    LOGICAL, INTENT(OUT)            :: is_inadmissible

    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_all(:, :, :)
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: k

    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)



    !/* internals objects       */

    INTEGER(our_int)                :: edu_lagged
    INTEGER(our_int)                :: future_idx
    INTEGER(our_int)                :: exp_a
    INTEGER(our_int)                :: exp_b
    INTEGER(our_int)                :: edu

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
    is_inadmissible = (edu .GE. edu_max - edu_start)
    IF(is_inadmissible) THEN
        payoffs_future(3) = zero_dble
    ELSE
        future_idx = mapping_state_idx(period + 1 + 1, exp_a + 1, &
                        exp_b + 1, edu + 1 + 1, 2)
        payoffs_future(3) = periods_emax(period + 1 + 1, future_idx + 1)
    END IF

	! Staying at home
    future_idx = mapping_state_idx(period + 1 + 1, exp_a + 1, &
                    exp_b + 1, edu + 1, 1)
    payoffs_future(4) = periods_emax(period + 1 + 1, future_idx + 1)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE create_draws(draws, num_draws, seed)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)  :: draws(:, :, :)

    INTEGER(our_int), INTENT(IN)                :: num_draws 
    INTEGER(our_int), INTENT(IN)                :: seed

    !/* internal objects        */

    INTEGER(our_int)                            :: seed_inflated(15)
    INTEGER(our_int)                            :: seed_size
    INTEGER(our_int)                            :: period
    INTEGER(our_int)                            :: j
    INTEGER(our_int)                            :: i

    REAL(our_dble)                              :: deviates(4)

    LOGICAL                                     :: READ_IN

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Allocate containers
    ALLOCATE(draws(num_periods, num_draws, 4))

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

            DO j = 1, num_draws

                2000 FORMAT(4(1x,f15.10))
                READ(12,2000) draws(period, j, :)

            END DO

        END DO

        CLOSE(12)

    ELSE

        DO period = 1, num_periods

            DO i = 1, num_draws

               CALL standard_normal(deviates)

               draws(period, i, :) = deviates

            END DO

        END DO

    END IF
   
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
FUNCTION inverse(A, n)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: n

    REAL(our_dble), INTENT(IN)      :: A(:, :)

    !/* internal objects        */

    INTEGER(our_int)                :: ipiv(n)
    INTEGER(our_int)                :: info

    REAL(our_dble)                  :: inverse(n, n)
    REAL(our_dble)                  :: work(n)
    
!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
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
!*******************************************************************************
!*******************************************************************************
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

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
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
!*******************************************************************************
!*******************************************************************************
SUBROUTINE store_results(mapping_state_idx, states_all, &
                periods_payoffs_systematic, states_number_period, &
                periods_emax, crit_val)

    !/* external objects        */


    INTEGER(our_int), INTENT(IN)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), INTENT(IN)    :: states_number_period(:)
    INTEGER(our_int), INTENT(IN)    :: states_all(:,:,:)

    REAL(our_dble), INTENT(IN)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)
    REAL(our_dble), INTENT(IN)      :: crit_val

    !/* internal objects        */

    INTEGER(our_int)                :: max_states_period
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! This is a break in design as otherwise I need to carry the integer up
    ! from the solution level.
    max_states_period = MAXVAL(states_number_period)

    IF (request == 'solve') THEN

        ! Write out results for the store results.
        1800 FORMAT(5(1x,i5))

        OPEN(UNIT=1, FILE='.mapping_state_idx.resfort.dat')

        DO period = 1, num_periods
            DO i = 1, num_periods
                DO j = 1, num_periods
                    DO k = 1, min_idx
                        WRITE(1, 1800) mapping_state_idx(period, i, j, k, :)
                    END DO
                END DO
            END DO
        END DO

        CLOSE(1)


        2000 FORMAT(4(1x,i5))

        OPEN(UNIT=1, FILE='.states_all.resfort.dat')

        DO period = 1, num_periods
            DO i = 1, max_states_period
                WRITE(1, 2000) states_all(period, i, :)
            END DO
        END DO

        CLOSE(1)


        1900 FORMAT(4(1x,f25.15))

        OPEN(UNIT=1, FILE='.periods_payoffs_systematic.resfort.dat')

        DO period = 1, num_periods
            DO i = 1, max_states_period
                WRITE(1, 1900) periods_payoffs_systematic(period, i, :)
            END DO
        END DO

        CLOSE(1)

        2100 FORMAT(i5)

        OPEN(UNIT=1, FILE='.states_number_period.resfort.dat')

        DO period = 1, num_periods
            WRITE(1, 2100) states_number_period(period)
        END DO

        CLOSE(1)


        2200 FORMAT(i5)

        OPEN(UNIT=1, FILE='.max_states_period.resfort.dat')

        WRITE(1, 2200) max_states_period

        CLOSE(1)


        2400 FORMAT(100000(1x,f25.15))

        OPEN(UNIT=1, FILE='.periods_emax.resfort.dat')

        DO period = 1, num_periods
            WRITE(1, 2400) periods_emax(period, :)
        END DO

        CLOSE(1)

    ! Write out value of criterion function if evaluated.
    ELSEIF (request == 'evaluate') THEN

        2500 FORMAT(1x,f25.15)

        OPEN(UNIT=1, FILE='.eval.resfort.dat')

        WRITE(1, 2500) crit_val

        CLOSE(1)

    END IF

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE read_specification(coeffs_a, coeffs_b, &
                coeffs_edu, coeffs_home, shocks_cholesky, & 
                num_points_interp)

    !
    !   This function serves as the replacement for the RespyCls and reads in
    !   all required information about the model parameterization. It just
    !   reads in all required information.
    !

    !/* external objects        */

    INTEGER(our_int), INTENT(OUT)   :: num_points_interp

    REAL(our_dble), INTENT(OUT)     :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(OUT)     :: coeffs_home(1)
    REAL(our_dble), INTENT(OUT)     :: coeffs_edu(3)
    REAL(our_dble), INTENT(OUT)     :: coeffs_a(6)
    REAL(our_dble), INTENT(OUT)     :: coeffs_b(6)
    
    !/* internal objects        */

    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Fix formatting
    1500 FORMAT(6(1x,f15.10))
    1510 FORMAT(f15.10)
    1520 FORMAT(6(1x,f20.10))

    1505 FORMAT(i10)
    1515 FORMAT(i10,1x,i10)

    ! Read model specification
    OPEN(UNIT=1, FILE='.model.resfort.ini')

        ! BASICS
        READ(1, 1505) num_periods
        READ(1, 1510) delta

        ! WORK
        READ(1, 1500) coeffs_a
        READ(1, 1500) coeffs_b

        ! EDUCATION
        READ(1, 1500) coeffs_edu
        READ(1, 1515) edu_start, edu_max

        ! HOME
        READ(1, 1500) coeffs_home

        ! SHOCKS
        DO j = 1, 4
            READ(1, 1520) (shocks_cholesky(j, k), k=1, 4)
        END DO

        ! SOLUTION
        READ(1, 1505) num_draws_emax
        READ(1, 1505) seed_emax

        ! PROGRAM
        READ(1, *) is_debug
        READ(1, 1505) num_procs

        ! INTERPOLATION
        READ(1, *) is_interpolated
        READ(1, 1505) num_points_interp

        ! ESTIMATION
        READ(1, 1505) num_agents_est
        READ(1, 1505) num_draws_prob
        READ(1, 1505) seed_prob
        READ(1, 1510) tau

        ! AUXILIARY
        READ(1, 1505) min_idx
        READ(1, *) is_myopic

        ! REQUUEST
        READ(1, *) request

        ! EXECUTABLES
        READ(1, *) exec_dir

    CLOSE(1)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE read_dataset(data_array, num_agents)

    !/* external objects        */

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)  :: data_array(:, :)

    INTEGER(our_int), INTENT(IN)                :: num_agents

    !/* internal objects        */

    INTEGER(our_int)                            :: j
    INTEGER(our_int)                            :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Allocate data container
    ALLOCATE(data_array(num_periods * num_agents, 8))

    ! Read observed data to double precision array
    OPEN(UNIT=1, FILE='.data.resfort.dat')

        DO j = 1, num_periods * num_agents
            READ(1, *) (data_array(j, k), k = 1, 8)
        END DO

    CLOSE(1, STATUS='delete')

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION clip_value_1(value, lower_bound, upper_bound)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)  :: lower_bound
    REAL(our_dble), INTENT(IN)  :: upper_bound
    REAL(our_dble), INTENT(IN)  :: value

    !/*  internal objects       */

    REAL(our_dble)              :: clip_value_1

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF(value < lower_bound) THEN

        clip_value_1 = lower_bound

    ELSEIF(value > upper_bound) THEN

        clip_value_1 = upper_bound

    ELSE

        clip_value_1 = value

    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION clip_value_2(value, lower_bound, upper_bound)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)  :: lower_bound
    REAL(our_dble), INTENT(IN)  :: upper_bound
    REAL(our_dble), INTENT(IN)  :: value(:)

    !/*  internal objects       */

    REAL(our_dble), ALLOCATABLE :: clip_value_2(:)
    
    INTEGER(our_int)            :: num_values
    INTEGER(our_int)            :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------


    num_values = SIZE(value, 1)
    
    ALLOCATE(clip_value_2(num_values))

    DO i = 1, num_values

        IF(value(i) < lower_bound) THEN

            clip_value_2(i) = lower_bound

        ELSEIF(value(i) > upper_bound) THEN

            clip_value_2(i) = upper_bound

        ELSE

            clip_value_2(i) = value(i)

        END IF
    
    END DO

END FUNCTION
!*******************************************************************************
!*******************************************************************************
END MODULE