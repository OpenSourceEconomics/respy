!*******************************************************************************
!******************************************************************************* 
MODULE master_auxiliary

    !/* external modules    */

    USE shared_auxiliary

    USE shared_constants

    USE solve_auxiliary

    USE mpi

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE fort_solve_parallel(periods_payoffs_systematic, states_number_period, &
                mapping_state_idx, periods_emax, states_all, coeffs_a, &
                coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, &
                is_interpolated, num_draws_emax, & 
                num_periods, num_points, edu_start, is_myopic, is_debug, & 
                edu_max, min_idx, delta, num_slaves, SLAVECOMM)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_all(:, :, :)

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(:, :)

    INTEGER(our_int), INTENT(IN)                    :: num_draws_emax
    INTEGER(our_int), INTENT(IN)                    :: num_periods
    INTEGER(our_int), INTENT(IN)                    :: num_points
    INTEGER(our_int), INTENT(IN)                    :: edu_start
    INTEGER(our_int), INTENT(IN)                    :: edu_max
    INTEGER(our_int), INTENT(IN)                    :: min_idx

    REAL(our_dble), INTENT(IN)                      :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(IN)                      :: coeffs_home(:)
    REAL(our_dble), INTENT(IN)                      :: coeffs_edu(:)
    REAL(our_dble), INTENT(IN)                      :: coeffs_a(:)
    REAL(our_dble), INTENT(IN)                      :: coeffs_b(:)
    REAL(our_dble), INTENT(IN)                      :: delta

    LOGICAL, INTENT(IN)                             :: is_interpolated
    LOGICAL, INTENT(IN)                             :: is_myopic
    LOGICAL, INTENT(IN)                             :: is_debug

    !NEw external
    INTEGER(our_int), INTENT(IN)                    :: num_slaves, SLAVECOMM


    !/* internal objects        */

    INTEGER(our_int), ALLOCATABLE                   :: states_all_tmp(:, :, :)

    INTEGER(our_int)                                :: max_states_period
    INTEGER(our_int)                                :: period


    !NEW internal
    INTEGER(our_int)                                :: ierr
    INTEGER(our_int)                                :: num_states, status, task

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

        ! Spawn the slaves for doing the actual work.
        CALL MPI_COMM_SPAWN('./slave', MPI_ARGV_NULL, num_slaves, MPI_INFO_NULL, & 
                0, MPI_COMM_WORLD, SLAVECOMM, MPI_ERRCODES_IGNORE, ierr)

        ! Request EMAX calculation
        task = 2
        CALL MPI_Bcast(task, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)

        !-----------------------------------------------------------------------
        ! While waiting for the slaves to finish, the master prepares himself.
        !-----------------------------------------------------------------------

        ! Allocate arrays
        ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
        ALLOCATE(states_all_tmp(num_periods, 100000, 4))
        ALLOCATE(states_number_period(num_periods))

        CALL fort_create_state_space(states_all_tmp, states_number_period, &
                    mapping_state_idx, max_states_period, num_periods, edu_start, &
                    edu_max)

        ALLOCATE(periods_emax(num_periods, max_states_period))

        ! Calculate the systematic payoffs
        ALLOCATE(states_all(num_periods, max_states_period, 4))
        states_all = states_all_tmp(:, :max_states_period, :)
        DEALLOCATE(states_all_tmp)

        ALLOCATE(periods_payoffs_systematic(num_periods, max_states_period, 4))

        CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, &
                num_periods, states_number_period, states_all, edu_start, &
                coeffs_a, coeffs_b, coeffs_edu, coeffs_home)
       
        ! The leading slave is kind enough to let the parent process know about the 
        ! intermediate outcomes.
        DO period = (num_periods - 1), 0, -1

            num_states = states_number_period(period + 1)

            CALL MPI_RECV(periods_emax(period + 1, :num_states), num_states, & 
                    MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, SLAVECOMM, status, & 
                    ierr)

        END DO

        ! Shut down orderly
        task = 1
        CALL MPI_Bcast(task, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
        CALL MPI_FINALIZE (ierr)

        ! Write out result to allow temporary testing against the scalar implementation.
        2500 FORMAT(1x,f25.15)
        OPEN(UNIT=1, FILE='.eval.resfort.dat')
        WRITE(1, 2500)  periods_emax(1, 1)
        CLOSE(1)

END SUBROUTINE

SUBROUTINE record_equality(is_equal)

LOGICAL :: is_equal

IF(.NOT. is_equal) THEN

OPEN (UNIT=5, FILE=".error.testing", STATUS="NEW")  ! Root directory
CLOSE (UNIT=5)
END IF


END SUBROUTINE

!*******************************************************************************
!*******************************************************************************
END MODULE
!*******************************************************************************
!******************************************************************************* 
PROGRAM master

    !/* external modules        */

    USE master_auxiliary    

    USE shared_constants
    
    USE shared_auxiliary

    USE solve_auxiliary

    USE solve_fortran

    USE mpi
    
    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */


    INTEGER(our_int), ALLOCATABLE   :: states_all_tmp(:, :, :)

    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx_parallel(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_number_period_parallel(:)
    INTEGER(our_int), ALLOCATABLE   :: states_all_parallel(:, :, :)

    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx_scalar(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_number_period_scalar(:)
    INTEGER(our_int), ALLOCATABLE   :: states_all_scalar(:, :, :)

    INTEGER(our_int)                :: status(MPI_STATUS_SIZE) 
    INTEGER(our_int)                :: max_states_period
    INTEGER(our_int)                :: num_draws_emax
    INTEGER(our_int)                :: num_draws_prob
    INTEGER(our_int)                :: num_agents_est
    INTEGER(our_int)                :: num_periods
    INTEGER(our_int)                :: num_slaves
    INTEGER(our_int)                :: num_points
    INTEGER(our_int)                :: num_states
    INTEGER(our_int)                :: SLAVECOMM
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: edu_start
    INTEGER(our_int)                :: edu_max
    INTEGER(our_int)                :: min_idx
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: ierr
    INTEGER(our_int)                :: task

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_systematic_parallel(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax_parallel(:, :)

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_systematic_scalar(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax_scalar(:, :)

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: crit_val
    REAL(our_dble)                  :: delta
    REAL(our_dble)                  :: tau

    LOGICAL                         :: is_interpolated
    LOGICAL                         :: is_myopic
    LOGICAL                         :: is_debug, is_equal

    CHARACTER(10)                   :: request
    CHARACTER(10)                   :: arg
    REAL(our_dble), ALLOCATABLE     :: periods_draws_emax(:, :, :)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Initialize MPI environment
    CALL MPI_INIT(ierr)

    ! Read in requested number of slaves from the command line.
    CALL GETARG(one_int, arg)
    IF (LEN_TRIM(arg) == 0) THEN
        num_slaves = 2
    ELSE
        read (arg,*) num_slaves
    END IF

    ! Read in model specification.
    CALL read_specification(num_periods, delta, coeffs_a, coeffs_b, &
            coeffs_edu, edu_start, edu_max, coeffs_home, shocks_cholesky, & 
            num_draws_emax, seed_emax, seed_prob, num_agents_est, is_debug, & 
            is_interpolated, num_points, min_idx, request, num_draws_prob, & 
            is_myopic, tau) 

    ! Create the required random draws for the EMAX calculation in case of 
    ! scalar execution.
    CALL create_draws(periods_draws_emax, num_periods, num_draws_emax, &
            seed_emax, is_debug)

    ! Solve the model for a given parametrization in parallel
    CALL fort_solve_parallel(periods_payoffs_systematic_parallel, & 
            states_number_period_parallel, mapping_state_idx_parallel, & 
            periods_emax_parallel, states_all_parallel, coeffs_a, &
            coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, &
            is_interpolated, num_draws_emax, num_periods, num_points, & 
            edu_start, is_myopic, is_debug, edu_max, min_idx, delta, & 
            num_slaves, SLAVECOMM)

    CALL fort_solve(periods_payoffs_systematic_scalar, &
            states_number_period_scalar, mapping_state_idx_scalar, & 
            periods_emax_scalar, states_all_scalar, coeffs_a, &
            coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, &
            is_interpolated, num_draws_emax, periods_draws_emax, & 
            num_periods, num_points, edu_start, is_myopic, is_debug, & 
            edu_max, min_idx, delta)

    ! Test equality of return arguments.
    is_equal = ALL(periods_emax_scalar .EQ. periods_emax_parallel)
    CALL record_equality(is_equal)

    is_equal = ALL(states_number_period_scalar .EQ. states_number_period_parallel)
    CALL record_equality(is_equal)

    is_equal = ALL(mapping_state_idx_scalar .EQ. mapping_state_idx_parallel)
    CALL record_equality(is_equal)

    is_equal = ALL(periods_emax_scalar .EQ. periods_emax_parallel)
    CALL record_equality(is_equal)

    is_equal = ALL(states_all_scalar .EQ. states_all_parallel)
    CALL record_equality(is_equal)





END PROGRAM
!*******************************************************************************
!******************************************************************************* 
