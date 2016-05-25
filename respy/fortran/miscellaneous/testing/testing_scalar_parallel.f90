!*******************************************************************************
!******************************************************************************* 
MODULE testing_auxiliary

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS

SUBROUTINE record_equality(is_equal)

LOGICAL :: is_equal

IF(.NOT. is_equal) THEN

OPEN (UNIT=5, FILE=".error.testing")  ! Root directory
CLOSE (UNIT=5)
END IF


END SUBROUTINE

!*******************************************************************************
!*******************************************************************************
END MODULE
!*******************************************************************************
!******************************************************************************* 
PROGRAM testing

    !/* external modules        */

    USE parallel_auxiliary 

    USE resfort_library

    USE testing_auxiliary    

!    USE shared_constants
    
 !   USE shared_auxiliary

  !  USE solve_auxiliary

   ! USE solve_fortran

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
    INTEGER(our_int)                :: task, stat

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
    LOGICAL                         :: is_debug, is_equal, file_exists

    CHARACTER(10)                   :: request
    CHARACTER(10)                   :: arg
    REAL(our_dble), ALLOCATABLE     :: periods_draws_emax(:, :, :)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Cleanup

inquire(file='.error.testing',exist=file_exists)
if ( file_exists ) then
    open(unit=1234, iostat=stat, file='.error.testing', status='old')
    close(1234, status='delete')
END IF


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
            is_myopic, tau, num_procs) 

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
            num_procs, SLAVECOMM)

    CALL fort_solve(periods_payoffs_systematic_scalar, &
            states_number_period_scalar, mapping_state_idx_scalar, & 
            periods_emax_scalar, states_all_scalar, coeffs_a, &
            coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, &
            is_interpolated, num_draws_emax, periods_draws_emax, & 
            num_periods, num_points, edu_start, is_myopic, is_debug, & 
            edu_max, min_idx, delta)

    ! Test equality of return arguments.
    is_equal = ALL(periods_payoffs_systematic_scalar .EQ. periods_payoffs_systematic_parallel)
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
