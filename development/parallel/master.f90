!*******************************************************************************
!******************************************************************************* 
PROGRAM master

    !/* external modules        */

    USE shared_constants
    
    USE shared_auxiliary

    USE solve_auxiliary

    USE solve_fortran

    USE mpi
    
    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */
    
    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_all_tmp(:, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_number_period(:)

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

    REAL(our_dble), ALLOCATABLE     :: periods_emax(:, :)

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: delta
    REAL(our_dble)                  :: tau

    LOGICAL                         :: is_interpolated
    LOGICAL                         :: is_myopic
    LOGICAL                         :: is_debug

    CHARACTER(10)                   :: request
    CHARACTER(10)                   :: arg

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Initialize MPI Environment
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

    ! Allocate arrays
    ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
    ALLOCATE(states_all_tmp(num_periods, 100000, 4))
    ALLOCATE(states_number_period(num_periods))

    CALL fort_create_state_space(states_all_tmp, states_number_period, &
                mapping_state_idx, max_states_period, num_periods, edu_start, &
                edu_max)

    ALLOCATE(periods_emax(num_periods, max_states_period))


    ! Spawn the slaves for doing the actual work.
    CALL MPI_COMM_SPAWN('./slave', MPI_ARGV_NULL, num_slaves, MPI_INFO_NULL, & 
            0, MPI_COMM_WORLD, SLAVECOMM, MPI_ERRCODES_IGNORE, ierr)

    ! Request EMAX calculation
    task = 2
    CALL MPI_Bcast(task, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)

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

END PROGRAM
!*******************************************************************************
!******************************************************************************* 
