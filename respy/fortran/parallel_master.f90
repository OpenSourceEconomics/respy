!*******************************************************************************
!******************************************************************************* 
PROGRAM master

    !/* external modules        */

    USE parallel_auxiliary    

    USE resfort_library 
    
    USE mpi
    
    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */
    
    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_all_tmp(:, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE   :: states_all(:, :, :)

    INTEGER(our_int)                :: status(MPI_STATUS_SIZE) 
    INTEGER(our_int)                :: max_states_period
    INTEGER(our_int)                :: num_draws_emax
    INTEGER(our_int)                :: num_draws_prob
    INTEGER(our_int)                :: num_agents_est
    INTEGER(our_int)                :: num_periods
    INTEGER(our_int)                :: num_points
    INTEGER(our_int)                :: num_states
    INTEGER(our_int)                :: SLAVECOMM
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: edu_start
    INTEGER(our_int)                :: num_procs
    INTEGER(our_int)                :: min_idx
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: ierr
    INTEGER(our_int)                :: task

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_prob(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax(:, :)
    REAL(our_dble), ALLOCATABLE     :: data_array(:, :)

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: crit_val
    REAL(our_dble)                  :: tau

    LOGICAL                         :: is_myopic
    LOGICAL                         :: is_debug

    CHARACTER(225)                  :: exec_dir
    CHARACTER(10)                   :: request
    CHARACTER(10)                   :: arg

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Initialize MPI environment
    CALL MPI_INIT(ierr)

    ! Read in model specification.
    CALL read_specification(num_periods, coeffs_a, coeffs_b, &
            coeffs_edu, edu_start, coeffs_home, shocks_cholesky, & 
            num_draws_emax, seed_emax, seed_prob, num_agents_est, is_debug, & 
            num_points, min_idx, request, num_draws_prob, & 
            is_myopic, tau, num_procs, exec_dir) 

    ! Execute on request.
    IF (request == 'solve') THEN

        ! Solve the model for a given parametrization in parallel.
        CALL fort_solve_parallel(periods_payoffs_systematic, states_number_period, &
                mapping_state_idx, periods_emax, states_all, coeffs_a, &
                coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, &
                num_draws_emax, num_periods, num_points, & 
                edu_start, is_myopic, is_debug, min_idx, & 
                num_procs, SLAVECOMM, exec_dir)

    ELSE IF (request == 'evaluate') THEN

        ! This part creates (or reads from disk) the draws for the Monte 
        ! Carlo integration of the choice probabilities. For is_debugging 
        ! purposes, these might also be read in from disk or set to zero/one.   
        CALL create_draws(periods_draws_prob, num_periods, num_draws_prob, &
                seed_prob, is_debug)

        ! Read observed dataset from disk.
        CALL read_dataset(data_array, num_periods, num_agents_est)

        ! Solve the model for a given parametrization in parallel
        ! periods_draws_emax
        CALL fort_solve_parallel(periods_payoffs_systematic, states_number_period, &
                mapping_state_idx, periods_emax, states_all, coeffs_a, &
                coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, &
                num_draws_emax, num_periods, num_points, & 
                edu_start, is_myopic, is_debug, min_idx, & 
                num_procs, SLAVECOMM, exec_dir)

        ! TODO: Parallelize
        CALL fort_evaluate(crit_val, periods_payoffs_systematic, & 
                mapping_state_idx, periods_emax, states_all, shocks_cholesky, & 
                num_periods, edu_start, data_array, & 
                num_agents_est, num_draws_prob, periods_draws_prob, tau)

    END IF

    ! Cleanup
    OPEN(UNIT=1, FILE='.model.resfort.ini')
    CLOSE(1, STATUS='delete')
    
    ! Store results. These are read in by the PYTHON wrapper and added to the 
    ! RespyCls instance.
    CALL store_results(mapping_state_idx, states_all, &
            periods_payoffs_systematic, states_number_period, periods_emax, &
            num_periods, min_idx, crit_val, request)


END PROGRAM
!*******************************************************************************
!******************************************************************************* 
