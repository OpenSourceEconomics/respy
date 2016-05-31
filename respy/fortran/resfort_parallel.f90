!******************************************************************************
!******************************************************************************
PROGRAM resfort_parallel

    !/* external modules        */

    USE parallel_constants

    USE parallel_auxiliary    

    USE resfort_library 
    
    USE mpi
    
    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */
    
    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE   :: states_all(:, :, :)

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

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize MPI environment
    CALL MPI_INIT(ierr)

    ! Read in model specification.
    CALL read_specification(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky)

    ! Start army of slaves that are available to help with the computations
    CALL MPI_COMM_SPAWN(TRIM(exec_dir) // '/resfort_parallel_slave', MPI_ARGV_NULL, (num_procs - 1), MPI_INFO_NULL, 0, MPI_COMM_WORLD, SLAVECOMM, MPI_ERRCODES_IGNORE, ierr)
    CALL MPI_Bcast(2, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)

    ! Execute on request.
    IF (request == 'solve') THEN

        ! Solve the model for a given parametrization in parallel.
        CALL fort_solve_parallel(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

    ELSE IF (request == 'evaluate') THEN

        ! This part creates (or reads from disk) the draws for the Monte Carlo integration of the choice probabilities. For is_debugging purposes, these might also be read in from disk or set to zero/one.   
        CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob)

        ! Read observed dataset from disk.
        CALL read_dataset(data_array, num_agents_est)

        ! Solve the model for a given parametrization in parallel.
        CALL fort_solve_parallel(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home)
        
        ! Evaluate criterion function in parallel
        CALL fort_evaluate(crit_val, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_array, periods_draws_prob)

    END IF

    ! Store results. These are read in by the PYTHON wrapper and added to the  RespyCls instance.
    CALL store_results(mapping_state_idx, states_all, periods_payoffs_systematic, states_number_period, periods_emax, crit_val)

    ! Shut down orderly
    CALL MPI_Bcast(1, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
    CALL MPI_FINALIZE (ierr)

END PROGRAM
!******************************************************************************
!******************************************************************************
