!******************************************************************************
!******************************************************************************
PROGRAM master

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

    ! Execute on request.
    IF (request == 'solve') THEN

        ! Solve the model for a given parametrization in parallel.
        CALL fort_solve_parallel(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky)

    ELSE IF (request == 'evaluate') THEN

        ! This part creates (or reads from disk) the draws for the Monte Carlo integration of the choice probabilities. For is_debugging purposes, these might also be read in from disk or set to zero/one.   
        CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob)

        ! Read observed dataset from disk.
        CALL read_dataset(data_array, num_agents_est)

        ! Solve the model for a given parametrization in parallel.
        CALL fort_solve_parallel(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky)
        
       ! TODO: Parallelize
        CALL fort_evaluate(crit_val, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_array, periods_draws_prob)

    END IF

    ! Cleanup
    OPEN(UNIT=1, FILE='.model.resfort.ini'); CLOSE(1, STATUS='delete')
    
    ! Store results. These are read in by the PYTHON wrapper and added to the  RespyCls instance.
    CALL store_results(mapping_state_idx, states_all, periods_payoffs_systematic, states_number_period, periods_emax, crit_val)

END PROGRAM
!******************************************************************************
!******************************************************************************
