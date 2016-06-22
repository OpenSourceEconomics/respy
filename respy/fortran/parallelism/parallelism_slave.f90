!******************************************************************************
!******************************************************************************
PROGRAM resfort_parallel_slave

    !/* external modules        */

    USE parallelism_constants
    
    USE parallelism_auxiliary

    USE resfort_library

    USE mpi
    
    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */

    INTEGER(our_int), ALLOCATABLE   :: num_emax_slaves(:, :)
    INTEGER(our_int), ALLOCATABLE   :: num_obs_slaves(:)

    INTEGER(our_int)                :: lower_bound
    INTEGER(our_int)                :: upper_bound
    INTEGER(our_int)                :: task

    REAL(our_dble), ALLOCATABLE     :: data_slave(:, :)

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: partial_crit
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: crit_val
    
    LOGICAL                         :: STAY_AVAILABLE = .TRUE.

    REAL(our_dble)                  :: newuoa_rhobeg
    REAL(our_dble)                  :: newuoa_rhoend    
    REAL(our_dble)                  :: bfgs_stpmx
    REAL(our_dble)                  :: bfgs_gtol 

    INTEGER(our_int)                :: newuoa_maxfun
    INTEGER(our_int)                :: bfgs_maxiter
    INTEGER(our_int)                :: newuoa_npt
    INTEGER(our_int)                :: num_procs
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: seed_sim

    CHARACTER(225)                  :: optimizer_used
    CHARACTER(225)                  :: exec_dir
    CHARACTER(10)                   :: request

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL MPI_INIT(ierr)

    CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    CALL MPI_COMM_SIZE(MPI_COMM_WORLD, num_slaves, ierr)

    CALL MPI_COMM_GET_PARENT(PARENTCOMM, ierr)


    CALL read_specification(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, edu_start, edu_max, delta, tau, seed_sim, seed_emax, seed_prob, num_procs, is_debug, is_interpolated, is_myopic, request, exec_dir, maxfun, paras_fixed, optimizer_used, newuoa_npt, newuoa_maxfun, newuoa_rhobeg, newuoa_rhoend, bfgs_epsilon, bfgs_gtol, bfgs_stpmx, bfgs_maxiter)


    IF (rank == zero_int) CALL logging_solution(1)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, edu_start, edu_max)  
    
    IF (rank == zero_int) CALL logging_solution(-1)



    CALL distribute_workload(num_emax_slaves, num_obs_slaves)

    CALL create_draws(periods_draws_emax, num_draws_emax, seed_emax, is_debug)

    
    DO WHILE (STAY_AVAILABLE)  

        
        CALL MPI_Bcast(task, 1, MPI_INT, 0, PARENTCOMM, ierr)

        IF (task == 1) THEN
            CALL MPI_FINALIZE(ierr)
            STAY_AVAILABLE = .FALSE.    
            CYCLE
        END IF


        CALL MPI_Bcast(x_all_current, 26, MPI_DOUBLE, 0, PARENTCOMM, ierr)

        CALL dist_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x_all_current)

        
    
        IF(task == 2) THEN

            IF (rank == zero_int) CALL logging_solution(2)

            CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start)

            IF (rank == zero_int) CALL logging_solution(-1)

            IF (rank == zero_int) CALL logging_solution(3)

            CALL fort_backward_induction_slave(periods_emax, periods_draws_emax, states_number_period, periods_payoffs_systematic, mapping_state_idx, states_all, shocks_cholesky, delta, is_debug, is_interpolated, is_myopic, edu_start, edu_max, num_emax_slaves, .True.)

            IF (rank == zero_int .AND. .NOT. is_myopic) THEN
                CALL logging_solution(-1)
            ELSEIF (rank == zero_int) THEN
                CALL logging_solution(-2)
            END IF

        ELSEIF (task == 3) THEN
            
            IF (.NOT. ALLOCATED(data_est)) THEN

                CALL read_dataset(data_est, num_agents_est)

                CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob, is_debug)

                lower_bound = SUM(num_obs_slaves(:rank)) + 1
                upper_bound = SUM(num_obs_slaves(:rank + 1))
    
                ALLOCATE(data_slave(num_obs_slaves(rank + 1), 8))

                data_slave = data_est(lower_bound:upper_bound, :)

            END IF

            CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start)

            CALL fort_backward_induction_slave(periods_emax, periods_draws_emax, states_number_period, periods_payoffs_systematic, mapping_state_idx, states_all, shocks_cholesky, delta, is_debug, is_interpolated, is_myopic, edu_start, edu_max, num_emax_slaves, .False.)

            CALL fort_evaluate(partial_crit, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_slave, periods_draws_prob, delta, tau, edu_start, edu_max)
           
            CALL MPI_REDUCE(partial_crit, crit_val, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

            IF (rank == zero_int) CALL MPI_SEND(crit_val, 1, MPI_DOUBLE, 0, 75, PARENTCOMM, ierr)            

        END IF    

    END DO

END PROGRAM
!******************************************************************************
!******************************************************************************