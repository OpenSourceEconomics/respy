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
    INTEGER(our_int)                :: num_states
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: count
    INTEGER(our_int)                :: task
    INTEGER(our_int)                :: k

    REAL(our_dble), ALLOCATABLE     :: periods_emax_slaves(:)
    REAL(our_dble), ALLOCATABLE     :: endogenous_slaves(:)
    REAL(our_dble), ALLOCATABLE     :: draws_emax(:, :)
    REAL(our_dble), ALLOCATABLE     :: data_slave(:, :)
    REAL(our_dble), ALLOCATABLE     :: exogenous(:, :)
    REAL(our_dble), ALLOCATABLE     :: predictions(:)
    REAL(our_dble), ALLOCATABLE     :: endogenous(:)
    REAL(our_dble), ALLOCATABLE     :: maxe(:)

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: payoffs_systematic(4)
    REAL(our_dble)                  :: shocks_cov(4, 4)
    REAL(our_dble)                  :: emax_simulated
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: partial_crit
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: shifts(4)
    REAL(our_dble)                  :: crit_val

    LOGICAL, ALLOCATABLE            :: is_simulated(:)
    
    LOGICAL                         :: STAY_AVAILABLE = .TRUE.
    LOGICAL                         :: any_interpolated
    LOGICAL                         :: is_head

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

    ! Initialize MPI environment
    CALL MPI_INIT(ierr)
    CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
    CALL MPI_COMM_SIZE(MPI_COMM_WORLD, num_slaves, ierr)
    CALL MPI_COMM_GET_PARENT(PARENTCOMM, ierr)

    ! Determine the role of head slave, which has additional responsibilites
    is_head = .False.
    IF(rank == zero_int) is_head = .True.

    ! Read in model specification.
    CALL read_specification(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, edu_start, edu_max, delta, tau, seed_sim, seed_emax, seed_prob, num_procs, is_debug, is_interpolated, is_myopic, request, exec_dir, maxfun, paras_fixed, optimizer_used, newuoa_npt, newuoa_maxfun, newuoa_rhobeg, newuoa_rhoend, bfgs_epsilon, bfgs_gtol, bfgs_stpmx, bfgs_maxiter)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, edu_start, edu_max)

    

    ! Determine workload and allocate communication information.
    ALLOCATE(num_emax_slaves(num_periods, num_slaves), num_obs_slaves(num_slaves), draws_emax(num_draws_emax, 4))

    CALL determine_workload(num_obs_slaves, (num_agents_est * num_periods))
    DO period = 1, num_periods
        CALL determine_workload(num_emax_slaves(period, :), states_number_period(period))   
    END DO

    ! Calculate the systematic payoffs

    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start)

    ! TODO: IS THIS THE RIGHT PLACE TO DO IT?
    ALLOCATE(periods_emax(num_periods, max_states_period))
    periods_emax = MISSING_FLOAT

    ! This part creates (or reads from disk) the draws for the Monte Carlo integration of the EMAX. For is_debugging purposes, these might  also be read in from disk or set to zero/one.   
    CALL create_draws(periods_draws_emax, num_draws_emax, seed_emax, is_debug)
    
    DO WHILE (STAY_AVAILABLE)  
        
        ! Waiting for request from master to perform an action.
        CALL MPI_Bcast(task, 1, MPI_INT, 0, PARENTCOMM, ierr)

        IF(task == 1) THEN
            CALL MPI_FINALIZE(ierr)
            STAY_AVAILABLE = .FALSE.
        END IF


        CALL MPI_Bcast(x_all_current, 26, MPI_DOUBLE, 0, PARENTCOMM, ierr)

        CALL dist_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x_all_current)
            
        CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start)

        






        ! Evaluate EMAX.
        IF(task == 2) THEN

            CALL fort_backward_induction_slave(num_emax_slaves, shocks_cholesky, .True.)

        ! Evaluate criterion function
        ELSEIF (task == 3) THEN


            CALL fort_backward_induction_slave(num_emax_slaves, shocks_cholesky, .False.)

            ! If the evaluation is requested for the first time. The data container is not allocated, so all preparations for the evaluation are taken.
            IF (.NOT. ALLOCATED(data_est)) THEN

                CALL read_dataset(data_est, num_agents_est)

                CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob, is_debug)

                ! Upper and lower bound of tasks
                lower_bound = SUM(num_obs_slaves(:rank)) + 1
                upper_bound = SUM(num_obs_slaves(:rank + 1))
    
                ! Allocate dataset
                ALLOCATE(data_slave(num_obs_slaves(rank + 1), 8))

                data_slave = data_est(lower_bound:upper_bound, :)

            END IF

            ! Evaluate criterion function    
            CALL fort_evaluate(partial_crit, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_slave, periods_draws_prob, delta, tau, edu_start, edu_max)
           
            CALL MPI_REDUCE(partial_crit, crit_val, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

            ! The leading slave updates the master 
            IF (is_head) CALL MPI_SEND(crit_val, 1, MPI_DOUBLE, 0, 75, PARENTCOMM, ierr)            

        END IF    

    END DO

END PROGRAM
!******************************************************************************
!******************************************************************************