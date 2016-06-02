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

    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE   :: num_emax_slaves(:, :)
    INTEGER(our_int), ALLOCATABLE   :: num_obs_slaves(:)
    INTEGER(our_int), ALLOCATABLE   :: states_all(:, :, :)

    INTEGER(our_int)                :: lower_bound
    INTEGER(our_int)                :: upper_bound
    INTEGER(our_int)                :: num_states
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: count
    INTEGER(our_int)                :: task
    INTEGER(our_int)                :: k

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_emax(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_prob(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax_slaves(:)
    REAL(our_dble), ALLOCATABLE     :: endogenous_slaves(:)
    REAL(our_dble), ALLOCATABLE     :: periods_emax(:, :)
    REAL(our_dble), ALLOCATABLE     :: draws_emax(:, :)
    REAL(our_dble), ALLOCATABLE     :: data_array(:, :)
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
    
!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize MPI environment
    CALL MPI_INIT(ierr)
    CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
    CALL MPI_COMM_SIZE(MPI_COMM_WORLD, num_slaves, ierr)
    CALL MPI_COMM_GET_PARENT(PARENTCOMM, ierr)

    ! Read in model specification.
    CALL read_specification(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky)

    ! Allocate arrays
    IF(rank == 0) CALL logging_solution(1)

    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, periods_emax, periods_payoffs_systematic)

    IF(rank == 0) CALL logging_solution(-1)

    ! Determine workload and allocate communication information.
    ALLOCATE(num_emax_slaves(num_periods, num_slaves), num_obs_slaves(num_slaves), draws_emax(num_draws_emax, 4))

    CALL determine_workload(num_obs_slaves, (num_agents_est * num_periods))
    DO period = 1, num_periods
        CALL determine_workload(num_emax_slaves(period, :), states_number_period(period))   
    END DO

    ! Calculate the systematic payoffs
    IF(rank == 0) CALL logging_solution(2)

    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

    IF(rank == 0) CALL logging_solution(-1)

    ! This part creates (or reads from disk) the draws for the Monte Carlo integration of the EMAX. For is_debugging purposes, these might  also be read in from disk or set to zero/one.   
    CALL create_draws(periods_draws_emax, num_draws_emax, seed_emax)

    is_head = .False.
    IF(rank == zero_int) is_head = .True.
    
    DO WHILE (STAY_AVAILABLE)  
        
        ! Waiting for request from master to perform an action.
        CALL MPI_Bcast(task, 1, MPI_INT, 0, PARENTCOMM, ierr)

        ! Shutting down operations.
        IF(task == 1) THEN

            CALL MPI_FINALIZE(ierr)
            STAY_AVAILABLE = .FALSE.

        ! Evaluate EMAX.
        ELSEIF(task == 2) THEN

            ! Set random seed. We need to set the seed here as well as this part of the code might be called using F2PY without any previous seed set. This ensures that the interpolation grid is identical across draws.
            seed_inflated(:) = 123

            CALL RANDOM_SEED(size=seed_size)

            CALL RANDOM_SEED(put=seed_inflated)

            ! Construct auxiliary objects
            shocks_cov = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))

            ! Shifts
            shifts = zero_dble
            shifts(1) = clip_value(EXP(shocks_cov(1, 1)/two_dble), zero_dble, HUGE_FLOAT)
            shifts(2) = clip_value(EXP(shocks_cov(2, 2)/two_dble), zero_dble, HUGE_FLOAT)

            IF(rank == 0) CALL logging_solution(3)

            DO period = (num_periods - 1), 0, -1

                ! Extract draws and construct auxiliary objects
                draws_emax = periods_draws_emax(period + 1, :, :)
                num_states = states_number_period(period + 1)
                ALLOCATE(periods_emax_slaves(num_states), endogenous_slaves(num_states))

                IF (rank == 0) CALL logging_solution(4, period, num_states)        

                ! Distinguish case with and without interpolation
                any_interpolated = (num_points_interp .LE. num_states) .AND. is_interpolated

                ! Upper and lower bound of tasks
                lower_bound = SUM(num_emax_slaves(period + 1, :rank))
                upper_bound = SUM(num_emax_slaves(period + 1, :rank + 1))
                
                 IF (any_interpolated) THEN

                    ! Allocate period-specific containers
                    ALLOCATE(is_simulated(num_states), endogenous(num_states), maxe(num_states), exogenous(num_states, 9), predictions(num_states))

                    ! Constructing indicator for simulation points
                    is_simulated = get_simulated_indicator(num_points_interp, num_states, period)
       
                    ! Constructing the dependent variable for all states, including the ones where simulation will take place. All information will be used in either the construction of the prediction model or the prediction step.
                    CALL get_exogenous_variables(exogenous, maxe, period, num_states, periods_payoffs_systematic, shifts, mapping_state_idx, periods_emax, states_all)

                    ! Initialize missing values
                    endogenous = MISSING_FLOAT
                    endogenous_slaves = MISSING_FLOAT

                    ! Construct dependent variables for the subset of interpolation points.
                    count = 1
                    DO k = lower_bound, upper_bound - 1

                        ! Skip over points that will be predicted
                        IF (.NOT. is_simulated(k + 1)) THEN
                            count = count + 1 
                            CYCLE
                        END IF

                        ! Extract payoffs
                        payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

                        ! Get payoffs
                        CALL get_future_value(emax_simulated, draws_emax, period, k, payoffs_systematic, mapping_state_idx, states_all, periods_emax, shocks_cholesky)

                        ! Construct dependent variable
                        endogenous_slaves(count) = emax_simulated - maxe(k + 1)
                        count = count + 1 

                    END DO
                    
                    ! Distribute exogenous information
                    CALL distribute_information(num_emax_slaves, period, endogenous_slaves, endogenous)
                    
                    ! Create prediction model based on the random subset of points where the EMAX is actually simulated and thus endogenous and exogenous variables are available. For the interpolation  points, the actual values are used.
                    CALL get_predictions(predictions, endogenous, exogenous, maxe, is_simulated, num_states, is_head)

                    ! Store results
                    periods_emax(period + 1, :num_states) = predictions

                    ! The leading slave updates the master period by period.
                    IF (rank == 0) CALL MPI_SEND(periods_emax(period + 1, :num_states), num_states, MPI_DOUBLE, 0, period, PARENTCOMM, ierr)    

                    ! Deallocate containers
                    DEALLOCATE(is_simulated, exogenous, maxe, endogenous, predictions)

                 ELSE

                    count =  1
                    DO k = lower_bound, upper_bound - 1

                        ! Extract payoffs
                        payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

                        CALL get_future_value(emax_simulated, draws_emax, period, k, payoffs_systematic, mapping_state_idx, states_all, periods_emax, shocks_cholesky)

                        ! Collect information
                        periods_emax_slaves(count) = emax_simulated

                        count = count + 1

                    END DO
                    
                    CALL distribute_information(num_emax_slaves, period, periods_emax_slaves, periods_emax(period + 1, :))
                    
                    ! The leading slave updates the master period by period.
                    IF (rank == 0) CALL MPI_SEND(periods_emax(period + 1, :num_states), num_states, MPI_DOUBLE, 0, period, PARENTCOMM, ierr)            
          
                END IF

                DEALLOCATE(periods_emax_slaves, endogenous_slaves)
    
            END DO
        
        ! Evaluate criterion function
        ELSEIF (task == 3) THEN

            ! If the evaluation is requested for the first time. The data container is not allocated, so all preparations for the evaluation are taken.
            IF (.NOT. ALLOCATED(data_array)) THEN

                CALL read_dataset(data_array, num_agents_est)

                CALL create_draws(periods_draws_prob, num_draws_prob, seed_prob)

                ! Upper and lower bound of tasks
                lower_bound = SUM(num_obs_slaves(:rank)) + 1
                upper_bound = SUM(num_obs_slaves(:rank + 1))
    
                ! Allocate dataset
                ALLOCATE(data_slave(num_obs_slaves(rank + 1), 8))

                data_slave = data_array(lower_bound:upper_bound, :)

            END IF

            ! Evaluate criterion function    
            CALL fort_evaluate(partial_crit, periods_payoffs_systematic, mapping_state_idx, periods_emax, states_all, shocks_cholesky, data_slave, periods_draws_prob)
            
            CALL MPI_REDUCE(partial_crit, crit_val, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

            ! The leading slave updates the master 
            IF (rank == 0) CALL MPI_SEND(crit_val, 1, MPI_DOUBLE, 0, 75, PARENTCOMM, ierr)            
          
        END IF    

    END DO

END PROGRAM
!******************************************************************************
!******************************************************************************