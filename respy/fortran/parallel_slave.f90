!*******************************************************************************
!******************************************************************************* 
MODULE slave_shared

    !/* external modules    */

    USE resfort_library 

    USE mpi

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE distribute_inter(num_emax_slaves, period, & 
                periods_emax_slaves, periods_emax, rank, num_states, & 
                PARENTCOMM)
    
    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: num_emax_slaves(:, :)
    INTEGER(our_int), INTENT(IN)    :: PARENTCOMM
    INTEGER(our_int), INTENT(IN)    :: num_states
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: rank
 
    REAL(our_dble), INTENT(IN)      :: periods_emax_slaves(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:)

    !/* internal objects        */

    INTEGER(our_int), ALLOCATABLE   :: rcounts(:)
    INTEGER(our_int), ALLOCATABLE   :: scounts(:)
    INTEGER(our_int), ALLOCATABLE   :: disps(:)

    INTEGER(our_int)                :: num_slaves
    INTEGER(our_int)                :: ierr
    INTEGER(our_int)                :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Auxiliary objects
    num_slaves = SIZE(num_emax_slaves, 2)

    ! Parameterize the communication.
    ALLOCATE(rcounts(num_slaves), scounts(num_slaves), disps(num_slaves))
    scounts(:) = num_emax_slaves(period + 1, :)
    rcounts(:) = scounts
    DO i = 1, num_slaves
        disps(i) = SUM(scounts(:i - 1)) 
    END DO
    
    ! Aggregate the EMAX contributions across the slaves.    
    CALL MPI_ALLGATHERV(periods_emax_slaves, scounts(rank + 1), MPI_DOUBLE, & 
            periods_emax, rcounts, disps, MPI_DOUBLE, & 
            MPI_COMM_WORLD, ierr)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE distribute_information(num_emax_slaves, period, & 
                periods_emax_slaves, periods_emax, rank, num_states, & 
                PARENTCOMM)
    
    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: num_emax_slaves(:, :)
    INTEGER(our_int), INTENT(IN)    :: PARENTCOMM
    INTEGER(our_int), INTENT(IN)    :: num_states
    INTEGER(our_int), INTENT(IN)    :: period
    INTEGER(our_int), INTENT(IN)    :: rank
 
    REAL(our_dble), INTENT(IN)      :: periods_emax_slaves(:)
    REAL(our_dble), INTENT(IN)      :: periods_emax(:, :)

    !/* internal objects        */

    INTEGER(our_int), ALLOCATABLE   :: rcounts(:)
    INTEGER(our_int), ALLOCATABLE   :: scounts(:)
    INTEGER(our_int), ALLOCATABLE   :: disps(:)

    INTEGER(our_int)                :: num_slaves
    INTEGER(our_int)                :: ierr
    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: periods_emax_subset(num_states)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Auxiliary objects
    num_slaves = SIZE(num_emax_slaves, 2)

    ! Parameterize the communication.
    ALLOCATE(rcounts(num_slaves), scounts(num_slaves), disps(num_slaves))
    scounts(:) = num_emax_slaves(period + 1, :)
    rcounts(:) = scounts
    DO i = 1, num_slaves
        disps(i) = SUM(scounts(:i - 1)) 
    END DO
    
    ! Aggregate the EMAX contributions across the slaves.    
    CALL MPI_ALLGATHERV(periods_emax_slaves, scounts(rank + 1), MPI_DOUBLE, & 
            periods_emax(period + 1, :), rcounts, disps, MPI_DOUBLE, & 
            MPI_COMM_WORLD, ierr)

    ! The leading slave updates the master period by period.
    periods_emax_subset = periods_emax(period + 1, :num_states)
    IF (rank == 0) THEN
        CALL MPI_SEND(periods_emax_subset, num_states, & 
            MPI_DOUBLE, 0, period, PARENTCOMM, ierr)            
    END IF

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE determine_workload(num_emax_slaves, num_periods, num_slaves, & 
            states_number_period)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(OUT)   :: num_emax_slaves(:, :)
    
    INTEGER(our_int), INTENT(IN)    :: states_number_period(:)
    INTEGER(our_int), INTENT(IN)    :: num_periods
    INTEGER(our_int), INTENT(IN)    :: num_slaves

    !/* internal objects        */

    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ALLOCATE(num_emax_slaves(num_periods, num_slaves))

    num_emax_slaves = zero_int

    DO period = 1, num_periods

        j = 1

        DO i = 1, states_number_period(period)
            
            IF (j .GT. num_slaves) THEN
            
                j = 1

            END IF

            num_emax_slaves(period, j) = num_emax_slaves(period, j) + 1

            j = j + 1

        END DO
    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE
!*******************************************************************************
!******************************************************************************* 
PROGRAM slave

    !/* external modules        */

    USE resfort_library

    USE slave_shared    

    USE mpi
    
    !/* setup                   */

    IMPLICIT NONE

    !/* objects                 */

    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE   :: states_all_tmp(:, :, :)
    INTEGER(our_int), ALLOCATABLE   :: num_emax_slaves(:, :)
    INTEGER(our_int), ALLOCATABLE   :: states_all(:, :, :)
    
    INTEGER(our_int)                :: max_states_period
    INTEGER(our_int)                :: lower_bound
    INTEGER(our_int)                :: upper_bound
    INTEGER(our_int)                :: num_periods
    INTEGER(our_int)                :: num_states
    INTEGER(our_int)                :: num_points
    INTEGER(our_int)                :: num_slaves
    INTEGER(our_int)                :: PARENTCOMM
    INTEGER(our_int)                :: edu_start
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: count
    INTEGER(our_int)                :: rank
    INTEGER(our_int)                :: ierr
    INTEGER(our_int)                :: task
    INTEGER(our_int)                :: k

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_emax(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax_slaves(:)
    REAL(our_dble), ALLOCATABLE     :: endogenous_slaves(:)
    REAL(our_dble), ALLOCATABLE     :: periods_emax(:, :)
    REAL(our_dble), ALLOCATABLE     :: draws_emax(:, :)

    REAL(our_dble)                  :: shocks_cholesky(4, 4)
    REAL(our_dble)                  :: payoffs_systematic(4)
    REAL(our_dble)                  :: shocks_cov(4, 4)
    REAL(our_dble)                  :: emax_simulated
    REAL(our_dble)                  :: coeffs_home(1)
    REAL(our_dble)                  :: coeffs_edu(3)
    REAL(our_dble)                  :: coeffs_a(6)
    REAL(our_dble)                  :: coeffs_b(6)
    REAL(our_dble)                  :: shifts(4)

    LOGICAL                         :: STAY_AVAILABLE = .TRUE.
    LOGICAL                         :: any_interpolated, is_head

     LOGICAL, ALLOCATABLE   :: is_simulated(:)
     REAL(our_dble), ALLOCATABLE :: endogenous(:)
     REAL(our_dble), ALLOCATABLE :: maxe(:), exogenous(:, :), predictions(:)
     
         INTEGER(our_int)                    :: seed_inflated(15)
    INTEGER(our_int)                    :: seed_size

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Initialize MPI environment
    CALL MPI_INIT(ierr)
    CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
    CALL MPI_COMM_SIZE(MPI_COMM_WORLD, num_slaves, ierr)
    CALL MPI_COMM_GET_PARENT(PARENTCOMM, ierr)

    ! Read in model specification.
    CALL read_specification(num_periods, coeffs_a, coeffs_b, & 
            coeffs_edu, edu_start, coeffs_home, shocks_cholesky, & 
            num_points)

    ALLOCATE(draws_emax(num_draws_emax, 4))

    ! Allocate arrays
    ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
    ALLOCATE(states_all_tmp(num_periods, 100000, 4))
    ALLOCATE(states_number_period(num_periods))

    IF(rank == 0) CALL logging_solution(1)

    CALL fort_create_state_space(states_all_tmp, states_number_period, &
            mapping_state_idx, max_states_period, num_periods, edu_start)

    IF(rank == 0) CALL logging_solution(-1)

    ALLOCATE(periods_emax(num_periods, max_states_period))

    ! Determine workload and allocate communication information.
    CALL determine_workload(num_emax_slaves, num_periods, num_slaves, & 
            states_number_period)

    states_all = states_all_tmp(:, :max_states_period, :)
    DEALLOCATE(states_all_tmp)

    ALLOCATE(periods_payoffs_systematic(num_periods, max_states_period, 4))

    ! Calculate the systematic payoffs
    IF(rank == 0) CALL logging_solution(2)

    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, &
            num_periods, states_number_period, states_all, edu_start, &
            coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

    IF(rank == 0) CALL logging_solution(-1)

    ! This part creates (or reads from disk) the draws for the Monte 
    ! Carlo integration of the EMAX. For is_debugging purposes, these might 
    ! also be read in from disk or set to zero/one.   
    CALL create_draws(periods_draws_emax, num_periods, num_draws_emax, & 
            seed_emax)

    periods_emax = MISSING_FLOAT

    IF(rank == zero_int) THEN
        is_head = .True.
    ELSE
        is_head = .False.
    END IF

    DO WHILE (STAY_AVAILABLE)  
        
        ! Waiting for request from master to perform an action.
        CALL MPI_Bcast(task, 1, MPI_INT, 0, PARENTCOMM, ierr)

        ! Shutting down operations.
        IF(task == 1) THEN

            CALL MPI_FINALIZE(ierr)
            STAY_AVAILABLE = .FALSE.

        ! Evaluate EMAX.
        ELSEIF(task == 2) THEN

            ! Set random seed. We need to set the seed here as well as this part of the
            ! code might be called using F2PY without any previous seed set. This
            ! ensures that the interpolation grid is identical across draws.
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
                ALLOCATE(periods_emax_slaves(num_states))

                ALLOCATE(endogenous_slaves(num_states))


                IF (rank == 0) THEN
                     CALL logging_solution(4, period, num_states)        
                END IF
                ! Distinguish case with and without interpolation
                any_interpolated = (num_points .LE. num_states) .AND. is_interpolated

                ! Upper and lower bound of tasks
                lower_bound = SUM(num_emax_slaves(period + 1, :rank))
                upper_bound = SUM(num_emax_slaves(period + 1, :rank + 1))
                
                 IF (any_interpolated) THEN

                    ! Allocate period-specific containers
                    ALLOCATE(is_simulated(num_states)); ALLOCATE(endogenous(num_states))
                    ALLOCATE(maxe(num_states)); ALLOCATE(exogenous(num_states, 9))
                    ALLOCATE(predictions(num_states))

                    ! Constructing indicator for simulation points
                    is_simulated = get_simulated_indicator(num_points, num_states, &
                                        period, num_periods)

       
                    ! Constructing the dependent variable for all states, including the
                    ! ones where simulation will take place. All information will be
                    ! used in either the construction of the prediction model or the
                    ! prediction step.
                    CALL get_exogenous_variables(exogenous, maxe, period, num_periods, &
                            num_states, periods_payoffs_systematic, shifts, &
                            edu_start, mapping_state_idx, periods_emax, &
                            states_all)



                    ! Initialize missing values
                    endogenous = MISSING_FLOAT
                    endogenous_slaves = MISSING_FLOAT

                    ! Construct dependent variables for the subset of interpolation
                    ! points.
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
                        CALL get_payoffs(emax_simulated, draws_emax, period, &
                                k, payoffs_systematic, edu_start, mapping_state_idx, &
                                states_all, num_periods, periods_emax, shocks_cholesky)

                        ! Construct dependent variable
                        endogenous_slaves(count) = emax_simulated - maxe(k + 1)
                        count = count + 1 



                    END DO
                    

                    ! Distribute exogenous information
                    ! TODO: POLYMORPHISM
                    CALL distribute_inter(num_emax_slaves, period, & 
                        endogenous_slaves, endogenous, rank, num_states, & 
                        PARENTCOMM)
                    
                    ! Create prediction model based on the random subset of points where
                    ! the EMAX is actually simulated and thus endogenous and
                    ! exogenous variables are available. For the interpolation
                    ! points, the actual values are used.
                    CALL get_predictions(predictions, endogenous, exogenous, maxe, &
                            is_simulated, num_points, num_states, is_head)

                    ! Store results
                    periods_emax(period + 1, :num_states) = predictions

                    ! TODO: Keep master updates, somebody please!!
                    ! The leading slave updates the master period by period.
                    IF (rank == 0) THEN
                        CALL MPI_SEND(periods_emax(period + 1, :num_states), num_states, & 
                            MPI_DOUBLE, 0, period, PARENTCOMM, ierr)    
                    END IF
                    ! Deallocate containers
                    DEALLOCATE(is_simulated); DEALLOCATE(exogenous); DEALLOCATE(maxe);
                    DEALLOCATE(endogenous); DEALLOCATE(predictions)


                 ELSE

                    count =  1
                    DO k = lower_bound, upper_bound - 1

                        ! Extract payoffs
                        payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

                        CALL get_payoffs(emax_simulated, draws_emax, &
                                period, k, payoffs_systematic, edu_start, &
                                mapping_state_idx, states_all, num_periods, &
                                periods_emax, shocks_cholesky)

                        ! Collect information
                        periods_emax_slaves(count) = emax_simulated

                        count = count + 1

                    END DO
                    
                    CALL distribute_information(num_emax_slaves, period, & 
                        periods_emax_slaves, periods_emax, rank, num_states, & 
                        PARENTCOMM)
    
          
                END IF

                DEALLOCATE(periods_emax_slaves)
                DEALLOCATE(endogenous_slaves)
    
            END DO
       
        END IF    

    END DO

END PROGRAM
!*******************************************************************************
!*******************************************************************************