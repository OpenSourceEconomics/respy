!*******************************************************************************
!******************************************************************************* 
MODULE slave_shared

    !/* external modules    */

    USE shared_auxiliary

    USE shared_constants

    USE mpi

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
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
    IF (rank == 0) THEN
        CALL MPI_SEND(periods_emax(period + 1, :num_states), num_states, & 
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

    USE shared_constants
    
    USE shared_auxiliary

    USE solve_auxiliary

    USE solve_fortran

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
    INTEGER(our_int)                :: num_draws_emax
    INTEGER(our_int)                :: num_draws_prob
    INTEGER(our_int)                :: num_agents_est
    INTEGER(our_int)                :: lower_bound
    INTEGER(our_int)                :: upper_bound
    INTEGER(our_int)                :: num_periods
    INTEGER(our_int)                :: num_states
    INTEGER(our_int)                :: num_points
    INTEGER(our_int)                :: num_slaves
    INTEGER(our_int)                :: PARENTCOMM
    INTEGER(our_int)                :: seed_prob
    INTEGER(our_int)                :: seed_emax
    INTEGER(our_int)                :: edu_start
    INTEGER(our_int)                :: edu_max
    INTEGER(our_int)                :: min_idx
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: count
    INTEGER(our_int)                :: rank
    INTEGER(our_int)                :: ierr
    INTEGER(our_int)                :: task
    INTEGER(our_int)                :: k

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_emax(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax_slaves(:)
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
    REAL(our_dble)                  :: delta
    REAL(our_dble)                  :: tau

    LOGICAL                         :: STAY_AVAILABLE = .TRUE.
    LOGICAL                         :: is_interpolated
    LOGICAL                         :: is_myopic
    LOGICAL                         :: is_debug

    CHARACTER(10)                   :: request


!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Initialize MPI environment
    CALL MPI_INIT(ierr)
    CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
    CALL MPI_COMM_SIZE(MPI_COMM_WORLD, num_slaves, ierr)
    CALL MPI_COMM_GET_PARENT(PARENTCOMM, ierr)

    ! Read in model specification.
    CALL read_specification(num_periods, delta, coeffs_a, coeffs_b, & 
            coeffs_edu, edu_start, edu_max, coeffs_home, shocks_cholesky, & 
            num_draws_emax, seed_emax, seed_prob, num_agents_est, is_debug, & 
            is_interpolated, num_points, min_idx, request, num_draws_prob, & 
            is_myopic, tau)

    ALLOCATE(draws_emax(num_draws_emax, 4))

    ! Allocate arrays
    ALLOCATE(mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2))
    ALLOCATE(states_all_tmp(num_periods, 100000, 4))
    ALLOCATE(states_number_period(num_periods))

    CALL fort_create_state_space(states_all_tmp, states_number_period, &
            mapping_state_idx, max_states_period, num_periods, edu_start, &
            edu_max)

    ALLOCATE(periods_emax(num_periods, max_states_period))

    ! Determine workload and allocate communication information.
    CALL determine_workload(num_emax_slaves, num_periods, num_slaves, & 
            states_number_period)


    states_all = states_all_tmp(:, :max_states_period, :)
    DEALLOCATE(states_all_tmp)

    ALLOCATE(periods_payoffs_systematic(num_periods, max_states_period, 4))

    ! Calculate the systematic payoffs
    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, &
            num_periods, states_number_period, states_all, edu_start, &
            coeffs_a, coeffs_b, coeffs_edu, coeffs_home)

    ! This part creates (or reads from disk) the draws for the Monte 
    ! Carlo integration of the EMAX. For is_debugging purposes, these might 
    ! also be read in from disk or set to zero/one.   
    CALL create_draws(periods_draws_emax, num_periods, num_draws_emax, & 
            seed_emax, is_debug)

    periods_emax = MISSING_FLOAT

    DO WHILE (STAY_AVAILABLE)  
        
        ! Waiting for request from master to perform an action.
        CALL MPI_Bcast(task, 1, MPI_INT, 0, PARENTCOMM, ierr)

        ! Shutting down operations.
        IF(task == 1) THEN

            CALL MPI_FINALIZE(ierr)
            STAY_AVAILABLE = .FALSE.

        ! Evaluate EMAX.
        ELSEIF(task == 2) THEN

            ! Construct auxiliary objects
            shocks_cov = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))

            DO period = (num_periods - 1), 0, -1

                ! Extract draws and construct auxiliary objects
                draws_emax = periods_draws_emax(period + 1, :, :)
                num_states = states_number_period(period + 1)
                ALLOCATE(periods_emax_slaves(num_states))


                ! Upper and lower bound of tasks
                lower_bound = SUM(num_emax_slaves(period + 1, :rank))
                upper_bound = SUM(num_emax_slaves(period + 1, :rank + 1))
                
                count =  1
                DO k = lower_bound, upper_bound - 1

                    ! Extract payoffs
                    payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

                    CALL get_payoffs(emax_simulated, num_draws_emax, draws_emax, &
                            period, k, payoffs_systematic, edu_max, edu_start, &
                            mapping_state_idx, states_all, num_periods, &
                            periods_emax, delta, shocks_cholesky)

                    ! Collect information
                    periods_emax_slaves(count) = emax_simulated

                    count = count + 1

                END DO

                CALL distribute_information(num_emax_slaves, period, & 
                        periods_emax_slaves, periods_emax, rank, num_states, & 
                        PARENTCOMM)
    
                DEALLOCATE(periods_emax_slaves)

            END DO
       
        END IF    

    END DO

END PROGRAM
!*******************************************************************************
!*******************************************************************************