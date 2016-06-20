!******************************************************************************
!******************************************************************************
MODULE parallelism_auxiliary

    !/* external modules        */

    USE parallelism_constants

    USE resfort_library

    USE mpi

    !/* setup                   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE distribute_information(num_emax_slaves, period, send_slave, recieve_slaves)
    
    ! DEVELOPMENT NOTES
    !
    ! The assumed-shape input arguments allow to use this subroutine repeatedly.

    !/* external objects        */

    REAL(our_dble), INTENT(INOUT)       :: recieve_slaves(:)    
     
    REAL(our_dble), INTENT(IN)          :: send_slave(:)

    INTEGER(our_int), INTENT(IN)        :: num_emax_slaves(num_periods, num_slaves)
    INTEGER(our_int), INTENT(IN)        :: period

    !/* internal objects        */

    INTEGER(our_int)                    :: rcounts(num_slaves)
    INTEGER(our_int)                    :: scounts(num_slaves)
    INTEGER(our_int)                    :: disps(num_slaves)
    INTEGER(our_int)                    :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Parameterize the communication.
    scounts = num_emax_slaves(period + 1, :)
    rcounts = scounts
    DO i = 1, num_slaves
        disps(i) = SUM(scounts(:i - 1)) 
    END DO
    
    CALL MPI_ALLGATHERV(send_slave, scounts(rank + 1), MPI_DOUBLE, recieve_slaves, rcounts, disps, MPI_DOUBLE, MPI_COMM_WORLD, ierr)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE determine_workload(jobs_slaves, jobs_total)

    !/* external objects        */

    INTEGER(our_int), INTENT(INOUT)     :: jobs_slaves(num_slaves)
    
    INTEGER(our_int), INTENT(IN)        :: jobs_total 

    !/* internal objects        */

    INTEGER(our_int)                    :: j
    INTEGER(our_int)                    :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    jobs_slaves = zero_int

    j = 1

    DO i = 1, jobs_total
            
        IF (j .GT. num_slaves) j = 1

        jobs_slaves(j) = jobs_slaves(j) + 1

        j = j + 1

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_evaluate_parallel(crit_val)

    !   This routine instructs the slaves to evaluate the criterion function and waits for the lead slave to send the result.

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)         :: crit_val

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL MPI_Bcast(3, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)

    CALL MPI_RECV(crit_val, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, SLAVECOMM, status, ierr)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_estimate_parallel(crit_val, success, message, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed, optimizer_used, maxfun, newuoa_npt, newuoa_rhobeg, newuoa_rhoend, newuoa_maxfun, bfgs_gtol, bfgs_maxiter, bfgs_stpmx)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: crit_val
        
    CHARACTER(150), INTENT(OUT)     :: message

    LOGICAL, INTENT(OUT)            :: success

    INTEGER(our_int), INTENT(IN)    :: newuoa_maxfun
    INTEGER(our_int), INTENT(IN)    :: newuoa_npt
    INTEGER(our_int), INTENT(IN)    :: maxfun
    INTEGER(our_int), INTENT(IN)    :: bfgs_maxiter

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)      :: newuoa_rhobeg
    REAL(our_dble), INTENT(IN)      :: newuoa_rhoend
    REAL(our_dble), INTENT(IN)      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)      :: coeffs_b(6)
    REAL(our_dble), INTENT(IN)      :: bfgs_stpmx
    REAL(our_dble), INTENT(IN)      :: bfgs_gtol

    CHARACTER(225), INTENT(IN)      :: optimizer_used

    LOGICAL, INTENT(IN)             :: paras_fixed(26) 

    !/* internal objects    */

    REAL(our_dble)                  :: x_free_start(COUNT(.not. paras_fixed))
    REAL(our_dble)                  :: x_free_final(COUNT(.not. paras_fixed))
    
    INTEGER(our_int)                :: iter
    INTEGER(our_int)                :: maxfun_int
    
    LOGICAL, PARAMETER              :: all_free(26) = .False.

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Some ingredients for the evaluation of the criterion function need to be created once and shared globally.
    CALL get_free_optim_paras(x_all_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, all_free)

    CALL get_free_optim_paras(x_free_start, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed)

    x_free_final = x_free_start

    IF (maxfun == zero_int) THEN

        success = .True.
        message = 'Single evaluation of criterion function at starting values.'

    ELSEIF (optimizer_used == 'FORT-NEWUOA') THEN

        ! This is required to keep the original design of the algorithm intact
        maxfun_int = MIN(maxfun, newuoa_maxfun) - 1 

        CALL newuoa(fort_criterion_parallel, x_free_final, newuoa_npt, newuoa_rhobeg, newuoa_rhoend, zero_int, maxfun_int, success, message, iter)
        
    ELSEIF (optimizer_used == 'FORT-BFGS') THEN

        CALL dfpmin(fort_criterion_parallel, fort_dcriterion_parallel, x_free_final, bfgs_gtol, bfgs_maxiter, bfgs_stpmx, maxfun, success, message, iter)

    END IF
    
    crit_val = fort_criterion_parallel(x_free_final)

    CALL logging_estimation_final(success, message, crit_val)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION fort_criterion_parallel(x)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)      :: x(:)
    REAL(our_dble)                  :: fort_criterion_parallel

    !/* internal objects    */

    REAL(our_dble), SAVE            :: value_step = HUGE_FLOAT
    
    INTEGER(our_int), SAVE          :: num_step = - one_int

    INTEGER(our_int)                :: num_states
    INTEGER(our_int)                :: period
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

    LOGICAL                         :: is_start
    LOGICAL                         :: is_step

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Ensuring that the criterion function is not evaluated more than specified. However, there is the special request of MAXFUN equal to zero which needs to be allowed.
    IF ((num_eval == maxfun) .AND. (maxfun .GT. zero_int)) THEN
        fort_criterion_parallel = -HUGE_FLOAT
        RETURN
    END IF

    ! Construct the full set of current parameters
    j = 1

    DO i = 1, 26

        IF(paras_fixed(i)) THEN

            x_all_current(i) = x_all_start(i)

        ELSE
            
            x_all_current(i) = x(j)
            j = j + 1

        END IF

    END DO


    !  Update parameter that each slave is working with.
    CALL MPI_Bcast(0, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
     
    CALL MPI_Bcast(x_all_current, 26, MPI_DOUBLE, MPI_ROOT, SLAVECOMM, ierr)


    ! Solve the model    
    CALL MPI_Bcast(2, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
    
    ! THis block is only temporary until the slave is extracted ....
    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, periods_emax, periods_payoffs_systematic, edu_start, edu_max)

    DO period = (num_periods - 1), 0, -1

        num_states = states_number_period(period + 1)
        
        CALL MPI_RECV(periods_emax(period + 1, :num_states) , num_states, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, SLAVECOMM, status, ierr)
        
    END DO


    CALL fort_evaluate_parallel(fort_criterion_parallel)



    num_eval = num_eval + 1

    is_start = (num_eval == 1)



    is_step = (value_step .GT. fort_criterion_parallel) 
 
    IF (is_step) THEN

        num_step = num_step + 1

        value_step = fort_criterion_parallel

    END IF

    
    CALL write_out_information(num_eval, fort_criterion_parallel, x_all_current, 'current')

    IF (is_start) THEN

        CALL write_out_information(zero_int, fort_criterion_parallel, x_all_current, 'start')

    END IF

    IF (is_step) THEN

        CALL write_out_information(num_step, fort_criterion_parallel, x_all_current, 'step')

        CALL logging_estimation_step(num_step, fort_criterion_parallel)

    END IF

    
END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION fort_dcriterion_parallel(x)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: x(:)
    REAL(our_dble)                  :: fort_dcriterion_parallel(SIZE(x))

    !/* internals objects       */

    REAL(our_dble)                  :: ei(COUNT(.NOT. paras_fixed))
    REAL(our_dble)                  :: d(COUNT(.NOT. paras_fixed))
    REAL(our_dble)                  :: f0
    REAL(our_dble)                  :: f1

    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    ei = zero_dble

    ! Evaluate baseline
    f0 = fort_criterion_parallel(x)

    DO j = 1, COUNT(.NOT. paras_fixed)

        ei(j) = one_dble

        d = bfgs_epsilon * ei

        f1 = fort_criterion_parallel(x + d)

        fort_dcriterion_parallel(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE fort_solve_parallel(periods_payoffs_systematic, states_number_period, mapping_state_idx, periods_emax, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start, edu_max)

    !/* external objects        */

    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE, INTENT(INOUT)    :: states_all(:, :, :)

    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE, INTENT(INOUT)      :: periods_emax(:, :)

    REAL(our_dble), INTENT(IN)                      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)                      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)                      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)                      :: coeffs_b(6)

    INTEGER(our_int), INTENT(IN)                    :: edu_start
    INTEGER(our_int), INTENT(IN)                    :: edu_max

    !/* internal objects        */

    INTEGER(our_int)                                :: num_states
    INTEGER(our_int)                                :: period

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
      
    CALL MPI_Bcast(2, 1, MPI_INT, MPI_ROOT, SLAVECOMM, ierr)
    
    CALL fort_create_state_space(states_all, states_number_period, mapping_state_idx, periods_emax, periods_payoffs_systematic, edu_start, edu_max)

    CALL fort_calculate_payoffs_systematic(periods_payoffs_systematic, states_number_period, states_all, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, edu_start)

    DO period = (num_periods - 1), 0, -1

        num_states = states_number_period(period + 1)
        
        CALL MPI_RECV(periods_emax(period + 1, :num_states) , num_states, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, SLAVECOMM, status, ierr)
        
    END DO

    CALL logging_solution(-1)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE