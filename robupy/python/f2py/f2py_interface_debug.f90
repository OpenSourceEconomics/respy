!*******************************************************************************
!*******************************************************************************
!
!   This subroutine is just a wrapper for selected functions of the ROBUFORT 
!   library. Its sole purpose is to serve as a wrapper for debugging purposes.
!
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_evaluate_fortran_bare(rslt, coeffs_a, coeffs_b, coeffs_edu, & 
                coeffs_home, shocks, edu_max, delta, edu_start, is_debug, & 
                is_interpolated, level, measure, min_idx, num_draws, & 
                num_periods, num_points, is_ambiguous, periods_eps_relevant, & 
                eps_cholesky, num_agents, num_sims, data_array, & 
                standard_deviates, max_states_period)

    !
    ! The presence of max_states_period breaks the equality of interfaces. 
    ! However, this is done for convenience reasons as it allows to allocate
    ! the results containers directly.
    !

    !/* external libraries      */

    USE robufort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: rslt 

    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_points
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: min_idx
    INTEGER, INTENT(IN)             :: num_agents, num_sims, max_states_period

    DOUBLE PRECISION, INTENT(IN)    :: periods_eps_relevant(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: standard_deviates(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: data_array(:, :)

    DOUBLE PRECISION, INTENT(IN)    :: eps_cholesky(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)
    DOUBLE PRECISION, INTENT(IN)    :: shocks(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(:)
    DOUBLE PRECISION, INTENT(IN)    :: level
    DOUBLE PRECISION, INTENT(IN)    :: delta 

    LOGICAL, INTENT(IN)             :: is_interpolated
    LOGICAL, INTENT(IN)             :: is_ambiguous
    LOGICAL, INTENT(IN)             :: is_debug

    CHARACTER(10), INTENT(IN)       :: measure

    !/* internal objects        */

    INTEGER, ALLOCATABLE            :: mapping_state_idx(:, :, :, :, :)
    INTEGER, ALLOCATABLE            :: states_number_period(:)
    INTEGER, ALLOCATABLE            :: states_all(:, :, :)

    DOUBLE PRECISION, ALLOCATABLE   :: periods_payoffs_systematic(:, :, :)
    DOUBLE PRECISION, ALLOCATABLE   :: periods_payoffs_ex_post(:, :, :)
    DOUBLE PRECISION, ALLOCATABLE   :: periods_future_payoffs(:, :, :)
    DOUBLE PRECISION, ALLOCATABLE   :: periods_emax(:, :)

    DOUBLE PRECISION, ALLOCATABLE   :: likl(:)

    INTEGER :: i, period, counts(4), s

    INTEGER(our_int)                :: exp_a, idx, k, j
    INTEGER(our_int)                :: exp_b
    INTEGER(our_int)                :: edu
    INTEGER(our_int)                :: edu_lagged, choice
    
DOUBLE PRECISION :: payoffs_systematic(4), deviates(num_sims, 4), & 
    likl_contrib, eps, conditional_deviates(num_sims, 4), disturbances(4), & 
    total_payoffs(4), payoffs_ex_post(4), future_payoffs(4), choice_probabilities(4)

    LOGICAL :: is_working

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Solve the model for given parametrization
    CALL solve_fortran_bare(mapping_state_idx, periods_emax, & 
            periods_future_payoffs, periods_payoffs_ex_post, & 
            periods_payoffs_systematic, states_all, & 
            states_number_period, coeffs_a, coeffs_b, coeffs_edu, & 
            coeffs_home, shocks, edu_max, delta, edu_start, & 
            is_debug, is_interpolated, level, measure, min_idx, num_draws, & 
            num_periods, num_points, is_ambiguous, periods_eps_relevant)



    ALLOCATE(likl(num_agents * num_periods))
    likl = 0.0 ! TO BE REPLACED WITH DOUBLE


    j = 1   ! TODO: Align with FORTRAN

    DO i = 0, num_agents - 1

        DO period = 0, num_periods -1

                        ! Distribute state space
            exp_a = INT(data_array(j, 5))
            exp_b = INT(data_array(j, 6))
            edu = INT(data_array(j, 7))
            edu_lagged = INT(data_array(j, 8))

            ! TODO: Break naming convention
            choice = INT(data_array(j, 3))

            ! Transform total years of education to additional years of
            ! education and create an index from the choice.
            edu = edu - edu_start

            ! This is only done for aligment
            idx = choice

            ! TODO: There is not a clear decision made on whether to let index
            ! run from zero and work with increments + 1 for selection ...

            ! Get state indicator to obtain the systematic component of the
            ! agents payoffs. These feed into the simulation of choice
            ! probabilities.
            k = mapping_state_idx(period + 1, exp_a + 1, exp_b + 1, edu + 1, edu_lagged + 1)
            payoffs_systematic = periods_payoffs_systematic(period + 1, k + 1, :)

            ! Extract relevant deviates from standard normal distribution.
            deviates = standard_deviates(period + 1, :, :)

            ! Prepare to calculate product of likelihood contributions.
            likl_contrib = 1.0

            ! TODO: ALign across implementations
            is_working = (choice == 1) .OR. (choice == 2)

            ! If an agent is observed working, then the the labor market shocks
            ! are observed and the conditional distribution is used to determine
            ! the choice probabilities.
            IF (is_working) THEN

                ! Calculate the disturbance, which follows a normal
                ! distribution.
                eps = LOG(data_array(j, 4)) - LOG(payoffs_systematic(idx))
                
                ! Construct independent normal draws implied by the observed
                ! wages.
                IF (choice == 1) THEN
                    deviates(:, idx) = eps / sqrt(shocks(idx, idx))
                ELSE
                    deviates(:, idx) = (eps - eps_cholesky(idx, 1) * deviates(:, 1)) / eps_cholesky(idx, idx)
                END IF
                
                ! Record contribution of wage observation. REPLACE 0.0
                likl_contrib =  likl_contrib * normal_pdf(eps, DBLE(0.0), sqrt(shocks(idx, idx)))

            END IF

            ! Determine conditional deviates. These correspond to the
            ! unconditional draws if the agent did not work in the labor market.

            ! TODO: Consider writing a helper function that aligns impelemtations
            ! more.

            DO s = 1, num_sims

                conditional_deviates(s, :) = MATMUL(deviates(s, :), TRANSPOSE(eps_cholesky))
            END DO

            counts = 0

            DO s = 1, num_sims
                ! Extract deviates from (un-)conditional normal distributions.
                disturbances = conditional_deviates(s, :)

                disturbances(1) = exp(disturbances(1))
                disturbances(2) = exp(disturbances(2))

                ! Calculate total payoff.
                CALL get_total_value(total_payoffs, payoffs_ex_post, & 
                        future_payoffs, period, num_periods, delta, &
                        payoffs_systematic, disturbances, edu_max, edu_start, & 
                        mapping_state_idx, periods_emax, k, states_all)
                
                ! Record optimal choices
                counts(MAXLOC(total_payoffs)) = counts(MAXLOC(total_payoffs)) + 1

            END DO

            ! Determine relative shares
            choice_probabilities = counts / DBLE(num_sims)

            ! Adjust  and record likelihood contribution
            likl_contrib = likl_contrib * choice_probabilities(idx)
            likl(j) = likl_contrib

            
        j = j + 1

        END DO


    END DO 

    PRINT *, likl
    
    ! Scaling
    DO i = 1, num_agents * num_periods
        likl(i) = clip_value(likl(i), DBLE(1e-20), DBLE(1.0e10))
    END DO

    likl = log(likl)


    rslt = -SUM(likl) / DBLE(num_agents * num_periods)


END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_solve_fortran_bare(mapping_state_idx, periods_emax, & 
                periods_future_payoffs, periods_payoffs_ex_post, & 
                periods_payoffs_systematic, states_all, states_number_period, & 
                coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, &
                edu_max, delta, edu_start, is_debug, is_interpolated, &
                level, measure, min_idx, num_draws, num_periods, num_points, &
                is_ambiguous, periods_eps_relevant, max_states_period)
    
    !
    ! The presence of max_states_period breaks the equality of interfaces. 
    ! However, this is required so that the size of the return arguments is
    ! known from the beginning.
    !

    !/* external libraries      */

    USE robufort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(OUT)            :: mapping_state_idx(num_periods, num_periods, num_periods, min_idx, 2)
    INTEGER, INTENT(OUT)            :: states_number_period(num_periods)
    INTEGER, INTENT(OUT)            :: states_all(num_periods, max_states_period, 4)

    DOUBLE PRECISION, INTENT(OUT)   :: periods_payoffs_systematic(num_periods, max_states_period, 4)
    DOUBLE PRECISION, INTENT(OUT)   :: periods_payoffs_ex_post(num_periods, max_states_period, 4)
    DOUBLE PRECISION, INTENT(OUT)   :: periods_future_payoffs(num_periods, max_states_period, 4)
    DOUBLE PRECISION, INTENT(OUT)   :: periods_emax(num_periods, max_states_period)

    INTEGER, INTENT(IN)             :: max_states_period


    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_points
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: min_idx

    DOUBLE PRECISION, INTENT(IN)    :: periods_eps_relevant(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(:)
    DOUBLE PRECISION, INTENT(IN)    :: shocks(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(:)
    DOUBLE PRECISION, INTENT(IN)    :: level
    DOUBLE PRECISION, INTENT(IN)    :: delta 
 
    LOGICAL, INTENT(IN)             :: is_interpolated
    LOGICAL, INTENT(IN)             :: is_ambiguous
    LOGICAL, INTENT(IN)             :: is_debug

    CHARACTER(10), INTENT(IN)       :: measure

    !/* internal objects        */

        ! This container are required as output arguments cannot be of 
        ! assumed-shape type
    
    INTEGER, ALLOCATABLE            :: mapping_state_idx_int(:, :, :, :, :)
    INTEGER, ALLOCATABLE            :: states_number_period_int(:)
    INTEGER, ALLOCATABLE            :: states_all_int(:, :, :)

    DOUBLE PRECISION, ALLOCATABLE   :: periods_payoffs_systematic_int(:, :, :)
    DOUBLE PRECISION, ALLOCATABLE   :: periods_payoffs_ex_post_int(:, :, : )
    DOUBLE PRECISION, ALLOCATABLE   :: periods_future_payoffs_int(:, :, :)
    DOUBLE PRECISION, ALLOCATABLE   :: periods_emax_int(:, :)

!-------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------- 
   
    CALL solve_fortran_bare(mapping_state_idx_int, periods_emax_int, & 
            periods_future_payoffs_int, periods_payoffs_ex_post_int, & 
            periods_payoffs_systematic_int, states_all_int, & 
            states_number_period_int, coeffs_a, coeffs_b, coeffs_edu, & 
            coeffs_home, shocks, edu_max, delta, edu_start, & 
            is_debug, is_interpolated, level, measure, min_idx, num_draws, & 
            num_periods, num_points, is_ambiguous, periods_eps_relevant)

    ! Assign to initial objects for return to PYTHON
    periods_payoffs_systematic = periods_payoffs_systematic_int   
    periods_payoffs_ex_post = periods_payoffs_ex_post_int  
    periods_future_payoffs = periods_future_payoffs_int  
    states_number_period = states_number_period_int 
    mapping_state_idx = mapping_state_idx_int 
    periods_emax = periods_emax_int 
    states_all = states_all_int

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_normal_pdf(rslt, x, mean, sd)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)      :: rslt

    DOUBLE PRECISION, INTENT(IN)       :: x
    DOUBLE PRECISION, INTENT(IN)       :: mean
    DOUBLE PRECISION, INTENT(IN)       :: sd

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt = normal_pdf(x, mean, sd)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_pinv(rslt, A, m)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(m, m)

    DOUBLE PRECISION, INTENT(IN)    :: A(m, m)
    
    INTEGER, INTENT(IN)             :: m

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Get Pseudo-inverse
    rslt = pinv(A, m)
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_svd(U, S, VT, A, m)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: S(m) 
    DOUBLE PRECISION, INTENT(OUT)   :: U(m, m)
    DOUBLE PRECISION, INTENT(OUT)   :: VT(m, m)

    DOUBLE PRECISION, INTENT(IN)    :: A(m, m)
    
    INTEGER, INTENT(IN)             :: m

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Get Singular-Value-Decomposition
    CALL svd(U, S, VT, A, m)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_payoffs_ambiguity(emax_simulated, payoffs_ex_post, &
                future_payoffs, num_draws, eps_relevant, period, k, &
                payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
                states_all, num_periods, periods_emax, delta, is_debug, &
                shocks, level, measure)

    !/* external libraries    */

    USE robufort_ambiguity

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: payoffs_ex_post(4)
    DOUBLE PRECISION, INTENT(OUT)   :: future_payoffs(4)
    DOUBLE PRECISION, INTENT(OUT)   :: emax_simulated

    INTEGER, INTENT(IN)             :: mapping_state_idx(:,:,:,:,:)
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

    DOUBLE PRECISION, INTENT(IN)    :: payoffs_systematic(:)
    DOUBLE PRECISION, INTENT(IN)    :: shocks(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: eps_relevant(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: delta
    DOUBLE PRECISION, INTENT(IN)    :: level

    LOGICAL, INTENT(IN)             :: is_debug

    CHARACTER(10), INTENT(IN)       :: measure

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Get the expected payoffs under ambiguity
    CALL get_payoffs_ambiguity(emax_simulated, payoffs_ex_post, &
                future_payoffs, num_draws, eps_relevant, period, k, & 
                payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
                states_all, num_periods, periods_emax, delta, is_debug, &
                shocks, level, measure)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_criterion_approx_gradient(rslt, x, eps, num_draws, &
                eps_relevant, period, k, payoffs_systematic, edu_max, &
                edu_start, mapping_state_idx, states_all, num_periods, &
                periods_emax, delta)

    !/* external libraries    */

    USE robufort_ambiguity

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(2)

    DOUBLE PRECISION, INTENT(IN)    :: eps_relevant(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: payoffs_systematic(:)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: delta
    DOUBLE PRECISION, INTENT(IN)    :: x(:)
    DOUBLE PRECISION, INTENT(IN)    :: eps

    INTEGER, INTENT(IN)             :: mapping_state_idx(:,:,:,:,:)
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Approximate the gradient of the criterion function
    rslt = criterion_approx_gradient(x, eps, num_draws, eps_relevant, &
            period, k, payoffs_systematic, edu_max, edu_start, mapping_state_idx, &
            states_all, num_periods, periods_emax, delta)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_simulate_emax(emax_simulated, payoffs_ex_post, &
                future_payoffs, num_periods, num_draws, period, k, &
                eps_relevant_emax, payoffs_systematic, edu_max, edu_start, &
                periods_emax, states_all, mapping_state_idx, delta)

    !/* external libraries    */

    USE robufort_emax

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: payoffs_ex_post(4)
    DOUBLE PRECISION, INTENT(OUT)   :: future_payoffs(4)
    DOUBLE PRECISION, INTENT(OUT)   :: emax_simulated

    DOUBLE PRECISION, INTENT(IN)    :: eps_relevant_emax(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: payoffs_systematic(:)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: delta

    INTEGER, INTENT(IN)             :: mapping_state_idx(:,:,:,:,:)
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Simulate expected future value
    CALL simulate_emax(emax_simulated, payoffs_ex_post, future_payoffs, &
            num_periods, num_draws, period, k, eps_relevant_emax, & 
            payoffs_systematic, edu_max, edu_start, periods_emax, states_all, &
            mapping_state_idx, delta)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_criterion(emax_simulated, x, num_draws, eps_relevant, &
                period, k, payoffs_systematic, edu_max, edu_start, & 
                mapping_state_idx, states_all, num_periods, periods_emax, & 
                delta)

    !/* external libraries    */

    USE robufort_ambiguity

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: emax_simulated

    DOUBLE PRECISION, INTENT(IN)    :: eps_relevant(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: payoffs_systematic(:)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: delta
    DOUBLE PRECISION, INTENT(IN)    :: x(:)

    INTEGER , INTENT(IN)            :: mapping_state_idx(:,:,:,:,:)
    INTEGER , INTENT(IN)            :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    emax_simulated = criterion(x, num_draws, eps_relevant, period, k, &
                        payoffs_systematic, edu_max, edu_start, &
                        mapping_state_idx, states_all, num_periods, &
                        periods_emax, delta)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_divergence_approx_gradient(rslt, x, cov, level, eps)

    !/* external libraries    */

    USE robufort_ambiguity

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)  :: rslt(2)
    DOUBLE PRECISION, INTENT(IN)   :: x(2)
    DOUBLE PRECISION, INTENT(IN)   :: eps
    DOUBLE PRECISION, INTENT(IN)   :: cov(4,4)
    DOUBLE PRECISION, INTENT(IN)   :: level

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Approximate the gradient of the KL divergence
    rslt = divergence_approx_gradient(x, cov, level, eps)

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_multivariate_normal(draws, mean, covariance, num_draws, dim)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    INTEGER, INTENT(IN)             :: num_draws
    INTEGER, INTENT(IN)             :: dim

    DOUBLE PRECISION, INTENT(OUT)   :: draws(num_draws, dim)
    DOUBLE PRECISION, INTENT(IN)    :: mean(dim)
    DOUBLE PRECISION, INTENT(IN)    :: covariance(dim, dim)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Generate multivariate normal deviates    
    CALL multivariate_normal(draws, mean, covariance)
    
END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_standard_normal(draw, dim)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    INTEGER, INTENT(IN)             :: dim
    
    DOUBLE PRECISION, INTENT(OUT)   :: draw(dim)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Generate standard normal deviates
    CALL standard_normal(draw)

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_determinant(det, A)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: det

    DOUBLE PRECISION, INTENT(IN)    :: A(:, :)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Get determinant
    det = determinant(A)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_cholesky(factor, matrix, n)

    !/* external libraries    */

    USE robufort_auxiliary
    
    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: factor(n,n)

    DOUBLE PRECISION, INTENT(IN)    :: matrix(:,:)

    INTEGER, INTENT(IN)             :: n

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Get Cholesky decomposition
    CALL cholesky(factor, matrix)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_inverse(inv, A, n)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: inv(n, n)

    DOUBLE PRECISION, INTENT(IN)    :: A(:, :)

    INTEGER, INTENT(IN)             :: n

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Get inverse
    inv = inverse(A, n)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_trace(rslt, A)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT) :: rslt

    DOUBLE PRECISION, INTENT(IN)  :: A(:,:)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Get trace
    rslt = trace_fun(A)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_divergence(div, x, cov, level)

    !/* external libraries    */

    USE robufort_ambiguity

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: div(1)

    DOUBLE PRECISION, INTENT(IN)    :: x(2)
    DOUBLE PRECISION, INTENT(IN)    :: cov(4,4)
    DOUBLE PRECISION, INTENT(IN)    :: level

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Calculate divergence
    div = divergence(x, cov, level)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_clipped_vector(Y, X, lower_bound, upper_bound, num_values)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: Y(num_values)

    DOUBLE PRECISION, INTENT(IN)    :: X(num_values)
    DOUBLE PRECISION, INTENT(IN)    :: lower_bound
    DOUBLE PRECISION, INTENT(IN)    :: upper_bound 

    INTEGER, INTENT(IN)             :: num_values

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL get_clipped_vector(Y, X, lower_bound, upper_bound, num_values)


END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_r_squared(r_squared, Y, P, num_agents)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: r_squared

    DOUBLE PRECISION, INTENT(IN)    :: Y(num_agents)
    DOUBLE PRECISION, INTENT(IN)    :: P(num_agents)
    
    INTEGER, INTENT(IN)              :: num_agents

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL get_r_squared(r_squared, Y, P, num_agents)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_point_predictions(Y, X, coeffs, num_agents)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)       :: Y(num_agents)

    DOUBLE PRECISION, INTENT(IN)        :: coeffs(:)
    DOUBLE PRECISION, INTENT(IN)        :: X(:,:)
    
    INTEGER, INTENT(IN)                 :: num_agents

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL point_predictions(Y, X, coeffs, num_agents)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_predictions(predictions, endogenous, exogenous, maxe, & 
                is_simulated, num_points, num_states)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)               :: predictions(num_states)

    DOUBLE PRECISION, INTENT(IN)                :: exogenous(:, :)
    DOUBLE PRECISION, INTENT(IN)                :: endogenous(:)
    DOUBLE PRECISION, INTENT(IN)                :: maxe(:)

    INTEGER, INTENT(IN)                         :: num_states
    INTEGER, INTENT(IN)                         :: num_points

    LOGICAL, INTENT(IN)                         :: is_simulated(:)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL get_predictions(predictions, endogenous, exogenous, maxe, & 
            is_simulated, num_points, num_states)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_random_choice(sample, candidates, num_candidates, num_points)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    INTEGER, INTENT(OUT)            :: sample(num_points)

    INTEGER, INTENT(IN)             :: candidates(:)
    INTEGER, INTENT(IN)             :: num_candidates
    INTEGER, INTENT(IN)             :: num_points

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

     CALL random_choice(sample, candidates, num_candidates, num_points)


END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_coefficients(coeffs, Y, X, num_covars, num_agents)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: coeffs(num_covars)

    DOUBLE PRECISION, INTENT(IN)    :: Y(:)
    DOUBLE PRECISION, INTENT(IN)    :: X(:,:)
    
    INTEGER, INTENT(IN)             :: num_covars
    INTEGER, INTENT(IN)             :: num_agents

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL get_coefficients(coeffs, Y, X, num_covars, num_agents)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_endogenous_variable(exogenous_variable, period, &
                num_periods, num_states, delta, periods_payoffs_systematic, &
                edu_max, edu_start, mapping_state_idx, periods_emax, &
                states_all, is_simulated, num_draws, shocks, level, & 
                is_ambiguous, is_debug, measure, maxe, eps_relevant)

    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)       :: exogenous_variable(num_states)

    DOUBLE PRECISION, INTENT(IN)        :: periods_payoffs_systematic(:, :, :)
    DOUBLE PRECISION, INTENT(IN)        :: periods_emax(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: eps_relevant(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: shocks(4, 4)
    DOUBLE PRECISION, INTENT(IN)        :: maxe(:)
    DOUBLE PRECISION, INTENT(IN)        :: delta
    DOUBLE PRECISION, INTENT(IN)        :: level
 
    INTEGER, INTENT(IN)                 :: mapping_state_idx(:, :, :, :, :)    
    INTEGER, INTENT(IN)                 :: states_all(:, :, :)    
    INTEGER, INTENT(IN)                 :: num_periods
    INTEGER, INTENT(IN)                 :: num_states
    INTEGER, INTENT(IN)                 :: edu_start
    INTEGER, INTENT(IN)                 :: num_draws
    INTEGER, INTENT(IN)                 :: edu_max
    INTEGER, INTENT(IN)                 :: period

    LOGICAL, INTENT(IN)                 :: is_simulated(:)
    LOGICAL, INTENT(IN)                 :: is_ambiguous
    LOGICAL, INTENT(IN)                 :: is_debug
    
    CHARACTER(10), INTENT(IN)           :: measure

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL get_endogenous_variable(exogenous_variable, period, num_periods, &
            num_states, delta, periods_payoffs_systematic, edu_max, &
            edu_start, mapping_state_idx, periods_emax, states_all, &
            is_simulated, num_draws, shocks, level, is_ambiguous, is_debug, & 
            measure, maxe, eps_relevant)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_exogenous_variables(independent_variables, maxe, &
                period, num_periods, num_states, delta, & 
                periods_payoffs_systematic, shifts, edu_max, edu_start, &
                mapping_state_idx, periods_emax, states_all)

    !/* external libraries    */

    USE robufort_emax

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)        :: independent_variables(num_states, 9)
    DOUBLE PRECISION, INTENT(OUT)        :: maxe(num_states)


    DOUBLE PRECISION, INTENT(IN)        :: periods_payoffs_systematic(:, :, :)
    DOUBLE PRECISION, INTENT(IN)        :: periods_emax(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: shifts(:)
    DOUBLE PRECISION, INTENT(IN)        :: delta
 
    INTEGER, INTENT(IN)                 :: mapping_state_idx(:, :, :, :, :)    
    INTEGER, INTENT(IN)                 :: states_all(:, :, :)    
    INTEGER, INTENT(IN)                 :: num_periods
    INTEGER, INTENT(IN)                 :: num_states
    INTEGER, INTENT(IN)                 :: edu_start
    INTEGER, INTENT(IN)                 :: edu_max
    INTEGER, INTENT(IN)                 :: period

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL get_exogenous_variables(independent_variables, maxe,  period, &
            num_periods, num_states, delta, periods_payoffs_systematic, &
            shifts, edu_max, edu_start, mapping_state_idx, periods_emax, &
            states_all)
            
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_simulated_indicator(is_simulated, num_points, & 
                num_states, period, num_periods, is_debug)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    LOGICAL, INTENT(OUT)            :: is_simulated(num_states)

    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_states
    INTEGER, INTENT(IN)             :: num_points
    INTEGER, INTENT(IN)             :: period

    LOGICAL, INTENT(IN)             :: is_debug

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    is_simulated = get_simulated_indicator(num_points, num_states, period, & 
                        num_periods, is_debug)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_payoffs(emax_simulated, payoffs_ex_post, future_payoffs, &
                num_draws, eps_relevant, period, k, payoffs_systematic, & 
                edu_max, edu_start, mapping_state_idx, states_all, num_periods, & 
                periods_emax, delta, is_debug, shocks, level, is_ambiguous, & 
                measure)


    !/* external libraries    */

    USE robufort_library

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)       :: emax_simulated
    DOUBLE PRECISION, INTENT(OUT)       :: payoffs_ex_post(4)
    DOUBLE PRECISION, INTENT(OUT)       :: future_payoffs(4)

    DOUBLE PRECISION, INTENT(IN)        :: payoffs_systematic(:)
    DOUBLE PRECISION, INTENT(IN)        :: eps_relevant(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: periods_emax(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: shocks(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: delta
    DOUBLE PRECISION, INTENT(IN)        :: level

    INTEGER, INTENT(IN)                 :: mapping_state_idx(:, :, :, :, :)
    INTEGER, INTENT(IN)                 :: states_all(:, :, :)
    INTEGER, INTENT(IN)                 :: num_periods
    INTEGER, INTENT(IN)                 :: num_draws
    INTEGER, INTENT(IN)                 :: edu_max
    INTEGER, INTENT(IN)                 :: edu_start
    INTEGER, INTENT(IN)                 :: period
    INTEGER, INTENT(IN)                 :: k 

    LOGICAL, INTENT(IN)                 :: is_ambiguous
    LOGICAL, INTENT(IN)                 :: is_debug

    CHARACTER(10), INTENT(IN)           :: measure

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL get_payoffs(emax_simulated, payoffs_ex_post, future_payoffs, &
                num_draws, eps_relevant, period, k, payoffs_systematic, & 
                edu_max, edu_start, mapping_state_idx, states_all, &
                num_periods, periods_emax, delta, is_debug, shocks, level, &
                is_ambiguous, measure)
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
