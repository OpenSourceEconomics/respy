!*******************************************************************************
!*******************************************************************************
!
!   This subroutine is just a wrapper for selected functions of the ROBUFORT 
!   library. Its sole purpose is to serve as a wrapper for debugging purposes.
!
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
                shifts, edu_max, edu_start, mapping_state_idx, periods_emax, &
                states_all, is_simulated, num_draws, shocks, level, is_debug, & 
                measure, maxe, eps_relevant)

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
    DOUBLE PRECISION, INTENT(IN)        :: shifts(:)
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
    LOGICAL, INTENT(IN)                 :: is_debug
    
    CHARACTER(10), INTENT(IN)           :: measure

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL get_endogenous_variable(exogenous_variable, period, num_periods, &
            num_states, delta, periods_payoffs_systematic, shifts, edu_max, &
            edu_start, mapping_state_idx, periods_emax, states_all, &
            is_simulated, num_draws, shocks, level, is_debug, measure, maxe, &
            eps_relevant)

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
                periods_emax, delta, is_debug, shocks, level, measure)


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

    LOGICAL, INTENT(IN)                 :: is_debug

    CHARACTER(10), INTENT(IN)           :: measure

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL get_payoffs(emax_simulated, payoffs_ex_post, future_payoffs, &
                num_draws, eps_relevant, period, k, payoffs_systematic, & 
                edu_max, edu_start, mapping_state_idx, states_all, &
                num_periods, periods_emax, delta, is_debug, shocks, level, &
                measure)
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
