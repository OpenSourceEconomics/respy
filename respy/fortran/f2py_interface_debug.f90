!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_normal_pdf(rslt, x, mean, sd)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

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

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(m, m)

    DOUBLE PRECISION, INTENT(IN)    :: A(m, m)
    
    INTEGER, INTENT(IN)             :: m

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt = pinv(A, m)
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_svd(U, S, VT, A, m)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: S(m) 
    DOUBLE PRECISION, INTENT(OUT)   :: U(m, m)
    DOUBLE PRECISION, INTENT(OUT)   :: VT(m, m)

    DOUBLE PRECISION, INTENT(IN)    :: A(m, m)
    
    INTEGER, INTENT(IN)             :: m

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL svd(U, S, VT, A, m)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_simulate_emax(emax_simulated, num_periods, & 
                num_draws_emax_int, period, k, draws_emax, payoffs_systematic, & 
                edu_max_int, edu_start, periods_emax, states_all, & 
                mapping_state_idx, delta_int, shocks_cholesky)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: emax_simulated

    DOUBLE PRECISION, INTENT(IN)    :: payoffs_systematic(:)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_emax(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: draws_emax(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: delta_int

    INTEGER, INTENT(IN)             :: mapping_state_idx(:,:,:,:,:)
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: edu_max_int
    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: k

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# Transfer auxiliary variable to global variable.
    num_draws_emax = num_draws_emax_int
    edu_max = edu_max_int
    delta = delta_int

    CALL simulate_emax(emax_simulated, num_periods, period, k, & 
            draws_emax, payoffs_systematic, edu_start, periods_emax, & 
            states_all, mapping_state_idx, shocks_cholesky)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_standard_normal(draw, dim)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    INTEGER, INTENT(IN)             :: dim
    
    DOUBLE PRECISION, INTENT(OUT)   :: draw(dim)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL standard_normal(draw)

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_determinant(det, A)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: det

    DOUBLE PRECISION, INTENT(IN)    :: A(:, :)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    det = determinant(A)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_inverse(inv, A, n)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: inv(n, n)

    DOUBLE PRECISION, INTENT(IN)    :: A(:, :)

    INTEGER, INTENT(IN)             :: n

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    inv = inverse(A, n)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_trace(rslt, A)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT) :: rslt

    DOUBLE PRECISION, INTENT(IN)  :: A(:,:)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    rslt = trace_fun(A)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_clip_value(clipped_value, value, lower_bound, upper_bound, & 
            num_values)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: clipped_value(num_values)

    DOUBLE PRECISION, INTENT(IN)        :: value(:)
    DOUBLE PRECISION, INTENT(IN)        :: lower_bound
    DOUBLE PRECISION, INTENT(IN)        :: upper_bound 

    INTEGER, INTENT(IN)                 :: num_values

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    clipped_value = clip_value(value, lower_bound, upper_bound)


END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_pred_info(r_squared, bse, Y, P, X, num_states, & 
                num_covars)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: bse(num_covars)
    DOUBLE PRECISION, INTENT(OUT)   :: r_squared
    
    DOUBLE PRECISION, INTENT(IN)    :: X(num_states, num_covars)
    DOUBLE PRECISION, INTENT(IN)    :: Y(num_states)
    DOUBLE PRECISION, INTENT(IN)    :: P(num_states)
    
    INTEGER, INTENT(IN)             :: num_states
    INTEGER, INTENT(IN)             :: num_covars

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL get_pred_info(r_squared, bse, Y, P, X, num_states, num_covars)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_point_predictions(Y, X, coeffs, num_states)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: Y(num_states)

    DOUBLE PRECISION, INTENT(IN)        :: coeffs(:)
    DOUBLE PRECISION, INTENT(IN)        :: X(:,:)
    
    INTEGER, INTENT(IN)                 :: num_states

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL point_predictions(Y, X, coeffs, num_states)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_predictions(predictions, endogenous, exogenous, maxe, & 
                is_simulated, num_points, num_states)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

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

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

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
SUBROUTINE wrapper_get_coefficients(coeffs, Y, X, num_covars, num_states)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)   :: coeffs(num_covars)

    DOUBLE PRECISION, INTENT(IN)    :: X(:,:)
    DOUBLE PRECISION, INTENT(IN)    :: Y(:)
    
    INTEGER, INTENT(IN)             :: num_covars
    INTEGER, INTENT(IN)             :: num_states

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    CALL get_coefficients(coeffs, Y, X, num_covars, num_states)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_endogenous_variable(exogenous_variable, period, &
                num_periods, num_states, delta_int, periods_payoffs_systematic, &
                edu_max_int, edu_start, mapping_state_idx, periods_emax, &
                states_all, is_simulated, num_draws_emax_int, maxe, draws_emax, & 
                shocks_cholesky)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: exogenous_variable(num_states)

    DOUBLE PRECISION, INTENT(IN)        :: periods_payoffs_systematic(:, :, :)
    DOUBLE PRECISION, INTENT(IN)        :: shocks_cholesky(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: periods_emax(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: draws_emax(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: maxe(:)
    DOUBLE PRECISION, INTENT(IN)        :: delta_int

    INTEGER, INTENT(IN)                 :: mapping_state_idx(:, :, :, :, :)    
    INTEGER, INTENT(IN)                 :: states_all(:, :, :)    
    INTEGER, INTENT(IN)                 :: num_draws_emax_int
    INTEGER, INTENT(IN)                 :: edu_max_int    
    INTEGER, INTENT(IN)                 :: num_periods
    INTEGER, INTENT(IN)                 :: num_states
    INTEGER, INTENT(IN)                 :: edu_start
    INTEGER, INTENT(IN)                 :: period


    LOGICAL, INTENT(IN)                 :: is_simulated(:)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    num_draws_emax = num_draws_emax_int
    edu_max = edu_max_int
    delta = delta_int

    CALL get_endogenous_variable(exogenous_variable, period, num_periods, &
            num_states, periods_payoffs_systematic, &
            edu_start, mapping_state_idx, periods_emax, states_all, &
            is_simulated, maxe, draws_emax, shocks_cholesky)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_exogenous_variables(independent_variables, maxe, &
                period, num_periods, num_states, delta_int, &
                periods_payoffs_systematic, shifts, edu_max_int, edu_start, &
                mapping_state_idx, periods_emax, states_all)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)        :: independent_variables(num_states, 9)
    DOUBLE PRECISION, INTENT(OUT)        :: maxe(num_states)


    DOUBLE PRECISION, INTENT(IN)        :: periods_payoffs_systematic(:, :, :)
    DOUBLE PRECISION, INTENT(IN)        :: periods_emax(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: shifts(:)
    DOUBLE PRECISION, INTENT(IN)        :: delta_int

    INTEGER, INTENT(IN)                 :: mapping_state_idx(:, :, :, :, :)    
    INTEGER, INTENT(IN)                 :: states_all(:, :, :)    
    INTEGER, INTENT(IN)                 :: edu_max_int
    INTEGER, INTENT(IN)                 :: num_periods
    INTEGER, INTENT(IN)                 :: num_states
    INTEGER, INTENT(IN)                 :: edu_start
    INTEGER, INTENT(IN)                 :: period

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# Transfer auxiliary variable to global variable.
    edu_max = edu_max_int
    delta = delta_int

    CALL get_exogenous_variables(independent_variables, maxe,  period, &
            num_periods, num_states, periods_payoffs_systematic, &
            shifts, edu_start, mapping_state_idx, periods_emax, &
            states_all)
            
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_simulated_indicator(is_simulated, num_points, & 
                num_states, period, is_debug_int, num_periods)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    LOGICAL, INTENT(OUT)            :: is_simulated(num_states)

    INTEGER, INTENT(IN)             :: num_periods
    INTEGER, INTENT(IN)             :: num_states
    INTEGER, INTENT(IN)             :: num_points
    INTEGER, INTENT(IN)             :: period

    LOGICAL, INTENT(IN)             :: is_debug_int

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    is_debug = is_debug_int
    is_simulated = get_simulated_indicator(num_points, num_states, period, & 
                        num_periods)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE wrapper_get_payoffs(emax_simulated, num_draws_emax_int, & 
                draws_emax, period, k, payoffs_systematic, edu_max_int, & 
                edu_start, mapping_state_idx, states_all, num_periods, & 
                periods_emax, delta_int, shocks_cholesky)


    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */

    DOUBLE PRECISION, INTENT(OUT)       :: emax_simulated

    DOUBLE PRECISION, INTENT(IN)        :: payoffs_systematic(:)
    DOUBLE PRECISION, INTENT(IN)        :: shocks_cholesky(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: periods_emax(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: draws_emax(:, :)
    DOUBLE PRECISION, INTENT(IN)        :: delta_int

    INTEGER, INTENT(IN)                 :: mapping_state_idx(:, :, :, :, :)
    INTEGER, INTENT(IN)                 :: states_all(:, :, :)
    INTEGER, INTENT(IN)                 :: num_draws_emax_int
    INTEGER, INTENT(IN)                 :: edu_max_int
    INTEGER, INTENT(IN)                 :: num_periods
    INTEGER, INTENT(IN)                 :: edu_start
    INTEGER, INTENT(IN)                 :: period
    INTEGER, INTENT(IN)                 :: k 
    
!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !# Transfer auxiliary variable to global variable.
    num_draws_emax = num_draws_emax_int
    edu_max = edu_max_int
    delta = delta_int

    CALL get_payoffs(emax_simulated, draws_emax, period, k, & 
            payoffs_systematic, edu_start, mapping_state_idx, & 
            states_all, num_periods, periods_emax, shocks_cholesky)
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
