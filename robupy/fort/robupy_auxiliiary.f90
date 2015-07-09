MODULE robupy_auxiliary

	!/*	external modules	*/

    USE robupy_program_constants

	!/*	setup	*/

	IMPLICIT NONE

    PRIVATE

    PUBLIC ::   get_future_payoffs_lib

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE get_future_payoffs_lib(future_payoffs, edu_max, edu_start, &
        mapping_state_idx, period, emax, k, states_all)

    !/* external libraries    */

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: future_payoffs(4)

    DOUBLE PRECISION, INTENT(IN)    :: emax(:,:)

    INTEGER, INTENT(IN)             :: k
    INTEGER, INTENT(IN)             :: period
    INTEGER, INTENT(IN)             :: edu_max
    INTEGER, INTENT(IN)             :: edu_start
    INTEGER, INTENT(IN)             :: states_all(:,:,:)
    INTEGER, INTENT(IN)             :: mapping_state_idx(:,:,:,:,:)

    !/* internals objects    */

    INTEGER(our_int)    :: exp_A
    INTEGER(our_int)    :: exp_B
    INTEGER(our_int)    :: edu
    INTEGER(our_int)    :: edu_lagged
    INTEGER(our_int)    :: future_idx

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

! Distribute state space
exp_A = states_all(period + 1, k + 1, 1)
exp_B = states_all(period + 1, k + 1, 2)
edu = states_all(period + 1, k + 1, 3)
edu_lagged = states_all(period + 1, k + 1, 4)

! Working in occupation A
future_idx = mapping_state_idx(period + 1 + 1, exp_A + 1 + 1, exp_B + 1, edu + 1, 1)
future_payoffs(1) = emax(period + 1 + 1, future_idx + 1)

!Working in occupation B
future_idx = mapping_state_idx(period + 1 + 1, exp_A + 1, exp_B + 1 + 1, edu + 1, 1)
future_payoffs(2) = emax(period + 1 + 1, future_idx + 1)

! Increasing schooling. Note that adding an additional year
! of schooling is only possible for those that have strictly
! less than the maximum level of additional education allowed.
IF (edu < edu_max - edu_start) THEN
    future_idx = mapping_state_idx(period + 1 + 1, exp_A + 1, exp_B + 1, edu + 1 + 1, 2)
    future_payoffs(3) = emax(period + 1 + 1, future_idx + 1)
ELSE
    future_payoffs(3) = -HUGE(future_payoffs)
END IF

! Staying at home
future_idx = mapping_state_idx(period + 1 + 1, exp_A + 1, exp_B + 1, edu + 1, 1)
future_payoffs(4) = emax(period + 1 + 1, future_idx + 1)

END SUBROUTINE
END MODULE
