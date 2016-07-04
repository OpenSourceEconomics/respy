!******************************************************************************
!******************************************************************************
MODULE evaluate_auxiliary

	!/*	external modules	  */

    USE shared_constants

    USE shared_auxiliary

	!/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
FUNCTION get_smoothed_probability(total_payoffs, idx, tau)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: idx

    REAL(our_dble), INTENT(IN)      :: total_payoffs(4)
    REAL(our_dble), INTENT(IN)      :: tau

    !/*  internal objects       */

    INTEGER(our_int), ALLOCATABLE   :: infos(:)
    
    REAL(our_dble)                  :: get_smoothed_probability
    REAL(our_dble)                  :: smoot_payoff(4)
    REAL(our_dble)                  :: maxim_payoff(4)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    maxim_payoff = MAXVAL(total_payoffs)

    CALL clip_value(smoot_payoff, EXP((total_payoffs - maxim_payoff)/tau), zero_dble, HUGE_FLOAT, infos)

    get_smoothed_probability = smoot_payoff(idx) / SUM(smoot_payoff)

END FUNCTION
!******************************************************************************
!******************************************************************************
PURE FUNCTION normal_pdf(x, mean, sd)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: mean
    REAL(our_dble), INTENT(IN)      :: sd
    REAL(our_dble), INTENT(IN)      :: x

    !/*  internal objects       */

    REAL(our_dble)                  :: normal_pdf
    REAL(our_dble)                  :: std

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    std = ((x - mean) / sd)

    normal_pdf = (one_dble / sd) * (one_dble / sqrt(two_dble * pi))

    normal_pdf = normal_pdf * exp( -(std * std) / two_dble)

END FUNCTION
!******************************************************************************
!******************************************************************************
END MODULE