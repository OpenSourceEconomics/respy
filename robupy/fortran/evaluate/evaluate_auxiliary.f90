!*******************************************************************************
!*******************************************************************************
MODULE evaluate_auxiliary

	!/*	external modules	*/

    USE robufort_constants

    USE shared_auxiliary

	!/*	setup	*/

	IMPLICIT NONE

    PUBLIC

CONTAINS
!*******************************************************************************
!*******************************************************************************
PURE FUNCTION normal_pdf(x, mean, sd)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: mean
    REAL(our_dble), INTENT(IN)      :: sd
    REAL(our_dble), INTENT(IN)      :: x

    !/*  internal objects       */

    REAL(our_dble)                  :: normal_pdf
    REAL(our_dble)                  :: std

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    std = ((x - mean) / sd)

    normal_pdf = (one_dble / sd) * (one_dble / sqrt(two_dble * pi))

    normal_pdf = normal_pdf * exp( -(std * std) / two_dble)

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION clip_value(value, lower_bound, upper_bound)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)  :: lower_bound
    REAL(our_dble), INTENT(IN)  :: upper_bound
    REAL(our_dble), INTENT(IN)  :: value

    !/*  internal objects       */

    REAL(our_dble)              :: clip_value

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF(value < lower_bound) THEN

        clip_value = lower_bound

    ELSEIF(value > upper_bound) THEN

        clip_value = upper_bound

    ELSE

        clip_value = value

    END IF

END FUNCTION
!*******************************************************************************
!*******************************************************************************
END MODULE