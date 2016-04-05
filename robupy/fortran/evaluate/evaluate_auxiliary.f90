!*******************************************************************************
!*******************************************************************************
MODULE evaluate_auxiliary

	!/*	external modules	*/

    USE shared_constants

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
!*******************************************************************************
!*******************************************************************************
END MODULE