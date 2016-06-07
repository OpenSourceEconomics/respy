MODULE shared_containers

    USE shared_constants

    !/* setup                   */

    IMPLICIT NONE

!******************************************************************************
!******************************************************************************

    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx(:, :, :, :, :)

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: data_array(:, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax(:, :)

!******************************************************************************
!******************************************************************************
END MODULE 