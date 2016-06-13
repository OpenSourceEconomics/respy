MODULE shared_containers

    ! This module declares all those variables that are required as part of the evaluation of the criterion function.

    USE shared_constants

    !/* setup                   */

    IMPLICIT NONE

!******************************************************************************
!******************************************************************************

    INTEGER(our_int), ALLOCATABLE   :: mapping_state_idx(:, :, :, :, :)
    INTEGER(our_int), ALLOCATABLE   :: states_number_period(:)
    INTEGER(our_int), ALLOCATABLE   :: states_all(:, :, :)

    REAL(our_dble), ALLOCATABLE     :: periods_payoffs_systematic(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_prob(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_draws_emax(:, :, :)
    REAL(our_dble), ALLOCATABLE     :: periods_emax(:, :)
    REAL(our_dble), ALLOCATABLE     :: data_est(:, :)

    REAL(our_dble)                  :: delta

!******************************************************************************
!******************************************************************************
END MODULE 