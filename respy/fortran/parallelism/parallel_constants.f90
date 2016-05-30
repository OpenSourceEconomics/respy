MODULE parallel_constants
    
    !/* external modules    */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE
    
!------------------------------------------------------------------------------
!   Parallel Inputs
!------------------------------------------------------------------------------
    
    ! MPI Constants
    INTEGER(our_int)            :: ierr
    INTEGER(our_int)            :: SLAVECOMM

    ! Auxiliary variables that allow for explicit shape arrays.

!******************************************************************************
!******************************************************************************
END MODULE 