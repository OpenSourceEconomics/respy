MODULE parallel_constants
    
    !/* external modules    */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE
    
!------------------------------------------------------------------------------
!   Parallel Inputs
!------------------------------------------------------------------------------
    
    ! MPI Variables
    INTEGER(our_int)            :: PARENTCOMM
    INTEGER(our_int)            :: SLAVECOMM
    INTEGER(our_int)            :: num_slaves
    INTEGER(our_int)            :: status
    INTEGER(our_int)            :: ierr
    INTEGER(our_int)            :: rank
        
!******************************************************************************
!******************************************************************************
END MODULE 