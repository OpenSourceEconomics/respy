MODULE parallel_constants
    
    !/* external modules    */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE
    
!------------------------------------------------------------------------------
!   Parallel Inputs
!------------------------------------------------------------------------------

    INTEGER(our_int)            :: ierr
    INTEGER(our_int)            :: SLAVECOMM
    INTEGER(our_int)            :: PARENTCOMM
        
!******************************************************************************
!******************************************************************************
END MODULE 