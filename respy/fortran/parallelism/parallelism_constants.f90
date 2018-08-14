MODULE parallelism_constants

#IF MPI_AVAILABLE

    !/* external modules    */

    USE shared_interface

    USE mpi

    !/* setup                   */

    IMPLICIT NONE

!------------------------------------------------------------------------------
!   Parallel Inputs
!------------------------------------------------------------------------------

    INTEGER(our_int)            :: status(MPI_STATUS_SIZE)
    INTEGER(our_int)            :: PARENTCOMM
    INTEGER(our_int)            :: SLAVECOMM
    INTEGER(our_int)            :: ierr
    INTEGER(our_int)            :: rank

!******************************************************************************
!******************************************************************************
#ENDIF

END MODULE
