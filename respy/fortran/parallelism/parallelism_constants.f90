MODULE parallelism_constants

    !/* external modules    */

    USE resfort_library

    USE mpi

    !/* setup                   */

    IMPLICIT NONE

!------------------------------------------------------------------------------
!   Parallel Inputs
!------------------------------------------------------------------------------

    INTEGER(our_int)            :: PARENTCOMM
    INTEGER(our_int)            :: SLAVECOMM
    INTEGER(our_int)            :: num_slaves
    INTEGER(our_int)            :: ierr
    INTEGER(our_int)            :: rank

    INTEGER(our_int)            :: status(MPI_STATUS_SIZE)

!******************************************************************************
!******************************************************************************
END MODULE
