MODULE parallelism_constants

#if MPI_AVAILABLE

    !/* external modules    */

    USE shared_constants

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
#endif

END MODULE
