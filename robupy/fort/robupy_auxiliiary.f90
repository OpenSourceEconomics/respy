MODULE robupy_auxiliary

	!/*	external modules	*/

    USE robupy_program_constants

	!/*	setup	*/

	IMPLICIT NONE

    PRIVATE

    PUBLIC ::   foo

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE foo (a)
      		integer a
      		print*, "Hello from Fortran!"
      		print*, "a=",a
END SUBROUTINE

END MODULE
