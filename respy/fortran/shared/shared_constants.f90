MODULE shared_constants

    !/*	setup	                */

    IMPLICIT NONE
    
!------------------------------------------------------------------------------- 
!	Parameters and Types
!------------------------------------------------------------------------------- 

    INTEGER, PARAMETER :: our_int   = selected_int_kind(9)
    INTEGER, PARAMETER :: our_dble  = selected_real_kind(15, 307)

    INTEGER(our_int), PARAMETER :: zero_int     = 0_our_int
    INTEGER(our_int), PARAMETER :: one_int      = 1_our_int
    INTEGER(our_int), PARAMETER :: two_int      = 2_our_int
    INTEGER(our_int), PARAMETER :: three_int    = 3_our_int

    REAL(our_dble), PARAMETER :: zero_dble      = 0.00_our_dble
    REAL(our_dble), PARAMETER :: one_dble       = 1.00_our_dble
    REAL(our_dble), PARAMETER :: two_dble       = 2.00_our_dble

    REAL(our_dble), PARAMETER :: pi        = 3.141592653589793238462643383279502884197_our_dble
    
    ! Variables that are part of the FORTRAN initialization file and are never 
    ! to be changed.
    INTEGER(our_int)            :: edu_max

    REAL(our_dble)              :: delta
    REAL(our_dble)              :: tau

    LOGICAL                     :: is_interpolated

    CHARACTER(225)              :: exec_dir

    ! Variables that need to be aligned across FORTRAN and PYTHON 
    ! implementations.
    INTEGER(our_int), PARAMETER :: MISSING_INT  = -99_our_int

    REAL(our_dble), PARAMETER :: MISSING_FLOAT  = -99.0_our_dble
    REAL(our_dble), PARAMETER :: SMALL_FLOAT    = 1.0e-5_our_dble
    REAL(our_dble), PARAMETER :: TINY_FLOAT     = 1.0e-8_our_dble
    REAL(our_dble), PARAMETER :: HUGE_FLOAT     = 1.0e20_our_dble

    ! Interpolation
    REAL(our_dble), PARAMETER :: INADMISSIBILITY_PENALTY = -40000.00_our_dble

!*******************************************************************************
!*******************************************************************************
END MODULE 