MODULE robupy_program_constants

	!/*	setup	*/

	IMPLICIT NONE

!------------------------------------------------------------------------------ 
!	parameters and types
!------------------------------------------------------------------------------ 

    INTEGER, PARAMETER ::   our_int 	= selected_int_kind(9),		   &
					        our_sgle 	= selected_real_kind(6,37),	&
						    our_dble    = selected_real_kind(15,307), &
						    zero_int 	= 0_our_int, &
							one_int 	= 1_our_int, &
							two_int 	= 2_our_int

    REAL(our_dble), PARAMETER  ::   zero_dble    = 0.00_our_dble, &
                                    quarter_dble = 0.25_our_dble, &
                                    half_dble    = 0.50_our_dble, &
                                    one_dble     = 1.00_our_dble, &
                                    two_dble     = 2.00_our_dble, &
                                    three_dble   = 3.00_our_dble, &
    pi_dble = 3.1415926535897932384626433832795028841971_our_dble


    REAL(our_dble), PARAMETER   :: upperClip = 0.999999999999_our_dble, &
                                   lowerClip = 0.000000000001_our_dble
        
!******************************************************************************
!******************************************************************************
END MODULE 