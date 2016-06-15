!******************************************************************************
!******************************************************************************
MODULE estimate_auxiliary

	!/*	external modules	*/

    USE shared_constants

	!/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE write_out_information(counter, fval, x, which)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: counter

    REAL(our_dble), INTENT(IN)      :: x(26)
    REAL(our_dble), INTENT(IN)      :: fval 

    CHARACTER(*), INTENT(IN)        :: which

    !/* internal objects        */

    INTEGER(our_int)                :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------
    
    20 FORMAT(5(1x,i10))
    30 FORMAT(1x,f45.15)

    OPEN(UNIT=1, FILE='opt_info_' // TRIM(which) // '.respy.log')

        WRITE(1, 20) counter
        WRITE(1, 30) fval

        DO i = 1, 26
            WRITE(1, 30) x(i)
        END DO

    CLOSE(1)

END SUBROUTINE

!******************************************************************************
!******************************************************************************
SUBROUTINE dist_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, x)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(OUT)     :: coeffs_home(1)
    REAL(our_dble), INTENT(OUT)     :: coeffs_edu(3)
    REAL(our_dble), INTENT(OUT)     :: coeffs_a(6)
    REAL(our_dble), INTENT(OUT)     :: coeffs_b(6)

    REAL(our_dble), INTENT(IN)      :: x(26)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Extract model ingredients
    coeffs_a = x(1:6)

    coeffs_b = x(7:12)

    coeffs_edu = x(13:15)

    coeffs_home = x(16:16)

    ! Note that the Cholesky decomposition is initially the upper triangular in this case. This is required to align the order of optimization parameters with the outline of the original authors. 
    shocks_cholesky = 0.0

    shocks_cholesky(1, 1:) = x(17:20)

    shocks_cholesky(2, 2:) = x(21:23)

    shocks_cholesky(3, 3:) = x(24:25)

    shocks_cholesky(4, 4:) = x(26:26)

    shocks_cholesky = TRANSPOSE(shocks_cholesky)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE get_free_optim_paras(x, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, paras_fixed)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)      :: shocks_cholesky(4, 4)
    REAL(our_dble), INTENT(IN)      :: coeffs_home(1)
    REAL(our_dble), INTENT(IN)      :: coeffs_edu(3)
    REAL(our_dble), INTENT(IN)      :: coeffs_a(6)
    REAL(our_dble), INTENT(IN)      :: coeffs_b(6)

    LOGICAL, INTENT(IN)             :: paras_fixed(26)

    REAL(our_dble), INTENT(OUT)     :: x(COUNT(.not. paras_fixed))

    !/* internal objects        */    

    REAL(our_dble)                  :: x_internal(26)

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    x_internal(1:6) = coeffs_a(:)
        
    x_internal(7:12) = coeffs_b(:)
        
    x_internal(13:15) = coeffs_edu(:)
        
    x_internal(16:16) = coeffs_home(:)

    x_internal(17:20) = shocks_cholesky(1, :)
        
    x_internal(21:23) = shocks_cholesky(2, 2:)
        
    x_internal(24:25) = shocks_cholesky(3, 3:)
        
    x_internal(26:26) = shocks_cholesky(4, 4:)


    j = 1

    DO i = 1, 26

        IF(paras_fixed(i)) CYCLE

        x(j) = x_internal(i)
        
        j = j + 1

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE