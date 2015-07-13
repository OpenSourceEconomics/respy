MODULE robupy_auxiliary

	!/*	external modules	*/

    USE robupy_program_constants

	!/*	setup	*/

	IMPLICIT NONE

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE ludcmp(A, d, indx)

    !/* external objects    */
    
    INTEGER(our_int), INTENT(INOUT) :: indx(:)

    REAL(our_dble), INTENT(INOUT)		:: a(:,:)
    REAL(our_dble), INTENT(INOUT)		:: d

    !/* internal objects    */

    INTEGER(our_int)			          :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: imax
    INTEGER(our_int)                :: n

    REAL(our_dble), ALLOCATABLE     :: vv(:)
    REAL(our_dble)				          :: dum
    REAL(our_dble)                  :: sums
    REAL(our_dble)                  :: aamax

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Auxiliary objects
    n = SIZE(A, DIM = 1)

    ! Initialize containers
    ALLOCATE(vv(n))

    ! Allocate containers
    d = one_dble

    ! Main
    DO i = 1, n

       aamax = zero_dble

       DO j = 1, n

          IF(abs(a(i, j)) > aamax) aamax = abs(a(i, j))

       END DO

       vv(i) = one_dble / aamax

    END DO

    DO j = 1, n

       DO i = 1, (j - 1)
    
          sums = a(i, j)
    
          DO k = 1, (i - 1)
    
             sums = sums - a(i, k)*a(k, j)
    
          END DO
    
       a(i,j) = sums
    
       END DO
    
       aamax = zero_dble
    
       DO i = j, n

          sums = a(i, j)

          DO k = 1, (j - 1)

             sums = sums - a(i, k)*a(k, j)

          END DO

          a(i, j) = sums

          dum = vv(i) * abs(sums)

          IF(dum >= aamax) THEN

            imax  = i

            aamax = dum

          END IF

       END DO

       IF(j /= imax) THEN

         DO k = 1, n

            dum = a(imax, k)

            a(imax, k) = a(j, k)

            a(j, k) = dum

         END DO

         d = -d

         vv(imax) = vv(j)

       END IF

       indx(j) = imax
       
       IF(a(j, j) == zero_dble) a(j, j) = tiny
       
       IF(j /= n) THEN
       
         dum = one_dble / a(j, j)
       
         DO i = (j + 1), n
       
            a(i, j) = a(i, j) * dum
       
         END DO
       
       END IF
    
    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE lubksb(A, B, indx)

    !/* external objects    */

    INTEGER(our_int), INTENT(IN)    :: indx(:)

    REAL(our_dble), INTENT(INOUT)		:: A(:, :)
    REAL(our_dble), INTENT(INOUT)   :: B(:)

    !/* internal objects    */

    INTEGER(our_int) 		            :: ii
    INTEGER(our_int)                :: n
    INTEGER(our_int)                :: ll
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: i

    REAL(our_dble)			            :: sums

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Auxiliary objects
    n = SIZE(A, DIM = 1)

    ! Allocate containers
    ii = zero_dble

    ! Main
    DO i = 1, n
    
       ll = indx(i)
    
       sums = B(ll)					 
    
       B(ll) = B(i)
    
       IF(ii /= zero_dble) THEN
    
         DO j = ii, (i - 1)
    
           	sums = sums - a(i, j) * b(j)

         END DO
    
       ELSE IF(sums /= zero_dble) THEN
    
         ii = i
    
       END IF
    
       b(i) = sums
    
    END DO
    
    DO i = n, 1, -1
    
       sums = b(i)
    
       DO j = (i + 1), n
    
          sums = sums - a(i, j) * b(j)
    
       END DO
    
       b(i) = sums / a(i, i)
    
    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE