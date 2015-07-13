MODULE robupy_auxiliary

	!/*	external modules	*/

    USE robupy_program_constants

	!/*	setup	*/

	IMPLICIT NONE

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE ludcmp(A, d, indx)

    !Defining arguments
    REAL(KIND=our_dble), INTENT(INOUT)		:: a(:,:)
    REAL(KIND=our_dble), INTENT(INOUT)		:: d
    INTEGER(KIND=our_int), INTENT(INOUT)	:: indx(:)
    !Defining local variables
    INTEGER(KIND=our_int)			:: i, j, k, imax, n
    REAL(KIND=our_dble), ALLOCATABLE:: vv(:)
    REAL(KIND=our_dble), PARAMETER	:: nmax = 500, tiny = 1.0e-20
    REAL(KIND=our_dble)				:: dum, sums, aamax
    !Algorithm
    n = size(A, 1)
    d = 1
    ALLOCATE(vv(n))
    DO i = 1, n
       aamax = 0.0
       DO j = 1,n
          IF(abs(a(i,j)) > aamax) aamax = abs(a(i,j))
       END DO
       vv(i) = 1.0/aamax
    END DO
    DO j = 1, n
       DO i = 1, (j - 1)
          sums = a(i,j)
          DO k = 1, (i - 1)
             sums = sums - a(i,k)*a(k,j)
          END DO
       a(i,j) = sums
       END DO
       aamax = 0.0
       DO i = j, n
          sums = a(i,j)
          DO k = 1, (j - 1)
             sums = sums - a(i,k)*a(k,j)
          END DO
          a(i,j) = sums
          dum = vv(i) * abs(sums)
          IF(dum >= aamax) THEN
            imax  = i
            aamax = dum
          END IF
       END DO
       IF(j /= imax) THEN
         DO k = 1, n
            dum = a(imax, k)
            a(imax, k) = a(j,k)
            a(j,k) = dum
         END DO
         d = -d
         vv(imax) = vv(j)
       END IF
       indx(j) = imax
       IF(a(j,j) == REAL(0)) a(j,j) = tiny
       IF(j /= n) THEN
         dum = 1.0/a(j,j)
         DO i = j+1, n
            a(i,j) = a(i,j)*dum
         END DO
       END IF
    END DO
END SUBROUTINE

!******************************************************************************
!******************************************************************************
SUBROUTINE lubksb(A, B, indx)
!
!This subroutine solves a set of n linear equations AX = B.
!
!Calls  :
!Version: 17.10.2008
!
IMPLICIT NONE
!Definition of arguments
REAL(KIND=our_dble), INTENT(INOUT)			:: A(:,:), B(:)
INTEGER(KIND=our_int), INTENT(IN)			:: indx(:)
!Definition of local variables
INTEGER(KIND=our_int) 		:: ii, n, ll, j, i
REAL(KIND=our_dble)			:: sums
!Algorithm
n  = size(A,1)
ii = 0
DO i = 1, n
   ll = indx(i)
   sums = B(ll)					!!will be there when needed
   B(ll) = B(i)
   IF(ii /= 0) THEN
     DO j = ii, (i - 1)
       	sums = sums - a(i,j)*b(j)
     END DO
   ELSE IF(sums /= 0.0) THEN
     ii = i
   END IF
   b(i) = sums
END DO
DO i = n, 1, -1
   sums = b(i)
   DO j = i+1, n
      sums = sums - a(i,j)*b(j)
   END DO
   b(i) = sums/a(i,i)
END DO
END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE