MODULE robupy_auxiliary

	!/*	external modules	*/

    USE robupy_program_constants

	!/*	setup	*/

	IMPLICIT NONE

    PRIVATE

    PUBLIC ::   trace_fun
    PUBLIC ::   inverse_fun
    PUBLIC ::   det_fun

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
FUNCTION det_fun(A)
!
!This function calculates the determinant of a matrix A.
!
REAL(KIND=our_dble), INTENT(IN)	:: A(:,:)
REAL(KIND=our_dble) 	:: det_fun
!Definition of local variables
REAL(KIND=our_dble)					:: d
REAL(KIND=our_dble), ALLOCATABLE	:: B(:,:)
INTEGER(KIND=our_int)				:: j, n
INTEGER(KIND=our_int), ALLOCATABLE	:: indx(:)
!Algorithm
n = size(A,1); ALLOCATE(B(n,n))
B = A
ALLOCATE(indx(n))
CALL ludcmp(B, d, indx)
DO j = 1,n
   d = d*B(j,j)
END DO
det_fun = d
END function
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
!********************************************************************
!********************************************************************
FUNCTION inverse_fun(A, k)
!
!This function calculates the inverse of a matrix A.
!
!Calls  : ludcmp, lubksb
!Version: 19.10.2008
!
IMPLICIT NONE
!Definition of arguments

INTEGER(KIND=our_int), INTENT(IN)		:: k
REAL(KIND=our_dble), INTENT(IN)		:: A(k,k)
REAL(KIND=our_dble) :: inverse_fun(k,k)
!Definition of local variables
REAL(KIND=our_dble)					:: d
REAL(KIND=our_dble), ALLOCATABLE	:: y(:,:), B(:,:)
INTEGER(KIND=our_int)				:: n, i, j
INTEGER(KIND=our_int), ALLOCATABLE	:: indx(:)
!Algorithm
n  = size(A,1)
ALLOCATE(y(n,n)); y = 0
ALLOCATE(B(n,n)); B = A
ALLOCATE(indx(n))
DO i = 1,n
   y(i,i) = 1
END DO
CALL ludcmp(B, d, indx)
DO j = 1,n
   CALL lubksb(B, y(:,j), indx)
END DO
inverse_fun = y
END FUNCTION

!******************************************************************************
!******************************************************************************
PURE FUNCTION trace_fun(A)

    REAL(KIND=our_dble), INTENT(IN)     :: A(:,:)
    REAL(our_dble) :: trace_fun

    !/* internals objects    */

    INTEGER(our_int)                    :: i

    INTEGER(our_int)                    :: n

    ! Initialize results
    trace_fun = zero_dble

    ! Get dimension
    n = SIZE(A, DIM = 1)

    ! Calculate trace
    DO i = 1, n

        trace_fun = trace_fun + A(i,i)

    END DO

END FUNCTION

!******************************************************************************
!******************************************************************************
END MODULE