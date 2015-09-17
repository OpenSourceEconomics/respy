MODULE robufort_auxiliary

	!/*	external modules	*/

    USE robufort_program_constants

	!/*	setup	*/

  IMPLICIT NONE

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE divergence(div, x, cov, level)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: div(1)

    REAL(our_dble), INTENT(IN)      :: x(2)
    REAL(our_dble), INTENT(IN)      :: cov(4,4)
    REAL(our_dble), INTENT(IN)      :: level

    !/* internals objects    */

    REAL(our_dble)                  :: alt_mean(4, 1) = zero_dble
    REAL(our_dble)                  :: old_mean(4, 1) = zero_dble
    REAL(our_dble)                  :: alt_cov(4,4)
    REAL(our_dble)                  :: old_cov(4,4)
    REAL(our_dble)                  :: inv_old_cov(4,4)
    REAL(our_dble)                  :: comp_a
    REAL(our_dble)                  :: comp_b(1, 1)
    REAL(our_dble)                  :: comp_c
    REAL(our_dble)                  :: rslt

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Construct alternative distribution
    alt_mean(1,1) = x(1)
    alt_mean(2,1) = x(2)
    alt_cov = cov

    ! Construct baseline distribution
    old_cov = cov

    ! Construct auxiliary objects.
    inv_old_cov = inverse(old_cov, 4)

    ! Calculate first component
    comp_a = trace_fun(MATMUL(inv_old_cov, alt_cov))

    ! Calculate second component
    comp_b = MATMUL(MATMUL(TRANSPOSE(old_mean - alt_mean), inv_old_cov), &
                old_mean - alt_mean)

    ! Calculate third component
    comp_c = LOG(determinant(alt_cov) / determinant(old_cov))

    ! Statistic
    rslt = half_dble * (comp_a + comp_b(1,1) - four_dble + comp_c)

    ! Divergence
    div = level - rslt

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE divergence_approx_gradient(rslt, x, cov, level, eps)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: rslt(2)

    REAL(our_dble), INTENT(IN)      :: x(2)
    REAL(our_dble), INTENT(IN)      :: eps
    REAL(our_dble), INTENT(IN)      :: cov(4,4)
    REAL(our_dble), INTENT(IN)      :: level

    !/* internals objects    */

    INTEGER(our_int)                :: k

    REAL(our_dble)                  :: ei(2)
    REAL(our_dble)                  :: d(2)
    REAL(our_dble)                  :: f0(1)
    REAL(our_dble)                  :: f1(1)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Initialize containers
    ei = zero_dble

    ! Evaluate baseline
    CALL divergence(f0, x, cov, level)

    ! Iterate over increments
    DO k = 1, 2

        ei(k) = one_dble

        d = eps * ei

        CALL divergence(f1, x + d, cov, level)

        rslt(k) = (f1(1) - f0(1)) / d(k)

        ei(k) = zero_dble

    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
FUNCTION inverse(A, k)

    !/* external objects    */

  INTEGER(our_int), INTENT(IN)  :: k

  REAL(our_dble), INTENT(IN)    :: A(k, k)

    !/* internal objects    */
  
  REAL(our_dble), ALLOCATABLE   :: y(:, :)
  REAL(our_dble), ALLOCATABLE   :: B(:, :)
  REAL(our_dble)                :: d
  REAL(our_dble)                :: inverse(k, k)

  INTEGER(our_int), ALLOCATABLE :: indx(:)  
  INTEGER(our_int)              :: n
  INTEGER(our_int)              :: i
  INTEGER(our_int)              :: j

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
  
  ! Auxiliary objects
  n  = size(A, 1)

  ! Allocate containers
  ALLOCATE(y(n, n))
  ALLOCATE(B(n, n))
  ALLOCATE(indx(n))

  ! Initialize containers
  y = zero_dble
  B = A

  ! Main
  DO i = 1, n
  
     y(i, i) = 1
  
  END DO

  CALL ludcmp(B, d, indx)

  DO j = 1, n
  
     CALL lubksb(B, y(:, j), indx)
  
  END DO
  
  ! Collect result
  inverse = y

END FUNCTION
!*******************************************************************************
!*******************************************************************************
FUNCTION determinant(A)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)    :: A(:, :)
    REAL(our_dble)                :: determinant

    !/* internal objects    */

    INTEGER(our_int), ALLOCATABLE :: indx(:)
    INTEGER(our_int)              :: j
    INTEGER(our_int)              :: n

    REAL(our_dble), ALLOCATABLE   :: B(:, :)
    REAL(our_dble)                :: d

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Auxiliary objects
    n  = size(A, 1)

    ! Allocate containers
    ALLOCATE(B(n, n))
    ALLOCATE(indx(n))

    ! Initialize containers
    B = A

    CALL ludcmp(B, d, indx)
    
    DO j = 1, n
    
       d = d * B(j, j)
    
    END DO
    
    ! Collect results
    determinant = d

END FUNCTION
!*******************************************************************************
!*******************************************************************************
PURE FUNCTION trace_fun(A)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)  :: A(:,:)
    REAL(our_dble)              :: trace_fun

    !/* internals objects    */

    INTEGER(our_int)            :: i
    INTEGER(our_int)            :: n

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Get dimension
    n = SIZE(A, DIM = 1)

    ! Initialize results
    trace_fun = zero_dble

    ! Calculate trace
    DO i = 1, n

        trace_fun = trace_fun + A(i, i)

    END DO

END FUNCTION
!*******************************************************************************
!*******************************************************************************
SUBROUTINE ludcmp(A, d, indx)

    !/* external objects    */
    
    INTEGER(our_int), INTENT(INOUT) :: indx(:)

    REAL(our_dble), INTENT(INOUT)   :: a(:,:)
    REAL(our_dble), INTENT(INOUT)   :: d

    !/* internal objects    */

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: imax
    INTEGER(our_int)                :: n

    REAL(our_dble), ALLOCATABLE     :: vv(:)
    REAL(our_dble)                  :: dum
    REAL(our_dble)                  :: sums
    REAL(our_dble)                  :: aamax

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Initialize containers
    imax = missing_int 
    
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
       
       IF(a(j, j) == zero_dble) a(j, j) = tiny_dble
       
       IF(j /= n) THEN
       
         dum = one_dble / a(j, j)
       
         DO i = (j + 1), n
       
            a(i, j) = a(i, j) * dum
       
         END DO
       
       END IF
    
    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE lubksb(A, B, indx)

    !/* external objects    */

    INTEGER(our_int), INTENT(IN)    :: indx(:)

    REAL(our_dble), INTENT(INOUT)   :: A(:, :)
    REAL(our_dble), INTENT(INOUT)   :: B(:)

    !/* internal objects    */

    INTEGER(our_int)                :: ii
    INTEGER(our_int)                :: n
    INTEGER(our_int)                :: ll
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: sums

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Auxiliary objects
    n = SIZE(A, DIM = 1)

    ! Allocate containers
    ii = zero_int

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
!*******************************************************************************
!*******************************************************************************
SUBROUTINE cholesky(factor, matrix)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: factor(:,:)

    REAL(our_dble), INTENT(IN)      :: matrix(:,:)

    !/* internal objects    */

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: n
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: j

    REAL(our_dble), ALLOCATABLE     :: clon(:, :)

    REAL(our_dble)                  :: sums

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Auxiliary objects
    n = size(matrix,1)
   
    ! Allocate containers
    ALLOCATE(clon(n,n))
    
    ! Apply Cholesky decomposition
    clon = matrix
    
    DO j = 1, n

      sums = 0.0
      
      DO k = 1, (j - 1)

        sums = sums + clon(j, k)**2

      END DO

      clon(j, j) = DSQRT(clon(j, j) - sums)
       
      DO i = (j + 1), n

        sums = zero_dble

        DO k = 1, (j - 1)

          sums = sums + clon(j, k)*clon(i, k)

        END DO

        clon(i, j) = (clon(i, j) - sums)/clon(j, j)

      END DO
    
    END DO
    
    ! Transfer information from matrix to factor
    DO i = 1, n
    
      DO j = 1, n  
    
        IF(i .LE. j) THEN
    
          factor(j, i) = clon(j, i) 
    
        END IF
    
      END DO
    
    END DO

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE standard_normal(draw)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: draw(:)

    !/* internal objects    */

    INTEGER(our_int)                :: g
    INTEGER(our_int)                :: dim
    
    REAL(our_dble), ALLOCATABLE     :: u(:)
    REAL(our_dble), ALLOCATABLE     :: r(:)

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

    ! Auxiliary objects
    dim = SIZE(draw)

    ! Allocate containers
    ALLOCATE(u(2 * dim)); ALLOCATE(r(2 * dim))

    ! Call uniform deviates
    CALL RANDOM_NUMBER(u)

    ! Apply Box-Muller transform
    DO g = 1, (2 * dim), 2

       r(g) = DSQRT(-two_dble * LOG(u(g)))*COS(two_dble *pi * u(g + one_int)) 
       r(g + 1) = DSQRT(-two_dble * LOG(u(g)))*SIN(two_dble *pi * u(g + one_int)) 

    END DO

    ! Extract relevant floats
    DO g = 1, dim 

       draw(g) = r(g)     

    END DO

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE multivariate_normal(draws, mean, covariance)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: draws(:, :)
    REAL(our_dble), INTENT(IN)      :: mean(:)
    REAL(our_dble), INTENT(IN)      :: covariance(:, :)
    
    !/* internal objects    */
    
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: num_draws
    INTEGER(our_int)                :: dim
    
    REAL(our_dble), ALLOCATABLE     :: z(:, :)
    REAL(our_dble), ALLOCATABLE     :: ch(:, :)

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 
  
    ! Auxiliary objects
    num_draws = SIZE(draws, 1)

    dim       = SIZE(draws, 2)

    ! Allocate containers
    ALLOCATE(z(dim, 1)); ALLOCATE(ch(dim, dim))

    ! Initialize containers
    ch = zero_dble

    ! Construct Cholesky decomposition
    CALL cholesky(ch, covariance) 

    ! Draw deviates
    DO i = 1, num_draws   
       
       CALL standard_normal(z(:, 1))
       
       draws(i, :) = MATMUL(ch, z(:, 1)) + mean(:)  
    
    END DO

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
END MODULE