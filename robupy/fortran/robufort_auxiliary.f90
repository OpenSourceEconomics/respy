MODULE robufort_auxiliary

	 !/*	external modules	*/

    USE robufort_constants

	 !/*	setup	*/

    IMPLICIT NONE

    PUBLIC

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE logging_solution(indicator, period)

    !/* external objects    */

    INTEGER(our_int), INTENT(IN)            :: indicator
    INTEGER(our_int), INTENT(IN), OPTIONAL  :: period

    !/* internals objects    */

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    OPEN(UNIT=99, FILE='logging.robupy.sol.log', ACCESS='APPEND')
    
    ! State space creation
    IF (indicator == 1) THEN
      WRITE(99, *) " Starting state space creation   "
      WRITE(99, *) ""

    ! Ex Ante Payoffs
    ELSEIF (indicator == 2) THEN
      WRITE(99, *) " Starting calculation of systematic payoffs  "
      WRITE(99, *) ""

    ! Backward induction procedure
    ELSEIF (indicator == 3) THEN
      WRITE(99, *) " Starting backward induction procedure "
      WRITE(99, *) ""
    
    ELSEIF (indicator == 4) THEN
      WRITE(99, *) " ... solving period ", period
      WRITE(99, *) ""
   
    ! Finishing
    ELSEIF (indicator == -1) THEN
        WRITE(99, *) " ... finished "
        WRITE(99, *) ""
        WRITE(99, *) ""
        
    END IF
  
  CLOSE(99)
    
END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
FUNCTION inverse(A, k)

    !/* external objects    */

    INTEGER(our_int), INTENT(IN)  :: k

    REAL(our_dble), INTENT(IN)    :: A(:, :)

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

    REAL(our_dble), INTENT(OUT)           :: draws(:, :)
    REAL(our_dble), INTENT(IN), OPTIONAL  :: mean(:)
    REAL(our_dble), INTENT(IN), OPTIONAL  :: covariance(:, :)
    
    !/* internal objects    */
    
    INTEGER(our_int)                :: num_draws
    INTEGER(our_int)                :: dim
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j  

    REAL(our_dble), ALLOCATABLE     :: covariance_internal(:, :)
    REAL(our_dble), ALLOCATABLE     :: mean_internal(:)
    REAL(our_dble), ALLOCATABLE     :: ch(:, :)
    REAL(our_dble), ALLOCATABLE     :: z(:, :)

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

    ! Auxiliary objects
    num_draws = SIZE(draws, 1)

    dim       = SIZE(draws, 2)

    ! Handle optional arguments
    ALLOCATE(mean_internal(dim)); ALLOCATE(covariance_internal(dim, dim))

    IF (PRESENT(mean)) THEN

      mean_internal = mean

    ELSE

      mean_internal = zero_dble

    END IF

    IF (PRESENT(covariance)) THEN

      covariance_internal = covariance

    ELSE

      covariance_internal = zero_dble

      DO j = 1, dim

        covariance_internal(j, j) = one_dble

      END DO

    END IF

    ! Allocate containers
    ALLOCATE(z(dim, 1)); ALLOCATE(ch(dim, dim))

    ! Initialize containers
    ch = zero_dble

    ! Construct Cholesky decomposition
    CALL cholesky(ch, covariance_internal) 

    ! Draw deviates
    DO i = 1, num_draws   
       
       CALL standard_normal(z(:, 1))
       
       draws(i, :) = MATMUL(ch, z(:, 1)) + mean_internal(:)  
    
    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_clipped_vector(Y, X, lower_bound, upper_bound, num_values)

    !/* external objects    */


    REAL(our_dble), INTENT(OUT)           :: Y(num_values)

    REAL(our_dble), INTENT(IN)            :: X(num_values)
    REAL(our_dble), INTENT(IN)            :: lower_bound
    REAL(our_dble), INTENT(IN)            :: upper_bound
    
    INTEGER(our_int), INTENT(IN)          :: num_values

    !/* internal objects    */
    
    INTEGER(our_int)                      :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    DO i = 1, num_values

        IF (X(i) .LT. lower_bound) THEN

            Y(i) = lower_bound

        ELSE IF (X(i) .GT. upper_bound) THEN

            Y(i) = upper_bound

        ELSE 

            Y(i) = X(i)

        END IF

    END DO


END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_predictions(Y, X, coeffs, num_agents)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: Y(num_agents)

    REAL(our_dble), INTENT(IN)      :: coeffs(:)
    REAL(our_dble), INTENT(IN)      :: X(:,:)
    
    INTEGER(our_int), INTENT(IN)    :: num_agents

    !/* internal objects    */

    INTEGER(our_int)                 :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    DO i = 1, num_agents

        Y(i) = DOT_PRODUCT(coeffs, X(i, :))

    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_coefficients(coeffs, Y, X, num_covars, num_agents)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: coeffs(num_covars)

    REAL(our_dble), INTENT(IN)      :: X(:, :)
    REAL(our_dble), INTENT(IN)      :: Y(:)
    
    INTEGER, INTENT(IN)             :: num_covars
    INTEGER, INTENT(IN)             :: num_agents

    !/* internal objects    */

    REAL(our_dble)                  :: A(num_covars, num_covars)
    REAL(our_dble)                  :: C(num_covars, num_covars)
    REAL(our_dble)                  :: D(num_covars, num_agents)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
   A = MATMUL(TRANSPOSE(X), X)

   C =  inverse(A, num_covars)

   D = MATMUL(C, TRANSPOSE(X))

   coeffs = MATMUL(D, Y) 

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_r_squared(r_squared, Y, P, num_agents)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: r_squared

    REAL(our_dble), INTENT(IN)      :: Y(num_agents)
    REAL(our_dble), INTENT(IN)      :: P(num_agents)
    
    INTEGER(our_int), INTENT(IN)    :: num_agents


    !/* internal objects    */


    REAL(our_dble)                  :: mean_observed
    REAL(our_dble)                  :: ss_residuals
    REAL(our_dble)                  :: ss_total
 
    INTEGER(our_int)                :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Calculate mean of observed data
    mean_observed = SUM(Y) / DBLE(num_agents)
    
    ! Sum of squared residuals
    ss_residuals = zero_dble

    DO i = 1, num_agents

        ss_residuals = ss_residuals + (Y(i) - P(i))**2

    END DO

    ! Sum of squared residuals
    ss_total = zero_dble

    DO i = 1, num_agents

        ss_total = ss_total + (Y(i) - mean_observed)**2

    END DO

    ! Construct result
    r_squared = one_dble - ss_residuals / ss_total

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE