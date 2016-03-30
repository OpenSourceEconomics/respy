MODULE robufort_auxiliary

    !/*	external modules	    */

    USE robufort_constants

	!/*	setup	                */

    IMPLICIT NONE
    
    PUBLIC

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_model_parameters(coeffs_a, coeffs_b, coeffs_edu, coeffs_home, &
                shocks_cov, shocks_cholesky, x)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: shocks_cholesky(:, :)
    REAL(our_dble), INTENT(OUT)     :: shocks_cov(:, :)
    REAL(our_dble), INTENT(OUT)     :: coeffs_home(:)
    REAL(our_dble), INTENT(OUT)     :: coeffs_edu(:)
    REAL(our_dble), INTENT(OUT)     :: coeffs_a(:)
    REAL(our_dble), INTENT(OUT)     :: coeffs_b(:)

    REAL(our_dble), INTENT(IN)      :: x(:)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Extract model ingredients
    coeffs_a = x(1:6)

    coeffs_b = x(7:12)

    coeffs_edu = x(13:15)

    coeffs_home = x(16:16)

    shocks_cholesky = 0.0

    shocks_cholesky(1:4, 1) = x(17:20)

    shocks_cholesky(2:4, 2) = x(21:23)

    shocks_cholesky(3:4, 3) = x(24:25)

    shocks_cholesky(4:4, 4) = x(26:26)

    ! Reconstruct the covariance matrix of reward shocks
    shocks_cov = MATMUL(shocks_cholesky, TRANSPOSE(shocks_cholesky))

END SUBROUTINE
!******************************************************************************
!******************************************************************************
FUNCTION clip_value(value, lower_bound, upper_bound)

    !/* external objects        */

    REAL(our_dble), INTENT(IN)  :: lower_bound
    REAL(our_dble), INTENT(IN)  :: upper_bound
    REAL(our_dble), INTENT(IN)  :: value

    !/*  internal objects       */

    REAL(our_dble)              :: clip_value

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    IF(value < lower_bound) THEN

        clip_value = lower_bound

    ELSEIF(value > upper_bound) THEN

        clip_value = upper_bound

    ELSE

        clip_value = value

    END IF

END FUNCTION
!*******************************************************************************
!*******************************************************************************
FUNCTION inverse(A, k)

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)  :: k

    REAL(our_dble), INTENT(IN)    :: A(:, :)

    !/* internal objects        */
  
    REAL(our_dble), ALLOCATABLE   :: y(:, :)
    REAL(our_dble), ALLOCATABLE   :: B(:, :)

    REAL(our_dble)                :: inverse(k, k)
    REAL(our_dble)                :: d

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

    !/* external objects        */

    REAL(our_dble)                :: determinant

    REAL(our_dble), INTENT(IN)    :: A(:, :)

    !/* internal objects        */

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

    !/* external objects        */

    REAL(our_dble)              :: trace_fun

    REAL(our_dble), INTENT(IN)  :: A(:,:)

    !/* internals objects       */

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

    !/* external objects        */
    
    INTEGER(our_int), INTENT(INOUT) :: indx(:)

    REAL(our_dble), INTENT(INOUT)   :: a(:,:)
    REAL(our_dble), INTENT(INOUT)   :: d

    !/* internal objects        */

    INTEGER(our_int)                :: imax
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: k
    INTEGER(our_int)                :: n

    REAL(our_dble), ALLOCATABLE     :: vv(:)


    REAL(our_dble)                  :: aamax
    REAL(our_dble)                  :: sums
    REAL(our_dble)                  :: dum

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Initialize containers
    imax = MISSING_INT 
    
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
       
       IF(a(j, j) == zero_dble) a(j, j) = TINY_FLOAT
       
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

    !/* external objects        */

    INTEGER(our_int), INTENT(IN)    :: indx(:)

    REAL(our_dble), INTENT(INOUT)   :: A(:, :)
    REAL(our_dble), INTENT(INOUT)   :: B(:)

    !/* internal objects        */

    INTEGER(our_int)                :: ii
    INTEGER(our_int)                :: ll
    INTEGER(our_int)                :: n
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
SUBROUTINE svd(U, S, VT, A, m)
    
    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: VT(:, :)
    REAL(our_dble), INTENT(OUT)     :: U(:, :)
    REAL(our_dble), INTENT(OUT)     :: S(:) 

    REAL(our_dble), INTENT(IN)      :: A(:, :)
    
    INTEGER(our_int), INTENT(IN)    :: m

    !/* internal objects        */

    INTEGER(our_int)                :: LWORK
    INTEGER(our_int)                :: INFO

    REAL(our_dble), ALLOCATABLE     :: IWORK(:)
    REAL(our_dble), ALLOCATABLE     :: WORK(:)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
  
    ! Auxiliary objects
    LWORK =  M * (7 + 4 * M)

    ! Allocate containers
    ALLOCATE(WORK(LWORK)); ALLOCATE(IWORK(8 * M))

    ! Call LAPACK routine
    CALL DGESDD( 'A', m, m, A, m, S, U, m, VT, m, WORK, LWORK, IWORK, INFO)

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
FUNCTION pinv(A, m)

    !/* external objects        */

    REAL(our_dble)                  :: pinv(m, m)

    REAL(our_dble), INTENT(IN)      :: A(:, :)
    
    INTEGER(our_int), INTENT(IN)    :: m


    !/* internal objects        */

    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: VT(m, m)
    REAL(our_dble)                  :: UT(m, m) 
    REAL(our_dble)                  :: U(m, m)
    REAL(our_dble)                  :: cutoff
    REAL(our_dble)                  :: S(m) 
 
!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL svd(U, S, VT, A, m)

    cutoff = 1e-15_our_dble * MAXVAL(S)

    DO i = 1, M

        IF (S(i) .GT. cutoff) THEN

            S(i) = one_dble / S(i)

        ELSE 

            S(i) = zero_dble

        END IF

    END DO

    UT = TRANSPOSE(U)

    DO i = 1, M

        pinv(i, :) = S(i) * UT(i,:)

    END DO

    pinv = MATMUL(TRANSPOSE(VT), pinv)

END FUNCTION
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_clipped_vector(Y, X, lower_bound, upper_bound, num_values)

    !/* external objects        */

    REAL(our_dble), INTENT(INOUT)       :: Y(:)

    REAL(our_dble), INTENT(IN)          :: lower_bound
    REAL(our_dble), INTENT(IN)          :: upper_bound
    REAL(our_dble), INTENT(IN)          :: X(:)
    
    INTEGER(our_int), INTENT(IN)        :: num_values

    !/* internal objects        */
    
    INTEGER(our_int)                    :: i

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
SUBROUTINE point_predictions(Y, X, coeffs, num_agents)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: Y(:)

    REAL(our_dble), INTENT(IN)      :: coeffs(:)
    REAL(our_dble), INTENT(IN)      :: X(:, :)
    
    INTEGER(our_int), INTENT(IN)    :: num_agents

    !/* internal objects        */

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

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: coeffs(:)

    INTEGER, INTENT(IN)             :: num_covars
    INTEGER, INTENT(IN)             :: num_agents

    REAL(our_dble), INTENT(IN)      :: X(:, :)
    REAL(our_dble), INTENT(IN)      :: Y(:)
    
    !/* internal objects        */

    REAL(our_dble)                  :: A(num_covars, num_covars)
    REAL(our_dble)                  :: C(num_covars, num_covars)
    REAL(our_dble)                  :: D(num_covars, num_agents)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
   A = MATMUL(TRANSPOSE(X), X)

   C =  pinv(A, num_covars)

   D = MATMUL(C, TRANSPOSE(X))

   coeffs = MATMUL(D, Y)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE get_r_squared(r_squared, observed, predicted, num_agents)

    !/* external objects        */

    REAL(our_dble), INTENT(OUT)     :: r_squared

    REAL(our_dble), INTENT(IN)      :: predicted(:)
    REAL(our_dble), INTENT(IN)      :: observed(:)
    
    INTEGER(our_int), INTENT(IN)    :: num_agents

    !/* internal objects        */

    REAL(our_dble)                  :: mean_observed
    REAL(our_dble)                  :: ss_residuals
    REAL(our_dble)                  :: ss_total
 
    INTEGER(our_int)                :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    
    ! Calculate mean of observed data
    mean_observed = SUM(observed) / DBLE(num_agents)
    
    ! Sum of squared residuals
    ss_residuals = zero_dble

    DO i = 1, num_agents

        ss_residuals = ss_residuals + (observed(i) - predicted(i))**2

    END DO

    ! Sum of squared residuals
    ss_total = zero_dble

    DO i = 1, num_agents

        ss_total = ss_total + (observed(i) - mean_observed)**2

    END DO

    ! Construct result
    r_squared = one_dble - ss_residuals / ss_total

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE