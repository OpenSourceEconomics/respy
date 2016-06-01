!******************************************************************************
!******************************************************************************
!
!   This module contains utilities that are used throughout the program but are
!   not tailored for it.
!
!******************************************************************************
!******************************************************************************
MODULE shared_utilities

    !/* external modules    */

    USE shared_lapack_interfaces
    
    USE shared_constants

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS

!******************************************************************************
!******************************************************************************
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

    INTEGER(our_int) , ALLOCATABLE     :: IWORK(:)
    REAL(our_dble), ALLOCATABLE     :: WORK(:)
    REAL(our_dble)                  :: A_sub(m, m)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    A_sub = A

    ! Auxiliary objects
    LWORK =  M * (7 + 4 * M)

    ! Allocate containers
    ALLOCATE(WORK(LWORK)); ALLOCATE(IWORK(8 * M))

    ! Call LAPACK routine
    CALL DGESDD( 'A', m, m, A_sub, m, S, U, m, VT, m, WORK, LWORK, IWORK, INFO)

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE random_choice(sample, candidates, num_candidates, num_draws)

  !
  ! Source
  ! ------
  !
  !   KNUTH, D. The art of computer programming. Vol II.
  !     Seminumerical Algorithms. Reading, Mass: AddisonWesley, 1969
  !

    !/* external objects    */

    INTEGER, INTENT(OUT)          :: sample(:)

    INTEGER, INTENT(IN)           :: num_candidates
    INTEGER, INTENT(IN)           :: candidates(:)
    INTEGER, INTENT(IN)           :: num_draws

    !/* internal objects    */

    INTEGER(our_int)              :: m
    INTEGER(our_int)              :: j
    INTEGER(our_int)              :: l

    REAL(our_dble)                :: u(1)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize auxiliary objects
    m = 0

    ! Draw random points from candidates
    DO j = 1, num_candidates

        CALL RANDOM_NUMBER(u)

        l = INT(DBLE(num_candidates - j + 1) * u(1)) + 1

        IF (l .GT. (num_draws - m)) THEN

            CONTINUE

        ELSE

            m = m + 1

            sample(m) = candidates(j)

            IF (m .GE. num_draws) THEN

                RETURN

            END IF

        END IF

    END DO


END SUBROUTINE
!******************************************************************************
!******************************************************************************
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

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

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
!******************************************************************************
!******************************************************************************
END MODULE