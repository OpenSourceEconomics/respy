!******************************************************************************
!******************************************************************************
MODULE shared_interfaces

    IMPLICIT NONE

    ABSTRACT INTERFACE

        FUNCTION interface_func(x)

            !/* external modules    */

            USE shared_constants

            !/* dummy arguments     *

            REAL(our_dble), INTENT(IN)      :: x(:)
            REAL(our_dble)                  :: interface_func

        END FUNCTION

        FUNCTION interface_dfunc(x)

            !/* external modules    */

            USE shared_constants

            !/* dummy arguments     *

            REAL(our_dble), INTENT(IN)      :: x(:)
            REAL(our_dble)                  :: interface_dfunc(SIZE(x))

        END FUNCTION

    END INTERFACE

    INTERFACE

        PURE SUBROUTINE SLSQP(M, MEQ, LA, N, X, XL, XU, F, C, G, A, ACC, ITER, MODE, W, LEN_W, JW, LEN_JW) !

            !/* external modules    */

            USE shared_constants

            !/* external objects    */

            INTEGER(our_int), INTENT(INOUT)     :: ITER
            INTEGER(our_int), INTENT(INOUT)     :: MODE

            REAL(our_dble), INTENT(INOUT)       :: X(N)
            REAL(our_dble), INTENT(INOUT)       :: ACC

            INTEGER(our_int), INTENT(IN)        :: JW(LEN_W)
            INTEGER(our_int), INTENT(IN)        :: LEN_JW
            INTEGER(our_int), INTENT(IN)        :: LEN_W
            INTEGER(our_int), INTENT(IN)        :: MEQ
            INTEGER(our_int), INTENT(IN)        :: LA
            INTEGER(our_int), INTENT(IN)        :: M
            INTEGER(our_int), INTENT(IN)        :: N

            REAL(our_dble), INTENT(IN)          :: A(LA, N + 1)
            REAL(our_dble), INTENT(IN)          :: G(N + 1)
            REAL(our_dble), INTENT(IN)          :: W(LEN_W)
            REAL(our_dble), INTENT(IN)          :: XL(N)
            REAL(our_dble), INTENT(IN)          :: XU(N)
            REAL(our_dble), INTENT(IN)          :: C(LA)
            REAL(our_dble), INTENT(IN)          :: F

        END SUBROUTINE


        PURE SUBROUTINE DGESDD(JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, IWORK, INFO)

            !/* external modules    */

            USE shared_constants

            !/* external objects    */

            INTEGER(our_int), INTENT(OUT)   :: INFO

            REAL(our_dble), INTENT(INOUT)   :: A(LDA, *)
            REAL(our_dble), INTENT(OUT)     :: S(*)
            REAL(our_dble), INTENT(OUT)     :: U(LDU, *)
            REAL(our_dble), INTENT(OUT)     :: VT(LDVT,*)
            REAL(our_dble), INTENT(OUT)     :: WORK(*)

            CHARACTER(1), INTENT(IN)        :: JOBZ

            INTEGER(our_int), INTENT(IN)    :: M
            INTEGER(our_int), INTENT(IN)    :: N
            INTEGER(our_int), INTENT(IN)    :: LDA
            INTEGER(our_int), INTENT(IN)    :: LDU
            INTEGER(our_int), INTENT(IN)    :: LDVT
            INTEGER(our_int), INTENT(IN)    :: LWORK

            INTEGER(our_int), INTENT(IN)    :: IWORK(*)

        END SUBROUTINE


        PURE SUBROUTINE DPOTRF(UPLO, N, A, LDA, INFO)

            !/* external modules    */

            USE shared_constants

            !/* external objects    */

            REAL(our_dble), INTENT(INOUT)   :: A(N, N)

            INTEGER(our_int), INTENT(IN)    :: INFO
            INTEGER(our_int), INTENT(IN)    :: LDA
            INTEGER(our_int), INTENT(IN)    :: N

            CHARACTER(1), INTENT(IN)        :: UPLO

        END SUBROUTINE


        PURE SUBROUTINE DGETRF(M, N, A, LDA, IPIV, INFO)

            !/* external modules    */

            USE shared_constants

            !/* external objects    */

            INTEGER(our_int), INTENT(OUT)   :: IPIV(N)
            INTEGER(our_int), INTENT(OUT)   :: INFO

            REAL(our_dble), INTENT(INOUT)   :: A(N, N)

            INTEGER(our_int), INTENT(IN)    :: LDA
            INTEGER(our_int), INTENT(IN)    :: M
            INTEGER(our_int), INTENT(IN)    :: N

         END SUBROUTINE


        PURE SUBROUTINE DGETRI(N, A, LDA, IPIV, WORK, LWORK, INFO)

            !/* external modules    */

            USE shared_constants

            !/* external objects    */

            REAL(our_dble), INTENT(INOUT)   :: WORK(LWORK)
            REAL(our_dble), INTENT(INOUT)   :: A(N, N)

            INTEGER(our_int), INTENT(IN)    :: IPIV(N)
            INTEGER(our_int), INTENT(IN)    :: LWORK
            INTEGER(our_int), INTENT(IN)    :: INFO
            INTEGER(our_int), INTENT(IN)    :: LDA
            INTEGER(our_int), INTENT(IN)    :: N

         END SUBROUTINE

    END INTERFACE

!******************************************************************************
!******************************************************************************
END MODULE
