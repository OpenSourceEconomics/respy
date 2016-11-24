!******************************************************************************
!******************************************************************************
!
!   Notes:
!
!       There exists and independent copy of this file at
!       RESPY/development/tests/random/modules. This is used as part of the
!       testing efforts. This is required to ensure independence of
!       each GITHUB repository.
!
!******************************************************************************
!******************************************************************************
PROGRAM dp3asim

  !****************************************************************************
  ! There are only minor modifications to the original source code.
  ! The input and output connections are opened and the commenting
  ! style is changed to comply with the FORTRAN95 standard. Line
  ! breaks are removed. Some consecutive IF statements were changed to ELSEIF.
  !
  ! The program is amended to read in random components to allow for testing
  ! its output against the RESPY package (if requested).
  !
  !****************************************************************************

  ! PEI: Interface to added functions
  USE PEI_ADDITIONS

  ! PEI: Interface to IMSL replacements
  USE IMSL_REPLACEMENTS

!********************************************************
!*  PROGRAM TO CONSTRUCT MONTE CARLO DATA FOR DP MODEL  *
!********************************************************
!*  TUITION COST INCLUDED                               *
!********************************************************
!*  VERSION WITH LAGGED SCHOOL AS STATE VARIABLE        *
!********************************************************
      INTEGER KSTATE(40,14000,4)
      INTEGER FSTATE(40,40,40,11,2)
      INTEGER KMAX(40)
      DIMENSION EMAX(40,14000),EMAX1(40,14000)
      DIMENSION BETA(2,6),A(4,4),RHO(4,4)
      DIMENSION SIGMA(4)
      DIMENSION EU1(5000,40),EU2(5000,40),C(5000,40),B(5000,40)
      DIMENSION RNN(4)
      DIMENSION PROB(40,4)
      DIMENSION PROB1(40,4)
      INTEGER X1,X2,E,T

      ! PEI: Open files
      open(9,file='in.txt')

      open(10,file='otest.txt');

      open(11,file='ftest.txt')


      READ(9,1500) NPER,NPOP,DRAW,DRAW1,TAU
      WRITE(10,1500) NPER,NPOP,DRAW,DRAW1,TAU
 1500 FORMAT(1x,i3,1x,i5,1x,f7.0,1x,f6.0,1x,f6.2)
!C     GAMA= 0.577
      WNA=-9.99
      do 1 J=1,2
      READ(9,1501) (BETA(J,k),k=1,6)
      WRITE(10,1501) (BETA(J,k),k=1,6)
 1501 FORMAT(6(1x,f10.6))
    1 continue
      READ(9,1502) CBAR1,CBAR2,CS,VHOME,DELTA
      WRITE(10,1502) CBAR1,CBAR2,CS,VHOME,DELTA
 1502 FORMAT(5(1x,f10.5))
      do 2 J=1,4
      READ(9,1503) (RHO(J,K),K=1,J)
      WRITE(10,1503) (RHO(J,K),K=1,J)
 1503 FORMAT(4(1x,f10.5))
    2 continue
      READ(9,1503) (SIGMA(J),J=1,4)
      WRITE(10,1503) (SIGMA(J),J=1,4)
!*********************
!*  TRANSFORMATIONS  *
!*********************
      CBAR1 = CBAR1*1000.00
      CBAR2 = CBAR2*1000.00
      CS    = CS   *1000.00
      VHOME = VHOME*1000.00
      DO 1007 J=3,4
        SIGMA(J) = SIGMA(J)*1000.0
 1007 CONTINUE
!*********************************************************
!*  TAKE THE CHOLESKY DECOMPOSITION OF RHO AND PUT IN A  *
!*********************************************************
      DO 3 J=2,4
      DO 4 K=1,J-1
        RHO(K,J) = RHO(J,K)
    4 CONTINUE
    3 CONTINUE
      CALL LFCDS(4,RHO,4,A,4,COND)
      DO 5 J=2,4
      DO 6 K=1,J-1
        A(J,K) = A(K,J)
    6 CONTINUE
    5 CONTINUE
      DO 7 J=1,4
        WRITE(10,1503) (RHO(J,K),K=1,4)
    7 CONTINUE
      DO 8 J=1,4
        DO 1008 K=1,4
         A(J,K)=A(J,K)*SIGMA(J)
 1008   CONTINUE
        WRITE(10,1503) (A(J,K),K=1,4)
    8 CONTINUE
!****************************
!*  CREATE THE STATE INDEX  *
!****************************
      DO 10 T=1,NPER
      K=0
      DO 20 E=10,20
         IF(E.GT.10+T-1) GOTO 20
      DO 21 X1=0,T-1
      DO 22 X2=0,T-1
        IF(X1+X2+E-10.LT.T) THEN
           DO 23 LS=0,1
           IF((LS.eq.0).and.((E-T).eq.9)) goto 23
           IF((LS.eq.1).and.(E.eq.10).and.(T.gt.1)) goto 23
           K=K+1
           KSTATE(T,K,1)=X1
           KSTATE(T,K,2)=X2
           KSTATE(T,K,3)=E
           KSTATE(T,K,4)=LS
           FSTATE(T,X1+1,X2+1,E-9,LS+1)=K
   23      CONTINUE
        ENDIF
   22 CONTINUE
   21 CONTINUE
   20 CONTINUE
      KMAX(T)=K
   10 CONTINUE
      do 24 t=1,nper
         write(10,2001) t,kmax(t)
 2001    format(' t=',i2,' kmax(t)=',i6)
   24 continue
!***************************
!*  DRAW RANDOM VARIABLES  *
!***************************
      CALL RNSET(1111111111)
!C     DO 30 T=1,NPER
      DO 31 J=1,DRAW
      DO 30 T=1,NPER
       CALL RNNOR(4,RNN)
       EU1(J,T) = exp(A(1,1)*RNN(1))
       EU2(J,T) = exp(A(2,1)*RNN(1)+A(2,2)*RNN(2))
       C(J,T)   = A(3,1)*RNN(1)+A(3,2)*RNN(2)+A(3,3)*RNN(3)
       B(J,T)   = A(4,1)*RNN(1)+A(4,2)*RNN(2)+A(4,3)*RNN(3)+A(4,4)*RNN(4)
   30 CONTINUE
   31 CONTINUE
!C  30 CONTINUE

  ! PEI: Create zero disturbances.
  CALL READ_IN_DISTURBANCES(EU1, EU2, C, B)

!*****************************************************************
!*  CONSTRUCT THE EXPECTED MAX OF THE TIME NPER VALUE FUNCTIONS  *
!*****************************************************************
      DO 40 K=1,KMAX(NPER)
        EMAX(NPER,K)=0.
!C       EMAX1(NPER,K)=0.
   40 CONTINUE
      DO 41 K=1,KMAX(NPER)
        X1=KSTATE(NPER,K,1)
        X2=KSTATE(NPER,K,2)
        E=KSTATE(NPER,K,3)
        LS=KSTATE(NPER,K,4)
        W1=exp(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2    +BETA(1,5)*X2+BETA(1,6)*X2**2)
        W2=exp(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2  +BETA(2,5)*X2+BETA(2,6)*X2**2)
        IF(E.GE.12) THEN
          CBAR = CBAR1 - CBAR2
         ELSE
          CBAR = CBAR1
        ENDIF
        IF(LS.eq.0) CBAR = CBAR - CS
      DO 42 J=1,DRAW
        V1 = W1*EU1(J,NPER)
        V2 = W2*EU2(J,NPER)
        IF(E.LE.19) THEN
          V3 = CBAR+C(J,NPER)
         ELSE
          V3 = CBAR - 400000.0
        ENDIF
        V4 = VHOME+B(J,NPER)
!C        write(10,2002) j,eu1(nper,j),eu2(nper,j),c(nper,j)
!C 2002   format(' j=',i2,' eu1=',f10.3,' eu2=',f10.3,' c=',f10.3)
!C        SUMV=EXP((V1-VMAX)/TAU)+EXP((V2-VMAX)/TAU)
!C     *            +EXP((V3-VMAX)/TAU)+EXP((V4-VMAX)/TAU)
        VMAX=AMAX1(V1,V2,V3,V4)
        EMAX(NPER,K)=EMAX(NPER,K)+VMAX
!C        EMAX1(NPER,K)=EMAX1(NPER,K)
!C     *       +TAU*(GAMA+LOG(SUMV)+VMAX/TAU)
   42 CONTINUE
      EMAX(NPER,K) = EMAX(NPER,K)/DRAW
   41 CONTINUE
!***********************************************************
!*  CONSTRUCT THE EXPECTED MAX OF THE VALUE FUNCTIONS FOR  *
!*  PERIODS 2 THROUGH NPER-1                               *
!***********************************************************
      DO 50 S=1,NPER-2
      T=NPER-S
      DO 51 K=1,KMAX(T)
        EMAX(T,K)=0.
!C       EMAX1(T,K)=0.
   51 CONTINUE
      DO 52 K=1,KMAX(T)
        X1=KSTATE(T,K,1)
        X2=KSTATE(T,K,2)
        E=KSTATE(T,K,3)
        LS=KSTATE(T,K,4)
        W1=exp(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2      +BETA(1,5)*X2+BETA(1,6)*X2**2)
        W2=exp(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2      +BETA(2,5)*X2+BETA(2,6)*X2**2)
        IF(E.GE.12) THEN
          CBAR = CBAR1 - CBAR2
         ELSE
          CBAR = CBAR1
        ENDIF
        IF(LS.eq.0) CBAR = CBAR - CS
      DO 53 J=1,DRAW
        V1=W1*EU1(J,T)    + DELTA*EMAX(T+1,FSTATE(T+1,X1+2,X2+1,E-9,1))
        V2=W2*EU2(J,T)    + DELTA*EMAX(T+1,FSTATE(T+1,X1+1,X2+2,E-9,1))
        IF(E.LE.19) then
          V3=CBAR+C(J,T)  + DELTA*EMAX(T+1,FSTATE(T+1,X1+1,X2+1,E-8,2))
         else
          V3=CBAR - 400000.0
        ENDIF
        V4=VHOME+B(J,T)   + DELTA*EMAX(T+1,FSTATE(T+1,X1+1,X2+1,E-9,1))
!C       SUMV=EXP((V1-VMAX)/TAU)+EXP((V2-VMAX)/TAU)
!C    *     +EXP((V3-VMAX)/TAU)+EXP(V4-VMAX)/TAU)
        VMAX=AMAX1(V1,V2,V3,V4)
        EMAX(T,K)=EMAX(T,K)+VMAX
!C       EMAX1(T,k)=EMAX1(T,K)
!C    *     +TAU*(GAMA+LOG(SUMV+VMAX/TAU)
   53 CONTINUE
      EMAX(T,K) = EMAX(T,K)/DRAW
   52 CONTINUE
   50 CONTINUE
!C      DO 54 T=2,NPER
!C      DO 55 K=1,KMAX(T)
!C        WRITE(10,2000) T,K,KSTATE(T,K,1),KSTATE(T,K,2),
!C     *   KSTATE(T,K,3),EMAX(T,K)
!C 2000   FORMAT(' T=',I2,' K=',I4,' X1=',I2,' X2=',I2,
!C     *    ' E=',I2,' EMAX=',F16.3)
!C   55 CONTINUE
!C   54 CONTINUE
!C      GOTO 999
!***********************************************************
!*  CONSTRUCT MONTE-CARLO DATA FOR PERIODS 1 THROUGH NPER  *
!***********************************************************
      do 58 t=1,nper
      do 59 j=1,4
        prob(t,j)=0.0
        prob1(t,j)=0.0
  59  continue
  58  continue
      wealth = 0.0
      DO 60 L=1,NPOP
        X1=0
        X2=0
        E=10
        LS1=1
      DO 61 T=1,NPER-1
       LS = LS1
       W1=exp(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2 +BETA(1,5)*X2+BETA(1,6)*X2**2)
       W2=exp(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2 +BETA(2,5)*X2+BETA(2,6)*X2**2)
       WAGE1=W1*EU1(L,T)
       WAGE2=W2*EU2(L,T)
       V1=WAGE1 + DELTA*EMAX(T+1,FSTATE(T+1,X1+2,X2+1,E-9,1))
       V2=WAGE2 + DELTA*EMAX(T+1,FSTATE(T+1,X1+1,X2+2,E-9,1))
       IF(E.GE.12) THEN
         CBAR = CBAR1 - CBAR2
        ELSE
         CBAR = CBAR1
       ENDIF
       IF(LS.eq.0) CBAR = CBAR - CS
       IF(E.LE.19) then
         V3=CBAR+C(L,T) + DELTA*EMAX(T+1,FSTATE(T+1,X1+1,X2+1,E-8,2))
         WAGE3=CBAR+C(L,T)
        ELSE
         V3=CBAR - 400000.0
         WAGE3=CBAR-400000.0
       ENDIF
       V4=VHOME+B(L,T) + DELTA*EMAX(T+1,FSTATE(T+1,X1+1,X2+1,E-9,1))
       WAGE4=VHOME+B(L,T)
       VMAX=AMAX1(V1,V2,V3,V4)
       SUMV=EXP((V1-VMAX)/TAU)+EXP((V2-VMAX)/TAU)   +EXP((V3-VMAX)/TAU)+EXP((V4-VMAX)/TAU)
       prob(t,1)=prob(t,1)+( EXP((v1-vmax)/tau) /sumv ) /npop
       prob(t,2)=prob(t,2)+( EXP((v2-vmax)/tau) /sumv ) /npop
       prob(t,3)=prob(t,3)+( EXP((v3-vmax)/tau) /sumv ) /npop
       prob(t,4)=prob(t,4)+( EXP((v4-vmax)/tau) /sumv ) /npop
       IF (VMAX .EQ. V1) THEN
         K=1
         LS1=0
       ELSE IF (VMAX .EQ. V2) THEN
         K=2
         LS1=0
       ELSE IF (VMAX .EQ. V3) THEN
         K=3
         LS1=1
       ELSE IF (VMAX .EQ. V4) THEN
         K=4
         LS1=0
       ENDIF
       prob1(t,k)=prob1(t,k)+1.0/npop
       IF(K.EQ.1) THEN
        WRITE(11,1000) L,NPER,K,WAGE1,X1,X2,E,LS
        X1=X1+1
        wealth = wealth + WAGE1*(DELTA**T)
       ENDIF
       IF(K.EQ.2) THEN
        WRITE(11,1000) L,NPER,K,WAGE2,X1,X2,E,LS
        X2=X2+1
        wealth = wealth + WAGE2*(DELTA**T)
       ENDIF
       IF(K.EQ.3) THEN
        WRITE(11,1000) L,NPER,K,WAGE3,X1,X2,E,LS
        E=E+1
        wealth = wealth + WAGE3*(DELTA**T)
       ENDIF
       IF(K.EQ.4) THEN
        WRITE(11,1000) L,NPER,K,WAGE4,X1,X2,E,LS
        wealth = wealth + WAGE4*(DELTA**T)
       ENDIF
   61 CONTINUE
       T = NPER
       LS = LS1
       W1=exp(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2 +BETA(1,5)*X2+BETA(1,6)*X2**2)
       W2=exp(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2 +BETA(2,5)*X2+BETA(2,6)*X2**2)
       V1=W1*EU1(L,T)
       V2=W2*EU2(L,T)
       IF(E.GE.12) THEN
         CBAR = CBAR1 - CBAR2
        ELSE
         CBAR = CBAR1
       ENDIF
       IF(LS.eq.0) CBAR = CBAR - CS
       IF(E.LE.19) then
         V3=CBAR+C(L,T)
        ELSE
         V3=CBAR - 400000.0
       ENDIF
       V4=VHOME+B(L,T)
       VMAX=AMAX1(V1,V2,V3,V4)
       SUMV=EXP((V1-VMAX)/TAU)+EXP((V2-VMAX)/TAU) +EXP((V3-VMAX)/TAU)+EXP((V4-VMAX)/TAU)
       prob(t,1)=prob(t,1)+( EXP((v1-vmax)/tau) /sumv ) /npop
       prob(t,2)=prob(t,2)+( EXP((v2-vmax)/tau) /sumv ) /npop
       prob(t,3)=prob(t,3)+( EXP((v3-vmax)/tau) /sumv ) /npop
       prob(t,4)=prob(t,4)+( EXP((v4-vmax)/tau) /sumv ) /npop
       IF (VMAX .EQ. V1) THEN
         K=1
       ELSE IF (VMAX .EQ. V2) THEN
         K=2
       ELSE IF (VMAX .EQ. V3) THEN
         K=3
       ELSE IF (VMAX .EQ. V4) THEN
         K=4
       ENDIF
       prob1(t,k)=prob1(t,k)+1.0/npop
       IF(K.EQ.1) THEN
        WRITE(11,1000) L,NPER,K,V1,X1,X2,E,LS
        wealth = wealth + WAGE1*(DELTA**T)
       ENDIF
       IF(K.EQ.2) THEN
        WRITE(11,1000) L,NPER,K,V2,X1,X2,E,LS
        wealth = wealth + WAGE2*(DELTA**T)
       ENDIF
       IF(K.EQ.3) THEN
        WRITE(11,1000) L,NPER,K,V3,X1,X2,E,LS
        wealth = wealth + WAGE3*(DELTA**T)
       ENDIF
       IF(K.EQ.4) THEN
        WRITE(11,1000) L,NPER,K,V4,X1,X2,E,LS
        wealth = wealth + WAGE4*(DELTA**T)
       ENDIF
   60 CONTINUE
      wealth = wealth/npop
      write(10,1060) wealth
 1060 format(' discounted wealth = ',f16.2)
      do 70 t=1,nper
        write(10,3000) t,(prob(t,j),j=1,4)
 3000   format(' t=',i3,' prob=',4f16.12)
   70 continue
      do 71 t=1,nper
        write(10,3001) t,(prob1(t,j),j=1,4)
 3001   format(' t=',i3,' prob1=',4f16.12)
   71 continue
 1000 FORMAT(1X,I5,1X,I3,1X,I1,1X,F10.2,4(1X,I3))
!C 999 CONTINUE
      STOP
      END
