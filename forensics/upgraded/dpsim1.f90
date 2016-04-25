!*******************************************************************************
!*******************************************************************************
PROGRAM dpsim1

  ! PEI: Interface to IMSL replacements
  USE IMSL_REPLACEMENTS
  
!**************************************************
!*  PROGRAM TO ESTIMATE DP MODEL BY SIMULATED ML  *
!**************************************************
!********************
!*  WAGES INCLUDED  *
!********************
!****************
!*  TRANSFORMS  *
!****************
!**********************************************
!*  ONLY SIMULATE HOME AND SCHOOL PROBS ONCE  *
!**********************************************
!*****************************
!*  SIMULATE EXPECTED MAX'S  *
!*****************************
!**************************************************************
!*  INCLUDE TUITION COST AND LAGGED SCHOOL AS STATE VARIABLE  *
!**************************************************************
!***************************************
!*  SIMULATE DISTRIBUTIONS OF CHOICES  *
!***************************************
      INTEGER KSTATE(13150,40,4)
      INTEGER FSTATE(40,40,40,11,2)
      INTEGER KMAX(40)
      DIMENSION EMAX(13150,40)
      DIMENSION BETA(2,6),A(4,4),RHO(4,4)
      DIMENSION SIGMA(4)
      DIMENSION EU1(10000,40),EU2(10000,40),C(10000,40),B(10000,40)
      DIMENSION RV(4)
      DIMENSION PROB(40,4)
      DIMENSION COUNT(40,4)
!C***DIMENSIONS ALLOW FOR 27 VARIABLES IN MODEL
      DIMENSION CPARM(27)
      DIMENSION IPARM(27)
      CHARACTER*8 NAME(31)
!C***DIMENSIONS ALLOW FOR 10000 PEOPLE AND UP TO 40 PERIODS
      INTEGER TIME(10000)
      INTEGER EDUC(10000,40),LSCHL(10000,40)
      INTEGER EXPER(10000,40,2)
      INTEGER X1,X2,E,T
      CHARACTER*4 TRANS

      ! PEI: Set up file connectors
      open(9,file='in1.txt'); open(10,file='ouncond41.txt') 
      open(11,file='funcond41.txt'); open(12,file='ocond41.txt')
      open(13,file='fcond41.txt'); open(14,file='ftest.txt')
      open(15,file='seed.txt'); open(16,file='emax41.txt')
      open(17,file='state41.txt')
      
      READ(9,1500) NPER,NPOP,DRAW,DRAW1,TAU,NITER,MAXIT,MAXSTP,TRANS
      WRITE(10,1500) NPER,NPOP,DRAW,DRAW1,TAU,NITER,MAXIT,MAXSTP,TRANS
      WRITE(12,1500) NPER,NPOP,DRAW,DRAW1,TAU,NITER,MAXIT,MAXSTP,TRANS
 1500 FORMAT(1x,i3,1x,i5,1x,f7.0,1x,f6.0,1x,f6.2,1x,i2,1x,i2,1X,I2,1x,A3)
!C     DRAW1=DRAW
      NPARM=27
      ALPHA=0.1
!C     GAMA=0.57
      WNA=-9.99
      PI=3.141592654
!*****************************************************************
!* READ IN REPLICATION NUMBER AND THE SEEDS FOR THIS REPLICATION *
!*****************************************************************
      READ(15,1313) IRUN,ISEED,ISEED1,ISEED2                               
      WRITE(10,1313) IRUN,ISEED,ISEED1,ISEED2                               
      WRITE(12,1313) IRUN,ISEED,ISEED1,ISEED2                               
 1313 FORMAT(I4,1x,I10,1x,I10,1x,I10)
!*****************************
!*  READ IN STARTING VALUES  *
!*****************************
      WRITE(10,1504) 
      WRITE(12,1504) 
 1504 FORMAT(' PARAMETER VECTOR:')
      DO 1 J=1,2
      READ(9,1501) (BETA(J,K),K=1,6)                         
      WRITE(10,1501) (BETA(J,K),K=1,6)                         
      WRITE(12,1501) (BETA(J,K),K=1,6)                         
 1501 FORMAT(6(1x,f10.6))
    1 CONTINUE
      READ(9,1502) CBAR1,CBAR2,CS,VHOME,DELTA
      WRITE(10,1502) CBAR1,CBAR2,CS,VHOME,DELTA
      WRITE(12,1502) CBAR1,CBAR2,CS,VHOME,DELTA
 1502 FORMAT(5(1x,f10.5))
      DO 2 J=1,4
      READ(9,1503) (RHO(J,K),K=1,J)
      WRITE(10,1503) (RHO(J,K),K=1,J)
      WRITE(12,1503) (RHO(J,K),K=1,J)
 1503 FORMAT(4(1x,f10.5))
    2 CONTINUE
      READ(9,1503) (SIGMA(J),J=1,4)
      WRITE(10,1503) (SIGMA(J),J=1,4)
      WRITE(12,1503) (SIGMA(J),J=1,4)
!*******************
!*  READ IN NAMES  *
!*******************
      WRITE(10,1607) 
      WRITE(12,1607) 
 1607 FORMAT(' NAMES OF PARAMETERS:')
      DO 201 J=1,2
       READ(9,1601) (NAME(K),K=(J-1)*6+1,J*6)
       WRITE(10,1601) (NAME(K),K=(J-1)*6+1,J*6) 
       WRITE(12,1601) (NAME(K),K=(J-1)*6+1,J*6) 
 1601  FORMAT(6(1X,A8))
  201 CONTINUE  
      READ(9,1602) (NAME(K),K=2*6+1,2*6+5)
      WRITE(10,1602) (NAME(K),K=2*6+1,2*6+5)
      WRITE(12,1602) (NAME(K),K=2*6+1,2*6+5)
 1602 FORMAT(5(1X,A8))
      DO 202 J=1,4
       READ(9,1603) (NAME(K),K=2*6+5+J*(J-1)/2+1,2*6+5+J*(J+1)/2)
       WRITE(10,1603) (NAME(K),K=2*6+5+J*(J-1)/2+1,2*6+5+J*(J+1)/2)
       WRITE(12,1603) (NAME(K),K=2*6+5+J*(J-1)/2+1,2*6+5+J*(J+1)/2)
 1603  FORMAT(4(1X,A8))
  202 CONTINUE
      READ(9,1603) (NAME(K),K=2*6+5+4*(4+1)/2+1,2*6+5+4*(4+1)/2+4)
      WRITE(10,1603) (NAME(K),K=2*6+5+4*(4+1)/2+1,2*6+5+4*(4+1)/2+4)
      WRITE(12,1603) (NAME(K),K=2*6+5+4*(4+1)/2+1,2*6+5+4*(4+1)/2+4)
!**************************
!*  READ IN IPARM VECTOR  *
!**************************
      DO 203 J=1,2
       READ(9,1604) (IPARM(K),K=(J-1)*6+1,J*6)
 1604  FORMAT(6I1)
  203 CONTINUE  
      READ(9,1605) (IPARM(K),K=2*6+1,2*6+5)
 1605 FORMAT(5I1)
      DO 204 J=1,4
      READ(9,1606) (IPARM(K),K=2*6+5+J*(J-1)/2+1,2*6+5+J*(J+1)/2)
 1606  FORMAT(4I1)
  204 CONTINUE
!******************
!*  READ IN DATA  *
!******************
      DO 7 I=1,NPOP
      READ(14,1002) IPPP,TIME(I),EXPER(I,1,1),EXPER(I,1,2),EDUC(I,1),LSCHL(I,1)
       DO 8 T=2,TIME(I)
       READ(14,1003) EXPER(I,T,1),EXPER(I,T,2),EDUC(I,T),LSCHL(I,T)
    8  CONTINUE
    7  CONTINUE
 1002 FORMAT(1X,I5,1X,I3,1X,1x,1X,10x,4(1X,I3))
 1003 FORMAT(11X,1x,1X,10x,4(1X,I3))
 1000 FORMAT(1X,I5,1X,I3,1X,I1,1X,F10.2,4(1X,I3))
 1001 FORMAT(11X,I1,1X,F10.2,4(1X,I3))
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
      DO 88 J=1,4
        DO 1008 K=1,4
         A(J,K)=A(J,K)*SIGMA(J)
 1008   CONTINUE
        WRITE(10,1503) (A(J,K),K=1,4)
        WRITE(12,1503) (A(J,K),K=1,4)
   88 CONTINUE
!***************************
!*  DRAW RANDOM VARIABLES  *
!***************************
      CALL RNSET(ISEED)
      NDRAW=DRAW
      IF(NPOP.GT.NDRAW) NDRAW=NPOP
!C***DRAW THE RANDOM VARIABLES FOR SIMULATING THE DP SOLUTION
      DO 10 J=1,NDRAW
      DO 9 T=1,NPER
       CALL RNNOR(4,RV)
       U1        = A(1,1)*RV(1)
       U2        = A(2,1)*RV(1)+A(2,2)*RV(2)
       C(J,T)    = A(3,1)*RV(1)+A(3,2)*RV(2)+A(3,3)*RV(3)
       B(J,T)    = A(4,1)*RV(1)+A(4,2)*RV(2)+A(4,3)*RV(3)+A(4,4)*RV(4)
       EU1(J,T)=EXP(U1)
       EU2(J,T)=EXP(U2)
!C      IF(j.le.2) then
!C       write(11,1111) T,EU1(J,T),EU2(J,T),C(J,T),B(J,T)
!C1111   format(' t=',I2,' DRAWS=',4F10.4)                
!C      ENDIF          
    9 CONTINUE
   10 CONTINUE
!*****************************************************************
!*  CONSTRUCT THE EXPECTED MAX OF THE TIME NPER VALUE FUNCTIONS  *
!*****************************************************************
!C****CREATE THE STATE INDEX FOR TIME = NPER
      K=0
      DO 15 E=10,20
         IF(E.GT.10+NPER-1)  GO TO 15
      DO 16 X1=0,NPER-1
      DO 17 X2=0,NPER-1
        IF(X1+X2+E-10.LT.NPER) THEN
          DO 18 LS=0,1 
          IF((LS.EQ.0).AND.((E-NPER).EQ.9)) GOTO 18
          IF((LS.EQ.1).AND.(E.EQ.10)) GOTO 18
           K=K+1
           KSTATE(K,NPER,1)=X1
           KSTATE(K,NPER,2)=X2
           KSTATE(K,NPER,3)=E
           KSTATE(K,NPER,4)=LS
           FSTATE(NPER,X1+1,X2+1,E-9,LS+1)=K
   18     CONTINUE
        ENDIF
   17 CONTINUE
   16 CONTINUE
   15 CONTINUE
      KMAX(NPER)=K
!C****INITIALIZE THE EXPECTED MAX TO ZERO
      DO 20 K=1,KMAX(NPER)
        EMAX(K,NPER)=0.0
   20 CONTINUE
       DO 7021 K=1,KMAX(NPER) 
       X1=KSTATE(K,NPER,1)
       X2=KSTATE(K,NPER,2)
       E=KSTATE(K,NPER,3)
       LS=KSTATE(K,NPER,4)
       W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
       W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
       IF(E.GE.12) CBAR = CBAR1 - CBAR2
       IF(E.LT.12) CBAR = CBAR1
       IF(LS.EQ.0) CBAR = CBAR - CS
       IF(E.GT.19) CBAR = CBAR - 50000.0
      DO 22 J=1,DRAW
        V1=W1*EU1(J,NPER)
        V2=W2*EU2(J,NPER)
        V3=CBAR+C(J,NPER)
        V4=VHOME+B(J,NPER) 
!C       WRITE(11,4150) J,V1,V2,V3,V4
!C4150   FORMAT(' DRAW=',I2,' V1=',F10.2,' V2=',F10.2,' V3=',f10.2,
!C    *     ' V4=',F10.2)
        VMAX=AMAX1(V1,V2,V3,V4)
        EMAX(K,NPER)=EMAX(K,NPER)+VMAX
   22 CONTINUE
      EMAX(K,NPER) = EMAX(K,NPER)/DRAW
 7021 CONTINUE
!***********************************************************
!*  CONSTRUCT THE EXPECTED MAX OF THE VALUE FUNCTIONS FOR  *
!*  PERIODS 2 THROUGH NPER-1                               *
!***********************************************************
      DO 30 IS=1,NPER-1
      T=NPER-IS
!C     WRITE(11,6330) T
!C6330 FORMAT(' WORKING ON PERIOD ',I2)
!C****CREATE THE STATE INDEX FOR TIME = T
      K=0
      DO 40 E=10,20
         IF(E.GT.10+T-1)  GOTO 40
      DO 41 X1=0,T-1
      DO 42 X2=0,T-1
        IF(X1+X2+E-10.LT.T) THEN
         DO 43 LS=0,1 
         IF((LS.EQ.0).AND.((E-T).EQ.9)) GOTO 43
         IF((LS.EQ.1).AND.(E.EQ.10).AND.(T.GT.1)) GOTO 43
           K=K+1
           KSTATE(K,T,1)=X1
           KSTATE(K,T,2)=X2
           KSTATE(K,T,3)=E
           KSTATE(K,T,4)=LS
           FSTATE(T,X1+1,X2+1,E-9,LS+1)=K
   43    CONTINUE
        ENDIF
   42 CONTINUE
   41 CONTINUE
   40 CONTINUE
      KMAX(T)=K
!C****INITIALIZE THE EXPECTED MAX TO ZERO
      DO 51 K=1,KMAX(T) 
        EMAX(K,T)=0.0
   51 CONTINUE
      DO 7052 K=1,KMAX(T)
      X1=KSTATE(K,T,1)
      X2=KSTATE(K,T,2)
      E=KSTATE(K,T,3)
      LS=KSTATE(K,T,4)
      W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
      W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
      IF(E.GE.12) THEN
        CBAR = CBAR1 - CBAR2
       ELSE
        CBAR = CBAR1
      ENDIF
      IF(LS.EQ.0) CBAR = CBAR - CS
      E1 = DELTA*EMAX(FSTATE(T+1,X1+2,X2+1,E-9,1),T+1)
      E2 = DELTA*EMAX(FSTATE(T+1,X1+1,X2+2,E-9,1),T+1)
      IF(E.LE.19) E3 = CBAR + DELTA*EMAX(FSTATE(T+1,X1+1,X2+1,E-8,2),T+1)
      IF(E.GT.19) E3 = CBAR - 50000.0                  
      E4 = VHOME + DELTA*EMAX(FSTATE(T+1,X1+1,X2+1,E-9,1),T+1)
      DO 53 J=1,DRAW
        V1=W1*EU1(J,T)  + E1    
        V2=W2*EU2(J,T)  + E2 
        V3=C(J,T)       + E3
        V4=B(J,T)       + E4
        VMAX=AMAX1(V1,V2,V3,V4)
!C       WRITE(11,4150) J,V1,V2,V3,V4
        EMAX(K,T)=EMAX(K,T)+VMAX
   53 CONTINUE
      EMAX(K,T) = EMAX(K,T)/DRAW
 7052 CONTINUE
!C***END OF LOOP OVER TIME PERIODS
   30 CONTINUE
!****************************************************
!*  SIMULATE THE CHOICES AT TIME T GIVEN THE STATE  *
!****************************************************
      do 56 j=1,4
      do 57 t=1,nper
        count(t,j)=0.0
  57  continue
  56  continue
      wealth = 0.0
      DO 60 I=1,NPOP
      DO 330 T=1,NPER-1  
         E=EDUC(I,T)
         X1=EXPER(I,T,1)
         X2=EXPER(I,T,2)
         LS=LSCHL(I,T)
         IF(E.GE.12) THEN
           CBAR = CBAR1 - CBAR2
          ELSE
           CBAR = CBAR1
         ENDIF
         IF(LS.EQ.0) CBAR = CBAR - CS
         W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
         W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
         WAGE1=W1*EU1(I,T)
         WAGE2=W2*EU2(I,T)
         V1 = WAGE1+DELTA*EMAX(FSTATE(T+1,X1+2,X2+1,E-9,1),T+1)
         V2 = WAGE2+DELTA*EMAX(FSTATE(T+1,X1+1,X2+2,E-9,1),T+1)   
         IF(E.LE.19) THEN 
           V3=C(I,T)+CBAR+DELTA*EMAX(FSTATE(T+1,X1+1,X2+1,E-8,2),T+1)
           WAGE3=CBAR+C(I,T)
          ELSE              
           V3=CBAR - 50000.0             
           WAGE3=CBAR-50000.0 
         ENDIF                                      
         V4 = B(I,T)+VHOME+DELTA*EMAX(FSTATE(T+1,X1+1,X2+1,E-9,1),T+1)
         WAGE4=VHOME+B(I,T)
       VMAX=AMAX1(V1,V2,V3,V4)
       IF(VMAX .EQ. V1) THEN
         K=1 
         WRITE(13,1000) I,T,K,WAGE1,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE1
       ENDIF
       IF(VMAX .EQ. V2) THEN
         K=2 
         WRITE(13,1000) I,T,K,WAGE2,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE2
       ENDIF
       IF(VMAX .EQ. V3) THEN
         K=3 
         WRITE(13,1000) I,T,K,WAGE3,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE3
       ENDIF
       IF(VMAX .EQ. V4) THEN
         K=4 
         WRITE(13,1000) I,T,K,WAGE4,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE4
       ENDIF
       COUNT(T,K) = COUNT(T,K) + 1.d0/NPOP
!C***END OF LOOP OVER TIME PERIODS
  330 CONTINUE
       T=NPER
       E=EDUC(I,T)
       X1=EXPER(I,T,1)
       X2=EXPER(I,T,2)
       LS=LSCHL(I,T)
       IF(E.GE.12) THEN
         CBAR = CBAR1 - CBAR2
        ELSE
         CBAR = CBAR1
       ENDIF
       IF(LS.EQ.0) CBAR = CBAR - CS
       IF(E.GT.19) CBAR = CBAR - 50000.0
       W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
       W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
       WAGE1=W1*EU1(I,T)
       WAGE2=W2*EU2(I,T)
       V1 = WAGE1
       V2 = WAGE2   
       V3 = C(I,T)+CBAR
       V4 = B(I,T)+VHOME
       VMAX=AMAX1(V1,V2,V3,V4)
       IF(VMAX .EQ. V1) THEN
         K=1 
         WRITE(13,1000) I,T,K,WAGE1,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE1
       ENDIF
       IF(VMAX .EQ. V2) THEN
         K=2 
         WRITE(13,1000) I,T,K,WAGE2,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE2
       ENDIF
       IF(VMAX .EQ. V3) THEN
         K=3 
         WRITE(13,1000) I,T,K,V3,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE3
       ENDIF
       IF(VMAX .EQ. V4) THEN
         K=4 
         WRITE(13,1000) I,T,K,V4,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE4
       ENDIF
       COUNT(T,K) = COUNT(T,K) + 1.d0/NPOP
!C***END OF LOOP OVER PEOPLE
   60 CONTINUE
      wealth = wealth/NPOP
      write(12,1060) wealth
 1060 format(' discounted wealth = ',f16.2)
      do 71 t=1,nper
        write(12,3000) t,(COUNT(t,j),j=1,4)
   71 continue
!***********************************************************
!*  CONSTRUCT MONTE-CARLO DATA FOR PERIODS 1 THROUGH NPER  *
!***********************************************************
!C     DO 54 T=2,4   
!C     DO 55 K=1,KMAX(T)
!C       WRITE(11,2000) T,K,KSTATE(K,T,1),KSTATE(K,T,2),
!C    *   KSTATE(K,T,3),EMAX(K,T)  
!C2000   FORMAT(' T=',I2,' K=',I4,' X1=',I2,' X2=',I2,
!C    *    ' E=',I2,' EMAX=',F16.3)
!C  55 CONTINUE
!C  54 CONTINUE
      do 58 j=1,4
      do 59 t=1,nper
        prob(t,j)=0.0
  59  continue
  58  continue
      wealth = 0.0
      DO 260 I=1,NPOP
        X1=0
        X2=0
        E=10
        LS1=1
      DO 61 T=1,NPER-1  
       LS = LS1
       W1=exp(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
       W2=exp(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
       WAGE1=W1*EU1(I,T) 
       WAGE2=W2*EU2(I,T)  
       IF(E.GE.12) THEN                 
         CBAR = CBAR1 - CBAR2 
        ELSE
         CBAR = CBAR1
       ENDIF
       IF(LS.eq.0) CBAR = CBAR - CS
       V1=WAGE1 + DELTA*EMAX(FSTATE(T+1,X1+2,X2+1,E-9,1),T+1)
       V2=WAGE2 + DELTA*EMAX(FSTATE(T+1,X1+1,X2+2,E-9,1),T+1) 
       IF(E.LE.19) THEN                 
         V3=CBAR+C(I,T) + DELTA*EMAX(FSTATE(T+1,X1+1,X2+1,E-8,2),T+1)
         WAGE3=CBAR+C(I,T)  
        ELSE
         V3=CBAR - 50000.0
         WAGE3=CBAR-50000.0
       ENDIF
       V4=VHOME+B(I,T) + DELTA*EMAX(FSTATE(T+1,X1+1,X2+1,E-9,1),T+1)
       WAGE4=VHOME+B(I,T)  
       VMAX=AMAX1(V1,V2,V3,V4)
!C      WRITE(11,2010) I,T,V1,V2,V3,V4
!C2010  FORMAT(' I=',I2,' T=',I2,' V1=',f10.2,' V2=',F10.2,' V3=',F10.2,
!C    *     ' V4=',F10.2) 
       IF (VMAX .EQ. V1) THEN
         K=1
         WRITE(11,1000) I,T,K,WAGE1,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE1
         X1=X1+1
         LS1=0
       ENDIF
       IF (VMAX .EQ. V2) THEN
         K=2
         WRITE(11,1000) I,T,K,WAGE2,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE2
         X2=X2+1
         LS1=0
       ENDIF
       IF (VMAX .EQ. V3) THEN
         K=3
         WRITE(11,1000) I,T,K,WAGE3,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE3
         E=E+1
         LS1=1
       ENDIF
       IF (VMAX .EQ. V4) THEN
         K=4
         WRITE(11,1000) I,T,K,WAGE4,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE4
         LS1=0
       ENDIF
       prob(t,k)=prob(t,k)+1.0/npop
   61 CONTINUE
       T=NPER           
       LS = LS1
       W1=exp(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
       W2=exp(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
       WAGE1=W1*EU1(I,T) 
       WAGE2=W2*EU2(I,T)  
       IF(E.GE.12) THEN                 
         CBAR = CBAR1 - CBAR2 
        ELSE
         CBAR = CBAR1
       ENDIF
       IF(LS.eq.0) CBAR = CBAR - CS
       IF(E.GT.19) CBAR = CBAR - 50000.0
       V1=WAGE1 
       V2=WAGE2  
       V3=CBAR+C(I,T) 
       V4=VHOME+B(I,T) 
       VMAX=AMAX1(V1,V2,V3,V4)
!C      WRITE(11,2010) I,T,V1,V2,V3,V4
       IF (VMAX .EQ. V1) THEN
         K=1
         WRITE(11,1000) I,T,K,WAGE1,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE1
       ENDIF
       IF (VMAX .EQ. V2) THEN
         K=2
         WRITE(11,1000) I,T,K,WAGE2,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE2
       ENDIF
       IF (VMAX .EQ. V3) THEN
         K=3
         WRITE(11,1000) I,T,K,V3,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE3
       ENDIF
       IF (VMAX .EQ. V4) THEN
         K=4
         WRITE(11,1000) I,T,K,V4,X1,X2,E,LS
         wealth = wealth + (DELTA**T)*WAGE4
       ENDIF
       prob(t,k)=prob(t,k)+1.0/npop
  260 CONTINUE
      wealth = wealth/NPOP
      write(10,1260) wealth
 1260 format(' discounted wealth = ',F16.2)
      do 70 t=1,nper
        write(10,3000) t,(prob(t,j),j=1,4)
   70 continue
 3000 FORMAT(' T=',I3,' PROB=',4F16.12)
!C 999 CONTINUE
      STOP
      END
