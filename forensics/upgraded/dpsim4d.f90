!*******************************************************************************
!*******************************************************************************
PROGRAM dpsim4d

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
!********************************
!*  INTERPOLATE EXPECTED MAX'S  *
!********************************
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
      DIMENSION EU1(2000,40),EU2(2000,40),C(2000,40),B(2000,40)
      DIMENSION RV(4)
      DIMENSION PROB(40,4)
      DIMENSION COUNT(40,4)
      DIMENSION RMAXV(13150,40)
!C***DIMENSIONS ALLOW FOR 27 VARIABLES IN MODEL
      DIMENSION CPARM(27)
      DIMENSION IPARM(27)
      CHARACTER*8 NAME(31)
!C***DIMENSIONS ALLOW FOR 1000 PEOPLE AND UP TO 40 PERIODS
      INTEGER TIME(1000)
      INTEGER EDUC(1000,40),LSCHL(1000,40)
      INTEGER EXPER(1000,40,2)
!C***DIMENSIONS ALLOW FOR 10 VARIABLES in INTERPOLATING REGRESSION
      DIMENSION RGAMA(15),SEGAMA(15),VGAMA(15)
      DIMENSION XY(15),XR(15)
      DIMENSION XX(15,15),XXI(15,15)
      DIMENSION XXIV(225)
!C***ALLOWS FOR FORTY PERIODS AND 500 POINTS IN INTERPOLATING
      DIMENSION EMAXS(13150,40)
      INTEGER TK(13150,40)
      INTEGER TK1(13150)
      DIMENSION Y(13150) 
      INTEGER X1,X2,E,T
      INTEGER TM20,T20  
      CHARACTER*4 TRANS

      ! PEI: Set up file connectors
      open(9,file='in1.txt'); open(10,file='ouncond41.txt') 
      open(11,file='funcond41.txt'); open(12,file='ocond41.txt')
      open(13,file='fcond41.txt'); open(14,file='ftest.txt')
      open(15,file='seed.txt'); open(16,file='emax41.txt')
      open(17,file='state41.txt')

      READ(9,1500) NPER,NPOP,DRAW,DRAW1,TAU,NITER,MAXIT,MAXSTP,TRANS,INTP,PMIN     
      WRITE(10,1500) NPER,NPOP,DRAW,DRAW1,TAU,NITER,MAXIT,MAXSTP,TRANS,INTP,PMIN     
      WRITE(12,1500) NPER,NPOP,DRAW,DRAW1,TAU,NITER,MAXIT,MAXSTP,TRANS,INTP,PMIN     
 1500 FORMAT(1x,i3,1x,i5,1x,f7.0,1x,f6.0,1x,f6.2,1x,i2,1x,i2,1X,I2,1x,A3,1x,I5,1x,F4.0)
!C     DRAW1=DRAW
      NPARM=27
!C***SET NUMBER OF PARAMS IN INTERPOLATING REGRESSION
      NPT=8  
      NPTA=8  
!C***SET NUMBER OF INTERPOLATING POINTS
!C     INTP=100 
!C     INTP=250 
!C     INTP=500 
!C     INTP=1000
!C     INTP=2000
!C     INTP=13150
!C**CHOOSE PERIOD AT WHICH INTERPOLATION BEGINS
!C     INTPER=3 
!C     INTPER=9 
!C     INTPER=11
!C     INTPER=15
!C     INTPER=19
!C     INTPER = 41
!C     INTPER = 40
      IF(INTP.EQ.100) INTPER = 7
      IF(INTP.EQ.200) INTPER = 8
      IF(INTP.EQ.250) INTPER = 9
      IF(INTP.EQ.500) INTPER = 11
      IF(INTP.EQ.1000) INTPER = 15
      IF(INTP.EQ.2000) INTPER = 19
      ALPHA=0.1
      GAMA=0.57
      TAUE=500.0
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
      DO 2007 I=1,(IRUN-1)*40*NPOP
       READ(14,*)
 2007 CONTINUE
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
       C(J,T) = A(3,1)*RV(1)+A(3,2)*RV(2)+A(3,3)*RV(3)
       B(J,T) = A(4,1)*RV(1)+A(4,2)*RV(2)+A(4,3)*RV(3)+A(4,4)*RV(4)
       EU1(J,T)=EXP(U1)
       EU2(J,T)=EXP(U2)
!C      IF(T.EQ.NPER) write(11,4448) J,(RV(L),L=1,4)
!C4448  FORMAT(' DRAW=',I2,' RNN=',4f10.4) 
    9 CONTINUE
   10 CONTINUE
      CALL RNGET(ISEED)
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
!C     write(11,1510) NPER,K
!C1510 format(' t=',I2,' states=',i5)
      IF(NPER.GE.INTPER) THEN
        IF(KMAX(NPER).GE.INTP) KKMAX=INTP 
        IF(KMAX(NPER).LT.INTP) KKMAX=KMAX(NPER) 
       ELSE
        KKMAX=KMAX(NPER)
      ENDIF
!C     write(11,1511) kkmax
!C1511 format(' kkmax=',i5)
!************************************
!*  chose the interpolating points  *
!************************************
      IF(NPER.GE.INTPER) THEN
!C     write(11,1512) NPER,INTPER 
!C1512 format(' A: NPER=',I2,' INTPER=',I2)
      CALL RNOPT(1)
      CALL RNSET(ISEED2)
!C     write(11,9666) NPER,KMAX,ISEED2
!C9666 format(' CALLING RNSRI, T= ',I2,' KMAX=',I5,' SEED=',I10)
      CALL RNSRI(KKMAX,KMAX(NPER),TK1)
!C     write(11,9667) (TK1(k),k=1,50)
!C9667 format(8I7)  
      CALL RNGET(ISEED2)
      DO 715 K=1,KKMAX
       TK(K,NPER) = TK1(K)
  715 CONTINUE
      ELSE
!C     write(11,1513) NPER,INTPER 
!C1513 format(' B: NPER=',I2,' INTPER=',I2)
       DO 716 K=1,KMAX(NPER)
        TK(K,NPER) = K
  716  CONTINUE
      ENDIF
!C     WRITE(11,4015) NPER,KMAX(NPER)
!C4015 FORMAT(' T=',I2,' KMAX=',I5)
!C****INITIALIZE THE EXPECTED MAX TO ZERO
      DO 520 IQ=1,NPTA
        XY(IQ) = 0.0
       DO 521 IR=1,NPTA
         XX(IQ,IR) = 0.0
  521  CONTINUE
  520 CONTINUE
      YBAR = 0.0
      YBAR1 = 0.0
      DO 7021 KK=1,KKMAX  
       K = TK(KK,NPER)   
!C      IF(NP.EQ.NPARM+1) WRITE(11,4016) NPER,KK,K,E-9,X1,X2,LS
!C4016  FORMAT(' T=',I2,' KK=',I2,' K=',I5,' E=',I2,' X1=',I2,
!C    *    ' X2=',I2,' LS=',I2)
       X1=KSTATE(K,NPER,1)
       X2=KSTATE(K,NPER,2)
       E=KSTATE(K,NPER,3)
       LS=KSTATE(K,NPER,4)
       W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
       W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
       IF(E.GE.12) THEN
         CBAR = CBAR1 - CBAR2
        ELSE
         CBAR = CBAR1
       ENDIF
       IF(LS.EQ.0) CBAR = CBAR - CS
       IF(E.GT.19) CBAR = CBAR - 50000.0
      EMAX(K,NPER) = 0.0
      DO 22 J=1,DRAW
        V1=W1*EU1(J,NPER)
        V2=W2*EU2(J,NPER)
        V3=CBAR+C(J,NPER)
        V4=VHOME+B(J,NPER) 
!C       WRITE(11,4149) J,EU1(J,NPER),EU2(J,NPER),C(J,NPER),B(J,NPER)
!C4149   FORMAT(/,' DRAW=',I2,' E1=',F10.4,' E2=',F10.4,' E3=',f10.4,
!C    *     ' E4=',F10.4)
!C       WRITE(11,4150) J,V1,V2,V3,V4
!C4150   FORMAT(/,' DRAW=',I2,' V1=',F10.2,' V2=',F10.2,' V3=',f10.2,
!C    *     ' V4=',F10.2)
        VMAX=AMAX1(V1,V2,V3,V4)
!C       SUMV=EXP((V1-VMAX)/TAUE)+EXP((V2-VMAX)/TAUE)
!C    *      +EXP((V3-VMAX)/TAUE)+EXP((V4-VMAX)/TAUE)
!C       EMAX(K,NPER)=EMAX(K,NPER)+TAUE*(GAMA+LOG(SUMV)+VMAX/TAUE)
        EMAX(K,NPER)=EMAX(K,NPER)+VMAX
   22 CONTINUE
      EMAX(K,NPER) = EMAX(K,NPER)/DRAW
      IF(NPER.GE.INTPER) THEN
      EV1 = W1*EXP(SIGMA(1)**2/2.0)             
      EV2 = W2*EXP(SIGMA(2)**2/2.0)             
      RMAXV(K,NPER) = AMAX1(EV1,EV2,CBAR,VHOME)
!C     write(11,1190) k,X1,X2,E,LS
!C1190 format(' k=',I5,' X1=',I2,' X2=',I2,' E=',I2,' LS=',I2)
!C     WRITE(11,1191) EV1,EV2,CBAR,VHOME
 1191 format(' V1=',f12.5,' V2=',f12.5,' V3=',f12.5,' V4=',F12.5, ' ME=',f12.5)                                          
      IF(RMAXV(K,NPER).GT.EMAX(K,NPER)) EMAX(K,NPER) = RMAXV(K,NPER) 
!C     ARG = (EMAX(K,NPER)-RMAXV(K,NPER))/RMAXV(K,NPER)
!C     write(11,1122) K,EMAX(K,NPER),RMAXV(K,NPER),ARG  
!C1122 format(' K=',I5,' EMAX=',f10.2,' maxe=',f10.2,' y=',f10.6)
!CC    Y(KK) = LOG((EMAX(K,NPER)-RMAXV(K,NPER))/RMAXV(K,NPER))
!CC    Y(KK) = (EMAX(K,NPER)-RMAXV(K,NPER))/RMAXV(K,NPER)
      Y(KK) = EMAX(K,NPER)-RMAXV(K,NPER)
!C     Y(KK) = EMAX(K,NPER)
      YBAR = YBAR + Y(KK)/KKMAX 
      YBAR1 = YBAR1 + EMAX(K,NPER)/KKMAX 
      XR(1) = 1.0
      XR(2) = PEI_SQRT(0.00001 + RMAXV(K,NPER) - EV1)
      XR(3) = PEI_SQRT(0.00001 + RMAXV(K,NPER) - EV2)
      XR(4) = PEI_SQRT(0.00001 + RMAXV(K,NPER) - CBAR)
      XR(5) = PEI_SQRT(0.00001 + RMAXV(K,NPER) - VHOME)
      XR(6) = (RMAXV(K,NPER) - EV1)
      XR(7) = (RMAXV(K,NPER) - EV2)
!C     XR(8) = (RMAXV(K,NPER) - CBAR)
      XR(8) = (RMAXV(K,NPER) - VHOME)
      DO 522 IQ = 1,NPTA
       XY(IQ) = XY(IQ) + XR(IQ)*Y(KK)
       DO 523 IR = 1,NPTA
        XX(IQ,IR) = XX(IQ,IR) + XR(IQ)*XR(IR)
  523  CONTINUE
  522 CONTINUE
      ENDIF
 7021 CONTINUE
      DO 8021 KK=1,KKMAX  
       K = TK(KK,NPER)
       EMAXS(KK,NPER) = EMAX(K,NPER)
 8021 CONTINUE   
      IF(NPER.GE.INTPER) THEN
!C     DO 5528 IQ=1,NPTA
!C     WRITE(11,5557) (XX(IQ,IR),IR=1,IQ) 
!C5557 FORMAT(' XX=',5f15.1)
!C5528 CONTINUE 
      CALL LINDS(NPTA,XX,15,XXI,15)
      DO 526 IQ=2,NPTA
      DO 527 IR=1,IQ-1
        XXI(IQ,IR) = XXI(IR,IQ)
  527 CONTINUE
  526 CONTINUE
      J=0
      DO 626 IQ=1,NPTA
      DO 627 IR=1,NPTA
        J=J+1
        XXIV(J) = XXI(IQ,IR) 
  627 CONTINUE
  626 CONTINUE  
!C     DO 5529 IQ=1,NPTA
!C     WRITE(11,5530) (XXI(IQ,IR),IR=1,IQ) 
!C5530 FORMAT(' XXI=',8f9.6)
!C5529 CONTINUE 
      DO 524 IQ=1,NPTA
        RGAMA(IQ) = 0.0
       DO 525 IR=1,NPTA
         RGAMA(IQ) = RGAMA(IQ) + XXI(IQ,IR)*XY(IR)
  525  CONTINUE
  524 CONTINUE
      WRITE(11,5553)
 5553 FORMAT(20X,'  C        EV1        EV2        EV3         EV4 ')  
      WRITE(11,5554) NPER,(RGAMA(IQ),IQ=1,NPTA)
!C***CALCULATE STANDARD ERRORS 
      SSE = 0.0
      SST = 0.0
!C     DMAX = 0.0 
      DO 528 KK=1,KKMAX  
       K=TK(KK,NPER)
       KKN=0
       DO 928 KKK=1,NPTA
        KKN = KKN + 1
  928  CONTINUE          
       X1=KSTATE(K,NPER,1)
       X2=KSTATE(K,NPER,2)
       E=KSTATE(K,NPER,3)
       LS=KSTATE(K,NPER,4)
       W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
       W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
       IF(E.GE.12) THEN                
        CBAR = CBAR1 - CBAR2
       ELSE                   
        CBAR = CBAR1
       ENDIF                   
       IF(LS.EQ.0) CBAR = CBAR - CS
       IF(E.GT.19) CBAR = CBAR - 50000.0
       EV1 = W1*EXP(SIGMA(1)**2/2.0)             
       EV2 = W2*EXP(SIGMA(2)**2/2.0)             
!C      DUMMY1 = 0.0                         
!C      DUMMY2 = 0.0                         
!C      DUMMY3 = 0.0                         
!C      IF(RMAXV(K,NPER).EQ.EV1) DUMMY1 = 1.0
!C      IF(RMAXV(K,NPER).EQ.EV2) DUMMY2 = 1.0
!C      IF(RMAXV(K,NPER).EQ.VHOME) DUMMY3 = 1.0
       XR2 = PEI_SQRT(0.00001 + RMAXV(K,NPER) - EV1)
       XR3 = PEI_SQRT(0.00001 + RMAXV(K,NPER) - EV2)
       XR4 = PEI_SQRT(0.00001 + RMAXV(K,NPER) - CBAR)
       XR5 = PEI_SQRT(0.00001 + RMAXV(K,NPER) - VHOME)
       XR6 = (RMAXV(K,NPER) - EV1)
       XR7  = (RMAXV(K,NPER) - EV2)
!C      XR8  = (RMAXV(K,NPER) - CBAR)
       XR8  = (RMAXV(K,NPER) - VHOME)
       XGAMA = RGAMA(1) + RGAMA(2)*XR2  + RGAMA(3)*XR3 & 
                        + RGAMA(4)*XR4  + RGAMA(5)*XR5 & 
                        + RGAMA(6)*XR6  + RGAMA(7)*XR7 &
                        + RGAMA(8)*XR8  
!C      YHAT = AMAX1(RMAXV(K,NPER),XGAMA)
       YHAT = AMAX1(0.0,XGAMA)
!C      IF(YHAT-RMAXV(K,NPER).GT.DMAX) DMAX = YHAT - RMAXV(K,NPER)
       SSE = SSE + ( Y(KK) - YHAT )**2 
       SST = SST + ( Y(KK) - YBAR )**2 
!C      WRITE(11,6551) KK,K,X1,X2,E,LS
!C6551  FORMAT(' KK=',I2,' K=',I5,' X1=',I2,' X2=',I2,' E=',I2,' LS=',I2)
!C      WRITE(11,5700) NPER,KK,K,EMAX(K,NPER),RMAXV(K,NPER)
!C      WRITE(11,6554) KK,Y(KK),XGAMA,YBAR,SSE,SST
  528 CONTINUE
      RSQ=(SST-SSE)/SST
      DO 561 IQ=1,NPTA
        IR=IQ*((NPTA)+1)-(NPTA)   
        VGAMA(IQ) = XXIV(IR)*SSE/(KKMAX-NPTA)
  561 CONTINUE
      DO 571 IR=1,NPTA    
        SEGAMA(IR) = SQRT(VGAMA(IR))
  571 CONTINUE
      WRITE(11,5555) NPER,(SEGAMA(IQ),IQ=1,NPTA)
      WRITE(11,6555) SSE,SST,RSQ,YBAR1,INTP
 6555 FORMAT(' SSE=',f12.0,' SST=',F13.0,' RSQ=',F7.5,' YBAR=',f8.1, ' INTP=',I5)
!C     WRITE(11,7555) DMAX 
!C7555 FORMAT(' DMAX = ',F12.3)
      DO 529 K=1,KMAX(NPER)
       X1=KSTATE(K,NPER,1)
       X2=KSTATE(K,NPER,2)
       E=KSTATE(K,NPER,3)
       LS=KSTATE(K,NPER,4)
       W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
       W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
       IF(E.GE.12) THEN
         CBAR = CBAR1 - CBAR2
        ELSE
         CBAR = CBAR1
       ENDIF
       IF(LS.EQ.0) CBAR = CBAR - CS
       IF(E.GT.19) CBAR = CBAR - 50000.0
       EV1 = W1*EXP(SIGMA(1)**2/2.0)             
       EV2 = W2*EXP(SIGMA(2)**2/2.0)             
       RMAXV(K,NPER) = AMAX1(EV1,EV2,CBAR,VHOME)
       XR2 = PEI_SQRT(0.00001 + RMAXV(K,NPER) - EV1)
       XR3 = PEI_SQRT(0.00001 + RMAXV(K,NPER) - EV2)
       XR4 = PEI_SQRT(0.00001 + RMAXV(K,NPER) - CBAR)
       XR5 = PEI_SQRT(0.00001 + RMAXV(K,NPER) - VHOME)
       XR6 = (RMAXV(K,NPER) - EV1)
       XR7 = (RMAXV(K,NPER) - EV2)
!C      XR8 = (RMAXV(K,NPER) - CBAR)
       XR8 = (RMAXV(K,NPER) - VHOME)
       XGAMA = RGAMA(1) + RGAMA(2)*XR2  + RGAMA(3)*XR3 &
                        + RGAMA(4)*XR4  + RGAMA(5)*XR5 &
                        + RGAMA(6)*XR6  + RGAMA(7)*XR7 &
                        + RGAMA(8)*XR8                 
       EMAX(K,NPER) = RMAXV(K,NPER) + AMAX1(0.0,XGAMA)
!C      EMAX(K,NPER) = AMAX1(RMAXV(K,NPER),XGAMA)
!CC     IF(NP.EQ.NPARM+1) write(11,4100) NPER,k,emax(k,NPER),RMAXV(K,NPER)
!C      write(16,1115) EV1,EV2,CBAR,EMAXS(K,NPER),EMAX(K,NPER)
!C1115  format(F10.2,1x,F10.2,1x,F10.2,1x,F10.2,1x,F10.2)                   
  529 CONTINUE 
!C     DMAX1 = DMAX
!C     DO 9553 K=1,KMAX(NPER)
!C      IF(EMAX(K,NPER)-RMAXV(K,NPER).GT.DMAX) THEN
!C        DD = EMAX(K,NPER)-RMAXV(K,NPER)
!C        IF(DD.GT.DMAX1) DMAX1 = DD     
!C        X1=KSTATE(K,NPER,1)
!C        X2=KSTATE(K,NPER,2)
!C        E=KSTATE(K,NPER,3)
!C        LS=KSTATE(K,NPER,4)
!C        write(11,9555) K,X1,X2,E,LS,EMAX(K,NPER),RMAXV(K,NPER)
!C      ENDIF                            
!C9553 CONTINUE  
!C     write(11,9551) DMAX1                        
!C9551 format(' DMAX1 = ',F10.0)                   
      DO 9021 KK=1,KKMAX  
       K = TK(KK,NPER)
!C      write(17,9921) KK,K,EMAX(K,NPER),EMAXS(KK,NPER)
!C9921  format(' KK=',I2,' K=',I5,' EMAX=',F10.2,' EMAXS=',F10.2)
       EMAX(K,NPER) = EMAXS(KK,NPER)
 9021 CONTINUE            
      ENDIF    
!C     goto 999 
!**********************************************************
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
!C     WRITE(11,5015) T,KMAX
!C5015 FORMAT(' T=',I2,' KMAX=',I5) 
      IF(T.GE.INTPER) THEN
        IF(KMAX(T).GE.INTP) KKMAX=INTP 
        IF(KMAX(T).LT.INTP) KKMAX=KMAX(T) 
       ELSE
        KKMAX=KMAX(T)
      ENDIF
      IF(T.GE.INTPER) THEN
       CALL RNOPT(1)
       CALL RNSET(ISEED2)
       CALL RNSRI(KKMAX,KMAX(T),TK1)
       CALL RNGET(ISEED2)
!C      write(11,9667) (TK1(k),k=1,KKMAX)
       DO 717 K=1,KKMAX
        TK(K,T) = TK1(K)
  717  CONTINUE
      ELSE
       DO 718 K=1,KMAX(T)
        TK(K,T) = K
  718  CONTINUE
      ENDIF
!C****INITIALIZE THE EXPECTED MAX TO ZERO
      DO 550 IQ=1,NPT
        XY(IQ) = 0.0
       DO 551 IR=1,NPT
         XX(IQ,IR) = 0.0
  551  CONTINUE
  550 CONTINUE
      YBAR = 0.0
      YBAR1 = 0.0
      DO 7052 KK=1,KKMAX                           
      K = TK(KK,T) 
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
      EMAX(K,T) = 0.0   
      DO 53 J=1,DRAW
        V1=W1*EU1(J,T)  + E1    
        V2=W2*EU2(J,T)  + E2 
        V3=C(J,T)       + E3
        V4=B(J,T)       + E4
        VMAX=AMAX1(V1,V2,V3,V4)
!C       SUMV=EXP((V1-VMAX)/TAUE)+EXP((V2-VMAX)/TAUE)
!C    *      +EXP((V3-VMAX)/TAUE)+EXP((V4-VMAX)/TAUE)
!C       EMAX(K,T)=EMAX(K,T)+TAUE*(GAMA+LOG(SUMV)+VMAX/TAUE)
        EMAX(K,T)=EMAX(K,T)+VMAX
   53 CONTINUE
      EMAX(K,T) = EMAX(K,T)/DRAW
      IF(T.GE.INTPER) THEN
      EW1 = W1*EXP(SIGMA(1)**2/2.0)
      EW2 = W2*EXP(SIGMA(2)**2/2.0)
      RMAXV(K,T) = AMAX1(EW1+E1,EW2+E2,E3,E4)
      IF(RMAXV(K,T).GT.EMAX(K,T)) EMAX(K,T) = RMAXV(K,T) 
!C     ARG = (EMAX(K,T)-RMAXV(K,T))/RMAXV(K,T)
!C     write(11,1190) k,X1,X2,E,LS
!C     WRITE(11,1191) E1+EW1,E2+EW2,E3,E4,RMAXV(K,T)     
!C     write(11,1122) K,EMAX(K,T),RMAXV(K,T),ARG  
!CC    Y(KK) = LOG((EMAX(K,T)-RMAXV(K,T))/RMAXV(K,T))
!CC    Y(KK) = (EMAX(K,T)-RMAXV(K,T))/RMAXV(K,T)
      Y(KK) = EMAX(K,T)-RMAXV(K,T)
!C     Y(KK) = EMAX(K,T)
      YBAR = YBAR + Y(KK)/KKMAX 
      YBAR1 = YBAR1 + EMAX(K,T)/KKMAX 
      XR(1) = 1.0
      XR(2) = PEI_SQRT(0.00001 + RMAXV(K,T) - EW1 - E1)
      XR(3) = PEI_SQRT(0.00001 + RMAXV(K,T) - EW2 - E2)
      XR(4) = PEI_SQRT(0.00001 + RMAXV(K,T) - E3)
      XR(5) = PEI_SQRT(0.00001 + RMAXV(K,T) - E4)
      XR(6) = (RMAXV(K,T) - EW1 - E1)
      XR(7) = (RMAXV(K,T) - EW2 - E2)
!C     XR(8) = (RMAXV(K,T) - E3)
      XR(8) = (RMAXV(K,T) - E4)
      DO 552 IQ = 1,NPT
       XY(IQ) = XY(IQ) + XR(IQ)*Y(KK)
       DO 553 IR = 1,NPT
        XX(IQ,IR) = XX(IQ,IR) + XR(IQ)*XR(IR)
  553  CONTINUE
  552 CONTINUE
      ENDIF
 7052 CONTINUE
      DO 8052 KK=1,KKMAX  
       K = TK(KK,T)   
       EMAXS(KK,T) = EMAX(K,T)   
 8052 CONTINUE   
      IF(T.GE.INTPER) THEN
!C     DO 5558 IQ=1,NPT
!C     WRITE(11,5557) (XX(IQ,IR),IR=1,IQ)
!C5558 CONTINUE 
      CALL LINDS(NPT,XX,15,XXI,15)
      DO 557 IQ=2,NPT
      DO 558 IR=1,IQ-1
        XXI(IQ,IR) = XXI(IR,IQ)
  558 CONTINUE
  557 CONTINUE
      J=0
      DO 628 IQ=1,NPT
      DO 629 IR=1,NPT
        J=J+1
        XXIV(J) = XXI(IQ,IR) 
  629 CONTINUE
  628 CONTINUE  
!C     DO 5559 IQ=1,NPT
!C     WRITE(11,5530) (XXI(IQ,IR),IR=1,NPT)
!C5559 CONTINUE 
      DO 554 IQ=1,NPT
        RGAMA(IQ) = 0.0
       DO 555 IR=1,NPT
         RGAMA(IQ) = RGAMA(IQ) + XXI(IQ,IR)*XY(IR)
  555  CONTINUE
  554 CONTINUE
      WRITE(11,5553)
      WRITE(11,5554) T,(RGAMA(IQ),IQ=1,NPT)
 5554 FORMAT(' T=',I2,' GAMA=',5F12.6,/,11x,5F12.6,/,11x,5f12.6)   
!C***CALCULATE STANDARD ERRORS 
      SSE = 0.0
      SST = 0.0
!C     DMAX = 0.0
      DO 562 KK=1,KKMAX 
       K=TK(KK,T)
       KKN=0
       DO 929 KKK=1,NPT
        KKN = KKN + 1
  929  CONTINUE          
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
       EW1 = W1*EXP(SIGMA(1)**2/2.0)             
       EW2 = W2*EXP(SIGMA(2)**2/2.0)             
       V1=EW1   + DELTA*EMAX(FSTATE(T+1,X1+2,X2+1,E-9,1),T+1)
       V2=EW2   + DELTA*EMAX(FSTATE(T+1,X1+1,X2+2,E-9,1),T+1)
       IF(E.LE.19) THEN 
         V3=CBAR  + DELTA*EMAX(FSTATE(T+1,X1+1,X2+1,E-8,2),T+1)
        ELSE              
         V3=CBAR  - 50000.0             
       ENDIF                                      
       V4=VHOME + DELTA*EMAX(FSTATE(T+1,X1+1,X2+1,E-9,1),T+1)
!C      DUMMY1 = 0.0                         
!C      DUMMY2 = 0.0                         
!C      DUMMY3 = 0.0                         
!C      IF(RMAXV(K,T).EQ.V1) DUMMY1 = 1.0
!C      IF(RMAXV(K,T).EQ.V2) DUMMY2 = 1.0
!C      IF(RMAXV(K,T).EQ.V4) DUMMY3 = 1.0
       XR2 = PEI_SQRT(0.00001 + RMAXV(K,T) - V1)
       XR3 = PEI_SQRT(0.00001 + RMAXV(K,T) - V2)
       XR4 = PEI_SQRT(0.00001 + RMAXV(K,T) - V3)
       XR5 = PEI_SQRT(0.00001 + RMAXV(K,T) - V4)
       XR6 = (RMAXV(K,T) - V1)
       XR7 = (RMAXV(K,T) - V2)
!C      XR8 = (RMAXV(K,T) - V3)
       XR8 = (RMAXV(K,T) - V4)
       XGAMA = RGAMA(1) + RGAMA(2)*XR2  + RGAMA(3)*XR3 &
                        + RGAMA(4)*XR4  + RGAMA(5)*XR5 & 
                        + RGAMA(6)*XR6  + RGAMA(7)*XR7 & 
                        + RGAMA(8)*XR8    
!C      YHAT = AMAX1(RMAXV(K,T),XGAMA)
       YHAT = AMAX1(0.0,XGAMA)
!C      IF(YHAT-RMAXV(K,T).GT.DMAX) DMAX = YHAT - RMAXV(K,T)
       SSE = SSE + ( Y(KK) - YHAT )**2 
       SST = SST + ( Y(KK) - YBAR )**2 
!C      WRITE(11,6551) KK,K,X1,X2,E,LS
!C      WRITE(11,5700) T,KK,K,EMAX(K,T),RMAXV(K,T)
!C      WRITE(11,6554) KK,Y(KK),XGAMA,YBAR,SSE,SST
!C6554  FORMAT(' KK=',I2,' Y=',F8.3,' YHAT=',F8.3,' YBAR=',F8.3,  
!C    *     ' SSE=',F8.3,' SST=',F8.3)
  562 CONTINUE
      RSQ = (SST - SSE)/SST 
      DO 560 IQ=1,NPT
        IR=IQ*(NPT+1)-NPT   
        VGAMA(IQ) = XXIV(IR)*SSE/(KKMAX-NPT)
  560 CONTINUE
      DO 570 IR=1,NPT
        SEGAMA(IR) = SQRT(VGAMA(IR))
  570 CONTINUE       
      WRITE(11,5555) T,(SEGAMA(IQ),IQ=1,NPT)
 5555 FORMAT(' T=',I2,' SE  =',5F12.6,/,11x,5F12.6,/,11x,5f12.6)  
      WRITE(11,6555) SSE,SST,RSQ,YBAR1,INTP
!C     WRITE(11,7555) DMAX 
      DO 559 K=1,KMAX(T)
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
       EW1 = W1*EXP(SIGMA(1)**2/2.0)             
       EW2 = W2*EXP(SIGMA(2)**2/2.0)             
       V1=EW1   + DELTA*EMAX(FSTATE(T+1,X1+2,X2+1,E-9,1),T+1)
       V2=EW2   + DELTA*EMAX(FSTATE(T+1,X1+1,X2+2,E-9,1),T+1)
       IF(E.LE.19) THEN 
         V3=CBAR  + DELTA*EMAX(FSTATE(T+1,X1+1,X2+1,E-8,2),T+1)
        ELSE              
         V3=CBAR  - 50000.0             
       ENDIF                                      
       V4=VHOME + DELTA*EMAX(FSTATE(T+1,X1+1,X2+1,E-9,1),T+1)
       RMAXV(K,T) = AMAX1(V1,V2,V3,V4)         
       XR2 = PEI_SQRT(0.00001 + RMAXV(K,T) - V1)
       XR3 = PEI_SQRT(0.00001 + RMAXV(K,T) - V2)
       XR4 = PEI_SQRT(0.00001 + RMAXV(K,T) - V3)
       XR5 = PEI_SQRT(0.00001 + RMAXV(K,T) - V4)
       XR6 = (RMAXV(K,T) - V1)
       XR7 = (RMAXV(K,T) - V2)
!C      XR8 = (RMAXV(K,T) - V3)
       XR8 = (RMAXV(K,T) - V4)
       XGAMA = RGAMA(1) + RGAMA(2)*XR2  + RGAMA(3)*XR3 &
                        + RGAMA(4)*XR4  + RGAMA(5)*XR5 &
                        + RGAMA(6)*XR6  + RGAMA(7)*XR7 & 
                        + RGAMA(8)*XR8                  
!C    *                  + RGAMA(8)*XR8  + RGAMA(9)*XR9  
!CC     EMAX(K,T) = RMAXV(K,T)*(1.0 + EXP(XGAMA))
!CC     EMAX(K,T) = RMAXV(K,T)*(1.0 + XGAMA)
       EMAX(K,T) = RMAXV(K,T) + AMAX1(0.0,XGAMA)
!C      EMAX(K,T) = AMAX1(RMAXV(K,T),XGAMA)
!CC     IF(NP.EQ.NPARM+1) write(11,4100) T,k,emax(K,T),RMAXV(K,T)
!C4100 FORMAT(' T=',i2,' K=',i3,' EMAX=',f12.2,' RMAXV=',F12.2)
  559 CONTINUE 
!C     DMAX1 = DMAX
!C     DO 9554 K=1,KMAX(T)
!C      IF(EMAX(K,T)-RMAXV(K,T).GT.DMAX) THEN
!C        DD = EMAX(K,T)-RMAXV(K,T)
!C        IF(DD.GT.DMAX1) DMAX1 = DD     
!C        X1=KSTATE(K,T,1)
!C        X2=KSTATE(K,T,2)
!C        E=KSTATE(K,T,3)
!C        LS=KSTATE(K,T,4)
!C        write(11,9555) K,X1,X2,E,LS,EMAX(K,T),RMAXV(K,T)
!C9555    format(' K=',I5,' X1=',I2,' X2=',I2,' E=',I2,' LS=',I1,
!C    *      ' EMAX=',F9.0,' MAXE=',F9.0)
!C      ENDIF                            
!C9554 CONTINUE                          
!C     write(11,9551) DMAX1                        
      DO 9052 KK=1,KKMAX  
       K = TK(KK,T)   
!C      write(11,9921) KK,K,EMAX(K,T),EMAXS(KK,T)
       EMAX(K,T) = EMAXS(KK,T)   
 9052 CONTINUE   
      ENDIF
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
      CME = 0.0
      CMX1 = 0.0
      CMX2 = 0.0
      DO 60 I=1,NPOP
      DO 330 T=1,NPER-1  
         E=EDUC(I,T)
         X1=EXPER(I,T,1)
         X2=EXPER(I,T,2)
         LS=LSCHL(I,T)
!C        WRITE(13,1333) I,T,E,X1,X2,LS
!C1333    FORMAT(' I=',I2,' T=',I2,' E=',I2,' X1=',I2,' X2=',I2,
!C    *      ' LS=',I2)
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
       ENDIF
       IF(VMAX .EQ. V2) THEN
         K=2 
         WRITE(13,1000) I,T,K,WAGE2,X1,X2,E,LS
       ENDIF
       IF(VMAX .EQ. V3) THEN
         K=3 
         WRITE(13,1000) I,T,K,WAGE3,X1,X2,E,LS
       ENDIF
       IF(VMAX .EQ. V4) THEN
         K=4 
         WRITE(13,1000) I,T,K,WAGE4,X1,X2,E,LS
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
         X1 = X1+1
       ENDIF
       IF(VMAX .EQ. V2) THEN
         K=2 
         WRITE(13,1000) I,T,K,WAGE2,X1,X2,E,LS
         X2 = X2 + 1
       ENDIF
       IF(VMAX .EQ. V3) THEN
         K=3 
         WRITE(13,1000) I,T,K,V3,X1,X2,E,LS
         E = E + 1   
       ENDIF
       IF(VMAX .EQ. V4) THEN
         K=4 
         WRITE(13,1000) I,T,K,V4,X1,X2,E,LS
       ENDIF
       COUNT(T,K) = COUNT(T,K) + 1.d0/NPOP
       CMX1 = CMX1 + X1                   
       CMX2 = CMX2 + X2                   
       CME  = CME  + E                    
!C***END OF LOOP OVER PEOPLE
   60 CONTINUE
       CMX1 = CMX1/NPOP                   
       CMX2 = CMX2/NPOP                   
       CME  = CME/NPOP  
      write(12,3001) CMX1,CMX2,CME 
 3001 format(' CONDITIONAL MEANS AT END OF LIFE :',/, '   X1 = ',F8.4,' X2=',F8.4,' E=',F8.4,/)                
      do 71 t=1,nper
        write(12,3000) t,(COUNT(t,j),j=1,4)
   71 continue
!***********************************************************
!*  CONSTRUCT MONTE-CARLO DATA FOR PERIODS 1 THROUGH NPER  *
!***********************************************************
      do 58 j=1,4
      do 59 t=1,nper
        prob(t,j)=0.0
  59  continue
  58  continue
      wealth = 0.0 
      UMX1 = 0.0 
      UMX2 = 0.0 
      UME  = 0.0 
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
       IF (VMAX .EQ. V1) THEN
         K=1
         WRITE(11,1000) I,T,K,WAGE1,X1,X2,E,LS
         wealth = wealth + (DELTA**(T-1))*WAGE1
         X1=X1+1
         LS1=0
       ENDIF
       IF (VMAX .EQ. V2) THEN
         K=2
         WRITE(11,1000) I,T,K,WAGE2,X1,X2,E,LS
         wealth = wealth + (DELTA**(T-1))*WAGE2
         X2=X2+1
         LS1=0
       ENDIF
       IF (VMAX .EQ. V3) THEN
         K=3
         WRITE(11,1000) I,T,K,WAGE3,X1,X2,E,LS
         wealth = wealth + (DELTA**(T-1))*WAGE3
         E=E+1
         LS1=1
       ENDIF
       IF (VMAX .EQ. V4) THEN
         K=4
         WRITE(11,1000) I,T,K,WAGE4,X1,X2,E,LS
         wealth = wealth + (DELTA**(T-1))*WAGE4
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
       IF (VMAX .EQ. V1) THEN
         K=1
         WRITE(11,1000) I,T,K,WAGE1,X1,X2,E,LS
         X1 = X1 + 1
         wealth = wealth + (DELTA**(T-1))*WAGE1
       ENDIF
       IF (VMAX .EQ. V2) THEN
         K=2
         WRITE(11,1000) I,T,K,WAGE2,X1,X2,E,LS
         X2 = X2 + 1
         wealth = wealth + (DELTA**(T-1))*WAGE2
       ENDIF
       IF (VMAX .EQ. V3) THEN
         K=3
         WRITE(11,1000) I,T,K,V3,X1,X2,E,LS
         E = E + 1
         wealth = wealth + (DELTA**(T-1))*WAGE3
       ENDIF
       IF (VMAX .EQ. V4) THEN
         K=4
         WRITE(11,1000) I,T,K,V4,X1,X2,E,LS
         wealth = wealth + (DELTA**(T-1))*WAGE4
       ENDIF
       prob(t,k)=prob(t,k)+1.0/npop
       UMX1 = UMX1 + X1            
       UMX2 = UMX2 + X2            
       UME  = UME  + E             
  260 CONTINUE
      wealth = wealth/NPOP
      UMX1 = UMX1/NPOP            
      UMX2 = UMX2/NPOP            
      UME  = UME/NPOP             
      write(10,3002) UMX1,UMX2,UME 
 3002 format(/,' UNCONDITIONAL MEANS AT END OF LIFE :',/, '   X1 = ',F8.4,' X2=',F8.4,' E=',F8.4,/)                
      write(10,1170) wealth
 1170 format(' discounted wealth = ',F16.2)
      do 70 t=1,nper
        write(10,3000) t,(prob(t,j),j=1,4)
   70 continue
 3000 FORMAT(' T=',I3,' PROB=',4F16.12)
!C     REWIND(unit=15) 
!C     WRITE(15,1313) IRUN+1,ISEED,ISEED1,ISEED2                               
      DO 4001 J=1,IRUN-1
       READ(17,*)
 4001 CONTINUE
      WRITE(17,4000) IRUN,UMX1,UMX2,UME        
 4000 FORMAT(1x,I3,1x,F8.4,1x,F8.4,1x,F8.4)
  999 CONTINUE
      STOP
      END
