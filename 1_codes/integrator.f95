       program earthtest 
       implicit real*8 (a-h,o-z) 
       dimension ff(6)
       external diff
       logical ok
       character*80 idfile 
       real l_p 
       real*8 q 
       common /params/ B_0,pm,q,r_e,dpm,c,pi 
       q=-4.8e-10 
       l_p=pi. 
       pm=9.1e-28 
       idfile = 'subtest.8' 
       call earth(l_p,idfile) 
       stop 
       end 






       subroutine earth(l_p,idfile) 

!  (in this code  cgs units are used ) 



      implicit real*8 (a-h,o-z) 
      character*80  idfile 
      dimension ff(6) 
      common /params/ B_0,pm,q,r_e,dpm,c,pi 
      real L, l_m, l_p, l_pp 
      external diff 
      logical OK 
      call startup 
! initial conditions 
      L=6.68 
      x=r_e*L 
      y=0. 
      z=0. 
! energy KE = 1.e6 ev 
      xe=1.e6*1.e-12 
      v=sqrt(2*xe/pm) 
! mirror lattitude 
      l_m=20.*pi/180. 
! B at the equator 
      B_e=B_0/L**3 


! pitch angle l_p 

      l_p=pi/4. 
      
      vx=0. 
      vy=v*sin(l_p) 
      vz=v*cos(l_p) 
! Larmor radius 
      r_L=pm*vy*c/(q*B_e) 
      x=x+r_L 
! integrator settings: 
      n=6 
      acc=1.e-12 
      h=1.e-1 
      hmin=1.e-13 
      jtest=0 
      
      ff(1)=x 
      ff(2)=y
      ff(3)=z 
      ff(4)=vx 
      ff(5)=vy 
      ff(6)=vz 
      t1=0. 
      dt=1.e-2 

      
      do 100 i=1,100*60 
       t2=t1+dt 
! to avoid crash 
         r1=sqrt(ff(1)**2+ff(2)**2+ff(3)**2) 
         if(r1.le.r_e) go to 1000 
! integration begins 
       call merson(t1,t2,ff,6,acc,h,hmin,jtest,OK,diff) 
      
      t1=t2 
      r=sqrt(ff(1)**2+ff(2)**2)/r_e 
      xre=ff(1)/r_e 
      yre=ff(2)/r_e     
      zre=ff(3)/r_e 
      open(unit=20,file=idfile) 
      write(20,66)t1,xre,yre,zre,ff(4)/1.e5,ff(5)/1.e5,ff(6)/1.e5 

66    format(7e15.7) 
100   continue 

1000  continue 
    
      return 
      end 
      
       

      subroutine startup 
      implicit real*8 (a-h,o-z) 
      common /params/ B_0,pm,q,r_e,dpm,c,pi 

! proton 
!     pm=1.6726e-24 
!     q=4.8e-10 

! Earth 
      r_e=6384.4e5 
      B_0=0.3 
      dpm=B_0*r_e**3 
!     print*,'dpm=',dpm 

! speed of light 
      c=3.e10 
      pi=acos(-1.) 
      return 
      end 
      

      subroutine diff(t,ff,ffp) 
      implicit real*8 (a-h,o-z) 
      common /params/ B_0,pm,q,r_e,dpm,c,pi 
      common /force/ flx,fly,flz 
      dimension ff(6), ffp(6) 
      real*8 mcnumber, magv, kineg
      logical outp

      x=ff(1) 
      y=ff(2) 
      z=ff(3) 
      vx=ff(4) 
      vy=ff(5) 
      vz=ff(6) 
      call forces(x,y,z,vx,vy,vz,fx,fy,fz) 
      ffp(1)=vx 
      ffp(2)=vy 
      ffp(3)=vz 
      ffp(4)=fx/pm 
      ffp(5)=fy/pm 
      ffp(6)=fz/pm 
      call check_collision(ff,outp)
      if (outp) then
         magv = (ff(4)**2 + ff(5)**2 + ff(6)**2)**.5
         kineg = .5*pm*(magv**2)
         if(kineg >= 2.5e-18) then
! Electron -> Ionized Atom (differentiate between oxygen and nitrogen for hypothesis)
! Reduce KE of electron by ionization energy in all three directions (random number distribution)
! Split the remaining energy among the two electrons (recrsively spawn new electron
! w/ E given by random number distribution)

! Nitloc.append(np.array([x[i],y[i],z[i]]))
            call random_number(mcnumber)

            ffp(4) = mcnumber * ff(4)
            ffp(5) = mcnumber * ff(5)
            ffp(6) = mcnumber * ff(6)
         else if(kineg >= 1.93e-18) then
! Oxloc.append(np.array([x[i],y[i],z[i]]))
            call random_number(mcnumber)
            ffp(4) = mcnumber * ff(4)
            ffp(5) = mcnumber * ff(5)
            ffp(6) = mcnumber * ff(6)
         else
! Elastic case
! Find a way to make this more accurate - good enough for now according to Jan
                ffp(4) = -ff(4)
                ffp(5) = -ff(5)
                ffp(6) = -ff(6)
         end if
       end if         
            
      return 
      end 

      subroutine check_collision(ff,outp)
      implicit real*8 (a-h,o-z) 
      common /params/ B_0,pm,q,r_e,dpm,c,pi
      dimension ff(6)
      real*8 mcnumber, magv
      logical outp
      magv = (ff(4)**2 + ff(5)**2 + ff(6)**2)**.5
! Itikawa Mason et al, table 11
      A = .025e-16
      prob = 1 - EXP(-.1*.5*A*magv)
      call random_number(mcnumber)
      outp = (prob >= mcnumber)
      return
      end 
    
    

      subroutine forces(x,y,z,vx,vy,vz,fxl,fyl,fzl) 
      implicit real*8 (a-h,o-z) 
      common /params/ B_0,pm,q,r_e,dpm,c,pi 

      call fields(x,y,z,bx,by,bz) 
! Lorentz force 
       fxl=q*(vy*bz-vz*by)/c 
       fyl=q*(vz*bx-vx*bz)/c 
       fzl=q*(vx*by-vy*bx)/c 

      return 
      end 


      subroutine fields(x,y,z,bx,by,bz) 
      implicit real*8 (a-h,o-z) 

      common /params/ B_0,pm,q,r_e,dpm,c,pi 

! aligned, centered dipole 
       r2=x**2+y**2+z**2 
       r=sqrt(r2) 
       r3=r**3 
       bx=3.*x/r3*z*dpm/r2 
       by=3.*y/r3*z*dpm/r2 
       bz=(3.*z*z/r2-1.)*dpm/r3 

      return 
      end 
      

      SUBROUTINE MERSON (X,XEND,Y,N,ACC,H,HMIN,JTEST,OK,DIFF) 
      implicit real*8 (a-h,o-z) 

! this is a nice integrator, one can force integration forward even 
! if the requested accuracy is not achieved 
!

!     CERN LIBRARY NO D 208. 
! 
!     REVISED VERSION JULY 1971. 
! 
!     THIS VERSION OF MERSON IS A MODIFICATION OF THE PROGRAM RECEIVED 
!     FROM KJELLER COMPUTER INSTALLATION NORWAY.THE MAIN DIFFERENCE 
!     BEEING A CHANGE OF THE TEST FOR STEPLENGTH HALVING. 
! 
! 
!     PURPOSE = STEP BY STEP INTEGRATION OF A SYSTEM OF FIRST ORDER 
!               DIFFERENTIAL EQUATIONS 
! 
!               DYK(X)/DX=FK(X,Y1(X),Y2(X),.....,YN(X)) , K=1(1)N 
!
!               WITH AUTOMATIC ERROR CONTROL USING THE METHOD DUE TO 
!               MERSON. 
! 
!     PARAMETERS 
! 
!     X       = START VALUE FOR THE DOMAIN OF INTEGRATION (INPUT POINT). 
!     XEND    = END VALUE FOR THE DOMAIN OF INTEGRATION (OUTPUT POINT). 
!     Y       = ARRAY CONTAINING THE DEPENDENT VARIABLES.WHEN ENTERING 
!               THE ROUTINE THE FUNCTION VALUES YK(X),K=1(1)N AND WHEN 
!               RETURNING TO THE CALLING PROGRAM THE COMPUTED FUNCTION 
!               VALUES YK(XEND),K=1(1)N. 
!     N       = THE NUMBER OF DIFFERENTIAL EQUATION,EQUAL OR LESS 100. 
!     ACC     = PRESCRIBED RELATIVE ACCURACY (TO BE OBTAINED FOR ALL 
!               FUNCTION VALUES IN THE ARRAY Y). 
!     H       = INITIAL STEPLENGTH. 
!     HMIN    = ABSOLUTE VALUE OF MINIMUM STEPLENGTH WANTED. 
!     JTEST   = TEST PARAMETER RELATED TO THE STEPLENGTH IN THE FOLLOW- 
!               ING WAY. 
!               JTEST = 0 , IF DURING THE CALCULATION WE GET 
!                           ABS(H).LT.HMIN (BY REPEATED HALVING OF THE 
!                           STEPLENGTH),THEN AN ERROR MESSAGE IS PRINT- 
!                           ED,OK SET EQUAL TO .FALSE. FOLLOWED BY RE- 
!                           TURN TO THE CALLING PROGRAM. 
!               JTEST = 1 , CHECKING AS FOR JTEST=0,BUT THE CALCULATION 
!                           WILL CONTINUE WITH THE FIXED STEPLENGTH HMIN 
!     OK      = A LOGICAL VARIABLE WHICH IS SET EQUAL TO .TRUE. WHEN EN- 
!               TERING THE ROUTINE.THE VALUE LATER IS DEPENDING OF JTEST 
!               IN THE FOLLOWING WAY 
!               JTEST = 1 , OK = .TRUE. ALWAYS. 
!               JTEST = 0 , OK = .FALSE. IF THE STEPLENGTH BECOMES TOO 
!               SMALL (SEE DESCRIPTION FOR JTEST). 
!     DIFF    = USER SUPPLIED SUBROUTINE FOR THE CALCULATION OF THE 
!               RIGHT HAND SIDES OF THE SYSTEM OF DIFFERENTIAL EQUATIONS 
!               (I.E.THE FIRST ORDER DERIVATIVES).CALLING SEQUENCE 
!               CALL DIFF(X,W,F) , WHERE 
!               X = THE CURRENT VALUE OF THE ARGUMENT 
!               W = AN ARRAY WITH THE ELEMENTS WK(X),K=1(1)N I.E. THE 
!                   FUNTION VALUES FOR WHICH THE DERIVATIVES ARE TO BE 
!                   COMPUTED. 
!               F = AN ARRAY WITH THE ELEMENTS FK(X),K=1(1)N (WE HAVE 
!                   FK(X)=FK(X,W1(X),W2(X),....,WN(X))) TO BE COMPUTED 
!                   BY  DIFF AND RETURNED TO MERSON. 
!               THE CHOSEN NAME FOR DIFF MUST APPEAR IN AN EXTERNAL 
!               STATEMENT IN THE PROGRAM CALLING MERSON. 
! 
!     THE ARRAYS IN THE COMMON BLOCK BELOW ARE ONLY USED INTERNALLY IN 
!     MERSON (AND DIFF) AND ARE TO FREE DISPOSAL OUTSIDE MERSON. 
!     THE MAXIMUM NUMBER OF EQUATIONS WHICH  MAY BE INTEGRATED ARE 100. 
!     THIS NUMBER MAY BE CHANGED BY CHANGING THE DIMENSION IN THE 
!     COMMON BLOCK BELOW ACCORDINGLY. 
! 
      COMMON / LOCAL / YZ(100) , A(100) , B(100) , F(100) , W(100) 
! 
      LOGICAL OK 
      DIMENSION Y(N) 
! 
!     RZERO IS A NUMBER WITH A MAGNITUDE OF ORDER EQUAL TO THE NOISE 
!     LEVEL OF THE MACHINE I.E. IN THE RANGE OF THE ROUNDING OFF ERRORS. 
!
      DATA RZERO / 1.E-13 / 
! 
!     CHECK NUMBER OF EQUATIONS,EQUAL OR LESS THAN 100. 
! 
      IF (N.GT.100) GO TO 86 
! 
      OK=.TRUE. 
! 
!     STORE INTERNALLY PARAMETERS IN LIST 
! 
      NN=N 
      DO 1 K=1,NN 
    1 W(K)=Y(K) 
      Z=X 
      ZEND=XEND 
      BCC=ACC 
      ZMIN=HMIN 
      ITEST=JTEST 
      S   =H 
      ISWH=0 
! 
    2 HSV=S 
      COF=ZEND-Z 
      IF (ABS(S).LT.ABS(COF)) GO TO 8 
      S=COF 
      IF (ABS(COF/HSV).LT.RZERO) GO TO 50 
      ISWH=1 
! 
!     IF ISWH=1 THEN  S IS EQUAL TO MAXIMUM POSSIBLE STEPLENGTH 
!     WITHIN THE REMAINING PART OF THE DOMAIN OF INTEGRATION. 
! 
    8 DO 10 K=1,NN 
   10 YZ(K)=W(K) 
   12 HT=.3333333333333*S 
! 
      CALL DIFF(Z,W,F) 
! 
      Z=Z+HT 
      DO 20 K=1,NN 
      A(K)=HT*F(K) 
   20 W(K)=A(K)+YZ(K) 
! 
      CALL DIFF(Z,W,F) 
! 
      DO 22 K=1,NN 
      A(K)=.5*A(K) 
   22 W(K)=.5*HT*F(K)+A(K)+YZ(K) 
! 
      CALL DIFF(Z,W,F) 
! 
      Z=Z+.5*HT 
      DO 24 K=1,NN 
      B(K)=4.5*HT*F(K) 
   24 W(K)=.25*B(K)+.75*A(K)+YZ(K) 
! 
      CALL DIFF(Z,W,F) 
! 
      Z=Z+.5*S 
      DO 26 K=1,NN 
      A(K)=2.*HT*F(K)+A(K) 
   26 W(K)=3.*A(K)-B(K)+YZ(K) 
! 
      CALL DIFF(Z,W,F) 
! 
      DO 28 K=1,NN 
      B(K)=-.5*HT*F(K)-B(K)+2.*A(K) 
      W(K)=W(K)-B(K) 
      A(K)=ABS(5.*BCC*W(K)) 
      B(K)=ABS(B(K)) 
      IF (ABS(W(K)).LE.RZERO) GO TO 28 
      IF(B(K).GT.A(K)) GO TO 60 
   28 CONTINUE 
! 
!     REQUIRED ACCURACY OBTAINED FOR ALL COMPUTED FUNCTION VALUES. 
! 
      IF (ISWH.EQ.1) GO TO 50 
! 
!     TEST IF STEPLENGTH DOUBLING IS POSSIBLE. 
! 
   40 DO 42 K=1,NN 
      IF(B(K).GT. .03125*A(K)) GO TO 2 
   42 CONTINUE 
      S=S+S 
      GO TO 2 
! 
!     CALCULATION FINISHED.REPLACE INPUT FUNCTION VALUES WITH THE FUNC- 
!     TION VALUES COMPUTED FOR THE OUTPUT POINT XEND.REPLACE INPUT STEP- 
!     LENGTH H WITH NEW COMPUTED STEPLENGTH. 
! 
   50 H=HSV 
      X=Z 
      DO 52 K=1,NN 
   52 Y(K)=W(K) 
! 
      RETURN 
! 
!     REQUIRED ACCURACY NOT OBTAINED. 
! 
! 
   60 COF=.5*S 
      IF (ABS(COF).GE.ZMIN) GO TO 80 
      IF (ITEST.EQ.0) GO TO 84 
! 
!     JTEST=1,CONTINUE WITH CONSTANT STEPLENGTH EQUAL HMIN. 
! 
      S=ZMIN 
      IF (HSV.LT.0.) S=-S 
      IF (ISWH.EQ.1) GO TO 50 
      GO TO 2 
! 
!     DO CALCULATIONS RELATED TO HALVING OF STEPLENGTH. 
! 
   80 DO 82 K=1,NN 
   82 W(K)=YZ(K) 
      Z=Z-S 
      S=COF 
      ISWH=0 
      GO TO 2 
! 
!     JTEST=0 AND ABS(S).LT.HMIN.PRINT ERROR MESSAGE,SET OK=.FALSE. AND 
!     RETURN TO CALLING PROGRAM. 
!
   84 print 88 
      PRINT 89 , ITEST , S , ZMIN , Z 
      OK=.FALSE. 
      GO TO 50 
! 
   86 PRINT 90 
      print 91, N 
      STOP 
! 
   88 FORMAT(//,5X,'*** SUBROUTINE MERSON ERROR ***') 
   89 Format('  JTEST =',I2,12X,4HH = ,E12.5,2X,7HHMIN = ,E12.5,2X,4HX = ,E12.5) 
   90 FORMAT(//,5X,31H*** SUBROUTINE MERSON ERROR ***) 
   91 Format(4H  N=,I4,' GREATER THAN THE MAXIMUM NUMBER OF EQUATIONS ALLOWED') 
!
      END  
