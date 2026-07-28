"""Interactive Bi2Se3/Bi2Te3 Ewald-sphere geometry and intensity viewer.

Run from the repository root with::

    uv run --extra visualization python examples/ewald_sphere_viewer.py

The figure uses one Cu K-alpha1 beam. Drag any panel to rotate it, then release
to synchronize all six panels. Use the radio buttons or I/C keys to switch
between the continuous intensity field and the zero-mosaic cylinder sections.
"""

from __future__ import annotations

import argparse
import base64
import math
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

WAVELENGTH_A = 1.540592925
WAVE_NUMBER_AINV = 2.0 * math.pi / WAVELENGTH_A
GLOBAL_LOG_MIN = -10.500151371945972
GLOBAL_LOG_MAX = -1.5001513719459711
TEXTURE_SHAPE = (96, 192)

_TEXTURE_DATA_B85 = """c-
ri}XLln<mL~iYdwL|bO0G&}XyLv0009t|1VDlyyeGW(9s~%wBCA+l)souH%+Ai)ojvnmzivh*JQ9YbnwfXzym9)dO32LCb0cm<+_
-V?b00qZ@6&$xTy5q5;-gjCCxicf)wYTVk3Mf(0Dl(Wcj#@QAgIWH{2XmUibqrUmx5j<$`t-d+>bv;o0ek?{|xkRm$w-
q#_$iZL;u+)Po6$~_Uzdj+@QZah1-*54S8+y!zWJ$Vm$h>FTe=>Eb?#WwOJ`f67=E2*q=OI+t|htsSFmE#}^3rJPwOaA>y_-
*GGb&HUN8l6Z(`wXK|2Exhw`1`gCLM=@VFn5&jYSxxgQQHdl<11mpN0KY6x^C9?!Osy)+@)$VjU9rk0hQLmEmnPlA72HdAJQoz5h
t!-@K$V|RWr8k<7?GET$yY<Mlr`i#)$k@$iPY?k{@W;?kK%U<wWf)3;2+)VWMPLY(dq+-
xIG)bt3x#|(6$|<7hkChyK>&lW)|X+tfgho+ColwZ{h{3#ilvZG)A6w1d9<e#G6-
9+0093O{4wz3IKQ55vqB6d7z%*kU)v^eWcnlbNg`LSw@yz_+s#@j8xK0oS{a9e-
P%}RpB4cIzrMb)g{5$0TC+14&z5S<HuP<)Ud|;>+(&vDhqS#0;qQ0<82SnDUoUGjBJ?E~3;?A38#snoV|Is9g=+J(dv?}oHy{upu
g#>D@o4z%&4C1KL+~*A&22o5C)1j2-Vh{uqun_>>z+2Ng;dCG)`%Inji-
J51NcMqGuZzy+h$}Km!L1eK!HeHrO^>g<SPxN53A);F`rFFd{%>kOU8@@7&I6@hRjtMtiDJxn=h8iRb=cMm3$)TFe<quSbe78kI*
kbjwZ!~1XBV$UB@z{d$vG4S822$?P)aI-QIbx+p1(EE|Z*1Kt^DF96jiL0$Xl!MY5F^{5JgEX{%Am#RIlIDFeIy6lgyJf2{Mzz>i
>lHQQ!n7)UT2g1!KcA3xo|v*iYRAf7E&s?}<}-MhYhaeIB<E+u`3N*F&Z!1@sV`UZ?&dFV@)+UM}wy>=b?wo=T-
19pR)jo)|*t)CO{C($o}J}bnG1QP;0fkuEPQbURy49AlBTIcfS#fzJ(Zao*a>qXEMhVb_#Sns2U@I#YVq_>B2_3jmT{jyWbCu89t
G;P!(8r1uU_DsMZqDR60<#<DpVO)Zl5qR=!9Ybb|6<VY1BwIfN9SXtEhHN?^1-rckHRvD6C;;SJ$WK^`P-
hEeApyVuoYk`@HltP{W|J}N&%o|A5B?PRFXy!>5f%#Y^cl1V<7wi3SG?5fU0$AdYMG!##U*04F?cFlAQJJJB<$wqCYHqHi$nr86%
Rcja#faKrq(&XyzI3~ao4^WTCX<No;~fue>i~O*PR*kUoLBC5+FJ>BLI+q45<XPkX9<CY}9igXAv;lIGR{{Xm>fSCKaEICG%A#tJ
7sa)QV}iZ47}WKk!7erAh@_HX&sr1N5lyCpv!`{1>xrT7=ml7zyy>K!Ej4JX5MOJKS#Pu|dfrW4E`lRH@M&N#}B@kV7w^3G|LoDw
j(~+(s!CJSOv$hGVDO?J(=4OgwBdA*K&9cc<W^p#NfiLziGi0Aw3L1c3Hb5`!xc^4U}Z7PGy*jbo{7k$kP$tYm^ma<=>^n5i_IwS
2^;V&S0Y7%YLx<_iT}1_|udF#NvVfG9t#`(yAY(0{S4%}6j50G<OA8U)ROwf_7ZGz<pMP#%YKF#f=ylp#F~!1#0FV<iKR!3^I-
<~OuQKOJe$58sbX-EsIc;QxNQ&4@51z&DcufB^vbp?wKyV10r57EfXErLf9mlfmO%7J;>U986|mWs>q)6g(Is=-
vKTp#K5>v5gA_{uKJZpVv?%m=XYJ&<H=Yh*H=*E`x~KLg3@beD#6Vc4R!TIvtih83VVCW61U_4yX0Ncx1C4sQF|ZBES}g$l&tW6l
m!j!@nH<-!E><1sKc#V8~IV1_-
EG#lZvo7%WL(@We9Ng#S<>7IJ9>AQy%};|j%!Lw_QhiFphH5;OwP&v=f?42flsQZ^=buD<e5!$(8^`$3y4!L$IA_}55sYa~}HCcS
$cEUY|CttV6Obm|$8mPJ5t!g3K<qX$e_g|M7GZ?ag*MXYkt+8F+R_oKr{f&P0GA?67%8UUa`kQKx>hRU7uUL)n$Ap-
m)u{G7cetCOd3GK7-WBBpx{ZQrn_T_auWfg+~*e32cQjOkuryMe}A<O5-!P-C3{gdz)fu9m#Mu4dS=;P-
e#%kx!uTFDr1qo<BW!GPQ{_f+4m+hEw3coSde)-|!yXRH^E(Od0lER%ky?TCLiyd+&@Q-
yrI{e=+YKsLJ3;<9dNPpJpd%@yquUm^-
M0gm#Dp0@u@XN=$)7U<90zY#<c6#^mmk+n~fC|Qs7g^%9Ztt`h+@r4p{Ta>P@$R32|NCVPO#qYufc2lE_2nAvM%JgLVE}%T*q-
aXeDmtE>{U_*>kPauQIy{D<*PR@dpWz91fF1MT3;6WHs{k)))4$7^_lGcG5F&^|8Bg^N-
z{)asa>{Fi@apB(*nNZ&eZX2h)^e43|4+pi|k%^o2JD@PrM9>a0@^8#%}}4JJUfRnL0Wq-
R6*LF;}r_<x7i76>ps0EhxT#Ym1Lxnd@0;$ab#yu)a<(*gQK!cE}Ek$}E*I@RbQ50MPZGX*onT;y1Sc{-
&&6Wxyj|L^8CbOELZpszrWethzbEH}G-
PNSFzn+{k!_aIt1J*`9!xCC4uJ|X~)FnR>9pU?n^jZUB2EGIvE0<WGZ>NDv672%@^Ff{;^E6|fQJVz>*un1%(6Vi7qo~HJsN~M%r
MT2MWz_b9O(Nu2m$fE|2a1<t!OkhdmQVt%`{vm!?ebBmpIr=dP&<9{jfq*Mw9h#nazRrBK2dNC6s&plb#kfsI#z7kzz`p_G-
^Sv|GF!Y@OuCd*Jfuu}M`j%#TC=v+5uX*R`b>2HYVc<TSUdpW7`DYd2&Ph@BM|}5wWLq4uUc^<qpv>T+>0m>!x(Q}U7w~cTs%Q^6
iTIn2i&bS#CD$CyvFe_g#WuiTPDCf{ICtThEsU+jb<UBp%5hA+RfWHS4A5a53A3{1|k4-
q~zL)S8v|l)VvY`MH47A8+oq+w>GlfrtqV8KN|QM5#|lRB>o?U`0=Vh>9kV`>8KF?`t7^7*F`G_H-
#U^u@<l2zPqhM_^G;3rE^*esPIGlKTP1CRiEkZ|J`gu8-
QsA8f?L1Ht4gpZHXgQsb(M>f*^F3F7IA+Q${*GeK(K>Phh$+)qQbyS#k;qknNDER#FbhHiCcLc6%^^e_r?h?sv-
tKvST>1fIaZLDe`zVYi-z$8(Rf-RsNxiHd|Bc0V!zSd!|betF%^9&_<{mfjr>IW^Rc`{7><e_DV=_`e&+zrID4YIRC35r?DcL*-
UIZ558G|6l?Lt?7EJ9MaQpI3ic6(@LpZ>*M&pTY!JE`+xVlB?8PS&|>_XIKjTnZjz921ir~14cQeG><oS^MPUy`{U$yEN0ONAwtW
F^6BYkp`T6oQ;{S4H0UFnzZH_rsEGKPZ0-nfG=(Uo;?lplQPm^f%3JwuZ5ZjXFV$96h-
k6JjG<|2)=XbwbO@M&{p)5cX>(2&mH`KViYRC5|cmhXju^J^L6#OKK(Q46h2zbg~ynS`q2<_rFCfA?;9Q^AW1bw3W^6nygNXOH4{
&Y5K5#VO<;{=vyHtp9z?;K_??p}5idcwv!3jPPP_$SoocfVUMz=Q&g;Yaj;V*g!R-@vOQt>-
tLq=8D{+cM2|IjADd;3uhq<#sb;;}fWcWas92E2744tglV(zf<_X_$>IpSb!h)0AcvHbgiCq%ZNm=yV&hDqB=5mU_SH>2rO9_ZS=
ZDx0pzjxpVbe+Q!Fh3?}f@;c+m3KPUd-{x!S&Y!ReKUY}J-
B@iW^(%E@4svlo~`pb_#+B`oic_c&vRcZBkkEDdHc?;0^`t!N;XBI!=!(E5AR}6tA5_8A|0!ivA_bwp()EWDM3gN%#l|51tfk5Vn
MJxga+Mw6_uIK+){FCYf(2?kLG67E@i5&UUZZ)K#j8@S8>PgXrs@>DPLqvk!Bh%^N6udR1Kr{9a+WNCH{;~Dvt6BVr`T+muHWtrR
7z|1lkw}ps`md)EkOAo1|Iqo1rbqN&LLm}aN`pbc#ACrd-`YTd{=@K(94KJ%-
^5aRBAq)L^XQl)nl^fR`|><%W)1Lf4)C+g+4GmTr%^49#MF6WQMXRSqhdD)TUcL#rZ#|y3H&MR|K##BgCDjXj82a?U8`jrVha1Xb
o1fEZN<hR;;>`qUo4Kuu~lwAe7GqcvngUnrdCUPJx(KRLrmj;@b88nET72qMI3Oj?QoQ+)C%R((~?g?6?j^2|MaIfO*fwiO@JYON
d1X?ck|7k{`9ux5l|Jr(&=frP^%=45F4UD0mdirzk~fYss6~`yNTZoHLq`a&2FdUlLP#3e*N`z!^tC#4)E~wCGwn&*T4Sy2H=<bN
}X=AcXQng?cz7r2MRPj0p=PI_p_hTm!E<95cP?(yVuv9vu??|Lt&e9m#<%)XG{ztZUR4!$S`HjU%kG}nc0*bZ>f9Mxqf|jme3PN3
N&W|uRZ`Pn9mdH52?=<0oV?==Z$tV=ai7CGAB@NyM#;_I3W7&2Lf4Q2b07pqmm`gT(jLczXdh~VGB~A6({iT|8Ax8<6QL_;Rosu?
_FCsfi+cY6jRxB(7+~<X%eH&Y7kM0_$mJto=6oLtTv;BMkcWh!E`oNY}8U#0d5Q4!^S26>I7acz~3!9K2A*Ef%<IYXmZGQupilu^
?VYMM5Z!190rAif}ccTa5zjV8G4JaKeip&Asa$YgM-cn3WUA`e}V2lS?qp5JH1S)KTsbmLv8c<tZF`=OD7T_{B*uV%%wqlEOr7v7
TTL>T(N{thwu}KbS|H-
w)%WFH3K_RphXj4kppN|_s=Z;ZK^I<YBdW!6^%$B5Qt>1&JpliRZNurXR56JfJ4V6!*9}5zCyEA3hJoaX!yVWR0rJemwA6i-
e+j_#0p)dt5<gyc`LMF;t52yF;YEkmx5aQjQ^Id4VK!c)rgTzB;cXt)SADzdv#TE39(=WMVY{J2H^Lrz+d2fo4}8g2U~YPeRxs#N
&tSc++V+W`{ugj6cVTK6NS#w^_#ahb-$bp@Jqb)7axAQYX#-
FN&ItNSoiinqdBi8c3%j;%m*BX*JYOo;HT>1XK#P``BgipAWh*XDT3`+KmYRfEUu#h{32Hwh*R^)hWH=OJ;P54@cSj`e~)s1@ihv
5NdMV~@kZ~g7BMjK0KYccdG+z9m(75D2ERPeeEHMISDmO9;Kwr!k=j|W5kF+3TM!Fx08k}ByFfy7J`at5u>RnM2mWZ(Yv2=cI2@k
5<0)Odc+tx{#54LMcI106UR;zsJ7hfcE|G8WMx*`%A%3|2JVf!qO$jjX0&LC&-
j`_3|Iy(E9362C$*xk&Am9LgBHtXZbvw0K|B`6z{(`_W$7-
E!EpCQPJMc1rAy)267`UzB`oHiB=9BLq=llL)>wgPcT8Ln~!eX&FJVWD2mCD(mNkAN$4~PLl6qth9aw+A}F#7L8UqOr1*60MV<Oc
GKd))tiG57Xy+V`-9A+n_kg@{Q2pr}emqEss<-D+Cj{NLyw0BCA=s$45297-yHPhg4^3Mre2*_!b|pk2U!vC;)RN&vx}>-
VAW0ouWK4&4F2!yqC<_!(MXuGwi7Lq-k`IX-
S62S^;p7%H|p&74omfbf$=28Tc3KHPzJuQ4CQSoq8+00KRHV`=!q_~rHN?fcMoLX;gx@|AKXcpxNVadeF*Q)|`pLBkAwLoi=&)iN
Fp9fu_f4}zI;B_BDK5odiUs1Xoft`t2p6}>Ur{WGp}rrw`#o^=};ua*S?6Cd~!ndFI8Eu<61{0{`WP;EU)W)l7bF(f8S>&-
N}XU)7{&zyFmqXf`Q48a$xg#Ubj9)UF#{_|{o3(wgP7TV_*-
D<+FphAiv)S7Khx6kV^?a;A({8;*q$>H_6oi?*p2&oxWVNX=M7w7FlaG!(UT7NbZ0JJ20Z1wP~Md3@&wkUkL=D?qCUp~L;mO>^T9
)ltBbgp=zTFJ*8Y9<E8k724Ev3#Xkh`V$=A_jx!nL?%R)$_}C-
hZHx^C^SC?FHd`DA9ZKqTePW_~r!g0oU_0EKg(c1Va&k|N5*FF^dS>7>e8$hx+!SQ;8kRNYLuIxkZv4$10r*XpG`EIR&##5Sb&Dv
ul7q5(;`O8Xor9*d^Zh0E~rCez{rz;+NCt$6{Cq0hI97>62}a&Kb>>%7t9H**UGHy?b2THioWtr5oqh*S&hmxl7%G`hSbM>rB;q*
VpHbv`bCLY~#3l-gND>(=6u-<x(!{)N!_-
3@%Zp0uWK7SC@iEiC~=$KYaX*C^ZMO)mE#PPbQM_kaJ&1+TPlx$d4n1W~b9Age@{MjGrvCgbR&Mr&)*`%PG)1B;md@6o<ab*IKP=
HfWX-pG}6Ik3|oEJvVw7HTHHYg81t>5ySw$<iM9K)@r5niS1xdEn(upf{A7ATKut8D(X8@GBB|J7Q;{;`J$;*%x~Fcfr#iWJX50H
JFuOkOSM`t={t~2<DUqfpN!sKE&O>lf_pNA8`7UGrqb*Q$Kw&7Ma`$+F@0w<hQe3sjYhqKPu@mC=r^{>d<FPP$)^n8!c+Kai!Tz7
hdpK`bL+`402D3ue<pf<ar8M_1pQ1J#^a}3RN?NP$+WK)Gaws&6zji*C6GvjVII%)PyM!uV~ExJCez-okh=ABEOBH?fXVEIua>u&
42Eg=qdb^R9EnOL<ADt@nd`E?KA8Z6@)NK<@MJ2Lgxf?(oBDQEg2mY*b2CULQ%FX6NN@^aKQC=EQEH;0{4&|#=;y~XX=`&*kN&W{
ElffhbbmjOu%FNP?9<vHUviW;I7r%@OkA5u{aS*2G70S4i6pRL_aiBsp9%jcZ*WfP-
8T>JnZNenn|Voi!`??SP=RtI$vSJRxAp!vNc8+rfKlq~gb3fFrasQi9-
PEJ4*rLU98}1B!;xrA4qqq~3iuo*jf6+ZT${~e!^0m5ps(PMa9jOLynY7oc=GIc_UTmC>7wM#VG8O*2I>$!0KdLPpm8O;I-
}WYce-3oyVY#e?MitJ5)KLcS)Xi37!Z!c;7O-Hl5lB+t@VBm$zT|Mn3UnW<;lNODZt-NVjt^z=-
|GIr}KBTCaXITNu;xRsAbcMNYG<9@2hu&90r9r$Wz!F<TVf}432O|y>GUAg5w_r+*Xrzhfl|E!pHcj9N?)W=KJJde?NhJJa>5upw
}JRec?p5SgC_N(srv+D;KiKXvpujTMqWMDuqlUg4Bu+DT+uUQ>e822Nt{AABrZkg>tRY0v|Q&m0~s#_Sp}0K!Pnuf~TX&@MtEF{Q
G<5zW*?eeK2L8KHJzPG6Ztnfz2C=X9|^i>$GzQTDJp)sFd>AR3aKa3HZEjmlHIX+v^LQgrkX6Head$=$$V7Md!3tuM{%zkk@vglM
5Kc?Tu$hL?V(+KAjD}8im3n<G?UoA6akUq%|r>q|hB$U6Ar*puTLjJ7>M~3((GckQk?}W}{xKR?6j45wuddT&dRTjV2^%ckInhy9
te122vrH^+2Z(aj4KvzOjxZ#6KBk_D}IEtV&feMl|pkS?a+$4{iE5GJ^*a2aoI?p#Aw$wcZ*?a0#^w1mF32uh)aS8Tu3YH{|F33o
!aty;{nH8Q`%WfiyxMgA5%Y`YBbzongpNu#^Jh(JZmU<Hw_23Q6_?s~?Lmk*oJiM>dx)7>Oq{xnj9mL%J9A??8}&ETg}lgYiKIs8
%f(bD3m367;!jN2WcsT*7BTD=_TQS|1S_AfyGeWD9fK=+!a41K6@zN4#xh8j~xK>?k#Q!=c6MaC?Dn$CK%7t^l%<YmlZin#~qyZK
T0}Z#El{R@FeJb0L>aC*xp_c-
;=G<<OwlD0d_RE|W$kAQv~A$mX{8Y?2D#$!cwqG~n49vc+!puYo8u1`FD=#8SCZrP1#j56nl$HfTBbczyoh$w??2i9}<uSUet23>
x%z=$B|D5)Pf51pPj*2U`1W$4BM^<Gx;_Qp%-bXfI<iXcQ8ONWkN8Sj-
?xY?O$*$OxP<9mmpk`<ZDNEDnbUH(V4djm}`Q*c`z9AQXuuQrV7Np;Ydw)M|}JtJCTA`n|oq{mBOX0{yPjX*C+PTD7ZGD&#vdsYD
FcSRR+dW-%Fb8kIsO{TFXf{#&+BWL8-+J*Me3KE-
~XZgVhy3a`yPSx``;92Ms`lvy|h9}V_>35C!EnWm5#Bc~Y7_RZ0R2QwazCx8b1Ib2l`xAaJqF$u=d&*Huh_0+5!V-
gJE504^9j1nHi4<L&ZmoE@WcNDuSupsHcvZIEUNtpyJlXNNxM0dc8<A`ZwfPW19EbjZ#T}=uxz8gOsZVa2-
@QH*ck}A}?J)`N+d~{^7fGxq}3x=S*AP$!QXe1Q$dtCNovuRJGl!<suDs<cdd+qv&K4`AlvpNYDiX5|+(38=YyoDuDS$uFNIfAX@
cq*OAX0sWnW%ET~ZJwTXIvv2$0okSnVDX>$T-L*VwOquZLx-
I)YvBos<@pdD_fq<+Nf}UVIV5`@L!|ShD)>+W414G>0Bp>@5egZmy^E{oH@CMpH_xxHAgiqlxKfI_RP4m-
I5MbZd<F@G@U4yDquI^(rNH`r_5?7Zjv6yYAzPY27bx^c&OkJk2ZT6X2=DcC80;A&LF=@8ehK|Mg#YIG^XunVmlx-Na;8?!r=kJp
(Vjv;Cm;k0W4(`hbXk!CYX$PFkpO@_fhJHItlmhfP;H!cVcQd!n&)SoR;`rJ0pl7R&mawgOw%*iknZ&m(wvCbYETMjc=!To1pm>h
hsSx$R|}7jV@DZS0O%+wRGIAlXc`g#*qiX#r2-u#Q_-
N;<?{H0;Rs~RB_R7W51HAJPC@T9s)cmaZ#Sug^hNN$|Ly1Z@22^?MxHBV3T$9VY>CEX^Pa>Yh}ldk9)Zkam*e<wUn7@*lkcutqlL
`W!(+QU5Qbk$Cn5c}nKTkM3DZ~pCsR(KB_5PnCkmPn-
;L8>4LlIz0>E@)fCYss+0~oO7OT}_HtE$0sgTEJK&w0+Y+wWekqC@#j!?F%H<~TS@JqmJ0y}=6&4K^@9R99vR-
*6vW|9AAVgUN*8a#BYVRJZaCXE7u2{1^52`%~!i^UU%FOl*6a~3N88TK($?DO1j-%Ps>Cw+&HCi{Qj4BQ-
N;aJ@I=aThxH2fpw`EHWH>zfBleV~gN$R-
_dk4N1<#19)jTg$=2VI6%7Xl=OmPvW2FgTAN#c1nOT2jWP3Mur=L9%u^G`s%aYSE2rI69Q~Z2{6KsehKhkMgFk|3tj&|4)8x4g5M
aoI`I@Hm(K^b!C)3(aU?p2&*w5hRs&{20z?Faf@XmK2Xy+hm98*Q>BnYnj)vSho7+&Y0)5^ZqNfVg2R5hU*dXVSa4>uvi6b{0JDj
!ywU9cL03npZA~_600fzX$o4N&ifDV4*)@bhC(Rcle#}R(e`x_Vve`i;v5HSdtZCG<46PhM9dE@y~F%x!Zc|=%ih&-
($oGF&_aj!{ABjDil_pNOVfgw_;c6ay`%*KF32aWy>75xggUk~QpJoeMBH}E{I)gO)oodyXYRswp0r%FuzOuc*FYnLL&J8-
BHj=FOkDYbj&-Fn7vl2Gx;IR;CW7@WaK*l*SH@EdC*5~BhBZ$}(n4^|~m`@{6j>4TvLP2hhDsB<`aPpZ;t)pEf@z-
a>DQzYh4p>=+JbA1Ma9cXw6Kb{6cBF>=S&Rd0$SwevYz>>s=!CbA?s-
!%64j|EaI*xyqLG9}mFHo1Wr}aaB@M24>+Pi*!(M&scX+TkMfTg0`d-
3M=%ga_SU=)zx{wE2Hfn4kI<?A;udS$?D0e;2OcAe?w#q;Z4HD(dx*9V!9j|S@hElL!@q8se5P{`B19@n7n$EdFWew^?qTJ1f*xo
oAKa8MhFKRk@LZr}g$r`H#i&>^fUSXikJLzRoyKmGCjZ7Y7rhlBlaRHZZBy1aSbt45E6qfE|6W9oC?7=lR&<_*9A|27D{0-
;Nl=#hws;AiiLYS(W*ynETnIOH^>qoHA;$sL)_%Xc5%T-QSTYy>}1bQG<CASMumwY|RfY?4K4E{EDzE3?1N=T1XH-
$)8as@6Sg7lH;Z68}k{NRJcs-ql4r@6)n~I0z@~g&?xDzI^-Qs#i}OODO~-{*!A67TRasTFRjyZO-
6d8bmXUK|z&ZQRvJd^bjXF2<7XIO46mG!;tWJB29YWi>Ff|yP7>T<KTf&+e4{z+;<?Q5%I{&bd@VvY1H$f0|9OW1^*KG_l^CT#lJ
<~ISRxQVW*x)rGTJv5J692OEr3(f=?Z@LV*BOzCx$hNZAx3@*<f+<>{T_L@aQ$L*AOeKOKZM8-w-
bJz}p=La+MxH!);^+F;s~^CfDXN<;&GDk6!>WHD%D;05p7JFp`Ql8j7au$WX5$X+1OL@J$H!k6!v3~B+)cRIra@$E$SuSj8s68iS
_bm-eKbQL&$C=51Nq_GDh0qZV<2z5VSZ8Yg5bm9>HKma0LqB9xQd}siOj9qIW60~bXTsDJ($82wItgo#ag!tvk?0but)B45_j>Oj
;yL^dqtpu&&WFkd$7)++ZHU+J}%MJv93=4%VoJs}{MHC_#+Br(Ka>D02*72b|Vh;Xg6nw)FL=*{@hJM0%vsj|Q94j|!)poa;b;&6
t)_$mb+5ti4gem+45SrdOEr<45B#PXXZFbw$TB9600+HTZ8w-PQ(Kz_Nm`fk{<s9a>Z->--
!_aLURTpjFzJ7TQSrJK_ghCNJ@@F?UXL+ZD*f-O*`i31*;zS-
f#1x9emV_*a^Ovu0+ff}AhaSZJ%>ww#V{aESzaeS+5PqgPclF`dKfY+>;%+UAOw)v0FW$bsEILF8{$cn4v|Txhm#^QxXoWR2GE3`
@=Nd2m`0IzOoSBK>M<LP~i-
BG#gcvn=7>)T2$s338GY+zs?|=DtTk#$4awufRezN!G<NNEPl}DJuPvBXL*Y7{R=_U6WWC~~Z&{w(r_{;mt>;Yqde~?GMn8j{k05
l4B;Oqn#&^|!(m%JOSKfk*@jT`t3DuqJVMmu-!U!SKBn0V;yGCZb0XBa&5AbtM&{aq)jrBf(W2Hy}ry}f&05AKp-
{U2LCr_|>j@aGGF5<3qkj<fcI`ARWr7BXlwDuoK^f9LY7<WrDm;!ns5U+L_!1L;5X9*xEjnxn-
^KDf_<)5zx#xqQ8d194gO`~n5)^Cn<$M5f~4*sKw7`8*b#N@XbQkyOmBVT`HIK!F$<cPtgLD;QKNoyFsG1se15fr3c{H|{7o41*+
|Kg@@Z%0G#gyzpR7?#MPA*289VcP%c5Q4Ej?c6DkI4K^Mo@k6$eNUhry04XI#hs(0dWwYTxq3t<2BnwkWrZX`9eo6Ak!~_@+Kfrm
7WGYK{Qf@c14hfw~q4A_L9+iNbIHn^z0F@_`@@Nz)UE;_#+vO7-i%N#`B++OfzPWD-%=`hEpF9EvfIy~-J&l(i-d$x)Y${}bx?E-
%1;)R%HJZR!97Se!xgh(K$~I-M-hFu4@QSIB)dw!Z7VsGTVPy(U^a1$80^TBc00<=7Zv5)!fB5UWrjt+OnnKxZ#LUCb^go_wj%2e
T6PLz!HsAgAAAY`y@6t#F+?4gRn)<BB{`u8H;y2iDi^Y?us>Jm#|Ma)_Ef=3Awx?_LtV>Fm!B3F7vh`ZpE~fEat@nTXr(doUDjEr
}6d>xeiu$jZKQ%{zW{BSq1tL-;{`Tvif4V6gvS}h)s#?o9Bq-
~T#F430Q#KKeeOS2p>F3vNzXX}Rn^VUBwEC}}!8S*MzMeyz)>j}hb3fKN@0L6=I+eZWOC$pZHeq}MAk&v^2qY7}JvNmt^OU;hjo3
buJflAI)PFSw*7*vAG69APM5J(Zu9JXehe@N-1UidlPYBMqllB92HWuz#EII-
74s*v6IC1GXaFivYJ_~5b?vnz!0{$@*sLwq^AW;~6iI_vDQK>YRRIQRQp+m~F{X?WMB`UR)1zw_a#1cM(0?*#T;=j83e<FD-
+5|ubfB?b<AY2eKj@}!Od2}pzVIT6}!rKH(=ZVF=dJgg;jY@`-{I@n|&c5?9P*+LdTqXN-I04}NQ+N-fumpS-WY;jYk>=H9Bdi81
V}AgU?kCaI;l|}vGooce@3Hs-76r-i-
|X|}PpHo)XV5OhKRE%0GZ>4fNDrJ2gNR1uJ8G|f{`szI<ILda*s6Cw|NN@v;8STLgTr|sg*lG;-M_$i`&{_ZCjeMN2_jpvRE!?-
Xkvf&r@#H}!>Lz5p2ANScuzn4?QcJI{bCyLFj_1nZ6X3#JSWuu-vdAT1Rk3JIEpUPxHzu{6m+4d{qE0ye%o~M$priue#pw;yP9wR
{O5OVkC3hiRL?IO5gi3LU;RI6`B@SF#14R^3#F!H=jOT<QPVl*{MFmHS9vpgrvKT4M;y93(z?FsBu!Ew9s3`JA8i7I9blWIw>fQ*
dhe|4lT#UB{$96ZdyE<VXY9q=*I@oKsB&NVtXGdf-|0Eqb6h81|NGz{-
+^Hhh^+CZvN3Ngp9`5d6uQz^Y`2SE1>!y&@`nQlqQVOv`IK}D#}vxtp>MJ&uZBD{eugB1pJ@Fo-hd|;py3%{i+7x?HVQ%Wu|>xtl
jv%HzFtip3lYP9#2*f=|3c`^dfu<5lgK=s<=7l7G@v(mTVt*VH0$SH>;Jv+<COkl`?M0$aamL%k;KrQl$v#zzX(6E|Kef(qI$D*q
GOPVL@JA`3sp|rMZXd^bUmzkc=>etftdwpoxC4~RKI$nA;aEh#$L2>e%^@gBM0E2{RWP|`^cj`2JC?*Yfh@rB#7>l$MAo@*!OTh`
)wusl>KP!tW)wUVg3yw$CAHz_4=yh5D=#L&j|uY>FV{Xi@b#c^H&g+{!-_x7Tu?;&n`c6_y_LCpYY`>_zAn-9AwoypaTn-qylI8`
MdYGHIHb9KS$)L-M)YKyzCUfzG;Hg0i6_b-d(~n{Hqf{EOJ47jp~J7CxQuE&w7+3aM48+yGoa@?k);8-
VFZ|&sMm&dv#fIiNReNmZbD#>tH$)(eW?wKCD>(;;Uu&DSOZ~JT0AQDKLJit8jLGQIG7=aZ~&uIQm|sesO(Pa7khO6zxgr6q@*Zl
=WE->lK_=zo~r32tP>!^?xnv-
6ext3WC6rsI?mzr>swaJLIpxU`aA(rqQk?ECK?!*&y$Fvo)yyHKc{~2@9NubI>pHfzBEb+k%5YJmxnEL97NA&(!<Ug<SYZ)TeJ5@
_%k)iK3%$u8{WWnRxh8TWIpf;(-If_RRLXdI6ZfM=AgE_X~W`v-mgg;KImgGN|~_48ZdaBc(<?=hM(;_-|<%U#{LLMGm2<fhF-
(29xo?sOIB0R>Z$z{hfv7Yf$iG2+UnuD4961$jDfN*p_N^JC(4JHH{z3GKMRiZX;zA6R>2N<s_L5+jf}*%;tRjUw?l7kV*V&Sf)g
&u_bEIOpKZNc)Y+IEjR1=fS!RF#<(E@U>N#9zTPZH%>q20Z;nB;zLvCUloBR(ZCLgZ?dEjF%a<irFsS&~2ojUa9|)Ja*H@jqTS3C
ocI{zkERK~7Y(G3639sMW!ZMV{Cy8X(zDvWA6z+WI>bhGB2mCIRgs`^g=J)gRpN!*wx-
B?7$&`xOQs?rrQ}Ao)SOS0F9ZjXe4h_tGGR&Rb#4t6Ea4Hpb@AC;*y4GI+pOmu2Qs(4Pxczhv{`r@Wi_m{D?-
F>j0^8%KM5QxTZMQ1<TIcMv6h7eNsVX;IrYY^(rEHDi-
=gfg(s2DIw~C7AAB0Qovra8vX|<~<r;_+=UgX0cR=7P~bc6a8dIh<M1N@Fex!EYCi_K1}6fp~lTvMdld;Yvv4IA0pWBd=>Y-
1RC5HWFyLUW|l>NJb#QlnW;I2PdlO!ebXqBvmv;TycMbUGT0=gV*rEj-
s0E}vapb;}_mI9UzqZ$hUuW2oG{x;iU|O<V$8I;)(I2czk9%xmCpKV2L-^VRAEG)oHjP2hj>498O+93Ptu$DV-CwkITFsJo7Msa`
L}>`Kb!Fw<y#eUqZJ$BOlODel;%VTi&#yD#9i8ljI5)I9v!Smep{0DS$K`DZ4JaBdP9Y&L_z-%-kibllb^hM_!k2Li4`1-
(zo@vKj^L025Q0s;3SSedrT0=4PbYB8$BJO*)VI`RsodgW~B(t`4jUoMKm`hKDUByd5%rCBi;q*})ofhm;9giOLF5|`S~v07grKG
@#e#PhVaKr9hIaX@3XF%`MDICSjGFIO$V`PGsVqCXDkLnf2fAQ86uSw$16q3f`#3l?&luC$z_i<MF~Y*(|k*Pf1-ubf-F_4DxO45
C<n95~=+=^n<j(k2u1`XXYODr+cHg78NiIxcRa`st$TvtKPO;5sJ+aXJPOIeo2d5s|27GInzvoulRHOr{=?Vv8a%I8WlqMA&_}E9
P_PL}(1qG-zS@+^@d8PXXO&?Rm0>W65-
fHiy%Gw5Q;bw$^78i)Yh2`vPoJgev2a_4vqau{#~ML!FF;8&sj1rC@rndiCO!D;Pi7q)PVe!9=!D$R$HAlbnegtoCbY*`-
rz1nS3&R0pn5GFK>M6G8i)gu00c@Nk|0%jti{W1#1RK>vW44o@~96>vp!wRWd_S}!KNhYB{CNCdZWo9i>`Gv58{AZUU}B9YmOLvN
y3KkatfwOrI?kWe<BpiF^z3`WyZ%#A=tiQop*Z|vP8f3no<o}Zt!tJ#plpwZ~`I>;R6(Fofc>!|hXp<RGL<3R>DcvE9=gtFE4+4*
_5SxWkkcG-Xg4y}0lJqe7^>5C=;(4VXmdFmryJYQ>dI;YJ_HtGkdn&D8;>oDsT0tOK?%*`2fJch^+DD-BB7k<;{@kg_j=4q$Xs^#
OpBQ=k>{^Z9+)!*;KUo@`*e82m*X=2^6F9yn|w_A-$K9xvj3#D?om`g<iE{j1e6R@a6I5!hXF4@LWI1-
h?;tE7lxnj9oDrAz0RKEXxF6KMdiD}!<9?#}q+%x-
SmceKu0|qf1*06N3_Q(@X<;&F?sDPeM=KzgJyVWOT@q+|YjY7<0F<BfQUmy_4R7R@@;Y(?@+U;holuM^U_4Qh{oKJ;4M_Mr*yY_g
B#cMoy>cN8*gg~b-
`ff6R<nhx@B3r6GuzOD;v3NWd35DYzeGt&HbRm7J10tGO2ol14aA3Ar?V!1Q;Uv`G&^ujd%7O&KSU7}ye&V$sXr*l8W}ijv;XMAO
#i?s^62NB2qoBR(={knY63W$jgX!Sl;LzgmM^Xg<4(6c&yg1b|q#@BTOw5tVWYVc*DxHVM2vCdw)|q@N;&WIIp|4E_y;?40kumE}
A5Z3g&C4BJDgSPUfoU@T3%LHV1QM0TVDn{a<FPv!OXUDL4$K_VYybgr*=()=m{A(wUyJ!%zECVzDj<tF2~Cq@gG$P01I`Tsb_=Qh
{^R1@&2Lt1i#eFaGpGg`%zKAcmoFHOB~qDeE}zfmvQVF=;SB6Lz+bBrbLj-6YJfk)>#|!+IwjzZ>Zk6^CT}jvon4-
MIzNYYItv&k(I?W_0*OMcHy&DS4wu{G@j}h#4+74jG&Bm?Y&sQ>o*>jrI+a2u=CkQ!JZ5wK*>u+U!-b?v-
+p@!0x47~CbU%YXYjTPh-uh7fk*<=p{PJE{{dijaX6h$huwB;J}~H1atVTbt6#W#jBj9Z&ii-wZSzv$AI>lmj1dIE<;b`CVR3Y1@
e~S0fM1Qrf#d3Dpigimpb;g^aty418HJ8umK$OoWzaZj(1cAr$xty%QSopI)x-CzwS|-
r(@aehOdWH$N9IwOO#c7Oq0;!_H?&nrI(}n&_}`~J{cqSlnKtOZChB`h^6WRZ`+)yVyi2PgUxE8`6)`>wd(`6v6086kopNcuBy;d
DUxgR0+)ozrMm<}*7jqP<JkRE_NKJzObndHZ{4>}mX-bzgG%2P9n1R1w*PFkUqSIbY`l8lmfp0EoC}N;_>?Yygo5O2HggLZbYm?B
2D7WUcSvkf<7?)rnkJp?X8(ju8+O9Eg>?HJIzQSDUjd3}~Wtbu&n}NTwt$|LV`FNUkYy^G+`G{`=txZTVF2q2BX;$bdBC<zl#@zx
L=CY2BQJzfoIcCTef}3dAA*Tq66ysDP6GUW>??<CHE5bDCSf6#Qe<$6CyFCRRNyQ#8*CFI}NX|8GN(khm?Ew|s`cQ!R{IM&W%#YA
H$DR$yK(RzRM<5o9_$(ybXowx(rz}FK?4U*mHSdV99OE*8j3gFcBo+%ebRu>@Svt+6_TwBHxXIHDst6MTOz;(f)N8JM@5teHyR0U
Sm<fj@A^i?=u*p;gSFF(N8BB*}=-8{-5x{;DVxLD476C_OiZv#y%k6d??a8@h_`ZH^oK0%psd(-
=Y+8axXq;*AO%{eM&>jck*<vxD3cE~l795j@^gM~m67B9EIsIYayUJ$M@e|kap;4z0vZ#m%NUbWC+~f+U^2K5{9yrzt$lHTkGgOz
v!+Y}Y&5*v0@NaBvVd*=EzGS)C>9lM4h(jl!AX)hkdJ0RTv-
)G1VilpXJL@zn`D`W?^;>lk=3o%;1d2fCh~#VSPN!K;`VM#K*sYC?^;tHr2?FRj=Xg{R9zH}P?CR59QKbj|boI1%(QB9Eu00q((0
Vdms5uU#%I)4IAhNrG{&#)e?e@;kTIE#WSR<g5k(mJD-
*d&G_bz&;)wKUWN*xjpPx!M?dHB9x$zw7jzzF{~k+1ipYQ5{*=NHXfz$l`?<O4(!U1Id5s-
4T5m#^+#+&sU!yu3U=>s`ONdwFx&siu5J2_0b|AW%fcK(2Z5{Pwz6OL_Es;&vbZEF0PCB;x1-j1NE`KaQ>SWLnpEZ(iT_%3-
sF3Kw-GQMfu!whjIK_RXso*S&TF&~R7VmoHvHzuaB7vmPCnf(SsBn8W4X?Q7`gR>q@c<NCP_DBX|FBe+-
uH12a`0N|TwEKOlg)GuDVx_jO!1PpvK()|>^Ay_)Ud;jsno10#vkT~&qJl=4&-
n)78;p6+e^HR{j@8c))4S_=E`Q57*7xjc)LBk@q-
p}T^LzlS{XC#;*o}W}80&m}!ZT2oNI_0QUPAB&9gZ@9gdHweFO}7+v8dW<|seIoNEp_4FPW$~2zfG50qvg)UWv`j_?ehpz>VpP<i
5qm803!tgGngVf3gzo9Xz_6CxFkgV$qd<nFHvfq^}6+}-
?YPF&}cL|M`rS8>)qa2vy|{1$QWc8KY_&6xuGSjRnLcxWE41lVxpeJBlJQB3!S*%eK*%-
KBYk0IHt;alFDXc9;28_K)6FFygg5{RI3&<F~3F4r&FmE3WZ9g^VJr3MXi;Ro;_Y)fe2Kw(G$yLQzupx6Vd*W`pjcOT<U_JL)$oV
`40wQ8_(P^SRHn=M$m^3;TISK*?PNCjM`LiLO+E}CR5;aeU&X*Y_#jyfKkv_AV>g##%y<34LeNyHd^<iz@KvsFe||v(!=2ZfRIiS
lP?nSK(ZhLpG0A6obf`Xl=2ybbQnC+C=|NT;7gS%g}7701|;XmCln@6B;qqk2$?Th_ur!m#tH&V4gjc1Nf#>BN)dwu!zWW{0-
Y@o3c8F^sP`e@FeNw;fXa{>UBOVmrW4R8z?>rzNDLA5t&mQD6E!wB#y78NCdcnr;sBkL;QQH%FBABIo1d|J>^t#W)oc<BpGsp()c
Xd#0#+XKfQ%gxfUnRS_SF(L!qQG8vDH@piSKxqfn*A7j1loYp6>otufe`ud}}vWu?D`TAWDy;g-
Ri2+o8hn>3qE_91A(LJo*fNI#25e#lkK<G_8RE)E!$4`ZjtjC1M8Zvo@#u=aQJtmEb${k~@<V07F)~GxgI()~luiC8u%s!`0s9Sv
jC((<bmk&)C{P`RuY+4exVlK(**<Z?<t-
&$yLj466F0cRv~jD~bTK1AvB~D6=OkjcUrNpg|C+bg8@Y{KFsLT@`FX(8*)?sdS;OaP{twAD&m-Qh0D^3TLX?s3h$&A`1R_-
M<2V?EC?k*n$CmJl7D&6m#JtF@%UpV<-
Zx*T4Stm%Ex<GJ{{@uHF6e*I!?^0tyBVye>Wp=Zcwtfs60sUt5HKIf?8yi?7|$@Z;E~aIs!ZJLPncRn3sPt1mwO`qNFxj?(}3(#=
o5etc1NOF^XtDqZeOSL?;FiH#e_{{*!f*lKYEOXCIx#F?Z1H^*zIt-M#oq|xbg2G5i@y}i3>MfTuTW_<mmvG*dao4ec7go($1-
lZ{B-hAt{7B^4!|CIWpmD2iVwZMm2{69|P$1(R$3bjhgvBP9QjmeTbGtEvTZIi+(GQ<xnm(-
SSbeb8boQ1r+<49F%g_C_IZWjNexjeFq;~&167j`nCK!^zn81qRIb1)tY94YubE}O+<@%MbGY!Wgu=?H#E0%ZS%v<os#li8GSkIw
@9`aHhkC=iPW%@RQ1wKZmZ4)HIkWVT3v*#Vfv4`<a;Wyg_RF6KD49`5qlEVf8@?67DBu!au}2w=lOvqPY@IF5B9=q>*4q4n4i%jF
`+GAfd(H;sR8_pcT^^X)Y6?L&0@lwE(hb5_q)8>P@b5P&C=ia4Ma3axLxAqE@`_<}glGai7yA1XDfx%yeB?BAu_4?ija^sE4A3WV
DKTEHsM1<X+G6D5ZY8Ve{mG5`bn6Ldc`Jy`H_j`E}pSf;vwsX)7G|IfgmS3eFVY-~aQ!)ch^H-GuZKfTBqd58e)F##xJ@R5fco-
z00PyhIrH(faKbGH9cX_UWR#$i2Y0H#VlPU9y5!kj|;{XhQCzupxO_*}Sg2IX|9Blw_T4+nY7KPcY)^?&~3`*s1)>5!)IPfcHR7V
8I#V&CQkzKvq$k@W{o3#SP57FY4bFMs)Xnbh%lJRYA9{c|V)l|miD??2=5b;-
+*fBEG_(PhyK5VrKKrTAB<qPSRr#_&Juum5nx5h{zn>#tqio)_#=q*$CpB;bS9l#XzUfVtbRNFqT>%1P~o^V_SMf0qwDiuisllD_
$~2>*kX@IP1;KL&@CqvEKXu~hh24#5{Ij5e!LE`kI=79lFK^g!@Ma--E|REQw~<j3Ju%&FqQ!Eb#QkxBdyKP3zlRe@$FFpPg#PK!
)~mPOgFMj_%sJ*@T?TCKc$SIFZ)0|09fWLbbfCgH9--)a@SYG`zLB86sG1}(HSgv4cp|H&Nu-
`pGjH>=?%QiKMF>qyCiCMRFG<Ep><<>$MaLni1C0IlEs0-
2+B_wz5W>aHCjG|f3Y<&n!_5CS^Zx%j`IgMV50>NnrauQ}VtKU{!_Y-
6}kf~<IGb>WLN$ru0dumAeDo46Vbzz{#^es%ojZ~ywQ|L`KIfs~8GfldykLfFVA4woNP{HW@`I(Pn9;4nIVJW1>>c6;@RjssMDFa
P#m{_p?&mpA!+(G-5se*O)3^ftc-CI?3ssrR}?x0r<gPsT6vRn9M4@qG@P!xw4OcmMQn|Mt&siU;B;{NjV+n}7bdfBUDqv{uB2-
rbM4F3&4I*<AcTd{+EFOm1K!{6v8bkm&_fY@j4^@99tf_|JcRRX&=*e^h?;*MI)UpH97UpjvEI08sYX1jKRtPglafJbwE=_~~kQJ
Q*|zz!EPI?<dYb{QB{E&Lo_gzd}>)`Nv;BoG13h0<bv<Ou=N_t)~Cy;9nczC$r@HMzw$i@32Cpt9bF^x)D_MH-
P@`3(XBxuyOt3qUcfz;oX@fP#gE<Fi#e``Sb6pKVt?EU8q#a#0u38kImr<wf;i0Sv(m|-
~RpuPvDbcvsv(Kg<KAsx1&;sWh$kRP9pqA;{S09Kc2u)JHrw0vB&S+6R<e~L!{Pi<(zV8iuSv|KL8xQ+?i{2YY~Hh!xrp0{hniQB
<xf(2J8O{=Fe}%KV`rTSXDdSYBF0)xpz4np*h`se%%b~pkvCw0EGpB4mP@Q^ZI!=Z5DDkyY5smo2+&_1E$vhH2e%x=KR&`t7^4b3
hi;(QcvUV<A<A)RS3+tG5eXx5?V_)A3okSJW@7yFH~w)t5>gIoo7sp)$xCl{jeB6%Tl<0_x`p~uI2+;4qI{3dH1JZUpHK0xck8Y3
_d+V10Z%aUjO>1cbyXjo1+cntL4V+`*+s`3+q1t|Fa=}=3y4}e>oOEIg&70@?iV*AAh<n+XT#^2LkavFa@^q?N5Jv-
44oGOv%wnJXQw%pFLy_@jqJ`|0?$LT>MYQ@zeHVEl8^~_CvjdNoVeOtJkkyoW}P!1NfwY0Gz$}>5Es_Rqqb;o<x6W&p<P!72BhY<
A3t$$FG$SU?T$*T0hB3U-
7h44e#?=&}zw$*mI}noq`KcQBsD*SYZK}JFY_K{4{5mFrXEc#orHCJEujTl8nP*X3WnC@6WfZ4Zy+!Fe-lN07l>%gQ;}*Sjqq^C0
%4m)LPYqW$6ASf_s_11JshJwrUBBhz?$5NRPwmRM5ypN<l8dzk>T=u>%4P|9YRgk<8N^9+?b!1rPXScfcCo$~dGn7(NkcumCivBh
vzV!w&HE@)UZ5>F7|yBlnMA>r3!|ivMIee!zaqmLH!a6G5|tMxpJv0RF5)N+rXWT*$RH2xcNvrH(AX@7jUhlbC~v<jJv|O(o&6==
fK0zb&X<wu%RI8$)2pG<s{Y(L1Zh4|x=-
&>SgL3gH9(@Q#r%yqn}7grR2<vye*R9meZty++ci*T`4|6z>%Z{%==I!9#JO&&7}D8{FYYtakP4?y~IJp;82<Ksuj>tkJ<8Z$AnS
z8f|Lp=SZ;8Fj}~zPx*NRf|Q!ZUY~`2>);Nf6lqW=tluD_5z@QO7r^FtE-AnL8U9*nZ{`&>)E9d2XTA-Kt3XE*OP^wW!y?SRpF~#
y?S-
s1m#l#dkoA<`0wq1UU&rq2cZZokm{VbtEU$it&~+nVd{dV&PAsb)H8`A^hjY9roR77$E1j?sn*5CX|;V0DUn5hors)STEy?06{`<
^w;2D%CWJp;>z*~T$!s=iSJFvzKtI#2XFX~<VHh)s1XB{|YEKq=mT{|~_muW%7W%4j)~&@M{3Gwr6ZE^^)k-
MNzX3zR52#}qT2HcA$^<M2Canlk84@6LNyWTI0cnK3F9Atlgr6DsBuJS=TGN3gkSP_D9xY@0{+FjKq@FInLfyi#)fSiAsu#hTbpR
w+X*e+Ka>*0u5edlLUFeyT3q%8zAQ>XP)$OvV*|;rK{L3z{=a%0@zrubv6GH%4p~xJeh({w1(Go~BNEQMf1wWnuVFtX!gI|a=o=C
_cV`nZ;7X%Nio;-^hLh$2M6cz?M3T_^S4ili|o-n091ZYDT3&0({KC+${;{WcR^;^GNf`5H|bDPNE3HWR(0Xx=kcoI`2-
&IJswBZs2={r}V*p-WzBs^~XWdfDW7w{Ow?akj9J9GaSn)OWrPX!7kTJ%yn9y=&;3R%?BeVaGncOB_=xHJ$~Mxt?dbVn|Kz-
!x=!WHrQFXQP_y#<t2RPhL#D}-KsyK?-
@x66aDHc4WmH=ZjNK=>D%)GwBW#ZhI4{$!zC%BLb;i(ba(@MU_7H<HSi%7vu=P(~fRh$pc@09v7#i+hb?(&i%kD6wOUa<6B?_-
630tz)@+o>ZmX?KJYCW1UhaU=RTVB7viI#!F3*rQa;4!yc>I>ItXI&C_nT(=5fES`MgefFm*lGNtY~ly7vp?MlkC$HlHs2G1-
NU?|4I5V+aW#R&clg6JTSMG8G8P6B?9b$^EiO#}j0>xdPbXP3{PU-
#P8LOPKyRNKAl=g%+Cn#Gty%LOm7cJ{3vf8Zp6l!eR&4n%~xiSXaA8hp5{cJx>b+}b*hr*lU0)keLX%N9zdd@^8Bafm?uO`doWlz
_YeHN5N9`svxl6^Mdv72_Tgm;gkM$`VNCOQk|KSFSgz`KVLJ!>yy>M-N@SNBM5lFyg-c3$@4IP&69PK-vK*OUiGSLuYY3U9#tl<Z
A8iS+952Y1W$^@Rx48mWw#|B+yjHGv#J~3i&7#k48h@W36yZ|5qqM{@sds`#=0Jr}8yee{eK`Tx+zQq$@2DMi1bLq1lNe^L7v2p+
u%ot^yJW<c~}u<UZWxL5E!&Rg4JO>9i{86Pr;h7tnB<>uU?*=I2Fj&jMe>Ixrna4{ZP(rP-
G#)LZRVy_ob{R9q4kl%VCxG)9Zl8wduxZup;0i%}!vg5tAS5?5vMCyVemg@n(n<Pc%fpl8oMQw96)7m+56S6+sOfGIb*BbjooQLp
3^0Z0!hNFi4|na&mMD3yvG*^WY~+!1l<WTYevmICQrAd#=s8?|yK;s(@B1NcuT(k>?RE>>hxK+j^B!aopT149vN%<fP;4d++b4RS
US(@&)WD?A?AfPvWn7h?lSQy3y!Zm|2KsZ2H<54p`6A!SfQeJX8aVdBU#@`O2wFcYarKmamVrZrh?4!hO7zbj%AFk2{jc;k6@TNn
aUw7YM%+8s8FNh{-
$r{FKhI{N<m`w}QToP&RuOSQ2L%0&x=kR3!P0m7mMIh=!M+c**(vU`LAP*8e%W1JacdD7Q?$RFmUf<2yOsDOhhhXutKWzmj5hd&u
5d`-
ch$+=sJ$6`)49(p$66EtqRWqHgf>DNyu6Hb4eoALOm#1(V%Ij1v%@1OrZk*>S~r^v(mlWxo<J6V)feh=1x=~Qc!bl(+8MV2sx%uU
Nir&m~kC;=@=|DN0^tFxQTNsj*n#)}244olPAKLI!z$IL2x5i7K%3^kvL9%e;?oRwsm``vslgwKq8B`NOHe*FJb`!5Cd^S4L;fBQ
^x<Fj`AZ(jfJhx`*&|KEceYdQMQx%52xaRK;Gr5#&gAHLr<wor^mE8Cun2;b0cP!ucZ@v3}NOC*^Q02TZ_+3V2H06$J6|E&yNk5R
Z<9<FkFoWmM8dxD%fgUvkw+}ntrvHl+|@;<EWe?ap<JVHG}u6AshcaV8H%QrI(|Gp%YXda-
4Q(PQR#t0gho)j0pG<$k~gyR1E{=4Ngpo?e-9-
?}HkVMEaYNc6{s~K*rah9gd5ogGx_h+v6c6IhrbPw*2BM;*G#x{=3;P3?kK8G<X*f%T281PX!zrlxO-1hMDch-A8-
}}FSI(G2|#=rx;u}$JgH71MQ>9ku+dX-{F0=RUrQ!-3q#{$kA$&Ny$AAG2ha!A`7<KE{-
sMlYga{o1b0XCk*3|R0rN1kvpo6l!c(NHkx_t*{qm(EZESc)-q*~S9SoCBN39}I?~Dez%3>^ag1DckEanfvIMU-
!H}oxQ;u=5TIcneqcqEMIM%p0?`MYOPi+Wg_kaIUBB>H9<GFi8vFnM7I3E9m$ly%hfvgu$qr~4&+Si#zgMsyv+ASB+v_PfM=7i0e
%c!W^~8$Fo%A-)4RO7JZl!B4!w|q8>`*DHCDqLM-
l2B(L(bKdbihU!`%7#nA<3$V>VE7PwyH1uu25P82<H5JX?9_iDm(7_36dU%a^y8r=^%f%O@kn>&HYun69umGGFV6l};~jU%tG#IB
nD_rEJu5sAS_e=R{xKGkWG85j5cb9Y+^y%}#$LUF+Sxe*5O-
RVx=bl+g(|<c@cY`3<=n#u4bU!$7We_43Wz*SEb|I^uVlwPHGMb7LJfcW`0w(!BV$2aA({$0L}w@l>u<XNy&OFW$U+^Wwagu&dZ4
0tn8aQ;4{}2;jbyLI)up1QJ_iPt?xgw|kYCO(*42@!R*#-
CV{#e~%na9EqoQr)p={&#%wwY0n;?LSl&3h69sc&Ltx^?N~Bbt~VVR)M5sS!r$|x>+svPlv~du;ijYCmIkk`&i%hMhp;bz#uiE!O
T~1^uH{q6>|Kj5nuv#-`yyBXELpVg48;>spJkU#rtr1){@XT9AO2ME^TOP(RjU^)%mEwE!NcN-bfMa4v06-
O5ra&mN)LnCYNK8RR3{+X0S`z~iuFb{8$6U!iDZUIZGzr5s)ckS5<Iy%oqIPM`@Fn_1gZpcb1?c5a9bE0naSmGm<$$+MF&BLB-
)N8QfzikYiW<3142^>9K9!9JMA=!5z7t@yh5QvZ!?%29+yeR^>a7TbC1w#cr2}dF(be{{M!V!!eBW*Pz$I?uq}y97aM)4a;=(=SQ
T`b6^^d3M)K8KIps5o>E!-<RDt^7*kVwy36sIw6RBUQ@K>omf-
b<rLEWq;gZYbPs%$5TWW=>Epun&wtQ~{Z?GHyIev67tgoBreY?Z|yiH7}dt6_&lfd!xl_Fa)=;>4z6Vh8j0$wU>T=^{1{Rx6;gsP
@ZX00#J>?iZK>nQ{f{YdIaJpkXSl;dBm~$wrk3Qg4Knk4P7(jL@vjrNdSw6DFUb%b_u;lrsU702+WH{vYR7fI&e&r+m%VGnIoTtG
|p&*F*UErjuN)QAxSg02i4mGDj<?-Buy6$D_dUS8(7Pk;2;x6k6TWO4KZ(f)|-
;cdF8;<xWg|!mtQD3jTTUm)8#Z8ol_=sB#tvz{PWn;mX<7MKgIUrjp4NrY2CiynA)iPMD#MpdW!tAVAB$Inlm(b$3|_XqXf-nJPX
`HZQKuDq$lBj~OdszfujSKde;!2(|dp!zujy<81HM`*%0>AfV8pau3tJx4-
=Q;i_O4QAqs&I6&+rvKOvC{QAq=Uiy$r1>`xw`pvuduX@>IJ_>&H?q8t<)gKm>&KfIx2OFROzW}h}{QU8*71lDy6e`!8z5MX!zkC
D?E))0xd&|ea{Q1LW*36|+$P8_`b@%aSz_cTP@x$xS<Hsx10Q<x0)w57bmm!s-
;RqUN^=Fw9r!U^VzAAg<fC7l6kF@W8`swvq@_;jepL38rd;QZ-ckPIt1;~Hop7Pb}w=Yf;CKj{<^rL6S)PJG|8+sA8g%V6?&`jmK
{saaTWKw&o-f0(4_W4X2g+dcq5{=$@D`S^|)bOGHlW8)0rgh$HBrGE6H7b+8e^P9B>M6SvkdKW`-
<4~~O%z~TAOP9~7%V`L^+S=E+(Exhzh~N)vq1)o(G!XI%`(Q2=@VfEWys9_NW^0V*)MGQzEQ8Y`Gang1Ul?&Z%*N#QlHgI;LQ^NM
S;d9Fcwc@2&9r-YbX(QY1vd7M{Nt7K-(GA{p0w_G-
$&*3E0#e8kMDWMH3<Gu0$$efb@~E>5Hm9D^~DB6X3zT%9UUP>t`U5sBFn$wA`$vY!VuM$DS-#vpx-zOd5;7g%*5`FIz1q?K^au#F
nZx%aKC~n@Z|uAFZL`pWpo}*G?P_z|;hI+MfUz9Dyb`dxDwf*=fNor*jO^dau`rn>bVWIi`4{*Q-
Yj9J<_{KRs(^f*!M+2A7H77@xjBOjU7xurmAw#UICNKR%k80Hhsvw$ZHATJ@AoOqY2oS9dSEse>8(2dVChyQ_*vMi<*s^;WIYY-H
U#q$%}(w4es(3KcBBTCsHK(1O6=7>AkO-K)z=Chpd<7}|L6?H}J?<qp|X_}PcKtM`9=+ly-
%EUi19sa(Fg>tzlZILy}U^!*O4i0euvM5pn8y<h?$_&N5<?N1-
?8i4~Pn@ZOuFW&$1({=7(M*k0T*FXL8{vxTPQ`yRcK;!P?Pq!622RC>6F0bLbiU6n+;PGez;Fy-e_1kwh6(?BOY4Tw6_RY)Fh=xA
3{?Ij%)0b~<n?X6eO*ktz@7`V)EP(iPp#J0d=c~{C@z0k29h(4HvN}{dzi7tx8Dz)+;uxdVc0GB-
r%vIg@{f}Bb~S3`K;{pbu^($*oYz8XGWMS8zhe2>1ry-
W)C6Ge`4Y*XiARPBG{r}eOgga7nZeK552Q1ZBQc=QB$0WhU^3y`W1^}*S`lgV?w_vwJ6C~FYX3g#ub_CkRHNNtk>NyHx++lVc1m7
3vVZmS@4)^g_m(={N<c-2(`w1A9j!)6$M;vyN#paoRq@Z3E?$TqWdiiKFyvGV;b$6>-
P@bfn2t^!B|yT*VqNU?=C+$OFe!tV2b&j)`p?DxC29%gB?`0{Kgi)3Q07yZ`-
#rY^LA7Vx_O8n42(9~ety$Q>?8OQ8uwx5_0v_CA9Va*EXDr~stGy}0FD)g@zWK)V!NGp%V?D00KgAuGIze+F8UO7<N=5&9!uPRGG
jb1oxWeJjQ_zL{Cyt~G6HZfQ|Vm2CmL~S5SG|}{x8BTt8qo59z7RkUxn}^y+7tV`7QXrxHtaC6Ff{<1V1PTDKpxvdtw$HUS|gB*A
yC^CEl~zj50QZMn&*rCwP}0|HtDW@z7vk{0K^OB%28u_)Mgq`_r0AXYvifOg3UhB!lr|#(8J%kN?Zxq(6_Rcy!?Ir!%>_aJ_d{_9
~EmM)LRJaZz~7XT5q@$3^O45T<x@{&Vn;cyvfK2A#%Wi5!*N_wTN<Cf;xWknZQ1vRCik-&PzV7K27-
&`5YR+P!<>|8mv!b2)xIeP`dK5wbYDv8!MI2Vhx}Krj)WXQ2HImIN>}{f}R-V!IrcP-EKPq2rh0|BU<BV*SC<G@e8{d?e=TbFcpC
|NQU2T!rOq1V0JE&z6U;{_?;7&p*A&>AB*ga5~}9&~OX&=i5&*KUds;r|{!N&irY&7TM$L(l7q@zy9Ud^Ap(=e%Z<SumAGD{`MlR
<L^ak-P62NgkKf^Z?WH&;78Pls61(0-JE5Q1Uvrr+h2cvS+NRP6Zly|Yvtw7zrJnzcLYb-vzx2NiIO<A{VqAZ|91Nseg6XZ<$>zO
RVRJOhdQ@@)-
8FJ9Ay6XdC}=ij?!D|p4Fit;vc3vR~OZQoH$9`vLgPK&R?tH@3SZnsC0oPQLX3wY7S59O&7C411PvNw*Ej0BrpWC#k5z;<EZ`ldN
pAY(5VCrs{OF|_`Y)Y&piN79suY&|F6C8>~8D2(*Bndd*Z~hWm^&(2%`7iJ3)Zxy%LS+y#wr3lvOP$RxHOEdy+M?QohWZdC$4`0z
|neRP4O#E$4#`0{3b?`<#9Hex7YGY7~}0*bT;Zo;gv2A}u2m@L>9d(k``xnFr<wcQ{~CXp9Cab#=zsIo|2jhwR_yoM2%aFpAn6Pe
m*eu~fjI()hM)_xPY1QZXjxUxq4FJvi=WZG0M)A&`nCmPjh@RZ~``)}M=yuRienWa0=N#lI$SW*V(>)W9HteJxEIXk0w~`lMjv4y
Iph0_NHZCtp9kXauC7Uz11-L$us#WSoMvu@3y(869Bq{B17!xhJ6G%eRM*2#R2(f7H&{xJZ%$b^PGjo7Xo@zlbun{-
lWf&70S6o*l&1NM;7EEz>^gSAq({qleQAxargLf9(8lj{9$d4dc=2LCGrts7N$Ty#Ms|H`fh_e`Ng+@i(r&dHu8>*U;dD5^w3?^a
yMxH|Jj-%|3pfiGEJ?<D5Sb)sgz~#c?g9B*X2CBHTEC^7trk<B-SpUu2FgfAsjtc_XZV8yK=OR0D6-
BWmKx6#n1LxqTTw>&H1g$AdrN{$t&js0fz($GwtY2JmlDB%VU=_^=k$(I?K&>AGm`@VHm-
NGMwXzsz6i9rw#Y1#z+i|8@rcU;WEc7Z_uAc+>bNX5UAvB&8=^sh48Ad=l)ZTr-joz^|S<e^dJrxqmY}86uMSyRlNelJ+P`tB-
Ix@CROhEr370`+Bs_1wb10eAKSxfr=$@OtDI*Q%zX-Tbr1Q2x7n%-;$_yI+d7-
1Kgj&)7qo?dIOlvU4JyS1DkRExY&Y!nS%e(=ne31rn@ge;?6Yty+$z|G7BkNB!L|&0(J5pDQS=?4mq44Nj-UZf0VWhNLv)4Ig~Cn
di`d`Eg?*HV7vzS=vPZzp#BR!{Eufv5Fq$P_Efdq?RHzms98WJ@h!=EAKq(8NQ4pnAHtT{nQr#`^`w<gA`8sXVhg;}uBPmw5&Xm8
ADrlisq6Py;PK$&xDSSZjiL1<3XOx)(}P-
kmrtgvJ;_2b9oXiQ2w0{;XaIr4H3asHg``K#AoF(PHTXiKknm_3YdA~L@do(#4&e{u0T3V02!Oa9e%44ljXhpJy?J_bQj2KmB&Na
)<5KNL8DkTZnpt1pWXOzm__bNVB+<3u+6j36v>vx>c%-
$JN&Hg{FlN9~@h88UlF=MfAs#HAAFa`3CRZTUym<EN`Bf)nW^ZjVm5#k^F`xA6*aS>$hQQW&llfwH&!J>)5!t3>=j!>ZXBVwhz-
5xr)*g*5q4C0hK3*#PXEFfj<GBGqaPhoCRXStYQn_*b`0Ja~M$E_}l0?=-
wRhNWWIRd=JmUfNPf>a@js9V;ny`vUMCSG$Xs@pyA2-UST+FGYt~^*)`uSJ$O!yT}_|hy9Xms!XaD#7-
6q=oOz250}t4W8HvPo5VvaRFu(_SfTU?RFYYa2{MxYRp6KW=3`3hE|B>PS|*{Z75!?lcP#Gk@dZZTQb3?bY#Ew<+X{HHb&|z7Gk)
-6+Ig&BS97&yIw;NgyNW4*SilUrS#{;t$v9T7S0LKSV4f6E>;h9ZzI0o`JdJ3ejC5;o-
<0cwy>@4+!7G51<72*H~J2JXb92hiuz=wUi6+Z?d$mShfJu$_a>w>ci?9Q2-Og6|ymxmbD3GucT_-wk@<@DCXjBEo<%Jxc2ix@yU
hJlgmj^K6vnmz*d=EUXRn^4Mjp8i<(Q?*rZGJ7P}P+jjdtPw`)+A%xbsjCG^b=5?5vMfTz7qr^o9utJs7`W8#hT;%`4%N<Z_XrQ>
fOt!$840{ONlmMd2)`MAfx-
&$WMkZ24#d2{VHfEeuX$#e#dL|9+n;_E%}T%}sc#XUy3fJL5CKfLz~#y^k7Z+`{Dhgm$4R0G#ConE(*-?J+j11+rS_-&*w(-
ro;e52dzz%`RbD&j#49^N*<61lUM%^x&RK=8{iA3WM*O7#|(*XQ-`<!Y@?yOxXERZIdF^*j|nj~KDRP}-
yL=~gYb=Lhe)EP5$(^AS$!!TIWMcZvU>pf6zT5wbuJ^W;JjMIx~e<yaz4y^y@Vii3V&!s<GB%Nd4(FZ+oEq;BMrA%|W-
UKwqGas9nd?jrw&lL8Mfh1{24JtFe7j!34`>~!0WawZ=3yR6%C4tZnTbEXX-apVT8%O8$s%8hon)2w774lR%P=&O;`p%Vn2r3o<e
_0#T+de16ZWN^oFH7E}SQxq4HL7P?plObaH$5&TZ@bIq@r~-{Gm@Ew6%*EUW5gAdTS&+PPl0Ah_KAOWm`s{Y>T!jm_aJteyI6c3-
I6LkFe|8zf3=)~f;t54!k$}sD0nzaAjY8f+BDIl$h%Pp`K!52TpIuy@pB}U;X_(fSv^rQse7-34?1!+wKAlwGd$2-
eD|Z8_QtR;a^7{Jn?5I=C?s-
gVg~n(B1G~@bc36yBwMM_aV=@`_YMGEl+1wydxC)bJFI(*#ox#^m4_l>FU{}c|uG}&A@14kZpH8QsT>+zy(H+f#1~@r8KR-
S0w<_6qz~%Bq66su_SjeRl(NHL|mq_iWlCiMQVbVyrY?0RDi)AaV{_*Mg`Pm7m{%q831S8N2HV53lqlDM5?<n3i5T9Jxpb0e)OS#
!SI66KCo?I`c_rlR+wp4AjTCGN{RLJE^)q1nptXE6fWZ1K-Q)*1^$bPX7eE<0P=%CvyXJbB#Mo8OuG>QM-
T}i0(Qh0wlByQfnKiIjGIWmLIzqg++*PE?YvtB8r<KYm%U#!*}%|^XiE|n_vR;SzVby_f`ZrEcn=yn|bSh`R_Un%GJ_xv`4j6)vm
;s-p=57&^IsU0;=+<bs=chp9!+aKPGClX)*8uU7CcDFwgPo*>2OnQHRKa(p~YV}6FRw?FEdjY3er&Ma8vRWhtU)c-$-BzO-
;qV~vznJEs;)+P&7=lk7|8%zU)mMm)6^k#Ht28>jUZ+(nWD<!?sogeNY<34IPPaD@j>h8gcq|$Yc<g4qQp9C4IRc47t%a{>RB|z&
h3IB|HNvC)Y_^Ql5*+>+!b{`2Q7H2l-o9g(zgR#9nvaUXVskiLzECVvsG)6$abT&#q;8?}!#W~rja&f_6VIYd<|!ej4^&c^A-
nWB9`AaJ;)^mKR|cj*8T7$z?-
(_Bl!ra8f;CUd=+ou39_}p=jas04HII*tqs%m4&gf3kC^H2rW=SM`D0zQMTjA#xtXW2%=>9*dpRk0y(IT_%ru*^vFHt`GzoD%DUs
P5Wm2sKK*=6G2T^m=>#uVn!Oa+8t&WnkHeNP|^69ZuHj{IxWz$bPP50@#pJJ%622EfsqA>?+KMu&42k1qf#e~DBkS16P!l}bG*;I
mSpkjtbJu}CQ3^Ehl4gH9WU$Zc#4%ostB1Np(S1;>~X3sCq7EeFiCY_32gRj9OjqiNS_ce*`3e;^nNM<UVa-
k?OIkw`cc4ETK>x6^Lj1sejb3e3m?E(?ZvVBvfttnE?cIQo@~%9JIyCZGl&;cgVT(~^P(kJ;*Q`+||Zcyd3J%@<1LO0`z2LrDUxf
X}sBwNfq>^4ZLO5-i+;KDWba+SV$iLMVQ*g$4B8Zk}<~JeCwz!BzYiCE_D_*C-
6GNUqu5b$Ei&cq)@Gfu&}%)$VkA{r<th;bb{D==XcwP8%$n!E!X8NyQ@pk7IXRBNuWQP;wu^Kcc)a0v&JGo?RyF7&ie^64=-
z)45`$-t6#(VySEqEa=<aKGb|h3WgYB?!*7VKaU2#9rU~HX01}lreYzl1FlLrbTo}C3V#586#G0K$QcWPNgD<t8mPeVCYi~X>vo-
iXbNt)5b#i_8lZ!<+y*<a=CFW2T5ZrWNV^~p^}6i_*p#KB0q3qx&S#Q0QF-DO1pWa1w9e^lC8vpnz@!bB1-
AwesT_&M<P3lfSfvi(!^(#K0lXbLQm5A+IMD&zfi>%}M?n8Vngp~#y#lsh0VmiObEw1t{9E+X*k{kz_;!r1kw-
0nE(BUYs5IKVkwgZ#4?qTL7;%2293G#7Z9P`bPQm^dEue8g>p&|j#Y`gNwHcK{Q2Qg!bBlgbUv{ZQ#)J_=6Na+8Lj!;hG59i_$>9
qF6Kd`L(dqf+6(|?yC%{LJj!#Zbj*pT5!aq;WFW`@t=ch;gb`5}sXA(LYpFtk=eLVDYw7zDn;YLlEvS45UItsBw8jCNUZ}d;EZk|
4S_U!4C$CqbEy%rcTYt?F{QmKN$v{rBRj?OM0KS7?jxjOAP@^PO<BVtjI43ML>kHVh-
e}~odxD7)Sa14MsL`W>D(Y=?e_fD>!JbV7)IRJNd*sYh+u`rmfyk3vj3nsL1EM2O156^&!;F)Jnu1<RO+@9MgWx;`GeKvd?W#2g)
nMor?_q)>uAk!F`qqGF~EA6ARtH(DtkFU;;d-
dXe)N9*TNyS32SP%$>VySA|=8f(b>%HT1^s%#}c4a?kQF6#wB>FA<TTQI{b7YAZZ@xYo6B--
(5N5!@gNO|2jyF+k_CbF=0d7+*q{1$fTEb<H<xgaCC2Er^oGMg-Bb|W$+;0{W-
W@4}c<Vh_*Bw^=EKTT7m)^C1HhyUG#b^s)9z@t;N_RZ5Y`NZS)=HU}-
@dI7uxP{NgImKCgfx~wv2FLqGNoFxSubZ}o*gN3i!k&aJor;5DW5KV@^BC5)MsG8Xaf+4RGtEAK13oxkJTXK!N|ig@B<TI6e3S%u
zG@#2-Jd5@Tf$>c=Zo_fASz`68lmoKVwaBd+0NPAFe~8CYMYhmkK#Z-oYXKLH@yk0Z1M~j!-
IB$V5o)bc_ZwhX0GXV)P46u*OZ8lFT1z0BjB*An1q~JQC*cWXyer@CRQG3z`{ni%1xZ{e!V*3O?>J=)?&ko(-
Q)DDI8u&13l2)(I4jM4^%kVIuSaPAW?vl}iOI>a7ovhTjeC0uzT&u2M)i6eNM}%1HH3!(Z|c7tey}qy+=~P*#bl*hLZp8ALED66v
N)snHb(2VF)flZ?S9!|#IO0Q?T?*d&_BfTS7PRWK2urZN1t@Fx>tEIfez&FrId?Ck%`0sgfOs>~M6maD~-f16K1-
J2=1MswAAH5au?=_phZT{`?u#zZ>^g}?1j6|3cJ)Fz`2RDd2#NeRx(<1ujufMdd>0h6MK@G6d>@uq8?Ub_^xNofdthQtyrb&gIBT
Uo!BO-Az1lG$2+wsm-N)G0+R5(Z)bO=^pm+PzLK?bR>{0~MpO>YsfF^T{1Ek4z>`87M~)=qhif34pb8J`EFQTBZptiTd&7_0=i-
iiOQzER8SMKE1lWJgz4!LK@7<%!FS7Gn*N&icVM`!9Q^YGo?fM$?~^H<FbvnmoG#7RCTa&eDmz-WoO^Q2e2qiO}Kvk{PpYS7tN$q
OvCWg#MWf<0)Bm74{Mkd3Yo&U>~}7oKD#-t1l80L{CGN!Gw<-`S%8~4Xn6aDvUUhGU9@@i^6l%V#|5W|iX_R6cb~ld?)R^+n-
LwGis7fSb&=-
v>)(I(_DMI6q}HX1oQ313uiw7BYDRT56#s+KBy9Lus6JWh{_&F;nWsLRz`wRZRs^eO&tAQJ+)rD16u8V$hZ~nKfA{wJSsBZrOTu#
Jmd~EQ{oTvUMpz9OMiibk-GBV@)w8o|P(j{UyTyMWHwo2pSHL)zzrURno*Yfyy1vOV#p_4sXT5??Nk`ILOI^9%#m)6eEoxv>hxn;
%L$r2seRI*vxui&9Y`W4{=$)M()#E14=K5#~*ZX%;0bMrt*JJ}c#Ion|jDb`x6LTt&?n|Z1T-nyqQMVZ0=Fvv-
U(<No;bQmbsFiie5d37O(izL-QUN0m%b@#k!hI%HKtH*g{_1Sy&k+Y&ClKi(wZWhgF)1j1u{~Aqc54Z%XqW&SO@9r;61&}c$}UFn
Q<x%^!Jrn=hY9R)a}dsq;h2cxz&GGGQ@Q)F>-Qn=m#|6Z>1-~Wjz>ZAymPd^RH2Y`tJ$<M{4}=8oy-
?fUM&a73s1q`B5x9gJm9eyAY2vGxf%TMviD8kA5FfA@isTvJCR%|7cp^>`e)0np;*MF<1=VfG8xN1P9{@n48G14i3P24Hd6mw6Y|
y$8<Wj_IO+a#;V%_FGedfH#DSmzM5xAc)GPVrRLJs@dGg)EZqcV;(x@Z()2TG3!dC>p<UNoWNv6vECGb|!CnI93w+Bn7;QZIS1TW
1>zBzQDEtCmT3#(}WCD&3od-
3MwdCAVBkKw2D?4|RUZ(f`gEL>n2O&zYGtdp&Q`%K|qA_eMQvWHILM~T;)Vo$zv*e!S^pjPQZfBX6SfBXJLHz=kLQjjCb$LZo=_r
>@B_WpU>F9g3(B;G>zu#@+QH`i9j7C)cQN}>DfJH?Ov8b5jNXCoaLb)XHJE|4yygL)=(f1$7Q;`<-
}@UjQ+Q%3Mp0RG;~Km7Rpi;hnSI{;H31aAd&w2i^y|Er}!_+}-
boom2!2e`!#7I6xb&8+0{`79cZV<}#I^X==avW<&gub})>&@;HU^405azqu${I5ZlI&*v%4Hj{#WyLp+B0d<Ck@5f7}|GjO1Pp7m
0J{&{{Y|)rZ@s88EBW5sUzVhMOQ6(Tp`GdrP0aS)OP&qm~Ec;{(2Kq3QN!vo>3UKk?y3ev9pbN4`&%lpd#c+3H`Fw1bPiM+(kwnx
XXJOTk@i*b=iQExQL~Jr9osT}u-
5RJs%)yUe{Yyu$F2KKvX6f4`$^6B3yXcqG>0C%IwMrR7_yYqNQY)mWa_MwA`ml_I9zU$EE{lKZ{N0PQrw_(YG{S_0sD_jS3L0JR%
hf7L3zvd)-U0qC5`}9?R%$t)oK8~&%7~H(X>%xJF+TlHXRu#31N_HJXRm)Sbbmyi(Vi-oQ+5HBVMsI&`(;%BggCqlM)Xh0{ljL$z
@QEuMj3iz^Vcl=AC&+77XNSX@!#s6(B$@D&@QJ_ApWC+vQJFDg^x05#lG^v5yVfSqYu+Ybx`o|FQH+*4E|4W@jo2V-
GJk}KrR>1DHN*Am#<fo77kYZ=!HCN0FEVDt>=9*DrNBSK*(inOhb7Q{-wg_|8+if;mc9k5M+5tqtU^-
5)_u!9S&ND?sJQuBC!U;ZY>KwL<LXNkOk_5kjNJcNf;mBA^abY`SSVL25bYu2a8K4lR*at01K#0ky6g1V!e03|Ajr6%9ATaO!x?Z
PlvC-B_45e0saN<^Q*-
TfJ+@gH^2la+QQ&SHEIcm2@Rmqcq+X{NFz_xKba=f=v6%A89I|AQEQ|e29$CkY>W!&++TqIBODIN2jG8*2w!b%f@-
A;%+YKnY~%q0XmpV+mPxvi&N+^sCU++@F`EcJ#N-*nnQYW7fCg-$VTcd!j(<t=%LMo3R<>dt34-95vqz7w`zZql_z-
(H(>*yTy7&Y20m49mfbS|EoOCn0Y~VW_L#luM_$X`U&?sBO4dj@T%P)n0OmzlEdom=E!*9NS|Dx*?G8qh!zj^)arXNFC;-
lj~RT1moJiBiCMGOX0=<B|C|NS?I5eb7r8amKj@#C|PK3-0G9yxy4B$BC2Rr>1vpZ@gjEGA<y8In-
v=H=7Fr21C=0TX~wCl8;#yy=7_3?@q!JA3!1KfS+7tC&<W?7{!L@PppVRHiQ8{q*DSPNGs4lPU02&!1eiBXZir`iCZuw6C6=S3Lq
IlP!&&{O-q}-
d&`WOtb@xCjt29;9o=WGeo}j*KglEt~mt&KNtA_>0#L`qD|;`(L~<z;VJNcE|bX;IBSpJy#2cE6Ae1>h?36dzli^TI;HwEz|YY}>
xT!Gpppae8xxIwJL?ckt)B&sY`fn`7$JU+GFUk{tVgxT2)ar5K>Q#5Josh1foRC4;IUaO7T=O?A9iyN;XpoTb0DB2bmY2+?X-
mto?`P9woo*%E1QM?-g5XqnjZe=<NtgbKU3k$H=B8%k_+&&CEn`kHB_k>$S-XU1el<*#r0{`D}j%3mFP<f=0g0R+-
3Q8cl<QIvvU6I*?HM11QpBCr%zvf`}(qE;n2tSzjThJbou(*SEp$`2RtWqmf=ekC!dDn|DR3b|M2zCCwTZr#~+$JcKGVM?_M76DO
gMv&sKZ-{-+;ab^QYeA`aZg-+lGNPw$`BY&=l^ioL^^-+lM$Feaaa|F=Ky`tu$Rej<scOq{&_!}qUG5-Nb7@2Efh9^mf<L=*U-
e%cQ|egCxX;DhH>;HB^X0A5nkM#j&tmPrEhA?yF&;^If<U#cY7e)8tclXg(b1jCOhfBwz8H`i4M5BSd(rhl~s{D<eLUcY(w&3WDg
1|X(%@REc&HUBQc|M7<`pK<ZS^$(e4Otw!>T1g{^!C<ge;l|nZc{8eJ(#G_oXiRmqd47G?2&>rOIgT;eIyq^lOq}WU4<7!XWB+?^
GIb5Q{HJmazF5p>;DOZ}ohfl;TfJt+DV`cX#m-E#*UGviOz<2-V2Wgmxu}_stv~1B|GCZ|?v19dxorTjg5yi2LN;7I(-
<=3G_>dgTjBBfmoA6r&z-zS#(<kO?sl})tmk|RBm@O-
1NjS`|BPRNA*z{R4iKsJ8VQR=r7QeJMAt_}pIE=qRftYd(XXIWX)KA(7Av=!g@B4Wto{e12SE7eZy*0-
=Wm~l?H@4%063P$9ZSW11}>E*cI?;N^^8l3%ES)#FJS#Fb!EUW`wlS;dR%-
zl`R(Hb}^JmS)IJT`?=0Pe?5MH{@KX>ePjbq1(RvJ-zkN4ObS!&LQLV2C-
%Q&9sr$+xzykpUAWZgw~H~qRYAwSe_87M`J)d$f5z|+3}Bg(t+T81ZpOx^&?J^{DjoOg*rYN2WD;BFji*y#i-
bnu+cMqrtFu;eS47<$>aae*KmT8Jeeux}7Z@|o|IsK`wmH=X_<LEqfWkHHRXY7vKA@%z<-
6bs9Zem`xB8vRo{38l*t0!=znwC(iP-7MDE`mp-d}w1{mbO}|KI>|oj}tB3c!EL(QOt*;?8x?E>4<BGlxhR>UR-
{9CNaHa&gwpxg`|VcC-xqrx4K4#?Q}Z=1^Qdh2q>B$jJ*ltO1A|y%#vWe@8$jOWoPd$@x(uVdhTYPc)9sPdZt*ltP9%K2kBSo<qD
nz8wqR7`^|RHsD{EyS^J0AHc;wa3CT>x^1;vbV3?&i@Os+@*#QE^vx0cn{>4o$#)dl;cjiwL^_Mzx-DZ8G4IF4KRb!#3=8I{58#G
y4Awwwtt+~pP6W&nDv``LxI>YEUCkmAMhlUpwg)01w}DS4QYE{AM0$VErQ>X&YoD>}N8Ic8Pv&30&$@u0iyx|=u!UlY#+|CP+Vy<
c#NPrvl`oZvITSSD6P}FH=}2NW3P&uH@<FfO;+w+xM!Qu>xwR6pkc}#!EQfz__yaC{eEh2<fzIaf`#j-
1RQIVQ95M<4<5bajGPPE#H=1{MEf$kj%-
BLEGT>1JvecfaK!u=u*yHniY&rpH6&L@fW1&k+CeZ!$B>XY_BOy!=S4d)WFjXiOGl?`(|51yWj5<7-
r?m&7v3N3*%Y%~MgIPrfYJ4OSOz_ukR?>+~u~bL}&0^BZ%-
|~=1&F)JKj7#mpbLm~w%)g2X?HudTs#ibBU+V=O+;{y#NYNMi;zQdfDm@NwY1;BC1QlTO@_i60PBZXJXh;<+m(Hvp1r<Q@YsR`%y
ac0#zV*e{znAfb|6#h^!x2<ChW5AXk}a)f{_4|$Q4`tqhly-admld(k;hqQVOQVN2GG4+8v84oT;|^{Z1_t*yas2jAjT}{01lZZK
is~Z25{Y{2O4{->-I$PL2<n`8~ha1Ep|STN@BRqVdx|JUP3#fGQQue8eIF_`zVi#e$+c9<P5d-
#j=zIqFvT!9=hz8~=Fh<nr0O=f+SX_0Ki<)AfV1%geKaMyXgT74{=8gOCC|oUeC9_VXo33Trp()nYnqSFwp;uG*jo4dACj2|V9GU
#&ym!m0ne27fLL0iD9n(Rq^P&d~{|#AXxtQMXZu!;T6JV9N{^r^gozM<d~Y*SV_|(>DQpBmr=|0Gg)XX*Rp?)lNC-
(Q$6^<H=V{MGh}c{yrmf`snzxMiCn!aiLnRlq${M!C}8uPWep|+6Jg+B8AD}3B*#FOd{g5>14RaU0vOvN=*J#xz#^B=rzGx)heVh
7{!#ehYzMgpQi;xCSspw#Si0Ut{)GAMhBoZ5?8Tpb9?=PXsS@H13%dhJ9Gl_I>u7PXu<0!F>@7FU(z|k`$hP0Ar%eyy>8pKf=gP%
hMzBH1<o%oA2Jn%{^{%}bSS;e;P53Xqum#dM#FxmNyVpZ%;rt6Zcz9tlhY3$_Suap37^A&BHZ|q^RpuVm%%>H0{C=_O7i7M9Q_7~
!IQ{UDuq<QByX(F5RXF3>IRu9kSbIvxrE0cZH&ate>qKhnWuj;5Bfp@l>6g!D&#P39lJ<h$nHiq4eRKc!!6zJmN7L){^U2yl}Rqo
G#Tms2q$lp?tI5GB*;$5pWK_tI$B!3Wdi;^JOPwxoshc}T>X<7e6=Z_)EwLk6<j_a;|kA{J;4P$rGj#QO1%Y_EBFEOGt(L_6I5Yb
;`XBQ#Z2{<e?)x-Pl5)AoQf-
Z^2Hbbkb=uJz4t?vXJ)F<%wRjurJyfRotYw4&()pzaOgPtE<b|=%_8}h`7GbXh!@J!ELN0R#J69_3cvSrlwVRy=T6!*KNsTvUuKt
;|98p^m7F`5e~L!Y2bT|ky*9ht#gLhhmX<D>ZN*H)m>ScF5)*ojF`~FM%bL09AKtahwEzeD`lzH4UnoXYm|(e;UE;D?O!_c?-
@p)D3l>42FS>+l1RnZv@LLo*TOh#%jlBL~C=?Edg28~#>v6kXP6t$8+SaQT5&?&S+`CViFpYl^>IaJ~nKl3iJoFi;XvCAK47(0*X
fK)0K^>+_rCcf%^7&jYm(6C<sl*;4L}^e-
`Aq88=2Rn0!e0b^&h}}R1l9omi2N9)K1L+bxnk9})f0;C=b#8{y92dmT1_Y+)dc^oS4)LVG8%AM3`!B3hQ|MoNq4P|<HrTRtn}50
3Ah?wm{=yA!C=slKwub5z!oVD7IzRB05x@>c2NTgxqy!?U`MT7fSNh>9kqnVV!%SDApw-@;|7ewzq^3flm(B*1-t-
!M1V=ELQeS2I|j8(04kJ7gh~T4t;vpPV(q6hne=`tnL=c}3i(_*84G#sCaqj7mTL?<@KvK8`Vtuvp&7$Jj()*TZgShVK!9vxiy=_
LW1m<ul}g4VeuqgdMpYC@R0dZlQv=H^R;$&rYu>fk94?O+bYh3q1YBPrkZ4Q}e<TiHjlqK;rGT-8T!mo<OvC?bSq-wUaK*<Sj>*W
cLE#6kV#gIq0za?UYvp_@;xVfPbP^_5fd-
|s*r330Juse%#A30K&t=i6RE9uh_C!+oa;*+}L@pU}?I^fV24Za*|Az>u;i#z&wa13|No<+X8BXOZz_T0G!hY0aQu1h|QMnV+lq3
fx{lHV2Jkk9E=;6&qC7%jAjWRZA68|*(SvzQ)-
S@NsBlwYVhd}1Yjm}^kYRKm^iLl$G<kQI`$jK9Bs2f5e)A>r1JDkYCSJUyJ(<tYV3G3^(`g@N?RMhUK0f%b=9)3t$<0}jnm)9Tgx
vfT(kVzc@j*=_}1z|Q&ZmpoQges%e?F;z5U@TSe5hB}Q3|bkAz&#v4<XTd|Zd$$W;jQ%B3I-
pceIXKjGO$9B@YyI?AB9c@&d(M|)jES-r`72VI<-
{5W*|mjHb6cbUknx)G7*nS8KkOO8&MF%g+J#g0LOx<1{ke=<erqw5^IbWt9cs?AQbWtJd-QYA_Gh!u@~J-
B#_}rE8#M4+kryo$@ON7Wk)MvL;QqU`0t{@hpWAZgMR=Yrtu0Tv$=HKzpG$VF=h_%4KwJT`EtDlFOTZYb_)uHN5KFkWMM2HDoeWK
-AiY)sfbH2q~01Zihl<DrQ`#r4Vc6aW?Poh5i2&k{Z1_#GD{d#NNr*8G<KMl0TFdSKJ0aRhsUT?dnFUGYap*26uQvpPgmQ$UaORF
YdK_;$u<N3^5~~5z*7!HMVFxWaCN>+<KXn{xKrA*%7Ay!0DSOW>-h5V<Lk@wlcRpS(e59eoL^o)etdb{%J_6V1`Sxi5St@~_R-nt
VKW!p=2M0l$wx#GAL78DeS9=+!KeWPMeUW5>W696_R9V9tIN|)A-p4mafJ-7+O^-
lc=r0uH_xw+JM~g#KU1oAj<278^XB!li}t=t%|+r5`P;#K`{eTK{Gbxs6;aV-dF$(IGejIG&@Ym2{9;sd@3sLZ@YCe(Z1?i%i)Ys
d#ej|riA8L=JyAQldG)(@Z=RpGb5XBt*XE7p+UL*Ty!+j&o1<F7E@#6BIod$+;QHB%r<c8~TTa^=ai9_W<JEtE@p0Ig0TT^yD;tQ
CHUWN5{^0S8Z(iOU7X4}#3`^zegT?;EljkpAKD+GZd<Ln2Cy*L^x$fn&moJ|`x#$;zdM*rfWvTte!<(1iym)+&_e|n{G>LzX2<7K
<q$d~Pr|`}3`pNZ^$EQ#%hk@ZQ^)G<0pZ04hx0=UdGMOx%+MTNPPeGr!=$A133@E8{`uNH9Nj+}nQ)c6zr9(MeKXZ}p<PblJCbmQ
ijdrsXv&&)i(-
|z8Jz47=9kz>ck5R&5B5QmmM`HBEi|xasUM*>tu^@grOKy*qn(am*Y7wJM)cN=q>ZIctaN7X5_{lWB#_A1tZ8{;k8(}cH3Tt>jmx
*}{5-tnChem;gC|6?e#4@@4uvNiDjbgBbIvaS`s^Qbf82$+%%9%o%pD#FxT_~HmLBP@~QyFZ&P{3t#cswrRD=e-
^Z8Yy_rO<T{_=o|(iKN;cvr#SLvXJMvJT9Bf6$trk26dnVJr6&=4(KckCL3Tjekfec;Y;)un^_}Zv)CMgTnm;vd=8s6fRDBSo5Ke
SBCT8ieqjqVW}8JP;d7vj^&<RtfRAecj_T(Kex^WUaeAU@DCHw$b7j_OCYSOV1XK70245<diCSe`wh)R2rK29FMI&I2;a{JF9~b`
AVflywbMa%WE-
;QH%3zD$?l!U>B~RkW^)Ii_ntKM`1b&`juX%QT+0Qv7Jf$bw=(fS8zMKH#)_^HCn7}5KOuxhzVx71Rm^9%2nC9nKW1^*4_k~%xX`
G!@@A~P@>EThg;8*i?iG%0ge)s0O<`J{U@Uz99+Vz|7zI}d>(DBv&LigzK^ycYxFSP^pfJmbQ&^#s9&u8Cd;21F40K=3VBf}3AJa
JX8UcY<$^--
sma>)2Q+4Hx5{PF$sPEf{1IN2zE;6Aco=lS~||M>PiyTg|`Qnk*}*KgmwzN)$qIh0BK59cb$&bWygGvNM2n8P^!f&0@1?%MU+@4k
K7%k8-|e4aja{ObJ=@18Wg;xYW7{=JPS?|yjy>Nut6@inekuJ`oY@7`Y5+yeS-
2f!wPN&L&f$1z|E|E=UF>i%?&DSi0$%V%e0@3xH3<w?EOi{~$&9PAr;6aK?9?jJmP`TU~lmGZcJ`L?%w_Uz@?hiMasKH34M@Z(kg
qI<6i2g1dVo{`X*5@)7$&@BeEQW2la73d??gQHH?DdkM`Uyjt7?HnCcBYFXs%NI$(`mKA=$~YyEHjgOn;NbsqS_^mnHQs~)UyMqw
-r^s+KN$Vlk{xd-
U^P0tE~A9U1MB*1KIKvKrp8a6+LOv>BW5v=Co#Ib4x=>?^6p63w=)2NFoOTVn3ycSj`B<crW|OX*^3UJBlsaOkyL7omKv3WMa<(%
wjC~u8qQMV^Dmg$)E1XxTf*mwEs08_6g5hvA|7%hFxG)*<DZ6)b9IfUnmmOcbAKu_vT(ROnXhtkb>57E9?TPnB_bY&jp9ewKL~!X
#t=!w0v_nS(dPNpNyR7Qagk{n>%apsk%1WNgqZJmV9>-
B{G<V&e?BSMJAr?=c!2Qvs!03#`IE!_Z2%7zE(ZW(AqzD+vaDkv8J*A{1>5_FPo7`5BPu>(z=#8l;s0s^|D3>@=`fiI{9obYhy4=
37b*hv^Cwr`xK6-_1v7w&tu#^hf$))E1-f|m>dARMpb$a>SWt&!9RJAlH6t!-
Ui`zf1C8U~7~n^$pUW2r#nRnWv)|4+WdgJy77PtwqE5!+W53Frxpu#q+Lelh0zP*n70FP}XfggdAty`XN2;IA<q1R*sa$0V#Ul=t
NGKGE#KRmZEDS$um|I0Gf{&;ik$A|Wl1n8b0a!;P4KNq~oUC9AbkFhdBf}5Uds!SFOrE0Owe0Bi8o303m&kQH+bW5WKWqTB`uReM
YI{c~mjDwaa*bZUW7*Zi6e>Kph^C^Kl7w;m)9!z73H<l)I>01;#C@3XU|*uN@7J2epg|#%h{Y1MJ6$QocBMk-
8E~rtH#P7%>29o4NxRkXA(_GuEH-QVb|o~Fja;gd?tp&@;rV&^2ksB>3&e73>GJj4XT69P_=8mGtzW%(c9J!LkrHVDCQ|)iR5fK!
p1ruLdzIiRxi-
>!_V)E<$to8M2KWbJM*lSYU}Me`N;I+a_y76VpI+1~3Yk>4oqPQLPd~jq49O?(%R`56fBMt=$GL6rl)_Sb@zY=b^Zj{DBN6hzHh&
KO`%_Z%i||iG+T!>Rh99`UP-@6O`}2SO&)>g0@TlZ6xuyQ<&wu~x57#NR5V>wg@pE}Xb?W+uzyAHtuj&@?l*)7P-
QWM`zyACzZ;%RM2SD)?=HOqV`llLT27Z+EJHW5cJ^k~4{rBH~e-
JQf<nrC>%YXmdU;c2JP$3;)fS)f^B`*K)m%sh{%j&LNt}z7;e*d@s{;xkj&FM$*PmMo!bf4J{G(CP|{tx^|9X<X1fBfaA7tKt<t(
EE17w>-j<2OCOe4_s4{@yo#{PEpIS})VO6Pf0VpZ@Y6zdwztf&bk0-wlio`HSJdl}v-<C)$54#mhJEUY)kO&3&s}73w~H3-
uc%6aFuO`i*a&c0($;b-&qdoxXba=CWwP`Y&}D)Pdp0AO1faKgQP_k$gkPe>kjr!Q$<--
9J4nxfM!Z^YN<}XL++2n*f;D@Gmy!&tANG-
1I3G?$Y6DzkLdpa9%i3pyNMj6#v6t1pmWf(h*dV2$_EcB84?xu9xzNY>Qm(Dxck4_u@LteX#ipIFK&hyS_OqyX102$R%H@m*ZB2N
Py12!{KKv7r@Vj|4TglumivzY=~Cc%}gX)&W8+Qxx0LPc~bK$Q1`*s572=W{@TgqaoH^w8$$VVHj-
(!D^UaNy+7CRzw`t!wg3{@@&^|u^^n8kFi3<7f8*lGRcBAbhbznB`VBacX0LPg<f7qM2qgyaY^Z*6ageu(<}81HDg3t!Xt)3sI*L
b^XN|B%CJ_pR+T`)eZ{J*(t%BhRBH91~Yx(-kw=a*ATJV%arU^IBE{}>1A)CdZk1d~9eo_2G$xO5XfC0RnM5EuyyQF-
$H!<d~zWw3-^Oi?+yMafx?;=m@`THNfy~-Qm9)~Y=<vaaGVuy!rUk2wVYyTYlV>>X3Z7-
el0*3*Gug{#n{_fi+btiHFI&yv`aMquE``zpFj2?YZ<4vabY?86v*J$^>`|@+q0`xX%%KG^Evq)n!$Ut}HfIXHh(71eh(@z<=2z(
kWhyh$fs(<tJvJsGhy%~6xCo>o|qVe<RargOo@#8eWzyRPK95z}we0{jss3t8U_7Fa509#~9RvX2zo<DpRdKd<Otok2K9$qb(#%b
0Lcq(hqumLcFI9P~^<x&9@w&H=ou~OVK3sCq}<W3tgKw#c0R=^MoWxm(~sa(u~=rNV|VfFu0@Q)lJqPuvZ(Hjohl^iBZU`f@xO;j
ZZMUPm3D(f`6^^`@xVse!BVAyLEqC59lhlhBFZ}*ni|K6K_1UGU7jdTH)Sf$yGSK5u7SHWWPcN4XCBkdF;*X&5F0|jXUhS-
^Iv}=i7K9i;J<{Irvd{?6qvj#^$V~5WVW*#0cd;Wljf3g8!7a+5>u26WddVG1_Ntw6|w#J*vrGo}8jfSB|Euhi3hG06E@@m)&t|`
?yzdWw)g+nea8>&ogP9<_$i2vS_sn=%U9~eMWdiKlZLi7CQ>L6?7(V5a+U&v>cGEnq`SSDZrLuv-
U>`Iw*o;7=Lb#vY<l*{`bCGGa^2z8$scNf!#w{y<lrig-LcQC8N5RNUTAyR6!tJVJLaWi3L(?yoOTrnHo;SB<PhS6BOop838+p~!
1Y-
6H%eA=&8+pSW>z?{50eYj*6Jj{s2C%9907$Xh9&;|4LdLfxD72<XYouvywX`j4LL8pvGzftH4R3s>*W6>q{c%hU{7V7nUP{$an{*
~ERcgvArK3kNzZ)NNfn=H13Q>lo<Vz=r=bSis$uX=FYD}^*jJl1U}ABCX_m3qeq)xB*tl`hg-
?G{HQl?q$L<mu3j2XpVfmWDs?5^@-
WMHTAIX1xTg*66@X7;1m6(W<50a@uIRGVqxucc*KuM$WHhfamC7MJLgl%{n2KI1+m`7I`vt_50b<_ZOedyadJ^=(YzC$uuT&=qwc
8wl|iHy0sk2Shh9r3rFjYCS%@h-
tbvElSv~JN4$Rl{>26?PF*?aKwDG>3rf`ua8lSZtw9S@`clX0pUU6}<=__?n=<^GLSwQRXzUw~`^>?=;Qnz2{%Hpy5a|+w)nVI~v
4*ZrVF-
0rx5v37=g@{<(m3)Rr^jv82^mB9WR`5(=CB$hbRq%IeZInp+?dB@`)nrsFDAl(hWMfIQnFAk?)!~=3YMydF0;o9wMHc!w5a%WB-
WG8S6PDTN~2bY*=2MrsSSm1^zRqTg(ND&gpYqg?86Mc-HafP2?ttVConYrT>JR!xKj*m3z={aPNIse$@<ad)p@_VAF?VzyO3E*Yi
Pd;<K&L&NvnuTLKN?W+o56yyxY$CHH<+L%h~wv-
#LHo{h9a&i%=qeH(o!!y171S?)x2PjS$o=UF9qEubw{t`tqn<iaV4HGDGQzm)b{{Uq63()i3x|bkHC|joIPfZ=PJ=TpicryL{qk;
Kq!=u_X<d9fdU&hDGF=_bLbHR~JXkQX!WL+dy?Ql%8DY{N@SFLfgs*jXWyP7|gd|=GiAV=bfAfSj1A=!l_)L)I7R?@rQe69`R0rZ
_6hy#>0Q>Km@wdnJB^V+D5zAZ5H-yGCEbT6UsGuFv)eV4wZ4JP-Ul%e$mK<b_7(q%(ho(c6;px4A3nloJu5%3V!nI8IgO7h_JK55
XT!}lOZ#?Lwm_WqknYRt)$!<Hkm5WIYKb19x5wYWWax97DS~fpN@wdIsui;hI(1u!=rwqklYKoO)|#jWc5$Rt>T6r&LQ#Q=D(i6k
0zg`^W{cwqS$D+YPqOQ!GQDxu2f^RI^E7)t%wdfD_x}B1%FwM8YveN92g2)G*@f28^wgzDCg6$6tt`OVb5PI8~QmT2=t4Yfza3#K
o#mig{P8xev1kQGJ`1@^iUY!#-x%E{a6x}3I1Y%-VG%kiDZt-;@?ZAKy&DYRN~m=H4Fb2_m{R{W*G5w5_r@A3QMHaZ-YsfN88#Q(
jG^$cZ~iCf8V0<z*M}gSBhAaY4|h3&X-
C5J~Ik>27ba8jU$xEWKuDo%|N2~mn@L^z~&?BKtc{}i?BA2wSXUa{`q_bff;f1Bly<{RGz|Mb@>7TzuRV12^m9$v_*wLVhB}6o7*
1<_*_<lf=4B+;o)DFx3CnCVFLg979=1A<B)_?$R;6whezF93jQXQ2l+e6Yynb#Ab&^3-
de}QkHZ_e1NbjyQ+Za_h^zsBs?};#i)qN8;!-v-
itn7Vu}R@V{#3eHg|A}#DHd^U1&4Hm&)ZoB`y`KNgi8hSuah8sDg^e8Bm6Z!jkpAUB8@-
FU&|&#kU&LRUs*u<x|2A`0xH<b+9r*!G}^r=e=vsf2iaKC3zTg=jvC=)W2rFMVl|}qps$9#cB7I{+Z@*aB<XGeZEYzc+cXt#ZG%M
TNmM$c3Gz3$b!w$jsZy)eDy2du7Vy}NiTr~MHcudyDX?dhO0{kq@;6OJol3%^lQz~?rWvO*X@g4u#~~7qQz2JZ*TL9NXY(Zr?Y7z
G^7@0}2q@v8-|KeR05Zt_(r7f00S2tJA(ZUlBQBeHTdP2bV?@F_Hv3LdW*1OSm!WauQc;nM$W8JT-@00bjO#)^i|}R>@fa-
eL=sYIvw4J(4ITkc<kDd6<8|yBK<zWgo7llQz`v06dk5fSbmj^48$`(T)fg=<e>lFMEtG4JZ{0--Lc4CaTFpUewZPxtTkPj{3o@C
r`|+^fWie`G0v46HF|Yuib^QVARCr8m)OP^-O%k0gkf{x3yC)Edr!x6crPgRd1A6`b!NK9-Ar#IW6!i1KLBHRF@SBZVrIgR4;t_-
l54q2D(k7}TyO8&O2h_Mccr<Aulp5fR6&i!d;&A&z(O7an13jn;tbn|CU`G#Ff)(%?whE-
e*Q(HaGW*F`G~{<XEGC0SA%?wR3r(en$KAh!^uR1Rg%J+_Dkl6-V{-
T+sR9fn!_EmMR}ojw<?{tFyADeMY7n5EA|8tlyEquO6jBkN!=#~EJO=#ze@wYzri#VNfPgv(fj};Rp;m=lq0(r<FtcOYwODO7yTj
phx?C<dTEHi#)8Vk&Y*x#zX$K5UT8&B}mx__#E5=aKVr^|whvU)DQ6_aa)`kpPY(he=VUUX?cvsDWv8wPcZ?wQK5H2|qm5W>skdc
ccY`Vi#6juL_FaHaM4KiE"""


@dataclass(frozen=True)
class EwaldCase:
    material: str
    lattice_a_A: float
    incidence_deg: float
    log_max: float
    texture_index: int


CASES = (
    EwaldCase("Bi2Se3", 4.143, 5.0, -2.5435566400182315, 0),
    EwaldCase("Bi2Se3", 4.143, 10.0, -2.332494993346455, 1),
    EwaldCase("Bi2Se3", 4.143, 15.0, -1.788306297359463, 2),
    EwaldCase("Bi2Te3", 4.386, 5.0, -2.009756843028878, 3),
    EwaldCase("Bi2Te3", 4.386, 10.0, -2.085877587602414, 4),
    EwaldCase("Bi2Te3", 4.386, 15.0, -1.5001513719459711, 5),
)


def incident_wavevector_Ainv(incidence_deg: float) -> NDArray[np.float64]:
    """Return the sample-frame incident wavevector for a grazing angle."""

    angle_rad = math.radians(float(incidence_deg))
    return np.array(
        [0.0, WAVE_NUMBER_AINV * math.cos(angle_rad), -WAVE_NUMBER_AINV * math.sin(angle_rad)],
        dtype=np.float64,
    )


def ki_frame_basis(ki_sample_Ainv: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return columns ``(x', y', z')`` with ``z'`` parallel to ``ki``."""

    ki = np.asarray(ki_sample_Ainv, dtype=np.float64)
    if ki.shape != (3,) or not np.all(np.isfinite(ki)):
        raise ValueError("ki_sample_Ainv must be one finite three-vector")
    magnitude = float(np.linalg.norm(ki))
    if magnitude <= 0.0:
        raise ValueError("ki_sample_Ainv must be nonzero")
    z_prime = ki / magnitude
    x_prime = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_prime = np.cross(z_prime, x_prime)
    y_prime /= np.linalg.norm(y_prime)
    return np.column_stack((x_prime, y_prime, z_prime))


def cylinder_shell_radii_Ainv(lattice_a_A: float, count: int = 5) -> NDArray[np.float64]:
    """Return the first distinct hexagonal reciprocal-cylinder radii."""

    if lattice_a_A <= 0.0:
        raise ValueError("lattice_a_A must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    reciprocal_scale = 4.0 * math.pi / (math.sqrt(3.0) * lattice_a_A)
    orders = {h * h + h * k + k * k for h in range(-6, 7) for k in range(-6, 7) if h != 0 or k != 0}
    radii = [reciprocal_scale * math.sqrt(order) for order in sorted(orders)]
    reachable = [radius for radius in radii if radius < 2.0 * WAVE_NUMBER_AINV]
    if len(reachable) < count:
        raise ValueError("requested more reachable reciprocal shells than were enumerated")
    return np.asarray(reachable[:count], dtype=np.float64)


def cylinder_ewald_intersections_Ainv(
    radius_Ainv: float,
    ki_sample_Ainv: NDArray[np.float64],
    *,
    sample_count: int = 721,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate both exact branches of one cylinder/Ewald-sphere intersection."""

    if radius_Ainv <= 0.0:
        raise ValueError("radius_Ainv must be positive")
    if sample_count < 3:
        raise ValueError("sample_count must be at least three")
    ki = np.asarray(ki_sample_Ainv, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * math.pi, sample_count, dtype=np.float64)
    x = radius_Ainv * np.cos(phi)
    y = radius_Ainv * np.sin(phi)
    discriminant = WAVE_NUMBER_AINV**2 - x * x - (y + ki[1]) ** 2
    valid = discriminant >= 0.0
    root = np.sqrt(np.maximum(discriminant, 0.0))
    upper = np.full((sample_count, 3), np.nan, dtype=np.float64)
    lower = np.full((sample_count, 3), np.nan, dtype=np.float64)
    upper[valid] = np.column_stack((x[valid], y[valid], -ki[2] + root[valid]))
    lower[valid] = np.column_stack((x[valid], y[valid], -ki[2] - root[valid]))
    return upper, lower


def central_rod_intersections_Ainv(
    ki_sample_Ainv: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the direct and regular intersections of the zero-radius central rod."""

    ki = np.asarray(ki_sample_Ainv, dtype=np.float64)
    return np.array(((0.0, 0.0, 0.0), (0.0, 0.0, -2.0 * ki[2])), dtype=np.float64)


@lru_cache(maxsize=1)
def _intensity_textures() -> NDArray[np.uint8]:
    packed = base64.b85decode("".join(_TEXTURE_DATA_B85.split()).encode("ascii"))
    raw = zlib.decompress(packed)
    expected = len(CASES) * TEXTURE_SHAPE[0] * TEXTURE_SHAPE[1]
    if len(raw) != expected:
        raise RuntimeError(f"embedded texture payload has {len(raw)} bytes; expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(CASES), *TEXTURE_SHAPE)


def _sphere_mesh(
    texture_index: int,
    stride: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    latitude = np.linspace(-0.5 * math.pi, 0.5 * math.pi, TEXTURE_SHAPE[0])[::stride]
    longitude = np.linspace(-math.pi, math.pi, TEXTURE_SHAPE[1], endpoint=False)[::stride]
    longitude = np.append(longitude, math.pi)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    x = WAVE_NUMBER_AINV * np.cos(lat_grid) * np.cos(lon_grid)
    y = WAVE_NUMBER_AINV * np.cos(lat_grid) * np.sin(lon_grid)
    z = WAVE_NUMBER_AINV * np.sin(lat_grid)
    texture = _intensity_textures()[texture_index][::stride, ::stride]
    texture = np.column_stack((texture, texture[:, 0])).astype(np.float64) / 255.0
    return x, y, z, texture


def _split_finite_segments(points: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    finite = np.all(np.isfinite(points), axis=1)
    boundaries = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    return [points[start:stop] for start, stop in boundaries.reshape(-1, 2) if stop - start > 1]


def _ki_graticule(ki_sample_Ainv: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    basis = ki_frame_basis(ki_sample_Ainv)
    curves: list[NDArray[np.float64]] = []
    longitude = np.linspace(-math.pi, math.pi, 145)
    for latitude_deg in (-60.0, -30.0, 0.0, 30.0, 60.0):
        latitude = math.radians(latitude_deg)
        local = np.vstack(
            (
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.full_like(longitude, math.sin(latitude)),
            )
        )
        curves.append(WAVE_NUMBER_AINV * (basis @ local).T)
    latitude = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 97)
    for longitude_deg in range(0, 360, 45):
        longitude_rad = math.radians(longitude_deg)
        local = np.vstack(
            (
                np.cos(latitude) * math.cos(longitude_rad),
                np.cos(latitude) * math.sin(longitude_rad),
                np.sin(latitude),
            )
        )
        curves.append(WAVE_NUMBER_AINV * (basis @ local).T)
    return curves


def _configure_axis(axis: object, title: str, elev: float, azim: float) -> None:
    axis.set_title(title, pad=2.0)
    axis.set_xlim(-WAVE_NUMBER_AINV, WAVE_NUMBER_AINV)
    axis.set_ylim(-WAVE_NUMBER_AINV, WAVE_NUMBER_AINV)
    axis.set_zlim(-WAVE_NUMBER_AINV, WAVE_NUMBER_AINV)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_axis_off()
    axis.view_init(elev=elev, azim=azim)


def _draw_frame(axis: object, ki_sample_Ainv: NDArray[np.float64]) -> None:
    basis = ki_frame_basis(ki_sample_Ainv)
    frame_length = 0.42 * WAVE_NUMBER_AINV
    for vector, label in zip(basis[:, :2].T, ("x'", "y'"), strict=True):
        end = frame_length * vector
        axis.plot((0.0, end[0]), (0.0, end[1]), (0.0, end[2]), color="0.28", linewidth=1.2)
        axis.text(*end, label, color="0.2")
    axis.quiver(
        0.0,
        0.0,
        0.0,
        *ki_sample_Ainv,
        color="tab:green",
        linewidth=2.0,
        arrow_length_ratio=0.09,
    )
    axis.text(*ki_sample_Ainv, "z' || ki", color="tab:green")


def _draw_graticule(axis: object, ki_sample_Ainv: NDArray[np.float64], *, alpha: float) -> None:
    for curve in _ki_graticule(ki_sample_Ainv):
        axis.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="0.25", alpha=alpha, linewidth=0.55)


def _draw_intensity(axis: object, case: EwaldCase, stride: int, color_map: object) -> None:
    x, y, z, normalized = _sphere_mesh(case.texture_index, stride)
    axis.plot_surface(
        x,
        y,
        z,
        facecolors=color_map(normalized),
        linewidth=0.0,
        antialiased=False,
        shade=False,
        rcount=x.shape[0],
        ccount=x.shape[1],
    )
    ki = incident_wavevector_Ainv(case.incidence_deg)
    _draw_graticule(axis, ki, alpha=0.38)
    _draw_frame(axis, ki)


def _draw_cylinder_sections(axis: object, case: EwaldCase) -> None:
    ki = incident_wavevector_Ainv(case.incidence_deg)
    _draw_graticule(axis, ki, alpha=0.48)
    for shell_index, radius in enumerate(cylinder_shell_radii_Ainv(case.lattice_a_A)):
        for branch in cylinder_ewald_intersections_Ainv(float(radius), ki):
            for segment in _split_finite_segments(branch):
                relative = segment + ki
                axis.plot(
                    relative[:, 0],
                    relative[:, 1],
                    relative[:, 2],
                    color="tab:blue",
                    linewidth=2.2 if shell_index == 0 else 1.25,
                )
    z_absolute = np.array((-WAVE_NUMBER_AINV - ki[2], WAVE_NUMBER_AINV - ki[2]))
    central_line = np.column_stack((np.zeros(2), np.zeros(2), z_absolute)) + ki
    axis.plot(*central_line.T, color="tab:orange", linewidth=2.2)
    central_points = central_rod_intersections_Ainv(ki) + ki
    axis.scatter(*central_points.T, color="tab:orange", edgecolor="0.15", s=34, depthshade=False)
    _draw_frame(axis, ki)


def run_viewer(*, initial_mode: str = "intensity", stride: int = 2) -> None:
    """Open the synchronized six-panel interactive viewer."""

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.widgets import RadioButtons

    if initial_mode not in {"intensity", "cylinders"}:
        raise ValueError("initial_mode must be 'intensity' or 'cylinders'")
    if stride not in {1, 2, 3, 4}:
        raise ValueError("stride must be one of 1, 2, 3, or 4")

    figure = plt.figure(figsize=(16.0, 9.2))
    axes = tuple(figure.add_subplot(2, 3, index + 1, projection="3d") for index in range(6))
    figure.subplots_adjust(left=0.02, right=0.86, bottom=0.05, top=0.91, wspace=0.01, hspace=0.07)
    color_map = plt.get_cmap("magma")
    normalization = LogNorm(vmin=10.0**GLOBAL_LOG_MIN, vmax=10.0**GLOBAL_LOG_MAX)
    scalar_map = ScalarMappable(norm=normalization, cmap=color_map)
    color_axis = figure.add_axes((0.9, 0.22, 0.018, 0.52))
    color_bar = figure.colorbar(scalar_map, cax=color_axis)
    color_bar.set_label("Σ |F|² ⊗ mosaic intensity  [Å² rad⁻²]")
    radio_axis = figure.add_axes((0.875, 0.79, 0.115, 0.1))
    radio = RadioButtons(
        radio_axis, ("intensity", "cylinders"), active=0 if initial_mode == "intensity" else 1
    )
    help_text = figure.text(
        0.875,
        0.12,
        "drag: rotate\nrelease: sync panels\nI: intensity\nC: cylinders\nR: reset\nQ: close",
        va="bottom",
    )
    help_text.set_in_layout(False)
    state = {"mode": initial_mode, "elev": 18.0, "azim": 122.0}

    def redraw() -> None:
        for axis, case in zip(axes, CASES, strict=True):
            axis.clear()
            _configure_axis(
                axis,
                f"{case.material} · {case.incidence_deg:g}°",
                state["elev"],
                state["azim"],
            )
            if state["mode"] == "intensity":
                _draw_intensity(axis, case, stride, color_map)
            else:
                _draw_cylinder_sections(axis, case)
        color_axis.set_visible(state["mode"] == "intensity")
        if state["mode"] == "intensity":
            figure.suptitle("Continuous SF ⊗ mosaic intensity · kᵢ-aligned sphere coordinates")
        else:
            figure.suptitle(
                "Zero-mosaic reciprocal-cylinder intersections · kᵢ-aligned sphere coordinates"
            )
        figure.canvas.draw_idle()

    def select_mode(label: str) -> None:
        state["mode"] = label
        redraw()

    def synchronize(event: object) -> None:
        if event.inaxes not in axes:
            return
        state["elev"] = float(event.inaxes.elev)
        state["azim"] = float(event.inaxes.azim)
        for axis in axes:
            axis.view_init(elev=state["elev"], azim=state["azim"])
        figure.canvas.draw_idle()

    def key_press(event: object) -> None:
        key = (event.key or "").lower()
        if key == "i":
            radio.set_active(0)
        elif key == "c":
            radio.set_active(1)
        elif key == "r":
            state["elev"], state["azim"] = 18.0, 122.0
            redraw()
        elif key == "q":
            plt.close(figure)

    radio.on_clicked(select_mode)
    figure.canvas.mpl_connect("button_release_event", synchronize)
    figure.canvas.mpl_connect("key_press_event", key_press)
    legend_handles = (
        Line2D((0,), (0,), color="tab:blue", linewidth=2.0, label="sphere-cylinder intersection"),
        Line2D((0,), (0,), color="tab:orange", linewidth=2.0, label="central rod Qr=0"),
        Line2D((0,), (0,), color="tab:green", linewidth=2.0, label="single incident beam / z'"),
    )
    figure.legend(handles=legend_handles, loc="lower center", ncols=3, frameon=False)
    redraw()
    plt.show()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("intensity", "cylinders"),
        default="intensity",
        help="initial view (the figure can switch modes interactively)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        choices=(1, 2, 3, 4),
        default=2,
        help="surface mesh stride; 1 is highest resolution and 4 is fastest",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    run_viewer(initial_mode=arguments.mode, stride=arguments.stride)
