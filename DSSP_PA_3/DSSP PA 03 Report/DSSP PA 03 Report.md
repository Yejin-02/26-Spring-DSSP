# DSSP PA 03 Report

20261099 양예진

# I. 배경 지식

## 1. Linear Predictive Coding

`Speech1.wav` 는 16kHz의 샘플링 레이트로 샘플링 된 약 56초의 음성으로, $16,000\times56=896,000$개의 샘플로 구성된 긴 수열이다. LPC는 소스/필터 모델링을 통해 필터의 출력인 `Speech1.wav` 전체를 저장하거나 전송하지 않고, 더 적은 메모리의 소스와 필터의 정보만으로 음성을 압축할 수 있게 한다.

음성 생성을 소스/필터 모델링 할 때, 입술 방사나 비음, glottal airflow shaping 등을 무시하고 excitation signal과 vocal tract filter만을 고려한다면 다음과 같이 모델링 할 수 있다.

- $p[n]$, Impulse Train (Pitch Period $P$) or $w[n]$, White Noise
- $h[n]$, Vocal Tract Filter ($H(z)$ only has pole inside the unit circle)
- $x[n]=h[n]*p[n]$, Speech Signal

$H(z)$가 극점만을 가진다고 하였으므로 다음과 같이 수식을 작성할 수 있다.

$$
H(z) = \frac{A}{1-\sum_{k=1}^{p}a_kz^{-k}}\tag{1}
$$

식 (1)을 LCCDE의 형태로 변형하면 다음과 같은 수식을 얻을 수 있다.

$$
x[n]=\sum_{k=1}^pa_kx[n-k]+Ap[n]\tag{2}
$$

즉, $A$와 $\mathbf{a}=[a_1, \cdots, a_p]^T$, 그리고 $p[n]$의 주기 $P$를 알고 있다면 유성음 $x[n]$을 완벽하게 복구할 수 있다. 무성음 $x[n]$의 경우 $p[n]$ 대신 stochastic하게 형성된 white-noise $w[n]$을 사용하여 복구할 수 있다.

## 2. LP Analysis Implementation

### A. Forward Prediction Filter

그렇다면 $\mathbf{a}=[a_1, \cdots, a_p]^T$를 어떻게 구할 수 있을까? $x[n]$과 $\hat{x}[n]$의 MSE를 최소화하는 $\mathbf{a}$를 찾아야 한다. 이는 Wiener Filter의 일종으로, Wiener-Hopf Equation을 통해 최적의 계수를 구할 수 있다.

- Input $\mathbf{u}[n-1]=\big[u[n-1], u[n-2],\cdots,u[n-M]\big]^T$
- Optimal Filter Coefficient $\mathbf{w}_f=\big[w_{f,1}, w_{f,2},\cdots,w_{f,M}\big]^T$
- Output $y_o[n]=\mathbf{w}_f^H\mathbf{u}[n-1]=\mathbf{u}^H[n-1]\mathbf{w}_f$ (음성에선 모두 Real-valued ⇒ 둘이 동치)
- Desired Response $u[n]$
- Estimation Error $f_M[n]=u[n]-y_o[n]$
- Cost Function $P_M=\text{MMSE}=E\big[f_M^2[n]\big]$

$$
\begin{aligned}
\mathbf{r}&=\mathbf{R}\mathbf{w}_f\\
E\big[u[n]\mathbf{u}[n-1]\big]&=E\big[\mathbf{u}[n-1]\mathbf{u}^H[n-1]\big]\mathbf{w}_f
\tag{3-1}
\end{aligned}
$$

$$
\begin{aligned}
P_M&=\sigma_d^2-\mathbf{r}^H\mathbf{w}_f\\
&=r(0)-\mathbf{r}^H\mathbf{w}_f
\end{aligned}\tag{3-2}
$$

실제 구현에서는 Ensemble Average를 Time Average로 대체하고, WSS 가정을 위해 신호를 윈도잉하는 과정에서 Autocorrelation Method를 적용하여, Normal Equation을 통해 계수를 구하게 된다.

$$
\begin{aligned}
\mathbf{\Phi}\hat{\mathbf{w}}&=\mathbf{z}\\
\mathbf{U}^T\mathbf{U}\hat{\mathbf{w}}&=\mathbf{z}
\end{aligned}\tag{4}
$$

### B. Forward Prediction Error Filter (Inverse of Vocal Tract Filter)

I.2.A. Forward Prediction Filter를 약간 조정하여, 필터의 출력 자체가 오차가 되게 할 수 있다.

- Input $\mathbf{u}_{M+1}[n]=\big[u[n], u[n-1], u[n-2],\cdots,u[n-M]\big]^T$
- Optimal Filter Coefficient $\mathbf{a}_M=\big[1, -w_{f,1}, -w_{f,2},\cdots,-w_{f,M}\big]^T$
- Output $y_o[n]=\mathbf{a}_M^H\mathbf{u}_{M+1}[n]=\mathbf{u}_{M+1}^H[n]\mathbf{a}_M$ (음성에선 모두 Real-valued ⇒ 둘이 동치)
- Desired Response $f_M[n]$

이 역시 Wiener Filter의 일종이므로 I.2.A.에서와 동일한 공식을 얻을 수 있으며, 그렇게 구한 계수들로 이뤄진 필터 $A(z)$는 $1-\sum_{k=1}^p a_kz^{-k}$, 즉 All-pole 모델 $H(z)$의 역필터가 된다.

### C. Augmented Wiener-Hopf Equation

식 (3-1)과 식 (3-2)를 다음과 같이 하나의 식으로 정리할 수 있다.

$$
\begin{bmatrix}
r(0)&\mathbf{r}^H\\
\mathbf{r}&\mathbf{R}
\end{bmatrix}
\begin{bmatrix}
1\\
-\mathbf{w}_f
\end{bmatrix}
=
\begin{bmatrix}
P_M\\
\mathbf{0}
\end{bmatrix}
\tag{5}
$$

이때 $\big[1, -w_{f,1},\cdots,-w_{f_M}\big]^T$이 $\mathbf{a}_M$이란 점을 이용하여 식 (5)를 다음과 같이 정리된 형태로 쓸 수 있다.

$$
\mathbf{R}_{M+1}\mathbf{a}_M
=
\begin{bmatrix}
P_M\\
\mathbf{0}
\end{bmatrix}
\tag{6}
$$

$\mathbf{R}_{M+1}$은 $\mathbf{R}$을 $M\times M$에서 $(M+1)\times(M+1)$로 확장한 것으로, Symmetric 성질과 Toeplitz 성질이 유지된다.

<div align="center">
  <img src="DSSP%20PA%2003%20Report/image.png" width="50%" alt="식 (6) 행렬">
  <p>Fig1. 식 (6) 행렬</p>
</div>

### D. Levinson-Durbin Algorithm

$M$차 필터에서 $a_{M,1}$부터 $a_{M,M}$까지 $M$개의 계수를 구할 때 $\mathbf{w}=\mathbf{R^{-1}}\mathbf{r}$을 구하기 위해 역행렬 연산을 하지 않고, $\mathbf{a}_1=[1, \kappa_1]^T$부터 시작하여 재귀적으로 구할 수 있다.

$$
\mathbf{a}_m=
\begin{bmatrix}
\mathbf{a}_{m-1}\\0
\end{bmatrix}+\kappa_m
\begin{bmatrix}
0\\\mathbf{a}^{B*}_{m-1}
\end{bmatrix}\tag{7}
$$

이때 식 (7)에서 쓰이는 반사 계수 $\kappa_m$은 다음과 같이 계산할 수 있다.

$$
\kappa_m=-\frac{\Delta_{m-1}}{P_{m-1}}\tag{8}
$$

식 (7)을 가정하고 식 (8)을 얻어내는 과정에서는 식 (6)이 쓰이며, $\kappa_m$을 구하기 위해 필요한 두 가지 변수는 다음과 같이 계산할 수 있다.

$$
\begin{aligned}
P_m&=P_{m-1}+\kappa_m(-\kappa_mP_{m-1})^*
\\&=P_{m-1}(1-|\kappa_m|^2)
\end{aligned}\tag{9}
$$

$$
\Delta_{m-1}=\mathbf{r}^{BT}_m\mathbf{a}_{m-1}=\sum_{l=0}^{m-1}r(l-m)a_{m-1,l}\tag{10}
$$

또한, $M$차 Forward LP의 오차와 $M$차 Backward LP의 오차 역시도 재귀적으로 계산할 수 있다.

$$
\begin{bmatrix}f_m(n) \\b_m(n)\end{bmatrix}=\begin{bmatrix}1 & \kappa_m^* \\\kappa_m & 1\end{bmatrix}\begin{bmatrix}f_{m-1}(n) \\b_{m-1}(n-1)\end{bmatrix}, \quad m = 1, 2, \dots, M\tag{11}
$$

### F. Lattice Predictors
<div align="center">
  <img src="DSSP%20PA%2003%20Report/image%201.png" width="50%" alt="Lattice Predictors 회로">
  <p>Fig2. Lattice Predictors 회로</p>
</div>

- Initialization: $f_0(n)=b_0(n)=u(n)$
- Recursion: $f_1(n)=f_0(n)+\kappa_1b_0(n-1)$, $b_1(n)=\kappa_1f_0(n)+b_0(n-1)$, $\cdots$

Lattice Predictors는 위와 같이 $\mathbf{a}_m$ 없이 오로지 $\kappa_1$부터 $\kappa_M$까지 $M$개의 반사 계수들만으로 구현된 필터이다. $f_0(n)=b_0(n)=u(n)$에서 출발하여, $M$차 Forward Prediction Error Filter의 출력과 $M$차 Backward Prediction Error Filter의 출력을 재귀적으로 얻을 수 있다.

### G. All-pole, All-pass Lattice Filter

$f_0(n)$에서 $f_M(n)$까지 구하는 과정에서 재귀적으로 구한 $\kappa_1$~$\kappa_M$을 계수를 이용하여, $f_M(n)$을 $f_0(n)$으로 복원하는 필터도 구성할 수 있다.

- Input: $f_M(n), b_{m}(-1)=0 \text{ for }m\in[0, M]$
- Recursion (Time 축에서 반복)
    1. $n=0$
        - $f_{M-1}(0)=f_M(0)-\kappa_Mb_{M-1}(-1)$
        - $b_{M}(0)=b_{M-1}(-1)+\kappa_Mf_{M-1}(0)$
        - $f_{M-2}(0)=f_{M-1}(0)-\kappa_{M-1}b_{M-2}(-1)$
        - $\green{b_{M-1}(0)}=b_{M-2}(-1)+\kappa_{M-1}f_{M-2}(0)$
        - $f_{M-3}(0)=f_{M-2}(0)-\kappa_{M-2}b_{M-3}(-1)$
        - $\red{b_{M-2}(0)}=b_{M-3}(-1)+\kappa_{M-2}f_{M-3}(0)$
        - $\cdots$
        
        ⇒ $f_{m}(0)\text{ for }m\in[0, M-1]$, $b_m(0)\text{ for }m\in[0,M]$을 알 수 있음
        
    2. $n=1$
        - $f_{M-1}(1)=f_M(1)-\kappa_M\green{b_{M-1}(0)}$
        - $b_{M}(1)=\green{b_{M-1}(0)}+\kappa_{M}f_{M-1}(1)$
        - $f_{M-2}(1)=f_{M-1}(1)-\kappa_{M-1}\red{b_{M-2}(0)}$
        - $b_{M-1}(1)=\red{b_{M-2}(0)}+\kappa_{M-1}f_{M-2}(1)$
        - $\cdots$
    3. $\cdots$

⇒ $f_0(n)\text{ for }n\in[0, \text{Length of Input}]$을 알 수 있음

# II. 결과

## 1. 코드 링크

II-2에 기재된 결과를 얻은 코드와 그 결과로 생성된 .wav 파일들을 깃허브에서 확인할 수 있다.

[26-Spring-DSSP/DSSP_PA_3 at main · Yejin-02/26-Spring-DSSP](https://github.com/Yejin-02/26-Spring-DSSP/tree/main/DSSP_PA_3)

## 2. 결과 분석

`Speech1.wav` 파일에서 0초~3.6초 구간의 첫 번째 발화에 대해 다음과 같은 결과를 얻었다.

### A.  Analysis - Lattice Predictor 입출력 비교


![Fig3. 원본 신호와 $f_{10}(n)$, 화이트 노이즈의 2번째~4번째 프레임 구간에서 시간 도메인, 주파수 도메인 비교](DSSP%20PA%2003%20Report/image%202.png)

Fig3. 원본 신호와 $f_{10}(n)$, 화이트 노이즈의 2번째~4번째 프레임 구간에서 시간 도메인, 주파수 도메인 비교

Lattice Predictor를 거쳐 생성된 10차 예측 오차 신호인 `f_10_full` 과, `f_10_full` 과 같은 에너지를 가지도록 랜덤하게 생성한 노이즈 신호인 `white_noise` 를 원본 신호 `data` 를 국소적으로 총 960개의 샘플에서 파형과 스펙트럼을 비교했다.

시간 도메인에서 파형을 비교한 결과 원본 신호는 해당 구간에서 준주기적 파형이 나타났으므로 유성음 구간임을 알 수 있었다. 또한, 스펙트럼에서도 포먼트를 확인할 수 있었다. 반면 `f_10_full` 은 원본 신호보다 진폭이 크게 줄어들어 적은 에너지를 가짐을 확인할 수 있었고, 주파수 도메인에서 확인한 결과 그 에너지의 감소가 저주파 대역에서 크게 일어났음을 확인할 수 있었다. 하지만 랜덤하게 생성된 노이즈처럼 완전히 평탄한 엔벨롭이 나타나지는 않았다. 합성 결과인 `1_analysis_f10.wav` 에서도 “The empty flask stood on the tin tray”라는 문장이 쉽게 인식 되었으나, 원본 신호보다 노이지하게 들렸다.

### B. Analysis - Pole-zero Plot

<div style="display: flex; justify-content: space-between; align-items: flex-start; width: 100%;">
  <div style="width: 32%; text-align: center;">
    <img src="DSSP%20PA%2003%20Report/image%203.png" style="width: 100%;" />
    <p style="font-size: 0.9em; line-height: 1.2; margin-top: 8px;">Fig4. (a) 3번째 프레임 LPC 결과</p>
  </div>
  <div style="width: 32%; text-align: center;">
    <img src="DSSP%20PA%2003%20Report/image%204.png" style="width: 100%;" />
    <p style="font-size: 0.9em; line-height: 1.2; margin-top: 8px;">(b) 4번째 프레임 LPC 결과</p>
  </div>
  <div style="width: 32%; text-align: center;">
    <img src="DSSP%20PA%2003%20Report/image%205.png" style="width: 100%;" />
    <p style="font-size: 0.9em; line-height: 1.2; margin-top: 8px;">(c) 80번째 프레임 LPC 결과</p>
  </div>
</div>

3, 4, 80번 프레임에서 LPC 분석을 통해 얻은 $a_k$ 계수들을 바탕으로 $A(z)$의 근, 즉 보컬 트랙 필터의 극점을 계산하여 pole-zero plot을 확인하였다.

10차 필터를 이용했으므로 10개의 극점이 나왔고, 이 값들이 모두 단위 원 내부에 있음이 확인되었다. 또, 음성은 real-valued 신호이므로 모든 극점이 켤레 쌍으로 존재함을 확인하였다.

3번 프레임과 80번 프레임의 극점은 상이하게 나타난 반면, 4번 프레임에서는 극점 위치가 거의 동일하게 나타났다. 이는 II.2.A.에서 확인할 수 있듯이 원 신호가 3번~4번 프레임에 걸쳐 Quasi-Stationary하게 나타나기 때문으로 분석된다. 실제로 해당 구간은 모두 The($\text{DH IY}$)에서 $\text{IY}$에 해당하는 구간으로, 같은 음소를 발음하는 중이므로 보컬 트랙을 모델링한 결과 비슷한 위치에 극점이 나타난다.

### C. Analysis - LPC의 역할

![Fig5. 원본 신호와 $f_{10}(n)$, 화이트 노이즈의 전체 구간 STFT 비교](DSSP%20PA%2003%20Report/image%206.png)

Fig5. 원본 신호와 $f_{10}(n)$, 화이트 노이즈의 전체 구간 STFT 비교

원 신호, `f_10_full`, `white_noise` 총 세 가지 신호를 STFT한 결과 위와 같은 스펙트로그램을 얻었다. II.2.B에서와 같은 방식으로 프레임 별 LPC가 예측한 극점의 위치를 바탕으로 유성음으로 추정되는 구간에서는 예측한 포먼트를 함께 출력하였다.

이때 유/무성음의 구별을 위해서 zero-crossing rate와 energy를 기준으로 삼았다. 그 결과 LPC 분석을 통해 얻은 극점의 위치가 `speech1.wav` 에서 나타나는 유성음 구간의 포먼트를 잘 모델링하고 있음을 확인하였다.

`f_10_full` 과 `white_noise` 는 음성 합성 단계에서 입력으로 쓰일 2종의 신호이다. `f_10_full` 은 10차 Lattice Predictor를 거친 결과, 원본 신호에서보다 포먼트 성분이 약하게 나타나게 되었음을 확인하였다. 반면, `white_noise` 는 랜덤하게 생성되었기 때문에 모든 대역에 걸쳐 에너지가 퍼져있는 형태로 나타났다.

### D. Synthesis - 원 신호와 합성 신호 비교

![Fig6. 원본 신호와 $f_{10}(n)$, 화이트 노이즈의 2번째~5번째 프레임 구간에서 시간 도메인, 주파수 도메인 비교](DSSP%20PA%2003%20Report/image%207.png)

Fig6. 원본 신호와 $f_{10}(n)$, 화이트 노이즈의 2번째~5번째 프레임 구간에서 시간 도메인, 주파수 도메인 비교

`f_10_full` 과 `white_noise` 를 M차 All-pole Lattice Filter의 입력으로 사용하여 얻은 합성 신호 2종 `m10_f10`, `m10_noise` 를 원본 신호와 두 가지 도메인에서 비교하였다.

`m10_f10` 은 원본 신호와 거의 유사한 결과가 형성되었으며, `2_synthesis_m10_f10.wav` 재생 결과 원본 신호와 똑같이 들렸다. 이를 통해 Lattice Filter가 완벽한 복원이 가능한 시스템임을 확인하였다.

`m10_noise` 는 시간 도메인 상에서 원본 신호와 크게 다른 파형이 나타났다. `m10_f10` 은 거의 0에 가까운 Reconstruction Error를 가지는 것과 다르게 큰 오류가 나타났다. 하지만 스펙트럼을 확인한 결과 기존 `white_noise` 의 PSD보다 원본 신호에 훨씬 유사해졌음을 확인할 수 있었다. 기존의 전대역에 걸쳐 평탄한 엔벨롭에서 신호와 유사한 포먼트를 가지는 엔벨롭으로 변화하였다.

![image.png](DSSP%20PA%2003%20Report/image%208.png)
Fig7. 원본 신호와 `m10_f10`, `m10_noise`의 전체 구간 STFT 비교

`2_synthesis_m10_noise.wav` 를 들어본 결과, 원본 신호와 완전히 다른, 귓속말처럼 들리는 음성이 확인되었다. 하지만 포먼트 정보가 반영 되었기 때문에 음성의 내용이 “The empty flask stood on the tin tray”임을 충분히 이해할 수 있었다. STFT의 결과에서도 포먼트 근처 주파수에서 에너지가 크게 나타나는 것을 확인할 수 있었다. 그러나 피치 정보는 STFT에서도 확인할 수 없었으며, 실제 음성 역시 단 하나의 음으로 말하는 것처럼 들려 인간의 음성과 상이하게 들렸다.

## 3. 추가 실험

`Speech1.wav` 파일에서 3.6초~7초 구간의 두 번째 발화에 대해 다음과 같은 결과를 얻었다.

| 입력 \ 필터 | 2차 필터 | 100차 필터 |
| --- | --- | --- |
| `f_2_full` | - | `3_synthesis_m100_f2.wav` |
| `f_10_full_new`  | - | `3_synthesis_m100_f10.wav` |
| `new_white_noise` | `3_synthesis_m2_noise.wav` | `3_synthesis_m100_noise.wav` |

2차 필터와 100차 필터에서의 반사 계수와 $f_2(n)$, $f_{100}(n)$을 추가적으로 얻어, 세 개의 합성 신호를 만들었다.

1. `3_synthesis_m2_noise.wav` 
    
    2차 필터에서 얻은 반사 계수는 차수가 충분히 크지 않기 때문에 under fitting 되어있다. 2차 필터는 단 하나의 포먼트밖에 모델링할 수 없기 때문에 유성음 구간의 음소 정보를 충분히 반영할 수 없다. 이로 인해 2차 필터에 화이트 노이즈를 입력한 결과인 `2_synthesis_m2_noise.wav` 에서는 발화를 완전히 인식하기 어려웠다.
    
2. `3_synthesis_m100_noise.wav` 
    
    100차 필터에서 얻은 반사 계수는 차수가 과도하게 크기 때문에 over fitting 되어있다. 보컬 트랙을 모델링하는 데서 그치지 않고 음성의 피치 정보까지 반영하게 된다. 이로 인해 100차 필터에 화이트 노이즈를 입력한 결과인 `3_synthesis_m100_noise.wav` 에서는 원본 신호와 비슷한 인토네이션이 들린다.
    
3. `3_synthesis_m100_f2.wav` , `3_synthesis_m100_f10.wav` 
    
    $M$차 필터에서 얻은 $f_M(n)$은 식 (2)의 $Ap[n]$에 해당하는 부분으로, `f_2_full` 이나 `f_10_full` 과 같이 $M$의 차원이 적당히 작을 경우에는 음성의 피치 정보를 담고 있는 부분이 된다. 이를 피치 정보까지 과적합 된 100차 필터의 입력으로 사용한 두 개의 합성 신호는 상당히 Artificial하게 들리는데, 두 피치 정보가 충돌하기 때문으로 추정된다.