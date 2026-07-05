# Derived Equations of Motion — 6-DOF R-P-P-R-R-R Arm

Derived with SymPy using Euler–Lagrange (§3 of brief).
Joint order: **[q1=θ1, q2=d1, q3=d2, q4=θ4, q5=θ5, q6=θ6]**

## Potential Energy  V(q)

```
V = 89271*q2/1000 - 50031*sqrt(2)*q3/2000 - 120663*sqrt(2)*sin(q5)*cos(q4)/100000 - 120663*sqrt(2)*cos(q5)/100000 - 923121*sqrt(2)/80000 + 1001601/20000
```

## Gravity Vector  G(q) = ∂V/∂q

```
G[1] = 0
G[2] = 89271/1000
G[3] = -50031*sqrt(2)/2000
G[4] = 120663*sqrt(2)*sin(q4)*sin(q5)/100000
G[5] = 120663*sqrt(2)*sin(q5)/100000 - 120663*sqrt(2)*cos(q4)*cos(q5)/100000
G[6] = 0
```

## Mass Matrix  M(q)  (6×6 symmetric, positive-definite)

```
M[1,1] = M[1,1] = 51*q3**2/20 - 123*q3*sin(q5)*cos(q4)/500 + 123*q3*cos(q5)/500 + 833*q3/400 - 3*sin(q4)**2*sin(q5)**2*sin(q6)**2/25000 + 181099*sin(q4)**2*sin(q5)**2/12000000 + 3*sin(q4)**2*sin(q6)**2/12500 - 553*sin(q4)**2/2400000 + 3*sin(q4)*sin(q5)*sin(q6)*cos(q6)/12500 - 3*sin(q4)*sin(q6)*cos(q4)*cos(q5)*cos(q6)/12500 + 3*sin(q5)*sin(q6)**2*cos(q4)*cos(q5)/12500 - 181099*sin(q5)*cos(q4)*cos(q5)/6000000 - 1107*sin(q5)*cos(q4)/10000 - 3*sin(q6)**2/25000 + 1107*cos(q5)/10000 + 11910211/24000000
M[1,3] = M[3,1] = -123*sqrt(2)*sin(q4)*sin(q5)/1000
M[1,4] = M[4,1] = sqrt(2)*(1476000*q3*sin(q5)*cos(q4) - 1440*sin(q4)*sin(q5)*sin(q6)*cos(q6) + 1440*sin(q5)**2*sin(q6)**2 - 181099*sin(q5)**2 - 1440*sin(q5)*sin(q6)**2*cos(q4)*cos(q5) + 181099*sin(q5)*cos(q4)*cos(q5) + 664200*sin(q5)*cos(q4) - 8197)/12000000
M[1,5] = M[5,1] = 3*sqrt(2)*(82000*q3*sin(q4 - q5) + 82000*q3*sin(q4 + q5) + 20389*sin(q4) + 36900*sin(q4 - q5) + 36900*sin(q4 + q5) - 40*sin(q4 - 2*q6) - 40*sin(q4 + 2*q6) - 20*sin(-q4 + q5 + 2*q6) - 20*sin(q4 - q5 + 2*q6) + 20*sin(q4 + q5 - 2*q6) - 20*sin(q4 + q5 + 2*q6) + 40*cos(q5 - 2*q6) - 40*cos(q5 + 2*q6))/4000000
M[1,6] = M[6,1] = -487*sqrt(2)*(sin(q5)*cos(q4) + cos(q5))/2000000
M[2,2] = M[2,2] = 91/10
M[2,3] = M[3,2] = -51*sqrt(2)/20
M[2,4] = M[4,2] = 123*sqrt(2)*sin(q4)*sin(q5)/1000
M[2,5] = M[5,2] = 123*sqrt(2)*(sin(q5) - cos(q4)*cos(q5))/1000
M[3,3] = M[3,3] = 51/10
M[3,5] = M[5,3] = -123*sin(q5)/500
M[4,4] = M[4,4] = -3*sin(q5)**2*sin(q6)**2/12500 + 181099*sin(q5)**2/6000000 + 8197/6000000
M[4,5] = M[5,4] = -3*cos(q5 - 2*q6)/50000 + 3*cos(q5 + 2*q6)/50000
M[4,6] = M[6,4] = 487*cos(q5)/1000000
M[5,5] = M[5,5] = 3*sin(q6)**2/12500 + 60927/2000000
M[6,6] = M[6,6] = 487/1000000
```

## Equations of Motion

    M(q) q̈ + C(q,q̇) q̇ + G(q) = τ

Substituting τ = M(q)·(q̈_d + Kd·ė + Kp·e + Ki·∫e) + C(q,q̇)·q̇ + G(q)
yields decoupled linear error dynamics: ë + Kd·ė + Kp·e + Ki·∫e = 0 (§5.1)

*(Full C(q,q̇) not printed here — see dynamics.py:_derive_symbolic for detail)*

## Total Mechanical Energy  E = T + V

    T = ½ q̇ᵀ M(q) q̇
    V = Σᵢ mᵢ · g · zᵢ(q)
    E(t) ≈ const  when τ = 0  (energy-conservation check for §9.3)