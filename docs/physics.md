# Physics of a glider winch launch

This document derives the model implemented in `winch_sim/` and explains how it is
solved with [diffrax](https://docs.kidger.site/diffrax/).

The system has three coupled parts:

1. the **glider**, a rigid body moving in the vertical plane (3 degrees of freedom:
   two translations and pitch),
2. the **rope**, 1000–1500 m of steel wire or synthetic (Dyneema) rope with its own
   mass, elasticity and aerodynamic drag, partly lying on the runway at the start,
3. the **winch**, a drum that reels the rope in, driven either by an ideal tension
   controller or a power-limited engine,

plus two "controllers": the **pilot** (elevator) and the **winch driver** (throttle).

```
  z ▲
    │            glider ✈  θ (pitch)
    │                   ○ hook
    │                    ╲
    │                     ╲   rope (sags under its own weight,
    │                       ╲       drag pushes it back)
    │                          ╲
    │                              ╲        β = elevation of the glider
    │                                   ╲       seen from the winch
    │ start                                   ╲___
 ═══╪═══════════════════════════════════════════════▣══  runway, winch drum ▣
    └──────────────────────────────────────────────▶ x
```

## 1. Frames, state and conventions

* Earth frame: $x$ horizontal, from the glider's start towards the winch, $z$ up.
  The runway is $z = 0$, the winch at $\mathbf r_w = (x_w, z_w)$ with
  $x_w \approx$ laid-out rope length and $z_w \approx 1$ m.
* Body frame: $x_b$ forward along the fuselage, $z_b$ towards the canopy.
  Unit vectors in earth coordinates:
  $\hat{\mathbf x}_b = (\cos\theta, \sin\theta)$,
  $\hat{\mathbf z}_b = (-\sin\theta, \cos\theta)$.
* Pitch attitude $\theta$ and pitching moment $M$ positive nose-up. In the plane,
  the moment of a force $\mathbf F$ applied at offset $\mathbf r$ from the CG is
  $M = \mathbf r \times \mathbf F = r_x F_z - r_z F_x$.
* A point with body coordinates $\mathbf b$ sits at
  $\mathbf r_\text{cg} + R(\theta)\mathbf b$ and moves with
  $\mathbf v + q\,\mathbf r^\perp$, where $\mathbf r^\perp = (-r_z, r_x)$.

State vector $y$ (all SI):

| symbol | meaning |
|---|---|
| $\mathbf r = (x, z)$, $\mathbf v$ | glider CG position and velocity (earth frame) |
| $\theta$, $q$ | pitch attitude and pitch rate |
| $\mathbf p_i, \mathbf u_i$, $i = 1..N-1$ | rope node positions and velocities |
| $L_0$ | unstretched length of rope still paid out |
| $v_r$ | reel-in speed at the drum |
| $\delta_e$ | elevator deflection |
| $e_I$ | pilot's integrated attitude error |

With $N = 12$ rope segments that is $6 + 4\cdot 11 + 4 = 54$ states.
(Using earth-frame instead of body-frame translational velocities is equivalent for a
3-DOF model and makes the coupling with the rope nodes simpler.)

## 2. Glider rigid body

$$
m_\text{tot}\,\dot{\mathbf v} = \mathbf F_\text{aero} + \sum_k \mathbf F_{\text{gnd},k}
    + \mathbf F_\text{hook} - m g\,\hat{\mathbf z}
$$
$$
I_{yy}\,\dot q = M_\text{aero} + \sum_k \mathbf r_k \times \mathbf F_{\text{gnd},k}
    + \mathbf r_\text{hook} \times \mathbf F_\text{hook},
\qquad \dot\theta = q, \qquad \dot{\mathbf r} = \mathbf v
$$

$m_\text{tot} = m + m_\text{end}$ includes the rope end assembly carried at the hook
(parachute, strop, weak link, half the top rope segment); its weight is part of
$\mathbf F_\text{hook}$.

### 2.1 Aerodynamics

Air-relative velocity $\mathbf v_a = \mathbf v - \mathbf w(z)$ with a power-law wind
profile $\mathbf w = (-W_{10}\,(z/10)^{1/7},\,0)$ (headwind blows towards $-x$).
Airspeed $V = |\mathbf v_a|$, and

$$
\alpha = \operatorname{atan2}(-\mathbf v_a\cdot\hat{\mathbf z}_b,\ \mathbf v_a\cdot\hat{\mathbf x}_b),
\qquad \bar q = \tfrac12\rho V^2,\qquad \hat q = \frac{q\,\bar c}{2V}.
$$

Coefficients (the fuselage-referenced $C_{L0}$ contains the wing incidence):

$$
C_L = \operatorname{softclip}\big(C_{L0} + C_{L\alpha}\alpha + C_{L\delta}\delta_e + C_{Lq}\hat q,\ C_{L\min},\ C_{L\max}\big)
$$
$$
C_D = C_{D0} + k\,C_L^2 + 1.3\sin^2(\alpha - \alpha_\text{stall})_+ ,\qquad k = \frac1{\pi e A\!R}
$$
$$
C_m = C_{m0} + C_{m\alpha}\alpha + C_{mq}\hat q + C_{m\delta}\delta_e
$$

Lift acts perpendicular to $\mathbf v_a$, drag against it:

$$
\mathbf F_\text{aero} = \bar q S\left(C_L\,\hat{\mathbf e}_v^\perp - C_D\,\hat{\mathbf e}_v\right),
\qquad \hat{\mathbf e}_v = \mathbf v_a / V,\qquad M_\text{aero} = \bar q S \bar c\, C_m.
$$

The soft clip gives a lift plateau at $C_{L\max}$ (stall) while staying differentiable.
Per-glider coefficients are generated from handbook data (`gliders.make_glider`):

* $C_{D0}$ from the best glide ratio: $(L/D)_\max = 1/(2\sqrt{C_{D0}k})$,
* $C_{L\alpha}$ from the Helmbold formula for the aspect ratio, +8 % for the tail,
* $C_{m\alpha} = -C_{L\alpha}\cdot$ static margin (15 % $\bar c$),
* $C_{mq}$, $C_{m\delta}$, $C_{L\delta}$, $C_{Lq}$ from tail volume and tail arm,
* $C_{m0}$ so the glider trims at best-glide $C_L$ with neutral elevator,
* $I_{yy} = m\,k_y^2$ with a radius of gyration of 1.1–1.4 m.

### 2.2 Tow hook

The winch ("CG") hook sits slightly ahead of and below the CG
($\mathbf b_\text{hook} \approx (0.25\text{–}0.35,\ -0.45)$ m). With cable tension
$T$ pulling at angle $\phi$ below the fuselage axis the hook moment is

$$
M_\text{hook} = T\left(|b_z|\cos\phi - b_x\sin\phi\right),
$$

nose-up while the cable is shallow (the well-known pitch-up tendency right after
lift-off) and nose-down once $\tan\phi > |b_z|/b_x$ in the steep part of the climb,
which is why the pilot holds more and more back stick as the launch proceeds.

### 2.3 Ground contact

Nose skid, main wheel and tail wheel are points $\mathbf b_k$ with a smooth
spring–damper that can only push, and regularised Coulomb friction:

$$
N_k = \sigma\!\left(\tfrac{\delta_k}{2\,\text{mm}}\right)\operatorname{softplus}(k_k\delta_k - c_k \dot z_k),
\qquad F_{x,k} = -\mu_k N_k \tanh(\dot x_k / 0.1)
$$

with penetration $\delta_k = -z_k$. This handles standing, the ground roll, lift-off
(and a touchdown, should it happen) with *one* continuous vector field — no mode
switching, which keeps the ODE smooth and makes `jit`/`vmap`/`grad` straightforward.

## 3. The rope: lumped masses with elasticity, weight and drag

Rope weight is large (steel ≈ 0.08 kg/m → ~100 kg of rope for 1200 m, heavier than
the pilot), its drag at 30 m/s is hundreds of newtons, and its stretch (1–2 % at
working load) acts like a bungee. All three shape the tension that actually reaches
the hook, the cable angle at the hook, and the oscillations during the ground run.

### 3.1 Discretisation

The paid-out rope is split into $N$ segments with equal rest length
$\ell_0 = L_0/N$. Node 0 is fixed at the winch, nodes $1..N-1$ are free point masses
$m_n = \mu\,\ell_0$, node $N$ is the hook (a point on the glider).

Reel-in is $\dot L_0 = -v_r$. Because *all* rest lengths and masses shrink uniformly,
the number of states stays fixed, which JAX needs. The price is a small inconsistency
in momentum bookkeeping (mass leaves every node instead of only at the drum); it
vanishes as $N \to \infty$ and the test suite checks that the release height
converges with $N$ (12 vs 24 segments differ by < 2 %).

### 3.2 Segment tension

For segment $j$ from node $a$ to node $b$: $\mathbf d = \mathbf p_b - \mathbf p_a$,
$\ell = |\mathbf d|$, $\hat{\mathbf e} = \mathbf d/\ell$, $\dot\ell = (\mathbf u_b-\mathbf u_a)\cdot\hat{\mathbf e}$.

$$
T_j = \operatorname{softplus}\!\left(E\!A\,\frac{\ell - \ell_0}{\ell_0}
      + c\left(\dot\ell - \ell\,\frac{\dot\ell_0}{\ell_0}\right)\right),
\qquad c = 2\zeta\sqrt{E\!A\,\mu}
$$

* softplus (width 2 N): a rope cannot push; a slack rope carries ~0 N.
* The damping term uses the *strain rate*, so reeling in does not by itself create
  damping forces. With $c = 2\zeta\sqrt{E\!A\,\mu}$ each segment's axial mode has
  damping ratio $\zeta$ independent of $\ell_0$.

### 3.3 Aerodynamic drag (cross-flow principle)

With the segment's mean velocity relative to the wind
$\mathbf v_\text{rel} = \tfrac12(\mathbf u_a + \mathbf u_b) - \mathbf w$,
split into tangential $\mathbf v_t = (\mathbf v_\text{rel}\cdot\hat{\mathbf e})\hat{\mathbf e}$
and normal $\mathbf v_n = \mathbf v_\text{rel} - \mathbf v_t$ parts:

$$
\mathbf F_{\text{drag},j} = -\tfrac12\rho\, d\, \ell\left(C_{Dn}|\mathbf v_n|\mathbf v_n
    + \pi C_f |\mathbf v_t|\mathbf v_t\right)
$$

($C_{Dn} \approx 1.2$ for a cylinder, $C_f \approx 0.01$–$0.02$). Half goes to each
end of the segment. Near the top of the launch the upper rope swings round the winch
at high speed, so this term grows as the launch progresses.

### 3.4 Node equations

$$
m_n\,\dot{\mathbf u}_i = T_{i+1}\hat{\mathbf e}_{i+1} - T_i\hat{\mathbf e}_i
   + \tfrac12(\mathbf F_{\text{drag},i} + \mathbf F_{\text{drag},i+1})
   - m_n g\,\hat{\mathbf z} + \mathbf F_{\text{gnd},i},
\qquad \dot{\mathbf p}_i = \mathbf u_i
$$

Ground contact of the rope lying on the grass uses the same spring–damper/friction law
as the wheels ($\mu \approx 0.4$–$0.5$), with the stiffness scaled to the node mass.
Dragging 100 kg of steel rope over grass costs ~500 N during the ground run.

### 3.5 Force on the glider

$$
\mathbf F_\text{hook} = -T_N\hat{\mathbf e}_N + \tfrac12\mathbf F_{\text{drag},N}
  - \tfrac12\rho\,(C_DA)_\text{chute}|\mathbf v_{a,h}|\mathbf v_{a,h} - m_\text{end}\,g\,\hat{\mathbf z}
$$

The **local** rope direction at the hook, $-\hat{\mathbf e}_N$, is steeper than the
straight line to the winch because of sag; it is this angle that triggers the
back-release and that the pilot sees.

### 3.6 Validation: elastic catenary

For a rope hanging still between two points, the exact solution is the elastic
catenary (Irvine 1981), parametrised by unstretched arc length $s$ with constant
horizontal tension $H$ and vertical tension $V(s) = V_0 + w s$:

$$
x(s) = \frac{Hs}{E\!A} + \frac Hw\left[\sinh^{-1}\frac{V}{H} - \sinh^{-1}\frac{V_0}{H}\right],\quad
z(s) = \frac{ws^2/2 + V_0 s}{E\!A} + \frac Hw\left[\sqrt{1+\tfrac{V^2}{H^2}} - \sqrt{1+\tfrac{V_0^2}{H^2}}\right]
$$

`cable.elastic_catenary` solves for $V_0$ and the unstretched length, and the test
suite checks that the lumped rope ($N = 20$) settles onto it within 1 % of the sag.
A consequence visible in the launch results: in the climb the tension at the hook
exceeds the tension at the winch by roughly $\mu g\,\Delta z$ (≈ 300 N for steel at
400 m).

### 3.7 Rope presets

| | diameter | $\mu$ | $E\!A$ | $C_{Dn}$ | $C_f$ |
|---|---|---|---|---|---|
| steel wire | 4.5 mm | 0.080 kg/m | 1.0 MN | 1.2 | 0.02 |
| Dyneema | 5.0 mm | 0.016 kg/m | 0.6 MN | 1.2 | 0.01 |

## 4. Winch

Both winch models share one equation for the drum, lumped as an effective mass
$M_\text{eff} = J/r_d^2$ at the rope:

$$
M_\text{eff}\,\dot v_r = \tau(t, \beta)\,F_\text{avail}(v_r) - T_1 - b\,v_r,
\qquad
F_\text{avail}(v) = \frac{P_\max}{\sqrt{v^2 + (P_\max/F_\max)^2}}
$$

$F_\text{avail}$ is a smooth $\min(F_\max, P_\max/v)$:

* **tension-controlled winch**: $P_\max \to \infty$, so the winch pulls with a
  prescribed tension $F_\max = k_T\,m g$ (default $k_T = 1.1$) — isolates the
  glider's behaviour;
* **engine winch**: finite power (default 200 kW, 12 kN), so the pull *drops* as the
  reel speeds up and *rises* when the glider slows the rope down (e.g. during
  rotation) — a real mechanism behind weak-link failures of light gliders.

The driver's throttle $\tau$ ramps up over 3 s after $t = 1$ s and fades to 40 %
as the glider elevation seen from the winch, $\beta = \operatorname{atan2}(z - z_w, x_w - x)$,
goes from 50° to 70°. For the engine winch the driver sets the throttle so that the
pull at 15 m/s reel speed is $k_T\,mg$. After release the drum is braked.

## 5. Pilot

The pilot flies pitch attitude. The reference is a smooth function of the state
(again no discrete phases):

$$
\theta_\text{ref} = (1 - s_\text{top})\Big[\theta_g + s_\text{rot}(h)\,(\theta_\text{climb} - \theta_g)
  + K_V\,\text{clip}(V - V_\text{target})\Big] + s_\text{top}(\beta)\,\theta_\text{top}
$$

* ground roll: rest attitude + 3° of back stick ($\theta_g$),
* rotation into the climb between 2 and 30 m height ($s_\text{rot}$, smoothstep),
* climb attitude 35–40°, raised when faster than the target speed, lowered when slower,
* nose lowered to 5° as $\beta$ goes from 55° to 70° ($s_\text{top}$),
* after release: glide attitude 0°.

Elevator: PID with saturation and a first-order stick lag,

$$
\delta_{e,\text{cmd}} = \delta_\max\tanh\!\frac{-(K_p e + K_i e_I) + K_d q}{\delta_\max},\quad
\dot\delta_e = \frac{\delta_{e,\text{cmd}} - \delta_e}{\tau_p},\quad
\dot e_I = a(h)\,e - (1-a(h))\,e_I
$$

with $e = \theta_\text{ref} - \theta$ and $a(h)$ a smooth "airborne" factor that stops
integrator wind-up while the tail is on the ground.

## 6. End of the launch — events

The launch ends at the first of:

| event | condition |
|---|---|
| `release` | local cable angle at the hook reaches 72° below horizontal (pilot pulls the release) |
| `back_release` | cable more than 110° below the fuselage axis (hook back-release) |
| `weak_link` | hook tension exceeds the glider's weak-link rating |
| `rope_in` | less than 30 m of rope left out |

## 7. Solving it with diffrax

* **Pure-JAX vector field** `dynamics.vector_field(t, y, args)`. `y` is an
  `equinox.Module` pytree (`State`), `args` holds all parameters as `equinox.Module`s
  with float leaves, so the whole solve can be `jit`-compiled, `vmap`-ed over gliders,
  ropes or winch settings, and differentiated (e.g. release height w.r.t. winch
  tension).
* **Smoothness by construction.** Every kink (rope slack, ground contact, friction,
  stall, pilot phases) is replaced by softplus / sigmoid / tanh / smoothstep, so an
  adaptive explicit solver never has to stop at a discontinuity.
* **Solver.** `Tsit5` with a `PIDController(rtol=atol=1e-6)`. The stiffest mode is the
  axial rope wave, $\omega \approx \sqrt{E\!A/\mu}/\ell_0$ (≈ 50 rad/s at the start,
  a few hundred near the top), which is resolved in ~2–3k steps. `Kvaerno5` (implicit)
  is available via `solver="kvaerno5"`; it gives the same answer but takes more steps,
  i.e. the problem is not stiff enough to benefit.
* **Events.** `diffrax.Event` with the four condition functions above
  (sign change, `direction=True`) and an `optimistix.Newton` root finder to locate
  the release instant exactly. `sol.event_mask` says which one fired.
* **Saving.** `SaveAt(subs=[SubSaveAt(ts=grid), SubSaveAt(t1=True)])` — the grid for
  plotting, `t1` for the exact state at release (with events, a plain `ts`+`t1`
  save puts the final point into the next free grid slot).
* **Two stages.** A second `diffeqsolve` from the release state with
  `attached = 0` continues the glider in free flight (push-over, pick-up of speed)
  while the rope falls.
* **Derived quantities** (tensions, angles, load factor, power) are recomputed from
  saved states with the same `evaluate` function, `vmap`-ed over time, so plots can
  never disagree with the dynamics.
* **Batching.** `simulate.solve_batch(launches)` stacks the parameter pytrees and
  `vmap`s the solve: all five catalogue gliders run as one compiled program in
  ~0.5 s after a few seconds of compilation.

## 8. Example results

1200 m rope, no wind, winch pull 1.1 × glider weight
(`uv run main.py [--rope steel] [--winch engine]`):

| glider | mass | $V_W$ | rope / winch | release height | time | ground roll | max V | max $T_\text{hook}$ | max L/W |
|---|---|---|---|---|---|---|---|---|---|
| Ka 8 | 290 kg | 100 km/h | Dyneema / tension | 454 m | 50 s | 46 m | **128 km/h** | 3.0 kN | 2.6 |
| LS4 | 360 kg | 130 km/h | Dyneema / tension | 456 m | 42 s | 69 m | **150 km/h** | 4.1 kN | 3.0 |
| ASK 21 | 470 kg | 150 km/h | Dyneema / tension | 481 m | 45 s | 52 m | **151 km/h** | 5.4 kN | 3.2 |
| ASG 29 | 420 kg | 150 km/h | Dyneema / tension | 465 m | 39 s | 75 m | **157 km/h** | 4.8 kN | 3.0 |
| DG-1000 | 620 kg | 150 km/h | Dyneema / tension | 493 m | 40 s | 64 m | **165 km/h** | 7.0 kN | 3.3 |
| Ka 8 | | | steel / tension | 434 m | 50 s | 57 m | 121 km/h | 3.3 kN | 2.5 |
| ASK 21 | | | steel / tension | 468 m | 45 s | 59 m | 146 km/h | 5.7 kN | 3.1 |
| DG-1000 | | | steel / tension | 484 m | 40 s | 70 m | 161 km/h | 7.3 kN | 3.2 |
| Ka 8 | | | Dyneema / engine | 444 m | 51 s | 70 m | 107 km/h | 3.7 kN | 2.2 |
| ASK 21 | | | Dyneema / engine | 452 m | 46 s | 65 m | 122 km/h | 6.0 kN | 2.6 |
| DG-1000 | | | Dyneema / engine | 449 m | 42 s | 81 m | 132 km/h | 7.9 kN | 2.5 |

Observations:

* Release heights of ~40 % of the rope length, 40–50 s launches and climb speeds of
  90–110 km/h are in line with typical club winch launches.
* Steel costs 10–20 m of height and lengthens the ground run versus Dyneema.
* A constant tension (tension winch) accelerates the glider hard in the ground run
  and initial climb; the speed peak just before full rotation exceeds the placarded
  winch speed $V_W$ for every glider (bold; barely for the ASK 21, by 28 % for the
  Ka 8). A real driver would use a lower setting for the Ka 8 and ramp more gently. The power-limited engine winch is
  self-limiting (pull falls with reel speed): lower peak speeds and load factors, at
  the cost of 20–50 m of release height.
* The peak load factor occurs during the rotation, which the simple pilot flies
  somewhat aggressively.

## 9. Limitations and possible extensions

* Pilot and winch driver are simple smooth control laws, not human models. Their
  parameters (rotation height, climb attitude, tension profile) dominate the results
  as much as the glider's aerodynamics do — just as in real life.
* The glider data are approximate estimates, not manufacturer data.
* No ground effect, no flaps (flap settings matter for the ASG 29/LS-type gliders),
  no wing drop / lateral dynamics, no cable parachute opening after release.
* Rope reel-in removes mass uniformly from all nodes (see 3.1). A "moving-node"
  formulation that removes mass only at the drum would be exact.
* Winch drum physics is intentionally minimal (one effective mass).
* Differentiability is in place but not yet used; natural next steps are optimising
  the tension profile or pilot schedule for release height subject to
  $V \le V_W$, $T \le$ weak link and $n \le n_\max$ with `jax.grad`.
