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
True airspeed $V = |\mathbf v_a|$, and

$$
\alpha = \operatorname{atan2}(-\mathbf v_a\cdot\hat{\mathbf z}_b,\ \mathbf v_a\cdot\hat{\mathbf x}_b),
\qquad \bar q = \tfrac12\rho(z) V^2,\qquad \hat q = \frac{q\,\bar c}{2V}.
$$

**Air density** follows the International Standard Atmosphere (`atmosphere.py`)
at the pressure height $h = h_\text{field} + z$, with a temperature offset
$\Delta T$ (`Env.field_elevation`, `Env.isa_dT`, CLI `--field-elevation`,
`--isa-dt`):

$$
T = T_0 - \lambda h + \Delta T,\qquad p = p_0\Big(\frac{T_0 - \lambda h}{T_0}\Big)^{g_0/(R\lambda)},
\qquad \rho = \frac{p}{R\,T}
$$

($T_0$ = 288.15 K, $p_0$ = 101325 Pa, $\lambda$ = 6.5 K/km; isothermal above
11 km). The same $\rho(z)$ is used for the glider, for every rope segment (at
its mid-height) and for the parachute. At 3000 m, $\rho$ is 26 % lower than at
sea level.

**Indicated vs true airspeed.** Instruments, placards and pilots use indicated
airspeed. Neglecting instrument error and compressibility, it equals the
equivalent airspeed:

$$
V_\text{IAS} = V\sqrt{\rho(z)/\rho_0},\qquad \rho_0 = 1.225\ \text{kg/m}^3 .
$$

Stall and the aerodynamic loads depend on $\bar q$, i.e. on IAS. The pilot's
target speed and the winch speed limit $V_W$ are therefore IAS values, while the
kinematics use TAS. Summaries report max/min IAS (against $V_W$ and stall) and
max TAS. On a hot, high airfield the glider needs more TAS, i.e. a longer ground
roll, for the same IAS.

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

The resolution is set in **segments per km** of laid-out rope
(`--segments-per-km`, default 10, i.e. 100 m segments, minimum 4 segments), so
short and long ropes are discretised alike. Too coarse a rope is badly wrong for
long ropes: 50 km of Dyneema gave ~3000 m release height at 20 segments but
2170 m at 500 segments (10/km).

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
catenary (Irvine 1981, ref. 11 in §10), parametrised by unstretched arc length $s$ with constant
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

### 3.7 Rope presets and rope input

| | diameter | $\mu$ | $E\!A$ | breaking load | $C_{Dn}$ | $C_f$ |
|---|---|---|---|---|---|---|
| steel wire | 4.5 mm | 0.080 kg/m | 1.0 MN | 17 kN | 1.2 | 0.02 |
| Dyneema | 5.0 mm | 0.016 kg/m | 0.6 MN | 27 kN | 1.2 | 0.01 |

These are generic, approximate values. A specific rope can be given in three ways:

* `cable.rope_from_datasheet(diameter_mm, mass_kg_per_100m, breaking_load_kN,
  elongation_at_break_pct | EA_N, CDn, Cf, ground_mu, end_mass_kg, end_CdA_m2, ...)`.
  Without `EA_N`, the stiffness is the secant value
  $E\!A = F_\text{break}/\varepsilon_\text{break}$. Synthetic ropes are stiffer
  at working loads, so give `EA_N` from the load–elongation curve at 20–30 % of
  the breaking load when you have it.
* A TOML file with the same keys (`--rope-file ropes/example_dyneema_6mm.toml`), or
  `base = "steel"` plus SI overrides of `Rope` fields
  (`ropes/example_steel_override.toml`).
* Single-field SI overrides on the command line: `--rope-param mu=0.09`.

The breaking load does not affect the dynamics. It is used to report the rope safety
factor $F_\text{break}/\max T$ in the summary table.

### 3.8 Long ropes: why the hook tension does not grow

Rope weight enters the tension balance through height, not length. Along a
hanging rope, $dT/dz = \mu g$ (plus drag), so the hook feels at most the tension
at the rope's lowest point plus $\mu g\,\Delta z$ (0.16 N/m × 3000 m ≈ 0.5 kN
for Dyneema). The remaining rope weight rests on the winch end, the sag, or the
ground. What a long rope does is **eat** tension between winch and hook: friction
of the part still lying on the grass ($\mu_g\, m_\text{rope} g$) and drag of
the long airborne part. ASK 21, engine winch, 10 segments/km:

| rope | length | rope mass | release height | max T winch | max T hook | rope on ground at 60 % of height |
|---|---|---|---|---|---|---|
| Dyneema | 1.2 km | 19 kg | 450 m | 6.2 kN | 6.1 kN | 0 % |
| Dyneema | 10 km | 160 kg | 2353 m | 5.6 kN | 4.8 kN | 5 % |
| Dyneema | 50 km | 800 kg | 2169 m | 5.6 kN | 2.9 kN | 81 % |
| steel | 10 km | 800 kg | 1500 m | 8.6 kN | 6.0 kN | 29 % |
| steel | 50 km | 4000 kg | no lift-off | 6.9 kN | 0.2 kN | 100 % |

Beyond some length, extra rope only adds drag and friction: the 50 km Dyneema
launch goes *lower* than the 10 km one. The 50 km steel rope (4 t, ~20 kN of
friction on grass) cannot be dragged by this winch at all.

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
* climb attitude 35–40°, raised when faster than the target speed (IAS), lowered when slower,
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
| `no_liftoff` | still on the ground after 120 s (e.g. rope too heavy to drag); a failed launch, no sensitivities |

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
* **Saving (two passes).** Pass 1 runs to the event with `SaveAt(t1=True)` only,
  so the time limit `t_max` (default 10⁵ s, `--t-max`) costs nothing, and
  `max_steps` is 5·10⁶. Pass 2 repeats the deterministic solve from 0 to the
  release time and saves 2000 evenly spaced points for plotting and diagnostics.
  Long and short launches get the same number of plot points.
* **Two stages.** A second `diffeqsolve` from the release state with
  `attached = 0` continues the glider in free flight (push-over, pick-up of speed)
  while the rope falls.
* **Derived quantities** (tensions, angles, load factor, power) are recomputed from
  saved states with the same `evaluate` function, `vmap`-ed over time, so plots can
  never disagree with the dynamics.
* **Gradients.** `sensitivity.release_height` is the same stage-1 solve with
  `SaveAt(t1=True)` and `RecursiveCheckpointAdjoint`. `eqx.filter_value_and_grad`
  over the whole `Launch` pytree gives every parameter's sensitivity at once (§8b).
* **Batching.** `simulate.solve_batch(launches)` stacks the parameter pytrees and
  `vmap`s the solve: all five catalogue gliders run as one compiled program in
  ~0.5 s after a few seconds of compilation.

## 8. Example results

1200 m rope, no wind, winch pull 1.1 × glider weight
(`uv run main.py [--rope steel] [--winch engine]`):

| glider | mass | $V_W$ | rope / winch | release height | time | ground roll | max V | max $T_\text{hook}$ | max L/W |
|---|---|---|---|---|---|---|---|---|---|
| Ka 8 | 290 kg | 100 km/h | Dyneema / tension | 454 m | 50 s | 46 m | **128 km/h** | 3.0 kN | 2.6 |
| ASK 13 | 460 kg | 120 km/h | Dyneema / tension | 470 m | 43 s | 52 m | **149 km/h** | 5.0 kN | 2.8 |
| LS4 | 360 kg | 130 km/h | Dyneema / tension | 456 m | 42 s | 69 m | **150 km/h** | 4.1 kN | 3.0 |
| ASK 21 | 470 kg | 150 km/h | Dyneema / tension | 481 m | 45 s | 52 m | **151 km/h** | 5.4 kN | 3.2 |
| ASG 29 | 420 kg | 150 km/h | Dyneema / tension | 465 m | 39 s | 75 m | **157 km/h** | 4.8 kN | 3.0 |
| DG-1000 | 620 kg | 150 km/h | Dyneema / tension | 493 m | 40 s | 64 m | **165 km/h** | 7.0 kN | 3.3 |
| Ka 8 | | | steel / tension | 434 m | 50 s | 57 m | 121 km/h | 3.3 kN | 2.5 |
| ASK 13 | | | steel / tension | 459 m | 44 s | 58 m | **144 km/h** | 5.2 kN | 2.7 |
| ASK 21 | | | steel / tension | 468 m | 45 s | 59 m | 146 km/h | 5.7 kN | 3.1 |
| DG-1000 | | | steel / tension | 484 m | 40 s | 70 m | 161 km/h | 7.3 kN | 3.2 |
| Ka 8 | | | Dyneema / engine | 444 m | 51 s | 70 m | 107 km/h | 3.7 kN | 2.2 |
| ASK 13 | | | Dyneema / engine | 441 m | 45 s | 69 m | 119 km/h | 5.9 kN | 2.3 |
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

## 8b. Sensitivities of the release height

Every `uv run main.py` run (unless `--no-sensitivity`; `--only rope` filters the table) computes $\partial h/\partial p$ for
all 80 model parameters (glider, rope, winch, pilot, environment, layout) in one
reverse-mode pass of `jax.grad` through the diffrax solve
(`sensitivity.py`, `RecursiveCheckpointAdjoint`). The release time is found by
the event root finder, and its dependence on the parameters is differentiated
implicitly. The gradients agree with central finite differences to < 0.5 %
(`tests/test_sensitivity.py`). Tiny sensitivities are only as accurate as the
solver tolerance, so the default is rtol = atol = 1e-8. A run takes ~10 s.

For each parameter the table gives its value and unit, $\partial h/\partial p$
in m per unit (angles per degree), the height change for a +10 % change, and
the elasticity $(p/h)\,\partial h/\partial p$ (% height per % parameter).
Everything else is held fixed: e.g. the winch pull, preset as 1.1 × weight, does
not follow a change of glider mass. These are local linearisations. The full
table goes to `results/sensitivity_<glider>_<rope>_<winch>.csv`.

ASK 21, 1200 m Dyneema, tension winch (h = 481 m), largest effects:

| parameter | value | ∂h/∂p | Δh for +10 % | elasticity |
|---|---|---|---|---|
| rope length | 1200 m | 0.380 m/m | +45.6 m | 0.95 |
| winch pull `F_max` | 5072 N | 0.0488 m/N | +24.7 m | 0.51 |
| glider mass | 470 kg | −0.464 m/kg | −21.8 m | −0.45 |
| winch fade start `fade_beta0` | 50° | 0.75 m/deg | +3.8 m | 0.08 |
| wing area | 17.95 m² | 1.92 m/m² | +3.4 m | 0.07 |
| rope diameter | 5 mm | −6.6 m/mm | −3.3 m | −0.07 |
| rope normal drag coefficient `CDn` | 1.2 | −27 m | −3.2 m | −0.07 |
| pilot target speed | 105 km/h | −1.08 m/(m/s) | −3.1 m | −0.07 |
| release cable angle | 72° | 0.41 m/deg | +3.0 m | 0.06 |
| climb attitude | 40° | 0.63 m/deg | +2.5 m | 0.05 |

Rope characteristics for the steel rope and the ASK 13 (`--only rope`): diameter
−2.8 m, `CDn` −2.7 m, mass per length −1.7 m, parachute drag area −0.4 m,
ground friction −0.4 m per +10 %. The axial stiffness EA has essentially **no**
influence on release height (< 1 cm for +10 %). Rope stretch shapes the
tension oscillations in the ground run (§3), not the energy that reaches the
glider. Rope drag (diameter × `CDn`) matters about twice as much as rope weight.

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
* The sensitivities (§8b) are local. For large changes (e.g. steel → Dyneema)
  run the simulation directly. The same gradients could also drive an
  optimisation of the tension profile or pilot schedule, subject to
  $V \le V_W$, $T \le$ weak link and $n \le n_\max$.

## 10. Related literature

**Winch-launch simulation**

1. A. Gäb, C. Santel, "Numerical Simulation of Glider Winch Launches",
   *Technical Soaring* 35(3), 2011.
   [ts.ostiv.org/index.php/ts/article/view/76](https://ts.ostiv.org/index.php/ts/article/view/76)
   ([PDF](https://ts.ostiv.org/index.php/ts/article/download/76/69)).
   RWTH Aachen, Matlab/Simulink. Models aircraft, pilot, winch, winch operator,
   cable, atmosphere and terrain. The cable is mass points joined by spring–damper
   links with drag, weight and ground reaction, the same approach as §3 here. Pilot
   and winch operator are PID controllers with reaction time and neuromuscular delay.
   Analyses: a reference launch, wind, and overly steep initial climbs. **The best
   reference to validate this model against.**
2. "Numeric Simulation of a Glider Winch Launch", study thesis, Chair of Flight
   Dynamics, RWTH Aachen, 2008.
   [publications.rwth-aachen.de/record/230129](http://publications.rwth-aachen.de/record/230129/files/3265.pdf)
   ([CORE](https://core.ac.uk/download/pdf/36588266.pdf)). The origin of (1).
   Findings: opening the throttle quickly destabilises the phugoid; the tow-hook
   position can be optimised; release height depends on cable type, tow distance,
   maximum winch force and especially cable drag. Lists ground effect and
   ground/cable friction as missing.
3. C. Santel, "An investigation of glider winch launch accidents utilizing
   multipoint aerodynamics models in flight simulation", diploma thesis,
   RWTH Aachen (cited in 1).
4. L. Bogan, "Glider Winch Launch Simulation" (web page, not peer-reviewed).
   [bogan.ca/soaring/winch/winch.html](https://bogan.ca/soaring/winch/winch.html).
   Prescribed flight path, MathCAD. Gives a good intuition for cable drag growing
   as the rope turns broadside.

**Optimal launch trajectories**

5. "Maximum Altitude Sailplane Winch-Launch Trajectories", *Aeronautical Quarterly*
   28(2), pp. 75–84, 1977.
   [doi:10.1017/S0001925900007976](https://doi.org/10.1017/S0001925900007976).
   With limits on C_L and cable acceleration, the optimal trajectory is
   climb–dive–climb; adding a reel-in speed limit makes it more realistic. The
   benchmark for a `jax.grad`-based optimisation (§9).
6. R. Eppler, "Windenschlepp und optimale Ausklinkhöhe" (winch launch and optimal
   release height), cited in (1).
7. Pierson and Chen, follow-up work on optimal sailplane trajectories,
   *Journal of Aircraft* (1979), *Optimal Control Applications and Methods* (1980).

**Safety and winch engineering**

8. H. Browning, "Boundaries of Safe Winch Launching", *Technical Soaring* 31(4),
   p. 95, 2007; T. Hills, "Safety Analysis of the Winch Launch", ibid., p. 101.
   [Index](https://soaringweb.org/Soaring_Index/Technical_Soaring/Technical_Soaring_issue.html).
9. "The Design and Development of Glider Launching Winches", *Technical Soaring*.
   [ts.ostiv.org/index.php/ts/article/view/942](https://ts.ostiv.org/index.php/ts/article/view/942/0).
   Design formulae for winch hardware.
10. B. S. Smith, Soaring Safety Foundation:
    [Winches](https://www.soaringsafety.org/publications/winches.pdf),
    [Winch launching revisited](https://www.soaringsafety.org/publications/Winch-launching-revisited.pdf).

**Cable / tether modelling**

11. H. M. Irvine, *Cable Structures*, MIT Press, 1981. The elastic catenary used
    for validation in §3.6.
12. Williams, Lansdorp, Ockels, "Modeling and control of a kite on a
    variable length flexible inelastic tether", AIAA-2007-6705 (cited in 1).
    Variable-length tether modelling, cf. the reel-in approach in §3.1.
13. "A quaternion-based model for optimal control of the SkySails airborne wind
    energy system", [arXiv:1508.05494](https://arxiv.org/abs/1508.05494).
    Optimal control with tether force and reel-out speed as variables.

**How this model compares with (1)**: the structure is the same (lumped-mass cable,
PID pilot, tension-controlled winch operator). Differences: (1) includes human
reaction and neuromuscular delays and a more detailed winch/operator model. This
model uses a fixed-size reel-in discretisation, smooth contact models (one
continuous ODE instead of mode switching), and JAX/diffrax for batching and
gradients. Obvious next steps: reproduce the reference launch from (1), and add
reaction delays to the pilot and winch driver.
