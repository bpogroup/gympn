# Theoretical analysis — draft propositions (ccf / s-ccf / cfpk)

*Draft for §5 of the EJOR paper (v2 outline, see `EJOR_PAPER_OUTLINE.md`).
Notation is LaTeX-ready ($...$). Four results, spanning both provenance-
exploiting mechanisms: (P1) ccf is an unbiased policy-gradient estimator
under the realized-component assumption (A1); a companion proposition
upgrades this to s-ccf, unbiased unconditionally via the static/topological
partition (A1$'$); (P2) ccf/s-ccf's per-decision advantage variance is
reduced by a factor equal to the number of independent causal components;
**(P3, NEW) `cfpk`'s coupling truncation is exact** — reusing one simulated
continuation for both branches of a fork after they provably reconverge to
the same state changes neither the estimator's target nor its unbiasedness,
and eliminates the reconverged tail's variance contribution entirely (not
merely reduces it). Assumptions are stated explicitly — P1's (A1) and P3's
(A3) are the ones a reviewer will scrutinize, so both are justified from
A-E PN / SMDP semantics rather than asserted.*

---

## Setup and notation

An A-E PN under policy $\pi_\theta$ induces a semi-Markov decision process. One
episode $\tau \sim \pi_\theta$ yields:

- **Decisions** $d = 1,\dots,D$: at decision $d$ the pre-action state is $s_d$,
  the chosen action is $a_d \sim \pi_\theta(\cdot\mid s_d)$, and $u_d$ is its
  decision clock (SMDP time). Let $\mathcal F_d$ denote the history up to but
  excluding $a_d$ (so $s_d$ and everything before are $\mathcal F_d$-measurable).
- **Rewards** $j$: scalar $r_j$ fired at time $t_j$.
- **Provenance DAG:** every token records the firing that produced it and the
  input tokens consumed; every reward $r_j$ is a deterministic function of the
  tokens consumed by its firing. For a reward $j$, let $\mathcal A(j)\subseteq\{1,\dots,D\}$
  be the set of decisions on its causal lineage (the backward reachable
  decisions in the DAG).

**Causal components.** Build an undirected graph on decisions with an edge
$(d,d')$ whenever $\exists j:\{d,d'\}\subseteq\mathcal A(j)$ (two decisions that
jointly caused a common reward). Its connected components partition the decisions
into **causal components** $C_1,\dots,C_M$; write $c(d)$ for $d$'s component.
Because all ancestors of a reward are unioned, each reward with $\mathcal A(j)\neq\varnothing$
has a well-defined component $c(j):=c(d)$ for any $d\in\mathcal A(j)$. (Rewards
with $\mathcal A(j)=\varnothing$ are policy-independent constants and drop out of
the gradient; we ignore them.) This is exactly the union-find partition computed
by `_redistribute_ccf`.

**Discounting.** SMDP discount from decision $d$: $\rho_d(t)=e^{-\beta\max(0,\,t-u_d)}$.

**The two credit signals.** For decision $d$:

$$
G_d^{\text{full}} \;=\!\! \sum_{j:\,t_j\ge u_d}\!\! \rho_d(t_j)\,r_j
\qquad\text{(full reward-to-go = }mc\_q\text{ = PPO)},
$$

$$
G_d^{\text{ccf}} \;=\!\! \sum_{\substack{j:\,t_j\ge u_d\\ c(j)=c(d)}}\!\! \rho_d(t_j)\,r_j
\qquad\text{(own-component reward-to-go = }ccf\text{)}.
$$

Their difference is the **cross-component future reward**

$$
\Delta_d \;=\; G_d^{\text{full}}-G_d^{\text{ccf}}
\;=\!\! \sum_{\substack{j:\,t_j\ge u_d\\ c(j)\ne c(d)}}\!\! \rho_d(t_j)\,r_j .
$$

The estimators use a learned critic baseline $V(s_d)$:

$$
\hat g^{\bullet}=\sum_{d=1}^{D}\nabla_\theta\log\pi_\theta(a_d\mid s_d)\,
\big(G_d^{\bullet}-V(s_d)\big),\qquad \bullet\in\{\text{full},\text{ccf}\}.
$$

We take as given the standard result that the full reward-to-go estimator is
unbiased, $\mathbb E[\hat g^{\text{full}}]=\nabla_\theta J(\theta)$ (policy-
gradient theorem, causality/GPOMDP form). All claims below are *relative* to it.

---

## Assumption

> **(A1) Component mean-exogeneity.** For every decision $d$,
> $$\mathbb E\big[\Delta_d \mid \mathcal F_d,\,a_d\big]\;=\;\mathbb E\big[\Delta_d \mid \mathcal F_d\big].$$
> That is, given the pre-action history, the *expected* cross-component future
> reward does not depend on which action is taken at $d$.

**Why A1 holds in an A-E PN (justification, not a further assumption).** A reward
$r_j$ is a deterministic function of the tokens consumed at its firing, and each
such token's entire history is recorded in the provenance DAG; hence $r_j$ is
causally influenced by $a_d$ **only if** some descendant of $a_d$ lies on $j$'s
lineage, i.e. **only if** $d\in\mathcal A(j)$, i.e. **only if** $c(j)=c(d)$.
Cross-component rewards ($c(j)\ne c(d)$) therefore have no causal path from
$a_d$: neither their values nor — since a lineage link to $d$ would itself place
them in $c(d)$ — their component membership can be altered by $a_d$. The only
residual channel is the *exogenous randomness* driving cross-component events
(arrival types, service times); A1 asserts these draws are, in conditional mean,
independent of $a_d$ given $\mathcal F_d$. This holds under the standard
simulation model of independent per-firing draws (Remark R3 discusses the shared-
RNG realization subtlety, which affects realized values but not the conditional
mean). Thus A1 is a property of A-E PN semantics plus exogenous-noise
independence, not an assumption about problem-specific structure.

---

## Proposition 1 (Unbiasedness)

> Under (A1), $\;\mathbb E[\hat g^{\text{ccf}}]=\mathbb E[\hat g^{\text{full}}]=\nabla_\theta J(\theta)$.
> The ccf estimator is unbiased.

**Proof.** The two estimators differ only through $\Delta_d$:
$$
\hat g^{\text{full}}-\hat g^{\text{ccf}}
=\sum_{d=1}^{D}\nabla_\theta\log\pi_\theta(a_d\mid s_d)\,\Delta_d .
$$
Fix $d$ and condition on $\mathcal F_d$. Since $a_d\sim\pi_\theta(\cdot\mid s_d)$
is drawn given $\mathcal F_d$ and $\Delta_d$ is realized afterwards,
$$
\mathbb E\!\left[\nabla\log\pi_\theta(a_d\mid s_d)\,\Delta_d \mid \mathcal F_d\right]
=\mathbb E_{a_d}\!\left[\nabla\log\pi_\theta(a_d\mid s_d)\;
\mathbb E[\Delta_d\mid \mathcal F_d,a_d]\right].
$$
By (A1) the inner expectation equals $\bar\Delta_d:=\mathbb E[\Delta_d\mid\mathcal F_d]$,
a quantity that does not depend on $a_d$. Hence
$$
=\bar\Delta_d\;\mathbb E_{a_d}\!\left[\nabla\log\pi_\theta(a_d\mid s_d)\mid\mathcal F_d\right]
=\bar\Delta_d\sum_{a}\nabla_\theta\pi_\theta(a\mid s_d)
=\bar\Delta_d\,\nabla_\theta\!\!\sum_{a}\pi_\theta(a\mid s_d)
=\bar\Delta_d\,\nabla_\theta 1=0,
$$
using $\nabla\log\pi=\nabla\pi/\pi$ and $\sum_a\pi_\theta(a\mid s_d)=1$. Summing
over $d$ and taking total expectation gives $\mathbb E[\hat g^{\text{full}}-\hat g^{\text{ccf}}]=0$.
Combined with $\mathbb E[\hat g^{\text{full}}]=\nabla_\theta J$, the claim
follows. $\qquad\blacksquare$

**Reading.** Cross-component future rewards act as an (action-mean-independent)
*control variate*: subtracting them removes variance without moving the expected
gradient — a valid policy-gradient baseline, but one derived *per decision from
the causal structure* rather than a single learned scalar.

---

## Proposition 2 (Variance reduction)

Idealized model isolating the mechanism. At decision $d$, suppose the future
reward-to-go splits across $K$ causal components,
$$
G_d^{\text{full}}=\sum_{k=1}^{K} R_k,\qquad d\in\text{component }1,\quad G_d^{\text{ccf}}=R_1,
$$
where $R_k$ is component $k$'s discounted reward-to-go, and assume, conditional on
$s_d$, the $R_k$ are **mutually independent** with variances $\sigma_k^2$.

> Let the critic be the conditional-mean baseline $V(s_d)=\mathbb E[G_d^{\text{full}}\mid s_d]$.
> Then the per-decision advantage variances satisfy
> $$
> \operatorname{Var}\!\big(G_d^{\text{full}}-V(s_d)\mid s_d\big)=\sum_{k=1}^{K}\sigma_k^2,
> \qquad
> \operatorname{Var}\!\big(G_d^{\text{ccf}}-V(s_d)\mid s_d\big)=\sigma_1^2 .
> $$
> In particular, if components have comparable variance $\sigma_k^2\approx\sigma^2$,
> ccf reduces the per-decision advantage variance by a factor $\approx K$, and the
> gradient-noise standard deviation by $\approx\sqrt{K}$.

**Proof.** By independence, $\operatorname{Var}(G_d^{\text{full}}\mid s_d)=\sum_k\sigma_k^2$;
subtracting the constant (in the randomness) $V(s_d)$ leaves it unchanged, giving
the first equality. For ccf, $G_d^{\text{ccf}}-V(s_d)=R_1-V(s_d)$; since $V(s_d)$
is $s_d$-measurable (constant given $s_d$),
$\operatorname{Var}(R_1-V(s_d)\mid s_d)=\operatorname{Var}(R_1\mid s_d)=\sigma_1^2$.
The cross-component terms $\sum_{k\ge2}R_k$ contribute their full variance
$\sum_{k\ge2}\sigma_k^2$ to the full estimator and zero to ccf. Under
$\sigma_k^2\approx\sigma^2$ the ratio is $K\sigma^2/\sigma^2=K$; variance scales
linearly and standard deviation as $\sqrt K$. $\qquad\blacksquare$

**Reading.** The full estimator pays the reward variance of *every* concurrent
component; ccf pays only the deciding component's. The gain grows with the number
of independent components in flight — the exact quantity the N-copies sweep
varies by construction.

---

## Corollary 1 (Monotone variance reduction — no idealization)

Proposition 2 quantifies the gain under mutually-independent, equal-variance
components. The following weaker statement drops both idealizations — it needs no
component-count model, no equal variances, and holds for *any* critic — at the
cost of giving a monotone inequality rather than a factor.

> **(A2) Component uncorrelatedness.** Conditional on $s_d$, the own-component
> reward-to-go $G_d^{\text{ccf}}$ and the cross-component reward-to-go $\Delta_d$
> are uncorrelated: $\operatorname{Cov}(G_d^{\text{ccf}},\Delta_d\mid s_d)=0$.
> (Implied by causal independence of distinct components — they share no lineage
> and, by the exogeneity used for (A1), no common stochastic driver; see R6.)
>
> **Claim.** Under (A2), for *any* baseline $V(s_d)$,
> $$
> \operatorname{Var}\!\big(G_d^{\text{full}}-V(s_d)\mid s_d\big)
> =\operatorname{Var}\!\big(G_d^{\text{ccf}}-V(s_d)\mid s_d\big)
> +\operatorname{Var}\!\big(\Delta_d\mid s_d\big)
> \;\ge\;
> \operatorname{Var}\!\big(G_d^{\text{ccf}}-V(s_d)\mid s_d\big),
> $$
> with equality **iff** $\operatorname{Var}(\Delta_d\mid s_d)=0$, i.e. iff decision
> $d$ has no cross-component future reward (in particular whenever $M=1$).

**Proof.** Write $A_d^{\text{full}}=G_d^{\text{full}}-V(s_d)=\big(G_d^{\text{ccf}}-V(s_d)\big)+\Delta_d=A_d^{\text{ccf}}+\Delta_d$.
Since $V(s_d)$ is $s_d$-measurable (constant given $s_d$), it cancels from every
centered second moment, so
$$
\operatorname{Var}(A_d^{\text{full}}\mid s_d)
=\operatorname{Var}(A_d^{\text{ccf}}\mid s_d)+\operatorname{Var}(\Delta_d\mid s_d)
+2\operatorname{Cov}(A_d^{\text{ccf}},\Delta_d\mid s_d).
$$
The covariance equals $\operatorname{Cov}(G_d^{\text{ccf}},\Delta_d\mid s_d)=0$ by
(A2) (again $V(s_d)$ is constant given $s_d$). Hence
$\operatorname{Var}(A_d^{\text{full}}\mid s_d)=\operatorname{Var}(A_d^{\text{ccf}}\mid s_d)+\operatorname{Var}(\Delta_d\mid s_d)$,
and since a variance is nonnegative the inequality follows, with equality iff
$\operatorname{Var}(\Delta_d\mid s_d)=0$. $\qquad\blacksquare$

**Reading.** ccf is **never worse** than the full (PPO) estimator in per-decision
advantage variance, for *any* critic and with *no* assumption on how many
components exist or how their variances compare, and it is **strictly better**
at exactly those decisions that have cross-component future reward. Proposition 2
is the special case that turns this qualitative "$\ge$" into the quantitative
"factor $K$" when the cross components are independent and comparably sized. This
is the unimpeachable version to cite when a reviewer questions P2's idealization;
P2 is then the interpretable magnitude under a clean model.

---

## Remarks (for the paper or rebuttal)

- **R1 — single-component collapse (the honest null).** If a shared resource
  links all cases, every reward's lineage eventually touches every decision and
  $M=1$: then $\Delta_d\equiv 0$, $G_d^{\text{ccf}}=G_d^{\text{full}}$, and ccf is
  *identically* PPO. This is why ccf ties PPO on single-pool queues (s1, grid) —
  not a failure but the theory's prediction ($K=1\Rightarrow$ no reduction).

- **R2 — contrast with `lrq` (why the middle path is needed).** `lrq` restricts
  credit to a *single token lineage*, dropping the rewards that *sibling*
  lineages — the roads not taken at $d$ — would have earned. Those are causal
  descendants of *alternative* actions at $d$ and are **not** mean-independent of
  $a_d$; dropping them violates the analogue of (A1) and biases the estimate
  (the measured "foreclosure bias" on s1). ccf keeps the whole component, so it
  never drops an $a_d$-dependent reward: unbiased by construction.

- **R3 — shared RNG.** With a single global random stream, changing $a_d$ can
  shift the *order* in which cross-component events consume random draws, so their
  *realized* values may differ across counterfactual actions. This affects
  realized $\Delta_d$ but not $\mathbb E[\Delta_d\mid\mathcal F_d,a_d]$ (the draws
  are i.i.d. and the cross-component subprocess distribution is unchanged), so
  (A1) — a statement about conditional means — is unaffected. Under common random
  numbers per stream the realizations coincide as well.

- **R4 — postpone.** The proposition concerns real action-decisions. Under
  token-flow-off, postpone decisions have no reward-causing descendants
  ($G^{\text{ccf}}=0$) and are instead credited by the SMDP-TD advantage via the
  postpone mask (as in `lrq2`); this leaves P1 intact for the action-decisions.

- **R5 — empirical corroboration.** P2 predicts the full estimator's per-decision
  credit magnitude grows $\sim K$ while ccf's stays $O(1)$ per component. The
  estimator diagnostic matches: on N independent copies the `mc_q` credit sum
  grows $\sim N^2$ (each of $\sim N$ decisions sees $\sim N$ components' rewards)
  while the ccf sum grows $\sim N$, and $|ccf-mc\_q|/\text{scale}$ rises
  $0.19\!\to\!0.94$ as $N:1\!\to\!8$.

- **R6 — (A2) from causal independence.** Distinct components share no lineage,
  so $G_d^{\text{ccf}}$ (a function of tokens in $c(d)$) and $\Delta_d$ (a
  function of tokens outside $c(d)$) depend on disjoint token sets. Given $s_d$,
  the exogenous draws governing those disjoint sets are independent (the
  per-firing-independence used for (A1)), so $G_d^{\text{ccf}}\perp\Delta_d\mid s_d$;
  in particular they are uncorrelated, which is all (A2) requires. Note (A2) is
  strictly weaker than this conditional independence and weaker than P2's mutual
  independence — the corollary needs only zero correlation, so it survives mild
  residual coupling that would break the stronger assumptions.

---

## When (A1) fails, and the static-component fix ($s\text{-}ccf$)

Proposition 1 rests on (A1): the cross-component reward $\Delta_d$ is mean-
independent of $a_d$. This holds when the component partition is fixed, but the
$ccf$ estimator computes it from the **realized trajectory** via union-find, and
that partition can itself **depend on the action** — which breaks (A1).

**The failure motif (an AND-join).** Let decision $d$ choose between routing a
token into an assembly/join transition ($a_d=A$) or to a standalone path
($a_d=B$). A second decision $d'$ feeds the same join and also carries a private
reward $r'$. Under $A$ the join fires, so $d$ and $d'$ are unioned into one
realized component and $r'\in c(d)$; under $B$ there is no join, the components
are separate, and $r'\notin c(d)$. But $r'$ fires under **both** actions (it is
$d'$'s private reward). So $ccf$ credits $d$ with $r'$ only under $A$: the
component **membership is action-dependent**, (A1) fails, and $\Delta_d$ is not
mean-independent of $a_d$. With $r' > r_1-r_{\text{join}}$ this is not merely a
variance issue — it **reverses the sign** of $d$'s credit difference, so $ccf$
prefers the suboptimal join. (Verified exactly on five hand-built motifs in
`assembly_probe.py`; $lrq$ suffers the analogous failure when the coupling is a
shared *resource* rather than a join.)

**The fix.** Compute the component partition from the **static net topology**,
not the realized trajectory: two decisions are in one component iff they *could*
co-cause a reward under **some** action sequence (topological reachability,
taking the union over all of a decision's competing actions). Call the resulting
estimator $s\text{-}ccf$: each decision is credited with the realized rewards of
its **static** component.

\begin{assumption}[Static (A1$'$)]
The component partition is a function of the net topology alone; in particular it
does not depend on $a_d$.
\end{assumption}

(A1$'$) holds **by construction** for $s\text{-}ccf$. In the motif above, $d$ and
$d'$ are statically coupled (the join is reachable from $d$'s route-to-join
action), so $r'\in c(d)$ under **both** $A$ and $B$; the $r'$ term therefore
cancels in $d$'s credit difference, exactly as it should.

\begin{proposition}[Unbiasedness of $s\text{-}ccf$]
Under (A1$'$), the excluded rewards $\{r_j: c(j)\neq c(d)\}$ are, by topological
construction, unreachable from $a_d$ under any action, hence action-independent
given $s_d$. Therefore the argument of Proposition~1 applies with (A1$'$) in
place of (A1), and $s\text{-}ccf$ is an unbiased policy-gradient estimator —
without assuming anything about the realized trajectory.
\end{proposition}

$s\text{-}ccf$ retains the variance reduction of $ccf$ on statically-independent
structure (there the static and realized partitions coincide) and reduces to
$mc\_q$ only when the whole net is one static component. It thus **dominates**
$ccf$: unbiased in the join/handoff regimes where $ccf$ (and $lrq$) flip, and
identical to $ccf$ on pure resource-assignment problems where the triggering
structure is absent.

\paragraph{Reading / status.} $ccf$ is the naive realized-component estimator;
$s\text{-}ccf$ is the principled one, with (A1) upgraded from fine print to a
checkable topological property (A1$'$). The distinction is invisible on pure
dynamic task assignment — assignment decisions do not create joins, so the
realized and static partitions agree, and both $ccf$ and $s\text{-}ccf$ behave
identically (confirmed: on ncopies and s1, $s\text{-}ccf\approx ccf$, both
beating / matching PPO as expected). It becomes decisive exactly at AND-join /
synchronization transitions. \emph{The bias and the fix are established at the
estimator level (exact, library-verified); a clean training-level demonstration
is elusive because the very structure that triggers the $ccf$ flip — a private
reward coupled through a join — is an intrinsically hard credit-assignment
problem in which even the unbiased baseline (PPO) fails to learn the optimum.}
See `spectrum-experiment` notes and `assembly_probe.py`.

---

## Proposition 3 (Causal-order GAE, `cgae`)

$ccf$ restricts *which* rewards a decision sees but keeps the Monte-Carlo
return-to-go ($\lambda=1$, no bootstrap). `cgae` keeps the same restriction
implicitly and changes *how* the retained rewards are propagated: it runs
GAE's own recursion along the **provenance DAG** instead of the trajectory
index. This section shows that the unbiasedness argument of Proposition 1
transfers to it unchanged, because the two estimators omit **the same mass**.

### Additional notation

For decisions $d\ne d'$ write $d\to d'$ when $d'$ consumes a token descending
from an output token of $d$ with no intervening decision (the *nearest
producer* relation of `_redistribute_cgae`); let $\mathrm{succ}(d)=\{d': d\to d'\}$
and let $\mathrm{desc}(d)$ be the reflexive-transitive closure. Note
$\mathrm{desc}(d)\subseteq c(d)$: a causal descendant shares a lineage with $d$,
hence a component.

**Single-owner assignment.** Each reward $j$ with $\mathcal A(j)\ne\varnothing$
is assigned to exactly one decision,
$$
\mathrm{own}(j)\;=\;\arg\max_{d\in\mathcal A(j)} u_d ,
$$
the latest decision on its lineage (ties broken by index). Write
$w_d=\sum_{j:\,\mathrm{own}(j)=d} \rho_d(t_j)\,r_j$ for the mass $d$ owns.
Ownership is a **partition** of the rewards: $\sum_d$ (owned) counts each
reward exactly once, in contrast to the lineage $Q$-samples of $lrq$, which
give the full mass of $r_j$ to *every* $d\in\mathcal A(j)$.

### Assumptions

> **(A2) Single causal successor.** $|\mathrm{succ}(d)|\le 1$ for every $d$.

> **(A2′) Clock-exogeneity of the cross-component critic.** Writing
> $V=V_{c(d)}+V_{\neg c(d)}$ for the decomposition of the critic into its own-
> and cross-component parts, $\mathbb E\big[\rho_d(u_{s})V_{\neg c(d)}(s_{s})\mid
> \mathcal F_d,a_d\big]$ does not depend on $a_d$, where $s=\mathrm{succ}(d)$.

(A2′) is needed only for $\lambda<1$; it is the statement that although $a_d$
may shift *when* the successor decision occurs, the discounted value of the
other components at that moment is, in conditional mean, unaffected. It holds
whenever $e^{-\beta t}V_{\neg c(d)}(t)$ has zero drift given $\mathcal F_d$, or
whenever the successor time is conditionally independent of $a_d$.

### Statement

> **(i)** Under (A1) and (A2), the $\lambda=1$, $V\equiv 0$ `cgae` estimator
> satisfies $\mathbb E[\hat g^{\text{cgae}}]=\nabla_\theta J(\theta)$.
>
> **(ii)** Under (A1), (A2) and (A2′), and with $V=V^\pi$, the $\lambda<1$
> estimator is a $\lambda$-weighted average of $k$-step causal advantage
> estimators, each unbiased; `cgae` is therefore unbiased in exactly the sense
> ordinary GAE is, and its $\lambda$ knob has the same bias-variance reading.
>
> **(iii)** (A2) may be dropped if the successor aggregation is replaced by the
> **flow-weighted sum** $\sum_{s\in\mathrm{succ}(d)} w(d\!\to\!s)\,A_s$ with
> $w(d\!\to\!s)$ the fraction of $s$'s consumed tokens descending from $d$, so
> that $\sum_{d\in\mathrm{pred}(s)} w(d\!\to\!s)=1$.

### Proof

**(i).** Under (A2) each $d$ has at most one successor, so the recursion
$A_d=w_d+\rho_d(u_{s})A_{s}$, $s=\mathrm{succ}(d)$, unrolls along a path and
terminates (the DAG is acyclic in decision time). Composing the SMDP discounts
telescopes, giving
$$
A_d\;=\;\sum_{k\in\mathrm{desc}(d)}\rho_d(u_k)\;\frac{w_k}{\rho_k(u_k)}
\;=\;\sum_{\substack{j:\ \mathrm{own}(j)\in\mathrm{desc}(d)}}\rho_d(t_j)\,r_j .
$$
Because ownership is a partition, every reward in the sum appears exactly once.
Now compare with $G_d^{\text{ccf}}$. Both omit every reward with
$c(j)\ne c(d)$: for $\mathrm{own}(j)\in\mathrm{desc}(d)\subseteq c(d)$ implies
$c(j)=c(d)$. They differ only in which *own-component* rewards are retained —
`cgae` keeps those whose owner is a causal descendant of $d$, $ccf$ keeps those
with $t_j\ge u_d$. Write this difference as $\Xi_d:=G_d^{\text{ccf}}-A_d$, so
$$
\hat g^{\text{full}}-\hat g^{\text{cgae}}
=\sum_{d}\nabla_\theta\log\pi_\theta(a_d\mid s_d)\,(\Delta_d+\Xi_d).
$$
The $\Delta_d$ term vanishes in expectation by Proposition 1. For $\Xi_d$:
a reward $j$ with $c(j)=c(d)$, $t_j \ge u_d$ and $\mathrm{own}(j)\notin\mathrm{desc}(d)$
has, by definition of $\mathrm{own}$, its whole lineage $\mathcal A(j)$ disjoint
from $\mathrm{desc}(d)$ — no descendant of $a_d$ lies on $j$'s lineage. By the
same A-E PN argument that justifies (A1) — $r_j$ is a deterministic function of
the tokens consumed at its firing, and those tokens' histories are recorded in
the DAG — $r_j$ is then causally uninfluenced by $a_d$, so
$\mathbb E[\Xi_d\mid\mathcal F_d,a_d]=\mathbb E[\Xi_d\mid\mathcal F_d]$.
Conditioning on $\mathcal F_d$ and applying the score-function identity exactly
as in Proposition 1,
$$
\mathbb E\!\left[\nabla\log\pi_\theta(a_d\mid s_d)\,\Xi_d\mid\mathcal F_d\right]
=\bar\Xi_d\,\nabla_\theta\!\!\sum_a \pi_\theta(a\mid s_d)=\bar\Xi_d\,\nabla_\theta 1=0 .
$$
Summing over $d$ and taking total expectation gives the claim. $\qquad\blacksquare$

**(ii).** With $V=V^\pi$ define the $k$-step causal advantage
$A_d^{(k)}=\sum_{i=0}^{k-1}\rho_d(u_{d_i})w_{d_i}+\rho_d(u_{d_k})V(s_{d_k})-V(s_d)$
along the successor path $d_0=d,\ d_{i+1}=\mathrm{succ}(d_i)$. Each
$A_d^{(k)}$ is the ordinary $k$-step SMDP advantage of the sub-process obtained
by restricting to $c(d)$; its own-component part is unbiased by the standard
argument, and its cross-component part is mean-independent of $a_d$ by (A1) for
the reward terms and (A2′) for the bootstrap term $\rho_d(u_{d_k})V_{\neg c(d)}$,
so both vanish in the gradient by the score-function identity. `cgae`'s
recursion with $\lambda<1$ is the geometric average
$A_d=(1-\lambda)\sum_{k\ge1}\lambda^{k-1}A_d^{(k)}$ (immediate by unrolling, as
for GAE), a convex combination of unbiased terms. $\qquad\blacksquare$

**(iii).** Without (A2) the unrolling of (i) double-counts: a reward owned by a
decision reachable from $d$ along two distinct paths is added once per path.
Flow weights repair exactly this. Since $\sum_{d\in\mathrm{pred}(s)}w(d\!\to\!s)=1$,
the total credit passed backward out of $s$ is $A_s$ regardless of how many
predecessors it has, so unrolling assigns each owned reward a total weight of
$1$ across all paths; the sum in (i) is recovered with each reward counted once,
and the rest of the proof is unchanged. On a chain $w\equiv1$ and the weighted
sum is the identity, so (iii) strictly generalizes (i). $\qquad\blacksquare$

### Remarks

- **R6 (What is actually new).** `cgae` = component-scoped GAE. The recursion
  is GAE's, so it inherits GAE's theory; the scoping is $ccf$'s, so it inherits
  Proposition 1. The only genuinely new obligation is the aggregation rule over
  $\mathrm{succ}(d)$, which is what (iii) settles.

- **R7 (The implementation uses `mean`, not the flow-weighted sum).**
  `_redistribute_cgae` averages over successors. On a chain this coincides with
  (i) exactly; under fan-out it is a shrinkage with no derivation, and is
  **not** covered by the proposition. This is a real gap between the shipped
  code and the theorem. It is empirically narrow: measured fan-out on
  `ncopies` $N{=}4$ — the benchmark on which the result rests — is $1.07$, with
  $93\%$ of decisions having $\le1$ causal successor, so (A2) holds on almost
  all of the data. On `s1`, where fan-out is $1.34$ ($34.5\%$ multi-successor),
  `cgae` measures as *null* against PPO, i.e. the regime the theorem does not
  cover is also the regime in which no claim is made. Shipping the
  flow-weighted variant of (iii) would close the gap; the prediction is that it
  leaves the $N{=}4$ numbers unchanged.

- **R8 (Relation to $lrq$'s bias).** The single-owner partition is what keeps
  $\Xi_d$ action-mean-independent. $lrq$ instead grants the *full* mass of
  $r_j$ to every $d\in\mathcal A(j)$ and drops everything else, which discards
  reward that *is* action-dependent through shared-resource foreclosure — a
  deleted gradient term rather than a baseline. That is the bias $alin$ inverts
  and the reason $lrq$ underperforms PPO on `s1` ($11.21$ vs $13.59$) while
  `cgae` sits at parity.

- **R9 (Empirical status).** `ncopies` $N{=}4$, 20 paired seeds, common random
  numbers: `cgae` $0.936\pm0.158$ normalized vs PPO $0.477\pm0.418$, a paired
  gain of $+0.459$ ($t$-test $p=1.1\times10^{-4}$, Wilcoxon $p=1.7\times10^{-4}$,
  sign test $17/20$), with $0/20$ seeds collapsing against PPO's $5/20$. At
  $N{=}2$, $+0.371$ ($p=0.024$). On `s1`, statistically indistinguishable from
  PPO — as the theory predicts, since there the component partition is a single
  blob (measured: one reward-bearing component holding $100\%$ of reward mass)
  and there is nothing to scope.

---

## Part II — DAG-replay counterfactual credit (`cfpk`) and coupling truncation

The estimators above *restrict* which realized rewards count toward a
decision's credit (a partition of the same single trajectory). `cfpk` takes
a different route entirely: it does not partition anything. At a decision
$d$ it **forks the simulator** and replays the true environment dynamics
forward under the taken action $a_d$ and, separately, under an alternative
$a_d'$, from the identical pre-decision state — an exact, model-based
counterfactual (the "model" is the real simulator, not a learned
approximation), never a filter on one realized trajectory.

### Setup and notation (extends Part I)

- **Fork.** At decision $d$ (state $s_d$, clock $u_d$), snapshot the
  simulator and roll out **two branches** from the identical snapshot: one
  continuing with the taken action $a_d$, one with an alternative $a_d'$.
  Write $\omega\in\{a_d,a_d'\}$ for a branch.
- **Continuation return.** For a branch $\omega$, let
  $$
  G_d(\omega)\;=\!\!\sum_{j:\,t_j\ge u_d}\!\!\rho_d(t_j)\,r_j^{(\omega)}
  $$
  be the SMDP-discounted return realized by that branch's simulated
  continuation (policy $\pi_\theta$ thereafter; a lookahead horizon $L$ or
  episode end, optionally closed off by a learned value-tail bootstrap
  $\gamma^{\cdot}V_\phi(\cdot)$ at the horizon — the proposition below is
  agnostic to which, see Remark R9).
- **The credit signal.** $I_d = G_d(a_d) - G_d(a_d')$, the *exact* realized
  counterfactual advantage of the taken action over the alternative — exact
  because both branches are simulated with the true environment, not
  estimated from a single trajectory's lineage.
- **State fingerprint.** $\phi(\cdot)$: the canonical, symmetry-collapsed
  (marking, clock) signature already used by the MCTS planner's
  transposition table (`gympn.mcts_planner.state_fingerprint`), reused here
  unchanged. $\phi$ collapses only token relabelings that are provably
  interchangeable (anonymous unit resources); it never merges two states
  that differ in anything that could affect future dynamics.
- **Lockstep rollout and reconvergence.** Simulate both branches one
  environment step at a time, in lockstep. Let
  $$
  \sigma \;=\; \min\{\text{round } k : \phi(s_d^{(a_d)}(k)) = \phi(s_d^{(a_d')}(k))\}
  $$
  (the first round, if any within the lookahead, at which the two branches'
  simulated states coincide), and let $s^\*$ denote that common state,
  reached at common elapsed time $t^\*-u_d$ (identical for both branches,
  since $\phi$ includes the clock). If no such round exists within the
  lookahead, $\sigma:=\infty$ and both branches are simulated to completion
  independently — the untruncated base case, exact and unbiased on its own
  by construction (it is literally the true environment).

### Assumption

> **(A3) Fingerprint sufficiency.** For any two simulated states $s,s'$ with
> $\phi(s)=\phi(s')$, the conditional law of all future (policy and
> exogenous) randomness given $s$ equals that given $s'$. Equivalently,
> $\phi$ is a bisimulation-consistent sufficient statistic for the SMDP's
> transition-and-reward kernel: two states with the same fingerprint induce
> *identical* distributions over every future trajectory.

**Why (A3) holds by construction.** The AEPN's dynamics depend on the
marking and the clock alone (transition enablement, timing, and the
policy's graph observation are all functions of these); $\phi$ discards
nothing but token identities within a class of tokens that are, by
construction, interchangeable in every downstream rule (same place, same
attributes, same enabled transitions) — the identical collapsing already
relied upon, and separately validated, for the MCTS transposition table. No
further assumption is needed: (A3) is a structural fact about the simulator,
not a problem-specific hypothesis.

### The coupling-truncated estimator

When $\sigma<\infty$, decompose each branch's return around the
reconvergence point $s^\*$:
$$
G_d(a_d) = P_{a_d} + \rho_d(t^\*)\,\tilde V(s^\*), \qquad
G_d(a_d') = P_{a_d'} + \rho_d(t^\*)\,\tilde V(s^\*),
$$
where $P_\omega$ is branch $\omega$'s own realized (pre-reconvergence)
discounted reward up to round $\sigma$ — generally different across
branches, since they took different actions at $d$ — and $\tilde V(s^\*)$ is
a **single** simulated sample of the continuation return from $s^\*$ to the
lookahead horizon (whatever procedure — full Monte Carlo, or a
value-tail-bootstrapped truncation — generates that sample; see R9),
**reused identically for both branches** rather than drawn twice. This is
exactly what `_paired_coupled_suffixes` computes: `A['disc'] + shared`,
`B['disc'] + shared` with the same `shared` scalar on both sides.

\begin{proposition}[P3 — Coupling truncation is exact]
Under (A3), for any realization of the pre-reconvergence randomness, the
coupling-truncated credit signal satisfies
$$
I_d^{\text{coupled}} \;=\; G_d(a_d) - G_d(a_d') \;=\; P_{a_d} - P_{a_d'},
$$
**exactly** — the shared continuation $\tilde V(s^\*)$ cancels identically,
regardless of its realized value. Consequently $I_d^{\text{coupled}}$ has
the same distribution as the naive estimator that draws two *independent*
continuation samples $\tilde V_a(s^\*),\tilde V_{a'}(s^\*)$ from $s^\*$ (one
per branch) and forms $I_d^{\text{naive}} = (P_{a_d}+\rho_d(t^\*)\tilde
V_a(s^\*)) - (P_{a_d'}+\rho_d(t^\*)\tilde V_{a'}(s^\*))$ **in expectation**,
and has **strictly lower variance** whenever
$\operatorname{Var}(\tilde V(s^\*)\mid s^\*)>0$: the reconverged tail
contributes exactly zero variance to $I_d^{\text{coupled}}$, versus
$\rho_d(t^\*)^2\cdot 2\operatorname{Var}(\tilde V(s^\*)\mid s^\*)$ (two
independent draws) to $I_d^{\text{naive}}$.
\end{proposition}

**Proof.** *Exact cancellation.* Immediate from the decomposition above: the
$\rho_d(t^\*)\tilde V(s^\*)$ term is literally the same real number added to
both $G_d(a_d)$ and $G_d(a_d')$, so it cancels in the difference for every
realization, not merely in expectation — no appeal to (A3) is even needed
for this half of the claim, only that the implementation reuses one draw.

*Same target.* (A3) gives $\phi(s_d^{(a_d)}(\sigma))=\phi(s_d^{(a_d')}(\sigma))$
$\Rightarrow$ both branches' post-$\sigma$ continuations are drawn from the
*identical* distribution, call its mean $V^\pi(s^\*)$. Hence
$\mathbb E[\tilde V(s^\*)]=\mathbb E[\tilde V_a(s^\*)]=\mathbb E[\tilde
V_{a'}(s^\*)]=V^\pi(s^\*)$, so
$\mathbb E[I_d^{\text{naive}}] = \mathbb E[P_{a_d}-P_{a_d'}] +
\rho_d(t^\*)(V^\pi(s^\*)-V^\pi(s^\*)) = \mathbb E[P_{a_d}-P_{a_d'}] =
\mathbb E[I_d^{\text{coupled}}]$: both estimators target the same quantity.
(Note the *naive* estimator is itself already unbiased for this quantity —
coupling truncation is not fixing a bias, it is removing variance the naive
sequential estimator was paying for nothing.)

*Variance.* Write $I_d^{\text{naive}} = (P_{a_d}-P_{a_d'}) +
\rho_d(t^\*)(\tilde V_a(s^\*)-\tilde V_{a'}(s^\*))$. The two continuation
draws are simulated independently (fresh exogenous randomness per branch
past $\sigma$) and independent of the pre-$\sigma$ randomness that generated
$P_{a_d},P_{a_d'}$ (future randomness given the state, by the Markov
property), so
$\operatorname{Var}(I_d^{\text{naive}}) = \operatorname{Var}(P_{a_d}-P_{a_d'})
+ \rho_d(t^\*)^2\cdot 2\operatorname{Var}(\tilde V(s^\*)\mid s^\*)$, while
$I_d^{\text{coupled}}=P_{a_d}-P_{a_d'}$ exactly, so
$\operatorname{Var}(I_d^{\text{coupled}}) = \operatorname{Var}(P_{a_d}-P_{a_d'})$
— strictly smaller whenever the reconverged tail has any variance at all.
$\qquad\blacksquare$

**Reading.** This is a stronger claim than a typical common-random-numbers
(CRN) coupling argument, which *reduces* shared-tail variance by inducing
positive correlation between the two draws; here the shared draw is not
merely correlated with itself across branches, it *is* the same draw, so the
tail's contribution to the paired difference is not reduced but
**eliminated**. And unlike CRN, this needs no alignment of the two branches'
random-number-stream *positions* at the reconvergence point (a much
stronger, not-generally-available condition) — only that they have reached
the same *state*, which determines the same future distribution regardless
of how each branch got there (Remark R7).

### Remarks

- **R7 — no RNG-stream alignment required.** A naive "couple the tails"
  implementation might assume the two branches' underlying random-number
  generators are in the same position when they reconverge in state — false
  in general, since the branches took a different number of intervening
  exogenous events to reach $s^\*$. The proof above never uses this: it only
  uses that $s^\*$ is the same state for both (A3), and that *one* draw from
  that state is reused for both rather than two draws being taken. This is
  why the mechanism is safe to implement with ordinary independent
  simulation, no CRN bookkeeping across the reconvergence boundary.

- **R8 — cost.** Once coupled, `env.step` calls for the remainder of the
  lookahead window drop from 2/round (both branches) to 0 (a single shared
  continuation is drawn once, outside the lockstep loop) — the mechanism
  `_test_coupling_truncation.py` confirms exactly on a controlled
  reconverging net (18$\to$12 `env.step` calls, 33% reduction) and
  `run_cfpk_coupling_wallclock.py` measures at real training scale on a
  contested topology designed to fork often (`g_exclusive_choice_joint`,
  full protocol scale): 1.26x wall-clock speedup, with `norm_final`
  *exactly* identical between the truncated and untruncated arms across all
  seeds — the zero-variance-contribution property of P3 predicts exactly
  this outcome (no change to what is learned, only to how it is computed).

- **R9 — agnostic to the tail procedure.** The proof only uses that
  $\tilde V(s^\*)$ is *some* random variable generated by a fixed procedure
  applied to the state $s^\*$ alone (Markov in $s^\*$) — it does not matter
  whether that procedure is pure Monte Carlo simulation to episode end or a
  value-tail-bootstrapped truncation at a finite lookahead ($\rho\cdot
  V_\phi(\cdot)$ appended at the horizon, as `_continue_discounted`
  implements). Both are valid instantiations; P3 covers `cfpk`'s actual
  production configuration (`value_tail=True`) without modification.

- **R10 — scope.** Coupling truncation is gated off for the lineage-credit
  suffix (`cf_lineage=True`; see `maybe_fork`'s `not lineage` guard): a
  post-hoc lineage analysis is defined per complete causal trace, and
  sharing one tail between two branches' separate traces has no well-defined
  per-branch lineage meaning. P3 applies to the plain (non-lineage) suffix,
  which is what `cfpk` uses in production.

\paragraph{Reading / status.} P3 completes the theory side of the paper's
two-mechanism structure: Part I (`s-ccf`) shows *what* to credit is safe to
restrict, given the right (static) partition; Part II (`cfpk` + coupling
truncation) shows the *cost* of computing an exact, unpartitioned
counterfactual credit is safe to reduce, given the right (state-sufficient)
reconvergence test. Both propositions share the same proof pattern — find a
quantity that is provably mean- (or, in P3, realization-) independent of the
thing being dropped, and show dropping it changes nothing that matters —
which is worth stating explicitly in the paper as the throughline connecting
Parts I and II, not two unrelated tricks.