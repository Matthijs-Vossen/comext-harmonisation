# Lukaszuk et al. (2022) — “LT method” for trade code harmonisation (implementation-ready extract)

This is a structured, code-oriented overview of the **LT method** (Lukaszuk & Torun, 2022) for harmonising trade data across classification revisions.  
It is written to be directly usable as a spec for implementing the method (and adapting it to **CN8 / Eurostat Comext**).

---

## 1) What problem the method solves

### 1.1 Why harmonising classifications is hard
When a classification revision happens, product codes can be:

- **1:1** unchanged  
- **m:1** multiple codes merge into one code  
- **1:n** one code splits into multiple codes  
- **m:n** general reassignment network  

The difficulty depends on the **chosen conversion direction**.  
In any direction, estimation is required exactly when an **initial-vintage code links to multiple target-vintage codes** (a one-to-many row in the concordance):

\[
|\{s : (k,s)\text{ is an allowed link}\}| > 1
\]

These are the cases where historical trade must be **allocated across multiple possible target codes** using unknown weights.

---

### 1.2 Core LT idea
The LT method infers conversion weights from the assumption that, within a revision-affected concordance group,
the **relative composition of trade is persistent over time**.

Concretely, it chooses weights **\(\beta\)** so that **scaled bilateral shares observed under the initial vintage in year**
\(t_0\), once converted, best match **scaled bilateral shares observed under the target vintage in year**
\(t_1\), while respecting the official concordance link constraints.

> The method is **direction-agnostic**: it can be applied **forward** (older → newer) or **backward** (newer → older), as long as trade is observed under both vintages in the chosen \(t_0\) and \(t_1\).


---

## 2) Key objects and definitions

### 2.1 Codes, concordance links, and “product groups”
Consider two vintages:

- initial-vintage codes: **\(k \in K\)**  
- target-vintage codes: **\(s \in S\)**  

The official concordance provides which pairs \((k, s)\) are allowed links.

**Product group = connected component (in the concordance network)**  
The method builds a **bipartite graph** (initial codes on one side, target codes on the other, concordance links as edges).  
A **product group \(g\)** is a **connected component** of this bipartite graph.

> Intuition: if two initial codes map to the same target code (directly or indirectly), they are linked and belong to the same group.

Weights are estimated **within groups only** (weights across groups are structurally zero).

---

### 2.2 Trade flows are bilateral product flows
The paper uses bilateral flows to increase observations and identify weights:

- **\(X^t_{ij,k}\)**: trade value (e.g., imports/exports) from exporter \(i\) to importer \(j\) in product \(k\) (in year \(t\)).

The paper’s baseline focuses on imports, but the method is symmetric for exports.

---

### 2.3 Within-group scaling into trade shares (crucial)
The method works on **scaled shares** rather than raw levels.

For each product group **\(g\)** and year **\(t\)**, define:

\[
x^t_{ij,k} \equiv
\frac{X^t_{ij,k}}
{\sum_{\hat i,\hat j}\sum_{\hat k \in g} X^t_{\hat i \hat j,\hat k}}
\]

So the denominator is the **total group trade** (over all pairs and products) for that group-year.

Properties:

- scaling is done **separately for each group and year**
- for each group-year: \(\sum_{i,j,k \in g} x^t_{ij,k} = 1\)
- prevents group-level shocks/trends from dominating the estimation

---

## 3) The LT conversion algorithm (baseline)

The method is described as three components:

---

### Step 1 — Build product groups and identify which need estimation
1. Build the concordance links between vintages.
2. Form **connected components** = product groups **\(g\)**.
3. **Estimate weights only** for groups that contain at least one **ex-ante ambiguous mapping** in the conversion direction  
   (i.e., at least one 1:n or m:n case).
4. Groups that are fully deterministic (only 0/1 mappings) can be converted directly (no optimisation needed).

---

### Step 2 — Prepare estimation data for each group
For each group **\(g\)**:

- Build a dataset of bilateral observations:
  - each observation = a country-pair \((i, j)\)
  - features = initial-vintage products \(k \in K_g\)
  - targets = target-vintage products \(s \in S_g\)

- Convert absolute flows \(X\) into within-group shares \(x\) (Section 2.3).

#### “Compliers” / switchers restriction
The paper restricts estimation to importers that actually switched between the vintages (to avoid mixing countries still reporting in old vintage).

(For EU CN revisions, adoption is typically uniform, but this maps to excluding non-standard partner aggregates / confidential categories from estimation.)

---

### Step 3 — Estimate conversion weights via constrained least squares
Let:

- **initial vintage** = the classification used in the data you start from (codes \(k \in K\))
- **target vintage** = the classification you want to convert into (codes \(s \in S\))

Choose two years:

- \(t_0\): a year observed under the **initial** vintage
- \(t_1\): a year observed under the **target** vintage

> The LT method is **direction-agnostic**: it can be used for **forward** conversion (older → newer) or **backward** conversion (newer → older). The ordering of years \(t_0\) and \(t_1\) is therefore not fixed.

#### Objective (baseline constrained least squares)
\[
\min_{\{\beta_{k,s}\}}
\sum_{s}\sum_{i,j}
\left(
x_{ij,s}^{t_1} - \sum_{k} x_{ij,k}^{t_0}\beta_{k,s}
\right)^2
\]

#### Constraints (three types)
1) **Non-negativity**
\[
\beta_{k,s} \ge 0 \quad \forall k,s
\]

2) **Row-sum-to-one per initial code (mass preservation)**
\[
\sum_s \beta_{k,s} = 1 \quad \forall k
\]

3) **Hard-fix deterministic weights implied by official concordance**
- impossible links: \(\beta_{k,s} = 0\)
- deterministic mapping in chosen direction: \(\beta_{k,s} = 1\) (and 0 for the rest of the row)

#### Estimation is performed group-by-group
Because codes outside a group cannot map into the group, optimisation is solved **separately for each group**.

---

## 4) Applying estimated weights to convert trade data

### 4.1 Convert one year of trade (initial → target)
After estimating \(\hat\beta_{k,s}\):

\[
\hat x_{ij,s}^{t_0\rightarrow t_1} = \sum_k x_{ij,k}^{t_0}\hat\beta_{k,s}
\]

In practice, apply weights to **absolute flows \(X\)** (then aggregate), not just shares.

---

### 4.2 Convert across multiple revisions by chaining matrices
To convert across multiple vintage steps (e.g. A→B→C), multiply conversion matrices:

\[
\hat\beta_{k,l} = \sum_s \hat\beta_{k,s}\hat\beta_{s,l}
\]

Applied to trade:

\[
\hat x_{ij,l} =
\sum_{s}\sum_{k} x_{ij,k}\hat\beta_{k,s}\hat\beta_{s,l}
\]

This preserves non-negativity and row-sum-to-one.

---

## 5) Why bilateral scaling matters (identification + stability)
The paper prefers bilateral flows over importer-only totals because:

1. bilateral detail adds identifying variation
2. fewer cases of “more unknowns than observations” vs coarser aggregation

They empirically validate persistence using autoregressive-style regressions of bilateral shares on lagged shares and find high persistence (~0.85–0.89).

---

## 7) Conditions for the method to work
The method assumes:

1. a valid official concordance between vintages
2. persistence: revision boundaries should not generate major structural breaks once codes are made comparable

They also note concordances may be broad (too many links); the estimation handles this by assigning near-zero weights to irrelevant links.

---

## 9) Mapping to CN8 / Eurostat Comext harmonisation
Direct adaptation steps:

1. Use Eurostat CN correspondence tables as the concordance network.
2. Define CN8 product groups as connected components.
3. Estimate weights using **annual bilateral CN8 flows** (stable), apply to monthly.
4. Exclude special/aggregate partner codes from weight estimation, but retain them in conversion outputs.
5. Chain across many CN revisions by multiplying estimated step matrices.
