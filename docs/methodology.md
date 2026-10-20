# Methodology

## Physical setting

Silicon wafers are sliced from a monocrystalline boule and polished to a
target thickness. In-line metrology records thickness deviations at nine
spatially distributed gauge groups $G_1,\ldots,G_9$. A section index
$i = 1,\ldots,n$ labels successive scan locations (or wafers), yielding the
data matrix

$$X_{\mathrm{raw}} \in \mathbb{R}^{n \times 9}, \qquad (X_{\mathrm{raw}})_{ij} = \text{thickness deviation at gauge } G_j.$$

Because the gauges sample overlapping regions of the same continuum
thickness field, the columns of $X_{\mathrm{raw}}$ are strongly correlated.
The analysis below isolates the dominant spatial modes of that field.

## Standardization

Each gauge is mapped to a common scale by the sample $z$-score

$$X_{ij} = \frac{(X_{\mathrm{raw}})_{ij} - \mu_j}{\sigma_j}, \qquad \mu_j = \frac{1}{n}\sum_{i=1}^{n}(X_{\mathrm{raw}})_{ij}, \qquad \sigma_j = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}\bigl((X_{\mathrm{raw}})_{ij}-\mu_j\bigr)^{2}}.$$

Without this step, gauges with larger metrology variance would dominate the
covariance spectrum irrespective of their physical coupling. A parallel
**median / MAD** scaling and a Mahalanobis gate on the $z$-scored rows are
used to test sensitivity to heavy-tailed thickness spikes.

PCA on the *centered but unscaled* matrix (raw covariance) is retained as a
contrast: it answers a different question (which gauges move most in absolute
units) and is not interchangeable with correlation PCA for mode
interpretation.

## Sample covariance and eigenmodes

The unbiased sample covariance of the standardized gauges is

$$C = \frac{1}{n-1}\, X^{\mathsf{T}} X \in \mathbb{R}^{9 \times 9}.$$

Principal component analysis solves the symmetric eigenproblem

$$C\, v_k = \lambda_k\, v_k, \qquad \lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_9 \ge 0,$$

where the orthonormal eigenvectors $v_k$ are the **loadings** and the
eigenvalues $\lambda_k$ are the variances along those modes. The fraction of
total variance explained by component $k$ is

$$\rho_k = \frac{\lambda_k}{\sum_{m=1}^{9}\lambda_m}.$$

### SVD equivalence

Write $\tilde{X} = X / \sqrt{n-1}$ and compute the thin SVD
$\tilde{X} = U\Sigma V^{\mathsf{T}}$. Then the columns of $V$ coincide with
the eigenvectors of $C$, and $\lambda_k = \Sigma_{kk}^{2}$.

### Bootstrap uncertainty

Resampling the $n$ rows with replacement and recomputing $\{\rho_k\}$ and the
leading loadings (after sign alignment to the full-sample fit) yields
percentile confidence intervals for both the spectrum and the mode geometry.

### Marchenko–Pastur noise edge

For aspect ratio $\gamma = p/n$ and population covariance $I_p$, sample
eigenvalues of pure noise concentrate in

$$\bigl[(1-\sqrt{\gamma})^{2},\, (1+\sqrt{\gamma})^{2}\bigr].$$

Eigenvalues above the upper edge $\lambda_{+}$ are treated as signal
candidates. With $n = 184$ and $p = 9$, $\lambda_{+} \approx (1+\sqrt{\gamma})^{2}$.

## Scores, SPC, and reconstruction

Projecting onto the leading $r$ eigenvectors gives

$$T = X V_r, \qquad \hat{X} = T V_r^{\mathsf{T}} = X V_r V_r^{\mathsf{T}}.$$

The first score column $T_{\cdot 1}$ is monitored as a Shewhart series with
center line $\bar{T}_{\cdot 1}$ and control limits
$\bar{T}_{\cdot 1} \pm 3\, s_{T_{\cdot 1}}$, providing a single global
thickness statistic for process control.

The RMSE $\|X - \hat{X}\|_{\mathrm{F}} / \sqrt{np}$ quantifies how many
eigenmodes are needed to recover the thickness field to a chosen tolerance.

## Sensor pruning

Gauges that land close in the PC1–PC2 loading plane are candidates for
removal. Re-fitting PCA after dropping $G_5$ (redundant with $G_1$), $G_6$
(PC2 driver), or both, compares $\rho_1$ and the two-component RMSE and
turns the loading geometry into a metrology-design statement.

## Cross-validation

A $K$-fold split over sections fits PCA on the training rows and reconstructs
held-out rows in the training loading basis. Agreement between in-sample and
CV RMSE indicates that the truncated eigen-expansion generalizes.

## Physical plane model

On the schematic gauge coordinates $(x_j, y_j)$, each section is fit by

$$x_{ij} \approx a_i + b_i x_j + c_i y_j,$$

i.e. a global offset plus a linear tilt. Proximity of this model's RMSE to
two-component PCA supports reading PC1/PC2 as offset and gradient modes.

## Permutation tests

Independently permuting each gauge column destroys cross-gauge correlation.
The fraction of permuted $\rho_1$ values at least as large as the observed
$\rho_1$ is a one-sided $p$-value for a dominant common mode.

Because a large $\rho_1$ mechanically deflates $\rho_2$, PC2 is tested
instead by the spatial $R^{2}$ of its loadings against schematic $(x,y)$,
with gauge labels permuted under the null.

## Interpretation on the wafer

- **PC1** aggregates gauges with like-signed loadings and captures a
  nearly global thickness offset across the wafer.
- **PC2** introduces sign changes among gauges and encodes spatial
  gradients or bow/warp patterns.
- Because survey coordinates are unavailable, wafer-map figures use a
  schematic $3\times 3$ layout; mode identity itself is coordinate-free.
