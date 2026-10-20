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
covariance spectrum irrespective of their physical coupling.

A parallel **median / MAD** scaling,

$$X_{ij}^{\mathrm{rob}} = \frac{(X_{\mathrm{raw}})_{ij} - m_j}{1.4826\,\mathrm{MAD}_j},$$

is used to test sensitivity to heavy-tailed thickness spikes. Outlying
sections are additionally flagged by a Mahalanobis gate on the
$z$-scored rows: keep row $i$ when
$d_i^{2} = (\mathbf{x}_i-\bar{\mathbf{x}})^{\mathsf{T}} C^{-1}(\mathbf{x}_i-\bar{\mathbf{x}})$
does not exceed the $97.5\%$ critical value of a $\chi^{2}_{9}$ law.

## Sample covariance and eigenmodes

The unbiased sample covariance of the standardized gauges is

$$C = \frac{1}{n-1}\, X^{\mathsf{T}} X \in \mathbb{R}^{9 \times 9}.$$

Principal component analysis solves the symmetric eigenproblem

$$C\, v_k = \lambda_k\, v_k, \qquad \lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_9 \ge 0,$$

where the orthonormal eigenvectors $v_k$ are the **loadings** (sensor
weights of each mode) and the eigenvalues $\lambda_k$ are the variances
along those modes. The fraction of total variance explained by component
$k$ is

$$\rho_k = \frac{\lambda_k}{\sum_{m=1}^{9}\lambda_m}.$$

### SVD equivalence

Write $\tilde{X} = X / \sqrt{n-1}$ and compute the thin SVD
$\tilde{X} = U\Sigma V^{\mathsf{T}}$. Then the columns of $V$ coincide with
the eigenvectors of $C$, and $\lambda_k = \Sigma_{kk}^{2}$. The
implementation exposes both routes and asserts numerical agreement.

### Bootstrap uncertainty

Resampling the $n$ rows with replacement and recomputing $\{\rho_k\}$ yields
percentile confidence intervals for the spectrum, used in the scree figure.

## Scores and reconstruction

Projecting the standardized sections onto the leading $r$ eigenvectors
gives the score matrix

$$T = X V_r, \qquad V_r = [v_1\,\cdots\,v_r],$$

and the truncated reconstruction

$$\hat{X} = T V_r^{\mathsf{T}} = X V_r V_r^{\mathsf{T}}.$$

The root-mean-square error $\|X - \hat{X}\|_{\mathrm{F}} / \sqrt{np}$
quantifies how many eigenmodes are needed to recover the thickness field
to a chosen tolerance.

## Interpretation on the wafer

- **PC1** aggregates gauges with like-signed loadings and captures a
  nearly global thickness offset across the wafer.
- **PC2** (and higher modes) introduce sign changes among gauges and
  therefore encode spatial gradients or bow/warp patterns.
- Gauge pairs that land close to each other in the PC1–PC2 loading plane
  contribute similarly to the retained modes and are candidates for
  physically redundant sensors.
- Because survey coordinates are unavailable, wafer-map figures use a
  schematic $3\times 3$ layout; mode identity itself is coordinate-free.
