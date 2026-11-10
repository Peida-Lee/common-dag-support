### K.6 Root-target compatibility and measured design

Let R be known structural roots, N the remaining coordinates and \(J\subseteq R\) the known target set. For each j, intersect its candidates across all available contrasts and impose \((a_j)_R=e_j\). Assume these sets are nonempty. Write \(C=A_{NR}\) for the root-to-N response block and complete its unobserved columns freely. The full root-response block is \(A_{:R}=\binom{I_R}{C}\). Coordinates in N may themselves have empty parent sets. The model permits arbitrary real coefficients and positive-definite correlated error covariances, with every acyclic subgraph on N allowed.

For every acyclic zero-diagonal coefficient matrix T on N, put

$$
B(T)=\begin{pmatrix}0&0\\(I-T)C&T\end{pmatrix},\qquad
A(T)=\begin{pmatrix}I&0\\C&(I-T)^{-1}\end{pmatrix}.
$$

Direct multiplication gives \((I-B)A=I\). A topological order of T, preceded by R, orders B. Every root response is C, independently of T. Conversely, the root-column equation \((I-B)A_{:R}=I_{:R}\) forces \(B_{NR}=(I-B_{NN})C\), so this parameterization covers every acyclic representation with the fixed root responses.

For \(u\in J\) and \(w\in N\), its direct coefficient is

$$
B_{wu}=C_{wu}-\sum_{m\in N\setminus\{w\}}T_{wm}C_{mu}.
$$

If \(C_{wu}=0\), take T=0. If some \(C_{mu}\ne0\), take only \(T_{wm}=C_{wu}/C_{mu}\) nonzero. This is acyclic and has \(B_{wu}=0\). If all other \(C_{mu}=0\), the identity forces \(B_{wu}=C_{wu}\). This proves Proposition 2 for fixed columns and, by its independent candidate choices, for their intersection support. If \(u\in R\setminus J\), the free completion \(C_{:u}=0\) leaves the full response column \(A_{:u}=e_u\) and gives a missing-edge representation.

For every constructed W, set \(\Omega_g=W\Sigma_gW^\top\) using the original positive-definite observed covariances. These matrices remain positive definite. Each selected target factorization has \(S_j=a_jh_j^\top+h_ja_j^\top\) and \(Wa_j=e_j\). Hence

$$
WS_jW^\top=e_j(Wh_j)^\top+(Wh_j)e_j^\top,
$$

which preserves (1) and all original covariance experiments. This response-preserving construction relates to the coefficient and noise-covariance ambiguity studied in LLC. Proposition 2 supplies the explicit acyclic root-target criterion. [22]

The zero-response condition is a population equality. When \(|N|\ge2\), an isolated response with \(C_{wu}\ne0\) can be approached by missing-edge experiments. Choose \(m\ne w\) and replace its zero u response by \(\delta\). In each contrast targeting u, replace \(a_u\) by \(a_u+\delta e_m\), keeping its partner h fixed. Every observed covariance converges to its original value and remains positive definite for sufficiently small \(\delta\). The construction \(T_{wm}=C_{wu}/\delta\) gives an acyclic missing-edge model for every nonzero \(\delta\). Thus the missing-edge class has zero covariance separation at this isolated-response experiment when coefficients and error scales are unrestricted. A finite-sample separation guarantee needs further restrictions. Its population zero pattern alone supplies no positive gap.

The A1 plan has eight known target coordinates among ten roots. Its nine readouts have no own-target acquisition. With all 19 coordinates retained, a missing row has 17 possible parents, while the eight available target columns have rank at most eight. The direct spanning-target shortcut therefore requires further information in this full-parent class. Proposition 2 explains the remaining mediated ambiguity even when all root responses are given. The measured procedures and their reference semantics are specified in Appendix D.

An exact rational check uses 24 systems of sizes four, five and six. Each checks targeted- and untargeted-root queries, target-factor recovery, acyclic zero-diagonal representations, all root responses and transformed target blocks. The existing affine-flat solver agrees on all 48 queries. The same systems verify the general T parameterization. The reproduction code supplies every matrix and decision.

