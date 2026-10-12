/**
 * math/linalg.js
 * Responsibility: small dense linear algebra for the model heads of Section 5 (p = 5 coefficients): square
 * matrices are row-major Float64Array of length p*p. Solve, inverse, log-determinant, symmetric
 * eigendecomposition and projection onto the positive semidefinite cone. Pure functions, no allocation
 * beyond their results.
 */

/**
 * p x p identity times a scalar.
 * @param {number} p size
 * @param {number} [s=1] diagonal value
 * @returns {Float64Array}
 * @see Section 5.2 (I / s0^2 terms)
 */
export function eye(p, s = 1) { const A = new Float64Array(p * p); for (let i = 0; i < p; i++) A[i * p + i] = s; return A; }

/**
 * Diagonal matrix from a vector.
 * @param {ArrayLike<number>} d diagonal
 * @returns {Float64Array}
 * @see Section 5.2, diag(tau_q^2)
 */
export function diag(d) { const p = d.length, A = new Float64Array(p * p); for (let i = 0; i < p; i++) A[i * p + i] = d[i]; return A; }

/**
 * Diagonal of a square matrix.
 * @param {Float64Array} A matrix
 * @param {number} p size
 * @returns {Float64Array}
 * @see Section 5.2 (coordinatewise variances)
 */
export function diagOf(A, p) { const d = new Float64Array(p); for (let i = 0; i < p; i++) d[i] = A[i * p + i]; return d; }

/**
 * Sum of two matrices (or a matrix and a scaled one).
 * @param {Float64Array} A first
 * @param {Float64Array} B second
 * @param {number} [b=1] scale of B
 * @returns {Float64Array} A + b B
 * @see Section 5.2
 */
export function add(A, B, b = 1) { const C = new Float64Array(A.length); for (let i = 0; i < A.length; i++) C[i] = A[i] + b * B[i]; return C; }

/**
 * Matrix product of two p x p matrices.
 * @param {Float64Array} A left
 * @param {Float64Array} B right
 * @param {number} p size
 * @returns {Float64Array}
 * @see Section 5.2, V_dp = s_S^2 A A^T
 */
export function mul(A, B, p) {
  const C = new Float64Array(p * p);
  for (let i = 0; i < p; i++) for (let k = 0; k < p; k++) { const a = A[i * p + k]; if (a === 0) continue; for (let j = 0; j < p; j++) C[i * p + j] += a * B[k * p + j]; }
  return C;
}

/**
 * Matrix-vector product.
 * @param {Float64Array} A p x p matrix
 * @param {ArrayLike<number>} x vector
 * @param {number} p size
 * @returns {Float64Array}
 * @see Section 5.2, w = A S
 */
export function mv(A, x, p) { const y = new Float64Array(p); for (let i = 0; i < p; i++) { let s = 0; for (let j = 0; j < p; j++) s += A[i * p + j] * x[j]; y[i] = s; } return y; }

/**
 * Transpose.
 * @param {Float64Array} A matrix
 * @param {number} p size
 * @returns {Float64Array}
 */
export function transpose(A, p) { const B = new Float64Array(p * p); for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) B[j * p + i] = A[i * p + j]; return B; }

function lu(A, p) {
  const M = Float64Array.from(A), piv = new Int32Array(p); let sign = 1, sing = false;
  for (let i = 0; i < p; i++) piv[i] = i;
  for (let c = 0; c < p; c++) {
    let best = c, bv = Math.abs(M[c * p + c]);
    for (let r = c + 1; r < p; r++) { const v = Math.abs(M[r * p + c]); if (v > bv) { bv = v; best = r; } }
    if (bv === 0) { sing = true; continue; }
    if (best !== c) { for (let j = 0; j < p; j++) { const t = M[c * p + j]; M[c * p + j] = M[best * p + j]; M[best * p + j] = t; } const t = piv[c]; piv[c] = piv[best]; piv[best] = t; sign = -sign; }
    const d = M[c * p + c];
    for (let r = c + 1; r < p; r++) { const f = (M[r * p + c] /= d); if (f !== 0) for (let j = c + 1; j < p; j++) M[r * p + j] -= f * M[c * p + j]; }
  }
  return { M, piv, sign, sing };
}

/**
 * Solve A x = b by LU decomposition with partial pivoting.
 * @param {Float64Array} A p x p matrix
 * @param {ArrayLike<number>} b right-hand side
 * @param {number} p size
 * @returns {Float64Array} x
 * @see Section 5.2 (Newton steps of every fit)
 */
export function solve(A, b, p) {
  const { M, piv } = lu(A, p), x = new Float64Array(p);
  for (let i = 0; i < p; i++) { let s = b[piv[i]]; for (let j = 0; j < i; j++) s -= M[i * p + j] * x[j]; x[i] = s; }
  for (let i = p - 1; i >= 0; i--) { let s = x[i]; for (let j = i + 1; j < p; j++) s -= M[i * p + j] * x[j]; x[i] = s / M[i * p + i]; }
  return x;
}

/**
 * Inverse of a p x p matrix.
 * @param {Float64Array} A matrix
 * @param {number} p size
 * @returns {Float64Array}
 * @see Section 5.2, A = (J+ + I/s0^2)^{-1}
 */
export function inv(A, p) {
  const { M, piv } = lu(A, p), X = new Float64Array(p * p), col = new Float64Array(p);
  for (let c = 0; c < p; c++) {
    for (let i = 0; i < p; i++) { let s = piv[i] === c ? 1 : 0; for (let j = 0; j < i; j++) s -= M[i * p + j] * col[j]; col[i] = s; }
    for (let i = p - 1; i >= 0; i--) { let s = col[i]; for (let j = i + 1; j < p; j++) s -= M[i * p + j] * col[j]; col[i] = s / M[i * p + i]; }
    for (let i = 0; i < p; i++) X[i * p + c] = col[i];
  }
  return X;
}

/**
 * Natural logarithm of |det A| (numpy.linalg.slogdet()[1]).
 * @param {Float64Array} A matrix
 * @param {number} p size
 * @returns {number}
 * @see Remark 4.4 (Laplace evidence of each component)
 */
export function logdet(A, p) { const { M, sing } = lu(A, p); if (sing) return -Infinity; let s = 0; for (let i = 0; i < p; i++) s += Math.log(Math.abs(M[i * p + i])); return s; }

/**
 * Eigendecomposition of a symmetric matrix by cyclic Jacobi rotations.
 * @param {Float64Array} S symmetric p x p matrix
 * @param {number} p size
 * @returns {{values:Float64Array, vectors:Float64Array}} vectors column-wise (row-major p x p, column i is eigenvector i)
 * @see Section 5.2 (projection onto the positive semidefinite cone)
 */
export function eigh(S, p) {
  const a = Float64Array.from(S), V = eye(p);
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) off += a[i * p + j] ** 2;
    if (off < 1e-30 * (1 + a.reduce((s, x) => s + x * x, 0))) break;
    for (let r = 0; r < p; r++) for (let q = r + 1; q < p; q++) {
      const arq = a[r * p + q]; if (Math.abs(arq) < 1e-300) continue;
      const th = (a[q * p + q] - a[r * p + r]) / (2 * arq);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < p; k++) { const akr = a[k * p + r], akq = a[k * p + q]; a[k * p + r] = c * akr - s * akq; a[k * p + q] = s * akr + c * akq; }
      for (let k = 0; k < p; k++) { const ark = a[r * p + k], aqk = a[q * p + k]; a[r * p + k] = c * ark - s * aqk; a[q * p + k] = s * ark + c * aqk; }
      for (let k = 0; k < p; k++) { const vkr = V[k * p + r], vkq = V[k * p + q]; V[k * p + r] = c * vkr - s * vkq; V[k * p + q] = s * vkr + c * vkq; }
    }
  }
  return { values: diagOf(a, p), vectors: V };
}

/**
 * Symmetrize and project onto the cone of matrices with eigenvalues at least `floor` (ghb_v4._psd).
 * @param {Float64Array} H matrix
 * @param {number} p size
 * @param {number} floor smallest eigenvalue kept
 * @returns {Float64Array}
 * @see Section 5.2, J+ (J projected onto the positive semidefinite cone)
 */
export function psd(H, p, floor) {
  const S = new Float64Array(p * p);
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) S[i * p + j] = (H[i * p + j] + H[j * p + i]) / 2;
  const { values, vectors: U } = eigh(S, p), out = new Float64Array(p * p);
  for (let k = 0; k < p; k++) { const l = Math.max(values[k], floor); for (let i = 0; i < p; i++) { const ui = U[i * p + k] * l; for (let j = 0; j < p; j++) out[i * p + j] += ui * U[j * p + k]; } }
  return out;
}
