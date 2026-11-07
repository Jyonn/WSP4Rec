"""Optional exact-real certificates; these do not certify float32 execution.

The policy is independently encoded here. No sampled examples are used to prove
universality. Parameters are interpreted as their exact stored float32 values.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path
import time
import numpy as np

_SHAPES = {'W1': (22, 32), 'b1': (32,), 'W2': (32, 32),
           'b2': (32,), 'W3': (32, 1), 'b3': (1,)}


def _state(state):
    if set(state) != set(_SHAPES):
        raise ValueError('exactly W1,b1,W2,b2,W3,b3 required')
    result = {}
    for key, shape in _SHAPES.items():
        a = np.asarray(state[key], dtype=np.float32)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f'invalid {key}')
        result[key] = a.copy()
    return result


def _fraction(value):
    return Fraction(float(value))


def source_npz_sha256(path):
    """Hash the actual source artifact, rather than a reserialized state."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def certify_written(state, task, permutation=None, timeout_ms=30000):
    """Prove |NN-policy| <= 1e-6 on every legal real/binary input.

    ``permutation`` means the deployed network consumes ``x[..., permutation]``.
    A SAT witness is exact (rational strings) as well as approximately numeric.
    UNKNOWN and missing Z3 are explicitly unverified outcomes.
    """
    started = time.monotonic()
    s = _state(state)
    if task not in (0, 1, 2):
        raise ValueError('task must be 0, 1, or 2')
    if timeout_ms <= 0:
        raise ValueError('timeout_ms must be positive')
    if permutation is not None:
        p = np.asarray(permutation)
        if p.shape != (22,) or not np.array_equal(np.sort(p), np.arange(22)):
            raise ValueError('permutation must contain each index 0..21 once')
        s['W1'] = s['W1'][np.argsort(p)]
    base = {'verified': False, 'arithmetic': 'exact real arithmetic of float32 parameters',
            'tolerance': '1/1000000', 'task': int(task),
            'domain': '20 independent binary fields; budget [1/5,1]; runtime [1/10,1]'}
    try:
        import z3
    except ImportError:
        return dict(base, status='unavailable', reason='optional z3-solver dependency unavailable',
                    elapsed_seconds=time.monotonic()-started)
    base['z3_version'] = z3.get_version_string()
    def q(v):
        f = _fraction(v)
        return z3.RealVal(f'{f.numerator}/{f.denominator}')
    flags = {i: z3.Bool(f'feature_{i}') for i in range(22) if i not in (14, 15)}
    x = [z3.If(flags[i], 1, 0) if i in flags else z3.Real(f'feature_{i}') for i in range(22)]
    def relu(v):
        return z3.If(v > 0, v, 0)
    def layer(inputs, weight, bias, activate):
        values = []
        for j in range(len(bias)):
            # Skip exact zero coefficients only, never small nonzero terms.
            v = z3.Sum([q(bias[j])] + [q(weight[i,j])*a for i,a in enumerate(inputs) if weight[i,j] != 0])
            values.append(z3.simplify(relu(v) if activate else v))
        return values
    h1 = layer(x, s['W1'], s['b1'], True)
    h2 = layer(h1, s['W2'], s['b2'], True)
    output = layer(h2, s['W3'], s['b3'], False)[0]
    def indicator(condition):
        return z3.If(condition, 1, 0)
    matches = [z3.And(flags[i], flags[i+4]) for i in range(4)]
    target = z3.Sum([w*indicator(m) for w,m in zip(((3,2,4,1),(1,4,2,3),(4,1,3,2))[task], matches)])
    target += 2*indicator(z3.And(flags[8], flags[9])) + indicator(z3.And(flags[16], flags[17]))
    target -= 4*relu(x[15]-x[14])
    graphic = z3.And(flags[10], flags[11])
    if task >= 1:
        graphic = z3.And(graphic, z3.Not(z3.And(flags[12], flags[13])))
    target -= 40*(indicator(graphic)+indicator(z3.And(flags[18], flags[19])))
    if task >= 1:
        target -= 40*indicator(z3.And(flags[8], z3.Not(flags[9])))
    if task == 2:
        target += 5*indicator(z3.And(matches[0], flags[8], flags[9]))
    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))
    solver.add(x[14] >= z3.RealVal('1/5'), x[14] <= 1,
               x[15] >= z3.RealVal('1/10'), x[15] <= 1)
    error = output-target
    solver.add(z3.Or(error > z3.RealVal('1/1000000'), error < z3.RealVal('-1/1000000')))
    verdict = solver.check()
    result = dict(base, status=str(verdict), elapsed_seconds=time.monotonic()-started)
    if verdict == z3.unsat:
        result['verified'] = True
    elif verdict == z3.unknown:
        result['reason'] = solver.reason_unknown()
    else:
        model = solver.model()
        def evaluated(expr):
            v = model.eval(expr, model_completion=True)
            return v, float(v.as_long()) if z3.is_int_value(v) else float(Fraction(v.numerator_as_long(), v.denominator_as_long()))
        exact, numeric = zip(*(evaluated(v) for v in x))
        out, out_num = evaluated(output)
        expected, expected_num = evaluated(target)
        err, err_num = evaluated(error)
        result['counterexample'] = {'features': list(numeric), 'features_exact': list(map(str,exact)),
                                    'network_output': out_num, 'oracle_output': expected_num,
                                    'error': err_num, 'network_output_exact': str(out),
                                    'oracle_output_exact': str(expected), 'error_exact': str(err)}
    return result


def fold_certificate(source, edited):
    """Recognize status folding and bound its exact-real output discrepancy.

    For every real input, compares source with status flags forced to one against
    edited with arbitrary flags. Absolute-matrix propagation uses the global
    1-Lipschitz property of ReLU; bias addition rounding is included exactly.
    """
    a, b = _state(source), _state(edited)
    recognized = (np.array_equal(a['W1'][:20], b['W1'][:20])
                  and np.all(b['W1'][20:22] == 0)
                  and all(np.array_equal(a[k], b[k]) for k in ('W2','b2','W3','b3')))
    result = {'recognized_fold': bool(recognized),
              'arithmetic': 'exact real arithmetic of float32 parameters',
              'float32_execution_certified': False}
    if not recognized:
        return dict(result, status='not_recognized')
    e1 = [abs(_fraction(b['b1'][j])-(_fraction(a['b1'][j])+_fraction(a['W1'][20,j])+_fraction(a['W1'][21,j]))) for j in range(32)]
    e2 = [sum((e1[i]*abs(_fraction(a['W2'][i,j])) for i in range(32)), Fraction(0)) for j in range(32)]
    bound = sum((e2[j]*abs(_fraction(a['W3'][j,0])) for j in range(32)), Fraction(0))
    return dict(result, status='recognized', output_error_bound_exact=str(bound),
                output_error_bound=float(bound), first_layer_bias_error_exact=list(map(str,e1)),
                scope='source status flags forced to 1 versus edited arbitrary flags, all real other inputs',
                bias_rounding_included=True)
