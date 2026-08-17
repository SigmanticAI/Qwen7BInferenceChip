"""int_model.py — pure-integer, synthesizable-shaped bit-exact model of the KVQ
datapath math (scale_from_amax / quant_one / dequant_one / f16->f32 widen).

This mirrors, op-for-op, the integer SystemVerilog I intend to emit in
cq_fp_pkg.sv (no `real`, no float). It is validated EXHAUSTIVELY against the
golden arbiter (golden/apex_golden/cq_codec.py) by prove.py in this directory.

Decompose fp16 magnitude bits b -> value = M * 2**E (M in [0,2047]):
  exp==0      : M=man,       E=-24     (zero / subnormal)
  1<=exp<=30  : M=1024+man,  E=exp-25  (normal)
  exp==31     : inf/nan, OUT OF CONTRACT (codec data is finite)

All rounding is round-half-to-even (RNE) computed from an EXACT integer
remainder — no float64 intermediate, so there is no double-rounding to reason
about: this model rounds the exact rational once, and prove.py confirms that
single result equals the golden float64->narrow chain over the whole domain.
"""

def qmax_of(bits):
    return (1 << (bits - 1)) - 1

def qmin_of(bits):
    return -(1 << (bits - 1))

EPS_F16 = 0x0400  # 2**-14, the smallest normal fp16 == EPS exactly


def _decode_f16(b):
    """b: 16-bit int -> (sign, M, E) with |value| = M*2**E, M in [0,2047]."""
    b &= 0xFFFF
    sign = (b >> 15) & 1
    exp = (b >> 10) & 0x1F
    man = b & 0x3FF
    if exp == 0:
        return sign, man, -24
    if exp <= 30:
        return sign, 1024 + man, exp - 25
    # exp==31: out of contract; clamp to a large finite (defensive only)
    return sign, 2047, 5   # ~65504-ish; never exercised by codec data


def _rne_div(N, D):
    """round-half-to-even of the exact nonneg rational N/D, BOUNDED — the
    op-for-op mirror of cq_fp_pkg::rne_div_bounded (F1.5): exact for
    floor(N/D) < 2**12, else saturates to 2**12. scale_from_amax's quotient
    is in [1024, 2048] (always exact); quant_one clamps to |code| <= 128
    afterwards, so the saturated value clamps to the same code the exact one
    would. D == 0 (out of contract) deterministically saturates, like the SV."""
    if D == 0 or N >= (D << 12):
        return 1 << 12
    q = N // D            # == the SV's 12-step restoring walk (exact here)
    r = N - q * D
    two_r = r << 1
    if two_r > D:
        q += 1
    elif two_r == D:
        if q & 1:
            q += 1
    return q


# ---------------------------------------------------------------------------
# scale_from_amax : s = max(amax/qmax, EPS), RNE to fp16 bits
# ---------------------------------------------------------------------------
def scale_from_amax(amax_f16, bits):
    amax_f16 &= 0xFFFF
    q = qmax_of(bits)
    sgn, Ma, Ea = _decode_f16(amax_f16)   # amax is sign-cleared by the amax walk
    if Ma == 0 or sgn:
        return EPS_F16                     # amax<=0 -> s=EPS (neg: a<EPS too)

    # binade j = floor(log2(Ma/q))  s.t.  q*2^j <= Ma < q*2^(j+1)
    j = 0
    if Ma >= q:
        while (q << (j + 1)) <= Ma:
            j += 1
    else:
        j = -1
        while (Ma << (-j)) < q:
            j -= 1
    e = j + Ea                             # binade of a = amax/q

    # EPS floor: a < 2^-14  <=>  e <= -15  -> result is EPS (0x0400)
    if e <= -15:
        return EPS_F16

    # normal fp16: m = RNE( Ma * 2^(10-j) / q ) in [1024,2048]
    shift = 10 - j
    if shift >= 0:
        m = _rne_div(Ma << shift, q)
    else:
        m = _rne_div(Ma, q << (-shift))
    if m == 2048:
        e += 1
        m = 1024
    ef = e + 15                            # >=1 (result >= EPS => normal)
    mant = m - 1024
    return ((ef & 0x1F) << 10) | (mant & 0x3FF)


# ---------------------------------------------------------------------------
# quant_one : q = clamp(RNE(x/s), qmin, qmax)   signed code
# ---------------------------------------------------------------------------
def quant_one(x_f16, s_f16, bits):
    sx, Mx, Ex = _decode_f16(x_f16)
    ss, Ms, Es = _decode_f16(s_f16)        # contract scale: positive, Ms!=0
    if Mx == 0:
        return 0
    shift = Ex - Es
    if shift >= 0:
        mag = _rne_div(Mx << shift, Ms)
    else:
        mag = _rne_div(Mx, Ms << (-shift))
    # quotient sign = sign(x) XOR sign(s): a (defensive, out-of-contract)
    # negative scale flips the code sign exactly as the golden float divide
    # does; RNE on the magnitude == RNE on the signed value (symmetric ties).
    q = -mag if (sx ^ ss) else mag
    lo, hi = qmin_of(bits), qmax_of(bits)
    if q < lo:
        q = lo
    if q > hi:
        q = hi
    return q


# ---------------------------------------------------------------------------
# dequant_one : xhat = code * s -> fp32 bits (EXACT, product <= 19 bits)
# ---------------------------------------------------------------------------
def _int_to_f32(sign, a, E):
    """|value| = a*2**E exactly, a integer >=0 with <=24 significant bits."""
    if a == 0:
        return sign << 31                  # signed zero
    k = a.bit_length() - 1                 # MSB position
    ef = E + k + 127                       # fp32 biased exponent
    if ef <= 0:
        # subnormal / underflow fp32 (not exercised by codec, defensive)
        sh = 1 - ef
        mant = (a << (23 - k)) >> sh if (23 - k) >= sh else (a << (23 - k))
        return (sign << 31) | (mant & 0x7FFFFF)
    if ef >= 255:
        return (sign << 31) | (0xFF << 23) # inf (not exercised)
    mant = (a << (23 - k)) & 0x7FFFFF if (23 - k) >= 0 else (a >> (k - 23)) & 0x7FFFFF
    return (sign << 31) | ((ef & 0xFF) << 23) | mant


def dequant_one(code, s_f16):
    ss, Ms, Es = _decode_f16(s_f16)
    P = code * Ms
    sign = 1 if (P < 0) else (1 if (code < 0 and Ms == 0) else 0)
    # signed-zero: product 0 keeps sign = code_sign XOR s_sign
    if P == 0:
        sign = ((1 if code < 0 else 0) ^ ss)
        return sign << 31
    a = -P if P < 0 else P
    sign = 1 if (code < 0) ^ (ss) else 0
    return _int_to_f32(sign, a, Es)


# ---------------------------------------------------------------------------
# widen f16 -> f32 (outlier lane, EXACT, purely combinational)
# ---------------------------------------------------------------------------
def widen_f16_f32(x_f16):
    sgn, M, E = _decode_f16(x_f16)
    if M == 0:
        return sgn << 31                   # preserve signed zero
    return _int_to_f32(sgn, M, E)
