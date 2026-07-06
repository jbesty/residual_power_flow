using StaticArrays: SVector

export compute_current_injection
export n_controls

# ── n_controls: number of control inputs per component type ──────────────────

n_controls(::SynchronousMachineStatic) = 2
n_controls(::ZIPLoad)                  = 2
n_controls(::Shunt)                    = 0
n_controls(::Branch)                   = 0
n_controls(::AsymmetricBranch)         = 0

# ── Single-bus current injections ─────────────────────────────────────────────

function compute_current_injection(
    component::SynchronousMachineStatic,
    voltage::Real,
    control::AbstractVector{<:Real},
)
    T_M = control[1]
    V_ref = control[2]
    i_d = T_M / voltage
    i_q = -component.K_DV * (V_ref - voltage)
    return SVector(i_d, i_q)
end

function compute_current_injection(
    component::ZIPLoad,
    voltage::Real,
    control::AbstractVector{<:Real},
)
    # The active and reactive power values are positive when consuming
    P = control[1]
    Q = control[2]
    ZIP_load_factor =
        component.a +
        component.b * (voltage / component.V_0) +
        component.c * (voltage / component.V_0)^2
    s = -complex(P, Q) * ZIP_load_factor
    i_complex = conj(s / voltage)
    return SVector(i_complex.re, i_complex.im)
end

function compute_current_injection(
    component::Shunt,
    voltage::Real,
    control::AbstractVector{<:Real},
)
    # Shunt conductance and susceptance
    i_complex = -complex(component.G, component.B) * voltage
    return SVector(i_complex.re, i_complex.im)
end

# ── Two-bus current injections ────────────────────────────────────────────────
#
# Returns (i_from, i_to) where each is a 2-element [i_d, i_q] SVector.
# v_from_mag and v_to_mag are voltage magnitudes at the from-bus and to-bus.
# theta = θ_to - θ_from (the branch angle variable).
# Convention: branch_currents vector in PowerFlowState stores from-end first.

function compute_current_injection(
    component::Union{Branch, AsymmetricBranch},
    v_from_mag::Real,
    v_to_mag::Real,
    theta::Real,
)
    sin_theta, cos_theta = sincos(theta)
    # From-end: local reference at from-bus (angle 0), to-bus at angle +theta
    v_from             = complex(v_from_mag, zero(v_from_mag))
    v_to_from_frame    = v_to_mag * complex(cos_theta, sin_theta)
    i_from = -(component.Y_11 * v_from + component.Y_12 * v_to_from_frame)
    # To-end: local reference at to-bus (angle 0), from-bus at angle -theta
    v_to               = complex(v_to_mag, zero(v_to_mag))
    v_from_to_frame    = v_from_mag * complex(cos_theta, -sin_theta)
    i_to = -(component.Y_22 * v_to + component.Y_21 * v_from_to_frame)
    return (SVector(i_from.re, i_from.im), SVector(i_to.re, i_to.im))
end


function compute_control_from_current_injection(
    component::SynchronousMachineStatic,
    voltage::Real,
    current_injection::AbstractVector{<:Real},
)
    i_d = current_injection[1]
    i_q = current_injection[2]
    T_M = i_d * voltage
    V_ref = voltage - i_q / component.K_DV
    return [T_M, V_ref]
end
