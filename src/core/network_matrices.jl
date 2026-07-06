using LinearAlgebra: Diagonal, I

# ── Network matrix queries (Ybus, admittance matrices) ──────────────────────

function convert_branches_to_network_matrices(branch_models, from_buses, to_buses)
    Yff = [branch_model.Y_11 for branch_model in branch_models]
    Yft = [branch_model.Y_12 for branch_model in branch_models]
    Ytf = [branch_model.Y_21 for branch_model in branch_models]
    Ytt = [branch_model.Y_22 for branch_model in branch_models]

    n_buses = maximum([from_buses..., to_buses...])

    C_f = (one(eltype(Yff)) + zero(eltype(Yff)) * im) * I(n_buses)[from_buses, :]
    C_t = (one(eltype(Yff)) + zero(eltype(Yff)) * im) * I(n_buses)[to_buses, :]

    Yf = Diagonal(Yff) * C_f + Diagonal(Yft) * C_t
    Yt = Diagonal(Ytf) * C_f + Diagonal(Ytt) * C_t

    return C_f, C_t, Yf, Yt
end

function compute_Ybus_all(
    branch_models,
    from_buses,
    to_buses,
    shunt_models = [],
    shunt_buses = [],
)
    C_f, C_t, Yf, Yt =
        convert_branches_to_network_matrices(branch_models, from_buses, to_buses)
    Ybus = C_f' * Yf + C_t' * Yt

    for (shunt_model, shunt_bus) in zip(shunt_models, shunt_buses)
        Ybus[shunt_bus, shunt_bus] += shunt_model.G + im * shunt_model.B
    end
    return Ybus, Yf, Yt
end

function compute_Ybus(
    branch_models,
    from_buses,
    to_buses,
    shunt_models = [],
    shunt_buses = [],
)
    return compute_Ybus_all(branch_models, from_buses, to_buses, shunt_models, shunt_buses)[1]
end
