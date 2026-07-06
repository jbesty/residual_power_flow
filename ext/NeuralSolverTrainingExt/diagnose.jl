# --- Diagnostics for a trained NeuralSolver ---
#
# Prints per-layer weight, gradient, and activation statistics along with
# normalised data sanity checks and LayerNorm parameter summaries.

function diagnose(solver::NeuralSolver, dataset)
    model = solver.model
    ps    = solver.parameters
    st    = solver.states
    t     = solver.transformation

    println("=" ^ 62)
    println("NeuralSolver Diagnostics")
    println("=" ^ 62)

    # 1. Normalised data sanity check
    println("\n--- Normalised data statistics ---")
    X_norm = normalize_controls(t, controls_matrix(dataset))
    Y_norm = normalize_voltages(t, voltages_matrix(dataset))

    println("Controls (normalised):")
    @printf("  global   mean = %+.4f  std = %.4f\n",
            mean(X_norm), std(X_norm; corrected=false))
    for j in 1:size(X_norm, 2)
        col = view(X_norm, :, j)
        @printf("  dim %3d  mean = %+.4f  std = %.4f\n",
                j, mean(col), std(col; corrected=false))
    end

    println("Voltages (normalised):")
    @printf("  global   mean = %+.4f  std = %.4f\n",
            mean(Y_norm), std(Y_norm; corrected=false))
    for j in 1:size(Y_norm, 2)
        col = view(Y_norm, :, j)
        @printf("  dim %3d  mean = %+.4f  std = %.4f\n",
                j, mean(col), std(col; corrected=false))
    end

    # 2. Per-layer weight statistics
    println("\n--- Per-layer weight statistics ---")
    for layer_name in keys(ps)
        layer_ps = ps[layer_name]
        isempty(keys(layer_ps)) && continue
        println("  $layer_name:")
        for param_name in keys(layer_ps)
            p = vec(layer_ps[param_name])
            @printf("    %-8s  L2 = %8.4f  mean = %+.4f  std = %.4f\n",
                    param_name, norm(p), mean(p), std(p; corrected=false))
        end
    end

    # 3. Per-layer gradient L2 norms
    println("\n--- Per-layer gradient L2 norms (MSE on training set) ---")
    X = Matrix(normalize_controls(t, controls_matrix(dataset))')
    Y = Matrix(normalize_voltages(t, voltages_matrix(dataset))')

    t_elapsed = @elapsed begin
        _, grads = Zygote.withgradient(ps) do p
            ŷ, _ = model(X, p, st)
            mean(abs2.(ŷ .- Y))
        end
    end
    t_elapsed > 5.0 && @printf("  (gradient computation took %.1f s)\n", t_elapsed)

    grad_ps = grads[1]
    for layer_name in keys(grad_ps)
        layer_grad = grad_ps[layer_name]
        layer_grad === nothing && continue
        isempty(keys(layer_grad)) && continue
        println("  $layer_name:")
        for param_name in keys(layer_grad)
            g = layer_grad[param_name]
            if g === nothing
                @printf("    %-8s  L2 = (no gradient)\n", param_name)
            else
                @printf("    %-8s  L2 = %.4e\n", param_name, norm(vec(g)))
            end
        end
    end

    # 4. Per-layer activation statistics
    println("\n--- Per-layer activation statistics (post-activation mean / std) ---")
    x_act = X
    for lname in keys(model.layers)
        layer    = model.layers[lname]
        ps_layer = ps[lname]
        st_layer = st[lname]
        y_act, _ = layer(x_act, ps_layer, st_layer)
        flat = vec(y_act)
        @printf("  %s  mean = %+.4f  std = %.4f  (size %s)\n",
                lname, mean(flat), std(flat; corrected=false), join(size(y_act), "×"))
        x_act = y_act
    end

    # 5. LayerNorm γ (scale) and β (bias) parameters
    println("\n--- LayerNorm γ (scale) and β (bias) parameters ---")
    found_any = false
    for layer_name in keys(ps)
        layer_ps = ps[layer_name]
        if haskey(layer_ps, :scale) && haskey(layer_ps, :bias)
            found_any = true
            γ = vec(layer_ps.scale)
            β = vec(layer_ps.bias)
            @printf("  %s:\n", layer_name)
            @printf("    γ (scale)  L2 = %8.4f  mean = %+.4f  std = %.4f\n",
                    norm(γ), mean(γ), std(γ; corrected=false))
            @printf("    β (bias)   L2 = %8.4f  mean = %+.4f  std = %.4f\n",
                    norm(β), mean(β), std(β; corrected=false))
        end
    end
    found_any || println("  (no LayerNorm layers found)")

    println("\n" * "=" ^ 62)
    return nothing
end
