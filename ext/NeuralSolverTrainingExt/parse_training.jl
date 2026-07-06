# --- Config-driven training recipes ---
#
# parse_training turns a plain table (e.g. a TOML [training] block) into a
# training recipe, so the optimiser schedule is data instead of hard-coded:
#
#   kind = "schedule"  -> phase = [{opt=...}, ...]              -> NeuralTraining
#                                                                  (or Vector{NeuralTraining})
#   kind = "best_of"   -> [[candidate]] each with phase = [...] -> BestOf
#
# Phase tables: opt ∈ {"adam","lbfgs"}; "iters" sets the iteration budget; Adam
# reads "lr", L-BFGS reads "m"/"g_tol"; both accept "print_every". Absent keys
# fall back to the AdamTraining / LBFGSTraining constructor defaults.

function _parse_phase(p::AbstractDict)
    g(k, d) = haskey(p, k) ? p[k] : d
    haskey(p, "opt") || throw(ArgumentError("training phase is missing the \"opt\" key"))
    opt = lowercase(String(p["opt"]))
    if opt == "adam"
        return AdamTraining(
            learning_rate = Float64(g("lr", 1.0e-3)),
            n_epochs      = Int(g("iters", 1000)),
            print_every   = Int(g("print_every", 0)),
        )
    elseif opt == "lbfgs"
        return LBFGSTraining(
            m              = Int(g("m", 10)),
            max_iterations = Int(g("iters", 1000)),
            g_tol          = Float64(g("g_tol", NaN)),
            print_every    = Int(g("print_every", 0)),
        )
    else
        throw(ArgumentError("unknown training optimiser $(repr(opt)); supported: \"adam\", \"lbfgs\""))
    end
end

function _parse_phase_list(d::AbstractDict)
    haskey(d, "phase") || throw(ArgumentError("training schedule is missing the \"phase\" array"))
    phases = NeuralTraining[_parse_phase(p) for p in d["phase"]]
    isempty(phases) && throw(ArgumentError("training \"phase\" array must be non-empty"))
    return phases
end

function parse_training(d::AbstractDict)
    kind = lowercase(String(haskey(d, "kind") ? d["kind"] : "schedule"))
    if kind == "schedule"
        phases = _parse_phase_list(d)
        return length(phases) == 1 ? phases[1] : phases
    elseif kind == "best_of"
        haskey(d, "candidate") || throw(ArgumentError(
            "best_of training is missing [[candidate]] entries"))
        candidates = [_parse_phase_list(c) for c in d["candidate"]]
        select_by = Symbol(lowercase(String(haskey(d, "select_by") ? d["select_by"] : "val_loss")))
        return BestOf(candidates; select_by = select_by)
    else
        throw(ArgumentError("unknown training kind $(repr(kind)); supported: \"schedule\", \"best_of\""))
    end
end
