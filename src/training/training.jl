using Flux
import Flux.Tracker: Params, gradient, data, update!

function _update_params_full!(pscb::AbstractArray, xs; kwargs...)
  #  @show pscb, kwargs
    for c in pscb
        apply!(c, xs; kwargs...)
    end
end
  
_update_params_full!(pscb, xs; kwargs...) = _update_params_full!([pscb], xs; kwargs...)
  
# ------------------------------------

#from Flux/optimise/train.jl
# function update!(opt, x, x̄)
    # update!(x, -Flux.Optimise.apply!(opt, x, data(x̄)))
# end
  
#from Flux/optimise/train.jl
# function update!(opt, xs::Params, gs)
    # for x in xs
        # update!(opt, x, gs[x])
    # end
# end

# macro interrupts(ex)
    # :(try $(esc(ex))
    # catch e
        # e isa InterruptException || rethrow()
        # throw(e)
    # end)
# end

#from Flux/optimise/train.jl
# struct StopException <: Exception end

#from Flux/optimise/train.jl
# function stop()
    # throw(StopException())
# end

function trainepochs!(loss, ps, data, opt;
    epochs = 1, 
    pscb = ParamsIdentity(), 
    epochcb = nothing, 
    batchcb = nothing)

    isnothing(epochcb) || epochcb(epoch = 0, epoch_start_time = time(), ps = ps)

    for e in 1:epochs
        epoch_start_time = time()

        epochdata = data isa Function ? data(e) : data
    
        for (b, d) in enumerate(epochdata)
            try
                gs = gradient(ps) do
                    loss(d...)
                end
                # _update_params_full!(pscb, ps, epoch = e, batch = b)
                isnothing(batchcb) || batchcb(ps, epoch = e, batch = b, batch_size = size(d[1])[end])
                update!(opt, ps, gs)
            catch ex
                if ex isa Flux.Optimise.StopException
                    break
                else
                    rethrow(ex)
                end
            end        
        end 
    
        try
            isnothing(epochcb) || epochcb(epoch = e, epoch_start_time = epoch_start_time, ps = ps)
        catch ex
            if ex isa Flux.Optimise.StopException
                break
            else
                rethrow(ex)
            end
        end
    
    end
end


mutable struct ParamsIdentity
end

apply!(o::ParamsIdentity, xs; kwargs...) = xs

mutable struct ParamsReport
    batches
end
  
function apply!(o::ParamsReport, xs; epoch, batch, kwargs...)
    if batch ∉ o.batches
        return xs
    end
    param_min = flatten((abs.(x.data) for x in xs)) |> minimum
    param_max = flatten((abs.(x.data) for x in xs)) |> maximum
    param_mean = flatten((abs.(x.data) for x in xs)) |> mean
    
    grad_min = flatten((x.grad for x in xs)) |> minimum
    grad_max = flatten((x.grad for x in xs)) |> maximum
    grad_mean = flatten((x.grad for x in xs)) |> mean
    grad_L2 = collect(flatten((x.grad for x in xs))) |> norm
    println("$batch: |θ| [$param_min;$param_max] mean: $param_mean; Δ [$grad_min;$grad_max] mean: $grad_mean L2: $grad_L2")
    xs
end
  
mutable struct ClipGradient
    lb::Float32
    ub::Float32
end
  
ClipGradient() = ClipGradient(-1f0, 1f0)
  
function apply!(o::ClipGradient, xs; kwargs...)
    for x in xs
        x.grad .= min.(max.(x.grad, o.lb), o.ub)
    end
    xs
end
  
mutable struct ClipGradientNorm
    max_norm::Float32
    norm_type::Float32
end
  
ClipGradientNorm() = ClipGradientNorm(1.0f0, 2f0)
  
  # see https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html
function apply!(o::ClipGradientNorm, xs; kwargs...)
    total_norm = 0
    for x in xs
        total_norm += norm(x.grad)^o.norm_type
    end
    total_norm = total_norm^(1. / o.norm_type)
    clip_coef = o.max_norm / (total_norm + 1e-6)
      
    if clip_coef < 1
  #    @show "PRE", collect(flatten((x.grad for x in xs))) |> norm, clip_coef
        for x in xs
            x.grad .*= clip_coef 
        end
  #    @show "AFT", collect(flatten((x.grad for x in xs))) |> norm, clip_coef
    end
    xs
end