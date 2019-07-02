function update!(opt, x, x̄)
    update!(x, apply!(opt, x, copy(data(x̄))))
end
  
function _update_params_full!(pscb::AbstractArray, xs; kwargs...)
  #  @show pscb, kwargs
    for c in pscb
        apply!(c, xs; kwargs...)
    end
end
  
_update_params_full!(pscb, xs; kwargs...) = _update_params_full!([pscb], xs; kwargs...)
  
  
function _update_params!(opt, xs)
    for x in xs
        Δ = Flux.Optimise.apply!(opt, x.data, x.grad)
        x.data .-= Δ
        Δ .= 0
    end
end
  
macro interrupts(ex)
    :(try $(esc(ex))
    catch e
        e isa InterruptException || rethrow()
        throw(e)
    end)
end
  
struct StopException <: Exception end
  
function stop()
    throw(StopException())
end

function trainepochs!(loss, ps, data, opt;
    epochs = 1, 
    pscb = ParamsIdentity(), 
    epochcb = (args...;kwargs...)->(), 
    batchcb = (args...;kwargs...)->())

  for e in 1:epochs
      epoch_start_time = time()
    
      for (b, d) in enumerate(data)
          try
              l = loss(d...)
              @interrupts back!(l)
              _update_params_full!(pscb, ps, epoch = e, batch = b)
              batchcb(ps, epoch = e, batch = b, batch_size = size(d[1])[end])
              _update_params!(opt, ps)
          catch ex
              if ex isa StopException
                  break
              else
                  rethrow(ex)
              end
          end        
      end 
    
      try
          epochcb(epoch = e, epoch_start_time = epoch_start_time, ps = ps)
      catch ex
          if ex isa StopException
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
