using Documenter, FluxGoodies

DocMeta.setdocmeta!(FluxGoodies, :DocTestSetup, quote 
    using Flux
    using FluxGoodies
end; recursive=true)

makedocs(;
    modules=[FluxGoodies],
    format=Documenter.HTML(assets=String[], prettyurls = false),
    pages=[
        "Home" => "index.md",
        "API" => [
            "Architectures" => "architectures.md",
            "Losses" => "losses.md",
            "Training" => "training.md",
        ]
    ],
    repo="https://github.com/dhonza/FluxGoodies.jl/blob/{commit}{path}#L{line}",
    sitename="FluxGoodies.jl",
    authors="Jan Drchal"
)

deploydocs(;
    repo="github.com/dhonza/FluxGoodies.jl"
)
