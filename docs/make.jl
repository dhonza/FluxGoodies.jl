using Documenter, FluxGoodies

makedocs(;
    modules=[FluxGoodies],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/dhonza/FluxGoodies.jl/blob/{commit}{path}#L{line}",
    sitename="FluxGoodies.jl",
    authors="Jan Drchal",
    assets=String[],
)

deploydocs(;
    repo="github.com/dhonza/FluxGoodies.jl",
)
