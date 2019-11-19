| **Build Status** | **Coverage** |
|:-----------------:|:-----------------:|
| [![Build Status][build-img]][build-url] | [![Codecov branch][codecov-img]][codecov-url] |

# SchwartzSmith.jl
SchwartzSmith.jl is a package for estimating the Schwartz Smith model.\
The implementation and notation was based on the original article by Eduardo Schwartz and James E. Smith (2000), called "Short-Term Variations and Long-Term Dynamics in Commodity Prices" (https://pdfs.semanticscholar.org/c298/e5c8c477e3da8941173e6cb593f05230006c.pdf)

To use the package you can do 
```julia
pkg> add https://github.com/LAMPSPUC/SchwartzSmith.jl.git
```

## Features

* Parameter estimation
    * Original and average time to maturity
    * Random or predefined seed
* Simulation
* Forecasting


[build-img]: https://travis-ci.org/LAMPSPUC/SchwartzSmith.jl.svg?branch=master
[build-url]: https://travis-ci.org/LAMPSPUC/SchwartzSmith.jl

[codecov-img]: https://codecov.io/gh/LAMPSPUC/SchwartzSmith.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/gh/LAMPSPUC/SchwartzSmith.jl?branch=master