# dekef
> Density estimation in kernel exponential families. 


<!---[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]--->

`dekef` is a python package providing tools for 

- computing density estimates in kernel exponential families using the score matching
  loss function and the negative log-likelihood loss function, 
- visualizing the density estimates when data are 1-dimensional or 
  the contour plot of the density estimates when data are 
  2-dimensional, and
- assessing the quality of density estimates using the correlation and 
  the mean integrated squared error ([MISE](https://en.wikipedia.org/wiki/Mean_integrated_squared_error)).
  
<!---
## Installation

OS X & Linux:

```sh
npm install my-crazy-module --save
```

Windows:

```sh
edit autoexec.bat
```
--->

## Examples

Please refer to Jupyter notebooks in `/examples` for examples of 
how to use functions to compute and visualize density estimates 
and assess their qualities. 

<!---
## Development setup

Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms.

```sh
make install
npm test
```
--->

<!---
## History

* 0.3
    * Add Jupyter notebooks in `/examples` - February 15, 2021
  
* 0.2
    * Update functions of visualizing density estimates  - February 14, 2021

* 0.1
    * Work in progress - February 11, 2021
--->

## Meta

Chenxi Zhou – chenxizhou.jayden@gmail.com

[https://github.com/zhou-chenxi/dekef](https://github.com/zhou-chenxi/dekef)

<!---
## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
--->

<!-- Markdown link & img dfn's -->
[project-stage-badge: Development]: https://img.shields.io/badge/Project%20Stage-Development-yellowgreen.svg
[project-stage-page]: https://blog.pother.ca/project-stages/
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
