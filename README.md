Benchmark run on A100 between a PyTorch eager implementation of attention, `flash_attn_unpadded_qkvpacked_func` from [HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention), and [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) from PyTorch nightlies.

## Specs

* PyTorch: 2.1.0.dev20230306+cu117
* flash-attn: commit 57ee618170e1adecbf787365cdf330c63768abd2
* GPU: one NVIDIA A100-SXM4-80GB

## Results

|bs|seqlen|headdim|nheads|PT eager (ms/forward)|PT native (ms/forward)|Hazy (ms/forward)|Hazy speedup over `scaled_dot_product_attention`|
|----------|-------|-------|------|---------------------|----------------------|-------------------------|----------------------------------------------|
|8         |64     |32     |12    |0.112                |0.052                 |0.042                    |1.251                                         |
|8         |64     |32     |16    |0.106                |0.053                 |0.048                    |1.109                                         |
|8         |64     |32     |24    |0.110                |0.054                 |0.049                    |1.104                                         |
|8         |64     |64     |12    |0.110                |0.056                 |0.045                    |1.250                                         |
|8         |64     |64     |16    |0.106                |0.058                 |0.044                    |1.308                                         |
|8         |64     |64     |24    |0.103                |0.065                 |0.037                    |1.791                                         |
|8         |64     |128    |12    |0.105                |0.061                 |0.038                    |1.592                                         |
|8         |64     |128    |16    |0.107                |0.061                 |0.043                    |1.418                                         |
|8         |64     |128    |24    |0.108                |0.060                 |0.038                    |1.597                                         |
|8         |128    |32     |12    |0.105                |0.058                 |0.044                    |1.307                                         |
|8         |128    |32     |16    |0.111                |0.059                 |0.045                    |1.319                                         |
|8         |128    |32     |24    |0.110                |0.051                 |0.038                    |1.358                                         |
|8         |128    |64     |12    |0.107                |0.063                 |0.035                    |1.776                                         |
|8         |128    |64     |16    |0.109                |0.058                 |0.040                    |1.475                                         |
|8         |128    |64     |24    |0.101                |0.057                 |0.040                    |1.432                                         |
|8         |128    |128    |12    |0.110                |0.064                 |0.041                    |1.578                                         |
|8         |128    |128    |16    |0.108                |0.090                 |0.049                    |1.842                                         |
|8         |128    |128    |24    |0.139                |0.131                 |0.050                    |2.634                                         |
|8         |256    |32     |12    |0.122                |0.050                 |0.031                    |1.607                                         |
|8         |256    |32     |16    |0.139                |0.056                 |0.041                    |1.372                                         |
|8         |256    |32     |24    |0.206                |0.100                 |0.040                    |2.515                                         |
|8         |256    |64     |12    |0.129                |0.083                 |0.040                    |2.082                                         |
|8         |256    |64     |16    |0.192                |0.100                 |0.031                    |3.194                                         |
|8         |256    |64     |24    |0.245                |0.172                 |0.056                    |3.085                                         |
|8         |256    |128    |12    |0.180                |0.146                 |0.122                    |1.203                                         |
|8         |256    |128    |16    |0.228                |0.183                 |0.136                    |1.344                                         |
|8         |256    |128    |24    |0.384                |0.319                 |0.230                    |1.383                                         |
|8         |512    |32     |12    |0.371                |0.125                 |0.052                    |2.408                                         |
|8         |512    |32     |16    |0.503                |0.151                 |0.061                    |2.477                                         |
|8         |512    |32     |24    |0.632                |0.229                 |0.108                    |2.115                                         |
|8         |512    |64     |12    |0.395                |0.194                 |0.105                    |1.849                                         |
|8         |512    |64     |16    |0.498                |0.232                 |0.123                    |1.891                                         |
|8         |512    |64     |24    |0.731                |0.382                 |0.252                    |1.518                                         |
|8         |512    |128    |12    |0.498                |0.428                 |0.300                    |1.425                                         |
|8         |512    |128    |16    |0.653                |0.523                 |0.375                    |1.392                                         |
|8         |512    |128    |24    |1.003                |0.910                 |0.705                    |1.290                                         |
|8         |1024   |32     |12    |1.184                |0.390                 |0.218                    |1.788                                         |
|8         |1024   |32     |16    |1.550                |0.478                 |0.308                    |1.552                                         |
|8         |1024   |32     |24    |2.314                |0.833                 |0.433                    |1.923                                         |
|8         |1024   |64     |12    |1.316                |0.568                 |0.377                    |1.505                                         |
|8         |1024   |64     |16    |1.735                |0.734                 |0.465                    |1.579                                         |
|8         |1024   |64     |24    |2.610                |1.254                 |0.822                    |1.526                                         |
|8         |1024   |128    |12    |1.656                |1.360                 |1.183                    |1.150                                         |
|8         |1024   |128    |16    |2.143                |1.781                 |1.532                    |1.162                                         |
|8         |1024   |128    |24    |3.089                |2.707                 |2.554                    |1.060                                         |
|16        |64     |32     |12    |0.121                |0.066                 |0.035                    |1.896                                         |
|16        |64     |32     |16    |0.116                |0.057                 |0.039                    |1.461                                         |
|16        |64     |32     |24    |0.106                |0.044                 |0.043                    |1.027                                         |
|16        |64     |64     |12    |0.107                |0.055                 |0.040                    |1.376                                         |
|16        |64     |64     |16    |0.106                |0.047                 |0.032                    |1.477                                         |
|16        |64     |64     |24    |0.103                |0.059                 |0.050                    |1.177                                         |
|16        |64     |128    |12    |0.112                |0.062                 |0.037                    |1.705                                         |
|16        |64     |128    |16    |0.109                |0.118                 |0.030                    |3.874                                         |
|16        |64     |128    |24    |0.128                |0.153                 |0.044                    |3.453                                         |
|16        |128    |32     |12    |0.110                |0.073                 |0.034                    |2.116                                         |
|16        |128    |32     |16    |0.103                |0.060                 |0.050                    |1.200                                         |
|16        |128    |32     |24    |0.158                |0.065                 |0.041                    |1.578                                         |
|16        |128    |64     |12    |0.105                |0.057                 |0.040                    |1.420                                         |
|16        |128    |64     |16    |0.135                |0.090                 |0.037                    |2.414                                         |
|16        |128    |64     |24    |0.188                |0.124                 |0.035                    |3.550                                         |
|16        |128    |128    |12    |0.134                |0.133                 |0.051                    |2.614                                         |
|16        |128    |128    |16    |0.179                |0.149                 |0.052                    |2.837                                         |
|16        |128    |128    |24    |0.275                |0.226                 |0.113                    |2.007                                         |
|16        |256    |32     |12    |0.210                |0.097                 |0.042                    |2.290                                         |
|16        |256    |32     |16    |0.269                |0.115                 |0.034                    |3.337                                         |
|16        |256    |32     |24    |0.387                |0.169                 |0.052                    |3.280                                         |
|16        |256    |64     |12    |0.244                |0.143                 |0.056                    |2.543                                         |
|16        |256    |64     |16    |0.330                |0.187                 |0.060                    |3.140                                         |
|16        |256    |64     |24    |0.476                |0.268                 |0.117                    |2.289                                         |
|16        |256    |128    |12    |0.379                |0.314                 |0.231                    |1.355                                         |
|16        |256    |128    |16    |0.499                |0.399                 |0.258                    |1.544                                         |
|16        |256    |128    |24    |0.698                |0.633                 |0.401                    |1.579                                         |
|16        |512    |32     |12    |0.666                |0.255                 |0.137                    |1.868                                         |
|16        |512    |32     |16    |0.853                |0.301                 |0.139                    |2.171                                         |
|16        |512    |32     |24    |1.263                |0.499                 |0.251                    |1.985                                         |
|16        |512    |64     |12    |0.770                |0.383                 |0.232                    |1.651                                         |
|16        |512    |64     |16    |0.981                |0.496                 |0.289                    |1.716                                         |
|16        |512    |64     |24    |1.476                |0.736                 |0.438                    |1.681                                         |
|16        |512    |128    |12    |0.978                |0.886                 |0.673                    |1.316                                         |
|16        |512    |128    |16    |1.315                |1.097                 |0.834                    |1.315                                         |
|16        |512    |128    |24    |2.011                |1.640                 |1.252                    |1.311                                         |
|16        |1024   |32     |12    |2.333                |0.812                 |0.444                    |1.826                                         |
|16        |1024   |32     |16    |3.019                |0.987                 |0.597                    |1.654                                         |
|16        |1024   |32     |24    |4.608                |1.559                 |0.906                    |1.721                                         |
|16        |1024   |64     |12    |2.651                |1.255                 |0.791                    |1.588                                         |
|16        |1024   |64     |16    |3.437                |1.481                 |0.982                    |1.509                                         |
|16        |1024   |64     |24    |5.282                |2.258                 |1.529                    |1.477                                         |
|16        |1024   |128    |12    |3.150                |2.712                 |2.461                    |1.102                                         |
|16        |1024   |128    |16    |4.190                |3.664                 |3.228                    |1.135                                         |
|16        |1024   |128    |24    |6.157                |5.308                 |4.874                    |1.089                                         |
|64        |64     |32     |12    |0.107                |0.092                 |0.027                    |3.428                                         |
|64        |64     |32     |16    |0.129                |0.117                 |0.034                    |3.448                                         |
|64        |64     |32     |24    |0.170                |0.148                 |0.045                    |3.288                                         |
|64        |64     |64     |12    |0.153                |0.130                 |0.042                    |3.083                                         |
|64        |64     |64     |16    |0.196                |0.176                 |0.055                    |3.174                                         |
|64        |64     |64     |24    |0.332                |0.233                 |0.101                    |2.312                                         |
|64        |64     |128    |12    |0.223                |0.217                 |0.097                    |2.234                                         |
|64        |64     |128    |16    |0.307                |0.308                 |0.143                    |2.156                                         |
|64        |64     |128    |24    |0.442                |0.440                 |0.193                    |2.282                                         |
|64        |128    |32     |12    |0.249                |0.142                 |0.043                    |3.336                                         |
|64        |128    |32     |16    |0.297                |0.181                 |0.057                    |3.157                                         |
|64        |128    |32     |24    |0.452                |0.290                 |0.115                    |2.525                                         |
|64        |128    |64     |12    |0.369                |0.230                 |0.101                    |2.287                                         |
|64        |128    |64     |16    |0.468                |0.310                 |0.138                    |2.245                                         |
|64        |128    |64     |24    |0.701                |0.472                 |0.219                    |2.153                                         |
|64        |128    |128    |12    |0.566                |0.467                 |0.191                    |2.441                                         |
|64        |128    |128    |16    |0.712                |0.628                 |0.320                    |1.964                                         |
|64        |128    |128    |24    |1.072                |0.882                 |0.363                    |2.429                                         |
|64        |256    |32     |12    |0.791                |0.322                 |0.175                    |1.838                                         |
|64        |256    |32     |16    |1.001                |0.472                 |0.213                    |2.216                                         |
|64        |256    |32     |24    |1.408                |0.634                 |0.344                    |1.844                                         |
|64        |256    |64     |12    |0.923                |0.513                 |0.242                    |2.120                                         |
|64        |256    |64     |16    |1.230                |0.688                 |0.355                    |1.936                                         |
|64        |256    |64     |24    |1.850                |0.993                 |0.466                    |2.132                                         |
|64        |256    |128    |12    |1.307                |1.125                 |0.652                    |1.724                                         |
|64        |256    |128    |16    |1.764                |1.507                 |0.895                    |1.684                                         |
|64        |256    |128    |24    |2.573                |2.084                 |1.258                    |1.656                                         |
|64        |512    |32     |12    |2.456                |0.952                 |0.533                    |1.787                                         |
|64        |512    |32     |16    |3.383                |1.235                 |0.646                    |1.911                                         |
|64        |512    |32     |24    |4.970                |1.746                 |0.929                    |1.878                                         |
|64        |512    |64     |12    |2.888                |1.410                 |0.782                    |1.804                                         |
|64        |512    |64     |16    |4.025                |1.873                 |1.092                    |1.714                                         |
|64        |512    |64     |24    |5.807                |2.673                 |1.525                    |1.753                                         |
|64        |512    |128    |12    |4.005                |3.222                 |2.473                    |1.303                                         |
|64        |512    |128    |16    |5.135                |4.164                 |3.265                    |1.275                                         |
|64        |512    |128    |24    |7.908                |6.160                 |4.768                    |1.292                                         |
|64        |1024   |32     |12    |9.167                |2.834                 |1.792                    |1.582                                         |
|64        |1024   |32     |16    |12.346               |3.701                 |2.367                    |1.564                                         |
|64        |1024   |32     |24    |18.261               |5.586                 |3.648                    |1.531                                         |
|64        |1024   |64     |12    |10.397               |4.314                 |2.998                    |1.439                                         |
|64        |1024   |64     |16    |13.884               |5.774                 |3.979                    |1.451                                         |
|64        |1024   |64     |24    |20.953               |8.356                 |5.869                    |1.424                                         |
|64        |1024   |128    |12    |12.442               |10.513                |9.131                    |1.151                                         |
|64        |1024   |128    |16    |16.654               |14.036                |12.500                   |1.123                                         |
|64        |1024   |128    |24    |25.053               |20.593                |18.381                   |1.120                                         |
