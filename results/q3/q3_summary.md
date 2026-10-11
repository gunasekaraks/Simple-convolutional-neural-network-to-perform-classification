| run | optimizer | best val acc | best epoch | final train loss | final val loss | final val acc | mean s/epoch |
|---|---|---|---|---|---|---|---|
| adam | Adam (lr=1e-3) | 0.7368 | 20 | 0.4455 | 0.8146 | 0.7368 | 8.0 |
| sgd | SGD (lr=1e-2) | 0.6987 | 16 | 0.8075 | 0.8831 | 0.6751 | 8.3 |
| sgd_m0.9 | SGD + momentum 0.9 (lr=1e-2) | 0.7241 | 19 | 0.5431 | 0.8133 | 0.7241 | 8.7 |
| sgd_m0.5 | SGD + momentum 0.5 (lr=1e-2) | 0.7132 | 18 | 0.7390 | 1.0769 | 0.5935 | 4.4 |
| sgd_m0.99 | SGD + momentum 0.99 (lr=1e-2) | 0.6606 | 18 | 0.6875 | 1.1066 | 0.5808 | 4.5 |
