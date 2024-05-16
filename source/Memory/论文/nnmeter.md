# nn-Meter: 可高效、准确地预测模型推理时间的系统

---

https://www.bilibili.com/video/BV1BM4y1A7T5/?vd_source=95cdaed537ef42447d87ba42e0af33be![image-20240511180959845](../../_static/figure/image-20240511180959845.png)

![image-20240511181017766](../../_static/figure/image-20240511181017766.png)

![image-20240511181044140](../../_static/figure/image-20240511181044140.png)

> op 不准，没考虑系统优化
>
> graph取决于数据集质量
>
> 所以nnmeter考虑kernel层面。

![image-20240511181204385](../../_static/figure/image-20240511181204385.png)

> 第一：每个算子尝试可能的融合，如果带来收益，就认为可融合。
>
> ​           然后深度优先搜索去找更多的融合可能。

![image-20240511181325255](../../_static/figure/image-20240511181325255.png)

![image-20240511181721677](../../_static/figure/image-20240511181721677.png)

![image-20240511181909649](../../_static/figure/image-20240511181909649.png)

> kernel 搜索空间过大，而random example会错过一些样本点。学出来的latency pattern不准。
>
> 设计会考虑的+重要的数据点(误差比较大的点)附近去采样。

![image-20240511182030005](../../_static/figure/image-20240511182030005.png)

![image-20240511182510018](../../_static/figure/image-20240511182510018.png)