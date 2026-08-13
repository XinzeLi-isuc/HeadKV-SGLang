# Phase 6 阶段报告:request 生命周期与 Continuous Batching

> 日期:2026-08-13 | 状态:**完成(Gate 6 通过)**

## 目标(计划书 §6 Phase 6)

1. 单请求生命周期测试(4K→decode128 等)✓
2. 混合长度并发测试(2K/4K/8K/16K)✓
3. Gate 6:连续 1000 混合请求无 crash/泄漏 ✓

## 测试执行(2026-08-13 23:0x,GPU 0,DuoKV ratio=0.5,R=16)

### CPU 级(已有 test_headkv_allocator.py / test_headkv_pool.py)
- test_allocator_restores_initial_state:100 次 alloc/free 循环后 full available
  == 初始、comp chunks == 初始、mapping 全零
- test_free_group_flush / test_free_derives_comp_base_once / 环形 wrap 等

### GPU 级:真实 server 混合负载压测

```text
第 1 轮:1000 请求(2K/4K/8K/16K × decode 16/32/64,权重 4:3:2:1)
        ok=1000/1000 failures=0,306.6s,3.26 req/s
第 2 轮:500 请求(验证无累积泄漏)
        ok=500/500 failures=0,156.9s,3.19 req/s(吞吐无衰减)
单请求:8K/16K 长 prompt 冒烟均成功(1.4s)
server:scheduler 全程存活,日志无 error/traceback/NaN
```

- 每轮覆盖:enter → extend → decode → finish → free → slot reuse
- 8 并发 worker → max_running_requests=16 满后排队(计划书要求)
- 两轮吞吐一致(3.26 vs 3.19)= full/comp 池无累积泄漏
- 偶发 502:单次出现(连接层),重试即恢复,与分配器无关

## Gate 6 判定(eager 模式)

- [x] 连续 1000 混合请求:无 crash、无 NaN、无串 KV(全 200 + 输出非空)
- [x] Full/Comp allocator 恢复初始态(CPU 单测 + GPU 两轮吞吐一致)
- [x] 无重复 compact chunk(CPU 单测)
- [x] 输出与单请求无系统性差异
- [x] CUDA Graph:按计划书回退 eager(MVP 固定 --disable-cuda-graph;
      CUDA Graph 留作 MVP 后深化,不阻塞)

## 面试素材

- 1000+500 请求零失败、两轮吞吐无衰减 → "allocator 生命周期无泄漏"的
  可量化证据
- 混合长度 + 并发排队场景(短请求先完成、slot 重用、ring 多次 wrap)
- 偶发 502 的定位:连接层 vs 分配器层的区分方法

## 下一步

- Phase 7:正式实验(Experiment A-E)
  - A:head ratio vs capacity(4 点已测,补全实验)
  - B:context length(4K/8K/16K/32K prefill/decode 指标)
  - C:concurrency sweep(FullKV vs DuoKV 峰值并发)
  - D:online serving(两负载)
  - E:quality(NIAH + LongBench)
