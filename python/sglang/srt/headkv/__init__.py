"""HeadKV:Policy-decoupled head-wise heterogeneous KV cache backend.

算法与配置层,零 SGLang 内部依赖(仅 numpy/torch),便于单测与 future
current-main port。runtime 只消费 HeadPolicy 输出的 KV-head mask + sink/recent。
"""
