# defaults.jl — 默认参数定义
#
# 集中存放模拟器的全局默认值；修改默认行为只需改此文件。

"""
    DefaultMPSTruncation

所有作用型接口（`apply!` / `apply_kraus!` / `measure!` / `simulate` 等）`trunc`
关键字的默认截断方案：按维数截断到 `50`。
"""
const DefaultMPSTruncation = truncdimcutoff(50, 1.0e-8)
