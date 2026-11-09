# 本次归档验证记录

日期：2026-11-09（Asia/Shanghai）。

## Material Passport

任务：独立项目整理与复现链路验证。证据来自本地原始源码、保存的 metrics、模型和数据；没有进行外部论文事实检索。历史实验整体状态为 ANALYZED，未声称完整论文结果 VERIFIED。

## 已执行

- 使用原 Python 3.8.13 / torch 2.2.2 / PyG 2.6.1 环境执行 `tools/smoke_test.py`。
- 16 节点 / 48 事件 / 2 关系合成图训练 2 epochs，训练后执行原训练器完整 test 与 relation breakdown。
- 新 checkpoint 包含 memory.node_updater、memory.rel_updater 以及 node_type_emb 的参数，排除动态 node_memory。
- Agent 加载新 checkpoint，按训练 batch 重放历史，rank_links 得到有限分数；重复查询结果相同；observe 确认状态改变；越界 ID 与倒序时间被拒绝。
- 源码 compileall；详细训练输出见 `repro/smoke.log`。

## 修正的代码缺陷

原 lightweight saver 使用任意 startswith 过滤：`memory.` 会漏存 updater 权重，`node_type` 又会匹配已训练的 node_type_emb.weight。现只按完整键或以 `.` 为边界的子键过滤，且单独列出 memory 的动态 buffer 名称。未改变模型 forward、损失、划分或状态学习语义。

## 未通过旧权重精确复原

旧 checkpoint 缺少已训练的 node_type embedding；按 seed 只能重建初始化值，无法恢复已训练值。Agent 故意拒绝此类 checkpoint。完整模型推理须用修正后的保存器重新训练。原存档仍完整保留。

## 验证范围限制

没有重新跑完多数据集 20 epoch 和全部第三方基线，也未在全新下载依赖的环境中验证全部可选分支。小图成功说明主要训练与推理链路可用，不代表历史论文数值已重现或模型统计设计无偏。

## 数据审计结果

实际扫描 41 份统一图与任务张量。40 份索引边界和互斥检查通过；Patent-US inductive 失败，train/test 交集 129（旧 test 共 131 事件）。结果完整保存在 data_audit.json 与 patent_us_split_overlap.json。原文件未修改；新预处理增加 eval 优先去重，训练包装入口对有重复索引的旧数据默认报错。

## 其他链路检查

- 48 份配置全部通过实际训练 CLI 参数解析，引用的数据文件全部存在。
- `tools/preprocessing_smoke.py` 通过：真实 DBLP JSONL ZIP parser、128-D 特征、任务视图与基线格式导出；日志见 preprocessing_smoke.log。
- JSON-lines 接口分别验证有效请求和越界错误返回；旧缺参 checkpoint 被明确拒绝。

## 归档完整性

原项目 3532 个普通文件、原始数据 29 个文件、三套基线 341 个文件及参考预处理 71 个文件逐一核对，无缺失。原始数据与基线副本文件大小全部一致；项目内大小差异仅为本次修改的 README、四个 Python 文件和其自动编译缓存。详细清单见 source_copy_check.json。

SHA256 manifest 共 2728 项，覆盖原始数据、处理数据、历史实验及 vendor 文件（排除缓存、node_modules 和 Git 元数据）。每项由实际文件内容计算，生成后核对大小一致。未额外重复读取全部 65 GB 进行第二遍哈希；用户可运行 verify_artifacts.py 重新检查。

已实际运行 US/CN/IN/software 原始统计；结果在 data/reports/raw_stats。Patent 新划分函数以重叠输入做了小例验证，确认 eval 优先去重并保留修复元数据。没有自动覆盖原有大图任务文件，也没有把旧指标标为修复后结果。

本次未重新全量扫描 DBLP/Amazon/Github 原始记录；统计入口和原始文件均已提供。AMiner raw_stats 使用原归档完整扫描记录，不将其冒充本次新扫描。
