# Live2D Automation 项目优化计划书

## 1. 文档目的

本文档用于将当前项目评估结论收敛为可执行的分阶段落地路线图，作为后续开发、验收与同步进度的统一依据。

适用范围：

- `mcp_server/` 服务端能力演进
- 图像分析、分层、PSD 打包、Cubism 自动化、导出验证链路
- 项目工程质量、安全性、可维护性、MCP app 化能力建设

---

## 2. 当前状态总结

### 2.1 当前阶段判断

- 项目阶段：`Alpha / PoC`
- 产品形态：`内部自动化工具原型`
- 自动化程度：`半自动`
- 交付物状态：`mock intermediate artifact`，尚非可直接投产的 Live2D 成品

### 2.2 当前已具备能力

- 基于 MCP 的多步骤会话式流水线
- 单图角色分析、面部特征提取、语义分层
- Cubism 模板映射与 PSD 打包
- Cubism 自动化计划生成、dispatch bundle、校准报告
- Native GUI 自动化 PoC、dry-run、partial execution、resume
- mock `.moc3` / `model3.json` / textures 输出与结构校验
- 基础输入校验、输出目录约束、远端上传 opt-in、安全脱敏

### 2.3 当前核心缺口

1. 缺少真实、稳定、可验证的 Cubism 最终导出闭环
2. 缺少可长期维护的稳定自动化执行层
3. 缺少真实数据集上的质量评估与回归体系
4. 核心实现文件过大，维护成本高
5. `opencli / MCP app` 路线停留在计划与调度层，执行层尚未打通

---

## 3. 总体目标

### 3.1 近期目标

将项目从“可演示的 Alpha PoC”推进到“可在受控环境中稳定使用的内部自动化工具”。

### 3.2 中期目标

将项目演进为“可作为 MCP app 后端核心”的稳定能力平台，支持人机协作、失败恢复、可观测和可追溯执行。

### 3.3 长期目标

在固定模板、固定 Cubism 版本、固定操作环境下，实现高成功率的近全自动流程，并具备进一步产品化基础。

---

## 4. 分阶段路线图

## P0：打通核心价值闭环

### 目标

让项目从“能生成 mock 中间产物”进入“能稳定产出真实可验收的 Cubism 导出结果”的阶段。

### 里程碑

#### M0.1 真实导出链路落地

工作内容：

- 明确最终导出策略：
  - 优先通过 Cubism Editor 自动化导出真实 `.moc3` / `model3.json` / textures
  - 移除对 placeholder `.moc3` 作为“成功交付”的依赖
- 区分两类导出状态：
  - `mock_export`
  - `real_cubism_export`
- 在导出结果中显式记录产物级别、来源、执行证据、导出方式

验收标准：

- 在固定环境下，至少 1 套模板 + 1 个样例角色可生成真实 Cubism 导出文件
- `validate_cubism_export` 能区分 mock 导出与真实导出
- CLI / MCP 返回结果中包含真实导出状态标识

预计工作量：

- `2-3 周`

#### M0.2 Native GUI 执行链稳定化

工作内容：

- 固化 Windows + Cubism 指定版本执行基线
- 完善 `launch/import/apply_template/export` 四步执行证据
- 增加截图、窗口探针、执行耗时、失败原因分类
- 增强重试与对话框恢复策略
- 完善 resume 的成功跳步与失败回滚语义

验收标准：

- 在固定 Windows 环境下连续执行 20 次，主流程成功率达到 `>= 80%`
- 每次失败都能归类到明确 failure code
- 每次执行都生成完整 artifact 集：plan、dispatch、execution、calibration、step evidence

预计工作量：

- `2-4 周`

#### M0.3 最小真实回归样本集

工作内容：

- 建立一组最小真实样本集，覆盖：
  - 标准胸像角色
  - 双眼清晰可见
  - 简单发型遮挡
  - 简单表情差异
- 为每个样本固化预期输出：
  - 必须分出的图层
  - 允许缺失的图层
  - 最低模板 coverage
  - 导出成功标准

验收标准：

- 样本集进入 `tests/` 外的专用回归目录或测试资源目录
- 每次关键变更后可执行最小回归检查
- 至少 3 个真实样本具备可复用基线结果

预计工作量：

- `1-2 周`

### P0 阶段交付标准

- 项目不再只依赖 mock 导出宣称成功
- 固定环境下具备真实导出能力
- 自动化执行具备基础稳定性和失败可观测性

### P0 总工期估算

- `5-9 周`

---

## P1：提升工程质量与可维护性

### 目标

降低后续功能迭代的回归风险，让项目从“可用 PoC”升级为“可维护的内部平台”。

### 里程碑

#### M1.1 拆分核心大文件

当前完成情况：

- [x] `mcp_server/tools/native_gui_controller.py`
  - 已拆分为 controller / script helpers / runtime helpers
  - 当前主控制文件约 `663` 行，已达到本项验收线
- [ ] `mcp_server/tools/cubism_automation.py`
- [ ] `mcp_server/secure_server_impl.py`

工作内容：

- 拆分 `mcp_server/secure_server_impl.py`
  - session 生命周期管理
  - MCP tool handlers
  - pipeline orchestration
  - error payload / validation glue
- 拆分 `mcp_server/tools/cubism_automation.py`
  - backend resolution
  - dispatch building
  - execution engine
  - calibration/report generation
- 拆分 `mcp_server/tools/native_gui_controller.py`
  - profile loading
  - powershell generation
  - action execution
  - probe / recovery / evidence

验收标准：

- 单文件原则上控制在 `<= 500-700 行`
- 新模块职责边界清晰
- 现有 CLI / MCP 接口行为保持兼容

预计工作量：

- `2-3 周`

#### M1.2 强化类型、测试与接口一致性

工作内容：

- 收紧类型边界，减少 `dict[str, Any]` 泛滥
- 为关键响应结构补充 dataclass / typed schema
- 补齐公共导出 API 一致性，例如 server export surface 与真实 tool surface 对齐
- 增加关键流程单元测试与集成测试

验收标准：

- 关键 pipeline 响应结构完成类型化
- `server.py` 对外接口与工具实现保持一致
- 新增测试覆盖关键成功路径、失败路径和 resume 路径

预计工作量：

- `1.5-2.5 周`

#### M1.3 提升结果可复现性与调试效率

工作内容：

- 为 mesh / 随机流程增加 deterministic seed
- 统一 artifact 命名与目录结构
- 增加 step-level trace id 与 job summary
- 让同一输入在同配置下尽量输出一致结果

验收标准：

- 同图同配置重复执行，结构性输出差异显著下降
- 关键调试信息能在单个 job summary 中汇总定位

预计工作量：

- `1-2 周`

### P1 阶段交付标准

- 项目核心代码可维护性明显提升
- 接口、类型、测试、artifact 结构更加稳定
- 回归和问题定位成本下降

### P1 总工期估算

- `4.5-7.5 周`

---

## P2：面向 MCP app 和平台化演进

### 目标

把项目从单机工具链提升为可持续演进的 MCP app 后端能力平台。

### 里程碑

#### M2.1 打通 opencli / connector-assisted 执行层

工作内容：

- 从“生成 connector intent”推进到“可真正执行 connector workflow”
- 明确 opencli 依赖、桥接协议、前置检查、失败反馈语义
- 将 `native_gui` 与 `opencli` 抽象为统一 backend contract

验收标准：

- `opencli` 不再只停留在 prepare 阶段
- 在至少 1 个受控环境下可执行完整 dispatch
- backend contract 对上层 CLI / MCP 保持统一

预计工作量：

- `2-4 周`

#### M2.2 持久化作业与进度同步能力

工作内容：

- 将 session / artifact metadata / resume context 从内存态转为持久化
- 增加 job 状态流转：
  - `queued`
  - `running`
  - `partial`
  - `blocked`
  - `success`
  - `error`
- 提供面向 MCP app 的进度查询与历史检索能力

验收标准：

- 服务重启后可保留 job 元数据与工件索引
- 可查询历史任务、失败原因、恢复入口

预计工作量：

- `2-3 周`

#### M2.3 建立质量评分与人工接管机制

工作内容：

- 增加自动质量评分：
  - 图层完整度
  - 模板覆盖率
  - 关键部件缺失
  - 导出可信度
- 定义人工接管点：
  - 自动执行前确认
  - apply template 前确认
  - export 前确认
  - 失败后手动恢复入口

验收标准：

- 每个 job 都能输出质量评分与建议动作
- MCP app 可以根据评分决定自动继续、暂停或请求人工确认

预计工作量：

- `2-3 周`

### P2 阶段交付标准

- 项目具备 MCP app 后端平台属性
- 支持历史可追踪、失败可恢复、人机协作可落地
- `native_gui` 与 `opencli` 路线开始并行可用

### P2 总工期估算

- `6-10 周`

---

## 5. 优先级总表

| 优先级 | 阶段目标 | 结果定位 | 预计工期 |
| --- | --- | --- | --- |
| P0 | 打通真实导出与稳定自动化 | 从 PoC 进入可用内测工具 | 5-9 周 |
| P1 | 降低维护成本与回归风险 | 从可用走向可维护 | 4.5-7.5 周 |
| P2 | 平台化与 MCP app 化 | 从内部工具走向平台后端 | 6-10 周 |

建议执行顺序：

1. 先完成 `P0`
2. 再完成 `P1`
3. 最后推进 `P2`

不建议跳过 `P0` 直接做 app 包装，否则会把 mock 能力包装成成品能力，后续返工成本更高。

---

## 6. 验收口径

### 6.1 功能验收

- 是否能稳定完成从输入图片到目标产物的完整流程
- 是否能明确区分 mock 成功与真实导出成功
- 是否能在失败时给出明确状态与恢复路径

### 6.2 工程验收

- 是否有稳定测试与最小真实回归样本
- 是否降低了核心模块复杂度
- 是否具备一致的 artifact、日志和状态模型

### 6.3 平台验收

- 是否支持 job 查询、历史追踪、resume、失败归因
- 是否适合作为 MCP app 的后端能力层

---

## 7. 风险与依赖

### 7.1 主要风险

1. Cubism GUI 自动化天然脆弱，受版本、语言、窗口标题、弹窗流程影响大
2. 图像输入风格分布复杂，当前启发式和轻量 refinement 泛化能力有限
3. 若过早追求“全自动成品”，会掩盖真实能力边界并抬高返工成本

### 7.2 外部依赖

- Windows 执行环境
- 指定版本的 Cubism Editor
- 可校准的 profile 与窗口探针规则
- 若使用 API part backend，还依赖远端服务可用性与合规配置

---

## 8. 建议的推进方式

### 8.1 推荐节奏

- 第 1 阶段：以 `P0` 为主，按周追踪真实导出成功率
- 第 2 阶段：在 `P0` 基本稳定后进入 `P1`，控制技术债增长
- 第 3 阶段：在固定环境稳定后推进 `P2`，建设 MCP app 化能力

### 8.2 推荐里程碑管理方式

- 每个里程碑必须有：
  - 目标
  - 输入条件
  - 输出产物
  - 验收标准
  - 风险记录
  - 当前状态

建议状态值：

- `not_started`
- `in_progress`
- `blocked`
- `partial`
- `done`

---

## 9. 进度同步模板

后续同步建议直接在本文件中追加更新，或按周维护相同格式。

### 9.1 周报模板

```md
## 进度更新 - YYYY-MM-DD

### 本周完成
- 

### 当前进行中
- 

### 当前阻塞项
- 

### 风险变化
- 

### 下周计划
- 
```

### 9.2 里程碑状态模板

```md
| 里程碑 | 状态 | 负责人 | 开始日期 | 目标日期 | 备注 |
| --- | --- | --- | --- | --- | --- |
| M0.1 真实导出链路落地 | not_started | TBD | TBD | TBD |  |
| M0.2 Native GUI 执行链稳定化 | not_started | TBD | TBD | TBD |  |
| M0.3 最小真实回归样本集 | not_started | TBD | TBD | TBD |  |
| M1.1 拆分核心大文件 | in_progress | TBD | TBD | TBD | `native_gui_controller.py` 已完成拆分；`cubism_automation.py` / `secure_server_impl.py` 继续推进 |
| M1.2 强化类型、测试与接口一致性 | not_started | TBD | TBD | TBD |  |
| M1.3 提升结果可复现性与调试效率 | not_started | TBD | TBD | TBD |  |
| M2.1 打通 opencli / connector-assisted 执行层 | not_started | TBD | TBD | TBD |  |
| M2.2 持久化作业与进度同步能力 | not_started | TBD | TBD | TBD |  |
| M2.3 建立质量评分与人工接管机制 | not_started | TBD | TBD | TBD |  |
```

---

## 10. 最终建议

当前项目最合理的定位不是“立即包装为全自动 Live2D 成品生成器”，而是：

- 短期做成稳定的内部自动化工具
- 中期做成支持人机协作的 MCP app 后端
- 长期再逐步提升自动化闭环与泛化能力

项目推进时，必须坚持以下原则：

1. 先补真实导出闭环，再谈产品包装
2. 先把固定环境跑稳，再谈泛化
3. 先控制核心技术债，再扩展后端和平台能力

这是当前投入产出比最高、返工风险最低的路线。
