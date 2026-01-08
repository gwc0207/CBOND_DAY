# CBOND_DAY

日频可转债因子研究与回测框架。

## 快速流程

1) 同步 raw data（数据读取略）
2) 构建 cleaned data（日级明细）
3) 计算因子并写入 factor data
4) 单因子批量回测（含报告）

一键流程：
```
python -m cbond_daily.run.pipeline_all
```

多因子回测（线性组合/回归权重）：
```
python -m cbond_daily.run.backtest
```

## 数据层与目录

- raw data：原始同步数据（由同步模块产出）
- cleaned data：清洗后的日级明细，路径：`D:/cbond_daily/clean_data/YYYY-MM/YYYYMMDD.parquet`
- factor data：因子结果，路径：`D:/cbond_daily/factor_data/YYYY-MM/YYYYMMDD.parquet`
- 回测结果：`D:/cbond_daily/results/<start>_<end>/<batch>/<timestamp>/...`

路径配置：`cbond_daily/config/paths_config.json`

## 配置入口

- raw data：`cbond_daily/config/raw_data_config.json`
- cleaned data：`cbond_daily/config/cleaned_data_config.json`
- 因子计算：`cbond_daily/config/factors_config.json`
- 单因子批量：`cbond_daily/config/factor_batch_config.json`
- 多因子回测：`cbond_daily/config/backtest_config.json`

## 因子开发

1) 新建或修改因子类，继承 `cbond_daily.factors.base.Factor`
2) 实现 `compute(self, data: pd.DataFrame) -> pd.Series`
3) 使用 `@FactorRegistry.register("factor_name")` 注册
4) 在 `factors_config.json` 中配置参数

例子：
```json
{
  "name": "your_factor",
  "params": {
    "param1": 10
  }
}
```

历史窗口（动态 lookback）：
- 若因子依赖历史序列，请覆写 `required_lookback()` 返回所需交易日数。
- pipeline 会按因子需求读取历史窗口，仅写入当日结果。

算子复用：
- 通用算子集中在 `cbond_daily/factors/operators.py`
- 因子内部推荐用 `ops.xxx()` 组合表达

因子列命名规则（用于 factor data 列名）：
- 无参数：`name_`
- 有参数：`name_param1_param2_`（按参数 key 排序）

## 因子可用字段（cleaned data）

因子计算的输入来自 cleaned data 日级宽表（`D:/cbond_daily/clean_data/YYYY-MM/YYYYMMDD.parquet`）。  
字段主要分为以下几类（按前缀归类）：

- 行情基础字段：`trade_date`, `code`, `prev_close_price`, `close_price`, `open_price`,
  `high_price`, `low_price`, `volume`, `amount`, `deal`
- TWAP/VWAP 分段：`twap_0930_1000`、`twap_1430_1500` 等；`vwap_0930_1000`、`vwap_1430_1500` 等
- `deriv_*`：衍生指标（久期、转股价值、纯债溢价率、波动率等）
- `base_*`：基础信息与估值字段（转股价、纯债价值、评级、触发条款等）
- `rating_*`：评级信息
- `info_*`：转债与股票静态信息（代码、名称、发行信息等）

说明：
- 字段是否存在取决于上游表结构与同步范围。
- 因子使用前建议先确认字段是否在当日文件中存在。
## 单因子批量（factor_batch）

入口：`python -m cbond_daily.run.factor_batch`

输出目录：
```
results/<start>_<end>/Single_Factor/<timestamp>/<factor>/<params>/
```

输出文件：
- `daily_returns.csv`
- `nav_curve.csv`
- `positions.csv`
- `diagnostics.csv`
- `factor_report.png`
- `ic_series.csv`
- `factor_bins.csv`
- `factor_metrics.csv`
- `summary.csv`（在 batch 目录）
- `nav_compare.png`（在 batch 目录）

`factor_report` 不再独立运行，随 `factor_batch` 自动生成。

## 多因子回测（backtest）

入口：`python -m cbond_daily.run.backtest`

支持：
- 手动权重或回归权重（`weight_source`）
- 分箱手动/自动（`bin_source`）
- 交易成本：`twap_bps + fee_bps`

输出目录：
```
results/<start>_<end>/Backtest/<timestamp>/<signal>/
```

输出文件：
- `daily_returns.csv`
- `nav_curve.csv`
- `positions.csv`
- `diagnostics.csv`
- `weights_history.csv`
- `backtest_report.png`
- `ic_series.csv`
- `factor_bins.csv`
- `factor_metrics.csv`
- `summary.csv`（在 batch 目录）

`backtest_report` 不再独立运行，随 `backtest` 自动生成。
