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

实盘日更（补数据 + 计算因子 + 生成交易表）：
```
python -m cbond_daily.run.live_daily
```

## 数据层与目录

- raw data：原始同步数据（由同步模块产出）
- cleaned data：清洗后的日级明细，路径：`D:/cbond_daily/clean_data/YYYY-MM/YYYYMMDD.parquet`
- factor data：因子结果，路径：`D:/cbond_daily/factor_data/YYYY-MM/YYYYMMDD.parquet`
- 回测结果：`D:/cbond_daily/results/<start>_<end>/<batch>/<timestamp>/...`

路径配置：`cbond_daily/config/paths_config.json5`

## 配置入口

- raw data：`cbond_daily/config/raw_data_config.json5`
- cleaned data：`cbond_daily/config/cleaned_data_config.json5`
- 因子计算：`cbond_daily/config/factors_config.json5`
- 单因子批量：`cbond_daily/config/factor_batch_config.json5`
- 多因子回测：`cbond_daily/config/backtest_config.json5`

## 因子开发

1) 新建或修改因子类，继承 `cbond_daily.factors.base.Factor`
2) 实现 `compute(self, data: pd.DataFrame) -> pd.Series`
3) 使用 `@FactorRegistry.register("factor_name")` 注册
4) 在 `factors_config.json5` 中配置参数

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
以下是当前完整字段列表（示例日 2024-01-03）：

`trade_date`, `code`, `prev_close_price`, `act_prev_close_price`, `close_price`, `open_price`,
`high_price`, `low_price`, `volume`, `amount`, `deal`, `update_time_x`, `twap_0930_1000`,
`twap_0930_0945`, `twap_0930_1015`, `twap_0930_1030`, `twap_0935_1000`, `twap_0945_1015`,
`twap_1000_1030`, `twap_1030_1100`, `twap_1100_1130`, `twap_1300_1330`, `twap_1330_1400`,
`twap_1400_1430`, `twap_1430_1500`, `twap_1442_1457`, `twap_1447_1457`, `twap_1452_1457`,
`update_time_y`, `vwap_0930_1000`, `vwap_0930_0945`, `vwap_0930_1015`, `vwap_0930_1030`,
`vwap_0935_1000`, `vwap_0945_1015`, `vwap_1000_1030`, `vwap_1030_1100`, `vwap_1100_1130`,
`vwap_1300_1330`, `vwap_1330_1400`, `vwap_1400_1430`, `vwap_1430_1500`,
`vwap_1442_1457`, `vwap_1447_1457`, `vwap_1452_1457`, `update_time`, `deriv_year_to_mat`,
`deriv_remain_size`, `deriv_current_yield`, `deriv_cb_conv_price`, `deriv_cb_call_price`,
`deriv_cb_put_price`, `deriv_turnover_rate`, `deriv_stock_code`, `deriv_stock_close_price`,
`deriv_bond_prem_ratio`, `deriv_debt_puredebt_ratio`, `deriv_puredebt_prem_ratio`,
`deriv_conv_value`, `deriv_ytm`, `deriv_duration`, `deriv_modify_duration`, `deriv_convexity`,
`deriv_base_rate`, `deriv_stock_volatility`, `deriv_pure_redemption_value`,
`deriv_redemption_prem_ratio`, `deriv_update_time`, `base_year_to_mat`, `base_remain_size`,
`base_current_yield`, `base_cb_conv_price`, `base_cb_put_price`, `base_turnover_rate`,
`base_stock_code`, `base_stock_close_price`, `base_bond_prem_ratio`, `base_debt_puredebt_ratio`,
`base_puredebt_prem_ratio`, `base_conv_value`, `base_ytm`, `base_duration`, `base_modify_duration`,
`base_convexity`, `base_base_rate`, `base_stock_volatility`, `base_pure_redemption_value`,
`base_redemption_prem_ratio`, `base_cb_prev_close_price`, `base_cb_act_prev_close_price`,
`base_cb_close_price`, `base_cb_volume`, `base_cb_amount`, `base_cb_deal`,
`base_stk_prev_close_price`, `base_stk_act_prev_close_price`, `base_stk_close_price`,
`base_stk_volume`, `base_stk_amount`, `base_stk_deal`, `base_rating`, `base_cb_call_price`,
`base_trigger_is_price`, `base_trigger_cum_days`, `base_trigger_reach_days`, `base_trigger_process`,
`base_trigger_date`, `base_in_trigger_process`, `base_trigger_price_revise`,
`base_trigger_is_price_revise`, `base_trigger_cum_days_revise`, `base_trigger_reach_days_revise`,
`base_trigger_process_revise`, `base_trigger_date_revise`, `base_update_time`, `rating_id`,
`rating_rating`, `rating_update_time`, `info_bond_id`, `info_exchange_mic`, `info_instrument_id`,
`info_instrument_id_mic`, `info_numeric_id`, `info_conv_id`, `info_sec_short_name`,
`info_stock_code`, `info_redemption_price`, `info_redemption_price_after_tax`, `info_publish_date`,
`info_fir_publish_date`, `info_update_time`, `info_instrument_name`

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
