# CBOND_DAY

日频可转债因子研究与回测框架。

## 快速流程
1) 同步 raw data（数据库 → 本地）
2) 构建 cleaned data（日级宽表）
3) 计算因子并写入 factor data
4) 单因子批量回测（含报告）

一键流程：
```bash
python -m cbond_daily.run.pipeline_all
```

多因子回测（线性组合 + 回归权重）：
```bash
python -m cbond_daily.run.backtest
```

实盘日更（补数据 + 算因子 + 生成交易表）：
```bash
python -m cbond_daily.run.live_daily
```

## 数据层与目录
- raw data：原始同步数据（`D:/cbond_daily/raw_data`）
- cleaned data：清洗后的日级宽表（`D:/cbond_daily/clean_data/YYYY-MM/YYYYMMDD.parquet`）
- factor data：因子结果（`D:/cbond_daily/factor_data/YYYY-MM/YYYYMMDD.parquet`）
- 回测结果：`D:/cbond_daily/results/<start>_<end>/<batch>/<timestamp>/...`

路径配置：`cbond_daily/config/paths_config.json5`

## 配置入口
- raw data：`cbond_daily/config/raw_data_config.json5`
- cleaned data：`cbond_daily/config/cleaned_data_config.json5`
- 单因子批量：`cbond_daily/config/factor_batch_config.json5`
- 多因子回测：`cbond_daily/config/backtest_config.json5`
- 实盘：`cbond_daily/config/live_config.json5`

## 因子开发
1) 新建或修改因子类，继承 `cbond_daily.factors.base.Factor`
2) 实现 `compute(self, data: pd.DataFrame) -> pd.Series`
3) 用 `@FactorRegistry.register("factor_name")` 注册
4) 在 `factor_batch_config.json5` 配置参数

历史窗口（动态 lookback）：
- 如因子依赖历史序列，覆写 `required_lookback()` 返回所需交易日数
- pipeline 会按因子需求读取历史窗口，仅写入目标日期结果

算子复用：
- 通用算子集中在 `cbond_daily/factors/operators.py`

因子列命名规则（用于 factor data 列名）：
- 无参数：`name_`
- 有参数：`name_param1_param2_`（参数 key 排序后拼接）

## 因子可用字段（cleaned data）
cleaned data 现在按表**全字段**落盘，并按表前缀区分来源（见 `cleaned_data_config.json5`）。

通用键：
- `trade_date`：交易日
- `instrument_code`：转债代码
- `exchange_code`：交易所（SH/SZ）
- `code`：`instrument_code.exchange_code` 拼接

### market_cbond.daily_price（无前缀）
- `prev_close_price`：昨收
- `act_prev_close_price`：实际昨收
- `close_price`：收盘
- `open_price`：开盘
- `high_price`：最高
- `low_price`：最低
- `volume`：成交量
- `amount`：成交额
- `deal`：成交笔数
- `update_time`：更新时间

### market_cbond.daily_twap（无前缀）
TWAP 时间窗字段：
- `twap_0930_0945`, `twap_0930_1000`, `twap_0930_1015`, `twap_0930_1030`
- `twap_0935_1000`, `twap_0935_1005`, `twap_0945_1000`, `twap_0945_1015`
- `twap_1000_1030`, `twap_1030_1100`, `twap_1100_1130`
- `twap_1300_1330`, `twap_1330_1400`, `twap_1400_1430`
- `twap_1430_1442`, `twap_1430_1500`, `twap_1442_1457`, `twap_1447_1457`, `twap_1452_1457`
- `update_time`

### market_cbond.daily_vwap（无前缀）
VWAP 时间窗字段（与 TWAP 对应）：
- `vwap_0930_0945`, `vwap_0930_1000`, `vwap_0930_1015`, `vwap_0930_1030`
- `vwap_0935_1000`, `vwap_0935_1005`, `vwap_0945_1000`, `vwap_0945_1015`
- `vwap_1000_1030`, `vwap_1030_1100`, `vwap_1100_1130`
- `vwap_1300_1330`, `vwap_1330_1400`, `vwap_1400_1430`
- `vwap_1430_1442`, `vwap_1430_1500`, `vwap_1442_1457`, `vwap_1447_1457`, `vwap_1452_1457`
- `update_time`

### market_cbond.daily_deriv（前缀 `deriv_`）
衍生指标字段（示例）：
- `deriv_year_to_mat`：剩余年限
- `deriv_remain_size`：剩余规模
- `deriv_current_yield`：当期收益率
- `deriv_cb_conv_price`：转股价
- `deriv_cb_call_price`：赎回价
- `deriv_cb_put_price`：回售价
- `deriv_turnover_rate`：换手率
- `deriv_stock_code`：正股代码
- `deriv_stock_close_price`：正股收盘
- `deriv_bond_prem_ratio`：转股溢价率
- `deriv_debt_puredebt_ratio`：纯债比例
- `deriv_puredebt_prem_ratio`：纯债溢价率
- `deriv_conv_value`：转股价值
- `deriv_ytm`：到期收益率
- `deriv_duration`：久期
- `deriv_modify_duration`：修正久期
- `deriv_convexity`：凸性
- `deriv_base_rate`：基准利率
- `deriv_stock_volatility`：正股波动率
- `deriv_pure_redemption_value`：纯债价值
- `deriv_redemption_prem_ratio`：赎回溢价率
- `deriv_update_time`：更新时间

### market_cbond.daily_base（前缀 `base_`）
基础指标字段（示例）：
- `base_year_to_mat`, `base_remain_size`, `base_current_yield`
- `base_cb_conv_price`, `base_cb_put_price`, `base_cb_call_price`
- `base_turnover_rate`, `base_stock_code`, `base_stock_close_price`
- `base_bond_prem_ratio`, `base_debt_puredebt_ratio`, `base_puredebt_prem_ratio`
- `base_conv_value`, `base_ytm`, `base_duration`, `base_modify_duration`, `base_convexity`
- `base_base_rate`, `base_stock_volatility`
- `base_pure_redemption_value`, `base_redemption_prem_ratio`
- `base_cb_prev_close_price`, `base_cb_act_prev_close_price`, `base_cb_close_price`
- `base_cb_volume`, `base_cb_amount`, `base_cb_deal`
- `base_stk_prev_close_price`, `base_stk_act_prev_close_price`, `base_stk_close_price`
- `base_stk_volume`, `base_stk_amount`, `base_stk_deal`
- `base_rating`
- `base_trigger_is_price`, `base_trigger_cum_days`, `base_trigger_reach_days`
- `base_trigger_process`, `base_trigger_date`
- `base_in_trigger_process`
- `base_trigger_price_revise`, `base_trigger_is_price_revise`
- `base_trigger_cum_days_revise`, `base_trigger_reach_days_revise`
- `base_trigger_process_revise`, `base_trigger_date_revise`
- `base_update_time`

### market_cbond.daily_rating（前缀 `rating_`）
- `rating_id`：评级记录 ID
- `rating_rating`：评级
- `rating_update_time`：更新时间

### metadata.cbond_info（前缀 `info_`）
- `info_bond_id`：债券 ID
- `info_exchange_mic`：交易所 MIC
- `info_instrument_id`：合约 ID
- `info_instrument_id_mic`：MIC 下合约 ID
- `info_numeric_id`：数值 ID
- `info_conv_id`：转债 ID
- `info_sec_short_name`：简称
- `info_stock_code`：正股代码
- `info_redemption_price`：赎回价
- `info_redemption_price_after_tax`：税后赎回价
- `info_publish_date`：发行日
- `info_fir_publish_date`：首次公告日
- `info_update_time`：更新时间
- `info_instrument_name`：名称

说明：
- 字段是否存在取决于上游表结构与同步范围
- 建议在因子计算前先检查当日文件字段

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
- `summary.csv`（batch 目录）
- `nav_compare.png`（batch 目录）

`factor_report` 随 `factor_batch` 自动生成。

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
- `summary.csv`（batch 目录）

`backtest_report` 随 `backtest` 自动生成。
