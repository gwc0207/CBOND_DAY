# CBOND_DAY

日频独立仓库骨架（待填充实现）。

## 流程说明

1) 同步 ODS（数据读取略）：按配置增量拉取并落盘到 ODS。
2) 构建 DWD：按日合并主表与扩展表，生成日级明细数据。
3) 计算因子到 DWS：读取 DWD，计算因子后按日写入 DWS。
4) 实验回测：使用 `python -m cbond_daily.run.factor_batch` 批量回测并输出对比结果。
5) 报表：输出 IC/IR、分箱统计、净值对比图等（`python -m cbond_daily.report.factor_report`）。

一键流程：
`python -m cbond_daily.run.pipeline_all`

## 因子开发指南

因子开发遵循以下约定：
- 因子类需继承 `cbond_daily.factors.base.Factor`
- 实现 `compute(self, data: pd.DataFrame) -> pd.Series`
- 使用 `FactorRegistry.register("factor_name")` 注册因子
- 配置通过 `cbond_daily/config/factors_config.json` 下发参数

内置因子范例（见 `cbond_daily/factors/builtin.py`）：
- `intraday_momentum`: `close_price / prev_close_price - 1`
- `liquidity`: `amount`（空值填 0）

新增因子步骤：
1) 在 `cbond_daily/factors/` 新建或修改文件，定义因子类
2) 用 `@FactorRegistry.register("your_factor_name")` 注册
3) 在 `cbond_daily/config/factors_config.json` 增加：
   ```json
   {
     "name": "your_factor_name",
     "params": {
       "param1": 10
     }
   }
   ```
4) 运行 `python -m cbond_daily.run.build_factors`

## DWD 配置与字段说明

当前 DWD 配置见 `cbond_daily/config/dwd_config.json`：
- `primary_table`: `market_cbond.daily_price`
- `merge_tables`: `market_cbond.daily_twap`, `market_cbond.daily_vwap`,
  `market_cbond.daily_deriv`, `market_cbond.daily_base`,
  `market_cbond.daily_rating`, `metadata.cbond_info`

合并规则：
- 主表 `daily_price` 作为基准，按 `trade_date + code` 合并其他日频表。
- `metadata.cbond_info` 按 `code` 去重后合并（无 `trade_date`）。
- 除 `trade_date`/`code` 外的列会按表前缀重命名：
  - `daily_deriv` -> `deriv_*`
  - `daily_base` -> `base_*`
  - `daily_rating` -> `rating_*`
  - `cbond_info` -> `info_*`

字段示例（来自 `dwd_daily` 最新样本，2025-12-31）：
- `daily_price` 主表：`trade_date`, `code`, `prev_close_price`, `act_prev_close_price`,
  `close_price`, `open_price`, `high_price`, `low_price`, `volume`, `amount`, `deal`,
  `update_time_x`
- `daily_twap`：`twap_0930_1000`, `twap_0930_0945`, `twap_0930_1015`, `twap_0930_1030`,
  `twap_0935_1000`, `twap_0945_1015`, `twap_1000_1030`, `twap_1030_1100`,
  `twap_1100_1130`, `twap_1300_1330`, `twap_1330_1400`, `twap_1400_1430`,
  `twap_1430_1500`, `twap_1442_1457`, `twap_1447_1457`, `twap_1452_1457`,
  `update_time_y`
- `daily_vwap`：`vwap_0930_1000`, `vwap_0930_0945`, `vwap_0930_1015`, `vwap_0930_1030`,
  `vwap_0935_1000`, `vwap_0945_1015`, `vwap_1000_1030`, `vwap_1030_1100`,
  `vwap_1100_1130`, `vwap_1300_1330`, `vwap_1330_1400`, `vwap_1400_1430`,
  `vwap_1430_1500`, `vwap_1442_1457`, `vwap_1447_1457`, `vwap_1452_1457`,
  `update_time`
- `daily_deriv`（前缀 `deriv_`）：`deriv_year_to_mat`, `deriv_remain_size`,
  `deriv_current_yield`, `deriv_cb_conv_price`, `deriv_cb_call_price`,
  `deriv_cb_put_price`, `deriv_turnover_rate`, `deriv_stock_code`,
  `deriv_stock_close_price`, `deriv_bond_prem_ratio`, `deriv_debt_puredebt_ratio`,
  `deriv_puredebt_prem_ratio`, `deriv_conv_value`, `deriv_ytm`, `deriv_duration`,
  `deriv_modify_duration`, `deriv_convexity`, `deriv_base_rate`,
  `deriv_stock_volatility`, `deriv_pure_redemption_value`,
  `deriv_redemption_prem_ratio`, `deriv_update_time`
- `daily_base`（前缀 `base_`）：`base_year_to_mat`, `base_remain_size`,
  `base_current_yield`, `base_cb_conv_price`, `base_cb_put_price`,
  `base_turnover_rate`, `base_stock_code`, `base_stock_close_price`,
  `base_bond_prem_ratio`, `base_debt_puredebt_ratio`, `base_puredebt_prem_ratio`,
  `base_conv_value`, `base_ytm`, `base_duration`, `base_modify_duration`,
  `base_convexity`, `base_base_rate`, `base_stock_volatility`,
  `base_pure_redemption_value`, `base_redemption_prem_ratio`,
  `base_cb_prev_close_price`, `base_cb_act_prev_close_price`,
  `base_cb_close_price`, `base_cb_volume`, `base_cb_amount`, `base_cb_deal`,
  `base_stk_prev_close_price`, `base_stk_act_prev_close_price`,
  `base_stk_close_price`, `base_stk_volume`, `base_stk_amount`, `base_stk_deal`,
  `base_rating`, `base_cb_call_price`, `base_trigger_is_price`,
  `base_trigger_cum_days`, `base_trigger_reach_days`, `base_trigger_process`,
  `base_trigger_date`, `base_in_trigger_process`, `base_trigger_price_revise`,
  `base_trigger_is_price_revise`, `base_trigger_cum_days_revise`,
  `base_trigger_reach_days_revise`, `base_trigger_process_revise`,
  `base_trigger_date_revise`, `base_update_time`
- `daily_rating`（前缀 `rating_`）：`rating_id`, `rating_rating`, `rating_update_time`
- `cbond_info`（前缀 `info_`）：`info_bond_id`, `info_exchange_mic`,
  `info_instrument_id`, `info_instrument_id_mic`, `info_numeric_id`, `info_conv_id`,
  `info_sec_short_name`, `info_stock_code`, `info_redemption_price`,
  `info_redemption_price_after_tax`, `info_publish_date`, `info_fir_publish_date`,
  `info_update_time`, `info_instrument_name`

说明：不同日期可能列集合相同但更新值不同，字段来源以配置与上游表结构为准。
