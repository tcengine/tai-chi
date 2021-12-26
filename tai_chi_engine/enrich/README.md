# Enrichment - The 1st phase of Tai-Chi Engine

> Enrichment **add more columns** to your dataframe. It defines how you want to enrich a column, ```Tai-Chi Engine``` will take care how the rule will apply to each row

> 以腰為軸 上下相隨

## For developers
* Every enrichment column will inherent the [```tai_chi_engine.enrich.basic.Enrich```](./basic.py) class.
* The ```__call__``` function will take a value from source column, and return the result to the destination column.
* ```is_lazy``` attribute:
    * If True, the function will be called only when the row of data is in the current training/valid batch, or preview a row. It's fitful for loading images, etc (oh, you don't want to load 10k images into memory at the same time).
    * If False, the transformation will be called for the entire column. It's fitful for things that can be executed fast and don't take up insane amount of memory.