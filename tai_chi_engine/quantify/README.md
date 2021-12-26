# Quantify - The 2nd phase of Tai-Chi Engine
> This step defines which columns will be used for X or Y, and how they are going to be transform to tensors. (Because model love tensors). These modules define such process, but no calculation happens during all the defining.

> 內外相合 用意不用力

## For developers
* Every quantify class will inherent the [```tai_chi_engine.quantify.basic.Quantify```](./basic.py) class.
* The ```__call__``` function will take a list of values from source columns, and should return a tensor for the model