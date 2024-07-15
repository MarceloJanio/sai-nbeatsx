SAI-NBEATSx: Modelo que integra autoatenção ao nbeatsx

## Citation

Para citar o artigo da versão base

```console
@article{olivares2021nbeatsx,
  title={Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx},
  author={Olivares, Kin G and Challu, Cristian and Marcjasz, Grzegorz and Weron, Rafa{\l} and Dubrawski, Artur},
  journal = {International Journal of Forecasting, submitted},
  volume = {Working Paper version available at arXiv:2104.05522},
  year={2021}
}
```

Para citar o artigo do SAI-NBEATSx

(aguardando resultado da submissão)

## Intruções


# criar ambiente virtual no conda

conda create --name nbeatsx_epf python=3.7.2
conda activate nbeatsx_epf

conda install -c anaconda numpy==1.16.1
conda install -c anaconda pandas==0.25.2
conda install -c conda-forge matplotlib==3.1.1
conda install -c anaconda seaborn==0.9.0
conda install -c anaconda scipy==1.5.2

conda install pytorch torchvision -c pytorch

conda install -c conda-forge jupyterlab
conda install -c conda-forge tqdm
conda install -c conda-forge hyperopt
conda install -c anaconda requests

# Executar sai-nbeats

python src/hyperopt_nbeatsx.py --dataset BR --space "nbeats_x" --data_augmentation 0 --random_validation 0 --n_val_weeks 52 --hyperopt_iters 300 --experiment_id "BRDA0RV0"
