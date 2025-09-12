# ConvALSR-Net

Our model code will be uploaded soon.
  
## Data Preprocessing

Please follw the [GeoSeg](https://github.com/WangLibo1995/GeoSeg) to preprocess the LoveDA, Potsdam and Vaihingen dataset.

## Training

"-c" means the path of the config, use different **config** to train different models.

```shell
python train_supervision_dp.py -c ./config/potsdam/convalsrnet.py
```

```shell
python train_supervision_dp.py -c ./config/vaihingen/convalsrnet.py
```

```shell
python train_supervision_dp.py -c ./config/loveda/convalsrnet.py
```

## Testing

**Vaihingen**
```shell
python test_vaihingen.py -c ./config/vaihingen/convalsrnet.py -o ./fig_results/convalsrnet_vaihingen/ --rgb -t "d4"
```

**Potsdam**
```shell
python test_potsdam.py -c ./config/potsdam/convalsrnet.py -o ./fig_results/convalsrnet_potsdam/ --rgb -t "d4"
```

**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))

Output RGB images (Offline testing, using the validation set for testing, directly output the mIOU results)
```shell
python test_loveda.py -c ./config/loveda/convalsrnet.py -o ./fig_results/convalsrnet_loveda_rgb --rgb --val -t "d4"
```
Output label images (need to be compressed and uploaded to the online testing website)
```shell
python test_loveda.py -c ./config/loveda/convalsrnet.py -o ./fig_results/convalsrnet_loveda_onlinetest -t "d4"
```


## Acknowledgement

Our training scripts come from [ConvLSR-Net](https://github.com/stdcoutzrh/ConvLSR-Net) which is based on [GeoSeg](https://github.com/WangLibo1995/GeoSeg). Thanks for the author's open-sourcing code.
- [ConvLSR-Net](https://github.com/stdcoutzrh/ConvLSR-Net)
- [GeoSeg(UNetFormer)](https://github.com/WangLibo1995/GeoSeg)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
