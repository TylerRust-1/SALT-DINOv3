# Prediction maps

Every prediction map behind the three benchmark tables, one PNG per test image, for all
fifteen printed rows: three input resolutions per task plus the data-matched controls.
128,151 maps over eleven benchmarks.

The maps are not in this repository. They are archived on Zenodo, split by task, because the
salient rows are three quarters of the bytes and most readers want one task:

| archive | rows | benchmarks | maps | size |
|---|---|---|---|---|
| `SALT-maps-cod` | 336, 504, 672 | CAMO, COD10K, CHAMELEON, NC4K | 19,419 | 435 MB |
| `SALT-maps-mas` | 336, 504, 672 in two training sets | MAS3K, RMAS | 9,840 | 228 MB |
| `SALT-maps-sod` | 336, 504, 672 in two training sets | DUTS-TE, DUT-OMRON, ECSSD, HKU-IS, PASCAL-S | 98,892 | 690 MB |

Each archive is self-describing: it carries its own `manifest.json`, so a tarball downloaded
alone is still checkable. Verify a download against `SHA256SUMS` in the record.

<!-- ZENODO-DOI -->
> **DOI:** [10.5281/zenodo.23554491](https://doi.org/10.5281/zenodo.23554491)

## Why these ship

The paper gates every per-image comparison on whether a baseline's released maps reproduce
the numbers that baseline is cited for. Fifteen released sets go through that gate and six
fail, one by 0.060 S-measure. A paper that applies that test to other people's maps has to
release its own maps to the same standard, or the audit is one-directional.

So each archive carries a `manifest.json` that records, per row and per benchmark, both the
number the paper prints and the number recomputed **from the shipped PNGs**:

```json
"cod672": {
  "run": "dinov3_cod672", "input_size": 672, "train_set": "CAMO-TR+COD10K-TR", "seed": 1024,
  "sets": {
    "CHAMELEON": {
      "n": 76,
      "reported":           {"mIoU": 0.8641, "Smeasure": 0.9425, "w_Fm": 0.9117, "MAE": 0.0157},
      "rescored_from_maps": {"mIoU": 0.8641, "Smeasure": 0.9425, "w_Fm": 0.9117, "MAE": 0.0157}
    }
  }
}
```

S-measure, the metric the gate uses, agrees to four decimals on all 54 row-and-benchmark
pairs. mIoU and max F-measure agree to within 0.0002: twelve values sit 0.0001 below the
printed number and one (`sod672_duts` DUT-OMRON mIoU, 0.8034 printed, 0.8032 rescored) sits
0.0002 below, because quantizing the soft maps to 8 bits moves a few pixels across the 0.5
threshold. One three-decimal value the paper prints would change on rescoring: `cod336`
CHAMELEON mIoU, 0.8155 printed as .816 and 0.8154 rescored.

## Layout

```
<row>/<BENCHMARK>/<image stem>.png
```

Row ids name the configuration: `cod672` is the camouflaged model at 672 px, `sod336` the
salient model at 336 px on the full training union, and `sod336_duts` the **data-matched
control** retrained on DUTS-TR alone, which is what the four salient intervals pair against
so that no interval rests on the larger training union. `mas672_marine` is the marine
control trained without COD10K.

Maps are **8-bit soft, not thresholded**, at native ground-truth resolution. Thresholding
would make S-measure and max F-measure unrecoverable, which is a large part of why some
released map sets cannot be checked against their own papers.

## Rescore them

```bash
tar xzf SALT-maps-cod-v1.0.0.tar.gz
python ../evaluation/stat_tests.py score \
    --maps cod672/CAMO --set CAMO --name check_camo
python ../evaluation/stat_tests.py compare --a check_camo --b birefnet_CAMO --metrics Smeasure
```

`score` needs the benchmark ground truth; `compare` runs offline against the per-image dumps
already in [`../evaluation/per_image/`](../evaluation/per_image/).

## Regenerate them

```bash
python eval.py --ckpt runs/salt_cod672/salt-best.pth --size 672 \
    --test_root /data/CAMO/Test --save_maps maps_out
```

`--save_maps` is the same code path that produced the archives, including running the
sigmoid and the upsample under bf16 autocast. That detail is load-bearing: promoting to
fp32 there shifts mIoU by about 0.0015, which is most of the measured S-measure seed spread.
