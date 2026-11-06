# datapipe

Dataset-aware pipeline for filing, registering, segmenting and compressing acquisition data.

Each stage operates on **one sample or session per invocation**, so the unit of work is also the
unit of failure: a stage that cannot establish what it is looking at refuses rather than guessing,
and nothing is moved until its identity has been confirmed.

## Installation

```sh
pip install datapipe-neuro
```

The distribution is `datapipe-neuro`; the import name and the CLI are both `datapipe`. An unrelated
`datapipe` project already holds that name on PyPI.

Individual stages need tools that are deliberately **not** install-time dependencies, because most
runs use only one stage: `register` and `segment` need `brainreg` and
`brainglobe-segmentation`, and `compress` shells out to the [`datapack`](https://github.com/larsrollik/datapack)
CLI (`datapack-neuro`) rather than importing it.

## Stages

| Command | What it does |
|---|---|
| `datapipe register <sample>` | brainreg-registers one serial-section sample, writing output beside the stitched directories |
| `datapipe segment <sample>` | opens napari and brainglobe-segmentation preloaded on that sample's registration |
| `datapipe compress <sample> <dest>` | compresses one sample's raw and analysis trees via datapack into `dest/<sample>` |
| `datapipe place <path>` | files one scan or session into a data tree, refusing on identity mismatch |
| `datapipe splits <root>` | finds acquisitions the recorder broke in two, and tells them from false starts |

### place

`place` routes on what the directory *is*, not on a flag you have to remember: a serial-section
scan is repacked into the tree, while an acquisition session is moved, because compression is an
earlier stage and has already run by the time anything is placed.

The subject is checked against the directory's own record and a **mismatch is refused, not
warned about**. For an acquisition session that record is the full directory prefix, not its middle
field. A histology run that holds more than one brain carries every id in its name, and is split
per animal afterwards.

### splits

A recording broken in two by a crash and a false start followed by a real recording look alike on
disk: two acquisitions, minutes apart, same subject. `splits` separates them on the gap between
them and on how much each half contains, reporting its reasoning per candidate rather than only a
verdict. Durations come from the sample counts in the `.npy` sidecars — the array shape is read,
never the array — so a 400 GB recording is classified without being loaded.

## Licence

Non-commercial research and academic use. See [LICENSE](LICENSE); commercial use requires a separate
written agreement.
