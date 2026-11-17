# IR-Line-Synth

Synthetic annotated data for a line of reflective IR distance sensors, built
from an existing YOLO dataset of its grayscale frames: cut annotated objects out
of real frames, make new ones by replaying them with another motion, place them
on real or generated backgrounds, and export the result as a new YOLO dataset
(or as long tapes) with the provenance of every example. A local web editor for
working by hand, and recipes run from the command line for whole batches.

It was written to add hard cases (crowded, overlapping objects) to the training
data of EggCounter, a detector for a 40-channel line, and works with any line:
the number of channels and the length of a frame come from the dataset (see
*Sensor line*).

## What it does

- **Manual editor.** Pick a source frame and click an annotated box to add the
  object to a bank. Flip, rotate, scale, change brightness, cut it by an
  intensity mask, then place it on a background frame by click or drag.
  Blending (replace, max, additive), soft edges and a maximum IoU between boxes
  are checked on every change; undo and redo work on the whole scene.
- **Variations.** Save N copies of a scene with random shifts of position,
  brightness, angle and scale of the added objects, from a seed.
- **Object generator.** Make new objects from real ones and pick them by hand.
  A frame is a space-time scan (rows are successive polls of the sensor line),
  so an object is replayed with a new motion: speed (long and short objects),
  drift across the belt (rolling), swings and lost contact (swinging, torn),
  then imperfections are added (pixel noise, the bright part moved towards an
  edge, uneven channels, dropped pixels). A filter compares each candidate with
  real objects of the same kind and only advises; chosen objects go to the
  bank and work like real ones. See `morph.py`.
- **Generated backgrounds.** Empty backgrounds stitched from blocks of empty
  sensor columns of real frames, so every channel keeps its own offset and
  noise (`background.py`). A tape has one set of sensors: the offsets of its
  channels are fixed by the first piece and stay the same along the whole tape,
  later pieces bring only their noise, so there are no seams. The spread of a
  generated background (largest minus smallest value) is limited to 3% of the
  full scale (7 of 255 levels), for frames and tapes alike. Rebuilt from a seed
  whenever a scene is rendered.
- **Autogenerator.** Generate a batch of candidate scenes by rules (object
  count, reflections, rotation, brightness range, insertion by box or mask,
  maximum IoU; background: the chosen one, a random real frame or a new
  generated one per image), review the previews and save only the selected ones.
- **Tapes.** A recording instead of a frame: all channels and any number of rows
  (up to 20000). Choose objects from the bank, a layout function (random, even
  steps, groups of objects close together), the number of objects and the length
  in rows, and generate. The tape can then be edited: add objects of the bank by
  click, delete or move them (drag or X/Y), undo and redo, change the background
  seed. Saved as a CSV of the signal, a CSV of the labels and a JSON description
  (`tape.py`).
- **Safe export.** Scenes are written as 8-bit PNG, YOLO labels and a
  provenance JSON per scene, into a folder that must lie outside the source
  dataset. Identical scenes are not saved twice, and a scene goes only into the
  split all of its sources come from (see *One split at a time*). The preview
  is the exact file content: intensities are rounded to the 256 levels before
  display.

## One split at a time

The output has the splits `train`, `val` and `test`, like the source. A scene
or a tape is saved only into the split that **every** one of its objects and
its background come from: a real background frame, or the frames whose empty
columns make a generated background. Anything else is refused with an error,
`val` and `test` included, so nothing of one source split reaches another.

- The autogenerator and tapes have a rule `split` (the page: *Часть набора*):
  pool objects, the fixed, real or generated background and the save all use
  it. Bank objects of another split are shown greyed out and not used.
  (`background_split`, the old name, is accepted only when it equals `split`.)
- The object generator takes its sources, statistics and filter from one split.
- The provenance of a scene or tape lists, for every object, the source frame
  (`frame`, e.g. `scan_0012`), its `split` and `box_index`; for the background,
  its split and the source `frames`.

**Stubs.** An object is *cut by the frame edge* (`clipped`) when its box comes
closer than 1 cell to any edge of the frame (as `TOUCH_TOLERANCE` in SensorLine)
or when one of its pixels above the threshold lies on the first or last row or
column. Stubs are never used to make new objects (neither picked automatically
nor given as sources), nor counted in the statistics and kinds of the filter,
nor taken from the bank by the autogenerator and tapes. The manual editor shows
them with a dashed box marked «обрубок» and still lets you place them by hand.

**The smallest new objects.** In a batch of N ≥ 10 new objects of one kind, the
⌊0.1·N⌋ shortest (rows above the threshold) and the ⌊0.1·N⌋ narrowest (sensors)
are removed; for the rolling and swinging kinds, which can still come out
small and faint, also the ⌊0.1·N⌋ smallest by area (pixels above the threshold)
and the ⌊0.2·N⌋ dimmest (mean of those pixels). An object in several lists is
removed once, ties broken by the seed. The smallest value of each measure left
is fixed as a threshold, and the removed objects are replaced by new ones of the
same kind, sources and filter that reach every threshold. The response
(`size_filter`) says how many were removed and replaced and gives the thresholds.

To build the datasets of one source split by split: start
`ir-line-synth serve --data path\to\main_flow`, choose *Часть набора* `train` in the
object generator, the autogenerator or the tape, collect bank objects from
`train` frames, generate and save; then the same with `val` and with `test`.

## Install and run

Python 3.12 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\ir-line-synth.exe serve --data path\to\data.yaml
```

Open `http://127.0.0.1:8009/` (`--port` changes the port). The output folder
is chosen in the page; by default `synthetic_data` in the current directory.
The editor prints the dataset and the sensor line it reads when it starts; page
exports do not record the dataset path, so check that line in the console.

`run.ps1` / `run.bat` install the package if needed, start the editor and open the
browser: `.\run.ps1 -Data <folder or data.yaml>`, or set the dataset once with
`setx IR_LINE_SYNTH_DATA "<folder>"` and start `run.bat` without arguments.

## Command line

```text
ir-line-synth serve --data DATASET [--port 8009]      the editor in the browser
ir-line-synth generate RECIPE [--output DIR] [--dry-run]
ir-line-synth verify OUTPUT --data DATASET            saved examples render again exactly
ir-line-synth constants [--output FILE]               the registry of fixed numbers
ir-line-synth fit-sensor --calibration PATH ... --output PROFILE     a sensor profile from measurements
ir-line-synth sensor-report --profile PROFILE --calibration PATH ... the check of a profile
ir-line-synth --version
```

- **generate** runs a recipe: one YAML file with the dataset, the split, one seed,
  the pool of objects and what to make (scenes, tapes). The same recipe on the same
  data makes the same files. Every run writes `runs/run_<time>_<recipe digest>.json`
  (program version, recipe and its SHA-256, the sensor line, what was made and what
  was refused) and a copy of the recipe into the output. See `docs/recipes.md`.
- **verify** renders every saved scene and tape of an output again from its own
  description and compares it with the recorded digest and with the files on disk.
  It exits with 1 and names the example when anything differs.
- **constants** prints `docs/constants.md`: the value, unit, origin (measured,
  tuned, external, bound, definition) and meaning of every fixed number of the
  program. The tests fail when a numeric constant has no entry or the document is
  out of date.
- **fit-sensor** fits a sensor profile (below) to calibration captures and, optionally,
  to recordings of the bare belt of a working line; **sensor-report** checks a profile
  against the same measurements and writes the result as Markdown.

## Sensor model

A line of reflective IR distance sensors is described by a *sensor profile*, a YAML
file: the response of every channel to distance and reflectance
(`offset + reflectance * gain * (d + shift)^-power`), the noise of the sensor, the range
of its ADC, an optional correction table, how frames are formed from polls and,
optionally, the bare belt of a working installation (channel levels and a noise with a
periodic part). Two profiles are shipped: `reference_line_40`, fitted to the
calibration captures and recordings of the reference line, and `example_line_32`, a
line given by nominal values with a spread between channels. The model, the data of the
reference line and what the model leaves out are in `docs/sensor_model.md`; the check
(response on held-out distances, noise, the 1070 frames of the dataset made again from
the recordings pixel for pixel, the belt noise) in `docs/sensor_validation.md`.

Objects can also be described by shape instead of cut from real frames (`shapes.py`,
`docs/objects.md`): a superquadric (sphere, ellipsoid, box, standing cylinder) lying on
the belt gives the height of its top under every channel in every poll (with roughness
and a reflectance map: smooth spread, dark patches, dropouts, glints), and the sensor
model turns the distance from the line into codes. The installation is the height of the
line over the belt, the pitch of the channels, the poll rate and the reflectance of the
belt. The kinds of motion of real objects (calm, slow with stalls, fast, rolling,
swinging with lost contact) move these objects too (`motion.py`). A whole record
without real data (`record.py`): objects pass over the bare belt of the profile, the line
sees the nearest surface, and the codes are cut into frames with boxes from the geometry.
A recipe with a `parametric` section makes such records by random rules without any
source dataset (`docs/recipes.md`); `verify` checks them without `--data`. On the page the
button «Без реальных данных» edits the same section, previews the frames of the first record
and saves all of them. `ir-line-synth object-report` fits what was not measured (pitch of the
channels, fall of the reflection with the tilt of the surface, belt speed, reflectance of an egg) to
the real eggs and writes `docs/object_validation.md`, with the margins that make geometric boxes
look like those of the annotators. Kinds of objects and motion ranges are tuned in rows and channels
of the reference line and keep their duration and distance on another line (`sensor_profile` in a
data.yaml names the profile of the line of a dataset).

## Sensor line

A line of reflective IR distance sensors is polled at a fixed rate while objects
pass under it; successive polls stacked together give a frame: one axis holds the
channels, the other the polls (time). The *frame geometry* of a dataset is the
number of channels, the number of polls in a frame and the image axis of the
channels. Scenes have the size of the frames, tapes have all the channels, and an
output holds frames of one size.

## Source dataset

`--data` takes a YAML file with a `data` section, or a folder that contains
`data.yaml`. Paths are relative to the YAML file:

```yaml
data:
  train: ./train          # images/ and labels/ (YOLO, class 0)
  val: ./val
  test: ./test            # optional
  sensor_axis: x          # image axis of the channels: x or y (default x)
  scale: 255.0            # intensity = (stored value - offset) / scale (default 255)
  offset: 0.0
  channels: 40            # optional: channels of the line
  frame_length: 40        # optional: polls in a frame
```

Frames must be grayscale and all of one size. Without `channels` and
`frame_length` the first frame of the dataset sets them. An EggCounter model
configuration has the same `data` section and can be passed directly.

## Output

```text
synthetic_data/
  data.yaml                   the same data section: ./train, ./val, ./test, scale 255
  synthetic_dataset.json      export format marker
  train/, val/ and test/
    images/synthetic_<id>.png
    labels/synthetic_<id>.txt
    provenance/synthetic_<id>.json   program version, source frames, boxes, transforms, statistics
  runs/                       records of the recipe runs (command line)
```

Tapes go into the same folder, next to the scenes:

```text
synthetic_data/tapes/train/, val/ and test/
  <name>.csv          the signal: a row per poll (time), a column per channel;
                      8-bit levels 0-255 (as in the PNG) or intensity 0-1, optional header
  <name>.labels.csv   one row per object: index, class, channel_min, row_min,
                      channel_max, row_max (boxes in channels and rows), source, kind
  <name>.json         the full tape description (it renders again from it), sources, statistics
```

The columns are always channels and the rows always time, whatever `sensor_axis`
of the source frames is. Identical tapes are not saved twice, and a tape goes only
into the split of all its objects and its background.

An output made before the `test` split existed gets `test: ./test` added to its
`data.yaml` on the next save.

Because the output describes itself with `data.yaml`, it can be used as a
training set directly, for example with `train.py --data synthetic_data` in
EggCounter, or as the source of the next editing session.

## Structure

```text
src/ir_line_synth/
  sensor.py       the sensor line: channels, polls per frame, axis
  dataset.py      reading the source dataset (YOLO frames, data.yaml)
  scene.py        objects cut from frames, their transforms, placement and blending
  morph.py        new objects: a real object replayed with another motion; the filter
  background.py   empty backgrounds from the empty channels of real frames
  generator.py    batches of scenes by rules
  tape.py         long recordings
  store.py        the export: PNG + YOLO + provenance, tapes as CSV; duplicates, splits
  workspace.py    the work over one dataset, shared by the page and the command line
  recipe.py       recipes and their runs
  verify.py       saved examples against their descriptions
  constants.py    the registry of fixed numbers
  sensor_model.py the sensor model and its profile files; profiles/ the shipped profiles
  framing.py      frames from polls: pooling, tiles, normalization
  calibration.py  a profile fitted to calibration captures and belt recordings
  sensor_report.py  the check of a profile against measurements
  __main__.py     the command line
  web/            the local web interface: server, JSON API, page
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Limitations

- Labels have a single class.
- The thresholds of the kinds of objects and the motion ranges are tuned on the reference
  line (40 channels, rows of 50 ms, channels 9 mm apart); on another line they keep their
  duration and distance when its profile is named (`sensor_profile`), but were not checked
  on the objects of such a line.
- The page text is mostly in Russian, the command line in English.
