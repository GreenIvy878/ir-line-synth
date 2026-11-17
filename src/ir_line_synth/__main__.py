"""The command line.

    ir-line-synth serve --data DATASET [--port 8009]      the editor in the browser
    ir-line-synth generate RECIPE [--output DIR] [--dry-run]
    ir-line-synth verify OUTPUT --data DATASET            saved examples render again exactly
    ir-line-synth constants [--output FILE]               the registry of fixed numbers (Markdown)
    ir-line-synth fit-sensor --calibration PATH ... --output PROFILE   a sensor profile from measurements
    ir-line-synth sensor-report --profile PROFILE --calibration PATH ...   the check of a profile
"""

import argparse
from pathlib import Path
import sys

from . import PROGRAM, __version__
from .dataset import load_data_description
from .recipe import run
from .verify import verify


def serve(args, parser):
    from .web.server import make_server
    from .web.views import EditorViews
    if not 0 <= args.port <= 65535:
        parser.error("port must be in [0, 65535]")
    try:
        data = load_data_description(args.data)
        views = EditorViews(data)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    # Exports do not record the dataset path, so say here which one the frames come from.
    print("Source dataset: " + ", ".join(f"{part} {data[part]}" for part in views.workspace.source_splits()))
    geometry = views.workspace.geometry
    print(f"Sensor line: {geometry.channels} channels, frames of {geometry.frame_length} polls "
          f"(channels on {geometry.sensor_axis})", flush=True)
    try:
        server = make_server(views, args.port)
    except OSError as error:
        parser.error(f"Port {args.port} is not available: another editor is probably still running. "
                     f"Stop it or choose another --port. ({error})")
    print(f"IR-Line-Synth: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def generate(args, parser):
    record = run(args.recipe, args.output, args.dry_run, report=lambda line: print(line, flush=True))
    for note in record["notes"]:
        print("  " + note)
    if args.dry_run:
        print("Dry run: nothing was written.")
    else:
        print(f"Record of the run: {record['record']}")
    return 0


def check(args, parser):
    results = verify(args.output, load_data_description(args.data))
    failed = [item for item in results if not item.ok]
    for item in failed:
        print(f"MISMATCH {item.kind} {item.path}: {item.problem}")
    print(f"{len(results) - len(failed)} of {len(results)} examples render again exactly")
    return 1 if failed else 0


def constants(args, parser):
    from .constants import markdown, unregistered
    missing, stale = unregistered()
    if missing or stale:
        parser.error(f"The registry is out of date: unregistered {missing}, no longer in the code {stale}")
    text = markdown()
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8", newline="\n")
        print(f"Written: {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def recordings_of(args):
    """The recordings named by --recordings and --only: (polls of each, their numbers, their files)."""
    from .calibration import number_ranges, numbered
    from .framing import read_recording
    if not args.recordings:
        return [], [], []
    files = numbered(args.recordings, number_ranges(args.only) if args.only else None)
    numbers = [int("".join(ch for ch in path.stem if ch.isdigit())) for path in files]
    return [read_recording(path, args.channels) for path in files], numbers, files


def fit_sensor(args, parser):
    from .calibration import fit_profile, read_captures
    from .sensor_model import FrameSpec, save_profile
    calibration = read_captures(args.calibration, args.clip)
    args.channels = calibration.channels
    recordings, _, files = recordings_of(args)
    model = fit_profile(calibration, args.name, args.boards, args.adc_max,
                        FrameSpec(args.pool, args.length, args.normalize), recordings, files)
    header = (f"# Sensor profile fitted by {PROGRAM} {__version__} (ir-line-synth fit-sensor); "
              "see docs/sensor_model.md.\n")
    save_profile(model, args.output, header)
    print(f"Written: {args.output} ({model.channels} channels"
          + (f", bare belt of {len(recordings)} recordings" if recordings else "") + ")")
    return 0


def sensor_report(args, parser):
    from .calibration import read_captures
    from .sensor_model import load_profile
    from .sensor_report import report
    model = load_profile(args.profile)
    calibration = read_captures(args.calibration, args.clip)
    args.channels = model.channels
    recordings, numbers, files = recordings_of(args)
    text = report(model, calibration, recordings, numbers, files, args.dataset, args.recordings)
    Path(args.output).write_text(text, encoding="utf-8", newline="\n")
    print(f"Written: {args.output}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog=PROGRAM, description="Synthetic annotated data for lines of reflective IR distance sensors")
    parser.add_argument("--version", action="version", version=f"{PROGRAM} {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    dataset_help = "A YAML file with a data section (for example a data.yaml), or a folder with data.yaml"

    command = commands.add_parser("serve", help="Open the editor in the browser")
    command.add_argument("--data", metavar="DATASET", required=True, help=dataset_help)
    command.add_argument("--port", type=int, default=8009)
    command.set_defaults(handler=serve)

    command = commands.add_parser("generate", help="Run a recipe: make and save scenes and tapes")
    command.add_argument("recipe", metavar="RECIPE", help="The recipe (YAML), see docs/recipes.md")
    command.add_argument("--output", metavar="DIR", help="Save here instead of the output of the recipe")
    command.add_argument("--dry-run", action="store_true", help="Make and check everything, write nothing")
    command.set_defaults(handler=generate)

    command = commands.add_parser("verify", help="Check that saved examples render again exactly")
    command.add_argument("output", metavar="OUTPUT", help="A synthetic output folder or one provenance file")
    command.add_argument("--data", metavar="DATASET", required=True, help=dataset_help)
    command.set_defaults(handler=check)

    command = commands.add_parser("constants", help="The registry of the program's fixed numbers (Markdown)")
    command.add_argument("--output", metavar="FILE", help="Write it to this file (UTF-8)")
    command.set_defaults(handler=constants)

    def measurements(command):
        command.add_argument("--calibration", metavar="PATH", required=True,
                             help="Calibration captures: a folder of data_<mm>.csv or a table channel,distance_mm,value")
        command.add_argument("--clip", type=int, default=2047,
                             help="Saturation of the captures: a node with a sample at or above it is left out")
        command.add_argument("--recordings", metavar="DIR", help="Recordings of a working line (CSV, a row per poll)")
        command.add_argument("--only", metavar="NUMBERS",
                             help="Only the recordings with these numbers at the end of the name, e.g. 2-22")

    command = commands.add_parser("fit-sensor", help="Fit a sensor profile to calibration captures and recordings")
    measurements(command)
    command.add_argument("--name", required=True, help="Name of the profile")
    command.add_argument("--boards", type=int, default=1, help="Boards of the line (channels of a board in a row)")
    command.add_argument("--adc-max", type=int, required=True, help="The largest code of the line in operation")
    command.add_argument("--pool", type=int, default=1, help="Polls averaged into a frame row")
    command.add_argument("--length", type=int, default=40, help="Rows of a frame")
    command.add_argument("--normalize", choices=("tile_minmax", "none"), default="none")
    command.add_argument("--output", metavar="PROFILE", required=True)
    command.set_defaults(handler=fit_sensor)

    command = commands.add_parser("sensor-report", help="Check a sensor profile against the measurements (Markdown)")
    command.add_argument("--profile", required=True, help="A profile file or the name of a shipped profile")
    measurements(command)
    command.add_argument("--dataset", metavar="DIR", help="A dataset with manifest.csv (name, split, recording, tile)")
    command.add_argument("--output", metavar="FILE", required=True)
    command.set_defaults(handler=sensor_report)
    return parser


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args, parser)
    except (ValueError, OSError) as error:
        print(f"{PROGRAM} {args.command}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
