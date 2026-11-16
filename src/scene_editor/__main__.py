"""The command line.

    scene-editor serve --data DATASET [--port 8009]      the editor in the browser
    scene-editor generate RECIPE [--output DIR] [--dry-run]
    scene-editor verify OUTPUT --data DATASET            saved examples render again exactly
    scene-editor constants [--output FILE]               the registry of fixed numbers (Markdown)
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
    profile = views.workspace.profile
    print(f"Sensor line: {profile.channels} channels, frames of {profile.frame_length} polls "
          f"(channels on {profile.sensor_axis})", flush=True)
    try:
        server = make_server(views, args.port)
    except OSError as error:
        parser.error(f"Port {args.port} is not available: another editor is probably still running. "
                     f"Stop it or choose another --port. ({error})")
    print(f"Synthetic Scene Editor: http://127.0.0.1:{server.server_port}/", flush=True)
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
