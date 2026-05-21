from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Sequence

import typer

from simcortex.preproc.fs_to_mni import app as fs_to_mni_app


app = typer.Typer(
    help="SimCortex (SC) CLI: preprocessing, segmentation, initial surfaces, and deformation."
)

# This lets commands forward Hydra-style arguments such as:
#   dataset.root=/path
#   trainer.max_epochs=100
#   --config-name train
#   --multirun
#
# Without this, Typer may incorrectly interpret some Hydra options as CLI options.
FORWARD_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


def _format_command(cmd: Sequence[str]) -> str:
    """Return a shell-readable command string for logging/debugging."""
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_module(
    module: str,
    overrides: Sequence[str] | None = None,
    *,
    torchrun: bool = False,
    nproc_per_node: int = 1,
) -> int:
    """
    Run a SimCortex stage module in a subprocess.

    Parameters
    ----------
    module:
        Python module path, e.g. ``simcortex.seg.train``.

    overrides:
        Extra arguments forwarded to the target module, typically Hydra overrides.

    torchrun:
        If True, launch the module with torch.distributed.run.

    nproc_per_node:
        Number of processes/GPUs per node when torchrun is enabled.

    Returns
    -------
    int
        The subprocess return code.
    """
    if nproc_per_node < 1:
        raise typer.BadParameter(
            f"--nproc-per-node must be >= 1, got {nproc_per_node}."
        )

    forwarded_args = list(overrides or [])

    if torchrun:
        # Safer than calling "torchrun" directly, because this guarantees the launcher
        # comes from the same Python environment as the active simcortex command.
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc_per_node}",
            "-m",
            module,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            module,
        ]

    cmd.extend(forwarded_args)

    typer.echo(f"[SimCortex CLI] Running: {_format_command(cmd)}")
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


# ---------------------------------------------------------------------
# Stage 1 — Preprocessing
# ---------------------------------------------------------------------
# Kept as a direct Typer sub-app to preserve the exact existing behavior of
# simcortex.preproc.fs_to_mni. We should revisit this after reviewing
# fs_to_mni.py to decide whether it should also be subprocess-launched.
app.add_typer(
    fs_to_mni_app,
    name="fs-to-mni",
    help="Stage 1 — Preprocessing: FreeSurfer/native space to MNI152.",
)


# ---------------------------------------------------------------------
# Stage 2 — Segmentation
# ---------------------------------------------------------------------
seg_app = typer.Typer(
    help="Stage 2 — Segmentation: 3D U-Net in MNI152 space."
)
app.add_typer(seg_app, name="seg")


@seg_app.command(
    "train",
    help="Train the segmentation model using Hydra config overrides.",
    context_settings=FORWARD_CONTEXT_SETTINGS,
)
def seg_train(
    ctx: typer.Context,
    torchrun: bool = typer.Option(
        False,
        "--torchrun",
        help="Launch training with torch.distributed.run for multi-GPU DDP.",
    ),
    nproc_per_node: int = typer.Option(
        1,
        "--nproc-per-node",
        min=1,
        help="Number of processes/GPUs per node when --torchrun is enabled.",
    ),
) -> None:
    raise typer.Exit(
        run_module(
            "simcortex.seg.train",
            ctx.args,
            torchrun=torchrun,
            nproc_per_node=nproc_per_node,
        )
    )


@seg_app.command(
    "infer",
    help="Run segmentation inference using Hydra config overrides.",
    context_settings=FORWARD_CONTEXT_SETTINGS,
)
def seg_infer(ctx: typer.Context) -> None:
    raise typer.Exit(
        run_module(
            "simcortex.seg.inference",
            ctx.args,
        )
    )


@seg_app.command(
    "eval",
    help="Evaluate segmentation predictions using Hydra config overrides.",
    context_settings=FORWARD_CONTEXT_SETTINGS,
)
def seg_eval(ctx: typer.Context) -> None:
    raise typer.Exit(
        run_module(
            "simcortex.seg.eval",
            ctx.args,
        )
    )


# ---------------------------------------------------------------------
# Stage 3 — Initial Surfaces
# ---------------------------------------------------------------------
initsurf_app = typer.Typer(
    help="Stage 3 — InitSurf: generate initial WM/pial surfaces, SDFs, and ribbon outputs."
)
app.add_typer(initsurf_app, name="initsurf")


@initsurf_app.command(
    "generate",
    help="Generate initial WM/pial surfaces from segmentation predictions.",
    context_settings=FORWARD_CONTEXT_SETTINGS,
)
def initsurf_generate(ctx: typer.Context) -> None:
    raise typer.Exit(
        run_module(
            "simcortex.initsurf.generate",
            ctx.args,
        )
    )


# ---------------------------------------------------------------------
# Stage 4 — Deformation
# ---------------------------------------------------------------------
deform_app = typer.Typer(
    help="Stage 4 — Deformation: train, infer, and evaluate cortical surface deformation."
)
app.add_typer(deform_app, name="deform")


@deform_app.command(
    "train",
    help="Train the deformation model using Hydra config overrides.",
    context_settings=FORWARD_CONTEXT_SETTINGS,
)
def deform_train(
    ctx: typer.Context,
    torchrun: bool = typer.Option(
        False,
        "--torchrun",
        help="Launch training with torch.distributed.run for multi-GPU DDP.",
    ),
    nproc_per_node: int = typer.Option(
        1,
        "--nproc-per-node",
        min=1,
        help="Number of processes/GPUs per node when --torchrun is enabled.",
    ),
) -> None:
    raise typer.Exit(
        run_module(
            "simcortex.deform.train",
            ctx.args,
            torchrun=torchrun,
            nproc_per_node=nproc_per_node,
        )
    )


@deform_app.command(
    "infer",
    help="Run deformation inference using Hydra config overrides.",
    context_settings=FORWARD_CONTEXT_SETTINGS,
)
def deform_infer(ctx: typer.Context) -> None:
    raise typer.Exit(
        run_module(
            "simcortex.deform.inference",
            ctx.args,
        )
    )


@deform_app.command(
    "eval",
    help="Evaluate deformed surfaces and optional collision metrics.",
    context_settings=FORWARD_CONTEXT_SETTINGS,
)
def deform_eval(ctx: typer.Context) -> None:
    raise typer.Exit(
        run_module(
            "simcortex.deform.eval",
            ctx.args,
        )
    )


if __name__ == "__main__":
    app()