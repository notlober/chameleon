import logging
import os
from pathlib import Path
from dataclasses import asdict

import torch

import cerebras.pytorch as cstorch
from configuration import parse_args
from data import get_dataloader
from model import ChameleonModel

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def main(model_config, run_config, cs_config):
    if run_config.backend == "CSX":
        backend = cstorch.backend(run_config.backend, use_cs_grad_accum=True)
    else:
        backend = cstorch.backend(run_config.backend)

    out_dir = Path(run_config.out_dir)

    if not backend.is_cpu:
        cstorch.amp.use_bfloat16(True)

    with backend.device:
        model = ChameleonModel(model_config)

    compiled_model = cstorch.compile(model, backend)

    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": run_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = cstorch.optim.AdamW(
        param_groups,
        lr=0.1,  # just a placeholder as we are using learning rate scheduling
        weight_decay=run_config.weight_decay,
        correct_bias=False,
        betas=(0.9, 0.95),
        eps=1e-5,
    )
    lr_scheduler = cstorch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        initial_learning_rate=run_config.learning_rate,
        decay_rate=run_config.lr_decay,
        total_iters=run_config.num_steps,
    )
    warmup_scheduler = cstorch.optim.lr_scheduler.LinearLR(
        optimizer,
        initial_learning_rate=0.0,
        end_learning_rate=1.0,
        total_iters=run_config.warmup_steps,
    )
    all_params = (
        p
        for param_group in optimizer.param_groups
        for p in param_group["params"]
    )

    if run_config.checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {run_config.checkpoint_path}")

        state_dict = cstorch.load(run_config.checkpoint_path)

        model.load_state_dict(state_dict["model"])
        if "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])
        if "lr_scheduler" in state_dict:
            lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        global_step = state_dict.get("global_step", 0)
    else:
        global_step = 0

    @cstorch.checkpoint_closure
    def save_checkpoint(step):
        checkpoint_path = out_dir.joinpath(f"checkpoint_{step}.mdl")
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "global_step": step,
            "model_config": asdict(model_config),
        }
        cstorch.save(           state_dict, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    @cstorch.trace
    def training_step(batch):
        nonlocal global_step
        input_ids, labels = batch
        loss = compiled_model(input_ids, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(all_params), run_config.gradient_clip_val
        )
        optimizer.step()
        if global_step < run_config.warmup_steps:
            warmup_scheduler.step()
        else:
            lr_scheduler.step()
        optimizer.zero_grad()
        global_step += 1
        return loss

    writer = cstorch.utils.tensorboard.SummaryWriter(
        log_dir=out_dir.joinpath("train")
    )

    @cstorch.step_closure
    def log_loss(loss, step):
        global_rate = (
            executor.profiler.get_global_rate()
            if hasattr(executor.profiler, "get_global_rate")
            else None
        )

        logger.info(
            f"| Step={step}, "
            f"Loss={loss.item():.5f}, "
            f"GlobalRate={global_rate:.2f} samples/sec"
            if global_rate
            else ""
        )
        print(
            f"Step={step}, Loss={loss.item():.5f}, GlobalRate={global_rate:.2f} samples/sec"
            if global_rate
            else f"Step={step}, Loss={loss.item():.5f}"
        )

        writer.add_scalar("loss", loss.item(), step)
        if global_rate:
            writer.add_scalar("samples_per_second", global_rate, step)

    data_path = os.path.join(run_config.data_file)
    dataloader = cstorch.utils.data.DataLoader(
        get_dataloader,
        data_path,
        run_config.sequence_length,
        run_config.batch_size,
        run_config.seed,
    )
    executor = cstorch.utils.data.DataExecutor(
        dataloader,
        num_steps=run_config.num_steps - global_step,
        checkpoint_steps=run_config.checkpoint_steps,
        cs_config=cs_config,
        writer=writer,
    )

    for step, batch in enumerate(executor, start=global_step + 1):
        if step > run_config.num_steps:
            break
        loss = training_step(batch)
        log_loss(loss, step)
        save_checkpoint(step)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    model_config, run_config, cs_config = parse_args()
    main(model_config, run_config, cs_config)