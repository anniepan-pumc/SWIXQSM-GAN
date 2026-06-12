import os
import torch
from tqdm import tqdm

def fit2d(
    gen, disc, train_dl, epochs,
    scheduler_G, scheduler_D,
    optimizer_G, optimizer_D,
    scaler,                 # torch.cuda.amp.GradScaler() or None
    L1_loss,                # nn.L1Loss
    gan_loss,               # nn.BCEWithLogitsLoss
    device,
    fid_cal,                # torchmetrics FrechetInceptionDistance
    save_path,
    rand_slice=True,
    use_amp=False,
    grad_accum_steps=1,
    content_w=0.2
):
    """
    Assumes:
      - disc is PatchGAN-like and returns logits of shape (B,1,H',W') or (B,1).
      - gan_loss is BCEWithLogitsLoss.
      - dataset returns (idx, lr_img, hr_img)
    """

    t_loss_G, t_loss_D = [], []
    fids_t2f, fids_q2s = [], []

    amp_enabled = bool(use_amp and device.type == "cuda" and scaler is not None)

    for epoch in range(epochs):
        gen.train()
        disc.train()

        e_loss_G, e_loss_D = [], []
        fid_t2f_sum = 0.0
        fid_q2s_sum = 0.0

        optimizer_G.zero_grad(set_to_none=True)
        optimizer_D.zero_grad(set_to_none=True)

        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for i, batch in enumerate(pbar):
            idx, lr_img, hr_img = batch

            if not rand_slice:
                lr_img = lr_img.view(lr_img.shape[0] * lr_img.shape[1], 1, lr_img.shape[2], lr_img.shape[3])
                hr_img = hr_img.view(hr_img.shape[0] * hr_img.shape[1], 1, hr_img.shape[2], hr_img.shape[3])

            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            accum = 1.0 / max(1, grad_accum_steps)
            do_step = ((i + 1) % grad_accum_steps == 0) or ((i + 1) == len(train_dl))

            # -------------------------
            # Discriminator update
            # -------------------------
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                with torch.no_grad():
                    fake_hr_detached = gen(lr_img)

                pred_real = disc(hr_img)
                pred_fake = disc(fake_hr_detached.detach())

                valid = torch.ones_like(pred_real)
                fake = torch.zeros_like(pred_fake)

                loss_D = 0.5 * (gan_loss(pred_real, valid) + gan_loss(pred_fake, fake))
                loss_D_scaled = loss_D * accum

            if amp_enabled:
                scaler.scale(loss_D_scaled).backward()
            else:
                loss_D_scaled.backward()

            # -------------------------
            # Generator update
            # -------------------------
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                fake_hr = gen(lr_img)
                pred_fake_for_G = disc(fake_hr)

                valid = torch.ones_like(pred_fake_for_G)

                adv = gan_loss(pred_fake_for_G, valid)
                content = L1_loss(fake_hr, hr_img)

                loss_G = adv + content_w * content
                loss_G_scaled = loss_G * accum

            if amp_enabled:
                scaler.scale(loss_G_scaled).backward()
            else:
                loss_G_scaled.backward()

            # -------------------------
            # Optimizer step (accum)
            # -------------------------
            if do_step:
                if amp_enabled:
                    scaler.step(optimizer_D)
                    scaler.step(optimizer_G)
                    scaler.update()
                else:
                    optimizer_D.step()
                    optimizer_G.step()

                optimizer_D.zero_grad(set_to_none=True)
                optimizer_G.zero_grad(set_to_none=True)

            e_loss_D.append(loss_D.detach().item())
            e_loss_G.append(loss_G.detach().item())

            # -------------------------
            # FID (fixed reset + sane uint8 conversion)
            # Your tensors are normalized to [-1, 1] (from Normalize(0.5, 0.5)).
            # Convert back to [0,255] uint8 for torchmetrics FID.
            # -------------------------
            with torch.no_grad():
                def to_uint8_3ch(x):
                    # x: (B,1,H,W) in [-1,1]
                    x = (x * 0.5 + 0.5).clamp(0, 1)          # -> [0,1]
                    x = (x * 255.0).round().to(torch.uint8)  # -> uint8
                    return x.expand(-1, 3, -1, -1).cpu()

                real_swi = to_uint8_3ch(lr_img)
                real_qsm = to_uint8_3ch(hr_img)
                fake_qsm = to_uint8_3ch(fake_hr)

                # real vs fake_qsm
                fid_cal.reset()
                fid_cal.update(real_qsm, real=True)
                fid_cal.update(fake_qsm, real=False)
                fid_t2f_sum += float(fid_cal.compute())

                # qsm vs swi
                fid_cal.reset()
                fid_cal.update(real_qsm, real=True)
                fid_cal.update(real_swi, real=False)
                fid_q2s_sum += float(fid_cal.compute())

            pbar.set_postfix({
                "loss_D": sum(e_loss_D) / max(1, len(e_loss_D)),
                "loss_G": sum(e_loss_G) / max(1, len(e_loss_G)),
            })

        if scheduler_G is not None:
            scheduler_G.step()
        if scheduler_D is not None:
            scheduler_D.step()

        avg_D = sum(e_loss_D) / max(1, len(e_loss_D))
        avg_G = sum(e_loss_G) / max(1, len(e_loss_G))
        t_loss_D.append(avg_D)
        t_loss_G.append(avg_G)

        fids_t2f.append(fid_t2f_sum / max(1, len(train_dl)))
        fids_q2s.append(fid_q2s_sum / max(1, len(train_dl)))

        print(f"{epoch+1}/{epochs} -- Gen Loss: {avg_G:.6f} -- Disc Loss: {avg_D:.6f}")
        print(f"{epoch+1}/{epochs} -- FID t2f: {fids_t2f[-1]:.6f} -- FID q2s: {fids_q2s[-1]:.6f}")

        os.makedirs(save_path, exist_ok=True)
        torch.save(gen.state_dict(), os.path.join(save_path, f"gen_{epoch}.pt"))
        torch.save(disc.state_dict(), os.path.join(save_path, f"disc_{epoch}.pt"))

    return t_loss_G, t_loss_D, fids_t2f, fids_q2s
