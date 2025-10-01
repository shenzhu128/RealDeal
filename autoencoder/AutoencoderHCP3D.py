import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import argparse
import nibabel as nib

from utils.dataset.hcp import HCPDataset
from utils.models.models3D import ResNetAutoencoder3d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def train(
    train_loader,
    model,
    optimizer,
    start_epoch=0,
    num_epochs=100,
    lr=1.0e-3,
    weight_decay=0.0,
    accumulation_steps=1,
    save_every=-1,
):
    mse_loss = nn.MSELoss()
    # bce_loss = nn.BCELoss()

    if save_every == -1:
        save_every = num_epochs

    nTrain = len(train_loader.dataset)
    print("nTrain =", nTrain)
    for epoch in range(start_epoch, start_epoch + num_epochs):
        tot_loss = 0
        optimizer.zero_grad()
        for i, batchX in enumerate(train_loader):
            x = batchX["data"].to("cuda")

            y = model(x)

            loss = mse_loss(y, x) / float(accumulation_steps)
            loss.backward()

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            batch_scale = x.shape[0] / nTrain
            tot_loss += (
                float(accumulation_steps) * batch_scale * loss.data.cpu().numpy()
            )

        if (epoch + 1) % save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "tot_loss": tot_loss,
                },
                "hcp3d_autoencoder_checkpoint_epoch{:03}.pth".format(epoch),
            )
            print("Saving checkpoint at epoch", epoch)

        print("Epoch", epoch, " loss =", tot_loss)

    return model


def test(test_loader, model):
    model = model.eval()

    mse_loss = nn.MSELoss()

    nTest = len(test_loader.dataset)
    print("nTest =", nTest)

    tot_loss = 0
    with torch.no_grad():
        for i, batchX in enumerate(test_loader):
            x = batchX["data"].to(device)

            y = model(x)

            loss = mse_loss(y, x)

            batch_scale = x.shape[0] / nTest
            tot_loss += batch_scale * loss.data.cpu().numpy()

            if i < 10:
                ## Output nifti images of reconstructions from first batches
                x = x.data.cpu().numpy()
                y = y.data.cpu().numpy()

                ximg = nib.Nifti1Image(np.squeeze(x[0]), affine=np.eye(4))
                nib.save(ximg, "orig3D_{:02}.nii.gz".format(i))

                yimg = nib.Nifti1Image(np.squeeze(y[0]), affine=np.eye(4))
                nib.save(yimg, "recon3D_{:02}.nii.gz".format(i))

        print("Test loss =", tot_loss)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="base path for HCP data")
    parser.add_argument("--checkpoint", help="checkpoint from which to start")
    parser.add_argument("--batch_size", type=int, default=4, help="the batch size")
    parser.add_argument(
        "--random_crop",
        type=int,
        default=0,
        help="size to random crop an image in each dimension (set to 0 for no cropping)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="the number of epochs to train"
    )
    parser.add_argument(
        "--learn_rate", type=float, default=1.0e-3, help="the learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1.0e-5, help="weight decay"
    )
    parser.add_argument(
        "--accumulate", type=int, default=1, help="number of gradients to accumulate"
    )
    parser.add_argument(
        "--randseed",
        type=int,
        default=42,
        help="random seed for random train/test split",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=-1,
        help="number of epochs to save checkpoint (set to -1 to only save at end)",
    )
    parser.add_argument(
        "--run_test", type=bool, default=True, help="whether to run testing at the end"
    )
    args = parser.parse_args()

    print("Arguments used:")
    print(args)

    hcp_data = HCPDataset(
        args.data_path, "T1w_cropped.nii.gz", random_crop=args.random_crop
    )
    num_data = len(hcp_data)

    generator = torch.Generator().manual_seed(args.randseed)
    num_train = (int)(0.8 * num_data)
    num_test = num_data - num_train
    print("Number of training data =", num_train)
    print("Number of testing data =", num_test, flush=True)
    [train_data, test_data] = random_split(
        hcp_data, [num_train, num_test], generator=generator
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    torch.manual_seed(args.randseed)
    model = ResNetAutoencoder3d(num_channels=64, nonlinearity=F.leaky_relu).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learn_rate, weight_decay=args.weight_decay
    )
    # optimizer = torch.optim.SGD(model.parameters(), lr = 1.0e-3, weight_decay = 1.0e-5)

    start_epoch = 0
    if args.checkpoint != None:
        print("Loading checkpoint:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

    if args.epochs > 0:
        model = model.train()
        model = train(
            train_loader,
            model,
            optimizer,
            start_epoch=start_epoch,
            num_epochs=args.epochs,
            accumulation_steps=args.accumulate,
            save_every=args.save_every,
        )

    if args.run_test == True:
        # Don't random crop at test time
        hcp_data = HCPDataset(args.data_path, "T1w_cropped.nii.gz", random_crop=0)

        generator = torch.Generator().manual_seed(args.randseed)
        [train_data, test_data] = random_split(
            hcp_data, [num_train, num_test], generator=generator
        )

        test_loader = DataLoader(
            test_data, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        test(test_loader, model)


if __name__ == "__main__":
    run()
