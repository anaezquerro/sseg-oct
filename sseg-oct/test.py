from trainers import AdversarialTrainer, ConvolutionalTrainer
import torch.nn as nn
from utils.data import *
from utils.metrics import *
from torch.optim import Adam, RMSprop
from models import *


from argparse import ArgumentParser


image_sizes = {
    'base': Base.IMAGE_SIZE, **BuilderSMP.image_sizes,
    'attnunet': AttentionUNet.IMAGE_SIZE,
    'deformunet': DeformUNet.IMAGE_SIZE
}


transform = t.Compose([
    t.RandomHorizontalFlip(p=0.5),
    t.RandomAffine(degrees=(-7, 7), translate=(0.1, 0.1), scale=(1, 1.2)),
    t.ElasticTransform(alpha=20.0, sigma=10.0)
])


if __name__ == '__main__':
    parser = ArgumentParser(description='Test the different approaches for pathological fluid detection in tomographic images')


    parser.add_argument('model', choices=['base', 'unet', 'linknet', 'pspnet', 'pan', 'attnunet', 'deformunet'], default='base', type=str, help='Approach to use')
    parser.add_argument('mode', choices=['kfold', 'train'], default='train', type=str, help='Mode to use')
    parser.add_argument('--device', '-d', type=str, default='0', help='CUDA device to use')
    parser.add_argument('--route', type=str, default='../OCT-dataset/', help='Local path where OCT dataset is stored')
    parser.add_argument('--image_size', type=int, nargs=2, default=None, help='Image size to use')
    parser.add_argument('--model_path', type=str, default=None, help='Folder to store checkpoints and results of the training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of the optimizer')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size of training')
    parser.add_argument('--num_epochs', type=int, default=100, help='NUmber of training epochs')
    parser.add_argument('--patience', type=int, default=30, help='Tolerance of epochs without improving validation metrics')
    parser.add_argument('--verbose', '-v', action='store_true', help='Output trace in console')
    parser.add_argument('-aug', action='store_true', help='Use data augmentation in training')
    parser.add_argument('-adversarial', action='store_true', help='Use adversarialin training')


    args = parser.parse_args()

    # get device
    device = torch.device(f'cuda:{args.device}' if args.device.isnumeric() and torch.cuda.is_available() else 'cpu')

    # make folders
    if args.model_path is None:
        args.model_path = f'../results/{args.model}' + ('-aug' if args.aug else '') + ('-adv' if args.adversarial else '')
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)


    # get default image size if not specified
    args.image_size = image_sizes[args.model] if not args.image_size else image_sizes

    # create metrics and optimizer
    metrics = {'acc': accuracy, 'prec': precision, 'rec': recall, 'fscore': fscore, 'iou': iou}
    optimizer = lambda params: Adam(params, lr=args.lr)

    # create model builders
    model_builders = {
        'base': Base.build,
        **{key: BuilderSMP(key) for key in ['unet', 'linknet', 'pspnet', 'pan']},
        'attnunet': lambda: AttentionUNet.build(in_channels=1, num_classes=1),
        'deformunet': lambda: DeformUNet.build(input_channels=1)
    }

    common_args = dict(criterion=(nn.BCEWithLogitsLoss() if not args.model in ['attnunet', 'deformunet'] else nn.BCELoss()), metrics=metrics)
    if args.mode == 'kfold':
        common_args.update(
            dict(
                device=device, optimizer_builder=optimizer,
                kfold=OCTDataset.kfold(
                    k=10, image_path=f'{args.route}/images', mask_path=f'{args.route}/masks/',
                    transform=transform if args.aug else None, rsize=args.image_size
                ),
                batch_size=args.batch_size, num_epochs=args.num_epochs, patience=args.patience,
                checkpoint_metric='fscore', save_model=args.model_path, verbose=args.verbose
            )
        )
        if args.adversarial:
            results = AdversarialTrainer.kfold(
                generator_builder=model_builders[args.model], image_size=args.image_size, **common_args
            )
        else:
            results = ConvolutionalTrainer.kfold(
                model_builder=model_builders[args.model], **common_args
            )
    else:
        model = model_builders[args.model]().to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        if args.adversarial:
            trainer = AdversarialTrainer(model, device=device, discriminator=Discriminator(in_channels=1, img_size=args.image_size),
                                         optimizer=optimizer, **common_args)
        else:
            trainer = ConvolutionalTrainer(model, device=device, optimizer=optimizer, **common_args)

        # create data
        train, val = OCTDataset.split(0.1, image_path=f'{args.route}/images/', mask_path=f'{args.route}/masks', transform=transform if args.aug else None, rsize=args.image_size)
        trainloader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
        valloader = DataLoader(val, batch_size=args.batch_size)

        trainer.train(trainloader, valloader, num_epochs=args.num_epochs, patience=args.patience, save_model=f'{args.model_path}/model.pt',
                      checkpoint_metric='fscore', verbose=args.verbose, display=args.verbose, save_plot=f'{args.model_path}/history.pdf')
