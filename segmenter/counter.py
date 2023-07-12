from models import *
from utils import count_parameters

if __name__ == '__main__':
    print('------------------ DEFAULT BENCHMARK ------------------')
    base = Base.build()
    print('Number of parameters of the baseline:', count_parameters(base, only_trainable=True))
    unet = BuilderSMP('unet')
    print('Number of parameters of U-Net:', count_parameters(unet(), only_trainable=True))
    linknet = BuilderSMP('linknet')
    print('Number of parameters of LinkNet:', count_parameters(linknet(), only_trainable=True))
    pspnet = BuilderSMP('pspnet')
    print('Number of parameters of PSPNet:', count_parameters(pspnet(), only_trainable=True))
    pan = BuilderSMP('pan')
    print('Number of parameters of PAN:', count_parameters(pan(), only_trainable=True))
    attnunet = AttentionUNet.build(in_channels=1, num_classes=1)
    print('Number of parameters of AttentionUnet:', count_parameters(attnunet, only_trainable=True))
    deformunet = DeformUNet.build(input_channels=1)
    print('Number of parameters of Deformable U-Net:', count_parameters(deformunet, only_trainable=True))

    print('------------------ ADVERSARIAL BENCHMARK ------------------')
    unet = BuilderSMP('unet')
    discriminator = Discriminator(in_channels=1, img_size=BuilderSMP.image_sizes['unet'])
    nparams = count_parameters(unet(), only_trainable=True) + count_parameters(discriminator, only_trainable=True)
    print('Number of parameters of U-Net:', nparams)

    linknet = BuilderSMP('linknet')
    discriminator = Discriminator(in_channels=1, img_size=BuilderSMP.image_sizes['linknet'])
    nparams = count_parameters(linknet(), only_trainable=True) + count_parameters(discriminator, only_trainable=True)
    print('Number of parameters of LinkNet:', nparams)

    pspnet = BuilderSMP('pspnet')
    discriminator = Discriminator(in_channels=1, img_size=BuilderSMP.image_sizes['pspnet'])
    nparams = count_parameters(pspnet(), only_trainable=True) + count_parameters(discriminator, only_trainable=True)
    print('Number of parameters of PSPNet:', nparams)

    pan = BuilderSMP('pan')
    discriminator = Discriminator(in_channels=1, img_size=BuilderSMP.image_sizes['pan'])
    nparams = count_parameters(pan(), only_trainable=True) + count_parameters(discriminator, only_trainable=True)
    print('Number of parameters of PAN:', nparams)
