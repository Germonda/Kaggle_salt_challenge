import os
import ntpath
import pandas as pd
from options.val_options import ValOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html


if __name__ == '__main__':
    opt = ValOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    pred_dict = dict()
    for i, data in enumerate(dataset):
        if i >= opt.num_val:
            break
        model.set_input(data)
        model.val()
        score = model.get_score()
        rle_mask = model.get_rle_mask()
        id = ntpath.basename(model.image_paths[0]).split('.')[0]
        pred_dict[id] = rle_mask
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 1 == 0:
            print('processing (%04d)-th image... %s. Score: %0.2f' % (i, img_path, score))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # save the website
    webpage.save()

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv("submission.csv")
