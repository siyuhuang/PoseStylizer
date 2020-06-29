import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import time

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

print("testing samples: %d/%d" % (opt.how_many,len(dataset)))

model = model.eval()

# test
images = []
count = 0
j = 0
for i, data in enumerate(dataset):
#     print(' process %d/%d img ..'%(i,opt.how_many))
    if i >= opt.how_many:
        break
    model.set_input(data)
    startTime = time.time()
    model.test()
    endTime = time.time()
    image = model.get_current_p2()
    images += [image]
    if not (i+1)%50:
        count += 1
        visualizer.save_images_uncurated(webpage, images, count)
        print(i)
        images = []
        j = 0

webpage.save()




