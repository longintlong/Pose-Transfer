import time
import cv2
import os
import shutil
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import time
from HPE.web_demo import extract_pose_main
from tool.generate_pose_map_fashion import compute_pose


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
extract_pose_main(opt.video_path, opt.source_path)
compute_pose(opt.dataroot+"demo-resize-annotation-test.csv",opt.dataroot+"testK",sigma=6)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

print(opt.how_many)
print(len(dataset))

model = model.eval()
print(model.training)

opt.how_many = 999999
fake_path = "./fake_results/demo"
shutil.rmtree(fake_path)
os.mkdir(fake_path)
# test
for i, data in enumerate(dataset):
    print(' process %d/%d img ..'%(i,opt.how_many))
    if i >= opt.how_many:
        break
    model.set_input(data)
    startTime = time.time()
    model.test()
    endTime = time.time()
    print(endTime-startTime)
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    img_path = [img_path]
    print(img_path)
    visualizer.save_images(webpage, visuals, img_path)
webpage.save()

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
videoWriter = cv2.VideoWriter('demo.avi', fourcc,
                                  30, (256,256))

pics = os.listdir(fake_path)
for pic in pics:
    read_img = cv2.imread(os.path.join(fake_path,pic))
    videoWriter.write(read_img)
    cv2.waitKey(20)
videoWriter.release()
# path = "./demo_data/test/"+"source_woman.jpg"
# img = cv2.imread(path)
# img = img[:,40:216,:]
# cv2.imwrite("./demo_data/test/"+"source_woman_.jpg",img)


