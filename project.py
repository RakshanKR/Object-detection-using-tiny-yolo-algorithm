from google.colab import drive
drive.mount('/content/drive')

!pip install gluoncv

%matplotlib inline

!pip install mxnet

from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
%cd 

im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/dog.jpg',
                          path='dog.jpg')
x, img1 = data.transforms.presets.yolo.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)

im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/eagle.jpg',
                          path='eagle.jpg')
x, img2 = data.transforms.presets.yolo.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)

im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/horses.jpg',
                          path='horses.jpg')
x, img3 = data.transforms.presets.yolo.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)

im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/person.jpg',
                          path='person.jpg')
x, img4 = data.transforms.presets.yolo.load_test(im_fname, short=512)

x, img3 = data.transforms.presets.yolo.load_test('/content/drive/MyDrive/project (1)/ts/00003.jpg', short=512) #change here: which img we taken  has input taken that has img 1,2,3,or4
class_IDs, scores, bounding_boxs = net(x)

ax = utils.viz.plot_bbox(img3, bounding_boxs[0], scores[0],#here also
                         class_IDs[0], class_names=net.classes)
plt.show()