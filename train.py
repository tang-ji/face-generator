import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default='input/', help="path to folder containing images")
parser.add_argument("--mode", required="pix2pix", choices=["pix2pix", "ICGAN"])
parser.add_argument("--label_dir", default='label.txt', help="where to put output files")
parser.add_argument("--output_dir", default='output/', help="where to put output files")
parser.add_argument("--attributes", default=2, help="how many attributes want to use")

parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--max_epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=1, help="update summaries every summary_freq batches")
parser.add_argument("--sample_freq", type=int, default=200, help="output samples every sample_freq batches")
parser.add_argument("--sample_dir", default='samples/', default=200, help="where to put output samples")
parser.add_argument("--save_freq", type=int, default=1000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, help="scale images to this size")
a = parser.parse_args()

if a.mode == 'pix2pix':
	from FaceGenerator import *
	gan = FaceGenerator(a)
	if a.checkpoint is not None:
		gan = pk.load(open('checkpoints/model.pkl', 'rb'))
	gan.train(epochs=a.max_epochs, batch_size=a.batch_size, sample_interval=a.sample_freq)
elif a.mode == 'ICGAN':
	from ICGAN_FaceGenerator import *
	gan = ICGAN(a)
	if a.checkpoint is not None:
		gan = pk.load(open('checkpoints/model.pkl', 'rb'))
	gan.train(epochs=a.max_epochs, batch_size=a.batch_size, sample_interval=a.sample_freq)