from skimage import exposure
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
from PIL import Image

#print(img)
# array = np.array(img)

# red = array[:,:,0]
# blue = array[:,:,1]
# green = array[:,:,2]

# # using the variable ax for single a Axes
# fig, ax = plt.subplots(2, 3)
# ax[0, 0].imshow(exposure.equalize_hist(red), cmap="gray")
# ax[0, 1].imshow(exposure.equalize_hist(blue), cmap="gray")
# ax[0, 2].imshow(exposure.equalize_hist(green), cmap="gray")

# ax[1, 0].imshow(red, cmap="gray")
# ax[1, 1].imshow(blue, cmap="gray")
# ax[1, 2].imshow(green, cmap="gray")


# preprocess_train = transforms.Compose([

#     transforms.Lambda(lambda x: x.convert('RGB')),
#     transforms.Resize((320, 320)),
#     transforms.RandomRotation(10),
#     transforms.RandomResizedCrop((320, 320), scale=(0.80, 1)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2), #color
#     transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.adjust_sharpness(x, sharpness_factor=1.3))], p=0.3),
#     transforms.Lambda(lambda x:equalization(x)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# fig, ax = plt.subplots(1, 10)
# img = Image.open('../images/000001-10.png')
# ax[0].imshow(img)
# for i in range(1, 10):
# 	new_img = preprocess_train(img)
# 	ax[i].imshow(np.transpose(new_img.numpy(), (1, 2, 0)))
# plt.show()


def equalization(img):

	array = np.array(img)
	if array.shape[2] > 1:
		red, blue, green = array[:,:,0], array[:,:,1], array[:,:,2]

		# normalize each channel
		red = exposure.equalize_hist(red)
		blue = exposure.equalize_hist(blue)
		green = exposure.equalize_hist(green)

		new_array = np.stack((red, blue, green), axis = 2)
		return Image.fromarray((new_array*255).astype(np.uint8))

	img = exposure.equalize_hist(np.array(img))
	return Image.fromarray(img*255).convert("L")


