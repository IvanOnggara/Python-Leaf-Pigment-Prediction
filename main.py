from __future__ import print_function
from tkinter import *
# from tkinter import ttk
import cv2
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import pandas as pd
import keras
from keras.layers import Dense, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential
import matplotlib.pylab as plt
import os
import numpy as np
from keras.optimizers import RMSprop,Adam,SGD,Adagrad,Adadelta,Adamax, Nadam
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.models import load_model
import csv

def buildModel(model,arsitektur,input_shape,num_classes):
	if arsitektur == "lenet":
		model.add(Conv2D(6, kernel_size=(5, 5),
	                 strides=(2, 2),
	                 activation='relu',
	                 input_shape=input_shape))

		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Conv2D(16, (5, 5), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())
		model.add(Dense(120, activation='relu'))
		model.add(Dense(num_classes))
		model.add(LeakyReLU(alpha=0.3))

	elif arsitektur == "alexnet":

		model.add(Conv2D(48, kernel_size=(11, 11),
	                 strides=(4, 4),
	                 activation='relu',
	                 input_shape=input_shape))
		
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
		model.add(Conv2D(128, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(172, (2, 2), activation='relu'))
		model.add(Conv2D(172, (2, 2), activation='relu'))
		model.add(Conv2D(128, (2, 2), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Flatten())
		model.add(Dense(4096, activation='relu'))
		model.add(Dense(4096, activation='relu'))
		model.add(Dense(1000, activation='relu'))
		model.add(Dense(num_classes))
		model.add(LeakyReLU(alpha=0.3))
	elif arsitektur == "shallow":
		model.add(Conv2D(32, kernel_size=(3, 3),
	        input_shape=input_shape,activation='relu'))
		# add FC layer
		model.add(Flatten())
		model.add(Dense(num_classes))
		model.add(LeakyReLU(alpha=0.3))

	return model


def trainCNN():
	f = filedialog.asksaveasfile(mode='w', defaultextension=".h5")
	name = f.name
	imagetype = mode1.get()
	batch_size = batch.get()
	epoch_size = epoch.get()
	optimizer = opt.get()
	if optimizer == "rmsprop":
		optimizer = RMSprop()
	elif optimizer == 'adam':
		optimizer = Adam()
	elif optimizer == 'adamax':
		optimizer = Adamax()
	elif optimizer == 'adadelta':
		optimizer = Adadelta()
	elif optimizer == 'nadam':
		optimizer = Nadam()
	elif optimizer == 'sgd':
		optimizer = SGD()
	elif optimizer == 'adagrad':
		optimizer = Adagrad()

	arsitektur = ars.get()
	csv1 = pd.read_csv('DATA INDEX/data-notnorm.csv')
	num_classes = 5
	labels=[]

	if arsitektur == "lenet" or arsitektur =="shallow":
		image_x_size = 32
		image_y_size = 32
	else:
		image_x_size = 120
		image_y_size = 120

	img_data_list=[]

	if imagetype == "rgb":
		num_channels = 3
		PATH = 'HANDPHONE'
		data_path = PATH
		img_list = os.listdir(data_path)

		print ('Loaded the images of dataset-')
		for img in img_list:
			input_img=cv2.imread(data_path + '/'+ img )
			input_img_resize=cv2.resize(input_img,(image_x_size,image_y_size))
			# print(input_img_resize)
			img_data_list.append(input_img_resize)
			for index, row in csv1.iterrows():
				if img.split('.')[0] == row['Nama']:
					labels.append([row['KloroGreen'],row['KloroRedEdge'],row['CaroGreen'],row['CaroRedEdge'],row['Antho']])
	else:
		num_channels = 10
		PATH = 'LABORATORIUM'
		data_path = PATH
		fold_list = os.listdir(data_path)

		print ('Loaded the images of dataset-')
		for fold in fold_list:
			img_list = os.listdir(data_path+"/"+fold)
			construct = np.zeros(shape=(image_x_size,image_y_size,10))
			dimensions = 0
			for img in img_list:

				input_img=cv2.imread(data_path+"/"+fold + '/'+ img )
				input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
				input_img_resize=cv2.resize(input_img,(image_x_size,image_y_size))
				construct[:,:,dimensions]=input_img_resize
				dimensions+=1

			img_data_list.append(construct)
			for index, row in csv1.iterrows():
				if str(fold) == row['Nama']:
					labels.append([row['KloroGreen'],row['KloroRedEdge'],row['CaroGreen'],row['CaroRedEdge'],row['Antho']])
					

	img_data = np.array(img_data_list)
	img_data = img_data.astype(np.floating)
	img_data /= 255

	# num_of_samples = img_data.shape[0]
	labels = np.array(labels)

	x,y = shuffle(img_data,labels, random_state=2)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
	x_train = x_train.reshape(x_train.shape[0], image_x_size, image_y_size, num_channels)
	x_test = x_test.reshape(x_test.shape[0], image_x_size, image_y_size, num_channels)
	input_shape = (image_x_size, image_y_size, num_channels)

	model = Sequential()
	model = buildModel(model,arsitektur,input_shape,num_classes)
	model.compile(loss="mean_squared_error",
	              optimizer=optimizer,
	              metrics=['accuracy'])

	class AccuracyHistory(keras.callbacks.Callback):
	    def on_train_begin(self, logs={}):
	        self.loss_val = []
	        self.loss = []

	    def on_epoch_end(self, batch, logs={}):
	        self.loss_val.append(logs.get('val_loss'))
	        self.loss.append(logs.get('loss'))
	        

	    def on_train_end(self, logs={}):
	    	
	    	model.save(str(name))

	class AccuracyA(keras.callbacks.Callback):
		
	    def on_train_begin(self, logs={}):
	    	pass
	        # print('TRAIN BEGINNNNNNN')
	        # root.after(100, open_top)

	    def on_epoch_end(self, batch, logs={}):
	    	pass
	        # print('EPOCH ENDDDDDDDD')
	        # root.mainloop()

	    def on_train_end(self, logs={}):
	    	pass
	    	# print('some')
	    	# self.pop.destroy()

	    # def open_top(self):
	    # 	self.pop = Toplevel()

	history = AccuracyHistory()
	abc = AccuracyA()

	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epoch_size,
	          verbose=1,
	          validation_data=(x_test, y_test),
	          callbacks=[history,abc])

	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	score1 = model.evaluate(x_train, y_train, verbose=1)
	print('train loss:', score1[0])
	print('train accuracy:', score1[1])
	graph1 = plt.figure(1)
	plt.plot(range(1, epoch_size+1), history.loss)
	plt.xlabel('Epochs')
	plt.ylabel('MSE in sample')
	graph1.show()

	graph2 = plt.figure(2)
	plt.plot(range(1, epoch_size+1), history.loss_val)
	plt.xlabel('Epochs')
	plt.ylabel('MSE out sample')
	graph2.show()

	insample.set(float(history.loss[-1]))
	outsample.set(float(history.loss_val[-1]))

def menuClick(frame):
	frame.tkraise()
	warningmono.set('')
	warning.set('')

def configure(frame):
	frame.configure(width = root.winfo_screenwidth(),height=root.winfo_screenheight())
	print(root.winfo_screenwidth())

def resizeWindow():	
	for frame in (mainFrameRight,trainingFrameRight,aboutFrameRight,howToFrameRight,addDataFrameRight
		,addDataRGBFrameRight,addDataMonoFrameRight):

		frame.bind("<Configure>",configure(frame))

def hoverButton():
	if mouse_pressed:
		for button in (mainMenu,howToUseMenu,aboutMenu,trainingMenu,addDataMenu):
			button.bind("<Enter>", lambda event, h=button: h.configure(bg="#FFC312"))
			button.bind("<Leave>", lambda event, h=button: h.configure(bg=leftSideItemColor))
	else:
		for button in (mainMenu,howToUseMenu,aboutMenu,trainingMenu,addDataMenu):
			button.unbind("<Enter>")
			button.unbind("<Leave>")

def makeImage(width,height,path):
	image = cv2.imread(path)

	if (image.shape[0]>image.shape[1]):
		height = height
		width = int((image).shape[1]/((image).shape[0]/height))
	else:
		width = width
		height = int((image).shape[0]/((image).shape[1]/width))

	dim = (width, height)

	image = cv2.resize(image,dim)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	img = Image.fromarray(image)
	img = ImageTk.PhotoImage(img)
	return img


def chooseImage():
	path = filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
	if path == "":
		pass
	else:
		allPath['imagetopred'] = path
		img = makeImage(245,245,path)
		imageToPrediction.configure(image=img)
		imageToPrediction.image = img

def chooseImagePredMono(filter):
	path = filedialog.askopenfilename(filetypes=[("Image File",'.tif')])
	if path == "":
		pass
	else:
		img = makeImage(100,100,path)
		
		if filter == 1:
			allPath['imagetopredmono350'] = path
			pictPred350Nm.configure(image=img)
			pictPred350Nm.image = img
		elif filter == 2:
			allPath['imagetopredmono400'] = path
			pictPred400Nm.configure(image=img)
			pictPred400Nm.image = img
		elif filter == 3:
			allPath['imagetopredmono450'] = path
			pictPred450Nm.configure(image=img)
			pictPred450Nm.image = img
		elif filter == 4:
			allPath['imagetopredmono500'] = path
			pictPred500Nm.configure(image=img)
			pictPred500Nm.image = img
		elif filter == 5:
			allPath['imagetopredmono550'] = path
			pictPred550Nm.configure(image=img)
			pictPred550Nm.image = img
		elif filter == 6:
			allPath['imagetopredmono600'] = path
			pictPred600Nm.configure(image=img)
			pictPred600Nm.image = img
		elif filter == 7:
			allPath['imagetopredmono650'] = path
			pictPred650Nm.configure(image=img)
			pictPred650Nm.image = img
		elif filter == 8:
			allPath['imagetopredmono700'] = path
			pictPred700Nm.configure(image=img)
			pictPred700Nm.image = img
		elif filter == 9:
			allPath['imagetopredmono750'] = path
			pictPred750Nm.configure(image=img)
			pictPred750Nm.image = img
		elif filter == 10:
			allPath['imagetopredmono800'] = path
			pictPred800Nm.configure(image=img)
			pictPred800Nm.image = img
	

def chooseImageAddRGB():
	pathAddRGB = filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
	if pathAddRGB == "":
		pass
	else:
		print('rgb')
		print(pathAddRGB)
		allPath['imageaddrgb'] = pathAddRGB
		img = makeImage(250,250,pathAddRGB)
		labelGambar.configure(image=img)
		labelGambar.image = img

def chooseImageAddMono(filter):
	path = filedialog.askopenfilename(filetypes=[("Image File",'.tif')])
	if path == "":
		pass
	else:
		img = makeImage(125,125,path)
		
		if filter == 1:
			allPath['imageaddmono350'] = path
			labelGambar350.configure(image=img)
			labelGambar350.image = img
		elif filter == 2:
			allPath['imageaddmono400'] = path
			labelGambar400.configure(image=img)
			labelGambar400.image = img
		elif filter == 3:
			allPath['imageaddmono450'] = path
			labelGambar450.configure(image=img)
			labelGambar450.image = img
		elif filter == 4:
			allPath['imageaddmono500'] = path
			labelGambar500.configure(image=img)
			labelGambar500.image = img
		elif filter == 5:
			allPath['imageaddmono550'] = path
			labelGambar550.configure(image=img)
			labelGambar550.image = img
		elif filter == 6:
			allPath['imageaddmono600'] = path
			labelGambar600.configure(image=img)
			labelGambar600.image = img
		elif filter == 7:
			allPath['imageaddmono650'] = path
			labelGambar650.configure(image=img)
			labelGambar650.image = img
		elif filter == 8:
			allPath['imageaddmono700'] = path
			labelGambar700.configure(image=img)
			labelGambar700.image = img
		elif filter == 9:
			allPath['imageaddmono750'] = path
			labelGambar750.configure(image=img)
			labelGambar750.image = img
		elif filter == 10:
			allPath['imageaddmono800'] = path
			labelGambar800.configure(image=img)
			labelGambar800.image = img

def chooseArc(type):
	pathArc = filedialog.askopenfilename(filetypes=[("File",'.h5')])
	if pathArc == "":
		pass
	else:
		partArc = pathArc.split("/")
		lengthArc = len(partArc)
		nameArc = partArc[lengthArc-1]
		if len(nameArc)>15:
			nameArc = nameArc[:15]+"..."
		if type == "rgb":
			buttonChooseArc.configure(text=nameArc)
			allPath['modeltopred'] = pathArc
		else:
			buttonArcMono.configure(text=nameArc)
			allPath['modeltopredmono'] = pathArc
		

def createRightFrame():
	frame = Frame(container,width=800,height=600,bg=rightSideBackgroundColor, relief="solid")
	frame.grid(row=0,column=0)
	return frame

def createImagePlace(frame,startImg,width,height):
	label = Label(frame,image=startImg,borderwidth=1, relief="solid",width=width,height=height)
	return label

def executePred(types):
	# maxKloroGreen = 1.857813
	# minKloroGreen = -0.11692

	# maxKloroRededge = 0.323876
	# minKloroRededge = -0.16118

	# maxKaroGreen = 1.137226
	# minKaroGreen = 0.157148

	# maxKaroRededge = 2.226155
	# minKaroRededge = 0.511603

	# maxAntho = 1.967888
	# minAntho = 0.065826
	if types == "rgb":
		warning.set('')
		imagepath = allPath['imagetopred']
		modelpath = allPath['modeltopred']

		image1 = []

		model = load_model(modelpath)

		image = cv2.imread(imagepath)
		image = cv2.resize(image,(32,32))
		# print(img)
		# # cv2.imshow("image",image)    
		# # cv2.waitKey(0)
		image1.append(image)

		image12 = np.array(image1)
		image12 = image12.astype('float32')
		image12 /= 255
		try:
			result = model.predict(image12)
		except Exception as e:
			# if str(e) == "Error when checking : expected conv2d_1_input to have shape (120, 120, 3) but got array with shape (32, 32, 3)":
			image1 = []
			image = cv2.imread(imagepath)
			image = cv2.resize(image,(120,120))
			image1.append(image)
			image12 = np.array(image1)
			image12 = image12.astype('float32')
			image12 /= 255
			try:
				result = model.predict(image12)
			except Exception as e:
				warning.set('Your File is wrong, \nmake sure you choose the correct one')
				return
		
		klorogreen = result[0][0]
		klororededge = result[0][1]
		karogreen = result[0][2]
		karorededge = result[0][3]
		antho = result[0][4]

		numKlorofilPredRGB.set(klorogreen)
		numKarotenoidPredRGB.set(karogreen)
		numKlorofilPredRGB2.set(klororededge)
		numKarotenoidPredRGB2.set(karorededge)
		numAntosianinPredRGB.set(antho)
		# print(str(klorogreen)+" "+str(klororededge)+" "+str(karogreen)+" "+str(karorededge)+" "+str(antho))
		print(result)
	else:
		warningmono.set('')
		imagepathmono = [allPath['imagetopredmono350'],allPath['imagetopredmono800'],allPath['imagetopredmono400'],
		allPath['imagetopredmono450'],allPath['imagetopredmono500'],allPath['imagetopredmono550'],allPath['imagetopredmono600']
		,allPath['imagetopredmono650'],allPath['imagetopredmono700'],allPath['imagetopredmono750']]

		modelpathmono = allPath['modeltopredmono']

		image2 = []

		model = load_model(modelpathmono)
		arrayimage10d = np.zeros(shape=(32,32,10))
		dimension = 0
		for img in imagepathmono:
			image = cv2.imread(img)
			image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			image = cv2.resize(image,(32,32))

			arrayimage10d[:,:,dimension] = image
			dimension+=1
			# print(img)
			# cv2.imshow("image",image)
			# cv2.waitKey(0)

		image2.append(arrayimage10d)

		image12 = np.array(image2)
		image12 = image12.astype('float32')
		image12 /= 255
		try:
			resultmono = model.predict(image12)
		except Exception as e:
			if str(e) == "Error when checking : expected conv2d_1_input to have shape (120, 120, 10) but got array with shape (32, 32, 3)":
				image2 = []
				arrayimage10d = np.zeros(shape=(120,120,10))
				dimension = 0
				for img in imagepathmono:
					image = cv2.imread(img)
					image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
					image = cv2.resize(image,(120,120))

					arrayimage10d[:,:,dimension] = image
					dimension+=1

				image2.append(arrayimage10d)

				image12 = np.array(image1)
				image12 = image12.astype('float32')
				image12 /= 255
				resultmono = model.predict(image12)
			else:
				warningmono.set('Your File is wrong, \nmake sure you choose \nthe correct one')

		predMonoKlo.set(resultmono[0][0])
		predMonoKar.set(resultmono[0][2])
		predMonoKlo2.set(resultmono[0][1])
		predMonoKar2.set(resultmono[0][3])
		predMonoAnt.set(resultmono[0][4])


def file_save():
	f = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
	if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
		return
	text2save = str(text.get(1.0, END)) # starts from `1.0`, not `0.0`
	f.write(text2save)
	f.close() # `()` was missing.

def changeAbout():
	if buttonChange.cget("text") == "Developer":
		pictAbout.configure(image=about1)
		buttonChange.configure(text="About Apps")
	else:
		pictAbout.configure(image=about)
		buttonChange.configure(text="Developer")

def changeHowTo(position):
	string = str(position)
	namaFile = "assets/howto"+string+".png"
	pict = PhotoImage(file=namaFile)
	pictHowTo.configure(image=pict)
	pictHowTo.image = pict

def addData(types):
	if types == 'mono':
		if allPath['imageaddmono350'] == '' and allPath['imageaddmono400'] == '' and allPath['imageaddmono450'] == '' and allPath['imageaddmono500'] == '' and allPath['imageaddmono550'] == '' and allPath['imageaddmono600'] == '' and allPath['imageaddmono650'] == '' and allPath['imageaddmono700'] == '' and allPath['imageaddmono750'] == '' and allPath['imageaddmono800'] == '':

			return
		else:
			try:
				klorogreenval = float(trainKlorofilMono.get())
				klororededgeval = float(trainKlorofilMono1.get())
				carogreenval = float(trainKarotenoidMono.get())
				carorededgeval = float(trainKarotenoidMono1.get())
				antoval = float(trainAntosianinMono.get())
			except Exception as e:
				print('error')
				return
			
			nomorData = 1
			nomorImage = 1
			imagepathaddmono = [allPath['imageaddmono350'],allPath['imageaddmono400'],allPath['imageaddmono450'],allPath['imageaddmono500']
			,allPath['imageaddmono550'],allPath['imageaddmono600'],allPath['imageaddmono650'],allPath['imageaddmono700']
			,allPath['imageaddmono750'],allPath['imageaddmono800']]
			
			listFold = os.listdir('LABORATORIUM')
			newFoldName = 'tambahanMono-'

			while os.path.exists('LABORATORIUM/tambahanMono-'+str(nomorData)):
				nomorData+=1

			os.makedirs('LABORATORIUM/'+newFoldName+str(nomorData))
			print(nomorData)

			for imagepath in imagepathaddmono:
				image = cv2.imread(imagepath)
				width = image.shape[0]
				if width>450:
					crop_img = image[300:700, 450:900]
					cv2.imwrite("LABORATORIUM/"+newFoldName+str(nomorData)+"/"+newFoldName+str(nomorData)+'-'+str(nomorImage)+'.tif',crop_img)
				else:
					cv2.imwrite("LABORATORIUM/"+newFoldName+str(nomorData)+"/"+newFoldName+str(nomorData)+'-'+str(nomorImage)+'.tif',image)
				nomorImage+=1

			os.rename('DATA INDEX/data-notnorm.csv', 'DATA INDEX/temporary.csv')
			first = open('DATA INDEX/temporary.csv')
			new = open('DATA INDEX/data-notnorm.csv', 'w', newline='')
			fieldnames = ['Nama','KloroGreen','KloroRedEdge','CaroGreen','CaroRedEdge','Antho']
			writer = csv.DictWriter(new, fieldnames=fieldnames)

			writer.writeheader()

			firstline = True
			for rows in csv.reader(first):
				if firstline:    #skip first line
					firstline = False
					continue

				if rows[0] == newFoldName+str(nomorData):
					pass
				else:
					writer.writerow({'Nama': rows[0], 'KloroGreen': rows[1], 'KloroRedEdge': rows[2], 
						'CaroGreen': rows[3], 'CaroRedEdge': rows[4], 'Antho': rows[5]})

			writer.writerow({'Nama': newFoldName+str(nomorData), 'KloroGreen': klorogreenval, 'KloroRedEdge': klororededgeval, 
					'CaroGreen': carogreenval, 'CaroRedEdge': carorededgeval, 'Antho': antoval})

			first.close()
			new.close()
			os.remove('DATA INDEX/temporary.csv')

	else:
		imagepathaddrgb = allPath['imageaddrgb']
		if imagepathaddrgb == '':
			return
		else:
			try:
				klorogreenval = float(numKlorofilRGB.get())
				klororededgeval = float(numKlorofilRGB1.get())
				carogreenval = float(numKarotenoidRGB.get())
				carorededgeval = float(numKarotenoidRGB1.get())
				antoval = float(numAntosianinRGB.get())
			except Exception as er:
				print('error')
				return
			

			nomorData = 1
			
			listImage = os.listdir('HANDPHONE')
			newName = 'tambahanRGB-'

			while os.path.exists('HANDPHONE/'+newName+str(nomorData)+'.jpg'):
				nomorData+=1

			image = cv2.imread(imagepathaddrgb)
			width = image.shape[0]
			cv2.imwrite("HANDPHONE/"+newName+str(nomorData)+'.jpg',image)

			os.rename('DATA INDEX/data-notnorm.csv', 'DATA INDEX/temporary.csv')
			first = open('DATA INDEX/temporary.csv')
			new = open('DATA INDEX/data-notnorm.csv', 'w', newline='')
			fieldnames = ['Nama','KloroGreen','KloroRedEdge','CaroGreen','CaroRedEdge','Antho']
			writer = csv.DictWriter(new, fieldnames=fieldnames)

			writer.writeheader()

			firstline = True
			for rows in csv.reader(first):
				if firstline:    #skip first line
					firstline = False
					continue

				if rows[0] == newName+str(nomorData):
					pass
				else:
					writer.writerow({'Nama': rows[0], 'KloroGreen': rows[1], 'KloroRedEdge': rows[2], 
						'CaroGreen': rows[3], 'CaroRedEdge': rows[4], 'Antho': rows[5]})

			writer.writerow({'Nama': newName+str(nomorData), 'KloroGreen': klorogreenval, 'KloroRedEdge': klororededgeval, 
					'CaroGreen': carogreenval, 'CaroRedEdge': carorededgeval, 'Antho': antoval})

			first.close()
			new.close()
			os.remove('DATA INDEX/temporary.csv')




root = Tk()
# root.resizable(False,False)
root.minsize(1055, 600)
root.geometry("1055x600+200+80")
root.title("Leaf Pigment Prediction")
img = PhotoImage(file='assets/icon.png')
root.tk.call('wm', 'iconphoto', root._w, img)
mouse_pressed = True

container = Frame(root)
allPath = {}
allPath['imagetopred'] = ""
allPath['modeltopred'] = ""
allPath['imagetopredmono350'] =""
allPath['imagetopredmono400'] =""
allPath['imagetopredmono450'] =""
allPath['imagetopredmono500'] =""
allPath['imagetopredmono550'] =""
allPath['imagetopredmono600'] =""
allPath['imagetopredmono650'] =""
allPath['imagetopredmono700'] =""
allPath['imagetopredmono750'] =""
allPath['imagetopredmono800'] =""
allPath['modeltopredmono'] = ""

allPath['imageaddrgb'] = ""
allPath['imageaddmono350'] = ""
allPath['imageaddmono400'] = ""
allPath['imageaddmono450'] = ""
allPath['imageaddmono500'] = ""
allPath['imageaddmono550'] = ""
allPath['imageaddmono600'] = ""
allPath['imageaddmono650'] = ""
allPath['imageaddmono700'] = ""
allPath['imageaddmono750'] = ""
allPath['imageaddmono800'] = ""

#ALL COLOR
leftSideBackgroundColor = "#1E2D44"
rightSideBackgroundColor = "#ECF0F1"
leftSideItemColor = "#9A9CA3"

#FRAME DI KIRI
mainFrameLeft = Frame(root,width=200,height=600,bg=leftSideBackgroundColor,borderwidth=1, relief="solid")

#FRAME DI KANAN

trainingFrameRight = createRightFrame()
aboutFrameRight = createRightFrame()
howToFrameRight = createRightFrame()
addDataFrameRight = createRightFrame()
addDataRGBFrameRight = createRightFrame()
addDataMonoFrameRight = createRightFrame()
mainFrameMonoRight = createRightFrame()
mainFrameRight = createRightFrame()
#=========================================================================================

#ICON MENU FRAME KIRI
imgMainMenu =PhotoImage(file='assets/leaf.png')
imgQuestion =PhotoImage(file='assets/question.png')
imgTraining = PhotoImage(file='assets/mind.png')
imgInfo =PhotoImage(file='assets/info.png')
imgAdd =PhotoImage(file='assets/plus.png')
logo = PhotoImage(file='assets/icon.png')
logo = logo.subsample(3,3)

#OBJECT DI FRAME KIRI + CONFIGURASINYA
appName = Label(mainFrameLeft,text="LEAF PIGMENT",fg="#ECF0F1",font="Calibre 24 bold",bg=leftSideBackgroundColor)
appName1 = Label(mainFrameLeft,text="PREDICTION",fg="#ECF0F1",font="Calibre 24 bold",bg=leftSideBackgroundColor)

mainMenu = Button(mainFrameLeft,text=" Main Menu",font=("Roboto Light",14,"bold"),fg="#012345",bg=leftSideItemColor,borderwidth=0,command=lambda: menuClick(mainFrameRight))
trainingMenu = Button(mainFrameLeft,text=" Training Menu",font=("Roboto Light",14,"bold"),fg="#012345",bg=leftSideItemColor,borderwidth=0,command=lambda: menuClick(trainingFrameRight)) 
aboutMenu = Button(mainFrameLeft,text=" About",font=("Roboto Light",14,"bold"),fg="#012345",bg=leftSideItemColor,borderwidth=0,command=lambda: menuClick(aboutFrameRight))
howToUseMenu = Button(mainFrameLeft,text=" How To Use",font=("Roboto Light",14,"bold"),fg="#012345",bg=leftSideItemColor,borderwidth=0,command=lambda: menuClick(howToFrameRight))
addDataMenu = Button(mainFrameLeft,text=" Add Data",font=("Roboto Light",14,"bold"),fg="#012345",bg=leftSideItemColor,borderwidth=0,command=lambda: menuClick(addDataFrameRight))
logoLa = Label(mainFrameLeft,image=logo,bg=leftSideBackgroundColor)

mainMenu.configure(image = imgMainMenu,compound=LEFT,width=220,anchor="w")
trainingMenu.configure(image = imgTraining,compound=LEFT,width=220,anchor="w")
aboutMenu.configure(image = imgInfo,compound=LEFT,width=220,anchor="w")
howToUseMenu.configure(image = imgQuestion,compound=LEFT,width=220,anchor="w")
addDataMenu.configure(image = imgAdd,compound=LEFT,width=220,anchor="w")

logoLa.grid(row=0,column=0,pady=(20,0))
appName.grid(row=1,column=0,ipadx=0)
appName1.grid(row=2,column=0,pady=(0,20),ipadx=0)
mainMenu.grid(row=3,column=0,ipady=10,ipadx=15)
trainingMenu.grid(row=4,column=0,ipady=10,ipadx=15)
addDataMenu.grid(row=5,column=0,ipady=10,ipadx=15)
howToUseMenu.grid(row=6,column=0,ipady=10,ipadx=15)
aboutMenu.grid(row=7,column=0,ipady=10,ipadx=15)


#============================================================================
#OBJECT + IMAGE UNTUK FRAME MAIN MENU
image = cv2.imread('assets/no-image.png')

image = cv2.resize(image,(245,245))

img = Image.fromarray(image)
img = ImageTk.PhotoImage(img)


frameForResult = Frame(mainFrameRight,borderwidth=2,relief="ridge",width=700,height=450,bg="#353B47")
imageToPrediction = Label(mainFrameRight,image=img,borderwidth=1, relief="solid",width=245,height=245)

buttonChooseImage = Button(mainFrameRight,text="Select Image",font="Calibre 15 bold",bg="#006266",fg="white",borderwidth=1,width=15,command=chooseImage)

# labelKarotenoid = Label(mainFrameRight,text="Carotenoid Index :",font="Calibre 15 bold",width=18,height=2,bg="#353B47",fg=rightSideBackgroundColor,anchor="e")
# labelKlorofil = Label(mainFrameRight,text=  "Chlorophyll Index :",font="Calibre 15 bold",width=18,bg="#353B47",fg=rightSideBackgroundColor,anchor="e")
# labelAntosianin = Label(mainFrameRight,text="Anthocyanin Index :",font="Calibre 15 bold",width=18,bg="#353B47",fg=rightSideBackgroundColor,anchor="e")

labelKarotenoid = Label(mainFrameRight,text="Carotenoid",font="Calibre 13 bold",width=18,height=2,bg="#353B47",fg=rightSideBackgroundColor,anchor="e")
labelKlorofil = Label(mainFrameRight,text=  "Chlorophyll",font="Calibre 13 bold",width=18,bg="#353B47",fg=rightSideBackgroundColor,anchor="e")
labelAntosianin = Label(mainFrameRight,text="Anthocyanin index : ",font="Calibre 13 bold",bg="#353B47",fg=rightSideBackgroundColor,anchor="e")

buttonChooseArc = Button(mainFrameRight,text="No File",command=lambda: chooseArc("rgb"),font="Calibre 13 bold",bg="#006266",fg="white",borderwidth=1)
labelArc = Label(mainFrameRight,text="no choosen architecture",font="Calibre 13 bold")
buttonExecute = Button(mainFrameRight,text="Execute",font="Calibre 20 bold",bg="#006266",fg="white",borderwidth=1,width=10,height=1,command=lambda: executePred("rgb"))
buttonChangeInput = Button(mainFrameRight)

numKlorofilPredRGB = DoubleVar()
numKlorofilPredRGB.set(0)
numKlorofilPredRGB2 = DoubleVar()
numKlorofilPredRGB2.set(0)
numKarotenoidPredRGB = DoubleVar()
numKarotenoidPredRGB.set(0)
numKarotenoidPredRGB2 = DoubleVar()
numKarotenoidPredRGB2.set(0)
numAntosianinPredRGB = DoubleVar()
numAntosianinPredRGB.set(0)
warning = StringVar()
warning.set("")

# Entry(mainFrameRight,font="Calibre 13 bold",justify='left',width=5,state='disabled',textvariable=numKlorofilPredRGB).place(relx=.72,rely=.35,anchor="center")
# Entry(mainFrameRight,font="Calibre 13 bold",justify='left',width=5,state='disabled',textvariable=numKarotenoidPredRGB).place(relx=.72,rely=.4,anchor="center")
# Entry(mainFrameRight,font="Calibre 13 bold",justify='left',width=5,state='disabled',textvariable=numAntosianinPredRGB).place(relx=.72,rely=.45,anchor="center")
Entry(mainFrameRight,font="Calibre 13 bold",justify='left',width=5,state='disabled',textvariable=numKlorofilPredRGB).place(relx=.6,rely=.35,anchor="center")
Entry(mainFrameRight,font="Calibre 13 bold",justify='left',width=5,state='disabled',textvariable=numKlorofilPredRGB2).place(relx=.86,rely=.35,anchor="center")
Entry(mainFrameRight,font="Calibre 13 bold",justify='left',width=5,state='disabled',textvariable=numKarotenoidPredRGB).place(relx=.6,rely=.45,anchor="center")
Entry(mainFrameRight,font="Calibre 13 bold",justify='left',width=5,state='disabled',textvariable=numKarotenoidPredRGB2).place(relx=.86,rely=.45,anchor="center")
Entry(mainFrameRight,font="Calibre 13 bold",justify='left',width=5,state='disabled',textvariable=numAntosianinPredRGB).place(relx=.75,rely=.5,anchor="center")

#KONFIGURASI FRAME MAIN MENU
frameForResult.place(relx=.5,rely=.5,anchor="center")
imageToPrediction.place(relx=.25, rely=.5, anchor="center")
buttonChooseImage.place(relx=.25, rely=.75, anchor="center")
# labelKarotenoid.place(relx=.55, rely=.4, anchor="center")
# labelKlorofil.place(relx=.55, rely=.35, anchor="center")
# labelAntosianin.place(relx=.55, rely=.45, anchor="center")
labelKarotenoid.place(relx=.6, rely=.4, anchor="center")
labelKlorofil.place(relx=.6, rely=.3, anchor="center")
labelAntosianin.place(relx=.5, rely=.5, anchor="w")
buttonChooseArc.place(relx=.67,rely=.57,anchor="w")
# labelArc.place(relx=.65,rely=.6,anchor="center")
buttonExecute.place(relx=.65,rely=.75,anchor="center")
Label(mainFrameRight,text="green index : ",font="Calibre 11 bold",height=2,bg="#353B47",fg=rightSideBackgroundColor,anchor="e").place(relx=.5,rely=.35,anchor='center')
Label(mainFrameRight,text="rededge index : ",font="Calibre 11 bold",height=2,bg="#353B47",fg=rightSideBackgroundColor,anchor="e").place(relx=.75,rely=.35,anchor='center')
Label(mainFrameRight,text="green index : ",font="Calibre 11 bold",height=2,bg="#353B47",fg=rightSideBackgroundColor,anchor="e").place(relx=.5,rely=.45,anchor='center')
Label(mainFrameRight,text="rededge index : ",font="Calibre 11 bold",height=2,bg="#353B47",fg=rightSideBackgroundColor,anchor="e").place(relx=.75,rely=.45,anchor='center')

Label(mainFrameRight,textvariable=warning,font="Calibre 12 bold",height=2,bg="#353B47",fg="red",anchor="e").place(relx=.65, rely=.65, anchor="center")
Label(mainFrameRight,text=" RGB Image",font="Calibre 16 bold",bg=rightSideBackgroundColor,fg="#012345",width=50,anchor="w").place(relx=.06,rely=.18,anchor="w")
Label(mainFrameRight,text="Select Architecture :",font="Calibre 13 bold",bg="#353B47",fg=rightSideBackgroundColor).place(relx=.55,rely=.57,anchor="center")
Button(mainFrameRight,text="Switch to Monochrome Mode",font="Calibre 14 bold",bg="#006266",fg="white",borderwidth=1,command=lambda: menuClick(mainFrameMonoRight)).place(relx=.06,rely=.06,anchor="w")
#=========================================================================================
#FRAME MAIN MENU MONOCHROME
imgMonoPred = cv2.imread('assets/no-image.png')
imgMonoPred = cv2.resize(imgMonoPred,(100,100))
imgMonoPred = Image.fromarray(imgMonoPred)
imgMonoPred = ImageTk.PhotoImage(imgMonoPred)

Button(mainFrameMonoRight,text="Switch to RGB Mode",font="Calibre 14 bold",bg="#006266",fg="white",borderwidth=1,command=lambda: menuClick(mainFrameRight)).place(relx=.06,rely=.06,anchor="w")
Frame(mainFrameMonoRight,borderwidth=2,relief="ridge",width=700,height=500,bg="#353B47").place(relx=.06,rely=.14,anchor="nw")
Label(mainFrameMonoRight,text=" Monochrome Image",font="Calibre 16 bold",bg=rightSideBackgroundColor,fg="#012345",width=50,anchor="w").place(relx=.06,rely=.18,anchor="w")

pictPred350Nm = Label(mainFrameMonoRight,image=imgMonoPred,borderwidth=1, relief="solid",width=100,height=100)
pictPred400Nm = Label(mainFrameMonoRight,image=imgMonoPred,borderwidth=1, relief="solid",width=100,height=100)
pictPred450Nm = Label(mainFrameMonoRight,image=imgMonoPred,borderwidth=1, relief="solid",width=100,height=100)
pictPred500Nm = Label(mainFrameMonoRight,image=imgMonoPred,borderwidth=1, relief="solid",width=100,height=100)
pictPred550Nm = Label(mainFrameMonoRight,image=imgMonoPred,borderwidth=1, relief="solid",width=100,height=100)
buttonPred350Nm = Button(mainFrameMonoRight,text="Select Image",font="Calibre 11 bold",bg="#006266",fg="white",borderwidth=1,width=11,command=lambda: chooseImagePredMono(1))
buttonPred400Nm = Button(mainFrameMonoRight,text="Select Image",font="Calibre 11 bold",bg="#006266",fg="white",borderwidth=1,width=11,command=lambda: chooseImagePredMono(2))
buttonPred450Nm = Button(mainFrameMonoRight,text="Select Image",font="Calibre 11 bold",bg="#006266",fg="white",borderwidth=1,width=11,command=lambda: chooseImagePredMono(3))
buttonPred500Nm = Button(mainFrameMonoRight,text="Select Image",font="Calibre 11 bold",bg="#006266",fg="white",borderwidth=1,width=11,command=lambda: chooseImagePredMono(4))
buttonPred550Nm = Button(mainFrameMonoRight,text="Select Image",font="Calibre 11 bold",bg="#006266",fg="white",borderwidth=1,width=11,command=lambda: chooseImagePredMono(5))

posX = .2
for pic in (pictPred350Nm,pictPred400Nm,pictPred450Nm,pictPred500Nm,pictPred550Nm):
	pic.place(relx = posX, rely = .35, anchor = "center")
	posX = posX+.15

posX = .2
for pic in (buttonPred350Nm,buttonPred400Nm,buttonPred450Nm,buttonPred500Nm,buttonPred550Nm):
	pic.place(relx = posX, rely = .45, anchor = "center")
	posX = posX+.15

pictPred600Nm = Label(mainFrameMonoRight,image=imgMonoPred,borderwidth=1, relief="solid",width=100,height=100)
pictPred650Nm = Label(mainFrameMonoRight,image=imgMonoPred,borderwidth=1, relief="solid",width=100,height=100)
pictPred700Nm = Label(mainFrameMonoRight,image=imgMonoPred,borderwidth=1, relief="solid",width=100,height=100)
pictPred750Nm = Label(mainFrameMonoRight,image=imgMonoPred,borderwidth=1, relief="solid",width=100,height=100)
pictPred800Nm = Label(mainFrameMonoRight,image=imgMonoPred,borderwidth=1, relief="solid",width=100,height=100)
buttonPred600Nm = Button(mainFrameMonoRight,text="Select Image",font="Calibre 11 bold",bg="#006266",fg="white",borderwidth=1,width=11,command=lambda: chooseImagePredMono(6))
buttonPred650Nm = Button(mainFrameMonoRight,text="Select Image",font="Calibre 11 bold",bg="#006266",fg="white",borderwidth=1,width=11,command=lambda: chooseImagePredMono(7))
buttonPred700Nm = Button(mainFrameMonoRight,text="Select Image",font="Calibre 11 bold",bg="#006266",fg="white",borderwidth=1,width=11,command=lambda: chooseImagePredMono(8))
buttonPred750Nm = Button(mainFrameMonoRight,text="Select Image",font="Calibre 11 bold",bg="#006266",fg="white",borderwidth=1,width=11,command=lambda: chooseImagePredMono(9))
buttonPred800Nm = Button(mainFrameMonoRight,text="Select Image",font="Calibre 11 bold",bg="#006266",fg="white",borderwidth=1,width=11,command=lambda: chooseImagePredMono(10))

Label(mainFrameMonoRight,text="350 nm",bg="#353b47",fg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.2,rely=.24,anchor="center")
Label(mainFrameMonoRight,text="400 nm",bg="#353b47",fg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.35,rely=.24,anchor="center")
Label(mainFrameMonoRight,text="450 nm",bg="#353b47",fg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.5,rely=.24,anchor="center")
Label(mainFrameMonoRight,text="500 nm",bg="#353b47",fg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.65,rely=.24,anchor="center")
Label(mainFrameMonoRight,text="550 nm",bg="#353b47",fg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.8,rely=.24,anchor="center")
Label(mainFrameMonoRight,text="600 nm",bg="#353b47",fg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.2,rely=.49,anchor="center")
Label(mainFrameMonoRight,text="650 nm",bg="#353b47",fg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.35,rely=.49,anchor="center")
Label(mainFrameMonoRight,text="700 nm",bg="#353b47",fg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.5,rely=.49,anchor="center")
Label(mainFrameMonoRight,text="750 nm",bg="#353b47",fg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.65,rely=.49,anchor="center")
Label(mainFrameMonoRight,text="800 nm",bg="#353b47",fg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.8,rely=.49,anchor="center")

posX = .2
for pic in (pictPred600Nm,pictPred650Nm,pictPred700Nm,pictPred750Nm,pictPred800Nm):
	pic.place(relx = posX, rely = .6, anchor = "center")
	posX = posX+.15

posX = .2
for pic in (buttonPred600Nm,buttonPred650Nm,buttonPred700Nm,buttonPred750Nm,buttonPred800Nm):
	pic.place(relx = posX, rely = .7, anchor = "center")
	posX = posX+.15

Label(mainFrameMonoRight,text="Chlorophyll",font="Calibre 11 bold",bg="#353B47",fg=rightSideBackgroundColor,anchor="w",width=10).place(relx=.2, rely=.77, anchor="center")
Label(mainFrameMonoRight,text="Carotenoid",font="Calibre 11 bold",bg="#353B47",fg=rightSideBackgroundColor,anchor="w",width=10).place(relx=.2, rely=.85, anchor="center")
Label(mainFrameMonoRight,text="Anthocyanin : ",font="Calibre 11 bold",bg="#353B47",fg=rightSideBackgroundColor,anchor="w",width=10).place(relx=.2, rely=.93, anchor="center")
Label(mainFrameMonoRight,text="green : ",font="Calibre 11 bold",bg="#353B47",fg=rightSideBackgroundColor,anchor="w",width=10).place(relx=.2, rely=.81, anchor="center")
Label(mainFrameMonoRight,text="rededge : ",font="Calibre 11 bold",bg="#353B47",fg=rightSideBackgroundColor,anchor="w",width=10).place(relx=.4, rely=.81, anchor="center")
Label(mainFrameMonoRight,text="green : ",font="Calibre 11 bold",bg="#353B47",fg=rightSideBackgroundColor,anchor="w",width=10).place(relx=.2, rely=.89, anchor="center")
Label(mainFrameMonoRight,text="rededge : ",font="Calibre 11 bold",bg="#353B47",fg=rightSideBackgroundColor,anchor="w",width=10).place(relx=.4, rely=.89, anchor="center")

# Label(mainFrameMonoRight,text="%",font="Calibre 11 bold",bg="#353B47",fg=rightSideBackgroundColor,anchor="e").place(relx=.32, rely=.75, anchor="center")
# Label(mainFrameMonoRight,text="%",font="Calibre 11 bold",bg="#353B47",fg=rightSideBackgroundColor,anchor="e").place(relx=.32, rely=.8, anchor="center")
# Label(mainFrameMonoRight,text="%",font="Calibre 11 bold",bg="#353B47",fg=rightSideBackgroundColor,anchor="e").place(relx=.32, rely=.85, anchor="center")

predMonoKlo = DoubleVar()
predMonoKlo.set(0)
predMonoKar = DoubleVar()
predMonoKar.set(0)
predMonoKlo2 = DoubleVar()
predMonoKlo2.set(0)
predMonoKar2 = DoubleVar()
predMonoKar2.set(0)
predMonoAnt = DoubleVar()
predMonoAnt.set(0)
warningmono = StringVar()
warningmono.set('')
Label(mainFrameMonoRight,textvariable=warningmono,font="Calibre 11 bold",height=3,bg="#353B47",fg="red",anchor="e").place(relx=.6, rely=.9, anchor="center")

# Entry(mainFrameMonoRight,font="Calibre 11 bold",justify='right',width=3,state='disabled',textvariable=predMonoKlo).place(relx=.29,rely=.75,anchor="center")
# Entry(mainFrameMonoRight,font="Calibre 11 bold",justify='right',width=3,state='disabled',textvariable=predMonoKar).place(relx=.29,rely=.8,anchor="center")
# Entry(mainFrameMonoRight,font="Calibre 11 bold",justify='right',width=3,state='disabled',textvariable=predMonoAnt).place(relx=.29,rely=.85,anchor="center")
Entry(mainFrameMonoRight,font="Calibre 11 bold",justify='right',width=5,state='disabled',textvariable=predMonoKlo).place(relx=.25,rely=.81,anchor="center")
Entry(mainFrameMonoRight,font="Calibre 11 bold",justify='right',width=5,state='disabled',textvariable=predMonoKlo2).place(relx=.48,rely=.81,anchor="center")
Entry(mainFrameMonoRight,font="Calibre 11 bold",justify='right',width=5,state='disabled',textvariable=predMonoKar).place(relx=.25,rely=.89,anchor="center")
Entry(mainFrameMonoRight,font="Calibre 11 bold",justify='right',width=5,state='disabled',textvariable=predMonoKar2).place(relx=.48,rely=.89,anchor="center")
Entry(mainFrameMonoRight,font="Calibre 11 bold",justify='right',width=5,state='disabled',textvariable=predMonoAnt).place(relx=.35,rely=.93,anchor="center")

Label(mainFrameMonoRight,text="Architecture :",font="Calibre 11 bold",bg="#353B47",fg=rightSideBackgroundColor,anchor="w",width=10).place(relx=.6, rely=.77, anchor="center")
buttonArcMono = Button(mainFrameMonoRight,text="No File",font="Calibre 11 bold",bg="#006266",fg="white",borderwidth=1,command=lambda: chooseArc("mono"))
buttonArcMono.place(relx=.68, rely=.77, anchor="w")

Button(mainFrameMonoRight,text="Execute",font="Calibre 13 bold",bg="#006266",fg="white",borderwidth=1,width=15,command=lambda: executePred("mono")).place(relx=.65, rely=.83, anchor="center")


#=========================================================================================
#FRAME TRAINING

Frame(trainingFrameRight,borderwidth=2,relief="ridge",width=700,height=400,bg="#353B47").place(relx=.5,rely=.38,anchor="center")
Label(trainingFrameRight,text=" Training Menu",font="Calibre 15 bold",bg=rightSideBackgroundColor,fg="#012345",width=50,anchor="w").place(relx=.06,rely=.1,anchor="w")
Label(trainingFrameRight,text=" Image Type : ",font="Calibre 15 bold",width=12,anchor="w",bg="#353B47",fg=rightSideBackgroundColor).place(relx=.2,rely=.16,anchor="center")
Label(trainingFrameRight,text=" Batch Size : ",font="Calibre 15 bold",width=12,anchor="w",bg="#353B47",fg=rightSideBackgroundColor).place(relx=.2,rely=.23,anchor="center")
Label(trainingFrameRight,text=" Epoch : ",font="Calibre 15 bold",width=12,anchor="w",bg="#353B47",fg=rightSideBackgroundColor).place(relx=.2,rely=.3,anchor="center")
Label(trainingFrameRight,text=" Optimizer : ",font="Calibre 15 bold",width=12,anchor="w",bg="#353B47",fg=rightSideBackgroundColor).place(relx=.2,rely=.37,anchor="center")
Label(trainingFrameRight,text=" Architecture : ",font="Calibre 15 bold",width=12,anchor="w",bg="#353B47",fg=rightSideBackgroundColor).place(relx=.2,rely=.55,anchor="center")

MODES = [
    ("RGB Images", "rgb"),
    ("Monochrome Images", "mnc")
]

ARSITEKTUR = [
	("ShallowNet","shallow"),
	("LeNet","lenet"),
	("AlexNet","alexnet")
	
]

OPTIMIZER = [
	("Adam","adam"),
	("RMSprop","rmsprop"),
	("Adamax","adamax"),
	("Adagrad","adagrad"),
	("SGD","sgd"),
	("Adadelta","adadelta"),
	("Nadam","nadam"),
]

mode1 = StringVar()
mode1.set("rgb") # initialize
place = .32

for text, mode in MODES:
	b = Radiobutton(trainingFrameRight, text=text,
		variable=mode1, value=mode ,font="Calibre 14 bold",bg="#353B47",fg=rightSideBackgroundColor,selectcolor="#353B47")
	b.place(relx=place,rely=.16,anchor="w")
	place = place+.2

batch = IntVar()
batch.set(0)
Entry(trainingFrameRight,font="Calibre 13 bold",justify='left',width=5,textvariable=batch).place(relx=.36,rely=.23,anchor="center")

epoch = IntVar()
epoch.set(0)
Entry(trainingFrameRight,font="Calibre 13 bold",justify='left',width=5,textvariable=epoch).place(relx=.36,rely=.3,anchor="center")

opt = StringVar()
opt.set("adam")
place = .32
posOpt = 0
for text, mode in OPTIMIZER:
	c = Radiobutton(trainingFrameRight, text=text,
		variable=opt, value=mode ,font="Calibre 14 bold",bg="#353B47",fg=rightSideBackgroundColor,selectcolor="#353B47")
	if posOpt>3:
		if relly == .37:
			place = .32
		else:
			pass

		relly = .45
	else:
		relly = .37
	c.place(relx=place,rely=relly,anchor="w")
	place = place+.15
	posOpt+=1

ars = StringVar()
ars.set("shallow")
place = .32
positionars = 0
for text, mode in ARSITEKTUR:
	c = Radiobutton(trainingFrameRight, text=text,
		variable=ars, value=mode ,font="Calibre 14 bold",bg="#353B47",fg=rightSideBackgroundColor,selectcolor="#353B47")
	c.place(relx=place,rely=.55,anchor="w")
	
	if positionars<1:
		place = place+.18
	else:
		place = place+.15
	positionars+=1

buttonTrainNew = Button(trainingFrameRight,text="Train New Model",font="Calibre 16 bold",bg="#006266",fg="white"
	,borderwidth=1,width=20,height=3,command=trainCNN) 
buttonTrainNew.place(relx=.5,rely=.7,anchor="center")

# progressBar = ttk.Progressbar(trainingFrameRight,orient=HORIZONTAL,length=300,mode='determinate')
# progressBar.place(relx=.5,rely=.65,anchor="center")

# labelProses = Label(trainingFrameRight,text=" No Process ",font="Calibre 12 bold",width=12)
# labelProses.place(relx=.5,rely=.7,anchor="center")

# frameProsesTrain = Frame(trainingFrameRight)

# ScrollBarTrain = Scrollbar(frameProsesTrain,borderwidth=2,relief="ridge")
# textBoxTrain = Text(frameProsesTrain,height = 8,width = 85,borderwidth=2)
# extext="=====================================No Process======================================"
# ScrollBarTrain.config(command=textBoxTrain.yview)
# textBoxTrain.config(yscrollcommand=ScrollBarTrain.set)
# textBoxTrain.configure(state=NORMAL)


# textBoxTrain.insert(END, extext)
# # textBoxTrain.configure(state=DISABLED)

# textBoxTrain.pack(side="left", fill="both", expand=True)
# ScrollBarTrain.pack(side="right", fill="y", expand=False)
# frameProsesTrain.place(relx=.5,rely=.86,anchor="center")
Label(trainingFrameRight,text=" MSE In Sample : ",font="Calibre 15 bold",anchor="w",fg='black').place(relx=.2,rely=.9,anchor="center")

Label(trainingFrameRight,text=" MSE Out Sample : ",font="Calibre 15 bold",anchor="w",fg='black').place(relx=.5,rely=.9,anchor="center")

insample = DoubleVar()
insample.set(0)
Entry(trainingFrameRight,font="Calibre 13 bold",justify='left',width=5,textvariable=insample,state='disabled').place(relx=.34,rely=.9,anchor="center")

outsample = DoubleVar()
outsample.set(0)
Entry(trainingFrameRight,font="Calibre 13 bold",justify='left',width=5,textvariable=outsample,state='disabled').place(relx=.64,rely=.9,anchor="center")
#=========================================================================================
#FRAME ADD DATA AWAL

buttonToRGB = Button(addDataFrameRight,text="RGB IMAGE",font="Calibre 16 bold",bg="#006266",fg="white"
	,borderwidth=1,width=20,height=5, command=lambda: menuClick(addDataRGBFrameRight))

buttonToMono = Button(addDataFrameRight,text="MONOCHROME IMAGE",font="Calibre 16 bold",bg="#006266",fg="white"
	,borderwidth=1,width=20,height=5, command=lambda: menuClick(addDataMonoFrameRight))

buttonToRGB.place(relx=.25,rely=.5,anchor="center")
buttonToMono.place(relx=.75,rely=.5,anchor="center")

#=========================================================================================
#FRAME ADD DATA RGB
Frame(addDataRGBFrameRight,borderwidth=2,relief="ridge",width=700,height=375,bg="#353B47").place(relx=.5,rely=.5,anchor="center")
backImage = PhotoImage(file="assets/reply.png")

buttonToBack = Button(addDataRGBFrameRight,image=backImage,font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=70,height=50,command=lambda: menuClick(addDataFrameRight))

imageNoRgb = cv2.imread('assets/no-image.png')
imageNoRgb = cv2.resize(imageNoRgb,(250,250))
imgNoRgb = Image.fromarray(imageNoRgb)
imgNoRgb = ImageTk.PhotoImage(imgNoRgb)

labelGambar = Label(addDataRGBFrameRight,image=imgNoRgb,borderwidth=1, relief="solid",width=250,height=250)
buttonChooseImageAdd = Button(addDataRGBFrameRight,text="Select Image",font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=20,command=chooseImageAddRGB)

labelKlorofilRGB = Label(addDataRGBFrameRight,text="Chlorophyll :",font="Calibre 13 bold",width=12)
labelKarotenoidRGB = Label(addDataRGBFrameRight,text="Carotenoid :",font="Calibre 13 bold",width=12)
labelAntosianinRGB = Label(addDataRGBFrameRight,text="Anthocyanin :",font="Calibre 13 bold",width=12)
# labelPercent1 = Label(addDataRGBFrameRight,text="%",font="Calibre 13 bold")
# labelPercent2 = Label(addDataRGBFrameRight,text="%",font="Calibre 13 bold")
# labelPercent3 = Label(addDataRGBFrameRight,text="%",font="Calibre 13 bold")

numKlorofilRGB = DoubleVar()
numKlorofilRGB.set(0)
numKarotenoidRGB = DoubleVar()
numKarotenoidRGB.set(0)
numKlorofilRGB1 = DoubleVar()
numKlorofilRGB1.set(0)
numKarotenoidRGB1 = DoubleVar()
numKarotenoidRGB1.set(0)
numAntosianinRGB = DoubleVar()
numAntosianinRGB.set(0)

buttonAddData = Button(addDataRGBFrameRight,text="ADD DATA TO RGB DATASET",font="Calibre 16 bold",bg="#006266",fg="white"
	,borderwidth=1,height=2,width=30,command=lambda: addData('rgb'))

labelKlorofilRGB.place(relx=.45,rely=.3,anchor="nw")
Label(addDataRGBFrameRight,text="green index :",font="Calibre 12 bold").place(relx=.45,rely=.38,anchor="nw")
Label(addDataRGBFrameRight,text="rededge index :",font="Calibre 12 bold").place(relx=.67,rely=.38,anchor="nw")

Entry(addDataRGBFrameRight,font="Calibre 13 bold",justify='right',width=5,textvariable=numKlorofilRGB).place(relx=.58,rely=.38,anchor="nw")
Entry(addDataRGBFrameRight,font="Calibre 13 bold",justify='right',width=5,textvariable=numKlorofilRGB1).place(relx=.82,rely=.38,anchor="nw")

# labelPercent1.place(relx=.71,rely=.3,anchor="nw")

labelKarotenoidRGB.place(relx=.45,rely=.45,anchor="nw")
Label(addDataRGBFrameRight,text="green index :",font="Calibre 12 bold").place(relx=.45,rely=.53,anchor="nw")
Label(addDataRGBFrameRight,text="rededge index :",font="Calibre 12 bold").place(relx=.67,rely=.53,anchor="nw")

Entry(addDataRGBFrameRight,font="Calibre 13 bold",justify='right',width=5,textvariable=numKarotenoidRGB).place(relx=.58,rely=.53,anchor="nw")
Entry(addDataRGBFrameRight,font="Calibre 13 bold",justify='right',width=5,textvariable=numKarotenoidRGB1).place(relx=.82,rely=.53,anchor="nw")

# labelPercent2.place(relx=.71,rely=.4,anchor="nw")

labelAntosianinRGB.place(relx=.45,rely=.6,anchor="nw")
Entry(addDataRGBFrameRight,font="Calibre 13 bold",justify='right',width=5,textvariable=numAntosianinRGB).place(relx=.63,rely=.6,anchor="nw")
# labelPercent3.place(relx=.71,rely=.5,anchor="nw")

# miniFrameForEntry.place(relx=.7,rely=.4,anchor="center")
buttonToBack.place(relx=.03,rely=.03,anchor="nw")
labelGambar.place(relx=.25,rely=.45,anchor="center")
buttonChooseImageAdd.place(relx=.25,rely=.7,anchor="center")
buttonAddData.place(relx=.5,rely=.9,anchor="center")
#=========================================================================================
#FRAME ADD DATA MONOCHROME
imageNoMono = cv2.imread('assets/no-image.png')
imageNoMono = cv2.resize(imageNoMono,(125,125))
imgNoMono = Image.fromarray(imageNoMono)
imgNoMono = ImageTk.PhotoImage(imgNoMono)

buttonToBack1 = Button(addDataMonoFrameRight,image=backImage,font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=70,height=50,command=lambda: menuClick(addDataFrameRight))

labelGambar350 = Label(addDataMonoFrameRight,image=imgNoMono,borderwidth=1, relief="solid",width=125,height=125)
labelGambar400 = Label(addDataMonoFrameRight,image=imgNoMono,borderwidth=1, relief="solid",width=125,height=125)
labelGambar450 = Label(addDataMonoFrameRight,image=imgNoMono,borderwidth=1, relief="solid",width=125,height=125)
labelGambar500 = Label(addDataMonoFrameRight,image=imgNoMono,borderwidth=1, relief="solid",width=125,height=125)
labelGambar550 = Label(addDataMonoFrameRight,image=imgNoMono,borderwidth=1, relief="solid",width=125,height=125)
labelGambar600 = Label(addDataMonoFrameRight,image=imgNoMono,borderwidth=1, relief="solid",width=125,height=125)
labelGambar650 = Label(addDataMonoFrameRight,image=imgNoMono,borderwidth=1, relief="solid",width=125,height=125)
labelGambar700 = Label(addDataMonoFrameRight,image=imgNoMono,borderwidth=1, relief="solid",width=125,height=125)
labelGambar750 = Label(addDataMonoFrameRight,image=imgNoMono,borderwidth=1, relief="solid",width=125,height=125)
labelGambar800 = Label(addDataMonoFrameRight,image=imgNoMono,borderwidth=1, relief="solid",width=125,height=125)

buttonChooseImageAddMono1 = Button(addDataMonoFrameRight,text="Select Image",font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=12,command=lambda: chooseImageAddMono(1))

buttonChooseImageAddMono2 = Button(addDataMonoFrameRight,text="Select Image",font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=12,command=lambda: chooseImageAddMono(2))

buttonChooseImageAddMono3 = Button(addDataMonoFrameRight,text="Select Image",font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=12,command=lambda: chooseImageAddMono(3))

buttonChooseImageAddMono4 = Button(addDataMonoFrameRight,text="Select Image",font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=12,command=lambda: chooseImageAddMono(4))

buttonChooseImageAddMono5 = Button(addDataMonoFrameRight,text="Select Image",font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=12,command=lambda: chooseImageAddMono(5))

buttonChooseImageAddMono6 = Button(addDataMonoFrameRight,text="Select Image",font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=12,command=lambda: chooseImageAddMono(6))

buttonChooseImageAddMono7 = Button(addDataMonoFrameRight,text="Select Image",font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=12,command=lambda: chooseImageAddMono(7))

buttonChooseImageAddMono8 = Button(addDataMonoFrameRight,text="Select Image",font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=12,command=lambda: chooseImageAddMono(8))

buttonChooseImageAddMono9 = Button(addDataMonoFrameRight,text="Select Image",font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=12,command=lambda: chooseImageAddMono(9))

buttonChooseImageAddMono10 = Button(addDataMonoFrameRight,text="Select Image",font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,width=12,command=lambda: chooseImageAddMono(10))

labelKlorofilMono = Label(addDataMonoFrameRight,text="Chlorophyll green index :",font="Calibre 10 bold")
labelKarotenoidMono = Label(addDataMonoFrameRight,text="Carotenoid green index :",font="Calibre 10 bold")
labelAntosianinMono = Label(addDataMonoFrameRight,text="Anthocyanin :",font="Calibre 11 bold")
labelKlorofilMono1 = Label(addDataMonoFrameRight,text="Chlorophyll rededge index  :",font="Calibre 10 bold")
labelKarotenoidMono1 = Label(addDataMonoFrameRight,text="Carotenoid rededge index :",font="Calibre 10 bold")
# labelPercentMono1 = Label(addDataMonoFrameRight,text="%",font="Calibre 13 bold")
# labelPercentMono2 = Label(addDataMonoFrameRight,text="%",font="Calibre 13 bold")
# labelPercentMono3 = Label(addDataMonoFrameRight,text="%",font="Calibre 13 bold")
trainKlorofilMono = DoubleVar()
trainKlorofilMono.set(0)
trainKlorofilMono1 = DoubleVar()
trainKlorofilMono1.set(0)
trainKarotenoidMono = DoubleVar()
trainKarotenoidMono.set(0)
trainKarotenoidMono1 = DoubleVar()
trainKarotenoidMono1.set(0)
trainAntosianinMono = DoubleVar()
trainAntosianinMono.set(0)

entryKlorofilMono = Entry(addDataMonoFrameRight,textvariable=trainKlorofilMono, font="Calibre 11 bold",justify='right',width=5)
entryKarotenoidMono = Entry(addDataMonoFrameRight,textvariable=trainKarotenoidMono, font="Calibre 11 bold",justify='right',width=5)
entryKlorofilMono1 = Entry(addDataMonoFrameRight,textvariable=trainKlorofilMono1, font="Calibre 11 bold",justify='right',width=5)
entryKarotenoidMono1 = Entry(addDataMonoFrameRight,textvariable=trainKarotenoidMono1, font="Calibre 11 bold",justify='right',width=5)
entryAntosianinMono = Entry(addDataMonoFrameRight,textvariable=trainAntosianinMono, font="Calibre 11 bold",justify='right',width=5)

buttonAddDataMono = Button(addDataMonoFrameRight,text="ADD DATA TO MONOCHROME DATASET",font="Calibre 13 bold",bg="#006266",fg="white"
	,borderwidth=1,height=2,command=lambda: addData('mono'))

buttonToBack1.place(relx=.03,rely=.03,anchor="nw")

Label(addDataMonoFrameRight,text="350 nm",bg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.1,rely=.17,anchor="center")
Label(addDataMonoFrameRight,text="400 nm",bg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.3,rely=.17,anchor="center")
Label(addDataMonoFrameRight,text="450 nm",bg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.5,rely=.17,anchor="center")
Label(addDataMonoFrameRight,text="500 nm",bg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.7,rely=.17,anchor="center")
Label(addDataMonoFrameRight,text="550 nm",bg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.9,rely=.17,anchor="center")
Label(addDataMonoFrameRight,text="600 nm",bg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.1,rely=.5,anchor="center")
Label(addDataMonoFrameRight,text="650 nm",bg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.3,rely=.5,anchor="center")
Label(addDataMonoFrameRight,text="700 nm",bg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.5,rely=.5,anchor="center")
Label(addDataMonoFrameRight,text="750 nm",bg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.7,rely=.5,anchor="center")
Label(addDataMonoFrameRight,text="800 nm",bg=rightSideBackgroundColor,font="Calibre 13 bold").place(relx=.9,rely=.5,anchor="center")

labelGambar350.place(relx=.1,rely=.3,anchor="center")
labelGambar400.place(relx=.3,rely=.3,anchor="center")
labelGambar450.place(relx=.5,rely=.3,anchor="center")
labelGambar500.place(relx=.7,rely=.3,anchor="center")
labelGambar550.place(relx=.9,rely=.3,anchor="center")
labelGambar600.place(relx=.1,rely=.63,anchor="center")
labelGambar650.place(relx=.3,rely=.63,anchor="center")
labelGambar700.place(relx=.5,rely=.63,anchor="center")
labelGambar750.place(relx=.7,rely=.63,anchor="center")
labelGambar800.place(relx=.9,rely=.63,anchor="center")

buttonChooseImageAddMono1.place(relx=.1,rely=.43,anchor="center")
buttonChooseImageAddMono2.place(relx=.3,rely=.43,anchor="center")
buttonChooseImageAddMono3.place(relx=.5,rely=.43,anchor="center")
buttonChooseImageAddMono4.place(relx=.7,rely=.43,anchor="center")
buttonChooseImageAddMono5.place(relx=.9,rely=.43,anchor="center")
buttonChooseImageAddMono6.place(relx=.1,rely=.76,anchor="center")
buttonChooseImageAddMono7.place(relx=.3,rely=.76,anchor="center")
buttonChooseImageAddMono8.place(relx=.5,rely=.76,anchor="center")
buttonChooseImageAddMono9.place(relx=.7,rely=.76,anchor="center")
buttonChooseImageAddMono10.place(relx=.9,rely=.76,anchor="center")

labelKlorofilMono.place(relx=.01,rely=.84,anchor="nw")
labelKarotenoidMono.place(relx=.01,rely=.90,anchor="nw")
labelKlorofilMono1.place(relx=.29,rely=.84,anchor="nw")
labelKarotenoidMono1.place(relx=.29,rely=.90,anchor="nw")
labelAntosianinMono.place(relx=.01,rely=.96,anchor="nw")
# labelPercentMono1.place(relx=.3,rely=.84,anchor="center")
# labelPercentMono2.place(relx=.3,rely=.90,anchor="center")
# labelPercentMono3.place(relx=.3,rely=.96,anchor="center")
entryKlorofilMono.place(relx=.21,rely=.84,anchor="nw")
entryKarotenoidMono.place(relx=.21,rely=.90,anchor="nw")
entryKlorofilMono1.place(relx=.51,rely=.84,anchor="nw")
entryKarotenoidMono1.place(relx=.51,rely=.90,anchor="nw")
entryAntosianinMono.place(relx=.21,rely=.96,anchor="nw")
buttonAddDataMono.place(relx=.78,rely=.89,anchor="center")

#=========================================================================================
#FRAME HOW TO USE MAIN MENU

Button(howToFrameRight,text="Main Menu (RGB)",font="Calibre 12 bold",bg="#006266",fg="white",borderwidth=1,width=15,command= lambda: changeHowTo(1)).place(relx=0.12,rely=0.06,anchor="center")
Button(howToFrameRight,text="Add Data (RGB)",font="Calibre 12 bold",bg="#006266",fg="white",borderwidth=1,width=15,command=lambda: changeHowTo(4)).place(relx=0.32,rely=0.06,anchor="center")
Button(howToFrameRight,text="Training Menu",font="Calibre 12 bold",bg="#006266",fg="white",borderwidth=1,width=15,command=lambda: changeHowTo(3)).place(relx=0.52,rely=0.06,anchor="center")

how = PhotoImage(file='assets/howto1.png')
pictHowTo = Label(howToFrameRight,image=how,width=790,height=530)
pictHowTo.place(relx=.001,rely=.1,anchor="nw")

Button(howToFrameRight,text="Main Menu (Monochrome)",font="Calibre 12 bold",bg="#006266",fg="white",borderwidth=1,width=21,command=lambda: changeHowTo(2)).place(relx=0.155,rely=0.12,anchor="center")
Button(howToFrameRight,text="Add Data (Monochrome)",font="Calibre 12 bold",bg="#006266",fg="white",borderwidth=1,width=21,command=lambda: changeHowTo(5)).place(relx=0.42,rely=0.12,anchor="center")

#=========================================================================================
#OBJECT + KONFIGURASI FRAME ABOUT
about = PhotoImage(file='assets/about.png')
about1 = PhotoImage(file='assets/about1.png')
pictAbout = Label(aboutFrameRight,image=about,width=800,height=600)
pictAbout["bg"] = aboutFrameRight["bg"]
pictAbout.place(relx=.5,rely=.5,anchor="center")

buttonChange = Button(aboutFrameRight,text="Developer",font="Calibre 12 bold",bg="#006266",fg="white",borderwidth=1,width=15,command=changeAbout)
buttonChange.place(relx=.1,rely=.05,anchor="center")

mainFrameLeft.pack(side=LEFT,fill="both")
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)
container.pack(side=RIGHT, fill="both",expand=True)

hoverButton()
# resizeWindow()

root.mainloop()