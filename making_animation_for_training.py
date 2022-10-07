"""## Let's make some animation HAHA

Here I am trying to make some animation from the training
"""

from pandas.core.frame import FilePathOrBuffer
import tcnae
import imp
imp.reload(tcnae)


# Number of the 600s simulationd used for training.
TrnSzSd = 40

# The wind speed for the training
u='16mps'

# The Channel (sensor) that training is being done on
ch = 'MyTB'

# Choosing 40 random simulations out of 90 availble one for every wind speed
rnd_seed = np.random.random_integers(1, high=90, size=TrnSzSd)
print(rnd_seed)

# To make it easier to follow, 
x = data_sets_norm['wspdX'][u].iloc[rnd_seed,1:]
y = data_sets_norm[ch][u].iloc[rnd_seed,1:]
print(x.shape)
print(y.shape)

tcn_ae = tcnae.TCNAE() # Use the parameters specified in the paper

epoch = 300

time = np.linspace(0,600,12000)
seeds = np.arange(0,90)
rem_seeds = seeds[~np.in1d(np.arange(seeds.size),rnd_seed)]
SeedNo=random.choice(rem_seeds)
epoch_xaxis = np.arange(1,301)

for i in range(0,epoch):
  history = tcn_ae.model.fit(x, y, 
                            batch_size=12, 
                            validation_split=0.1, 
                            shuffle=True,
                            verbose=1)
  y_pred = tcn_ae.model.predict(data_sets_norm['wspdX'][u].iloc[SeedNo:SeedNo+1,1:])
  fig,ax = plt.subplots(1,2,figsize=(16,9),gridspec_kw={'width_ratios': [2, 1]})
  ax[0].plot(time, data_sets_norm[ch][u].iloc[SeedNo,1:],label ='Simulation', linewidth=2)
  ax[0].plot(time,y_pred[0,:,:],label='Prediction',alpha=0.6, linewidth=2)
  ax[0].set_ylabel('Scaled Fore-Aft moment @ Tower Bottom [-]')
  ax[0].grid()
  ax[0].legend()
  ax[0].set_xlabel('Time [s]')
  a = ax[1].hist(data_sets_norm[ch][u].iloc[SeedNo,1:], density=True, bins =20, orientation='horizontal',label ='Simulation',alpha = 0.75)
  _ = ax[1].hist(y_pred[0,:,0],label='Prediction',bins = a[1], orientation='horizontal',alpha = 0.7,density=True)
  ax[1].grid()
  ax[1].legend()
  ax[1].set_xlabel('Probablity Density [-]')
  fig.suptitle(f"Epoch No {i}")
  #plt.title()
#fig.savefig('Animation_MyB/TCN_EncDec_UniWind_'+u+'_'+'Norm'+ch+'_epoch_'+str(i+1)+'.png', dpi=200, facecolor='w',
#          edgecolor='w', orientation='portrait', format='png', transparent=False)
  fig.savefig('Animation_MyB/TCN_EncDec_UniWind_'+u+'_'+'Norm'+ch+'_epoch_'+str(i+1).zfill(3)+'.png', dpi=200, facecolor='w',
            edgecolor='w', orientation='portrait', format='png', transparent=False)
  plt.close('all')


import cv2
import os


image_folder = 'Animation_MyB'
video_name = 'video.avi'

images = os.listdir(image_folder)
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape


fourcc = cv2.VideoWriter_fourcc(*'MPEG')
video = cv2.VideoWriter(video_name, fourcc, 5.0, (width,height))


for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

import cv2
import os


image_folder = 'Animation_MyB'
video_name = 'video.avi'

images = os.listdir(image_folder)
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape


fourcc = cv2.VideoWriter_fourcc(*'MPEG')
video = cv2.VideoWriter(video_name, fourcc, 10, (width,height))


for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

#!pip install opencv-python

os.listdir(image_folder)
print(images)

import pickle
f = open('history_16mps_MyTB.pckl','wb')
pickle.dump(history,f)
f.close()

!pip install ffmpeg

os.system('ffmpeg -n -framerate 8  -i "Animation_MyB/TCN_EncDec_UniWind_16mps_NormMyTB_epoch_%00d.png"  -vf "fps25,format=yuv420p" movie.mp4')