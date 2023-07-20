

## Import the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
# %matplotlib inline
from scipy import signal
## Load the dataset
d = pd.read_csv('/content/data.csv')
#d

des=pd.read_csv('/content/design_sf.csv')
#des

des_dict={row[1]['id']: row[1]['sfreq'] for row in des[['id', 'sfreq']].iterrows()}
des_dict

## View the no of column and rows present in the dataset
d.shape

plt.figure(figsize=(14, 4))
(d['Marker']).astype(np.int).plot()

t = d.timestamps

## First 5 column is eeg signal and last column is used as a tag
eeg = np.array(d.iloc[:, 1:5])
tag = np.array(d.loc[:, 'Marker'])

prev = 0
for i, t in enumerate(tag):
    if t != 0:
        prev = t
    elif i != 0:
        tag[i] = prev

plt.plot(tag)
plt.show()

## Now see the data present in eeg 
eeg.shape

## Plot eeg siganls
plt.plot(eeg)

## Plot the colorbar map of eeg data in this legends identify discrete label of discrete points
_ = plt.specgram(eeg[:, 2], NFFT=128, Fs=200, noverlap=64)
plt.colorbar()

# Remove DC offset which is the mean amplitude displacement from zero.
hp_cutoff_Hz = 1.0
fs_Hz = 200
b, a = signal.butter(2, hp_cutoff_Hz/(fs_Hz / 2.0), 'highpass')
eeg = signal.lfilter(b, a, eeg, axis=0)

_ = plt.specgram(eeg[:, 0], NFFT=128, Fs=200, noverlap=64)
plt.colorbar()

## Notch filter that is used to remove the fixed frequency noise source
notch_freq_Hz = np.array([60.0])
freq_Hz = notch_freq_Hz
bp_stop_Hz = freq_Hz + 3.0*np.array([-1, 1])  # set the stop band
b, a = signal.butter(3, bp_stop_Hz/(fs_Hz / 2.0), 'bandstop')
eeg = signal.lfilter(b, a, eeg, axis=0)
print("Notch filter removing: " + str(bp_stop_Hz[0]) + "-" + str(bp_stop_Hz[1]) + " Hz")

_ = plt.specgram(eeg[:, 0], NFFT=128, Fs=200, noverlap=64)
plt.colorbar()

plt.plot(eeg[1:200, 0])
plt.plot(eeg[-300:, 0])

eeg = eeg[50:,]
tag = tag[50:,]

plt.plot(eeg[1:200, 0])
plt.plot(eeg[-300:, 0])

## Filter from 5 to 35 Hz, helps remove 60Hz noise also helps remove the DC line noise
## 100 is half the sampling rate (250Hz/2)
b, a = signal.butter(3, (5.0/100, 50.0/100), btype='bandpass') 
b, a

#eeg_f = eeg 
eeg_f = signal.lfilter(b, a, eeg, axis=0)

t_sec = np.arange(len(eeg[:, 0])) / fs_Hz

plt.figure(figsize=(10,5))
plt.subplot(1,1,1)
plt.plot(t_sec,eeg_f)
plt.xlabel('Time (sec)')
plt.ylabel('Power (uV)')
plt.title('Signal')
plt.show()

## Import the mlab library
import matplotlib.mlab as mlab

## here import the FastICA module that is fast independent analysis algorithm used to separate independent sources from a mixed signal.
from sklearn.decomposition import FastICA
ica = FastICA()
sources = ica.fit_transform(eeg_f)
means = ica.mean_.copy()
mixing = ica.mixing_.copy()

## Look at the plots to find the eyeblink component
for i in range(ica.components_.shape[0]):
    plt.figure(figsize=(16,4))
    plt.plot(sources[:, i])
    plt.title(i)

plt.figure()
plt.plot(eeg_f[500:9000, 0])

word_starts = []
prev_t = None

for i, t in enumerate(tag):
    if t != -1 and t != 0 and t != prev_t:
        w = t #word_dict[t]
        word_starts.append( {'index': i, 
                             'word': t,
                             'dict': w} )
    prev_t = t

## this confirms that there's ~2.5 seconds between words
np.diff([x['index'] for x in word_starts]) / 200.0

trial_types = np.array([w['word'] for w in word_starts])

t_before = 0.5
t_after  = 1.0
eeg_trials = np.zeros((4, len(word_starts), int((t_before+t_after)*200)))
time = np.arange(0, eeg_trials.shape[2], 1) / 200.0 - t_before

for c in range(4):
    for i in range(len(word_starts)):
        d = word_starts[i]
        start = d['index']
        if start < 100:
            continue
        # 100 samples = 0.5s, 400 samples = 2.0 s
        # we want 0.5s before the stimulus presentation and 2.0 seconds after
        eeg_trials[c, i, :] = eeg_f[int(start-t_before*200):int(start+t_after*200), c]

## Moving average is used to analyze data points by creating a series of averages of different subsets of the full data set.
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

N_AVG = 10
eeg_trials_smooth = np.apply_along_axis(moving_average, axis=2, arr=eeg_trials, n=N_AVG)
eeg_trials_smooth.shape

values = [str(v) for k,v in des_dict.items()]
values[2] = 'CTRL'

for i in range(4):
    plt.figure(figsize=(10,4))
    for t in range(len(np.unique(trial_types))):
        x= time[(N_AVG-1):]
        y= np.mean(eeg_trials_smooth[i][trial_types==(t+1)],axis=0)
        err= np.std(eeg_trials_smooth[i][trial_types==(t+1)],axis=0) / np.sqrt(eeg_trials_smooth[i][trial_types==(t+1)].shape[0])
        plt.plot(x, y)
        plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=4)
        plt.legend(values)    
    plt.xlabel('Time since stimulus (s)', fontsize=15)
    plt.ylabel('EEG amplitude\n(arbitrary units)', fontsize=15)
    plt.savefig('./amplt_by_time_ch{}.png'.format(i))

## This function is made to combine the above 4 grphas into one graph
## get_power_spectrum is computes the rotationally averaged power spectra of a series of images and averages 
#these spectra into one spectrum for the whole set of images
def get_power_spect(data):
    spec_PSDperHz, _, _ = mlab.specgram(data, NFFT=164, window=mlab.window_hanning, Fs=fs_Hz, noverlap=28)
    return spec_PSDperHz.mean(axis=1)

eeg_spect = np.apply_along_axis(get_power_spect, axis=2, arr=eeg_trials)

plt.plot(eeg_spect.mean(axis=0)[trial_types==1].mean(axis=0))
plt.plot(eeg_spect.mean(axis=0)[trial_types==2].mean(axis=0))
plt.plot(eeg_spect.mean(axis=0)[trial_types==3].mean(axis=0))

# channels are concatenated horizontally
eeg_trials_X = eeg_trials.swapaxes(0,1).reshape((eeg_trials.shape[1], eeg_trials.shape[0]*eeg_trials.shape[2])) 
eeg_trials_Y = trial_types

eeg_spect_X = eeg_spect.swapaxes(0,1).reshape((eeg_spect.shape[1], eeg_spect.shape[0]*eeg_spect.shape[2]))
eeg_trials_X = np.concatenate([eeg_trials_X, eeg_spect_X], axis=1)

eeg_trials_X = np.concatenate(eeg_trials, axis=0)
eeg_trials_Y = np.repeat(trial_types, 4)

eeg_trials_X_smooth = np.vstack([moving_average(eeg_trials_X[r,:], n=10) for r in range(eeg_trials_X.shape[0])])

## LogisticRegression model is used to get the accuracy score of the dataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(eeg_trials_X_smooth, eeg_trials_Y, stratify=eeg_trials_Y, test_size=.3)

model = LogisticRegression(penalty='l1', solver='liblinear', multi_class='auto')
scores = cross_val_score(model, X=eeg_trials_X_smooth, y=eeg_trials_Y, scoring='accuracy', cv=5)

scores.mean()

model.fit(eeg_trials_X_smooth, eeg_trials_Y)
plt.figure(figsize=(14, 4))
plt.plot(model.coef_.T)
plt.legend(values)

## Another classifier that is used to get the accuracy score of the model is RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=2000, oob_score=True)
scores = cross_val_score(model, X=eeg_trials_X_smooth, y=eeg_trials_Y, scoring='accuracy', cv=5)
scores.mean()

model.fit(X_train, y_train)
model.score(X_test, y_test)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=2000)
scores = cross_val_score(model, X=eeg_trials_X[np.isin(eeg_trials_Y, [1,2])], y=eeg_trials_Y[np.isin(eeg_trials_Y, [1,2])], scoring='accuracy', cv=5)
scores.mean()