from get_features.extractFeatures import extract_features
from torchvision import transforms
from PIL import Image
import pickle
from sklearn import preprocessing
from pandas import np
import os
SCALER_TYPE = {'standard':'preprocessing.StandardScaler()',
               'minmax'  :'preprocessing.MinMaxScaler(feature_range=(0,1))'
              }

#extract features to perform classification
def extract_feature(filename):
    features = 'logspec'
    # Get spectrogram parameters
    params = {'window': 'hamming',
              'win_length': 40,
              'hop_length': 10,
              'ndft': 800,
              'nfreq': 200,
              'segment_size': 300}
    features_data = extract_features(filename, features, params)

    if features_data is not None:
        name, ext = os.path.splitext(filename)
        out_filename = name + '.pkl'
        with open(out_filename, "wb") as fout:
            pickle.dump(features_data, fout)

#transform extracted log spectrogram features into image
def transform_image(filename):
    data = None
    with open(filename, "rb") as finTr:
        audio_file = pickle.load(finTr)
    #Features Post-processing
    if data is None:
        #return (audio_file)
        data = audio_file[0][0].astype(np.float32)
        data_seg = audio_file[0][1]

    data= normalize(data,'minmax')
    data_shape = data.shape
    data = spec_to_gray(data)
    num_in_ch = 3

    assert len(data) == data_shape[0]
    #print("RGB Data values: ", data)
    return data, data_seg

#post-processing of features before turning into image
def normalize(data,scaling):
    # re-arrange array from (N, C, F, T) to (C, -1, F)
    nsegs = data.shape[0]
    nch = data.shape[1]
    nfreq = data.shape[2]
    ntime = data.shape[3]
    rearrange = lambda x: x.transpose(1, 0, 3, 2).reshape(nch, -1, nfreq)
    data = rearrange(data)

    scaler = eval(SCALER_TYPE[scaling])

    for ch in range(nch):
            # get scaling values from training data
        scale_values = scaler.fit(data[ch])

        # apply to all
        data[ch] = scaler.transform(data[ch])

    # Shape the data back to (N,C,F,T)
    rearrange = lambda x: x.reshape(nch, -1, ntime, nfreq).transpose(1, 0, 3, 2)
    data = rearrange(data)
    return data

def spec_to_gray(data):

        """
        Convert normalized spectrogram to 3-channel gray image (identical data on each channel)
            and apply AlexNet image pre-processing

        Input: data
                - shape (N,C,H,W) = (num_spectrogram_segments, 1, Freq, Time)
                - data range [0.0, 1.0]
        """

        # AlexNet preprocessing
        alexnet_preprocess = transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Convert format to uint8, flip the frequency axis to orientate image upward,
        #   duplicate into 3 channels
        data = np.clip(data, 0.0, 1.0)
        data = (data * 255.0).astype(np.uint8)
        data = np.flip(data, axis=2)
        data = np.moveaxis(data, 1, -1)
        data = np.repeat(data, 3, axis=-1)

        data_tensor = list()
        for i, seg in enumerate(data):
            img = Image.fromarray(seg, mode='RGB')
            data_tensor.append(alexnet_preprocess(img))

        return data_tensor

