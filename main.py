import os
import pickle

import torch
import torchvision
from flask import Flask, render_template, request, flash, url_for
from pandas import np
from torch import nn
from werkzeug.utils import secure_filename, redirect
from get_features.extractFeatures import extract_features
from torchvision import transforms
from PIL import Image
import torch.nn.functional as f


UPLOAD_FOLDER = r'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
dir = r'C:\Users\Musifah\PycharmProjects\SERApplication'
#device = torch.device("cpu")
# device = torch.device("cuda:0")

#File size limit is 16mb
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['wav', 'mp3'])
SCALER_TYPE = {'standard':'preprocessing.StandardScaler()',
               'minmax'  :'preprocessing.MinMaxScaler(feature_range=(0,1))'
              }
IEMOCAP_EMO_CODES = {0: 'ang', 1:'sad', 2:'hap', 3:'neu'}
filename = ''

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def setup_model(in_ch=3):
    model = torchvision.models.alexnet(pretrained=True)  # Trained on 1000 classes from ImageNet
    features = model.features
    avgpool = model.avgpool
    classifier = model.classifier

    if in_ch != 3:
        features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        init_layer(features[0])

    model.classifier[6] = torch.nn.Linear(4096, 4)
    model.load_state_dict(torch.load(r'C:\Users\Musifah\PycharmProjects\SERApplication\alexnet_mixup.pth'))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    return model, device

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


# def _normalize(self, scaling):
#
#         '''
#         calculate normalization factor from training dataset and apply to
#            the whole dataset
#         '''
#
#         # get data range
#         input_range = self._get_data_range()
#
#         # re-arrange array from (N, C, F, T) to (C, -1, F)
#         nsegs = self.train_data.shape[0]
#         nch = self.train_data.shape[1]
#         nfreq = self.train_data.shape[2]
#         ntime = self.train_data.shape[3]
#         rearrange = lambda x: x.transpose(1, 0, 3, 2).reshape(nch, -1, nfreq)
#         self.train_data = rearrange(self.train_data)
#         self.val_data = rearrange(self.val_data)
#         self.test_data = rearrange(self.test_data)
#
#         # scaler type
#         scaler = eval(SCALER_TYPE[scaling])
#
#         for ch in range(nch):
#             # get scaling values from training data
#             scale_values = scaler.fit(self.train_data[ch])
#
#             # apply to all
#             self.train_data[ch] = scaler.transform(self.train_data[ch])
#             self.val_data[ch] = scaler.transform(self.val_data[ch])
#             self.test_data[ch] = scaler.transform(self.test_data[ch])
#
#         # Shape the data back to (N,C,F,T)
#         rearrange = lambda x: x.reshape(nch, -1, ntime, nfreq).transpose(1, 0, 3, 2)
#         self.train_data = rearrange(self.train_data)
#         self.val_data = rearrange(self.val_data)
#         self.test_data = rearrange(self.test_data)
#
#         print(f'\nDataset normalized with {scaling} scaler')
#         print(f'\tRange before normalization: {input_range}')
#         print(f'\tRange after  normalization: {self._get_data_range()}')
#
# def _get_data_range(self):
#         # get data range
#         trmin = np.min(self.train_data)
#         evmin = np.min(self.val_data)
#         tsmin = np.min(self.test_data)
#         dmin = np.min(np.array([trmin, evmin, tsmin]))
#
#         trmax = np.max(self.train_data)
#         evmax = np.max(self.val_data)
#         tsmax = np.max(self.test_data)
#         dmax = np.max(np.array([trmax, evmax, tsmax]))
#
#         return [dmin, dmax]
def output_emotion(model, audio_file, num_segs, device):
    test_loader = torch.utils.data.DataLoader(audio_file, batch_size=1, shuffle=False)
    test_preds_segs = []

    model.eval()
    for i, batch in enumerate(test_loader):
        test_data_batch = batch

        # Send to correct device
        test_data_batch = test_data_batch.to(device)

        # Forward
        test_preds_batch = model(test_data_batch)
        test_preds_segs.append(f.log_softmax(test_preds_batch, dim=1).cpu())

    # Accumulate results for val data
    test_preds_segs = np.vstack(test_preds_segs)
    test_preds = get_preds(audio_file, num_segs, test_preds_segs)

    # Make sure everything works properly
    # assert len(test_preds) == test_dataset.n_actual_samples
    # test_wa = test_dataset.weighted_accuracy(test_preds)
    # test_ua = test_dataset.unweighted_accuracy(test_preds)

    # results = (test_loss, test_wa * 100, test_ua * 100)

    # if return_matrix:
    #     test_conf = test_dataset.confusion_matrix_iemocap(test_preds)
    #     return results, test_conf
    # else:
    #     return results
    return test_preds


def get_preds(audio_file, num_segs,seg_preds):
    """
    Get predictions for all utterances from their segments' prediction.
    This function will accumulate the predictions for each utterance by
    taking the maximum probability along the dimension 0 of all segments
    belonging to that particular utterance.
    """
    n_actual_samples = len(num_segs)
    preds = np.empty(
        shape=(n_actual_samples, 4), dtype="float")

    end = 0
    for v in range(n_actual_samples):
        start = end
        end = start + num_segs[v]

        preds[v] = np.average(seg_preds[start:end], axis=0)

    preds = np.argmax(preds, axis=1)
    return preds

def get_emotion(prediction):

    if prediction == 0:
        emotion = "angry"
    elif prediction == 1:
        emotion = "sad"
    elif prediction == 2:
        emotion = "happy"
    else:
        emotion = "neutral"

    return emotion

def get_prediction(filename):
    name, ext = os.path.splitext(filename)
    print("Audio file", name)
    head, tail = os.path.split(name)
    file_dest = os.path.join(dir,UPLOAD_FOLDER)
    tail = tail + "_conv"
    os.system('C:/ffmpeg/bin/ffmpeg.exe -i {} -acodec pcm_s16le -ar 16000 {}/{}.wav'.format(filename, file_dest, tail))
    extract_feature(os.path.join(dir,UPLOAD_FOLDER, tail+".wav"))
    model, device = setup_model()
    img, img_seg = transform_image(os.path.join(dir,UPLOAD_FOLDER,tail) + '.pkl')

    #get prediction from model --> unsure if right code
    with torch.no_grad():
        prediction  = output_emotion(model, img,img_seg, device=device)
        print("Prediction classes: ", prediction[0])

    emotion = get_emotion(prediction[0])
    os.remove("{}/{}.wav".format(file_dest, filename))
    return "Audio File is classified as " + emotion + " emotion"

@app.route('/')
def load_file():
    return render_template('upload.html')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if request.form['btn_identifier'] == 'submit_btn':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                global filename
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return render_template('upload.html', filename=filename)
        elif request.form['btn_identifier'] == 'classify_btn':
            prediction_idx = get_prediction(os.path.join(dir,UPLOAD_FOLDER,filename))
            flash(prediction_idx)
            return redirect(request.url)

@app.route('/display/<filename>')
def display_audio(filename):
 	#print('display_image filename: ' + filename)
 	return redirect(url_for('static', filename='uploads/' + filename, code=301))

if __name__ == '__main__':
    app.run(debug=True)
    #prediction_idx = get_prediction(os.path.join(dir, UPLOAD_FOLDER, 'dia0_utt0.wav'))
