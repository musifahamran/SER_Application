
from flask import Flask, render_template, request, flash, url_for
from werkzeug.utils import secure_filename, redirect
from get_features.audio_processing import extract_feature, transform_image
import torch
from model import SER_AlexNet,SER_AlexNet_GAP, SER_RESNET
from pandas import np
import os
import torch.nn.functional as f


UPLOAD_FOLDER = r'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Change path to project location
dir = r'C:\Users\Musifah\Documents\Github\SER_Application'

#Change pretrained model in models folder
pretrained_model = 'fcn_model.pth'

#File size limit is 16mb
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

#Allow only audio file
ALLOWED_EXTENSIONS = set(['wav', 'mp3'])

filename = ''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#setup model
def setup_model():
    model_type = pretrained_model.split('_')
    model = SER_models(network=model_type[0])
    model.load_state_dict(torch.load(os.path.join(dir,'models',pretrained_model),map_location=torch.device('cpu')))
    device = torch.device("cpu")
    model.to(device)
    return model, device

def SER_models(network='alexnet'):
    preT_models = ['']
    if network == 'alexnet':
        model = SER_AlexNet(pretrained=True)
    elif network == 'fcn':
        model= SER_AlexNet_GAP(pretrained=True)
    elif network == 'resnet':
        model = SER_RESNET(pretrained=True)
    return model


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

    return test_preds

def get_preds(audio_file, num_segs,seg_preds):
    """
    Get predictions for utterance from their segments' prediction.
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

    #Remove converted files after prediction is done
    os.remove("{}/{}.wav".format(file_dest,tail))
    os.remove("{}/{}.pkl".format(file_dest,tail))
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
 	return redirect(url_for('static', filename='uploads/' + filename, code=301))

if __name__ == '__main__':
    app.run(debug=True)
